#!/usr/bin/env python3
"""
Active Acoustic Inference Demo
================================
Demonstrates active inference for room acoustic characterization.

A Gaussian Process agent adaptively selects probe frequencies,
using simulated lock-in (synchronous) detection to measure the
complex room transfer function H(f) with sub-noise-floor sensitivity.

Three panels show:
  Top:    Amplitude |H(f)| — GP mean ± 2σ vs. ground truth
  Middle: Phase ∠H(f)     — GP mean vs. ground truth
  Bottom: Posterior std σ(f) — shows where uncertainty remains high

The agent always probes the frequency of maximum posterior variance σ²(f).
This is the epistemic-value-maximising policy under a Gaussian generative
model — equivalent to minimising expected free energy (active inference).

To run interactively on your laptop:
    Comment out the matplotlib.use('Agg') line below.
    Then: python active_acoustic_inference.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  — edit this section before running
# ═══════════════════════════════════════════════════════════════════════════════

MODE          = 'real'    # 'sim' for simulation, 'real' for real microphone + speaker

# Real audio settings (only used when MODE = 'real')
INPUT_DEVICE  = 3     # None = system default mic.  Run list_devices() to find index.
OUTPUT_DEVICE = 3     # None = system default speaker.
DRIVE_AMP     = 0.5    # Speaker output level 0.0–1.0.  0.05 is very quiet.
SAMPLE_RATE   = 44100    # Hz — standard audio rate
N_STEPS       = 60       # Number of probe frequencies
T_PER_STEP    = 3.0      # Lock-in integration time per frequency (seconds)
F_MIN         = 50       # Hz — lowest frequency to probe
F_MAX         = 2000     # Hz — highest frequency to probe

# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SYNTHETIC ROOM MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class SyntheticRoom:
    """
    Room transfer function H(f) as a sum of Lorentzian (resonant) modes.

        H(f) = Σ_n  A_n · f₀ₙ² / (f₀ₙ² − f² + i·f·f₀ₙ/Q_n)

    Peak magnitude of mode n at resonance = A_n · Q_n.
    Normalised so max |H(f)| = 1 over the frequency grid.
    """

    MODES = [  # (f₀ Hz,   Q,    A)
        (  87,    8,   0.9),
        ( 134,    6,   0.7),
        ( 178,   10,   1.0),
        ( 243,    5,   0.5),
        ( 312,    7,   0.6),
        ( 445,    6,   0.4),
        ( 623,    8,   0.5),
        ( 890,    5,   0.3),
        (1240,    6,   0.25),
        (1680,    7,   0.20),
    ]

    def __init__(self, f_grid):
        self.f_grid = f_grid
        self.H = self._build(f_grid)
        self.H /= np.max(np.abs(self.H))   # normalise to peak = 1

    def _build(self, f):
        H = np.zeros(len(f), dtype=complex)
        for f0, Q, A in self.MODES:
            H += A * f0**2 / (f0**2 - f**2 + 1j * f * f0 / Q)
        return H

    def at(self, f):
        """Complex H at arbitrary frequency f (interpolated)."""
        re = np.interp(f, self.f_grid, self.H.real)
        im = np.interp(f, self.f_grid, self.H.imag)
        return re + 1j * im


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SIMULATED LOCK-IN MEASUREMENT
# ═══════════════════════════════════════════════════════════════════════════════

class SimulatedLockin:
    """
    Models a lock-in (synchronous detection) measurement at a single frequency.

    Physical basis:
        - Emit sinusoid at f, integrate microphone signal for time T
        - Cross-correlate with reference → extract Re{H} and Im{H}
        - Noise is incoherent with reference → averages down as 1/√T

    Measurement model:
        H_measured(f) = H_true(f) + ε,   ε ~ CN(0,  N₀(f) / T)

    N₀(f):  noise power spectral density — HVAC model, stronger at low f.
    """

    def __init__(self, room, base_noise_psd=3e-4):
        self.room     = room
        self.base_psd = base_noise_psd

    def noise_psd(self, f):
        """Pink-ish HVAC noise: N₀(f) = base × (1 + 400/f)."""
        return self.base_psd * (1.0 + 400.0 / f)

    def measure(self, f, T=2.0):
        """Return noisy complex measurement of H(f)."""
        H   = self.room.at(f)
        var = self.noise_psd(f) / T
        eps = np.sqrt(var / 2) * (np.random.randn() + 1j * np.random.randn())
        return H + eps

    def obs_noise_var(self, f, T=2.0):
        """Observation noise variance used in the GP likelihood."""
        return self.noise_psd(f) / T


# ═══════════════════════════════════════════════════════════════════════════════
# 2b. REAL LOCK-IN MEASUREMENT  (microphone + speaker via sounddevice)
# ═══════════════════════════════════════════════════════════════════════════════

class RealLockin:
    """
    Real lock-in measurement using your microphone and speaker.

    At each step:
      1. Emit a pure sinusoid at frequency f through the speaker for T seconds
      2. Simultaneously record the microphone
      3. Apply synchronous detection (multiply by sin and cos reference,
         then average) — this rejects all noise incoherent with the tone
      4. Return complex H(f) = in-phase + j·quadrature

    The drive amplitude is kept low (DRIVE_AMP = 0.05 by default) so the
    tone is imperceptible under typical room noise.  Lock-in averaging
    pulls the signal out of the noise floor regardless.

    Call calibrate_noise() once at startup to measure the actual noise
    floor — this feeds into the GP likelihood for accurate uncertainty.
    """

    def __init__(self, fs=44100, drive_amplitude=0.05,
                 input_device=None, output_device=None):
        if not AUDIO_AVAILABLE:
            raise RuntimeError("sounddevice not installed.  Run: pip3 install sounddevice")
        self.fs    = fs
        self.A     = drive_amplitude
        self.in_d  = input_device
        self.out_d = output_device
        self._noise_psd_estimate = 1e-6   # default until calibrate_noise() is called

    def calibrate_noise(self, T_cal=4.0):
        """
        Record T_cal seconds of silence (speaker off) to estimate the
        ambient noise power spectral density N₀(f).
        This is used by the GP to set realistic observation uncertainties.
        Call this once before starting the inference loop.
        """
        print(f'\nCalibrating noise floor — {T_cal:.0f}s silent recording.')
        print('Please stay quiet and keep the room as still as possible...')
        N   = int(T_cal * self.fs)
        rec = sd.rec(N, samplerate=self.fs, channels=1, dtype='float32',
                     device=self.in_d)
        sd.wait()
        x = rec[:, 0].astype(float)

        # Estimate broadband PSD via Welch method
        _, psd = welch(x, fs=self.fs, nperseg=min(self.fs, len(x)//4))
        self._noise_psd_estimate = float(np.mean(psd))
        print(f'Noise floor: {10*np.log10(self._noise_psd_estimate):.1f} dBFS/Hz  '
              f'(broadband average)\n')

    def measure(self, f, T=3.0):
        """
        Emit sinusoid at f Hz for T seconds, return complex H(f).

        Lock-in detection math:
            Play:    s(t) = A · sin(2πft)
            Record:  x(t) = |H|·A · sin(2πft + φ) + noise(t)
            Extract: I = 2·mean(x·sin(2πft)) / A  →  |H|·cos(φ)
                     Q = 2·mean(x·cos(2πft)) / A  →  |H|·sin(φ)
                     H = I + jQ = |H|·e^{jφ}
        """
        N = int(T * self.fs)
        t = np.arange(N) / self.fs

        # Output signal: mono sinusoid
        out = (self.A * np.sin(2 * np.pi * f * t)).astype(np.float32)

        # Play and record simultaneously
        rec = sd.playrec(out[:, np.newaxis], samplerate=self.fs,
                         channels=1, dtype='float32',
                         input_device=self.in_d,
                         output_device=self.out_d)
        sd.wait()
        x = rec[:, 0].astype(float)

        # Trim first and last 10% to avoid speaker/mic transients
        trim = max(int(0.10 * N), 1)
        x = x[trim: N - trim]
        t = t[trim: N - trim]

        # Synchronous detection
        I = 2.0 * np.mean(x * np.sin(2 * np.pi * f * t)) / self.A
        Q = 2.0 * np.mean(x * np.cos(2 * np.pi * f * t)) / self.A

        return I + 1j * Q

    def obs_noise_var(self, f, T=3.0):
        """Observation noise variance for GP likelihood: N₀/T."""
        return self._noise_psd_estimate / T


def list_devices():
    """Print all available audio devices with their index numbers."""
    if not AUDIO_AVAILABLE:
        print("sounddevice not installed.  Run: pip3 install sounddevice")
        return
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    print(f"\nDefault input:  {sd.default.device[0]}")
    print(f"Default output: {sd.default.device[1]}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  GAUSSIAN PROCESS  (the generative model)
# ═══════════════════════════════════════════════════════════════════════════════

def _k52(f1, f2, ls, sv):
    """Matérn-5/2 covariance kernel k(f1, f2)."""
    r = np.abs(f1[:, None] - f2[None, :]) / ls
    return sv * (1.0 + np.sqrt(5)*r + 5.0*r**2/3.0) * np.exp(-np.sqrt(5)*r)


class GP:
    """
    Gaussian Process over the complex transfer function H(f).

    Real and imaginary parts are modelled as independent GPs sharing
    a Matérn-5/2 kernel — a good default for physical functions that
    are smooth but can have sharp (resonant) features.

    Prior:      H(f) ~ GP(0, k(f, f'))
    Posterior:  updated analytically (GP regression) after each measurement.

    Key output: posterior std σ(f) — drives the agent's action selection.
    """

    def __init__(self, f_grid, length_scale=45.0, signal_var=0.5, jitter=1e-6):
        self.f   = f_grid
        self.ls  = length_scale
        self.sv  = signal_var
        self.jit = jitter

        # Prior variance is signal_var at every point (k(f,f) = sv for Matérn)
        self._prior_var = signal_var * np.ones(len(f_grid))

        # Accumulated observations
        self._fo  = []   # observed frequencies
        self._yr  = []   # Re{H} observations
        self._yi  = []   # Im{H} observations
        self._nv  = []   # per-observation noise variances

        # Posterior — initialised to prior
        self.mean_re = np.zeros(len(f_grid))
        self.mean_im = np.zeros(len(f_grid))
        self.var     = self._prior_var.copy()

    def update(self, f_new, y_new, obs_noise_var):
        """Incorporate one new observation; recompute posterior analytically."""
        self._fo.append(f_new)
        self._yr.append(y_new.real)
        self._yi.append(y_new.imag)
        self._nv.append(obs_noise_var)

        fo = np.array(self._fo)
        yr = np.array(self._yr)
        yi = np.array(self._yi)

        # Kernel matrices
        Koo  = _k52(fo, fo, self.ls, self.sv)
        Koo += np.diag(self._nv) + self.jit * np.eye(len(fo))   # noise + jitter
        Kgo  = _k52(self.f, fo, self.ls, self.sv)                # (n_grid × n_obs)

        # Cholesky solve — numerically stable inversion of Koo
        L        = np.linalg.cholesky(Koo)
        alpha_r  = np.linalg.solve(L.T, np.linalg.solve(L, yr))
        alpha_i  = np.linalg.solve(L.T, np.linalg.solve(L, yi))

        # Posterior mean
        self.mean_re = Kgo @ alpha_r
        self.mean_im = Kgo @ alpha_i

        # Posterior variance: var(f) = k(f,f) − k(f,X)ᵀ Koo⁻¹ k(X,f)
        #   Using Cholesky:  V = L⁻¹ Kgoᵀ  (shape: n_obs × n_grid)
        #   var reduction  = sum_j V[j,:]²
        V        = np.linalg.solve(L, Kgo.T)
        self.var = np.maximum(self._prior_var - np.sum(V**2, axis=0), 0.0)

    @property
    def mean(self):
        return self.mean_re + 1j * self.mean_im

    @property
    def std(self):
        return np.sqrt(self.var)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  ACTIVE AGENT  (action selection policy)
# ═══════════════════════════════════════════════════════════════════════════════

class ActiveAgent:
    """
    Selects the next probe frequency to maximise information gain.

    Policy:  f* = argmax_f  σ²(f)

    This is the epistemic-value-maximising policy under a Gaussian model —
    equivalent to minimising expected free energy when the sole goal is to
    characterise H(f) (no pragmatic term).

    An exclusion zone around recently measured frequencies prevents the
    agent from redundantly re-probing where the posterior is already tight.
    """

    def __init__(self, f_grid, excl_hz=20.0):
        self.f_grid  = f_grid
        self.excl_hz = excl_hz
        self.history = []   # list of (f, y) tuples

    def select(self, gp):
        """Return f* = argmax σ²(f), excluding recent measurement sites."""
        var = gp.var.copy()
        for f_prev, _ in self.history[-8:]:
            var[np.abs(self.f_grid - f_prev) < self.excl_hz] = 0.0
        return self.f_grid[np.argmax(var)]

    def record(self, f, y):
        self.history.append((f, y))


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  FIGURE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_panels(ax_a, ax_p, ax_v, f_grid, gp, agent, f_last, room=None):
    """Clear and redraw all three panels.  room=None in real-audio mode."""

    f_meas = np.array([h[0] for h in agent.history])
    y_meas = np.array([h[1] for h in agent.history])
    mean   = gp.mean
    std    = gp.std

    ax_a.cla();  ax_p.cla();  ax_v.cla()

    # ── Amplitude ─────────────────────────────────────────────────────────────
    if room is not None:
        ax_a.plot(f_grid, np.abs(room.H),
                  color='#bbbbbb', lw=1.5, label='True |H(f)|')
    ax_a.fill_between(f_grid,
                      np.maximum(np.abs(mean) - 2*std, 0),
                      np.abs(mean) + 2*std,
                      color='steelblue', alpha=0.20, label='±2σ')
    ax_a.plot(f_grid, np.abs(mean),
              color='steelblue', lw=1.5, label='GP mean')
    if len(f_meas):
        ax_a.scatter(f_meas, np.abs(y_meas),
                     c='crimson', s=22, zorder=5,
                     label=f'Measurements  (n={len(f_meas)})')
    ax_a.axvline(f_last, color='limegreen', lw=1.5, ls='--',
                 alpha=0.9, label='Last probe')
    ax_a.set_ylabel('|H(f)|')
    ax_a.set_xlim(f_grid[0], f_grid[-1])
    ax_a.grid(True, alpha=0.25)
    ax_a.legend(fontsize=7.5, ncol=3, loc='upper right')

    # ── Phase ─────────────────────────────────────────────────────────────────
    if room is not None:
        ax_p.plot(f_grid, np.angle(room.H),
                  color='#bbbbbb', lw=1.5, label='True ∠H(f)')
    ax_p.plot(f_grid, np.angle(mean),
              color='steelblue', lw=1.5, label='GP mean')
    if len(f_meas):
        ax_p.scatter(f_meas, np.angle(y_meas),
                     c='crimson', s=22, zorder=5)
    ax_p.axvline(f_last, color='limegreen', lw=1.5, ls='--', alpha=0.9)
    ax_p.set_ylabel('∠H(f)  (rad)')
    ax_p.set_xlim(f_grid[0], f_grid[-1])
    ax_p.set_ylim(-np.pi - 0.3, np.pi + 0.3)
    ax_p.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax_p.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax_p.grid(True, alpha=0.25)
    ax_p.legend(fontsize=7.5, loc='upper right')

    # ── Posterior std ─────────────────────────────────────────────────────────
    ax_v.fill_between(f_grid, 0, std, color='tomato', alpha=0.30)
    ax_v.plot(f_grid, std, color='tomato', lw=1.5, label='σ(f)')
    ax_v.axvline(f_last, color='limegreen', lw=1.5, ls='--', alpha=0.9,
                 label='Last probe  (was argmax σ)')
    ax_v.set_ylim(bottom=0)
    ax_v.set_xlim(f_grid[0], f_grid[-1])
    ax_v.set_ylabel('Posterior std  σ(f)')
    ax_v.set_xlabel('Frequency  (Hz)', fontsize=9)
    ax_v.grid(True, alpha=0.25)
    ax_v.legend(fontsize=7.5, loc='upper right')


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN SIMULATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run(lockin, room=None, n_steps=60, T=3.0, step_pause=0.15):
    """
    Run the active inference loop with a live updating plot.

    Args:
        lockin:     SimulatedLockin or RealLockin instance
        room:       SyntheticRoom (shows ground truth in grey) or None
        n_steps:    number of probe frequencies
        T:          integration time per frequency (seconds)
        step_pause: pause between steps in sim mode
    """
    f_grid = np.linspace(F_MIN, F_MAX, 600)
    gp     = GP(f_grid)
    agent  = ActiveAgent(f_grid)

    plt.ion()
    fig = plt.figure(figsize=(13, 9), facecolor='#f9f9f9')
    gs  = gridspec.GridSpec(3, 1, hspace=0.52, top=0.91, bottom=0.07,
                            left=0.09, right=0.97)
    ax_a = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1])
    ax_v = fig.add_subplot(gs[2])

    mode_label = 'Simulation' if room is not None else 'Real Audio'
    print(f'\n{"─"*52}')
    print(f'  Active Acoustic Inference  [{mode_label}]')
    print(f'  {n_steps} steps  |  {T:.1f}s integration per step')
    print(f'{"─"*52}')
    print(f'{"Step":>5}  {"Probe (Hz)":>10}  {"Max σ":>10}  {"|H| measured":>14}')
    print('─' * 45)

    for step in range(1, n_steps + 1):

        # ── Active inference loop ─────────────────────────────────────────────
        f_next = agent.select(gp)           # ACTION:      argmax σ²(f)
        y      = lockin.measure(f_next, T)  # OBSERVATION: lock-in measurement
        nv     = lockin.obs_noise_var(f_next, T)
        gp.update(f_next, y, nv)            # PERCEPTION:  GP posterior update
        agent.record(f_next, y)

        print(f'{step:>5}  {f_next:>10.1f}  {np.max(gp.std):>10.5f}  {abs(y):>14.6f}')

        # ── Update live plot ──────────────────────────────────────────────────
        _draw_panels(ax_a, ax_p, ax_v, f_grid, gp, agent, f_next, room=room)
        fig.suptitle(
            f'Active Acoustic Inference  [{mode_label}]  |  Step {step}/{n_steps}'
            f'  |  Last probe: {f_next:.0f} Hz'
            f'  |  Max σ: {np.max(gp.std):.4f}',
            fontsize=11, fontweight='bold'
        )
        fig.canvas.draw()
        plt.pause(step_pause if room is not None else 0.01)

    print('─' * 45)
    print('Done — close the plot window to exit.')
    plt.ioff()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    if MODE == 'sim':
        # ── Simulation mode — no hardware needed ─────────────────────────────
        f_grid = np.linspace(F_MIN, F_MAX, 600)
        room   = SyntheticRoom(f_grid)
        lockin = SimulatedLockin(room)
        run(lockin, room=room, n_steps=N_STEPS, T=T_PER_STEP, step_pause=0.15)

    elif MODE == 'real':
        # ── Real audio mode ───────────────────────────────────────────────────
        # Uncomment the next line to see your audio device list and indices:
        # list_devices()

        lockin = RealLockin(fs=SAMPLE_RATE,
                            drive_amplitude=DRIVE_AMP,
                            input_device=INPUT_DEVICE,
                            output_device=OUTPUT_DEVICE)
        lockin.calibrate_noise(T_cal=4.0)
        run(lockin, room=None, n_steps=N_STEPS, T=T_PER_STEP)

    else:
        print(f"Unknown MODE '{MODE}'. Set MODE = 'sim' or MODE = 'real'.")