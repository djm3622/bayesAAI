#!/usr/bin/env python3
"""
Active Acoustic Inference Demo
================================
Demonstrates active inference for room acoustic characterization.

A Gaussian Process agent adaptively selects probe frequencies,
using simulated lock-in (synchronous) detection to measure the
complex room transfer function H(f) with sub-noise-floor sensitivity.

Three panels show, updating live after every single measurement:
  Top:    Amplitude |H(f)| — GP mean ± 2σ vs. ground truth
  Middle: Phase ∠H(f)     — GP mean vs. ground truth
  Bottom: Posterior std σ(f) — shows where uncertainty remains high

The agent always probes the frequency of maximum posterior variance σ²(f).
This is the epistemic-value-maximising policy under a Gaussian generative
model — equivalent to minimising expected free energy (active inference).

Running this file:
    - On a laptop with a display, it opens a live matplotlib window that
      redraws after every measurement (true real-time animation).
    - In a headless environment (no display, e.g. CI/sandbox), it instead
      renders the same frame-by-frame evolution to an MP4 so you can still
      watch the inference unfold.
"""

import os
import sys

import matplotlib

# Headless environments (no display) can't open an interactive window, so we
# fall back to the non-interactive Agg backend and render a video instead.
HEADLESS = sys.platform.startswith('linux') and not os.environ.get('DISPLAY')
if HEADLESS:
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)


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
# 5.  LIVE VISUALIZER  (persistent artists, updated in place every step)
# ═══════════════════════════════════════════════════════════════════════════════

class LiveVisualizer:
    """
    Three-panel figure whose artists are created ONCE and then updated in
    place after every measurement — this is what makes real-time / animated
    playback fast (no re-creating axes, legends, etc. every frame).
    """

    def __init__(self, f_grid, room):
        self.f_grid = f_grid
        self.room   = room

        self.fig = plt.figure(figsize=(13, 9), facecolor='#f9f9f9')
        self.title = self.fig.suptitle('', fontsize=11, fontweight='bold')

        gs = gridspec.GridSpec(3, 1, hspace=0.52, top=0.91, bottom=0.07,
                                left=0.09, right=0.97)
        self.ax_a = self.fig.add_subplot(gs[0])
        self.ax_p = self.fig.add_subplot(gs[1])
        self.ax_v = self.fig.add_subplot(gs[2])

        # ── Static "ground truth" curves (drawn once) ────────────────────────
        self.ax_a.plot(f_grid, np.abs(room.H), color='#bbbbbb', lw=1.5,
                        label='True |H(f)|')
        self.ax_p.plot(f_grid, np.angle(room.H), color='#bbbbbb', lw=1.5,
                        label='True ∠H(f)')

        # ── Dynamic artists — amplitude panel ────────────────────────────────
        self.mean_a_line, = self.ax_a.plot([], [], color='steelblue', lw=1.5,
                                            label='GP mean')
        self.fill_a  = self.ax_a.fill_between(f_grid, 0, 0, color='steelblue',
                                               alpha=0.20, label='±2σ')
        self.scat_a  = self.ax_a.scatter([], [], c='crimson', s=22, zorder=5,
                                          label='Measurements  (n=0)')
        self.vline_a = self.ax_a.axvline(f_grid[0], color='limegreen', lw=1.5,
                                          ls='--', alpha=0.9, label='Last probe')
        self.ax_a.set_ylabel('|H(f)|')
        self.ax_a.set_ylim(0, max(1.15, np.max(np.abs(room.H)) * 1.15))
        self.leg_a = self.ax_a.legend(fontsize=7.5, ncol=3, loc='upper right')

        # ── Dynamic artists — phase panel ────────────────────────────────────
        self.mean_p_line, = self.ax_p.plot([], [], color='steelblue', lw=1.5,
                                            label='GP mean')
        self.scat_p  = self.ax_p.scatter([], [], c='crimson', s=22, zorder=5)
        self.vline_p = self.ax_p.axvline(f_grid[0], color='limegreen', lw=1.5,
                                          ls='--', alpha=0.9)
        self.ax_p.set_ylabel('∠H(f)  (rad)')
        self.ax_p.set_ylim(-np.pi - 0.3, np.pi + 0.3)
        self.ax_p.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        self.ax_p.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        self.ax_p.legend(fontsize=7.5, loc='upper right')

        # ── Dynamic artists — posterior std panel ────────────────────────────
        self.std_line, = self.ax_v.plot([], [], color='tomato', lw=1.5,
                                         label='σ(f)')
        self.fill_v  = self.ax_v.fill_between(f_grid, 0, 0, color='tomato',
                                               alpha=0.30)
        self.vline_v = self.ax_v.axvline(f_grid[0], color='limegreen', lw=1.5,
                                          ls='--', alpha=0.9,
                                          label='Last probe  (was argmax σ)')
        self.ax_v.set_ylabel('Posterior std  σ(f)')
        self.ax_v.set_ylim(0, 0.75)
        self.ax_v.legend(fontsize=7.5, loc='upper right')

        for ax in [self.ax_a, self.ax_p, self.ax_v]:
            ax.set_xlim(50, 2000)
            ax.set_xlabel('Frequency  (Hz)', fontsize=9)
            ax.grid(True, alpha=0.25)

        self.fig.canvas.draw()

    def update(self, step, n_steps, f_last, gp, agent):
        """Update all artists in place to reflect the current GP posterior."""
        mean = gp.mean
        std  = gp.std
        f_meas = np.array([h[0] for h in agent.history])
        y_meas = np.array([h[1] for h in agent.history])

        # Lines
        self.mean_a_line.set_data(self.f_grid, np.abs(mean))
        self.mean_p_line.set_data(self.f_grid, np.angle(mean))
        self.std_line.set_data(self.f_grid, std)

        # fill_between objects have no set_data — remove & redraw the polygon
        self.fill_a.remove()
        self.fill_a = self.ax_a.fill_between(
            self.f_grid, np.maximum(np.abs(mean) - 2*std, 0), np.abs(mean) + 2*std,
            color='steelblue', alpha=0.20)
        self.fill_v.remove()
        self.fill_v = self.ax_v.fill_between(
            self.f_grid, 0, std, color='tomato', alpha=0.30)

        # Measurement scatter
        self.scat_a.set_offsets(np.column_stack([f_meas, np.abs(y_meas)])
                                 if len(f_meas) else np.empty((0, 2)))
        self.scat_p.set_offsets(np.column_stack([f_meas, np.angle(y_meas)])
                                 if len(f_meas) else np.empty((0, 2)))

        # Last-probe marker
        for vline in (self.vline_a, self.vline_p, self.vline_v):
            vline.set_xdata([f_last, f_last])

        # Rescale std axis so small late-stage values stay visible
        self.ax_v.set_ylim(0, max(0.05, np.max(std) * 1.2))

        self.title.set_text(
            f'Active Acoustic Inference  |  Step {step}/{n_steps}'
            f'  |  Last probe: {f_last:.0f} Hz'
            f'  |  Max σ remaining: {np.max(std):.4f}'
            f'  |  n obs: {len(f_meas)}'
        )

    def savefig(self, path, dpi=130):
        self.fig.savefig(path, dpi=dpi, bbox_inches='tight')


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  SHARED SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def _setup():
    """Build the room, lock-in, GP, agent, and visualizer for a fresh run."""
    f_grid = np.linspace(50, 2000, 600)
    room   = SyntheticRoom(f_grid)
    lockin = SimulatedLockin(room)
    gp     = GP(f_grid)
    agent  = ActiveAgent(f_grid)
    viz    = LiveVisualizer(f_grid, room)
    return f_grid, room, lockin, gp, agent, viz


def _step(lockin, gp, agent, T):
    """One active-inference cycle: select → measure → update. Returns f_next."""
    f_next = agent.select(gp)           # ACTION:      argmax σ²(f)
    y      = lockin.measure(f_next, T)  # OBSERVATION: lock-in measurement
    nv     = lockin.obs_noise_var(f_next, T)
    gp.update(f_next, y, nv)            # PERCEPTION:  GP posterior update
    agent.record(f_next, y)
    return f_next


# ═══════════════════════════════════════════════════════════════════════════════
# 7a.  LIVE INTERACTIVE MODE  (real display, updates after every measurement)
# ═══════════════════════════════════════════════════════════════════════════════

def run_live(n_steps=80, T=2.0, pause=0.05):
    """
    Open an interactive window and update it after EVERY measurement.
    Requires a real display (run this locally on your laptop, not headless).
    """
    plt.ion()
    f_grid, room, lockin, gp, agent, viz = _setup()
    plt.show(block=False)

    print(f'{"Step":>5}  {"Probe (Hz)":>10}  {"Max σ":>10}')
    print('─' * 30)
    for step in range(1, n_steps + 1):
        f_next = _step(lockin, gp, agent, T)
        viz.update(step, n_steps, f_next, gp, agent)
        viz.fig.canvas.draw_idle()
        viz.fig.canvas.flush_events()
        plt.pause(pause)
        print(f'{step:>5}  {f_next:>10.1f}  {np.max(gp.std):>10.5f}')
    print('─' * 30)
    print('Done. Close the window to exit.')

    plt.ioff()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 7b.  ANIMATION-EXPORT MODE  (headless-safe — renders the evolution to video)
# ═══════════════════════════════════════════════════════════════════════════════

def run_animation(n_steps=80, T=2.0, out_path='inference_evolution.mp4', fps=8):
    """
    Render the full step-by-step evolution to a video file, one frame per
    measurement. Works without a display, so this is what runs in headless /
    CI / sandboxed environments where run_live() can't open a window.
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    f_grid, room, lockin, gp, agent, viz = _setup()

    def frame(step):
        f_next = _step(lockin, gp, agent, T)
        viz.update(step, n_steps, f_next, gp, agent)
        return []

    # init_func=lambda: [] prevents FuncAnimation's implicit extra call to
    # frame() during setup, which would otherwise double-count step 1.
    anim   = FuncAnimation(viz.fig, frame, frames=range(1, n_steps + 1),
                            init_func=lambda: [], blit=False, repeat=False)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(viz.fig)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))
    if HEADLESS:
        path = run_animation(n_steps=180, T=2.0,
                              out_path=os.path.join(out_dir, 'inference_evolution.mp4'))
        print(f'No display detected — saved real-time evolution video to {path}')
    else:
        run_live(n_steps=180, T=2.0, pause=0.05)
