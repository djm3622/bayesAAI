#!/usr/bin/env python3
import matplotlib
matplotlib.use('TkAgg')   # reliable live-update backend on macOS
"""
Active Acoustic Inference — Two-Channel Multi-Output GP
=======================================================
Stage 2: one speaker, two microphones, simulation only.

Room model
----------
Physically derived rectangular room (Lx × Ly × Lz).  Each speaker-mic
pair has its own transfer function H_i(f), all sharing the same room poles
but with different residues determined by source/mic positions.

Multi-output GP
---------------
A single GP models both channels jointly using a coregionalized kernel:

    k((f,i), (f',j)) = k_freq(f,f') · B[i,j]

where k_freq is the non-stationary Matérn-5/2 and B is a 2×2
coregionalization matrix learned from data via a rank-1 factorization:

    B = w·wᵀ + diag(κ)

  w  — shared component (same-room correlation between channels)
  κ  — channel-specific residual variance (independent component)

This encodes the physical constraint that both mics are in the same room:
a measurement in channel 0 informs the posterior in channel 1, weighted
by the learned inter-channel correlation.

Active sensing policy
---------------------
The agent selects the (frequency, channel) pair that maximises the joint
posterior variance — the sum of posterior std across both channels.  This
naturally directs measurements to wherever the joint uncertainty is highest,
exploiting cross-channel information sharing automatically.

Note on mode count
------------------
Transfer functions are precomputed on a dense grid at initialisation using
chunked NumPy, then interpolated for each lock-in step.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Room geometry and acoustics
ROOM = dict(
    Lx    = 5.2,    # m — length
    Ly    = 4.1,    # m — width
    Lz    = 2.9,    # m — height
    alpha = 0.15,   # mean absorption coefficient (0 = rigid, 1 = anechoic)
    c     = 343.0,  # m/s
)

# Source and microphone positions [x, y, z] metres
#   Avoid placing source/mic exactly on a wall or room centre —
#   that zeros out certain mode shapes and makes them invisible.
SOURCE_POS = np.array([1.2, 0.9, 1.5])
MIC_POSITIONS = [
    np.array([3.8, 2.7, 1.1]),   # mic 0 — original position
    np.array([3.8, 2.6, 1.1]),   # mic 1 — different wall proximity / height
]

# Frequency axis
F_MIN  = 20.0    # Hz
F_MAX  = 500.0    # Hz
N_GRID = 1200      # display / GP grid points

# Active inference
N_STEPS    = 300   # probe steps
T_PER_STEP = 0.5   # lock-in integration time per step (s)

# Noise (simulated HVAC-like pink floor)
NOISE_FLOOR = 3e-5  # base PSD amplitude (Pa² / Hz) — reduce for cleaner demo

# Non-stationary GP kernel parameters
SCHROEDER_HZ = 350.0   # estimated Schroeder frequency — tune to room
LS_BELOW     =  10.0   # Hz — tight length scale below Schroeder
LS_ABOVE     =  45.0   # Hz — smooth length scale above Schroeder


# ═══════════════════════════════════════════════════════════════════════════
# 1.  RECTANGULAR ROOM MODEL
# ═══════════════════════════════════════════════════════════════════════════

class RectangularRoom:
    """
    Physically derived modal model of a rectangular room.

    Attributes
    ----------
    f_modes : ndarray  — modal frequencies (Hz), sorted ascending
    Q_modes : ndarray  — modal Q values (frequency-dependent via Sabine)
    T60     : float    — Sabine reverberation time (s)
    n_modes : int      — total number of modes enumerated

    Key methods
    -----------
    residues(src, mic)          → ndarray of Aₙ for one speaker-mic pair
    transfer_function(f, s, m)  → H(f) array (modal sum, exact but slower)
    H_grid                      → precomputed H on the display grid (fast)
    at_grid(f, src, mic)        → interpolated scalar H(f) for lock-in use
    schroeder_freq()            → estimated Schroeder frequency (Hz)
    """

    def __init__(self, Lx, Ly, Lz, alpha, c=343.0):
        self.Lx = Lx;  self.Ly = Ly;  self.Lz = Lz
        self.alpha = alpha;  self.c = c
        self.V = Lx * Ly * Lz
        self.S = 2.0 * (Lx*Ly + Ly*Lz + Lx*Lz)
        self.T60 = 0.161 * self.V / (alpha * self.S)

        # Q = π·f·T60 / (3·ln10)  — derived from modal decay rate
        self._q_factor = np.pi * self.T60 / (3.0 * np.log(10.0))

        self._enumerate_modes(f_max=F_MAX * 1.05)

        print(f"\n── Rectangular Room ─────────────────────────────────────")
        print(f"   Dimensions : {Lx:.2f} × {Ly:.2f} × {Lz:.2f} m")
        print(f"   Volume     : {self.V:.1f} m³     Surface : {self.S:.1f} m²")
        print(f"   Absorption : α = {alpha:.2f}")
        print(f"   T₆₀        : {self.T60:.2f} s  (Sabine)")
        print(f"   Modes ≤ {F_MAX:.0f} Hz : {self.n_modes:,}")
        print(f"   Schroeder f : {self.schroeder_freq():.0f} Hz  (estimated)")
        print(f"─────────────────────────────────────────────────────────\n")

    # ------------------------------------------------------------------
    def _enumerate_modes(self, f_max):
        """Enumerate all (l, m, n) modes with f_lmn ≤ f_max."""
        c  = self.c
        Lx, Ly, Lz = self.Lx, self.Ly, self.Lz

        l_max = int(2 * f_max * Lx / c) + 2
        m_max = int(2 * f_max * Ly / c) + 2
        n_max = int(2 * f_max * Lz / c) + 2

        # Build index arrays efficiently using broadcasting
        l = np.arange(0, l_max + 1)
        m = np.arange(0, m_max + 1)
        n = np.arange(0, n_max + 1)

        ll, mm, nn = np.meshgrid(l, m, n, indexing='ij')
        ll = ll.ravel();  mm = mm.ravel();  nn = nn.ravel()

        # Remove DC mode (0,0,0)
        mask_dc  = (ll == 0) & (mm == 0) & (nn == 0)
        ll = ll[~mask_dc];  mm = mm[~mask_dc];  nn = nn[~mask_dc]

        # Modal frequencies
        f_vals = (c / 2.0) * np.sqrt((ll/Lx)**2 + (mm/Ly)**2 + (nn/Lz)**2)

        # Keep only modes within range
        keep = f_vals <= f_max
        ll = ll[keep];  mm = mm[keep];  nn = nn[keep];  f_vals = f_vals[keep]

        # Sort by frequency
        order    = np.argsort(f_vals)
        self._l  = ll[order].astype(np.int32)
        self._m  = mm[order].astype(np.int32)
        self._n  = nn[order].astype(np.int32)

        self.f_modes = f_vals[order]
        self.Q_modes = self._q_factor * self.f_modes

        # Eigenfunction normalisation weights εₗεₘεₙ / V
        el = np.where(self._l == 0, 1.0, 2.0)
        em = np.where(self._m == 0, 1.0, 2.0)
        en = np.where(self._n == 0, 1.0, 2.0)
        self._norm_weights = el * em * en / self.V

        self.n_modes = len(self.f_modes)

    # ------------------------------------------------------------------
    def schroeder_freq(self):
        """Schroeder frequency: f_S ≈ 2000 · sqrt(T60/V)  (Hz)."""
        return 2000.0 * np.sqrt(self.T60 / self.V)

    # ------------------------------------------------------------------
    def mode_shape(self, pos):
        """
        Evaluate all mode shapes at position pos = [x, y, z].
        Returns ndarray of shape (n_modes,).
        """
        x, y, z = pos
        return (np.cos(self._l * np.pi * x / self.Lx) *
                np.cos(self._m * np.pi * y / self.Ly) *
                np.cos(self._n * np.pi * z / self.Lz))

    # ------------------------------------------------------------------
    def residues(self, src_pos, mic_pos):
        """
        Residue amplitudes Aₙ for a given source-mic pair.
            Aₙ = norm_weight_n · ψₙ(src) · ψₙ(mic)
        Returns ndarray of shape (n_modes,).
        """
        psi_s = self.mode_shape(src_pos)
        psi_m = self.mode_shape(mic_pos)
        return self._norm_weights * psi_s * psi_m

    # ------------------------------------------------------------------
    def transfer_function(self, f_array, src_pos, mic_pos,
                          chunk=1000, normalise=True):
        """
        Compute H(f) over f_array using the full modal sum.

        Parameters
        ----------
        f_array   : 1-D frequency array (Hz)
        src_pos   : [x, y, z] source position
        mic_pos   : [x, y, z] microphone position
        chunk     : modes processed per batch (memory control)
        normalise : if True, scale so max|H| = 1

        Returns complex ndarray of same length as f_array.
        """
        f  = np.asarray(f_array, dtype=float)
        An = self.residues(src_pos, mic_pos)
        fn = self.f_modes
        Qn = self.Q_modes
        H  = np.zeros(len(f), dtype=complex)

        for i in range(0, self.n_modes, chunk):
            sl    = slice(i, i + chunk)
            fn_c  = fn[sl, np.newaxis]           # (chunk, 1)
            Qn_c  = Qn[sl, np.newaxis]
            An_c  = An[sl, np.newaxis]
            f_row = f[np.newaxis, :]             # (1, N_freq)
            H    += np.sum(An_c / (fn_c**2 - f_row**2
                           + 1j * f_row * fn_c / Qn_c), axis=0)

        if normalise:
            peak = np.max(np.abs(H))
            if peak > 0:
                H /= peak
        return H

    # ------------------------------------------------------------------
    def precompute_grid(self, f_grid, src_pos, mic_pos):
        """
        Precompute and store H(f) on a dense grid for fast interpolation.
        Call once per source-mic pair before the inference loop.
        """
        print(f"Precomputing H(f) on {len(f_grid)}-point grid "
              f"({self.n_modes:,} modes) … ", end='', flush=True)
        H = self.transfer_function(f_grid, src_pos, mic_pos, normalise=True)
        print("done.")
        self._H_grid_f    = f_grid
        self._H_grid_H    = H
        self._H_grid_norm = 1.0   # already normalised
        return H

    # ------------------------------------------------------------------
    def at_grid(self, f):
        """
        Interpolated H(f) from the precomputed grid — fast scalar lookup
        for use inside the lock-in measurement loop.
        """
        re = np.interp(f, self._H_grid_f, self._H_grid_H.real)
        im = np.interp(f, self._H_grid_f, self._H_grid_H.imag)
        return complex(re, im)


# ═══════════════════════════════════════════════════════════════════════
# 2.  SIMULATED LOCK-IN  (draws from RectangularRoom)
# ═══════════════════════════════════════════════════════════════════════

class RoomLockin:
    """
    Simulated lock-in for one speaker and one or more microphones.

    Each mic channel is an independent RoomLockin internally, but the
    multi-channel interface measure(f, channel, T) selects which channel
    to measure.  H_true is a list of precomputed transfer functions,
    one per mic.

    Noise model:  PSD(f) = noise_floor · (1 + 400/f)   [pink-ish / HVAC]
    Lock-in variance after integration time T:  σ² = PSD(f) / T
    """

    def __init__(self, room, src_pos, mic_positions,
                 f_grid, noise_floor=3e-5):
        self.room = room
        self.src  = src_pos
        self.mics = mic_positions
        self.nf   = noise_floor
        self.n_channels = len(mic_positions)

        # Precompute H on the display grid for each mic.
        # RectangularRoom.precompute_grid stores one grid internally so we
        # call transfer_function directly for channels beyond the first.
        print(f"Precomputing {self.n_channels} transfer functions...")
        self.H_true = []
        for i, mic in enumerate(mic_positions):
            print(f"  Channel {i} (mic at {mic}) ... ", end='', flush=True)
            H = room.transfer_function(f_grid, src_pos, mic, normalise=True)
            self.H_true.append(H)
            print("done.")

        # Store interpolation grids per channel
        self._f_grid = f_grid

    def noise_psd(self, f):
        return self.nf * (1.0 + 400.0 / f)

    def _H_at(self, f, channel):
        """Interpolated H(f) for a given channel."""
        re = np.interp(f, self._f_grid, self.H_true[channel].real)
        im = np.interp(f, self._f_grid, self.H_true[channel].imag)
        return complex(re, im)

    def measure(self, f, channel, T=0.5):
        """Return noisy complex H(f) for the given channel."""
        H_true = self._H_at(f, channel)
        var    = self.noise_psd(f) / T
        noise  = np.sqrt(var / 2) * (np.random.randn() + 1j * np.random.randn())
        return H_true + noise

    def obs_noise_var(self, f, T=0.5):
        return self.noise_psd(f) / T


# ═══════════════════════════════════════════════════════════════════════
# 3.  NON-STATIONARY GP
# ═══════════════════════════════════════════════════════════════════════

def _length_scale(f, schroeder=SCHROEDER_HZ,
                  ls_lo=LS_BELOW, ls_hi=LS_ABOVE):
    """
    Frequency-dependent GP length scale with sigmoid blend.

    Transitions smoothly from ls_lo (tight, resolves isolated modes)
    to ls_hi (smooth, modal overlap region) centred on the Schroeder
    frequency.  The width parameter (30 Hz) controls transition sharpness.
    """
    f = np.asarray(f, dtype=float)
    blend = 1.0 / (1.0 + np.exp(-(f - schroeder) / 30.0))
    return ls_lo + (ls_hi - ls_lo) * blend


def _k52_nonstat(f1, f2, signal_var):
    """
    Non-stationary Matérn-5/2 covariance kernel.

    Uses geometric mean of local length scales at f1 and f2, which
    preserves positive-definiteness of the covariance matrix.
    """
    ls1 = _length_scale(f1)
    ls2 = _length_scale(f2)
    ls  = np.sqrt(ls1[:, np.newaxis] * ls2[np.newaxis, :])
    r   = np.abs(f1[:, np.newaxis] - f2[np.newaxis, :]) / ls
    return signal_var * (1.0 + np.sqrt(5)*r + 5*r**2/3) * np.exp(-np.sqrt(5)*r)


class MultiOutputGP:
    """
    Two-channel multi-output Gaussian Process.

    Models H₀(f) and H₁(f) jointly using a coregionalized kernel:

        k((f,i),(f',j)) = k_freq(f,f') · B[i,j]

    where B is a 2×2 positive-definite coregionalization matrix with
    a rank-1 + diagonal structure:

        B = w·wᵀ + diag(κ)

    w  — shared weight vector (length 2): encodes same-room correlation
    κ  — per-channel residual variance:   encodes channel-specific variation

    This means each measurement updates both channels' posteriors,
    weighted by the learned inter-channel correlation.

    Parameters
    ----------
    f_grid     : frequency grid for display / posterior evaluation
    n_channels : number of microphone channels (2 for this stage)
    signal_var : prior variance per channel
    w_init     : initial shared weight (correlation strength)
    jitter     : Cholesky stabilisation

    Key outputs
    -----------
    mean(ch)  : complex posterior mean for channel ch
    std(ch)   : posterior std for channel ch
    var_joint : sum of posterior variance across all channels — drives agent
    """

    def __init__(self, f_grid, n_channels=2, signal_var=0.4,
                 w_init=0.5, jitter=1e-6):
        self.f          = f_grid
        self.nc         = n_channels
        self.sv         = signal_var
        self.jit        = jitter
        n               = len(f_grid)

        # Coregionalization: B = w·wᵀ + diag(κ)
        # w: shared weights, κ: residual per channel
        self.w  = np.full(n_channels, w_init)          # shared component
        self.kappa = np.full(n_channels, signal_var * (1 - w_init**2))

        # Per-channel prior variance
        self._prior_var = signal_var * np.ones(n)

        # Observation storage: lists indexed by measurement order
        self._f_obs  = []   # frequency of each observation
        self._ch_obs = []   # channel index of each observation
        self._yr     = []   # real part of observation
        self._yi     = []   # imag part of observation
        self._nv     = []   # observation noise variance

        # Posterior state per channel
        self.mean_re = [np.zeros(n) for _ in range(n_channels)]
        self.mean_im = [np.zeros(n) for _ in range(n_channels)]
        self.var     = [self._prior_var.copy() for _ in range(n_channels)]

    def _B(self):
        """2×2 coregionalization matrix B = w·wᵀ + diag(κ)."""
        return np.outer(self.w, self.w) + np.diag(self.kappa)

    def _k_joint(self, f1, ch1, f2, ch2):
        """
        Joint kernel k((f1,ch1),(f2,ch2)) = k_freq(f1,f2) · B[ch1,ch2].
        f1, f2 are 1-D arrays; ch1, ch2 are integer channel indices.
        Returns a (len(f1), len(f2)) matrix.
        """
        return _k52_nonstat(f1, f2, self.sv) * self._B()[ch1, ch2]

    def update(self, f_new, channel, y_new, obs_noise_var):
        """
        Add one measurement and recompute joint posterior for all channels.
        """
        self._f_obs.append(f_new)
        self._ch_obs.append(channel)
        self._yr.append(y_new.real)
        self._yi.append(y_new.imag)
        self._nv.append(obs_noise_var)

        fo  = np.array(self._f_obs)
        cho = np.array(self._ch_obs)
        yr  = np.array(self._yr)
        yi  = np.array(self._yi)
        N   = len(fo)

        # Build joint observation covariance Koo (N×N)
        B   = self._B()
        Koo = np.zeros((N, N))
        for a in range(N):
            for b in range(N):
                Koo[a, b] = (_k52_nonstat(fo[a:a+1], fo[b:b+1], self.sv)[0, 0]
                             * B[cho[a], cho[b]])
        Koo += np.diag(self._nv) + self.jit * np.eye(N)

        L       = np.linalg.cholesky(Koo)
        alpha_r = np.linalg.solve(L.T, np.linalg.solve(L, yr))
        alpha_i = np.linalg.solve(L.T, np.linalg.solve(L, yi))

        # Update posterior for each channel
        for ch in range(self.nc):
            # Cross-covariance between grid points (channel ch) and observations
            Kgo = np.zeros((len(self.f), N))
            for b in range(N):
                Kgo[:, b] = (_k52_nonstat(self.f, fo[b:b+1], self.sv)[:, 0]
                             * B[ch, cho[b]])

            self.mean_re[ch] = Kgo @ alpha_r
            self.mean_im[ch] = Kgo @ alpha_i
            V                = np.linalg.solve(L, Kgo.T)
            self.var[ch]     = np.maximum(
                self._prior_var - np.sum(V**2, axis=0), 0.0)

    def mean(self, ch):
        return self.mean_re[ch] + 1j * self.mean_im[ch]

    def std(self, ch):
        return np.sqrt(self.var[ch])

    @property
    def var_joint(self):
        """Sum of posterior variance across all channels — for agent policy."""
        return sum(self.var)


# ═══════════════════════════════════════════════════════════════════════
# 4.  ACTIVE AGENT  (multi-channel)
# ═══════════════════════════════════════════════════════════════════════

class ActiveAgent:
    """
    Selects the next (frequency, channel) pair to maximise joint
    posterior variance reduction across all channels.

    Policy:  (f*, ch*) = argmax_{f,ch}  Σ_ch  var_ch(f)

    The channel is chosen as the one with highest variance at f*.
    An exclusion zone around recently probed (f, ch) pairs prevents
    redundant re-measurement.
    """

    def __init__(self, f_grid, n_channels=2, excl_factor=1.2):
        self.f_grid      = f_grid
        self.nc          = n_channels
        self.excl_factor = excl_factor
        self.history     = []   # list of (f, channel, y)

    def select(self, mgp):
        """Return (f_next, channel_next)."""
        # Joint variance across channels
        var_joint = mgp.var_joint.copy()

        # Apply exclusion zones for recent probes
        for f_prev, ch_prev, _ in self.history[-10:]:
            excl = self.excl_factor * float(
                _length_scale(np.array([f_prev]))[0])
            var_joint[np.abs(self.f_grid - f_prev) < excl] *= 0.5

        f_next  = self.f_grid[np.argmax(var_joint)]

        # Pick the channel with highest individual variance at f_next
        idx     = np.argmin(np.abs(self.f_grid - f_next))
        ch_next = int(np.argmax([mgp.var[ch][idx] for ch in range(self.nc)]))

        return f_next, ch_next

    def record(self, f, channel, y):
        self.history.append((f, channel, y))


# ═══════════════════════════════════════════════════════════════════════
# 5.  VALIDATION PLOT  (static, shown before inference starts)
# ═══════════════════════════════════════════════════════════════════════

CH_COLORS = ['steelblue', 'crimson']   # one colour per channel

def plot_room_model(room, f_grid, lockin):
    """Show true |H(f)| for both channels with modal frequencies marked."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), facecolor='#f9f9f9',
                              sharex=True)
    fig.subplots_adjust(hspace=0.35, top=0.88, bottom=0.10,
                        left=0.08, right=0.97)
    fs_hz = room.schroeder_freq()
    fig.suptitle(
        f"Room Model — {room.Lx:.1f}×{room.Ly:.1f}×{room.Lz:.1f} m  |  "
        f"α={room.alpha:.2f}  T₆₀={room.T60:.2f}s  |  "
        f"Schroeder ≈{fs_hz:.0f} Hz  |  {lockin.n_channels} channels",
        fontsize=10, fontweight='bold'
    )
    ax_a, ax_p = axes
    modal_below = room.f_modes[(room.f_modes >= F_MIN) &
                               (room.f_modes <= min(fs_hz * 1.5, F_MAX))]
    for fm in modal_below:
        ax_a.axvline(fm, color='#cccccc', lw=0.5, alpha=0.6)
        ax_p.axvline(fm, color='#cccccc', lw=0.5, alpha=0.6)
    ax_a.axvline(fs_hz, color='darkorange', lw=1.5, ls='--', alpha=0.8)
    ax_p.axvline(fs_hz, color='darkorange', lw=1.5, ls='--', alpha=0.8)

    for ch in range(lockin.n_channels):
        H = lockin.H_true[ch]
        ax_a.plot(f_grid, np.abs(H),    color=CH_COLORS[ch], lw=1.2,
                  label=f'Ch {ch} |H(f)|')
        ax_p.plot(f_grid, np.angle(H),  color=CH_COLORS[ch], lw=1.2,
                  label=f'Ch {ch} ∠H(f)')

    ax_a.set_ylabel('|H(f)|');  ax_a.set_xlim(F_MIN, F_MAX)
    ax_a.grid(True, alpha=0.22);  ax_a.legend(fontsize=8, loc='upper right')
    ax_p.set_ylabel('∠H(f) (rad)');  ax_p.set_xlabel('Frequency (Hz)')
    ax_p.set_ylim(-np.pi-0.3, np.pi+0.3)
    ax_p.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax_p.set_yticklabels(['-π','-π/2','0','π/2','π'])
    ax_p.grid(True, alpha=0.22);  ax_p.legend(fontsize=8, loc='upper right')
    plt.pause(0.1)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 6.  LIVE INFERENCE PLOT  (two-channel)
# ═══════════════════════════════════════════════════════════════════════

def draw_inference(axes, f_grid, mgp, agent, f_last, ch_last,
                   lockin, schroeder):
    """
    Four-panel live plot:
      Row 0: amplitude  |H(f)| — both channels
      Row 1: phase      ∠H(f) — both channels
      Row 2: posterior std σ(f) — both channels + joint
      Row 3: measurement allocation histogram per channel
    """
    for ax in axes:
        ax.cla()

    ax_a, ax_p, ax_v, ax_h = axes

    history = agent.history
    schroeder_kw = dict(color='darkorange', lw=1.0, ls=':', alpha=0.6)

    # ── Amplitude & Phase ──────────────────────────────────────────
    for ch in range(mgp.nc):
        col   = CH_COLORS[ch]
        H_tr  = lockin.H_true[ch]
        mean  = mgp.mean(ch)
        std   = mgp.std(ch)

        # amplitude
        ax_a.plot(f_grid, np.abs(H_tr), color='#cccccc', lw=1.0)
        ax_a.fill_between(f_grid,
                          np.maximum(np.abs(mean) - 2*std, 0),
                          np.abs(mean) + 2*std,
                          color=col, alpha=0.12)
        ax_a.plot(f_grid, np.abs(mean), color=col, lw=1.4,
                  label=f'Ch {ch} GP mean')

        # phase
        ax_p.plot(f_grid, np.angle(H_tr), color='#cccccc', lw=1.0)
        ax_p.plot(f_grid, np.angle(mean),  color=col, lw=1.4,
                  label=f'Ch {ch} GP mean')

        # measurements for this channel
        f_ch = [h[0] for h in history if h[1] == ch]
        y_ch = [h[2] for h in history if h[1] == ch]
        if f_ch:
            ax_a.scatter(f_ch, np.abs(y_ch),    c=col, s=14, zorder=5,
                         alpha=0.7, label=f'Ch {ch} n={len(f_ch)}')
            ax_p.scatter(f_ch, np.angle(y_ch),  c=col, s=14, zorder=5, alpha=0.7)

    ax_a.axvline(f_last, color='limegreen', lw=1.2, ls='--', alpha=0.85)
    ax_a.axvline(schroeder, **schroeder_kw)
    ax_a.set_ylabel('|H(f)|');  ax_a.set_xlim(f_grid[0], f_grid[-1])
    ax_a.grid(True, alpha=0.20);  ax_a.legend(fontsize=7, ncol=4, loc='upper right')

    ax_p.axvline(f_last, color='limegreen', lw=1.2, ls='--', alpha=0.85)
    ax_p.axvline(schroeder, **schroeder_kw)
    ax_p.set_ylabel('∠H(f) (rad)');  ax_p.set_xlim(f_grid[0], f_grid[-1])
    ax_p.set_ylim(-np.pi-0.3, np.pi+0.3)
    ax_p.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax_p.set_yticklabels(['-π','-π/2','0','π/2','π'])
    ax_p.grid(True, alpha=0.20);  ax_p.legend(fontsize=7, loc='upper right')

    # ── Posterior std ──────────────────────────────────────────────
    for ch in range(mgp.nc):
        ax_v.plot(f_grid, mgp.std(ch), color=CH_COLORS[ch],
                  lw=1.3, label=f'σ ch {ch}')
    ax_v.plot(f_grid, np.sqrt(mgp.var_joint), color='black',
              lw=1.0, ls='--', alpha=0.5, label='√(joint var)')
    ax_v.axvline(f_last, color='limegreen', lw=1.2, ls='--', alpha=0.85,
                 label=f'Probe {f_last:.0f}Hz ch{ch_last}')
    ax_v.axvline(schroeder, **schroeder_kw)
    ax_v.set_ylim(bottom=0);  ax_v.set_xlim(f_grid[0], f_grid[-1])
    ax_v.set_ylabel('Posterior std σ(f)')
    ax_v.grid(True, alpha=0.20);  ax_v.legend(fontsize=7, loc='upper right')

    # ── Measurement allocation bar chart ───────────────────────────
    counts = [sum(1 for h in history if h[1] == ch) for ch in range(mgp.nc)]
    ax_h.bar(range(mgp.nc), counts,
             color=CH_COLORS[:mgp.nc], alpha=0.75, edgecolor='white')
    ax_h.set_xticks(range(mgp.nc))
    ax_h.set_xticklabels([f'Ch {ch}' for ch in range(mgp.nc)])
    ax_h.set_ylabel('Measurements')
    ax_h.set_xlabel('Frequency (Hz)', fontsize=9)
    ax_h.grid(True, alpha=0.20, axis='y')
    # Label bars
    for i, c in enumerate(counts):
        ax_h.text(i, c + 0.3, str(c), ha='center', fontsize=9)


# ═══════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ═══════════════════════════════════════════════════════════════════════

def run():
    # ── Build room and lock-in ───────────────────────────────────────
    room   = RectangularRoom(**ROOM)
    f_grid = np.linspace(F_MIN, F_MAX, N_GRID)
    lockin = RoomLockin(room, SOURCE_POS, MIC_POSITIONS, f_grid,
                        noise_floor=NOISE_FLOOR)
    fs_hz  = room.schroeder_freq()

    # ── Validation plot ──────────────────────────────────────────────
    plt.ion()
    fig_val = plot_room_model(room, f_grid, lockin)
    fig_val.canvas.draw()
    plt.pause(1.5)

    # ── Inference figure (4 panels) ──────────────────────────────────
    fig = plt.figure(figsize=(13, 11), facecolor='#f9f9f9')
    gs  = gridspec.GridSpec(4, 1, hspace=0.48, top=0.92, bottom=0.06,
                            left=0.09, right=0.97,
                            height_ratios=[2, 2, 2, 1])
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    mgp   = MultiOutputGP(f_grid, n_channels=lockin.n_channels)
    agent = ActiveAgent(f_grid, n_channels=lockin.n_channels)

    n_ch  = lockin.n_channels
    print(f'{"─"*62}')
    print(f'  Two-Channel Active Inference — Rectangular Room')
    print(f'  {N_STEPS} steps  |  {n_ch} channels  |  '
          f'{T_PER_STEP:.2f}s integration per step')
    print(f'{"─"*62}')
    print(f'{"Step":>5}  {"Probe Hz":>10}  {"Ch":>4}  '
          f'{"Max σ joint":>12}  {"|H| meas":>12}')
    print('─' * 50)

    for step in range(1, N_STEPS + 1):
        f_next, ch_next = agent.select(mgp)
        y               = lockin.measure(f_next, ch_next, T_PER_STEP)
        nv              = lockin.obs_noise_var(f_next, T_PER_STEP)
        mgp.update(f_next, ch_next, y, nv)
        agent.record(f_next, ch_next, y)

        max_sigma = float(np.max(np.sqrt(mgp.var_joint)))
        print(f'{step:>5}  {f_next:>10.1f}  {ch_next:>4}  '
              f'{max_sigma:>12.5f}  {abs(y):>12.6f}')

        draw_inference(axes, f_grid, mgp, agent,
                       f_next, ch_next, lockin, fs_hz)
        fig.suptitle(
            f'Two-Channel Active Inference  |  '
            f'Step {step}/{N_STEPS}  |  '
            f'Probe: {f_next:.0f} Hz  ch{ch_next}  |  '
            f'Max σ: {max_sigma:.4f}',
            fontsize=10, fontweight='bold'
        )
        fig.canvas.draw()
        plt.pause(0.01)

    print('─' * 50)
    print('Done.')
    plt.ioff()
    plt.show(block=True)
    try:
        input('\nPress Enter to close figures...')
    except (EOFError, KeyboardInterrupt):
        pass
    plt.close('all')


if __name__ == '__main__':
    run()