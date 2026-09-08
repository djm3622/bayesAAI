#!/usr/bin/env python3
import matplotlib
matplotlib.use('TkAgg')   # reliable live-update backend on macOS
"""
Active Acoustic Inference — Physically Derived Rectangular Room
================================================================
Stage 1: single speaker, single microphone, simulation only.

Room model
----------
For a rigid-walled rectangular room (Lx × Ly × Lz), the pressure
eigenfunctions and eigenfrequencies are known analytically:

    f_lmn = (c/2) · sqrt((l/Lx)² + (m/Ly)² + (n/Lz)²)

    ψ_lmn(r) = cos(lπx/Lx) · cos(mπy/Ly) · cos(nπz/Lz)

The transfer function between source rₛ and mic rₘ is:

    H(f) = Σ_n  Aₙ / (fₙ² − f² + j·f·fₙ/Qₙ)

    where  Aₙ = εₗεₘεₙ · ψₙ(rₛ) · ψₙ(rₘ) / V
           εᵢ = 1 if index = 0, else 2   (eigenfunction normalisation)

Modal Q values are derived from the Sabine reverberation time:

    T₆₀  = 0.161 · V / (α · S)          [Sabine formula]
    Qₙ   = π · fₙ · T₆₀ / (3 · ln 10)  [Q from modal decay rate]

This gives physically correct frequency-dependent Q: modes at higher
frequency have proportionally higher Q (more cycles per decay).

Above the Schroeder frequency (~150–300 Hz for a typical room) modes
overlap heavily and cannot be resolved individually.  The GP inference
still works throughout; the CAP pole estimator (Stage 2) will target
the isolated-mode region below Schroeder.

Note on mode count
------------------
Up to 2000 Hz a room of this size has ~50 000 modes.  The transfer
function is precomputed on a dense grid at initialisation using chunked
NumPy operations (~2 s), then interpolated for each lock-in step.
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
MIC_POS    = np.array([3.8, 2.7, 1.1])

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
SCHROEDER_HZ = 500.0   # estimated Schroeder frequency — tune to room
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
    Simulated lock-in measurements using the rectangular room model.

    Noise model:  PSD(f) = noise_floor · (1 + 400/f)   [pink-ish / HVAC]
    Lock-in variance after integration time T:  σ² = PSD(f) / T
    """

    def __init__(self, room, src_pos, mic_pos,
                 f_grid, noise_floor=3e-5):
        self.room = room
        self.src  = src_pos
        self.mic  = mic_pos
        self.nf   = noise_floor
        # Precompute H on the display grid
        self.H_true = room.precompute_grid(f_grid, src_pos, mic_pos)

    def noise_psd(self, f):
        return self.nf * (1.0 + 400.0 / f)

    def measure(self, f, T=0.5):
        """Return noisy complex H(f) measurement."""
        H_true = self.room.at_grid(f)
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


class GP:
    """
    Gaussian Process over the complex room transfer function H(f).

    Real and imaginary parts are modelled as independent GPs sharing
    the same Matérn-5/2 kernel with a frequency-dependent length scale:
      • tight  (LS_BELOW Hz) below the Schroeder frequency
      • smooth (LS_ABOVE Hz) above it

    Posterior updated analytically after each lock-in measurement.
    Key output: posterior std σ(f) — drives the active probe selection.
    """

    def __init__(self, f_grid, signal_var=0.4, jitter=1e-6):
        self.f   = f_grid
        self.sv  = signal_var
        self.jit = jitter

        n = len(f_grid)
        self._prior_var = signal_var * np.ones(n)
        self._fo = [];  self._yr = [];  self._yi = [];  self._nv = []

        self.mean_re = np.zeros(n)
        self.mean_im = np.zeros(n)
        self.var     = self._prior_var.copy()

    def update(self, f_new, y_new, obs_noise_var):
        self._fo.append(f_new)
        self._yr.append(y_new.real)
        self._yi.append(y_new.imag)
        self._nv.append(obs_noise_var)

        fo = np.array(self._fo)
        yr = np.array(self._yr)
        yi = np.array(self._yi)

        Koo  = _k52_nonstat(fo, fo, self.sv)
        Koo += np.diag(self._nv) + self.jit * np.eye(len(fo))
        Kgo  = _k52_nonstat(self.f, fo, self.sv)

        L        = np.linalg.cholesky(Koo)
        alpha_r  = np.linalg.solve(L.T, np.linalg.solve(L, yr))
        alpha_i  = np.linalg.solve(L.T, np.linalg.solve(L, yi))

        self.mean_re = Kgo @ alpha_r
        self.mean_im = Kgo @ alpha_i
        V            = np.linalg.solve(L, Kgo.T)
        self.var     = np.maximum(self._prior_var - np.sum(V**2, axis=0), 0.0)

    @property
    def mean(self): return self.mean_re + 1j * self.mean_im

    @property
    def std(self):  return np.sqrt(self.var)


# ═══════════════════════════════════════════════════════════════════════
# 4.  ACTIVE AGENT
# ═══════════════════════════════════════════════════════════════════════

class ActiveAgent:
    """
    Selects the next probe frequency as argmax of posterior variance.

    An exclusion zone around recently measured frequencies prevents
    the agent re-probing where the posterior is already tight.
    The exclusion radius is tied to the local length scale so it
    automatically tightens below the Schroeder frequency.
    """

    def __init__(self, f_grid, excl_factor=1.2):
        self.f_grid      = f_grid
        self.excl_factor = excl_factor   # exclusion = factor × local ls
        self.history     = []

    def select(self, gp):
        var = gp.var.copy()
        for f_prev, _ in self.history[-10:]:
            excl = self.excl_factor * float(_length_scale(np.array([f_prev]))[0])
            var[np.abs(self.f_grid - f_prev) < excl] = 0.0
        return self.f_grid[np.argmax(var)]

    def record(self, f, y):
        self.history.append((f, y))


# ═══════════════════════════════════════════════════════════════════════
# 5.  VALIDATION PLOT  (static, shown before inference starts)
# ═══════════════════════════════════════════════════════════════════════

def plot_room_model(room, f_grid, H_true):
    """
    Show the true room transfer function with modal frequencies marked.
    Helps verify the room model before running inference.
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), facecolor='#f9f9f9',
                              sharex=True)
    fig.subplots_adjust(hspace=0.35, top=0.88, bottom=0.10,
                        left=0.08, right=0.97)
    fig.suptitle(
        f"Room Model Validation  |  "
        f"{room.Lx:.1f} × {room.Ly:.1f} × {room.Lz:.1f} m  |  "
        f"α = {room.alpha:.2f}  |  T₆₀ = {room.T60:.2f} s  |  "
        f"{room.n_modes:,} modes ≤ {F_MAX:.0f} Hz",
        fontsize=10, fontweight='bold'
    )

    ax_a, ax_p = axes

    # ── Amplitude ──
    ax_a.plot(f_grid, np.abs(H_true), color='steelblue', lw=1.2)
    ax_a.set_ylabel('|H(f)|')
    ax_a.set_xlim(F_MIN, F_MAX)
    ax_a.grid(True, alpha=0.25)

    # Mark modal frequencies (only those in display range, below Schroeder)
    fs_hz = room.schroeder_freq()
    modal_below = room.f_modes[(room.f_modes >= F_MIN) &
                               (room.f_modes <= min(fs_hz * 1.5, 500.0))]
    for fm in modal_below:
        ax_a.axvline(fm, color='tomato', lw=0.6, alpha=0.5)

    # Schroeder frequency marker
    ax_a.axvline(fs_hz, color='darkorange', lw=1.5, ls='--', alpha=0.8)

    legend_elements = [
        Line2D([0], [0], color='steelblue',   lw=1.5, label='|H(f)| true'),
        Line2D([0], [0], color='tomato',      lw=1.0, label='Modal freqs (below 1.5·fₛ)'),
        Line2D([0], [0], color='darkorange',  lw=1.5, ls='--',
               label=f'Schroeder fₛ ≈ {fs_hz:.0f} Hz'),
    ]
    ax_a.legend(handles=legend_elements, fontsize=8, loc='upper right')

    # ── Phase ──
    ax_p.plot(f_grid, np.angle(H_true), color='steelblue', lw=1.2)
    ax_p.set_ylabel('∠H(f)  (rad)')
    ax_p.set_xlabel('Frequency  (Hz)')
    ax_p.set_ylim(-np.pi - 0.3, np.pi + 0.3)
    ax_p.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax_p.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax_p.grid(True, alpha=0.25)
    for fm in modal_below:
        ax_p.axvline(fm, color='tomato', lw=0.6, alpha=0.5)
    ax_p.axvline(fs_hz, color='darkorange', lw=1.5, ls='--', alpha=0.8)

    plt.pause(0.1)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 6.  LIVE INFERENCE PLOT
# ═══════════════════════════════════════════════════════════════════════

def draw_inference(ax_a, ax_p, ax_v, f_grid, gp, agent,
                   f_last, H_true, schroeder):

    f_meas = np.array([h[0] for h in agent.history])
    y_meas = np.array([h[1] for h in agent.history])
    mean   = gp.mean
    std    = gp.std

    ax_a.cla();  ax_p.cla();  ax_v.cla()

    # Amplitude
    ax_a.plot(f_grid, np.abs(H_true), color='#bbbbbb', lw=1.2, label='True |H|')
    ax_a.fill_between(f_grid,
                      np.maximum(np.abs(mean) - 2*std, 0),
                      np.abs(mean) + 2*std,
                      color='steelblue', alpha=0.18, label='±2σ')
    ax_a.plot(f_grid, np.abs(mean), color='steelblue', lw=1.5, label='GP mean')
    if len(f_meas):
        ax_a.scatter(f_meas, np.abs(y_meas), c='crimson', s=18, zorder=5,
                     label=f'n = {len(f_meas)}')
    ax_a.axvline(f_last, color='limegreen', lw=1.3, ls='--', alpha=0.85)
    ax_a.axvline(schroeder, color='darkorange', lw=1.0, ls=':', alpha=0.6)
    ax_a.set_ylabel('|H(f)|');  ax_a.set_xlim(f_grid[0], f_grid[-1])
    ax_a.grid(True, alpha=0.22)
    ax_a.legend(fontsize=7.5, ncol=4, loc='upper right')

    # Phase
    ax_p.plot(f_grid, np.angle(H_true), color='#bbbbbb', lw=1.2, label='True ∠H')
    ax_p.plot(f_grid, np.angle(mean),   color='steelblue', lw=1.5, label='GP mean')
    if len(f_meas):
        ax_p.scatter(f_meas, np.angle(y_meas), c='crimson', s=18, zorder=5)
    ax_p.axvline(f_last, color='limegreen', lw=1.3, ls='--', alpha=0.85)
    ax_p.axvline(schroeder, color='darkorange', lw=1.0, ls=':', alpha=0.6)
    ax_p.set_ylabel('∠H(f)  (rad)')
    ax_p.set_xlim(f_grid[0], f_grid[-1])
    ax_p.set_ylim(-np.pi - 0.3, np.pi + 0.3)
    ax_p.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax_p.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax_p.grid(True, alpha=0.22)
    ax_p.legend(fontsize=7.5, loc='upper right')

    # Posterior std
    ax_v.fill_between(f_grid, 0, std, color='tomato', alpha=0.28)
    ax_v.plot(f_grid, std, color='tomato', lw=1.5, label='σ(f)  posterior std')
    ax_v.axvline(f_last, color='limegreen', lw=1.3, ls='--', alpha=0.85,
                 label=f'Last probe  {f_last:.0f} Hz')
    ax_v.axvline(schroeder, color='darkorange', lw=1.0, ls=':', alpha=0.6,
                 label=f'Schroeder  {schroeder:.0f} Hz')
    ax_v.set_ylim(bottom=0)
    ax_v.set_xlim(f_grid[0], f_grid[-1])
    ax_v.set_ylabel('Posterior std  σ(f)')
    ax_v.set_xlabel('Frequency  (Hz)', fontsize=9)
    ax_v.grid(True, alpha=0.22)
    ax_v.legend(fontsize=7.5, loc='upper right')


# ═══════════════════════════════════════════════════════════════════════
# 7.  CAP ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════
#
# Fits the Common Acoustic Poles model to measurements collected below
# the Schroeder frequency:
#
#     H(f) = Σ_n  Aₙ / (fₙ² − f² + j·f·fₙ/Qₙ)
#
# The problem is nonlinear in the poles {fₙ, Qₙ} but linear in the
# residues {Aₙ} given fixed poles.  We exploit this structure:
#
#   Step 1 — Initialise pole frequencies from peaks of |GP mean| below
#             the Schroeder frequency.
#   Step 2 — Given poles, solve for residues by linear least squares.
#   Step 3 — Given residues, refine poles by Levenberg-Marquardt using
#             the scipy least_squares (trf method) with analytic structure.
#   Step 4 — Iterate steps 2–3 until convergence.
#
# Validation: compare estimated poles to the room's true modal
# frequencies and Q values (known from RectangularRoom).

from scipy.optimize import least_squares
from scipy.signal import find_peaks


def _lorentzian(f, fn, Qn):
    """Complex Lorentzian for mode n evaluated at frequencies f."""
    return 1.0 / (fn**2 - f**2 + 1j * f * fn / Qn)


def _build_basis(f_meas, f_poles, Q_poles):
    """
    Build the complex basis matrix Φ of shape (N_meas, N_poles).
    Column n is the Lorentzian for pole n at all measurement frequencies.
    Residues then solve:  Φ @ A ≈ H_meas  (linear least squares).
    """
    Phi = np.zeros((len(f_meas), len(f_poles)), dtype=complex)
    for n in range(len(f_poles)):
        Phi[:, n] = _lorentzian(f_meas, f_poles[n], Q_poles[n])
    return Phi


def _solve_residues(Phi, H_meas):
    """
    Linear least squares for residues A given basis Phi.
    Stacks real and imaginary parts so numpy lstsq sees a real system.
    """
    Phi_ri = np.vstack([Phi.real, Phi.imag])
    H_ri   = np.concatenate([H_meas.real, H_meas.imag])
    A, _, _, _ = np.linalg.lstsq(Phi_ri, H_ri, rcond=None)
    return A


def _residual_fn(params, f_meas, H_meas):
    """
    Flattened real residual vector for scipy.optimize.least_squares.
    params = [f1, Q1, f2, Q2, ..., fN, QN]
    Residues are re-solved analytically at each evaluation (VarPro).
    """
    f_poles = params[0::2]
    Q_poles = params[1::2]
    Phi     = _build_basis(f_meas, f_poles, Q_poles)
    A       = _solve_residues(Phi, H_meas)
    H_hat   = Phi @ A
    res     = H_meas - H_hat
    return np.concatenate([res.real, res.imag])


def _init_poles_from_gp(gp, f_grid, f_cap_max, n_poles_max=20,
                         min_separation_hz=4.0):
    """
    Initialise pole frequencies from peaks of |GP mean| below f_cap_max.
    Q is estimated from the half-power bandwidth of each peak.
    """
    mask    = f_grid <= f_cap_max
    f_sub   = f_grid[mask]
    amp_sub = np.abs(gp.mean[mask])
    df      = f_sub[1] - f_sub[0]

    min_sep_idx = max(1, int(min_separation_hz / df))
    peaks, props = find_peaks(amp_sub, distance=min_sep_idx, prominence=0.002)

    if len(peaks) == 0:
        f_init = np.linspace(f_sub[0] + 10, f_cap_max - 10, 5)
        return f_init, np.full(len(f_init), 10.0)

    # Keep top n_poles_max by prominence
    order = np.argsort(props['prominences'])[::-1][:n_poles_max]
    peaks = np.sort(peaks[order])
    f_init = f_sub[peaks]

    # Estimate Q from half-power bandwidth
    Q_init = np.zeros(len(peaks))
    for i, pk in enumerate(peaks):
        half = amp_sub[pk] / np.sqrt(2)
        left = pk
        while left > 0 and amp_sub[left] > half:
            left -= 1
        right = pk
        while right < len(amp_sub) - 1 and amp_sub[right] > half:
            right += 1
        bw = (right - left) * df
        Q_init[i] = np.clip(f_init[i] / bw if bw > 0 else 10.0, 2.0, 200.0)

    return f_init, Q_init


class CAPEstimator:
    """
    Common Acoustic Poles estimator — single speaker-mic channel.

    Usage
    -----
    cap = CAPEstimator(f_cap_max=SCHROEDER_HZ)
    cap.fit(gp, agent, f_grid)   # initialise from GP, fit to measurements
    cap.print_comparison(room)   # compare estimated vs true poles
    H_cap = cap.predict(f_grid)  # evaluate fitted model on frequency grid
    """

    def __init__(self, f_cap_max=None, n_poles_max=20,
                 min_separation_hz=8.0):
        self.f_cap_max   = f_cap_max or SCHROEDER_HZ
        self.n_poles_max = n_poles_max
        self.min_sep     = min_separation_hz
        self.f_poles     = None
        self.Q_poles     = None
        self.residues    = None

    def fit(self, gp, agent, f_grid, verbose=True):
        """Fit to measurements below f_cap_max collected by the active agent."""
        f_all  = np.array([h[0] for h in agent.history])
        y_all  = np.array([h[1] for h in agent.history])
        mask   = f_all <= self.f_cap_max
        f_meas = f_all[mask]
        H_meas = y_all[mask]

        if len(f_meas) < 4:
            print("CAP: insufficient measurements below Schroeder frequency.")
            return self

        if verbose:
            print(f'\n{"─"*55}')
            print(f'  CAP Estimator — {len(f_meas)} measurements '
                  f'below {self.f_cap_max:.0f} Hz')

        # Step 1: initialise poles from GP peaks
        f_init, Q_init = _init_poles_from_gp(
            gp, f_grid, self.f_cap_max,
            n_poles_max=self.n_poles_max,
            min_separation_hz=self.min_sep
        )
        N_poles = len(f_init)
        if verbose:
            print(f'  Initial poles ({N_poles}): '
                  f'{", ".join(f"{f:.1f}" for f in f_init)} Hz')

        # Steps 2–4: VarPro nonlinear least squares
        p0      = np.empty(2 * N_poles)
        p0[0::2] = f_init
        p0[1::2] = Q_init

        lb = np.empty_like(p0);  ub = np.empty_like(p0)
        lb[0::2] = F_MIN;        ub[0::2] = self.f_cap_max
        lb[1::2] = 1.0;          ub[1::2] = 500.0

        result = least_squares(
            _residual_fn, p0,
            bounds=(lb, ub),
            args=(f_meas, H_meas),
            method='trf',
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
            max_nfev=5000, verbose=0
        )

        self.f_poles = result.x[0::2]
        self.Q_poles = result.x[1::2]

        # Final residue solve
        Phi           = _build_basis(f_meas, self.f_poles, self.Q_poles)
        self.residues = _solve_residues(Phi, H_meas)

        H_hat      = Phi @ self.residues
        rms_err    = np.sqrt(np.mean(np.abs(H_meas - H_hat)**2))
        rms_signal = np.sqrt(np.mean(np.abs(H_meas)**2))

        if verbose:
            print(f'  Converged : {result.nfev} evals, '
                  f'RMS error = {rms_err:.4f} '
                  f'({100*rms_err/rms_signal:.1f}% of signal RMS)')
            print(f'{"─"*55}')
        return self

    def predict(self, f_array):
        """Evaluate fitted CAP model on a frequency array."""
        if self.f_poles is None:
            raise RuntimeError("Call fit() first.")
        f = np.asarray(f_array)
        H = np.zeros(len(f), dtype=complex)
        for n in range(len(self.f_poles)):
            H += self.residues[n] * _lorentzian(f, self.f_poles[n],
                                                  self.Q_poles[n])
        return H

    def print_comparison(self, room):
        """Compare estimated poles to true room poles."""
        if self.f_poles is None:
            return
        true_f = room.f_modes[room.f_modes <= self.f_cap_max]
        true_Q = room.Q_modes[room.f_modes <= self.f_cap_max]

        print(f'\n{"─"*65}')
        print(f'  CAP Pole Comparison  (below {self.f_cap_max:.0f} Hz)')
        print(f'{"─"*65}')
        print(f'  {"Est f (Hz)":>12}  {"Est Q":>8}  │  '
              f'{"True f (Hz)":>12}  {"True Q":>8}  │  {"Δf (Hz)":>8}')
        print(f'  {"─"*12}  {"─"*8}  │  {"─"*12}  {"─"*8}  │  {"─"*8}')
        for idx in np.argsort(self.f_poles):
            ef = self.f_poles[idx];  eQ = self.Q_poles[idx]
            near = np.argmin(np.abs(true_f - ef))
            print(f'  {ef:>12.2f}  {eQ:>8.2f}  │  '
                  f'{true_f[near]:>12.2f}  {true_Q[near]:>8.2f}  │  '
                  f'{ef - true_f[near]:>+8.2f}')
        print(f'{"─"*65}')
        print(f'  True poles in range : {len(true_f)}')
        print(f'  Estimated poles     : {len(self.f_poles)}')
        print(f'{"─"*65}\n')


def plot_cap_result(f_grid, H_true, gp, cap, schroeder):
    """Final comparison: true H(f) vs GP mean vs CAP model."""
    mask      = f_grid <= cap.f_cap_max
    H_cap     = cap.predict(f_grid[mask])

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), facecolor='#f9f9f9',
                              sharex=True)
    fig.subplots_adjust(hspace=0.35, top=0.90, bottom=0.09,
                        left=0.08, right=0.97)
    fig.suptitle(
        f'CAP Model vs GP vs True  |  '
        f'{len(cap.f_poles)} poles estimated below {cap.f_cap_max:.0f} Hz',
        fontsize=11, fontweight='bold'
    )
    ax_a, ax_p = axes

    # Amplitude
    ax_a.plot(f_grid, np.abs(H_true),   color='#aaaaaa', lw=1.2,
              label='True |H(f)|')
    ax_a.plot(f_grid, np.abs(gp.mean),  color='steelblue', lw=1.5,
              ls='--', alpha=0.7, label='GP mean')
    ax_a.plot(f_grid[mask], np.abs(H_cap), color='crimson', lw=2.0,
              label='CAP model')
    for fn in cap.f_poles:
        ax_a.axvline(fn, color='crimson', lw=0.7, alpha=0.35)
    ax_a.axvline(schroeder, color='darkorange', lw=1.2, ls=':', alpha=0.7,
                 label=f'Schroeder {schroeder:.0f} Hz')
    ax_a.set_ylabel('|H(f)|');  ax_a.set_xlim(f_grid[0], f_grid[-1])
    ax_a.grid(True, alpha=0.22)
    ax_a.legend(fontsize=8, loc='upper right', ncol=2)

    # Phase
    ax_p.plot(f_grid, np.angle(H_true),  color='#aaaaaa', lw=1.2,
              label='True ∠H(f)')
    ax_p.plot(f_grid, np.angle(gp.mean), color='steelblue', lw=1.5,
              ls='--', alpha=0.7, label='GP mean')
    ax_p.plot(f_grid[mask], np.angle(H_cap), color='crimson', lw=2.0,
              label='CAP model')
    for fn in cap.f_poles:
        ax_p.axvline(fn, color='crimson', lw=0.7, alpha=0.35)
    ax_p.axvline(schroeder, color='darkorange', lw=1.2, ls=':', alpha=0.7)
    ax_p.set_ylabel('∠H(f)  (rad)');  ax_p.set_xlabel('Frequency  (Hz)')
    ax_p.set_ylim(-np.pi - 0.3, np.pi + 0.3)
    ax_p.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax_p.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax_p.grid(True, alpha=0.22)
    ax_p.legend(fontsize=8, loc='upper right', ncol=2)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ═══════════════════════════════════════════════════════════════════════

def run():
    # ── Build room ──────────────────────────────────────────────────
    room   = RectangularRoom(**ROOM)
    f_grid = np.linspace(F_MIN, F_MAX, N_GRID)
    lockin = RoomLockin(room, SOURCE_POS, MIC_POS, f_grid,
                        noise_floor=NOISE_FLOOR)
    H_true = lockin.H_true
    fs_hz  = room.schroeder_freq()

    # ── Validation plot ──────────────────────────────────────────────
    plt.ion()
    fig_val = plot_room_model(room, f_grid, H_true)
    fig_val.canvas.draw()
    plt.pause(1.5)

    # ── GP inference figure ──────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9), facecolor='#f9f9f9')
    gs  = gridspec.GridSpec(3, 1, hspace=0.50, top=0.91, bottom=0.07,
                            left=0.09, right=0.97)
    ax_a = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1])
    ax_v = fig.add_subplot(gs[2])

    gp    = GP(f_grid)
    agent = ActiveAgent(f_grid)

    print(f'{"─"*58}')
    print(f'  Active Acoustic Inference — Rectangular Room Simulation')
    print(f'  {N_STEPS} steps  |  {T_PER_STEP:.2f} s integration per step')
    print(f'{"─"*58}')
    print(f'{"Step":>5}  {"Probe Hz":>10}  {"Max σ":>10}  {"|H| meas":>12}')
    print('─' * 42)

    for step in range(1, N_STEPS + 1):
        f_next = agent.select(gp)
        y      = lockin.measure(f_next, T_PER_STEP)
        nv     = lockin.obs_noise_var(f_next, T_PER_STEP)
        gp.update(f_next, y, nv)
        agent.record(f_next, y)

        print(f'{step:>5}  {f_next:>10.1f}  {np.max(gp.std):>10.5f}  '
              f'{abs(y):>12.6f}')

        draw_inference(ax_a, ax_p, ax_v,
                       f_grid, gp, agent, f_next, H_true, fs_hz)
        fig.suptitle(
            f'Active Inference — Rectangular Room  |  '
            f'Step {step}/{N_STEPS}  |  '
            f'Probe: {f_next:.0f} Hz  |  '
            f'Max σ: {np.max(gp.std):.4f}',
            fontsize=10, fontweight='bold'
        )
        fig.canvas.draw()
        plt.pause(0.01)

    # ── CAP estimation ───────────────────────────────────────────────
    cap = CAPEstimator(f_cap_max=fs_hz)
    cap.fit(gp, agent, f_grid)
    cap.print_comparison(room)

    fig_cap = plot_cap_result(f_grid, H_true, gp, cap, fs_hz)
    fig_cap.canvas.draw()

    print('─' * 42)
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