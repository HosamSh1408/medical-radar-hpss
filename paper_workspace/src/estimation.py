"""DFT baseline + NLS harmonic estimator (Eq.17, Lk=2)."""
import numpy as np
from . import config as C


def psd_fft(d_hat, nfft=C.NLS_NFFT, fs=C.FS_RADAR):
    """Folded one-sided PSD of complex or real signal (Eq.17).

    Preserves 100% of quadrature energy without projection loss onto [1, 1].
    P(f) = |X(+f)|^2 + |X(-f)|^2 for f >= 0.
    """
    d = np.asarray(d_hat, dtype=complex)
    X = np.fft.fft(d, n=nfft)
    P = np.abs(X) ** 2
    half = nfft // 2
    Pf = P[:half + 1].copy()
    Pf[1:half] += P[nfft - 1:half:-1]
    f = np.arange(half + 1) * fs / nfft
    return f, Pf



def dft_estimate(d_hat, fs=C.FS_RADAR):
    f, S = psd_fft(d_hat)
    m = (f >= C.BAND_LOW_HZ) & (f <= C.BAND_HIGH_HZ)
    f, S = f[m], S[m]
    return float(f[np.argmax(S)] * 60.0), f, S


def nls_estimate(d_hat, f_coarse_hz, fs=C.FS_RADAR,
                 lk=C.NLS_LK, radius_bpm=C.NLS_RADIUS_BPM,
                 step_bpm=C.NLS_GRID_STEP_BPM, weights=None,
                 dual_center=False):
    """NLS harmonic sum (Eq.17) with experiment options.

    weights: per-harmonic multipliers (default equal). dual_center: also
    evaluate around DFT peak and keep the higher normalized score (robust
    to coarse octave errors; ablation only, not paper spec).
    """
    f, S = psd_fft(d_hat)
    # interpolate helper
    def at(ff):
        return np.interp(ff, f, S, left=0, right=0)
    if weights is None:
        weights = [1.0] * lk
    if np.isnan(f_coarse_hz):
        # fallback to DFT peak as centre
        c, _, _ = dft_estimate(d_hat, fs)
        centre = c
    else:
        centre = f_coarse_hz * 60.0
    centres = [centre]
    if dual_center:
        c, _, _ = dft_estimate(d_hat, fs)
        if abs(c - centre) > radius_bpm:
            centres.append(c)
    best, bestf = -1, centre
    for centre in centres:
        lo = max(C.BAND_LOW_HZ * 60, centre - radius_bpm)
        hi = min(C.BAND_HIGH_HZ * 60, centre + radius_bpm)
        grid = np.arange(lo, hi + 1e-9, step_bpm)
        for g in grid:
            score = sum(w * at(g * h / 60.0)
                        for h, w in enumerate(weights, start=1))
            if score > best:
                best, bestf = score, g
    return float(bestf), f, S
