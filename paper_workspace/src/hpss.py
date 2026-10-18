"""Branch B: median-filtered HPSS extension (Fitzgerald-style).

Shared upstream: CSD -> 30s windows -> STFT rect 48/hop 16/NFFT 256 (hop 16,
i.e. 1 s; the plan's parenthetical '32 samples' is a typo: 32 samples @16 Hz
would be a 2 s hop, contradicting 2 s overlap on a 3 s window).
Shared downstream: Wiener phase reuse + iSTFT -> Kaiser bandpass -> NLS.

Separation (on magnitude spectrogram |X|, F x T):
  S_harm = median(|X|, kernel=(1, l_harm))   # smooth along TIME -> stable harmonic lines
  S_perc = median(|X|, kernel=(l_perc, 1))   # smooth along FREQ -> broadband transients
  M_harm = S_harm^gamma / (S_harm^gamma + S_perc^gamma + eps)
  |X_hpss| = M_harm * |X|
Note axis convention: |X| rows=frequency, cols=time, so time smoothing uses
a (1, l_harm) footprint and frequency smoothing a (l_perc, 1) footprint.

Hyperparameters (plan): l_harm=7 frames (~7 s, spans full 1-2 s breath cycles),
l_perc=21 bins (~1.3 Hz @0.0625 Hz/bin, wider than the narrow breathing line),
gamma=2, eps=1e-12.

Caveat: STFT here is two-sided complex FFT (F=256, shared with the NMF
branch for a fair A/B), so frequency-median spans both positive and negative
halves. Standard HPSS uses a one-sided spectrum; absolute numbers could shift
under a one-sided reimplementation, but the A/B comparison is unaffected.
"""
import numpy as np
from scipy.ndimage import median_filter
from . import config as C

L_HARM = 7
L_PERC = 21
GAMMA = 2.0
EPS = 1e-12


def hpss_filter(mag, l_harm=L_HARM, l_perc=L_PERC, gamma=GAMMA):
    mag = np.maximum(np.asarray(mag, dtype=float), 0.0)
    mag_s = np.fft.fftshift(mag, axes=0)
    s_harm = median_filter(mag_s, size=(1, l_harm), mode="nearest")
    s_perc = median_filter(mag_s, size=(l_perc, 1), mode="nearest")
    hg = s_harm ** gamma
    pg = s_perc ** gamma
    mask_s = hg / (hg + pg + EPS)
    filt = np.fft.ifftshift(mask_s * mag_s, axes=0)
    mask = np.fft.ifftshift(mask_s, axes=0)
    return filt, mask, np.fft.ifftshift(s_harm, axes=0), np.fft.ifftshift(s_perc, axes=0)


def mitigate_hpss(x, l_harm=L_HARM, l_perc=L_PERC, gamma=GAMMA):
    """Full Branch-B block mirroring rbm_nmf.mitigate diagnostics."""
    from .rbm_nmf import stft_complex, istft_complex, env_normalize
    X, mag, phase = stft_complex(x)
    mag_filt_raw, mask, s_harm, s_perc = hpss_filter(mag, l_harm, l_perc, gamma)
    mag_filt = env_normalize(mag_filt_raw)
    Xhat = mag_filt * np.exp(1j * phase)
    x_hat = istft_complex(Xhat, len(np.asarray(x).ravel()))
    # continuity proxy for minimal-subset analogy: fraction of mask energy kept
    kept_frac = float(np.sum(mag_filt_raw) / (np.sum(mag) + EPS))
    return {"x_hat": x_hat, "X": X, "mag": mag, "mag_filt": mag_filt,
            "mask": mask, "kept_frac": kept_frac}
