"""Kaiser bandpass + autocorrelation + coarse peak (Breathing rate estimation)."""
import numpy as np
from scipy import signal as sig
from . import config as C


def design_kaiser():
    taps = sig.firwin(C.FIR_TAPS, [C.BAND_LOW_HZ, C.BAND_HIGH_HZ],
                      fs=C.FS_RADAR, window=("kaiser", C.KAISER_BETA),
                      pass_zero=False)
    return taps


_TAPS = design_kaiser()


def bandpass(x_hat):
    """Zero-phase FIR on real and imag separately (complex preserved)."""
    x_hat = np.asarray(x_hat)
    r = sig.filtfilt(_TAPS, [1.0], np.real(x_hat))
    im = sig.filtfilt(_TAPS, [1.0], np.imag(x_hat))
    return r + 1j * im


def autocorr_coarse(d_hat, fs=C.FS_RADAR, prominence=0.10):
    """Autocorrelation then mean peak-gap -> f_coarse (Hz). Falls back to NaN.

    Follows paper Methods: peak detection on r(t), exclude gaps outside
    the 18-180 bpm physiological range, and compute average inter-peak time.
    Calculates autocorrelation directly on complex d_hat without projection loss.
    """
    d = np.asarray(d_hat, dtype=complex)
    d = d - np.mean(d)
    n = len(d)
    if np.allclose(d, 0):
        return np.nan
    X = np.fft.fft(d, n=2 * n)
    r = np.real(np.fft.ifft(np.abs(X) ** 2))[:n]
    r = r / (r[0] + 1e-12)

    peaks, _ = sig.find_peaks(r, distance=max(1, int(fs / 3.0)), prominence=prominence)
    pk = np.r_[0, peaks]
    gaps = np.diff(pk) / fs
    gaps = gaps[(gaps >= 1.0 / 3.0) & (gaps <= 1.0 / 0.3)]
    if len(gaps) == 0:
        return np.nan
    return 1.0 / float(np.mean(gaps))

