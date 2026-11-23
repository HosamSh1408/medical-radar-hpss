"""Score-based 1D/2D HR-BR Viterbi tracker and spectral observation extractors.

Reproduces and cleanly upgrades the signal-processing and tracking formulation from:
  H. Shibly, "Probabilistically Informed Respiratory-Rate Tracking Using Contactless
  Continuous-Wave Medical Radar for Unconstrained Neonatal Monitoring"
directly on top of the repository's audited Beltrao et al. (2022) + HPSS baseline.

All symbol conventions adhere strictly to
`paper_orchestra_bundle/raw_materials/notation_glossary.tex`.
"""
from __future__ import annotations

from functools import lru_cache
import warnings

import numpy as np
from scipy import signal as sig
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

from paper_workspace.src import config as C
from paper_workspace.src.estimation import dft_estimate, psd_fft
from paper_workspace.src.hpss import mitigate_hpss
from paper_workspace.src.rbm_nmf import detect_rbm_hoyer, istft_complex, stft_complex

BR_GRID_2BPM = np.arange(18.0, 90.0 + 1e-9, 2.0, dtype=np.float64)   # 37 bins (paper Sec. 4.4)
BR_GRID_1BPM = np.arange(18.0, 90.0 + 1e-9, 1.0, dtype=np.float64)   # 73 bins (Beltrao fine grid)
HR_GRID_2BPM = np.arange(90.0, 210.0 + 1e-9, 2.0, dtype=np.float64)  # 61 bins (paper Sec. 4.4)

_HARM_ORDERS = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)


@lru_cache(maxsize=16)
def _design_bandpass_taps(
    low_hz: float,
    high_hz: float,
    fs: float = float(C.FS_RADAR),
    taps: int = int(C.FIR_TAPS),
    beta: float = float(C.KAISER_BETA),
) -> np.ndarray:
    """Design Kaiser-windowed FIR bandpass taps matching `filtering.design_kaiser`."""
    return sig.firwin(
        int(taps),
        [float(low_hz), float(high_hz)],
        fs=float(fs),
        window=("kaiser", float(beta)),
        pass_zero=False,
    )


def complex_bandpass(
    x: np.ndarray,
    low_hz: float,
    high_hz: float,
    fs: float = float(C.FS_RADAR),
) -> np.ndarray:
    """Zero-phase Kaiser FIR filter preserving complex I/Q linearity & invariants."""
    arr = np.asarray(x)
    if arr.size < 12:
        return arr.astype(np.complex128 if np.iscomplexobj(arr) else np.float64)
    taps = _design_bandpass_taps(float(low_hz), float(high_hz), float(fs))
    padlen = min(3 * len(taps), arr.size - 1)
    if np.iscomplexobj(arr):
        re = sig.filtfilt(taps, [1.0], np.real(arr).astype(np.float64), padlen=padlen)
        im = sig.filtfilt(taps, [1.0], np.imag(arr).astype(np.float64), padlen=padlen)
        return re + 1j * im
    return sig.filtfilt(taps, [1.0], arr.astype(np.float64), padlen=padlen)


def folded_welch_psd(
    sig_win: np.ndarray,
    fs: float = float(C.FS_RADAR),
    nperseg: int = 256,
    noverlap: int = 128,
    nfft: int = int(C.NLS_NFFT),
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided complex Welch PSD folded symmetrically onto f >= 0.

    Folding P(+f) + P(-f) guarantees invariance to global phase rotation e^{j phi},
    I/Q sign flip (-I, -Q), and I/Q channel swap (Q, I) = j * conj(I + j Q).
    """
    s = np.asarray(sig_win, dtype=np.complex128).ravel()
    if s.size == 0 or not np.all(np.isfinite(s)):
        half = nfft // 2
        return np.arange(half + 1, dtype=np.float64) * fs / nfft, np.zeros(half + 1, dtype=np.float64)
    s = s - np.mean(s)
    if float(np.sum(np.abs(s) ** 2)) <= 1e-30:
        half = nfft // 2
        return np.arange(half + 1, dtype=np.float64) * fs / nfft, np.zeros(half + 1, dtype=np.float64)

    n_seg = min(int(nperseg), s.size)
    n_ov = min(int(noverlap), max(0, n_seg - 1))
    n_fft = max(int(nfft), n_seg)
    _, p_two = sig.welch(
        s,
        fs=float(fs),
        window="hann",
        nperseg=n_seg,
        noverlap=n_ov,
        nfft=n_fft,
        return_onesided=False,
        detrend=False,
        scaling="density",
    )
    p_two = np.real(p_two).astype(np.float64)
    half = n_fft // 2
    pf = p_two[: half + 1].copy()
    pf[1:half] += p_two[n_fft - 1 : half : -1]
    pf = np.maximum(pf, 0.0)
    freqs = np.arange(half + 1, dtype=np.float64) * float(fs) / float(n_fft)
    return freqs, pf


def _sample_and_max_normalize(freqs_hz: np.ndarray, psd: np.ndarray, grid_bpm: np.ndarray) -> np.ndarray:
    """Interpolate PSD onto candidate BPM grid and max-normalize to [0, 1]."""
    grid = np.asarray(grid_bpm, dtype=np.float64)
    vals = np.interp(grid / 60.0, freqs_hz, psd, left=0.0, right=0.0)
    vals = np.maximum(vals, 0.0)
    vmax = float(np.max(vals)) if vals.size else 0.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        return np.zeros_like(grid, dtype=np.float64)
    return vals / vmax


def extract_br_welch_spectrum(
    d_win: np.ndarray,
    grid_bpm: np.ndarray = BR_GRID_2BPM,
    fs: float = float(C.FS_RADAR),
    nperseg: int = 256,
    noverlap: int = 128,
) -> np.ndarray:
    """Extract max-normalized BR Welch evidence spectrum over `grid_bpm`."""
    freqs, pf = folded_welch_psd(d_win, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return _sample_and_max_normalize(freqs, pf, grid_bpm)


def extract_br_harmonic_nls_spectrum(
    d_win: np.ndarray,
    grid_bpm: np.ndarray = BR_GRID_1BPM,
    lk: int = int(C.NLS_LK),
    fs: float = float(C.FS_RADAR),
) -> np.ndarray:
    """Extract max-normalized harmonic-NLS score surface (Beltrao Eq. 17) over `grid_bpm`."""
    d = np.asarray(d_win, dtype=np.complex128).ravel()
    if d.size == 0 or not np.all(np.isfinite(d)) or float(np.sum(np.abs(d) ** 2)) <= 1e-30:
        return np.zeros(len(grid_bpm), dtype=np.float64)
    freqs, pf = psd_fft(d, nfft=int(C.NLS_NFFT), fs=float(fs))
    grid = np.asarray(grid_bpm, dtype=np.float64)
    score = np.zeros_like(grid, dtype=np.float64)
    for h in range(1, int(lk) + 1):
        score += np.interp(grid * (float(h) / 60.0), freqs, pf, left=0.0, right=0.0)
    score = np.maximum(score, 0.0)
    smax = float(np.max(score)) if score.size else 0.0
    if not np.isfinite(smax) or smax <= 0.0:
        return np.zeros_like(grid, dtype=np.float64)
    return score / smax


def extract_br_anchored_nls_spectrum(
    d_win: np.ndarray,
    f_coarse_hz: float,
    grid_bpm: np.ndarray = BR_GRID_1BPM,
    lk: int = int(C.NLS_LK),
    fs: float = float(C.FS_RADAR),
    radius_bpm: float = float(C.NLS_RADIUS_BPM),
    off_band_floor: float = 0.05,
) -> np.ndarray:
    """Autocorrelation-anchored Beltrao Eq. 17 harmonic-NLS score mitigating f0/2 aliasing.

    Weights S(b) = P(b) + P(2b) by a smooth Gaussian window centered on `f_coarse_hz * 60`
    (falling back to folded DFT peak if NaN) with an off-band floor >= 0.05 so Viterbi
    retains smooth log-gradients while suppressing subharmonic half-rate locks.
    """
    d = np.asarray(d_win, dtype=np.complex128).ravel()
    if d.size == 0 or not np.all(np.isfinite(d)) or float(np.sum(np.abs(d) ** 2)) <= 1e-30:
        return np.zeros(len(grid_bpm), dtype=np.float64)
    freqs, pf = psd_fft(d, nfft=int(C.NLS_NFFT), fs=float(fs))
    grid = np.asarray(grid_bpm, dtype=np.float64)
    score = np.zeros_like(grid, dtype=np.float64)
    for h in range(1, int(lk) + 1):
        score += np.interp(grid * (float(h) / 60.0), freqs, pf, left=0.0, right=0.0)

    if np.isfinite(f_coarse_hz):
        centre = float(f_coarse_hz) * 60.0
    else:
        centre, _, _ = dft_estimate(d, fs=float(fs))
    centre = float(np.clip(centre, grid[0], grid[-1]))
    gauss = np.exp(-0.5 * ((grid - centre) / float(radius_bpm)) ** 2)
    envelope = np.maximum(float(off_band_floor), gauss)
    score = np.maximum(score * envelope, 0.0)

    smax = float(np.max(score)) if score.size else 0.0
    if not np.isfinite(smax) or smax <= 0.0:
        return np.zeros_like(grid, dtype=np.float64)
    return score / smax


def extract_hr_csd_velocity_spectrum(
    x_win: np.ndarray,
    grid_hr_bpm: np.ndarray = HR_GRID_2BPM,
    fs: float = float(C.FS_RADAR),
    use_hpss: bool = False,
    nperseg: int = 256,
    noverlap: int = 128,
) -> np.ndarray:
    """Extract secondary cardiac (90-210 bpm) evidence from CSD complex velocity.

    Uses complex first-difference dv/dt = (x[m] - x[m-1]) * fs + zero-phase [1.5, 3.5] Hz
    Kaiser bandpass so the cardiac observation branch maintains strict invariance to amplitude
    scaling, global phase rotation, I/Q sign flip, and I/Q channel swap.
    """
    x = np.asarray(x_win, dtype=np.complex128).ravel()
    if x.size < 16 or not np.all(np.isfinite(x)):
        return np.zeros(len(grid_hr_bpm), dtype=np.float64)
    vx = np.diff(x) * float(fs)
    vx_bp = complex_bandpass(vx, 1.5, 3.5, fs=float(fs))
    if use_hpss:
        vx_bp = mitigate_hpss(vx_bp)["x_hat"]
        vx_bp = complex_bandpass(vx_bp, 1.5, 3.5, fs=float(fs))
    freqs, pf = folded_welch_psd(vx_bp, fs=float(fs), nperseg=nperseg, noverlap=noverlap)
    return _sample_and_max_normalize(freqs, pf, grid_hr_bpm)


def extract_hr_dacm_velocity_spectrum(
    x_win: np.ndarray,
    grid_hr_bpm: np.ndarray = HR_GRID_2BPM,
    fs: float = float(C.FS_RADAR),
    nperseg: int = 256,
    noverlap: int = 128,
) -> np.ndarray:
    """Extract secondary cardiac (90-210 bpm) evidence via scale-invariant DACM velocity."""
    x = np.asarray(x_win, dtype=np.complex128).ravel()
    if x.size < 16 or not np.all(np.isfinite(x)):
        return np.zeros(len(grid_hr_bpm), dtype=np.float64)
    xc = x - np.mean(x)
    p_mean = float(np.mean(np.abs(xc) ** 2))
    if not np.isfinite(p_mean) or p_mean <= 1e-30:
        return np.zeros(len(grid_hr_bpm), dtype=np.float64)
    omega = (
        np.imag(np.conj(xc[:-1]) * np.diff(xc))
        / (np.abs(xc[:-1]) ** 2 + 1e-12 * p_mean)
        * float(fs)
    )
    omega_bp = complex_bandpass(omega, 1.5, 3.5, fs=float(fs))
    freqs, pf = folded_welch_psd(omega_bp, fs=float(fs), nperseg=nperseg, noverlap=noverlap)
    return _sample_and_max_normalize(freqs, pf, grid_hr_bpm)


def extract_hr_br_notched_velocity_spectrum(
    x_win: np.ndarray,
    br_est_bpm: float,
    grid_hr_bpm: np.ndarray = HR_GRID_2BPM,
    fs: float = float(C.FS_RADAR),
    max_harmonic: int = 5,
    nperseg: int = 256,
    noverlap: int = 128,
) -> np.ndarray:
    """Extract secondary HR evidence after two-sided LS projection of radar BR harmonics."""
    x = np.asarray(x_win, dtype=np.complex128).ravel()
    if x.size < 16 or not np.all(np.isfinite(x)):
        return np.zeros(len(grid_hr_bpm), dtype=np.float64)
    xc = x - np.mean(x)
    if float(np.sum(np.abs(xc) ** 2)) <= 1e-30:
        return np.zeros(len(grid_hr_bpm), dtype=np.float64)

    x_res = xc
    if np.isfinite(br_est_bpm) and float(br_est_bpm) > 3.0:
        f_br = float(br_est_bpm) / 60.0
        t_sec = np.arange(xc.size, dtype=np.float64) / float(fs)
        cols = []
        for k in range(1, int(max_harmonic) + 1):
            fk = float(k) * f_br
            if fk < 0.5 * float(fs) - 1e-6:
                cols.append(np.exp(2j * np.pi * fk * t_sec))
                cols.append(np.exp(-2j * np.pi * fk * t_sec))
        if cols:
            basis = np.column_stack(cols)
            coef, *_ = np.linalg.lstsq(basis, xc, rcond=None)
            x_res = xc - basis @ coef

    vx = np.diff(x_res) * float(fs)
    vx_bp = complex_bandpass(vx, 1.5, 3.5, fs=float(fs))
    freqs, pf = folded_welch_psd(vx_bp, fs=float(fs), nperseg=nperseg, noverlap=noverlap)
    return _sample_and_max_normalize(freqs, pf, grid_hr_bpm)


def legacy_ad_session_pipeline(
    bI: np.ndarray,
    bQ: np.ndarray,
    fs: float = float(C.FS_RADAR),
    win_s: int = int(C.WIN_S),
    update_s: int = int(C.UPDATE_S),
) -> dict[str, np.ndarray]:
    """Diagnostic reproduction of attached draft Sec. 4.2-4.3 (AD phase unwrap + raw NMF).

    Retained strictly as an isolated forensic bridge to quantify why the draft's
    `Naive FFT` (~17.47 bpm) and `No Joint 2D Viterbi` (~17.40 bpm) underperformed.
    """
    i_arr = np.asarray(bI, dtype=np.float64).ravel()
    q_arr = np.asarray(bQ, dtype=np.float64).ravel()
    n_samp = min(i_arr.size, q_arr.size)
    i_arr, q_arr = i_arr[:n_samp], q_arr[:n_samp]

    finite = np.isfinite(i_arr) & np.isfinite(q_arr)
    i_d = i_arr - float(np.mean(i_arr[finite])) if np.any(finite) else np.zeros_like(i_arr)
    q_d = q_arr - float(np.mean(q_arr[finite])) if np.any(finite) else np.zeros_like(q_arr)

    g_i = float(np.sqrt(2.0 * np.mean(i_d[finite] ** 2))) if np.any(finite) else 1.0
    g_q = float(np.sqrt(2.0 * np.mean(q_d[finite] ** 2))) if np.any(finite) else 1.0
    g_i = g_i if g_i > 1e-12 else 1.0
    g_q = g_q if g_q > 1e-12 else 1.0
    i_n = i_d / g_i
    q_n = q_d / g_q

    if np.sum(finite) > 2 and float(np.std(i_n[finite])) > 0.0 and float(np.std(q_n[finite])) > 0.0:
        rho = float(np.clip(np.corrcoef(i_n[finite], q_n[finite])[0, 1], -0.999, 0.999))
    else:
        rho = 0.0
    if abs(rho) > 0.5:
        q_dec = (q_n - rho * i_n) / float(np.sqrt(max(1.0 - rho * rho, 1e-12)))
    else:
        q_dec = q_n

    sc = i_n + 1j * q_dec
    theta = np.unwrap(np.angle(sc))
    pv = np.diff(theta) * float(fs)

    # Session-level STFT + raw unnormalized Frobenius NMF + Hoyer RBM rejection (Sec. 4.3)
    _, mag, phase = stft_complex(theta)
    mag_pos = np.maximum(mag, 0.0)
    model = NMF(
        n_components=int(C.NMF_K),
        init="nndsvda",
        tol=1e-4,
        max_iter=2000,
        random_state=int(C.NMF_RANDOM_STATE),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        W = model.fit_transform(mag_pos)
    H = model.components_
    keep = detect_rbm_hoyer(H, s_thresh=0.55, amp_factor=2.0)
    mag_filt = (W[:, keep] @ H[keep, :]) if np.any(keep) else np.zeros_like(mag_pos)
    theta_nmf = np.real(istft_complex(mag_filt * np.exp(1j * phase), theta.size))

    theta_bp_raw = complex_bandpass(theta, 0.3, 1.5, fs=float(fs))
    theta_bp_nmf = complex_bandpass(theta_nmf, 0.3, 1.5, fs=float(fs))
    pv_bp = complex_bandpass(pv, 1.5, 3.5, fs=float(fs))

    w_samp = int(round(win_s * fs))
    step = int(round(update_s * fs))
    starts = list(range(0, n_samp - w_samp + 1, step))
    n_win = len(starts)

    naive_fft = np.full(n_win, np.nan, dtype=np.float64)
    nmf_nojv = np.full(n_win, np.nan, dtype=np.float64)
    p_hr_ad = np.zeros((n_win, HR_GRID_2BPM.size), dtype=np.float64)

    for idx_w, s0 in enumerate(starts):
        s1 = s0 + w_samp
        p_raw = extract_br_welch_spectrum(theta_bp_raw[s0:s1], BR_GRID_2BPM, fs=float(fs))
        p_nmf = extract_br_welch_spectrum(theta_bp_nmf[s0:s1], BR_GRID_2BPM, fs=float(fs))
        naive_fft[idx_w] = float(BR_GRID_2BPM[int(np.argmax(p_raw))]) if float(np.max(p_raw)) > 0 else np.nan
        nmf_nojv[idx_w] = float(BR_GRID_2BPM[int(np.argmax(p_nmf))]) if float(np.max(p_nmf)) > 0 else np.nan
        pv_seg = pv_bp[s0 : min(s1 - 1, pv_bp.size)]
        p_hr_ad[idx_w] = extract_br_welch_spectrum(pv_seg, HR_GRID_2BPM, fs=float(fs))

    return {
        "diag_ad_naive_fft": naive_fft,
        "diag_ad_nmf_nojv": nmf_nojv,
        "P_hr_ad_diag_2bpm": p_hr_ad,
        "n_removed_session": np.array(int(np.sum(~keep)), dtype=np.int64),
    }


def legacy_ad_sensitivity_pipeline(
    bI: np.ndarray,
    bQ: np.ndarray,
    fs: float = float(C.FS_RADAR),
    win_s: int = int(C.WIN_S),
    update_s: int = int(C.UPDATE_S),
    detrend_before_nmf: bool = True,
    bandpass_before_nmf: bool = True,
    nmf_init: str = "nndsvd",
) -> dict[str, np.ndarray]:
    """Audit sensitivity variant of legacy AD + session NMF testing drift & NMF init."""
    i_arr = np.asarray(bI, dtype=np.float64).ravel()
    q_arr = np.asarray(bQ, dtype=np.float64).ravel()
    n_samp = min(i_arr.size, q_arr.size)
    i_arr, q_arr = i_arr[:n_samp], q_arr[:n_samp]

    finite = np.isfinite(i_arr) & np.isfinite(q_arr)
    i_d = i_arr - float(np.mean(i_arr[finite])) if np.any(finite) else np.zeros_like(i_arr)
    q_d = q_arr - float(np.mean(q_arr[finite])) if np.any(finite) else np.zeros_like(q_arr)

    g_i = float(np.sqrt(2.0 * np.mean(i_d[finite] ** 2))) if np.any(finite) else 1.0
    g_q = float(np.sqrt(2.0 * np.mean(q_d[finite] ** 2))) if np.any(finite) else 1.0
    g_i = g_i if g_i > 1e-12 else 1.0
    g_q = g_q if g_q > 1e-12 else 1.0
    i_n = i_d / g_i
    q_n = q_d / g_q

    if np.sum(finite) > 2 and float(np.std(i_n[finite])) > 0.0 and float(np.std(q_n[finite])) > 0.0:
        rho = float(np.clip(np.corrcoef(i_n[finite], q_n[finite])[0, 1], -0.999, 0.999))
    else:
        rho = 0.0
    if abs(rho) > 0.5:
        q_dec = (q_n - rho * i_n) / float(np.sqrt(max(1.0 - rho * rho, 1e-12)))
    else:
        q_dec = q_n

    theta = np.unwrap(np.angle(i_n + 1j * q_dec))
    theta_in = theta.copy()
    if detrend_before_nmf and theta_in.size > 2:
        theta_in = sig.detrend(theta_in, type="linear")
    if bandpass_before_nmf:
        theta_in = np.real(complex_bandpass(theta_in, 0.3, 1.5, fs=float(fs)))

    _, mag, phase = stft_complex(theta_in)
    mag_pos = np.maximum(mag, 0.0)
    model = NMF(
        n_components=int(C.NMF_K),
        init=str(nmf_init),
        tol=1e-4,
        max_iter=2000,
        random_state=int(C.NMF_RANDOM_STATE),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        W = model.fit_transform(mag_pos)
    H = model.components_
    keep = detect_rbm_hoyer(H, s_thresh=0.55, amp_factor=2.0)
    mag_filt = (W[:, keep] @ H[keep, :]) if np.any(keep) else np.zeros_like(mag_pos)
    theta_nmf = np.real(istft_complex(mag_filt * np.exp(1j * phase), theta_in.size))
    theta_bp_nmf = complex_bandpass(theta_nmf, 0.3, 1.5, fs=float(fs))

    w_samp = int(round(win_s * fs))
    step = int(round(update_s * fs))
    starts = list(range(0, n_samp - w_samp + 1, step))
    nmf_sens = np.full(len(starts), np.nan, dtype=np.float64)
    for idx_w, s0 in enumerate(starts):
        p_nmf = extract_br_welch_spectrum(theta_bp_nmf[s0 : s0 + w_samp], BR_GRID_2BPM, fs=float(fs))
        nmf_sens[idx_w] = float(BR_GRID_2BPM[int(np.argmax(p_nmf))]) if float(np.max(p_nmf)) > 0 else np.nan

    return {
        "diag_ad_nmf_sens": nmf_sens,
        "n_removed_session": np.array(int(np.sum(~keep)), dtype=np.int64),
    }


def compute_hqi_matrix(
    P_hr_norm: np.ndarray,
    hr_grid_bpm: np.ndarray = HR_GRID_2BPM,
    br_grid_bpm: np.ndarray = BR_GRID_2BPM,
    m_h: float = 0.001,
    eps_h: float = 1e-300,
) -> np.ndarray:
    """Compute exact alternative-aware Harmonic Quality Index H_t(i, j) (paper Eq. 5)."""
    p_hr = np.maximum(np.asarray(P_hr_norm, dtype=np.float64).ravel(), 0.0)
    hr = np.asarray(hr_grid_bpm, dtype=np.float64).ravel()
    br = np.asarray(br_grid_bpm, dtype=np.float64).ravel()

    n_h, n_b = hr.size, br.size
    if p_hr.size != n_h:
        raise ValueError(f"P_hr_norm length {p_hr.size} != hr_grid_bpm length {n_h}")
    if n_h == 0 or n_b == 0 or float(np.max(p_hr)) <= 0.0:
        return np.zeros((n_h, n_b), dtype=np.float64)

    r_ij = hr[:, None] / (br[None, :] + 1e-6)
    q_hr = float(np.max(p_hr)) / (float(np.mean(p_hr)) + 1e-10)
    a_t = (5.0 + 20.0 * float(np.clip((q_hr - 2.0) / 18.0, 0.0, 1.0))) * float(m_h) * 10.0

    # Harmonic neighborhood: |r_ij - n| < 0.10 for n in {2, 3, 4, 5}
    diff = np.abs(r_ij[:, :, None] - _HARM_ORDERS[None, None, :])
    harm_mask = np.any(diff < 0.10, axis=2)  # (N_H, N_B)

    p_col = np.broadcast_to(p_hr[:, None], (n_h, n_b))
    has_harm = np.any(harm_mask, axis=0)
    has_nh = np.any(~harm_mask, axis=0)

    p_harm = np.where(has_harm, np.max(np.where(harm_mask, p_col, -np.inf), axis=0), 0.0)
    p_nh = np.where(has_nh, np.max(np.where(~harm_mask, p_col, -np.inf), axis=0), 0.0)
    p_harm = np.maximum(p_harm, 0.0)
    p_nh = np.maximum(p_nh, 0.0)

    contrast = np.clip(p_nh / (p_harm + float(eps_h)), 0.0, 1.0)
    gauss = np.sum(
        np.exp(-0.5 * ((r_ij[:, :, None] - _HARM_ORDERS[None, None, :]) / 0.05) ** 2),
        axis=2,
    )
    return contrast[None, :] * a_t * gauss


def compute_hqi_1d_penalty(
    P_hr_norm: np.ndarray,
    hr_grid_bpm: np.ndarray = HR_GRID_2BPM,
    br_grid_bpm: np.ndarray = BR_GRID_2BPM,
    m_h: float = 0.001,
    eps_h: float = 1e-300,
) -> np.ndarray:
    """Slice Eq. 5 HQI at the dominant HR spectral peak for 1D-HQI ablation."""
    p_hr = np.asarray(P_hr_norm, dtype=np.float64).ravel()
    if p_hr.size == 0 or float(np.max(p_hr)) <= 0.0:
        return np.zeros(len(br_grid_bpm), dtype=np.float64)
    h_mat = compute_hqi_matrix(p_hr, hr_grid_bpm=hr_grid_bpm, br_grid_bpm=br_grid_bpm, m_h=m_h, eps_h=eps_h)
    i_star = int(np.argmax(p_hr))
    return h_mat[i_star, :]


def compute_transparent_aux_prior(
    candidates_t: list[tuple[float, float]] | np.ndarray,
    hr_rough_t: float,
    br_grid_bpm: np.ndarray = BR_GRID_2BPM,
    sigma_s: float = 2.0,
    w_s: float = 1.5,
) -> np.ndarray:
    """Reproducible baseline-view candidate prior S_t(j) (attached paper Eq. 7)."""
    grid = np.asarray(br_grid_bpm, dtype=np.float64).ravel()
    prior = np.zeros_like(grid, dtype=np.float64)
    for c_raw, p_raw in candidates_t:
        c = float(c_raw)
        pc = float(p_raw)
        if not (np.isfinite(c) and np.isfinite(pc)) or c <= 0.0:
            continue
        if np.isfinite(hr_rough_t) and abs(c - 0.5 * float(hr_rough_t)) <= 4.0:
            pc *= 0.1
        if pc < 0.1:
            continue
        pc = min(pc, 1.0)
        prior += float(w_s) * np.exp(-0.5 * ((grid - c) / float(sigma_s)) ** 2) * pc
    return prior


def decode_viterbi_1d(
    P_br_seq: np.ndarray,
    br_grid_bpm: np.ndarray = BR_GRID_2BPM,
    sigma_br: float = 4.0,
    w_br: float = 1.0,
    eps_p: float = 1e-300,
    hqi_1d_seq: np.ndarray | None = None,
    w_hqi: float = 4.0,
    aux_prior_seq: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Exact 1D BR max-sum Viterbi decoder isolating temporal & HQI contributions."""
    p_br = np.maximum(np.asarray(P_br_seq, dtype=np.float64), 0.0)
    br = np.asarray(br_grid_bpm, dtype=np.float64).ravel()
    n_t, n_b = p_br.shape
    if br.size != n_b:
        raise ValueError(f"P_br_seq width {n_b} != br_grid_bpm length {br.size}")
    if n_t == 0:
        return np.empty(0, dtype=np.float64), 0.0

    psi_br = -0.5 * ((br[:, None] - br[None, :]) / float(sigma_br)) ** 2
    v_prev = float(w_br) * np.log(p_br[0] + float(eps_p))
    if aux_prior_seq is not None:
        v_prev = v_prev + np.asarray(aux_prior_seq[0], dtype=np.float64)

    back = np.zeros((n_t, n_b), dtype=np.int32)
    for t in range(1, n_t):
        scores = v_prev[:, None] + psi_br  # (prev_j, curr_j)
        bp = np.argmax(scores, axis=0)
        back[t] = bp
        best_prev = scores[bp, np.arange(n_b)]
        emit = float(w_br) * np.log(p_br[t] + float(eps_p))
        if hqi_1d_seq is not None:
            emit = emit - float(w_hqi) * np.asarray(hqi_1d_seq[t], dtype=np.float64)
        if aux_prior_seq is not None:
            emit = emit + np.asarray(aux_prior_seq[t], dtype=np.float64)
        v_prev = emit + best_prev

    j_cur = int(np.argmax(v_prev))
    best_score = float(v_prev[j_cur])
    path_idx = np.empty(n_t, dtype=np.int32)
    path_idx[-1] = j_cur
    for t in range(n_t - 1, 0, -1):
        j_cur = int(back[t, j_cur])
        path_idx[t - 1] = j_cur
    return br[path_idx], best_score


def precompute_hqi_sequence(
    P_hr_seq: np.ndarray,
    hr_grid_bpm: np.ndarray = HR_GRID_2BPM,
    br_grid_bpm: np.ndarray = BR_GRID_2BPM,
    m_h: float = 0.001,
    eps_h: float = 1e-300,
) -> np.ndarray:
    """Precompute Eq. 5 HQI tensor (T, N_H, N_B) once per session for fast sweeps."""
    p_hr = np.asarray(P_hr_seq, dtype=np.float64)
    n_t = p_hr.shape[0]
    n_h = len(hr_grid_bpm)
    n_b = len(br_grid_bpm)
    out = np.empty((n_t, n_h, n_b), dtype=np.float64)
    for t in range(n_t):
        out[t] = compute_hqi_matrix(
            p_hr[t], hr_grid_bpm=hr_grid_bpm, br_grid_bpm=br_grid_bpm, m_h=m_h, eps_h=eps_h
        )
    return out


def decode_viterbi_2d_joint(
    P_br_seq: np.ndarray,
    P_hr_seq: np.ndarray,
    br_grid_bpm: np.ndarray = BR_GRID_2BPM,
    hr_grid_bpm: np.ndarray = HR_GRID_2BPM,
    w_br: float = 0.5,
    w_hqi: float = 4.0,
    sigma_br: float = 4.0,
    sigma_hr: float = 12.0,
    eps_p: float = 1e-300,
    aux_prior_seq: np.ndarray | None = None,
    use_hqi: bool = True,
    hqi_matrix_seq: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact separable 2-pass 2D HR-BR max-sum Viterbi decoder (paper Eqs. 6, 8, 9, 10).

    Exploits the exact additive separability of Psi(x_t, x_{t-1}) = Psi_HR + Psi_BR
    to reduce per-frame complexity from O(N_H^2 N_B^2) to O(N_H N_B (N_H + N_B))
    with zero mathematical approximation. Uses C-contiguous last-axis reductions
    and preallocated buffers for speed.
    """
    p_br = np.maximum(np.asarray(P_br_seq, dtype=np.float64), 0.0)
    p_hr = np.maximum(np.asarray(P_hr_seq, dtype=np.float64), 0.0)
    br = np.asarray(br_grid_bpm, dtype=np.float64).ravel()
    hr = np.asarray(hr_grid_bpm, dtype=np.float64).ravel()

    n_t, n_b = p_br.shape
    n_h = hr.size
    if p_hr.shape != (n_t, n_h):
        raise ValueError(f"P_hr_seq shape {p_hr.shape} != ({n_t}, {n_h})")
    if br.size != n_b:
        raise ValueError(f"br_grid_bpm length {br.size} != {n_b}")
    if n_t == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), 0.0
    if use_hqi and hqi_matrix_seq is not None and hqi_matrix_seq.shape != (n_t, n_h, n_b):
        raise ValueError(f"hqi_matrix_seq shape {hqi_matrix_seq.shape} != ({n_t}, {n_h}, {n_b})")

    # Initial frame score Phi_1 (Eq. 8)
    r_ij = hr[:, None] / (br[None, :] + 1e-6)
    r_prior = -0.5 * ((r_ij - 3.5) / 2.5) ** 2
    v = (
        np.log(p_hr[0][:, None] + float(eps_p))
        + np.log(p_br[0][None, :] + float(eps_p))
        + r_prior
    )
    if aux_prior_seq is not None:
        v = v + np.asarray(aux_prior_seq[0], dtype=np.float64)[None, :]

    # C-contiguous transition matrices: axis 1 is previous state for last-axis max
    psi_br = -0.5 * ((br[:, None] - br[None, :]) / float(sigma_br)) ** 2  # (B_curr, B_prev)
    psi_hr = -0.5 * ((hr[:, None] - hr[None, :]) / float(sigma_hr)) ** 2  # (H_curr, H_prev)

    bp_br = np.zeros((n_t, n_h, n_b), dtype=np.int16)  # (t, i_prev, j_curr) -> j_prev
    bp_hr = np.zeros((n_t, n_b, n_h), dtype=np.int16)  # (t, j_curr, i_curr) -> i_prev

    buf_br = np.empty((n_h, n_b, n_b), dtype=np.float64)
    buf_hr = np.empty((n_b, n_h, n_h), dtype=np.float64)
    log_hr = np.log(p_hr + float(eps_p))
    log_br = np.log(p_br + float(eps_p))

    for t in range(1, n_t):
        # Pass 1: maximize over j' (last axis) for each (i', j)
        np.add(v[:, None, :], psi_br[None, :, :], out=buf_br)
        arg_b = np.argmax(buf_br, axis=2)  # (H_prev, B_curr)
        bp_br[t] = arg_b
        mid = np.take_along_axis(buf_br, arg_b[:, :, None], axis=2)[:, :, 0]  # (H_prev, B_curr)

        # Pass 2: maximize over i' (last axis) for each (j, i)
        mid_t = np.ascontiguousarray(mid.T)  # (B_curr, H_prev)
        np.add(mid_t[:, None, :], psi_hr[None, :, :], out=buf_hr)
        arg_h = np.argmax(buf_hr, axis=2)  # (B_curr, H_curr)
        bp_hr[t] = arg_h
        best_prev = np.take_along_axis(buf_hr, arg_h[:, :, None], axis=2)[:, :, 0].T  # (H_curr, B_curr)

        emit = log_hr[t][:, None] + float(w_br) * log_br[t][None, :]
        if use_hqi:
            hqi_t = (
                hqi_matrix_seq[t]
                if hqi_matrix_seq is not None
                else compute_hqi_matrix(p_hr[t], hr_grid_bpm=hr, br_grid_bpm=br)
            )
            emit = emit - float(w_hqi) * hqi_t
        if aux_prior_seq is not None:
            emit = emit + np.asarray(aux_prior_seq[t], dtype=np.float64)[None, :]
        v = emit + best_prev

    flat_idx = int(np.argmax(v))
    i_cur, j_cur = np.unravel_index(flat_idx, v.shape)
    best_score = float(v[i_cur, j_cur])

    br_idx = np.empty(n_t, dtype=np.int32)
    hr_idx = np.empty(n_t, dtype=np.int32)
    br_idx[-1] = int(j_cur)
    hr_idx[-1] = int(i_cur)
    for t in range(n_t - 1, 0, -1):
        i_prev = int(bp_hr[t, j_cur, i_cur])
        j_prev = int(bp_br[t, i_prev, j_cur])
        i_cur, j_cur = i_prev, j_prev
        hr_idx[t - 1] = i_cur
        br_idx[t - 1] = j_cur
    return br[br_idx], hr[hr_idx], best_score


def decode_viterbi_2d_bruteforce(
    P_br_seq: np.ndarray,
    P_hr_seq: np.ndarray,
    br_grid_bpm: np.ndarray,
    hr_grid_bpm: np.ndarray,
    w_br: float = 0.5,
    w_hqi: float = 4.0,
    sigma_br: float = 4.0,
    sigma_hr: float = 12.0,
    eps_p: float = 1e-300,
    aux_prior_seq: np.ndarray | None = None,
    use_hqi: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Unfactored O(T N_H^2 N_B^2) reference 2D Viterbi solver for unit-test parity."""
    p_br = np.maximum(np.asarray(P_br_seq, dtype=np.float64), 0.0)
    p_hr = np.maximum(np.asarray(P_hr_seq, dtype=np.float64), 0.0)
    br = np.asarray(br_grid_bpm, dtype=np.float64).ravel()
    hr = np.asarray(hr_grid_bpm, dtype=np.float64).ravel()

    n_t, n_b = p_br.shape
    n_h = hr.size
    n_s = n_h * n_b

    r_ij = hr[:, None] / (br[None, :] + 1e-6)
    v0 = (
        np.log(p_hr[0][:, None] + float(eps_p))
        + np.log(p_br[0][None, :] + float(eps_p))
        - 0.5 * ((r_ij - 3.5) / 2.5) ** 2
    )
    if aux_prior_seq is not None:
        v0 = v0 + np.asarray(aux_prior_seq[0], dtype=np.float64)[None, :]
    v_prev = v0.reshape(n_s)

    ii, jj = np.meshgrid(np.arange(n_h), np.arange(n_b), indexing="ij")
    h_flat = hr[ii.ravel()]
    b_flat = br[jj.ravel()]
    psi_full = (
        -0.5 * ((h_flat[None, :] - h_flat[:, None]) / float(sigma_hr)) ** 2
        - 0.5 * ((b_flat[None, :] - b_flat[:, None]) / float(sigma_br)) ** 2
    )  # (prev_s, curr_s)

    back = np.zeros((n_t, n_s), dtype=np.int32)
    for t in range(1, n_t):
        scores = v_prev[:, None] + psi_full
        bp = np.argmax(scores, axis=0)
        back[t] = bp
        best_prev = scores[bp, np.arange(n_s)]
        emit = np.log(p_hr[t][:, None] + float(eps_p)) + float(w_br) * np.log(p_br[t][None, :] + float(eps_p))
        if use_hqi:
            emit = emit - float(w_hqi) * compute_hqi_matrix(p_hr[t], hr_grid_bpm=hr, br_grid_bpm=br)
        if aux_prior_seq is not None:
            emit = emit + np.asarray(aux_prior_seq[t], dtype=np.float64)[None, :]
        v_prev = emit.reshape(n_s) + best_prev

    s_cur = int(np.argmax(v_prev))
    best_score = float(v_prev[s_cur])
    br_idx = np.empty(n_t, dtype=np.int32)
    hr_idx = np.empty(n_t, dtype=np.int32)
    for t in range(n_t - 1, -1, -1):
        i_c, j_c = np.unravel_index(s_cur, (n_h, n_b))
        hr_idx[t] = int(i_c)
        br_idx[t] = int(j_c)
        if t > 0:
            s_cur = int(back[t, s_cur])
    return br[br_idx], hr[hr_idx], best_score
