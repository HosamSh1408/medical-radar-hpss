"""Sliding-window runner: 30 s window, 28 s overlap (update every 2 s).

Branch A (Beltrao baseline): NMF RBM mitigation -> NLS.
Branch B (HPSS extension): median-filter HPSS -> identical downstream NLS.
Shared upstream (CSD/STFT) and downstream (iSTFT/bandpass/NLS) for fair A/B.
"""
import time
import numpy as np
from . import config as C
from .demod import csd
from .rbm_nmf import mitigate
from .hpss import mitigate_hpss
from .filtering import bandpass, autocorr_coarse
from .estimation import dft_estimate, nls_estimate


def process_window(bI_win, bQ_win):
    x = csd(bI_win, bQ_win)
    d_raw = bandpass(x)
    t0 = time.perf_counter()
    rbm = mitigate(d_raw)
    t_nmf = time.perf_counter() - t0
    x_hat = rbm["x_hat"]
    d_hat = bandpass(x_hat)
    f_coarse = autocorr_coarse(d_hat)
    f_nls, _, _ = nls_estimate(d_hat, f_coarse)
    # Branch B: identical downstream, HPSS separation instead of NMF
    t1 = time.perf_counter()
    hpss = mitigate_hpss(d_raw)
    t_hpss = time.perf_counter() - t1
    d_hpss = bandpass(hpss["x_hat"])
    f_coarse_h = autocorr_coarse(d_hpss)
    f_hpss, _, _ = nls_estimate(d_hpss, f_coarse_h)
    # Unmitigated baselines on the same bandpassed complex signal d_raw:
    f_dft, _, _ = dft_estimate(d_raw)
    f_nls_nonmf, _, _ = nls_estimate(d_raw, autocorr_coarse(d_raw))
    return {
        "f_dft": f_dft, "f_nls": f_nls_nonmf, "f_nls_nmf": f_nls,
        "f_hpss": f_hpss, "f_coarse_h": f_coarse_h * 60 if np.isfinite(f_coarse_h) else np.nan,
        "f_coarse": f_coarse * 60 if np.isfinite(f_coarse) else np.nan,
        "n_removed": rbm["n_removed"], "kept_frac": hpss["kept_frac"],
        "nmf_converged": rbm["nmf_converged"],
        "t_nmf": t_nmf, "t_hpss": t_hpss,
    }


def run(bI, bQ, fs=C.FS_RADAR, win_s=C.WIN_S, update_s=C.UPDATE_S):
    bI = np.asarray(bI, dtype=float)
    bQ = np.asarray(bQ, dtype=float)
    W = win_s * fs
    S = update_s * fs
    out = []
    for s in range(0, len(bI) - W + 1, S):
        r = process_window(bI[s:s + W], bQ[s:s + W])
        r["t_start_s"] = s / fs
        out.append(r)
    return out
