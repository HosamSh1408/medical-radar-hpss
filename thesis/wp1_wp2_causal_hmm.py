"""WP1 (Front-End Symmetry), WP2 (Causal COLA-Hann HPSS), and Rung 1 (Causal/Online HMM).

All mathematical symbols and module docstrings conform strictly to
`paper_orchestra_bundle/raw_materials/notation_glossary.tex`.
Never mutates any locked `paper_workspace/` source or cache file.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import median_filter
from scipy.special import logsumexp

from paper_workspace.src import config as C
from paper_workspace.src.demod import csd
from thesis import viterbi_tracker as VT


# ---------------------------------------------------------------------------
# WP1: Lean Quadrature Front-End Operators (Hardware Clip Masking & ZCA)
# ---------------------------------------------------------------------------

def detect_adc_rail_mask(
    bI: np.ndarray,
    bQ: np.ndarray,
    low: float = -1024.0,
    high: float = 1023.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect 11-bit signed ADC saturation rail hits matching `exp0_forensics.py`."""
    bi = np.asarray(bI, dtype=np.float64).ravel()
    bq = np.asarray(bQ, dtype=np.float64).ravel()
    clip_i = (~np.isfinite(bi)) | (bi <= float(low)) | (bi >= float(high))
    clip_q = (~np.isfinite(bq)) | (bq <= float(low)) | (bq >= float(high))
    return clip_i, clip_q, (clip_i | clip_q)


def interp_masked_1d(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linearly interpolate masked samples across a 1D window with constant edge fill."""
    x = np.asarray(arr, dtype=np.float64).ravel().copy()
    m = np.asarray(mask, dtype=bool).ravel()
    if x.size == 0 or not np.any(m):
        return x
    valid = (~m) & np.isfinite(x)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return np.zeros_like(x, dtype=np.float64)
    if n_valid == 1:
        val = float(x[valid][0])
        x[~valid] = val
        return x
    idx_all = np.arange(x.size, dtype=np.float64)
    x[m] = np.interp(idx_all[m], idx_all[valid], x[valid])
    return x


def csd_clip_masked(
    bI: np.ndarray,
    bQ: np.ndarray,
    mode: str = "channel",
    clip_mask: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
    low: float = -1024.0,
    high: float = 1023.0,
) -> np.ndarray:
    """Apply ADC rail masking before locked mean-centered complex signal demodulation `csd()`.

    Parameters
    ----------
    mode:
        `"channel"` interpolates I and Q independently (preserving clean Q when only I clips).
        `"joint"` applies a shared `clip_any` mask to both channels (preserving exact
        $O(2) \\times \\mathbb{R}_{>0}$ equivariance given an external hardware mask).
    """
    bi = np.asarray(bI, dtype=np.float64).ravel()
    bq = np.asarray(bQ, dtype=np.float64).ravel()
    if clip_mask is None:
        clip_i, clip_q, clip_any = detect_adc_rail_mask(bi, bq, low=low, high=high)
    elif isinstance(clip_mask, tuple):
        clip_i, clip_q = clip_mask
        clip_any = np.asarray(clip_i, dtype=bool) | np.asarray(clip_q, dtype=bool)
    else:
        clip_any = np.asarray(clip_mask, dtype=bool).ravel()
        clip_i = clip_any
        clip_q = clip_any

    if mode == "channel":
        bi_clean = interp_masked_1d(bi, clip_i)
        bq_clean = interp_masked_1d(bq, clip_q)
    elif mode == "joint":
        bi_clean = interp_masked_1d(bi, clip_any)
        bq_clean = interp_masked_1d(bq, clip_any)
    else:
        raise ValueError(f"Unsupported clip_mask mode: {mode!r}")

    return csd(bi_clean, bq_clean)


def csd_sym_whitened(
    bI: np.ndarray,
    bQ: np.ndarray,
    lam: float = 0.10,
) -> np.ndarray:
    """Symmetry-preserving shrinkage ZCA I/Q whitening.

    Because $C_\\lambda = (1-\\lambda)C + \\lambda(\\tfrac{1}{2}\\operatorname{tr}C)I_2$ is
    orthogonal-equivariant, $(R C_\\lambda R^T)^{-1/2} = R C_\\lambda^{-1/2} R^T$ for every
    $R \\in O(2)$. Thus `csd_sym_whitened` is strictly $O(2) \\times \\mathbb{R}_{>0}$-equivariant:
    scaling -> exact invariance; phase rotation $e^{j\\phi}$ -> exact $e^{j\\phi}$;
    I/Q sign flip -> sign flip; I/Q channel swap -> $j \\overline{x(t)}$ (folded PSD invariant).
    """
    bi = np.asarray(bI, dtype=np.float64).ravel()
    bq = np.asarray(bQ, dtype=np.float64).ravel()
    if bi.size == 0 or bi.size != bq.size:
        raise ValueError("bI and bQ must be non-empty arrays of equal length")

    ui = bi - float(np.mean(bi))
    uq = bq - float(np.mean(bq))
    u = np.column_stack([ui, uq])  # (N, 2)
    cov = (u.T @ u) / float(bi.size)
    tr = float(np.trace(cov))
    if not np.isfinite(tr) or tr <= 1e-30:
        return np.zeros(bi.size, dtype=np.complex128)

    shrink = float(np.clip(lam, 0.0, 1.0))
    mu = 0.5 * tr
    c_lam = (1.0 - shrink) * cov + shrink * mu * np.eye(2, dtype=np.float64)

    eigvals, eigvecs = np.linalg.eigh(c_lam)
    eigvals = np.maximum(eigvals, 1e-12 * mu)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    w = u @ inv_sqrt.T
    return (w[:, 0] + 1j * w[:, 1]).astype(np.complex128)


def csd_clip_and_whiten(
    bI: np.ndarray,
    bQ: np.ndarray,
    lam: float = 0.10,
    mode: str = "channel",
    clip_mask: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Apply ADC rail masking first, then $O(2)$-equivariant shrinkage ZCA whitening."""
    x_clip = csd_clip_masked(bI, bQ, mode=mode, clip_mask=clip_mask)
    return csd_sym_whitened(np.real(x_clip), np.imag(x_clip), lam=lam)


# ---------------------------------------------------------------------------
# WP2: COLA-Hann STFT Synthesis & Strictly Causal / In-Band HPSS
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def cola_hann_window(win: int = int(C.STFT_WIN)) -> np.ndarray:
    """Strictly positive half-sample periodic Hann window with exact quadratic COLA.

    For `win=48, hop=16`: $\\sum_m w[n - mH] = 1.5$ and $\\sum_m w^2[n - mH] = 1.125$.
    """
    n = np.arange(int(win), dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * (n + 0.5) / float(win)))


def stft_complex_windowed(
    x: np.ndarray,
    win: int = int(C.STFT_WIN),
    hop: int = int(C.STFT_HOP),
    nfft: int = int(C.STFT_NFFT),
    window: str = "cola_hann",
    pad_edges: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Two-sided complex STFT supporting locked `rect` and padded `cola_hann`."""
    arr = np.asarray(x, dtype=np.complex128).ravel()
    orig_len = int(arr.size)
    win_i, hop_i, nfft_i = int(win), int(hop), int(nfft)

    if window == "rect":
        work = np.pad(arr, (0, win_i - orig_len)) if orig_len < win_i else arr
        frames = np.lib.stride_tricks.sliding_window_view(work, win_i)[::hop_i].T  # (win, T)
        X = np.fft.fft(frames, n=nfft_i, axis=0)
        return X, np.abs(X), np.angle(X), 0

    if window != "cola_hann":
        raise ValueError(f"Unsupported window: {window!r}")

    w = cola_hann_window(win_i)
    if pad_edges:
        pad_l = win_i - hop_i
        rem = (orig_len + 2 * pad_l - win_i) % hop_i
        pad_r = pad_l + ((hop_i - rem) % hop_i)
        work = np.pad(arr, (pad_l, pad_r), mode="constant")
    else:
        pad_l = 0
        work = np.pad(arr, (0, max(0, win_i - orig_len)), mode="constant")

    frames = np.lib.stride_tricks.sliding_window_view(work, win_i)[::hop_i].T * w[:, None]
    X = np.fft.fft(frames, n=nfft_i, axis=0)
    return X, np.abs(X), np.angle(X), pad_l


def istft_complex_windowed(
    X: np.ndarray,
    sig_len: int,
    win: int = int(C.STFT_WIN),
    hop: int = int(C.STFT_HOP),
    window: str = "cola_hann",
    pad_left: int = 32,
) -> np.ndarray:
    """LSEE-MSTFTM overlap-add iSTFT matching analysis window geometry cleanly."""
    win_i, hop_i = int(win), int(hop)
    n_out = int(sig_len)
    n_frames = X.shape[1]
    segs = np.fft.ifft(X, axis=0)[:win_i, :]  # (win, T)

    if window == "rect":
        out = np.zeros(n_out, dtype=np.complex128)
        cnt = np.zeros(n_out, dtype=np.float64)
        for m in range(n_frames):
            s = m * hop_i
            e = min(s + win_i, n_out)
            if e > s:
                out[s:e] += segs[: e - s, m]
                cnt[s:e] += 1.0
        cnt[cnt == 0.0] = 1.0
        return out / cnt

    if window != "cola_hann":
        raise ValueError(f"Unsupported window: {window!r}")

    w = cola_hann_window(win_i)
    buf_len = max(int(pad_left) + n_out + win_i, (n_frames - 1) * hop_i + win_i)
    ola_num = np.zeros(buf_len, dtype=np.complex128)
    ola_den = np.zeros(buf_len, dtype=np.float64)
    w_sq = w ** 2
    weighted = segs * w[:, None]
    for m in range(n_frames):
        s = m * hop_i
        ola_num[s : s + win_i] += weighted[:, m]
        ola_den[s : s + win_i] += w_sq

    rec = ola_num / np.maximum(ola_den, 1e-12)
    return rec[int(pad_left) : int(pad_left) + n_out]


def env_normalize_scale_inv(mag_clean: np.ndarray) -> np.ndarray:
    """Homogeneous-degree-1 block-level per-frame normalization (`f(c * mag) == c * f(mag)` for `c > 0`).

    Note: uses `np.median(frame_max)` over all frames of the 30 s block (`env_norm="block"`),
    which is 30 s block-causal (`[t-30s, t]`) rather than sample-by-sample frame-causal.
    """
    mag = np.maximum(np.asarray(mag_clean, dtype=np.float64), 0.0)
    if mag.size == 0:
        return mag
    g_max = float(np.max(mag))
    if not np.isfinite(g_max) or g_max <= 0.0:
        return mag
    rel_eps = 1e-12 * g_max
    frame_max = np.max(mag, axis=0, keepdims=True) + rel_eps
    med_max = float(np.median(frame_max))
    if med_max <= 1e-12 * g_max:
        return mag
    return mag * np.clip(med_max / frame_max, 0.5, 2.0)


def env_normalize_causal(mag_clean: np.ndarray) -> np.ndarray:
    """Strictly frame-causal homogeneous-degree-1 per-frame envelope normalizer.

    For each frame `m = 0..T-1`, computes `frame_max[m] = max(mag[:, m])`,
    `g_max_m = max(frame_max[:m+1])`, `rel_eps_m = 1e-12 * g_max_m`, and
    `med_m = median(frame_max[:m+1] + rel_eps_m)`, scaling `mag[:, m]` by
    `clip(med_m / (frame_max[m] + rel_eps_m), 0.5, 2.0)`. Guaranteed `0.0`
    future-frame leakage and exact `O(2) x R_{>0}` equivariance.
    """
    mag = np.maximum(np.asarray(mag_clean, dtype=np.float64), 0.0)
    if mag.size == 0:
        return mag
    out = mag.copy()
    frame_max = np.max(mag, axis=0)
    for m in range(frame_max.size):
        hist = frame_max[: m + 1]
        g_max_m = float(np.max(hist))
        if not np.isfinite(g_max_m) or g_max_m <= 0.0:
            continue
        rel_eps_m = 1e-12 * g_max_m
        med_m = float(np.median(hist + rel_eps_m))
        if med_m <= 1e-12 * g_max_m:
            continue
        scale_m = float(np.clip(med_m / (float(frame_max[m]) + rel_eps_m), 0.5, 2.0))
        out[:, m] = mag[:, m] * scale_m
    return out


@lru_cache(maxsize=8)
def _symmetric_band_masks(
    nfft: int,
    fs: float,
    band_low_hz: float,
    band_high_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mirror-symmetric positive and negative passband boolean masks in `fftshift` order."""
    freqs = np.fft.fftshift(np.fft.fftfreq(int(nfft), d=1.0 / float(fs)))
    pos = (freqs >= float(band_low_hz) - 1e-12) & (freqs <= float(band_high_hz) + 1e-12)
    neg = (freqs <= -float(band_low_hz) + 1e-12) & (freqs >= -float(band_high_hz) - 1e-12)
    if int(np.sum(pos)) != int(np.sum(neg)):
        raise RuntimeError(f"Asymmetric pos/neg band counts: {np.sum(pos)} vs {np.sum(neg)}")
    return pos, neg


def hpss_filter_causal(
    mag: np.ndarray,
    l_harm: int = 7,
    l_perc: int = 21,
    gamma: float = 2.0,
    lookahead: int = 0,
    in_band: bool = True,
    fs: float = float(C.FS_RADAR),
    band_low_hz: float = 0.25,
    band_high_hz: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal or bounded-lookahead HPSS over two-sided `fftshift` STFT magnitude.

    When `lookahead=0`, `filt[:, m]`, `mask[:, m]`, `s_harm[:, m]`, and `s_perc[:, m]`
    depend STRICTLY on frames `<= m` (`0.0` perturbation from future frames `> m`).
    When `in_band=True`, vertical median filtering runs strictly inside mirror-symmetric
    `[+band_low, +band_high]` and `[-band_high, -band_low]` bands to yield `M_{H,band}`.
    The default `band_low_hz=0.25` (bin 4 = `15.0 bpm` at `fs=16 Hz, nfft=256`) acts
    as the discrete-FFT guard preserving nominal `\\bandLow = 0.30 Hz` (`18.0 bpm`, since
    bin 5 lies at `0.3125 Hz = 18.75 bpm`).
    """
    lh = int(l_harm)
    lp = int(l_perc)
    la = int(lookahead)
    if la < 0 or la >= lh:
        raise ValueError(f"lookahead={la} must satisfy 0 <= lookahead < l_harm={lh}")

    m_in = np.maximum(np.asarray(mag, dtype=np.float64), 0.0)
    n_f, n_t = m_in.shape
    mag_s = np.fft.fftshift(m_in, axes=0)

    # Asymmetric causal temporal median via sliding window view
    pad_left = lh - 1 - la
    mag_pad = np.pad(mag_s, ((0, 0), (pad_left, la)), mode="edge")
    win_view = np.lib.stride_tricks.sliding_window_view(mag_pad, window_shape=lh, axis=1)
    s_harm = np.median(win_view, axis=2)  # (F, T)

    if not in_band:
        s_perc = median_filter(mag_s, size=(lp, 1), mode="nearest")
        hg = s_harm ** float(gamma)
        pg = s_perc ** float(gamma)
        # Strictly causal per-frame scale-equivariant denominator stabilizer
        frame_scale = np.maximum(np.mean(hg + pg, axis=0, keepdims=True), 1e-30)
        eps_rel = 1e-12 * frame_scale
        mask_s = hg / (hg + pg + eps_rel)
    else:
        pos_mask, neg_mask = _symmetric_band_masks(n_f, float(fs), float(band_low_hz), float(band_high_hz))
        s_perc = np.zeros_like(mag_s)
        s_perc[pos_mask, :] = median_filter(mag_s[pos_mask, :], size=(lp, 1), mode="nearest")
        s_perc[neg_mask, :] = median_filter(mag_s[neg_mask, :], size=(lp, 1), mode="nearest")
        inband = pos_mask | neg_mask
        mask_s = np.zeros_like(mag_s)
        hg = s_harm[inband, :] ** float(gamma)
        pg = s_perc[inband, :] ** float(gamma)
        frame_scale = np.maximum(np.mean(hg + pg, axis=0, keepdims=True), 1e-30)
        eps_rel = 1e-12 * frame_scale
        mask_s[inband, :] = hg / (hg + pg + eps_rel)

    filt = np.fft.ifftshift(mask_s * mag_s, axes=0)
    mask = np.fft.ifftshift(mask_s, axes=0)
    return filt, mask, np.fft.ifftshift(s_harm, axes=0), np.fft.ifftshift(s_perc, axes=0)


def mitigate_hpss_v2(
    x: np.ndarray,
    window: str = "cola_hann",
    lookahead: int = 0,
    in_band: bool = True,
    l_harm: int = 7,
    l_perc: int = 21,
    gamma: float = 2.0,
    fs: float = float(C.FS_RADAR),
    band_low_hz: float = 0.25,
    band_high_hz: float = 3.0,
    env_norm: str = "block",
) -> dict[str, np.ndarray | float]:
    """Radar-native causal/in-band HPSS block matching `mitigate_hpss` output keys.

    Parameters
    ----------
    env_norm:
        - `"block"` (default): applies 30 s block-level `env_normalize_scale_inv` matching `v1`.
          Strictly block-causal `[t-30s, t]` (zero future samples past block end `t`), though
          earlier frames within the block couple to the block's median frame amplitude.
        - `"causal"`: applies `env_normalize_causal`, guaranteeing exact `0.0` future-frame
          leakage on `mag_filt[:, <=m]` when `lookahead=0` (with `istft` retaining its
          48-sample / 3 s overlap-add synthesis support and zero-phase outer `bandpass()`).
        - `"none"`: skips envelope normalization (`mag_filt = mag_filt_raw`).
    """
    arr = np.asarray(x, dtype=np.complex128).ravel()
    X, mag, phase, pad_l = stft_complex_windowed(arr, window=window)
    mag_filt_raw, mask, s_harm, s_perc = hpss_filter_causal(
        mag,
        l_harm=l_harm,
        l_perc=l_perc,
        gamma=gamma,
        lookahead=lookahead,
        in_band=in_band,
        fs=fs,
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
    )
    if env_norm == "block":
        mag_filt = env_normalize_scale_inv(mag_filt_raw)
    elif env_norm == "causal":
        mag_filt = env_normalize_causal(mag_filt_raw)
    elif env_norm == "none":
        mag_filt = mag_filt_raw
    else:
        raise ValueError(f"Unsupported env_norm: {env_norm!r}")
    Xhat = mag_filt * np.exp(1j * phase)
    x_hat = istft_complex_windowed(Xhat, arr.size, window=window, pad_left=pad_l)
    denom = float(np.sum(mag))
    kept_frac = float(np.sum(mag_filt_raw) / denom) if denom > 0.0 else 0.0
    return {
        "x_hat": x_hat,
        "X": X,
        "mag": mag,
        "mag_filt": mag_filt,
        "mask": mask,
        "s_harm": s_harm,
        "s_perc": s_perc,
        "kept_frac": kept_frac,
    }


# ---------------------------------------------------------------------------
# Rung 1: Strictly Causal Online Forward-HMM, Fixed-Lag & Offline 1D Decoders
# ---------------------------------------------------------------------------

def build_gaussian_log_transition(
    grid_bpm: np.ndarray = VT.BR_GRID_1BPM,
    sigma_br: float = 2.0,
) -> np.ndarray:
    """Build exact row-stochastic Gaussian random-walk log-transition matrix `(N_B, N_B)`."""
    grid = np.asarray(grid_bpm, dtype=np.float64).ravel()
    diff = grid[:, None] - grid[None, :]
    raw = -0.5 * (diff / float(sigma_br)) ** 2
    return raw - logsumexp(raw, axis=1, keepdims=True)


def _emission_log_likelihood(
    S_seq: np.ndarray,
    eps: float = 1e-2,
    beta: float = 1.0,
) -> np.ndarray:
    s = np.maximum(np.asarray(S_seq, dtype=np.float64), 0.0)
    return float(beta) * np.log(s + float(eps))


def _posterior_summaries(log_post: np.ndarray, grid: np.ndarray) -> dict[str, np.ndarray]:
    lp = log_post - logsumexp(log_post, axis=1, keepdims=True)
    p = np.exp(lp)
    est_mmse = p @ grid
    est_map = grid[np.argmax(lp, axis=1)]
    n_b = grid.size
    ent = -np.sum(p * np.where(p > 1e-300, lp, 0.0), axis=1) / max(np.log(float(n_b)), 1e-12)
    ent = np.clip(ent, 0.0, 1.0)
    var = np.sum(p * (grid[None, :] - est_mmse[:, None]) ** 2, axis=1)
    sd = np.sqrt(np.maximum(var, 0.0))
    return {
        "est_mmse": est_mmse,
        "est_map": est_map,
        "post_entropy": ent,
        "post_sd_bpm": sd,
        "log_post": lp,
    }


def _can_use_scaled_hmm(eps: float, beta: float, grid_size: int) -> bool:
    return float(eps) >= 1e-12 and 0.0 <= float(beta) <= 10.0 and grid_size > 0


def _posterior_summaries_prob(p: np.ndarray, grid: np.ndarray) -> dict[str, np.ndarray]:
    row_sums = np.sum(p, axis=1, keepdims=True)
    pn = p / np.maximum(row_sums, 1e-300)
    est_mmse = pn @ grid
    est_map = grid[np.argmax(pn, axis=1)]
    n_b = grid.size
    lp = np.where(pn > 1e-300, np.log(np.maximum(pn, 1e-300)), 0.0)
    ent = -np.sum(pn * lp, axis=1) / max(np.log(float(n_b)), 1e-12)
    ent = np.clip(ent, 0.0, 1.0)
    var = np.sum(pn * (grid[None, :] - est_mmse[:, None]) ** 2, axis=1)
    sd = np.sqrt(np.maximum(var, 0.0))
    return {
        "est_mmse": est_mmse,
        "est_map": est_map,
        "post_entropy": ent,
        "post_sd_bpm": sd,
        "log_post": np.log(np.maximum(pn, 1e-300)),
    }


def hmm_forward_filter(
    S_seq: np.ndarray,
    grid_bpm: np.ndarray = VT.BR_GRID_1BPM,
    sigma_br: float = 2.0,
    eps: float = 1e-2,
    beta: float = 1.0,
) -> dict[str, np.ndarray]:
    """Strictly causal (`0 s` lookahead) Bayesian Forward HMM filter."""
    grid = np.asarray(grid_bpm, dtype=np.float64).ravel()
    s_mat = np.maximum(np.asarray(S_seq, dtype=np.float64), 0.0)
    n_t, n_b = s_mat.shape
    if n_b != grid.size:
        raise ValueError(f"S_seq width {n_b} != grid_bpm size {grid.size}")
    if n_t == 0:
        empty = np.empty(0, dtype=np.float64)
        return {
            "est_mmse": empty,
            "est_map": empty,
            "post_entropy": empty,
            "post_sd_bpm": empty,
            "log_evidence": empty,
            "log_alpha": np.empty((0, n_b), dtype=np.float64),
        }

    log_T = build_gaussian_log_transition(grid, sigma_br)
    if _can_use_scaled_hmm(eps, beta, n_b):
        T_mat = np.exp(log_T)
        emit = (s_mat + float(eps)) ** float(beta)
        alpha = np.empty((n_t, n_b), dtype=np.float64)
        log_z = np.empty(n_t, dtype=np.float64)
        a = emit[0] / float(n_b)
        c0 = float(a.sum())
        if not np.isfinite(c0) or c0 <= 0.0:
            a = np.full(n_b, 1.0 / float(n_b), dtype=np.float64)
            log_z[0] = -np.inf
        else:
            a = a / c0
            log_z[0] = float(np.log(c0))
        alpha[0] = a
        for t in range(1, n_t):
            a = (alpha[t - 1] @ T_mat) * emit[t]
            ct = float(a.sum())
            if not np.isfinite(ct) or ct <= 0.0:
                a = np.full(n_b, 1.0 / float(n_b), dtype=np.float64)
                log_z[t] = -np.inf
            else:
                a = a / ct
                log_z[t] = float(np.log(ct))
            alpha[t] = a
        res = _posterior_summaries_prob(alpha, grid)
        res["log_evidence"] = log_z
        res["log_alpha"] = res["log_post"]
        res["alpha"] = alpha
        return res

    log_emit = _emission_log_likelihood(s_mat, eps=eps, beta=beta)
    log_alpha = np.empty((n_t, n_b), dtype=np.float64)
    log_z = np.empty(n_t, dtype=np.float64)

    log_pi = np.full(n_b, -np.log(float(n_b)), dtype=np.float64)
    unnorm0 = log_pi + log_emit[0]
    z0 = float(logsumexp(unnorm0))
    log_z[0] = z0
    log_alpha[0] = unnorm0 - z0

    for t in range(1, n_t):
        log_pred = logsumexp(log_alpha[t - 1, :, None] + log_T, axis=0)
        unnorm = log_emit[t] + log_pred
        zt = float(logsumexp(unnorm))
        log_z[t] = zt
        log_alpha[t] = unnorm - zt

    res = _posterior_summaries(log_alpha, grid)
    res["log_evidence"] = log_z
    res["log_alpha"] = log_alpha
    res["alpha"] = np.exp(log_alpha)
    return res


def hmm_fixed_lag_smoother(
    S_seq: np.ndarray,
    lag_hops: int = 4,
    grid_bpm: np.ndarray = VT.BR_GRID_1BPM,
    sigma_br: float = 2.0,
    eps: float = 1e-2,
    beta: float = 1.0,
) -> dict[str, np.ndarray]:
    """Bounded-lookahead fixed-lag Forward-Backward smoother (`0` access past `t + lag_hops`)."""
    lag = int(lag_hops)
    if lag < 0:
        raise ValueError(f"lag_hops must be >= 0, got {lag}")
    fwd = hmm_forward_filter(S_seq, grid_bpm=grid_bpm, sigma_br=sigma_br, eps=eps, beta=beta)
    if lag == 0:
        return fwd

    grid = np.asarray(grid_bpm, dtype=np.float64).ravel()
    s_mat = np.maximum(np.asarray(S_seq, dtype=np.float64), 0.0)
    n_t, n_b = s_mat.shape
    if n_t == 0:
        return fwd

    log_T = build_gaussian_log_transition(grid, sigma_br)
    if _can_use_scaled_hmm(eps, beta, n_b):
        T_mat = np.exp(log_T)
        emit = (s_mat + float(eps)) ** float(beta)
        alpha = fwd["alpha"]
        p_post = np.empty((n_t, n_b), dtype=np.float64)
        for t in range(n_t):
            tau_end = min(n_t - 1, t + lag)
            b = np.ones(n_b, dtype=np.float64)
            for tau in range(tau_end, t, -1):
                b = T_mat @ (emit[tau] * b)
                bmax = float(b.max())
                if bmax > 0.0 and np.isfinite(bmax):
                    b = b / bmax
                else:
                    b = np.ones(n_b, dtype=np.float64)
            pt = alpha[t] * b
            ps = float(pt.sum())
            p_post[t] = (pt / ps) if ps > 0.0 and np.isfinite(ps) else alpha[t]
        res = _posterior_summaries_prob(p_post, grid)
        res["log_evidence"] = fwd["log_evidence"]
        return res

    log_emit = _emission_log_likelihood(s_mat, eps=eps, beta=beta)
    log_alpha = fwd["log_alpha"]
    log_post = np.empty((n_t, n_b), dtype=np.float64)

    for t in range(n_t):
        tau_end = min(n_t - 1, t + lag)
        log_beta = np.zeros(n_b, dtype=np.float64)
        for tau in range(tau_end, t, -1):
            log_beta = logsumexp(log_T + (log_emit[tau] + log_beta)[None, :], axis=1)
        unnorm = log_alpha[t] + log_beta
        log_post[t] = unnorm - float(logsumexp(unnorm))

    res = _posterior_summaries(log_post, grid)
    res["log_evidence"] = fwd["log_evidence"]
    return res


def hmm_offline_fb_smoother(
    S_seq: np.ndarray,
    grid_bpm: np.ndarray = VT.BR_GRID_1BPM,
    sigma_br: float = 2.0,
    eps: float = 1e-2,
    beta: float = 1.0,
) -> dict[str, np.ndarray]:
    """Full-session $O(T N_B^2)$ Forward-Backward marginal MMSE/MAP smoother."""
    fwd = hmm_forward_filter(S_seq, grid_bpm=grid_bpm, sigma_br=sigma_br, eps=eps, beta=beta)
    grid = np.asarray(grid_bpm, dtype=np.float64).ravel()
    s_mat = np.maximum(np.asarray(S_seq, dtype=np.float64), 0.0)
    n_t, n_b = s_mat.shape
    if n_t == 0:
        return fwd

    log_T = build_gaussian_log_transition(grid, sigma_br)
    if _can_use_scaled_hmm(eps, beta, n_b):
        T_mat = np.exp(log_T)
        emit = (s_mat + float(eps)) ** float(beta)
        beta_arr = np.ones((n_t, n_b), dtype=np.float64)
        for t in range(n_t - 2, -1, -1):
            b = T_mat @ (emit[t + 1] * beta_arr[t + 1])
            bmax = float(b.max())
            beta_arr[t] = (b / bmax) if (bmax > 0.0 and np.isfinite(bmax)) else np.ones(n_b, dtype=np.float64)
        p_post = fwd["alpha"] * beta_arr
        row_s = np.sum(p_post, axis=1, keepdims=True)
        p_post = np.where(row_s > 0.0, p_post / np.maximum(row_s, 1e-300), fwd["alpha"])
        res = _posterior_summaries_prob(p_post, grid)
        res["log_evidence"] = fwd["log_evidence"]
        return res

    log_emit = _emission_log_likelihood(s_mat, eps=eps, beta=beta)
    log_beta = np.zeros((n_t, n_b), dtype=np.float64)
    for t in range(n_t - 2, -1, -1):
        log_beta[t] = logsumexp(log_T + (log_emit[t + 1] + log_beta[t + 1])[None, :], axis=1)

    log_post = fwd["log_alpha"] + log_beta
    log_post -= logsumexp(log_post, axis=1, keepdims=True)
    res = _posterior_summaries(log_post, grid)
    res["log_evidence"] = fwd["log_evidence"]
    return res


def hmm_offline_viterbi_prob(
    S_seq: np.ndarray,
    grid_bpm: np.ndarray = VT.BR_GRID_1BPM,
    sigma_br: float = 2.0,
    eps: float = 1e-2,
    beta: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Exact row-normalized max-sum MAP Viterbi decoder paired with Forward-HMM."""
    grid = np.asarray(grid_bpm, dtype=np.float64).ravel()
    log_emit = _emission_log_likelihood(S_seq, eps=eps, beta=beta)
    n_t, n_b = log_emit.shape
    if n_b != grid.size:
        raise ValueError(f"S_seq width {n_b} != grid_bpm size {grid.size}")
    if n_t == 0:
        return np.empty(0, dtype=np.float64), 0.0

    log_T = build_gaussian_log_transition(grid, sigma_br)
    v_prev = np.full(n_b, -np.log(float(n_b)), dtype=np.float64) + log_emit[0]
    back = np.zeros((n_t, n_b), dtype=np.int32)
    idx_b = np.arange(n_b, dtype=np.int32)

    for t in range(1, n_t):
        scores = v_prev[:, None] + log_T
        bp = np.argmax(scores, axis=0)
        back[t] = bp
        v_prev = log_emit[t] + scores[bp, idx_b]

    j_cur = int(np.argmax(v_prev))
    best_score = float(v_prev[j_cur])
    path_idx = np.empty(n_t, dtype=np.int32)
    path_idx[-1] = j_cur
    for t in range(n_t - 1, 0, -1):
        j_cur = int(back[t, j_cur])
        path_idx[t - 1] = j_cur
    return grid[path_idx], best_score
