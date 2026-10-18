"""
NMF-based RBM mitigation (Eq.10-11).
STFT: rectangular 3 s (48 @16Hz), hop 1 s (16), NFFT 256.
NMF K=11 Frobenius (sklearn). Adaptive threshold + sparsity -> mask.
Wiener-like phase reuse + overlap-add iSTFT.
"""
import numpy as np
import warnings
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from . import config as C


def stft_complex(x, win=C.STFT_WIN, hop=C.STFT_HOP, nfft=C.STFT_NFFT):
    x = np.asarray(x).ravel()
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    frames = []
    for s in range(0, len(x) - win + 1, hop):
        seg = x[s:s + win]
        spec = np.fft.fft(seg, n=nfft)
        frames.append(spec)
    # tail: pad last partial frame for full coverage
    X = np.stack(frames, axis=1)  # (F, T)
    mag = np.abs(X)
    phase = np.angle(X)
    return X, mag, phase


def detect_rbm(H, alpha=C.RBM_ALPHA, frac=C.RBM_SPARSITY_FRAC):
    """Paper qualitative rule: strong (stronger than average) + sparse (short duration).

    H: (K, T). avg = mean(H). Candidate if max(h_i) > alpha*avg and
    top-3 time bins hold > frac of row energy.
    Returns boolean mask keep (True=keep, False=RBM).
    """
    K, T = H.shape
    avg = np.mean(H) + 1e-12
    keep = np.ones(K, dtype=bool)
    for i in range(K):
        h = H[i]
        peak = np.max(h)
        if peak > alpha * avg:
            order = np.sort(h)[::-1]
            topk = order[:min(3, T)].sum()
            tot = h.sum() + 1e-12
            if topk / tot > frac:
                keep[i] = False
    return keep


def hoyer_sparsity(h):
    """Hoyer sparsity Eq.4 (attached §4.3). All-zero rows -> 0."""
    h = np.asarray(h, dtype=float).ravel()
    n = h.size
    if n <= 1 or not np.any(h):
        return 0.0
    l1 = np.abs(h).sum()
    l2 = float(np.sqrt((h ** 2).sum()))
    if l2 <= 0:
        return 0.0
    return float((np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1))


def detect_rbm_hoyer(H, s_thresh=0.55, amp_factor=2.0):
    """Attached locked rule: reject iff s(hk) > 0.55 AND max(hk) > 2*mean(H).

    Operates on RAW sklearn output (no V/W/H normalization)."""
    H = np.asarray(H, dtype=float)
    avg = float(np.mean(H)) if H.size else 0.0
    keep = np.ones(H.shape[0], dtype=bool)
    for i in range(H.shape[0]):
        h = H[i]
        if hoyer_sparsity(h) > s_thresh and float(np.max(h)) > amp_factor * avg:
            keep[i] = False
    return keep


def env_normalize(mag_clean):
    """Renormalize per-frame peak magnitude so removing an RBM burst does not
    punch a near-zero amplitude hole in the reconstructed 30-s time waveform."""
    frame_max = np.max(mag_clean, axis=0, keepdims=True) + 1e-12
    med_max = float(np.median(frame_max))
    if med_max <= 1e-11:
        return mag_clean
    return mag_clean * np.clip(med_max / frame_max, 0.5, 2.0)


def nmf_filter(mag, phase, win=C.STFT_WIN, hop=C.STFT_HOP, nfft=C.STFT_NFFT,
               alpha_H=None, l1_ratio=None, preset=None):
    # preset='attached_locked': raw V, tol 1e-4, 2000 iters, Hoyer rule.
    # default ('beltrao_tuned'): magnitude-normalized V, 500 iters, α/frac rule.
    if preset is None:
        preset = getattr(C, "NMF_PRESET", "beltrao_tuned")
    if preset == "attached_locked":
        mag_n = np.maximum(np.asarray(mag, dtype=float), 0.0)  # NO normalization
        model = NMF(n_components=C.NMF_K, init="nndsvda", tol=1e-4,
                    max_iter=2000, random_state=C.NMF_RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            W = model.fit_transform(mag_n)
        H = model.components_
        keep = detect_rbm_hoyer(H)
        mag_filt = (W[:, keep] @ H[keep, :]) if np.any(keep) else np.zeros_like(mag_n)
        Xhat = mag_filt * np.exp(1j * phase)
        return Xhat, mag_filt, keep, W, H, int(model.n_iter_)
    # Normalize spectrogram (paper Fig.4b shows normalized |X|); NMF is
    # scale-sensitive and raw magnitudes explode on RBM spikes (H.mean ~1e15),
    # making a fixed alpha incomparable across windows.
    scale = float(np.max(mag)) + 1e-12
    mag_n = np.maximum(mag, 0) / scale
    if alpha_H is None:
        alpha_H = C.NMF_ALPHA_H
    if l1_ratio is None:
        l1_ratio = C.NMF_L1_RATIO
    model = NMF(n_components=C.NMF_K, init="nndsvda",
                max_iter=C.NMF_MAX_ITER, random_state=C.NMF_RANDOM_STATE,
                alpha_H=alpha_H, alpha_W=0.0, l1_ratio=l1_ratio)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        W = model.fit_transform(mag_n)
    H = model.components_
    keep = detect_rbm(H)
    # Reconstruct via Wiener ratio mask on |X| + per-frame envelope normalization
    WH_all = (W @ H) + 1e-12
    WH_keep = (W[:, keep] @ H[keep, :]) if np.any(keep) else WH_all
    mag_filt = env_normalize((WH_keep / WH_all) * np.maximum(mag, 0))
    Xhat = mag_filt * np.exp(1j * phase)
    return Xhat, mag_filt, keep, W, H, int(model.n_iter_)


def istft_complex(Xhat, sig_len, win=C.STFT_WIN, hop=C.STFT_HOP):
    F, T = Xhat.shape
    out = np.zeros(sig_len, dtype=complex)
    cnt = np.zeros(sig_len, dtype=float)
    idx = 0
    for t in range(T):
        seg = np.fft.ifft(Xhat[:, t], n=C.STFT_NFFT)[:win]
        s = idx
        e = min(s + win, sig_len)
        out[s:e] += seg[:e - s]
        cnt[s:e] += 1.0
        idx += hop
        if idx >= sig_len:
            break
    cnt[cnt == 0] = 1.0
    return out / cnt


def mitigate(x, alpha_H=None, l1_ratio=None, preset=None):
    """Full RBM block: x -> x_hat, plus diagnostics."""
    X, mag, phase = stft_complex(x)
    Xhat, mag_filt, keep, W, H, n_iter = nmf_filter(
        mag, phase, alpha_H=alpha_H, l1_ratio=l1_ratio, preset=preset)
    x_hat = istft_complex(Xhat, len(x))
    return {
        "x_hat": x_hat, "X": X, "mag": mag,
        "mag_filt": mag_filt, "keep": keep,
        "n_removed": int(np.sum(~keep)), "W": W, "H": H,
        "nmf_n_iter": n_iter,
        "nmf_converged": bool(n_iter < (2000 if preset == "attached_locked" else C.NMF_MAX_ITER)),
    }
