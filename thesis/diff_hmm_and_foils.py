"""Stage 2 Models: Symmetric Equivariant Differentiable 1D-HMM (Rung 2) & Unconstrained Foils (Rung 3).

Provides:
  1. `extract_salience_features_window` (`(N_B, 6)` O(2)xR+ invariant salience tensor).
  2. `DifferentiableSymmetricHMM` (`R2` linear `11`-param & residual MLP `43`-param models,
     strictly zero-offset frequency-shift-equivariant, supporting causal Forward and Offline FB
     analytical BPTT training via L-BFGS-B).
  3. `CausalDilatedTCN` (`15,817`-param strictly causal dilated Conv1D net with exact NumPy BPTT + Adam),
     `fit_predict_spectral_mlp` (`14,593` params), and `fit_predict_ridge_spectral` (`439` params)
     trained without frequency equivariance constraints as Rung 3 shortcut-learning foils.

All symbol conventions adhere strictly to
`paper_orchestra_bundle/raw_materials/notation_glossary.tex`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from scipy import optimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.neural_network import MLPRegressor

from paper_workspace.src import config as C
from paper_workspace.src.estimation import dft_estimate, psd_fft
from thesis import viterbi_tracker as VT

HARM_SAL_RADICES = (1.0, 2.0, 3.0, 0.5)
N_SAL_CHANNELS = 6
FILTERSAFE_CH_IDX = (0, 4, 5)  # P_clean(1f), A_coarse(f), P_raw(1f) strictly inside [0.3, 3.0] Hz across [18, 90] bpm


def _max_norm_psd(
    sig_arr: np.ndarray,
    grid_hz: np.ndarray,
    freqs_hz: np.ndarray,
    pf: np.ndarray,
) -> np.ndarray:
    vals = np.interp(grid_hz, freqs_hz, pf, left=0.0, right=0.0)
    vals = np.maximum(vals, 0.0)
    vmax = float(np.max(vals)) if vals.size else 0.0
    if not np.isfinite(vmax) or vmax <= 1e-30:
        return np.zeros_like(grid_hz, dtype=np.float64)
    return vals / vmax


def extract_salience_features_window(
    d_raw: np.ndarray,
    d_clean: np.ndarray,
    f_coarse_hz: float,
    grid_bpm: np.ndarray = VT.BR_GRID_1BPM,
    fs: float = float(C.FS_RADAR),
    eps0: float = 1e-3,
) -> np.ndarray:
    """Compute `(N_B, 6)` strictly $O(2) \\times \\mathbb{R}_{>0}$-invariant log-salience tensor.

    Channels:
      0: log(P_clean(1 f) + eps0)  [fundamental PSD]
      1: log(P_clean(2 f) + eps0)  [2nd harmonic]
      2: log(P_clean(3 f) + eps0)  [3rd harmonic]
      3: log(P_clean(0.5 f) + eps0) [subharmonic suppression cue]
      4: log(A_coarse(f) + eps0)   [autocorrelation coarse anchor envelope]
      5: log(P_raw(1 f) + eps0)    [pre-HPSS raw fundamental PSD]
    """
    grid = np.asarray(grid_bpm, dtype=np.float64).ravel()
    n_b = grid.size
    eps = float(eps0)
    out = np.full((n_b, N_SAL_CHANNELS), np.log(eps), dtype=np.float64)

    dc = np.asarray(d_clean, dtype=np.complex128).ravel()
    dr = np.asarray(d_raw, dtype=np.complex128).ravel()
    clean_ok = dc.size > 0 and np.all(np.isfinite(dc)) and float(np.sum(np.abs(dc) ** 2)) > 1e-30
    raw_ok = dr.size > 0 and np.all(np.isfinite(dr)) and float(np.sum(np.abs(dr) ** 2)) > 1e-30

    if clean_ok:
        freqs_c, pf_c = psd_fft(dc, nfft=int(C.NLS_NFFT), fs=float(fs))
        for ch_idx, mult in enumerate(HARM_SAL_RADICES):
            p_norm = _max_norm_psd(dc, (grid / 60.0) * float(mult), freqs_c, pf_c)
            out[:, ch_idx] = np.log(p_norm + eps)

        if np.isfinite(f_coarse_hz):
            centre = float(f_coarse_hz) * 60.0
        else:
            centre, _, _ = dft_estimate(dc, fs=float(fs))
        if np.isfinite(centre):
            centre = float(np.clip(centre, grid[0], grid[-1]))
            gauss = np.exp(-0.5 * ((grid - centre) / float(C.NLS_RADIUS_BPM)) ** 2)
            env = np.maximum(0.05, gauss)
        else:
            env = np.full(n_b, 0.05, dtype=np.float64)
        out[:, 4] = np.log(env + eps)

    if raw_ok:
        freqs_r, pf_r = psd_fft(dr, nfft=int(C.NLS_NFFT), fs=float(fs))
        p_raw = _max_norm_psd(dr, grid / 60.0, freqs_r, pf_r)
        out[:, 5] = np.log(p_raw + eps)

    return out


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = float(np.exp(-x))
        return 1.0 / (1.0 + z)
    z = float(np.exp(x))
    return z / (1.0 + z)


def build_mixture_transition_and_grad(
    grid_bpm: np.ndarray,
    log_sig_slow: float,
    log_sig_jump: float,
    logit_pi: float,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build heavy-tailed 2-Gaussian symmetric row-stochastic transition matrix & Jacobians.

    Contains NO global uniform component, guaranteeing exact frequency-shift equivariance
    across arbitrary `[q_min, q_max]` grids without midpoint shrinkage.
    """
    grid = np.asarray(grid_bpm, dtype=np.float64).ravel()
    diff2 = (grid[:, None] - grid[None, :]) ** 2

    sig_s = float(np.exp(log_sig_slow))
    sig_j = float(np.exp(log_sig_jump))
    pi_j = _sigmoid(float(logit_pi))

    inv_s = 1.0 / sig_s
    inv_j = 1.0 / sig_j
    inv_s2 = inv_s * inv_s
    inv_j2 = inv_j * inv_j

    gs = inv_s * np.exp(-0.5 * diff2 * inv_s2)
    gj = inv_j * np.exp(-0.5 * diff2 * inv_j2)
    k_mat = (1.0 - pi_j) * gs + pi_j * gj

    row_sum = np.maximum(np.sum(k_mat, axis=1, keepdims=True), 1e-300)
    t_mat = k_mat / row_sum

    dk_dlogss = (1.0 - pi_j) * gs * (diff2 * inv_s2 - 1.0)
    dk_dlogsj = pi_j * gj * (diff2 * inv_j2 - 1.0)
    dk_dlogitpi = pi_j * (1.0 - pi_j) * (gj - gs)

    dt_dlogss = (dk_dlogss - t_mat * np.sum(dk_dlogss, axis=1, keepdims=True)) / row_sum
    dt_dlogsj = (dk_dlogsj - t_mat * np.sum(dk_dlogsj, axis=1, keepdims=True)) / row_sum
    dt_dlogitpi = (dk_dlogitpi - t_mat * np.sum(dk_dlogitpi, axis=1, keepdims=True)) / row_sum

    return t_mat, (dt_dlogss, dt_dlogsj, dt_dlogitpi)


def _symmetric_edge_conv5(s_pt: np.ndarray, k1: float, k2: float) -> tuple[np.ndarray, np.ndarray]:
    s_pad = np.pad(s_pt, ((0, 0), (2, 2)), mode="edge")
    ell = (
        float(k2) * s_pad[:, :-4]
        + float(k1) * s_pad[:, 1:-3]
        + s_pad[:, 2:-2]
        + float(k1) * s_pad[:, 3:-1]
        + float(k2) * s_pad[:, 4:]
    )
    return ell, s_pad


def _symmetric_edge_conv5_grad(
    g_ell: np.ndarray,
    s_pad: np.ndarray,
    k1: float,
    k2: float,
) -> tuple[np.ndarray, float, float]:
    g_k1 = float(np.sum(g_ell * (s_pad[:, 1:-3] + s_pad[:, 3:-1])))
    g_k2 = float(np.sum(g_ell * (s_pad[:, :-4] + s_pad[:, 4:])))
    n_t, n_b = g_ell.shape
    g_pad = np.zeros((n_t, n_b + 4), dtype=np.float64)
    g_pad[:, :-4] += float(k2) * g_ell
    g_pad[:, 1:-3] += float(k1) * g_ell
    g_pad[:, 2:-2] += g_ell
    g_pad[:, 3:-1] += float(k1) * g_ell
    g_pad[:, 4:] += float(k2) * g_ell
    g_pad[:, 2] += g_pad[:, 0] + g_pad[:, 1]
    g_pad[:, -3] += g_pad[:, -2] + g_pad[:, -1]
    return g_pad[:, 2:-2], g_k1, g_k2


def hmm_emission_and_grad(
    phi_seq: np.ndarray,
    theta: np.ndarray,
    mode: str = "linear",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate strictly shift-equivariant emission log-scores `ell` `(T, N_B)`."""
    phi = np.asarray(phi_seq, dtype=np.float64)
    if mode == "linear":
        # theta: w(6), k1, k2, log_ss, log_sj, logit_pi  -> 11 params
        w = theta[:6]
        k1, k2 = float(theta[6]), float(theta[7])
        s_pt = phi @ w  # (T, N_B)
        cache: dict[str, Any] = {"s_pt": s_pt, "k1": k1, "k2": k2, "phi": phi}
    elif mode == "linear_filtersafe":
        # theta: w_fs(3) on (ch0, ch4, ch5), k1, k2, log_ss, log_sj, logit_pi -> 8 params
        phi_fs = phi[:, :, FILTERSAFE_CH_IDX]
        w = theta[:3]
        k1, k2 = float(theta[3]), float(theta[4])
        s_pt = phi_fs @ w  # (T, N_B)
        cache = {"s_pt": s_pt, "k1": k1, "k2": k2, "phi_fs": phi_fs}
    elif mode == "mlp":
        # theta: w_res(6), W1(6x4=24), b1(4), W2(4), k1, k2, log_ss, log_sj, logit_pi -> 43 params
        w_res = theta[:6]
        W1 = theta[6:30].reshape(6, 4)
        b1 = theta[30:34]
        W2 = theta[34:38]
        k1, k2 = float(theta[38]), float(theta[39])
        h_act = np.tanh(phi @ W1 + b1)  # (T, N_B, 4)
        s_pt = (phi @ w_res) + (h_act @ W2)
        cache = {"s_pt": s_pt, "h_act": h_act, "W1": W1, "W2": W2, "k1": k1, "k2": k2, "phi": phi}
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")

    ell, s_pad = _symmetric_edge_conv5(s_pt, k1, k2)
    cache["s_pad"] = s_pad
    return ell, cache


def hmm_emission_grad_params(
    g_ell: np.ndarray,
    cache: dict[str, Any],
    n_params: int,
    mode: str = "linear",
) -> np.ndarray:
    """Backprop emission score gradients `g_ell` into flat parameter vector."""
    grad = np.zeros(n_params, dtype=np.float64)
    g_spt, g_k1, g_k2 = _symmetric_edge_conv5_grad(
        g_ell, cache["s_pad"], cache["k1"], cache["k2"]
    )
    if mode == "linear":
        phi = cache["phi"]
        grad[:6] = np.einsum("tb,tbc->c", g_spt, phi)
        grad[6] = g_k1
        grad[7] = g_k2
    elif mode == "linear_filtersafe":
        phi_fs = cache["phi_fs"]
        grad[:3] = np.einsum("tb,tbc->c", g_spt, phi_fs)
        grad[3] = g_k1
        grad[4] = g_k2
    elif mode == "mlp":
        phi = cache["phi"]
        grad[:6] = np.einsum("tb,tbc->c", g_spt, phi)
        h_act = cache["h_act"]
        W2 = cache["W2"]
        grad[34:38] = np.einsum("tb,tbh->h", g_spt, h_act)
        g_h = g_spt[:, :, None] * W2[None, None, :]
        g_pre = g_h * (1.0 - h_act * h_act)
        grad[30:34] = np.sum(g_pre, axis=(0, 1))
        grad[6:30] = np.einsum("tbc,tbh->ch", phi, g_pre).ravel()
        grad[38] = g_k1
        grad[39] = g_k2
    return grad


def forward_hmm_predictions_and_bptt(
    ell: np.ndarray,
    t_mat: np.ndarray,
    grid_bpm: np.ndarray,
    decoder: str = "causal_forward",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run causal Forward or Offline Forward-Backward HMM retaining BPTT intermediates."""
    grid = np.asarray(grid_bpm, dtype=np.float64).ravel()
    n_t, n_b = ell.shape
    l_shift = ell - np.max(ell, axis=1, keepdims=True)
    e_mat = np.exp(l_shift)  # in (0, 1], safe from overflow

    alpha = np.empty((n_t, n_b), dtype=np.float64)
    a_prev = np.empty((n_t, n_b), dtype=np.float64)
    c_norm = np.empty(n_t, dtype=np.float64)

    a0 = np.full(n_b, 1.0 / float(n_b), dtype=np.float64)
    a_prev[0] = a0
    t_a = a0 * e_mat[0]
    c0 = float(np.sum(t_a))
    if not np.isfinite(c0) or c0 <= 0.0:
        alpha[0] = a0
        c_norm[0] = 1.0
    else:
        alpha[0] = t_a / c0
        c_norm[0] = c0

    for t in range(1, n_t):
        at = alpha[t - 1] @ t_mat
        a_prev[t] = at
        ta = at * e_mat[t]
        ct = float(np.sum(ta))
        if not np.isfinite(ct) or ct <= 0.0:
            alpha[t] = a0
            c_norm[t] = 1.0
        else:
            alpha[t] = ta / ct
            c_norm[t] = ct

    if decoder == "causal_forward":
        post = alpha
        cache = {"e_mat": e_mat, "alpha": alpha, "a_prev": a_prev, "c_norm": c_norm}
    elif decoder == "offline_fb":
        beta = np.ones((n_t, n_b), dtype=np.float64)
        m_tilde = np.empty((n_t, n_b), dtype=np.float64)
        s_norm = np.ones(n_t, dtype=np.float64)
        for t in range(n_t - 2, -1, -1):
            mt = e_mat[t + 1] * beta[t + 1]
            m_tilde[t + 1] = mt
            ut = t_mat @ mt
            st = float(np.sum(ut))
            if not np.isfinite(st) or st <= 0.0:
                beta[t] = 1.0
                s_norm[t] = 1.0
            else:
                beta[t] = ut / st
                s_norm[t] = st
        gamma_unnorm = alpha * beta
        z_norm = np.maximum(np.sum(gamma_unnorm, axis=1, keepdims=True), 1e-300)
        post = gamma_unnorm / z_norm
        cache = {
            "e_mat": e_mat,
            "alpha": alpha,
            "a_prev": a_prev,
            "c_norm": c_norm,
            "beta": beta,
            "m_tilde": m_tilde,
            "s_norm": s_norm,
            "z_norm": z_norm[:, 0],
            "post": post,
        }
    else:
        raise ValueError(f"Unsupported decoder: {decoder!r}")

    est_mmse = post @ grid
    est_map = grid[np.argmax(post, axis=1)]
    lp = np.where(post > 1e-300, np.log(np.maximum(post, 1e-300)), 0.0)
    ent = -np.sum(post * lp, axis=1) / max(np.log(float(n_b)), 1e-12)
    var = np.sum(post * (grid[None, :] - est_mmse[:, None]) ** 2, axis=1)
    sd = np.sqrt(np.maximum(var, 0.0))

    preds = {
        "est_mmse": est_mmse,
        "est_map": est_map,
        "post_entropy": np.clip(ent, 0.0, 1.0),
        "post_sd_bpm": sd,
    }
    return preds, cache


def bptt_hmm_adjoint(
    g_yhat: np.ndarray,
    t_mat: np.ndarray,
    grid_bpm: np.ndarray,
    cache: dict[str, Any],
    decoder: str = "causal_forward",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact gradients `(g_ell, g_T)` from prediction gradient `g_yhat`."""
    grid = np.asarray(grid_bpm, dtype=np.float64).ravel()
    alpha = cache["alpha"]
    e_mat = cache["e_mat"]
    c_norm = cache["c_norm"]
    n_t, n_b = alpha.shape

    g_ell = np.zeros((n_t, n_b), dtype=np.float64)
    g_T = np.zeros((n_b, n_b), dtype=np.float64)

    if decoder == "causal_forward":
        g_post = g_yhat[:, None] * grid[None, :]
        acc = np.zeros(n_b, dtype=np.float64)
        for t in range(n_t - 1, -1, -1):
            gt = g_post[t] + acc
            at = alpha[t]
            dt = float(np.dot(gt, at))
            diff_g = gt - dt
            g_ell[t] += diff_g * at
            if t > 0:
                g_tilde = diff_g / c_norm[t]
                g_a = g_tilde * e_mat[t]
                g_T += np.outer(alpha[t - 1], g_a)
                acc = t_mat @ g_a
        return g_ell, g_T

    if decoder == "offline_fb":
        beta = cache["beta"]
        m_tilde = cache["m_tilde"]
        s_norm = cache["s_norm"]
        z_norm = cache["z_norm"]
        post = cache["post"]

        g_post = g_yhat[:, None] * grid[None, :]
        d_p = np.sum(g_post * post, axis=1, keepdims=True)
        g_unnorm = (g_post - d_p) / z_norm[:, None]

        g_alpha = g_unnorm * beta
        g_beta = (g_unnorm * alpha).copy()

        # Backward adjoint wrt reverse beta-sweep (runs forward in time t = 0 .. T-2)
        for t in range(n_t - 1):
            gb = g_beta[t]
            bt = beta[t]
            g_u = (gb - float(np.dot(gb, bt))) / s_norm[t]
            mt1 = m_tilde[t + 1]
            g_T += np.outer(g_u, mt1)
            g_mt1 = t_mat.T @ g_u
            g_ell[t + 1] += g_mt1 * mt1
            g_beta[t + 1] += g_mt1 * e_mat[t + 1]

        # Backward adjoint wrt forward alpha-sweep (runs backward in time t = T-1 .. 0)
        acc = np.zeros(n_b, dtype=np.float64)
        for t in range(n_t - 1, -1, -1):
            gt = g_alpha[t] + acc
            at = alpha[t]
            dt = float(np.dot(gt, at))
            diff_g = gt - dt
            g_ell[t] += diff_g * at
            if t > 0:
                g_tilde = diff_g / c_norm[t]
                g_a = g_tilde * e_mat[t]
                g_T += np.outer(alpha[t - 1], g_a)
                acc = t_mat @ g_a
        return g_ell, g_T

    raise ValueError(f"Unsupported decoder: {decoder!r}")


class DifferentiableSymmetricHMM:
    """Rung 2 structured state-space model (<50 params, exact O(2)xR+ & zero-offset shift eq.)."""

    def __init__(
        self,
        mode: str = "linear",
        decoder: str = "causal_forward",
        grid_bpm: np.ndarray = VT.BR_GRID_1BPM,
        reg_l2: float = 5e-3,
        seed: int = 20261011,
    ) -> None:
        if mode not in ("linear", "linear_filtersafe", "mlp"):
            raise ValueError(f"Unknown mode {mode!r}")
        if decoder not in ("causal_forward", "offline_fb"):
            raise ValueError(f"Unknown decoder {decoder!r}")
        self.mode = mode
        self.decoder = decoder
        self.grid_bpm = np.asarray(grid_bpm, dtype=np.float64).ravel()
        self.reg_l2 = float(reg_l2)
        self.seed = int(seed)
        self.theta, self.bounds = self._init_theta_and_bounds()
        self.n_params = int(self.theta.size)
        self.train_history_: dict[str, Any] = {}

    def _init_theta_and_bounds(self) -> tuple[np.ndarray, list[tuple[float, float]]]:
        t_init = np.array([np.log(2.0), np.log(10.0), -5.0], dtype=np.float64)
        t_bnds = [
            (float(np.log(1.0)), float(np.log(6.0))),
            (float(np.log(5.0)), float(np.log(15.0))),
            (-7.0, -4.5),
        ]
        k_init = np.array([0.0, 0.0], dtype=np.float64)
        k_bnds = [(0.0, 0.5), (0.0, 0.5)]
        w_init = np.array([2.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        w_bnds = [(2.0, 5.0), (0.0, 3.0), (0.0, 3.0), (-0.5, 0.5), (0.0, 5.0), (-0.5, 2.0)]

        if self.mode == "linear":
            theta = np.concatenate([w_init, k_init, t_init])
            bounds = w_bnds + k_bnds + t_bnds
            return theta, bounds

        if self.mode == "linear_filtersafe":
            w_fs_init = np.array([2.0, 1.0, 0.0], dtype=np.float64)
            w_fs_bnds = [(2.0, 5.0), (0.0, 5.0), (-0.5, 2.0)]
            theta = np.concatenate([w_fs_init, k_init, t_init])
            bounds = w_fs_bnds + k_bnds + t_bnds
            return theta, bounds

        rng = np.random.default_rng(self.seed)
        w1 = rng.normal(0.0, 0.05, size=24)
        b1 = np.zeros(4, dtype=np.float64)
        w2 = rng.normal(0.0, 0.03, size=4)
        theta = np.concatenate([w_init, w1, b1, w2, k_init, t_init])
        bounds = w_bnds + [(-2.0, 2.0)] * 28 + [(-0.10, 0.10)] * 4 + k_bnds + t_bnds
        return theta, bounds

    def _trans_params(self, theta: np.ndarray) -> tuple[float, float, float]:
        return float(theta[-3]), float(theta[-2]), float(theta[-1])

    def _regularization_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        g_reg = np.zeros_like(theta)
        if self.reg_l2 <= 0.0:
            return 0.0, g_reg
        prior = np.zeros_like(theta)
        prior[0] = 2.0
        # Do not L2-penalize transition log-scales; penalize emission departures toward untempered prior
        mask = np.ones_like(theta)
        mask[-3:] = 0.0
        diff = (theta - prior) * mask
        loss_reg = float(self.reg_l2 * np.sum(diff ** 2))
        g_reg = 2.0 * self.reg_l2 * diff
        return loss_reg, g_reg

    def loss_and_grad(
        self,
        theta: np.ndarray,
        sessions_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> tuple[float, np.ndarray]:
        """Compute session-equalized smooth RMSE + L2 regularization & exact gradient.

        `sessions_data` elements are `(phi_seq, ref_seq, valid_mask)`.
        """
        lss, lsj, lpi = self._trans_params(theta)
        t_mat, (dt_ss, dt_sj, dt_pi) = build_mixture_transition_and_grad(
            self.grid_bpm, lss, lsj, lpi
        )

        n_sess = len(sessions_data)
        total_loss = 0.0
        total_grad = np.zeros_like(theta)
        g_t_acc = np.zeros_like(t_mat)

        for phi_seq, ref_seq, valid_mask in sessions_data:
            ell, e_cache = hmm_emission_and_grad(phi_seq, theta, mode=self.mode)
            preds, f_cache = forward_hmm_predictions_and_bptt(
                ell, t_mat, self.grid_bpm, decoder=self.decoder
            )
            yhat = preds["est_mmse"]
            n_val = max(1, int(np.sum(valid_mask)))
            err = np.where(valid_mask, yhat - ref_seq, 0.0)
            mse_s = float(np.sum(err * err) / float(n_val))
            smooth_rmse = float(np.sqrt(mse_s + 1e-4))
            total_loss += smooth_rmse / float(n_sess)

            # dLoss/d_yhat = (1 / n_sess) * (1 / smooth_rmse) * err / n_val
            g_yhat = err / (float(n_sess) * smooth_rmse * float(n_val))
            g_ell, g_t = bptt_hmm_adjoint(
                g_yhat, t_mat, self.grid_bpm, f_cache, decoder=self.decoder
            )
            g_t_acc += g_t
            total_grad += hmm_emission_grad_params(g_ell, e_cache, theta.size, mode=self.mode)

        total_grad[-3] += float(np.sum(g_t_acc * dt_ss))
        total_grad[-2] += float(np.sum(g_t_acc * dt_sj))
        total_grad[-1] += float(np.sum(g_t_acc * dt_pi))

        l_reg, g_reg = self._regularization_and_grad(theta)
        return float(total_loss + l_reg), (total_grad + g_reg)

    def fit(
        self,
        sessions_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        maxiter: int = 80,
    ) -> "DifferentiableSymmetricHMM":
        res = optimize.minimize(
            fun=lambda th: self.loss_and_grad(th, sessions_data),
            x0=self.theta.copy(),
            method="L-BFGS-B",
            jac=True,
            bounds=self.bounds,
            options={"maxiter": int(maxiter), "ftol": 1e-8, "gtol": 1e-5},
        )
        self.theta = np.asarray(res.x, dtype=np.float64)
        self.train_history_ = {
            "success": bool(res.success),
            "nit": int(res.nit),
            "nfev": int(res.nfev),
            "fun": float(res.fun),
        }
        return self

    def predict_session(
        self,
        phi_seq: np.ndarray,
        grid_bpm: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        grid = self.grid_bpm if grid_bpm is None else np.asarray(grid_bpm, dtype=np.float64).ravel()
        lss, lsj, lpi = self._trans_params(self.theta)
        t_mat, _ = build_mixture_transition_and_grad(grid, lss, lsj, lpi)
        ell, _ = hmm_emission_and_grad(phi_seq, self.theta, mode=self.mode)
        preds, _ = forward_hmm_predictions_and_bptt(ell, t_mat, grid, decoder=self.decoder)
        return preds

    def interpretable_params(self) -> dict[str, float]:
        lss, lsj, lpi = self._trans_params(self.theta)
        if self.mode == "linear_filtersafe":
            return {
                "w1_p1_clean": float(self.theta[0]),
                "w2_p2_harm": 0.0,
                "w3_p3_harm": 0.0,
                "w4_phalf_sub": 0.0,
                "w5_anchor": float(self.theta[1]),
                "w6_p1_raw": float(self.theta[2]),
                "sigma_slow_bpm": float(np.exp(lss)),
                "sigma_jump_bpm": float(np.exp(lsj)),
                "pi_jump": float(_sigmoid(lpi)),
                "k1": float(self.theta[3]),
                "k2": float(self.theta[4]),
            }
        out = {
            "w1_p1_clean": float(self.theta[0]),
            "w2_p2_harm": float(self.theta[1]),
            "w3_p3_harm": float(self.theta[2]),
            "w4_phalf_sub": float(self.theta[3]),
            "w5_anchor": float(self.theta[4]),
            "w6_p1_raw": float(self.theta[5]),
            "sigma_slow_bpm": float(np.exp(lss)),
            "sigma_jump_bpm": float(np.exp(lsj)),
            "pi_jump": float(_sigmoid(lpi)),
        }
        if self.mode == "linear":
            out["k1"] = float(self.theta[6])
            out["k2"] = float(self.theta[7])
        else:
            out["k1"] = float(self.theta[38])
            out["k2"] = float(self.theta[39])
        return out


# ---------------------------------------------------------------------------
# Rung 3: Unconstrained ML Foils (Exact Zero Frequency-Shift Equivariance Bias)
# ---------------------------------------------------------------------------

def flatten_salience_tensor(phi_seq: np.ndarray) -> np.ndarray:
    """Flatten `(T, N_B, 6)` salience tensor into `(T, N_B * 6)` unconstrained vector."""
    arr = np.asarray(phi_seq, dtype=np.float64)
    return arr.reshape(arr.shape[0], -1)


class CausalDilatedTCN:
    """Strictly causal 15,817-parameter dilated 1D-CNN/TCN baseline (pure NumPy + Adam).

    Takes flattened `(T, 438)` standardized spectral vector and applies:
      Layer 0: Linear(438 -> 24) + ReLU
      Blocks 1..3: Residual strictly causal dilated Conv1D(24 -> 24, kernel=3, dilations=1,2,4) + ReLU
      Readout: Linear(24 -> 1)
    Total parameters: `438*24 + 24 + 3*(3*24*24 + 24) + 24 + 1 = 15,817`.
    """

    def __init__(
        self,
        in_dim: int = 438,
        channels: int = 24,
        dilations: tuple[int, ...] = (1, 2, 4),
        kernel_size: int = 3,
        seed: int = 20261011,
    ) -> None:
        self.in_dim = int(in_dim)
        self.channels = int(channels)
        self.dilations = tuple(int(d) for d in dilations)
        self.kernel_size = int(kernel_size)
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)
        c = self.channels
        k = self.kernel_size
        self.params: dict[str, np.ndarray] = {
            "W0": rng.normal(0.0, np.sqrt(2.0 / self.in_dim), size=(self.in_dim, c)),
            "b0": np.zeros(c, dtype=np.float64),
        }
        for idx, _ in enumerate(self.dilations):
            self.params[f"Wk_{idx}"] = rng.normal(0.0, np.sqrt(2.0 / (k * c)), size=(k, c, c))
            self.params[f"bk_{idx}"] = np.zeros(c, dtype=np.float64)
        self.params["Wo"] = rng.normal(0.0, 1.0 / np.sqrt(c), size=(c,))
        self.params["bo"] = np.array([0.0], dtype=np.float64)

        self.param_keys = tuple(self.params.keys())
        self.n_params = int(sum(arr.size for arr in self.params.values()))
        self.mean_ = np.zeros(self.in_dim, dtype=np.float64)
        self.scale_ = np.ones(self.in_dim, dtype=np.float64)
        self.best_val_rmse_: float = float("nan")
        self.weight_decay_: float = 1e-3
        self.dropout_p_: float = 0.0

    def _forward_raw(
        self,
        x_std: np.ndarray,
        dropout_p: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        dp = float(dropout_p)
        if not (0.0 <= dp < 1.0):
            raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p!r}")
        use_drop = dp > 0.0 and rng is not None
        scale_keep = 1.0 / (1.0 - dp) if use_drop else 1.0

        n_t = x_std.shape[0]
        z0 = x_std @ self.params["W0"] + self.params["b0"]
        h_in0 = np.maximum(0.0, z0)
        if use_drop:
            m0 = (rng.random(h_in0.shape) >= dp).astype(np.float64) * scale_keep
            h = h_in0 * m0
        else:
            m0 = None
            h = h_in0
        caches: list[dict[str, Any]] = []

        for idx, dil in enumerate(self.dilations):
            wk = self.params[f"Wk_{idx}"]
            bk = self.params[f"bk_{idx}"]
            y_lin = np.broadcast_to(bk, (n_t, self.channels)).copy()
            for k_pos in range(self.kernel_size):
                shift = (self.kernel_size - 1 - k_pos) * dil
                if shift == 0:
                    y_lin += h @ wk[k_pos]
                elif shift < n_t:
                    y_lin[shift:] += h[: n_t - shift] @ wk[k_pos]
            act = np.maximum(0.0, y_lin)
            if use_drop:
                mk = (rng.random(act.shape) >= dp).astype(np.float64) * scale_keep
                act_d = act * mk
            else:
                mk = None
                act_d = act
            h_next = h + act_d
            caches.append({"h_in": h, "y_lin": y_lin, "act": act, "mk": mk, "dil": dil})
            h = h_next

        yhat = h @ self.params["Wo"] + float(self.params["bo"][0])
        return yhat, {"x_std": x_std, "z0": z0, "h_in0": h_in0, "m0": m0, "blocks": caches, "h_final": h}

    def _backward_raw(
        self,
        g_yhat: np.ndarray,
        fcache: dict[str, Any],
        weight_decay: float = 0.0,
    ) -> dict[str, np.ndarray]:
        grads: dict[str, np.ndarray] = {}
        h_final = fcache["h_final"]
        grads["Wo"] = h_final.T @ g_yhat + weight_decay * self.params["Wo"]
        grads["bo"] = np.array([float(np.sum(g_yhat))], dtype=np.float64)

        g_h = g_yhat[:, None] * self.params["Wo"][None, :]
        n_t = g_h.shape[0]

        for idx in range(len(self.dilations) - 1, -1, -1):
            bc = fcache["blocks"][idx]
            h_in = bc["h_in"]
            y_lin = bc["y_lin"]
            mk = bc["mk"]
            dil = int(bc["dil"])
            wk = self.params[f"Wk_{idx}"]

            g_act = g_h * mk if mk is not None else g_h
            g_y = np.where(y_lin > 0.0, g_act, 0.0)
            g_wk = np.zeros_like(wk)
            g_h_prev = g_h.copy()

            for k_pos in range(self.kernel_size):
                shift = (self.kernel_size - 1 - k_pos) * dil
                if shift == 0:
                    g_wk[k_pos] = h_in.T @ g_y + weight_decay * wk[k_pos]
                    g_h_prev += g_y @ wk[k_pos].T
                elif shift < n_t:
                    g_wk[k_pos] = h_in[: n_t - shift].T @ g_y[shift:] + weight_decay * wk[k_pos]
                    g_h_prev[: n_t - shift] += g_y[shift:] @ wk[k_pos].T
                else:
                    g_wk[k_pos] = weight_decay * wk[k_pos]

            grads[f"Wk_{idx}"] = g_wk
            grads[f"bk_{idx}"] = np.sum(g_y, axis=0)
            g_h = g_h_prev

        if fcache["m0"] is not None:
            g_h = g_h * fcache["m0"]
        g_z0 = np.where(fcache["z0"] > 0.0, g_h, 0.0)
        x_std = fcache["x_std"]
        grads["W0"] = x_std.T @ g_z0 + weight_decay * self.params["W0"]
        grads["b0"] = np.sum(g_z0, axis=0)
        return grads

    def session_mse_loss_and_grad(
        self,
        sessions_x_std: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        weight_decay: float = 1e-3,
        dropout_p: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, dict[str, np.ndarray]]:
        total_loss = 0.0
        acc_grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        n_s = len(sessions_x_std)

        for x_std, y_ref, v_mask in sessions_x_std:
            yhat, fcache = self._forward_raw(x_std, dropout_p=dropout_p, rng=rng)
            n_val = max(1, int(np.sum(v_mask)))
            err = np.where(v_mask, yhat - y_ref, 0.0)
            total_loss += float(np.sum(err * err) / float(n_val)) / float(n_s)
            g_yhat = (2.0 / (float(n_s) * float(n_val))) * err
            # Add weight decay only once cleanly on average over batch
            g_s = self._backward_raw(g_yhat, fcache, weight_decay=0.0)
            for k in self.param_keys:
                acc_grads[k] += g_s[k]

        if weight_decay > 0.0:
            for k in self.param_keys:
                if k.startswith("W"):
                    w = self.params[k]
                    total_loss += 0.5 * float(weight_decay) * float(np.sum(w * w))
                    acc_grads[k] += float(weight_decay) * w
        return total_loss, acc_grads

    def fit(
        self,
        train_sessions: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        val_sessions: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
        lr: float = 1.5e-3,
        weight_decay: float = 1e-3,
        dropout_p: float = 0.0,
        max_epochs: int = 120,
        patience: int = 20,
    ) -> "CausalDilatedTCN":
        # Standardize strictly from training valid windows
        x_tr_valid = np.concatenate([flatten_salience_tensor(p)[v] for p, _, v in train_sessions], axis=0)
        y_tr_valid = np.concatenate([y[v] for _, y, v in train_sessions], axis=0)
        self.mean_ = np.mean(x_tr_valid, axis=0)
        sd = np.std(x_tr_valid, axis=0)
        self.scale_ = np.where(sd > 1e-6, sd, 1.0)
        self.params["bo"] = np.array([float(np.mean(y_tr_valid))], dtype=np.float64)
        self.weight_decay_ = float(weight_decay)
        self.dropout_p_ = float(dropout_p)

        tr_std = [((flatten_salience_tensor(p) - self.mean_) / self.scale_, y, v) for p, y, v in train_sessions]
        val_std = (
            [((flatten_salience_tensor(p) - self.mean_) / self.scale_, y, v) for p, y, v in val_sessions]
            if val_sessions
            else None
        )

        m_adam = {k: np.zeros_like(v) for k, v in self.params.items()}
        v_adam = {k: np.zeros_like(v) for k, v in self.params.items()}
        best_val = float("inf")
        best_params = {k: v.copy() for k, v in self.params.items()}
        no_imp = 0
        rng_drop = np.random.default_rng(self.seed + 7919) if self.dropout_p_ > 0.0 else None

        for ep in range(1, int(max_epochs) + 1):
            loss, grads = self.session_mse_loss_and_grad(
                tr_std,
                weight_decay=weight_decay,
                dropout_p=self.dropout_p_,
                rng=rng_drop,
            )
            gnorm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.values())))
            clip_coef = min(1.0, 5.0 / max(gnorm, 1e-12))

            b1, b2, eps = 0.9, 0.999, 1e-8
            bc1 = 1.0 - b1 ** ep
            bc2 = 1.0 - b2 ** ep
            for k in self.param_keys:
                gk = grads[k] * clip_coef
                m_adam[k] = b1 * m_adam[k] + (1.0 - b1) * gk
                v_adam[k] = b2 * v_adam[k] + (1.0 - b2) * (gk * gk)
                m_hat = m_adam[k] / bc1
                v_hat = v_adam[k] / bc2
                self.params[k] -= float(lr) * m_hat / (np.sqrt(v_hat) + eps)

            if val_std:
                val_rmses = []
                for xs, yv, vv in val_std:
                    if not np.any(vv):
                        continue
                    yh, _ = self._forward_raw(xs, dropout_p=0.0, rng=None)
                    val_rmses.append(float(np.sqrt(np.mean((yh[vv] - yv[vv]) ** 2))))
                score = float(np.mean(val_rmses)) if val_rmses else loss
            else:
                score = loss

            if score < best_val - 1e-5:
                best_val = score
                best_params = {k: v.copy() for k, v in self.params.items()}
                no_imp = 0
            else:
                no_imp += 1
                if val_std and no_imp >= int(patience):
                    break

        self.params = best_params
        self.best_val_rmse_ = float(best_val)
        return self

    def predict_session(self, phi_seq: np.ndarray) -> np.ndarray:
        x_std = (flatten_salience_tensor(phi_seq) - self.mean_) / self.scale_
        yhat, _ = self._forward_raw(x_std, dropout_p=0.0, rng=None)
        return np.clip(yhat, 12.0, 120.0)


def fit_select_causal_tcn(
    train_sessions: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    val_sessions: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None,
    lr: float = 1.5e-3,
    grid: tuple[tuple[float, float], ...] = ((1e-3, 0.0), (1e-2, 0.0), (1e-2, 0.15), (5e-2, 0.25)),
    max_epochs: int = 100,
    patience: int = 18,
    seed: int = 20261011,
) -> tuple[CausalDilatedTCN, dict[str, Any]]:
    """Select `(weight_decay, dropout_p)` for `CausalDilatedTCN` strictly on inner validation."""
    in_dim = int(flatten_salience_tensor(train_sessions[0][0]).shape[1])
    best_model: CausalDilatedTCN | None = None
    best_val = float("inf")
    best_cfg: tuple[float, float] = (float(grid[0][0]), float(grid[0][1]))
    grid_val_rmses: dict[str, float] = {}

    for idx, (wd, dp) in enumerate(grid):
        mdl = CausalDilatedTCN(in_dim=in_dim, channels=24, dilations=(1, 2, 4), seed=int(seed) + 1009 * idx)
        mdl.fit(
            train_sessions,
            val_sessions,
            lr=float(lr),
            weight_decay=float(wd),
            dropout_p=float(dp),
            max_epochs=int(max_epochs),
            patience=int(patience),
        )
        key = f"wd={float(wd):g},drop={float(dp):g}"
        grid_val_rmses[key] = float(mdl.best_val_rmse_)
        if best_model is None or mdl.best_val_rmse_ < best_val:
            best_val = float(mdl.best_val_rmse_)
            best_model = mdl
            best_cfg = (float(wd), float(dp))

    assert best_model is not None
    info = {
        "selected_weight_decay": float(best_cfg[0]),
        "selected_dropout_p": float(best_cfg[1]),
        "best_val_rmse": float(best_val),
        "grid_val_rmses": grid_val_rmses,
    }
    return best_model, info


def fit_predict_spectral_mlp(
    train_sessions: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    val_sessions: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    test_phi_list: list[np.ndarray],
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
    seed: int = 20261011,
) -> dict[str, Any]:
    """Fit unconstrained Spectral MLP `(438 -> 32 -> 16 -> 1)` with inner-val `alpha` & early-stop selection."""
    x_tr = np.concatenate([flatten_salience_tensor(p)[v] for p, _, v in train_sessions], axis=0)
    y_tr = np.concatenate([y[v] for _, y, v in train_sessions], axis=0)
    mean_x = np.mean(x_tr, axis=0)
    sd = np.std(x_tr, axis=0)
    scale_x = np.where(sd > 1e-6, sd, 1.0)
    x_tr_s = (x_tr - mean_x) / scale_x

    val_sess_s = [
        ((flatten_salience_tensor(p) - mean_x) / scale_x, y, v)
        for p, y, v in val_sessions
        if np.any(v)
    ]
    has_val = bool(val_sess_s)

    best_overall_val = float("inf")
    best_alpha = float(alpha_grid[0])
    best_mlp: MLPRegressor | None = None
    best_overall_coefs: list[np.ndarray] | None = None
    best_overall_inters: list[np.ndarray] | None = None
    grid_val_rmses: dict[str, float] = {}

    for idx_a, alpha_val in enumerate(alpha_grid):
        mlp = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=float(alpha_val),
            batch_size=min(256, max(16, x_tr_s.shape[0] // 8)),
            learning_rate_init=1e-3,
            max_iter=1,
            warm_start=True,
            shuffle=True,
            random_state=int(seed) + 1009 * idx_a,
        )

        best_val = float("inf")
        best_coefs = None
        best_inters = None
        no_imp = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for _ in range(120):
                mlp.fit(x_tr_s, y_tr)
                if has_val:
                    sess_rmses = [
                        float(np.sqrt(np.mean((mlp.predict(xs[vv]) - yv[vv]) ** 2)))
                        for xs, yv, vv in val_sess_s
                    ]
                    score = float(np.mean(sess_rmses))
                else:
                    pt = mlp.predict(x_tr_s)
                    score = float(np.sqrt(np.mean((pt - y_tr) ** 2)))
                if score < best_val - 1e-5:
                    best_val = score
                    best_coefs = [c.copy() for c in mlp.coefs_]
                    best_inters = [b.copy() for b in mlp.intercepts_]
                    no_imp = 0
                else:
                    no_imp += 1
                    if has_val and no_imp >= 20:
                        break

        grid_val_rmses[str(float(alpha_val))] = float(best_val)
        if best_mlp is None or best_val < best_overall_val:
            best_overall_val = float(best_val)
            best_alpha = float(alpha_val)
            best_mlp = mlp
            best_overall_coefs = best_coefs
            best_overall_inters = best_inters

    assert best_mlp is not None
    if best_overall_coefs is not None and best_overall_inters is not None:
        best_mlp.coefs_ = best_overall_coefs
        best_mlp.intercepts_ = best_overall_inters

    n_params = int(sum(c.size for c in best_mlp.coefs_) + sum(b.size for b in best_mlp.intercepts_))
    preds = [
        np.clip(best_mlp.predict((flatten_salience_tensor(p) - mean_x) / scale_x), 12.0, 120.0)
        for p in test_phi_list
    ]
    return {
        "model": best_mlp,
        "mean": mean_x,
        "scale": scale_x,
        "n_params": n_params,
        "preds": preds,
        "selected_alpha": float(best_alpha),
        "best_val_rmse": float(best_overall_val),
        "grid_val_rmses": grid_val_rmses,
    }


def _svd_weighted_ridge_paths(
    x_tr_s: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    alphas: np.ndarray,
) -> list[tuple[np.ndarray, float]]:
    """Exact SVD closed-form weighted Ridge solutions matching `sklearn.linear_model.Ridge`."""
    w = np.asarray(w_tr, dtype=np.float64)
    w_sum = float(np.sum(w))
    xw_mean = np.sum(x_tr_s * w[:, None], axis=0) / w_sum
    yw_mean = float(np.sum(y_tr * w) / w_sum)
    xc = x_tr_s - xw_mean[None, :]
    yc = y_tr - yw_mean
    sw_sqrt = np.sqrt(w)
    u_mat, s_vals, vt_mat = np.linalg.svd(xc * sw_sqrt[:, None], full_matrices=False)
    uty = u_mat.T @ (yc * sw_sqrt)
    paths: list[tuple[np.ndarray, float]] = []
    for alpha_v in alphas:
        coef = vt_mat.T @ ((s_vals / (s_vals * s_vals + float(alpha_v))) * uty)
        inter = float(yw_mean - np.dot(xw_mean, coef))
        paths.append((coef, inter))
    return paths


def fit_predict_ridge_spectral(
    train_sessions: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    test_phi_list: list[np.ndarray],
    train_patient_ids: list[int] | None = None,
    alphas: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit unconstrained Ridge regression `(438 -> 1, 439 params)` with patient-LOPO CV when available."""
    alphas_arr = np.logspace(-1, 5, 13) if alphas is None else np.asarray(alphas, dtype=np.float64).ravel()
    if train_patient_ids is not None and len(train_patient_ids) != len(train_sessions):
        raise ValueError("train_patient_ids length must match train_sessions")

    xs, ys, sw, valid_sess = [], [], [], []
    for idx_s, (phi, y, v) in enumerate(train_sessions):
        n_val = int(np.sum(v))
        if n_val == 0:
            continue
        xv = flatten_salience_tensor(phi)[v]
        yv = np.asarray(y[v], dtype=np.float64)
        xs.append(xv)
        ys.append(yv)
        sw.append(np.full(n_val, 1.0 / float(n_val), dtype=np.float64))
        if train_patient_ids is not None:
            valid_sess.append((xv, yv, int(train_patient_ids[idx_s])))

    x_tr = np.concatenate(xs, axis=0)
    y_tr = np.concatenate(ys, axis=0)
    w_tr = np.concatenate(sw, axis=0)
    w_tr = w_tr * (len(w_tr) / np.sum(w_tr))

    mean_x = np.mean(x_tr, axis=0)
    sd = np.std(x_tr, axis=0)
    scale_x = np.where(sd > 1e-6, sd, 1.0)
    x_tr_s = (x_tr - mean_x) / scale_x

    distinct_pids = sorted({pid for _, _, pid in valid_sess}) if train_patient_ids is not None else []
    if len(distinct_pids) >= 2:
        alpha_sess_rmses: list[list[float]] = [[] for _ in range(alphas_arr.size)]
        for hold_pid in distinct_pids:
            in_xs, in_ys, in_sw, val_xy = [], [], [], []
            for xv, yv, spid in valid_sess:
                if spid == hold_pid:
                    val_xy.append((xv, yv))
                else:
                    in_xs.append(xv)
                    in_ys.append(yv)
                    in_sw.append(np.full(xv.shape[0], 1.0 / float(xv.shape[0]), dtype=np.float64))
            x_in = np.concatenate(in_xs, axis=0)
            y_in = np.concatenate(in_ys, axis=0)
            w_in = np.concatenate(in_sw, axis=0)
            w_in = w_in * (len(w_in) / np.sum(w_in))

            m_in = np.mean(x_in, axis=0)
            sd_in = np.std(x_in, axis=0)
            sc_in = np.where(sd_in > 1e-6, sd_in, 1.0)
            x_in_s = (x_in - m_in) / sc_in
            paths = _svd_weighted_ridge_paths(x_in_s, y_in, w_in, alphas_arr)
            val_xy_s = [((xv - m_in) / sc_in, yv) for xv, yv in val_xy]

            for idx_a, (coef_a, inter_a) in enumerate(paths):
                for xvs, yvs in val_xy_s:
                    pred_v = xvs @ coef_a + inter_a
                    alpha_sess_rmses[idx_a].append(float(np.sqrt(np.mean((pred_v - yvs) ** 2))))

        mean_cv_rmses = [float(np.mean(rmses)) for rmses in alpha_sess_rmses]
        best_idx = int(np.argmin(mean_cv_rmses))
        alpha_best = float(alphas_arr[best_idx])
        ridge = Ridge(alpha=alpha_best)
        ridge.fit(x_tr_s, y_tr, sample_weight=w_tr)
        inner_cv_mode = "patient_lopo"
        inner_cv_rmse = float(mean_cv_rmses[best_idx])
    else:
        ridge = RidgeCV(alphas=alphas_arr)
        ridge.fit(x_tr_s, y_tr, sample_weight=w_tr)
        alpha_best = float(ridge.alpha_)
        inner_cv_mode = "gcv_fallback"
        inner_cv_rmse = float("nan")

    n_params = int(ridge.coef_.size + 1)
    preds = [
        np.clip(ridge.predict((flatten_salience_tensor(p) - mean_x) / scale_x), 12.0, 120.0)
        for p in test_phi_list
    ]
    return {
        "model": ridge,
        "mean": mean_x,
        "scale": scale_x,
        "n_params": n_params,
        "preds": preds,
        "selected_alpha": float(alpha_best),
        "inner_cv_mode": inner_cv_mode,
        "inner_lopo_rmse": inner_cv_rmse,
    }
