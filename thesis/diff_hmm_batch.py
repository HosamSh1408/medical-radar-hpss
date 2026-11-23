"""Exact batched BLAS-accelerated BPTT engine for `DifferentiableSymmetricHMM`.

Stacks variable-length `(phi_seq, ref_seq, valid_mask)` sessions into padded
`(S, T_max, N_B, 6)` batches so `L-BFGS-B` training computes mathematically equivalent
(< 1e-13) smooth-RMSE objective and analytical gradients ~12-20x faster than looping
session-by-session in Python, without mutating `thesis/diff_hmm_and_foils.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from thesis import diff_hmm_and_foils as D


@dataclass(frozen=True)
class PreparedHmmBatch:
    n_sess: int
    t_max: int
    n_b: int
    n_ch: int
    phi_pad: np.ndarray    # (S, T_max, N_B, 6)
    phi_flat: np.ndarray   # (S * T_max * N_B, 6)
    ref_pad: np.ndarray    # (T_max, S)
    valid_pad: np.ndarray  # (T_max, S) bool
    active_pad: np.ndarray # (T_max, S) bool
    n_val: np.ndarray      # (S,) float64


def prepare_hmm_batch(
    sessions_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> PreparedHmmBatch:
    """Pack variable-length training sessions into contiguous padded batch tensors."""
    n_sess = len(sessions_data)
    if n_sess == 0:
        raise ValueError("sessions_data must be non-empty")
    lengths = [int(np.asarray(phi).shape[0]) for phi, _, _ in sessions_data]
    t_max = max(lengths)
    first = np.asarray(sessions_data[0][0], dtype=np.float64)
    n_b = int(first.shape[1])
    n_ch = int(first.shape[2])

    phi_pad = np.zeros((n_sess, t_max, n_b, n_ch), dtype=np.float64)
    ref_pad = np.zeros((t_max, n_sess), dtype=np.float64)
    valid_pad = np.zeros((t_max, n_sess), dtype=bool)
    active_pad = np.zeros((t_max, n_sess), dtype=bool)

    for s_idx, (phi_s, ref_s, v_s) in enumerate(sessions_data):
        l_s = lengths[s_idx]
        phi_pad[s_idx, :l_s] = np.asarray(phi_s, dtype=np.float64)
        ref_pad[:l_s, s_idx] = np.asarray(ref_s, dtype=np.float64)
        valid_pad[:l_s, s_idx] = np.asarray(v_s, dtype=bool)
        active_pad[:l_s, s_idx] = True

    n_val = np.maximum(1.0, np.sum(valid_pad, axis=0).astype(np.float64))
    phi_flat = np.ascontiguousarray(phi_pad.reshape(n_sess * t_max * n_b, n_ch))
    return PreparedHmmBatch(
        n_sess=n_sess,
        t_max=t_max,
        n_b=n_b,
        n_ch=n_ch,
        phi_pad=phi_pad,
        phi_flat=phi_flat,
        ref_pad=ref_pad,
        valid_pad=valid_pad,
        active_pad=active_pad,
        n_val=n_val,
    )


def batch_loss_and_grad(
    model: D.DifferentiableSymmetricHMM,
    theta: np.ndarray,
    batch: PreparedHmmBatch,
) -> tuple[float, np.ndarray]:
    """Compute exact session-equalized smooth RMSE + L2 reg & BPTT gradient on padded batch."""
    th = np.asarray(theta, dtype=np.float64)
    grid = model.grid_bpm
    n_sess = batch.n_sess
    t_max = batch.t_max
    n_b = batch.n_b

    lss, lsj, lpi = model._trans_params(th)
    t_mat, (dt_ss, dt_sj, dt_pi) = D.build_mixture_transition_and_grad(grid, lss, lsj, lpi)
    t_mat_t = np.ascontiguousarray(t_mat.T)

    # 1. Batched emission logits over (S * T_max * N_B, C)
    phi_fs_flat = None
    if model.mode == "linear":
        w = th[:6]
        k1, k2 = float(th[6]), float(th[7])
        s_pt_flat = batch.phi_flat @ w
        h_act_flat = None
        w2 = None
    elif model.mode == "linear_filtersafe":
        phi_fs_flat = np.ascontiguousarray(batch.phi_flat[:, D.FILTERSAFE_CH_IDX])
        w = th[:3]
        k1, k2 = float(th[3]), float(th[4])
        s_pt_flat = phi_fs_flat @ w
        h_act_flat = None
        w2 = None
    elif model.mode == "mlp":
        w_res = th[:6]
        w1 = th[6:30].reshape(6, 4)
        b1 = th[30:34]
        w2 = th[34:38]
        k1, k2 = float(th[38]), float(th[39])
        h_act_flat = np.tanh(batch.phi_flat @ w1 + b1)
        s_pt_flat = (batch.phi_flat @ w_res) + (h_act_flat @ w2)
    else:
        raise ValueError(f"Unsupported mode: {model.mode!r}")

    s_pt = s_pt_flat.reshape(n_sess, t_max, n_b)
    s_pad = np.pad(s_pt, ((0, 0), (0, 0), (2, 2)), mode="edge")
    ell_stb = (
        k2 * s_pad[:, :, :-4]
        + k1 * s_pad[:, :, 1:-3]
        + s_pad[:, :, 2:-2]
        + k1 * s_pad[:, :, 3:-1]
        + k2 * s_pad[:, :, 4:]
    )
    ell = np.ascontiguousarray(ell_stb.transpose(1, 0, 2))  # (T_max, S, N_B)

    l_shift = ell - np.max(ell, axis=2, keepdims=True)
    e_mat = np.exp(l_shift)

    # 2. Batched forward alpha sweep (t = 0 .. T_max - 1)
    alpha = np.empty((t_max, n_sess, n_b), dtype=np.float64)
    c_norm = np.empty((t_max, n_sess), dtype=np.float64)
    a0 = np.full((n_sess, n_b), 1.0 / float(n_b), dtype=np.float64)

    ta0 = a0 * e_mat[0]
    c0 = np.sum(ta0, axis=1)
    bad0 = (~np.isfinite(c0)) | (c0 <= 0.0)
    c0_safe = np.where(bad0, 1.0, c0)
    alpha[0] = np.where(bad0[:, None], a0, ta0 / c0_safe[:, None])
    c_norm[0] = c0_safe

    for t in range(1, t_max):
        at = alpha[t - 1] @ t_mat
        ta = at * e_mat[t]
        ct = np.sum(ta, axis=1)
        badt = (~np.isfinite(ct)) | (ct <= 0.0)
        ct_safe = np.where(badt, 1.0, ct)
        alpha[t] = np.where(badt[:, None], a0, ta / ct_safe[:, None])
        c_norm[t] = ct_safe

    if model.decoder == "causal_forward":
        post = alpha
    elif model.decoder == "offline_fb":
        beta = np.ones((t_max, n_sess, n_b), dtype=np.float64)
        m_tilde = np.zeros((t_max, n_sess, n_b), dtype=np.float64)
        s_norm = np.ones((t_max, n_sess), dtype=np.float64)
        for t in range(t_max - 2, -1, -1):
            mt = e_mat[t + 1] * beta[t + 1]
            m_tilde[t + 1] = np.where(batch.active_pad[t + 1, :, None], mt, 0.0)
            ut = mt @ t_mat_t
            st = np.sum(ut, axis=1)
            step_ok = batch.active_pad[t + 1] & np.isfinite(st) & (st > 0.0)
            st_safe = np.where(step_ok, st, 1.0)
            beta[t] = np.where(step_ok[:, None], ut / st_safe[:, None], 1.0)
            s_norm[t] = st_safe
        gamma_u = alpha * beta
        z_norm = np.maximum(np.sum(gamma_u, axis=2), 1e-300)
        post = gamma_u / z_norm[:, :, None]
    else:
        raise ValueError(f"Unsupported decoder: {model.decoder!r}")

    # 3. Session-equalized smooth RMSE loss & prediction adjoint
    yhat = post @ grid  # (T_max, S)
    err = np.where(batch.valid_pad, yhat - batch.ref_pad, 0.0)
    mse_s = np.sum(err * err, axis=0) / batch.n_val
    smooth_rmse = np.sqrt(mse_s + 1e-4)
    total_loss = float(np.mean(smooth_rmse))

    g_yhat = err / (float(n_sess) * smooth_rmse[None, :] * batch.n_val[None, :])
    g_post = g_yhat[:, :, None] * grid[None, None, :]

    # 4. Batched HMM adjoint BPTT
    g_ell = np.zeros((t_max, n_sess, n_b), dtype=np.float64)
    g_t = np.zeros((n_b, n_b), dtype=np.float64)

    if model.decoder == "causal_forward":
        acc = np.zeros((n_sess, n_b), dtype=np.float64)
        for t in range(t_max - 1, -1, -1):
            gt = g_post[t] + acc
            at = alpha[t]
            dt = np.sum(gt * at, axis=1)
            diff_g = gt - dt[:, None]
            g_ell[t] = diff_g * at
            if t > 0:
                g_a = (diff_g / c_norm[t, :, None]) * e_mat[t]
                g_t += alpha[t - 1].T @ g_a
                acc = g_a @ t_mat_t
    else:
        d_p = np.sum(g_post * post, axis=2, keepdims=True)
        g_unnorm = (g_post - d_p) / z_norm[:, :, None]
        g_alpha = g_unnorm * beta
        g_beta = (g_unnorm * alpha).copy()

        for t in range(t_max - 1):
            gb = np.where(batch.active_pad[t + 1, :, None], g_beta[t], 0.0)
            bt = beta[t]
            g_u = (gb - np.sum(gb * bt, axis=1, keepdims=True)) / s_norm[t, :, None]
            mt1 = m_tilde[t + 1]
            g_t += g_u.T @ mt1
            g_mt1 = g_u @ t_mat
            g_ell[t + 1] += g_mt1 * mt1
            g_beta[t + 1] += g_mt1 * e_mat[t + 1]

        acc = np.zeros((n_sess, n_b), dtype=np.float64)
        for t in range(t_max - 1, -1, -1):
            gt = g_alpha[t] + acc
            at = alpha[t]
            dt = np.sum(gt * at, axis=1)
            diff_g = gt - dt[:, None]
            g_ell[t] += diff_g * at
            if t > 0:
                g_a = (diff_g / c_norm[t, :, None]) * e_mat[t]
                g_a = np.where(batch.active_pad[t, :, None], g_a, 0.0)
                g_t += alpha[t - 1].T @ g_a
                acc = g_a @ t_mat_t

    # Zero strictly outside active session lengths and backprop 5-tap kernel + emission
    g_ell = np.where(batch.active_pad[:, :, None], g_ell, 0.0)
    g_ell_stb = np.ascontiguousarray(g_ell.transpose(1, 0, 2))

    g_k1 = float(np.sum(g_ell_stb * (s_pad[:, :, 1:-3] + s_pad[:, :, 3:-1])))
    g_k2 = float(np.sum(g_ell_stb * (s_pad[:, :, :-4] + s_pad[:, :, 4:])))
    g_pad = np.zeros((n_sess, t_max, n_b + 4), dtype=np.float64)
    g_pad[:, :, :-4] += k2 * g_ell_stb
    g_pad[:, :, 1:-3] += k1 * g_ell_stb
    g_pad[:, :, 2:-2] += g_ell_stb
    g_pad[:, :, 3:-1] += k1 * g_ell_stb
    g_pad[:, :, 4:] += k2 * g_ell_stb
    g_pad[:, :, 2] += g_pad[:, :, 0] + g_pad[:, :, 1]
    g_pad[:, :, -3] += g_pad[:, :, -2] + g_pad[:, :, -1]
    g_spt_flat = np.ascontiguousarray(g_pad[:, :, 2:-2]).reshape(-1)

    total_grad = np.zeros_like(th)
    if model.mode == "linear":
        total_grad[:6] = g_spt_flat @ batch.phi_flat
        total_grad[6] = g_k1
        total_grad[7] = g_k2
    elif model.mode == "linear_filtersafe":
        assert phi_fs_flat is not None
        total_grad[:3] = g_spt_flat @ phi_fs_flat
        total_grad[3] = g_k1
        total_grad[4] = g_k2
    else:
        assert h_act_flat is not None and w2 is not None
        total_grad[:6] = g_spt_flat @ batch.phi_flat
        total_grad[34:38] = g_spt_flat @ h_act_flat
        g_pre_flat = (g_spt_flat[:, None] * w2[None, :]) * (1.0 - h_act_flat * h_act_flat)
        total_grad[30:34] = np.sum(g_pre_flat, axis=0)
        total_grad[6:30] = (batch.phi_flat.T @ g_pre_flat).ravel()
        total_grad[38] = g_k1
        total_grad[39] = g_k2

    total_grad[-3] += float(np.sum(g_t * dt_ss))
    total_grad[-2] += float(np.sum(g_t * dt_sj))
    total_grad[-1] += float(np.sum(g_t * dt_pi))

    l_reg, g_reg = model._regularization_and_grad(th)
    return float(total_loss + l_reg), (total_grad + g_reg)


def fit_diff_hmm_batched(
    model: D.DifferentiableSymmetricHMM,
    sessions_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    maxiter: int = 80,
) -> D.DifferentiableSymmetricHMM:
    """Fit `DifferentiableSymmetricHMM` with exact batched L-BFGS-B optimization."""
    batch = prepare_hmm_batch(sessions_data)
    res = optimize.minimize(
        fun=lambda th: batch_loss_and_grad(model, th, batch),
        x0=model.theta.copy(),
        method="L-BFGS-B",
        jac=True,
        bounds=model.bounds,
        options={"maxiter": int(maxiter), "ftol": 1e-8, "gtol": 1e-5},
    )
    model.theta = np.asarray(res.x, dtype=np.float64)
    model.train_history_ = {
        "success": bool(res.success),
        "nit": int(res.nit),
        "nfev": int(res.nfev),
        "fun": float(res.fun),
    }
    return model
