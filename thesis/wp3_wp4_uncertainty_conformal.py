"""Stage 3 (WP3/WP4): Bayesian & Spectral Uncertainty Quantification, Selective Prediction,
Quiet-Apnea Pause Safety, OOD/WGN Abstention & Finite-K Hierarchical Conformal Calibration.

Implements `thesis/analysis_plan_v3_ai.md` Section 5 and Section 9 (`stage3-uncertainty-v1`).
All mathematical symbols and identifiers adhere strictly to
`paper_orchestra_bundle/raw_materials/notation_glossary.tex`.
Never mutates any file inside `paper_workspace/`, `beltrao_data/`, or prior stage caches.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy import optimize, stats

from thesis import io_utils as IO
from thesis import viterbi_tracker as VT


@lru_cache(maxsize=16)
def _bootstrap_index_matrix(n_p: int, n_boot: int, seed: int) -> np.ndarray:
    """Cache the `(n_boot, n_p)` patient-cluster resample index matrix matching `IO.patient_clustered_bootstrap`."""
    rng = np.random.default_rng(int(seed))
    return np.array([rng.integers(0, n_p, size=n_p) for _ in range(n_boot)], dtype=np.intp)


def fast_patient_clustered_bootstrap(
    patient_values: dict[int, np.ndarray],
    n_boot: int = 10000,
    seed: int = IO.RANDOM_SEED,
) -> dict[str, float]:
    """Vectorized 10,000-resample patient-clustered bootstrap matching `IO.patient_clustered_bootstrap` within `< 1e-15`."""
    pats = sorted(p for p, v in patient_values.items() if np.asarray(v).size > 0)
    if not pats:
        return {"mean": float("nan"), "patient_mean_of_means": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    pat_means = np.array([float(np.mean(patient_values[p])) for p in pats], dtype=np.float64)
    s_vec = np.array([float(np.sum(patient_values[p])) for p in pats], dtype=np.float64)
    c_vec = np.array([float(np.asarray(patient_values[p]).size) for p in pats], dtype=np.float64)
    idx_mat = _bootstrap_index_matrix(len(pats), int(n_boot), int(seed))
    boot_means = np.sum(s_vec[idx_mat], axis=1) / np.sum(c_vec[idx_mat], axis=1)
    obs = float(np.sum(s_vec) / np.sum(c_vec))
    return {
        "mean": obs,
        "patient_mean_of_means": float(pat_means.mean()),
        "ci95_lo": float(np.percentile(boot_means, 2.5)),
        "ci95_hi": float(np.percentile(boot_means, 97.5)),
    }


# Uniform discrete distribution standard deviation on locked 1-bpm grid [18, 90] bpm (N_B = 73)
UNIFORM_GRID_SD_1BPM: float = float(np.std(VT.BR_GRID_1BPM, ddof=0))

COV_GRID_51: np.ndarray = np.linspace(0.50, 1.00, 51, dtype=np.float64)
COV_GRID_91: np.ndarray = np.linspace(0.10, 1.00, 91, dtype=np.float64)

ALPHA_CONFORMAL_GRID: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)

PROXY_ORDER: tuple[str, ...] = (
    "U1_Posterior_Entropy",
    "U2_Posterior_SD",
    "U3_Spectral_Entropy_Clean",
    "U4_Spectral_Entropy_Raw",
    "U5_HQI_Harmonic_Deficit",
    "U6a_ZeroTuned_Additive_SQI",
    "U6b_ZeroTuned_Geometric_SQI",
    "U6c_LOPO_Calibrated_SQI",
)

PROXY_META: dict[str, dict[str, Any]] = {
    "U1_Posterior_Entropy": {
        "short_label": "U1 Post-Entropy",
        "symbol_latex": r"$H_t$",
        "n_tuned_params": 0,
        "uses_temporal_hmm": True,
        "uses_spectral_ood": False,
        "o2_scale_invariant": True,
        "block_lookahead_s": 0.0,
    },
    "U2_Posterior_SD": {
        "short_label": "U2 Post-SD",
        "symbol_latex": r"$\tilde{\sigma}_{\mathrm{post},t}$",
        "n_tuned_params": 0,
        "uses_temporal_hmm": True,
        "uses_spectral_ood": False,
        "o2_scale_invariant": True,
        "block_lookahead_s": 0.0,
    },
    "U3_Spectral_Entropy_Clean": {
        "short_label": "U3 Spec-Ent (HPSS)",
        "symbol_latex": r"$H_{\mathrm{spec},t}$",
        "n_tuned_params": 0,
        "uses_temporal_hmm": False,
        "uses_spectral_ood": True,
        "o2_scale_invariant": True,
        "block_lookahead_s": 0.0,
    },
    "U4_Spectral_Entropy_Raw": {
        "short_label": "U4 Spec-Ent (Raw)",
        "symbol_latex": r"$H_{\mathrm{raw},t}$",
        "n_tuned_params": 0,
        "uses_temporal_hmm": False,
        "uses_spectral_ood": True,
        "o2_scale_invariant": True,
        "block_lookahead_s": 0.0,
    },
    "U5_HQI_Harmonic_Deficit": {
        "short_label": "U5 1-HQI",
        "symbol_latex": r"$1 - \mathrm{HQI}_t$",
        "n_tuned_params": 0,
        "uses_temporal_hmm": False,
        "uses_spectral_ood": True,
        "o2_scale_invariant": True,
        "block_lookahead_s": 0.0,
    },
    "U6a_ZeroTuned_Additive_SQI": {
        "short_label": "U6a Additive-SQI (0p)",
        "symbol_latex": r"$1 - \mathrm{SQI}_{\mathrm{add},t}$",
        "n_tuned_params": 0,
        "uses_temporal_hmm": True,
        "uses_spectral_ood": True,
        "o2_scale_invariant": True,
        "block_lookahead_s": 0.0,
    },
    "U6b_ZeroTuned_Geometric_SQI": {
        "short_label": "U6b Geom-SQI (0p)",
        "symbol_latex": r"$1 - \mathrm{SQI}_{\mathrm{geom},t}$",
        "n_tuned_params": 0,
        "uses_temporal_hmm": True,
        "uses_spectral_ood": True,
        "o2_scale_invariant": True,
        "block_lookahead_s": 0.0,
    },
    "U6c_LOPO_Calibrated_SQI": {
        "short_label": "U6c LOPO-SQI (5p)",
        "symbol_latex": r"$1 - \mathrm{SQI}_{\mathrm{LOPO},t}$",
        "n_tuned_params": 5,
        "uses_temporal_hmm": True,
        "uses_spectral_ood": True,
        "o2_scale_invariant": True,
        "block_lookahead_s": 0.0,
    },
}

CONFORMAL_REGIME_ORDER: tuple[str, ...] = (
    "C1_Naive_RandomWindowSplit_50_50",
    "C2_Naive_LOPO_PooledWindow",
    "C3_SessionCluster_Hierarchical_LOPO",
    "C4_PatientExchangeable_Hierarchical_LOPO",
    "C5_LocallyAdaptive_BayesConformal_Hierarchical_LOPO",
)


# ---------------------------------------------------------------------------
# 1. Intrinsic Spectral, Harmonic & Bayesian Posterior Uncertainty Operators
# ---------------------------------------------------------------------------

def spectral_shannon_entropy_from_log_salience(
    log_p_col: np.ndarray,
    eps0: float = 1e-3,
) -> np.ndarray:
    """Compute normalized discrete Shannon entropy `H_spec in [0, 1]` from log-salience `(T, N_B)`.

    Given `log_p_col[t, q] = log(P_norm(f_q) + eps0)`, recovers `P_norm(f_q) >= 0`, normalizes
    across `q = 1..N_B` to a probability mass function `\\tilde{P}_t(f_q)`, and computes
    `-\\sum_q \\tilde{P}_t(f_q) \\ln \\tilde{P}_t(f_q) / \\ln N_B \\in [0, 1]`.
    Strictly $O(2) \\times \\mathbb{R}_{>0}$-invariant and `0 s` block-causal.
    """
    arr = np.asarray(log_p_col, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    n_t, n_b = arr.shape
    if n_t == 0 or n_b <= 1:
        return np.zeros(n_t, dtype=np.float64)

    p = np.maximum(np.exp(arr) - float(eps0), 0.0)
    row_sums = np.sum(p, axis=1, keepdims=True)
    unif = 1.0 / float(n_b)
    pn = np.where(row_sums > 1e-12, p / np.maximum(row_sums, 1e-12), unif)
    lp = np.where(pn > 1e-300, np.log(np.maximum(pn, 1e-300)), 0.0)
    ent = -np.sum(pn * lp, axis=1) / np.log(float(n_b))
    return np.clip(ent, 0.0, 1.0)


def harmonic_quality_index_from_salience(
    phi_seq: np.ndarray,
    eps0: float = 1e-3,
    half_width_bins: int = 2,
) -> np.ndarray:
    """Compute Harmonic Quality Index `\\hqiScore in [0, 1]` from `(T, N_B, 6)` salience tensor.

    Measures fundamental peak energy concentration (`+-2 bpm` around `argmax_q P_clean(1f)`)
    plus aligned 2nd-harmonic support `0.5 * P_clean(2f)` against total passband energy and
    subharmonic (`0.5f`) background. Strictly $O(2) \\times \\mathbb{R}_{>0}$-invariant.
    """
    phi = np.asarray(phi_seq, dtype=np.float64)
    if phi.ndim == 2:
        phi = phi[None, :, :]
    n_t, n_b, _ = phi.shape
    if n_t == 0:
        return np.empty(0, dtype=np.float64)

    p1 = np.maximum(np.exp(phi[:, :, 0]) - float(eps0), 0.0)
    p2 = np.maximum(np.exp(phi[:, :, 1]) - float(eps0), 0.0)
    p_half = np.maximum(np.exp(phi[:, :, 3]) - float(eps0), 0.0)

    hw = max(0, int(half_width_bins))
    hqi = np.zeros(n_t, dtype=np.float64)
    for t in range(n_t):
        row1 = p1[t]
        tot1 = float(np.sum(row1))
        if not np.isfinite(tot1) or tot1 <= 1e-12:
            hqi[t] = 0.0
            continue
        idx = int(np.argmax(row1))
        lo = max(0, idx - hw)
        hi = min(n_b, idx + hw + 1)
        fund_mass = float(np.sum(row1[lo:hi])) + 0.5 * float(p2[t, idx])
        bg_mass = tot1 + 0.5 * float(np.max(p_half[t])) + 1e-6
        hqi[t] = fund_mass / (fund_mass + bg_mass)
    return np.clip(hqi, 0.0, 1.0)


def causal_temporal_spectral_entropy(
    phi_seq: np.ndarray,
    history_hops: int = 3,
    eps0: float = 1e-3,
) -> np.ndarray:
    """Strictly causal short-lag mixture spectral entropy over `[max(0, t-history_hops), t]`.

    In real breathing sessions, adjacent 30-s windows overlap by 28 s (`93.3%`) and share a
    coherent harmonic ridge, keeping the causal mixture entropy low; in independent WGN windows
    or phase-scrambled surrogates, spectral peaks jump randomly across `[18, 90] bpm`.
    Guaranteed `0.0` future-frame leakage and exact $O(2) \\times \\mathbb{R}_{>0}$ invariance.
    """
    phi = np.asarray(phi_seq, dtype=np.float64)
    if phi.ndim == 2:
        phi = phi[None, :, :]
    n_t, n_b, _ = phi.shape
    if n_t == 0 or n_b <= 1:
        return np.zeros(n_t, dtype=np.float64)

    p1 = np.maximum(np.exp(phi[:, :, 0]) - float(eps0), 0.0)
    row_sums = np.sum(p1, axis=1, keepdims=True)
    pn = np.where(row_sums > 1e-12, p1 / np.maximum(row_sums, 1e-12), 1.0 / float(n_b))

    lag = max(0, int(history_hops))
    out = np.zeros(n_t, dtype=np.float64)
    denom_log = np.log(float(n_b))
    for t in range(n_t):
        t0 = max(0, t - lag)
        m_bar = np.mean(pn[t0 : t + 1], axis=0)
        m_bar = np.maximum(m_bar, 1e-300)
        out[t] = -float(np.sum(m_bar * np.log(m_bar))) / denom_log
    return np.clip(out, 0.0, 1.0)


def extract_zero_tuned_proxies_session(
    phi_seq: np.ndarray,
    post_entropy: np.ndarray,
    post_sd_bpm: np.ndarray,
    grid_sd_unif: float = UNIFORM_GRID_SD_1BPM,
    eps0: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Compute all 7 zero-tuned uncertainty proxies `u_{s,w} in [0, 1]` (`SQI_{s,w} = 1 - u_{s,w}`).

    Parameters
    ----------
    phi_seq:
        `(T, N_B, 6)` or single-window `(N_B, 6)` $O(2) \\times \\mathbb{R}_{>0}$-invariant log-salience tensor.
    post_entropy:
        `(T,)` causal Forward-HMM normalized posterior Shannon entropy `H_t in [0, 1]`.
    post_sd_bpm:
        `(T,)` causal Forward-HMM posterior standard deviation `\\sigma_{post,t}` in bpm.
    """
    phi = np.asarray(phi_seq, dtype=np.float64)
    if phi.ndim == 2:
        phi = phi[None, :, :]
    if phi.ndim != 3 or phi.shape[2] < 6:
        raise ValueError(f"phi_seq must have shape (T, N_B, >=6) or (N_B, >=6); got {phi.shape}")
    n_t = int(phi.shape[0])
    u1_ent = np.clip(np.asarray(post_entropy, dtype=np.float64).ravel(), 0.0, 1.0)
    u2_sd = np.clip(np.asarray(post_sd_bpm, dtype=np.float64).ravel() / float(grid_sd_unif), 0.0, 1.0)
    if u1_ent.size != n_t or u2_sd.size != n_t:
        raise ValueError(
            f"post_entropy ({u1_ent.size}) and post_sd_bpm ({u2_sd.size}) must match phi_seq length ({n_t})"
        )
    u3_sc = spectral_shannon_entropy_from_log_salience(phi[:, :, 0], eps0=eps0)
    u4_sr = spectral_shannon_entropy_from_log_salience(phi[:, :, 5], eps0=eps0)
    hqi = harmonic_quality_index_from_salience(phi, eps0=eps0)
    u5_hqi_def = np.clip(1.0 - hqi, 0.0, 1.0)

    # Strictly causal 4-window (6 s history) temporal mixture spectral entropy tail barrier
    # Separates real breathing (95th pct <= 0.913, including >5 s quiet-apnea pauses) from OOD WGN (>= 0.945)
    h_mix_causal = causal_temporal_spectral_entropy(phi, history_hops=3, eps0=eps0)
    b_ood_causal = np.clip((h_mix_causal - 0.90) / 0.10, 0.0, 1.0)

    # U6a: Zero-tuned additive Bayesian-OOD composite (0.45 * sd_norm + 0.35 * H_post + 0.20 * b_OOD)
    u6a_add = np.clip(0.45 * u2_sd + 0.35 * u1_ent + 0.20 * b_ood_causal, 0.0, 1.0)

    # U6b: Zero-tuned equal-exponent geometric-mean confidence composite
    # SQI_geom = ((1 - sd_norm) * (1 - H_post) * (1 - b_OOD)) ** (1/3)
    sqi_geom = (
        np.maximum(1.0 - u2_sd, 0.0)
        * np.maximum(1.0 - u1_ent, 0.0)
        * np.maximum(1.0 - b_ood_causal, 0.0)
    ) ** (1.0 / 3.0)
    u6b_geom = np.clip(1.0 - sqi_geom, 0.0, 1.0)

    return {
        "U1_Posterior_Entropy": u1_ent,
        "U2_Posterior_SD": u2_sd,
        "U3_Spectral_Entropy_Clean": u3_sc,
        "U4_Spectral_Entropy_Raw": u4_sr,
        "U5_HQI_Harmonic_Deficit": u5_hqi_def,
        "U6a_ZeroTuned_Additive_SQI": u6a_add,
        "U6b_ZeroTuned_Geometric_SQI": u6b_geom,
    }


def build_base5_feature_matrix(proxy_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Stack the 5 foundational uncertainty channels `[U1, U2, U3, U4, U5]` as `(T, 5)`."""
    return np.column_stack([
        proxy_dict["U1_Posterior_Entropy"],
        proxy_dict["U2_Posterior_SD"],
        proxy_dict["U3_Spectral_Entropy_Clean"],
        proxy_dict["U4_Spectral_Entropy_Raw"],
        proxy_dict["U5_HQI_Harmonic_Deficit"],
    ])


def fit_lopo_calibrated_sqi_weights(
    train_X_list: list[np.ndarray],
    train_abs_err_list: list[np.ndarray],
    ridge_lam: float = 100.0,
) -> dict[str, np.ndarray]:
    """Fit monotone non-negative Ridge-NNLS (`Z-NNLS`) SQI calibrator on training patients only.

    Standardizes the 5 foundational uncertainty features `[U1..U5]` using training-fold
    mean and standard deviation, centers training absolute errors `|e_{s,w}|`, and solves:
      `min_{w >= 0} ||X_z w - (y - mean(y))||_2^2 + ridge_lam * ||w||_2^2`
    normalizing `w` onto the probability simplex `\\sum_j w_j = 1`.
    """
    X_tr = np.concatenate(train_X_list, axis=0)
    y_tr = np.concatenate(train_abs_err_list, axis=0)
    ok = np.all(np.isfinite(X_tr), axis=1) & np.isfinite(y_tr)
    X_tr = X_tr[ok]
    y_tr = y_tr[ok]

    n_feat = X_tr.shape[1]
    if X_tr.shape[0] < n_feat + 2:
        return {
            "weights": np.full(n_feat, 1.0 / float(n_feat), dtype=np.float64),
            "mu_X": np.zeros(n_feat, dtype=np.float64),
            "sd_X": np.ones(n_feat, dtype=np.float64),
            "q_lo": np.array(0.0),
            "q_hi": np.array(1.0),
        }

    mu_X = np.mean(X_tr, axis=0)
    sd_X = np.maximum(np.std(X_tr, axis=0, ddof=0), 1e-6)
    Xz = (X_tr - mu_X[None, :]) / sd_X[None, :]
    yc = y_tr - float(np.mean(y_tr))

    sqrt_lam = float(np.sqrt(max(0.0, float(ridge_lam))))
    A_aug = np.vstack([Xz, sqrt_lam * np.eye(n_feat, dtype=np.float64)])
    y_aug = np.r_[yc, np.zeros(n_feat, dtype=np.float64)]
    w_raw, _ = optimize.nnls(A_aug, y_aug)
    w_sum = float(np.sum(w_raw))
    if not np.isfinite(w_sum) or w_sum <= 1e-12:
        w = np.full(n_feat, 1.0 / float(n_feat), dtype=np.float64)
    else:
        w = w_raw / w_sum

    raw_tr = Xz @ w
    q_lo = float(np.percentile(raw_tr, 1.0))
    q_hi = float(np.percentile(raw_tr, 99.0))
    if q_hi <= q_lo + 1e-9:
        q_lo, q_hi = float(np.min(raw_tr)), float(np.max(raw_tr) + 1e-6)

    return {
        "weights": w,
        "mu_X": mu_X,
        "sd_X": sd_X,
        "q_lo": np.array(q_lo, dtype=np.float64),
        "q_hi": np.array(q_hi, dtype=np.float64),
    }


def apply_lopo_calibrated_sqi(
    X_eval: np.ndarray,
    calibrator: dict[str, np.ndarray],
) -> np.ndarray:
    """Apply frozen training-fold `Z-NNLS` calibrator to test-fold or OOD feature matrix `(T, 5)`.

    Note: Because `q_lo < q_hi` is a strictly positive affine scaling, order statistics (and thus
    Risk-Coverage curves, AURC, E-AURC, AUGRC, and OOD AUROC) are invariant to the `[0, 1]`
    clipping bounds, while `1 / (1 + exp(-raw))` or linear scaling preserves strict monotonicity
    even in extreme OOD tails without saturating ties.
    """
    mu_X = np.asarray(calibrator["mu_X"], dtype=np.float64)
    sd_X = np.asarray(calibrator["sd_X"], dtype=np.float64)
    w = np.asarray(calibrator["weights"], dtype=np.float64)
    Xz = (np.asarray(X_eval, dtype=np.float64) - mu_X[None, :]) / sd_X[None, :]
    raw = Xz @ w
    # Strictly monotone logistic link into (0, 1) so extreme OOD values never clip to identical ties
    return 1.0 / (1.0 + np.exp(-np.clip(raw, -40.0, 40.0)))


# ---------------------------------------------------------------------------
# 2. Selective Prediction: Risk-Coverage, AURC, E-AURC & AUGRC (NeurIPS 2024)
# ---------------------------------------------------------------------------

def compute_risk_coverage_curve(
    errors: np.ndarray,
    uncertainty: np.ndarray,
    cov_grid: np.ndarray = COV_GRID_51,
) -> dict[str, Any]:
    """Compute selective risk R(c_cov), generalized risk GR(c_cov), AURC, E-AURC & AUGRC.

    Adheres strictly to `notation_glossary.tex` (`\\covLevel`, `\\selRisk`, `\\aurcMetric`,
    `\\eaurcMetric`) and Traub et al. (NeurIPS 2024) Generalized Risk `GR(c) = (k(c)/N) * R(c)`.
    Uses deterministic stable merge-sort (`kind="mergesort"`) so ties break reproducibly by
    temporal window order.
    """
    e = np.asarray(errors, dtype=np.float64).ravel()
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    ok = np.isfinite(e) & np.isfinite(u)
    e = e[ok]
    u = u[ok]
    n = int(e.size)
    c_grid = np.asarray(cov_grid, dtype=np.float64).ravel()
    if n == 0 or c_grid.size < 2:
        raise ValueError("errors and uncertainty must contain at least 1 valid finite pair")

    ae = np.abs(e)
    se2 = e ** 2
    gt10 = (ae > 10.0).astype(np.float64) * 100.0

    # Sort by uncertainty ascending (stable mergesort)
    ord_u = np.argsort(u, kind="mergesort")
    ae_u = ae[ord_u]
    se2_u = se2[ord_u]
    gt10_u = gt10[ord_u]

    # Sort by true absolute error ascending for exact Oracle baseline
    ord_o = np.argsort(ae, kind="mergesort")
    ae_o = ae[ord_o]
    se2_o = se2[ord_o]

    # Prefix cumulative sums for O(1) risk evaluation at every coverage point
    csum_ae_u = np.cumsum(ae_u)
    csum_se2_u = np.cumsum(se2_u)
    csum_gt10_u = np.cumsum(gt10_u)

    csum_ae_o = np.cumsum(ae_o)
    csum_se2_o = np.cumsum(se2_o)

    k_arr = np.clip(np.round(c_grid * float(n)).astype(int), 1, n)
    eff_cov = k_arr.astype(np.float64) / float(n)
    idx_k = k_arr - 1

    risk_rmse = np.sqrt(csum_se2_u[idx_k] / k_arr.astype(np.float64))
    risk_mae = csum_ae_u[idx_k] / k_arr.astype(np.float64)
    risk_gt10 = csum_gt10_u[idx_k] / k_arr.astype(np.float64)

    oracle_rmse = np.sqrt(csum_se2_o[idx_k] / k_arr.astype(np.float64))
    oracle_mae = csum_ae_o[idx_k] / k_arr.astype(np.float64)

    full_rmse = float(np.sqrt(csum_se2_u[-1] / float(n)))
    full_mae = float(csum_ae_u[-1] / float(n))
    full_gt10 = float(csum_gt10_u[-1] / float(n))

    rand_rmse = np.full_like(c_grid, full_rmse)
    rand_mae = np.full_like(c_grid, full_mae)

    # Generalized Risk GR(c) = (k(c) / N) * R(c) (Traub et al., NeurIPS 2024)
    gen_risk_rmse = eff_cov * risk_rmse
    gen_risk_mae = csum_ae_u[idx_k] / float(n)
    oracle_gen_rmse = eff_cov * oracle_rmse
    oracle_gen_mae = csum_ae_o[idx_k] / float(n)
    rand_gen_rmse = eff_cov * full_rmse
    rand_gen_mae = eff_cov * full_mae

    span = float(c_grid[-1] - c_grid[0])
    aurc_rmse = float(np.trapezoid(risk_rmse, c_grid) / span)
    aurc_oracle_rmse = float(np.trapezoid(oracle_rmse, c_grid) / span)
    aurc_rand_rmse = full_rmse
    e_aurc_rand_rmse = float(aurc_rmse - aurc_rand_rmse)
    e_aurc_oracle_rmse = float(aurc_rmse - aurc_oracle_rmse)
    denom_rmse = aurc_rand_rmse - aurc_oracle_rmse
    eff_sel_rmse = float((aurc_rand_rmse - aurc_rmse) / denom_rmse) if denom_rmse > 1e-12 else 0.0

    aurc_mae = float(np.trapezoid(risk_mae, c_grid) / span)
    aurc_oracle_mae = float(np.trapezoid(oracle_mae, c_grid) / span)
    aurc_rand_mae = full_mae
    e_aurc_rand_mae = float(aurc_mae - aurc_rand_mae)
    e_aurc_oracle_mae = float(aurc_mae - aurc_oracle_mae)
    denom_mae = aurc_rand_mae - aurc_oracle_mae
    eff_sel_mae = float((aurc_rand_mae - aurc_mae) / denom_mae) if denom_mae > 1e-12 else 0.0

    augrc_rmse = float(np.trapezoid(gen_risk_rmse, c_grid) / span)
    augrc_oracle_rmse = float(np.trapezoid(oracle_gen_rmse, c_grid) / span)
    augrc_rand_rmse = float(np.trapezoid(rand_gen_rmse, c_grid) / span)
    e_augrc_rand_rmse = float(augrc_rmse - augrc_rand_rmse)
    e_augrc_oracle_rmse = float(augrc_rmse - augrc_oracle_rmse)

    augrc_mae = float(np.trapezoid(gen_risk_mae, c_grid) / span)
    augrc_oracle_mae = float(np.trapezoid(oracle_gen_mae, c_grid) / span)
    augrc_rand_mae = float(np.trapezoid(rand_gen_mae, c_grid) / span)
    e_augrc_rand_mae = float(augrc_mae - augrc_rand_mae)
    e_augrc_oracle_mae = float(augrc_mae - augrc_oracle_mae)

    def _at_cov(arr_vals: np.ndarray, target_c: float) -> float:
        idx = int(np.argmin(np.abs(c_grid - float(target_c))))
        return float(arr_vals[idx])

    return {
        "n_windows": n,
        "cov_grid": c_grid.tolist(),
        "risk_rmse": risk_rmse.tolist(),
        "risk_mae": risk_mae.tolist(),
        "risk_gt10_pct": risk_gt10.tolist(),
        "oracle_risk_rmse": oracle_rmse.tolist(),
        "oracle_risk_mae": oracle_mae.tolist(),
        "random_risk_rmse": rand_rmse.tolist(),
        "random_risk_mae": rand_mae.tolist(),
        "gen_risk_rmse": gen_risk_rmse.tolist(),
        "gen_risk_mae": gen_risk_mae.tolist(),
        "oracle_gen_risk_rmse": oracle_gen_rmse.tolist(),
        "random_gen_risk_rmse": rand_gen_rmse.tolist(),
        "full_risk_rmse": full_rmse,
        "full_risk_mae": full_mae,
        "full_gt10_pct": full_gt10,
        "risk_rmse_at_90": _at_cov(risk_rmse, 0.90),
        "risk_rmse_at_80": _at_cov(risk_rmse, 0.80),
        "risk_rmse_at_70": _at_cov(risk_rmse, 0.70),
        "risk_rmse_at_50": _at_cov(risk_rmse, 0.50),
        "risk_mae_at_80": _at_cov(risk_mae, 0.80),
        "risk_mae_at_70": _at_cov(risk_mae, 0.70),
        "gt10_pct_at_80": _at_cov(risk_gt10, 0.80),
        "gt10_pct_at_70": _at_cov(risk_gt10, 0.70),
        "aurc_rmse": aurc_rmse,
        "aurc_oracle_rmse": aurc_oracle_rmse,
        "aurc_random_rmse": aurc_rand_rmse,
        "e_aurc_vs_random_rmse": e_aurc_rand_rmse,
        "e_aurc_vs_oracle_rmse": e_aurc_oracle_rmse,
        "selective_efficiency_rmse": eff_sel_rmse,
        "aurc_mae": aurc_mae,
        "aurc_oracle_mae": aurc_oracle_mae,
        "aurc_random_mae": aurc_rand_mae,
        "e_aurc_vs_random_mae": e_aurc_rand_mae,
        "e_aurc_vs_oracle_mae": e_aurc_oracle_mae,
        "selective_efficiency_mae": eff_sel_mae,
        "augrc_rmse": augrc_rmse,
        "augrc_oracle_rmse": augrc_oracle_rmse,
        "augrc_random_rmse": augrc_rand_rmse,
        "e_augrc_vs_random_rmse": e_augrc_rand_rmse,
        "e_augrc_vs_oracle_rmse": e_augrc_oracle_rmse,
        "augrc_mae": augrc_mae,
        "augrc_oracle_mae": augrc_oracle_mae,
        "augrc_random_mae": augrc_rand_mae,
        "e_augrc_vs_random_mae": e_augrc_rand_mae,
        "e_augrc_vs_oracle_mae": e_augrc_oracle_mae,
    }


def _safe_wilcoxon_p(deltas: np.ndarray) -> float:
    arr = np.asarray(deltas, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    nz = arr[np.abs(arr) > 1e-15]
    if nz.size < 2:
        return 1.0
    try:
        p = float(stats.wilcoxon(arr, zero_method="wilcox", method="auto").pvalue)
        return p if np.isfinite(p) else 1.0
    except ValueError:
        return 1.0


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    if xa.size < 3 or float(np.std(xa)) <= 1e-12 or float(np.std(ya)) <= 1e-12:
        return float("nan")
    val = float(stats.spearmanr(xa, ya).statistic)
    return val if np.isfinite(val) else float("nan")


def evaluate_cohort_selective_prediction(
    sessions: list[str],
    preds_by_sess: dict[str, np.ndarray],
    unc_by_proxy_sess: dict[str, dict[str, np.ndarray]],
    ref_valid_map: dict[str, tuple[np.ndarray, np.ndarray]],
    cov_grid: np.ndarray = COV_GRID_51,
    include_curves: bool = True,
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, float]]]]:
    """Evaluate dual Equal-Session Macro-Averaged & Window-Pooled Risk-Coverage across proxies.

    Returns
    -------
    summary_by_proxy:
        Dictionary keyed by proxy name containing both `session_macro` (with 10,000-resample
        patient-clustered bootstrap 95% CIs and paired Wilcoxon + Holm-Bonferroni p-values)
        and `window_pooled` Risk-Coverage metrics.
    per_sess_by_proxy:
        Per-session dictionary for CSV export and clean-room verification.
    """
    valid_sessions = [m for m in sessions if np.any(ref_valid_map[m][1])]
    c_grid = np.asarray(cov_grid, dtype=np.float64).ravel()
    proxy_names = [p for p in PROXY_ORDER if p in unc_by_proxy_sess]

    summary_out: dict[str, Any] = {}
    per_sess_out: dict[str, dict[str, dict[str, float]]] = {}
    raw_p_sess: list[float] = []
    raw_p_pat: list[float] = []

    for pname in proxy_names:
        u_map = unc_by_proxy_sess[pname]
        sess_rc_list: list[dict[str, Any]] = []
        per_sess_p: dict[str, dict[str, float]] = {}
        by_pat_eaurc: dict[int, list[float]] = {}
        by_pat_d80: dict[int, list[float]] = {}
        deltas_eaurc_s: list[float] = []
        deltas_d80_s: list[float] = []
        sp_sess_list: list[float] = []

        err_pool_list: list[np.ndarray] = []
        u_pool_list: list[np.ndarray] = []

        for m in valid_sessions:
            rf, vv = ref_valid_map[m]
            err_m = np.asarray(preds_by_sess[m][vv], dtype=np.float64) - np.asarray(rf[vv], dtype=np.float64)
            u_m = np.asarray(u_map[m][vv], dtype=np.float64)
            rc_m = compute_risk_coverage_curve(err_m, u_m, cov_grid=c_grid)
            sess_rc_list.append(rc_m)

            da_m = float(rc_m["e_aurc_vs_random_rmse"])
            d80_m = float(rc_m["risk_rmse_at_80"] - rc_m["full_risk_rmse"])
            sp_m = _safe_spearman(u_m, np.abs(err_m))
            if np.isfinite(sp_m):
                sp_sess_list.append(sp_m)

            deltas_eaurc_s.append(da_m)
            deltas_d80_s.append(d80_m)
            pid = IO.patient_id_of(m)
            by_pat_eaurc.setdefault(pid, []).append(da_m)
            by_pat_d80.setdefault(pid, []).append(d80_m)

            per_sess_p[m] = {
                "n_windows": int(rc_m["n_windows"]),
                "full_risk_rmse": float(rc_m["full_risk_rmse"]),
                "risk_rmse_at_90": float(rc_m["risk_rmse_at_90"]),
                "risk_rmse_at_80": float(rc_m["risk_rmse_at_80"]),
                "risk_rmse_at_70": float(rc_m["risk_rmse_at_70"]),
                "risk_rmse_at_50": float(rc_m["risk_rmse_at_50"]),
                "aurc_rmse": float(rc_m["aurc_rmse"]),
                "e_aurc_vs_random_rmse": da_m,
                "e_aurc_vs_oracle_rmse": float(rc_m["e_aurc_vs_oracle_rmse"]),
                "augrc_rmse": float(rc_m["augrc_rmse"]),
                "e_augrc_vs_random_rmse": float(rc_m["e_augrc_vs_random_rmse"]),
                "delta_rmse_at_80": d80_m,
                "spearman_u_abserr": float(sp_m) if np.isfinite(sp_m) else 0.0,
            }

            err_pool_list.append(err_m)
            u_pool_list.append(u_m)

        per_sess_out[pname] = per_sess_p

        # Equal-session macro-averaged curves & integrals
        mean_risk_rmse = np.mean([r["risk_rmse"] for r in sess_rc_list], axis=0)
        mean_risk_mae = np.mean([r["risk_mae"] for r in sess_rc_list], axis=0)
        mean_risk_gt10 = np.mean([r["risk_gt10_pct"] for r in sess_rc_list], axis=0)
        mean_oracle_rmse = np.mean([r["oracle_risk_rmse"] for r in sess_rc_list], axis=0)
        mean_rand_rmse = np.mean([r["random_risk_rmse"] for r in sess_rc_list], axis=0)
        mean_gen_rmse = np.mean([r["gen_risk_rmse"] for r in sess_rc_list], axis=0)
        mean_oracle_gen_rmse = np.mean([r["oracle_gen_risk_rmse"] for r in sess_rc_list], axis=0)
        mean_rand_gen_rmse = np.mean([r["random_gen_risk_rmse"] for r in sess_rc_list], axis=0)

        pat_eaurc_map = {p: np.asarray(v, dtype=np.float64) for p, v in by_pat_eaurc.items()}
        pat_d80_map = {p: np.asarray(v, dtype=np.float64) for p, v in by_pat_d80.items()}
        boot_eaurc = fast_patient_clustered_bootstrap(pat_eaurc_map, n_boot=10000, seed=IO.RANDOM_SEED)
        boot_d80 = fast_patient_clustered_bootstrap(pat_d80_map, n_boot=10000, seed=IO.RANDOM_SEED)

        pat_means_eaurc = np.array([float(np.mean(v)) for v in pat_eaurc_map.values()], dtype=np.float64)
        p_w_sess = _safe_wilcoxon_p(np.asarray(deltas_eaurc_s, dtype=np.float64))
        p_w_pat = _safe_wilcoxon_p(pat_means_eaurc)
        raw_p_sess.append(p_w_sess)
        raw_p_pat.append(p_w_pat)

        aurc_sess = float(np.mean([r["aurc_rmse"] for r in sess_rc_list]))
        aurc_ora_sess = float(np.mean([r["aurc_oracle_rmse"] for r in sess_rc_list]))
        aurc_rnd_sess = float(np.mean([r["full_risk_rmse"] for r in sess_rc_list]))
        denom_s = aurc_rnd_sess - aurc_ora_sess

        sess_macro: dict[str, Any] = {
            "n_sessions": len(valid_sessions),
            "n_patients": len(pat_eaurc_map),
            "full_risk_rmse_sess": aurc_rnd_sess,
            "full_risk_mae_sess": float(np.mean([r["full_risk_mae"] for r in sess_rc_list])),
            "full_gt10_sess_pct": float(np.mean([r["full_gt10_pct"] for r in sess_rc_list])),
            "risk_rmse_sess_at_90": float(np.mean([r["risk_rmse_at_90"] for r in sess_rc_list])),
            "risk_rmse_sess_at_80": float(np.mean([r["risk_rmse_at_80"] for r in sess_rc_list])),
            "risk_rmse_sess_at_70": float(np.mean([r["risk_rmse_at_70"] for r in sess_rc_list])),
            "risk_rmse_sess_at_50": float(np.mean([r["risk_rmse_at_50"] for r in sess_rc_list])),
            "gt10_sess_pct_at_80": float(np.mean([r["gt10_pct_at_80"] for r in sess_rc_list])),
            "gt10_sess_pct_at_70": float(np.mean([r["gt10_pct_at_70"] for r in sess_rc_list])),
            "aurc_rmse_sess": aurc_sess,
            "aurc_oracle_rmse_sess": aurc_ora_sess,
            "aurc_random_rmse_sess": aurc_rnd_sess,
            "e_aurc_vs_random_rmse_sess": float(aurc_sess - aurc_rnd_sess),
            "e_aurc_vs_oracle_rmse_sess": float(aurc_sess - aurc_ora_sess),
            "selective_efficiency_rmse_sess": float((aurc_rnd_sess - aurc_sess) / denom_s) if denom_s > 1e-12 else 0.0,
            "augrc_rmse_sess": float(np.mean([r["augrc_rmse"] for r in sess_rc_list])),
            "augrc_oracle_rmse_sess": float(np.mean([r["augrc_oracle_rmse"] for r in sess_rc_list])),
            "augrc_random_rmse_sess": float(np.mean([r["augrc_random_rmse"] for r in sess_rc_list])),
            "e_augrc_vs_random_rmse_sess": float(np.mean([r["e_augrc_vs_random_rmse"] for r in sess_rc_list])),
            "e_augrc_vs_oracle_rmse_sess": float(np.mean([r["e_augrc_vs_oracle_rmse"] for r in sess_rc_list])),
            "e_aurc_ci95_lo": float(boot_eaurc["ci95_lo"]),
            "e_aurc_ci95_hi": float(boot_eaurc["ci95_hi"]),
            "delta_rmse_at_80_sess": float(boot_d80["mean"]),
            "delta_rmse_at_80_ci95_lo": float(boot_d80["ci95_lo"]),
            "delta_rmse_at_80_ci95_hi": float(boot_d80["ci95_hi"]),
            "wins_sess": f"{int(np.sum(np.asarray(deltas_eaurc_s) < 0.0))}/{len(deltas_eaurc_s)}",
            "wins_pat": f"{int(np.sum(pat_means_eaurc < 0.0))}/{len(pat_means_eaurc)}",
            "p_wilcoxon_sess": p_w_sess,
            "p_wilcoxon_patient": p_w_pat,
            "spearman_u_abserr_sess_mean": float(np.mean(sp_sess_list)) if sp_sess_list else float("nan"),
        }
        if include_curves:
            sess_macro["cov_grid"] = c_grid.tolist()
            sess_macro["risk_rmse_curve"] = mean_risk_rmse.tolist()
            sess_macro["risk_mae_curve"] = mean_risk_mae.tolist()
            sess_macro["risk_gt10_pct_curve"] = mean_risk_gt10.tolist()
            sess_macro["oracle_risk_rmse_curve"] = mean_oracle_rmse.tolist()
            sess_macro["random_risk_rmse_curve"] = mean_rand_rmse.tolist()
            sess_macro["gen_risk_rmse_curve"] = mean_gen_rmse.tolist()
            sess_macro["oracle_gen_risk_rmse_curve"] = mean_oracle_gen_rmse.tolist()
            sess_macro["random_gen_risk_rmse_curve"] = mean_rand_gen_rmse.tolist()

        # Window-pooled Risk-Coverage
        all_err = np.concatenate(err_pool_list)
        all_u = np.concatenate(u_pool_list)
        pool_rc = compute_risk_coverage_curve(all_err, all_u, cov_grid=c_grid)
        pool_rc["spearman_u_abserr_pool"] = _safe_spearman(all_u, np.abs(all_err))
        if not include_curves:
            for ck in (
                "cov_grid", "risk_rmse", "risk_mae", "risk_gt10_pct",
                "oracle_risk_rmse", "oracle_risk_mae", "random_risk_rmse", "random_risk_mae",
                "gen_risk_rmse", "gen_risk_mae", "oracle_gen_risk_rmse", "random_gen_risk_rmse",
            ):
                pool_rc.pop(ck, None)

        summary_out[pname] = {
            "meta": dict(PROXY_META.get(pname, {})),
            "session_macro": sess_macro,
            "window_pooled": pool_rc,
        }

    # Holm-Bonferroni adjustment across the uncertainty proxy family
    holm_sess = IO.holm_bonferroni(raw_p_sess) if raw_p_sess else []
    holm_pat = IO.holm_bonferroni(raw_p_pat) if raw_p_pat else []
    for idx, pname in enumerate(proxy_names):
        summary_out[pname]["session_macro"]["p_wilcoxon_sess_holm"] = float(holm_sess[idx])
        summary_out[pname]["session_macro"]["p_wilcoxon_patient_holm"] = float(holm_pat[idx])

    return summary_out, per_sess_out


# ---------------------------------------------------------------------------
# 3. Quiet-Apnea Pause Safety Audit (`1024 Hz Resp.-Trigger`, `> 5.0 s` gaps)
# ---------------------------------------------------------------------------

def evaluate_apnea_pause_safety(
    sessions: list[str],
    unc_by_proxy_sess: dict[str, dict[str, np.ndarray]],
    valid_map: dict[str, np.ndarray],
    pause_map: dict[str, np.ndarray],
    safety_ratio_limit: float = 1.15,
) -> dict[str, Any]:
    """Stratify uncertainty scores and abstention rates on `> 5.0 s` pause vs regular windows.

    Verifies that selective abstention does not disproportionately reject real quiet-apnea /
    respiratory pause windows (`resp_max_gap_s > 5.0 s`) as false low-SNR dropouts.
    """
    proxy_names = [p for p in PROXY_ORDER if p in unc_by_proxy_sess]
    out: dict[str, Any] = {}

    for pname in proxy_names:
        u_map = unc_by_proxy_sess[pname]
        u_pause_list: list[np.ndarray] = []
        u_reg_list: list[np.ndarray] = []
        u_all_list: list[np.ndarray] = []
        is_pause_all: list[np.ndarray] = []

        # Also track within-session top-(1-c) rejection indicator
        sess_rej_at: dict[str, list[np.ndarray]] = {"0.90": [], "0.80": [], "0.70": []}
        sess_pause_flags: list[np.ndarray] = []

        for m in sessions:
            vv = np.asarray(valid_map[m], dtype=bool)
            if not np.any(vv):
                continue
            um = np.asarray(u_map[m][vv], dtype=np.float64)
            pm = np.asarray(pause_map[m][vv], dtype=bool)
            u_all_list.append(um)
            is_pause_all.append(pm)
            if np.any(pm):
                u_pause_list.append(um[pm])
            if np.any(~pm):
                u_reg_list.append(um[~pm])

            # Within-session deterministic mergesort ranking
            n_m = um.size
            ord_m = np.argsort(um, kind="mergesort")
            sess_pause_flags.append(pm)
            for c_str, c_val in (("0.90", 0.90), ("0.80", 0.80), ("0.70", 0.70)):
                k_keep = max(1, min(n_m, int(round(c_val * float(n_m)))))
                rej_mask = np.ones(n_m, dtype=bool)
                rej_mask[ord_m[:k_keep]] = False
                sess_rej_at[c_str].append(rej_mask)

        u_all = np.concatenate(u_all_list)
        p_all = np.concatenate(is_pause_all)
        u_p = np.concatenate(u_pause_list) if u_pause_list else np.empty(0, dtype=np.float64)
        u_r = np.concatenate(u_reg_list) if u_reg_list else np.empty(0, dtype=np.float64)

        # Global pooled ranking (deterministic mergesort)
        n_tot = u_all.size
        ord_pool = np.argsort(u_all, kind="mergesort")
        pool_cov_stats: dict[str, Any] = {}
        sess_cov_stats: dict[str, Any] = {}
        p_sess_cat = np.concatenate(sess_pause_flags)

        for c_str, c_val in (("0.90", 0.90), ("0.80", 0.80), ("0.70", 0.70)):
            k_pool = max(1, min(n_tot, int(round(c_val * float(n_tot)))))
            rej_pool = np.ones(n_tot, dtype=bool)
            rej_pool[ord_pool[:k_pool]] = False
            rp_pool = float(100.0 * np.mean(rej_pool[p_all])) if np.any(p_all) else 0.0
            rr_pool = float(100.0 * np.mean(rej_pool[~p_all])) if np.any(~p_all) else 0.0
            ratio_pool = float(rp_pool / max(rr_pool, 1e-9))
            pool_cov_stats[c_str] = {
                "rej_pause_pct": rp_pool,
                "rej_regular_pct": rr_pool,
                "excess_pause_rej_pp": float(rp_pool - rr_pool),
                "rejection_risk_ratio": ratio_pool,
            }

            rej_s = np.concatenate(sess_rej_at[c_str])
            rp_s = float(100.0 * np.mean(rej_s[p_sess_cat])) if np.any(p_sess_cat) else 0.0
            rr_s = float(100.0 * np.mean(rej_s[~p_sess_cat])) if np.any(~p_sess_cat) else 0.0
            ratio_s = float(rp_s / max(rr_s, 1e-9))
            sess_cov_stats[c_str] = {
                "rej_pause_pct": rp_s,
                "rej_regular_pct": rr_s,
                "excess_pause_rej_pp": float(rp_s - rr_s),
                "rejection_risk_ratio": ratio_s,
            }

        if u_p.size > 0 and u_r.size > 0:
            mwu = stats.mannwhitneyu(u_p, u_r, alternative="two-sided")
            auroc_pause = float(mwu.statistic) / float(u_p.size * u_r.size)
            rank_biserial = 2.0 * auroc_pause - 1.0
            p_mwu = float(mwu.pvalue)
        else:
            auroc_pause = 0.5
            rank_biserial = 0.0
            p_mwu = 1.0

        ratio_80_pool = float(pool_cov_stats["0.80"]["rejection_risk_ratio"])
        ratio_80_sess = float(sess_cov_stats["0.80"]["rejection_risk_ratio"])

        out[pname] = {
            "n_pause_windows": int(u_p.size),
            "n_regular_windows": int(u_r.size),
            "pause_fraction_pct": float(100.0 * u_p.size / max(1, n_tot)),
            "mean_u_pause": float(np.mean(u_p)),
            "sd_u_pause": float(np.std(u_p, ddof=1)),
            "mean_u_regular": float(np.mean(u_r)),
            "sd_u_regular": float(np.std(u_r, ddof=1)),
            "mean_sqi_pause": float(1.0 - np.mean(u_p)),
            "mean_sqi_regular": float(1.0 - np.mean(u_r)),
            "auroc_pause_vs_regular": auroc_pause,
            "rank_biserial_pause_vs_regular": rank_biserial,
            "p_mannwhitney_two_sided": p_mwu,
            "pooled_threshold_rejection": pool_cov_stats,
            "within_session_threshold_rejection": sess_cov_stats,
            "apnea_safe_at_80pct_pooled": bool(ratio_80_pool <= float(safety_ratio_limit)),
            "apnea_safe_at_80pct_within_sess": bool(ratio_80_sess <= float(safety_ratio_limit)),
        }
    return out


# ---------------------------------------------------------------------------
# 4. OOD, WGN, Phase-Scrambled Surrogate & Firmware-Dropout Abstention Audit
# ---------------------------------------------------------------------------

def exact_auroc_and_fpr95(
    in_dist_u: np.ndarray,
    ood_u: np.ndarray,
) -> dict[str, float]:
    """Compute exact Mann-Whitney / rank AUROC and FPR@95%TPR (`1 = OOD`, `0 = In-Dist`).

    Higher `u` (lower `SQI = 1 - u`) indicates greater OOD / abstention suspicion.
    """
    u_in = np.asarray(in_dist_u, dtype=np.float64).ravel()
    u_ood = np.asarray(ood_u, dtype=np.float64).ravel()
    u_in = u_in[np.isfinite(u_in)]
    u_ood = u_ood[np.isfinite(u_ood)]
    n0 = int(u_in.size)
    n1 = int(u_ood.size)
    if n0 == 0 or n1 == 0:
        return {"auroc": float("nan"), "fpr_at_95tpr": float("nan"), "mean_in": float("nan"), "mean_ood": float("nan")}

    all_scores = np.concatenate([u_in, u_ood])
    ranks = stats.rankdata(all_scores, method="average")
    r_ood = float(np.sum(ranks[n0:]))
    u_stat = r_ood - 0.5 * float(n1 * (n1 + 1))
    auroc = float(u_stat / float(n0 * n1))

    # Threshold capturing >= 95% of OOD windows (5th percentile of OOD uncertainty)
    thresh_95 = float(np.percentile(u_ood, 5.0))
    fpr_95 = float(np.mean(u_in >= thresh_95))

    return {
        "n_in_dist": n0,
        "n_ood": n1,
        "auroc": auroc,
        "fpr_at_95tpr": fpr_95,
        "mean_u_in_dist": float(np.mean(u_in)),
        "sd_u_in_dist": float(np.std(u_in, ddof=1)) if n0 > 1 else 0.0,
        "mean_u_ood": float(np.mean(u_ood)),
        "sd_u_ood": float(np.std(u_ood, ddof=1)) if n1 > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# 5. 5-Regime Finite-K Conformal Prediction Audit (C1..C5)
# ---------------------------------------------------------------------------

def finite_sample_conformal_order_index(n_cal: int, alpha: float) -> tuple[int, bool]:
    """Return 1-based order-statistic index `ceil((n_cal + 1) * (1 - alpha))` and vacuity flag.

    By the distribution-free exchangeability lemma (Vovk et al., 2005; Dunn et al., JASA 2023;
    Angelopoulos et al., ICLR 2024), exact coverage `>= 1 - alpha` over `n_cal` exchangeable
    calibration units requires the `k`-th smallest conformity score where
    `k = ceil((n_cal + 1) * (1 - alpha))`. When `k > n_cal` (i.e. `alpha < 1 / (n_cal + 1)`),
    the required quantile is `+inf`, making the prediction interval mathematically vacuous
    `[-inf, +inf]`.
    """
    k_idx = int(np.ceil(float(n_cal + 1) * (1.0 - float(alpha)) - 1e-12))
    k_idx = max(1, k_idx)
    return k_idx, bool(k_idx > int(n_cal))


def empirical_order_statistic(scores: np.ndarray, alpha: float) -> tuple[float, int, bool]:
    """Return exact `(q_hat, order_idx_1based, is_vacuous)` for 1D conformity scores."""
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n_cal = int(s.size)
    k_idx, vac = finite_sample_conformal_order_index(n_cal, alpha)
    if vac or n_cal == 0:
        return float("inf"), k_idx, True
    return float(s[k_idx - 1]), k_idx, False


def evaluate_finite_k_conformal_suite(
    cohort: str,
    preds_by_sess: dict[str, np.ndarray],
    post_sd_by_sess: dict[str, np.ndarray],
    ref_valid_map: dict[str, tuple[np.ndarray, np.ndarray]],
    alphas: tuple[float, ...] = ALPHA_CONFORMAL_GRID,
    min_sd_floor_bpm: float = 1.0,
    seed: int = IO.RANDOM_SEED,
) -> dict[str, Any]:
    """Run the 5-regime Finite-K Conformal Audit (`C1..C5`) across `alpha in {0.05, 0.10, 0.20, 0.30}`.

    Regimes
    -------
    - `C1_Naive_RandomWindowSplit_50_50`:
        Random 50/50 window split across all valid windows (ignores 93.3% window overlap
        and patient clustering).
    - `C2_Naive_LOPO_PooledWindow`:
        Patient-disjoint LOPO split, but computes the conformal quantile over all pooled
        training windows (`N_cal ~ 6,000`) as if windows were exchangeable.
    - `C3_SessionCluster_Hierarchical_LOPO`:
        Patient-disjoint LOPO split where each training session `s in {1..S_cal}` is
        summarized by its `(1 - alpha)` within-session residual quantile `Q_{1-alpha}(s)`,
        and `q_hat` is the `ceil((S_cal + 1)(1 - alpha))` order statistic across sessions.
    - `C4_PatientExchangeable_Hierarchical_LOPO`:
        Exact distribution-free patient-exchangeable hierarchical conformal prediction
        (Dunn et al., JASA 2023): each calibration infant `k in {1..K_cal}` is summarized by
        its within-patient `(1 - alpha)` residual quantile `Q_{1-alpha}(k)`, and `q_hat` is
        the `ceil((K_cal + 1)(1 - alpha))` order statistic across the `K_cal` infants.
    - `C5_LocallyAdaptive_BayesConformal_Hierarchical_LOPO`:
        Locally-adaptive patient-exchangeable hierarchical conformal prediction using Causal
        Forward-HMM posterior scale `sigma_tilde_{s,w} = max(sigma_{post,s,w}, min_sd_floor_bpm)`:
        conformity score `s_{s,w} = |y_{s,w} - yhat_{s,w}| / sigma_tilde_{s,w}`, yielding
        heteroscedastic time-varying intervals `yhat_{s,w} +- q_hat_adapt * sigma_tilde_{s,w}`.
    """
    sessions = [m for m in IO.get_cohort_sessions(cohort, include_zero_valid=False) if np.any(ref_valid_map[m][1])]
    folds = [f for f in IO.iter_lopo_folds(cohort) if any(m in sessions for m in f.test_sessions)]
    n_patients_total = len(folds)
    k_cal_patients = n_patients_total - 1
    alpha_min_patient = 1.0 / float(k_cal_patients + 1)

    # Pre-extract per-session valid residuals |y - yhat| and normalized residuals |y - yhat| / sigma_tilde
    abs_err_sess: dict[str, np.ndarray] = {}
    sd_scale_sess: dict[str, np.ndarray] = {}
    norm_err_sess: dict[str, np.ndarray] = {}
    for m in sessions:
        rf, vv = ref_valid_map[m]
        ae = np.abs(np.asarray(preds_by_sess[m][vv], dtype=np.float64) - np.asarray(rf[vv], dtype=np.float64))
        sd = np.maximum(np.asarray(post_sd_by_sess[m][vv], dtype=np.float64), float(min_sd_floor_bpm))
        abs_err_sess[m] = ae
        sd_scale_sess[m] = sd
        norm_err_sess[m] = ae / sd

    # Prepare deterministic 50/50 random window split for C1
    all_ae = np.concatenate([abs_err_sess[m] for m in sessions])
    all_pat_tags = np.concatenate([np.full(abs_err_sess[m].size, IO.patient_id_of(m), dtype=int) for m in sessions])
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(all_ae.size)
    n_half = all_ae.size // 2
    c1_cal_idx = perm[:n_half]
    c1_te_idx = perm[n_half:]
    c1_cal_scores = all_ae[c1_cal_idx]
    c1_te_scores = all_ae[c1_te_idx]
    c1_te_pats = all_pat_tags[c1_te_idx]

    by_alpha: dict[str, Any] = {}

    for alpha in alphas:
        a_key = f"{alpha:.2f}"
        target_cov = 1.0 - float(alpha)
        k_req_min_patients = int(np.ceil(1.0 / float(alpha) - 1.0 - 1e-12))
        pat_ord_idx, pat_is_vacuous = finite_sample_conformal_order_index(k_cal_patients, alpha)

        # C1: Random 50/50 window split
        q_c1, idx_c1, vac_c1 = empirical_order_statistic(c1_cal_scores, alpha)
        c1_pat_covs = [
            float(np.mean(c1_te_scores[c1_te_pats == p] <= q_c1))
            for p in sorted(set(c1_te_pats.tolist()))
        ]
        c1_summary = {
            "target_coverage": target_cov,
            "n_cal_units": int(c1_cal_scores.size),
            "order_stat_index": idx_c1,
            "is_vacuous": vac_c1,
            "q_hat_bpm": None if vac_c1 else q_c1,
            "mean_interval_width_bpm": None if vac_c1 else float(2.0 * q_c1),
            "empirical_coverage_patient_mean": float(np.mean(c1_pat_covs)),
            "empirical_coverage_patient_sd": float(np.std(c1_pat_covs, ddof=1)),
            "empirical_coverage_patient_min": float(np.min(c1_pat_covs)),
            "empirical_coverage_pooled": float(np.mean(c1_te_scores <= q_c1)),
            "n_patients_undercovered": int(sum(1 for c in c1_pat_covs if c < target_cov)),
            "coverage_gap_patient_mean": float(np.mean(c1_pat_covs) - target_cov),
        }

        # LOPO regimes C2, C3, C4, C5
        folds_c2, folds_c3, folds_c4, folds_c5 = [], [], [], []
        pool_hits_c2, pool_hits_c3, pool_hits_c4, pool_hits_c5 = [], [], [], []
        c5_low_sd_widths, c5_high_sd_widths = [], []
        c5_low_sd_covs, c5_high_sd_covs = [], []

        for fold in folds:
            tr_ms = [m for m in fold.train_sessions if m in sessions]
            te_ms = [m for m in fold.test_sessions if m in sessions]
            tr_pats = sorted({IO.patient_id_of(m) for m in tr_ms})

            te_ae = np.concatenate([abs_err_sess[m] for m in te_ms])
            te_sd = np.concatenate([sd_scale_sess[m] for m in te_ms])
            te_ne = np.concatenate([norm_err_sess[m] for m in te_ms])

            # C2: Naive LOPO Pooled-Window
            tr_ae_pool = np.concatenate([abs_err_sess[m] for m in tr_ms])
            q_c2, idx_c2, vac_c2 = empirical_order_statistic(tr_ae_pool, alpha)
            hits_c2 = te_ae <= q_c2
            pool_hits_c2.append(hits_c2)
            folds_c2.append({
                "patient_id": int(fold.patient_id),
                "n_test_windows": int(te_ae.size),
                "n_cal_units": int(tr_ae_pool.size),
                "order_stat_index": idx_c2,
                "is_vacuous": vac_c2,
                "q_hat_bpm": None if vac_c2 else q_c2,
                "interval_width_bpm": None if vac_c2 else float(2.0 * q_c2),
                "empirical_coverage": float(np.mean(hits_c2)),
            })

            # C3: Session-Cluster Hierarchical LOPO (each session summarized by (1-alpha) quantile)
            sess_qs = np.array([float(np.quantile(abs_err_sess[m], target_cov)) for m in tr_ms], dtype=np.float64)
            q_c3, idx_c3, vac_c3 = empirical_order_statistic(sess_qs, alpha)
            hits_c3 = np.ones(te_ae.size, dtype=bool) if vac_c3 else (te_ae <= q_c3)
            pool_hits_c3.append(hits_c3)
            folds_c3.append({
                "patient_id": int(fold.patient_id),
                "n_test_windows": int(te_ae.size),
                "n_cal_units": int(sess_qs.size),
                "order_stat_index": idx_c3,
                "is_vacuous": vac_c3,
                "q_hat_bpm": None if vac_c3 else q_c3,
                "interval_width_bpm": None if vac_c3 else float(2.0 * q_c3),
                "empirical_coverage": float(np.mean(hits_c3)),
            })

            # C4: Patient-Exchangeable Hierarchical LOPO (Dunn et al., JASA 2023)
            pat_qs_abs = np.array([
                float(np.quantile(np.concatenate([abs_err_sess[m] for m in tr_ms if IO.patient_id_of(m) == p]), target_cov))
                for p in tr_pats
            ], dtype=np.float64)
            q_c4, idx_c4, vac_c4 = empirical_order_statistic(pat_qs_abs, alpha)
            hits_c4 = np.ones(te_ae.size, dtype=bool) if vac_c4 else (te_ae <= q_c4)
            pool_hits_c4.append(hits_c4)
            folds_c4.append({
                "patient_id": int(fold.patient_id),
                "n_test_windows": int(te_ae.size),
                "n_cal_units": int(pat_qs_abs.size),
                "order_stat_index": idx_c4,
                "is_vacuous": vac_c4,
                "q_hat_bpm": None if vac_c4 else q_c4,
                "interval_width_bpm": None if vac_c4 else float(2.0 * q_c4),
                "empirical_coverage": float(np.mean(hits_c4)),
            })

            # C5: Locally-Adaptive Bayesian-Conformal Patient-Exchangeable Hierarchical LOPO
            pat_qs_norm = np.array([
                float(np.quantile(np.concatenate([norm_err_sess[m] for m in tr_ms if IO.patient_id_of(m) == p]), target_cov))
                for p in tr_pats
            ], dtype=np.float64)
            q_c5, idx_c5, vac_c5 = empirical_order_statistic(pat_qs_norm, alpha)
            if vac_c5:
                hits_c5 = np.ones(te_ne.size, dtype=bool)
                w_c5_mean = None
            else:
                radii_c5 = q_c5 * te_sd
                hits_c5 = te_ae <= radii_c5
                w_c5_arr = 2.0 * radii_c5
                w_c5_mean = float(np.mean(w_c5_arr))
                med_sd = float(np.median(te_sd))
                low_m = te_sd <= med_sd
                high_m = te_sd > med_sd
                if np.any(low_m):
                    c5_low_sd_widths.append(float(np.mean(w_c5_arr[low_m])))
                    c5_low_sd_covs.append(float(np.mean(hits_c5[low_m])))
                if np.any(high_m):
                    c5_high_sd_widths.append(float(np.mean(w_c5_arr[high_m])))
                    c5_high_sd_covs.append(float(np.mean(hits_c5[high_m])))
            pool_hits_c5.append(hits_c5)
            folds_c5.append({
                "patient_id": int(fold.patient_id),
                "n_test_windows": int(te_ne.size),
                "n_cal_units": int(pat_qs_norm.size),
                "order_stat_index": idx_c5,
                "is_vacuous": vac_c5,
                "q_hat_norm": None if vac_c5 else q_c5,
                "interval_width_bpm": w_c5_mean,
                "empirical_coverage": float(np.mean(hits_c5)),
            })

        def _agg_lopo_regime(fold_rows: list[dict[str, Any]], pool_hits: list[np.ndarray]) -> dict[str, Any]:
            any_vac = any(bool(r["is_vacuous"]) for r in fold_rows)
            covs_f = np.array([float(r["empirical_coverage"]) for r in fold_rows], dtype=np.float64)
            widths_f = [float(r["interval_width_bpm"]) for r in fold_rows if r["interval_width_bpm"] is not None]
            cal_counts = [int(r["n_cal_units"]) for r in fold_rows]
            return {
                "target_coverage": target_cov,
                "n_folds": len(fold_rows),
                "n_cal_units_per_fold": int(fold_rows[0]["n_cal_units"]),
                "n_cal_units_min": int(min(cal_counts)),
                "n_cal_units_max": int(max(cal_counts)),
                "order_stat_index": int(fold_rows[0]["order_stat_index"]),
                "is_vacuous": any_vac,
                "mean_interval_width_bpm": None if any_vac else float(np.mean(widths_f)),
                "sd_interval_width_bpm": None if (any_vac or len(widths_f) < 2) else float(np.std(widths_f, ddof=1)),
                "empirical_coverage_patient_mean": float(np.mean(covs_f)),
                "empirical_coverage_patient_median": float(np.median(covs_f)),
                "empirical_coverage_patient_sd": float(np.std(covs_f, ddof=1)) if covs_f.size > 1 else 0.0,
                "empirical_coverage_patient_min": float(np.min(covs_f)),
                "empirical_coverage_patient_max": float(np.max(covs_f)),
                "empirical_coverage_pooled": float(np.mean(np.concatenate(pool_hits))),
                "n_patients_undercovered": int(np.sum(covs_f < target_cov)),
                "coverage_gap_patient_mean": float(np.mean(covs_f) - target_cov),
                "valid_distribution_free_guarantee": bool((not any_vac) and (float(np.mean(covs_f)) >= target_cov)),
                "folds": fold_rows,
            }

        c5_agg = _agg_lopo_regime(folds_c5, pool_hits_c5)
        if not c5_agg["is_vacuous"] and c5_low_sd_widths and c5_high_sd_widths:
            c5_agg["low_uncertainty_half_width_bpm"] = float(np.mean(c5_low_sd_widths))
            c5_agg["high_uncertainty_half_width_bpm"] = float(np.mean(c5_high_sd_widths))
            c5_agg["low_uncertainty_half_coverage"] = float(np.mean(c5_low_sd_covs))
            c5_agg["high_uncertainty_half_coverage"] = float(np.mean(c5_high_sd_covs))

        by_alpha[a_key] = {
            "alpha": float(alpha),
            "target_coverage": target_cov,
            "k_cal_patients": k_cal_patients,
            "alpha_min_patient_exchangeable": alpha_min_patient,
            "patient_order_stat_index": pat_ord_idx,
            "patient_exchangeable_is_vacuous": pat_is_vacuous,
            "min_cal_patients_required_for_finite_interval": k_req_min_patients,
            "regimes": {
                "C1_Naive_RandomWindowSplit_50_50": c1_summary,
                "C2_Naive_LOPO_PooledWindow": _agg_lopo_regime(folds_c2, pool_hits_c2),
                "C3_SessionCluster_Hierarchical_LOPO": _agg_lopo_regime(folds_c3, pool_hits_c3),
                "C4_PatientExchangeable_Hierarchical_LOPO": _agg_lopo_regime(folds_c4, pool_hits_c4),
                "C5_LocallyAdaptive_BayesConformal_Hierarchical_LOPO": c5_agg,
            },
        }

    return {
        "cohort": cohort,
        "n_sessions": len(sessions),
        "n_patients_total": n_patients_total,
        "k_cal_patients_per_fold": k_cal_patients,
        "alpha_min_patient_exchangeable": alpha_min_patient,
        "min_cal_patients_for_95pct": 19,
        "by_alpha": by_alpha,
    }
