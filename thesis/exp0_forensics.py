"""EXP-0: Pre-registered observational 31-session failure forensics & EDF validation.

Reads ONLY from locked v1 `.npz` caches (verified SHA-256) and raw CSV/EDF inputs;
performs zero estimator tuning. Quantifies lag-pull effects vs physical error modes
(I/Q imbalance/collinearity, UCD/rail/DAC telemetry, cardiac/harmonic bleed-in, and
twin co-bedding) across `cached_nmf_lag`, `branch_neutral_raw_nls_lag`, and
`pure_metadata_clock_lag`, then exports CSV/JSON/MD deliverables and PDF+PNG figures.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

from paper_workspace.src import config as C
from paper_workspace.src import evaluation as EV
from thesis import edf_extractor as EDFX
from thesis import io_utils as IO

COHORT_ORDER = ("pool30_sync", "all_paired", "retained23", "PAPER_18")
OKABE_ITO = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#333333",
}


def _configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _iq_covariance_stats(bI: np.ndarray, bQ: np.ndarray) -> tuple[float, float, float]:
    """Return (sd_ratio >= 1, |corr(I, Q)|, 2x2 covariance condition number)."""
    i_c = bI - float(np.mean(bI))
    q_c = bQ - float(np.mean(bQ))
    sd_i = float(np.std(i_c))
    sd_q = float(np.std(q_c))
    if sd_i <= 1e-12 or sd_q <= 1e-12:
        return float("inf"), 1.0, float("inf")
    sd_ratio = float(max(sd_i / sd_q, sd_q / sd_i))
    corr_val = float(abs(np.corrcoef(i_c, q_c)[0, 1]))
    cov = np.cov(np.vstack([i_c, q_c]))
    eigvals = np.linalg.eigvalsh(cov)
    cond = float(max(eigvals) / max(1e-12, min(eigvals)))
    return sd_ratio, corr_val, cond


def _compute_radar_window_telemetry(df_radar: pd.DataFrame, n_win: int) -> dict[str, np.ndarray | dict]:
    """Compute per-window & whole-session radar CSV telemetry (strictly observational)."""
    bI = df_radar["I value"].to_numpy(dtype=float)
    bQ = df_radar["Q value"].to_numpy(dtype=float)
    dac_i = df_radar["DAC-I-Offset"].to_numpy(dtype=int)
    dac_q = df_radar["DAC-Q-Offset"].to_numpy(dtype=int)
    ucd_nonchild = (df_radar["UCD Status"].astype(str) != "ChildPresent").to_numpy(dtype=bool)
    clipped = (bI <= -1024.0) | (bI >= 1023.0) | (bQ <= -1024.0) | (bQ >= 1023.0)

    # Firmware error flag columns (exclude numeric Error 1..4 except non-zero and categorical flags)
    hw_err = np.zeros(len(df_radar), dtype=bool)
    if "Lin error" in df_radar.columns:
        hw_err |= df_radar["Lin error"].astype(str).str.lower() == "true"

    ddi = np.abs(np.diff(dac_i, prepend=dac_i[0]))
    ddq = np.abs(np.diff(dac_q, prepend=dac_q[0]))
    dac_big = (ddi > 1) | (ddq > 1)
    dac_dither = (ddi == 1) | (ddq == 1)

    w_samp = C.WIN_S * C.FS_RADAR
    step = C.UPDATE_S * C.FS_RADAR

    sd_ratio_w = np.full(n_win, np.nan)
    abs_corr_w = np.full(n_win, np.nan)
    cond_w = np.full(n_win, np.nan)
    clip_frac_w = np.full(n_win, np.nan)
    ucd_nonchild_w = np.full(n_win, np.nan)
    dac_jump_w = np.zeros(n_win, dtype=bool)

    for w in range(n_win):
        s0 = w * step
        s1 = s0 + w_samp
        sr, ac, cnd = _iq_covariance_stats(bI[s0:s1], bQ[s0:s1])
        sd_ratio_w[w] = sr
        abs_corr_w[w] = ac
        cond_w[w] = cnd
        clip_frac_w[w] = float(np.mean(clipped[s0:s1]))
        ucd_nonchild_w[w] = float(np.mean(ucd_nonchild[s0:s1]))
        dac_jump_w[w] = bool(np.any(dac_big[s0:s1]))

    # Session summaries computed excluding initial 256-sample boot transient where appropriate
    post_boot = slice(min(260, len(bI)), len(bI))
    sr_sess, ac_sess, cond_sess = _iq_covariance_stats(bI[post_boot], bQ[post_boot])
    nc_idx = np.flatnonzero(ucd_nonchild)
    dur_min = max(1e-6, len(bI) / (60.0 * C.FS_RADAR))

    sess_summary = {
        "iq_sd_ratio_session": sr_sess,
        "iq_abs_corr_session": ac_sess,
        "iq_cond_session": cond_sess,
        "radar_clip_frac_session": float(np.mean(clipped[post_boot])),
        "ucd_nonchild_frac_session": float(np.mean(ucd_nonchild)),
        "ucd_mid_session_dropout": bool(len(nc_idx) > 0 and nc_idx.max() > 260),
        "dac_large_jumps_session": int(np.sum(dac_big)),
        "dac_dither_rate_per_min": float(np.sum(dac_dither) / dur_min),
        "hw_error_frac": float(np.mean(hw_err)),
    }
    return {
        "sd_ratio_w": sd_ratio_w,
        "abs_corr_w": abs_corr_w,
        "cond_w": cond_w,
        "clip_frac_w": clip_frac_w,
        "ucd_nonchild_w": ucd_nonchild_w,
        "dac_jump_w": dac_jump_w,
        "session_summary": sess_summary,
    }


def _window_scores(est: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Compute window-level error metrics using locked strict `< delta` Acc thresholds."""
    e = est - ref
    abs_e = np.abs(e)
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(abs_e)),
        "bias": float(np.mean(e)),
        "acc3": float(100.0 * np.mean(abs_e < 3.0)),
        "acc6": float(100.0 * np.mean(abs_e < 6.0)),
        "acc10": float(100.0 * np.mean(abs_e < 10.0)),
        "gross_gt10_pct": float(100.0 * np.mean(abs_e > 10.0)),
        "r": float(np.corrcoef(est, ref)[0, 1]) if (np.std(est) > 0 and np.std(ref) > 0) else float("nan"),
        "oracle_sd": float(np.std(ref)),
    }


def _classify_dominant_mode(row: dict) -> str:
    """Deterministic observational tag identifying primary empirical failure driver."""
    if row["is_zero_valid"]:
        return "zero_valid_saturation"
    if row["is_zero_overlap"]:
        return "zero_overlap"
    if abs(row["delta_lag_pull_rmse"]) >= 2.0:
        return "lag_pull_nmf"
    if row["iq_sd_ratio_session"] >= 3.5 or row["iq_abs_corr_session"] >= 0.78:
        return "iq_collinearity_skew"
    if row["ucd_mid_session_dropout"] and row["rmse_hpss_neutral"] >= 10.0:
        return "mid_session_ucd_dropout"
    if row["rmse_hpss_neutral"] <= 6.5:
        return "clean_baseline"
    if row["cobedding_twins"]:
        return "twin_cobedding"
    if row["position"] == "side":
        return "side_posture_low_snr"
    if row["pct_win_hr_collision_band"] >= 10.0 or row["bias_hpss_neutral"] >= 2.5:
        return "cardiac_or_positive_bias"
    if row["oracle_sd_neutral"] >= 6.0:
        return "high_ref_volatility"
    return "mixed_spectral_tracking"


def _aggregate_cohort(
    session_records: list[dict],
    window_frames: dict[str, pd.DataFrame],
    cohort_name: str,
    regime_key: str,
) -> dict:
    """Compute equal-session and pooled-window cohort summaries for a cohort x lag regime."""
    ids = IO.get_cohort_sessions(cohort_name, include_zero_valid=False)
    recs = [r for r in session_records if r["measurement"] in ids]
    est_h_list, est_n_list, est_nls_list, ref_list = [], [], [], []
    per_h_rmse, per_n_rmse, per_nls_rmse = [], [], []
    per_h_mae, per_h_acc3, per_h_acc6, per_h_acc10, per_h_gt10 = [], [], [], [], []
    per_h_r, per_n_r, per_sd = [], [], []

    for rec in recs:
        df_w = window_frames[(rec["measurement"], regime_key)]
        ok = df_w["valid"].to_numpy(bool)
        if not np.any(ok):
            continue
        eh = df_w.loc[ok, "est_hpss"].to_numpy(float)
        en = df_w.loc[ok, "est_nmf"].to_numpy(float)
        enls = df_w.loc[ok, "est_nls"].to_numpy(float)
        rf = df_w.loc[ok, "ref"].to_numpy(float)

        est_h_list.append(eh)
        est_n_list.append(en)
        est_nls_list.append(enls)
        ref_list.append(rf)

        sh = _window_scores(eh, rf)
        sn = _window_scores(en, rf)
        snls = _window_scores(enls, rf)
        per_h_rmse.append(sh["rmse"])
        per_n_rmse.append(sn["rmse"])
        per_nls_rmse.append(snls["rmse"])
        per_h_mae.append(sh["mae"])
        per_h_acc3.append(sh["acc3"])
        per_h_acc6.append(sh["acc6"])
        per_h_acc10.append(sh["acc10"])
        per_h_gt10.append(sh["gross_gt10_pct"])
        if np.isfinite(sh["r"]):
            per_h_r.append(sh["r"])
        if np.isfinite(sn["r"]):
            per_n_r.append(sn["r"])
        per_sd.append(sh["oracle_sd"])

    # Constant baselines matching `final_review.py` exact definitions
    const_loso_list = []
    const_global_list = []
    pooled_ref = np.concatenate(ref_list)
    gmean = float(pooled_ref.mean())
    for i, rf in enumerate(ref_list):
        const_global_list.append(float(np.sqrt(np.mean((rf - gmean) ** 2))))
        others = np.concatenate([ref_list[j] for j in range(len(ref_list)) if j != i])
        med_other = float(np.median(others))
        const_loso_list.append(float(np.sqrt(np.mean((rf - med_other) ** 2))))

    all_eh = np.concatenate(est_h_list)
    all_en = np.concatenate(est_n_list)
    rmc = IO.rmcorr_ancova(est_h_list, ref_list)
    ba_clust = IO.clustered_bland_altman(est_h_list, ref_list)
    ba_pool = EV.bland_altman(all_eh, pooled_ref)

    return {
        "n_sessions": len(ref_list),
        "n_windows": int(len(pooled_ref)),
        "rmse_sess_hpss": float(np.mean(per_h_rmse)),
        "rmse_sess_nmf": float(np.mean(per_n_rmse)),
        "rmse_sess_nls": float(np.mean(per_nls_rmse)),
        "rmse_pool_hpss": float(np.sqrt(np.mean((all_eh - pooled_ref) ** 2))),
        "rmse_pool_nmf": float(np.sqrt(np.mean((all_en - pooled_ref) ** 2))),
        "rmse_session_median_hpss": float(np.median(per_h_rmse)),
        "mae_sess_hpss": float(np.mean(per_h_mae)),
        "mae_session_median_hpss": float(np.median(per_h_mae)),
        "acc3_sess_hpss": float(np.mean(per_h_acc3)),
        "acc6_sess_hpss": float(np.mean(per_h_acc6)),
        "acc10_sess_hpss": float(np.mean(per_h_acc10)),
        "gross_gt10_pool_hpss_pct": float(100.0 * np.mean(np.abs(all_eh - pooled_ref) > 10.0)),
        "gross_gt10_sess_hpss_pct": float(np.mean(per_h_gt10)),
        "r_within_sess_mean_hpss": float(np.mean(per_h_r)),
        "r_within_sess_mean_nmf": float(np.mean(per_n_r)),
        "sessions_r_hpss_lt_0_3": int(np.sum(np.array(per_h_r) < 0.3)),
        "r_within_rmcorr_hpss": rmc["r_within"],
        "r_within_rmcorr_df": rmc["df"],
        "r_within_rmcorr_p": rmc["p_value"],
        "const_loso": float(np.mean(const_loso_list)),
        "const_global_oracle": float(np.mean(const_global_list)),
        "session_mean_oracle": float(np.mean(per_sd)),
        "ba_pooled_hpss": {
            "bias": float(np.mean(all_eh - pooled_ref)),
            "loa_lo": float(np.mean(all_eh - pooled_ref) - 1.96 * np.std(all_eh - pooled_ref, ddof=1)),
            "loa_up": float(np.mean(all_eh - pooled_ref) + 1.96 * np.std(all_eh - pooled_ref, ddof=1)),
        },
        "ba_clustered_hpss": ba_clust,
    }


def _attribution_analysis(df_sess: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlations across the 28 primary `pool30_sync` sessions."""
    sub = df_sess[df_sess["in_pool30_sync"]].copy()
    covariates = [
        ("iq_sd_ratio_session", "I/Q SD ratio ($\\rho_{IQ}$)"),
        ("iq_abs_corr_session", "I/Q collinearity ($|r_{IQ}|$)"),
        ("radar_clip_frac_session", "Radar 11-bit rail saturation frac"),
        ("ucd_nonchild_frac_session", "UCD != ChildPresent fraction"),
        ("dac_large_jumps_session", "DAC large offset jumps (>1 LSB)"),
        ("dac_dither_rate_per_min", "DAC dither steps / min"),
        ("pct_win_near_hr_10bpm", "Windows within 10 bpm of EDF HR"),
        ("pct_win_near_half_hr_10bpm", "Windows within 10 bpm of HR/2"),
        ("pct_win_hr_collision_band", "Gross error near HR or HR/2"),
        ("pct_win_br_double_octave", "Harmonic double-octave ratio [1.8,2.2]"),
        ("ref_sat_frac_256hz", "EDF Resp saturation fraction"),
        ("oracle_sd_neutral", "Reference BR SD within session"),
        ("ref_slope_abs_mean_bpm_min", "Reference |dBR/dt| volatility"),
        ("minmove_frac", "NMF minimal-movement window frac"),
        ("lag_pull_abs_s", "Baseline NMF vs neutral lag shift (s)"),
    ]
    rows = []
    for col, desc in covariates:
        x = sub[col].to_numpy(float)
        ok = np.isfinite(x)
        xv = x[ok]
        if int(ok.sum()) < 3 or float(np.std(xv)) == 0.0:
            rho_rmse, p_rmse = 0.0, 1.0
            rho_exc, p_exc = 0.0, 1.0
            rho_r, p_r = 0.0, 1.0
        else:
            rho_rmse, p_rmse = stats.spearmanr(xv, sub.loc[ok, "rmse_hpss_neutral"])
            rho_exc, p_exc = stats.spearmanr(xv, sub.loc[ok, "excess_rmse_hpss_neutral"])
            rho_r, p_r = stats.spearmanr(xv, sub.loc[ok, "r_hpss_neutral"])
        rows.append({
            "covariate": col,
            "description": desc,
            "n_sessions": int(ok.sum()),
            "spearman_rho_rmse_neutral": float(rho_rmse),
            "p_rmse_neutral": float(p_rmse),
            "spearman_rho_excess_rmse": float(rho_exc),
            "p_excess_rmse": float(p_exc),
            "spearman_rho_within_r": float(rho_r),
            "p_within_r": float(p_r),
        })
    df_attr = pd.DataFrame(rows).sort_values("spearman_rho_excess_rmse", key=lambda s: s.abs(), ascending=False)
    return df_attr.reset_index(drop=True)


def _save_fig_pair(fig: plt.Figure, stem: str) -> list[str]:
    IO.BUNDLE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = IO.BUNDLE_FIG_DIR / f"{stem}.pdf"
    png_path = IO.BUNDLE_FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf_path.relative_to(IO.ROOT)), str(png_path.relative_to(IO.ROOT))]


def _plot_session_decomposition(df_sess: pd.DataFrame, summary: dict) -> list[str]:
    sub = df_sess[df_sess["in_all_paired_eval"]].sort_values("rmse_hpss_neutral", ascending=False).reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.9, 4.2), gridspec_kw={"width_ratios": [1.85, 1.0]}
    )

    y = np.arange(len(sub))
    ax1.barh(y, sub["rmse_hpss_neutral"], height=0.72, color=OKABE_ITO["blue"], alpha=0.88,
             label=r"HPSS + NLS ($\hat{\tau}_s$ neutral)")
    ax1.scatter(sub["rmse_hpss_cached"], y, color=OKABE_ITO["vermilion"], marker="d", s=22, zorder=4,
                label=r"HPSS + NLS ($\hat{\tau}_s$ cached v1)")
    ax1.scatter(sub["oracle_sd_neutral"], y, color=OKABE_ITO["green"], marker="|", s=45, linewidths=1.5, zorder=5,
                label=r"$\mathrm{SD}(f^{\mathrm{ref}})$ floor")
    c_loso = summary["regimes"]["branch_neutral_raw_nls_lag"]["pool30_sync"]["const_loso"]
    ax1.axvline(c_loso, color=OKABE_ITO["gray"], linestyle="--", linewidth=1.1,
                label=rf"$\mathrm{{const}}_{{\mathrm{{LOSO}}}}$ ({c_loso:.2f} bpm)")

    # Annotate top outlier / lag-pull sessions directly without crowding
    mode_tag = {"lag_pull_nmf": " [lag]", "iq_collinearity_skew": " [I/Q]", "mid_session_ucd_dropout": " [UCD]"}
    ylabels = [f"{r['measurement']}{mode_tag.get(r['dominant_failure_mode'], '')}" for _, r in sub.iterrows()]
    ax1.set_yticks(y)
    ax1.set_yticklabels(ylabels, fontsize=6.8)
    ax1.invert_yaxis()
    ax1.set_xlim(0.0, 22.0)
    ax1.set_xlabel(r"Within-Session Error $\mathrm{RMSE}_s$ (bpm)")
    ax1.set_title("(a) 31-Session Baseline Decomposition")
    ax1.legend(loc="center right", bbox_to_anchor=(0.99, 0.43), frameon=True, framealpha=0.92, fontsize=6.6)
    ax1.grid(axis="x", alpha=0.25)

    # Panel B: Cohort x Lag Regime summary
    x = np.arange(len(COHORT_ORDER))
    w = 0.26
    r_cached = [summary["regimes"]["cached_nmf_lag"][c]["rmse_sess_hpss"] for c in COHORT_ORDER]
    r_neutral = [summary["regimes"]["branch_neutral_raw_nls_lag"][c]["rmse_sess_hpss"] for c in COHORT_ORDER]
    r_meta = [summary["regimes"]["pure_metadata_clock_lag"][c]["rmse_sess_hpss"] for c in COHORT_ORDER]
    r_loso = [summary["regimes"]["branch_neutral_raw_nls_lag"][c]["const_loso"] for c in COHORT_ORDER]
    r_orac = [summary["regimes"]["branch_neutral_raw_nls_lag"][c]["session_mean_oracle"] for c in COHORT_ORDER]

    ax2.bar(x - w, r_cached, width=w, color=OKABE_ITO["vermilion"], label="Cached v1 lag")
    ax2.bar(x, r_neutral, width=w, color=OKABE_ITO["blue"], label="Neutral lag")
    ax2.bar(x + w, r_meta, width=w, color=OKABE_ITO["sky"], label="Metadata lag")
    ax2.plot(x, r_loso, "k--o", markersize=4, linewidth=1.1, label=r"$\mathrm{const}_{\mathrm{LOSO}}$")
    ax2.plot(x, r_orac, color=OKABE_ITO["green"], linestyle=":", marker="^", markersize=4, linewidth=1.2,
             label=r"$\overline{\mathrm{SD}}_{\mathrm{ref}}$ floor")

    ax2.set_xticks(x)
    ax2.set_xticklabels(["pool30\n(S=28)", "all\n(S=31)", "ret23\n(S=23)", "P18\n(S=18)"], fontsize=7.2)
    ax2.set_ylim(0.0, 14.5)
    ax2.set_ylabel(r"Equal-Session $\mathrm{RMSE}_{\mathrm{sess}}$ (bpm)")
    ax2.set_title("(b) Multiverse Cohort Sensitivity")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=6.5)
    ax2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _save_fig_pair(fig, "exp0_session_error_decomposition")


def _plot_physics_drivers(
    df_sess: pd.DataFrame,
    window_frames: dict[tuple[str, str], pd.DataFrame],
) -> list[str]:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.9, 2.95))

    # Panel A: I/Q collinearity vs session RMSE (pool30_sync)
    sub = df_sess[df_sess["in_all_paired_eval"]].copy()
    norm_col = sub["iq_abs_corr_session"].clip(0, 1)
    sc = ax1.scatter(
        sub["iq_sd_ratio_session"],
        sub["rmse_hpss_neutral"],
        c=norm_col,
        cmap="viridis",
        s=32,
        edgecolor="#222222",
        linewidth=0.5,
    )
    for mid in ("11.1", "9.2", "9.3", "12.1", "3.3"):
        row = sub[sub["measurement"] == mid]
        if not row.empty:
            rx = float(row["iq_sd_ratio_session"].iloc[0])
            ry = float(row["rmse_hpss_neutral"].iloc[0])
            offset = (-22, 4) if mid == "11.1" else (4, 3)
            ax1.annotate(mid, (rx, ry), textcoords="offset points", xytext=offset, fontsize=6.8)
    ax1.set_xscale("log")
    ax1.set_xticks([1, 2, 4, 6])
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax1.set_xlabel(r"I/Q Imbalance Ratio $\rho_{IQ}$ (log)")
    ax1.set_ylabel(r"$\mathrm{RMSE}_s$ (bpm, neutral)")
    ax1.set_title(r"(a) Quadrature Skew ($\rho_{IQ}, |r_{IQ}|$)")
    cb = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label(r"$|r_{IQ}|$", fontsize=7)
    ax1.grid(alpha=0.25)

    # Panel B: Cardiac proximity histogram on accurate vs gross-error windows
    w_list = [window_frames[(m, "branch_neutral_raw_nls_lag")] for m in IO.get_cohort_sessions("pool30_sync")]
    w_all = pd.concat(w_list, ignore_index=True)
    w_all = w_all[w_all["valid"]].copy()
    abs_err = np.abs(w_all["est_hpss"] - w_all["ref"])
    hr_prox = np.minimum(
        np.abs(w_all["est_hpss"] - w_all["hr_rtrig_bpm"]),
        np.abs(w_all["est_hpss"] - 0.5 * w_all["hr_rtrig_bpm"]),
    )
    ok_hp = np.isfinite(hr_prox)
    gross = ok_hp & (abs_err > 10.0)
    good = ok_hp & (abs_err < 3.0)
    bins = np.linspace(0, 60, 16)
    ax2.hist(hr_prox[good], bins=bins, density=True, color=OKABE_ITO["green"], alpha=0.55, label=r"$|e_{s,w}| < 3$ bpm")
    ax2.hist(hr_prox[gross], bins=bins, density=True, histtype="step", linewidth=1.6, color=OKABE_ITO["vermilion"], label=r"$|e_{s,w}| > 10$ bpm")
    ax2.axvline(10.0, color=OKABE_ITO["gray"], linestyle="--", linewidth=0.9)
    ax2.set_xlabel(r"$\min(|\hat{f}_b - \mathrm{HR}|, |\hat{f}_b - \mathrm{HR}/2|)$ (bpm)")
    ax2.set_ylabel("Density")
    ax2.set_title("(b) Cardiac Bleed-In Proximity")
    ax2.legend(loc="upper right", fontsize=6.5, frameon=True)
    ax2.grid(alpha=0.25)

    # Panel C: Case diagnostic on worst outlier 9.2 (keep NaN gaps so invalid intervals do not bridge)
    w92 = window_frames[("9.2", "branch_neutral_raw_nls_lag")].sort_values("t_center_s").copy()
    valid_mask = w92["valid"].to_numpy(bool)
    v92 = w92[valid_mask]
    t_all = w92["t_center_s"].to_numpy(float) / 60.0
    ref_trace = np.where(valid_mask, w92["ref"].to_numpy(float), np.nan)
    est_trace = np.where(valid_mask, w92["est_hpss"].to_numpy(float), np.nan)
    ax3.plot(t_all, ref_trace, color=OKABE_ITO["green"], linewidth=1.1, label=r"$f^{\mathrm{ref}}$")
    ax3.plot(t_all, est_trace, color=OKABE_ITO["blue"], linewidth=0.9, alpha=0.85, label=r"$\hat{f}_b$ (HPSS)")
    if np.any(np.isfinite(w92["hr_1hz_30s"])):
        hr_half_trace = np.where(valid_mask, 0.5 * w92["hr_1hz_30s"].to_numpy(float), np.nan)
        ax3.plot(t_all, hr_half_trace, color=OKABE_ITO["orange"], linestyle="--", linewidth=0.9, label=r"$\mathrm{HR}/2$")
    ucd_bad = v92["ucd_nonchild_w"] > 0
    if np.any(ucd_bad):
        ax3.scatter(v92.loc[ucd_bad, "t_center_s"] / 60.0, v92.loc[ucd_bad, "est_hpss"], color=OKABE_ITO["vermilion"], s=14, zorder=5, label=r"$\mathrm{UCD} \neq \mathrm{Child}$")
    rmse92 = float(np.sqrt(np.mean((v92["est_hpss"] - v92["ref"]) ** 2)))
    ax3.set_xlabel("Recording Time (min)")
    ax3.set_ylabel("Rate (bpm)")
    ax3.set_title(rf"(c) Outlier 9.2 Trace ($\mathrm{{RMSE}}={rmse92:.2f}$ bpm)")
    ax3.legend(loc="upper right", fontsize=6.2, frameon=True)
    ax3.grid(alpha=0.25)

    fig.tight_layout(w_pad=1.2)
    return _save_fig_pair(fig, "exp0_physics_error_drivers")


def _plot_edf256_validation(window_frames: dict[tuple[str, str], pd.DataFrame]) -> list[str]:
    w_list = [window_frames[(m, "branch_neutral_raw_nls_lag")] for m in IO.get_cohort_sessions("pool30_sync")]
    df_w = pd.concat(w_list, ignore_index=True)
    df_w = df_w[df_w["valid"]].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 3.2))

    # Panel A: 1024 Hz Resp.-Trigger vs 1 Hz IntelliVue RespRate Bland-Altman
    m_br = np.isfinite(df_w["br_trig_bpm"]) & np.isfinite(df_w["ref"])
    x_br = 0.5 * (df_w.loc[m_br, "br_trig_bpm"] + df_w.loc[m_br, "ref"])
    d_br = df_w.loc[m_br, "br_trig_bpm"] - df_w.loc[m_br, "ref"]
    rng = np.random.default_rng(IO.RANDOM_SEED)
    idx_sub = rng.choice(len(x_br), size=min(2500, len(x_br)), replace=False)
    ax1.scatter(x_br.iloc[idx_sub], d_br.iloc[idx_sub], s=4, alpha=0.25, color=OKABE_ITO["blue"], rasterized=True)
    bias_br = float(d_br.mean())
    sd_br = float(d_br.std(ddof=1))
    ax1.axhline(bias_br, color=OKABE_ITO["vermilion"], linewidth=1.2, label=rf"$\hat{{\mu}}_d = {bias_br:+.2f}$ bpm")
    ax1.axhline(bias_br + 1.96 * sd_br, color=OKABE_ITO["gray"], linestyle="--", linewidth=0.9,
                label=rf"LoA [{bias_br - 1.96 * sd_br:+.1f}, {bias_br + 1.96 * sd_br:+.1f}]")
    ax1.axhline(bias_br - 1.96 * sd_br, color=OKABE_ITO["gray"], linestyle="--", linewidth=0.9)
    ax1.set_ylim(-25, 31)
    ax1.set_xlabel(r"Mean of $\mathrm{BR}_{\mathrm{trig}}$ and $f^{\mathrm{ref}}$ (bpm)")
    ax1.set_ylabel(r"$\mathrm{BR}_{\mathrm{trig}} - f^{\mathrm{ref}}$ (bpm)")
    ax1.set_title(rf"(a) 1024 Hz Resp-Trigger vs 1 Hz ($|\mathcal{{W}}|={int(m_br.sum())}$)")
    ax1.legend(loc="upper right", fontsize=6.6, frameon=True)
    ax1.grid(alpha=0.25)

    # Panel B: 1024 Hz R-Trigger HR vs 1 Hz HeartRate stratified by SignalIQ
    m_hr = np.isfinite(df_w["hr_rtrig_bpm"]) & np.isfinite(df_w["hr_1hz_30s"]) & (df_w["hr_1hz_30s"] > 50)
    hi_iq = m_hr & (df_w["signaliq_mean"] >= 50)
    lo_iq = m_hr & (df_w["signaliq_mean"] < 50)
    d_hr = df_w.loc[hi_iq, "hr_rtrig_bpm"] - df_w.loc[hi_iq, "hr_1hz_30s"]
    if int(hi_iq.sum()) > 0:
        idx_hi = rng.choice(int(hi_iq.sum()), size=min(2500, int(hi_iq.sum())), replace=False)
        ax2.scatter(df_w.loc[hi_iq, "hr_1hz_30s"].iloc[idx_hi], df_w.loc[hi_iq, "hr_rtrig_bpm"].iloc[idx_hi],
                    s=5, alpha=0.25, color=OKABE_ITO["green"], zorder=2,
                    label=rf"SignalIQ $\geq$ 50 (MAE={np.mean(np.abs(d_hr)):.2f})", rasterized=True)
    if int(lo_iq.sum()) > 0:
        ax2.scatter(df_w.loc[lo_iq, "hr_1hz_30s"], df_w.loc[lo_iq, "hr_rtrig_bpm"], s=10, alpha=0.75,
                    color=OKABE_ITO["orange"], zorder=4, label="SignalIQ < 50", rasterized=True)
    ax2.plot([80, 220], [80, 220], "k--", linewidth=1.0, label="Identity")
    ax2.set_xlim(80, 220)
    ax2.set_ylim(80, 220)
    ax2.set_xlabel(r"Monitor 1 Hz $\mathrm{HR}$ (bpm)")
    ax2.set_ylabel(r"1024 Hz $\mathrm{R\text{-}Trigger}$ $\mathrm{HR}$ (bpm)")
    ax2.set_title("(b) Cardiac Trigger Concordance by SignalIQ")
    ax2.legend(loc="upper left", fontsize=6.5, frameon=True)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    return _save_fig_pair(fig, "exp0_edf256_reference_agreement")


def run_exp0(write_outputs: bool = True) -> dict:
    """Execute full read-only EXP-0 pipeline and return summary dictionary."""
    IO.verify_all_baseline_caches()
    ledger = IO.load_cohort_ledger()
    lags_map = IO.load_locked_neutral_lags()
    all_sessions = IO.get_cohort_sessions("all_paired", include_zero_valid=True)

    window_frames: dict[tuple[str, str], pd.DataFrame] = {}
    session_records: list[dict] = []

    p18_set = set(IO.get_cohort_sessions("PAPER_18"))
    ret23_set = set(IO.get_cohort_sessions("retained23"))
    p30_set = set(IO.get_cohort_sessions("pool30_sync"))
    all_eval_set = set(IO.get_cohort_sessions("all_paired"))

    edf_br_errs_all = []
    edf_hr_errs_hi_iq = []
    pause_gt5_count = 0
    total_valid_neutral_win = 0

    for m in all_sessions:
        cache = IO.load_locked_cache(m)
        df_radar = IO.load_radar_full(m)
        edf = EDFX.load_edf_recording(m)

        est_nmf = np.asarray(cache["est"], float)
        est_hpss = np.asarray(cache["est_hpss"], float)
        est_nls = np.asarray(cache["est_nls"], float)
        est_dft = np.asarray(cache["est_dft"], float)
        minmove_cached = np.asarray(cache["minmove"], bool)
        valid_cached = np.asarray(cache["valid"], bool)
        n_win = len(est_hpss)
        centers_s = np.arange(n_win, dtype=float) * C.UPDATE_S + C.WIN_S / 2.0

        telem = _compute_radar_window_telemetry(df_radar, n_win)
        sess_telem = telem["session_summary"]

        lag_info = lags_map[m]
        regime_lags = {
            "cached_nmf_lag": int(lag_info["cached_nmf_lag_s"]),
            "branch_neutral_raw_nls_lag": int(lag_info["neutral_raw_nls_lag_s"]),
            "pure_metadata_clock_lag": int(lag_info["pure_metadata_lag_s"]),
        }

        per_reg_sc: dict[str, dict] = {}
        ref_sat_frac_sess = float(np.mean(np.abs(edf.resp_256) >= 2.99))

        for reg_name, lag_val in regime_lags.items():
            ext = EDFX.extract_window_references(edf, centers_s, lag_s=lag_val)
            if reg_name == "cached_nmf_lag":
                # Guarantee bit-for-bit reproducibility with locked `v1.0-student-baseline`
                ref_arr = np.asarray(cache["ref"], float)
                v_arr = valid_cached
                assert np.array_equal(v_arr, ext["valid_vwin"] & np.isfinite(ext["br_1hz_30s"]) & np.isfinite(est_nmf)), m
            else:
                ref_arr = ext["br_1hz_30s"]
                v_arr = ext["valid_vwin"] & np.isfinite(ref_arr) & np.isfinite(est_nmf) & np.isfinite(est_hpss)

            df_w = pd.DataFrame({
                "measurement": m,
                "t_center_s": centers_s,
                "est_nmf": est_nmf,
                "est_hpss": est_hpss,
                "est_nls": est_nls,
                "est_dft": est_dft,
                "ref": ref_arr,
                "valid": v_arr,
                "minmove": minmove_cached,
                "br_trig_bpm": ext["br_trig_bpm"],
                "br_256_spec_bpm": ext["br_256_spec_bpm"],
                "resp_max_gap_s": ext["resp_max_gap_s"],
                "resp_rate_slope_bpm_s": ext["resp_rate_slope_bpm_s"],
                "hr_1hz_30s": ext["hr_1hz_30s"],
                "hr_rtrig_bpm": ext["hr_rtrig_bpm"],
                "hr_ecg256_bpm": ext["hr_ecg256_bpm"],
                "signaliq_mean": ext["signaliq_mean"],
                "sd_ratio_w": telem["sd_ratio_w"],
                "abs_corr_w": telem["abs_corr_w"],
                "clip_frac_w": telem["clip_frac_w"],
                "ucd_nonchild_w": telem["ucd_nonchild_w"],
            })
            window_frames[(m, reg_name)] = df_w

            if np.any(v_arr):
                sh = _window_scores(est_hpss[v_arr], ref_arr[v_arr])
                sn = _window_scores(est_nmf[v_arr], ref_arr[v_arr])
                snls = _window_scores(est_nls[v_arr], ref_arr[v_arr])
                per_reg_sc[reg_name] = {"hpss": sh, "nmf": sn, "nls": snls, "n_valid": int(v_arr.sum())}
            else:
                nan_sc = {k: float("nan") for k in ("rmse", "mae", "bias", "acc3", "acc6", "acc10", "gross_gt10_pct", "r", "oracle_sd")}
                per_reg_sc[reg_name] = {"hpss": nan_sc, "nmf": nan_sc, "nls": nan_sc, "n_valid": 0}

        # Detailed cardiac & harmonic statistics at primary branch_neutral_raw_nls_lag
        df_neu = window_frames[(m, "branch_neutral_raw_nls_lag")]
        vn = df_neu["valid"].to_numpy(bool)
        if np.any(vn):
            total_valid_neutral_win += int(vn.sum())
            eh_v = df_neu.loc[vn, "est_hpss"].to_numpy(float)
            rf_v = df_neu.loc[vn, "ref"].to_numpy(float)
            hr_v = df_neu.loc[vn, "hr_rtrig_bpm"].to_numpy(float)
            hr1_v = df_neu.loc[vn, "hr_1hz_30s"].to_numpy(float)
            br_t_v = df_neu.loc[vn, "br_trig_bpm"].to_numpy(float)
            siq_v = df_neu.loc[vn, "signaliq_mean"].to_numpy(float)
            gap_v = df_neu.loc[vn, "resp_max_gap_s"].to_numpy(float)

            ratio = eh_v / np.maximum(1e-6, rf_v)
            near_hr = float(100.0 * np.mean(np.abs(eh_v - hr_v) <= 10.0)) if np.any(np.isfinite(hr_v)) else float("nan")
            near_hhr = float(100.0 * np.mean(np.abs(eh_v - 0.5 * hr_v) <= 10.0)) if np.any(np.isfinite(hr_v)) else float("nan")
            hr_coll = float(100.0 * np.mean((np.abs(eh_v - rf_v) > 10.0) & ((np.abs(eh_v - hr_v) <= 10.0) | (np.abs(eh_v - 0.5 * hr_v) <= 10.0)))) if np.any(np.isfinite(hr_v)) else float("nan")
            slope_mean = float(np.mean(np.abs(df_neu.loc[vn, "resp_rate_slope_bpm_s"])) * 60.0)

            m_br_t = np.isfinite(br_t_v)
            if np.any(m_br_t):
                edf_br_errs_all.append(br_t_v[m_br_t] - rf_v[m_br_t])
            m_hr_hi = np.isfinite(hr_v) & np.isfinite(hr1_v) & (hr1_v > 50) & (siq_v >= 50)
            if np.any(m_hr_hi):
                edf_hr_errs_hi_iq.append(hr_v[m_hr_hi] - hr1_v[m_hr_hi])
            pause_gt5_count += int(np.sum(gap_v > 5.0))

            pct_half = float(100.0 * np.mean((ratio >= 0.4) & (ratio <= 0.6)))
            pct_dbl = float(100.0 * np.mean((ratio >= 1.8) & (ratio <= 2.2)))
            pct_harm = float(100.0 * np.mean(((ratio >= 0.6) & (ratio < 0.8)) | ((ratio > 1.25) & (ratio <= 1.6))))
            hr_mean = float(np.nanmean(hr_v))
            hr1_mean = float(np.nanmean(hr1_v))
            siq_mean = float(np.mean(siq_v))
            mm_frac = float(np.mean(df_neu.loc[vn, "minmove"]))
        else:
            near_hr = near_hhr = hr_coll = slope_mean = float("nan")
            pct_half = pct_dbl = pct_harm = hr_mean = hr1_mean = siq_mean = mm_frac = float("nan")

        lrow = ledger[ledger["measurement"] == m].iloc[0]
        sc_c = per_reg_sc["cached_nmf_lag"]
        sc_n = per_reg_sc["branch_neutral_raw_nls_lag"]
        sc_m = per_reg_sc["pure_metadata_clock_lag"]

        rec = {
            "measurement": m,
            "patient_group": int(lrow["patient_group"]),
            "position": str(lrow["position"]),
            "cobedding_twins": bool(lrow["cobedding_twins"]),
            "lid_status": str(lrow["lid_status"]),
            "in_paper18": m in p18_set,
            "in_retained23": m in ret23_set,
            "in_pool30_sync": m in p30_set,
            "in_all_paired_eval": m in all_eval_set,
            "is_zero_overlap": m in IO.ZERO_OVERLAP_IDS,
            "is_zero_valid": m in IO.ZERO_VALID_IDS,
            "lag_cached_s": regime_lags["cached_nmf_lag"],
            "lag_neutral_s": regime_lags["branch_neutral_raw_nls_lag"],
            "lag_meta_s": regime_lags["pure_metadata_clock_lag"],
            "lag_pull_abs_s": abs(regime_lags["cached_nmf_lag"] - regime_lags["branch_neutral_raw_nls_lag"]),
            "n_total_win": n_win,
            "n_valid_cached": sc_c["n_valid"],
            "n_valid_neutral": sc_n["n_valid"],
            "ref_sat_frac_256hz": ref_sat_frac_sess,
            "rmse_hpss_cached": sc_c["hpss"]["rmse"],
            "rmse_nmf_cached": sc_c["nmf"]["rmse"],
            "rmse_nls_cached": sc_c["nls"]["rmse"],
            "bias_hpss_cached": sc_c["hpss"]["bias"],
            "acc6_hpss_cached": sc_c["hpss"]["acc6"],
            "gross_gt10_hpss_cached": sc_c["hpss"]["gross_gt10_pct"],
            "r_hpss_cached": sc_c["hpss"]["r"],
            "rmse_hpss_neutral": sc_n["hpss"]["rmse"],
            "rmse_nmf_neutral": sc_n["nmf"]["rmse"],
            "rmse_nls_neutral": sc_n["nls"]["rmse"],
            "mae_hpss_neutral": sc_n["hpss"]["mae"],
            "bias_hpss_neutral": sc_n["hpss"]["bias"],
            "acc3_hpss_neutral": sc_n["hpss"]["acc3"],
            "acc6_hpss_neutral": sc_n["hpss"]["acc6"],
            "acc10_hpss_neutral": sc_n["hpss"]["acc10"],
            "gross_gt10_hpss_neutral": sc_n["hpss"]["gross_gt10_pct"],
            "r_hpss_neutral": sc_n["hpss"]["r"],
            "r_nmf_neutral": sc_n["nmf"]["r"],
            "oracle_sd_cached": sc_c["hpss"]["oracle_sd"],
            "oracle_sd_neutral": sc_n["hpss"]["oracle_sd"],
            "excess_rmse_hpss_neutral": sc_n["hpss"]["rmse"] - sc_n["hpss"]["oracle_sd"],
            "delta_lag_pull_rmse": sc_c["hpss"]["rmse"] - sc_n["hpss"]["rmse"],
            "rmse_hpss_meta": sc_m["hpss"]["rmse"],
            "r_hpss_meta": sc_m["hpss"]["r"],
            "iq_sd_ratio_session": sess_telem["iq_sd_ratio_session"],
            "iq_abs_corr_session": sess_telem["iq_abs_corr_session"],
            "iq_cond_session": sess_telem["iq_cond_session"],
            "radar_clip_frac_session": sess_telem["radar_clip_frac_session"],
            "ucd_nonchild_frac_session": sess_telem["ucd_nonchild_frac_session"],
            "ucd_mid_session_dropout": sess_telem["ucd_mid_session_dropout"],
            "dac_large_jumps_session": sess_telem["dac_large_jumps_session"],
            "dac_dither_rate_per_min": sess_telem["dac_dither_rate_per_min"],
            "hw_error_frac": sess_telem["hw_error_frac"],
            "hr_1hz_mean": hr1_mean,
            "hr_rtrig_mean": hr_mean,
            "signaliq_mean": siq_mean,
            "pct_win_near_hr_10bpm": near_hr,
            "pct_win_near_half_hr_10bpm": near_hhr,
            "pct_win_hr_collision_band": hr_coll,
            "pct_win_br_half_octave": pct_half,
            "pct_win_br_double_octave": pct_dbl,
            "pct_win_br_harmonic_band": pct_harm,
            "ref_slope_abs_mean_bpm_min": slope_mean,
            "minmove_frac": mm_frac,
        }
        rec["dominant_failure_mode"] = _classify_dominant_mode(rec)
        session_records.append(rec)

    df_sess = pd.DataFrame(session_records)
    df_attr = _attribution_analysis(df_sess)

    # Aggregate 4 cohorts x 3 lag regimes
    regimes_summary = {}
    for reg_key in ("cached_nmf_lag", "branch_neutral_raw_nls_lag", "pure_metadata_clock_lag"):
        regimes_summary[reg_key] = {
            coh: _aggregate_cohort(session_records, window_frames, coh, reg_key)
            for coh in COHORT_ORDER
        }

    # LOPO split verification metadata
    lopo_info = {
        coh: [
            {"patient_id": f.patient_id, "n_train": len(f.train_sessions), "n_test": len(f.test_sessions), "test": list(f.test_sessions)}
            for f in IO.iter_lopo_folds(coh)
        ]
        for coh in ("pool30_sync", "all_paired")
    }

    br_errs = np.concatenate(edf_br_errs_all) if edf_br_errs_all else np.empty(0)
    hr_errs = np.concatenate(edf_hr_errs_hi_iq) if edf_hr_errs_hi_iq else np.empty(0)

    outliers = {}
    for mid in ("9.2", "11.1", "12.1", "9.3", "3.3", "7.2", "1.3", "5.1"):
        row = df_sess[df_sess["measurement"] == mid].iloc[0].to_dict()
        outliers[mid] = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}

    summary_payload = {
        "experiment": "EXP-0",
        "provenance": IO.git_head_info(),
        "baseline_caches_verified": 33,
        "regimes": regimes_summary,
        "edf_256hz_validation": {
            "br_trig_vs_1hz_n_win": int(len(br_errs)),
            "br_trig_vs_1hz_bias_bpm": float(np.mean(br_errs)),
            "br_trig_vs_1hz_mae_bpm": float(np.mean(np.abs(br_errs))),
            "hr_rtrig_vs_1hz_signaliq_ge50_n_win": int(len(hr_errs)),
            "hr_rtrig_vs_1hz_signaliq_ge50_mae_bpm": float(np.mean(np.abs(hr_errs))),
            "valid_windows_with_pause_gt5s": int(pause_gt5_count),
            "total_valid_neutral_windows_31sessions": int(total_valid_neutral_win),
        },
        "lopo_folds": lopo_info,
        "outliers": outliers,
        "top_attribution_spearman": df_attr.head(8).to_dict(orient="records"),
    }

    if write_outputs:
        _configure_matplotlib()
        fig_files: list[str] = []
        fig_files.extend(_plot_session_decomposition(df_sess, summary_payload))
        fig_files.extend(_plot_physics_drivers(df_sess, window_frames))
        fig_files.extend(_plot_edf256_validation(window_frames))
        summary_payload["figures_exported"] = fig_files

        IO.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        IO.BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

        df_sess.to_csv(IO.AUDIT_DIR / "forensics_31sessions.csv", index=False)
        df_attr.to_csv(IO.BUNDLE_DIR / "forensics_attribution.csv", index=False)
        IO.write_json_locked(IO.BUNDLE_DIR / "metrics_wp0_forensics.json", summary_payload)
        IO.write_json_locked(IO.AUDIT_DIR / "forensics_windows_summary.json", summary_payload)
        _write_dense_ch2_addendum(summary_payload, df_attr)

    return summary_payload


def _write_dense_ch2_addendum(summary: dict, df_attr: pd.DataFrame) -> None:
    """Export dense PaperOrchestra Chapter 2 baseline & EXP-0 forensics specification."""
    r_c = summary["regimes"]["cached_nmf_lag"]
    r_n = summary["regimes"]["branch_neutral_raw_nls_lag"]
    edf_v = summary["edf_256hz_validation"]
    top_attr = df_attr.head(5)

    lines = [
        "# Chapter 2 Dense Input: Baseline Audit & EXP-0 Failure Forensics",
        "",
        "Strict symbol adherence: [notation_glossary.tex](file:///Users/hosamshibly/Downloads/medical_radar/paper_orchestra_bundle/raw_materials/notation_glossary.tex). "
        "Claim constraints: `L0-L3` internal validation only (`L2` exact Beltrao cohort reproduction blocked; `L4` external and `L5` clinical claims prohibited).",
        "",
        "## 1. Cohort Multiverse & Lag-Leakage Decomposition (`EXP-0` Locked Outputs)",
        f"- **Primary Confirmatory Cohort (`pool30_sync`, $S=28$ synced sessions, $11$ patients, $|\\mathcal{{W}}|={r_n['pool30_sync']['n_windows']}$):** "
        f"Under cached `v1` lags, `HPSS+NLS` obtains $\\rmseSess = {r_c['pool30_sync']['rmse_sess_hpss']:.3f}\\,\\mathrm{{bpm}}$ "
        f"($\\constLoso = {r_c['pool30_sync']['const_loso']:.3f}$, $\\oracleSd = {r_c['pool30_sync']['session_mean_oracle']:.3f}$, "
        f"$\\rWithinSessMean = {r_c['pool30_sync']['r_within_sess_mean_hpss']:.3f}$, ANCOVA $\\rWithin = {r_c['pool30_sync']['r_within_rmcorr_hpss']:.3f}$).",
        f"- **Branch-Neutral Synchronization Impact (`branch_neutral_raw_nls_lag`):** "
        f"Fixing synchronization lags without candidate leakage reduces `pool30_sync` $\\rmseSess$ from "
        f"${r_c['pool30_sync']['rmse_sess_hpss']:.3f}$ to ${r_n['pool30_sync']['rmse_sess_hpss']:.3f}\\,\\mathrm{{bpm}}$ "
        f"(full ``all_paired`` cohort: ${r_c['all_paired']['rmse_sess_hpss']:.3f} \\to {r_n['all_paired']['rmse_sess_hpss']:.3f}\\,\\mathrm{{bpm}}$), "
        f"isolating Session `12.1` as a pure NMF lag-pull artifact (`{summary['outliers']['12.1']['rmse_hpss_cached']:.2f}` -> `{summary['outliers']['12.1']['rmse_hpss_neutral']:.2f}` bpm).",
        f"- **Sensitivity Cohorts (`neutral`):** `retained23` $\\rmseSess = {r_n['retained23']['rmse_sess_hpss']:.3f}\\,\\mathrm{{bpm}}$; "
        f"`PAPER_18` $\\rmseSess = {r_n['PAPER_18']['rmse_sess_hpss']:.3f}\\,\\mathrm{{bpm}}$ ($\\rmsePool = {r_n['PAPER_18']['rmse_pool_hpss']:.3f}\\,\\mathrm{{bpm}}$).",
        "",
        "## 2. True Physical Error Modes Persisting Under Neutral Synchronization",
        f"- **Quadrature Imbalance & Collinearity (`WP1` Target):** Session `11.1` ($\\rho_{{IQ}}={summary['outliers']['11.1']['iq_sd_ratio_session']:.2f}$, "
        f"$|r_{{IQ}}|={summary['outliers']['11.1']['iq_abs_corr_session']:.3f}$, clipping `{100*summary['outliers']['11.1']['radar_clip_frac_session']:.2f}%`) "
        f"persists at $\\mathrm{{RMSE}}_s = {summary['outliers']['11.1']['rmse_hpss_neutral']:.2f}\\,\\mathrm{{bpm}}$ across all lags.",
        f"- **Mid-Session Hardware / UCD Dropouts (`WP1` / `WP4` Target):** Session `9.2` exhibits mid-session `UCD Status != ChildPresent` "
        f"(`9.2` $\\mathrm{{RMSE}}_s = {summary['outliers']['9.2']['rmse_hpss_neutral']:.2f}\\,\\mathrm{{bpm}}$).",
        f"- **High-Rate Reference Verification:** `1024 Hz Resp.-Trigger` reproduces `1 Hz RespRate` across "
        f"${edf_v['br_trig_vs_1hz_n_win']}$ windows (bias ${edf_v['br_trig_vs_1hz_bias_bpm']:+.2f}\\,\\mathrm{{bpm}}$, MAE ${edf_v['br_trig_vs_1hz_mae_bpm']:.2f}\\,\\mathrm{{bpm}}$), "
        f"and `1024 Hz R-Trigger` matches `1 Hz HeartRate` (`SignalIQ >= 50`, MAE ${edf_v['hr_rtrig_vs_1hz_signaliq_ge50_mae_bpm']:.2f}\\,\\mathrm{{bpm}}$).",
        "",
        "## 3. Top Observational Covariates Explaining Excess `pool30_sync` Error",
    ]
    for _, row in top_attr.iterrows():
        lines.append(
            f"- `{row['covariate']}` ({row['description']}): Spearman $\\rho_{{\\mathrm{{excess}}}} = {row['spearman_rho_excess_rmse']:+.3f}$ "
            f"($p={row['p_excess_rmse']:.4f}$), $\\rho_r = {row['spearman_rho_within_r']:+.3f}$."
        )
    (IO.BUNDLE_DIR / "idea_dense_ch2_wp0.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    res = run_exp0(write_outputs=True)
    p30_c = res["regimes"]["cached_nmf_lag"]["pool30_sync"]
    p30_n = res["regimes"]["branch_neutral_raw_nls_lag"]["pool30_sync"]
    print(
        f"EXP-0 COMPLETE | pool30_sync cached={p30_c['rmse_sess_hpss']:.4f} bpm "
        f"| neutral={p30_n['rmse_sess_hpss']:.4f} bpm | rmcorr={p30_n['r_within_rmcorr_hpss']:.3f} "
        f"| figures={len(res['figures_exported'])}"
    )
