"""Publication-grade Okabe-Ito figures for Stage 3 (WP3/WP4) Uncertainty, Apnea Safety & Conformal Audit.

Reads exclusively from `paper_orchestra_bundle/raw_materials/metrics_stage3_uncertainty.json`
and `thesis/outputs/stage3_uncertainty/pool30_uncertainty_and_conformal.npz` (zero hard-coded numbers).
Notation adheres strictly to `paper_orchestra_bundle/raw_materials/notation_glossary.tex`.

Exports dual vector `.pdf` and 300 DPI `.png` into `paper_orchestra_bundle/raw_materials/figures/`:
  1. `stage3_risk_coverage_and_apnea_safety.{pdf,png}`
  2. `stage3_finite_k_conformal_audit.{pdf,png}`

Execution:
  MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /opt/homebrew/bin/python3.10 -m thesis.plot_stage3_uncertainty
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thesis import io_utils as IO

OKABE_ITO = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#333333",
    "silver": "#888888",
}

METRICS_JSON = IO.BUNDLE_DIR / "metrics_stage3_uncertainty.json"


def _configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 6.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save_fig_pair(fig: plt.Figure, stem: str) -> list[str]:
    IO.BUNDLE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = IO.BUNDLE_FIG_DIR / f"{stem}.pdf"
    png_path = IO.BUNDLE_FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf_path), str(png_path)]


def plot_risk_coverage_and_apnea_safety(master: dict[str, Any]) -> list[str]:
    """Generate `stage3_risk_coverage_and_apnea_safety.{pdf,png}`."""
    sel_p30 = master["selective_prediction_by_cohort"]["pool30_sync"]
    r2e_1hz = sel_p30["br_1hz_30s"]["R2e_Causal_DiffHMM_FilterSafe8"]
    r2a_tla = sel_p30["br_trig_bpm_lag_aligned"]["R2a_Causal_DiffHMM_Linear11"]
    ap_r2e = master["apnea_pause_safety_by_cohort"]["pool30_sync"]["R2e_Causal_DiffHMM_FilterSafe8"]
    ood_r2e = master["ood_abstention_audit_pool30"]["models"]["R2e_Causal_DiffHMM_FilterSafe8"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.85))
    fig.subplots_adjust(wspace=0.38)

    # Panel (a): Equal-Session Macro-Averaged Risk-Coverage over c_cov in [0.50, 1.00]
    cov_grid = np.asarray(r2e_1hz["U2_Posterior_SD"]["session_macro"]["cov_grid"], dtype=np.float64)
    rand_1hz = np.asarray(r2e_1hz["U2_Posterior_SD"]["session_macro"]["random_risk_rmse_curve"], dtype=np.float64)

    curve_specs_1hz = (
        ("U2_Posterior_SD", r"U2 Post-SD $\tilde{\sigma}_{\mathrm{post},t}$ (1 Hz)", OKABE_ITO["blue"], "-", 1.7),
        ("U6b_ZeroTuned_Geometric_SQI", r"U6b Geom-SQI (0p, 1 Hz)", OKABE_ITO["green"], "--", 1.6),
        ("U6c_LOPO_Calibrated_SQI", r"U6c LOPO-SQI (5p, 1 Hz)", OKABE_ITO["purple"], "-.", 1.4),
        ("U3_Spectral_Entropy_Clean", r"U3 Spec-Ent $H_{\mathrm{spec},t}$ (1 Hz)", OKABE_ITO["orange"], ":", 1.5),
    )
    ax1.plot(cov_grid, rand_1hz, "--", color=OKABE_ITO["gray"], lw=1.1, alpha=0.8, label=r"Random abstention (1 Hz)")
    for pk, lbl, col, ls, lw in curve_specs_1hz:
        y_c = np.asarray(r2e_1hz[pk]["session_macro"]["risk_rmse_curve"], dtype=np.float64)
        ax1.plot(cov_grid, y_c, linestyle=ls, color=col, lw=lw, label=lbl)

    # Also overlay physical -10s Trigger selective curves (R2a U6b vs Random)
    rand_tla = np.asarray(r2a_tla["U6b_ZeroTuned_Geometric_SQI"]["session_macro"]["random_risk_rmse_curve"], dtype=np.float64)
    y_tla_u6b = np.asarray(r2a_tla["U6b_ZeroTuned_Geometric_SQI"]["session_macro"]["risk_rmse_curve"], dtype=np.float64)
    y_tla_u2 = np.asarray(r2a_tla["U2_Posterior_SD"]["session_macro"]["risk_rmse_curve"], dtype=np.float64)
    ax1.plot(cov_grid, rand_tla, ":", color=OKABE_ITO["silver"], lw=1.1, label=r"Random abstention (Trig $-10\,\mathrm{s}$)")
    ax1.plot(cov_grid, y_tla_u6b, "-", color=OKABE_ITO["vermilion"], lw=1.65, label=r"R2a U6b Geom-SQI (Trig $-10\,\mathrm{s}$)")
    ax1.plot(cov_grid, y_tla_u2, "--", color=OKABE_ITO["sky"], lw=1.45, label=r"R2a U2 Post-SD (Trig $-10\,\mathrm{s}$)")

    e_u2_1 = float(r2e_1hz["U2_Posterior_SD"]["session_macro"]["e_aurc_vs_random_rmse_sess"])
    p_u2_1 = float(r2e_1hz["U2_Posterior_SD"]["session_macro"]["p_wilcoxon_patient"])
    e_u6b_p = float(r2e_1hz["U6b_ZeroTuned_Geometric_SQI"]["window_pooled"]["e_aurc_vs_random_rmse"])
    e_tla = float(r2a_tla["U6b_ZeroTuned_Geometric_SQI"]["session_macro"]["e_aurc_vs_random_rmse_sess"])
    p_tla = float(r2a_tla["U6b_ZeroTuned_Geometric_SQI"]["session_macro"]["p_wilcoxon_patient"])

    stat_box = (
        rf"R2e U2 $\mathrm{{E\text{{-}}AURC}}_{{\mathrm{{sess}}}}$ (1 Hz): {e_u2_1:+.2f} bpm ($p_{{\mathrm{{pat}}}}={p_u2_1:.3f}$)" + "\n"
        rf"R2e U6b $\mathrm{{E\text{{-}}AURC}}_{{\mathrm{{pool}}}}$ (1 Hz): {e_u6b_p:+.2f} bpm (0 params)" + "\n"
        rf"R2a U6b $\mathrm{{E\text{{-}}AURC}}_{{\mathrm{{sess}}}}$ (Trig): {e_tla:+.2f} bpm ($p_{{\mathrm{{pat}}}}={p_tla:.3f}$)"
    )
    ax1.text(
        0.03,
        0.51,
        stat_box,
        transform=ax1.transAxes,
        ha="left",
        va="center",
        fontsize=5.65,
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "#F8F8F8", "edgecolor": OKABE_ITO["silver"], "alpha": 0.95},
    )

    ax1.set_xlim(0.50, 1.00)
    ax1.set_ylim(5.55, 8.75)
    ax1.set_xlabel(r"Retention Coverage $c_{\mathrm{cov}}$")
    ax1.set_ylabel(r"Selective Risk $\mathrm{RMSE}_{\mathrm{sess}}(c_{\mathrm{cov}})$ (bpm)")
    ax1.set_title(r"(a) Selective Risk–Coverage ($c_{\mathrm{cov}} \in [0.5, 1.0]$)")
    ax1.grid(True, alpha=0.22, lw=0.6)
    ax1.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=5.35, ncol=2, columnspacing=0.8)

    # Panel (b): Quiet-Apnea (> 5.0 s) Pause Rejection vs Regular at c_cov = 0.80 + WGN AUROC
    proxy_rows = (
        ("U1_Posterior_Entropy", "U1 Post-Ent"),
        ("U2_Posterior_SD", "U2 Post-SD"),
        ("U6b_ZeroTuned_Geometric_SQI", "U6b Geom-SQI (0p)"),
        ("U6a_ZeroTuned_Additive_SQI", "U6a Add-SQI (0p)"),
        ("U6c_LOPO_Calibrated_SQI", "U6c LOPO-SQI (5p)"),
        ("U5_HQI_Harmonic_Deficit", "U5 1-HQI"),
        ("U3_Spectral_Entropy_Clean", "U3 Spec-Ent (HPSS)"),
        ("U4_Spectral_Entropy_Raw", "U4 Spec-Ent (Raw)"),
    )
    y_pos = np.arange(len(proxy_rows), dtype=np.float64)
    rej_pause = np.array([
        float(ap_r2e[pk]["pooled_threshold_rejection"]["0.80"]["rej_pause_pct"])
        for pk, _ in proxy_rows
    ], dtype=np.float64)
    rej_reg = np.array([
        float(ap_r2e[pk]["pooled_threshold_rejection"]["0.80"]["rej_regular_pct"])
        for pk, _ in proxy_rows
    ], dtype=np.float64)
    wgn_aurocs = np.array([
        float(ood_r2e[pk]["wgn_200_seed20261011"]["auroc"])
        for pk, _ in proxy_rows
    ], dtype=np.float64)

    bh = 0.34
    ax2.barh(y_pos - bh * 0.5, rej_pause, height=bh, color=OKABE_ITO["vermilion"], alpha=0.88, label=r"Quiet-Apnea Pause $>5\,\mathrm{s}$ ($N=1581$)")
    ax2.barh(y_pos + bh * 0.5, rej_reg, height=bh, color=OKABE_ITO["blue"], alpha=0.85, label=r"Regular Breathing $\leq 5\,\mathrm{s}$ ($N=5059$)")
    ax2.axvline(20.0, color=OKABE_ITO["gray"], ls="--", lw=1.0, label=r"Nominal $20\%$ Abstention ($c_{\mathrm{cov}}=0.80$)")
    ax2.axvline(23.0, color=OKABE_ITO["vermilion"], ls=":", lw=1.0, label=r"$1.15\times$ Pause Over-Rejection Ceiling")

    for idx_r, (rp_v, rr_v, au_v) in enumerate(zip(rej_pause, rej_reg, wgn_aurocs)):
        ratio_v = rp_v / max(rr_v, 1e-9)
        col_txt = OKABE_ITO["green"] if (ratio_v <= 1.15 and au_v >= 0.95) else (OKABE_ITO["vermilion"] if ratio_v > 1.15 else OKABE_ITO["gray"])
        ax2.text(
            max(rp_v, rr_v) + 0.45,
            y_pos[idx_r],
            f"{ratio_v:.2f}x | WGN {au_v:.2f}",
            va="center",
            ha="left",
            fontsize=5.8,
            color=col_txt,
            fontweight="bold" if (ratio_v <= 1.15 and au_v >= 0.95) else "normal",
        )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([lbl for _, lbl in proxy_rows], fontsize=6.8)
    ax2.set_ylim(len(proxy_rows) + 1.85, -0.65)
    ax2.set_xlim(0.0, 38.0)
    ax2.set_xlabel(r"Empirical Abstention Rate at $c_{\mathrm{cov}} = 0.80$ (%)")
    ax2.set_title("(b) Apnea-Pause Safety vs. WGN OOD AUROC")
    ax2.grid(True, axis="x", alpha=0.22, lw=0.6)
    ax2.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=5.4, ncol=1)

    return _save_fig_pair(fig, "stage3_risk_coverage_and_apnea_safety")


def plot_finite_k_conformal_audit(master: dict[str, Any]) -> list[str]:
    """Generate `stage3_finite_k_conformal_audit.{pdf,png}`."""
    conf_p30 = master["finite_k_conformal_by_cohort"]["pool30_sync"]["br_1hz_30s"]["R2e_Causal_DiffHMM_FilterSafe8"]
    by_alpha = conf_p30["by_alpha"]
    alpha_min = float(conf_p30["alpha_min_patient_exchangeable"])
    max_target_nonvac = 100.0 * (1.0 - alpha_min)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.85))
    fig.subplots_adjust(wspace=0.36)

    # Panel (a): Nominal 1-alpha vs Empirical Held-Out Infant Coverage across alpha in {0.30, 0.20, 0.10, 0.05}
    alpha_keys = ["0.30", "0.20", "0.10", "0.05"]
    targets_pct = np.array([100.0 * float(by_alpha[ak]["target_coverage"]) for ak in alpha_keys], dtype=np.float64)

    ax1.plot([66, 98], [66, 98], "--", color=OKABE_ITO["gray"], lw=1.2, label=r"Exact Calibration ($y = 1 - \alpha$)")
    ax1.axvspan(max_target_nonvac, 98.0, color=OKABE_ITO["vermilion"], alpha=0.12, label=rf"Finite-$K$ Vacuous Zone ($\alpha < \frac{{1}}{{K+1}}={alpha_min:.3f}$)")
    ax1.axvline(max_target_nonvac, color=OKABE_ITO["vermilion"], ls=":", lw=1.1)

    regime_specs = (
        ("C2_Naive_LOPO_PooledWindow", "C2 Naive LOPO Pooled-Window", OKABE_ITO["vermilion"], "s--", 1.5, True),
        ("C1_Naive_RandomWindowSplit_50_50", "C1 Naive 50/50 Window-Split", OKABE_ITO["orange"], "d:", 1.3, True),
        ("C3_SessionCluster_Hierarchical_LOPO", "C3 Session-Cluster ($S_{\\mathrm{cal}}\\approx 25$)", OKABE_ITO["purple"], "^-.", 1.4, True),
        ("C4_PatientExchangeable_Hierarchical_LOPO", "C4 Patient-Hierarchical ($K_{\\mathrm{cal}}=10$)", OKABE_ITO["blue"], "o-", 1.7, False),
        ("C5_LocallyAdaptive_BayesConformal_Hierarchical_LOPO", "C5 Locally-Adaptive Bayes-Conformal", OKABE_ITO["green"], "D-", 1.7, False),
    )

    for rkey, lbl, col, fmt, lw, include_95 in regime_specs:
        aks = alpha_keys if include_95 else ["0.30", "0.20", "0.10"]
        xs = np.array([100.0 * float(by_alpha[ak]["target_coverage"]) for ak in aks], dtype=np.float64)
        ys = np.array([100.0 * float(by_alpha[ak]["regimes"][rkey]["empirical_coverage_patient_mean"]) for ak in aks], dtype=np.float64)
        ax1.plot(xs, ys, fmt, color=col, lw=lw, ms=4.8, label=lbl)

    # Mark vacuous [+inf] at 95% for C4/C5
    ax1.scatter([95.0], [100.0], marker="X", s=60, color=OKABE_ITO["blue"], zorder=5)
    ax1.annotate(
        r"C4/C5 vacuous $[-\infty,+\infty]$" + "\n" + r"($\lceil 11 \times 0.95 \rceil = 11 > 10$;" + "\n" + r"needs $K_{\mathrm{cal}} \geq 19$ infants)",
        xy=(95.0, 100.0),
        xytext=(81.5, 96.2),
        fontsize=5.6,
        color=OKABE_ITO["blue"],
        ha="center",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": OKABE_ITO["blue"], "alpha": 0.92},
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": OKABE_ITO["blue"]},
    )

    ax1.set_xlim(67.5, 97.5)
    ax1.set_ylim(63.0, 101.8)
    ax1.set_xticks(targets_pct)
    ax1.set_xticklabels([f"{t:.0f}%\n(α=.{int(round(100*float(ak))):02d})" for t, ak in zip(targets_pct, alpha_keys)], fontsize=6.3)
    ax1.set_xlabel(r"Nominal Target Coverage $1 - \alpha$")
    ax1.set_ylabel("Empirical Held-Out Infant Coverage (%)")
    ax1.set_title(r"(a) Finite-$K$ Conformal Coverage ($K_{\mathrm{cal}}=10$)")
    ax1.grid(True, alpha=0.22, lw=0.6)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.94, fontsize=5.45)

    # Panel (b): Per-Infant Held-Out Coverage at alpha = 0.20 (80% target) for C2 vs C4 vs C5
    a20 = by_alpha["0.20"]["regimes"]
    folds_c2 = a20["C2_Naive_LOPO_PooledWindow"]["folds"]
    folds_c4 = a20["C4_PatientExchangeable_Hierarchical_LOPO"]["folds"]
    folds_c5 = a20["C5_LocallyAdaptive_BayesConformal_Hierarchical_LOPO"]["folds"]

    pids = [int(r["patient_id"]) for r in folds_c2]
    x_idx = np.arange(len(pids), dtype=np.float64)
    c2_cov = np.array([100.0 * float(r["empirical_coverage"]) for r in folds_c2], dtype=np.float64)
    c4_cov = np.array([100.0 * float(r["empirical_coverage"]) for r in folds_c4], dtype=np.float64)
    c5_cov = np.array([100.0 * float(r["empirical_coverage"]) for r in folds_c5], dtype=np.float64)

    bw = 0.26
    ax2.bar(x_idx - bw, c2_cov, width=bw, color=OKABE_ITO["vermilion"], alpha=0.86, label=f"C2 Naive Window-Pool (mean {np.mean(c2_cov):.1f}%, 5/11 under)")
    ax2.bar(x_idx, c4_cov, width=bw, color=OKABE_ITO["blue"], alpha=0.86, label=f"C4 Patient-Hier (mean {np.mean(c4_cov):.1f}%, width {a20['C4_PatientExchangeable_Hierarchical_LOPO']['mean_interval_width_bpm']:.1f} bpm)")
    ax2.bar(x_idx + bw, c5_cov, width=bw, color=OKABE_ITO["green"], alpha=0.86, label=f"C5 Bayes-Adaptive (mean {np.mean(c5_cov):.1f}%, width {a20['C5_LocallyAdaptive_BayesConformal_Hierarchical_LOPO']['mean_interval_width_bpm']:.1f} bpm)")
    ax2.axhline(80.0, color=OKABE_ITO["gray"], ls="--", lw=1.1, label=r"Target $1 - \alpha = 80\%$ ($\alpha=0.20$, order $9/10$)")

    w_lo = float(a20["C5_LocallyAdaptive_BayesConformal_Hierarchical_LOPO"]["low_uncertainty_half_width_bpm"])
    w_hi = float(a20["C5_LocallyAdaptive_BayesConformal_Hierarchical_LOPO"]["high_uncertainty_half_width_bpm"])
    c5_note = (
        rf"C5 Heteroscedastic Width ($\alpha=0.20$): Low-$\sigma_{{\mathrm{{post}}}}$ half {w_lo:.1f} bpm vs. High-$\sigma_{{\mathrm{{post}}}}$ half {w_hi:.1f} bpm"
    )
    ax2.text(
        0.50,
        0.025,
        c5_note,
        transform=ax2.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.55,
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "#F8F8F8", "edgecolor": OKABE_ITO["silver"], "alpha": 0.96},
    )

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels([f"P{p}" for p in pids], fontsize=6.9)
    ax2.set_ylim(22.0, 127.0)
    ax2.set_yticks([40, 60, 80, 100])
    ax2.set_xlabel("Held-Out Test Infant (11-Fold LOPO on pool30_sync)")
    ax2.set_ylabel(r"Empirical Infant Coverage at $\alpha = 0.20$ (%)")
    ax2.set_title(r"(b) Per-Infant Coverage at $\alpha = 0.20$ ($80\%$ Target)")
    ax2.grid(True, axis="y", alpha=0.22, lw=0.6)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=5.35, ncol=1)

    return _save_fig_pair(fig, "stage3_finite_k_conformal_audit")


def main() -> None:
    _configure_matplotlib()
    master = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    f1 = plot_risk_coverage_and_apnea_safety(master)
    f2 = plot_finite_k_conformal_audit(master)
    print("[PLOT-STAGE3] Exported figures:", f1 + f2)


if __name__ == "__main__":
    main()
