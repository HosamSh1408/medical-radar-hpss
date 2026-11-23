"""Publication-grade Okabe-Ito figures for Stage 4 (`WP5`, `L4` Gleichauf 2024 External Benchmark).

Reads exclusively from `paper_orchestra_bundle/raw_materials/metrics_stage4_external.json`
(zero hard-coded numbers). Adheres strictly to `notation_glossary.tex` typography.

Exports dual vector `.pdf` and 300 DPI `.png` into `paper_orchestra_bundle/raw_materials/figures/`:
  1. `stage4_external_frequency_transfer.{pdf,png}`
  2. `stage4_external_uncertainty_and_conformal.{pdf,png}`

Execution:
  MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /opt/homebrew/bin/python3.10 -m thesis.plot_stage4_external
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

METRICS_JSON = IO.BUNDLE_DIR / "metrics_stage4_external.json"


def _configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 6.2,
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


def plot_frequency_transfer_and_ladder(master: dict[str, Any]) -> list[str]:
    """Generate `stage4_external_frequency_transfer.{pdf,png}`."""
    ps = master["per_scenario"]
    rates_nom = sorted({float(ps[s]["f_nom_bpm"]) for s in ps})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.05))
    fig.subplots_adjust(wspace=0.28)

    # Panel (a): Ground truth vs predicted mean across 05-80 bpm (mean across normal/deep at each rate)
    ax1.axvspan(0.0, 28.0, color="#F2E6E6", alpha=0.55, zorder=0, label=r"Sub-band / Kaiser skirt ($< 28\;\mathrm{bpm}$)")
    ax1.axvline(28.0, color=OKABE_ITO["vermilion"], ls=":", lw=1.0, zorder=1)
    ax1.plot([0.0, 86.0], [0.0, 86.0], "--", color=OKABE_ITO["gray"], lw=1.1, label=r"Ideal $\hat{f}_b = f^{\mathrm{ref}}$")

    curve_specs = (
        ("R2e_Causal_DiffHMM_FilterSafe8", "R2 FilterSafe-8 (8p, 0 s)", OKABE_ITO["blue"], "-o", 1.8, 4.2),
        ("R1a_Online_Forward_MMSE_0s", "R1 Forward-MMSE (3p, 0 s)", OKABE_ITO["green"], "--s", 1.4, 3.6),
        ("R3a_Causal_TCN_16k", "R3 Causal TCN (15.8k)", OKABE_ITO["vermilion"], "-^", 1.5, 4.0),
        ("R3c_Ridge_Spectral_439", "R3 Ridge (439p)", OKABE_ITO["orange"], "-.D", 1.35, 3.5),
        ("R3b_Spectral_MLP_15k", "R3 Spectral MLP (14.6k)", OKABE_ITO["purple"], ":v", 1.45, 3.8),
    )

    for km, lbl, col, fmt, lw, ms in curve_specs:
        xs, ys = [], []
        for fn in rates_nom:
            matching = [s for s in ps if float(ps[s]["f_nom_bpm"]) == fn]
            xs.append(float(np.mean([float(ps[s]["laser_f_mid_bpm"]) for s in matching])))
            ys.append(float(np.mean([float(ps[s]["models"][km]["mean_bpm"]) for s in matching])))
        ax1.plot(xs, ys, fmt, color=col, lw=lw, ms=ms, label=lbl, zorder=3 if km.startswith("R2") else 2)

    ax1.text(14.5, 83.0, "Stopband / FIR Skirt\n(< 28 bpm)", ha="center", va="top", fontsize=6.2, color=OKABE_ITO["gray"])
    ax1.set_xlim(2.0, 84.0)
    ax1.set_ylim(2.0, 86.0)
    ax1.set_xlabel(r"Laser Reference Rate $f^{\mathrm{ref}}$ (bpm)")
    ax1.set_ylabel(r"Zero-Touch Predicted Mean $\hat{f}_b$ (bpm)")
    ax1.set_title("(a) Physical Rate Transfer (05–80 bpm)")
    ax1.grid(True, alpha=0.22, lw=0.6)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=5.8)

    # Panel (b): Grouped bar chart of In-Band (>= 28 bpm, N=14) vs 80 bpm extreme tachypnea (N=2)
    reg = master["regimes"]["steady_state_excl_sync_pulses"]
    inb = reg["inband_ge28bpm"]
    ext80 = reg["inband_extreme_80bpm"]

    bar_models = (
        ("R0_HPSS_NLS_Locked", "R0\nNLS\n(0p)"),
        ("R1a_Online_Forward_MMSE_0s", "R1a\nFwd\n(3p)"),
        ("R1f_Offline_FB_MMSE_Inf", "R1f\nFB\n(3p)"),
        ("R2e_Causal_DiffHMM_FilterSafe8", "R2e\nFS-8\n(8p)"),
        ("R2a_Causal_DiffHMM_Linear11", "R2a\nLin\n(11p)"),
        ("R2d_E2E_Causal_H5_DiffHMM_Linear11", "R2d\nH5\n(11p)"),
        ("R3a_Causal_TCN_16k", "R3a\nTCN\n(16k)"),
        ("R3b_Spectral_MLP_15k", "R3b\nMLP\n(15k)"),
        ("R3c_Ridge_Spectral_439", "R3c\nRidge\n(439)"),
    )
    x = np.arange(len(bar_models), dtype=np.float64)
    w = 0.37
    v_inb = np.array([float(inb[m]["rmse_scen"]) for m, _ in bar_models], dtype=np.float64)
    v_80 = np.array([float(ext80[m]["rmse_scen"]) for m, _ in bar_models], dtype=np.float64)

    ax2.bar(x - 0.5 * w, v_inb, width=w, color=OKABE_ITO["blue"], alpha=0.9, label=r"In-Band $\geq 28\,\mathrm{bpm}$ ($N=14$)")
    ax2.bar(x + 0.5 * w, v_80, width=w, color=OKABE_ITO["vermilion"], alpha=0.88, label=r"Extreme $80\,\mathrm{bpm}$ ($N=2$)")

    for i, (vi, v8) in enumerate(zip(v_inb, v_80)):
        ax2.text(x[i] - 0.5 * w, vi + 0.16, f"{vi:.2f}", ha="center", va="bottom", fontsize=5.3, rotation=90)
        ax2.text(x[i] + 0.5 * w, v8 + 0.16, f"{v8:.2f}", ha="center", va="bottom", fontsize=5.3, rotation=90)

    ax2.axvline(5.5, color=OKABE_ITO["gray"], ls="--", lw=0.9)
    ax2.text(2.5, 13.0, r"Equivariant Rungs 0–2 ($\leq 0.07\;\mathrm{bpm}$)", ha="center", va="top", fontsize=6.4, color=OKABE_ITO["blue"], fontweight="bold")
    ax2.text(7.0, 13.0, "Rung 3 Foils", ha="center", va="top", fontsize=6.4, color=OKABE_ITO["vermilion"], fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl in bar_models], fontsize=6.3)
    ax2.set_ylim(0.0, 13.7)
    ax2.set_ylabel(r"$\mathrm{RMSE}_{\mathrm{scen}}$ (bpm)")
    ax2.set_title("(b) Passband Error Across Rungs (24.2 GHz Zero-Touch)")
    ax2.grid(True, axis="y", alpha=0.22, lw=0.6)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.01, 0.86), frameon=True, framealpha=0.95, fontsize=6.0)

    return _save_fig_pair(fig, "stage4_external_frequency_transfer")


def plot_uncertainty_and_conformal(master: dict[str, Any]) -> list[str]:
    """Generate `stage4_external_uncertainty_and_conformal.{pdf,png}`."""
    wp4_r2e = master["wp4_uncertainty_transfer"]["R2e_Causal_DiffHMM_FilterSafe8"]
    ps = master["per_scenario"]
    rates_nom = sorted({float(ps[s]["f_nom_bpm"]) for s in ps})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.05))
    fig.subplots_adjust(wspace=0.28)

    # Panel (a): Proxy profile across 05-80 bpm (mean over normal/deep)
    ax1.axvspan(0.0, 28.0, color="#F2E6E6", alpha=0.55, zorder=0, label=r"Sub-band / stopband ($< 28\;\mathrm{bpm}$)")
    ax1.axvline(28.0, color=OKABE_ITO["vermilion"], ls=":", lw=1.0)

    p_specs = (
        ("U2_Posterior_SD", r"U2 Post-SD $\tilde{\sigma}_{\mathrm{post},t}$", OKABE_ITO["blue"], "-o"),
        ("U6b_ZeroTuned_Geometric_SQI", r"U6b Geom-SQI $(1-\mathrm{SQI}_{\mathrm{geom}})$", OKABE_ITO["green"], "--s"),
        ("U3_Spectral_Entropy_Clean", r"U3 Spec-Ent $H_{\mathrm{spec},t}$", OKABE_ITO["orange"], "-.^"),
    )
    prof = wp4_r2e["mean_proxy_by_scenario"]
    for pk, lbl, col, fmt in p_specs:
        xs, ys = [], []
        for fn in rates_nom:
            ms = [s for s in ps if float(ps[s]["f_nom_bpm"]) == fn]
            xs.append(float(np.mean([float(ps[s]["laser_f_mid_bpm"]) for s in ms])))
            ys.append(float(np.mean([float(prof[s][pk]) for s in ms])))
        ax1.plot(xs, ys, fmt, color=col, lw=1.6, ms=3.9, label=lbl)

    au_u2 = float(wp4_r2e["stopband_05_10bpm_vs_inband_ood"]["U2_Posterior_SD"]["auroc"])
    au_u6b = float(wp4_r2e["stopband_05_10bpm_vs_inband_ood"]["U6b_ZeroTuned_Geometric_SQI"]["auroc"])
    au_u3 = float(wp4_r2e["stopband_05_10bpm_vs_inband_ood"]["U3_Spectral_Entropy_Clean"]["auroc"])
    au_sync = float(wp4_r2e["sync_pulse_transient_vs_steady_inband"]["U3_Spectral_Entropy_Clean"]["auroc"])
    box_a = (
        "Stopband (05–10 bpm) OOD AUROC:\n"
        f"U2 Post-SD: {au_u2:.4f} | U6b Geom-SQI: {au_u6b:.4f}\n"
        f"U3 Spec-Ent: {au_u3:.4f} | Sync Pulse: {au_sync:.3f}"
    )
    ax1.text(
        0.97,
        0.61,
        box_a,
        transform=ax1.transAxes,
        ha="right",
        va="center",
        fontsize=6.0,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#F8F8F8", "edgecolor": OKABE_ITO["silver"], "alpha": 0.95},
    )

    ax1.set_xlim(2.0, 84.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel(r"Laser Reference Rate $f^{\mathrm{ref}}$ (bpm)")
    ax1.set_ylabel(r"Normalized Uncertainty $u \in [0, 1]$")
    ax1.set_title("(a) Zero-Parameter Uncertainty Across 05–80 bpm")
    ax1.grid(True, alpha=0.22, lw=0.6)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=5.8)

    # Panel (b): Zero-touch NICU-to-Gleichauf conformal transfer (C4 vs C5, Full-K=11 frozen quantiles)
    cf = master["conformal_transfer"]["primary_full_k11_frozen_quantiles"]["R2e_Causal_DiffHMM_FilterSafe8"]
    alphas = ("0.10", "0.20", "0.30")
    x = np.arange(len(alphas), dtype=np.float64)
    bw = 0.19

    c4_inb = [float(cf["inband_ge28bpm"][a]["c4_cov_scen_mean"]) for a in alphas]
    c5_inb = [float(cf["inband_ge28bpm"][a]["c5_cov_scen_mean"]) for a in alphas]
    c4_a20 = [float(cf["all_20"][a]["c4_cov_scen_mean"]) for a in alphas]
    c5_a20 = [float(cf["all_20"][a]["c5_cov_scen_mean"]) for a in alphas]

    w4 = [float(cf["inband_ge28bpm"][a]["c4_mean_width_bpm"]) for a in alphas]
    w5 = [float(cf["inband_ge28bpm"][a]["c5_mean_width_scen_mean_bpm"]) for a in alphas]

    ax2.bar(x - 1.5 * bw, c4_inb, width=bw, color=OKABE_ITO["blue"], alpha=0.90, label=r"C4 In-Band ($\geq 28\,\mathrm{bpm}$)")
    ax2.bar(x - 0.5 * bw, c5_inb, width=bw, color=OKABE_ITO["green"], alpha=0.90, label=r"C5 Bayes-Conf In-Band ($\geq 28$)")
    ax2.bar(x + 0.5 * bw, c4_a20, width=bw, color=OKABE_ITO["sky"], alpha=0.88, label=r"C4 All 20 (incl. $<28$)")
    ax2.bar(x + 1.5 * bw, c5_a20, width=bw, color=OKABE_ITO["orange"], alpha=0.88, label=r"C5 Bayes-Conf All 20")

    for i, a in enumerate(alphas):
        targ = 1.0 - float(a)
        ax2.hlines(targ, x[i] - 2.2 * bw, x[i] + 2.2 * bw, colors=OKABE_ITO["gray"], linestyles="--", lw=1.2)
        ax2.text(x[i] - 1.5 * bw, c4_inb[i] + 0.015, f"{w4[i]:.1f}", ha="center", va="bottom", fontsize=5.4, color=OKABE_ITO["blue"], rotation=90)
        ax2.text(x[i] - 0.5 * bw, c5_inb[i] + 0.015, f"{w5[i]:.1f}", ha="center", va="bottom", fontsize=5.4, color=OKABE_ITO["green"], rotation=90)
        ax2.text(x[i] + 1.5 * bw, c5_a20[i] + 0.015, f"{c5_a20[i]:.2f}", ha="center", va="bottom", fontsize=5.4, color=OKABE_ITO["gray"])

    ax2.plot([], [], "--", color=OKABE_ITO["gray"], lw=1.2, label=r"Nominal Target $1-\alpha$")
    ax2.set_xticks(x)
    ax2.set_xticklabels([r"$\alpha=0.10$" + "\n(90%)", r"$\alpha=0.20$" + "\n(80%)", r"$\alpha=0.30$" + "\n(70%)"])
    ax2.set_ylim(0.0, 1.19)
    ax2.set_ylabel("Empirical Coverage (Fraction of Windows)")
    ax2.set_title("(b) Zero-Touch Conformal Transfer (Width bpm / Cov)")
    ax2.grid(True, axis="y", alpha=0.22, lw=0.6)
    ax2.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=5.8)

    return _save_fig_pair(fig, "stage4_external_uncertainty_and_conformal")


def main() -> None:
    _configure_matplotlib()
    master = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    f1 = plot_frequency_transfer_and_ladder(master)
    f2 = plot_uncertainty_and_conformal(master)
    print(f"[PLOT-STAGE4] Exported figures: {f1 + f2}")


if __name__ == "__main__":
    main()
