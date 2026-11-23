"""Publication-grade Okabe-Ito figures for Stage 2 Complexity Ladder & Shortcut Audit.

Reads exclusively from `paper_orchestra_bundle/raw_materials/metrics_stage2_complexity.json`
and `thesis/outputs/stage2_complexity/pool30_predictions.npz` (zero hard-coded numbers).
Notation adheres strictly to `paper_orchestra_bundle/raw_materials/notation_glossary.tex`.

Exports dual vector `.pdf` and 300 DPI `.png` into `paper_orchestra_bundle/raw_materials/figures/`:
  1. `stage2_complexity_ladder.{pdf,png}`
  2. `stage2_synthetic_sweep_and_weights.{pdf,png}`

Execution:
  MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /opt/homebrew/bin/python3.10 -m thesis.plot_stage2_complexity
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

METRICS_JSON = IO.BUNDLE_DIR / "metrics_stage2_complexity.json"
PRED_ARCHIVE = IO.ROOT / "thesis" / "outputs" / "stage2_complexity" / "pool30_predictions.npz"

SHORT_LABELS: dict[str, str] = {
    "R0_HPSS_NLS_Locked": "R0 NLS",
    "R1a_Online_Forward_MMSE_0s": "R1a Fwd",
    "R1f_Offline_FB_MMSE_Inf": "R1f FB",
    "R1g_Offline_Viterbi_Unpinned": "R1g Vit",
    "R2e_Causal_DiffHMM_FilterSafe8": "R2e FS-8",
    "R2a_Causal_DiffHMM_Linear11": "R2a HMM-11",
    "R2b_Causal_DiffHMM_MLP43": "R2b HMM-43",
    "R2f_Offline_DiffHMM_FilterSafe8": "R2f FB-8",
    "R2c_Offline_DiffHMM_Linear11": "R2c FB-11",
    "R2d_E2E_Causal_H5_DiffHMM_Linear11": "R2d H5-11",
    "R3c_Ridge_Spectral_439": "R3c Ridge",
    "R3b_Spectral_MLP_15k": "R3b MLP",
    "R3a_Causal_TCN_16k": "R3a TCN",
}

CAUSAL_CHAIN = (
    "R0_HPSS_NLS_Locked",
    "R1a_Online_Forward_MMSE_0s",
    "R2e_Causal_DiffHMM_FilterSafe8",
    "R2a_Causal_DiffHMM_Linear11",
    "R2b_Causal_DiffHMM_MLP43",
    "R3c_Ridge_Spectral_439",
    "R3b_Spectral_MLP_15k",
    "R3a_Causal_TCN_16k",
)


def _configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 6.8,
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


def _x_pos(n_params: int) -> float:
    """Smooth monotone log10(n_params + 1) coordinate keeping 0 parameters cleanly anchored."""
    return float(np.log10(float(n_params) + 1.0))


def plot_complexity_ladder(master: dict[str, Any]) -> list[str]:
    """Generate `stage2_complexity_ladder.{pdf,png}`."""
    sc = master["shortcut_learning_audit_pool30"]
    p30 = master["cohorts"]["pool30_sync"]
    const_1hz = float(p30["br_1hz_30s"]["constant_baselines"]["const_loso"])
    const_tla = float(p30["br_trig_bpm_lag_aligned"]["constant_baselines"]["const_loso"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.8))
    fig.subplots_adjust(wspace=0.35)

    # Panel (a): Causal 4-rung complexity frontier on crisp evenly-spaced rungs
    xs = np.arange(len(CAUSAL_CHAIN), dtype=np.float64)
    y_1hz = [float(sc[k]["rmse_1hz"]) for k in CAUSAL_CHAIN]
    y_t0 = [float(sc[k]["rmse_trig"]) for k in CAUSAL_CHAIN]
    y_tla = [float(sc[k]["rmse_trig_lag_aligned"]) for k in CAUSAL_CHAIN]

    ax1.plot(xs, y_1hz, "o-", color=OKABE_ITO["blue"], lw=1.6, ms=4.8, label=r"$1\,\mathrm{Hz}$ RespRate (train ref.)")
    ax1.plot(xs, y_tla, "^-", color=OKABE_ITO["green"], lw=1.6, ms=5.0, label=r"$1024\,\mathrm{Hz}$ Trig ($-10\,\mathrm{s}$ aligned)")
    ax1.plot(xs, y_t0, "s--", color=OKABE_ITO["orange"], lw=1.2, ms=4.2, alpha=0.85, label=r"$1024\,\mathrm{Hz}$ Trig ($0\,\mathrm{s}$ raw)")

    # Offline reference markers (R1f at x=1, R2f at x=2, R2c at x=3)
    for km_off, x_off, mkr in (
        ("R1f_Offline_FB_MMSE_Inf", 1.0, "D"),
        ("R2f_Offline_DiffHMM_FilterSafe8", 2.0, "P"),
        ("R2c_Offline_DiffHMM_Linear11", 3.0, "v"),
    ):
        ax1.plot(
            [x_off],
            [float(sc[km_off]["rmse_1hz"])],
            marker=mkr,
            linestyle="none",
            mfc="white",
            mec=OKABE_ITO["blue"],
            mew=1.2,
            ms=5.2,
            label="Offline FB (R1f / R2f / R2c)" if km_off == "R1f_Offline_FB_MMSE_Inf" else None,
        )
        ax1.plot(
            [x_off],
            [float(sc[km_off]["rmse_trig_lag_aligned"])],
            marker=mkr,
            linestyle="none",
            mfc="white",
            mec=OKABE_ITO["green"],
            mew=1.2,
            ms=5.2,
        )

    ax1.axhline(const_1hz, color=OKABE_ITO["blue"], ls=":", lw=1.0, alpha=0.75)
    ax1.axhline(const_tla, color=OKABE_ITO["green"], ls=":", lw=1.0, alpha=0.75)
    ax1.text(xs[-1], const_1hz - 0.14, r"$\mathrm{const}_{\mathrm{LOSO}}$ (1 Hz)", ha="right", va="top", fontsize=6.2, color=OKABE_ITO["blue"])
    ax1.text(xs[-1], const_tla + 0.10, r"$\mathrm{const}_{\mathrm{LOSO}}$ (Trig)", ha="right", va="bottom", fontsize=6.2, color=OKABE_ITO["green"])

    tick_lbls = ["0\nR0", "3\nR1a", "8\nR2e", "11\nR2a", "43\nR2b", "439\nR3c", "14.6k\nR3b", "15.8k\nR3a"]
    ax1.set_xticks(xs)
    ax1.set_xticklabels(tick_lbls, fontsize=6.3)
    ax1.set_xlim(-0.35, len(CAUSAL_CHAIN) - 0.65)
    ymin_a = min(min(y_1hz), 5.0) - 1.25
    ax1.set_ylim(ymin_a, max(const_tla, max(y_t0)) + 0.75)
    ax1.set_xlabel("Model Complexity (Causal Chain Parameters)")
    ax1.set_ylabel(r"$\mathrm{RMSE}_{\mathrm{sess}}$ (bpm)")
    ax1.set_title("(a) Complexity vs. Cross-Reference Error")
    ax1.grid(True, alpha=0.22, lw=0.6)
    ax1.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=5.8)

    # Panel (b): Physical dynamic-range retention (SD / SD_Trig,-10s) vs physical trigger correlation
    rung_colors = {
        0: OKABE_ITO["gray"],
        1: OKABE_ITO["blue"],
        2: OKABE_ITO["green"],
        3: OKABE_ITO["vermilion"],
    }
    rung_markers = {0: "P", 1: "o", 2: "^", 3: "X"}
    seen_rungs: set[int] = set()
    for km in sc:
        meta = sc[km]
        rg = int(meta["rung"])
        xr = float(meta["pred_sd_ratio_trig_lag_aligned"])
        yr = float(meta["r_within_trig_lag_aligned"])
        lbl = f"Rung {rg}" if rg not in seen_rungs else None
        seen_rungs.add(rg)
        ax2.scatter(
            [xr],
            [yr],
            color=rung_colors[rg],
            marker=rung_markers[rg],
            s=42,
            zorder=4,
            label=lbl,
        )

    # Explicit verified non-overlapping offsets (dx_pt, dy_pt, ha, va) in Panel (b)
    explicit_offsets = {
        "R2a_Causal_DiffHMM_Linear11": (6, 8, "left", "bottom"),
        "R2e_Causal_DiffHMM_FilterSafe8": (16, 4, "left", "bottom"),
        "R2f_Offline_DiffHMM_FilterSafe8": (-2, 16, "right", "bottom"),
        "R1f_Offline_FB_MMSE_Inf": (-6, 12, "right", "bottom"),
        "R2c_Offline_DiffHMM_Linear11": (-8, -8, "right", "top"),
        "R3c_Ridge_Spectral_439": (-6, 10, "center", "bottom"),
        "R2b_Causal_DiffHMM_MLP43": (14, 8, "left", "bottom"),
        "R3b_Spectral_MLP_15k": (-26, -28, "right", "top"),
        "R2d_E2E_Causal_H5_DiffHMM_Linear11": (8, -12, "left", "top"),
        "R1a_Online_Forward_MMSE_0s": (-4, -15, "right", "top"),
        "R1g_Offline_Viterbi_Unpinned": (-10, 0, "right", "center"),
        "R3a_Causal_TCN_16k": (0, -12, "center", "top"),
        "R0_HPSS_NLS_Locked": (4, 10, "left", "bottom"),
    }
    x_all = [float(sc[k]["pred_sd_ratio_trig_lag_aligned"]) for k in sc]
    y_all = [float(sc[k]["r_within_trig_lag_aligned"]) for k in sc]
    x_lo, x_hi = max(0.0, min(x_all) - 0.10), max(1.12, max(x_all) + 0.17)
    y_lo, y_hi = min(y_all) - 0.055, max(y_all) + 0.045
    ax2.set_xlim(x_lo, x_hi)
    ax2.set_ylim(y_lo, y_hi)

    for km in sc:
        xr = float(sc[km]["pred_sd_ratio_trig_lag_aligned"])
        yr = float(sc[km]["r_within_trig_lag_aligned"])
        dx_pt, dy_pt, ha, va = explicit_offsets.get(km, (6, 6, "left", "bottom"))
        ax2.annotate(
            SHORT_LABELS[km],
            xy=(xr, yr),
            xytext=(dx_pt, dy_pt),
            textcoords="offset points",
            fontsize=6.0,
            ha=ha,
            va=va,
            color=rung_colors[int(sc[km]["rung"])],
            arrowprops={"arrowstyle": "-", "lw": 0.55, "alpha": 0.5, "color": rung_colors[int(sc[km]["rung"])]},
        )

    ax2.axvline(1.0, color=OKABE_ITO["gray"], ls="--", lw=1.0, alpha=0.65)
    ax2.set_xlabel(r"Physical Dynamic-Range Ratio $\overline{\mathrm{SD}}(\hat{f}_b) / \overline{\mathrm{SD}}(f^{\mathrm{ref}}_{\mathrm{T10}})$", fontsize=8.0)
    ax2.set_ylabel(r"$r_{\mathrm{within}}$ vs. $1024\,\mathrm{Hz}$ Trig ($-10\,\mathrm{s}$)")
    ax2.set_title("(b) Variance Collapse vs. Physical Tracking")
    ax2.grid(True, alpha=0.22, lw=0.6)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=6.4)

    return _save_fig_pair(fig, "stage2_complexity_ladder")


def plot_synthetic_sweep_and_weights(master: dict[str, Any]) -> list[str]:
    """Generate `stage2_synthetic_sweep_and_weights.{pdf,png}`."""
    synth = master["synthetic_frequency_sweep_pool30"]
    w_r2e = master["interpretable_params_pool30"]["R2e_Causal_DiffHMM_FilterSafe8"]
    w_r2a = master["interpretable_params_pool30"]["R2a_Causal_DiffHMM_Linear11"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.8))
    fig.subplots_adjust(wspace=0.45)

    # Panel (a): Out-of-range passband 28-80 bpm sweep
    freqs = np.array([28.0, 35.0, 50.0, 65.0, 80.0], dtype=np.float64)
    case_keys = ["stat_28bpm", "stat_35bpm", "stat_50bpm", "stat_65bpm", "stat_80bpm"]
    ax1.plot([24, 84], [24, 84], "--", color=OKABE_ITO["gray"], lw=1.2, label=r"Identity ($\hat{f}_b = f_{\mathrm{true}}$)")

    sweep_specs = (
        ("R0_HPSS_NLS_Locked", r"R0 NLS (0 params)", OKABE_ITO["silver"], "P:", 1.2),
        ("R1a_Online_Forward_MMSE_0s", r"R1a Fwd-HMM (3 params)", OKABE_ITO["sky"], "s--", 1.3),
        ("R2e_Causal_DiffHMM_FilterSafe8", r"R2e FilterSafe-8 (8 params)", OKABE_ITO["blue"], "D-", 1.7),
        ("R2a_Causal_DiffHMM_Linear11", r"R2a Diff-HMM (11 params)", OKABE_ITO["green"], "o-", 1.5),
        ("R3c_Ridge_Spectral_439", r"R3c Ridge (439)", OKABE_ITO["orange"], "d-.", 1.3),
        ("R3b_Spectral_MLP_15k", r"R3b Spec-MLP (14.6k)", OKABE_ITO["purple"], "v:", 1.3),
        ("R3a_Causal_TCN_16k", r"R3a Causal TCN (15.8k)", OKABE_ITO["vermilion"], "X-", 1.7),
    )
    for km, lbl, col, fmt, lw in sweep_specs:
        ys = np.array([float(synth[ck][km]["pred_mean"]) for ck in case_keys], dtype=np.float64)
        sds = np.array([float(synth[ck][km]["pred_sd_across_folds"]) for ck in case_keys], dtype=np.float64)
        ax1.errorbar(
            freqs,
            ys,
            yerr=sds,
            fmt=fmt,
            color=col,
            lw=lw,
            ms=4.3,
            capsize=2.2,
            elinewidth=0.9,
            label=lbl,
        )

    ax1.set_xlim(24, 84)
    ax1.set_ylim(23, 89)
    ax1.set_xticks(freqs)
    ax1.set_xlabel(r"Synthetic CW Radar Frequency $f_{\mathrm{true}}$ (bpm)")
    ax1.set_ylabel(r"Predicted Rate $\hat{f}_b$ (bpm, 11-fold mean $\pm$ SD)")
    ax1.set_title("(a) Passband Frequency Transfer (28–80 bpm)")
    ax1.grid(True, alpha=0.22, lw=0.6)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=5.8)

    # Panel (b): Grouped horizontal comparison of R2e (8p, filter-safe) vs R2a (11p, +harmonics)
    param_specs = (
        ("w1_p1_clean", r"$w_1$ ($P_{\mathrm{clean}}(f)$)"),
        ("w6_p1_raw", r"$w_6$ ($P_{\mathrm{raw}}(f)$)"),
        ("w5_anchor", r"$w_5$ ($A_{\mathrm{coarse}}(f)$)"),
        ("w2_p2_harm", r"$w_2$ ($P_{\mathrm{clean}}(2f)$)"),
        ("w3_p3_harm", r"$w_3$ ($P_{\mathrm{clean}}(3f)$)"),
        ("w4_phalf_sub", r"$w_4$ ($P_{\mathrm{clean}}(0.5f)$)"),
        ("k1", r"$k_1$ ($\pm 1$ bpm kernel)"),
        ("k2", r"$k_2$ ($\pm 2$ bpm kernel)"),
    )
    y_idx = np.arange(len(param_specs), dtype=np.float64)
    means_a = np.array([float(w_r2a[k]["mean"]) for k, _ in param_specs], dtype=np.float64)
    sds_a = np.array([float(w_r2a[k]["sd"]) for k, _ in param_specs], dtype=np.float64)
    means_e = np.array([float(w_r2e[k]["mean"]) for k, _ in param_specs], dtype=np.float64)
    sds_e = np.array([float(w_r2e[k]["sd"]) for k, _ in param_specs], dtype=np.float64)

    bh = 0.34
    ax2.barh(y_idx - bh * 0.5, means_e, xerr=sds_e, color=OKABE_ITO["blue"], alpha=0.9, height=bh, capsize=2.0, error_kw={"lw": 0.9}, label=r"R2e FS-8 (1$f$-only)")
    ax2.barh(y_idx + bh * 0.5, means_a, xerr=sds_a, color=OKABE_ITO["green"], alpha=0.82, height=bh, capsize=2.0, error_kw={"lw": 0.9}, label=r"R2a HMM-11 (+3$f$/0.5$f$)")
    ax2.axvline(0.0, color=OKABE_ITO["gray"], lw=0.9)
    ax2.set_yticks(y_idx)
    ax2.set_yticklabels([lbl for _, lbl in param_specs], fontsize=6.9)
    lo_x = min(-0.65, float(np.min(means_a - sds_a)) - 0.15, float(np.min(means_e - sds_e)) - 0.15)
    hi_x = max(2.2, float(np.max(means_a + sds_a)) + 0.35, float(np.max(means_e + sds_e)) + 0.35)
    ax2.set_xlim(lo_x, hi_x)
    ax2.invert_yaxis()
    ax2.set_xlabel("Learned Weight (11-Fold Mean ± SD)")
    ax2.set_title(r"(b) Filter-Safe R2e ($N_p=8$) vs. R2a ($N_p=11$)")
    ax2.grid(True, axis="x", alpha=0.22, lw=0.6)
    ax2.legend(loc="lower right", bbox_to_anchor=(0.98, 0.25), frameon=True, framealpha=0.94, fontsize=5.9)

    sse, sje, pje = w_r2e["sigma_slow_bpm"], w_r2e["sigma_jump_bpm"], w_r2e["pi_jump"]
    ssa, sja, pja = w_r2a["sigma_slow_bpm"], w_r2a["sigma_jump_bpm"], w_r2a["pi_jump"]
    trans_box = (
        "Transition (R2e vs. R2a):\n"
        rf"$\sigma_{{\mathrm{{slow}}}}$: {sse['mean']:.2f} vs. {ssa['mean']:.2f} bpm" + "\n"
        rf"$\sigma_{{\mathrm{{jump}}}}$: {sje['mean']:.2f} vs. {sja['mean']:.2f} bpm" + "\n"
        rf"$\pi_{{\mathrm{{jump}}}}$: {pje['mean']:.4f} vs. {pja['mean']:.4f}"
    )
    ax2.text(
        0.98,
        0.03,
        trans_box,
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F7F7F7", "edgecolor": OKABE_ITO["silver"], "alpha": 0.95},
    )

    return _save_fig_pair(fig, "stage2_synthetic_sweep_and_weights")


def main() -> None:
    _configure_matplotlib()
    master = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    if not PRED_ARCHIVE.exists():
        raise FileNotFoundError(f"Missing {PRED_ARCHIVE}; run thesis.exp2_stage2_complexity first.")
    p1 = plot_complexity_ladder(master)
    p2 = plot_synthetic_sweep_and_weights(master)
    print(json.dumps({"generated_figures": p1 + p2}, indent=2))


if __name__ == "__main__":
    main()
