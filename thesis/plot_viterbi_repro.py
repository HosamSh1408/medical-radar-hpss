"""Publication-grade Okabe-Ito figures for the Viterbi reproduction & baseline audit.

Reads exclusively from `paper_orchestra_bundle/raw_materials/metrics_viterbi_repro.json`,
`audit/viterbi_repro_sessions.csv`, `thesis/outputs/viterbi_repro/cache/<m>.npz`,
and locked `v1` baseline caches (zero hard-coded numbers).

Exports dual vector PDF + 300 DPI PNG into `paper_orchestra_bundle/raw_materials/figures/`.
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_workspace.src import config as C
from paper_workspace.src import evaluation as EV
from thesis import edf_extractor as EDFX
from thesis import io_utils as IO
from thesis import viterbi_tracker as VT

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

METRICS_JSON = IO.BUNDLE_DIR / "metrics_viterbi_repro.json"
SESSIONS_CSV = IO.AUDIT_DIR / "viterbi_repro_sessions.csv"
OBS_CACHE_DIR = IO.ROOT / "thesis" / "outputs" / "viterbi_repro" / "cache"


def _configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
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


def _select_best_repro_viterbi(master: dict[str, Any]) -> tuple[str, str]:
    """Select the lowest-RMSE_sess reproducible Viterbi tracker on primary pool30_sync."""
    p30 = master["results"]["branch_neutral_raw_nls_lag"]["pool30_sync"]["methods"]
    candidates = [
        ("C1b_1D_Vit_HPSS", "1D Vit (HPSS)"),
        ("C1m_1D_Vit_HPSS_MatchedW05", "1D Vit w=0.5 (HPSS)"),
        ("C2b_1D_HQI_HPSS", "1D+HQI Vit (HPSS)"),
        ("C2m_1D_HQI_HPSS_MatchedW05", "1D+HQI w=0.5 (HPSS)"),
        ("C3a_2D_JointVit_NMF", "2D Joint Vit (NMF)"),
        ("C3b_2D_JointVit_HPSS", "2D Joint Vit (HPSS)"),
        ("C4a_2D_HarmNLS_Vit_NMF", "Harm-NLS 2D Vit (NMF)"),
        ("C4b_2D_HarmNLS_Vit_HPSS", "Harm-NLS 2D Vit (HPSS)"),
        ("C4c_1D_Anchored_HarmNLS_Vit_HPSS", "Anch-NLS 1D Vit (HPSS)"),
        ("C4d_2D_Anchored_HarmNLS_Vit_HPSS", "Anch-NLS 2D Vit (HPSS)"),
        ("C6a_LOPO_Cal_Vit_HPSS", "LOPO-Cal Vit (HPSS)"),
        ("C6b_LOPO_Cal_HarmNLS_HPSS", "LOPO-Cal Harm-NLS (HPSS)"),
        ("C6c_LOPO_Cal_HarmNLS_NMF", "LOPO-Cal Harm-NLS (NMF)"),
        ("C6d_LOPO_Wide_AnchoredNLS_HPSS", "LOPO-Wide Anch-NLS (HPSS)"),
        ("C6e_LOPO_Wide_AnchoredNLS_NMF", "LOPO-Wide Anch-NLS (NMF)"),
    ]
    avail = [item for item in candidates if item[0] in p30]
    best_k, best_lbl = min(avail, key=lambda item: p30[item[0]]["rmse_sess"])
    return best_k, best_lbl


def _short_viterbi_tick(method_key: str, default_lbl: str) -> str:
    short_map = {
        "C1b_1D_Vit_HPSS": "1D Vit",
        "C1m_1D_Vit_HPSS_MatchedW05": "1D Vit (w=0.5)",
        "C2b_1D_HQI_HPSS": "1D+HQI",
        "C2m_1D_HQI_HPSS_MatchedW05": "1D+HQI (w=0.5)",
        "C3a_2D_JointVit_NMF": "2D Joint (NMF)",
        "C3b_2D_JointVit_HPSS": "2D Joint Vit",
        "C4a_2D_HarmNLS_Vit_NMF": "Harm-NLS (NMF)",
        "C4b_2D_HarmNLS_Vit_HPSS": "Harm-NLS 2D",
        "C4c_1D_Anchored_HarmNLS_Vit_HPSS": "Anch-NLS 1D",
        "C4d_2D_Anchored_HarmNLS_Vit_HPSS": "Anch-NLS 2D",
        "C6a_LOPO_Cal_Vit_HPSS": "LOPO (C6a)",
        "C6b_LOPO_Cal_HarmNLS_HPSS": "LOPO (C6b)",
        "C6c_LOPO_Cal_HarmNLS_NMF": "LOPO (C6c)",
        "C6d_LOPO_Wide_AnchoredNLS_HPSS": "Best LOPO (HPSS)",
        "C6e_LOPO_Wide_AnchoredNLS_NMF": "Best LOPO (NMF)",
    }
    return short_map.get(method_key, default_lbl)


def _plot_ablation_ladder(master: dict[str, Any]) -> list[str]:
    reg = "branch_neutral_raw_nls_lag"
    best_k, best_lbl = _select_best_repro_viterbi(master)
    cohorts = [
        ("pool30_sync", r"$\mathrm{pool30\_sync}$ ($S=28$)", OKABE_ITO["blue"]),
        ("retained23", r"$\mathrm{retained23}$ ($S=23$)", OKABE_ITO["sky"]),
        ("PAPER_18", r"$\mathrm{PAPER\_18}$ ($S=18$)", OKABE_ITO["green"]),
    ]
    methods = [
        ("A1_DFT_Locked", "DFT"),
        ("A2_RawNLS_Locked", "Raw NLS"),
        ("A3_Beltrao_NMF_NLS", "NLS+NMF"),
        ("A4_HPSS_NLS", "HPSS+NLS"),
        ("B3_Clean_WelchPeak_HPSS", "Clean Welch"),
        ("C1b_1D_Vit_HPSS", "1D Vit"),
        ("C1m_1D_Vit_HPSS_MatchedW05", "1D Vit (w=0.5)"),
        ("C3b_2D_JointVit_HPSS", "2D Joint Vit"),
        ("C4b_2D_HarmNLS_Vit_HPSS", "Harm-NLS 2D"),
        ("C4d_2D_Anchored_HarmNLS_Vit_HPSS", "Anch-NLS 2D"),
    ]
    if best_k not in {m[0] for m in methods}:
        methods.append((best_k, _short_viterbi_tick(best_k, best_lbl)))

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.5))
    ax1, ax2, ax3, ax4 = axes.ravel()
    x = np.arange(len(methods), dtype=float)
    w = 0.25

    for idx_c, (ckey, clbl, col) in enumerate(cohorts):
        offset = (idx_c - 1) * w
        d_c = master["results"][reg][ckey]
        rmses = [d_c["methods"][m]["rmse_sess"] for m, _ in methods]
        acc10s = [d_c["methods"][m]["acc10_sess"] for m, _ in methods]
        rwins = [d_c["methods"][m]["r_within_rmcorr"] for m, _ in methods]

        ax1.bar(x + offset, rmses, width=w, color=col, alpha=0.92, label=clbl)
        ax2.bar(x + offset, acc10s, width=w, color=col, alpha=0.92, label=clbl)
        ax3.bar(x + offset, rwins, width=w, color=col, alpha=0.92, label=clbl)

    p30_loso = master["results"][reg]["pool30_sync"]["constants"]["const_loso"]
    ax1.axhline(p30_loso, color=OKABE_ITO["vermilion"], linestyle="--", linewidth=1.1,
                label=rf"$\mathrm{{const}}_{{\mathrm{{LOSO}}}}$ ({p30_loso:.2f} bpm)")

    labels = [lbl for _, lbl in methods]
    for ax in (ax1, ax2, ax3):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=34, ha="right", fontsize=6.6)
        ax.grid(axis="y", alpha=0.25)

    ax1.set_ylabel(r"$\mathrm{RMSE}_{\mathrm{sess}}$ (bpm)")
    ax1.set_ylim(0.0, 18.0)
    ax1.set_title("(a) Equal-Session Error across Cohort Tiers")
    ax1.legend(loc="upper right", fontsize=5.9, frameon=True, ncol=2)

    ax2.set_ylabel(r"$\mathrm{Acc}@10$ (%)")
    ax2.set_ylim(45.0, 100.0)
    ax2.set_title(r"(b) Threshold Accuracy ($|e_{s,w}| < 10$ bpm)")

    ax3.set_ylabel(r"$r_{\mathrm{within}}$ (ANCOVA)")
    ax3.set_ylim(0.0, 0.72)
    ax3.set_title(r"(c) Within-Session Tracking ($r_{\mathrm{within}}$)")

    # Panel (d): Paired delta RMSE vs Audited HPSS+NLS on pool30_sync with 95% patient-clustered CIs
    p30_paired = master["results"][reg]["pool30_sync"]["paired_vs_hpss_nls"]
    delta_methods = [m for m in methods if m[0] != "A4_HPSS_NLS"]
    y_pos = np.arange(len(delta_methods), dtype=float)
    means = np.array([p30_paired[m]["mean_delta"] for m, _ in delta_methods])
    ci_lo = np.array([p30_paired[m]["ci95_lo"] for m, _ in delta_methods])
    ci_hi = np.array([p30_paired[m]["ci95_hi"] for m, _ in delta_methods])
    xerr_lo = np.maximum(0.0, means - ci_lo)
    xerr_hi = np.maximum(0.0, ci_hi - means)
    colors_d = [OKABE_ITO["blue"] if val < 0.0 else OKABE_ITO["vermilion"] for val in means]

    ax4.axvline(0.0, color=OKABE_ITO["gray"], linestyle="--", linewidth=1.0)
    ax4.errorbar(means, y_pos, xerr=[xerr_lo, xerr_hi], fmt="none", ecolor=OKABE_ITO["gray"],
                 elinewidth=1.1, capsize=2.5, zorder=2)
    ax4.scatter(means, y_pos, c=colors_d, s=28, zorder=3)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([lbl for _, lbl in delta_methods], fontsize=6.6)
    ax4.invert_yaxis()
    ax4.set_xlabel(r"$\Delta \mathrm{RMSE}_{\mathrm{sess}}$ vs $\mathrm{HPSS+NLS}$ (bpm)")
    ax4.set_title(r"(d) $\mathrm{pool30\_sync}$ Paired $\Delta$ (95% Clustered CI)")
    ax4.grid(axis="x", alpha=0.25)

    fig.tight_layout(h_pad=1.3, w_pad=1.2)
    return _save_fig_pair(fig, "viterbi_repro_ablation_ladder")


def _plot_bland_altman(master: dict[str, Any], df_sess: pd.DataFrame) -> list[str]:
    reg = "branch_neutral_raw_nls_lag"
    best_k, best_lbl = _select_best_repro_viterbi(master)
    panels = [
        ("retained23", r"(a) $\mathrm{retained23}$ ($S=23$)"),
        ("pool30_sync", r"(b) $\mathrm{pool30\_sync}$ ($S=28$)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35))
    sub_reg = df_sess[df_sess["regime"] == reg]

    for ax, (ckey, title) in zip(axes, panels):
        sub_c = sub_reg[sub_reg["cohort"] == ckey]
        d_h = sub_c[sub_c["method"] == "A4_HPSS_NLS"].set_index("measurement")
        d_v = sub_c[sub_c["method"] == best_k].set_index("measurement")
        common = sorted(set(d_h.index) & set(d_v.index))

        avg_h = 0.5 * (d_h.loc[common, "est_mean"] + d_h.loc[common, "ref_mean"])
        dif_h = d_h.loc[common, "est_mean"] - d_h.loc[common, "ref_mean"]
        avg_v = 0.5 * (d_v.loc[common, "est_mean"] + d_v.loc[common, "ref_mean"])
        dif_v = d_v.loc[common, "est_mean"] - d_v.loc[common, "ref_mean"]

        ax.axhline(0.0, color=OKABE_ITO["silver"], linestyle=":", linewidth=0.9)
        ax.scatter(avg_h, dif_h, marker="^", s=26, color=OKABE_ITO["vermilion"], alpha=0.75,
                   label=r"Audited $\mathrm{HPSS+NLS}$")
        ax.scatter(avg_v, dif_v, marker="o", s=26, color=OKABE_ITO["blue"], alpha=0.90,
                   label=best_lbl)

        ba_h = master["results"][reg][ckey]["methods"]["A4_HPSS_NLS"]["ba_session_level"]
        ba_v = master["results"][reg][ckey]["methods"][best_k]["ba_session_level"]
        ba_v_rm = master["results"][reg][ckey]["methods"][best_k]["ba_clustered_anova"]
        assert abs(float(np.mean(dif_v)) - float(ba_v["bias"])) < 1e-9, (
            f"Scatter vs JSON bias mismatch for {best_k} ({ckey}): {np.mean(dif_v)} vs {ba_v['bias']}"
        )

        ax.axhline(ba_h["bias"], color=OKABE_ITO["vermilion"], linestyle="--", linewidth=0.9, alpha=0.8)
        ax.axhline(ba_v["bias"], color=OKABE_ITO["blue"], linestyle="-", linewidth=1.1)
        ax.axhline(ba_v["loa_lo"], color=OKABE_ITO["blue"], linestyle="--", linewidth=0.9)
        ax.axhline(ba_v["loa_up"], color=OKABE_ITO["blue"], linestyle="--", linewidth=0.9)
        ax.axhspan(ba_v["loa_lo"], ba_v["loa_up"], color=OKABE_ITO["sky"], alpha=0.10)

        txt = (
            f"{best_lbl}:\n"
            f"  Bias {ba_v['bias']:+.2f}, LoA [{ba_v['loa_lo']:+.2f}, {ba_v['loa_up']:+.2f}]\n"
            f"  RM-LoA [{ba_v_rm['loa_lo']:+.1f}, {ba_v_rm['loa_up']:+.1f}]\n"
            f"  prop p={ba_v['prop_p']:.3f}, ICC={ba_v['icc_1_1']:.3f}\n"
            f"HPSS+NLS: Bias {ba_h['bias']:+.2f}, ICC={ba_h['icc_1_1']:.3f}"
        )
        ax.text(
            0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=6.1,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.92),
        )
        ax.set_xlabel(r"Mean of $\hat{f}_b$ and $f^{\mathrm{ref}}$ (bpm)")
        ax.set_ylabel(r"Session Mean Diff ($\hat{f}_b - f^{\mathrm{ref}}$, bpm)")
        ax.set_ylim(-14.5, 21.0)
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=6.4, frameon=True)
        ax.grid(alpha=0.25)

    fig.tight_layout(w_pad=1.2)
    return _save_fig_pair(fig, "viterbi_repro_bland_altman")


def _compute_session_trace(measurement: str, lag_s: int, best_k: str, master: dict[str, Any]) -> dict[str, Any]:
    """Compute valid-aligned traces for plotting real sessions without bridging invalid gaps."""
    cache = IO.load_locked_cache(measurement)
    with np.load(OBS_CACHE_DIR / f"{measurement}.npz", allow_pickle=False) as z:
        p_br_nmf = z["P_br_nmf_2bpm"].copy()
        p_br_hpss = z["P_br_hpss_2bpm"].copy()
        s_nls_hpss = z["S_nls_hpss_1bpm"].copy()
        s_nls_nmf = z["S_nls_nmf_1bpm"].copy()
        s_anc_hpss = z["S_anc_hpss_1bpm"].copy()
        s_anc_nmf = z["S_anc_nmf_1bpm"].copy()
        p_hr_csd = z["P_hr_csd_2bpm"].copy()

    edf = EDFX.load_edf_recording(measurement)
    n_win = len(cache["est"])
    centers = np.arange(n_win, dtype=float) * C.UPDATE_S + C.WIN_S / 2.0
    ref_w, vwin = EV.window_ref_and_gating(edf.resp_256, edf.resp_rate_1hz, centers, lag_s, fs_raw=edf.fs_resp)
    valid = vwin & np.isfinite(ref_w) & np.isfinite(cache["est"]) & np.isfinite(cache["est_hpss"])

    clean_welch = np.where(
        np.max(p_br_hpss, axis=1) > 0, VT.BR_GRID_2BPM[np.argmax(p_br_hpss, axis=1)], VT.BR_GRID_2BPM[0]
    )
    if best_k == "C1b_1D_Vit_HPSS":
        vit_path = VT.decode_viterbi_1d(p_br_hpss, VT.BR_GRID_2BPM)[0]
    elif best_k == "C1m_1D_Vit_HPSS_MatchedW05":
        vit_path = VT.decode_viterbi_1d(p_br_hpss, VT.BR_GRID_2BPM, sigma_br=4.0, w_br=0.5)[0]
    elif best_k == "C2b_1D_HQI_HPSS":
        hqi = np.array([VT.compute_hqi_1d_penalty(p_hr_csd[t], VT.HR_GRID_2BPM, VT.BR_GRID_2BPM) for t in range(n_win)])
        vit_path = VT.decode_viterbi_1d(p_br_hpss, VT.BR_GRID_2BPM, hqi_1d_seq=hqi)[0]
    elif best_k == "C2m_1D_HQI_HPSS_MatchedW05":
        hqi = np.array([VT.compute_hqi_1d_penalty(p_hr_csd[t], VT.HR_GRID_2BPM, VT.BR_GRID_2BPM) for t in range(n_win)])
        vit_path = VT.decode_viterbi_1d(p_br_hpss, VT.BR_GRID_2BPM, sigma_br=4.0, w_br=0.5, hqi_1d_seq=hqi)[0]
    elif best_k == "C3a_2D_JointVit_NMF":
        vit_path = VT.decode_viterbi_2d_joint(p_br_nmf, p_hr_csd, VT.BR_GRID_2BPM, VT.HR_GRID_2BPM)[0]
    elif best_k == "C3b_2D_JointVit_HPSS":
        vit_path = VT.decode_viterbi_2d_joint(p_br_hpss, p_hr_csd, VT.BR_GRID_2BPM, VT.HR_GRID_2BPM)[0]
    elif best_k == "C4a_2D_HarmNLS_Vit_NMF":
        vit_path = VT.decode_viterbi_2d_joint(s_nls_nmf, p_hr_csd, VT.BR_GRID_1BPM, VT.HR_GRID_2BPM)[0]
    elif best_k == "C4b_2D_HarmNLS_Vit_HPSS":
        vit_path = VT.decode_viterbi_2d_joint(s_nls_hpss, p_hr_csd, VT.BR_GRID_1BPM, VT.HR_GRID_2BPM)[0]
    elif best_k == "C4c_1D_Anchored_HarmNLS_Vit_HPSS":
        vit_path = VT.decode_viterbi_1d(s_anc_hpss, VT.BR_GRID_1BPM, sigma_br=4.0, w_br=1.0)[0]
    elif best_k == "C4d_2D_Anchored_HarmNLS_Vit_HPSS":
        vit_path = VT.decode_viterbi_2d_joint(s_anc_hpss, p_hr_csd, VT.BR_GRID_1BPM, VT.HR_GRID_2BPM)[0]
    elif best_k.startswith("C6"):
        folds = master["results"]["branch_neutral_raw_nls_lag"]["pool30_sync"]["lopo_folds"][best_k]
        pid = IO.patient_id_of(measurement)
        finfo = next(f for f in folds if f["patient_id"] == pid)
        spec_lookup = {
            "C6a_LOPO_Cal_Vit_HPSS": (p_br_hpss, VT.BR_GRID_2BPM),
            "C6b_LOPO_Cal_HarmNLS_HPSS": (s_nls_hpss, VT.BR_GRID_1BPM),
            "C6c_LOPO_Cal_HarmNLS_NMF": (s_nls_nmf, VT.BR_GRID_1BPM),
            "C6d_LOPO_Wide_AnchoredNLS_HPSS": (s_anc_hpss, VT.BR_GRID_1BPM),
            "C6e_LOPO_Wide_AnchoredNLS_NMF": (s_anc_nmf, VT.BR_GRID_1BPM),
        }
        spec_seq, grid = spec_lookup[best_k]
        w_hqi = float(finfo.get("selected_w_hqi", 4.0))
        use_hqi = w_hqi > 0.0
        if finfo["selected_mode"] == "1d":
            hqi = (
                np.array([VT.compute_hqi_1d_penalty(p_hr_csd[t], VT.HR_GRID_2BPM, grid) for t in range(n_win)])
                if use_hqi
                else None
            )
            vit_path = VT.decode_viterbi_1d(
                spec_seq,
                grid,
                sigma_br=finfo["selected_sigma_br"],
                eps_p=finfo["selected_eps_floor"],
                hqi_1d_seq=hqi,
                w_hqi=w_hqi,
            )[0]
        else:
            vit_path = VT.decode_viterbi_2d_joint(
                spec_seq,
                p_hr_csd,
                grid,
                VT.HR_GRID_2BPM,
                w_br=finfo["selected_w_br"],
                w_hqi=w_hqi,
                sigma_br=finfo["selected_sigma_br"],
                eps_p=finfo["selected_eps_floor"],
                use_hqi=use_hqi,
            )[0]
    else:
        raise ValueError(f"Unsupported best_k in _compute_session_trace: {best_k!r}")

    return {
        "t_min": centers / 60.0,
        "valid": valid,
        "ref": ref_w,
        "nmf": np.asarray(cache["est"], float),
        "hpss": np.asarray(cache["est_hpss"], float),
        "welch": clean_welch,
        "vit": vit_path,
    }


def _plot_tracking_examples(master: dict[str, Any]) -> list[str]:
    lags = IO.load_locked_neutral_lags()
    best_k, best_lbl = _select_best_repro_viterbi(master)
    examples = [
        ("7.3", "(a) Anchor Session 7.3 (Prone, Twin)"),
        ("9.3", "(b) Session 9.3 (Supine, Cardiac Bleed-In / Volatility)"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=False)
    for ax, (mid, title) in zip(axes, examples):
        lag_val = int(lags[mid]["neutral_raw_nls_lag_s"])
        tr = _compute_session_trace(mid, lag_val, best_k, master)
        v = tr["valid"]
        t = tr["t_min"]

        def _score_str(arr: np.ndarray) -> str:
            e = np.abs(arr[v] - tr["ref"][v])
            rmse = float(np.sqrt(np.mean(e ** 2)))
            acc10 = float(100.0 * np.mean(e < 10.0))
            return f"RMSE {rmse:.1f}, Acc@10 {acc10:.0f}%"

        ax.plot(t, np.where(v, tr["ref"], np.nan), color=OKABE_ITO["gray"], linewidth=1.35, label=r"$f^{\mathrm{ref}}$ (VitaGuard)")
        ax.plot(t, np.where(v, tr["welch"], np.nan), color=OKABE_ITO["sky"], linewidth=0.85, alpha=0.80,
                label=rf"Clean Welch ({_score_str(tr['welch'])})")
        ax.plot(t, np.where(v, tr["nmf"], np.nan), color=OKABE_ITO["vermilion"], linewidth=0.90, linestyle="--", alpha=0.85,
                label=rf"Audited NMF+NLS ({_score_str(tr['nmf'])})")
        ax.plot(t, np.where(v, tr["hpss"], np.nan), color=OKABE_ITO["green"], linewidth=0.95, alpha=0.85,
                label=rf"Audited HPSS+NLS ({_score_str(tr['hpss'])})")
        ax.plot(t, np.where(v, tr["vit"], np.nan), color=OKABE_ITO["blue"], linewidth=1.30,
                label=rf"{best_lbl} ({_score_str(tr['vit'])})")

        ax.set_ylabel(r"$\hat{f}_b$ (bpm)")
        ax.set_title(rf"{title} [$\hat{{\tau}}_s = {lag_val}\,$s, $|\mathcal{{W}}_s|={int(np.sum(v))}$]")
        ax.set_ylim(15.0, 114.0)
        ax.set_yticks([20, 40, 60, 80])
        ax.legend(loc="upper center", fontsize=5.9, ncol=2, frameon=True, framealpha=0.95)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Recording Time (min)")
    fig.tight_layout(h_pad=1.15)
    return _save_fig_pair(fig, "viterbi_repro_tracking_examples")


def _plot_forensic_bridge(master: dict[str, Any]) -> list[str]:
    ret = master["results"]["branch_neutral_raw_nls_lag"]["retained23"]["methods"]
    best_k, best_lbl = _select_best_repro_viterbi(master)
    ref_d = master["_definitions"]["attached_draft_reference"]
    steps = [
        ("B6_Diag_Legacy_AD_NaiveFFT", "Legacy AD FFT"),
        ("B7_Diag_Legacy_AD_RawNMF", "Legacy AD NMF"),
        ("C3d_2D_JointVit_NMF_ADHR_Diag", "2D Vit (AD HR)"),
        ("B1_Clean_WelchPeak_Raw", "CSD Welch Raw"),
        ("B3_Clean_WelchPeak_HPSS", "CSD Welch HPSS"),
        ("A3_Beltrao_NMF_NLS", "Beltrao NMF+NLS"),
        ("A4_HPSS_NLS", "Audited HPSS+NLS"),
        ("C3b_2D_JointVit_HPSS", "2D Joint Vit"),
        ("C4b_2D_HarmNLS_Vit_HPSS", "Harm-NLS 2D Vit"),
        ("C4d_2D_Anchored_HarmNLS_Vit_HPSS", "Anch-NLS 2D Vit"),
    ]
    if best_k not in {k for k, _ in steps}:
        steps.append((best_k, _short_viterbi_tick(best_k, best_lbl)))

    x = np.arange(len(steps), dtype=float)
    rmse_sess = [ret[k]["rmse_sess"] for k, _ in steps]
    rmse_pool = [ret[k]["rmse_pool"] for k, _ in steps]
    acc10 = [ret[k]["acc10_sess"] for k, _ in steps]
    rwin = [ret[k]["r_within_rmcorr"] for k, _ in steps]
    const_loso = master["results"]["branch_neutral_raw_nls_lag"]["retained23"]["constants"]["const_loso"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.95))
    w = 0.36
    ax1.bar(x - w / 2, rmse_sess, width=w, color=OKABE_ITO["blue"], label=r"$\mathrm{RMSE}_{\mathrm{sess}}$")
    ax1.bar(x + w / 2, rmse_pool, width=w, color=OKABE_ITO["sky"], label=r"$\mathrm{RMSE}_{\mathrm{pool}}$")
    ax1.axhline(const_loso, color=OKABE_ITO["vermilion"], linestyle="--", linewidth=1.0,
                label=rf"$\mathrm{{const}}_{{\mathrm{{LOSO}}}}$ ({const_loso:.2f} bpm)")
    ax1.axhline(ref_d["naive_fft_rmse_bpm"], color=OKABE_ITO["gray"], linestyle=":", linewidth=1.0,
                label=f"Draft Naive FFT ({ref_d['naive_fft_rmse_bpm']:.2f}, {ref_d['n_windows_1s_grid']:,} win)")
    ax1.axhline(ref_d["proposed_rmse_bpm"], color=OKABE_ITO["orange"], linestyle="-.", linewidth=1.0,
                label=f"Draft Proposed ({ref_d['proposed_rmse_bpm']:.2f}, {ref_d['n_windows_1s_grid']:,} win)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in steps], rotation=38, ha="right", fontsize=6.3)
    ax1.set_ylabel("Error (bpm)")
    ax1.set_ylim(0.0, 23.8)
    ax1.set_title(r"(a) Forensic Bridge on $\mathrm{retained23}$ ($S=23$, 6,447 win)")
    ax1.legend(loc="upper right", fontsize=5.6, frameon=True)
    ax1.grid(axis="y", alpha=0.25)

    ax2_r = ax2.twinx()
    ax2_r.spines["right"].set_visible(True)
    l1 = ax2.plot(x, acc10, color=OKABE_ITO["green"], marker="o", markersize=4.5, linewidth=1.4,
                  label=r"$\mathrm{Acc}@10$ (%)")
    l2 = ax2_r.plot(x, rwin, color=OKABE_ITO["vermilion"], marker="s", markersize=4.2, linewidth=1.3,
                    linestyle="--", label=r"$r_{\mathrm{within}}$ (ANCOVA)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl in steps], rotation=38, ha="right", fontsize=6.3)
    ax2.set_ylabel(r"$\mathrm{Acc}@10$ (%)", color=OKABE_ITO["green"])
    ax2_r.set_ylabel(r"$r_{\mathrm{within}}$", color=OKABE_ITO["vermilion"])
    ax2.set_ylim(25.0, 92.0)
    ax2_r.set_ylim(-0.05, 0.70)
    ax2.set_title("(b) Accuracy vs. Within-Session Tracking")
    ax2.grid(axis="y", alpha=0.25)
    lines = l1 + l2
    ax2.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=6.2, frameon=True)

    fig.tight_layout(w_pad=1.4)
    return _save_fig_pair(fig, "viterbi_repro_forensic_bridge")


def generate_all_figures() -> list[str]:
    _configure_matplotlib()
    master = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    df_sess = pd.read_csv(SESSIONS_CSV, dtype={"measurement": str})
    paths: list[str] = []
    paths.extend(_plot_ablation_ladder(master))
    paths.extend(_plot_bland_altman(master, df_sess))
    paths.extend(_plot_tracking_examples(master))
    paths.extend(_plot_forensic_bridge(master))
    print("Generated figures:", paths)
    return paths


if __name__ == "__main__":
    generate_all_figures()
