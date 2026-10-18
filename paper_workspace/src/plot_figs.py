"""Generate Fig.5-style (I/Q + freq track) and Fig.4-style (NMF panels) for a measurement."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from .data_loader import load_radar
from .demod import csd
from .rbm_nmf import mitigate

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)


def fig5_style(measurement="7.3"):
    from .run_all import run_one
    r = run_one(measurement)
    bI, bQ = load_radar(measurement)
    t_raw = np.arange(len(bI)) / 16 / 60
    t_win = r["centers_s"] / 60
    v = r["valid"]

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(t_raw, bI, label="I", alpha=0.6, lw=0.8)
    ax[0].plot(t_raw, bQ, label="Q", alpha=0.6, lw=0.8)
    ax[0].set_ylabel("Amplitude")
    ax[0].legend(loc="upper right")
    ax[0].set_title(f"Beltrão Fig. 5 Reproduction: Radar I/Q & Frequency Track ({measurement})")

    # Plot frequency tracks on window time grid
    ref_masked = np.where(v, r["ref"], np.nan)
    est_masked = np.where(v, r["est"], np.nan)
    ax[1].plot(t_win, ref_masked, "r-", label="Reference respiration rate (impedance pneumography)", lw=1.5)
    ax[1].plot(t_win, est_masked, "b-", label="Radar NLS+NMF", alpha=0.85, lw=1.2)
    ax[1].set_xlabel("Time (minutes)")
    ax[1].set_ylabel("Breathing Rate (bpm)")
    ax[1].legend(loc="upper right")
    ax[1].set_ylim(0, 100)

    fig.tight_layout()
    p = OUT / f"fig5_{measurement}.png"
    fig.savefig(p, dpi=150)
    print("saved", p, r["full_nls_nmf"])
    return p


def fig4_style(measurement="7.3", start_s=60, dur_s=60):
    from .filtering import bandpass
    bI, bQ = load_radar(measurement)
    s0 = int(start_s * 16)
    W = int(dur_s * 16)
    x = bandpass(csd(bI[s0:s0 + W], bQ[s0:s0 + W]))
    out = mitigate(x)
    t = np.arange(len(x)) / 16
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(t, np.real(x), label="I (raw)", lw=1)
    ax[0].plot(t, np.imag(x), label="Q (raw)", lw=1)
    ax[0].legend(loc="upper right")
    ax[0].set_title(f"Beltrão Fig. 4 Reproduction: NMF Artifact Mitigation ({measurement}, t={start_s}s)")
    ax[0].set_ylabel("Raw Demod")
    ax[1].plot(t, np.real(out["x_hat"]), label="I (mitigated)", lw=1)
    ax[1].plot(t, np.imag(out["x_hat"]), label="Q (mitigated)", lw=1)
    ax[1].set_ylabel("Cleaned")
    ax[1].set_xlabel("Window Time (seconds)")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    p = OUT / f"fig4_{measurement}_{start_s}s.png"
    fig.savefig(p, dpi=150)
    print("saved", p, "removed", out["n_removed"])
    return p


def fig_bland_altman(measurement="7.3"):
    """Supp. Fig. 4 style Bland-Altman agreement plot."""
    from .run_all import run_one
    from . import evaluation as EV
    r = run_one(measurement)
    v = r["valid"]
    est = r["est"][v]
    ref = r["ref"][v]
    ba = EV.bland_altman(est, ref)

    mean_val = (est + ref) / 2.0
    diff_val = est - ref

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean_val, diff_val, alpha=0.35, edgecolors="none", color="#1f77b4", s=25)
    ax.axhline(ba["bias"], color="red", linestyle="-", lw=1.8,
               label=f"Mean Bias: {ba['bias']:+.2f} bpm")
    ax.axhline(ba["loa_up"], color="darkred", linestyle="--", lw=1.2,
               label=f"+1.96 SD: {ba['loa_up']:+.2f} bpm")
    ax.axhline(ba["loa_lo"], color="darkred", linestyle="--", lw=1.2,
               label=f"-1.96 SD: {ba['loa_lo']:+.2f} bpm")
    ax.axhline(0, color="gray", linestyle=":", lw=0.8)
    ax.set_xlabel("Mean Breathing Frequency (bpm)", fontsize=11)
    ax.set_ylabel("Radar - Reference Difference (bpm)", fontsize=11)
    ax.set_title(f"Bland-Altman Agreement: Measurement {measurement}", fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = OUT / f"bland_altman_{measurement}.png"
    fig.savefig(p, dpi=150)
    print(f"saved {p} (bias={ba['bias']:+.2f}, sd={ba['sd']:.2f}, loa=[{ba['loa_lo']:+.2f}, {ba['loa_up']:+.2f}])")
    return p


def fig6_style():
    """Beltrão Fig. 6 reproduction: Scenario comparison bar chart."""
    from .result_tables import build_table
    lines, _ = build_table(agg="permeas")
    scenarios = ["ALL", "supine", "prone", "single", "twins"]
    methods = ["DFT", "NLS", "NLS+NMF", "HPSS"]

    rmse_data = {m: [] for m in methods}
    acc10_data = {m: [] for m in methods}

    for scen in scenarios:
        for m in methods:
            match = [s for sc, meth, _, s in lines if sc == scen and meth == m]
            if match:
                rmse_data[m].append(match[0]["rmse"])
                acc10_data[m].append(match[0]["acc@10"])
            else:
                rmse_data[m].append(np.nan)
                acc10_data[m].append(np.nan)

    x = np.arange(len(scenarios))
    w = 0.20
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]
    for i, m in enumerate(methods):
        ax1.bar(x + (i - 1.5) * w, rmse_data[m], w, label=m, color=colors[i], alpha=0.85)
        ax2.bar(x + (i - 1.5) * w, acc10_data[m], w, label=m, color=colors[i], alpha=0.85)

    ax1.set_xticks(x); ax1.set_xticklabels(scenarios, fontsize=11)
    ax1.set_ylabel("RMSE (bpm)", fontsize=11)
    ax1.set_title("Breathing Rate Estimation RMSE by Scenario (Fig. 6a)", fontsize=12)
    ax1.legend(loc="upper right")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.set_xticks(x); ax2.set_xticklabels(scenarios, fontsize=11)
    ax2.set_ylabel("Accuracy within ±10 bpm (%)", fontsize=11)
    ax2.set_title("Accuracy within ±10 bpm by Scenario (Fig. 6b)", fontsize=12)
    ax2.legend(loc="lower right")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    p = OUT / "fig6_scenario_comparison.png"
    fig.savefig(p, dpi=150)
    print("saved", p)
    return p


if __name__ == "__main__":
    fig5_style("7.3")
    fig4_style("7.3")
    fig_bland_altman("7.3")
    fig6_style()
