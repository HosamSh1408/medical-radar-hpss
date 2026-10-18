"""Visual comparison between NMF (Beltrão 2022 Branch A) and HPSS (Branch B).

Generates:
1. outputs/nmf_vs_hpss_spectrogram.png:
   Time-frequency spectrogram & waveform comparison on an artifact window.
2. outputs/nmf_vs_hpss_tracking.png:
   25-minute respiratory tracking, error time-series, and Bland-Altman for session 7.3.
3. outputs/nmf_vs_hpss_cohort_benchmark.png:
   18-session cohort performance comparison (RMSE, Acc@10, Acc@6, Runtime).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from .data_loader import load_radar
from .demod import csd
from .rbm_nmf import mitigate, stft_complex
from .hpss import mitigate_hpss
from .run_batch import CACHE
from .protocols import MEASURE_INFO
from .cohort import PAPER_18
from .result_tables import build_table

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)


def plot_spectrogram_comparison(measurement="7.3", start_s=60, dur_s=30):
    """Figure 1: Mechanism & Spectrogram Separation on an artifact window."""
    from .filtering import bandpass
    bI, bQ = load_radar(measurement)
    s0 = int(start_s * 16)
    W = int(dur_s * 16)
    x = bandpass(csd(bI[s0:s0 + W], bQ[s0:s0 + W]))

    out_nmf = mitigate(x)
    out_hpss = mitigate_hpss(x)

    # Spectrogram grids
    # STFT: NFFT=256, fs=16 Hz -> df = 16/256 = 0.0625 Hz = 3.75 bpm per bin
    # We display frequencies from 0 to 2.5 Hz (0 to 150 bpm), bins 0 to 40
    freq_bins = np.arange(40) * (16.0 / 256.0) * 60.0  # bpm
    t_frames = np.arange(out_nmf["mag"].shape[1]) * 1.0  # 1s hop

    mag_raw = out_nmf["mag"][:40, :]
    mag_nmf = out_nmf["mag_filt"][:40, :]
    mag_hpss = out_hpss["mag_filt"][:40, :]

    # Normalize for visual comparison
    vmax = np.percentile(mag_raw, 98)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0])

    # (a) Raw Spectrogram
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.pcolormesh(t_frames, freq_bins, mag_raw, shading="auto", cmap="magma",
                         vmin=0, vmax=vmax)
    ax0.set_title("(a) Raw Demodulated Spectrogram\n(Breathing line + Transient Artifact)", fontsize=11)
    ax0.set_ylabel("Breathing Rate (bpm)", fontsize=10)
    ax0.set_xlabel("Time (s)", fontsize=10)
    fig.colorbar(im0, ax=ax0, orientation="horizontal", pad=0.18, label="Magnitude")

    # (b) NMF Mitigated
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.pcolormesh(t_frames, freq_bins, mag_nmf, shading="auto", cmap="magma",
                         vmin=0, vmax=vmax)
    ax1.set_title(f"(b) NMF Filtered Spectrogram\n({out_nmf['n_removed']}/11 RBM bases pruned)", fontsize=11)
    ax1.set_xlabel("Time (s)", fontsize=10)
    fig.colorbar(im1, ax=ax1, orientation="horizontal", pad=0.18, label="Magnitude")

    # (c) HPSS Mitigated
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.pcolormesh(t_frames, freq_bins, mag_hpss, shading="auto", cmap="magma",
                         vmin=0, vmax=vmax)
    ax2.set_title("(c) HPSS Filtered Spectrogram\n(Harmonic Mask M_harm * |X|)", fontsize=11)
    ax2.set_xlabel("Time (s)", fontsize=10)
    fig.colorbar(im2, ax=ax2, orientation="horizontal", pad=0.18, label="Magnitude")

    # (d) Reconstructed Real Signals Comparison (Time domain)
    t_sig = np.arange(len(x)) / 16.0
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(t_sig, np.real(x), color="gray", alpha=0.55, lw=1.2, label="Raw CSD Demodulated (with motion spike)")
    ax3.plot(t_sig, np.real(out_nmf["x_hat"]), color="#1f77b4", lw=1.6, label="NMF Reconstruction (clean breathing)")
    ax3.plot(t_sig, np.real(out_hpss["x_hat"]), color="#2ca02c", lw=1.6, linestyle="--", label="HPSS Reconstruction (clean breathing)")
    ax3.set_title("(d) Time-Domain Signal Recovery: Raw vs NMF vs HPSS", fontsize=11)
    ax3.set_xlabel("Window Time (seconds)", fontsize=10)
    ax3.set_ylabel("Normalized Amplitude", fontsize=10)
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    p = OUT / "nmf_vs_hpss_spectrogram.png"
    fig.savefig(p, dpi=200)
    print("Saved:", p)
    return p


def plot_tracking_and_agreement(measurement="7.3"):
    """Figure 2: 25-minute respiratory tracking, error residuals, and Bland-Altman."""
    from .run_all import run_one
    from . import evaluation as EV

    r = run_one(measurement)
    v = r["valid"]
    t_min = r["centers_s"][v] / 60.0
    ref = r["ref"][v]
    est_nmf = r["est"][v]
    est_hpss = r["est_hpss"][v]

    err_nmf = np.abs(est_nmf - ref)
    err_hpss = np.abs(est_hpss - ref)

    ba_nmf = EV.bland_altman(est_nmf, ref)
    ba_hpss = EV.bland_altman(est_hpss, ref)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1.0, 1.2])

    # (a) Full Time-Series Tracking
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t_min, ref, "k-", lw=1.8,
             label="Reference respiration rate (impedance pneumography)", zorder=3)
    ax0.plot(t_min, est_nmf, color="#1f77b4", alpha=0.85, lw=1.2,
             label=f"Radar NLS + NMF (RMSE={np.sqrt(np.mean((est_nmf-ref)**2)):.2f} bpm)")
    ax0.plot(t_min, est_hpss, color="#2ca02c", alpha=0.85, lw=1.2, linestyle="--",
             label=f"Radar HPSS (RMSE={np.sqrt(np.mean((est_hpss-ref)**2)):.2f} bpm)")
    ax0.set_title(f"Continuous Respiratory Rate Tracking: Session {measurement} (Prone-Twins, N={len(ref)})", fontsize=12)
    ax0.set_xlabel("Time (minutes)", fontsize=10)
    ax0.set_ylabel("Breathing Rate (bpm)", fontsize=10)
    ax0.set_ylim(20, 80)
    ax0.legend(loc="upper right", framealpha=0.9)
    ax0.grid(True, alpha=0.3)

    # (b) Absolute Error Time Series
    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(t_min, err_nmf, color="#1f77b4", alpha=0.7, lw=1.0, label=f"NMF Error (MAE={err_nmf.mean():.2f} bpm)")
    ax1.plot(t_min, err_hpss, color="#2ca02c", alpha=0.7, lw=1.0, linestyle="--", label=f"HPSS Error (MAE={err_hpss.mean():.2f} bpm)")
    ax1.axhline(6.0, color="orange", linestyle=":", lw=1.2,
                 label="±6 bpm acceptance threshold")
    ax1.set_title("Absolute Estimation Error vs Time", fontsize=11)
    ax1.set_xlabel("Time (minutes)", fontsize=10)
    ax1.set_ylabel("|Radar - Reference| (bpm)", fontsize=10)
    ax1.set_ylim(0, 20)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # (c) Bland-Altman for NMF
    ax2 = fig.add_subplot(gs[2, 0])
    mean_nmf = (est_nmf + ref) / 2.0
    diff_nmf = est_nmf - ref
    ax2.scatter(mean_nmf, diff_nmf, color="#1f77b4", alpha=0.35, s=20, edgecolors="none")
    ax2.axhline(ba_nmf["bias"], color="red", lw=1.6, label=f"Bias: {ba_nmf['bias']:+.2f} bpm")
    ax2.axhline(ba_nmf["loa_up"], color="darkred", linestyle="--", lw=1.2, label=f"+1.96 SD: {ba_nmf['loa_up']:+.2f}")
    ax2.axhline(ba_nmf["loa_lo"], color="darkred", linestyle="--", lw=1.2, label=f"-1.96 SD: {ba_nmf['loa_lo']:+.2f}")
    ax2.axhline(0, color="gray", linestyle=":", lw=0.8)
    ax2.set_title(f"NLS + NMF Bland-Altman (RMSE={np.sqrt(np.mean((est_nmf-ref)**2)):.2f} bpm)", fontsize=11)
    ax2.set_xlabel("Mean (Radar & Reference) (bpm)", fontsize=10)
    ax2.set_ylabel("Radar - Reference (bpm)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # (d) Bland-Altman for HPSS
    ax3 = fig.add_subplot(gs[2, 1])
    mean_hpss = (est_hpss + ref) / 2.0
    diff_hpss = est_hpss - ref
    ax3.scatter(mean_hpss, diff_hpss, color="#2ca02c", alpha=0.35, s=20, edgecolors="none")
    ax3.axhline(ba_hpss["bias"], color="red", lw=1.6, label=f"Bias: {ba_hpss['bias']:+.2f} bpm")
    ax3.axhline(ba_hpss["loa_up"], color="darkred", linestyle="--", lw=1.2, label=f"+1.96 SD: {ba_hpss['loa_up']:+.2f}")
    ax3.axhline(ba_hpss["loa_lo"], color="darkred", linestyle="--", lw=1.2, label=f"-1.96 SD: {ba_hpss['loa_lo']:+.2f}")
    ax3.axhline(0, color="gray", linestyle=":", lw=0.8)
    ax3.set_title(f"HPSS Bland-Altman (RMSE={np.sqrt(np.mean((est_hpss-ref)**2)):.2f} bpm)", fontsize=11)
    ax3.set_xlabel("Mean (Radar & Reference) (bpm)", fontsize=10)
    ax3.set_ylabel("Radar - Reference (bpm)", fontsize=10)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    p = OUT / "nmf_vs_hpss_tracking.png"
    fig.savefig(p, dpi=200)
    print("Saved:", p)
    return p


def plot_cohort_benchmark():
    """Figure 3: 18-session cohort benchmark comparison (RMSE, Acc@10, Acc@6, Runtime)."""
    lines_full, _ = build_table(agg="permeas", minimal=False)
    lines_min, _ = build_table(agg="permeas", minimal=True)

    scenarios = ["ALL", "supine", "prone", "single", "twins"]
    scen_labels = ["All (18)", "Supine (11)", "Prone (7)", "Single (14)", "Twins (4)"]

    methods = ["NLS", "NLS+NMF", "HPSS"]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c"]

    rmse_data = {m: [] for m in methods}
    acc10_data = {m: [] for m in methods}
    acc6_data = {m: [] for m in methods}

    for scen in scenarios:
        for m in methods:
            match = [s for sc, meth, _, s in lines_full if sc == scen and meth == m]
            if match:
                rmse_data[m].append(match[0]["rmse"])
                acc10_data[m].append(match[0]["acc@10"])
                acc6_data[m].append(match[0]["acc@6"])

    # Add Minimal Movement Prone & ALL as highlight categories
    highlight_scens = ["Prone (MinMove)", "All (MinMove)"]
    for m in methods:
        mp = [s for sc, meth, _, s in lines_min if sc == "prone" and meth == m][0]
        ma = [s for sc, meth, _, s in lines_min if sc == "ALL" and meth == m][0]
        rmse_data[m].extend([mp["rmse"], ma["rmse"]])
        acc10_data[m].extend([mp["acc@10"], ma["acc@10"]])
        acc6_data[m].extend([mp["acc@6"], ma["acc@6"]])

    all_scen_labels = scen_labels + highlight_scens
    x = np.arange(len(all_scen_labels))
    w = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (a) RMSE comparison
    ax_rmse = axes[0, 0]
    for i, m in enumerate(methods):
        ax_rmse.bar(x + (i - 1) * w, rmse_data[m], w, label=m, color=colors[i], alpha=0.85)
    ax_rmse.set_xticks(x); ax_rmse.set_xticklabels(all_scen_labels, rotation=25, ha="right", fontsize=9)
    ax_rmse.set_ylabel("RMSE (bpm) [Lower is Better]", fontsize=10)
    ax_rmse.set_title("(a) Breathing Rate Estimation RMSE by Scenario", fontsize=11)
    ax_rmse.legend(loc="upper right")
    ax_rmse.grid(True, axis="y", alpha=0.3)

    # (b) Acc@10 comparison
    ax_acc10 = axes[0, 1]
    for i, m in enumerate(methods):
        ax_acc10.bar(x + (i - 1) * w, acc10_data[m], w, label=m, color=colors[i], alpha=0.85)
    ax_acc10.set_xticks(x); ax_acc10.set_xticklabels(all_scen_labels, rotation=25, ha="right", fontsize=9)
    ax_acc10.set_ylabel("Accuracy within ±10 bpm (%) [Higher is Better]", fontsize=10)
    ax_acc10.set_title("(b) Accuracy within ±10 bpm by Scenario", fontsize=11)
    ax_acc10.legend(loc="lower right")
    ax_acc10.grid(True, axis="y", alpha=0.3)
    ax_acc10.set_ylim(60, 100)

    # (c) Acc@6 comparison
    ax_acc6 = axes[1, 0]
    for i, m in enumerate(methods):
        ax_acc6.bar(x + (i - 1) * w, acc6_data[m], w, label=m, color=colors[i], alpha=0.85)
    ax_acc6.set_xticks(x); ax_acc6.set_xticklabels(all_scen_labels, rotation=25, ha="right", fontsize=9)
    ax_acc6.set_ylabel("Accuracy within ±6 bpm (%) [Higher is Better]", fontsize=10)
    ax_acc6.set_title("(c) Accuracy within ±6 bpm by Scenario", fontsize=11)
    ax_acc6.legend(loc="lower right")
    ax_acc6.grid(True, axis="y", alpha=0.3)
    ax_acc6.set_ylim(45, 80)

    # (d) Computational Latency & Execution Profile
    ax_time = axes[1, 1]
    nmf_times = []
    hpss_times = []
    for measurement in PAPER_18:
        cache = CACHE / f"{measurement}.npz"
        if not cache.exists():
            raise RuntimeError("Current caches are required for measured timing; run --batch --force")
        with np.load(cache, allow_pickle=False) as z:
            nmf_times.append(float(z["t_nmf"]))
            hpss_times.append(float(z["t_hpss"]))
    nmf_ms = float(np.mean(nmf_times))
    hpss_ms = float(np.mean(hpss_times))
    speedup = nmf_ms / hpss_ms
    bench_methods = ["HPSS (measured)", "NMF (measured)"]
    times_ms = [hpss_ms, nmf_ms]
    time_colors = ["#2ca02c", "#1f77b4"]
    bars = ax_time.bar(bench_methods, times_ms, color=time_colors, alpha=0.85, width=0.5)
    for bar, t in zip(bars, times_ms):
        ax_time.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                     f"{t:.1f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_time.set_ylabel("Execution Time per 30-s Window (ms)", fontsize=10)
    ax_time.set_title(f"(d) Measured Python latency (HPSS/NMF = {speedup:.2f}×)", fontsize=11)
    ax_time.set_ylim(0, max(times_ms) * 1.35)
    ax_time.grid(True, axis="y", alpha=0.3)
    ax_time.annotate(f"{speedup:.2f}× lower measured time\n(cohort mean)",
                     xy=(0, hpss_ms), xytext=(0.25, nmf_ms * 0.55),
                     arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5),
                     fontsize=10, color="darkgreen", fontweight="bold")

    fig.tight_layout()
    p = OUT / "nmf_vs_hpss_cohort_benchmark.png"
    fig.savefig(p, dpi=200)
    print("Saved:", p)
    return p


if __name__ == "__main__":
    plot_spectrogram_comparison("7.3", start_s=60, dur_s=30)
    plot_tracking_and_agreement("7.3")
    plot_cohort_benchmark()
