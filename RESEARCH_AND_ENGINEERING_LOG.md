# Research, Engineering & Audit Log: Contactless Neonatal Radar Respiration Monitoring

**Project:** Independent Reimplementation of Beltrão et al. (*Scientific Reports*, 2022) and Harmonic–Percussive Source Separation (HPSS) Extension  
**Pipeline Version:** `main-v3-audit-fix-2` (Provenance-Locked)  
**Dataset:** 35 Radar CSVs (`16 Hz` 24 GHz CW I/Q), 34 Reference EDFs (`256 Hz` impedance pneumography + `1 Hz` monitor breathing rate), 33 paired local NICU measurements ($7,105$ valid 30-s windows across 31 evaluable sessions; $5,466$ valid windows across the 18-session benchmark cohort).

---

## Table of Contents

1. [Executive Summary & Scientific Claim Boundaries](#1-executive-summary--scientific-claim-boundaries)
2. [Chronological History of Versions & Experiments](#2-chronological-history-of-versions--experiments)
3. [Deep-Dive: The "Raw NLS Paradox" & Mathematical Problems Solved](#3-deep-dive-the-raw-nls-paradox--mathematical-problems-solved)
4. [Canonical Pipeline Architecture & Code Walkthrough](#4-canonical-pipeline-architecture--code-walkthrough)
5. [Verified Benchmark & Multiverse Results](#5-verified-benchmark--multiverse-results)
6. [Data, Cohort & Provenance Audit](#6-data-cohort--provenance-audit)
7. [Lessons Learned & Master's Thesis Roadmap](#7-lessons-learned--masters-thesis-roadmap)

---

## 1. Executive Summary & Scientific Claim Boundaries

This repository implements an audited, provenance-locked signal-processing pipeline for contactless neonatal breathing rate estimation from $24\text{ GHz}$ continuous-wave (CW) radar in the Neonatal Intensive Care Unit (NICU). It addresses two research objectives:

1. **Forensic Reimplementation of Beltrão et al. (2022):** Reconstructing the four-stage baseline pipeline—Complex Signal Demodulation (`CSD`), Short-Time Fourier Transform (`STFT`) with Non-Negative Matrix Factorization (`NMF`) Random Body Movement (`RBM`) mitigation, bandpass filtering, and Harmonic Nonlinear Least Squares (`NLS`) estimation.
2. **Deterministic HPSS Extension:** Replacing the iterative, non-convex NMF optimization stage ($\mathcal{O}(I \cdot K \cdot F \cdot T)$) with closed-form, zero-frequency-centered **Harmonic–Percussive Source Separation (HPSS)** ($\mathcal{O}(F \cdot T)$) using orthogonal median filters and soft Wiener masking.

### Summary of Key Findings
* **Both Mitigated Branches Beat Raw Unmitigated Estimation:** Across the 18-session benchmark (`PAPER_18`, $N=5,466$ windows), both `NLS+NMF` (**6.86 bpm** RMSE, **85.4%** Acc@10) and `HPSS+NLS` (**6.80 bpm** RMSE, **85.8%** Acc@10) substantially outperform `DFT` (**10.11 bpm** RMSE, **74.8%** Acc@10) and unmitigated `Raw NLS` (**7.49 bpm** RMSE, **82.4%** Acc@10) across every clinical subgroup (`ALL`, `supine`, `prone`, `single`, `twins`).
* **Statistical Equivalence in Accuracy Across All Cohort Variants:** Across all four predefined local cohort variants ($18, 23, 28,$ and $31$ evaluable sessions), the paired $\Delta\text{RMSE} = \text{RMSE}_{\text{HPSS}} - \text{RMSE}_{\text{NMF}}$ lies within $[-0.096, +0.025]\text{ bpm}$, with every 95% session-clustered and patient-clustered bootstrap confidence interval spanning zero.
* **~6–7× Separation-Stage Speedup & 100% Determinism:** HPSS's fixed-cost median filtering is **~6–7× faster** than iterative NMF per 30-s window (same-process, same-window per-window means; in the `main-v3-audit-fix-2` rebuild NMF $\approx 17.8$ ms vs. HPSS $\approx 2.8$ ms, $\approx 6.4\times$, for `all_paired_local` in [`reconstruction_multiverse.csv`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/outputs/audits/reconstruction_multiverse.csv)). Absolute runtimes are strongly environment/load dependent and only the ratio is reported (a controlled benchmark is pending, WS8); the batch builder is single-process/sequential. HPSS eliminates NMF's convergence failures (NMF hits its 500-iteration limit without meeting tolerance on **~89%** of `PAPER_18` valid windows).

### Permitted vs. Prohibited Scientific Claims

| Claim Level | Description | Status | Justification |
| :--- | :--- | :--- | :--- |
| **L0 — Computational Reproducibility** | Code runs deterministically from raw inputs with cryptographic cache verification. | **PASS** | 15/15 unit, provenance, and golden-value regression tests and 33/33 SHA-256 provenance-locked `.npz` caches pass. |
| **L1 — Constrained Reimplementation** | Published equations/parameters are mapped; unspecified protocol steps are explicitly documented. | **PASS** | All inferred parameters are isolated in [`config.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/config.py) and [`unresolved_protocol_items.yaml`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/metadata/unresolved_protocol_items.yaml). |
| **L2 — Exact Paper Reproduction** | Exact recovery of Beltrão et al.'s original 18 session IDs and $4,964$-window denominator. | **BLOCKED** | Original subject IDs are not listed in the paper; `PAPER_18` ($5,466$ windows) is ranked by local reference validity. |
| **L3 — Internal Signal-Processing Equivalence & Speedup** | HPSS matches NMF accuracy while running 6.3–6.9× faster on this dataset. | **EXPLORATORY PASS** | Verified across 4 cohort variants with 2,000-resample bootstrap CIs. |
| **L4 — External Generalization** | Performance holds on an untouched hospital site, radar hardware, or infant cohort. | **NOT TESTED** | Single-center NICU dataset only. |
| **L5 — Clinical Readiness / Superiority** | Safe to replace wired bedside monitors or trigger clinical alarms. | **BLOCKED** | No prospective clinical utility, alarm-fatigue, or embedded hardware study. |

---

## 2. Chronological History of Versions & Experiments

All intermediate prototypes and historical branches were audited, distilled into lessons learned, and retired so the final repository contains only the clean, reproducible pipeline.

### Phase 1: Initial Extraction & Naive Prototype (`v1`)
* **What was tried:** Direct translation of the paper's block diagram (`CSD -> 3-s rectangular STFT -> 11-component NMF -> iSTFT -> Bandpass -> NLS`) with initial heuristic thresholds ($\alpha = 1.8$, sparsity $> 0.5$) and unmasked 1-Hz cross-correlation lag search.
* **What went wrong:**
  1. Unmitigated `Raw NLS` (`7.46 bpm` RMSE) unexpectedly beat both `NLS+NMF` (`8.35 bpm`) and `HPSS+NLS` (`7.80 bpm`).
  2. Runtime bars in early plotting scripts were hard-coded rather than read from cached `perf_counter()` measurements.
  3. Reference channels were mislabeled as capnography instead of **impedance pneumography** (`256 Hz` raw waveform + `1 Hz` monitor-derived `RespRate`).

### Phase 2: Real-Valued Phase Demodulation & Joint HR–BR Viterbi Exploration (`p1` / `p2`)
* **What was tried:** Explored an alternative architecture inspired by follow-up radar literature (`p1_run.py`, `p2_run.py`, `preprocess.py`, `spectra.py`, `decoder.py`):
  * Algebraic ellipse fitting on $(I, Q)$ to correct gain/phase imbalance, followed by arctangent phase unwrapping $\theta(t) = \text{unwrap}(\text{atan2}(Q, I))$.
  * Welch power spectral density estimation and a 2D state-space **Viterbi decoder** tracking Breathing Rate ($18\text{–}90\text{ bpm}$) and Heart Rate ($90\text{–}210\text{ bpm}$) jointly with a Harmonic Quality Index (HQI) penalty when $n \cdot f_{\text{BR}} \approx f_{\text{HR}}$.
* **Why it was retired from the main submission:**
  * While joint HR–BR Viterbi decoding smoothed trajectory jumps on anchor session `7.3`, mixing phase demodulation and Viterbi tracking with Beltrão et al.'s CSD + memoryless NLS pipeline confounded the controlled A/B comparison between NMF and HPSS.
  * **Lesson learned:** To rigorously evaluate HPSS as a drop-in replacement for NMF, every upstream (`CSD`, `Bandpass`, `STFT`) and downstream (`iSTFT`, `Bandpass`, `Autocorr + NLS`, `Sync`) block must be held 100% identical.

### Phase 3: Alternative Gate & Lag Sensitivity Exploration (`beltrao_v2` / `sensitivity_v2`)
* **What was tried:** Tested alternative NMF basis-rejection rules (`max_over_median` vs. Hoyer sparsity, $\alpha \in \{1.8, 3.0\}$), whole-window vs. center-sample reference saturation gating, and alternative lag search ranges (`[-180, +180] s` vs. metadata-anchored `±45 s`).
* **Findings:**
  * Unconstrained `[-180, +180] s` lag searches produced spurious correlations on short or heavily saturated sessions (`1.2`, `10.4`), whereas anchoring the search within `±45 s` of the EDF/CSV header timestamp offset prevented false alignments.
  * Whole-window saturation gating (`zero saturated seconds in the 30 s window`) paired with 30-s window-averaged reference rates aligned much more closely with the paper's 30-s sliding window formulation than point-sampling the 1-Hz monitor rate.

### Phase 4: Fail-Closed Provenance & Cohort Multiverse (`v3`)
* **What was built:**
  * Added [`provenance.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/provenance.py) to bind every `.npz` cache file to the exact SHA-256 hashes of all 14 core Python modules, raw radar CSV and reference EDF files, Python/OS platform strings, and package versions.
  * Added [`cohort.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/cohort.py), [`build_metadata.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/build_metadata.py), and [`reconstruction_multiverse.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/reconstruction_multiverse.py) to evaluate all 33 paired local sessions across four transparent cohort definitions (`all_paired_local`, `pool30_no_zero_overlap`, `retained23_named_exclusions`, `current_reference_validity_ranked_18`) rather than relying solely on an outcome-ranked 18-session subset.

### Phase 5: Forensic Resolution of the "Raw NLS Paradox" (`main-v3-audit-fix-1`)
* **What was solved:** Conducted a first-principles mathematical audit of why `STFT -> Mitigation -> iSTFT` was degrading clean windows, identified four interacting signal-processing flaws plus one lag-correlation flaw, verified the fixes in a non-destructive ablation, integrated them into the active pipeline, and rebuilt all 33 provenance-locked caches, tables, and figures.

### Phase 6: Documentation-to-Code Reconciliation Audit (`main-v3-audit-fix-2`)
* **What was solved:** A doubt-first line-by-line audit of the code against the documentation. Twelve documentation/implementation discrepancies were corrected (audit **A1–A12**: filter type/band, reference sampling rate, harmonic-order description, multiverse window counts, "8-worker" runtime fiction, provenance file set, EDF start-datetime API, saturation-gate wording, `env_normalize` description, dead-code labeling, STFT tail-frame comment, timing drift), plus two audit-blocking issues (audit **A13** NMF-convergence percentage, **A14** a documented `--all-paired` rebuild flag that did not exist). The cache-dependency set was extended to include `protocols.py`, the pipeline version bumped to `main-v3-audit-fix-2`, and **golden-value regression tests** were added that bind the band/sampling/cohort/metric/multiverse claims to code and artifacts. Details in [`AUDIT_PLAN.md`](file:///Users/hosamshibly/Downloads/medical_radar/AUDIT_PLAN.md) and [`audit/AUDIT_FINDINGS.md`](file:///Users/hosamshibly/Downloads/medical_radar/audit/AUDIT_FINDINGS.md).

---

## 3. Deep-Dive: The "Raw NLS Paradox" & Mathematical Problems Solved

### The Paradox
Prior to `main-v3-audit-fix-1`, running the 18-session benchmark yielded an counter-intuitive result:
* **Unmitigated `Raw NLS`:** `7.46 bpm` RMSE (`81.8%` Acc@10)
* **`NLS + NMF`:** `8.35 bpm` RMSE (`79.5%` Acc@10) — *worse than doing nothing!*
* **`HPSS + NLS`:** `7.80 bpm` RMSE (`80.7%` Acc@10) — *also worse than doing nothing!*

In Beltrão et al. (2022), NMF artifact mitigation **reduced** overall error (`Fig. 6` vs. `Fig. 7`). Why did passing the signal through `STFT -> NMF/HPSS -> iSTFT` hurt performance in our earlier implementation, even on minimal-movement windows?

### Root Cause 1: Out-of-Band Spectral Leakage from the 3-s Rectangular STFT
* **Mathematical Mechanism:** The pipeline uses a 3-s rectangular STFT window ($M = 48$ samples at $f_s = 16\text{ Hz}$) zero-padded to $N_{\text{FFT}} = 256$. A rectangular window has a Dirichlet kernel spectrum with $-13\text{ dB}$ first sidelobes. Raw demodulated radar $x[n] = I[n] + jQ[n]$ contains large sub-$0.3\text{ Hz}$ baseline wander and DC drift ($10\text{–}20\times$ larger than the millimeter chest wall displacement).
  * When `stft_complex` is inverted directly without modification, the positive and negative sidelobes of the sub-$0.3\text{ Hz}$ drift cancel out exactly in overlap-add (`istft_complex`).
  * However, **as soon as any time-varying mask (NMF or HPSS) alters the spectrogram**, that exact phase cancellation is broken, and the $-13\text{ dB}$ sidelobes of the sub-$0.3\text{ Hz}$ drift leak directly into the $0.3\text{–}3.0\text{ Hz}$ ($18\text{–}180\text{ bpm}$) breathing band—surviving the post-iSTFT bandpass filter!
* **The Fix:** In [`run_measurement.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/run_measurement.py#L22), we apply `d_raw = bandpass(x)` ($0.3\text{–}3.0\text{ Hz}$ 64-tap Kaiser FIR, $\beta=6.5$, matching the paper's Methods band) **before** entering the STFT in both `mitigate(d_raw)` and `mitigate_hpss(d_raw)`.

### Root Cause 2: Post-Masking Amplitude "Holes" Breaking `autocorr_coarse`
* **Mathematical Mechanism:** When a 2–4 second RBM burst ($8\times$ amplitude) occurs inside a 30-s window and is suppressed by NMF or HPSS, the reconstructed time-domain waveform $\hat{x}[n]$ has near-zero amplitude during those 2–4 seconds and normal breathing amplitude elsewhere, or residual amplitude modulation at the burst boundaries.
  * Downstream, [`autocorr_coarse`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/filtering.py#L15-L30) searches for peaks in the normalized autocorrelation with `prominence=0.10` and picks the **first** lag peak (`peaks[0]`, highest frequency) to initialize the fine NLS search (`±5 bpm`).
  * Envelope dips and boundary steps flatten the fundamental autocorrelation peak below `0.10` prominence or create a false high-frequency ripple peak, causing `autocorr_coarse` to jump to $18\text{ bpm}$ or $80+\text{ bpm}$.
* **The Fix:** Consistent with Beltrão et al.'s spectrogram normalization before time-domain resynthesis, we added [`env_normalize`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/rbm_nmf.py#L44-L48) after spectrogram masking in both [`rbm_nmf.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/rbm_nmf.py#L71) and [`hpss.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/hpss.py#L57), scaling each active STFT time column by its peak magnitude (`peaks > 0.2 * median(peaks)`). This restores a uniform-envelope breathing wave across suppressed artifact segments.

### Root Cause 3: Direct Low-Rank Synthesis vs. Soft Wiener Ratio Masking in NMF
* **Mathematical Mechanism:** Replacing $|X|$ directly with $W_{\text{keep}} H_{\text{keep}}$ forces every clean frame through a rank-$\le 11$ approximation, smoothing sharp harmonic peaks even when 0 bases are flagged as RBM. Furthermore, $\alpha = 1.8$ in Eq. 11 was overly aggressive on pre-bandpassed spectra, pruning valid respiratory bases.
* **The Fix:** In [`rbm_nmf.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/rbm_nmf.py#L67-L70) and [`config.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/config.py#L29-L30), we set `RBM_ALPHA = 3.0` and `RBM_SPARSITY_FRAC = 0.6` and implemented a **Wiener ratio mask**:
  $$\hat{M}_{\text{NMF}}(f,t) = \frac{W_{\text{keep}} H_{\text{keep}}}{W H + \varepsilon} \odot |X(f,t)|$$
  When no RBM basis is removed ($W_{\text{keep}} H_{\text{keep}} = WH$), the mask is identically $1.0$, guaranteeing zero distortion on clean windows.

### Root Cause 4: Missing `fftshift` Along the Frequency Axis in Two-Sided Complex HPSS
* **Mathematical Mechanism:** `stft_complex` produces a two-sided 256-bin FFT in standard NumPy order:
  * Rows `0 .. 127`: $0\text{ Hz}$ to $+8\text{ Hz}$ (breathing band $+0.3\text{ to }+3.0\text{ Hz}$ is at rows `5 .. 48`).
  * Rows `128 .. 255`: $-8\text{ Hz}$ to $-0.0625\text{ Hz}$ (negative breathing band is at rows `208 .. 251`).
  * Previously, [`hpss_filter`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/hpss.py#L22-L48) ran `median_filter(mag, size=(21, 1), mode="nearest")` directly on the unshifted array. For positive breathing bins (`rows 6 .. 10`), the 21-bin vertical filter hit the boundary at `row 0` and repeated `row 0` (`mode="nearest"`) 5–10 times instead of wrapping smoothly across DC ($0\text{ Hz}$) into the negative frequencies (`rows 255, 254, ...`)!
* **The Fix:** [`hpss_filter`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/hpss.py#L38-L45) now shifts DC to the center (`mag_s = np.fft.fftshift(mag, axes=0)`), applies the orthogonal median filters ($L_{\text{harm}} = 7$ frames $= 7\text{ s}$, $L_{\text{perc}} = 21$ bins $= 1.31\text{ Hz}$), and shifts the mask back (`np.fft.ifftshift(mask_s, axes=0)`).

### Root Cause 5: Unmasked Reference Contamination in Lag Synchronization
* **Mathematical Mechanism:** [`sync_arrays`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/evaluation.py#L31-L67) previously computed Pearson correlation across candidate lags using all 1-Hz reference samples, including intervals where the reference impedance pneumograph was saturated (`|Resp| >= 2.99`).
* **The Fix:** [`sync_arrays`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/evaluation.py#L31-L67) now restricts correlation strictly to saturation-free 30-s windows (`vwin`) against the 30-s window-average reference (`ref_w`).

### Summary of Before vs. After Repair (`PAPER_18` Per-Measurement Mean)

| Method | Before Fix RMSE (bpm) | After Fix RMSE (bpm) | Before Fix Acc@10 (%) | After Fix Acc@10 (%) | Behavior vs. `Raw NLS` |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`DFT` (No Mit.)** | 10.22 | **10.11** | 74.6% | **74.8%** | Baseline coarse spectral peak |
| **`NLS` (No Mit.)** | 7.46 | **7.49** | 81.8% | **82.4%** | Harmonic model without RBM removal |
| **`NLS + NMF`** | 8.35 *(worse)* | **6.86** *(−0.63 bpm)* | 79.5% *(worse)* | **85.4%** *(+3.0 pp)* | **Beats Raw NLS across all 5 scenarios** |
| **`HPSS + NLS`** | 7.80 *(worse)* | **6.80** *(−0.69 bpm)* | 80.7% *(worse)* | **85.8%** *(+3.4 pp)* | **Best overall; 6.9× faster than NMF** |

---

## 4. Canonical Pipeline Architecture & Code Walkthrough

### End-to-End Signal Flow (`run_window` in [`run_measurement.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/run_measurement.py))

```text
Raw 24 GHz Radar I/Q (16 Hz, 30-s window = 480 samples, 2-s slide)
   │
   ▼
[1. Complex Signal Demodulation]  demod.py: csd(bI, bQ)
   │  x[n] = (I[n] - mean(I)) + j*(Q[n] - mean(Q))
   ▼
[2. Pre-STFT Bandpass Filter]     filtering.py: bandpass(x)
   │  64-tap zero-phase Kaiser FIR [0.3, 3.0] Hz (β=6.5, 18–180 bpm) -> d_raw
   ├──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
   ▼ (Branch 1: DFT)              ▼ (Branch 2: Raw NLS)          ▼ (Branch 3: NLS+NMF)          ▼ (Branch 4: HPSS+NLS)
dft_estimate(d_raw)            autocorr_coarse(d_raw)         rbm_nmf.py: mitigate(d_raw)    hpss.py: mitigate_hpss(d_raw)
Zero-padded FFT (4,096)        + nls_estimate(d_raw)          • 3-s rect STFT (hop=1s, 256)  • 3-s rect STFT (hop=1s, 256)
                                                              • 11-comp NNDSVD NMF           • fftshift along freq axis
                                                              • RBM rule (α=3.0, sp>0.6)     • Median filters:
                                                              • Wiener mask + env_normalize    L_harm=(1,7), L_perc=(21,1)
                                                              • iSTFT + bandpass             • Soft Wiener mask (γ=2.0)
                                                              • autocorr_coarse + NLS        • ifftshift + env_normalize
                                                                                             • iSTFT + bandpass
                                                                                             • autocorr_coarse + NLS
```

### Module-by-Module Inventory (`paper_workspace/src/`)

| File | Role in Pipeline | Key Functions & Mathematical Equations |
| :--- | :--- | :--- |
| [`config.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/config.py) | Locked system constants | `FS_RADAR = 16`, `WIN_S = 30`, `UPDATE_S = 2`, `BAND_LOW_HZ = 0.3`, `BAND_HIGH_HZ = 3.0`, `KAISER_BETA = 6.5`, `FIR_TAPS = 64`, `NMF_K = 11`, `RBM_ALPHA = 3.0`, `RBM_SPARSITY_FRAC = 0.6`. |
| [`protocols.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/protocols.py) | Clinical metadata | Maps each measurement ID (`1.1`–`12.3`) to posture (`supine`/`prone`), co-bedding `twins` status, and incubator `lid` state from `annotations.pdf`. |
| [`cohort.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/cohort.py) | Cohort definitions | Defines `PAPER_18` (18 sessions), `retained23()` (23 sessions), `pool30()` (30 sessions), and `local_pool()` (all 33 paired sessions). |
| [`data_loader.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/data_loader.py) | Raw I/O & pairing | Loads $16\text{ Hz}$ `bI, bQ` CSV columns and $256\text{ Hz}$ `Resp` + $1\text{ Hz}$ `RespRate` EDF channels; parses recording start timestamps. |
| [`demod.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/demod.py) | Complex demodulation | `csd(bI, bQ)` removes per-window DC offsets to form $x[n] = \tilde{I}[n] + j\tilde{Q}[n]$. |
| [`filtering.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/filtering.py) | Bandpass & coarse pitch | `bandpass(x)` applies a zero-phase 64-tap Kaiser FIR (β=6.5) over $0.3\text{–}3.0\text{ Hz}$; `autocorr_coarse(x)` finds the first prominent lag peak in $[18, 180]\text{ bpm}$. |
| [`estimation.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/estimation.py) | Frequency estimators | `dft_estimate(x)` (4,096-pt zero-padded FFT peak); `nls_estimate(x, f_coarse)` searches $L_k=2$ harmonics over $f_0 \pm 5\text{ bpm}$ ($0.1\text{ bpm}$ grid) with $f_0 < 0.75\text{ Hz}$ subharmonic check. |
| [`rbm_nmf.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/rbm_nmf.py) | Baseline NMF branch | `stft_complex`, `istft_complex`, `decompose_nmf`, `detect_rbm` (Eq. 11 + Hoyer sparsity), Wiener ratio mask, and `env_normalize`. |
| [`hpss.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/hpss.py) | Proposed HPSS branch | `hpss_filter` (`fftshift`-centered $(1,7)$ harmonic and $(21,1)$ percussive median filters, soft mask $M_H = S_H^2 / (S_H^2 + S_P^2 + \varepsilon)$) and `mitigate_hpss`. |
| [`lags.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/lags.py) | Timestamp lag anchor | Computes header clock offset between radar CSV and reference EDF and constrains lag search to $\pm 45\text{ s}$ around the metadata anchor. |
| [`evaluation.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/evaluation.py) | Gating, sync & metrics | `window_ref_and_gating` ($|V| \ge 2.99$ saturation gate; a 30-s window is valid iff it contains **zero** saturated seconds), `sync_arrays` (saturation-masked lag search anchored to the metadata lag), and `scores` (RMSE, Acc@3/6/10, Bias). |
| [`provenance.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/provenance.py) | Cache integrity | Computes SHA-256 over all 15 core modules (incl. `protocols.py`), raw inputs, and package versions; fails closed if any cache is stale or modified. |
| [`run_measurement.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/run_measurement.py) | 4-way session runner | Runs all 4 branches on every 30-s window and records per-window `perf_counter()` separation times (`t_nmf`, `t_hpss`) and `nmf_converged`. |
| [`run_all.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/run_all.py) | Session orchestrator | Pairs radar estimates with synchronized reference targets and aggregates scenario metrics. |
| [`run_batch.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/run_batch.py) | Sequential batch engine | Single-process loop that builds and verifies `.npz` caches in `paper_workspace/outputs/cache/`; `--all-paired` rebuilds all 33 paired sessions. |
| [`compare_ab.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/compare_ab.py) | Single-session demo | Prints 4-way metrics on anchor session `7.3` (`python3 main.py --demo`). |
| [`result_tables.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/result_tables.py) | Paper tables | Generates `Fig. 6` (per-measurement mean), `Fig. 7` (minimal movement), and pooled tables (`python3 main.py --tables`). |
| [`build_metadata.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/build_metadata.py) | Cohort ledger builder | Writes [`cohort_ledger.csv`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/metadata/cohort_ledger.csv) with SHA-256 hashes and exclusion provenance for all 33 paired sessions. |
| [`reconstruction_multiverse.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/reconstruction_multiverse.py) | Multiverse evaluator | Computes 4-cohort comparison tables and 2,000-resample bootstrap CIs (`python3 main.py --multiverse`). |
| [`plot_figs.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/plot_figs.py) | Reproduction figures | Generates `fig4_7.3_60s.png`, `fig5_7.3.png`, `bland_altman_7.3.png`, and `fig6_scenario_comparison.png`. |
| [`plot_nmf_vs_hpss.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/plot_nmf_vs_hpss.py) | Extension figures | Generates `nmf_vs_hpss_spectrogram.png`, `nmf_vs_hpss_tracking.png`, and `nmf_vs_hpss_cohort_benchmark.png`. |
| [`test_pipeline.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/test_pipeline.py) | Automated test suite | 15 unit, mathematical, provenance, and golden-value regression tests (`python3 main.py --test`). |

---

## 5. Verified Benchmark & Multiverse Results

### A. Anchor Session `7.3` (`N = 594` Valid Windows, `lag = -33 s`, `corr = 0.44`)
Reproduced via `python3 main.py --demo`:

| Method | Acc@1 (%) | Acc@3 (%) | Acc@5 (%) | Acc@6 (%) | Acc@10 (%) | RMSE (bpm) | MAE (bpm) | Bias (bpm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DFT` (No Mit.)** | 35.2% | 61.3% | 72.6% | 78.3% | 90.1% | 6.66 | 3.97 | +1.96 |
| **`NLS` (No Mit.)** | 33.7% | 64.3% | 77.4% | 83.5% | 94.1% | 4.49 | 3.06 | +0.90 |
| **`NLS + NMF`** | 33.0% | 64.5% | 78.3% | 81.6% | 93.9% | **4.85** | **3.20** | +1.55 |
| **`HPSS + NLS`** | 33.7% | 64.1% | 77.4% | 82.2% | 94.1% | **4.79** | **3.18** | +1.36 |

### B. 18-Session Benchmark (`PAPER_18`, `Fig. 6` Per-Measurement Scenario Mean, $N = 5,466$)
Reproduced via `python3 main.py --tables`:

| Scenario | $N_{\text{meas}}$ | $N_{\text{win}}$ | Method | Acc@1 (%) | Acc@3 (%) | Acc@5 (%) | Acc@6 (%) | Acc@10 (%) | RMSE (bpm) | MAE (bpm) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ALL** | 18 | 5,466 | `DFT` | 12.6 | 32.8 | 51.3 | 58.6 | 74.8 | 10.11 | 7.52 |
| **ALL** | 18 | 5,466 | `NLS` | 14.1 | 36.6 | 56.4 | 63.9 | 82.4 | 7.49 | 5.82 |
| **ALL** | 18 | 5,466 | `NLS+NMF` | 15.0 | 38.5 | 58.9 | 67.1 | 85.4 | **6.86** | **5.35** |
| **ALL** | 18 | 5,466 | `HPSS` | **15.2** | **39.1** | **59.0** | **67.3** | **85.8** | **6.80** | **5.32** |
| **supine** | 11 | 3,120 | `DFT` | 10.6 | 29.3 | 46.8 | 53.7 | 69.8 | 11.54 | 8.65 |
| **supine** | 11 | 3,120 | `NLS` | 11.9 | 33.7 | 52.0 | 59.2 | 78.1 | 8.35 | 6.54 |
| **supine** | 11 | 3,120 | `NLS+NMF` | 13.0 | 35.3 | 54.3 | 62.1 | 81.3 | **7.63** | **6.00** |
| **supine** | 11 | 3,120 | `HPSS` | **13.7** | **36.9** | **55.1** | **62.7** | **81.5** | **7.55** | **5.94** |
| **prone** | 7 | 2,346 | `DFT` | 15.9 | 38.3 | 58.4 | 66.3 | 82.6 | 7.85 | 5.74 |
| **prone** | 7 | 2,346 | `NLS` | 17.8 | 41.1 | 63.3 | 71.4 | 89.0 | 6.12 | 4.69 |
| **prone** | 7 | 2,346 | `NLS+NMF` | **18.2** | **43.6** | **66.2** | **74.9** | 91.7 | **5.65** | **4.32** |
| **prone** | 7 | 2,346 | `HPSS` | 17.6 | 42.6 | 65.1 | 74.4 | **92.4** | **5.62** | 4.35 |
| **single** | 14 | 4,002 | `DFT` | 12.2 | 33.2 | 52.4 | 60.1 | 76.5 | 9.79 | 7.27 |
| **single** | 14 | 4,002 | `NLS` | 13.4 | 35.6 | 55.5 | 63.3 | 82.4 | 7.54 | 5.91 |
| **single** | 14 | 4,002 | `NLS+NMF` | 14.3 | 37.4 | 58.3 | 66.7 | 85.2 | **6.89** | 5.42 |
| **single** | 14 | 4,002 | `HPSS` | **14.3** | **38.2** | **58.5** | **67.0** | **85.9** | **6.76** | **5.37** |
| **twins** | 4 | 1,464 | `DFT` | 14.1 | 31.4 | 47.3 | 53.5 | 68.7 | 11.23 | 8.38 |
| **twins** | 4 | 1,464 | `NLS` | 16.6 | 40.2 | 59.6 | 66.3 | 82.3 | 7.31 | 5.52 |
| **twins** | 4 | 1,464 | `NLS+NMF` | 17.5 | **42.3** | **61.3** | **68.2** | **85.8** | **6.77** | **5.08** |
| **twins** | 4 | 1,464 | `HPSS` | **18.2** | **42.3** | 60.9 | 68.1 | 85.1 | **6.94** | 5.17 |

### C. Minimal Movement Windows (`Fig. 7`, $\le 2$ Removed NMF Bases, $N = 4,468 / 5,466 = 81.7\%$)

| Scenario | $N_{\text{meas}}$ | $N_{\text{win}}$ | `DFT` RMSE (Acc@10) | `NLS` RMSE (Acc@10) | `NLS+NMF` RMSE (Acc@10) | `HPSS+NLS` RMSE (Acc@10) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ALL** | 18 | 4,468 | 9.73 bpm (76.0%) | 7.12 bpm (83.6%) | **6.58 bpm (86.5%)** | **6.60 bpm (86.6%)** |
| **supine** | 11 | 2,615 | 11.16 bpm (71.0%) | 8.02 bpm (79.3%) | **7.37 bpm (82.3%)** | **7.42 bpm (82.1%)** |
| **prone** | 7 | 1,853 | 7.49 bpm (83.7%) | 5.72 bpm (90.5%) | **5.34 bpm (93.3%)** | **5.33 bpm (93.5%)** |
| **single** | 14 | 3,216 | 9.50 bpm (77.5%) | 7.19 bpm (83.5%) | **6.63 bpm (86.3%)** | **6.61 bpm (86.6%)** |
| **twins** | 4 | 1,252 | 10.54 bpm (70.7%) | 6.90 bpm (84.0%) | **6.41 bpm (87.3%)** | **6.59 bpm (86.5%)** |

### D. Four-Variant Cohort Multiverse (`HPSS - NMF`)
Reproduced via `python3 main.py --multiverse` (stored in [`paper_workspace/outputs/audits/reconstruction_multiverse.csv`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/outputs/audits/reconstruction_multiverse.csv)):

| Cohort Variant | Evaluable Sessions | Patient Groups | Valid Windows | NMF RMSE (bpm) | HPSS RMSE (bpm) | $\Delta\text{RMSE}$ (HPSS−NMF) | 95% Session Bootstrap CI | Patient-Clustered 95% CI | $\Delta\text{Acc@10}$ (pp) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `all_paired_local` | 31 / 33 | 12 | 7,105 | 8.349 | 8.361 | **+0.012** | $[-0.456, +0.372]$ | $[-0.659, +0.517]$ | $-0.35\text{ pp}$ |
| `pool30_no_zero_overlap` | 28 / 30 | 12 | 6,642 | 8.181 | 8.085 | **−0.096** | $[-0.574, +0.287]$ | $[-0.713, +0.464]$ | $-0.28\text{ pp}$ |
| `retained23_named_exclusions` | 23 / 23 | 10 | 6,447 | 7.796 | 7.821 | **+0.025** | $[-0.202, +0.276]$ | $[-0.223, +0.340]$ | $+0.19\text{ pp}$ |
| `current_reference_validity_ranked_18` | 18 / 18 | 10 | 5,466 | 6.863 | 6.798 | **−0.065** | $[-0.249, +0.099]$ | $[-0.341, +0.165]$ | $+0.39\text{ pp}$ |

---

## 6. Data, Cohort & Provenance Audit

1. **Raw File Inventory (`beltrao_data/`):**
   * **35** radar CSV files (`16 Hz` I/Q samples) and **34** reference EDF files (`256 Hz` impedance pneumography + `1 Hz` `RespRate`).
   * **33** locally paired measurements mapped in [`data_loader.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/data_loader.py).
   * Unpaired files: `1.2` and `3.2` have radar CSVs but no reference EDFs; `10.4` has an EDF but no radar CSV.
   * Zero-valid-window sessions: `1.3` and `5.1` yield $0$ valid windows under the active saturation gate (a window is valid iff it contains **zero** seconds with `|V| >= 2.99`; any single saturated second rejects the whole 30-s window) and are transparently logged as non-evaluable (`31/33` evaluable in `all_paired_local`).
2. **Protocol Transcription Verification (`annotations.pdf` 36/36 Exact Match):**
   * Visual and OCR inspection of [`beltrao_data/annotations.pdf`](file:///Users/hosamshibly/Downloads/medical_radar/beltrao_data/annotations.pdf) (Supplementary Tables 2a–2d, Pages 3–6) confirms a **36/36 exact match** with `MEASURE_INFO` in [`protocols.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/protocols.py) (including row `7.2` on Page 5: `co-bedding = yes (twins)`, `supine = x`, `lid = closed`).
3. **NMF Convergence Diagnostic:**
   * `scikit-learn` NMF (`K=11, init='nndsvda', max_iter=500, tol=1e-4`) meets its stopping tolerance before 500 iterations in **10.7%** of valid windows in `PAPER_18` (**12.5%** across all 31 evaluable sessions). Rather than hiding `ConvergenceWarning`, [`rbm_nmf.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/rbm_nmf.py) records `nmf_converged` in every cache file.
4. **Synchronization Lag-Leakage & Motion-Severity Audit (`audit/`):**
   * **Branch-Neutral & Symmetric Lag Verification (`WS3`):** In the default evaluation (`active_nmf_synced`), the fine cross-correlation lag (`meta_lag ± 60 s`) is selected using `NLS+NMF` predictions, giving NMF a one-sided selection advantage. When evaluated under **symmetric per-method own-lag** (`per_method_own_synced`) or **branch-neutral raw-NLS lag** (`branch_neutral_raw_nls_synced`), `HPSS+NLS` achieves lower RMSE than `NLS+NMF` across **all four cohort definitions** (`PAPER_18`, `retained23`, `pool30`, and `all_paired_local`; e.g., `6.785` vs. `6.863 bpm` on `PAPER_18` under own-lag, and `7.135` vs. `7.301 bpm` under branch-neutral raw-NLS lag).
   * **Motion-Severity Partition (`WS2`):** On the `4,468 / 5,466` (`81.7%`) minimal-movement windows of `PAPER_18` (`n_removed <= 2`, reproducing Fig. 7), `NLS+NMF` and `HPSS+NLS` are nearly tied (`6.582` vs. `6.604 bpm`). On the **998 heavy-movement windows (`n_removed > 2`)**, `HPSS+NLS` outperforms `NLS+NMF` by **−0.488 bpm** (`7.561` vs. `8.050 bpm`).

---

## 7. Lessons Learned & Master's Thesis Roadmap

### Key Engineering & Scientific Lessons Learned
1. **Never treat STFT/iSTFT as a transparent identity when applying time-varying masks:** A rectangular STFT is only a perfect-reconstruction transform when unmodified. Time-varying spectral masks break Dirichlet sidelobe cancellation, turning out-of-band DC/drift into in-band noise unless the signal is bandpassed *before* the STFT.
2. **Time-domain estimators are sensitive to amplitude holes left by frequency-domain masks:** Suppressing transient motion bursts creates temporal envelope dropouts that distort autocorrelation peak prominence unless per-frame envelope normalization (`env_normalize`) is applied prior to iSTFT resynthesis.
3. **Two-sided complex spectrograms require zero-frequency centering (`fftshift`) before 2D spatial filtering:** Applying vertical median filters across an unshifted FFT array clamps at $0\text{ Hz}$ instead of bridging negative and positive low-frequency respiratory bins.
4. **Provenance-locked caching is essential for scientific integrity:** Binding `.npz` caches to cryptographic SHA-256 hashes of source code, input data, and package versions prevents stale numbers from contaminating downstream tables or figures.

### Roadmap to Elevate to a Master's Thesis
1. **Causal, COLA-Compliant & Trajectory-Aware HPSS:** Replace the rectangular STFT with a 75%-overlap Hann COLA STFT, replace centered temporal median filters with causal one-sided quantile filters for zero-lookahead streaming, and introduce slope-adaptive harmonic filtering for non-stationary neonatal tachypnea.
2. **Unified Multi-Vital (HR–BR) State-Space Tracking:** Combine the dual-band HPSS front-end with algebraic ellipse I/Q calibration and a joint Heart Rate + Breathing Rate Viterbi decoder to eliminate memoryless window-to-window jumps and harmonic collisions.
3. **Semi-Synthetic Physics Benchmark:** Inject real NICU RBM bursts onto simulated ground-truth neonatal chest waveforms across controlled Signal-to-Artifact Ratios ($\text{SAR} \in [-20, +10]\text{ dB}$) to evaluate performance in the severe-motion regime where reference EDF sensors saturate.
4. **HPSS-Intrinsic Signal Quality Index (SQI) & Leave-One-Infant-Out CV:** Use the HPSS percussive-to-harmonic energy ratio as a real-time abstention gate (Risk–Coverage curves) and validate hyperparameter selection under grouped Leave-One-Infant-Out cross-validation.
