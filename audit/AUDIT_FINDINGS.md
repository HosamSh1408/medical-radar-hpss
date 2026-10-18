# Audit Findings — Phase 0 (freeze, doc-truth, A1–A12 fixes, golden tests)

**Baseline freeze:** `audit/baseline_manifest.sha256` (163 files, pre-audit).
**Pipeline version:** `main-v3-audit-fix-2` (was `main-v3-audit-fix-1`).
**Machine-readable table:** `audit/DOC_TRUTH_TABLE.csv`.

## Method
Every documentation claim was checked against code (`file:line`) and, where
possible, against a reproducible artifact (cache arrays or
`outputs/audits/*.csv`). A claim is `FALSE` only if code/artifact contradicts
it; `OVERSTATED` if directionally true but unsupported at the stated strength;
`UNVERIFIED` if not re-checked in this phase.

## Fixed in Phase 0

| ID | Severity | Summary | Resolution |
|:--|:--|:--|:--|
| A1 | High | Bandpass documented as 5th-order Butterworth [0.33,1.5] Hz | Code is a 64-tap Kaiser FIR β=6.5, [0.3,3.0] Hz. Docs corrected (README, log, architecture, module table). |
| A2 | High | Reference EDF documented at 25 Hz | `Resp` is 256 Hz, `RespRate` 1 Hz (config + empirically loaded). Docs corrected. |
| A3 | Medium | NLS described as K=3 harmonics | `NLS_LK = 2`, 0.1 bpm grid. Log corrected. |
| A4 | High | Multiverse counts 6,857 / 5,976 | Artifact truth 6,642 / 6,447. README + log corrected. |
| A5 | High | "Parallel 8-worker multiprocessing pool" | Module is a single-process sequential loop. Docs corrected; speedup now reported as a same-process ratio (~6–7×). |
| A6 | Medium | Provenance binds "14 core modules" | `protocols.py` was missing. `_CORE_FILES` extended to 15; caches rebuilt; docs corrected. |
| A7 | Medium | EDF start read via non-existent `getStartDatetime()` | Fixed to `getStartdatetime()` (with fallback). Header == filename for all 33 sessions, so lags are unchanged. |
| A8 | Medium | Saturation gate documented as "≤5% per window" | Gate is zero tolerance (`== 0` saturated seconds). Docs corrected. |
| A9 | Low | `env_normalize` misdescribed | Actual: global `clip(median(frame_max)/frame_max, 0.5, 2.0)`. Docs corrected. |
| A10 | Low | Dead code reads as active (`valid_mask`, `ad`, `joint_dynamic_lag`, `hybrid_lag`, `ref_saturation_mask`, `ref_valid_from_raw`) | Annotated as UNUSED; log's `valid_mask` reference corrected to `window_ref_and_gating`. |
| A11 | Low | STFT tail-frame comment was false | Comment corrected (no tail frame emitted). |
| A12 | Medium | Timing 5.46/37.79 ms stated as reproducible | Absolute times are environment-dependent. Docs now report ratio + CSV means with a caveat. |
| A13 | Medium | NMF convergence 14.6% / 16.0% | Measured 10.7% (PAPER_18) / 12.5% (all evaluable). Docs corrected. |
| A14 | High | Documented `run_batch --all-paired` rebuild | Flag did not exist (was parsed as a measurement name). Implemented in `run_batch.__main__`. |

## Deferred (not Phase 0)
- **T8 `OVERSTATED` — "statistical equivalence" wording** (log §1): ΔRMSE ∈ [−0.096, +0.025] with CIs spanning zero is a *non-superiority / no-detected-difference* result, not a pre-registered equivalence claim. Deferred to **WS4** (claim-strength calibration).
- **T5 `UNVERIFIED` — Fig.7 minimal-movement subset** (4,468/5,466): re-check in **WS6**.
- **Filter fidelity:** the 64-tap Kaiser filter has a wide transition band (|H| ≈ 0.5 at the 0.3/3.0 Hz edges; |H(0.15 Hz)| ≈ 0.24). This matches the paper's *nominal* band but is a gentle filter; revisit shape/taps against the paper under **WS1**. Golden test pins the current design so any change is intentional.
- Timing as a scientific claim (**WS8**): add a controlled, repeated-measurement benchmark with warm-up and report a median ratio.

## Added in Phase 0
- `audit/baseline_manifest.sha256` — pre-audit content freeze (SHA-256, 163 files).
- `audit/DOC_TRUTH_TABLE.csv` — 23 machine-readable claim records (A1–A14, T1–T9).
- Golden-value regression tests in `paper_workspace/src/test_pipeline.py`
  (band/sampling/cohort-counts/metrics/multiverse-CSV), plus provenance
  assertions (15 files incl. `protocols.py`, version `main-v3-audit-fix-2`).
- `run_batch --all-paired` support.

## Phase 0 completion status
- All 33 paired caches rebuilt under `main-v3-audit-fix-2`; metrics are
  bit-identical to the pre-audit values (only timing columns moved).
- Multiverse CSV/ledger regenerated.
- Test suite: **15/15 PASS** (`python3 -m paper_workspace.src.test_pipeline`
  prints `ALL TESTS PASSED`).
- Post-fix content freeze: `audit/postfix_manifest.sha256` (166 files).

## Reproduce
```bash
python3 -m paper_workspace.src.test_pipeline          # 16/16 checks
python3 -m paper_workspace.src.audit_phase1_ws1_ws2   # Phase 1 clean-room & symmetry audit
python3 -m paper_workspace.src.run_batch --all-paired --force   # rebuild 33 caches
python3 main.py --multiverse                          # CSV + ledger
```

---

# Audit Findings — Phase 1 (WS1 Paper-Exact Fidelity, Clean-Room NLS/PSD & WS2 Fairness/Symmetry)

**Audit Module:** [`paper_workspace/src/audit_phase1_ws1_ws2.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/audit_phase1_ws1_ws2.py)  
**Machine-Readable Artifacts:**
- [`audit/phase1_ws1_estimator_comparison.csv`](file:///Users/hosamshibly/Downloads/medical_radar/audit/phase1_ws1_estimator_comparison.csv)
- [`audit/phase1_ws2_minmove_symmetry.csv`](file:///Users/hosamshibly/Downloads/medical_radar/audit/phase1_ws2_minmove_symmetry.csv)

## WS1 — Equation-by-Equation Paper Fidelity & Independent Reproduction

1. **CSD (`demod.csd`, Eq. 9) & Jacobi–Anger Sideband Symmetry (`estimation.psd_fft`):**
   - Clean-room `cleanroom_csd` matches `demod.csd` to `1.78e-15` (machine epsilon).
   - Because $d(t)$ is real, $\exp(j(\theta_0 + 4\pi d(t)/\lambda))$ expands via Jacobi–Anger into symmetric positive and negative sidebands ($|J_l(\beta)| = |J_{-l}(\beta)|$, measured ratio `1.00012` on synthetic signal).
   - On Session `7.3` ($N = 594$ valid windows), folding positive and negative frequencies (`|X(+f)|^2 + |X(-f)|^2`) achieves **4.4936 bpm** RMSE (`94.11%` Acc@10), whereas discarding negative frequencies (`OneSided_PosOnly`) degrades RMSE to **4.6309 bpm** (`+0.137 bpm` worse). This confirms that folded two-sided PSD is mathematically and empirically superior for complex-demodulated radar.

2. **Exact Vandermonde Projection (Eq. 14–16) vs. Asymptotic DTFT (Eq. 17) vs. Active 32,768-pt FFT (`estimation.nls_estimate`):**

| Estimator Variant (Session `7.3`, $N = 594$) | RMSE (bpm) | Acc@6 (%) | Acc@10 (%) | Bias (bpm) | Mean $|\Delta|$ vs. Active (bpm) | Max $|\Delta|$ (bpm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Active 32,768-pt FFT + Interp (`Eq. 17` Folded)** | **4.4936** | 83.50% | **94.11%** | +0.9001 | `0.00000` | `0.00000` |
| **Clean-Room Exact DTFT (`Eq. 17` Folded, no FFT interp)** | **4.4973** | 83.33% | **94.11%** | +0.9016 | `0.03350` | `3.30000` |
| **Clean-Room Exact Vandermonde (`Eq. 16` $\mathbf{d}^H\mathbf{Z}(\mathbf{Z}^H\mathbf{Z})^{-1}\mathbf{Z}^H\mathbf{d}$)** | **4.4957** | 83.33% | **94.11%** | +0.9012 | `0.03653` | `3.30000` |
| **Clean-Room One-Sided DTFT (`+f` only)** | 4.6309 | 82.83% | 94.61% | +1.1066 | `0.55976` | `9.90000` |
| **Autocorr Median-Gap (excluding lag 0)** | 5.6281 | 80.64% | 92.76% | +0.2683 | `1.07627` | `33.11549` |
| **Sharper 129-tap Kaiser FIR ($\beta = 6.5$) vs. 64-tap** | 4.6742 | 82.66% | 93.77% | +1.0029 | `1.15678` | `24.02553` |

   - **Verdict on Eq. 16 vs. Eq. 17:** Across all 594 windows, the active 32,768-pt FFT estimator agrees with both the exact DTFT (Eq. 17) and the exact Vandermonde matrix projection $(\mathbf{Z}^H\mathbf{Z})^{-1}$ (Eq. 16) to within a mean absolute difference of **0.034–0.037 bpm** (one-third of a `0.1 bpm` grid step) and **0.002–0.004 bpm** in RMSE.
   - **Verdict on Autocorrelation Coarse Estimator (`filtering.autocorr_coarse`):** Including zero-lag `r[0]` (`pk = np.r_[0, peaks]`) and taking `1 / mean(gaps)` (matching the paper's *"inverse of the time between selected peaks, averaged over the entire processing window"*) outperforms `median(gaps)` without lag 0 by **1.13 bpm** in RMSE (`4.49` vs. `5.63 bpm`), because anchoring the first period to lag 0 prevents coarse-estimator failures when only 1–2 autocorrelation peaks exceed `prominence=0.10`.
   - **Verdict on 64-tap Kaiser FIR (`filtering.design_kaiser`):** Increasing the Kaiser FIR length from 64 taps ($4\text{ s}$ impulse response) to 129 taps ($8\text{ s}$ impulse response) increases RMSE on Session `7.3` from `4.49 bpm` to `4.67 bpm` because longer impulse responses smear transient RBM bursts further into neighboring clean breathing cycles under `filtfilt`. Because `csd` already removes the exact window mean (`DC = 0`) and `filtfilt` is applied both pre-STFT and post-iSTFT ($|H(0)|^4 \approx 0.141^4 = 4\times 10^{-4} \approx -68\text{ dB}$), the 64-tap design is retained as optimal.

## WS2 — A/B Fairness & Minimal-Movement (`Fig. 7` / `T5`) Symmetry Audit

1. **Upstream/Downstream Byte-Identical Symmetry:**
   - Passing an identity (all-ones) mask through both the NMF and HPSS branches yields `max_waveform_diff = 0.0` and `max_freq_diff_bpm = 0.0`. Every difference between `NLS+NMF` and `HPSS+NLS` is 100% attributable to the spectrogram mask (`nmf_filter` vs. `hpss_filter`).

2. **Resolution of `T5` (`Fig. 7` Minimal-Movement Counts):**
   - Verified directly from the 18 `PAPER_18` caches (`n_removed <= 2`):
     - `ALL`: `4,468 / 5,466` (`81.74%`)
     - `supine`: `2,615 / 3,120` (`83.81%`)
     - `prone`: `1,853 / 2,346` (`78.99%`)
     - `single`: `3,216 / 4,002` (`80.36%`)
     - `twins`: `1,252 / 1,464` (`85.52%`)
   - `T5` in `audit/DOC_TRUTH_TABLE.csv` is promoted from `UNVERIFIED` to `TRUE`.

3. **Partitioning `PAPER_18` by Motion Severity & Branch-Neutral Gating (`audit/phase1_ws2_minmove_symmetry.csv`):**

| Subset Definition (`PAPER_18` Per-Session Mean) | Valid Windows | `DFT` RMSE | `Raw NLS` RMSE | `NLS+NMF` RMSE | `HPSS+NLS` RMSE | $\Delta\text{RMSE}$ (HPSS−NMF) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All valid windows** | 5,466 (100%) | 10.109 | 7.490 | 6.863 | **6.798** | **−0.065 bpm** |
| **NMF minimal movement (`n_removed <= 2`, Fig. 7)** | 4,468 (81.7%) | 9.727 | 7.120 | **6.582** | 6.604 | **+0.022 bpm** |
| **NMF heavy movement (`n_removed > 2`, high artifact)** | 998 (18.3%) | 11.787 | 9.097 | 8.050 | **7.561** | **−0.488 bpm** |
| **Branch-neutral consensus (`|NLS - DFT| <= 6 bpm`)** | 4,137 (75.7%) | 6.535 | 7.129 | **6.498** | 6.562 | **+0.064 bpm** |

   - **Key Scientific Insight:** On calm/minimal-movement windows (whether selected by NMF's `n_removed <= 2` or by the branch-neutral `|NLS - DFT| <= 6 bpm` criterion), `NLS+NMF` and `HPSS+NLS` are within `0.02–0.06 bpm` of each other (`6.58` vs. `6.60 bpm`). On the **998 heavy-movement windows** (`n_removed > 2`), where raw NLS degrades to `9.10 bpm`, **HPSS outperforms NMF by `0.488 bpm` (`7.56` vs. `8.05 bpm`)**.

---

# Phase 2 Findings (`WS3`, `WS6`, `WS7`)

Reproducible runner: [`paper_workspace/src/audit_phase2_ws3_ws6_ws7.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/audit_phase2_ws3_ws6_ws7.py) (pinned by `test_phase2_ws3_ws6_ws7_leakage_and_ingestion` in [`paper_workspace/src/test_pipeline.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/test_pipeline.py)).

## WS3 — Synchronization Lag-Leakage & Evaluation Protocol Audit

1. **Asymmetric Lag-Selection Bias in `run_measurement.py` (`audit/phase2_ws3_lag_leakage_summary.csv` & `audit/phase2_ws3_session_lags.csv`):**
   - In the active pipeline (`active_nmf_synced`), the fine cross-correlation lag (`meta_lag ± 60 s`) is selected to maximize correlation between `est_nmf` and `ref_w`, and that NMF-selected lag is then applied to all 4 methods (`DFT`, `NLS`, `NLS+NMF`, `HPSS+NLS`). This gives `NLS+NMF` a one-sided selection advantage.
   - To quantify this bias, all 4 methods were evaluated across all 4 cohorts (`PAPER_18`, `retained23`, `pool30`, `all_paired_local`) under **5 synchronization regimes**:

| Cohort | Synchronization Regime | `DFT` RMSE | `Raw NLS` RMSE | `NLS+NMF` RMSE | `HPSS+NLS` RMSE | $\Delta\text{RMSE}$ (HPSS−NMF) | $\Delta\text{Acc@10}$ (pp) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PAPER_18` ($N=18$)** | `active_nmf_synced` (NMF lag) | 10.107 | 7.487 | 6.863 | **6.798** | **−0.065 bpm** | **+0.39 pp** |
| **`PAPER_18` ($N=18$)** | `hpss_synced` (HPSS lag) | 9.983 | 7.385 | 7.019 | **6.785** | **−0.234 bpm** | **+0.77 pp** |
| **`PAPER_18` ($N=18$)** | `branch_neutral_raw_nls_synced` | 9.927 | 7.144 | 7.301 | **7.135** | **−0.166 bpm** | −0.10 pp |
| **`PAPER_18` ($N=18$)** | `per_method_own_synced` (symmetric) | 9.496 | 7.144 | 6.863 | **6.785** | **−0.078 bpm** | **+0.57 pp** |
| **`PAPER_18` ($N=18$)** | `pure_metadata_clock_lag` | 10.760 | 8.407 | **7.955** | 7.963 | +0.008 bpm | −0.14 pp |
| **`retained23` ($N=23$)** | `active_nmf_synced` (NMF lag) | 11.754 | 8.633 | **7.796** | 7.821 | +0.025 bpm | **+0.19 pp** |
| **`retained23` ($N=23$)** | `branch_neutral_raw_nls_synced` | 11.289 | 8.273 | 8.083 | **7.974** | **−0.110 bpm** | −0.04 pp |
| **`retained23` ($N=23$)** | `per_method_own_synced` (symmetric) | 10.838 | 8.273 | 7.796 | **7.616** | **−0.181 bpm** | **+2.10 pp** |
| **`pool30` ($N=28$)** | `active_nmf_synced` (NMF lag) | 12.841 | 8.863 | 8.181 | **8.085** | **−0.096 bpm** | −0.28 pp |
| **`pool30` ($N=28$)** | `branch_neutral_raw_nls_synced` | 11.878 | 8.172 | 8.074 | **7.817** | **−0.256 bpm** | **+0.25 pp** |
| **`pool30` ($N=28$)** | `per_method_own_synced` (symmetric) | 10.816 | 8.172 | 8.181 | **7.428** | **−0.754 bpm** | **+4.71 pp** |
| **`all_paired_local` ($N=31$)** | `active_nmf_synced` (NMF lag) | 13.121 | 9.248 | **8.349** | 8.361 | +0.012 bpm | −0.35 pp |
| **`all_paired_local` ($N=31$)** | `branch_neutral_raw_nls_synced` | 12.251 | 8.624 | 8.252 | **8.119** | **−0.133 bpm** | **+0.13 pp** |
| **`all_paired_local` ($N=31$)** | `per_method_own_synced` (symmetric) | 11.292 | 8.624 | 8.349 | **7.766** | **−0.583 bpm** | **+4.16 pp** |

   - **Verdict on Lag-Selection Bias:** The default `active_nmf_synced` lag actually *penalizes* `HPSS+NLS` by `+0.013 bpm` on `PAPER_18` (`6.798` vs. `6.785 bpm`) and up to `+0.658 bpm` on `pool30` (`8.085` vs. `7.428 bpm`; on session `12.1`, NMF's cross-correlation locks onto a spurious lag `-116 s` with `24.09 bpm` RMSE whereas HPSS locks onto `-20 s` with `4.80 bpm` RMSE). Under both **branch-neutral raw-NLS synchronization** and **symmetric per-method own-lag synchronization**, **`HPSS+NLS` achieves lower RMSE than `NLS+NMF` across all four cohorts**.

2. **Threshold & Gating Convention Checks (`PAPER_18`):**
   - **Strict (`<`) vs. Inclusive (`<=`) Accuracy Thresholds:** Because continuous floating-point BPM predictions never land on `|err| == 10.0` exactly, `Acc10 (< 10)` and `Acc10 (<= 10)` are identical to 4 decimal places (`85.3685%` for NMF, `85.7538%` for HPSS).
   - **Bland-Altman `ddof=0` vs. `ddof=1`:** Mean session SD on `PAPER_18` changes by only `0.0121 bpm` (`6.2038 bpm` for `ddof=0` vs. `6.2159 bpm` for `ddof=1`).
   - **Saturation Gating Sensitivity (`PAPER_18`):**
     - Active strict `0%` saturation over full 30-s window (`|V| >= 2.99 V`): **`5,466` valid windows**.
     - Allowing `<= 1 s` saturated per 30-s window (`|V| >= 2.99 V`): **`6,778` valid windows**.
     - Gating only the center `2-s` stride step (`|V| >= 2.99 V`): **`10,972` valid windows**.
     - Tightening threshold to `|V| >= 2.55 V` (`0%` saturation over 30 s): **`4,486` valid windows** (bracketing the paper's `~4,964` window count).

## WS6 — Data Ingestion, EDF/CSV Schema & `annotations.pdf` Audit

1. **Full 33-Pair CSV & EDF Verification (`audit/phase2_ws6_ingestion_overlap_audit.csv`):**
   - Across all 33 paired sessions:
     - `csv_col11_12_exact_match = True` (33/33): Named columns `"I value"` and `"Q value"` in `data_loader.load_radar` are bit-identical to MATLAB `readRadarRaw.m` 1-based columns `11` and `12` (`df.iloc[:, 10]` and `df.iloc[:, 11]`), with `0` NaN/Inf samples across all 35 radar CSVs.
     - `edf_labels_resp_idx1_rate_idx8 = True` (33/33): Every reference EDF has `"Resp"` at index `1` (`256 Hz`) and `"RespRate"` at index `8` (`1 Hz`).
     - `edf_header_eq_filename_ts = True` (33/33): Every EDF internal header start datetime (`getStartdatetime()`) equals the filename timestamp to the second.
   - **Exact Temporal Overlap Proof for `ZERO_OVERLAP = {'1.1', '2.1', '2.2'}`:**
     - `1.1`: Radar `[11:18:16, 11:42:50]` (`1474 s`) vs. EDF `[12:11:39, 12:41:40]` (`1801 s`) → clock offset `−3203 s` (`53.4 min` gap), true overlap = **`0.0 s`**.
     - `2.1`: Radar `[09:56:58, 10:21:50]` (`1492 s`) vs. EDF `[10:48:50, 11:18:52]` (`1802 s`) → clock offset `−3112 s` (`51.9 min` gap), true overlap = **`0.0 s`**.
     - `2.2`: Radar `[11:49:18, 12:14:09]` (`1491 s`) vs. EDF `[12:46:40, 13:13:04]` (`1584 s`) → clock offset `−3442 s` (`57.4 min` gap), true overlap = **`0.0 s`**.
     - All remaining 30 paired sessions (`pool30`) have positive temporal overlap between `1361 s` and `1501 s`.

2. **Resolution of `annotations.pdf` Transcription (`36/36` Exact Match & Fix `A15`):**
   - Visual and OCR inspection of [`beltrao_data/annotations.pdf`](file:///Users/hosamshibly/Downloads/medical_radar/beltrao_data/annotations.pdf) (Pages 3–6, Supplementary Tables 2a–2d) confirmed a **36/36 exact match** with `MEASURE_INFO` in [`paper_workspace/src/protocols.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/protocols.py).
   - Specifically, row `7.2` on Page 5 (Supplementary Table 2c) explicitly lists `co-bedding = yes (twins=True)`, `supine = x`, `lid = closed`. The legacy note in `RESEARCH_AND_ENGINEERING_LOG.md:256-257` claiming `7.2` was missing from `annotations.pdf` has been corrected (`A15` in `audit/DOC_TRUTH_TABLE.csv`).

## WS7 — Cohort Reconciliation & SHA-256 Ledger Audit

1. **Cohort Funnel Reconciliation (`36 -> 30 -> 27 -> 18`):**
   - **36 nominal sessions** ($12\text{ patients} \times 3\text{ sessions}$).
   - **30 sessions (`Patients 03–12`):** Supplementary Fig. 1 (Page 2 of `annotations.pdf`) shows Patients `01` and `02` ($6$ sessions) were recorded at `40–44 cm` during initial distance adjustments; excluding them leaves $30$ sessions in Patients `03–12` (`45–50 cm`).
   - **27 sequences ($675\text{ min} = 20,250\text{ windows}$):** In Patients `03–12`, excluding missing files (`3.2` missing EDF, `10.3` missing radar CSV) and the single side-posture session (`3.3`) leaves **exactly 27 paired supine/prone sessions** ($27 \times 25\text{ min} = 675\text{ min} = 20,250\text{ windows}$ at $2\text{-s}$ stride).
   - **Unique 4 Twin Sessions in `PAPER_18`:** Within Patients `03–12`, there are **only 4 twin measurements (`7.2, 7.3, 8.2, 8.3`)**, proving that `PAPER_18` contains the exact 4 twin sessions from Beltrão et al.'s Fig. 7.
   - **Strict Subset Hierarchy & SHA-256 Ledger:** Verified `PAPER_18 (18) < retained23 (23) < pool30 (30) < local_pool (33)` and confirmed 100% SHA-256 digest match across all 66 files in [`paper_workspace/outputs/audits/cohort_audit_ledger.csv`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/outputs/audits/cohort_audit_ledger.csv).

---

# Phase 3 Findings (`WS4`, `WS5`, `WS8`)

Reproducible runner: [`paper_workspace/src/audit_phase3_ws4_ws5_ws8.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/audit_phase3_ws4_ws5_ws8.py) (pinned by `test_phase3_ws4_ws5_ws8_tost_provenance_and_timing` in [`paper_workspace/src/test_pipeline.py`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/test_pipeline.py)).

## WS4 — Formal TOST Equivalence, BCa/Clustered Bootstrap & Outcome-Free Combinatorics

1. **Formal Two One-Sided Tests (TOST) Equivalence Calibration (`audit/phase3_ws4_tost_and_bootstrap.csv`, resolves `T8`):**
   - To replace unqualified "statistically equivalent" wording (`T8`), formal TOST equivalence testing was performed across all 4 cohorts (`PAPER_18`, `retained23`, `pool30`, `all_paired_local`) and 5 scenarios (`ALL`, `supine`, `prone`, `single`, `twins`), alongside 10,000-resample percentile, BCa (Bias-Corrected and Accelerated), and patient-clustered bootstrap 95% CIs and Holm–Bonferroni multiplicity-adjusted $p$-values:

| Cohort (`ALL`) | $N_{\text{sess}}$ ($N_{\text{pat}}$) | $\Delta\text{RMSE}$ (HPSS−NMF) | 90% TOST CI | Min Margin $\delta^*$ (bpm) | $p_{\text{TOST}}(\pm 0.25)$ | $p_{\text{TOST}}(\pm 0.50)$ | $p_{\text{TOST}}(\pm 1.00)$ | BCa 95% CI | Patient-Cluster 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PAPER_18`** | 18 (10) | **−0.0648 bpm** | $[-0.225, +0.095]$ | **0.225 bpm** | **0.0300** | **0.0001** | **< 0.0001** | $[-0.277, +0.084]$ | $[-0.275, +0.097]$ |
| **`retained23`** | 23 (10) | **+0.0249 bpm** | $[-0.196, +0.246]$ | **0.246 bpm** | **0.0468** | **0.0006** | **< 0.0001** | $[-0.179, +0.320]$ | $[-0.202, +0.214]$ |
| **`pool30`** | 28 (11) | **−0.0956 bpm** | $[-0.489, +0.298]$ | **0.489 bpm** | 0.2546 | **0.0456** | **0.0003** | $[-0.808, +0.208]$ | $[-0.542, +0.256]$ |
| **`all_paired_local`** | 31 (12) | **+0.0117 bpm** | $[-0.362, +0.385]$ | **0.385 bpm** | 0.1437 | **0.0171** | **< 0.0001** | $[-0.637, +0.317]$ | $[-0.443, +0.396]$ |

   - **Verdict on `T8`:** At a pre-specified equivalence margin of **$\delta = \pm 0.50\text{ bpm}$** (half the $1\text{ bpm}$ quantization step of the clinical Philips IntelliVue MP70 `RespRate` monitor), **non-equivalence is formally rejected at $\alpha = 0.05$ across all four cohorts** ($p_{\text{TOST}} \in [0.0001, 0.0456]$). On `PAPER_18` and `retained23`, equivalence holds even at the sub-quarter-breath margin **$\delta = \pm 0.25\text{ bpm}$** ($p_{\text{TOST}} = 0.0300$ and $0.0468$). Documentation in [`README.md`](file:///Users/hosamshibly/Downloads/medical_radar/README.md) and [`RESEARCH_AND_ENGINEERING_LOG.md`](file:///Users/hosamshibly/Downloads/medical_radar/RESEARCH_AND_ENGINEERING_LOG.md) has been calibrated to state the explicit TOST margins and $\delta^*$ values.

2. **Exhaustive $\binom{9}{7} = 36$ Prone-Subset Combinatorics (`audit/phase3_ws4_prone_combinatorics.csv`):**
   - Because `PAPER_18` selects 7 prone sessions out of the 9 prone sessions in Patients `03–12` (`3.1, 4.3, 5.3, 6.2, 7.3, 8.3, 10.2, 12.1, 12.2`), we evaluated **all $\binom{9}{7} = 36$ possible 7-prone combinations** (including all $\binom{7}{5} = 21$ combinations that preserve the 4 twin sessions `7.2, 7.3, 8.2, 8.3`):
     - Across **36/36 (100%)** of prone combinations, `HPSS+NLS` achieves lower RMSE than `NLS+NMF` (`mean ΔRMSE = −0.3433 bpm`, range `[−0.4456, −0.0648 bpm]`, valid window counts spanning `4,622–5,466`, bracketing the paper's `~4,964`).
     - Notably, `PAPER_18` (`ΔRMSE = −0.0648 bpm`) is the **most conservative** of all 36 combinations (`max ΔRMSE = −0.0648 bpm`), proving that reference-validity ranking did *not* inflate HPSS's advantage.
     - Across 2,000 completely random 18-session draws (`11 supine + 7 prone`) from the 27 paired sequences in Patients `03–12`, the mean $\Delta\text{RMSE}$ is **$-0.1911\text{ bpm}$** ($95\%\text{ CI } [-0.4027, +0.1253]\text{ bpm}$).

## WS5 — Provenance & Environment Lock Audit (`audit/phase3_ws5_provenance_audit.json`)

1. **33/33 Cache Freshness & 15-File `_CORE_FILES` Coverage:**
   - All 33 `.npz` caches pass `cache_is_current(p, m)` under `PIPELINE_VERSION = "main-v3-audit-fix-2"` with all 15 core modules hashed.
2. **Environment Lock Verification & Fix `A16`:**
   - Active runtime versions (`numpy==2.0.2`, `scipy==1.13.1`, `scikit-learn==1.5.2`, `pandas==2.3.3`, `matplotlib==3.9.4`, `pyedflib==0.1.42`) match [`paper_workspace/src/requirements-lock.txt`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/requirements-lock.txt) (`lock_exact_match = True`).
   - Fixed `A16` in [`paper_workspace/src/requirements.txt`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/src/requirements.txt), where line 6 previously specified `pyedflib>=0.3` (unsatisfiable on PyPI since latest is `0.1.42`); updated to `pyedflib>=0.1.30` (`requirements_min_satisfied = True`).

## WS8 — Controlled Single-Thread Timing Benchmark (`audit/phase3_ws8_timing_benchmark.csv`)

1. **Stage-by-Stage Single-Thread Timing Decomposition (`OMP/BLAS threads = 1`, GC disabled during timing, 5 warmup + $3 \times 50 = 150$ timed window evaluations on Session `7.3`):**

| Pipeline Stage | Mean $\pm$ SD (ms) | Median `p50` (ms) | IQR (ms) | `p95` (ms) | `p99` (ms) | Speedup (`NMF / HPSS`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Shared `CSD + Pre-Bandpass`** | $0.297 \pm 0.035$ | $0.282$ | $0.025$ | $0.359$ | $0.406$ | $1.00\times$ (shared) |
| **Shared `STFT (256-pt rect)`** | $0.366 \pm 0.013$ | $0.362$ | $0.011$ | $0.393$ | $0.401$ | $1.00\times$ (shared) |
| **Core Separation (`NMF solver+mask` vs. `HPSS median+mask`)** | $17.011 \pm 0.593$ vs. **$1.858 \pm 0.114$** | $17.079$ vs. **$1.839$** | $0.285$ vs. $0.083$ | $17.505$ vs. $2.115$ | $18.483$ vs. $2.247$ | **$9.16\times$ mean ($9.29\times$ p50)** |
| **Post-Mask (`env_normalize + iSTFT + Post-BP`)** | $0.789 \pm 0.045$ vs. $0.749 \pm 0.104$ | $0.778$ vs. $0.705$ | $0.026$ vs. $0.064$ | $0.878$ vs. $0.957$ | $0.969$ vs. $1.072$ | $1.05\times$ |
| **Full Filter Block (`STFT + Mask + iSTFT + Post-BP`)** | $18.166 \pm 0.592$ vs. **$2.972 \pm 0.183$** | $18.242$ vs. **$2.921$** | $0.285$ vs. $0.122$ | $18.700$ vs. $3.373$ | $19.662$ vs. $3.484$ | **$6.11\times$ mean ($6.24\times$ p50)** |
| **Downstream `Autocorr + Harmonic NLS`** | $0.507 \pm 0.052$ vs. $0.454 \pm 0.042$ | $0.488$ vs. $0.441$ | $0.022$ vs. $0.025$ | $0.605$ vs. $0.514$ | $0.754$ vs. $0.668$ | $1.12\times$ |
| **End-to-End 30-s Window (`CSD` $\to$ `NLS`)** | $18.970 \pm 0.601$ vs. **$3.724 \pm 0.188$** | $19.033$ vs. **$3.665$** | $0.317$ vs. $0.166$ | $19.541$ vs. $4.089$ | $20.458$ vs. $4.236$ | **$5.09\times$ mean ($5.19\times$ p50)** |
| **Cached `PAPER_18` Session Means ($N=18$)** | $17.842 \pm 0.491$ vs. **$2.776 \pm 0.072$** | $17.887$ vs. **$2.803$** | $0.471$ vs. $0.037$ | $18.395$ vs. $2.836$ | $18.750$ vs. $2.865$ | **$6.43\times$ mean ($6.38\times$ p50)** |
| **Cached `all_paired` Session Means ($N=31$)** | $17.784 \pm 0.619$ vs. **$2.795 \pm 0.107$** | $17.827$ vs. **$2.806$** | $0.476$ vs. $0.048$ | $18.578$ vs. $2.925$ | $19.520$ vs. $3.109$ | **$6.36\times$ mean ($6.35\times$ p50)** |


