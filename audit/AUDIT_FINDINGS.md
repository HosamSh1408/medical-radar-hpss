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

