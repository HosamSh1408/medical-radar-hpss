# Analysis Plan v2 — Pre-Registered Master's Thesis Experimental Protocol

**Status:** Pre-registered lock prior to `WP1` (`v1.0-student-baseline` frozen at commit `6b19fdd`, pipeline `main-v3-audit-fix-2`).  
**Companion Machine Schema:** [`analysis_plan_v2.yaml`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/metadata/analysis_plan_v2.yaml)  
**Predecessor:** [`analysis_plan_v1.md`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/metadata/analysis_plan_v1.md)  
**Frozen Governance Anchors:** [`claim_gate_status.json`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/metadata/claim_gate_status.json), [`unresolved_protocol_items.yaml`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/metadata/unresolved_protocol_items.yaml), [`notation_glossary.tex`](file:///Users/hosamshibly/Downloads/medical_radar/paper_orchestra_bundle/raw_materials/notation_glossary.tex)

---

## 1. Objective & Scope

Evaluate whether a physics-calibrated quadrature front-end (`WP1`), radar-native COLA/directional causal dual-band HPSS (`WP2`), joint cardiorespiratory state-space Viterbi tracking (`WP3`), and HPSS-intrinsic Signal Quality Index (`WP4`) overcome the tracking and abstention limitations of Beltrão et al.'s (2022) NMF and locked `v1` HPSS pipelines across uncurated NICU recordings, while strictly preserving `v1.0-student-baseline` reproducibility and claim boundaries (`L0`–`L5`).

---

## 2. Pre-Registered Cohort Hierarchy & 12-Fold LOPO Contract

| Cohort Tier | Role in Thesis | Sessions ($S$) | Patients | Definition |
| :--- | :--- | :---: | :---: | :--- |
| **`pool30_sync`** | **Single Primary Confirmatory Cohort** | `28` | `11` (`2..12`) | `pool30` excluding zero-valid-window sessions `1.3` and `5.1`; excludes proven `0.0 s` clock-overlap pairs (`1.1, 2.1, 2.2`). |
| **`all_paired`** | Pre-registered widest sensitivity tier | `31` | `12` (`1..12`) | All locally paired evaluable recordings (isolates zero-overlap pairs `1.1, 2.1, 2.2`). |
| **`retained23`** | Pre-registered protocol sensitivity tier | `23` | `10` (`3..12`) | `pool30` minus the 7 named protocol exclusions in `cohort.py`. |
| **`PAPER_18`** | **Historical benchmark anchor only** | `18` | `10` (`3..12`) | Reference-validity-ranked 18-session subset (`5,466` windows). |
| **Non-evaluable / Unpaired** | Preserved missingness ledger | `5` (`1.3, 5.1`; `1.2, 3.2, 10.3`) | — | Never silently dropped; reported explicitly in the cohort flowchart & saturation audit. |

* **Leave-One-Patient-Out (`LOPO`) Integrity:** All tunable parameters (I/Q ellipse conditioning thresholds, directional HPSS ridge slopes, Viterbi transition matrices, and SQI calibration weights) MUST be estimated strictly on the $K-1$ training patients for held-out Patient $k$ (`pool30_sync`: `11` folds over Patients `2..12`; `all_paired`: `12` folds over Patients `1..12`). Zero test-patient reference statistics may enter normalization or model fitting.

---

## 3. Branch-Neutral Synchronization Contract

Per Audit Phase 2 (`WS3`), optimizing synchronization lag separately per candidate estimator leaks evaluation outcomes (e.g., shifting `12.1` from `lag = -116 s` [`18.63 bpm`] to `lag = -21 s` [`3.47 bpm`]). Every `WP1`–`WP5` run locks one shared session alignment lag $\hat{\tau}_s$ before comparing algorithms:

1. **Primary Confirmatory Regime (`branch_neutral_raw_nls_lag`):** Metadata clock offset (`-round(metadata_lag_s) ± 60 s`) refined **once** using frozen `v1` upstream `est_nls` on saturation-masked `evaluation.sync_arrays()` (`lags.py` / `audit/phase2_ws3_session_lags.csv`), held invariant across `F0..F2c`, `HPSS`, `Viterbi`, and `SQI`.
2. **Mandatory Sensitivity Regime (`pure_metadata_clock_lag`):** Zero-search file-header clock offset `-round(metadata_lag_s)` (`0.0 s` correlation search).
3. **Diagnostic Reference (`cached_nmf_lag`):** Retained exclusively in `EXP-0` to bridge cleanly to `v1.0-student-baseline` JSONs (`final_review_independent.json`).

---

## 4. Dual Ground-Truth Reference Strategy (`256 Hz` / `1024 Hz` + `1 Hz`)

1. **Primary Monitor Target (`\brRef`, `RespRate_1Hz_30sAvg`):** `30-s` window average over strictly non-saturated `1 Hz RespRate` windows (`|Resp| < 2.99 V/Ohm` everywhere inside the `30-s` window, identical to `evaluation.window_ref_and_gating`), guaranteeing exact continuity with locked `v1` baselines.
2. **Secondary Physical Waveform & Cardiorespiratory Targets (`thesis/edf_extractor.py`):**
   * **Respiratory Waveform & Breath Triggers:** Direct `256 Hz Resp` (`[0.25, 2.0] Hz` spectral peak) and `1024 Hz Resp.-Trigger` inter-breath interval (`IBI`) median rate over `[12, 120] bpm`, eliminating `1 bpm` integer monitor quantization and flagging true quiet pauses (`> 5 s`).
   * **Cardiac Ground Truth (`[1.0, 6.0] Hz`):** `1024 Hz R-Trigger` inter-beat rate (`[60, 230] bpm`), `256 Hz ECG` envelope crosscheck, and `SignalIQ`-gated `1 Hz HeartRate` for evaluating `WP2b` (`EXP-3`) cardiac ridge SNR and `WP3` (`EXP-4`) 2D coupled tracking.

---

## 5. Locked Estimands, Statistical Inference & Mandatory Controls

* **Error & Agreement Metrics (`notation_glossary.tex`):**
  * Primary error KPI: $\mathrm{RMSE}_{\mathrm{sess}} = \frac{1}{S}\sum_{s=1}^S \mathrm{RMSE}_s$ (equal-session weighting).
  * Secondary error KPIs: $\mathrm{RMSE}_{\mathrm{pool}}$ (explicitly distinguished from $\mathrm{RMSE}_{\mathrm{sess}}$), $\mathrm{MAE}_{\mathrm{sess}}$, $\mathrm{Acc}@3/6/10$ (strict `<`), and `>10 bpm` gross-error tail %.
  * Within-session tracking: Primary **`r_within` (`\rWithin`, Bland–Altman 1995 ANCOVA `rmcorr`)** plus comparative **`r_within_sess_mean` (`\rWithinSessMean`, unweighted session Pearson mean)**.
  * Selective prediction (`WP4`): Risk–Coverage curves ($c_{\mathrm{cov}} \in [0.5, 1.0]$), `AURC`, and Excess-AURC (`E-AURC`).
* **Clustered Non-Parametric Inference:** No window-level i.i.d. p-values across overlapping `2-s` windows. Report `10,000`-resample patient-clustered BCa bootstrap `95% CIs` (`seed = 20261011`), two-sided paired Wilcoxon signed-rank tests on session/patient deltas, TOST at `±0.50 bpm` where equivalence is tested, and **Holm–Bonferroni** adjustment across primary ablation steps.
* **Mandatory 5-Control Pillar (Every Major Stage `WP1`–`WP4`):**
  1. Constant & inertia controls (`const_loso`, `const_global_oracle`, stiff-HMM control).
  2. Complex White Gaussian Noise (`N=200` windows × `480` samples, `seed = 20261011`).
  3. Phase-scrambled radar & time-shuffled reference surrogates.
  4. Mathematical group invariances (`< 1e-6 bpm` under amplitude scaling, global phase rotation $e^{j\pi/3}$, I/Q sign flip, and channel swap).
  5. Deterministic stationary + chirp synthetic recovery (`< 0.1 bpm`).

---

## 6. Pre-Registered Stage Gates & Claim Boundaries

* **Gate `G1` (`W05`, `WP1`):** Winning LOPO calibrator lowers `pool30_sync` $\mathrm{RMSE}_{\mathrm{sess}}$ under `branch_neutral_raw_nls_lag`, improves `11.1`/`12.1`, preserves clean baselines (`7.3`, `4.1`), and passes `< 1e-6 bpm` invariances.
* **Gate `G2` (`W09`, `WP2`):** Quantifies Accuracy–Latency–Runtime Pareto frontier (`lookahead in {0,1,2,4,8} s`); advances **2D `(BR, HR)` Viterbi** in `WP3` only if `EXP-3` cardiac extraction reaches `|err_HR| <= 10 bpm` on `>= 50%` of `SignalIQ`-valid windows (else **1D BR Viterbi + HQI cardiac suppression**).
* **Gate `G3` (`W13`, `WP3` — Confirmatory Efficacy):** Strictly beats `const_loso` (`8.124 bpm` `pool30`) with Holm-adjusted `p < 0.05` and patient-clustered BCa `95% CI < 0`, improves $\bar{r}_{\mathrm{within}}^{(\mathrm{sess})}$ & `r_within`, reduces `>10 bpm` gross errors, and survives phase-scrambled anti-inertia control.
* **Gate `G4` (`W16`, `WP4` — Hard Freeze):** SQI achieves `E-AURC < 0` across $c_{\mathrm{cov}} \in [0.5, 1.0]$, `AUROC >= 0.95` on WGN + `UCD != ChildPresent` + surrogates, and verified quiet-apnea preservation. **All hyperparameters freeze at `G4` prior to `EXP-6`–`EXP-8`.**
* **Claim-Gate Compliance (`L0`–`L5`):** Internal LOPO findings (`L3+`) remain explicitly separated from untouched external benchmark generalization (`L4`, `external_data/gleichauf2024/`), exact Beltrão cohort reproduction (`L2` — permanently blocked), and bedside clinical efficacy (`L5` — out of scope).
