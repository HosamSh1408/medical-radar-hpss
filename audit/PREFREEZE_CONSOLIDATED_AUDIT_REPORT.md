# Pre-Freeze Consolidated Audit Report — BTU Cottbus M.Sc. AI Master's Thesis

**Workspace:** `/Users/hosamshibly/Downloads/medical_radar`  
**Git Anchor:** `681de7e` (`HEAD -> thesis/main`)  
**Audit Date:** `2026-11-23`  
**Reference Contracts:** `.agents/rules/thesis-governance.md`, `paper_orchestra_bundle/raw_materials/notation_glossary.tex`, `thesis/analysis_plan_v3_ai.md` (including Section 10 Stage 4 pre-registration), `paper_workspace/metadata/claim_gate_status.json`, `paper_workspace/metadata/unresolved_protocol_items.yaml`, and `audit/postfix_manifest.sha256`

---

## Executive Summary & Freeze-Readiness Verdict

### **VERDICT: `PASS` (Freeze-Ready for LaTeX / PaperOrchestra Assembly)**

- **Blocking Defects:** **`0`**
- **Consolidated Non-Blocking Findings:** **`14`** non-blocking errata / methodological caveats (`PF-01`–`PF-14`: `11` non-blocking errata, `3` non-blocking methodological caveats) + **`1`** sandbox environment note (`PF-15`), deduplicated and reconciled across **Axis A** (`A1`–`A6`), **Axis B** (`B-1`–`B-6`), **Axis C** (`ERR-C1`–`ERR-C7`), and **Phase 2 Adversarial Challenge** (`P2-1`–`P2-4`).
- **Companion Machine-Readable Table:** [`audit/PREFREEZE_DISCREPANCY_TABLE.csv`](PREFREEZE_DISCREPANCY_TABLE.csv) (`15` data rows `PF-01`–`PF-15`, `10` RFC-4180 columns, 1-to-1 parity with Section 5).

### Acceptance Criteria Status (`ORIGINAL_REQUEST.md`)

| Acceptance Criterion | Status | Evidence Summary |
| :--- | :--- | :--- |
| **1. `paper_workspace.src.test_pipeline` under `/usr/bin/python3` (Python 3.9.6)** | **PASS** | **19/19** baseline tests passed (`ALL TESTS PASSED`, exit code `0`, `14.159 s`) under `DEVELOPER_DIR=/Library/Developer/CommandLineTools PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m paper_workspace.src.test_pipeline` (verified in Axis A; zero code changes since). |
| **2. `pytest --rootdir=.` on `tests/` under `/opt/homebrew/bin/python3.10` (Python 3.10.19)** | **PASS** (with `tests/` path per Dual Python Contract) | `pytest --rootdir=. tests/ -v` passes **35/35** (`42.69 s`, exit code `0`, 0 failures/warnings) across all 6 thesis suites (`test_wp0_pipeline.py` 6/6, `test_viterbi_reproduction.py` 9/9, `test_stage1_parsimony.py` 7/7, `test_stage2_complexity.py` 5/5, `test_stage3_uncertainty.py` 5/5, `test_stage4_external.py` 3/3). Running bare `pytest --rootdir=.` without `tests/` collects `paper_workspace/src/test_pipeline.py` under Python 3.10.19 (`49 passed, 5 failed` solely due to `cache_is_current()` enforcing Python 3.9.6; logged as `PF-01`). |
| **3. SHA-256 manifests & baseline immutability** | **PASS** | **`154/154`** frozen baseline files (`paper_workspace/` 78/78 + `beltrao_data/` 76/76) and **`33/33`** locked `v1` `.npz` caches in `audit/postfix_manifest.sha256` verify intact (`177/182` total; the 5 diffs are documented Phase 5/P0 updates in `6b19fdd`/`57e7cc0`). **`49/49`** `external_data/gleichauf2024/` files match `extracted_manifest.json`. `thesis/outputs/stage3_uncertainty/frozen_stage4_models.npz` verifies intact at SHA-256 `af33a83e157f7d2efb0f47e50c9544ee19bacb7da88ab0e0235ccead029cbcc4`. Zero pre-existing workspace files modified. |
| **4. Cross-artifact, statistical, leakage, conformal, notation, claim-gate & Nuances #1–#5 reconciliation** | **PASS** | Cross-stage numerical parity holds to `<= 1.78e-15`; Stage 3 NPZ recomputation holds to `<= 5.33e-15` (risk-coverage) and `0.000e+00` (`C4`/`C5`); Holm-Bonferroni recomputations across Stages 0–3 primary families match to `0.00e+00`; strict patient-LOPO anti-leakage verified; all 13 dual `.pdf`/`.png` figures, 71/71 `seed_references.bib`–`lit_review_matrix.csv` entries, all 41 pre-existing `audit/` files, and all 5 known nuances (`R4.1`–`R4.5`) reconciled with zero `L2`/`L5` violations. |
| **5. Consolidated deliverable & explicit `PASS`/`FAIL` verdict** | **PASS** | Delivered cleanly in `audit/PREFREEZE_CONSOLIDATED_AUDIT_REPORT.md` and `audit/PREFREEZE_DISCREPANCY_TABLE.csv` with zero pre-existing files altered. |

---

## 1. Baseline, Dual-Python & SHA-256 Integrity Verification

### 1.1 Dual Python Contract Execution
1. **Locked Baseline Suite (Python 3.9.6):**
   - Command: `DEVELOPER_DIR=/Library/Developer/CommandLineTools PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m paper_workspace.src.test_pipeline`
   - Outcome: **19/19 tests PASS** (`ALL TESTS PASSED`, exit code `0`, `14.159 s`).
   - Operational note (`PF-15`): Under macOS sandbox permissions on `/var/db/xcode_select_link`, `/usr/bin/python3` and `/usr/bin/git` require `DEVELOPER_DIR=/Library/Developer/CommandLineTools`.
   - Side effect (`PF-03`): `test_phase4_ws9_ws10_adversarial_and_traceability` (`paper_workspace/src/test_pipeline.py:458`) calls `run_ws10_traceability_audit(str(tr_json))`, touching the mtime of `audit/phase4_ws10_traceability_audit.json` while preserving an identical SHA-256 (`880b9a32...`).
2. **Thesis Suite (Python 3.10.19):**
   - Command (`tests/`): `PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/homebrew/bin/python3.10 -m pytest --rootdir=. tests/ -v`
   - Outcome: **35/35 tests PASS** (`42.69 s`, exit code `0`, 0 failures, 0 skipped, 0 warnings):
     - `tests/test_wp0_pipeline.py`: `6/6`
     - `tests/test_viterbi_reproduction.py`: `9/9`
     - `tests/test_stage1_parsimony.py`: `7/7`
     - `tests/test_stage2_complexity.py`: `5/5`
     - `tests/test_stage3_uncertainty.py`: `5/5`
     - `tests/test_stage4_external.py`: `3/3`
   - Bare `pytest --rootdir=.` behavior (`PF-01`): Because no `pytest.ini`, `pyproject.toml`, `setup.cfg`, or `tox.ini` exists at repo root, running bare `pytest --rootdir=.` collects `54` items (`35` in `tests/` + `19` in `paper_workspace/src/test_pipeline.py`), resulting in **`49 passed, 5 failed` (`50.18 s`)**. All 5 failures (`test_golden_cohort_counts` at `test_pipeline.py:201`, `test_golden_paper18_metrics` at `:217`, `test_golden_multiverse_csv_matches_caches` at `:247`, `test_phase1_ws1_ws2_cleanroom_and_symmetry` at `audit_phase1_ws1_ws2.py:344`, `test_phase3_ws4_ws5_ws8_tost_provenance_and_timing` at `:405`) arise intentionally from `provenance.cache_is_current()` enforcing Python 3.9.6 + `requirements-lock.txt` on locked `v1` caches.
   - Side effect & coverage notes (`PF-03`): `tests/test_viterbi_reproduction.py:355` (`test_v2_schema_cohort_scatter_and_cleanroom_audit`) invokes `AUDIT.run_audit()`, overwriting `audit/viterbi_godmode_audit.md` (`thesis/audit_viterbi_repro.py:729`) during test execution. Additionally, Stage 1–4 JSON contract tests assert against pre-computed `audit/stage*_godmode_audit.json` rather than recomputing Holm-Bonferroni, per-model `conformal_transfer`, or negative-`dt` decimation directly inside unit tests.

### 1.2 SHA-256 Manifests & Artifact Checksums
- **`audit/postfix_manifest.sha256` (`184` lines = `2` header comments + `182` file entries; `PF-02`):**
  - **Frozen baseline subset (`paper_workspace/` `78/78` + `beltrao_data/` `76/76`):** **`154/154` matched** (`bad_files = []`).
  - **Locked `v1` cache subset (`paper_workspace/outputs/cache/*.npz`):** **`33/33` matched**.
  - **Full manifest:** **`177/182` matched**; the `5/182` mismatches are all outside `paper_workspace/` and `beltrao_data/` and trace to legitimate commits *after* Phase 4 (`9c7ffc9`):
    1. `README.md` (`c37dafb05c29...` -> `dd11c6603284...`, updated in `6b19fdd`)
    2. `RESEARCH_AND_ENGINEERING_LOG.md` (`b7ef2bc8f927...` -> `526d966768a5...`, updated in `6b19fdd`)
    3. `audit/AUDIT_FINDINGS.md` (`ce80923ef98d...` -> `8a7495cec686...`, updated in `6b19fdd`)
    4. `audit/DOC_TRUTH_TABLE.csv` (`8cf50cf05e05...` -> `50a0cac7c1b9...`, updated in `6b19fdd`)
    5. `.gitignore` (`91e497c8279c...` -> `29bca15d3098...`, updated in `57e7cc0`)
- **`external_data/gleichauf2024/extracted_manifest.json`:** **`49/49` matched** (`0` mismatches, `3.577 GB`).
- **Stage 3/4 Frozen Model Archive (`thesis/outputs/stage3_uncertainty/frozen_stage4_models.npz`):**
  - Verified SHA-256: **`af33a83e157f7d2efb0f47e50c9544ee19bacb7da88ab0e0235ccead029cbcc4`**
  - Matches `metrics_stage3_uncertainty.json` (`frozen_stage4_models_sha256`), `metrics_stage4_external.json` (`frozen_stage4_archive_sha256`), `stage3_uncertainty_report.md:97`, `stage3_godmode_audit.json`, `stage4_external_report.md:4`, `stage4_godmode_audit.json`, and `thesis/analysis_plan_v3_ai.md:197` verbatim.
- **Stage 0/Viterbi Module Hashes (`metrics_viterbi_repro.json`):** `tracker_sha256` (`a3498c36...`) matches `thesis/viterbi_tracker.py`; `runner_sha256` (`a35d6fbda...`) matches `thesis/exp_viterbi_reproduction.py`.

### 1.3 `thesis/outputs/` Inventory (`693.19 MB`) & Git Hygiene
- **`thesis/outputs/` breakdown (5 stage directories; `EXP-0` writes directly to `audit/forensics_*` and bundle):**
  - `viterbi_repro/`: **`140.67 MB`** (33 `.npz` caches, `schema_version="viterbi-repro-v2"`)
  - `stage1_parsimony/`: **`146.37 MB`** (31 session `.npz` excluding zero-valid `1.3`/`5.1`, plus 28 `surrogates_5seed/*.npz`, `schema_version="stage1-parsimony-v3"`)
  - `stage2_complexity/`: **`340.63 MB`** (31 `.npz` + 28 surrogates + `pool30_predictions.npz` `3.34 MB`; caches carry `stage2-complexity-v2` and stale `diff_sha256="ea2a492a..."` vs current `diff_hmm_and_foils.py` `3bcfa437...` skipped by `exp2_stage2_complexity.py:105`, with live spot-check on Session `7.3` confirming `0.0` float32 max-abs diff on `phi_h0` and `phi_h5`, logged as `PF-05`)
  - `stage3_uncertainty/`: **`6.60 MB`** (`frozen_stage4_models.npz` + `pool30_uncertainty_and_conformal.npz` `6.66 MB`)
  - `stage4_external/`: **`58.92 MB`** (20 scenario `.npz` caches + `gleichauf20_predictions_and_uncertainty.npz` `4.72 MB`)
- **Git working tree (`PF-04`):**
  - Branch `thesis/main` @ `681de7e`.
  - `3` modified tracked files: `paper_orchestra_bundle/raw_materials/lit_review_matrix.csv`, `paper_orchestra_bundle/raw_materials/seed_references.bib`, `thesis/io_utils.py`.
  - Pre-existing untracked (`??`): `thesis/*.py` + `thesis/analysis_plan_v3_ai.md`, `thesis/outputs/` (`693.19 MB`, not in `.gitignore`), `tests/test_viterbi_reproduction.py` + `tests/test_stage{1..4}_*.py`, `paper_orchestra_bundle/raw_materials/metrics_*.json` (5 files) + `figures/` (10 PDF/PNG pairs), and `audit/` (21 Stage 0–4 reports/CSVs/JSONs), plus **2 new consolidated audit deliverables** (`audit/PREFREEZE_CONSOLIDATED_AUDIT_REPORT.md`, `audit/PREFREEZE_DISCREPANCY_TABLE.csv`).
  - Ignored/bytecode (`!!`): `.DS_Store`, `.claude/`, `.pytest_cache/`, `paper_workspace/src/__pycache__/*.cpython-310.pyc` (from Python 3.10 `pytest` collection).

---

## 2. Cross-Stage Numerical, Statistical, Conformal & Anti-Leakage Audit (WP0 & Stages 0–4)

### 2.1 Cross-Stage Numerical Parity & Stage 3 NPZ Recomputation
- **Locked `R0_HPSS_NLS_Locked` / `A4_HPSS_NLS` parity (`br_1hz_30s`, `branch_neutral_raw_nls_lag`) across `WP0`, `Stage 0`, `Stage 1.1`, `Stage 2.2`, `Stage 3 (c_cov=1.0)`:**
  - `pool30_sync`: `7.817309063606 bpm` (`max_diff = 0.00e+00`)
  - `retained23`: `7.973471981087 bpm` (`max_diff = 8.88e-16`)
  - `PAPER_18`: `7.135087890250 bpm` (`max_diff = 1.78e-15`)
- **`const_loso` parity across `WP0`–`Stage 2.2`:** `8.095849818985` (`pool30_sync`), `8.566345630397` (`retained23`), `8.202759388777` (`PAPER_18`), `max_diff = 0.00e+00`.
- **Inter-stage model parity:** `Stage 1.1` vs `Stage 2.2` (`R0`, `R1a`, `R1f`, `R1g`) and `Stage 2.2` vs `Stage 3 (c_cov=1.0)` (`R1a`, `R2e`, `R2a`, `R2d`) match across all 3 cohorts within **`< 1e-12`**.
- **Stage 3 raw NPZ recomputation (Phase 2 Challenger against `pool30_uncertainty_and_conformal.npz` + `pool30_predictions.npz`):**
  - `R2e` `U2_Posterior_SD` Risk-Coverage (`sess`/`pool` `AURC`, `E-AURC`, `AUGRC` across `1 Hz` and `Trig`): **`max_diff <= 5.33e-15`** (`0.000e+00` on `1 Hz` `sess AURC` and `sess AUGRC`).
  - `R2e` `C4`/`C5` empirical coverage (`alpha in {0.05, 0.10, 0.20, 0.30}`): **`max_diff = 0.000e+00`** for both `empirical_coverage_patient_mean` and `empirical_coverage_pooled`.

### 2.2 Headline Progression & External Verification (`Gate G1–G5`)
- **Stage 1 (`pool30_sync`, `br_1hz_30s`):** `R0_HPSS_NLS_Locked` (`7.8173 bpm`) -> `R1a_Online_Forward_MMSE_0s` (`6.9904 bpm`, `0 s` causal) -> `R1f_Offline_FB_MMSE_Inf` (`6.3952 bpm`), `R1g_Offline_Viterbi_Unpinned` (`6.4978 bpm`).
- **Stage 2 (`pool30_sync`, `br_1hz_30s`):** `R2e_Causal_DiffHMM_FilterSafe8` (`6.8486 bpm`, 8 interpretable parameters, `H0`) improves over `R0` and `R1a` while maintaining exact `O(2) x R_{>0}` (`H5`) / `SO(2) x R_{>0}` (`H0`) equivariance and `0.0` causality drift. Black-box foils (`R3c` `5.7985`, `R3a` `6.0751`, `R3b` `6.5844`) achieve lower internal RMSE via cohort-mean shortcutting.
- **Stage 3 (`Gate G4: PASS_AND_HARD_FREEZE_WP1_TO_WP4`):** `U2_Posterior_SD` selective prediction achieves `sess_eaurc_rnd = -0.245`, `wp_eaurc_rnd = -0.405` (`p_wilcoxon_sess_holm = 0.01364`, `p_wilcoxon_patient_holm = 0.07812`).
- **Stage 4 (`Gate G5: PASS_L4_ZERO_TOUCH_TRANSFER_VERIFIED`, `external_data/gleichauf2024/`):** On `inband_ge28bpm` (`N=14`), `R2e` achieves `rmse_scen = 0.0197 bpm` (`slope = 0.9994`) vs `R2a` `0.0189` (`1.0004`), `R2d` `0.0383` (`0.9985`), `R1a` `0.0300` (`0.9991`), cleanly defeating all black-box foils (`R3a` `3.1782`, `slope=0.8623`; `R3b` `3.5022`, `0.9293`; `R3c` `3.6353`, `0.7090`). `U6b_ZeroTuned_Geometric_SQI` reaches stopband-vs-inband OOD AUROC **`0.9998`** (`> 0.90`).

### 2.3 Statistical Testing, Anchor Guard & Anti-Leakage Controls
- **Holm-Bonferroni Recomputation:** Independent read-only recomputation across all confirmatory Stage 0 (`4` families, `m=21`), Stage 1 (`wp1_ablation` `m=6`, `wp2_ablation` `m=6`, `rung1_lopo.paired_vs_R0` `m=9`, `paired_vs_const_loso` `m=10`), Stage 2 (`paired_vs_R0` `m=12`, `paired_vs_R1a` `m=9`, `paired_vs_const_loso` `m=13`), and Stage 3 (`U1..U6c`, `m=8`) families matches stored JSON p-values to **`0.00e+00`**. All `N=11` patient-level Wilcoxon p-values respect the exact discrete floor `2 / 2^11 = 0.0009765625`.
  - **Omission (`PF-10`):** `metrics_stage1_parsimony.json` `rung1_lopo.paired_vs_R1g_offline_viterbi` (`n=8`) stores raw `p_wilcoxon_sess`, `p_wilcoxon_patient`, and `tost_*` without `p_wilcoxon_sess_holm` / `p_wilcoxon_patient_holm`.
- **Stage 1 `R1h`/`R1i` Anchor Check (`exp1_stage1_parsimony.py:618`):** `_build_rung1_lopo` uses `anchor_r0` (`H0`, `r_within = 0.422904`) as the `rmcorr >= 0.95 * base_r` guard anchor even for `R1h` (`H5`) and `R1i` (`H6`). Phase 2 counterfactual re-execution using own `H5`/`H6` anchors (`0.384252` / `0.409607`) showed **`0/11` LOPO folds differ (`delta_rmse = +0.000000 bpm`)**.
- **`TOST`, `rmcorr_ancova`, `patient_clustered_bootstrap`:** `TOST` (`+-0.50 bpm`) on `WP2 H5 vs H0` yields `p=0.4139` (sess) / `0.5190` (pat) on `1 Hz`, correctly retaining `H0`. `rmcorr_ancova` (`io_utils.py:228-264`) demeans by session (`S=28`, `df=6611`, matching `\rWithin` in `notation_glossary.tex`), and `fast_patient_clustered_bootstrap` preserves the exact ratio-of-sums equal-session estimator (`io_utils.py:306-330`).
- **Patient-LOPO Anti-Leakage & `R3a`/`R3b` Refit Asymmetry (`PF-09`):** All scalers, ridge/MLP/TCN selectors (`diff_hmm_and_foils.py:816-821, 935-942, 1062-1119`), SQI logistic weights (`U6c`), and conformal order statistics fit strictly inside training folds. At the Stage 3 archive freeze (`exp3_stage3_uncertainty.py:768-786`), `R3c` (Ridge) and `R2*` refit on all `11` `pool30_sync` patients, whereas `R3a` (TCN) and `R3b` (MLP) hold out `val_pat_full = 12` for early stopping/selection and freeze weights trained on **`10`** patients (conservative, zero leakage, but diverges from §10.1 prose).

### 2.4 Conformal Prediction Validity & Section 10 Pre-Registration Reconciliation
- **Finite-`K` Vacuity (`wp3_wp4_uncertainty_conformal.py:944-967`):** `finite_sample_conformal_order_index(n_cal, alpha)` computes `ceil((n_cal + 1)(1 - alpha) - 1e-12)`. For `K_cal = 10` (LOPO) and `K = 11` (full cohort), `alpha_min = 1/11 = 0.0909`. At `alpha = 0.05`, `k_idx > n_cal` returns `q_hat = inf` (`is_vacuous = True`, `NaN` in `frozen_stage4_models.npz` index `0`).
- **Two-Level Conformal Construction vs. Dunn et al. (2023) (`PF-08`):** `C4`/`C5` (`wp3_wp4_uncertainty_conformal.py:1110-1133`) compute `np.quantile(..., 1 - alpha)` within each calibration patient and take the `ceil((K_cal+1)(1-alpha))`-th order statistic across patients (`empirical` Stage 3 coverage `0.9756/0.8952/0.7595` for `C4` and `0.9632/0.8636/0.7661` for `C5` at `90%/80%/70%`). Unlike Dunn et al. (JASA 2023) subsampling or CDF-pooling (`>= 1 - alpha`), a two-level quantile construction has a worst-case marginal lower bound of `(1 - alpha)^2` (`0.81/0.64/0.49`) in general finite samples, requiring explicit framing in the thesis.
- **Stage 3 JSON Naming Quirk (`PF-10`):** `wp3_wp4_uncertainty_conformal.py:1140,1192-1195` stores **full** `C5` interval widths (`2 * q * sigma_tilde`) under `"low_uncertainty_half_width_bpm"` (`29.3007`) and `"high_uncertainty_half_width_bpm"` (`49.8223`), where `"half"` denotes the median-split half of windows (`plot_stage3_uncertainty.py:272`).
- **Stage 4 Cross-Model `q_hat` Reuse (`PF-07`):** `exp4_stage4_external.py:826-889` applies `R2e`'s frozen `q_hat` (`C4`: `19.5384, 13.4303, 8.2776`; `C5`: `14.6719, 9.0125, 6.6338`) across `CONFORMAL_MODELS = ("R2e_*", "R2a_*", "R1a_*")`. Phase 2 recomputation confirmed this inflates `R1a`'s `C5` `q_hat_norm` by **`1.816x–2.020x`** (`14.6719` vs `R1a`'s own `8.0812` at `alpha=0.10`), distorting `R1a` `subband_lt28bpm` `C5` coverage (`0.8909, w=81.03` vs own `0.4548, w=44.61`). Because `Gate G5`, `stage4_external_report.md` Table `3B`, `stage4_external_uncertainty_and_conformal` Panel (b), and `stage4_godmode_audit` exclusively evaluate `R2e`, no reported claim or gate is impacted.
- **Section 10 Pre-Registration Reconciliation (`PF-11`, `PF-14`):** `analysis_plan_v3_ai.md` §10 matches Stage 4 execution on primary (`K=11`) vs sensitivity (`LOPO-Mean`), `14+6=20` strata, 64-tap Kaiser FIR, decimation, laser GT, and window regimes. Minor deviations: (i) §10.1 (`line 197`) writes `frozen_r2*_full_refit_theta` whereas `.npz` keys omit `frozen_` (`r2*_full_refit_theta`); (ii) `R3a`/`R3b` train on 10 vs 11 patients (`PF-09`); and (iii) numeric `Gate G5` thresholds (`slope >= 0.95`, `best_r3_rmse > 3.0`, `min_20bpm_r3 > 10.0`, `u6b_auroc > 0.90`) and `sync_pulse` OOD AUROC (`exp4_stage4_external.py:793,914-939`) were operationalized at Stage 4 execution without explicit numeric pre-registration in §10.

---

## 3. Bundle Artifacts, 13 Dual Figures, Notation Glossary, Claim-Gate Governance & Pre-Existing `audit/` Reports

1. **`paper_orchestra_bundle/raw_materials/` JSONs & CSVs:**
   - All `6` bundle JSONs (`metrics_wp0_forensics.json`, `metrics_viterbi_repro.json`, `metrics_stage1_parsimony.json`, `metrics_stage2_complexity.json`, `metrics_stage3_uncertainty.json`, `metrics_stage4_external.json`) pass strict RFC-8259 parsing (`allow_nan=False`), and `metrics_wp0_forensics.json` (`5c4d78f7...`) is byte-identical to `audit/forensics_windows_summary.json`.
   - `forensics_attribution.csv` (`15` covariates) and `idea_dense_ch2_wp0.md` (`20` lines) match `metrics_wp0_forensics.json` exactly. Two prose caveats in `idea_dense_ch2_wp0.md` (`PF-14`): line `3` (`"L4 external ... prohibited"`) reflects WP0 scope prior to Stage 4 verification, and line `11` uses `\rho_{IQ}=5.86` (empirical I/Q SD ratio) not defined in `notation_glossary.tex`.
2. **References (`seed_references.bib` & `lit_review_matrix.csv`):**
   - `71` BibTeX entries (`0` brace errors) match `71` CSV rows 1-to-1 across `bib_key`, `year`, `doi`, `url`, and clusters (`cluster1`: 13, `cluster2`: 12, `cluster3`: 12, `cluster4`: 11, `cluster5`: 14, `cluster6`: 9). Header comment at `seed_references.bib:3-14` still reads `60-Paper` and `5 clusters` (`PF-14`).
3. **All 13 Dual Figures (`paper_orchestra_bundle/raw_materials/figures/`):**
   - `13` `.pdf` (1 page each) + `13` `.png` (300 DPI each), 100% paired and visually verified legible with Okabe-Ito palettes and accurate metrics.
   - **Erratum (`PF-12`):** `thesis/plot_stage4_external.py:84` -> `stage4_external_frequency_transfer.{pdf,png}` Panel (a) legend displays `Ideal f^{est} = f^{ref}` instead of canonical `\hat{f}_b = f^{\mathrm{ref}}` (`\brEst = \brRef` per `notation_glossary.tex:96-97`).
   - **Erratum (`PF-13`):** `exp0_edf256_reference_agreement.pdf`, `exp0_physics_error_drivers.pdf`, and `exp0_session_error_decomposition.pdf` embed `Type3` fonts because `thesis/exp0_forensics.py:43-55` omits `"pdf.fonttype": 42` / `"ps.fonttype": 42` (all 10 Stage 0–4 PDFs emit `Type0` TrueType CID fonts).
4. **`notation_glossary.tex` (`PF-14`):**
   - Strictly enforces `d(t)` vs `x(t)` vs `\hat{d}(t)`, `\rmseSess` vs `\rmsePool`, `\rWithin` vs `\rWithinSessMean`, `M_H` vs `M_{H,\mathrm{band}}`, strict `< \delta`, and `c_{\mathrm{cov}}`. Lines `150, 178-186, 198, 200, 210-211, 217-218, 226` retain `[WP1 TODO]`–`[WP5 TODO]` placeholder tags in Column 4 awaiting final thesis appendix mapping to `thesis/*.py`.
5. **Claim-Gate Governance (`claim_gate_status.json`, `unresolved_protocol_items.yaml`):**
   - Frozen `claim_gate_status.json` (`2026-10-18`) blocks `L2` (`exact_reproduction`) and `L5` (`clinical_superiority`), while `L3+` (`Gate G1–G4`) and `L4` (`Gate G5`) are tracked in `analysis_plan_v3_ai.md`, `metrics_stage3_uncertainty.json`, and `metrics_stage4_external.json`. Regex/manual audit across bundle and `audit/*.md` confirmed **zero** forbidden `L2` or `L5` claims.
6. **All 41 Pre-Existing `audit/` Files:**
   - Inventory verified directly: **`9` `.json`**, **`19` `.csv`**, **`11` `.md`**, **`2` `.sha256`** (`41` total).
   - All `9` JSONs pass strict RFC-8259 parsing (`all_passed: true` on `stage{1..4}_godmode_audit.json`, `all_clean: true` on `phase4_ws10_traceability_audit.json`, `all_33_caches_current: true` on `phase3_ws5_provenance_audit.json`).
   - All `19` CSVs are strictly rectangular (`bad_lens = 0`); `phase2_ws3_session_lags.csv` contains 12 `nan` cells strictly on zero-valid sessions `1.3` and `5.1`.
   - Minor rounding erratum (`PF-02`): `audit/AUDIT_FINDINGS.md:139` displays `7.974` for `retained23` synced `HPSS+NLS` RMSE (rounded from 4-dp CSV `7.9735` in `audit/phase2_ws3_lag_leakage_summary.csv:9`), whereas unrounded `7.973471981087` rounds to `7.973` in `viterbi_repro_reconciliation.md:64` and `idea_dense_ch2_wp0.md:8`.

---

## 4. Explicit Reconciliation of the Five Known Nuances (`R4.1`–`R4.5`)

### 4.1 Nuance #1 — Stale Filter Specification Prose in `metrics_stage3_uncertainty.json`
- **Observation:** `thesis/exp3_stage3_uncertainty.py:1291-1295` -> `metrics_stage3_uncertainty.json` (`gate_g4_verdict.hard_freeze_manifest_for_stage4_l4.stage4_l4_evaluation_protocol`) and `audit/stage3_uncertainty_report.md:106` state `"Within locked 0.40-1.50 Hz (24-90 bpm) 4th-order Butterworth bandpass"` and `"secondary_out_of_passband_low_rate_stratum_bpm": [10.0, 24.0]`.
- **Verification:** Actual locked DSP in `paper_workspace/src/filtering.py` (`F._TAPS`, `FIR_TAPS=64`, `[0.30, 3.00] Hz`, `KAISER_BETA=6.5` double-pass `filtfilt`) yields measured attenuation: `5 bpm = -30.77 dB`, `10 bpm = -22.98 dB`, `18 bpm = -12.05 dB`, `20 bpm = -9.99 dB`, `24 bpm = -6.61 dB`, `28 bpm = -4.12 dB`, `30 bpm = -3.17 dB`, `45 bpm = -0.13 dB`, `80 bpm = -0.00 dB`. Both `analysis_plan_v3_ai.md:200` and `metrics_stage4_external.json` (`hardware_protocol_disclosure` and `14 + 6 = 20` strata) implement this exact Kaiser FIR filter, and zero figures repeat `"Butterworth"`.
- **Disposition:** **Non-blocking erratum (`PF-06`)**. Never cite the stale Stage 3 Butterworth string in `PaperOrchestra` inputs.

### 4.2 Nuance #2 — `paper_workspace/src/hpss.py` (`H0`) Nyquist Reflection Asymmetry
- **Observation:** `paper_workspace/src/hpss.py:37-39` applies `median_filter(mag_s, size=(21, 1), mode="nearest")` across `fftshift` (`NFFT=256`). Index `0` is the unpaired Nyquist bin (`-fs/2`) while indices `1..255` are conjugate pairs; `mode="nearest"` causes asymmetric clamping at `idx=0` vs `idx=255`.
- **Verification:** Under `iq_channel_swap` (`1j * np.conj(dr)`), `H0` exhibits `0.4303` max mask diff at paired bins, reduced by downstream `bandpass()` to `2.578e-05 bpm` (`45BPM_normal`) / `1.158e-05 bpm` (`80BPM_deep`) in Stage 4 and `<= 8.63e-04 bpm` in Stage 2 (`SO(2) x R_{>0} <= 2.84e-14 bpm`). Mirror-symmetric `H5` (`_symmetric_band_masks`) achieves `1.42e-14`–`2.84e-14 bpm` across full `O(2) x R_{>0}`.
- **Disposition:** **Verified physical/mathematical behavior (0 defects)**; explain explicitly in Ch. 3/4 as a frozen `v1` baseline property motivating `H5`'s symmetric positive/negative band filtering.

### 4.3 Nuance #3 — `gleichauf2024` Laser vs. Radar Clock Offsets & Monotonicity
- **Observation:** Across all 20 `gleichauf2024` scenarios, 17 show `laser_t0` 8–21 s before `iq_t0`, while 3 exhibit large clock offsets: `20BPM_normal` (~49.88 days apart), `50BPM_normal` (~3586.45 s / 1 h), and `50BPM_deep` (~3587.89 s / 1 h).
- **Verification:** `exp4_stage4_external.py:187-208` (`extract_laser_steady_rate`) slices strictly in laser time `[t_l[0]+100.0, t_l[0]+700.0]` (`laser_f_mid_bpm` agrees with `f_nom` within `0.04 bpm` across all 20 scenarios: e.g., `19.9781`, `49.9681`), while radar windows (`432/442` steady-state) are gated strictly on radar `t_s` `[10.0 s, dur - 10.0 s]`. Although 11/20 raw `iq_values_*.csv` files have 1–3 negative raw `dt` samples (down to `-3.5183 ms`), `t_ds10 = t[::10]` (`exp4_stage4_external.py:169`) has **`0` non-positive `dt` across all 20 scenarios** (`min_dt10_ms >= +0.9816 ms`), guaranteeing strictly monotonic `np.interp`.
- **Disposition:** **Verified clean (0 defects)**.

### 4.4 Nuance #4 — Frozen `claim_gate_status.json` (`2026-10-18`) vs. `L3+` and `L4` Gates
- **Observation:** `paper_workspace/metadata/claim_gate_status.json` remains frozen at `2026-10-18` (`"generalizable_superiority": "not_evaluable"`), while `thesis/` tracks `Gate G1–G4` (`L3+`) and `Gate G5` (`L4`).
- **Verification:** Per `.agents/rules/thesis-governance.md`, `paper_workspace/` is strictly read-only. Stage 3 (`metrics_stage3_uncertainty.json`: `PASS_AND_HARD_FREEZE_WP1_TO_WP4`) and Stage 4 (`metrics_stage4_external.json`: `PASS_L4_ZERO_TOUCH_TRANSFER_VERIFIED`) extend the governance chain cleanly without mutating baseline metadata or asserting forbidden `L2`/`L5` claims.
- **Disposition:** **Compliant (`PF-02`)**; pair the frozen baseline gate file with Stage 3/4 gate manifests in Ch. 1/7.

### 4.5 Nuance #5 — `frozen_stage4_models.npz` SHA-256 & Stage 4 Consistency
- **Observation & Verification:** `thesis/outputs/stage3_uncertainty/frozen_stage4_models.npz` SHA-256 is verified at `af33a83e157f7d2efb0f47e50c9544ee19bacb7da88ab0e0235ccead029cbcc4`, matching Stage 3 and Stage 4 manifests identically. All 42 arrays (`r2*` parameters, `r3*` weights, `c4_*`/`c5_*` quantiles) match JSON manifests within `< 1e-12`, and index `0` (`alpha=0.05`) `NaN` entries in `c4_*`/`c5_*` correctly reflect finite-`K` vacuity (`alpha_min = 1/11 = 0.0909`).
- **Disposition:** **Verified intact (0 defects)**; note key naming (`r2*_full_refit_theta` without `frozen_`) in `PF-14` and Stage 4 cross-model reuse in `PF-07`.

---

## 5. Consolidated Discrepancy Register & PaperOrchestra Manuscript Safeguards

### 5.1 Unified Discrepancy Table (`PF-01`–`PF-15`, 1-to-1 with `PREFREEZE_DISCREPANCY_TABLE.csv`)

| id | severity | category | source_axes | location | symbol_or_figure | observed | expected | impact_on_gates | proposed_fix |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PF-01** | Non-blocking erratum | Test Discovery & Dual Python | Axis A (`A1`) | Repo root (missing `pytest.ini`/`pyproject.toml`); `paper_workspace/src/test_pipeline.py:201,217,247,405`; `audit_phase1_ws1_ws2.py:344` | `cache_is_current()`, `pytest --rootdir=.` | Bare `pytest --rootdir=.` under Python 3.10.19 collects `paper_workspace/src/test_pipeline.py` and fails 5/54 checks due to `cache_is_current()` enforcing Python 3.9.6. | Dual Python Contract: run `test_pipeline` via `/usr/bin/python3` (3.9.6, 19/19 PASS) and `pytest --rootdir=. tests/` via `/opt/homebrew/bin/python3.10` (3.10.19, 35/35 PASS). | None (both suites pass 100% under their respective Dual Python Contract invocations). | Add `pytest.ini` with `testpaths = tests` and `norecursedirs = paper_workspace beltrao_data external_data`, or mandate `pytest --rootdir=. tests/`. |
| **PF-02** | Non-blocking erratum | SHA-256 Manifests & Governance | Axis A (`A2`), Axis C (`ERR-C7`) | `audit/postfix_manifest.sha256:1-184`; `audit/AUDIT_FINDINGS.md:139`; `paper_workspace/metadata/claim_gate_status.json:1-24` | `postfix_manifest.sha256`, `retained23` RMSE (`7.974` vs `7.973`), `claim_gate_status.json` | `postfix_manifest.sha256` has 182 entries with 177/182 matching (5 modified post-Phase-4 in `6b19fdd`/`57e7cc0`: `.gitignore`, `README.md`, `RESEARCH_AND_ENGINEERING_LOG.md`, `AUDIT_FINDINGS.md`, `DOC_TRUTH_TABLE.csv`); `AUDIT_FINDINGS.md:139` prints `7.974` from 4-dp `7.9735`; `claim_gate_status.json` stays frozen at `2026-10-18`. | Frozen baseline subset (`paper_workspace/` + `beltrao_data/`) is 154/154 intact; unrounded `retained23` RMSE `7.97347198...` rounds to `7.973`; `L3+`/`L4` gates tracked in Stage 3/4 manifests. | None (154/154 baseline + 33/33 `v1` caches + 49/49 `gleichauf2024` match 100%; 0 `L2`/`L5` violations). | Document `154/154` baseline subset vs `182` total in consolidated report, note `7.97347 -> 7.973` rounding, pair frozen `claim_gate_status.json` with Stage 3/4 gate manifests, and generate `thesis_freeze_manifest.sha256` at freeze. |
| **PF-03** | Non-blocking erratum | Test Hygiene & Coverage | Axis A (`A3`), Axis B (`B-6`) | `paper_workspace/src/test_pipeline.py:458`; `tests/test_viterbi_reproduction.py:355`; `tests/test_stage{1..4}_*.py` | `run_ws10_traceability_audit()`, `AUDIT.run_audit()` | Running `test_pipeline` updates mtime of `audit/phase4_ws10_traceability_audit.json` (identical SHA-256); running `pytest tests/` rewrites `audit/viterbi_godmode_audit.md`; Stage 1–4 tests rely on precomputed godmode JSONs without independent Holm, per-model conformal, or jitter decimation unit tests. | Tests should be read-only on `audit/` (or write to `tmp_path`) and directly exercise Holm-Bonferroni, per-model conformal transfer, and negative-`dt` decimation. | None on reported results (all godmode gates 7/7 PASS; Holm recomputed to `0.00e+00` independently). | Add `write_report: bool = False` (default `False` in pytest) to `thesis/audit_viterbi_repro.py`, keep frozen `paper_workspace/` untouched, and add unit tests for Holm, `conformal_transfer`, and timestamp-jitter decimation. |
| **PF-04** | Non-blocking erratum | Git Hygiene | Axis A (`A4`) | Repo root (`.gitignore`, `thesis/outputs/`, `paper_orchestra_bundle/`, `audit/`, `tests/`, `thesis/`) | `git status --short`, `thesis/outputs/` (`693.19 MB`) | 3 modified tracked files (`lit_review_matrix.csv`, `seed_references.bib`, `thesis/io_utils.py`), 48+ untracked code/test/bundle/audit files, `693.19 MB` untracked `thesis/outputs/` not in `.gitignore`, plus `.DS_Store` and `paper_workspace/src/__pycache__/*.cpython-310.pyc`. | Clean `thesis/main` commit tracking thesis code/tests/bundle/audit artifacts while gitignoring heavy binary outputs and cleaning Python 3.10 bytecode from `paper_workspace/src/__pycache__/`. | None on numerical or statistical validity. | Add `thesis/outputs/` to `.gitignore`, commit tracked `thesis/`, `tests/`, `audit/`, and `paper_orchestra_bundle/` files on `thesis/main`, and remove `*.cpython-310.pyc` from `paper_workspace/src/__pycache__/`. |
| **PF-05** | Non-blocking erratum | Cache Provenance | Axis A (`A5`) | `thesis/outputs/stage2_complexity/cache/*.npz`; `thesis/exp2_stage2_complexity.py:105` | `schema_version`, `diff_sha256` | Stage 2 salience caches carry `schema_version=stage2-complexity-v2` and `diff_sha256=ea2a492a...` older than current `thesis/diff_hmm_and_foils.py` (`3bcfa437...`); `exp2_stage2_complexity.py:105` skips `diff_sha256` check for `v2`/`v3`. | Current `schema_version=stage2-complexity-v4` matching `diff_hmm_and_foils.py` hash (`3bcfa437...`). | None (live spot-check on Session `7.3` confirms `0.0` float32 max-abs diff on `phi_h0` and `phi_h5`; Stage 2 godmode 7/7 PASS). | Optionally rebuild `thesis/outputs/stage2_complexity/cache/` with `--force` at final freeze (or decouple salience-feature file hash from trainer hash). |
| **PF-06** | Non-blocking erratum | Filter Metadata (Nuance #1) | Axis B (`B-1`), Axis C (`ERR-C1`) | `thesis/exp3_stage3_uncertainty.py:1291-1295`; `paper_orchestra_bundle/raw_materials/metrics_stage3_uncertainty.json`; `audit/stage3_uncertainty_report.md:106` | `stage4_l4_evaluation_protocol.primary_inband_rationale`, `secondary_out_of_passband_low_rate_stratum_bpm` | Frozen Stage 3 metadata prose says `"0.40-1.50 Hz (24-90 bpm) 4th-order Butterworth bandpass"` and `[10.0, 24.0]` bpm stratum. | Actual locked filter in `paper_workspace/src/filtering.py`, `analysis_plan_v3_ai.md` §10, and `metrics_stage4_external.json` is `64-tap zero-phase Kaiser FIR [0.30, 3.00] Hz (beta=6.5)` with `subband_lt28bpm = {05, 10, 20} BPM`. | None (Stage 4 DSP and strata follow the true Kaiser FIR and exhaustive `14+6=20` partition; zero figures repeat Butterworth). | Do not cite the stale Butterworth prose in `PaperOrchestra` inputs; reference `paper_workspace/src/filtering.py` and `metrics_stage4_external.json`, and patch `exp3_stage3_uncertainty.py:1291-1295` if Stage 3 is re-versioned. |
| **PF-07** | Non-blocking erratum | Conformal Transfer | Axis B (`B-2`), Phase 2 (`P2-1`) | `thesis/exp3_stage3_uncertainty.py:1189,1199`; `thesis/exp4_stage4_external.py:826-889`; `paper_orchestra_bundle/raw_materials/metrics_stage4_external.json` | `conformal_transfer`, `CONFORMAL_MODELS`, `c4_q_hat_bpm`, `c5_q_hat_norm` | `exp4_stage4_external.py` applies `R2e`'s frozen `q_hat` (`C4`: `19.5384/13.4303/8.2776`; `C5`: `14.6719/9.0125/6.6338`) to `R2a` and `R1a`, inflating `R1a` `C5` `q_hat_norm` by `1.816x–2.020x` vs `R1a`'s own Stage 3 quantiles (`8.0812/5.2348/3.2838`) and distorting `R1a` `subband_lt28bpm` coverage (`0.8909` vs `0.4548` at `alpha=0.10`). | Each model evaluated in `conformal_transfer` should use its own Stage-3 calibrated `q_hat_bpm` / `q_hat_norm`. | None on `Gate G5`, `stage4_external_report.md` Table `3B`, `stage4_external_uncertainty_and_conformal` figure, or `stage4_godmode_audit` (all evaluate `R2e` only). | In `PaperOrchestra` manuscripts, report Stage 4 conformal transfer strictly for `R2e` (never cite `conformal_transfer[*][R1a_*\|R2a_*]`); freeze per-model quantiles in `frozen_stage4_models.npz` if Stage 3/4 is re-run. |
| **PF-08** | Non-blocking methodological caveat | Conformal Theory | Axis B (`B-3`) | `thesis/wp3_wp4_uncertainty_conformal.py:944-967,1110-1133` | `finite_sample_conformal_order_index()`, `empirical_order_statistic()`, `C4`/`C5` | `C4`/`C5` compute `np.quantile(1-alpha)` per calibration patient followed by `ceil((K_cal+1)(1-alpha))`-th order statistic across `K_cal` patients (worst-case marginal lower bound `(1-alpha)^2` vs Dunn et al. 2023 subsampling/CDF-pooling `1-alpha` bound); at `alpha=0.05` (`K=10/11`, `alpha_min=1/11=0.0909`), `k_idx > K` yields vacuous `q_hat=inf`/`NaN`. | Exact theoretical characterization distinguishing two-level patient-quantile conformal from Dunn et al. (2023) Theorem 1 subsampling/CDF-pooling, with explicit finite-`K` vacuity disclosure. | None (empirically `C4`/`C5` exceed target coverage at `90%/80%/70%` in Stage 3 and `C5` achieves `1.0000` inband in Stage 4; `alpha=0.05` is explicitly flagged `is_vacuous=True`). | Frame `C4`/`C5` in thesis text as *patient-quantile hierarchical conformal* inspired by Dunn et al. (2023) with finite-`K` vacuity floor `alpha_min = 1/(K_cal+1)` and note the two-level quantile construction. |
| **PF-09** | Non-blocking methodological caveat | Cross-Cohort Training Asymmetry | Axis B (`B-4`), Phase 2 (`P2-3`) | `thesis/exp3_stage3_uncertainty.py:768-786`; `thesis/analysis_plan_v3_ai.md:197` | `val_pat_full`, `r3a_tcn_*`, `r3b_mlp_*`, `r3c_ridge_*` | `R3a` (TCN) and `R3b` (MLP) frozen Stage 4 weights are trained on 10 `pool30_sync` patients (holding out `val_pat_full=12` for early stopping/selection) without refitting on all 11 patients, unlike `R3c` (Ridge) and `R2*` (all 11 patients) and §10.1 prose. | Exact alignment between §10.1 prose and `frozen_stage4_models.npz` training sample counts. | None (zero leakage; conservative for `R3a`/`R3b`; `R3c` refits on all 11 patients and still fails external transfer with RMSE `3.6353` vs `R2e` `0.0197`). | Disclose `val_patient_id=12` (`N_train_patients=10` for `R3a`/`R3b` vs `11` for `R3c`/`R2*`) in Stage 4 tables and §10 reconciliation notes. |
| **PF-10** | Non-blocking erratum | Multiple Testing & Key Naming | Axis B (`B-5`) | `thesis/exp1_stage1_parsimony.py`; `metrics_stage1_parsimony.json`; `thesis/wp3_wp4_uncertainty_conformal.py:1140,1192-1195` | `paired_vs_R1g_offline_viterbi`, `low_uncertainty_half_width_bpm`, `high_uncertainty_half_width_bpm` | `rung1_lopo.paired_vs_R1g_offline_viterbi` (`n=8`) lacks `p_wilcoxon_sess_holm` and `p_wilcoxon_patient_holm` keys; `C5` stores full widths (`2*q*sd`) under `low/high_uncertainty_half_width_bpm` (where `"half"` means median-split half of windows). | Holm-Bonferroni adjusted p-values if citing `paired_vs_R1g` comparisons; unambiguous `full_width` key naming. | None (all primary Stage 0–3 confirmatory families pass Holm verification to `0.00e+00`; `plot_stage3_uncertainty.py:272` labels median-split halves correctly). | Apply Holm step-down correction if citing `R1g` paired p-values in text; clarify in Stage 3 tables/captions that `low/high_uncertainty_half_width_bpm` stores full interval width (`2*q*sigma_tilde`) for each median-split half. |
| **PF-11** | Non-blocking methodological caveat | Pre-Registration vs Execution | Phase 2 (`P2-4`) | `thesis/exp4_stage4_external.py:793,914-939`; `thesis/analysis_plan_v3_ai.md` §10 | `gate_g5_l4_verdict`, `sync_pulse_transient_vs_steady_inband` | `exp4_stage4_external.py` defines numeric `Gate G5` thresholds (`slope >= 0.95`, `best_r3_rmse > 3.0`, `min_20bpm_r3 > 10.0`, `u6b_auroc > 0.90`) and `sync_pulse_transient` OOD AUROC not numerically itemized in §10. | Pre-registered numeric gate thresholds alongside §10 hypotheses. | None (thresholds operationalize §1/§4/§9 hypotheses conservatively; actual values clear margins widely: slope `0.9994`, best `R3` `3.1782`, `U6b` AUROC `0.9998`). | Explicitly distinguish pre-registered §10 directional criteria from operationalized `Gate G5` threshold checks and exploratory `sync_pulse` AUROC in Ch. 7. |
| **PF-12** | Non-blocking erratum | Figure Notation | Axis C (`ERR-C2`) | `thesis/plot_stage4_external.py:84`; `paper_orchestra_bundle/raw_materials/figures/stage4_external_frequency_transfer.{pdf,png}` | Panel (a) legend label | Panel (a) legend displays `Ideal f^{est} = f^{ref}`. | Canonical notation per `notation_glossary.tex:96-97` is `Ideal \hat{f}_b = f^{\mathrm{ref}}` (`\brEst = \brRef`). | None on numerical accuracy (cosmetic legend notation erratum). | Update `thesis/plot_stage4_external.py:84` to use `\hat{f}_b = f^{\mathrm{ref}}` and regenerate `stage4_external_frequency_transfer.{pdf,png}` after freeze. |
| **PF-13** | Non-blocking erratum | PDF Font Embedding | Axis C (`ERR-C3`) | `thesis/exp0_forensics.py:43-55`; `paper_orchestra_bundle/raw_materials/figures/exp0_{edf256_reference_agreement,physics_error_drivers,session_error_decomposition}.pdf` | `_configure_matplotlib()` | The three `exp0_*.pdf` figures embed `Type3` fonts because `_configure_matplotlib()` omits `pdf.fonttype=42` and `ps.fonttype=42` (other 4 plotting scripts emit `Type0` TrueType CID). | All 13 publication PDFs embed `Type0`/`Type42` TrueType vector fonts per IEEE/thesis standards. | None on PNG visual rendering (all 13 PNGs at 300 DPI render cleanly). | Add `'pdf.fonttype': 42` and `'ps.fonttype': 42` to `_configure_matplotlib()` in `thesis/exp0_forensics.py` and re-export the 3 `exp0_*.{pdf,png}` figures after freeze. |
| **PF-14** | Non-blocking erratum | Bundle Text, Glossary & Key Prefixes | Axis C (`ERR-C4`, `ERR-C5`, `ERR-C6`), Phase 2 (`P2-2`) | `seed_references.bib:3-14`; `idea_dense_ch2_wp0.md:3,11`; `notation_glossary.tex:150,178-226`; `analysis_plan_v3_ai.md:197` | `60-Paper` comment, `L4` prohibited, `\rho_{IQ}`, `[WPx TODO]`, `frozen_r2*_full_refit_theta` | `seed_references.bib` header says `60-Paper` / `5 clusters` (actual `71`/`6`); `idea_dense_ch2_wp0.md:3` says `"L4 external ... prohibited"` (pre-Stage-4) and line `11` uses undefined `\rho_{IQ}`; `notation_glossary.tex` retains `[WP1..5 TODO]` tags; `analysis_plan_v3_ai.md:197` writes `frozen_r2*_full_refit_theta` for `.npz` keys that omit `frozen_`. | `71-Paper` / `6 clusters` header; Ch. 2-scoped `L4` disclosure + defined `\rho_{IQ}`; implemented `thesis/*.py` paths in glossary; exact `.npz` key names `r2*_full_refit_theta`. | None (`71/71` bib/csv parity exact; all NPZ arrays match Stage 3/4 JSON manifests within `<1e-12`). | Update `seed_references.bib` header, scope `idea_dense_ch2_wp0.md` to Ch. 2/WP0 and define `\rho_{IQ}` in `PaperOrchestra` inputs, replace `[WPx TODO]` tags in glossary appendix, and note `.npz` key naming omits `frozen_`. |
| **PF-15** | Info (environment) | Sandbox Toolchain | Axis A (`A6`) | `/usr/bin/python3`, `/usr/bin/git` (macOS sandbox) | `xcrun`, `DEVELOPER_DIR` | Sandboxed `/usr/bin/python3` and `/usr/bin/git` fail on `/var/db/xcode_select_link` (`Operation not permitted`) without `DEVELOPER_DIR=/Library/Developer/CommandLineTools`. | Pass `DEVELOPER_DIR=/Library/Developer/CommandLineTools` for `/usr/bin/python3` (3.9.6) and `/usr/bin/git` inside sandboxed environments. | None on codebase or numerical outputs. | Include `DEVELOPER_DIR=/Library/Developer/CommandLineTools` in documented verification commands for sandboxed runners. |

### 5.2 Mandatory Safeguards for `PaperOrchestra` Manuscript Drafting
1. **Never cite Stage 3's stale Butterworth prose (`PF-06`):** Always cite the true locked `64-tap zero-phase Kaiser FIR [0.30, 3.00] Hz (beta=6.5)` (`paper_workspace/src/filtering.py`) and `subband_lt28bpm = {05, 10, 20} BPM` (`metrics_stage4_external.json`).
2. **Never cite `R1a` or `R2a` blocks from `metrics_stage4_external.json["conformal_transfer"]` (`PF-07`):** Report Stage 4 external conformal coverage/widths strictly for `R2e_Causal_DiffHMM_FilterSafe8`.
3. **Frame `C4`/`C5` accurately (`PF-08`, `PF-10`):** Describe `C4`/`C5` as *two-level patient-quantile hierarchical conformal* inspired by Dunn et al. (2023) with finite-`K` vacuity `alpha_min = 1/(K_cal+1)` (`alpha=0.05` vacuous), and clarify that Stage 3 `low/high_uncertainty_half_width_bpm` represents full interval width (`2 * q * sigma_tilde`) on each median-split half of windows.
4. **Disclose `R3a`/`R3b` 10-patient vs `R3c`/`R2*` 11-patient training (`PF-09`) and `Gate G5` operationalization (`PF-11`).**
5. **Enforce strict `L2`/`L5` claim boundaries (`PF-02`, `PF-14`):** Scope `idea_dense_ch2_wp0.md`'s `"L4 prohibited"` remark to Chapter 2 / WP0 (`L4` verified in Ch. 7), maintain zero `L2` (exact Beltrão reproduction) or `L5` (clinical bedside utility) claims, and adhere strictly to `notation_glossary.tex` macros (`\brEst`, `\brRef`, `\rmseSess`, `\rmsePool`, `\rWithin`, `\rWithinSessMean`, `\accTau{\delta}`, `\covLevel`).

---

## 6. Pre-Freeze / Post-Freeze Action Checklist & Verification Commands

### 6.1 Recommended Actions (Non-Blocking; Safe Post-Audit Cleanup Before LaTeX Build)
- **Git & Artifact Hygiene (`PF-01`, `PF-04`):** Add `thesis/outputs/` to `.gitignore`; add `pytest.ini` (`testpaths = tests`, `norecursedirs = paper_workspace beltrao_data external_data`); remove `paper_workspace/src/__pycache__/*.cpython-310.pyc`; commit tracked `thesis/`, `tests/`, `audit/`, and `paper_orchestra_bundle/` files on `thesis/main`; generate `audit/thesis_freeze_manifest.sha256` covering the full thesis freeze.
- **Cosmetic Plot & Text Fixes (`PF-12`, `PF-13`, `PF-14`):** Update `thesis/plot_stage4_external.py:84` (`\hat{f}_b = f^{\mathrm{ref}}`) and re-export `stage4_external_frequency_transfer.{pdf,png}`; add `"pdf.fonttype": 42, "ps.fonttype": 42` to `thesis/exp0_forensics.py:43-55` and re-export the 3 `exp0_*.{pdf,png}` figures; update `seed_references.bib` header (`71-Paper`, `6 clusters`) and resolve `[WPx TODO]` tags in `notation_glossary.tex` for the thesis appendix.
- **Test Hygiene (`PF-03`):** Add `write_report: bool = False` (default `False` in `pytest`) to `thesis/audit_viterbi_repro.py` and add unit tests for Holm-Bonferroni, per-model `conformal_transfer`, and negative-`dt` decimation.

### 6.2 Independent Verification Commands

1. **Frozen Baseline Suite (Python 3.9.6):**
   ```bash
   DEVELOPER_DIR=/Library/Developer/CommandLineTools PYTHONDONTWRITEBYTECODE=1 \
     /usr/bin/python3 -m paper_workspace.src.test_pipeline
   ```
   *(Expected: `ALL TESTS PASSED`, 19/19, exit code `0`.)*

2. **Thesis Suite (Python 3.10.19):**
   ```bash
   PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     /opt/homebrew/bin/python3.10 -m pytest --rootdir=. tests/ -v
   ```
   *(Expected: `35 passed`, exit code `0`.)*

3. **Read-Only SHA-256 & CSV Parity Verification:**
   ```bash
   shasum -a 256 thesis/outputs/stage3_uncertainty/frozen_stage4_models.npz
   # Expected: af33a83e157f7d2efb0f47e50c9544ee19bacb7da88ab0e0235ccead029cbcc4
   ```
   ```bash
   PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.10 -c '
   import csv
   rows = list(csv.reader(open("audit/PREFREEZE_DISCREPANCY_TABLE.csv", newline="", encoding="utf-8")))
   assert len(rows) == 16 and all(len(r) == 10 for r in rows), f"Bad CSV shape: {len(rows)}"
   print("PREFREEZE_DISCREPANCY_TABLE.csv OK:", len(rows) - 1, "rows x", len(rows[0]), "cols")
   '
   ```
