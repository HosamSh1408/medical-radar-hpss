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
python3 -m paper_workspace.src.test_pipeline          # 15/15 checks
python3 -m paper_workspace.src.run_batch --all-paired --force   # rebuild 33 caches
python3 main.py --multiverse                          # CSV + ledger
```
