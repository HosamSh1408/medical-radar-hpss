---
name: thesis-experiment-runner
description: >-
  Executes WP1-WP5 thesis experiments, 12-fold Leave-One-Patient-Out (LOPO) evaluations,
  mandatory invariance and negative controls, statistical tests, and provenance-locked JSON
  exports into paper_orchestra_bundle. Use when designing, running, benchmarking, or
  debugging any experimental pipeline, ablation study, or statistical evaluation.
---

# Thesis Experiment Runner

Standard workflow for executing `EXP-0` through `EXP-8` experiments with zero data leakage and reproducible provenance.

## Execution Environment
- **Dual Python Runtime Contract:**
  - **Frozen Baseline (`main-v3-audit-fix-2`):** Invoke `/usr/bin/python3` (`Python 3.9.6`, matching `paper_workspace/src/requirements-lock.txt`) when verifying `paper_workspace.src.test_pipeline` (`19/19 PASS` against locked `.npz` caches). Note: `/usr/bin/python3` resolves packages from user site-packages outside the agent sandbox (`BypassSandbox: true` if invoked via agent).
  - **New Thesis Modules (`WP1–WP5`):** Invoke `/opt/homebrew/bin/python3.10` (`Python 3.10.19`) with `MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` for all new `thesis/` pipelines, statistics, and `.npz` read-only consumers:
    ```bash
    MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /opt/homebrew/bin/python3.10 -m <module>
    ```
- **Statistical Cross-Checks:** Use `/usr/local/bin/Rscript` (`R 4.5.1`; `lme4` and `boot` installed; `rmcorr`/`TOSTER` optional) when running secondary mixed-effects/repeated-measures verification.

## Experimental Protocol Checklist
1. **Cohort Stratification Hierarchy:**
   - Primary confirmatory cohort: `pool30_sync` ($N=28$ synced sessions) and `all_paired` ($N=31$ evaluable sessions, $12$ patients).
   - Sensitivity cohort: `retained23` ($N=23$).
   - Historical comparison anchor only: `PAPER_18` ($N=18$ sessions, $5,466$ valid windows).
   - External benchmark (`L4`): `external_data/gleichauf2024/` (zero training/tuning on external recordings).
2. **Splitting & Synchronization Guardrails:**
   - Optimize parameters (ellipse windows, directional slopes, Viterbi transitions, SQI weights) strictly inside the 11 training patients per **LOPO fold (`Patient 1..12`)**.
   - Lock per-session radar-to-EDF lag $\hat{\tau}_s$ once via branch-neutral anchoring (`evaluation.sync_arrays` / metadata offset); never re-tune lag per candidate estimator.
3. **Mandatory Pillar-IV Control Suite:**
   Every proposed estimator (`WP1`–`WP4`) MUST pass all five checks before advancing a gate:
   - Constant baselines (`const_loso`, `const_global_oracle`, stiff-HMM control).
   - Complex White Gaussian Noise (`N=200`, fixed seed `20261011`) abstention / AUROC check.
   - Time-shuffled / phase-scrambled surrogate degradation check.
   - Mathematical invariances (`< 1e-6` bpm deviation under scaling, phase rotation, I/Q sign flip, channel swap).
   - Deterministic synthetic tone + chirp recovery check (`< 0.1` bpm error).
4. **Statistical Reporting Standard:**
   - Never run window-level i.i.d. p-values over overlapping 2-s windows.
   - Compute per-session metrics (`RMSE_sess`, `MAE_sess`, `Acc@3/6/10`, `r_within`), pooled-window context (`RMSE_pool`), `>10 bpm` tail fraction, repeated-measures Bland–Altman LoA, and **10,000-resample patient-clustered BCa bootstrap 95% CIs** with Holm–Bonferroni correction or TOST (`TOST_MARGIN = 0.50`).
5. **Provenance Export & Validation:**
   - Export locked numerical outputs directly to `paper_orchestra_bundle/raw_materials/metrics_chX.json` (with input file hashes and Python/git version).
   - Verify package integrity (`test_pipeline.py` unit checks) and inspect all generated CSV/JSON tables before invoking figure generation or `PaperOrchestra`.
