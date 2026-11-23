# Stage 2 Independent Clean-Room Audit Report (`stage2-complexity-v4`)

Executed with zero statistical/LOPO imports from `thesis.exp1_stage1_parsimony`, `thesis.exp2_stage2_complexity`, or `thesis.io_utils`.

**Overall Verdict:** `7/7 GATES PASSED`

## Gate 1 — Frozen Baseline Lock & Stage 1.1 Parity: **PASS**
- Checked `154/154` frozen hashes (`bad: 0`), `33/33` locked caches (`bad: 0`), `external_data/` untouched `True` (`52` files), git checked `False`.
- Exact Stage 1.1 parity (`R0, R1a, R1f, R1g` across 3 cohorts x 2 refs x 5 metrics): `max diff = 0.00e+00`.

## Gate 2 — Clean-Room 4-Way Metric Recomputation & CSV Parity: **PASS**
- Max diff across `13` models x `4` refs x `5` metrics: `2.22e-16`; `pred_sd_ratio` (`1Hz` & `T10`) diff: `0.00e+00`; CSV (`28` rows) diff: `1.14e-05`.

## Gate 3 — Finite-Diff Gradients, Bounds & SVD Ridge Parity: **PASS**
- Central FD (`h=1e-6`) max diff: `R2=9.02e-09`, `TCN=2.19e-07`; `sigma_slow >= 1.0` enforced: `True`; SVD vs `sklearn.Ridge` diff: `9.77e-15`.

## Gate 4 — O(2)xR+ Symmetry, 0.0 Causality & Frequency-Shift Equivariance: **PASS**
- Clean `H5+R2` O(2)xR+ diff: `2.13e-14 bpm`; causality diffs: `R2e=0.0`, `R2a=0.0`, `R2b=0.0`, `TCN=0.0`.
- Operator `+10 bpm` shift (`[-60, 180]`): `R2e=1.42e-14`, `R2a=7.11e-15`, `R2b=0.00e+00`; `[18, 90]` vs `[18, 120]` `< 0.05 bpm`: `True`; calibrated/radar shift probes ok: `True`; two-axis metadata ok: `True`.

## Gate 5 — LOPO Anti-Leakage (`11/10/10`), `sigma >= 1.0` Floor & R3 Tuning Logs: **PASS**
- Disjointness/val errors: `0`; `R2` bound violations: `0`; `R3`/convergence errors: `0`.

## Gate 6 — `-10 s` Lag Alignment, Synthetic Sweep & 5-Seed Surrogates: **PASS**
- `R0 vs Trig (-10 s)` peak lag: `+0 s`; passband `28-80 bpm` mean MAE: `R0=0.05`, `R2e=0.02`, `R2a=0.01 bpm`; 5-seed surrogate max `|r_within|`: `0.0285`.

## Gate 7 — Provenance, Strict RFC-8259 JSON & Completeness: **PASS**
- Schema `stage2-complexity-v4` | strict `allow_nan=False`: `True` | cohort/ref errors: `0` | WP4 fields ok: `True`.

