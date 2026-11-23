# Stage 4 Independent Clean-Room Audit Report (`stage4-external-v1`)

Executed with zero statistical, decimation, selective-risk, or conformal imports from `thesis.exp1..4`, `thesis.wp3_wp4_uncertainty_conformal`, or `thesis.io_utils`.

**Overall Verdict:** **7/7 GATES PASSED**

## Gate 1 — Frozen Baseline & `49/49` External `gleichauf2024` SHA-256 Lock: **PASS**
- Checked `154/154` frozen baseline hashes (`bad: 0`), `33/33` locked caches (`bad: 0`), `frozen_stage4_models.npz` SHA-256 `True`.
- Independently hashed all `49/49` files in `external_data/gleichauf2024/extracted_manifest.json` (`bad: 0`, zero external mutation).

## Gate 2 — Exhaustive `14 + 6 = 20` Partition & Clean-Room Laser Ground Truth: **PASS**
- Disjoint exhaustive partition verified (`14` in-band `>= 28 bpm` + `6` sub-band `< 28 bpm` = `20/20`).
- Clean-room `2.5 kHz` laser FFT recomputation on 4 sentinels: `max diff = 0.00e+00` (`max |f_laser - f_nom| = 0.0362 bpm`).

## Gate 3 — Clean-Room Raw-CSV $\to$ `16 Hz` $\to$ Model Re-Execution: **PASS**
- Independently ingested raw ~2 kHz I/Q CSVs and re-ran frozen `R0/R1a/R2e/R2a/R3c/R3b/R3a` on 3 sentinels (`max diff = 0.00e+00 < 1e-10`).

## Gate 4 — Clean-Room Stratified Metrics & CSV Parity: **PASS**
- Recomputed all `9` strata x `14` model rows + `all_windows` sensitivity (`max diff = 0.00e+00 < 1e-12`, CSV diff `4.97e-05`).

## Gate 5 — Clean-Room `WP4` Uncertainty & Conformal Transfer: **PASS**
- Recomputed `U1..U6b` `all_20` / `inband` Risk-Coverage, `05-10 bpm` stopband OOD (`AUROC`), and `C4`/`C5` coverage (`max wp4 diff = 0.00e+00`, `max conformal diff = 0.00e+00`).

## Gate 6 — Clean-Room 24.2 GHz $O(2)\times\mathbb{R}_{>0}$ Invariance & Causality: **PASS**
- Clean-room `H5` $O(2)\times\mathbb{R}_{>0}$ max diff across `R2e` + `U1/U2/U6b`: `1.42e-14` (`< 1e-6`), exact `0.0` causality: `0.00e+00`.

## Gate 7 — RFC-8259 JSON & Gate `G5` (`L4`) Contract: **PASS**
- Strict JSON (`allow_nan=False`): `True`, schema `stage4-external-v1`, verdict `PASS_L4_ZERO_TOUCH_TRANSFER_VERIFIED`.

