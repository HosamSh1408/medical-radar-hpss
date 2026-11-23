# Stage 3 Independent Clean-Room Audit Report (`stage3-uncertainty-v1`)

Executed with zero statistical, LOPO, selective-risk, or conformal imports from `thesis.exp1_stage1_parsimony`, `thesis.exp2_stage2_complexity`, `thesis.exp3_stage3_uncertainty`, `thesis.wp3_wp4_uncertainty_conformal`, or `thesis.io_utils`.

**Overall Verdict:** **7/7 GATES PASSED**

## Gate 1 — Frozen Baseline Lock & Stage 1.1 / Stage 2.2 Parity: **PASS**
- Checked `154/154` frozen hashes (`bad: 0`), `33/33` locked caches (`bad: 0`), `external_data/` untouched `True` (`52` files).
- Exact `c_cov = 1.00` parity: Stage 1.1 `max diff = 1.69e-14`, Stage 2.2 `max diff = 2.31e-14`.

## Gate 2 — Clean-Room Selective Risk-Coverage & CSV Parity: **PASS**
- Clean-room proxy recomputation diff (`U1..U6c` across `R2e/R2a/R2d`): `max diff = 0.00e+00` (`U6c` LOPO weight diff `0.00e+00`).
- Clean-room `AURC / E-AURC / AUGRC` across 4 references x 8 proxies: `max diff = 4.05e-15`, CSV<->JSON diff `9.93e-06`.

## Gate 3 — Clean-Room Quiet-Apnea (`> 5.0 s`) Pause Safety Audit: **PASS**
- Verified `1581` pause windows vs `5059` regular windows across pooled & within-session $c_{\mathrm{cov}} \in \{0.90, 0.80, 0.70\}$ (`max diff = 0.00e+00`).
- Rej@80% risk ratios: `U2_Posterior_SD = 1.02x (SAFE)`, `U6b_Geom_SQI = 1.13x (SAFE)` vs memoryless `U3_Spec_Ent = 1.39x (OVER-REJ)`.

## Gate 4 — Clean-Room WGN (`N=200`, `seed=20261011`), 5-Seed Surrogate & UCD OOD Audit: **PASS**
- Clean-room OOD AUROC diffs (`U1..U6c`): WGN `0.00e+00`, 5-seed surrogates `0.00e+00`, UCD telemetry `0.00e+00` (`counts match: True`).
- WGN AUROCs (`R2e`): `U3_Spec_Ent = 0.9796`, `U6a_Add_SQI = 0.9935`, `U6b_Geom_SQI = 0.9986`, `U6c_LOPO_SQI = 0.9635` (all `>= 0.95`).

## Gate 5 — Clean-Room 5-Regime Finite-$K$ Conformal Audit: **PASS**
- Clean-room `C1 / C2 / C3 / C4 / C5` conformal recomputation: `max diff = 0.00e+00` (`cross-cohort vacuity verified: True`).
- Verified finite-$K$ vacuity at $\alpha = 0.05 < 1/11$, systematic patient under-coverage of naive window-pooled `C2` across all $\alpha$, and valid $\ge 1-\alpha$ coverage of patient-exchangeable `C4` and locally-adaptive Bayesian-conformal `C5` at $\alpha \in \{0.10, 0.20, 0.30\}$.

## Gate 6 — Clean-Room $O(2) \times \mathbb{R}_{>0}$ Invariance, Causality & LOPO Disjointness: **PASS**
- Clean-room proxy $O(2) \times \mathbb{R}_{>0}$ max diff (`U1..U6c`): `2.22e-15` (`< 1e-6`), causality perturbation diff (`U1..U6c + b_OOD`): `0.00e+00` (`== 0.0`), low-amp `0.2 mm` OOD barrier `0.00e+00`, 11-fold LOPO patient disjointness: `True`.

## Gate 7 — Gate `G4` Hard-Freeze Manifest & RFC-8259 JSON Contract: **PASS**
- Strict RFC-8259 JSON (`allow_nan=False`): `True`, schema `stage3-uncertainty-v1`, Gate G4 verdict `PASS_AND_HARD_FREEZE_WP1_TO_WP4`.
- Frozen Stage 4 archive (`frozen_stage4_models.npz`): exists `True`, SHA-256 verified `True`, required keys `True`, finite arrays `True`.

