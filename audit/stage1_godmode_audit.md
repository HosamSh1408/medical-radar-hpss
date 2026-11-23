# Stage 1.1 Independent Clean-Room Audit Report (`stage1-parsimony-v3`)

Executed with zero imports from `thesis.exp1_stage1_parsimony` or `thesis.io_utils`.

**Overall Verdict:** `7/7 GATES PASSED`

## Gate 1 — Frozen Baseline SHA-256 & F0 Parity: **PASS**
- Checked `154/154` frozen files (`bad: 0`), `33/33` locked `v1` caches (`bad: 0`), and `max |F0 - est_hpss|` across 31 sessions (`0.00e+00 bpm`).

## Gate 2 — Clean-Room Metric, TOST, Wilcoxon, Raw Clip & CSV Parity: **PASS**
- Max clean-room vs JSON diff across 8 methods x dual references: `1.87e-14`.
- CSV (`28` rows) vs JSON mean max diff: `1.13e-05`; raw ADC clip fraction max diff: `4.83e-05`.

## Gate 3 — Clean-Room HMM, Scaled vs `logsumexp` & Stage 0 Viterbi Parity: **PASS**
- Clean-room explicit DP/FB loop max diff: `2.56e-13`; scaled vs `logsumexp` max diff: `1.28e-13`.
- Stage 0 `decode_viterbi_1d` vs Stage 1 `hmm_offline_viterbi_prob` across all 28 sessions at `sigma=2.0, eps=1e-3`: `0.00 bpm`.

## Gate 4 — Causality, O(2) x R+ Invariance & Two-Axis Contract: **PASS**
- Causal future diff: `hpss_l0=0.0`, `mitigate_v2(causal)=0.0` (vs `block` intra-window diag `89.25`), `hmm_fwd=0.0`, `hmm_lag4=0.0`.
- Clean-room O(2) x R+ max diff across `F2a/b/c, F1b, H1, H2, H5, H5(causal), H6`: `0.00e+00 bpm`; two-axis contract valid: `True`.

## Gate 5 — LOPO Anti-Leakage, Frontier Monotonicity & Equivariance: **PASS**
- Fold disjointness errors across 3 cohorts: `0`; causal `R1a..R1e` pinning/fallback errors: `0`; monotonicity violations `R1a >= R1c >= R1d >= R1e >= R1f` on `1 Hz` & `Trig`: `0`; shifted-grid equivariance ok: `True`.

## Gate 6 — 5-Seed Surrogates (+ `R1f`), Lag Audit & Gates G1/G2: **PASS**
- 5-seed scrambled surrogates (`|r_within| < 0.05` & real margin `> 1.0 bpm`): `True`; Monitor lag audit (`peak >= +6 s`): `True`; Gates `G1=RETAIN_F0_CSD_LOCKED`, `G2=RETAIN_H0_LOCKED`: `True`.

## Gate 7 — Provenance, 71/71 BibTeX Parity & Cohort Completeness: **PASS**
- Schema `stage1-parsimony-v3` | `71/71` BibTeX↔CSV parity: `True` | Cohort window count errors: `0`.

