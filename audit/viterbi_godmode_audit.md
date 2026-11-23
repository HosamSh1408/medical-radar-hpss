# God-Mode Independent Clean-Room Audit Report

Executed with **zero** imports from `thesis.*` against raw EDFs, locked `.npz` caches, and `metrics_viterbi_repro.json`.
**Overall Verdict:** `7/7 GATES PASSED`


## Gate 1 — Frozen Baseline Lock: **PASS**
- Checked `154/154` frozen files (`bad: 0`) and `33/33` locked `v1` caches (`bad: 0`).

## Gate 2 — Raw EDF (1.0 Hz Verified) & Clean-Room Metric Recomputation: **PASS**
- Checked EDF `RespRate` sample rate across `pool30_sync` (`bad: 0`), re-decoded 5 key trackers (including `C6e_LOPO_Wide_AnchoredNLS_NMF`) from raw NPZs + clean-room Viterbi.
- Max discrepancy across `RMSE_sess`, `RMSE_pool`, `MAE_sess`, strict `< 10` `Acc@10`, demeaned-OLS `r_within`, `ICC(1,1)`, BA LoA, `const_loso`, `const_lopo_median`, and `p_wilcoxon_patient`: `2.22e-16`.

## Gate 3 — Clean-Room Eq. 5 HQI & 2D Viterbi DP Parity (`7.3`): **PASS**
- Unfactored brute-force vs separable 2D Viterbi on real `7.3` subgrid: path diff `0.0` bpm, score diff `0.00e+00`.
- Full-session `7.3` `C3b` vs exported CSV: max diff `7.11e-15`.

## Gate 4 — Cohort-Scoped Bland–Altman Scatter-to-JSON Parity: **PASS**
- Cohorts exported in CSV: `['PAPER_18', 'all_paired', 'pool30_sync', 'retained23']`.
- Max Bias/SD/LoA discrepancy across all `(regime, cohort, method)` combinations (including all LOPO families): `7.11e-15`.

## Gate 5 — LOPO Anti-Leakage & Grid Pinning Audit: **PASS**
- Fold disjointness & coverage errors across all regimes/cohorts: `0`.
- `pool30_sync` LOPO diagnostics (`C6a–C6e`): `{"C6a_LOPO_Cal_Vit_HPSS": {"grid_mode": "narrow", "n_folds": 11, "n_fallback": 10, "n_upper_pinned": 0, "n_any_boundary_pinned": 2}, "C6b_LOPO_Cal_HarmNLS_HPSS": {"grid_mode": "narrow", "n_folds": 11, "n_fallback": 0, "n_upper_pinned": 11, "n_any_boundary_pinned": 11}, "C6c_LOPO_Cal_HarmNLS_NMF": {"grid_mode": "narrow", "n_folds": 11, "n_fallback": 0, "n_upper_pinned": 11, "n_any_boundary_pinned": 11}, "C6d_LOPO_Wide_AnchoredNLS_HPSS": {"grid_mode": "wide", "n_folds": 11, "n_fallback": 0, "n_upper_pinned": 0, "n_any_boundary_pinned": 11}, "C6e_LOPO_Wide_AnchoredNLS_NMF": {"grid_mode": "wide", "n_folds": 11, "n_fallback": 0, "n_upper_pinned": 0, "n_any_boundary_pinned": 11}}`.

## Gate 6 — Multi-Seed Surrogates, Invariances & HR Gate G2: **PASS**
- Max HR `<= 10 bpm` across all 9 HR extractors (incl. DACM, BR-Notched, 480-pt): `39.84%` (Gate `G2 >= 50%`: `False`).
- 20-seed time-shuffle `r_within` ok: `True`; 5-seed phase-scramble `r_within` ok: `True`; I/Q group invariance max diff: `0.00e+00` bpm.

## Gate 7 — Provenance, Git Head & Completeness: **PASS**
- Schema `viterbi-repro-v2` | Git `681de7e` (`thesis/main`) | Methods: `33`.
- Window completeness: `pool30_sync` (`28 / 6,640` bad: `0`), `retained23` (`23 / 6,447` bad: `0`).

