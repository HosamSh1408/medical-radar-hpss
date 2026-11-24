# Chapter 4 Dense Outline: Parsimony and Causal Inductive Bias (`Stages 1 & 2`, `Rung 0–3`)

**Notation Contract:** Strictly [`notation_glossary.tex`](notation_glossary.tex).  
**Authoritative Data Sources:** `metrics_stage1_parsimony.json`, `metrics_stage2_complexity.json`, `audit/GODMODE_AUDIT_REPORT.md` (§1.1, §1.3, Pillar 3, §4.1–4.2).  
**Figures:** **Figure 4.1** (`manuscript/figures/tikz_ch4_rungs.tex` — Equivariant `Rung 1–2` vs. Shortcut `Rung 3` Architecture) + frozen `stage2_complexity_ladder.pdf`, `stage2_synthetic_sweep_and_weights.pdf`.

---

## 4.1 Stage 1 Representation Parsimony: Gates `G1` & `G2` (`metrics_stage1_parsimony.json`)
- **Dual-Reference Protocol & VitaGuard Group-Delay Discovery:** Contrast smoothed `br_1hz_30s` with raw physical `1024 Hz Resp.-Trigger` (`br_trig_bpm` bias `+2.15` pool / `+1.65` sess, MAE `5.03`, `r_within = 0.606` at `0 s`, peaking at `+10 s` `r_within = 0.706` vs. pause-count `br_trig_count_bpm` `-6.83` sess / `-13.52` on `>5 s` gaps).
- **`WP1` Gate `G1` Verdict — `RETAIN_F0_CSD_LOCKED`:** None of the 6 I/Q preconditioning variants (`F1a`, `F1b` clipping interp; `F2a/b/c` symmetric Gram–Schmidt whitening; `F3` joint) achieve statistical support (`p_pat,Holm = 1.000` across all 6 on both references; `F2*` degrades clean sentinel Session `2.3` from `11.73` to `13.71–14.31 bpm` and Session `11.1` from `12.65` to `12.78–13.21`), while `F1a`/`F3` break `O(2)` rotation invariance.
- **`WP2` Gate `G2` Verdict — `RETAIN_H0_LOCKED` (`64-tap Kaiser FIR [0.30, 3.00] Hz, beta = 6.5`):**
  - Strictly apply Guardrail **G1**: locked filtering is `64-tap zero-phase Kaiser FIR [0.30, 3.00] Hz (beta = 6.5)` (`paper_workspace/src/filtering.py`), NOT Butterworth.
  - Strictly causal STFT replacements (`H5_Causal_COLA_InBand_L0`: `0 s` lookahead, `2.35 ms/win`, `1.47x` speedup; `H6_Causal_Rect_Full_L0`: `3.41 ms/win`, `1.02x`) fail `±0.50 bpm` TOST equivalence (`H5` `8.267` 1 Hz / `9.398` Trig, $p_{\mathrm{TOST,pat}} = 0.519 / 0.431$; `H6` `8.079 / 9.303`, `p_TOST,pat = 0.027 / 0.186`), while `H0` (`3.0 s` lookahead = `48` samples, `L_H = 3` hops, verified exactly by sample fuzzing `GM-12`) stays locked as primary (`H5` retained as `0 s` E2E ablation).
- **`Rung 1` Causality Frontier & Shifted-Grid / Phase-Scramble Controls:**
  - `R1a_Online_Forward_MMSE_0s` (`3p`, `0s` block lookahead): `6.9904 bpm` (`1 Hz`, `82.6%` Acc@10, `r_within = 0.4545`), `8.624` (`Trig 0s`) / `8.1973` (`Trig -10s`), beating `R0` (`7.8173 / 8.9676 / 8.7268`).
  - Document the Stage 1 causality gap vs. `R1f_Offline_FB_MMSE_Inf` (`6.3952` / `7.5971`) and audited $n=8$ Holm comparison vs. `R1g_Offline_Viterbi_Unpinned` (`6.4978`; Guardrail **G4** / `GM-07`).
  - Prove anti-inertia & shift equivariance: legacy `sig=8` shifts bias `+0.26 bpm` and memoryless Naive MMSE shifts `+2.82 bpm` on `[18, 120]`, whereas equivariant `R1a/R1f` have **`+0.00 bpm` shift**, and 5-seed radar phase scrambling collapses correlation to `0.006` (`+1.52 bpm` worse RMSE).

## 4.2 Stage 2 Complexity Ladder (`Rung 0–3`): BPTT Physics vs. Neural Shortcuts (`metrics_stage2_complexity.json`)
- **Formulation of Differentiable Equivariant 1D-HMM (`FilterSafe-8`, `R2e`):**
  - End-to-end exact log-sum-exp Forward-Backward (`mensch2018differentiable`, `riou2023pesto`), verified against central finite differences to `max abs err <= 4.77e-09` (`max rel err <= 3.48e-06` on 8p/11p/43p and `4.50e-07` on 15,817p `CausalDilatedTCN` — Guardrail **G6** / `GM-11`).
  - Zero-offset linear harmonic comb score $\sum w_m \phi_m(q) + w_5 \phi_{\mathrm{anchor}}(q) + w_6 \phi_{\mathrm{raw}}(q)$ ensures **exact translation equivariance** (`+10 bpm` operator diff `1.42e-14 bpm`; shifted-grid `[18, 120]` bias shift `0.0000 bpm`) and **$SO(2)\times\mathbb{R}_{>0}$ invariance** (`<= 3.38e-14`; E2E causal `H5`/`R2d` full $O(2)\times\mathbb{R}_{>0}$ `<= 2.31e-14`, `0.0` sequence causality diff — `GM-12`).
- **4-Rung Blind Test Results (`pool30_sync`, trained ONLY on `1 Hz`, Guardrail G6 & Figure 4.1 / `stage2_complexity_ladder.pdf`):**
  - Present full `R0–R3` table (`RMSE_sess`, `RMSE_pool`, `r_within`, `SD/SD_ref` variance ratio, `paired_vs_R0`, `paired_vs_R1a`) across `1 Hz` and `Trig -10s`:
    - `R2e_Causal_DiffHMM_FilterSafe8` (`8p`, `0s` block): `1 Hz` **`6.8486 bpm`** ($\Delta_{\mathrm{R0}} = -0.9687\;[-1.1347, -0.8034]$, `11/11` infants, $p_{\mathrm{pat}} = 0.000977$, **$p_{\mathrm{pat,Holm}} = 0.0117$**, $r_{\mathrm{within}} = 0.4651$); `Trig -10s` **`8.0634 bpm`** ($\Delta_{\mathrm{R0}} = -0.6635\;[-0.8827, -0.4457]$, `10/11` infants, $p_{\mathrm{pat}} = 0.001953$, **$p_{\mathrm{pat,Holm}} = 0.0215$**, $r_{\mathrm{within}} = 0.4410$). **ONLY causal model surviving patient-level Holm on BOTH references!**
    - Embrace **`R2e` vs. `R1a` Holm-NS (`GM-04`)**: `1 Hz` $\Delta = -0.1418$ `[-0.3354, +0.0353]`, $p_{\mathrm{sess}} = 0.0264$ ($p_{\mathrm{sess,Holm}} = 0.2109$), $p_{\mathrm{pat}} = 0.0830$ ($p_{\mathrm{pat,Holm}} = 0.6641$); `Trig -10s` $\Delta = -0.1340$, $p_{\mathrm{pat}} = 0.5771$ — demonstrating the statistical ceiling of an 11-infant NICU cohort where `8` parameters already saturate achievable LOPO accuracy.
    - Contrast **`R3c_Ridge_Spectral_439`**, **`R3b_Spectral_MLP_15k`**, **`R3a_Causal_TCN_16k`** (`GM-09`: disclose `R3a/b` trained on `N=10` with `Patient 12` held out vs. `N=11` for `R3c/R2*`): apparent in-band `1 Hz` gains (`5.7985 / 6.5844 / 6.0751 bpm`) vanish on `Trig -10s` (`8.1018 / 8.5150 / 8.4982 bpm`, ALL $p_{\mathrm{pat,Holm}} = 1.00$ vs `R0`), shrink prediction variance (`SD/SD_{T10} = 0.55 / 0.52`), and fail synthetic out-of-range frequency sweeps (`28–80 bpm` MAE `2.01–12.45 bpm` vs. `0.01–0.04 bpm` for `R2e` — `stage2_synthetic_sweep_and_weights.pdf`).
- **Hardware Delay & Data Purity Proofs:**
  - `[-20 s, +20 s]` VitaGuard Lag Sweep (`GM-14`): `5/6` unshifted models (`R0, R1a, R2e, R3c, R3b`) achieve optimal physical agreement identically at `+8..+10 s`.
  - All `31/31` Stage 2 salience caches (`248/248` windows) verified bit-exact (`worst_max_abs_diff = 0.0`, `GM-13`).
