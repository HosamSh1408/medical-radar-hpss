# Chapter 5 Dense Outline: Selective Uncertainty, Apnea-Pause Safety & Finite-$K$ Conformal Audit (`Stage 3`)

**Notation Contract:** Strictly [`notation_glossary.tex`](notation_glossary.tex).  
**Authoritative Data Sources:** `metrics_stage3_uncertainty.json`, `audit/GODMODE_AUDIT_REPORT.md` (§1.2, §2.1–2.2, `GM-01`–`GM-03`, `GM-05`, `GM-07`).  
**Frozen Figures (2 PDFs):** `stage3_risk_coverage_and_apnea_safety.pdf`, `stage3_finite_k_conformal_audit.pdf`.

---

## 5.1 Second Independent Proof of Parsimony Inside Uncertainty Quantification (`U1–U6c`)
- **Eight Strictly Causal Uncertainty Proxies (0.0 s lookahead diff verified across all 8):**
  - Intrinsic HMM posterior uncertainty (`0` fitted params): `U1_Posterior_Entropy` ($H_{\mathrm{post},t}$), `U2_Posterior_SD` ($\tilde{\sigma}_{\mathrm{post},t}$, strictly **zero-parameter & threshold-free** — Guardrail **G2** / `GM-01`).
  - Spectral/Harmonic OOD proxies (`0` fitted params): `U3_Spectral_Entropy_Clean`, `U4_Spectral_Entropy_Raw`, `U5_HQI_Harmonic_Deficit` (explicitly state overlapping-denominator `HQI` range `[0.0655, 0.4351]`, Spearman $\rho = 0.999945$ vs non-overlapping, $\Delta\eaurcMetric = +0.0032\,\mathrm{bpm}$ per Guardrail **G8** / `GM-05`).
  - Zero-weight composites vs. 5-parameter LOPO Ridge (`U6c`): `U6a_ZeroTuned_Additive_SQI`, `U6b_ZeroTuned_Geometric_SQI` (**zero-fitted-weight heuristic barrier composite** combining $\tilde{\sigma}_{\mathrm{post},t}$ with fixed cutoff $\tau_{\mathrm{ood}} = 0.90$ and causal `history_hops = 3`), and 5-parameter `U6c_LOPO_Calibrated_SQI`.
- **Selective Prediction & Risk–Coverage Curves (`pool30_sync`, $c_{\mathrm{cov}} \in [0.50, 1.00]$, `R2e` primary; `stage3_risk_coverage_and_apnea_safety.pdf`):**
  - **`U2_Posterior_SD` (`0` params, `0` thresholds):** `10/11` infants improved on `R2e` `1 Hz`, $\eaurcMetric_{\mathrm{sess}} = -0.2454\;[-0.4069, -0.1168]\,\mathrm{bpm}$, $p_{\mathrm{sess}} = 0.0017$, **$p_{\mathrm{pat}} = 0.0098$**, `6.8486 -> 6.6890 bpm` at $\covLevel = 0.80$; on `Trig -10s`: `-0.2929 [-0.4555, -0.1436] bpm`, `9/11` infants, $p_{\mathrm{pat}} = 0.0244$, `8.0634 -> 7.8798 bpm`.
  - **`U6b` exact reference-wise figures (`GM-01`):** `1 Hz` $\eaurcMetric_{\mathrm{sess}} = -0.2398\;[-0.4982, -0.0257]\,\mathrm{bpm}$, `8/11` infants, $p_{\mathrm{pat}} = 0.2402$, `6.8486 -> 6.5956 bpm`; `Trig -10s` `-0.2932 [-0.5500, -0.0699] bpm`, `8/11` infants, $p_{\mathrm{pat}} = 0.1230$, `8.0634 -> 7.7569 bpm`.
  - **Failure of 5-Parameter `U6c_LOPO_Calibrated_SQI`:** `1 Hz` 95% CI crosses zero (`-0.2688 [-0.5926, +0.0266]`, $p_{\mathrm{pat}} = 0.2402$, `8/11`) AND fails quiet-apnea safety (`1.2266x > 1.15x`), yielding a second independent proof of parsimony.

## 5.2 The Quiet-Apnea Pause Safety vs. OOD Paradox & Barrier Sensitivity
- **Stratified Pause-Safety Test ($N_{\mathrm{pause}} = 1581$ vs. $N_{\mathrm{reg}} = 5059$ windows with inter-breath gap `> 5.0 s`, target rejection ratio $\le 1.15\times$ at $\covLevel = 0.80$):**
  - Memoryless spectral proxies fail pause safety catastrophically: `U3` `1.3941x`, `U4` `1.4191x`, `U5` `1.1946x`, `U6c` `1.2266x`. Physical mechanism: genuine neonatal central pauses drop harmonic peak SNR, flattening normalized STFT entropy.
  - Pure Markov `U2` is pause-safe (`1.0200x` pooled / `1.0645x` sess) because the discrete transition prior bridges silent episodes smoothly, but yields weaker WGN OOD detection (`0.8406` AUROC, `0.4515` FPR@95).
- **Resolution by `U6b` + Honest $\tau_{\mathrm{ood}}$ Stress Test (`GM-01`):**
  - `U6b` ($\tau_{\mathrm{ood}} = 0.90$, `hops = 3`) satisfies quiet-apnea safety (`1.1275x` pooled / `1.1209x` sess `<= 1.15x`), achieves **`0.9986` WGN OOD AUROC** (`0.0036` FPR@95%TPR; `0.9055` AUROC on `ucd_nonchild >= 50%` partial-window waveform segments, `1.000` firmware UCD telemetry), `0.0` false barrier triggers (`max_b_ood = 0.0`) on synthetic `0.2 mm` shallow breathing (`10 dB SNR`), and exact $O(2)\times\mathbb{R}_{>0}$ invariance (`<= 8.88e-16` on `H5`).
  - Report our full $\tau_{\mathrm{ood}}$ sweep (`audit/GODMODE_AUDIT_REPORT.md` §2.1): because NICU 95th-pct mixed spectral entropy is `0.9130` vs. WGN 5th-pct `0.9498`, all $\tau_{\mathrm{ood}} \le 0.88$ fail apnea safety (`1.177x–1.316x`), confirming `[0.90, 0.92]` as the sole safe barrier band ($\tau_{\mathrm{ood}} = 0.92$ reaches `1.0284x` safety & $p_{\mathrm{pat}} = 0.0322$).

## 5.3 Finite-$K$ Conformal Audit: From Silent Naive Failure to Hierarchical Bounds (`stage3_finite_k_conformal_audit.pdf`)
- **HARD GUARDRAIL (`GM-02`):** Explicitly discard/blacklist `valid_distribution_free_guarantee` in `metrics_stage3_uncertainty.json` (`146` false `True` cells on `C2`/`C3`). Explain why `C1` (50/50 random window) and `C2` (LOPO pooled windows) silently fail held-out infants at `80%` target (`C2`: `0.772` mean cov, `min = 0.398`, `5/11` undercovered; `13.14 / 17.00 bpm` widths).
- **Analytic Vacuity Proof at $K_{\mathrm{cal}} = 10$:** Minimum exchangeable miscoverage is $\alpha_{\min} = 1/(K_{\mathrm{cal}}+1) = 1/11 \approx 0.0909$; at $1-\alpha = 0.95$ ($\alpha = 0.05$), rank index $\lceil (10+1)(0.95)\rceil = 11 > 10 \implies \mathrm{width} = \infty$ (requiring $K_{\mathrm{cal}} \ge 19$).
- **Two-Level Patient-Quantile (`C4`/`C5`) vs. Dunn et al. (JASA 2023) Theorem 1 (`GM-03` & `GM-07`):**
  - Formulate `C4` (unweighted) & `C5` (normalized by $\tilde{\sigma}_{\mathrm{post},t}$) clearly as conservative two-level patient-quantile bounds (`(1-\alpha)^2` worst-case vs. Dunn's `1-\alpha` CDF-pool/subsample).
  - Present full head-to-head table at $\alpha \in \{0.05, 0.10, 0.20, 0.30\}$: at $\alpha = 0.20$ (`80%`), `C5` (`0.8636` mean, `0.5856` min, `2/11` undercovered, `24.02 bpm` **full width**, tightening from `30.54 bpm` high-$\sigma$ to `17.96 bpm` low-$\sigma$ vs. `C4` `25.57 bpm`) matches `Dunn-Pool` (`0.8785`, `0.5903`, `1/11`, `25.05 bpm`) and outperforms single-draw `Dunn-Sub1` (`4/11` undercovered, `0.5277` min).
  - Freeze Gate `G4` artifacts (`frozen_stage4_models.npz`, `af33a83e...`) for zero-touch Stage 4 evaluation.
