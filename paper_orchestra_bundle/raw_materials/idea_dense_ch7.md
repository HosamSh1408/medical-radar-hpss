# Chapter 7 Dense Outline: Discussion, Clinical Readiness Gap, Limitations & Conclusion

**Notation Contract:** Strictly [`notation_glossary.tex`](notation_glossary.tex).  
**Authoring Scope:** English M.Sc. AI Thesis Chapter 7 (~7–9 pages); synthesizes cross-chapter scientific insights while strictly upholding `L2` / `L5` epistemic boundaries and BTU ethics/AI disclosure standards.  
**Key Citations:** `beltrao2022contactless`, `senechal2026next`, `krbec2024emerging`, `dunn2023distribution`, `gibbs2024conformal`, `barber2023conformal`, `angelopoulos2024conformal`, `riou2023pesto`, `geirhos2020shortcut`, `gleichauf2024time`, `liao2025noncontact`, `hari2025contactless`, `schellenberger2020dataset`.

---

## 7.1 Dual Confirmation of the Central Parsimony Thesis
- **Convergence of Two Independent Lines of Evidence:**
  - *Line 1 (Point Respiratory-Rate Estimation, Ch. 4 & 6):* Encoding physical shift equivariance, $SO(2)\times\mathbb{R}_{>0}$ baseband invariance, and causal Markov continuity in 8 interpretable parameters (`R2e_Causal_DiffHMM_FilterSafe8`) beats memoryless `R0` across both monitor-trend (`1 Hz` $p_{\mathrm{pat,Holm}} = 0.0117$) and physical (`Trig -10s` $p_{\mathrm{pat,Holm}} = 0.0215$) NICU references, saturates cleanly near `R1a` (`3p`, Holm-NS `0.0830`), and transfers zero-touch across external 24.2 GHz hardware (`0.0197 bpm`, `slope = 0.9994`) where `439–15,817`-parameter regressors collapse (`3.18–3.64 bpm`).
  - *Line 2 (Uncertainty Quantification & Safety, Ch. 5):* Parameter-free `U2_Posterior_SD` ($p_{\mathrm{pat}} = 0.0098$, `10/11` infants, `1.02x` quiet-apnea safety) and zero-fitted-weight barrier composite `U6b` ($\tau_{\mathrm{ood}} = 0.90$, `1.13x` safety, `0.9986` WGN AUROC, `0.9998` Stage 4 stopband AUROC) outperform 5-parameter LOPO ridge-calibrated `U6c`, which fails both 95% CI exclusion and apnea pause safety (`1.23x > 1.15x`).

## 7.2 Quantitative Clinical Readiness Gap (`L4 != L5`)
- **Why Stationary `L4` Perfection Does Not Imply Bedside `L5` Readiness:**
  - Contrast stationary phantom accuracy (`0.0197 bpm`, `0` outliers) against live preterm NICU uncertainty: even with causal physics-constrained inference, held-out `80%`-target hierarchical conformal bounds (`C5`) require **`24.02 bpm` average full width** (`17.96 bpm` low-$\tilde{\sigma}$ vs. `30.54 bpm` high-$\tilde{\sigma}$), with `2/11` hardest preterm infants dropping as low as `0.5856` coverage; `95%` patient-exchangeable sets are mathematically vacuous ($\alpha_{\min} = 1/11$) until $K_{\mathrm{cal}} \ge 19$ infants.
  - Under real preterm NICU confounding—cardiac `HR/2` collision ($\rho = +0.696$), severe I/Q collinearity (Session `11.1`, $\rho_{IQ} = 5.86$, `12.65 bpm`), `UCD` firmware interruptions, and spontaneous RBM—single-channel 24 GHz CW radar without range gating cannot yet replace wired alarms (`L5` strictly blocked).
- **Non-Deployability of Label-Dependent Online Conformal (`gibbs2024conformal`):** Prove mathematically why Adaptive Conformal Inference (updating $\alpha_{t+1}$ from observed $y_t$) cannot operate autonomously at an unmonitored bedside where live reference $\brRef_t$ does not exist (`barber2023conformal`, `angelopoulos2024conformal`).

## 7.3 Limitations & Honest Methodological Boundaries
- Cohort size ($K=11$ unique infants, $S=28$ synced sessions) limits power between close physics models (`R2e` vs `R1a`) and prevents 95% patient-level conformal guarantees.
- `U6b` relies on one empirically informed barrier cutoff ($\tau_{\mathrm{ood}} = 0.90$, safe band `[0.90, 0.92]`); `HQI` denominator compression (`[0.0655, 0.4351]`, $\rho = 0.999945$); `05–10 bpm` neonatal severe bradypnea lies below locked `0.30 Hz` FIR passband unless wider front-ends or lower cutoff frequencies are clinically validated.
- Unresolved `beltrao2022contactless` original cohort details block `L2` permanently.

## 7.4 Future Engineering & Clinical Research Directions
- Multi-antenna MIMO / wideband FMCW (`liao2025noncontact`, `hari2025contactless`) + optical/ToF (`gleichauf2021automated`) hardware providing range-bin isolation to suppress lateral cardiac/RBM clutter.
- Larger multicenter neonatal calibration cohorts ($K_{\mathrm{cal}} \ge 19\text{--}30+$) enabling valid $95\%$ hierarchical conformal prediction sets (`dunn2023distribution`) and safe quiet-apnea alarming protocols (`krbec2024emerging`).

## 7.5 Ethics, Generative AI Governance & Conclusion
- Reproducibility lock (`5e458c1`, dual Python `3.9.6` / `3.10.19` contract, open provenance logs), clinical safety ethics of avoiding silent pause-censoring and false clinical readiness claims, and concise concluding synthesis.
