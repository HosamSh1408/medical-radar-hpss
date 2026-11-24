# Chapter 6 Dense Outline: Zero-Touch External Hardware Transfer (`Stage 4`, `L4`)

**Notation Contract:** Strictly [`notation_glossary.tex`](notation_glossary.tex).  
**Authoritative Data Sources:** `metrics_stage4_external.json`, `audit/stage4_external_report.md`, `audit/GODMODE_AUDIT_REPORT.md` (§4.3, `GM-06`, `GM-08`, `GM-09`, `GM-15`).  
**Frozen Figures (2 PDFs):** `stage4_external_frequency_transfer.pdf`, `stage4_external_uncertainty_and_conformal.pdf`.

---

## 6.1 External Benchmark Protocol (`gleichauf2024`, Zero-Touch Governance)
- **Dataset & Hardware Domain Shift (`gleichauf2024time`):** Public Scientific Data neonatal thorax simulator (`20` controlled scenarios: `{05, 10, 20, 30, 40, 45, 50, 55, 60, 80} bpm` $\times$ `{normal, deep}` sinusoidal excursion) recorded with an independent **$24.2\,\mathrm{GHz}$ Six-Port CW interferometer** (`~2 kHz` raw sampled, deterministically downsampled via two-stage `resample_poly` to $16\,\mathrm{Hz}$; measured bandpass passband ripple `[+0.0023, -0.0003] dB` — essentially lossless, `GM-12`) against sub-millimeter laser interferometer ground truth ($f_{\mathrm{gt}}$).
- **Pre-Registered Parameterization & Guardrails (`PF-06`, `GM-08`, `GM-15`):**
  - Primary point estimate: `Full-Cohort K=11` refit from frozen Stage 3 archive (`frozen_stage4_models.npz`, SHA-256 `af33a83e...`); sensitivity check: `LOPO-Mean` (zero parameter updates, zero adaptation).
  - Locked bandpass: **`64-tap zero-phase Kaiser FIR [0.30, 3.00] Hz (beta = 6.5)`** (`18–180 bpm`).
  - Mutually exclusive & exhaustive `20/20` strata partition: **Primary In-Band (`inband_ge28bpm`, 14 scenarios)** (`>= 28.0 bpm`) vs. **Secondary Sub-Band / FIR Skirt (`subband_lt28bpm`, 6 scenarios)** (`{05, 10, 20} bpm`), separating the `-10 dB` Kaiser skirt `20 bpm` (2) from hardware stopband `{05, 10} bpm` (4). Explicitly distinguish `analysis_plan_v3_ai.md` §10 qualitative criteria from operationalized numeric `Gate G5` thresholds (`GM-15`).

## 6.2 Zero-Touch Point Tracking & Shortcut Collapse (`stage4_external_frequency_transfer.pdf`)
- **Steady-State `inband_ge28bpm` (14 scenarios, $6,049$ windows):**
  - **Physics-Constrained `R0–R2` Achieves Near-Machine Precision & Unit Slope:**
    - `R2e_Causal_DiffHMM_FilterSafe8` (**Primary**): **`0.0197 bpm`** scenario RMSE (`0.0262` pool), `0.0184 bpm` MAE, `100%` Acc@3/6/10, linear calibration slope **`0.9994`**, Pearson `r = 0.999999`, `0` steady-state outliers `> 0.5 bpm` (`max_abs_err = 0.158 bpm`), `80 bpm` extreme tachypnea `0.0089 bpm`, `LOPO-Mean` sensitivity `0.023 bpm`.
    - `R0`: `0.0284 bpm` (`slope = 0.9990`, `max = 0.096`); `R1a`: `0.0300 bpm` (`slope = 0.9991`); `R2a`: `0.0189 bpm` (`slope = 1.0004`); `R1f`: `0.015 bpm` (`slope = 1.000`).
  - **`Rung 3` Unconstrained Shortcuts Collapse Across Frequency Boundaries (`GM-09` & `GM-14`):**
    - `R3c_Ridge_Spectral_439` (trained on ALL `N=11` NICU infants!): **`3.6353 bpm`** in-band (`slope = 0.7090`), failing `80 bpm` extreme tachypnea (`10.4652 bpm` RMSE -> outputs `~69.5 bpm`) and `20 bpm` skirt (`11.7783 bpm` -> pulls toward NICU mean `30.5–32.6 bpm`).
    - `R3b_Spectral_MLP_15k` (`N=10`): `3.5022 bpm` in-band (`bias +3.402`, `20 bpm` skirt **`32.7040 bpm`** -> predicts `51.2–54.1 bpm`).
    - `R3a_Causal_TCN_16k` (`N=10`): `3.1782 bpm` in-band (`slope = 0.8623`, `80 bpm` `7.5332 bpm`, `20 bpm` skirt **`23.4714 bpm`** -> predicts `42.7–44.0 bpm`).
- **`20 bpm` Kaiser Skirt vs. `05–10 bpm` Hardware Stopband & Physical Harmonics:**
  - At `20 bpm = 0.333 Hz`, fundamental magnitude is attenuated at the `0.30 Hz` FIR transition skirt while $2f_b = 40\,\mathrm{bpm}$ sits unattenuated: shift-equivariant `R0` (`0.0694`), `R1a` (`0.1610`), `R2e` (`0.0949`), and `R2a` (`0.0245`) track the attenuated fundamental cleanly, whereas at `05–10 bpm` (hardware passband rejection) ALL models lock onto higher Bessel harmonics (`40–50 bpm`, `subband_lt28bpm` `25.25–40.52 bpm`).

## 6.3 External Uncertainty Transfer, Conformal Transfer & Invariance Checks (`stage4_external_uncertainty_and_conformal.pdf`)
- **Zero-Touch OOD Detection (`WP4` on `R2e`):** `U6b_ZeroTuned_Geometric_SQI` separates `05–10 bpm` hardware-stopband corruption (`N=1731`) from valid `inband_ge28bpm` (`N=6049`) with **`0.9998` AUROC** (`0.0000` FPR@95; `mean u = 0.0683` vs `0.2615`), while `U2_Posterior_SD` hits `0.9995` AUROC (`0.0000` FPR@95), and both detect start/end sync-pulse transients at `0.742–0.743` AUROC (reducing pooled `all_20` RMSE at `80%` coverage from `17.973` to `0.893` / `1.006 bpm`).
- **Zero-Touch Conformal Transfer — Strictly `R2e_Causal_DiffHMM_FilterSafe8` (`GM-06`):**
  - Frozen `Full-K=11` quantiles (`C4`: `39.08 / 26.86 / 16.56 bpm`; adaptive `C5`: **`29.34 / 18.02 / 13.27 bpm`** full width at 90% / 80% / 70%) achieve **`100.0%` coverage** (`min = 1.000`) across all 14 `inband_ge28bpm` stationary scenarios (`normal` and `deep`). Strictly omit `R1a`/`R2a` conformal transfer fields due to `q_hat` reuse bug (`GM-06`).
- **External `O(2) x R+` Invariance & Exact `0.0` Causality (`GM-12`):** `H5` invariance `<= 2.84e-14`, frozen `H0` `SO(2)xR+` `<= 2.84e-14`, `Gate G5` Verdict: **`PASS_L4_ZERO_TOUCH_TRANSFER_VERIFIED`**.
