# BTU Cottbus-Senftenberg Master's Thesis Authoring & Governance Guidelines

**Program:** Master of Science (M.Sc.) in Artificial Intelligence — Faculty 1 (Module 13600, 30 ECTS: 75% Written Thesis, 25% Oral Defense / *Kolloquium*)  
**Language:** English monograph (~75–90 pages, A4 `11pt` KOMA-Script `scrreprt`, compiled with XeTeX via `tectonic`)  
**Frozen Experimental Basis:** Branch `thesis/main` sealed at commit `5e458c1` (`audit/thesis_freeze_manifest.sha256` `113/113 OK`; frozen Stage 4 model archive `thesis/outputs/stage3_uncertainty/frozen_stage4_models.npz` SHA-256 `af33a83e157f7d2efb0f47e50c9544ee19bacb7da88ab0e0235ccead029cbcc4`)

---

## 1. Mandatory Structural & Integrity Requirements (BTU Module 13600)

1. **Front-Matter & AI Disclosure:**
   - Dedicated Faculty 1 Cover Page (`manuscript/frontmatter/titlepage.tex`) populated via centralized macros in `manuscript/metadata.tex` (Final Thesis Title, Candidate Name, Matriculation No., Chair, 1st & 2nd Examiners, Submission Date).
   - Bound **Declaration of Authorship (*Eigenständigkeitserklärung*)** (`manuscript/frontmatter/declaration.tex`) with explicit documentation of generative AI tooling (including `PaperOrchestra` / AI-assisted editorial workflows) and scientific software provenance (`Python 3.9.6` locked `v1` baseline in `paper_workspace/src/`; `Python 3.10.19` `WP1–WP5` pipeline in `thesis/`).
   - Standalone English **Abstract** (~300–350 words) plus optional toggled German *Zusammenfassung* (`manuscript/frontmatter/abstract.tex`).
2. **Single Authoritative Notation Contract:**
   - All chapters, equations, tables, and captions MUST load and use macros from [`paper_orchestra_bundle/raw_materials/notation_glossary.tex`](raw_materials/notation_glossary.tex).
   - Five mandatory symbol disambiguations:
     1. $\chestDisp$ (`\chestDisp`, physical chest displacement in mm) $\neq$ $\csdSignal$ (`\csdSignal`, mean-centered complex baseband) $\neq$ $\chestDispEst$ (`\chestDispEst`, uncalibrated bandpass-filtered `d_raw`, `d_hat`, or `d_hpss`).
     2. $\rmseSess$ (`\rmseSess`, primary equal-session macro-mean; e.g., `6.798 bpm` HPSS on `PAPER_18`) $\neq$ $\rmsePool$ (`\rmsePool`, window-pooled RMSE; `7.217 bpm` on `PAPER_18`). Remember that legacy JSON field `rmse_*_per_window` stores the equal-session mean!
     3. $\rWithin$ (`\rWithin`, Bland–Altman ANCOVA `rmcorr`) $\neq$ $\rWithinSessMean$ (`\rWithinSessMean`, unweighted mean of per-session Pearson $r_s$; `0.465` on `PAPER_18`). Never compute pooled cross-session Pearson $r$.
     4. $\hpssMask$ (`\hpssMask`, locked two-sided `fftshift` Wiener mask $\hpssHarm^\gamma / (\hpssHarm^\gamma + \hpssPerc^\gamma + \varepsilon)$) $\neq$ $\hpssMaskBand$ (`\hpssMaskBand`, WP2 band-restricted/directional mask $|f_k| \in [\bandLow, \bandHigh]$).
     5. $\nmfW, \nmfH$ (bold NMF basis and activation matrices) $\neq$ $\hpssHarm, \hpssPerc$ (HPSS harmonic/percussive spectrograms).
   - Always write strict inequality $|e_{s,w}| < \delta$ for `\accTau{\delta}` (`\mathrm{Acc}@\delta`), rectangular STFT `48/16/256` (`N_win=48`, `H_stft=16`, `N_FFT=256`) for locked `v1`, and `\covLevel` ($c_{\mathrm{cov}}$, never bare `c`) for selective-prediction coverage.
   - Cite strictly from [`paper_orchestra_bundle/raw_materials/seed_references.bib`](raw_materials/seed_references.bib) (`71` verified BibTeX keys across 6 clusters).

---

## 2. Eight Non-Negotiable Pre-Drafting Truth Guardrails

Every chapter outline and LaTeX drafting pass MUST obey all 8 guardrails below (derived from `audit/PREFREEZE_CONSOLIDATED_AUDIT_REPORT.md` and `audit/GODMODE_AUDIT_REPORT.md`):

1. **G1 — Locked Bandpass Filter & Exhaustive Stage 4 Strata (`PF-06` / `GM-08`):**
   - The locked radar bandpass across Stages 0–4 is ALWAYS the **64-tap zero-phase Kaiser FIR filter** over $[0.30, 3.00]\,\mathrm{Hz}$ ($18\text{--}180\,\mathrm{bpm}$, $\kaiserBeta = 6.5$, implemented via `scipy.signal.firwin` + `filtfilt` in `paper_workspace/src/filtering.py`).
   - The exhaustive Stage 4 partition of `gleichauf2024` (`20/20` scenarios, `100%` coverage) is **`inband_ge28bpm`** (`14` scenarios: `{30, 40, 45, 50, 55, 60, 80} bpm` $\times$ `{normal, deep}`) and **`subband_lt28bpm`** (`6` scenarios: `{05, 10, 20} bpm` $\times$ `{normal, deep}`).
   - **Ignore** stale Stage 3 metadata text mentioning `"0.40-1.50 Hz Butterworth"` or `"[10.0, 24.0] bpm"`.

2. **G2 — Uncertainty Naming, $\tau_{\mathrm{ood}}$ Sensitivity & Exact `p`-Values (`GM-01`):**
   - Frame **`U2_Posterior_SD`** ($\tilde{\sigma}_{\mathrm{post},t}$) as strictly **zero-parameter and threshold-free** (`10/11` infants improved on `R2e` `1 Hz`, $\eaurcMetric_{\mathrm{sess}} = -0.2454\;[-0.4069, -0.1168]\,\mathrm{bpm}$, $p_{\mathrm{pat}} = 0.0098$; `Trig -10s` $-0.2929\;[-0.4555, -0.1436]\,\mathrm{bpm}$, $p_{\mathrm{pat}} = 0.0244$; `1.02x` quiet-apnea safety).
   - Frame **`U6b_ZeroTuned_Geometric_SQI`** strictly as a **zero-fitted-weight heuristic barrier composite** (fixed cutoffs $\tau_{\mathrm{ood}} = 0.90$, causal `history_hops = 3`). Never call `U6b` "zero-threshold" or "parameter-free"! Cite our `[0.90, 0.92]` sensitivity check: all $\tau_{\mathrm{ood}} \le 0.88$ and threshold-free `b_ood = h_mix` fail `< 1.15x` apnea pause safety (`1.177x–1.316x`), whereas $\tau_{\mathrm{ood}} = 0.90$ achieves `1.1275x` pooled / `1.1209x` session safety, `0.9986` WGN AUROC (`0.0036` FPR@95%TPR), `0.9998` Stage 4 stopband AUROC, and $\tau_{\mathrm{ood}} = 0.92$ reaches `1.0284x` safety and $p_{\mathrm{pat}} = 0.0322$ (WGN `0.9968`).
   - Keep `U6b` on `R2e` **`1 Hz` ($p_{\mathrm{pat}} = 0.2402$, $\eaurcMetric_{\mathrm{sess}} = -0.2398$)** strictly distinct from **`Trig -10s` ($p_{\mathrm{pat}} = 0.1230$, $\eaurcMetric_{\mathrm{sess}} = -0.2932$)**.

3. **G3 — Finite-$K$ Conformal Theory & JSON Flag Blacklist (`PF-08` / `GM-02` / `GM-03`):**
   - **BLACKLIST** the field `"valid_distribution_free_guarantee"` in `metrics_stage3_uncertainty.json` (`146` non-exchangeable cells in `C2`/`C3`, including `33` at vacuous $\alpha = 0.05$, carry `true` erroneously).
   - Frame `C4`/`C5` as **two-level patient-quantile hierarchical conformal calibration inspired by Dunn et al. (2023)** \citep{dunn2023distribution} (noting its conservative $(1-\alpha)^2$ finite-sample vs. Dunn Theorem 1 single-$1-\alpha$ target via CDF pooling/subsampling).
   - Cite our audited head-to-head comparison on `R2e` `1 Hz` ($K_{\mathrm{cal}} = 10$ infants/fold, $\alpha_{\min} = 1/11 \approx 0.0909$): all three (`C4/C5`, `Dunn-Pool`, `Dunn-Sub1`) are **vacuous ($\mathrm{width} = \infty$)** at $\alpha = 0.05$ ($K_{\mathrm{cal}} \ge 19$ required); at $\alpha = 0.20$, adaptive `C5` (`0.8636` mean cov, `0.5856` min, `2/11` undercovered, `24.02 bpm` mean width) closely tracks `Dunn-Pool` (`0.8785`, `0.5903`, `1/11`, `25.05 bpm`) and clearly outperforms single-draw `Dunn-Sub1` (`4/11` undercovered, `0.5277` min).

4. **G4 — Full Interval Width Semantics & Audited `R1g` Holm Table (`PF-10` / `GM-07`):**
   - In `metrics_stage3_uncertainty.json`, `low_uncertainty_half_width_bpm` and `high_uncertainty_half_width_bpm` store **full interval widths** ($2\hat{q}_{\mathrm{norm}}\tilde{\sigma}$: `17.96` / `30.54 bpm` at 80%; `29.30` / `49.82 bpm` at 90%). Never double them or write `±18.0 bpm`!
   - When citing Stage 1 `paired_vs_R1g_offline_viterbi`, use the audited $n=8$ Holm values (`audit/GODMODE_AUDIT_REPORT.md` §1.3).

5. **G5 — Stage 4 Conformal Scope (`PF-07` / `GM-06`):**
   - Cite `metrics_stage4_external.json["conformal_transfer"]` **exclusively for `R2e_Causal_DiffHMM_FilterSafe8`** (never for `R1a`/`R2a`, whose stored fields reused `R2e`'s `q_hat`).

6. **G6 — Exact Holm `p`-Values, Fairness Disclosures & VitaGuard Lag Proof (`GM-04` / `PF-09` / `PF-11` / `GM-14`):**
   - `R2e` (`8p`) vs. `R0` (`0p`) on `pool30_sync`: `1 Hz` $\Delta = -0.9687\;\mathrm{bpm}$ `[-1.1347, -0.8034]`, `11/11` infants, $p_{\mathrm{pat}} = 0.000977$, **$p_{\mathrm{pat,Holm}} = 0.0117$** ($m=12$ family); `Trig -10s` $\Delta = -0.6635\;\mathrm{bpm}$ `[-0.8827, -0.4457]`, `10/11` infants, $p_{\mathrm{pat}} = 0.001953$, **$p_{\mathrm{pat,Holm}} = 0.0215$** — the ONLY causal architecture passing patient-level Holm on both references.
   - Explicitly report **`R2e` vs. `R1a` as Holm-non-significant** (`1 Hz` $\Delta = -0.1418\;\mathrm{bpm}$ `[-0.3354, +0.0353]`, $p_{\mathrm{sess}} = 0.0264$, $p_{\mathrm{sess,Holm}} = 0.2109$, $p_{\mathrm{pat}} = 0.0830$, $p_{\mathrm{pat,Holm}} = 0.6641$; `Trig -10s` $\Delta = -0.1340\;\mathrm{bpm}$, $p_{\mathrm{pat}} = 0.5771$): this saturation at 8 parameters reinforces the central parsimony thesis.
   - Explicitly disclose `R3a`/`R3b` used `N=10` training infants (`Patient 12` held out for early-stopping checkpoint selection) vs. `N=11` for `R3c` and `R2*` — noting `R3c` (trained on all `11` infants) still suffers catastrophic external shortcut failure (`3.6353 bpm` in-band, `10.4652 bpm` at `80 bpm`, `11.7783 bpm` on `20 bpm` skirt).
   - Distinguish `analysis_plan_v3_ai.md` §10 qualitative criteria from operationalized `Gate G5` numeric thresholds, and cite the `[-20 s, +20 s]` VitaGuard lag sweep (`5/6` unshifted models peak identically at `+8..+10 s`, proving monitor hardware latency).

7. **G7 — Hard Claim-Gate Lock (`L0–L5`):**
   - **Prohibited (`BLOCKED`):** `L2` (exact Beltrão 2022 cohort/denominator/solver reproduction) and `L5` (clinical bedside utility, autonomous apnea alarming, clinical superiority).
   - **Authorized:** `L0–L1` (verified provenance & parity), `L3 / L3+` (independent internal LOPO comparisons on `pool30_sync` and sensitivity cohorts), and `L4` (zero-touch stationary physical/hardware transfer on 24.2 GHz `gleichauf2024`: `0` steady-state outliers `>0.5 bpm`, `R2e = 0.0197 bpm`, `slope = 0.9994`). Frame `24.02 bpm` (`80%`) conformal intervals as a fundamental physical boundary of single-channel 24 GHz CW radar on preterm infants.

8. **G8 — `HQI` Formula & Glossary Layout (`GM-05` / `GM-10`):**
   - If writing the implemented formula of `HQI`, state the exact overlapping-denominator compression (`[0.0655, 0.4351]`, Spearman $\rho = 0.999945$, $\Delta\eaurcMetric = +0.0032\,\mathrm{bpm}$ vs. non-overlapping).
   - Never edit frozen `notation_glossary.tex`; render Appendix A under `16.0 cm` A4 geometry with `\emergencystretch=1.5em` so compiled warnings stay `<= 5pt`.

---

## 3. Chapter & Appendix Architecture (7-Chapter Monograph, ~75–90 pp.)

| Unit | Title | Page Target | Primary Sources & Frozen Figures |
| :--- | :--- | :---: | :--- |
| Front Matter | Cover, BTU *Eigenständigkeitserklärung* + AI/Software Disclosure, English Abstract, TOC/LOF/LOT | 6–8 | `metadata.tex`, `declaration.tex`, `abstract.tex` |
| **Ch. 1** | Introduction, Clinical Motivation, Research Questions & Claim-Gate Architecture (`L0–L5`) | 8–10 | `idea_dense_ch1.md`, **Fig. 1.1** (`tikz_ch1_architecture.tex`) |
| **Ch. 2** | Radar Physics (`WP1`) and Systematic Literature Review (`71` Refs, `6` Clusters) | 12–14 | `idea_dense_ch2.md`, `lit_review_matrix.csv`, `seed_references.bib` |
| **Ch. 3** | Baseline Forensic Audit (`EXP-0`) and Stage 0 Viterbi Reproduction | 12–14 | `idea_dense_ch3.md`, `metrics_wp0_forensics.json`, `metrics_viterbi_repro.json`, 7 `exp0_*` / `viterbi_repro_*` PDFs |
| **Ch. 4** | Parsimony & Causal Inductive Bias: Stage 1 (`WP1`/`WP2`/`R1`) & Stage 2 Complexity Ladder (`R0–R3`) | 14–16 | `idea_dense_ch4.md`, `metrics_stage1_parsimony.json`, `metrics_stage2_complexity.json`, **Fig. 4.1** (`tikz_ch4_rungs.tex`) + 2 `stage2_*.pdf` |
| **Ch. 5** | Selective Uncertainty, Pause-Safety vs. OOD Paradox & Finite-$K$ Conformal Audit (`Stage 3`) | 12–14 | `idea_dense_ch5.md`, `metrics_stage3_uncertainty.json`, 2 `stage3_*.pdf` |
| **Ch. 6** | Zero-Touch External Hardware Transfer on 24.2 GHz Neonatal Phantom (`Stage 4`, `L4`) | 9–11 | `idea_dense_ch6.md`, `metrics_stage4_external.json`, 2 `stage4_*.pdf` |
| **Ch. 7** | Discussion, Clinical Readiness Gap (`24.02 bpm`), Limitations, Ethics & Conclusion | 7–9 | `idea_dense_ch7.md` |
| **App. A–C** | Master Notation Glossary (`notation_glossary.tex`), Extended Statistical/Holm Tables, Reproducibility (`5e458c1`) | 8–10 | `notation_glossary.tex`, `audit/thesis_freeze_manifest.sha256` |
