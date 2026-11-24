# Master Briefing Prompt for Modular Thesis Drafting (`PaperOrchestra` / Direct LaTeX)

Use this prompt to generate or refine Chapters `2–7` and Appendices `B–C` consistently inside the BTU Cottbus-Senftenberg M.Sc. AI Master's Thesis scaffold ([`manuscript/main_thesis.tex`](../manuscript/main_thesis.tex)), preserving byte-level immutability across commit `5e458c1`.

---

## Prompt Template

> You are drafting **Chapter `<N>`** of the BTU Cottbus-Senftenberg Master's Thesis (*M.Sc. Artificial Intelligence*, Module 13600, English monograph, KOMA-Script `scrreprt`).
>
> **Mandatory Authoritative Inputs (Read Before Writing a Single Word):**
> 1. Structural, epistemic & statistical guardrails: [`paper_orchestra_bundle/guidelines_thesis.md`](guidelines_thesis.md) (obey **ALL 8 Guardrails G1–G8** without exception).
> 2. Authoritative symbol & macro contract: [`paper_orchestra_bundle/raw_materials/notation_glossary.tex`](raw_materials/notation_glossary.tex) (loaded via `\def\NOTATIONMACROSONLY{1}` in preamble; strictly use macros `\brEst`, `\brRef`, `\rmseSess`, `\rmsePool`, `\rWithin`, `\rWithinSessMean`, `\covLevel`, `\eaurcMetric`, `\hpssMask`, `\hpssMaskBand`, `\chestDisp`, `\csdSignal`, `\chestDispEst`, `\accTau{...}`, etc.).
> 3. Target chapter blueprint: [`paper_orchestra_bundle/raw_materials/idea_dense_ch<N>.md`](raw_materials/) and its cited frozen JSON source (`metrics_*.json`) plus [`audit/GODMODE_AUDIT_REPORT.md`](../audit/GODMODE_AUDIT_REPORT.md).
> 4. Verified references (`71` keys only): [`paper_orchestra_bundle/raw_materials/seed_references.bib`](raw_materials/seed_references.bib).
> 5. Frozen figures directory: [`paper_orchestra_bundle/raw_materials/figures/`](raw_materials/figures/) (reference exact `.pdf` stems via `\includegraphics` cleanly inside `\begin{figure}` environments).
>
> **Hard Prohibitions:**
> - **NEVER** modify any Python file, data file, JSON file, `.pdf`/`.png` figure, `notation_glossary.tex`, `seed_references.bib`, `idea_dense_ch2_wp0.md`, or `audit/thesis_freeze_manifest.sha256`.
> - **NEVER** assert `L2` (exact Beltrão 2022 cohort/window/solver reproduction) or `L5` (clinical bedside utility or autonomous apnea safety).
> - **NEVER** state `"Butterworth"` for the locked bandpass (locked bandpass is **`64-tap zero-phase Kaiser FIR [0.30, 3.00] Hz, beta = 6.5`**).
> - **NEVER** call `U6b_ZeroTuned_Geometric_SQI` `"zero-threshold"` or `"parameter-free"` (call it a **zero-fitted-weight heuristic barrier composite** with fixed $\tau_{\mathrm{ood}} = 0.90$, causal `history_hops = 3`; report both `1 Hz` $p_{\mathrm{pat}} = 0.2402$ and `Trig -10s` $p_{\mathrm{pat}} = 0.1230$; reserve `"zero-parameter / threshold-free"` strictly for `U2_Posterior_SD` at $p_{\mathrm{pat}} = 0.0098$).
> - **NEVER** read `"valid_distribution_free_guarantee"` from `metrics_stage3_uncertainty.json` (`146` false `True` cells on `C2`/`C3`).
> - **NEVER** double `low/high_uncertainty_half_width_bpm` (`17.96` / `30.54 bpm` at `80%` are already **full widths**) or cite `conformal_transfer` in Stage 4 for `R1a`/`R2a` (`R2e` only).
>
> **Verification Protocol After Drafting Chapter `<N>`:**
> 1. Compile [`manuscript/main_thesis.tex`](../manuscript/main_thesis.tex) via `/opt/homebrew/bin/tectonic --keep-logs --outdir build main_thesis.tex` inside `manuscript/`.
> 2. Confirm `0` LaTeX errors, `0` undefined citations/references, and `0` `Overfull \hbox > 5pt` warnings.
> 3. Render updated PDF pages to PNGs via PyMuPDF (`fitz`) and inspect tables/figures visually.
> 4. Verify `shasum -a 256 -c audit/thesis_freeze_manifest.sha256` still passes **`113/113 OK`**.
