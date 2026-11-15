---
name: paper-orchestra-bridge
description: >-
  Prepares chapter-scoped raw_materials inputs for google-research/paper-orchestra, runs
  pre-flight schema and notation validation, executes modular chapter or companion IEEE
  manuscript passes, and runs post-generation LaTeX-to-cache truth and BibTeX audits. Use
  whenever assembling dense outlines, updating paper_orchestra_bundle/, or auditing drafted chapters.
---

# PaperOrchestra Bridge & Truth Audit

Safely couples provenance-locked experiments in `paper_workspace/` to modular `google-research/paper-orchestra` runs while preventing numerical drift, citation hallucinations, or notation collisions.

## Critical Guardrails
- **Modular Scope:** Never feed an entire 75–95 page thesis into one monolithic `PaperOrchestra` run. Run **chapter-by-chapter** (`Ch. 1–6`) using a shared master shell (`main_thesis.tex`), plus an optional unified 8–10 page `IEEEtran` spin-off pass after Gate `G5`.
- **No Autonomous Quantitative Plotting:** Always pass `--use_plotting False` (or restrict autonomous diagramming strictly to conceptual block/architecture flowcharts that are visually inspected afterward). All quantitative results plots MUST come pre-rendered from `paper_workspace/`.
- **Strict Claim Boundaries:** Inject [claim_gate_status.json](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/metadata/claim_gate_status.json) (`L0–L5`) and [unresolved_protocol_items.yaml](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/metadata/unresolved_protocol_items.yaml) constraints into `guidelines_thesis.md` so the Refinement Agent cannot invent `L4`/`L5` claims or exact-original-cohort claims.

## Step-by-Step Workflow
1. **Stage Chapter Bundle (`paper_orchestra_bundle/raw_materials/`):**
   - `idea_dense_chX.md`: frozen derivations using macros from [notation_glossary.tex](file:///Users/hosamshibly/Downloads/medical_radar/paper_orchestra_bundle/raw_materials/notation_glossary.tex).
   - `exp_log_chX.md` & `metrics_chX.json`: exact tables exported directly from SHA-256 `.npz` caches.
   - `figures/`: verified vector `.pdf` charts generated via `publication-figures`.
   - `seed_references.bib` & `lit_review_matrix.csv`: pre-verified 60+ seed references.
2. **Pre-Flight Consistency Check:**
   Verify 1-to-1 `bib_key`/`year`/`cluster`/DOI parity between `.bib` and `.csv`, braced BibTeX acronyms, balanced LaTeX braces, zero macro collisions (`\covLevel` vs `\speedLight`), and valid Python identifiers before invoking `paper_writing_cli.py`.
3. **Post-Generation `EXP-8B` Verification Gate:**
   - **Numeric Truth Audit:** Extract every metric, p-value, confidence interval, session count, and window count in generated `.tex` files and confirm 100% agreement against `metrics_ch*.json` and `audit/DOC_TRUTH_TABLE.csv`.
   - **Citation Audit:** Verify every `\cite{...}` added by the Literature Review Agent resolves to a real DOI/venue via OpenAlex/PubMed/arXiv skills and matches the cited empirical claim accurately.
   - **Compile Check:** Build cleanly with `tectonic-latex-thesis` before human polishing.
