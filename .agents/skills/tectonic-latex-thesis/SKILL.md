---
name: tectonic-latex-thesis
description: >-
  Compiles thesis LaTeX chapters and master manuscripts locally using Tectonic (XeTeX + BibTeX),
  enforces notation_glossary.tex macro wiring and BTU Cottbus front-matter requirements, checks
  bibliography resolution and layout warnings, and renders PDF pages via PyMuPDF for visual
  inspection. Use whenever editing, assembling, debugging, or verifying .tex or .bib files.
---

# Tectonic LaTeX Thesis Builder & Visual Verifier

Builds publication-grade LaTeX PDFs locally using `/opt/homebrew/bin/tectonic` (`0.16.9`) and visually inspects rendered pages using `PyMuPDF` (`fitz`).

## Standard Thesis Macro Wiring
1. **Preamble of `main_thesis.tex` (loads macros without printing glossary tables):**
   ```latex
   \def\NOTATIONMACROSONLY{1}
   \input{paper_orchestra_bundle/raw_materials/notation_glossary.tex}
   ```
2. **Appendix / Nomenclature chapter body (renders full tables):**
   ```latex
   \let\NOTATIONMACROSONLY\undefined
   \input{paper_orchestra_bundle/raw_materials/notation_glossary.tex}
   ```
3. **Required packages in `main_thesis.tex`:** `amsmath`, `amssymb`, `booktabs`, `longtable`, `array`, `graphicx`, `hyperref`, `siunitx`.

## BTU Cottbus M.Sc. AI Front-Matter Checklist
- **Title Page:** University (*BTU Cottbus-Senftenberg*), Faculty 1, degree (*Master of Science in Artificial Intelligence*), Module `13600`, thesis title, student name & matriculation number, supervisors/examiners, submission date.
- **Declaration of Authorship (*Eigenständigkeitserklärung*):** Bound declaration including explicit disclosure and documentation of all generative AI tooling (`PaperOrchestra` drafting workflow grounded in provenance-locked Python logs) and plagiarism-check consent per BTU standards.

## Build & Verification Workflow
1. **Compile PDF:**
   ```bash
   /opt/homebrew/bin/tectonic -X compile <main_or_chapter>.tex --keep-logs
   ```
   *(macOS Note: `tectonic` panics inside the agent seatbelt sandbox due to `reqwest` -> `system-configuration` SCDynamicStore access; run `tectonic` outside sandbox or in user terminal, and pass `-C` once the local TeXLive CTAN cache is warmed.)*
2. **Zero-Error Log Audit:**
   Inspect stdout and `.log`/`.blg` output and fix EVERY occurrence of:
   - `Citation '...' undefined` or `Empty `thebibliography'`
   - `There were undefined references` / `multiply-defined labels`
   - `Overfull \hbox` (`> 5 pt`) or `Missing character` warnings
   - Title capitalization leaks in `.bib` (missing `{...}` around acronyms/proper nouns).
3. **Visual PDF Page Inspection:**
   Render target pages (title page, equations, longtables, figure floats) to PNGs in `<scratch_dir>/pdf_pages/` via `/opt/homebrew/bin/python3.10` + `fitz` (`PyMuPDF`, `150–200 dpi`), then inspect them with `read_file` to catch clipped equations, awkward page breaks, or overlapping labels before delivering drafts.
