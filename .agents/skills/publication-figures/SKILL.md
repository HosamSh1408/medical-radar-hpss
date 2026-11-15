---
name: publication-figures
description: >-
  Generates IEEE and Master's thesis publication figures using Matplotlib with locked notation
  labels, Okabe-Ito colorblind-safe palettes, vector PDF + 300 DPI PNG export, and mandatory
  visual inspection. Use when creating spectrograms, tracking plots, Bland-Altman plots,
  Risk-Coverage curves, or ablation comparison figures.
---

# Publication Figures Standard

Creates deterministic, typography-consistent figures grounded directly in provenance-locked `.npz` or `.json` artifacts (never hard-coding numbers in plotting scripts).

## Rendering & File Conventions
1. **Interpreter & Cache Env:**
   ```bash
   MPLCONFIGDIR=/tmp/mpl /opt/homebrew/bin/python3.10 <plot_script.py>
   ```
2. **Dual Export:** Save every figure in `paper_orchestra_bundle/raw_materials/figures/` as both **vector `.pdf`** (embedded directly in LaTeX) and **`.png` (`dpi=300`, `bbox_inches='tight'`)** for quick preview.
3. **Sizing for Thesis & IEEE Spin-Off:**
   - Full-width figure: `figsize=(6.8, 3.6–4.4)` inches (`9–10 pt` axis/tick fonts matching document body text).
   - Half-width / IEEE single-column figure: `figsize=(3.5, 2.5–3.0)` inches.

## Typography, Notation & Palette Rules
- **Notation Parity:** Axis labels, legends, and annotations MUST match [notation_glossary.tex](file:///Users/hosamshibly/Downloads/medical_radar/paper_orchestra_bundle/raw_materials/notation_glossary.tex) (e.g., $\mathrm{RMSE}_{\mathrm{sess}}$ vs. $\mathrm{RMSE}_{\mathrm{pool}}$, $\bar{r}_{\mathrm{within}}^{(\mathrm{sess})}$, $M_H$, $\hat{f}_b$, $f^{\mathrm{ref}}$, $c_{\mathrm{cov}}$). NEVER print raw Python identifiers (`rmse_hpss_per_window`, `d_hpss`, `est_nls`) inside figure legends or axis titles.
- **Colorblind-Safe Palette (Okabe–Ito):** `#0072B2` (Blue — proposed/HPSS), `#D55E00` (Vermilion — NMF baseline), `#009E73` (Bluish Green — reference/oracle), `#E69F00` (Orange), `#CC79A7` (Reddish Purple), `#56B4E9` (Sky Blue), `#333333` (Dark Gray — threshold lines); pair colors with distinct linestyles/markers so figures remain legible in grayscale.
- **Plot-Type Checklists:**
  - *Bland–Altman:* Plot bias ($\hat{\mu}_d$), repeated-measures 95% LoA bands ($\mathrm{LoA}_{\mathrm{lo}}, \mathrm{LoA}_{\mathrm{up}}$), and annotate $S$ sessions / $|\mathcal{W}|$ windows.
  - *Risk–Coverage / AURC:* Plot normalized coverage $c_{\mathrm{cov}} \in [0.5, 1.0]$ on x-axis, annotate $\mathrm{AURC}$ & E-AURC, and overlay random-abstention baseline.
  - *STFT / HPSS Spectrograms:* Label frequency in $\mathrm{Hz}$ (with secondary $\mathrm{bpm}$ ticks or clear `[0.3, 3.0] Hz` bracket) and consistent dB dynamic range across panels (`S_H`, `S_P`, `M_H`).

## Mandatory Visual Verification
Inspect every exported `.png` / rendered `.pdf` with `read_file` before adding it to `paper_orchestra_bundle/` to confirm zero clipped tick labels, overlapping legends, or truncated titles.
