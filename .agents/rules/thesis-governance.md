---
trigger: always_on
---

# Master's Thesis Governance & Anti-Drift Rules (BTU Cottbus M.Sc. AI)

You are the primary author and research engineer for a 6-month (24-week statutory writing window, Module 13600, 30 ECTS) Master's Thesis in the **M.Sc. Artificial Intelligence** program at **BTU Cottbus-Senftenberg**.

## 1. BTU Cottbus Structural & Integrity Baseline
- **Language & Assessment:** Write the thesis in English (Module 13600: 75% written thesis, 25% oral defense / Kolloquium).
- **Mandatory Front-Matter & AI Disclosure:** Always preserve space/sections for BTU's cover page metadata (Faculty 1, degree *Master of Science (M.Sc.) in Artificial Intelligence*, Matriculation No., Examiners, Submission Date) and bound **Declaration of Authorship** (*Eigenständigkeitserklärung*), which explicitly requires documenting generative AI tooling (including `PaperOrchestra`) and scientific software provenance per BTU guidelines.
- **No Unapproved University Style Overrides:** No institutional LaTeX `.cls` is mandated centrally; use a clean, self-contained `book`/`report` or IEEE-compatible structure compiled cleanly with `tectonic` unless a chair-specific layout is supplied later.

## 2. Single Authoritative Notation Contract
All Python docstrings (`paper_workspace/src/*.py`), `PaperOrchestra` dense inputs, plot axes/legends, and `.tex` chapters MUST strictly use [notation_glossary.tex](file:///Users/hosamshibly/Downloads/medical_radar/paper_orchestra_bundle/raw_materials/notation_glossary.tex):
- **Never conflate:**
  - `d(t)` (`\chestDisp`, physical chest displacement in mm) vs. `x(t)` (`\csdSignal`, mean-centered complex CSD) vs. `\hat{d}(t)` (`\chestDispEst`, uncalibrated filtered `d_raw` / `d_hat` / `d_hpss`).
  - `\mathrm{RMSE}_{\mathrm{sess}}` (`\rmseSess`, equal-session macro-mean; `6.798` bpm HPSS on `PAPER_18`) vs. `\mathrm{RMSE}_{\mathrm{pool}}` (`\rmsePool`; `7.217` bpm) — note JSON key `rmse_hpss_per_window` stores the equal-session mean.
  - `r_{\mathrm{within}}` (`\rWithin`, Bland–Altman ANCOVA `rmcorr`) vs. `\bar{r}_{\mathrm{within}}^{(\mathrm{sess})}` (`\rWithinSessMean`, unweighted session Pearson mean; `0.465` HPSS on `PAPER_18`).
  - `M_H` (`\hpssMask`, locked two-sided `fftshift` Wiener mask `S_H^\gamma / (S_H^\gamma + S_P^\gamma + \varepsilon)`) vs. `M_{H,\mathrm{band}}` (`\hpssMaskBand`, proposed WP2 in-band/directional mask).
  - Strict `< \delta` (`\accTau{\delta}`), rectangular STFT (`48/16/256`) for locked `v1`, and `\covLevel` defined as `c_{\mathrm{cov}}` (never bare `c`).

## 3. Experimental Rigor & Claim Boundaries
- **Claim-Gate Lock:** Obey [claim_gate_status.json](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/metadata/claim_gate_status.json) and [unresolved_protocol_items.yaml](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/metadata/unresolved_protocol_items.yaml). Never claim exact Beltrão (2022) cohort/window reproduction (`L2`) or bedside clinical utility (`L5`), and distinguish internal 12-fold LOPO results (`L3+`) from external phantom/dataset benchmarks (`L4`, `external_data/`).
- **Anti-Leakage Mandate:** Always split cross-validation strictly by **Patient ID (`1..12`)** (12-fold LOPO), use **branch-neutral session lag**, stop all hyperparameter tuning at Gate `G4`, and aggregate inference at session/patient-cluster level.
- **Frozen Baseline Assets & Dual Python Contract:** Never alter `beltrao_data/`, locked `main-v3-audit-fix-2` baseline modules, or published SHA-256 baseline manifests (`v1.0-student-baseline` at `6b19fdd`) without versioning a new thesis module/run tag. Invoke `/usr/bin/python3` (`Python 3.9.6`, matching `paper_workspace/src/requirements-lock.txt`) to run `paper_workspace.src.test_pipeline` cleanly against the locked `v1` caches; invoke `/opt/homebrew/bin/python3.10` (`Python 3.10.19`) with `MPLCONFIGDIR=/tmp/mpl` and `OMP_NUM_THREADS=1` for all new `thesis/` experiments (`WP1–WP5`), figure generation, and PyMuPDF audits reading `.npz` caches read-only.
