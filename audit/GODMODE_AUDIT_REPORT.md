# God Mode Adversarial Audit & Review Report — Frozen Pre-Manuscript Baseline (`5e458c1`)

**Workspace:** `/Users/hosamshibly/Downloads/medical_radar`  
**Audited Commit:** `5e458c1` (`HEAD -> thesis/main`, **0 tracked files modified during or after test execution**)  
**Audit Date:** `2026-11-23`  
**Scope:** Independent, zero-trust falsification across **6 Adversarial Pillars** covering `thesis/`, `tests/`, `paper_orchestra_bundle/raw_materials/`, `audit/`, frozen `paper_workspace/`, `beltrao_data/`, and `external_data/gleichauf2024/`.  
**Standalone Probe Artifacts (in scratch):** [`p1_p2_results.json`](file:///Users/hosamshibly/.gemini/antigravity/brain/3dcc8a3f-13b7-4f57-b5ce-a2b537115779/scratch/p1_p2_results.json), [`p3_p4_results.json`](file:///Users/hosamshibly/.gemini/antigravity/brain/3dcc8a3f-13b7-4f57-b5ce-a2b537115779/scratch/p3_p4_results.json), [`p5_results.json`](file:///Users/hosamshibly/.gemini/antigravity/brain/3dcc8a3f-13b7-4f57-b5ce-a2b537115779/scratch/p5_results.json).

---

## Executive Verdict: **`PASS WITH MANDATORY FRAMING SAFEGUARDS`**

- **Critical Computational / Leakage / Silent Data Defects:** **`0`** (BPTT gradients, sequence causality, $O(2)\times\mathbb{R}_{>0}$ symmetry, `31/31` Stage 2 salience caches, `154/154` frozen baseline + `49/49` external + `113/113` freeze SHA-256 hashes, `-20s..+20s` lag optimality, and `19/19 + 35/35` test suites all passed under adversarial re-verification).
- **New Findings Surfaced Beyond the Pre-Freeze Audit (`PF-01..PF-15`):**
  1. **`GM-01` (`MAJOR`):** `U6b` is **NOT strictly "zero-threshold"** — it uses a single fixed barrier cutoff $\tau_{\mathrm{ood}} = 0.90$ and history window `history_hops = 3` (`b_ood = clip((h_mix - 0.90)/0.10, 0, 1)`), and all $\tau_{\mathrm{ood}} \le 0.88$ (including threshold-free `b_ood = h_mix`) **fail `< 1.15x` apnea pause safety (`1.177x–1.316x`)**, with only **`[0.90, 0.92]`** satisfying both apnea safety and `> 0.996` WGN AUROC (moreover, `U6b` on `1 Hz` `R2e` has $p_{\mathrm{pat}} = 0.2402$, while `0.1230` was on `Trig -10s`; at $\tau_{\mathrm{ood}} = 0.92$, `U6b` reaches `1.028x` pause safety and $p_{\mathrm{pat}} = 0.0322$!). Only **`U2_Posterior_SD`** is genuinely 0-parameter and 0-threshold ($p_{\mathrm{pat}} = 0.0098$).
  2. **`GM-02` (`MAJOR`):** [wp3_wp4_uncertainty_conformal.py:1186](file:///Users/hosamshibly/Downloads/medical_radar/thesis/wp3_wp4_uncertainty_conformal.py#L1186) defines `valid_distribution_free_guarantee` via empirical `mean(covs_f) >= target_cov`, causing **146 non-patient-exchangeable cells (`8` in `C2`, `138` in `C3`, including `33` cells at $\alpha = 0.05$)** in `metrics_stage3_uncertainty.json` to carry `"valid_distribution_free_guarantee": true` falsely.
  3. **`GM-04` (`DEFENSE-FRAMING`):** **`R2e` (`8p`) vs. `R1a` (`3p`) is Holm-NS on NICU LOPO** (`1 Hz`: $\Delta = -0.1418\;\mathrm{bpm}$ `[-0.3354, +0.0353]`, $p_{\mathrm{sess,Holm}} = 0.2109$, $p_{\mathrm{pat}} = 0.0830$, $p_{\mathrm{pat,Holm}} = 0.6641$; `Trig -10s`: $\Delta = -0.1340$, $p_{\mathrm{sess}} = 0.3052$, $p_{\mathrm{pat}} = 0.5771$). Framing this transparently reinforces parsimony rather than weakening it.
  4. **`GM-05` (`MINOR`):** `HQI` ([wp3_wp4_uncertainty_conformal.py:218-220](file:///Users/hosamshibly/Downloads/medical_radar/thesis/wp3_wp4_uncertainty_conformal.py#L218-L220)) double-counts `fund_mass` in `bg_mass = tot1 + ...`, compressing its `[0, 1]` range to `[0.0655, 0.4351]` (median `0.2075` vs. exact non-overlapping `[0.0700, 0.7193]`), while preserving monotonic rank ($\rho = 0.999945$, $\Delta\mathrm{E\text{-}AURC} = +0.0032\,\mathrm{bpm}$).
  5. **`GM-10` (`MINOR`):** Tectonic compilation of [notation_glossary.tex](file:///Users/hosamshibly/Downloads/medical_radar/paper_orchestra_bundle/raw_materials/notation_glossary.tex) passes cleanly (`0` errors, `0` `TODO`s) but emits `7` `Overfull \hbox > 5pt` warnings (`max 11.99pt`) under default `article` table geometry.
- **Full Discrepancy & Verification Register:** [`audit/GODMODE_FINDINGS.csv`](GODMODE_FINDINGS.csv) (`16` rows `GM-01`–`GM-16`, `10` columns).

---

## Pillar 1 — First-Principles Mathematical & Statistical Verification

### 1.1 Analytical BPTT Gradient Parity (`GM-11` — Verified Exact)
Central finite-difference checks ($\varepsilon = 10^{-6}$ across all parameters for 8p/11p HMMs, 16 stratified coordinates for 43p MLP, and 28 stratified tensor coordinates for the 15,817-param TCN with `dropout_p=0.0`):

| Architecture | Decoder | Coords Tested | Max Abs Error | Max Rel Error ($\varepsilon=10^{-6}$) | Diagnostic Verdict |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `DifferentiableSymmetricHMM` (`linear_filtersafe`, 8p) | `causal_forward` | All 8 × 5 draws | `2.95e-09` | `1.18e-07` | Exact |
| `DifferentiableSymmetricHMM` (`linear_filtersafe`, 8p) | `offline_fb` | All 8 × 5 draws | `3.56e-09` | `3.10e-07` | Exact |
| `DifferentiableSymmetricHMM` (`linear`, 11p) | `causal_forward` | All 11 × 5 draws | `4.38e-09` | `9.39e-08` | Exact |
| `DifferentiableSymmetricHMM` (`linear`, 11p) | `offline_fb` | All 11 × 5 draws | `4.31e-09` | `3.48e-06` | Exact |
| `DifferentiableSymmetricHMM` (`mlp`, 43p) | `causal_forward` | 16–43 × 5 draws | `4.77e-09` (`1.36e-10` @ $10^{-5}$) | `4.38e-04` (`2.46e-06` @ $10^{-5}$)\* | Exact |
| `DifferentiableSymmetricHMM` (`mlp`, 43p) | `offline_fb` | 16 × 5 draws | `4.53e-09` | `1.70e-05` | Exact |
| `CausalDilatedTCN` (`438->24->1`, 15,817p) | Causal Dilated | 28 stratified | `4.50e-07` | `1.85e-06` | Exact |

\*`mlp__causal_forward` worst coordinate (`draw=2, j=33`) had tiny analytical gradient `g_ana = 5.51212e-5`; float64 loss subtraction at $\varepsilon=10^{-6}$ gave `8.46e-10` abs error (`1.36e-10` / `2.46e-6` relative at $\varepsilon=10^{-5}$), confirming zero gradient implementation error.

### 1.2 `HQI` Scale Compression (`GM-05` — Minor Mathematical Quirk)
- At [wp3_wp4_uncertainty_conformal.py:218-220](file:///Users/hosamshibly/Downloads/medical_radar/thesis/wp3_wp4_uncertainty_conformal.py#L218-L220), `hqi = fund_mass / (fund_mass + bg_mass)` includes `fund_mass`'s $P_1$ peak bins inside both `fund_mass` and `bg_mass = tot1 + 0.5*max(p_half) + 1e-6`, writing $x / (2x + R)$ instead of $x / (x + R)$.
- Across all `6,640` valid `pool30_sync` windows: code `HQI` spans `[0.0655, 0.4351]` (`med=0.2075`, `p95=0.3422`) vs. non-overlapping `[0.0700, 0.7193]` (`med=0.2612`, `p95=0.5174`).
- Because the transform is near-monotone ($\rho_{\mathrm{Spearman}} = 0.999945$), `U5` selective-prediction E-AURC changes by only `+0.0032 bpm` (`-0.2284` vs. `-0.2253 bpm`), leaving all Stage 3 findings invariant.

### 1.3 Complete Holm-Bonferroni Audit (`GM-04`, `GM-07`)
- **Confirmatory Stage 1–3 families recomputed:** `max_diff = 0.0` across all `wp1_ablation`, `wp2_ablation`, `rung1_lopo` (`vs R0`, `vs const_loso`), Stage 2 (`vs R0`, `vs R1a`, `vs const_loso`), and Stage 3 (`U1..U6c`).
- **Previously missing Stage 1 `paired_vs_R1g_offline_viterbi` Holm table (`n=8`, `pool30_sync`):**

| Model vs. `R1g` Offline Viterbi | 1 Hz $\Delta$ | 1 Hz $p_{\mathrm{sess}}$ ($p_{\mathrm{sess,Holm}}$) | 1 Hz $p_{\mathrm{pat}}$ ($p_{\mathrm{pat,Holm}}$) | Raw `Trig` $\Delta$ | Raw `Trig` $p_{\mathrm{pat}}$ ($p_{\mathrm{pat,Holm}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `R1a_Online_Forward_MMSE_0s` | `+0.493` | `0.0426` (`0.1347`) | `0.0137` (`0.0957`) | `+0.748` | `0.0010` (`0.0078`) |
| `R1b_Online_Forward_MAP_0s` | `+0.960` | `0.0000` (`0.0000`) | `0.0010` (`0.0078`) | `+1.110` | `0.0010` (`0.0078`) |
| `R1f_Offline_FB_MMSE_Inf` | **`-0.103`** | `0.0048` (`0.0239`) | `0.0674` (`0.2021`) | `-0.087` | `0.1016` (`0.2031`) |
| `R1h_E2E_Causal_H5_OnlineForward_0s` | `+0.639` | `0.0032` (`0.0190`) | `0.0244` (`0.1465`) | `+0.988` | `0.0010` (`0.0078`) |

- **`R2e` vs. `R1a` boundary (`GM-04`):** `1 Hz` $\Delta = -0.1418$ `[-0.3354, +0.0353]`, `9/11` infants, $p_{\mathrm{sess}} = 0.0264$ ($p_{\mathrm{sess,Holm}} = 0.2109$), $p_{\mathrm{pat}} = 0.0830$ ($p_{\mathrm{pat,Holm}} = 0.6641$); `Trig -10s` $\Delta = -0.1340$ `[-0.4968, +0.1664]`, `7/11`, $p_{\mathrm{sess}} = 0.3052$, $p_{\mathrm{pat}} = 0.5771$. Meanwhile both `R1a` (`6.990`) and `R2e` (`6.849`) decisively beat `R0` (`7.817`).

---

## Pillar 2 — Conformal Prediction & Uncertainty Quantification Stress Tests

### 2.1 `U6b` Cutoff ($\tau_{\mathrm{ood}}$) & History (`history_hops`) Stress Test (`GM-01`)
Recomputed directly from Stage 2 `phi_h0` caches (`pool30_sync`, `N=200` Stage 3 WGN, and `gleichauf2024` `05–10 bpm` stopband vs `>= 28 bpm` inband):

| Config (`hops`, $\tau_{\mathrm{ood}}$) | `R2e` 1 Hz $\mathrm{E\text{-}AURC}$ | Wins ($K=11$) | $p_{\mathrm{pat}}$ | Apnea `>5 s` Ratio (`80%` pool) | Safe ($\le 1.15\times$) | WGN AUROC (`FPR@95`) | Gleichauf Stopband AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `hops=3`, `None` (`b_ood = h_mix`) | `-0.2396` | `8/11` | `0.3203` | `1.316x` | **NO** | `0.9990` (`0.0030`) | `1.0000` |
| `hops=3`, $\tau=0.75$ | `-0.2365` | `8/11` | `0.3203` | `1.316x` | **NO** | `0.9990` (`0.0030`) | `1.0000` |
| `hops=3`, $\tau=0.80$ | `-0.2307` | `8/11` | `0.3203` | `1.316x` | **NO** | `0.9990` (`0.0030`) | `1.0000` |
| `hops=3`, $\tau=0.85$ | `-0.2308` | `8/11` | `0.3203` | `1.302x` | **NO** | `0.9990` (`0.0030`) | `1.0000` |
| `hops=3`, $\tau=0.88$ | `-0.2424` | `8/11` | `0.2783` | `1.177x` | **NO** | `0.9989` (`0.0030`) | `1.0000` |
| **`hops=3`, $\tau=0.90$ (Locked)** | **`-0.2398`** | **`8/11`** | **`0.2402`** | **`1.1275x`** (`1.121x` sess) | **YES** | **`0.9986`** (`0.0036`) | **`0.9998`** |
| **`hops=3`, $\tau=0.92$** | **`-0.2282`** | **`9/11`** | **`0.0322`** | **`1.0284x`** | **YES** | **`0.9968`** (`0.0075`) | **`0.9997`** |
| `hops=3`, $\tau=0.95$ | `-0.2194` | `9/11` | `0.0244` | `1.0075x` | **YES** | `0.9482` (`0.2444`) | `0.9992` |
| `hops=0`, $\tau=0.90$ | `-0.2329` | `9/11` | `0.0420` | `1.0666x` | **YES** | `0.9509` (`0.1645`) | `0.9992` |
| `hops=1`, $\tau=0.90$ | `-0.2376` | `8/11` | `0.0830` | `1.1099x` | **YES** | `0.9955` (`0.0139`) | `0.9992` |
| `hops=2`, $\tau=0.90$ | `-0.2403` | `8/11` | `0.1230` | `1.1275x` | **YES** | `0.9979` (`0.0060`) | `0.9998` |
| `hops=5`, $\tau=0.90$ | `-0.2240` | `8/11` | `0.4131` | `1.1275x` | **YES** | `0.9989` (`0.0018`) | `0.9998` |

**Mechanism exposed:** `hmix_nicu_p95 = 0.9130` vs. `hmix_wgn_p05 = 0.9498`. Thus $\tau_{\mathrm{ood}} \in [0.90, 0.92]$ sits naturally inside the physical gap above normal NICU entropy while rejecting WGN/stopband before `0.95`, and all causal windows `hops in {1, 2, 3, 5}` work cleanly at `0.90`. However, because $\tau \le 0.88$ fails apnea safety and $\tau_{\mathrm{ood}} = 0.90$ explicitly references that empirical gap (`wp3_wp4_uncertainty_conformal.py:294`), **`U6b` MUST be framed as a zero-fitted-weight, 1-cutoff heuristic barrier composite** (while **`U2_Posterior_SD`** stands as the strictly parameter-free & cutoff-free uncertainty metric). Also note precision: `U6b` on `R2e` gives $p_{\mathrm{pat}} = 0.2402$ on `1 Hz` and $p_{\mathrm{pat}} = 0.1230$ on `Trig -10s` (`E-AURC = -0.2932`).

### 2.2 Dunn et al. (JASA 2023) Theorem 1 vs. Two-Level `C4`/`C5` (`GM-03`)
Evaluated on the exact 11-fold `pool30_sync` `br_1hz_30s` `R2e` residuals (`K_cal = 10` infants/fold):

| Target $1-\alpha$ | Form | Method | Vacuous? | Patient Mean Cov | Worst Infant Min | Infants Undercovered | Mean Width [Low / High SD] (bpm) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **`0.95`** ($\alpha=0.05$) | All | `Two-Level C4/C5`, `Dunn-Pool`, `Dunn-Sub1` | **YES** | `1.0000` | `1.0000` | `0/11` | `inf` ($\frac{10}{11} < 0.95$) |
| **`0.90`** ($\alpha=0.10$) | Unw (`C4`) | `Two-Level C4` | No | `0.9756` | `0.8750` | `1/11` | `38.84` |
| **`0.90`** ($\alpha=0.10$) | Unw | `Dunn-2023 CDF-Pool` | No | `0.9874` | `0.9144` | `0/11` | `44.26` |
| **`0.90`** ($\alpha=0.10$) | Unw | `Dunn-2023 Subsampling (B=200)` | No | `0.9083` | `0.7020` | `4/11` | `30.61` |
| **`0.90`** ($\alpha=0.10$) | Adapt (`C5`) | `Two-Level C5` | No | `0.9632` | `0.8472` | `1/11` | `39.18` [`29.30` / `49.82`] |
| **`0.90`** ($\alpha=0.10$) | Adapt | `Dunn-2023 CDF-Pool` | No | `0.9880` | `0.9213` | `0/11` | `52.82` [`39.51` / `67.17`] |
| **`0.90`** ($\alpha=0.10$) | Adapt | `Dunn-2023 Subsampling (B=200)` | No | `0.9056` | `0.7010` | `4/11` | `33.83` [`25.35` / `43.01`] |
| **`0.80`** ($\alpha=0.20$) | Unw (`C4`) | `Two-Level C4` | No | `0.8952` | `0.5509` | `2/11` | `25.57` |
| **`0.80`** ($\alpha=0.20$) | Unw | `Dunn-2023 CDF-Pool` | No | `0.8778` | `0.5926` | `2/11` | `23.01` |
| **`0.80`** ($\alpha=0.20$) | Unw | `Dunn-2023 Subsampling (B=200)` | No | `0.8164` | `0.5003` | `4/11` | `20.93` |
| **`0.80`** ($\alpha=0.20$) | Adapt (`C5`) | **`Two-Level C5`** | No | **`0.8636`** | **`0.5856`** | `2/11` | **`24.02`** [`17.96` / `30.54`] |
| **`0.80`** ($\alpha=0.20$) | Adapt | `Dunn-2023 CDF-Pool` | No | `0.8785` | `0.5903` | `1/11` | `25.05` [`18.74` / `31.86`] |
| **`0.80`** ($\alpha=0.20$) | Adapt | `Dunn-2023 Subsampling (B=200)` | No | `0.8174` | `0.5277` | `4/11` | `22.64` [`16.95` / `28.79`] |
| **`0.70`** ($\alpha=0.30$) | Adapt (`C5`) | `Two-Level C5` vs `Dunn-Pool` vs `Sub1` | No | `0.7661` vs `0.7657` vs `0.7232` | `0.4398` vs `0.4213` vs `0.4010` | `3` / `4` / `4` | `17.62` vs `17.69` vs `17.06` |

### 2.3 False `valid_distribution_free_guarantee` Enumeration (`GM-02`) & Stage 4 Own-Quantiles (`GM-06`)
- `GM-02`: Across `metrics_stage3_uncertainty.json["finite_k_conformal_by_cohort"]`, **`146`** non-patient-exchangeable cells emit `"valid_distribution_free_guarantee": true`: `C2` (`3/2/2/1` at $\alpha = 0.05/0.10/0.20/0.30$) and `C3` (`30/36/36/36` across $\alpha = 0.05/0.10/0.20/0.30$).
- `GM-06`: Recomputing each model's **own** `Full-K=11` `C5` $\hat{q}_{\mathrm{norm}}$ on `pool30_sync` `1 Hz` confirms `R2e` (`14.672 / 9.012 / 6.634`) is exact, while stored JSON `conformal_transfer` reused `R2e` onto `R2a` (own: `12.515 / 7.817 / 4.627`) and `R1a` (own via `r1a_mode_cfg`: `6.974 / 4.862 / 2.971`), inflating `R1a` widths by `1.854x–2.233x` and inflating `R1a` `subband_lt28bpm` coverage at $\alpha=0.10$ from `0.4090` to `0.8909`. (`Gate G5`, Table `3B`, and Stage 4 figures evaluate `R2e` exclusively.)

---

## Pillar 3 — Randomized Causality & Group-Invariance Fuzzing (`GM-12` — Verified Clean)

- **Sample-Level Causality (`H5` vs. `H0`):**
  - `H5_Causal_COLA_InBand_L0`: `h5_raw_window_causality_max_diff = 0.0` across 5 random future perturbations ($+10^4$ noise spikes after window end).
  - `H0_Locked_Rect_Full_L3`: Perturbing samples $t \ge 240$ leaves STFT frames $m \le 11$ unchanged (`0.0`) and HPSS mask frames $m \le 9$ (`m+3=12` ending at `240`) bit-identical (**`0.0`**), turning non-zero precisely at frame $m = 10$ (`0.4103`, touching sample `240` inside `+3`-hop frame `13` spanning `[208, 256)`) — exact empirical verification of the `48-sample = 3.0 s` lookahead horizon.
- **Sequence-Level Causality (16 random future cutoffs across NICU `7.3` & `gleichauf2024` `45BPM_normal`):**
  - `max_abs_diff == 0.0` for `R2e_est`, `R2e_sd`, `R2d_est`, `R3a_tcn`, `U1`, `U2`, `U3`, `U5`, and `U6b`.
- **20 Random $O(2)\times\mathbb{R}_{>0}$ + Complex DC Transforms ($\theta \in [0, 2\pi)$, $g \in [0.05, 50]$, $c_I,c_Q \in [-500, 500]$, 10 swaps):**
  - Full $O(2)\times\mathbb{R}_{>0}$ + DC (`H5` / `R2d`): `phi_h5 <= 2.31e-14`, `R2d_est <= 2.13e-14 bpm`, `R2d_sd <= 5.77e-15 bpm`.
  - $SO(2)\times\mathbb{R}_{>0}$ + DC (`H0` / `R2e`): `phi_h0 <= 3.38e-14`, `R2e_est <= 1.42e-14 bpm`, `U2 <= 8.33e-17`, `U6b <= 2.22e-16`.
  - Reflection Swap `I<->Q` (`H0` Nyquist bin `idx=0` clamping): `phi_h0 <= 1.96e-3`, `R2e_est <= 1.29e-4 bpm`, `U2 <= 3.11e-6`, `U6b <= 5.24e-6`.

---

## Pillar 4 — All-31 Cache Verification, VitaGuard Lag Sweep & Stage 4 Forensics

### 4.1 All-31 Stage 2 Salience Caches (`GM-13` — `PF-05` Closed)
Recomputing 8 stratified windows per session across ALL `31/31` Stage 2 cache files (`248/248` windows total) vs. live `extract_salience_features_window`: **`worst_max_abs_diff_h0 = 0.0`**, **`worst_max_abs_diff_h5 = 0.0`** (in float32).

### 4.2 VitaGuard `[-20 s, +20 s]` Lag Sweep (`GM-14` — Hardware Monitor Latency Proof)
Equal-session `RMSE_sess` (bpm) across `pool30_sync` ($S=28$) shifting EDF trigger reference by $\Delta\tau$ (where `+10 s` corresponds to `lag_s = lag_val + 10`, i.e., `Trig -10s`):

| EDF Offset $\Delta\tau$ | `R0` (`0p`) [$r_{\mathrm{within}}$] | `R1a` (`3p`) | `R2e` (`8p`) [$r_{\mathrm{within}}$, `U2` $\mathrm{E\text{-}AURC}$] | `R3c` (`439p`) | `R3b` (`15k`) | `R3a` (`16k`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `-10 s` | `9.767` (`0.254`) | `9.306` | `9.270` (`0.258`, `-0.155`) | `8.421` | `8.976` | `8.641` |
| `0 s` (Raw Sync) | `8.968` (`0.343`) | `8.624` | `8.473` (`0.352`, `-0.277`) | `8.255` | `8.791` | `8.586` |
| `+4 s` | `8.773` (`0.381`) | `8.365` | `8.231` (`0.397`, `-0.228`) | `8.130` | `8.610` | **`8.438`** (min) |
| `+8 s` | **`8.706`** (`0.401`) | `8.221` | `8.071` (`0.430`, `-0.252`) | `8.108` | `8.555` | `8.463` |
| **`+10 s` (`Trig -10s`)** | `8.727` (**`0.402`**) | **`8.197`** (min) | **`8.063`** (`0.441`, **`-0.293`**) | **`8.102`** (min) | **`8.515`** (min) | `8.498` |
| `+12 s` | `8.852` (`0.395`) | `8.230` | `8.149` (**`0.446`**, `-0.290`) | `8.157` | `8.570` | `8.555` |
| `+20 s` | `9.751` (`0.341`) | `8.811` | `9.001` (`0.405`, `-0.295`) | `8.531` | `8.897` | `8.886` |

Five of six unshifted architectures (`R0`, `R1a`, `R2e`, `R3c`, `R3b`) reach their minimum error or maximum $r_{\mathrm{within}}$ identically at `+8..+10 s`, proving `+10 s` is a hardware VitaGuard monitor delay, NOT estimator cherry-picking.

### 4.3 Stage 4 External `gleichauf2024` Forensics (`GM-09`, `GM-15`)
- **Zero Steady-State Outliers:** Across all 14 `inband_ge28bpm` steady-state `[10 s, T-10 s]` scenarios, `steady_inband_outliers_gt_0p5_bpm = []` (`max_abs_err`: `R0 = 0.096`, `R2e = 0.158`, `R1a = 0.162 bpm`), while excluded edge transients reach `78.9–155.5 bpm`.
- **Two-Stage `resample_poly` Gain Ripple:** Within `[+0.0023, -0.0003] dB` across `5–80 bpm` (essentially lossless).
- **Rung 3 Shortcut Collapse (`20 bpm` Skirt vs `>= 28 bpm` Inband):** On the `-10 dB` attenuated `20 bpm` skirt, `R3c` (trained on all 11 infants!), `R3a`, and `R3b` snap toward the `~45–50 bpm` NICU training mean (`30.5–32.6`, `42.7–44.0`, and `51.2–54.1 bpm`; `RMSE = 11.78 / 23.47 / 32.70 bpm`), while equivariant `R2e` (`20.055–20.057 bpm`, `RMSE = 0.095`) and `R2a` (`20.001–20.003 bpm`, `RMSE = 0.024`) remain phase-locked cleanly.

---

## Pillar 5 — Manuscript-Readiness, BibTeX, LaTeX & Post-Suite Immutability (`GM-10`, `GM-16`)

1. **71-Entry BibTeX & CSV Audit:** `71/71` entries in [seed_references.bib](file:///Users/hosamshibly/Downloads/medical_radar/paper_orchestra_bundle/raw_materials/seed_references.bib) have `0` brace imbalances, `0` duplicate keys, `0` invalid DOIs, `0` inner-title acronym casing leaks (`{NICU}`, `{I/Q}`, `{CW}`, etc. properly braced), and `0` mismatches vs. [lit_review_matrix.csv](file:///Users/hosamshibly/Downloads/medical_radar/paper_orchestra_bundle/raw_materials/lit_review_matrix.csv).
2. **Tectonic Glossary Check (`GM-10`):** Compiling [notation_glossary.tex](file:///Users/hosamshibly/Downloads/medical_radar/paper_orchestra_bundle/raw_materials/notation_glossary.tex) exits `0` with `0` LaTeX errors and `0` remaining `TODO`s, flagging `7` `Overfull \hbox > 5pt` lines (`140`: `11.99pt`, `194`: `11.78pt`, `185`: `11.44pt`, `187`: `10.77pt`, `196/197/199`: `5.58pt`) to clean during LaTeX appendix layout.
3. **All 13 Dual Figures:** `13/13` paired `.pdf`/`.png` files, `1` page each, `Type0` TrueType vector fonts (`0` Type3), `300 DPI` PNGs.
4. **Post-Suite Immutability Verified Live:** After running **both** `/usr/bin/python3 -m paper_workspace.src.test_pipeline` (`ALL TESTS PASSED`, 19/19) **and** bare `/opt/homebrew/bin/python3.10 -m pytest --rootdir=. -v` (`35 passed`, 40.74 s), `git status --short` on `5e458c1` stays completely empty, `audit/thesis_freeze_manifest.sha256` verifies **`113/113 OK`**, `audit/postfix_manifest.sha256` baseline verifies **`154/154` + `33/33` caches OK**, `external_data/gleichauf2024` verifies **`49/49` OK`**, and `find paper_workspace -name "*.cpython-310.pyc"` returns empty.

---

## Pillar 6 — BTU Cottbus Module 13600 Defense Checklist

See the full hostile 3-examiner panel simulation (**Signal Processing Prof**, **Statistical ML Prof**, **Clinical Neonatologist**) in [`audit/DEFENSE_REDTEAM_QA.md`](DEFENSE_REDTEAM_QA.md), and inject these **6 non-negotiable drafting guardrails** into `PaperOrchestra` / Ch. 1–7:
1. Frame **`U2_Posterior_SD`** as the 0-param/0-threshold uncertainty score ($p_{\mathrm{pat}}=0.0098$) and **`U6b`** strictly as a *zero-fitted-weight heuristic barrier composite* ($\tau_{\mathrm{ood}}=0.90, H=3$; cite 1 Hz $p_{\mathrm{pat}}=0.2402$ vs Trig -10s $0.1230$, noting $[0.90, 0.92]$ stability).
2. Ignore `valid_distribution_free_guarantee` on `C2`/`C3` (`GM-02`), frame `C4`/`C5` as *two-level patient-quantile hierarchical conformal inspired by Dunn et al. (2023)* (`GM-03`), and cite Stage 4 `conformal_transfer` strictly for `R2e` (`GM-06`).
3. Frame `R2e` (`8p`) vs. `R1a` (`3p`) honestly as Holm-NS internal refinement (`GM-04`) confirming rather than weakening the parsimony thesis.
4. Clarify `HQI` raw scale compression (`[0.065, 0.435]`, $\rho=0.999945$, `GM-05`) and `low/high_uncertainty_half_width_bpm` full-width semantics (`GM-07`).
5. Cite ONLY locked `64-tap Kaiser FIR [0.30, 3.00] Hz (beta=6.5)` (`GM-08`), disclose `R3a/b` `N=10` vs `R3c/R2*` `N=11` (`GM-09`), and explicitly distinguish `Gate G5` operationalized numeric cutoffs from §10 pre-registration (`GM-15`).
6. Hold **zero `L2` (exact Beltrao reproduction) or `L5` (clinical bedside utility)** claims: frame `24.02 bpm` (`80%`) conformal bounds and `gleichauf2024` stationary `L4` transfer as quantitative boundaries defining what multi-antenna MIMO/FMCW hardware must solve next.
