# Analysis plan v1 — independent NMF/HPSS reconstruction

Status: computational lock for the next confirmatory run; not a claim of exact
Beltrão reproduction and not a clinical validation protocol.

## Objective

Compare the locked NMF implementation with the frozen HPSS implementation on
paired radar/reference recordings available in this repository, while keeping
cohort membership and evaluation rules separate from the results.

## Primary estimand

The primary internal estimand is the paired difference in per-measurement
respiratory-rate RMSE:

`HPSS RMSE - NMF RMSE` in bpm.

Negative values favor HPSS. Each measurement contributes one summary, so long
recordings do not receive disproportionate weight. Patient-group clustered
summaries are a required sensitivity analysis because several measurements come
from the same patient.

## Secondary estimands

- Acc@6 and Acc@10, reported in percentage points;
- MAE and signed bias in bpm;
- measured per-window runtime;
- valid-window count, invalid-window count, and NMF convergence rate;
- failure and missingness rates by measurement.

## Frozen evaluation rules

- Use the active provenance-checked pipeline and its locked NMF/HPSS settings.
- Do not tune HPSS using the outcome of any cohort variant.
- Report all four predefined cohort variants in the reconstruction multiverse.
- Report per-measurement summaries and pooled-window summaries separately.
- Use deterministic bootstrap resampling with seed `20261011`; the current
  report includes measurement-level and patient-group-level intervals.
- Preserve zero-valid-window measurements in the cohort ledger and report them
  as non-evaluable rather than silently removing them.

## Superiority gate

No superiority margin is declared in this repository because a clinically or
engineering-justified margin requires domain input that is not available here.
Until a margin is justified and frozen, results may be described only as
exploratory internal comparisons. Crossing zero is not treated as proof of
equivalence, and a nominal improvement is not treated as clinical benefit.

## Claim boundary

This plan cannot establish exact paper reproduction because the original cohort
and several protocol details remain unresolved. It also cannot establish
generalizable or clinical superiority without untouched external data,
patient-level clustering, failure analysis, clinical utility endpoints, and
target-hardware evidence.
