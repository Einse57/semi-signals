# Detector A + Detector B (video-only) noisy-OR fusion: preregistration addendum

Written 2026-10-25 23:43 PDT. At this time no Detector B (VSViG LOPO) held-out or inner-fold
probability has been produced or looked at (the first outer fold is still training).
Addendum to `docs/vsvig_lopo_preregistration.md`; everything not listed here follows that document.

## Rule
* Per second: p_F = 1 - (1 - p_A) * (1 - p_B) (`src/fusion.noisy_or`), on the raw (unsmoothed)
  per-second probabilities. p_A = Detector A LOPO out-of-fold probability (`results/detector_a/oof_probs.npz`),
  p_B = Detector B LOPO outer-fold probability. If the two series differ in length, the shorter length is used.
* Post-processing and scoring identical to Detector A (5 s smooth, events >= 3 s, merge < 30 s,
  tolerance 30 s before / 60 s after; Wilson, Poisson, patient bootstrap; delays; per-patient AUROC of p_F).

## Thresholds
* **Primary (nested):** for each outer fold, Detector A is retrained with its fixed LightGBM recipe on the
  same 3 patient-grouped inner folds used for Detector B; the inner out-of-fold p_A and p_B are fused the
  same way, and the threshold is chosen with the same rule (lowest 0.05-grid value with FA/h <= 1, else 0.95).
* **Secondary:** fixed 0.5.

## Decision: "fusion helps"
Under the primary rule, fusion helps if, versus Detector A alone on the same 33 seizures / 3.73 interictal h,
it detects more seizures with FA/h no higher, or has lower FA/h with no fewer seizures detected.
Differences are also reported as a patient-bootstrap 95% interval (2000 resamples, seed 0); with 33 seizures
we expect wide intervals and will say so. Any other comparison (other rules, weights, max-fusion, delays as a
criterion) is exploratory and labelled as such.
