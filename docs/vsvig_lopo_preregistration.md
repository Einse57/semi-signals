# Detector B (VSViG-base) leave-one-patient-out retrain: preregistration

Written 2026-10-25 23:02 PDT, before any VSViG training or scoring on held-out patients.

## Why
The released VSViG-base weights were trained on all 14 WU-SAHZU patients, so their scores on this set
are not an honest estimate. This retrain gives a leave-one-patient-out (LOPO) estimate for the same
architecture on our own keypoints, scored exactly like Detector A.

## Data
* Same 36 labelled videos and per-second labels as Detector A (`src/detector_a/labels.py`):
  interictal = t < EEG onset and all of `free` videos; ictal = clinical onset to +120 s; everything else excluded.
  `pat04_no-Sz2P` and `pat05_Sz4PG` excluded (no label row).
* Keypoints: the same semi-signals extraction used for Detector A (subject track, VSViG joint order).
  Patches are rebuilt on the training machine from the original videos with the same
  `extract_vsvig_patches` at the same frame indices (128 px Gaussian crop, min-max, 32x32 RGB).
* Window = 30 samples (5 s at 6/s) ending at each whole second; window label = label of its end second.
  Windows whose end second is excluded are not used for training or scoring.

## Model and training (fixed, no tuning on held-out patients)
* Architecture: VSViG-base (`third_party/vsvig/VSViG.py`), **random initialisation** (released weights not used).
* Loss: BCE with pos_weight = n_interictal / n_ictal of the training fold.
* Optimiser: AdamW, lr 3e-4, weight decay 1e-2, cosine decay to 0, batch 32, 10 epochs, fp16 autocast, seed 0.
* Training windows: every ictal window; interictal windows at a 2 s stride.
* Augmentation: none. Last-epoch weights are used (no early stopping, no checkpoint selection).

## Outer loop and scoring
* 14 outer folds, one per held-out patient. The held-out patient's windows at a 1 s stride give a
  per-second probability (seconds before the first full window = 0).
* Post-processing and scoring are identical to Detector A (`src/detector_a/scoring.py`): 5 s trailing smooth,
  events >= 3 s, merge gaps < 30 s, SzCORE-style tolerance 30 s before / 60 s after;
  Wilson CI on recall, Poisson CI on FA/h, patient bootstrap, delays vs clinical and EEG onset, per-patient AUROC.

## Threshold rules
* **Primary (nested):** inside each outer fold, 3 patient-grouped inner folds (patients sorted by ID, assigned
  round-robin) are trained with the same recipe; on the pooled inner out-of-fold probabilities, the lowest
  threshold on the 0.05 grid (0.05-0.95) with FA/h <= 1 is chosen, else 0.95.
  (Detector A used inner LOPO; 3 inner folds are used here for compute. This is the only planned difference.)
* **Secondary:** fixed threshold 0.5.
* Run order: all 14 outer models first (secondary rule + AUROC), then the inner folds.
  If the inner folds cannot finish by Tuesday 2026-10-27 18:00 PT, the primary rule is reported as not run.

## Reported alongside
* Released VSViG-base weights on the same windows (leaky, for reference only).
* Comparison with Detector A on the same 33 seizures and 3.73 interictal hours.
