# Detector A (keypoint motion features + LightGBM): leave-one-patient-out results

Run: 2026-10-25 ~22:20 PT, after the preregistration in `docs/detector_a_preregistration.md` (20:53 PT).
Data: WU-SAHZU EMU video set (14 patients, 36 videos; 33 labelled seizures, 3.73 h of interictal video).
Keypoints: `src/batch_extract.py` (RTMPose via rtmlib, 6 samples/s). Code: `python -m src.detector_a.lopo`.

## Headline (preregistered primary rule: nested inner-LOPO threshold)

| measure | value |
|---|---|
| seizures detected | 29/33 = 0.88 (Wilson 95% 0.73-0.95; patient bootstrap 0.77-0.97) |
| false alarms | 16 in 3.73 interictal h = 4.3/h (Poisson 95% 2.5-7.0; patient bootstrap 1.5-6.8) |
| delay vs clinical onset | median 18 s (IQR 5 to 34) |
| delay vs EEG onset | median 28 s (IQR 13 to 50) |
| per-second AUROC, held-out patient | mean 0.78 (range 0.69-0.92) |

Secondary rule (fixed threshold 0.5): 33/33 detected (Wilson 0.90-1.00), but 16.4 false alarms/h
(Poisson 12.5-21.0); median delay 1 s vs clinical onset.

Per-patient tables: `results/detector_a/RESULTS.md` (generated, kept out of the repo with the data).

## How to read this

* In every outer fold, no threshold on the 0.05 grid reached the inner-fold target of 1 false alarm/h,
  so the preregistered fallback (0.95) applied everywhere. The final all-patient model also uses 0.95.
  The 1/h target is not met on held-out patients (4.3/h).
* Interictal time is small (3.73 h) and concentrated in three patients (pat01 1.17 h, pat02 0.92 h,
  pat04 0.50 h); 11 of 16 false alarms come from pat01 and pat02. The FA/h estimate is therefore wide,
  and most patients contribute almost no interictal time.
* Many videos start shortly before onset, so the 30 s pre-onset tolerance makes early detections easy to
  count. Recall on this set is optimistic compared with continuous monitoring.
* Missed seizures (nested rule): pat04 Sz1, pat08 (1 of 3), pat10 (1 of 2), pat12 (1 of 4).
* Video-only signal. No EEG/iEEG is used by Detector A.

## Deviations from the preregistration

1. Band powers use 0.5-1.5 Hz and 1.5-3 Hz instead of 3-8 Hz (a 3-8 Hz band is above Nyquist at 6 samples/s).
   This was recorded before scoring.
2. `pat05_Sz4PG` was excluded because Label.xlsx has no row for it (unlabelled video, as with
   `pat04_no-Sz2P`). This was found when the scorer first ran, before any results were seen.

No parameters were changed after scoring.

## Artifacts

* `detector_a_all.txt` / `detector_a_all.json`: LightGBM model trained on all 14 patients plus threshold 0.95
  and the feature list (used by `src/run_seizure_demo.py --detector-a`). Kept outside the repo.
