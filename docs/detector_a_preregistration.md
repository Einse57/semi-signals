# Detector A: pose-feature seizure detector, preregistered protocol

Written 2026-10-25 (Sun) 21:05 PT, **before** any features were computed or any model trained/scored.
Changes after this point are listed under "Deviations" with a reason and a timestamp.

## Data
- WU-SAHZU-EMU-Video (VSViG, Apache-2.0): 14 patients, 33 motor seizures (+ pat03/pat04 `free.mp4` seizure-free clips).
- Labels: `Label.xlsx`, EEG onset and clinical onset per video. No seizure offset is given.
- Keypoints: semi-signals `src/batch_extract.py` (RTMPose-m body7 COCO-17 + RTMPose-m hand5, YOLOX-tiny person detector, score thr 0.3), sampled at 6 fps, subject chosen by IoU tracking.

## Per-second labels
- **interictal (0):** t < EEG onset, and all of `free.mp4`.
- **ictal (1):** clinical onset <= t < clinical onset + 120 s (clipped to video end).
- **excluded (not trained on, not counted as false alarm time):** EEG onset <= t < clinical onset (EEG-only phase), and t >= clinical onset + 120 s (unknown ictal/post-ictal state, no offset annotation).

## Features (causal: each second uses only the trailing window up to that second)
For each region (head, l_arm, r_arm, l_hand, r_hand, l_leg, r_leg, overall; COCO-17 joints as in `src/config.py` BODY_REGIONS; hands use wrist joints):
- speed of joints normalised by subject box diagonal; over trailing 4 s and 10 s: mean, max, std;
- jerk (2nd difference of speed) mean abs, trailing 4 s;
- band power of normalised displacement in 0.5-1.5 Hz and 1.5-3.0 Hz (trailing 10 s; 6 fps limits the band to < 3 Hz);
- visibility: fraction of frames with region score >= 0.3, trailing 4 s.
Plus: L/R asymmetry of arm, hand and leg speed (|L-R|/(L+R)), fraction of frames with a tracked subject, number of persons detected (mean, trailing 4 s), hand keypoint motion (mean speed of 21 hand points per side, trailing 4 s).
Missing joints are NaN (LightGBM handles NaN natively).

## Model (fixed, no tuning on held-out patients)
- LightGBM binary, `num_leaves=15, n_estimators=300, learning_rate=0.05, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=n_neg/n_pos`, seed 0.
- **Leave-one-patient-out** over the 14 patients.
- Post-processing: probability smoothed with a trailing 5 s mean; events = runs above threshold lasting >= 3 s; events closer than 30 s merged.
- **Threshold:** chosen per outer fold using only training patients: inner LOPO out-of-fold predictions on the 13 training patients; threshold = lowest value on a 0.05 grid with inner FA/h <= 1.0; if none, 0.95.

## Scoring (SzCORE-style, event based)
- Reference event = [clinical onset, clinical onset + 120 s] clipped; tolerance 30 s before onset and 60 s after the event end.
- TP: reference event overlapped by a predicted event. Missed otherwise.
- FA: predicted event not overlapping any (tolerance-extended) reference event and lying in interictal time. Predicted events wholly inside excluded post-ictal time are ignored.
- FA/h denominator: interictal hours.
- Report: seizure recall (Wilson 95% CI, plus patient-bootstrap CI), FA/h (Poisson exact 95% CI, plus patient-bootstrap CI), detection delay vs clinical onset and vs EEG onset (median, IQR), per-patient table, per-second AUROC per fold.
- Secondary (reported, not used for choices): fixed threshold 0.5 results.

## Caveats stated up front
- 14 patients, 33 seizures; interictal time is short (~hours), so FA/h is very uncertain.
- Seizures in this dataset are all motor (selected for video-visible semiology); results do not generalise to non-motor seizures.
- Recordings come from one centre, one camera setup.

## Deviations
(none yet)
