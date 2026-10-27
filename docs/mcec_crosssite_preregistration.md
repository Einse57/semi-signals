# Detector A, body-only variant: MCEC_Seizure cross-site check (preregistration)

Written 2026-10-27 07:40 PDT (committed b2e1dc5). At this time **no body-only model has been trained and no MCEC probability
has been computed or looked at.** The MCEC files were opened only to read their format
(keypoint layout, frame rate, folder labels, segment lengths).
Everything not listed here follows `docs/detector_a_preregistration.md`.
Changes after this point go under "Deviations" with a reason and a timestamp.

## External data: MCEC_Seizure
* Kuang, Wang, Sun, Zhao, Shi, Zhu (2024), Mendeley Data V1, doi:10.17632/pyxwfn94xm.1, **CC BY 4.0**.
  Preprocessing code: github.com/kuirh/GATCN_MAE. Hospital not named; China.
  The archive (`MEEC_Seizure.7z`, sha256 c4d6fac4…24f3) holds **skeleton CSVs only, no video**.
  It lives outside the repo, under the work directory at `external/mcec/`. No SWEC data is involved.
* Format: one CSV per ~2 s segment (`frame, landmark_id, x, y, z, visibility`), 50 frames each.
  33 MediaPipe-Pose landmarks; x and y are normalised to the (half-)frame width and height.
  Frame rate is **25 fps** (frame index = 25 × file time + const in the seizure folders).
  The GATCN_MAE code shows each camera picture is the left or right half of a split-screen frame, and
  that segments were saved only while MediaPipe detected a person (min_detection_confidence 0.65).
* Patients: 4563, 4581, 4600, 4613. Folder `<id>` / `<id>_<k>` = seizure segment; `<id>_norm` = non-seizure.
* **Seizures (5):**
  * 4563: ~2.7 min (3 frame-contiguous runs)
  * 4581_1: ~3.1 min
  * 4581_2: 28 s (starts 54 s after 4581_1 ends; counted as a separate seizure, as released)
  * 4600: 14 s
  * 4613: **2 s**
* **Annotation:** the dataset has **no explicit onset or offset fields.** The seizure folder spans the
  annotated seizure window intersected with frames in which MediaPipe found the person. So
  **"onset" = first frame of the seizure folder** and **"end" = last frame.** This onset may be later than
  the clinical onset if the person was not detected at first. Its EEG-vs-clinical type is unknown.
* **Non-seizure:** ~52.9 min in total, made of 147 frame-contiguous runs of 2–60 s (median about 12–43 s by
  patient), sampled from recordings of about 2–3.7 h.

## Model: "Detector A-body"
Identical to Detector A (same code path `src/detector_a/features.py`, same LightGBM parameters, seed and
`scale_pos_weight`; same per-second VSViG labels), except:
1. **Dropped features (detector- or hand-model-dependent):** `lhand_fine4`, `rhand_fine4`
   (RTMPose 21-point hands, absent in MCEC), `npers4` (MCEC's MediaPipe is single-person) and
   `has_subject4` (MCEC saved only frames where the person was detected, so it is constant by construction).
   The wrist-based `l_hand` and `r_hand` regions and `asym_hand` are kept, since they use COCO joints 9 and 10.
2. **Scale normaliser:** the subject box diagonal (YOLOX box in Detector A) is replaced by the diagonal of the
   bounding box of visible (score >= 0.3) COCO-17 joints. It is applied the same way on VSViG and on MCEC.
3. **Keypoints on VSViG:** the existing RTMPose COCO-17 body keypoints (`data/kp/*.npz`, 6 fps).

Training: all 14 VSViG patients (same records as Detector A, `pat04_no-Sz2P` and `pat05_Sz4PG` excluded).
Threshold: the same rule as Detector A's final model, applied to A-body on all 14 patients (inner LOPO,
lowest 0.05-grid value with FA/h <= 1, else 0.95). For Detector A this gave 0.95. Whatever the rule returns
is used, and it is frozen before MCEC is scored. Alarm rule unchanged: trailing 5 s mean, events = runs at or
above the threshold lasting >= 3 s, merged if the gap is < 30 s. **No tuning on MCEC.**

## MCEC to Detector A input mapping
* MediaPipe to COCO-17: nose 0→0, l_eye 2→1, r_eye 5→2, l_ear 7→3, r_ear 8→4, l_shoulder 11→5, r_shoulder 12→6,
  l_elbow 13→7, r_elbow 14→8, l_wrist 15→9, r_wrist 16→10, l_hip 23→11, r_hip 24→12, l_knee 25→13,
  r_knee 26→14, l_ankle 27→15, r_ankle 28→16. z is ignored. MediaPipe `visibility` is used as the joint score
  (same 0.3 threshold).
* Coordinates: **primary** uses x and y as released (normalised, 1:1). **Sensitivity** (reported, not used for
  any choice) scales x by 8/9 (half of a 16:9 frame) and by 2/3 (half of 4:3).
* Resampling 25 → 6 fps: for each target time k/6 s, take the nearest frame within 1/12 s.
  A target with no frame within that tolerance is "no subject" (missing).
* Records: each seizure folder is one record from its first to last frame; frame gaps inside it are missing.
  Each frame-contiguous non-seizure run is one record. Features are computed per record, so records do not
  share history, and the first seconds use partial windows, exactly as at the start of a VSViG video.
* Per-second labels: seizure record seconds in [onset, onset + 120 s) are ictal and later seconds are
  excluded (the Detector A rule; this only affects 4581_1). Non-seizure records are interictal.

## Metrics (MCEC)
* **Seizures detected (primary):** an alarm event overlapping [onset − 30 s, onset + 60 s]. With Wilson 95% CI.
  Because records start at the onset, pre-onset alarms are impossible, and an alarm needs >= 3 s above the
  threshold, so **4613 (2 s) is undetectable by design**. It stays in the denominator, and recall excluding
  it is also shown.
* **Secondary detection rule** (Detector A's scorer): overlap with [onset − 30, min(onset+120, end) + 60].
* **FA/h:** alarm events overlapping interictal seconds in the non-seizure runs, divided by interictal hours.
  Poisson exact 95% CI. Note that many runs are shorter than the warm-up and smoothing time.
* **Detection delay** from the MCEC onset (alarm start − onset), per seizure.
* **Per-second AUROC** of the raw probability, ictal vs interictal, pooled and per patient. 95% CI by
  record-level bootstrap (2000 resamples, seed 0, seizure and non-seizure records resampled separately).
  Approximate given n = 4 patients.
* Per-patient table. Secondary threshold 0.5 also reported.

## Reference: A-body on VSViG LOPO
To separate the loss from dropping hands and changing the normaliser from the site shift, A-body is also run
through the preregistered VSViG LOPO (nested threshold rule, same scorer). It is reported next to full
Detector A (29/33, 4.3 FA/h, mean AUROC 0.78), under both the Detector A scorer and the onset window
[onset − 30, onset + 60] (full Detector A is rescored from its saved events under the onset window too).

## Confounds stated up front
* Keypoint model mismatch: MediaPipe Pose (MCEC) vs RTMPose-m (training). They differ in noise, jitter,
  failure modes and score semantics (MediaPipe visibility vs RTMPose confidence).
* Unknown frame aspect ratio and resolution, and a split-screen half frame.
* Detection-gated segments: frames without a detected person were never saved. This biases toward visible
  motion and removes the "subject lost" signal.
* Non-contiguous, short non-seizure runs: FA/h is not continuous-monitoring FA/h.
* n = 4 patients and 5 seizures (one of them 2 s). Any result is anecdotal and CIs will be very wide.

## Deviations
* 2026-10-27 ~07:55 PDT (before any scoring): the "frame-contiguous runs" for non-seizure folders are chains of
  2-s files contiguous in both frame index and filename time. Non-seizure folders mix several source videos with
  overlapping frame indices, so merging by frame index alone wrongly deduplicated them. Found from record counts
  (17 runs, 697 s, instead of 147 runs, 3,174 s), not from any score.
* Results: `docs/mcec_crosssite_results.md`.
