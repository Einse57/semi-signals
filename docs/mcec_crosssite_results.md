# MCEC_Seizure cross-site check: Detector A-body (measurements)

Preregistration: `semi-signals/docs/mcec_crosssite_preregistration.md`, committed b2e1dc5
(timestamp fix aac2348) before any model was trained or any MCEC probability was computed.
Code: `semi-signals/src/detector_a/{body_only,mcec,mcec_crosssite}.py`. Run on 2026-10-27, about 07:45–08:00 PDT.
Data: MCEC_Seizure (CC BY 4.0), skeletons only.

## Frozen model
* Detector A-body was trained on all 14 VSViG patients (36 records), using 91 features
  (Detector A's 95 minus `lhand_fine4`, `rhand_fine4`, `npers4`, `has_subject4`).
  The subject scale comes from the box around the visible joints.
* The preregistered threshold rule gave **0.95**, the same as full Detector A.
* Alarm rule: 5 s trailing mean, at least 3 s above the threshold, events merged when the gap is under 30 s.
  Nothing was tuned on MCEC.

## Reference: A-body vs full Detector A on VSViG LOPO (33 seizures, 3.73 interictal h, nested threshold 0.95 in every fold)
| Model | Detection window | Detected | Wilson 95% | FA (FA/h, Poisson 95%) | Median delay vs clinical onset | Mean per-patient AUROC |
|---|---|---|---|---|---|---|
| Full Detector A | Detector A code: clin−30 … min(clin+120, end)+60 | 29/33 | 0.73–0.95 | 16 (4.3/h, 2.5–7.0) | 18 s | 0.78 |
| Full Detector A | onset: clin−30 … clin+60 | 26/33 | 0.62–0.89 | 16 (4.3/h) | 13 s | 0.78 |
| A-body | code window | 29/33 | 0.73–0.95 | 14 (3.75/h, 2.1–6.3) | 15 s | 0.75 |
| A-body | onset window | 26/33 | 0.62–0.89 | 14 (3.75/h) | 12 s | 0.75 |

**Dropping the hand and detector features costs almost nothing on VSViG:**
the same detections, slightly fewer false alarms, and an AUROC 0.03 lower.
So any loss on MCEC comes from the site shift, the keypoint model and the labels, not from removing hands.

## MCEC (5 seizures / 4 patients; 147 non-seizure runs, 3,174 s = 0.88 h)
Primary analysis: coordinates as released (1:1), frozen threshold 0.95, onset window [onset−30, onset+60].
The onset is the first frame of the seizure folder.

| | Result |
|---|---|
| Seizures detected | **2/5** (Wilson 95% 0.12–0.77). Excluding 4613, which is 2 s long and undetectable by design: 2/4 |
| False alarms | **10 in 0.88 h = 11.3/h** (Poisson 95% 5.4–20.9) |
| Delays from MCEC onset | 4563: 43 s; 4600: 11 s |
| Pooled per-second AUROC | **0.52** (record bootstrap 95% 0.35–0.74). Excluding 4613: 0.52 |
| Detector A code window | Same: 2/5, 10 FA |

Per patient (frozen 0.95):
| Patient | Seizures detected | Delay | FA / non-seizure h | FA/h | AUROC |
|---|---|---|---|---|---|
| 4563 | 1/1 | 43 s | 2 / 0.31 | 6.5 | 0.68 |
| 4581 | 0/2 (4581_1: max smoothed p 0.94; 4581_2: 0.50) | – | 2 / 0.31 | 6.6 | 0.42 |
| 4600 | 1/1 (14 s seizure) | 11 s | 5 / 0.08 | 61 | 0.70 |
| 4613 | 0/1 (2 s seizure; smoothed p 0.98 but too short for a 3 s event) | – | 1 / 0.19 | 5.4 | 0.98 (2 ictal seconds) |

Secondary threshold 0.5:
* Onset window: 2/5 detected (delays 2 s and 5 s), with 43 FA (48.8/h).
* Code window: 3/5, adding 4581_1 at 166 s.

Aspect-ratio sensitivity (preregistered):
* x × 8/9: 2/5, 10 FA, AUROC 0.52
* x × 2/3: 2/5, 9 FA, AUROC 0.52

The result does not depend on the aspect assumption.

## Exploratory (not preregistered, labelled as such)
* **Warm-up false alarms.** 5 of the 10 false alarms start in the first 10 s of a non-seizure run.
  This is the warm-up effect noted as a confound. Counting only alarms that start 10 s or more into a run
  gives 5 FA in about 0.57 h, roughly 9/h, which is still above VSViG's 3.75/h.
* **Feature medians.** Overall speed 4 s mean, in box diagonals per second:

  | | Non-seizure | Seizure |
  |---|---|---|
  | VSViG | 0.078 | 0.147 |
  | MCEC | 0.072 | 0.079 |

  Peak speed over 10 s:

  | | Non-seizure | Seizure |
  |---|---|---|
  | VSViG | 0.19 | 0.56 |
  | MCEC | 0.20 | 0.24 |

  MCEC's non-seizure movement level matches VSViG's, so the scale normalisation transfers.
  But MCEC's labelled seizure segments contain barely more movement than its non-seizure segments,
  while VSViG ictal seconds move about twice as much.

## Deviations
* **Non-seizure runs.** "Frame-contiguous" runs were implemented as chains of 2-s files that are contiguous in
  both frame index and filename time. A first version merged all frames by index and silently deduplicated
  frames from different source videos, giving 17 runs and 697 s. That was caught before scoring, using the
  loader's record counts only; the corrected loader gives 147 runs and 3,174 s, as in the preregistration.
* No other deviations.

## Read
With 4 patients and 5 seizures this is anecdotal, but it points one way.
* **Generalisation is poor:** per-second ranking is at chance (0.52), 2/5 seizures are detected, and FA/h is
  about 3× VSViG's.
* **The hands are not the cause:** A-body matches full A on VSViG.
* **Pixel scaling is not the cause either:** baseline motion lines up across the sites, and the aspect
  sensitivity changes nothing.
* **Likely causes:**
  * The MCEC seizure segments are, kinematically, mostly low-motion. Possible reasons are different
    semiology, annotation windows spanning quiet phases, or MediaPipe tracking dropouts (detection-gated saving).
  * MediaPipe and RTMPose produce different keypoint noise.

  These cannot be separated without video.
* **What would answer it:** a larger, video-level external set such as the IEEE DataPort 50-patient set,
  ideally with keypoints re-extracted by our own RTMPose pipeline.
