# Optional face stage: MediaPipe Face Landmarker

Off by default. With `--face` (viewer, live view, batch extraction) or the launcher's **Face (MediaPipe)**
box, semi-signals adds dense face landmarks to the body + hand pipeline. Body, hand, Detector A and
VSViG (Detector B) inputs are the same with face on or off: the face stage only adds outputs, and
its signals are not fed to either detector.

## Pipeline

1. Body pose (RTMPose-m, COCO-17) as before.
2. Face region, set by `--face-preprocess` (`src/face_preprocess.py`; same flag in the viewer, live view,
   batch extraction, annotation tool and segment export):
   - `none` (default): the frame is taken to be the face region already (for example a face camera or a
     clip cropped to the face). The landmarker runs on the whole frame, padded to a square.
   - `detect`: find, crop and align the face. The first frame (or the first after a loss) is searched on the
     whole frame and then on four overlapping tiles (0.6 x the longer side). The face is then cut out as a
     256 x 256 crop centred on the mesh, 1.8 x the face extent, rotated so that the outer eye corners
     (landmarks 33 and 263) are level. The landmarks are mapped back to frame pixels. Later frames reuse the
     previous frame's landmarks to place the crop. One face per frame; with several people this is the
     face found first, which is not necessarily the person the body tracker selected.
   - `coco-head`: head crop from the COCO head keypoints of the body pose (the method before this flag
     existed; use it for room video where the face is a small part of the frame),
     `src/face_estimator.head_crop_bbox`:
     - centre = mean of the visible head keypoints (nose, eyes, ears with score >= 0.3);
     - face width estimated from each visible keypoint pair (ear-ear = 1.0, eye-eye = 0.45, eye-ear
       = 0.35, nose-eye = 0.28, ...; the largest estimate wins);
     - fallback when fewer than two head keypoints are visible: 0.4 x shoulder width, else
       0.12 x the longer side of the person box;
     - square crop, side = 2 x face width (minimum 48 px), zero-padded past the frame edge.

   The default changed from the head crop to `none`. Room video (for example EMU cameras) should use
   `--face-preprocess coco-head` or `detect`. With `--face` off nothing changes.
3. MediaPipe Face Landmarker (IMAGE mode, one face per crop): 478 3D landmarks (468 mesh + 10 iris)
   and 52 blendshape scores; optionally the facial transformation matrix.

**Where it runs.** MediaPipe runs its bundled TFLite models on its own CPU runtime (TFLite +
XNNPACK). It does not use OpenVINO or ONNX Runtime, so with `--device gpu` the body and hand models
run on the Intel iGPU while the face stage runs on the CPU.

## Setup

```bash
pip install mediapipe                                  # optional dependency
python scripts/fetch_models.py --face --skip-data      # models/face_landmarker.task, sha256-checked
```

The model bundle is not in this repository. Source:
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
(SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`, pinned in `scripts/fetch_models.py`).

## Viewer and live view

- `python -m src.run_seizure_demo --input clip.mp4 --face` (`--no-face-mesh` draws contours only).
- `python -m src.run_live --face`.
- Key `m` switches face tracking on and off while running; the model loads the first time it is switched on.
  If MediaPipe or the model file is missing, the viewer keeps running and shows why face is unavailable.
- The current mode (`Mode: Body + Hands` / `Mode: Body + Hands + Face`) is shown on screen.
  In the seizure viewer the mode is in the legend under the video; with face on it is also drawn on the video.
  With face off the rendered video (`--save`) is unchanged.
- Face motion graphs (seizure viewer: under the body motion graphs; live view: under the region charts):

| Region | Landmarks |
|---|---|
| mouth | lips (outer + inner contour) |
| l_eye / r_eye | eyelids + iris |
| l_face / r_face | face oval, brow, eyelids, cheek and mouth corner of one side |

`l_` / `r_` are the subject's left and right (as in COCO-17 and MediaPipe's `*Left` / `*Right` blendshapes),
not the viewer's. A mirrored image swaps them.

Region motion = mean landmark displacement between consecutive frames after a 2D similarity alignment
(rotation, scale, translation fitted on all 478 points), divided by the face width (landmarks 234-454)
and the time step, in face widths per second. The alignment removes rigid head movement, so the graphs
show movement within the face.

## batch_extract output (`--face`, schema `semi-signals.face/1`)

Added to each `<video>.npz` only with `--face`; without it the file is unchanged.
F = sampled frames, P = 3 person slots (same order as `body_kpts`).

| Key | Shape | dtype | Content |
|---|---|---|---|
| `face_landmarks` | (F, P, 478, 3) | float32 | x, y in frame pixels; z in pixels on the x scale, relative to the face centre (MediaPipe convention, smaller = closer to the camera). NaN = no face |
| `face_blendshapes` | (F, P, 52) | float32 | blendshape scores 0..1, names in `meta["face"]["blendshape_names"]` (index 0 is `_neutral`). NaN = no face |
| `face_scores` | (F, P) | float32 | 1.0 = a face passed MediaPipe's detection and presence thresholds (0.5 each); NaN = no face or no person. The Tasks API reports no per-face confidence |
| `face_boxes` | (F, P, 4) | float32 | crop x1, y1, x2, y2 that was used (axis-aligned box around the rotated crop with `detect`); NaN = no crop (with `coco-head` also kept when no face was found in it) |
| `face_person` | (F,) | int16 | person slot that holds the face; -1 = no face. With `none` / `detect` (one face per frame) the slot is the person whose visible head keypoints are nearest to the face centre (0 when no person was found); with `coco-head` it is the tracked subject |
| `face_transform` | (F, P, 4, 4) | float32 | only with `--face-transform`: MediaPipe's facial transformation matrix (canonical face model to the detected face, in the crop's camera frame). NaN = no face |

`meta["face"]`: model name, URL, file name, runtime, schema, number of landmarks, blendshape names,
crop method, `preprocess` mode, whether the transform was saved, and `provisional: true` (the face keys
may still change). `src/points_export.py` uses `face_person` when present.

```bash
python -m src.batch_extract --out data/extract --face --face-transform data/pat10/Sz2P.mp4                 # frame = face
python -m src.batch_extract --out data/extract --face --face-preprocess coco-head data/pat10/Sz2P.mp4    # room video
```

## Measurements

`scripts/measure_face.py` times the per-frame pipeline (detection, body pose, hand pose, then face for the
selected person) with face off and on, on the same pre-decoded frames, alternating off/on/off/on, without
rendering. MediaPipe runs on its own TFLite/XNNPACK CPU runtime regardless of `--device`.

```
python scripts/measure_face.py --input clip.mp4 --device cpu --frames 150
python scripts/measure_face.py --input clip.mp4 --device gpu --frames 150   # OpenVINO (Intel iGPU)
```

Recorded 2026-10-28 (150 timed frames after 10 warm-up frames, mean of the two runs per setting):

| Host | Device (pose stack) | Clip | Face off FPS | Face on FPS | Face stage | Conditions |
|---|---|---|---|---|---|---|
| Linux VM, Intel Xeon, 8 vCPU | CPU, ONNX Runtime | 1280x720 test clip, face visible (160/160) | 7.6 | 7.9 | 10.8 ms | shared VM, load average 4 to 19 during the runs; run-to-run spread larger than the face cost |
| Linux VM, Intel Xeon, 8 vCPU | CPU, ONNX Runtime | WU-SAHZU-EMU pat10 Sz2P from 20 s, faces pixelated (0/160) | 17.8 | 17.5 | 2.8 ms | same as above; hands not detected in this segment |
| Core Ultra 9 285H | CPU / Arc 140T iGPU | | pending | pending | | to be recorded in a desktop-session scheduled task |

## Model card and licence notes

MediaPipe Face Landmarker (BlazeFace short-range detector, Face Mesh V2, Blendshape V2), Google,
Apache License 2.0. Points from the model cards that apply here:

- Intended use is AR / entertainment / selfie effects. The cards state the models are not intended for
  human life-critical decisions. In semi-signals the face outputs are research annotations, not a diagnosis.
- The landmarks and blendshapes do not provide facial recognition or identification; surveillance and
  identity recognition are out of scope.
- The models were built for selfie-style views: quality drops when the face is turned more than about 80
  degrees from the camera, rolled, less than half visible, or small/far away (BlazeFace short-range: faces
  within about 2 m; head yaw up to 90, roll and pitch up to 45 degrees). Patients in bed filmed from an
  EMU ceiling or wall camera are outside the training distribution. The head crop helps with distance, not
  with pose or occlusion (blankets, electrodes, hands, tubing).
- Fairness evaluations in the cards report results across regions, skin tones and genders (Face Mesh
  error 2.06-3.27% of the interocular distance across regions); blendshape training used lab captures
  and synthetic (GHUM) faces.
- Blendshape `*Left` / `*Right` refer to the subject's side; mirrored video swaps them.

Model cards: https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf,
https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf,
https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf

## Privacy

Dense face landmarks (and to a lesser extent blendshapes) describe face shape and may count as
biometric identifiers under HIPAA, GDPR and state biometric laws, even without any image. Treat
`face_*` arrays like the video they came from. A points-only export (no frames, crops or thumbnails;
optional canonical-pose normalisation and iris removal; random segment IDs) reduces re-identification
risk; it does not guarantee de-identification.

## Segment export with video (provisional)

`src/segment_export.py` writes, per annotated event, a raw clip (local only), de-identified clips (face
blur, landmarks on a blank canvas), landmarks and blendshapes in the `movement` layout, BIDS-style
`events.tsv` and `scans.tsv`, and a private mapping. File layout, naming and sidecars:
[SEGMENT_STORAGE.md](SEGMENT_STORAGE.md). The format is provisional.

## Points-only segment export

`src/points_export.py` (schema `semi-signals.points/1`) turns annotated events into segment files for a
face/body annotation dataset without any image content.

```
python -m src.batch_extract --out data/extract --face --face-transform clip.mp4
python -m src.points_export --npz data/extract/clip.npz --annotations clip.annotation.json \
       --out export/points --mapping private/points_mapping.csv --mode both --drop-iris --canonical-face
```

| Option | Effect |
|---|---|
| `--mode body/face/both` | which annotated events to export (default both) |
| `--pre-s`, `--post-s` | seconds before the clinical onset and after the clinical end (default 10, 10) |
| `--default-len-s` | event length when no end was marked (default 30) |
| `--drop-iris` | drop the 10 iris points (468-477) |
| `--canonical-face` | remove head pose and position with MediaPipe's facial transformation matrix (needs `--face-transform` at extraction) |
| `--keep-transform` | also store the 4x4 matrices (head pose and position) |
| `--no-face` | body and hands only |
| `--include-notes` | also export free-text notes (off by default: notes can contain names) |

One `<segment_id>.npz` per event (`segment_id` = 16 random hex characters), selected person only:

| Key | Shape | Content |
|---|---|---|
| `t` | (T,) | seconds relative to the marked clinical onset |
| `body_kpts`, `body_scores` | (T, 17, 2), (T, 17) | COCO-17, frame pixels; NaN when no person was selected |
| `hand_kpts`, `hand_scores` | (T, 2, 21, 2), (T, 2, 21) | left, right |
| `face_landmarks` | (T, 478 or 468, 3) | image pixels (as in batch_extract) or, with `--canonical-face`, pose-free coordinates in units of the outer-eye-corner distance (landmarks 33 to 263) |
| `face_blendshapes`, `face_scores` | (T, 52), (T,) | as in batch_extract |
| `face_transform` | (T, 4, 4) | only with `--keep-transform` |
| `meta` | JSON | schema, segment id, event (mode, relative clinical end and EEG onset, event class / seizure type / side / face term when set), sample rate, frame size, layouts, face options |

Not exported: frames, crops, thumbnails, video names, paths, dates, annotator, patient id, and notes
(unless `--include-notes`). The `.npz` zip entries carry the fixed 1980-01-01 time stamp; the file's own
modification time on disk is set by the file system. The private mapping CSV (`segment_id`, source `.npz`,
annotation file, event mode, onset in the source, export time) must be outside `--out`; keep it with the
source videos.

Canonical face: image coordinates (x right, y down, z away from the camera) are converted to MediaPipe's
canonical-model axes (x, -y, -z), centred on the mean of the 468 mesh points, rotated by R^T (R = the
rotation part of the facial transformation matrix) and divided by the outer-eye-corner distance. The axis
convention was checked against the canonical face model in the model bundle
(`geometry_pipeline_metadata_landmarks.binarypb`): on rolled (+-30 degrees) and perspective-warped copies of
MediaPipe's test image, the remaining rotation to the canonical model was 3-10 degrees after
normalisation and 30-54 degrees before.

Limits: rigid normalisation removes head pose and position, not face shape. Dense landmarks and their
motion can still be biometric identifiers, so a points-only export reduces re-identification risk and is
not guaranteed de-identification. Handle the segments under the same agreements as the source videos.
Not implemented: shape anonymisation (for example replacing per-subject shape with a mean face and keeping
only motion), and a dataset-level manifest.
