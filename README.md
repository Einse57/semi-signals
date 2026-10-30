# semi-signals

Real-time skeleton-based motion analysis — live body + hand tracking with per-region L/R motion quantification, accelerated via OpenVINO on Intel iGPU.

Inspired by [VSViG (Xu et al., ECCV 2024)](https://arxiv.org/abs/2311.14775) but extends its input pipeline with 21-keypoint hand tracking and finger-level motion. See [JOINT_MAPPING.md](src/JOINT_MAPPING.md) for a detailed comparison.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements-demo.txt
pip uninstall -y opencv-contrib-python opencv-python-headless   # keep a single OpenCV package
pip install --force-reinstall --no-deps opencv-python
python -m src.run_live
```

On Windows, `start.cmd` does the same setup for you (see [Seizure detection](#seizure-detection-research)).
`requirements.txt` is the larger development set (install torch first, see its header).

### CLI Options

```bash
python -m src.run_live --cam 1                  # specific webcam
python -m src.run_live --input video.mp4        # video file
python -m src.run_live --list-cams              # probe available cameras
python -m src.run_live --no-hands               # body only (~40 FPS vs ~25-30)
python -m src.run_live --det-score 0.6          # detection confidence threshold
python -m src.run_live --det-interval 10        # detect every N frames
python -m src.run_live --face                   # optional MediaPipe face landmarks (see below)
```

**Runtime keys:** `q` quit · `h` toggle hands · `m` toggle face · `c` cycle camera

## Models

All ONNX checkpoints are auto-downloaded by [rtmlib](https://github.com/Tau-J/rtmlib) from [OpenMMLab](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose).

| Model | Task | Input | Output |
|-------|------|-------|--------|
| [YOLOX-tiny](https://arxiv.org/abs/2107.08430) | Person detection | 416×416 | Bounding boxes |
| [RTMPose-m](https://arxiv.org/abs/2303.07399) body | Body pose | 256×192 crop | 17 keypoints (COCO) |
| [RTMPose-m](https://arxiv.org/abs/2303.07399) hand | Hand pose | 256×256 crop | 21 keypoints × 2 hands |
| [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (optional, `--face`) | Face mesh + blendshapes | `--face-preprocess`: whole frame (default), detect + align, or head crop from the body keypoints | 478 3D landmarks, 52 blendshapes |

The face stage is off by default. It needs `pip install mediapipe` and `python scripts/fetch_models.py --face`
(the model file is downloaded and sha256-checked, not stored here). MediaPipe runs on its own CPU runtime
(TFLite + XNNPACK), also when the other models run on the iGPU. Output schema, face regions and model card
notes: [docs/FACE_LANDMARKS.md](docs/FACE_LANDMARKS.md). By default the face stage treats each frame as the face
region (`--face-preprocess none`); for room video use `--face-preprocess coco-head` (head crop from the body
pose) or `detect` (find, crop and align the face).

Optional OpenVINO face backend (`--face-backend openvino --face-device CPU|GPU|NPU|AUTO`; MediaPipe is the default): [docs/FACE_OPENVINO.md](docs/FACE_OPENVINO.md).

## Performance

Tested on Intel Core Ultra 7 165U (Meteor Lake) — Xe-LPG iGPU, AI Boost NPU 3720.

| Configuration | FPS |
|---------------|-----|
| CPU (ONNX Runtime) | ~6 |
| iGPU (OpenVINO) — body only | ~40 |
| iGPU (OpenVINO) — body + hands | ~25–30 |

OpenVINO targets the iGPU via a monkey-patch in `config.py` (rtmlib's built-in OpenVINO backend hardcodes CPU). The NPU compiles all models but is slower — YOLOX ~9× slower, RTMPose ~1.2× slower.

## Project Layout

```
semi-signals/
├── start.cmd, start.ps1     # Windows: set up .venv and open the launcher
├── requirements-demo.txt    # demo and launcher
├── requirements-dev.txt     # tests
├── requirements.txt         # development set (torch installed separately)
├── src/
│   ├── run_live.py          # live body + hand tracking (entry point)
│   ├── config.py            # model URLs, device config, OpenVINO GPU patch
│   ├── detector.py, pose_estimator.py, hand_estimator.py   # YOLOX-tiny, RTMPose-m body and hand
│   ├── motion.py, display.py                               # per-region motion, overlay and charts
│   ├── face_estimator.py, face_preprocess.py, face_ov/     # optional face landmarks (MediaPipe / OpenVINO)
│   ├── launcher.py, run_seizure_demo.py, review_video.py   # launcher, live viewer, annotation tool
│   ├── detector_a/, detector_b/, fusion.py                 # seizure detectors
│   ├── annotation.py, labels.py                            # ELAN .eaf / JSON annotations, label sets
│   ├── segment_export.py, segment_store.py, fill_guard.py  # segment export and the fill check
│   ├── points_export.py                                    # points-only export
│   ├── deid_*.py                                           # de-identification methods, UI and checks
│   └── JOINT_MAPPING.md     # VSViG joint mapping comparison
├── scripts/fetch_models.py  # fetch and verify models and sample data
├── labels/                  # body.json, face.json label sets
├── docs/                    # setup, formats, results
├── third_party/vsvig/       # vendored VSViG code (Apache-2.0)
├── tests/                   # pytest (see Tests)
└── README.md
```

## Tests

```bash
pip install -r requirements-demo.txt -r requirements-dev.txt
pytest
```

`pytest.ini` collects `tests/` only. Tests that need local files are skipped without them:
- face models: `python scripts/fetch_models.py --face --face-check --skip-data --skip-vsvig`
- MediaPipe's sample image (five tests): download
  [business-person.png](https://storage.googleapis.com/mediapipe-assets/business-person.png) to
  `data/facetest/business-person.png`, or set `FACE_TEST_IMAGE` to its path
- OpenVINO face parity on a video: set `FACE_OV_CLIP` to a short clip with one face

## Seizure detection (research)

> **Research use only. Not a medical device and not a diagnostic.** Trained on one hospital's data and
> **not yet validated across sites.**

semi-signals also includes a video-only seizure detector built on the semi-signals pose pipeline, plus an explainable
live viewer and an annotation tool:

- **Detector A**: causal per-second pose features (region speeds, 0.5-1.5 Hz and 1.5-3 Hz rhythm, visibility,
  people in view) and a LightGBM classifier; alarm when the 5 s mean stays at or above 0.95 for 3 s.
  Trained and evaluated on the [VSViG](https://github.com/xuyankun/VSViG) WU-SAHZU EMU videos
  (Apache-2.0; Xu, Sawan et al., ECCV 2024), 14 patients, 35 recordings.
- **Detector B (reference)**: the released VSViG-base model (vendored code in `third_party/vsvig`, Apache-2.0).
- **Live viewer**: probability strip with alarm shading, top drivers from LightGBM `pred_contrib`, and per-region
  motion graphs; review your own video and save ELAN-compatible `.eaf` annotations.
- **Measurements** (each preregistered before scoring, see `docs/*_preregistration.md`):
  leave-one-patient-out on VSViG, 29/33 seizures detected with the scoring code's window (clinical onset - 30 s
  to min(onset + 120 s, recording end) + 60 s; 26/33 with an onset-only -30/+60 s window) at 4.3 false alarms/h.
  On a second site (MCEC_Seizure, 4 patients, skeletons only) a frozen body-only model caught 2/5 at 11.3/h,
  with chance-level per-second ranking: [docs/mcec_crosssite_results.md](docs/mcec_crosssite_results.md).

### Quick start: double-click start.cmd (or run .\start.ps1)
On Windows, double-click `start.cmd` in the repo folder. It sets up `.venv` on first use (it asks first) and
opens the launcher window: **Fetch models & sample data**, pick a video, then **Play with live scores** or
**Annotate this video** (progress, review window and control panel; every change is saved immediately).
The viewer stays open at the end of the video (`R` replay, `O` open another video, `Q` quit) or loops with
**Loop playback** / `--loop`. Keys are shown in a footer strip in each window (`h` hides it):
viewer `q` quit, `space` pause, `f` fast, `r` replay, `o` open; annotation `o` onset, `e` end, `t` type,
`n` note, `x` exclude, `u` undo, `j`/`l` -5/+5 s, `s` save now. On Linux/macOS: `python -m src.launcher`.
Details: [docs/SEIZURE_DETECTION.md](docs/SEIZURE_DETECTION.md#quick-start-double-click-startcmd-or-run-startps1).

From a terminal:
```bash
pip install -r requirements-demo.txt
python scripts/fetch_models.py            # pose models, VSViG-base -> OpenVINO IR, Detector A, Label.xlsx + 2 sample clips (sha256-checked)
python -m src.run_seizure_demo --input data/pat10/Sz2P.mp4 --labels data/Label.xlsx [--loop]
python -m src.launcher                    # the launcher window (what start.cmd opens)
python -m src.review_video                # load your own video and annotate it
python -m src.review_video --mode face    # annotate face events (face mesh render, face label set)
```

### Annotation modes and label sets
The annotation tool has three modes (launcher **Annotation mode**, or `--mode`): **Body/Hands** (default),
**Face** (the render adds the MediaPipe face mesh; see [docs/FACE_LANDMARKS.md](docs/FACE_LANDMARKS.md)) and
**Both** (`o` body onset, `f` face onset). Face events use the same timing fields as body events (clinical
onset, clinical end, optional EEG onset), an optional note and an optional side (`d`: L / R / both). Each
event stores its `mode`; face events go to their own `Face_event` EAF tier and never into `Seizure`, and
Detector A is compared with body events only. Files written before modes existed load as body events.

The fields come from `labels/body.json` and `labels/face.json` (schema `semi-signals.labels/1`). The face
set starts without a term list, so `t` does nothing for face events. To add a vetted list later (for
example from the ILAE seizure semiology glossary or a clinical expert), with no code change:

1. Edit `labels/face.json`: fill `term.choices` with the terms and `term.source` with where they come from
   (keep `term.field` = `face_term` and `term.tier` = `Face_term`).
2. Restart the annotation tool. `t` on a face event then opens the list; terms are saved in the
   `Face_term` EAF tier (a child of `Face_event`) and in `face_term` in the JSON.

`labels/<mode>.yaml` is read instead of the `.json` file when it exists and PyYAML is installed;
`SEMI_SIGNALS_LABELS=<folder>` points to another folder of label files. The timing fields are fixed and
shared by both modes.

### Points-only segment export
`python -m src.points_export --npz <extract>.npz --annotations <stem>.annotation.json --out <folder>
--mapping <private.csv> [--mode face] [--drop-iris] [--canonical-face]` writes one file per annotated event
with keypoints, scores, blendshapes and time relative to the clinical onset: no frames, crops, names, paths
or dates, random segment ids, and a separate private mapping file. Dense face landmarks may still be
biometric identifiers, so this reduces re-identification risk; it is not de-identification. Details:
[docs/FACE_LANDMARKS.md](docs/FACE_LANDMARKS.md#points-only-segment-export).

### Segment export with video (provisional format)
`python -m src.segment_export --input clip.mp4 --annotations clip.eaf --out <export> --raw-root <local raw folder>
[--face-preprocess detect]` writes per annotated event a raw clip (under `--raw-root` only), a face-filled
clip, a landmarks-only clip, landmarks and blendshapes (netCDF, `movement` layout), BIDS-style `events.tsv` /
`scans.tsv` and a private mapping CSV. Names follow BEP047, a proposal, not part of BIDS. Provisional format:
[docs/SEGMENT_STORAGE.md](docs/SEGMENT_STORAGE.md).

Face methods ([docs/FACE_DEID_TRADEOFF.md](docs/FACE_DEID_TRADEOFF.md)):

- **Fill** (shareable default): faces painted one flat colour, with a YuNet box fallback for faces the
  landmarker misses. YuNet then checks every output frame, fills any face it finds and checks again, then
  checks the written video once more; a face still found marks the export not shareable and lists the frames.
- **GHOST** face swap: reduced risk, not de-identified; research use only, behind an acknowledgement.
- **Blur** (`--deid blur`): local review only. Mosaic16 is not de-identifying.

The UI selector (launcher Face or Both mode, and "De-ID export..." in the annotation panel) offers Original,
Fill and GHOST with a before/after preview. Next to it, the Landmarker picker lets you run the face landmarks on
MediaPipe (CPU, the default) or OpenVINO on any device OpenVINO finds (CPU, GPU, NPU, AUTO); if a device
doesn't load, it drops back to CPU with a note, and the export sidecar records the device it actually used.
Third-party code, weights and donor images are not redistributed;
you fetch them yourself: [docs/FACE_SWAP_LICENCES.md](docs/FACE_SWAP_LICENCES.md).
`python scripts/fetch_models.py --ghost --accept-ghost-licences --skip-data --skip-vsvig` fetches the GHOST
code (Apache-2.0) and four files from the GHOST GitHub releases, sha256-checked, into the git-ignored
`third_party_local/`: the generator (Apache-2.0), InsightFace `backbone.pth` and `glintr100.onnx`
(non-commercial research only) and `iresnet.py` (MIT). Without `--accept-ghost-licences` it asks first and
fetches nothing unless you answer `y`. Donor images are not fetched.

Model weights are not in this repository. Details: [docs/SEIZURE_DETECTION.md](docs/SEIZURE_DETECTION.md),
[docs/detector_a_results.md](docs/detector_a_results.md),
[docs/mcec_crosssite_results.md](docs/mcec_crosssite_results.md). Third-party attribution: [NOTICE](NOTICE).

## References

- Xu et al., *"VSViG: Real-time Video-based Seizure Detection via Skeleton-based Spatiotemporal ViG"*, ECCV 2024. [arXiv:2311.14775](https://arxiv.org/abs/2311.14775)
- Jiang et al., *"RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose"*, 2023. [arXiv:2303.07399](https://arxiv.org/abs/2303.07399)
- Ge et al., *"YOLOX: Exceeding YOLO Series in 2021"*. [arXiv:2107.08430](https://arxiv.org/abs/2107.08430)
- Google, MediaPipe Face Landmarker (BlazeFace, Face Mesh V2, Blendshape V2), Apache-2.0. [ai.google.dev/edge/mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker); Kartynnik et al., *"Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs"*, 2019. [arXiv:1907.06724](https://arxiv.org/abs/1907.06724)

**What this project can do today:**
- Rule-based alerts on motion thresholds (e.g., sustained bilateral tremor)
- L/R asymmetry detection as an indicator
- Motion time-series recording for offline analysis
- Clip export on motion spikes for review
