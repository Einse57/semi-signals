# Face Landmarker on OpenVINO (optional backend)

`src/face_ov` runs the three models of the MediaPipe Face Landmarker bundle (BlazeFace short-range
detector, Face Mesh V2, Blendshape V2) with OpenVINO instead of MediaPipe Tasks. The graph around the
models (detection decoding, NMS, crops, landmark projection, tracking, smoothing, facial transformation
matrix) is re-implemented in Python from the MediaPipe sources named in each file's header. Outputs have
the same layout as the MediaPipe backend: 478 landmarks, 52 blendshapes, optional 4x4 facial
transformation matrix, up to `num_faces` faces per frame.

MediaPipe Tasks stays the default backend (`--face-backend mediapipe`).

## Setup

```bash
pip install openvino                                    # already in requirements.txt
python scripts/fetch_models.py --face --skip-data       # models/face_landmarker.task, sha256-checked
```

The bundle is fetched by the user and checked against the pinned SHA-256
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`. On first use the `.tflite` files and
the geometry metadata are extracted to `~/.cache/face_ov/<sha256[:16]>/` (`FACE_OV_CACHE` overrides) and
read with the OpenVINO TFLite frontend; the conversion happens locally at load time. No model weights or
converted (IR) files are stored in this repository.

## Use

```bash
python -m src.segment_export ... --face-backend openvino --face-device CPU
python -m src.deid_check ... --face-backend openvino
python -m src.batch_extract ... --face --face-backend openvino
python -m src.run_seizure_demo --input clip.mp4 --face --face-backend openvino
```

`--face-device` takes `CPU` (default), `GPU`, `NPU` or `AUTO` and is passed to OpenVINO as the device
name. On CPU the models run with f32 precision. Only CPU results are given below; other devices are
selectable but no parity results are given for them here.

In Python:

```python
from src.face_estimator import FaceEstimator
est = FaceEstimator(backend="openvino", device="CPU", num_faces=8, blendshapes=False, transform=True)
faces = est.all_faces(frame_bgr)        # list of (478, 3) landmark arrays in frame pixels
```

`src.face_ov.FaceLandmarkerOV(task, device, video=..., num_faces=..., blendshapes=..., transform=...)`
gives `detect(rgb, timestamp_ms)` (all faces) and `process()` (first face). In video mode each face is
tracked from its own landmarks and the detector runs only while fewer than `num_faces` faces are tracked;
landmark smoothing applies to a single face only, as in MediaPipe. `blendshapes=False` skips the
blendshape model (scores are NaN).

## Parity with MediaPipe Tasks (CPU)

Both backends compared frame by frame on six clips (four GRID clips of 75 frames, 360x288; two Intel
sample videos, first 300 frames, 768x432, one with two faces), OpenVINO 2026.4.1 and MediaPipe 1.2.0 on
the CPU of a Linux VM (Intel Xeon, 8 vCPU), IMAGE and VIDEO modes, `num_faces` 1 and 8. Faces are
paired by landmark centre. `scripts/face_ov_parity.py` runs the single-face comparison.

| Measure | Result (range over clips and modes) | Target |
|---|---|---|
| Frames with the same faces found | 100% on every clip | >= 99.5% |
| Landmark error / inter-iris distance, mean | 0.0002-0.0011 | <= 0.005 |
| Landmark error / inter-iris distance, p95 | 0.0005-0.0023 | <= 0.01 |
| Blendshape Pearson r, mean (coefficients with std > 0.02) | 0.998-0.9999 | >= 0.98 |
| Blendshape Pearson r, lowest coefficient (std > 0.02) | 0.995 | >= 0.9 |
| Transformation matrix rotation difference, p95 (max) | 0.03-0.16 deg (0.36 deg) | <= 1 deg |
| Face fill with segment_export (fill + fill check) | same faces filled and the same unmasked-face frames (none) on every frame; filled area within 0.3% of the frame | |

Per-frame time of the landmarker call on the same VM (median, VIDEO mode, `num_faces` 1, blendshapes and
matrix on, clips with a face in every frame): MediaPipe 6.4-6.7 ms, OpenVINO CPU 2.9-4.0 ms. Shared VM;
times vary between runs.

## Licences and model card

| Component | Source | Licence | In this repository |
|---|---|---|---|
| Face Landmarker bundle (BlazeFace short-range, Face Mesh V2, Blendshape V2) | <https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker> | Apache-2.0 | not redistributed / fetched by user; converted locally at load time |
| MediaPipe graph and calculators re-implemented in `src/face_ov/geometry.py`, `landmarker.py`, `face_geometry.py` | <https://github.com/google-ai-edge/mediapipe> | Apache-2.0, modified (file headers, `LICENSES/Apache-2.0.txt`, NOTICE) | yes |
| OpenVINO | <https://github.com/openvinotoolkit/openvino> | Apache-2.0 | installed with pip |

Model cards:
[BlazeFace short range](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf),
[Face Mesh V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf),
[Blendshape V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf).
The cards state the models are not intended for human life-critical decisions; the other model card
notes in [FACE_LANDMARKS.md](FACE_LANDMARKS.md#model-card-and-licence-notes) apply to both backends.
