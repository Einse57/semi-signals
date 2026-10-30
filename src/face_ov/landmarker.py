# Derived from MediaPipe (https://github.com/google-ai-edge/mediapipe), Apache-2.0, modified:
#   mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_graph.cc,
#   face_landmarks_detector_graph.cc, face_blendshapes_graph.cc,
#   mediapipe/tasks/cc/vision/face_detector/face_detector_graph.cc,
#   mediapipe/calculators/util/association_norm_rect_calculator.cc
# Copyright The MediaPipe Authors. Re-implemented in Python; see LICENSES/Apache-2.0.txt and NOTICE.
"""MediaPipe Face Landmarker models run with OpenVINO (CPU, GPU, NPU or AUTO).

Pipeline (mirrors mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_graph.cc, Apache-2.0):
  1. face detector (BlazeFace short range) on a 128x128 letterbox of the image, [-1, 1] RGB;
     SSD decode + weighted NMS; eye-keypoint rotation; rect x1.5.
  2. face landmarks (Face Mesh V2) on a 256x256 rotated crop of that rect, [0, 1] RGB; presence logit
     -> sigmoid -> threshold; landmarks projected back to the image.
  3. blendshapes (52) from the 146-landmark subset in pixels.
  4. optional facial transformation matrix (face_geometry.py).
  Up to num_faces faces per frame. VIDEO mode: the next-frame rects come from the landmarks (tracking);
  the detector runs only while fewer than num_faces faces are tracked, and its rects replace tracked
  rects they overlap (IoU > min_tracking_confidence; association_norm_rect_calculator.cc). The
  landmarks are smoothed with the One Euro filter before the blendshapes only when num_faces == 1, as
  MediaPipe does.

No MediaPipe code or runtime is used; geometry.py holds the re-implemented calculators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry as G
from .bundle import GEOMETRY_FILE, unpack_task
from .face_geometry import FaceGeometry

DEVICES = ("CPU", "GPU", "NPU", "AUTO")


@dataclass
class FaceResult:
    landmarks: np.ndarray        # (478, 3) normalised x, y (image width/height), z (image width units)
    blendshapes: np.ndarray      # (52,)
    presence: float              # sigmoid of the landmark model's presence logit
    det_score: float             # detector score of the detection that started this track (NaN if tracked)
    roi: G.Rect                  # rect the landmarks were computed in
    tracked: bool                # True when the rect came from the previous frame's landmarks
    transform: np.ndarray | None = None   # (4, 4) facial transformation matrix when requested


class FaceLandmarkerOV:
    def __init__(self, task_path, device: str = "CPU", video: bool = True, smooth: bool | None = None,
                 min_detection_confidence: float = 0.5, min_presence_confidence: float = 0.5,
                 min_suppression_threshold: float = 0.5, expected_sha256: str | None = None,
                 precision: str | None = "auto", cache_dir: str | None = None, core=None,
                 num_faces: int = 1, blendshapes: bool = True, transform: bool = False,
                 min_tracking_confidence: float = 0.5):
        import openvino as ov
        device = device.upper()
        if device not in DEVICES and not device.startswith(("AUTO:", "HETERO:", "GPU.")):
            raise ValueError(f"unknown OpenVINO device {device!r}")
        self.device = device
        self.video = bool(video)
        self.num_faces = max(1, int(num_faces))
        if smooth and self.num_faces > 1:
            raise ValueError("landmark smoothing supports a single face (num_faces == 1), as in MediaPipe")
        self.smooth = (self.video and self.num_faces == 1) if smooth is None else bool(smooth)
        self.blendshapes, self.min_track = bool(blendshapes), float(min_tracking_confidence)
        self.min_det, self.min_pres, self.nms_thr = min_detection_confidence, min_presence_confidence, min_suppression_threshold
        self.paths = unpack_task(task_path, expected_sha256)
        self.geometry = FaceGeometry(self.paths[GEOMETRY_FILE]) if transform else None
        self.core = core or ov.Core()
        if cache_dir:
            self.core.set_property({"CACHE_DIR": str(cache_dir)})
        cfg = {}
        # "auto": f32 on CPU (the CPU plugin otherwise picks bf16 on CPUs that have it, which moves the
        # blendshapes by ~0.01-0.02), the device default elsewhere (f16 on GPU and NPU).
        if precision == "auto":
            precision = "f32" if device == "CPU" else None
        self.precision = precision
        if precision:
            cfg["INFERENCE_PRECISION_HINT"] = precision
        self.compile_ms = {}
        import time
        self._models = {}
        specs = [("det", "face_detector.tflite", [1, 128, 128, 3]),
                 ("lmk", "face_landmarks_detector.tflite", [1, 256, 256, 3])]
        if self.blendshapes:
            specs.append(("bs", "face_blendshapes.tflite", [1, 146, 2]))
        for key, fname, shape in specs:
            m = self.core.read_model(str(self.paths[fname]))
            m.reshape({m.inputs[0]: shape})          # static shapes (NPU needs them; harmless elsewhere)
            t0 = time.perf_counter()
            cm = self.core.compile_model(m, device, cfg)
            self.compile_ms[key] = (time.perf_counter() - t0) * 1e3
            self._models[key] = cm
        self._det_req = self._models["det"].create_infer_request()
        self._lmk_req = self._models["lmk"].create_infer_request()
        self._bs_req = self._models["bs"].create_infer_request() if self.blendshapes else None
        det = self._models["det"]
        self._det_boxes = next(o for o in det.outputs if o.get_partial_shape()[-1].get_length() == 16)
        self._det_scores = next(o for o in det.outputs if o.get_partial_shape()[-1].get_length() == 1)
        lmk = self._models["lmk"]
        self._lmk_out = next(o for o in lmk.outputs if o.get_partial_shape()[-1].get_length() == 1434)
        self._pres_out = next(o for o in lmk.outputs if o.get_partial_shape().rank.get_length() == 4
                              and o.get_partial_shape()[-1].get_length() == 1)
        self.anchors = G.ssd_anchors(**G.SHORT_RANGE_ANCHORS)
        self.reset()

    # ---- state -----------------------------------------------------------------------------------
    def reset(self):
        self._prev_rois: list[G.Rect] = []
        self._smoother = G.OneEuroLandmarks(**G.ONE_EURO)
        self._size = None

    def close(self):
        self._models.clear()

    # ---- raw stages (also used by the raw-tensor parity check) -----------------------------------
    def detector_input(self, rgb: np.ndarray) -> np.ndarray:
        import cv2
        h, w = rgb.shape[:2]
        side, _, _ = G.letterbox_params(w, h)
        r = G.Rect(0.5, 0.5, side / w, side / h, 0.0)
        M = G.roi_warp_matrix(r, w, h, 128, 128)
        t = cv2.warpPerspective(rgb, M, (128, 128), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return (t.astype(np.float32) / 127.5 - 1.0)[None]

    def landmark_input(self, rgb: np.ndarray, roi: G.Rect) -> np.ndarray:
        import cv2
        h, w = rgb.shape[:2]
        M = G.roi_warp_matrix(roi, w, h, 256, 256)
        t = cv2.warpPerspective(rgb, M, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return (t.astype(np.float32) / 255.0)[None]

    def run_detector(self, x: np.ndarray):
        r = self._det_req.infer({0: x})
        return np.asarray(r[self._det_boxes]), np.asarray(r[self._det_scores])

    def run_landmarks(self, x: np.ndarray):
        r = self._lmk_req.infer({0: x})
        return np.asarray(r[self._lmk_out]).reshape(478, 3), float(np.asarray(r[self._pres_out]).reshape(-1)[0])

    def run_blendshapes(self, x: np.ndarray) -> np.ndarray:
        r = self._bs_req.infer({0: x})
        return np.asarray(r[self._models["bs"].outputs[0]]).reshape(-1)

    # ---- pipeline --------------------------------------------------------------------------------
    def detect_rois(self, rgb: np.ndarray, max_faces: int | None = None):
        """Detector -> up to max_faces (score, rect), highest score first."""
        h, w = rgb.shape[:2]
        boxes, scores = self.run_detector(self.detector_input(rgb))
        s, b, k = G.decode_detections(boxes, scores, self.anchors, self.min_det)
        if s.size == 0:
            return []
        out = []
        for score, box, kps in G.weighted_nms(s, b, k, self.nms_thr, max_out=max_faces or self.num_faces):
            box, kps = G.project_detection(box, kps, w, h)
            out.append((score, G.detection_to_roi(box, kps, w, h)))
        return out

    def detect_roi(self, rgb: np.ndarray):
        r = self.detect_rois(rgb, 1)
        return (r[0][1], r[0][0]) if r else (None, float("nan"))

    def detect(self, rgb: np.ndarray, timestamp_ms: int | None = None) -> list[FaceResult]:
        """Every face (up to num_faces) in one RGB uint8 frame (H, W, 3). VIDEO mode needs increasing
        timestamp_ms for smoothing."""
        h, w = rgb.shape[:2]
        if self._size != (w, h):
            self.reset()
            self._size = (w, h)
        # (rect, det_score, tracked) in association order: tracked rects first, detections after.
        cands = [(r, float("nan"), True) for r in self._prev_rois] if self.video else []
        if len(cands) < self.num_faces:
            for score, roi in self.detect_rois(rgb):
                cands = [c for c in cands if not G.rect_iou(c[0], roi) > self.min_track]
                cands.append((roi, score, False))
        cands = cands[:self.num_faces]
        faces, next_rois = [], []
        for roi, det_score, tracked in cands:
            raw, logit = self.run_landmarks(self.landmark_input(rgb, roi))
            presence = float(1.0 / (1.0 + np.exp(-logit)))
            if not presence > self.min_pres:
                continue
            lm = G.project_landmarks(raw, roi, w, h)
            next_rois.append(G.landmarks_to_roi(lm, w, h))
            if self.smooth and timestamp_ms is not None:
                lm = self._smoother.apply(lm, int(timestamp_ms) * 1000, w, h)
            bs = (self.run_blendshapes(G.blendshape_input(lm, w, h)).astype(np.float32) if self.blendshapes
                  else np.full(52, np.nan, np.float32))
            tf = self.geometry.transform(lm, w, h) if self.geometry is not None else None
            faces.append(FaceResult(lm, bs, presence, det_score, roi, tracked, tf))
        self._prev_rois = next_rois if self.video else []
        if not faces:
            self._smoother.reset()
        return faces

    def process(self, rgb: np.ndarray, timestamp_ms: int | None = None) -> FaceResult | None:
        """First face of detect() (single-face convenience)."""
        faces = self.detect(rgb, timestamp_ms)
        return faces[0] if faces else None
