"""MediaPipe Face Landmarker models run with OpenVINO (CPU, GPU, NPU or AUTO).

Pipeline (mirrors mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_graph.cc, Apache-2.0):
  1. face detector (BlazeFace short range) on a 128x128 letterbox of the image, [-1, 1] RGB;
     SSD decode + weighted NMS; eye-keypoint rotation; rect x1.5.
  2. face landmarks (Face Mesh V2) on a 256x256 rotated crop of that rect, [0, 1] RGB; presence logit
     -> sigmoid -> threshold; landmarks projected back to the image.
  3. blendshapes (52) from the 146-landmark subset in pixels.
  VIDEO mode: the next-frame rect comes from the landmarks (tracking; the detector is skipped while a
  face is tracked) and the landmarks are smoothed with the One Euro filter before the blendshapes, as
  MediaPipe does for num_faces == 1.

No MediaPipe code or runtime is used; geometry.py holds the re-implemented calculators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry as G
from .bundle import unpack_task

DEVICES = ("CPU", "GPU", "NPU", "AUTO")


@dataclass
class FaceResult:
    landmarks: np.ndarray        # (478, 3) normalised x, y (image width/height), z (image width units)
    blendshapes: np.ndarray      # (52,)
    presence: float              # sigmoid of the landmark model's presence logit
    det_score: float             # detector score of the detection that started this track (NaN if tracked)
    roi: G.Rect                  # rect the landmarks were computed in
    tracked: bool                # True when the rect came from the previous frame's landmarks


class FaceLandmarkerOV:
    def __init__(self, task_path, device: str = "CPU", video: bool = True, smooth: bool | None = None,
                 min_detection_confidence: float = 0.5, min_presence_confidence: float = 0.5,
                 min_suppression_threshold: float = 0.5, expected_sha256: str | None = None,
                 precision: str | None = "auto", cache_dir: str | None = None, core=None):
        import openvino as ov
        device = device.upper()
        if device not in DEVICES and not device.startswith(("AUTO:", "HETERO:", "GPU.")):
            raise ValueError(f"unknown OpenVINO device {device!r}")
        self.device = device
        self.video = bool(video)
        self.smooth = self.video if smooth is None else bool(smooth)
        self.min_det, self.min_pres, self.nms_thr = min_detection_confidence, min_presence_confidence, min_suppression_threshold
        self.paths = unpack_task(task_path, expected_sha256)
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
        for key, fname, shape in (("det", "face_detector.tflite", [1, 128, 128, 3]),
                                  ("lmk", "face_landmarks_detector.tflite", [1, 256, 256, 3]),
                                  ("bs", "face_blendshapes.tflite", [1, 146, 2])):
            m = self.core.read_model(str(self.paths[fname]))
            m.reshape({m.inputs[0]: shape})          # static shapes (NPU needs them; harmless elsewhere)
            t0 = time.perf_counter()
            cm = self.core.compile_model(m, device, cfg)
            self.compile_ms[key] = (time.perf_counter() - t0) * 1e3
            self._models[key] = cm
        self._det_req = self._models["det"].create_infer_request()
        self._lmk_req = self._models["lmk"].create_infer_request()
        self._bs_req = self._models["bs"].create_infer_request()
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
        self._prev_roi = None
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
    def detect_roi(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        boxes, scores = self.run_detector(self.detector_input(rgb))
        s, b, k = G.decode_detections(boxes, scores, self.anchors, self.min_det)
        if s.size == 0:
            return None, float("nan")
        score, box, kps = G.weighted_nms(s, b, k, self.nms_thr, max_out=1)[0]
        box, kps = G.project_detection(box, kps, w, h)
        return G.detection_to_roi(box, kps, w, h), score

    def process(self, rgb: np.ndarray, timestamp_ms: int | None = None) -> FaceResult | None:
        """One RGB uint8 frame (H, W, 3). VIDEO mode needs increasing timestamp_ms."""
        h, w = rgb.shape[:2]
        if self._size != (w, h):
            self.reset()
            self._size = (w, h)
        tracked = self.video and self._prev_roi is not None
        det_score = float("nan")
        if tracked:
            roi = self._prev_roi
        else:
            roi, det_score = self.detect_roi(rgb)
            if roi is None:
                self._smoother.reset()
                return None
        raw, logit = self.run_landmarks(self.landmark_input(rgb, roi))
        presence = float(1.0 / (1.0 + np.exp(-logit)))
        if not presence > self.min_pres:
            self._prev_roi = None
            self._smoother.reset()
            return None
        lm = G.project_landmarks(raw, roi, w, h)
        if self.video:
            self._prev_roi = G.landmarks_to_roi(lm, w, h)
        if self.smooth and timestamp_ms is not None:
            lm = self._smoother.apply(lm, int(timestamp_ms) * 1000, w, h)
        bs = self.run_blendshapes(G.blendshape_input(lm, w, h))
        return FaceResult(lm, bs.astype(np.float32), presence, det_score, roi, tracked)
