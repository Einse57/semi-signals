"""Where the face landmarker looks: optional face preprocessing (--face-preprocess).

  none       (default) the frames are already the face region: the whole frame, padded to a square,
             goes to MediaPipe Face Landmarker (whose own BlazeFace detector finds the face in it).
  detect     find, crop and align the face: a full-frame landmarker pass (and, if that finds nothing,
             a pass over four overlapping tiles) locates the face; a square crop around it is rotated
             so the outer eye corners are level (roll removed), resized to --face-size and passed to
             the landmarker again. The next frame starts from the previous face (tracking) and falls
             back to a full search when the face is lost.
  coco-head  crop from the COCO-17 head keypoints of the body pose (src/face_estimator.head_crop_bbox);
             needs body keypoints, so only the body pipeline tools can use it.

All modes return landmarks in original-frame pixels (x, y; z in px on the x scale), so downstream code
does not depend on the mode. "detect" also returns the 2x3 affine frame -> aligned crop and the roll
angle that was removed.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

MODES = ("none", "detect", "coco-head")
DEFAULT_MODE = "none"
EYE_OUTER = (33, 263)        # subject's right / left outer eye corner (image left / right for a frontal face)


def full_square_box(h: int, w: int) -> np.ndarray:
    """Square box centred on the frame with side max(h, w); square_crop pads the short side."""
    side = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0
    return np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], np.float32)


def roll_deg(landmarks: np.ndarray) -> float:
    """Image-plane angle of the line from landmark 33 to 263 (degrees, y down: positive = clockwise)."""
    a, b = landmarks[EYE_OUTER[0], :2], landmarks[EYE_OUTER[1], :2]
    return math.degrees(math.atan2(float(b[1] - a[1]), float(b[0] - a[0])))


def align_affine(landmarks: np.ndarray, size: int, scale: float = 1.8) -> np.ndarray:
    """2x3 affine frame -> aligned square crop of side `size`: centred on the mesh, eye line level,
    face extent * scale fitted to the crop."""
    p = np.asarray(landmarks, np.float64)[:468, :2]
    cx, cy = p.mean(0)
    ext = float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])))
    ext = max(ext, 8.0) * scale
    A = cv2.getRotationMatrix2D((cx, cy), roll_deg(landmarks), size / ext)
    A[:, 2] += (size / 2.0 - cx, size / 2.0 - cy)
    return A


def map_back(lm_crop: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Landmarks in aligned-crop px -> frame px (z rescaled by the crop scale)."""
    Ai = cv2.invertAffineTransform(A)
    xy = lm_crop[:, :2].astype(np.float64) @ Ai[:, :2].T + Ai[:, 2]
    s = math.hypot(A[0, 0], A[0, 1])
    return np.concatenate([xy, lm_crop[:, 2:3].astype(np.float64) / s], 1).astype(np.float32)


def crop_box_of(A: np.ndarray, size: int) -> np.ndarray:
    """Axis-aligned frame-px box around the rotated crop (for face_boxes / blur fallbacks)."""
    Ai = cv2.invertAffineTransform(A)
    c = np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float64) @ Ai[:, :2].T + Ai[:, 2]
    return np.array([c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()], np.float32)


class FacePreprocessor:
    """Face landmarks for one subject per frame, with the chosen preprocessing (see module doc)."""

    def __init__(self, estimator, mode: str = DEFAULT_MODE, size: int = 256, crop_scale: float = 1.8):
        if mode not in MODES:
            raise ValueError(f"face preprocess must be one of {MODES}, not {mode!r}")
        self.est, self.mode, self.size, self.crop_scale = estimator, mode, int(size), float(crop_scale)
        self._track = None

    def reset(self):
        self._track = None

    def _search(self, frame):
        h, w = frame.shape[:2]
        r = self.est.estimate_crop(frame, full_square_box(h, w))
        if r is not None:
            return r
        side = 0.6 * max(h, w)              # four overlapping square tiles
        for cy in (side / 2, h - side / 2):
            for cx in (side / 2, w - side / 2):
                r = self.est.estimate_crop(frame, np.array([cx - side / 2, cy - side / 2,
                                                            cx + side / 2, cy + side / 2], np.float32))
                if r is not None:
                    return r
        return None

    def _aligned(self, frame, lm_hint):
        A = align_affine(lm_hint, self.size, self.crop_scale)
        crop = cv2.warpAffine(frame, A, (self.size, self.size), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
        r = self.est.detect_image(crop)
        if r is None:
            return None
        r["landmarks"] = map_back(r["landmarks"], A)
        r["bbox"] = crop_box_of(A, self.size)
        r["affine"] = A.astype(np.float32)
        r["roll_deg"] = np.float32(roll_deg(lm_hint))
        r["crop"] = crop
        return r

    def run(self, frame, body_kpts=None, body_scores=None, body_bbox=None) -> dict | None:
        """Face dict (landmarks (478, 3) frame px, blendshapes, score, bbox, transform; detect mode also
        affine, roll_deg, crop) or None."""
        if self.mode == "coco-head":
            if body_kpts is None or not np.isfinite(np.asarray(body_kpts, float)).any():
                return None
            return self.est.estimate(frame, body_kpts, body_scores, body_bbox=body_bbox)
        h, w = frame.shape[:2]
        if self.mode == "none":
            return self.est.estimate_crop(frame, full_square_box(h, w))
        r = self._aligned(frame, self._track) if self._track is not None else None
        if r is None:
            hint = self._search(frame)
            r = self._aligned(frame, hint["landmarks"]) if hint is not None else None
            if r is None and hint is not None:      # found but the aligned pass failed: keep the search result
                r = hint
        self._track = r["landmarks"] if r is not None else None
        return r
