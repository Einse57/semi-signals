"""Fill fallback and unmasked-face check for the fill export (src/segment_export.py).

1. Fallback: per frame, YuNet (OpenCV Zoo face_detection_yunet, MIT) runs on the input frame. Every YuNet
   face whose centre is not inside a landmark mask (subject or bystander hull) gets its box, padded by
   `pad` x box size on each side and clipped to the frame, filled the same way as the landmark hulls.
2. Check: YuNet runs again on every fill output frame. A frame with a detection at or above the score
   threshold is "unmasked". Any unmasked frame, or no YuNet model at all, means the fill export is not
   marked shareable; the sidecar lists the frame indices (UnmaskedFaceCheck).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_NAME = "face_detection_yunet_2023mar.onnx"
FETCH_HINT = "python scripts/fetch_models.py --face-check --skip-data --skip-vsvig"
DEFAULT_SCORE = 0.6                       # same as deid_check.YuNetDetector (used for the measurements)
DEFAULT_PAD = 0.2


def missing_message(path) -> str:
    return (f"YuNet model not found ({path}): fill runs on landmarks only, faces the landmarker misses are not "
            f"filled, and this export cannot be marked shareable. Fetch it with: {FETCH_HINT}")


def pad_box(row, shape, pad=DEFAULT_PAD):
    """YuNet row (x, y, w, h, ...) -> (x0, y0, x1, y1) padded by pad * w / pad * h each side, clipped."""
    h, w = shape[:2]
    x, y, bw, bh = (float(v) for v in row[:4])
    x0, y0 = int(np.floor(x - pad * bw)), int(np.floor(y - pad * bh))
    x1, y1 = int(np.ceil(x + bw * (1 + pad))), int(np.ceil(y + bh * (1 + pad)))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def uncovered(rows, landmark_mask):
    """YuNet rows whose box centre is outside the landmark mask (uint8, >0 = masked)."""
    h, w = landmark_mask.shape[:2]
    keep = []
    for r in rows:
        cx, cy = int(round(float(r[0]) + float(r[2]) / 2)), int(round(float(r[1]) + float(r[3]) / 2))
        inside = 0 <= cx < w and 0 <= cy < h and landmark_mask[cy, cx] > 0
        if not inside:
            keep.append(r)
    return keep


class FillGuard:
    """YuNet fallback boxes and the unmasked-face check. `detector` (anything with detect(frame) -> YuNet
    rows) replaces the model file, e.g. in tests."""

    def __init__(self, model_path=None, score=DEFAULT_SCORE, pad=DEFAULT_PAD, detector=None):
        self.score, self.pad, self.problem = float(score), float(pad), None
        self.det = detector
        if self.det is None:
            p = Path(model_path) if model_path else None
            if p is None or not p.is_file():
                self.problem = missing_message(p)
            else:
                from src.deid_check import YuNetDetector
                self.det = YuNetDetector(p, score=self.score)
        self.reset()

    @property
    def available(self) -> bool:
        return self.det is not None

    def reset(self):
        self.frames, self.unmasked, self.n_boxes, self.box_frames = 0, [], 0, 0

    def _faces(self, frame):
        return [r for r in self.det.detect(frame) if float(r[-1]) >= self.score]

    def boxes(self, frame, landmark_mask):
        """Padded boxes to fill for YuNet faces the landmark mask does not cover."""
        if not self.available:
            return []
        out = [b for b in (pad_box(r, frame.shape, self.pad) for r in uncovered(self._faces(frame), landmark_mask))
               if b is not None]
        self.n_boxes += len(out)
        self.box_frames += bool(out)
        return out

    def check(self, out_frame, index) -> bool:
        """True when `out_frame` (fill output) has no YuNet face; otherwise records `index` as unmasked."""
        if not self.available:
            return False
        self.frames += 1
        if self._faces(out_frame):
            self.unmasked.append(int(index))
            return False
        return True

    def result(self) -> dict:
        ran = self.available
        clean = ran and not self.unmasked
        return dict(Ran=ran, Detector="YuNet (OpenCV Zoo face_detection_yunet_2023mar)", ScoreThreshold=self.score,
                    FramesChecked=self.frames, UnmaskedFrames=list(self.unmasked), UnmaskedFrameCount=len(self.unmasked),
                    FallbackBoxesFilled=self.n_boxes, FramesWithFallbackBoxes=self.box_frames, BoxPad=self.pad,
                    Clean=clean,
                    Reason=None if clean else (self.problem if not ran else
                                               f"YuNet found a face in {len(self.unmasked)} fill output frame(s)"))

    def summary(self, name="") -> str:
        r = self.result()
        head = f"[fill check] {name}: " if name else "[fill check] "
        if not r["Ran"]:
            return head + "NOT SHAREABLE: unmasked-face check not run. " + (self.problem or "")
        s = (f"{r['FramesChecked']} frames checked, {r['FallbackBoxesFilled']} YuNet fallback box(es) filled in "
             f"{r['FramesWithFallbackBoxes']} frame(s); ")
        if r["Clean"]:
            return head + s + "no unmasked face: shareable"
        fr = r["UnmaskedFrames"]
        show = ", ".join(map(str, fr[:50])) + (f", ... ({len(fr)} total)" if len(fr) > 50 else "")
        return head + s + f"NOT SHAREABLE: YuNet found a face in {len(fr)} output frame(s): {show}"
