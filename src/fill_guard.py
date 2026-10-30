"""Fill fallback and unmasked-face check for the fill export (src/segment_export.py).

1. Fallback: per frame, YuNet (OpenCV Zoo face_detection_yunet, MIT) runs on the input frame. Every YuNet
   face whose centre is not inside a landmark mask (subject or bystander hull) gets its box, padded by
   `pad` x box size on each side and clipped to the frame, filled the same way as the landmark hulls.
2. Check: YuNet runs again on every fill output frame. A frame with a detection at or above the score
   threshold is "unmasked". Any unmasked frame, or no YuNet model at all, means the fill export is not
   marked shareable; the sidecar lists the frame indices (UnmaskedFaceCheck).
3. Second pass (check_fix): a flagged frame gets every flagged YuNet box filled as well (same padding,
   whether or not a landmark hull covers it) and is checked once more. Only frames still flagged after that
   count as unmasked.
4. Encoded file (check_encoded): after the fill video is written, it is decoded again and every frame is
   checked with the same detector and threshold, so MP4 compression cannot bring back a detectable face.
   The export is shareable only when both checks pass (UnmaskedFaceCheck.EncodedOutputCheck).
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
        self.first_pass, self.n_second, self.second_frames = [], 0, 0
        self.encoded = None

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

    def check_fix(self, out_frame, index, refill):
        """Check with one second pass. `refill(boxes)` re-renders the fill frame with these extra boxes.
        Returns (frame to write, clean)."""
        if not self.available:
            return out_frame, False
        self.frames += 1
        rows = self._faces(out_frame)
        if not rows:
            return out_frame, True
        self.first_pass.append(int(index))
        bx = [b for b in (pad_box(r, out_frame.shape, self.pad) for r in rows) if b is not None]
        if bx:
            self.n_second += len(bx)
            self.second_frames += 1
            out_frame = refill(bx)
            if not self._faces(out_frame):
                return out_frame, True
        self.unmasked.append(int(index))
        return out_frame, False

    def check_encoded(self, path, expected=None) -> dict:
        """Decode the written fill video at `path` and check every frame again (same detector and threshold).
        `expected`: number of frames written; a shorter read-back fails the check."""
        if not self.available:
            self.encoded = dict(Ran=False, FramesChecked=0, ExpectedFrames=expected, UnmaskedFrames=[],
                                UnmaskedFrameCount=0, Clean=False, Reason=self.problem)
            return self.encoded
        import cv2
        cap, n, bad = cv2.VideoCapture(str(path)), 0, []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if self._faces(frame):
                    bad.append(n)
                n += 1
        finally:
            cap.release()
        if n == 0:
            reason = f"could not read the written fill video back ({Path(path).name})"
        elif expected is not None and n != int(expected):
            reason = f"read {n} of {int(expected)} frames back from the written fill video"
        elif bad:
            reason = f"YuNet found a face in {len(bad)} frame(s) of the encoded fill video"
        else:
            reason = None
        self.encoded = dict(Ran=True, FramesChecked=n, ExpectedFrames=expected, UnmaskedFrames=bad,
                            UnmaskedFrameCount=len(bad), Clean=reason is None, Reason=reason)
        return self.encoded

    def result(self) -> dict:
        ran = self.available
        enc = self.encoded
        clean = ran and not self.unmasked and (enc is None or enc["Clean"])
        reasons = []
        if not ran:
            reasons.append(self.problem)
        else:
            if self.unmasked:
                reasons.append(f"YuNet found a face in {len(self.unmasked)} fill output frame(s)")
            if enc is not None and not enc["Clean"]:
                reasons.append(enc["Reason"])
        return dict(Ran=ran, Detector="YuNet (OpenCV Zoo face_detection_yunet_2023mar)", ScoreThreshold=self.score,
                    FramesChecked=self.frames, UnmaskedFrames=list(self.unmasked), UnmaskedFrameCount=len(self.unmasked),
                    FallbackBoxesFilled=self.n_boxes, FramesWithFallbackBoxes=self.box_frames, BoxPad=self.pad,
                    FirstPassUnmaskedFrames=list(self.first_pass), SecondPassBoxesFilled=self.n_second,
                    FramesWithSecondPassBoxes=self.second_frames,
                    EncodedOutputCheck=None if enc is None else dict(enc),
                    Clean=clean,
                    Reason=None if clean else "; ".join(r for r in reasons if r))

    def summary(self, name="") -> str:
        r = self.result()
        head = f"[fill check] {name}: " if name else "[fill check] "
        if not r["Ran"]:
            return head + "NOT SHAREABLE: unmasked-face check not run. " + (self.problem or "")
        s = (f"{r['FramesChecked']} frames checked, {r['FallbackBoxesFilled']} YuNet fallback box(es) filled in "
             f"{r['FramesWithFallbackBoxes']} frame(s); second pass on {r['FramesWithSecondPassBoxes']} frame(s) "
             f"({r['SecondPassBoxesFilled']} box(es)); ")
        enc = r["EncodedOutputCheck"]
        if enc is not None:
            s += f"encoded file re-check: {enc['FramesChecked']} frames, {enc['UnmaskedFrameCount']} flagged; "
        if r["Clean"]:
            return head + s + "no unmasked face: shareable"

        def show(fr):
            return ", ".join(map(str, fr[:50])) + (f", ... ({len(fr)} total)" if len(fr) > 50 else "")
        bad = []
        if r["UnmaskedFrames"]:
            bad.append(f"YuNet found a face in {len(r['UnmaskedFrames'])} output frame(s): {show(r['UnmaskedFrames'])}")
        if enc is not None and not enc["Clean"]:
            bad.append(enc["Reason"] + (f": {show(enc['UnmaskedFrames'])}" if enc["UnmaskedFrames"] else ""))
        return head + s + "NOT SHAREABLE: " + "; ".join(bad)
