"""Offline batch keypoint extraction (no display).

For each input video, samples frames at a fixed rate and writes an .npz with:
  t            (F,)        video timestamp of each sampled frame [s]
  frame_idx    (F,)        source frame index
  body_kpts    (F, P, 17, 2)  COCO-17 keypoints for up to P persons (NaN = none)
  body_scores  (F, P, 17)
  body_boxes   (F, P, 5)   x1,y1,x2,y2,score
  sel          (F,)        index into P of the tracked subject (-1 = none)
  hand_kpts    (F, 2, 21, 2)  left/right hand of the tracked subject (NaN = none)
  hand_scores  (F, 2, 21)
  patches      (F, 15, 32, 32, 3) uint8, optional (--patches): VSViG-style
               Gaussian-weighted patches around 15 joints of the subject
  meta         json string (fps, size, models, ...)

With --face (off by default; without it the file is unchanged), MediaPipe Face Landmarker on a head
crop of every person with a pose (crop from the COCO head keypoints, src/face_estimator.py):
  face_landmarks   (F, P, 478, 3) float32  x, y in frame px; z in px on the x scale, relative to the
                                           face centre (MediaPipe convention). NaN = no face.
  face_blendshapes (F, P, 52)     float32  MediaPipe blendshape scores 0..1 (names in
                                           meta["face"]["blendshape_names"]). NaN = no face.
  face_scores      (F, P)         float32  1.0 = face returned (passed the detection and presence
                                           thresholds); NaN = no face or no person.
  face_boxes       (F, P, 4)      float32  head crop x1,y1,x2,y2 used (NaN = no crop)
  face_transform   (F, P, 4, 4)   float32  only with --face-transform: MediaPipe facial transformation
                                           matrix (canonical face model -> detected face, crop camera)
Schema details: docs/FACE_LANDMARKS.md.

Usage:
  python -m src.batch_extract --out OUT_DIR [--fps 6] [--device cuda] [--patches] [--face] video1.mp4 ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# VSViG joint order (15 joints, 5 partitions of 3) expressed as COCO-17 indices:
# head(nose, l_eye, r_eye), r_arm(sho, elb, wri), l_arm, r_leg(hip, knee, ank), l_leg
VSVIG_COCO_ORDER = [0, 1, 2, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15]

MAX_PERSONS = 3


def _gauss_kernel(size: int = 128, sigma_frac: float = 0.3) -> np.ndarray:
    sigma = size * sigma_frac
    ax = np.arange(size) - (size - 1) / 2
    k = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2))
    k = (k - k.min()) / (k.max() - k.min())
    return k[:, :, None].astype(np.float32)


_KERNEL = None


def extract_vsvig_patches(img: np.ndarray, kpts15: np.ndarray, ksize: int = 128) -> np.ndarray:
    """Replicates VSViG extract_patches (crop 128px centred on joint, Gaussian
    weight, per-patch min-max to 0..255, resize x1/4) for 15 given joints."""
    import cv2
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _gauss_kernel(ksize)
    h, w = img.shape[:2]
    out = np.zeros((15, ksize // 4, ksize // 4, 3), np.uint8)
    half = ksize // 2
    for j in range(15):
        x, y = kpts15[j]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        x0, y0 = int(x) - half, int(y) - half
        crop = np.zeros((ksize, ksize, 3), np.float32)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(w, x0 + ksize), min(h, y0 + ksize)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
        crop *= _KERNEL
        mn, mx = crop.min(), crop.max()
        if mx > mn:
            crop = (crop - mn) / (mx - mn) * 255.0
        out[j] = cv2.resize(crop, (ksize // 4, ksize // 4), interpolation=cv2.INTER_LINEAR).clip(0, 255).astype(np.uint8)
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class SubjectTracker:
    """Keeps the same person across frames by IoU; (re)initialises on the
    largest*score box when lost."""

    def __init__(self, iou_thr: float = 0.2, max_lost: int = 12):
        self.box = None
        self.lost = 0
        self.iou_thr = iou_thr
        self.max_lost = max_lost

    def update(self, boxes: list[np.ndarray]) -> int:
        if not boxes:
            self.lost += 1
            if self.lost > self.max_lost:
                self.box = None
            return -1
        if self.box is not None:
            ious = [_iou(self.box, b) for b in boxes]
            k = int(np.argmax(ious))
            if ious[k] >= self.iou_thr:
                self.box, self.lost = boxes[k][:4].copy(), 0
                return k
            self.lost += 1
            if self.lost <= self.max_lost:
                return -1
        areas = [(b[2] - b[0]) * (b[3] - b[1]) * (b[4] if len(b) > 4 else 1.0) for b in boxes]
        k = int(np.argmax(areas))
        self.box, self.lost = boxes[k][:4].copy(), 0
        return k


def process_video(path: Path, out_path: Path, det, pose, hand, fps: float, patches: bool, face=None) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = 1.0 / fps
    next_t = 0.0
    T, FI, BK, BS, BB, SEL, HK, HS, PT = [], [], [], [], [], [], [], [], []
    FL, FB, FS, FX, FT = [], [], [], [], []
    face_tf = face is not None and getattr(face, "_transform", False)
    tracker = SubjectTracker()
    fi = -1
    t0 = time.time()
    while True:
        ok = cap.grab()
        if not ok:
            break
        fi += 1
        t = fi / src_fps
        if t + 1e-6 < next_t:
            continue
        next_t += step
        ok, frame = cap.retrieve()
        if not ok:
            break
        boxes = det.detect(frame)
        boxes = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:MAX_PERSONS]
        bk = np.full((MAX_PERSONS, 17, 2), np.nan, np.float32)
        bs = np.zeros((MAX_PERSONS, 17), np.float32)
        bb = np.full((MAX_PERSONS, 5), np.nan, np.float32)
        hk = np.full((2, 21, 2), np.nan, np.float32)
        hs = np.zeros((2, 21), np.float32)
        sel = tracker.update(boxes)
        persons = pose.estimate(frame, boxes) if boxes else []
        for i, p in enumerate(persons):
            bk[i], bs[i] = p["keypoints"], p["scores"]
            b = np.asarray(p["bbox"], np.float32)
            bb[i, :len(b)] = b[:5]
        if sel >= 0 and hand is not None:
            hr = hand.estimate(frame, bk[sel], bs[sel])
            for j, side in enumerate(("left", "right")):
                if hr[side] is not None:
                    hk[j], hs[j] = hr[side]["keypoints"], hr[side]["scores"]
        if face is not None:
            from src.face_estimator import empty_face
            fe = empty_face(MAX_PERSONS)
            for i, p in enumerate(persons):
                fr = face.estimate(frame, bk[i], bs[i], body_bbox=bb[i])
                if fr is not None:
                    for k_ in ("landmarks", "blendshapes", "score", "bbox", "transform"):
                        fe[k_][i] = fr[k_]
                elif np.isfinite(bk[i]).any():
                    from src.face_estimator import head_crop_bbox
                    bx = head_crop_bbox(bk[i], bs[i], H, W, body_bbox=bb[i])
                    if bx is not None:
                        fe["bbox"][i] = bx
            FL.append(fe["landmarks"]); FB.append(fe["blendshapes"]); FS.append(fe["score"])
            FX.append(fe["bbox"]); FT.append(fe["transform"])
        if patches:
            if sel >= 0:
                PT.append(extract_vsvig_patches(frame, bk[sel][VSVIG_COCO_ORDER]))
            else:
                PT.append(np.zeros((15, 32, 32, 3), np.uint8))
        T.append(t); FI.append(fi); BK.append(bk); BS.append(bs); BB.append(bb)
        SEL.append(sel); HK.append(hk); HS.append(hs)
        if len(T) % 600 == 0:
            el = time.time() - t0
            print(f"  {path.name}: {t:.0f}s of {n_frames / src_fps:.0f}s, {len(T) / el:.1f} samples/s", flush=True)
    cap.release()
    meta = dict(video=str(path.name), src_fps=src_fps, n_frames=n_frames, width=W, height=H,
                sample_fps=fps, max_persons=MAX_PERSONS, vsvig_coco_order=VSVIG_COCO_ORDER,
                elapsed_s=time.time() - t0)
    arrs = dict(t=np.array(T, np.float64), frame_idx=np.array(FI, np.int64),
                body_kpts=np.array(BK).reshape(-1, MAX_PERSONS, 17, 2),
                body_scores=np.array(BS).reshape(-1, MAX_PERSONS, 17),
                body_boxes=np.array(BB).reshape(-1, MAX_PERSONS, 5),
                sel=np.array(SEL, np.int16),
                hand_kpts=np.array(HK).reshape(-1, 2, 21, 2),
                hand_scores=np.array(HS).reshape(-1, 2, 21),
                meta=np.array(json.dumps(meta)))
    if face is not None:
        from src.face_estimator import N_FACE, N_BLEND, BLENDSHAPE_NAMES, FACE_TASK_URL
        arrs["face_landmarks"] = np.array(FL, np.float32).reshape(-1, MAX_PERSONS, N_FACE, 3)
        arrs["face_blendshapes"] = np.array(FB, np.float32).reshape(-1, MAX_PERSONS, N_BLEND)
        arrs["face_scores"] = np.array(FS, np.float32).reshape(-1, MAX_PERSONS)
        arrs["face_boxes"] = np.array(FX, np.float32).reshape(-1, MAX_PERSONS, 4)
        if face_tf:
            arrs["face_transform"] = np.array(FT, np.float32).reshape(-1, MAX_PERSONS, 4, 4)
        meta["face"] = dict(model="MediaPipe Face Landmarker", model_url=FACE_TASK_URL,
                            model_file=os.path.basename(face.model_path), runtime="mediapipe (TFLite/XNNPACK, CPU)",
                            schema="semi-signals.face/1", n_landmarks=N_FACE, blendshape_names=BLENDSHAPE_NAMES,
                            crop="head keypoints (src/face_estimator.head_crop_bbox)", transform=bool(face_tf))
        arrs["meta"] = np.array(json.dumps(meta))
    if patches:
        arrs["patches"] = np.array(PT, np.uint8).reshape(-1, 15, 32, 32, 3)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, out_path)
    return meta


def _limit_ort_threads(n: int) -> None:
    """Cap ONNX Runtime CPU threads (rtmlib creates sessions without options)."""
    import onnxruntime as ort
    _orig = ort.InferenceSession

    class _Session(_orig):
        def __init__(self, path_or_bytes, sess_options=None, *args, **kw):
            if sess_options is None:
                sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = n
            sess_options.inter_op_num_threads = 1
            super().__init__(path_or_bytes, sess_options, *args, **kw)

    ort.InferenceSession = _Session


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--device", default="cuda", help="cuda | dml | cpu | gpu (OpenVINO iGPU)")
    ap.add_argument("--backend", default=None, help="onnxruntime | openvino (default by device)")
    ap.add_argument("--patches", action="store_true")
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--threads", type=int, default=2, help="OpenCV threads")
    ap.add_argument("--ort-threads", type=int, default=0, help="ONNX Runtime intra-op threads (0 = default)")
    ap.add_argument("--det-thr", type=float, default=0.3, help="person detector score threshold")
    ap.add_argument("--face", action="store_true",
                    help="add MediaPipe face landmarks + blendshapes per person (off by default)")
    ap.add_argument("--face-transform", action="store_true",
                    help="with --face: also save MediaPipe's facial transformation matrix per face")
    ap.add_argument("--face-model", default=None, help="face_landmarker.task (default models/face_landmarker.task)")
    a = ap.parse_args(argv)

    import cv2
    cv2.setNumThreads(a.threads)
    if a.ort_threads > 0:
        _limit_ort_threads(a.ort_threads)
    backend = a.backend or ("openvino" if a.device == "gpu" else "onnxruntime")
    if a.device == "dml":  # DirectML (any DX12 GPU on Windows, pip install onnxruntime-directml)
        from rtmlib.tools import base as _rb
        _rb.RTMLIB_SETTINGS["onnxruntime"]["dml"] = "DmlExecutionProvider"
    from src.detector import PersonDetector
    from src.pose_estimator import PoseEstimator
    from src.hand_estimator import HandEstimator
    det = PersonDetector(device=a.device, backend=backend, score_thr=a.det_thr)
    pose = PoseEstimator(device=a.device, backend=backend)
    hand = None if a.no_hands else HandEstimator(device=a.device, backend=backend)
    face = None
    if a.face:
        from src.face_estimator import FaceEstimator
        face = FaceEstimator(model_path=a.face_model, transform=a.face_transform)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for v in a.videos:
        v = Path(v)
        name = f"{v.parent.name}_{v.stem}.npz"
        op = out / name
        if op.exists():
            print(f"skip {name}", flush=True); continue
        print(f"start {v}", flush=True)
        try:
            meta = process_video(v, op, det, pose, hand, a.fps, a.patches, face=face)
            print(f"done {name} {json.dumps(meta)}", flush=True)
        except Exception as e:  # keep going on other videos
            print(f"FAIL {v}: {e!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
