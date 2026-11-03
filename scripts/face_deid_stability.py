"""Frame-to-frame face tracking stability of de-identified clips against the original.

  python scripts/face_deid_stability.py --original orig.mp4 --deid fill=fill.mp4 warp=warp.mp4 --json out.json

Each clip goes through its own Face Landmarker (src/face_preprocess.py, mode detect; no smoothing), frame by
frame. Per clip:
  face_frames       frames with a face (of 478 landmarks each)
  transitions_100   face lost or regained, per 100 frames
  longest_gap       longest run of frames without a face (inside the clip)
  step_median       median frame-to-frame landmark displacement, rigid head motion removed (similarity
                    alignment on all 478 points, as in src/face_estimator.face_region_speeds), in face widths
  hf_median         median high-frequency residual: aligned landmarks minus a centred 5-frame moving average
                    (runs of >= 5 frames with a face), in face widths; higher = more flicker
Raw clips stay local; the JSON holds numbers only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.face_estimator import FACE_WIDTH_PAIR, FaceEstimator, _align  # noqa: E402
from src.face_preprocess import FacePreprocessor  # noqa: E402

WIN = 5


def landmarks(path, model=None):
    est = FaceEstimator(model_path=model)
    pre = FacePreprocessor(est, "detect")
    cap, out = cv2.VideoCapture(str(path)), []
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            r = pre.run(f)
            out.append(None if r is None else np.asarray(r["landmarks"], float)[:, :2])
    finally:
        cap.release()
        est.close()
    return out


def _fw(p):
    return float(np.linalg.norm(p[FACE_WIDTH_PAIR[0]] - p[FACE_WIDTH_PAIR[1]]))


def stability(seq) -> dict:
    n = len(seq)
    has = np.array([s is not None for s in seq])
    trans = int(np.count_nonzero(has[1:] != has[:-1])) if n > 1 else 0
    gap = cur = 0
    for h in has:
        cur = 0 if h else cur + 1
        gap = max(gap, cur)
    steps, hf = [], []
    for t in range(1, n):
        if has[t] and has[t - 1] and _fw(seq[t]) > 1e-6:
            steps.append(np.linalg.norm(seq[t] - _align(seq[t - 1], seq[t]), axis=1).mean() / _fw(seq[t]))
    t = 0
    while t < n:                                   # runs of consecutive frames with a face
        if not has[t]:
            t += 1
            continue
        e = t
        while e < n and has[e]:
            e += 1
        if e - t >= WIN:
            ref = seq[t]
            fw = _fw(ref)
            A = np.stack([_align(seq[i], ref) for i in range(t, e)]) / max(fw, 1e-6)
            for j in range(WIN // 2, e - t - WIN // 2):
                sm = A[j - WIN // 2:j + WIN // 2 + 1].mean(0)
                hf.append(float(np.linalg.norm(A[j] - sm, axis=1).mean()))
        t = e
    return dict(frames=n, face_frames=int(has.sum()),
                transitions_100=100.0 * trans / n if n else None, transitions=trans, longest_gap=int(gap),
                step_median=float(np.median(steps)) if steps else None, step_n=len(steps),
                hf_median=float(np.median(hf)) if hf else None, hf_n=len(hf))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--original", required=True, help="raw clip (local only)")
    p.add_argument("--deid", nargs="+", required=True, help="name=path of each de-identified clip (same frames)")
    p.add_argument("--face-model", default=None)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    res = {"original": stability(landmarks(a.original, a.face_model))}
    for item in a.deid:
        name, path = item.split("=", 1)
        res[name] = stability(landmarks(path, a.face_model))
    txt = json.dumps(res, indent=1)
    if a.json:
        Path(a.json).write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
