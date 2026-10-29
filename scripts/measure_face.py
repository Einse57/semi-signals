"""Per-frame time of the pose pipeline with and without the optional face stage (no rendering).

  python scripts/measure_face.py --input clip.mp4 --device cpu      # ONNX Runtime on the CPU
  python scripts/measure_face.py --input clip.mp4 --device gpu      # OpenVINO on the Intel iGPU

Per frame: person detection, body pose, hand pose (wrist crops), then face (head crop + MediaPipe) for
the selected person. Face runs on MediaPipe's own CPU runtime whatever --device is. The same frames
are timed with face off and on (frames decoded beforehand; first --warmup frames not counted).
Prints one line per configuration and a JSON summary.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--device", default="cpu", help="cpu (ONNX Runtime) | gpu (OpenVINO iGPU)")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--start", type=float, default=0.0, help="start time [s]")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--face-model", default=None)
    ap.add_argument("--face-backend", default="mediapipe", choices=("mediapipe", "openvino"))
    ap.add_argument("--face-device", default="CPU", type=str.upper)
    ap.add_argument("--json", default=None, help="also write the summary here")
    a = ap.parse_args(argv)

    import cv2
    from src.detector import PersonDetector
    from src.pose_estimator import PoseEstimator
    from src.hand_estimator import HandEstimator
    from src.face_estimator import FaceEstimator

    backend = "openvino" if a.device == "gpu" else "onnxruntime"
    det = PersonDetector(device=a.device, backend=backend)
    pose = PoseEstimator(device=a.device, backend=backend)
    hand = HandEstimator(device=a.device, backend=backend)
    face = FaceEstimator(model_path=a.face_model, backend=a.face_backend, device=a.face_device)

    cap = cv2.VideoCapture(a.input)
    cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    frames = []
    while len(frames) < a.frames + a.warmup:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    def run(with_face):
        st = {"det": [], "pose": [], "hand": [], "face": [], "total": []}
        found = 0
        for i, f in enumerate(frames):
            t0 = time.perf_counter()
            boxes = det.detect(f)
            t1 = time.perf_counter()
            persons = pose.estimate(f, boxes[:1]) if len(boxes) else []
            t2 = time.perf_counter()
            if persons:
                hand.estimate(f, persons[0]["keypoints"], persons[0]["scores"])
            t3 = time.perf_counter()
            if with_face and persons:
                r = face.estimate(f, persons[0]["keypoints"], persons[0]["scores"], body_bbox=persons[0]["bbox"])
                found += (r is not None) and i >= a.warmup
            t4 = time.perf_counter()
            if i >= a.warmup:
                for k, v in zip(st, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)):
                    st[k].append(v * 1000)
        ms = {k: float(np.mean(v)) if v else 0.0 for k, v in st.items()}
        return dict(face=with_face, fps=1000.0 / ms["total"], ms=ms, faces_found=found if with_face else None,
                    frames=len(st["total"]))

    res = [run(False), run(True), run(False), run(True)]       # alternate to spread drift
    out = dict(host=platform.node(), cpu=platform.processor() or platform.machine(), os=platform.platform(),
               device=a.device, backend=backend, face_runtime="mediapipe (TFLite/XNNPACK, CPU)",
               input=os.path.basename(a.input), runs=res)
    for r in res:
        print(f"{a.device:4s} face={'on ' if r['face'] else 'off'} {r['fps']:6.1f} frames/s  "
              + "  ".join(f"{k} {v:.1f} ms" for k, v in r["ms"].items())
              + (f"  faces {r['faces_found']}/{r['frames']}" if r["face"] else ""))
    print(json.dumps(out))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    face.close()


if __name__ == "__main__":
    main()
