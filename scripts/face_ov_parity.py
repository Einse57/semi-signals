"""Parity of the OpenVINO face backend (src/face_ov) against MediaPipe Tasks on CPU.

  python scripts/face_ov_parity.py --device CPU --clips a.mp4 b.mpg --max-frames 300 --out parity.json
  python scripts/face_ov_parity.py --dump-raw raw.npz ...   # also save identical model inputs/outputs

Two comparisons per clip, both on whole frames:
  video: MediaPipe VIDEO mode vs face_ov video=True (tracking + One Euro smoothing in both)
  image: MediaPipe IMAGE mode vs face_ov video=False (detector on every frame, no smoothing)
Metrics: detection agreement (%), landmark error in px / inter-ocular distance (iris centres 468-473 of
the MediaPipe result; mean and p95 over landmarks and frames), blendshape MAE and per-channel Pearson r.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from face_ov import FaceLandmarkerOV  # noqa: E402

TASK = ROOT / "models" / "face_landmarker.task"


def read_frames(path, max_frames):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def mp_run(frames, fps, video):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(TASK), delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_faces=1, output_face_blendshapes=True)
    out = []
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        for i, f in enumerate(frames):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f))
            r = lm.detect_for_video(img, int(round(i * 1000 / fps))) if video else lm.detect(img)
            if not r.face_landmarks:
                out.append(None)
                continue
            l = np.array([[p.x, p.y, p.z] for p in r.face_landmarks[0]], np.float32)
            b = np.array([c.score for c in sorted(r.face_blendshapes[0], key=lambda c: c.index)], np.float32)
            out.append((l, b))
    return out


def ov_run(frames, fps, video, device, precision=None, dump=None):
    est = FaceLandmarkerOV(TASK, device=device, video=video, precision=precision)
    out = []
    for i, f in enumerate(frames):
        r = est.process(f, int(round(i * 1000 / fps)) if video else None)
        out.append(None if r is None else (r.landmarks, r.blendshapes))
        if dump is not None and r is not None and len(dump["lmk_in"]) < 40 and i % 3 == 0:
            h, w = f.shape[:2]
            from face_ov import geometry as G
            dx = est.detector_input(f)
            db, ds = est.run_detector(dx)
            lx = est.landmark_input(f, r.roi)
            lr, lp = est.run_landmarks(lx)
            bx = G.blendshape_input(r.landmarks, w, h)
            dump["det_in"].append(dx); dump["det_boxes"].append(db); dump["det_scores"].append(ds)
            dump["lmk_in"].append(lx); dump["lmk_out"].append(lr.reshape(1, 1, 1, -1)); dump["lmk_pres"].append([lp])
            dump["bs_in"].append(bx); dump["bs_out"].append(est.run_blendshapes(bx))
    return out, est.compile_ms


def compare(ref, test, w, h):
    n = len(ref)
    both = [(a, b) for a, b in zip(ref, test) if a is not None and b is not None]
    agree = sum((a is None) == (b is None) for a, b in zip(ref, test)) / max(n, 1) * 100
    res = dict(frames=n, ref_faces=sum(a is not None for a in ref), test_faces=sum(b is not None for b in test),
               detection_agreement_pct=round(agree, 2))
    if not both:
        return res
    errs, bsr, bst = [], [], []
    for (la, ba), (lb, bb) in both:
        pa, pb = la[:, :2] * [w, h], lb[:, :2] * [w, h]
        iod = float(np.linalg.norm(pa[468] - pa[473]))
        errs.append(np.linalg.norm(pa - pb, axis=1) / max(iod, 1e-6))
        bsr.append(ba); bst.append(bb)
    e = np.concatenate(errs)
    bsr, bst = np.array(bsr), np.array(bst)
    cors = []
    for c in range(1, bsr.shape[1]):
        if bsr[:, c].std() > 1e-3 and bst[:, c].std() > 1e-6:
            cors.append(float(np.corrcoef(bsr[:, c], bst[:, c])[0, 1]))
    res.update(landmark_err_iod_mean=float(e.mean()), landmark_err_iod_p95=float(np.percentile(e, 95)),
               blendshape_mae=float(np.abs(bsr - bst).mean()), blendshape_mae_max_channel=float(np.abs(bsr - bst).mean(0).max()),
               blendshape_r_median=float(np.median(cors)) if cors else float("nan"),
               blendshape_r_min=float(np.min(cors)) if cors else float("nan"), blendshape_r_channels=len(cors))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--precision", default=None, help="OpenVINO INFERENCE_PRECISION_HINT, e.g. f32")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-raw", default=None)
    ap.add_argument("--no-mediapipe", action="store_true", help="only run face_ov (reference read from --ref)")
    ap.add_argument("--ref", default=None, help="npz of MediaPipe results saved by an earlier run")
    ap.add_argument("--save-ref", default=None)
    a = ap.parse_args()
    report = dict(device=a.device, precision=a.precision, clips={})
    dump = {k: [] for k in ("det_in", "det_boxes", "det_scores", "lmk_in", "lmk_out", "lmk_pres", "bs_in", "bs_out")} if a.dump_raw else None
    refs = dict(np.load(a.ref, allow_pickle=True)) if a.ref else {}
    saved = {}
    for clip in a.clips:
        frames, fps = read_frames(clip, a.max_frames)
        h, w = frames[0].shape[:2]
        name = Path(clip).stem
        rep = {}
        for mode, video in (("video", True), ("image", False)):
            key = f"{name}:{mode}"
            ref = list(refs[key]) if key in refs else mp_run(frames, fps, video)
            saved[key] = np.array(ref, dtype=object)
            test, cms = ov_run(frames, fps, video, a.device, a.precision, dump if mode == "video" else None)
            rep[mode] = compare(ref, test, w, h)
            rep["compile_ms"] = {k: round(v, 1) for k, v in cms.items()}
            print(name, mode, json.dumps(rep[mode]), flush=True)
        report["clips"][name] = dict(size=[w, h], fps=fps, **rep)
    Path(a.out).write_text(json.dumps(report, indent=1))
    if a.save_ref:
        np.savez(a.save_ref, **saved, allow_pickle=True)
    if dump is not None:
        np.savez(a.dump_raw, **{k: np.concatenate([np.asarray(x) for x in v]) if k not in ("lmk_pres", "bs_out") else np.asarray(v) for k, v in dump.items()})


if __name__ == "__main__":
    main()
