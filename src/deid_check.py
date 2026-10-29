"""Checks for a de-identified clip against its raw original: detector hit rate, identity similarity and
motion preservation.

  python -m src.deid_check --original raw.mp4 --deid fill.mp4 [--embedder-config emb.json] [--json out.json]

1. detector hit rate: fraction of de-identified frames in which a face detector still finds a face
   (default: MediaPipe Face Landmarker, whole frame + 4 tiles). For fill this should be ~0; for a face
   swap it should stay high (the swapped face is still a face) and the identity pass decides.
2. identity similarity: cosine similarity between face embeddings of original and de-identified frames
   (and donor, if given). Embedder is pluggable (Embedder protocol); `sface` (OpenCV Zoo SFace, Apache-2.0,
   on YuNet detections, MIT) is built in; the common ArcFace weights are not licensed for this use
   (docs/FACE_SWAP_LICENCES.md). Model files are fetched by scripts/fetch_models.py --face-check.
3. motion preservation: per-frame MediaPipe landmarks and blendshapes on both clips; Pearson correlation of
   face-region motion (head motion removed, src/face_estimator.FaceMotion regions) and of each blendshape
   over time. Fill removes the face, so this pass reports no face there.
The thresholds that make a clip acceptable are not set here; the numbers are reported only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class FaceDetector(Protocol):
    def count(self, frame: np.ndarray) -> int: ...
    def close(self) -> None: ...


class Embedder(Protocol):
    name: str
    def faces(self, frame: np.ndarray) -> list:
        """Faces found in the frame, in the embedder's own format (e.g. YuNet rows)."""
        ...
    def embed(self, frame: np.ndarray, face) -> np.ndarray | None:
        """L2-normalised identity embedding of `face` (one item of faces(), possibly from another frame of
        the same size), or None."""
        ...
    def close(self) -> None: ...


class EmbedderNotConfigured(RuntimeError):
    pass


class MediaPipeDetector:
    """Face count with the MediaPipe Face Landmarker (BlazeFace detector inside), whole frame + 4 tiles."""

    def __init__(self, model_path=None, num_faces=8):
        from src.face_estimator import FaceEstimator
        self.est = FaceEstimator(model_path=model_path, num_faces=num_faces, blendshapes=False)

    def count(self, frame):
        from src.segment_export import other_faces
        return len(other_faces(self.est, frame, None))

    def close(self):
        self.est.close()


_EMBEDDERS: dict[str, type] = {}


def register_embedder(name):
    def deco(cls):
        _EMBEDDERS[name] = cls
        return cls
    return deco


def make_embedder(config_path=None) -> Embedder:
    if not config_path:
        raise EmbedderNotConfigured("identity pass skipped: no embedder configured (--embedder-config); see "
                                    "docs/FACE_SWAP_LICENCES.md for embedders and their licences")
    cfg = json.loads(Path(config_path).read_text())
    if cfg.get("licence_reviewed") is not True:
        raise EmbedderNotConfigured("embedder licence_reviewed is not true")
    cls = _EMBEDDERS.get(cfg.get("backend"))
    if cls is None:
        raise EmbedderNotConfigured(f"no embedder registered for {cfg.get('backend')!r} (known: {sorted(_EMBEDDERS)})")
    return cls(cfg)


class YuNetDetector:
    """Face count with YuNet (OpenCV Zoo, MIT), cv2.FaceDetectorYN."""
    name = "YuNet (OpenCV Zoo)"

    def __init__(self, model_path, score=0.6):
        self.model_path, self.score, self._size = str(model_path), score, None
        self.det = cv2.FaceDetectorYN.create(self.model_path, "", (320, 320), score, 0.3, 5000)

    def detect(self, frame):
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self.det.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self.det.detect(frame)
        return [] if faces is None else list(faces)

    def count(self, frame):
        return len(self.detect(frame))

    def close(self):
        pass


@register_embedder("sface")
class SFaceEmbedder:
    """SFace (OpenCV Zoo face_recognition_sface, Apache-2.0) on YuNet detections (OpenCV Zoo, MIT).
    Config: {"backend": "sface", "licence_reviewed": true, "model": ".../face_recognition_sface_2021dec.onnx",
    "detector": ".../face_detection_yunet_2023mar.onnx"}. OpenCV's SFace example treats cosine >= 0.363
    as the same identity (opencv_zoo models/face_recognition_sface/sface.py)."""
    name = "SFace (OpenCV Zoo) + YuNet"
    same_identity_threshold = 0.363

    def __init__(self, cfg):
        for k in ("model", "detector"):
            if not cfg.get(k) or not Path(cfg[k]).exists():
                raise EmbedderNotConfigured(f"sface: {k} not set or not found ({cfg.get(k)})")
        self.yunet = YuNetDetector(cfg["detector"])
        self.rec = cv2.FaceRecognizerSF.create(str(cfg["model"]), "")

    def faces(self, frame):
        return self.yunet.detect(frame)

    def embed(self, frame, face):
        crop = self.rec.alignCrop(frame, np.asarray(face, np.float32))
        v = self.rec.feature(crop).ravel().astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    def close(self):
        pass


def frames(path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f
    cap.release()


def hit_rate(path, detector: FaceDetector) -> dict:
    n = hits = 0
    for f in frames(path):
        n += 1
        hits += detector.count(f) > 0
    return dict(frames=n, frames_with_face=hits, hit_rate=hits / n if n else None)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _iou(a, b):
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    u = aw * ah + bw * bh - iw * ih
    return iw * ih / u if u > 0 else 0.0


def _stats(v, thr=None):
    if not v:
        return dict(n=0)
    v = np.asarray(v, float)
    d = dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), max=float(v.max()))
    if thr is not None:
        d["frac_at_or_above_threshold"] = float((v >= thr).mean())
    return d


def identity_similarity(orig, deid, embedder: Embedder, donor=None) -> dict:
    """Cosine similarity of identity embeddings, every face the embedder finds in each original frame:
    - detected: against the face the embedder's detector finds at the same place (IoU >= 0.3) in the
      de-identified frame (no value when it finds none there, e.g. fill);
    - same_location: against the de-identified frame cropped and aligned at the ORIGINAL face's position,
      so a value exists even when no face is detected (what is left at that spot).
    - donor_vs_deid: de-identified faces against the donor image, if given."""
    det, loc, don = [], [], []
    d_emb = None
    if donor is not None:
        df = embedder.faces(donor)
        d_emb = embedder.embed(donor, max(df, key=lambda f: f[2] * f[3])) if len(df) else None
    for fo, fd in zip(frames(orig), frames(deid)):
        fds = embedder.faces(fd)
        for face in embedder.faces(fo):
            eo = embedder.embed(fo, face)
            if eo is None:
                continue
            el = embedder.embed(fd, face)
            if el is not None:
                loc.append(cosine(eo, el))
            best = max(fds, key=lambda g: _iou(face, g), default=None)
            if best is not None and _iou(face, best) >= 0.3:
                ed = embedder.embed(fd, best)
                if ed is not None:
                    det.append(cosine(eo, ed))
                    if d_emb is not None:
                        don.append(cosine(d_emb, ed))
    thr = getattr(embedder, "same_identity_threshold", None)
    return dict(embedder=embedder.name, same_identity_threshold=thr, detected=_stats(det, thr),
                same_location=_stats(loc, thr), donor_vs_deid=_stats(don, thr))


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def region_motion(lm_seq, dt=1.0):
    """(T, R) face-region speeds (src/face_estimator.face_region_speeds: rigid head motion removed, face
    widths per second); NaN where either frame has no face."""
    from src.face_estimator import FACE_REGIONS, face_region_speeds
    names = list(FACE_REGIONS)
    out = np.full((len(lm_seq), len(names)), np.nan)
    for t in range(1, len(lm_seq)):
        if lm_seq[t] is not None and lm_seq[t - 1] is not None:
            sp = face_region_speeds(lm_seq[t - 1], lm_seq[t], dt)
            out[t] = [sp[r] for r in names]
    return out


def motion_correlation(orig, deid, lmk_orig, lmk_deid) -> dict:
    """Pearson r between original and de-identified clips for face-region speeds and blendshapes, over frames
    with a face in both. Use a separate landmarker per clip (see identity_similarity)."""
    from src.face_estimator import BLENDSHAPE_NAMES, FACE_REGIONS
    lo, bo, ld, bd = [], [], [], []
    for fo, fd in zip(frames(orig), frames(deid)):
        ro, rd = lmk_orig.run(fo), lmk_deid.run(fd)
        lo.append(ro["landmarks"] if ro else None)
        ld.append(rd["landmarks"] if rd else None)
        bo.append(ro["blendshapes"] if ro else np.full(52, np.nan, np.float32))
        bd.append(rd["blendshapes"] if rd else np.full(52, np.nan, np.float32))
    n_both = sum(a is not None and b is not None for a, b in zip(lo, ld))
    mo, md = region_motion(lo), region_motion(ld)
    bo, bd = np.asarray(bo), np.asarray(bd)
    return dict(frames=len(lo), frames_face_in_both=n_both,
                region_motion_r={r: pearson(mo[:, i], md[:, i]) for i, r in enumerate(list(FACE_REGIONS))},
                blendshape_r={n: pearson(bo[:, i], bd[:, i]) for i, n in enumerate(BLENDSHAPE_NAMES) if i > 0})


def main(argv=None):
    p = argparse.ArgumentParser(description="Detector, identity and motion checks for a de-identified clip")
    p.add_argument("--original", required=True, help="raw clip (local only)")
    p.add_argument("--deid", required=True, help="de-identified clip with the same frames")
    p.add_argument("--donor", default=None, help="donor face image (face swap only)")
    p.add_argument("--embedder-config", default=None)
    p.add_argument("--face-model", default=None)
    p.add_argument("--yunet", default=None, help="also count faces with YuNet (face_detection_yunet_2023mar.onnx)")
    p.add_argument("--json", default=None, help="write the results here (keep it with the raw files)")
    a = p.parse_args(argv)
    from src.face_estimator import FaceEstimator
    from src.face_preprocess import FacePreprocessor
    res = {}
    det = MediaPipeDetector(a.face_model)
    try:
        res["detector"] = dict(original=hit_rate(a.original, det), deid=hit_rate(a.deid, det),
                               detector="MediaPipe Face Landmarker (whole frame + 4 tiles)")
    finally:
        det.close()
    if a.yunet:
        yn = YuNetDetector(a.yunet)
        res["detector_yunet"] = dict(original=hit_rate(a.original, yn), deid=hit_rate(a.deid, yn), detector=yn.name)
    ests = []

    def lmk():                                   # fresh estimator + tracker: no state shared between clips
        e = FaceEstimator(model_path=a.face_model)
        ests.append(e)
        return FacePreprocessor(e, "detect")
    try:
        try:
            emb = make_embedder(a.embedder_config)
            donor = cv2.imread(a.donor) if a.donor else None
            res["identity"] = identity_similarity(a.original, a.deid, emb, donor)
            emb.close()
        except EmbedderNotConfigured as e:
            res["identity"] = dict(skipped=str(e))
        res["motion"] = motion_correlation(a.original, a.deid, lmk(), lmk())
    finally:
        for e in ests:
            e.close()
    txt = json.dumps(res, indent=1)
    if a.json:
        Path(a.json).write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
