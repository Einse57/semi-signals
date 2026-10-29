"""Checks for a de-identified clip against its raw original. SCAFFOLD for the face-swap work; the detector
pass and the motion pass run today with MediaPipe, the identity pass needs a configured embedder.

  python -m src.deid_check --original raw.mp4 --deid fill.mp4 [--embedder-config emb.json] [--json out.json]

1. detector hit rate: fraction of de-identified frames in which a face detector still finds a face
   (default: MediaPipe Face Landmarker, whole frame + 4 tiles). For fill this should be ~0; for a face
   swap it should stay high (the swapped face is still a face) and the identity pass decides.
2. identity similarity: cosine similarity between face embeddings of original and de-identified frames
   (and donor, if given). Embedder is pluggable (Embedder protocol); none is bundled, because the common
   ArcFace weights are not licensed for this use (docs/FACE_SWAP_LICENCES.md).
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
    def embed(self, frame: np.ndarray, landmarks: np.ndarray | None) -> np.ndarray | None:
        """L2-normalised identity embedding of the face (landmarks in frame px), or None."""
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


def identity_similarity(orig, deid, embedder: Embedder, lmk_orig, lmk_deid, donor=None) -> dict:
    """Separate landmarkers per clip: a tracking landmarker shared between the two would carry the
    original's face position into the de-identified frames."""
    sims, donor_sims = [], []
    d_emb = embedder.embed(donor, None) if donor is not None else None
    for fo, fd in zip(frames(orig), frames(deid)):
        ro, rd = lmk_orig.run(fo), lmk_deid.run(fd)
        eo = embedder.embed(fo, ro["landmarks"]) if ro else None
        ed = embedder.embed(fd, rd["landmarks"]) if rd else None
        if eo is not None and ed is not None:
            sims.append(cosine(eo, ed))
        if ed is not None and d_emb is not None:
            donor_sims.append(cosine(d_emb, ed))
    s = lambda v: dict(n=len(v), mean=float(np.mean(v)), max=float(np.max(v))) if v else dict(n=0)
    return dict(embedder=embedder.name, original_vs_deid=s(sims), donor_vs_deid=s(donor_sims))


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
    ests = []

    def lmk():                                   # fresh estimator + tracker: no state shared between clips
        e = FaceEstimator(model_path=a.face_model)
        ests.append(e)
        return FacePreprocessor(e, "detect")
    try:
        try:
            emb = make_embedder(a.embedder_config)
            donor = cv2.imread(a.donor) if a.donor else None
            res["identity"] = identity_similarity(a.original, a.deid, emb, lmk(), lmk(), donor)
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
