"""Detector A-body (preregistered in docs/mcec_crosssite_preregistration.md).

Same features / LightGBM recipe as Detector A, minus hand-model and detector-dependent features,
with the subject scale taken from the bounding box of visible COCO-17 joints instead of the detector box.
"""
from __future__ import annotations
import glob, json, os
import numpy as np

from .features import compute_features, feature_names, SCORE_THR
from .labels import read_labels, video_key, second_labels

DROP = ("lhand_fine4", "rhand_fine4", "npers4", "has_subject4")
ALL_NAMES = feature_names()
KEEP = [i for i, n in enumerate(ALL_NAMES) if n not in DROP]
BODY_NAMES = [ALL_NAMES[i] for i in KEEP]


def kp_boxes(kpts: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """(F,P,17,2),(F,P,17) -> (F,P,5) box of joints with score >= thr (NaN if < 2 such joints)."""
    k = kpts.astype(np.float64).copy()
    ok = scores >= SCORE_THR
    k[~ok] = np.nan
    with np.errstate(all="ignore"):
        x0 = np.nanmin(k[..., 0], axis=-1); x1 = np.nanmax(k[..., 0], axis=-1)
        y0 = np.nanmin(k[..., 1], axis=-1); y1 = np.nanmax(k[..., 1], axis=-1)
    b = np.stack([x0, y0, x1, y1, np.ones_like(x0)], axis=-1)
    b[ok.sum(-1) < 2] = np.nan
    return b


def body_features(d: dict) -> np.ndarray:
    d = dict(d)
    d["body_boxes"] = kp_boxes(np.asarray(d["body_kpts"]), np.asarray(d["body_scores"]))
    F = len(d["t"])
    if "hand_kpts" not in d:
        d["hand_kpts"] = np.zeros((F, 2, 21, 2), np.float32)
        d["hand_scores"] = np.zeros((F, 2, 21), np.float32)
    X = compute_features(d)
    return X[:, KEEP]


def load_vsvig(kp_dir, xlsx, cache, exclude=("pat04_no-Sz2P",)):
    lab = read_labels(xlsx)
    recs = []
    os.makedirs(cache, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(kp_dir, "*.npz"))):
        name = os.path.basename(f)[:-4]
        if name in exclude:
            continue
        pat, stem = name.split("_", 1)
        k = video_key(stem)
        if k and (pat, k) not in lab:
            print(f"SKIP {name}: no row in Label.xlsx", flush=True)
            continue
        eeg, clin = lab[(pat, k)] if k else (None, None)
        cf = os.path.join(cache, name + ".body.npz")
        if os.path.exists(cf):
            X = np.load(cf)["X"]
        else:
            z = np.load(f)
            d = {key: z[key] for key in ("t", "body_kpts", "body_scores", "body_boxes", "sel", "hand_kpts", "hand_scores")}
            X = body_features(d); np.savez(cf, X=X)
        y = second_labels(len(X), eeg, clin)
        recs.append(dict(name=name, pat=pat, X=X, y=y, eeg=eeg, clin=clin))
        print("feat", name, X.shape, flush=True)
    return recs
