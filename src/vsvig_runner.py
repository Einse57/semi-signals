"""Detector B: VSViG (ECCV 2024) skeleton-patch model on semi-signals keypoints.

Windows: 30 samples at 6 fps (5 s), patches (30,15,3,32,32) 0..255 float, kpts (30,15,3) = (x_px, y_px, score)
in VSViG joint order (src.batch_extract.VSVIG_COCO_ORDER). Output: sigmoid probability.

Caveat: the released VSViG-base weights were trained on all 14 WU-SAHZU patients
(within-patient split), so scores on that dataset are not an honest estimate.
"""
from __future__ import annotations

import numpy as np

from src.batch_extract import VSVIG_COCO_ORDER

WIN = 30


def build_windows(d: dict, stride: int = 3):
    """Returns (ends_t, patches (N,30,15,3,32,32) float32, kpts (N,30,15,3) float32, valid (N,))."""
    F = len(d["t"])
    P = d["body_kpts"].shape[1]
    sel = d["sel"].astype(int)
    idx = np.clip(sel, 0, P - 1)
    K = d["body_kpts"][np.arange(F), idx][:, VSVIG_COCO_ORDER]          # (F,15,2)
    S = d["body_scores"][np.arange(F), idx][:, VSVIG_COCO_ORDER]        # (F,15)
    kp = np.concatenate([np.nan_to_num(K), S[..., None]], -1).astype(np.float32)
    kp[sel < 0] = 0
    pt = d["patches"]                                                   # (F,15,32,32,3) uint8
    ends, PW, KW, V = [], [], [], []
    for e in range(WIN, F + 1, stride):
        s = e - WIN
        ends.append(d["t"][e - 1])
        PW.append(pt[s:e]); KW.append(kp[s:e]); V.append((sel[s:e] >= 0).mean())
    if not ends:
        return np.zeros(0), np.zeros((0, WIN, 15, 3, 32, 32), np.float32), np.zeros((0, WIN, 15, 3), np.float32), np.zeros(0)
    PW = np.ascontiguousarray(np.stack(PW).transpose(0, 1, 2, 5, 3, 4), dtype=np.float32)
    return np.array(ends), PW, np.ascontiguousarray(np.stack(KW)), np.array(V)


def load_torch_model(weights: str):
    import torch
    from third_party.vsvig.VSViG import VSViG_base
    m = VSViG_base()
    sd = torch.load(weights, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd)
    m.eval()
    return m


def predict_torch(m, PW, KW, batch: int = 16):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(PW), batch):
            out.append(m(torch.from_numpy(PW[i:i + batch]), torch.from_numpy(KW[i:i + batch])).reshape(-1).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def per_second(ends, probs, n_seconds):
    """Max window probability among windows ending inside each second."""
    out = np.full(n_seconds, np.nan)
    for t, p in zip(ends, probs):
        s = int(np.floor(t))
        if 0 <= s < n_seconds:
            out[s] = p if np.isnan(out[s]) else max(out[s], p)
    return out
