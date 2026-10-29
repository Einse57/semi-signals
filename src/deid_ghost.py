"""GHOST face swap (`ghost` backend of src/deid_swap.py), CPU port. RESEARCH USE ONLY.

GHOST (github.com/ai-forever/ghost) is an AEI-Net generator conditioned on an ArcFace identity embedding.
Its code and weights are NOT part of this repository: the configuration points at a local, git-ignored
checkout and at weights fetched with pinned sha256 (docs/FACE_SWAP_LICENCES.md). The generator weights
carry no licence and the ArcFace parts are InsightFace non-commercial research models, so output made with
this backend is for research only and is marked "reduced risk, not de-identified".

Differences from upstream inference.py (CUDA, fp16, MXNet): float32 on the configured torch device
(default cpu); faces are aligned from the MediaPipe landmarks handed to `swap` (5 points, upstream
insightface 0.2.1 `estimate_norm` with its 224-px multi-pose templates) instead of SCRFD; the blend mask is
the feathered MediaPipe face hull instead of the MXNet 2d106 landmark mask; no super-resolution.

Configuration:

    {"backend": "ghost", "licence_reviewed": true, "research_use_acknowledged": true,
     "models": {"code": "/path/third_party_local/ghost",          # upstream checkout (network/AEI_Net.py)
                "generator": "/path/G_unet_2blocks.pth",
                "id_embedder": "/path/backbone.pth"},             # iresnet100; iresnet.py beside it
     "donor": "/path/donor.png", "donor_landmarks": "/path/donor_landmarks.npy",
     "donor_licence": "...", "device": "cpu", "num_blocks": 2}
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

# insightface 0.2.1 python-package utils/face_align.py (MIT), 112-px multi-pose templates
_SRC112 = np.array([
    [[51.642, 50.115], [57.617, 49.990], [35.740, 69.007], [51.157, 89.050], [57.025, 89.702]],
    [[45.031, 50.118], [65.568, 50.872], [39.677, 68.111], [45.177, 86.190], [64.246, 86.758]],
    [[39.730, 51.138], [72.270, 51.138], [56.000, 68.493], [42.463, 87.010], [69.537, 87.010]],
    [[46.845, 50.872], [67.382, 50.118], [72.737, 68.111], [48.167, 86.758], [67.236, 86.190]],
    [[54.796, 49.990], [60.771, 50.115], [76.673, 69.007], [55.388, 89.702], [61.257, 89.050]]], np.float32)
CROP = 224


def five_points(lm) -> np.ndarray:
    """(5, 2) image-left eye, image-right eye, nose tip, image-left and image-right mouth corner from
    MediaPipe face-mesh landmarks (468 or 478 points)."""
    p = np.asarray(lm, np.float32)[:, :2]
    return np.stack([(p[33] + p[133]) / 2, (p[362] + p[263]) / 2, p[1], p[61], p[291]]).astype(np.float32)


def _umeyama(src, dst):
    """2x3 similarity transform src -> dst (least squares)."""
    ms, md = src.mean(0), dst.mean(0)
    s, d = src - ms, dst - md
    U, S, Vt = np.linalg.svd(d.T @ s / len(src))
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    c = (S * np.diag(D)).sum() / (s ** 2).sum(1).mean()
    M = np.zeros((2, 3))
    M[:, :2] = c * R
    M[:, 2] = md - c * R @ ms
    return M


def estimate_norm(kps, size=CROP):
    """insightface 0.2.1 estimate_norm(mode='None'): the best of the five pose templates."""
    best, err = None, np.inf
    for tpl in _SRC112 * (size / 112.0):
        M = _umeyama(kps.astype(np.float64), tpl.astype(np.float64))
        e = np.linalg.norm(np.c_[kps, np.ones(5)] @ M.T - tpl, axis=1).sum()
        if e < err:
            best, err = M, e
    return best


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class GhostSwapper:
    name = "ghost"

    def __init__(self, cfg: dict):
        import torch
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        m = cfg["models"]
        code = str(Path(m["code"]).resolve())
        if code not in sys.path:
            sys.path.insert(0, code)
        from network.AEI_Net import AEI_Net
        self.dev = torch.device(cfg.get("device", "cpu"))
        G = AEI_Net("unet", num_blocks=int(cfg.get("num_blocks", 2)), c_id=512)
        G.load_state_dict(torch.load(m["generator"], map_location="cpu"))
        self.G = G.eval().float().to(self.dev)
        emb = Path(m["id_embedder"])
        ires = Path(m.get("id_embedder_code") or emb.with_name("iresnet.py"))
        arc = _load_module("ghost_iresnet", ires).iresnet100(fp16=False)
        arc.load_state_dict(torch.load(str(emb), map_location="cpu"))
        self.arc = arc.eval().float().to(self.dev)
        self.donor_licence = cfg.get("donor_licence")
        donor = cv2.imread(str(cfg["donor"]))
        if donor is None:
            raise ValueError(f"cannot read donor image {cfg['donor']}")
        dlm = np.load(cfg["donor_landmarks"]) if cfg.get("donor_landmarks") else self._detect(donor)
        crop = cv2.warpAffine(donor, estimate_norm(five_points(dlm)), (CROP, CROP), borderValue=0.0)
        self.donor_crop = crop
        with torch.no_grad():
            x = self._to_t(crop[None, :, :, ::-1])               # upstream feeds the source crop as RGB
            self.embed = self.arc(F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=True))

    @staticmethod
    def _detect(img):
        from src.face_estimator import FaceEstimator
        est = FaceEstimator(num_faces=1)
        try:
            faces = est.all_faces(img)
        finally:
            est.close()
        if not faces:
            raise ValueError("no face found in the donor image; give donor_landmarks")
        return faces[0]

    def _to_t(self, arr):
        t = self.torch.from_numpy(np.ascontiguousarray(arr)).float().to(self.dev) / 255.0
        return ((t - 0.5) / 0.5).permute(0, 3, 1, 2)

    def close(self):
        pass

    def reset(self):
        pass

    def swap(self, frame, faces):
        out = frame.copy()
        for lm in faces:
            if lm is None:
                continue
            lm = np.asarray(lm, np.float32)
            if not np.isfinite(lm[:, :2]).all():
                continue
            out = self._one(out, lm)
        return out

    def _one(self, frame, lm):
        torch = self.torch
        M = estimate_norm(five_points(lm))
        crop = cv2.warpAffine(frame, M, (CROP, CROP), borderValue=0.0)
        tgt = cv2.resize(crop, (256, 256))[None, :, :, ::-1]   # upstream: BGR crop -> RGB, 256 px
        with torch.no_grad():
            y, _ = self.G(self._to_t(tgt), self.embed)
        y = ((y.permute(0, 2, 3, 1)[0] * 0.5 + 0.5).clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
        y = cv2.resize(y[:, :, ::-1], (CROP, CROP))
        h, w = frame.shape[:2]
        Mi = cv2.invertAffineTransform(M)
        sw = cv2.warpAffine(y, Mi, (w, h), borderMode=cv2.BORDER_REPLICATE)
        mask = np.zeros((h, w), np.uint8)
        hull = cv2.convexHull(lm[:468, :2].astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        r = max(3, int(0.06 * np.sqrt(cv2.contourArea(hull) + 1)))
        mask = cv2.erode(mask, np.ones((r, r), np.uint8))
        k = 2 * r + 1
        a = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32)[..., None] / 255.0
        return (a * sw + (1 - a) * frame).astype(np.uint8)
