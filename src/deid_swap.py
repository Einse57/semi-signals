"""Pluggable face-swap de-identification (`--deid swap`; `ghost` is accepted as an alias).

A swapper replaces every face in a frame with a donor identity while keeping expression and head
motion. The interface is model-agnostic so GHOST (github.com/ai-forever/ghost) or a substitute can be
plugged in once its code and weights pass the licence audit (docs/FACE_SWAP_LICENCES.md). Backends:
`warp` (src/deid_warp.py: landmark-driven warp of a synthetic donor, OpenCV and numpy only) and `ghost`
(placeholder that refuses to load).

Configuration is a JSON file (`--swap-config`, or SEMI_SIGNALS_SWAP_CONFIG):

    {"backend": "ghost",
     "licence_reviewed": true,              # set only after the audit approves every file below
     "models": {"generator": "/path/G.pth", "face_detector": "...", "id_embedder": "..."},
     "donor": "/path/donor.png",            # synthetic or clearly licensed face, never a real person without a licence
     "donor_licence": "CC0 (synthetic, source URL)",
     "device": "cpu"}

A `warp` configuration:

    {"backend": "warp", "licence_reviewed": true,
     "models": {"face_mesh": "models/canonical_face_model.obj"},   # scripts/fetch_models.py --face-check
     "donor": "/path/donor.png", "donor_landmarks": "/path/donor_landmarks.npy",
     "donor_licence": "synthetic: rendered by src/deid_warp.render_donor (seed 2)",
     "geometry": "donor"}                   # or "subject"; see src/deid_warp.py

Without a valid configuration every entry point raises SwapNotConfigured with what is missing; nothing is
downloaded automatically.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

ENV_CONFIG = "SEMI_SIGNALS_SWAP_CONFIG"


class SwapNotConfigured(RuntimeError):
    """Raised when --deid swap is requested without approved, configured models."""


class FaceSwapper(Protocol):
    name: str

    def swap(self, frame: np.ndarray, faces: Sequence[np.ndarray]) -> np.ndarray:
        """Return a copy of the BGR frame with every face in `faces` (MediaPipe landmarks (478, 3) in frame
        px) replaced by the donor identity. How much of the input identity remains is measured with
        src/deid_check.py, not guaranteed by the interface."""
        ...

    def close(self) -> None:
        ...


REQUIRED = {"ghost": ("generator", "face_detector", "id_embedder"), "warp": ("face_mesh",)}
_REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def load_config(path: str | os.PathLike | None) -> dict:
    p = path or os.environ.get(ENV_CONFIG)
    if not p:
        raise SwapNotConfigured(
            "--deid swap needs a face-swap configuration (--swap-config <json> or "
            f"{ENV_CONFIG}). No swap models are configured in this repository; see docs/FACE_SWAP_LICENCES.md "
            "for the licence audit that must approve them first.")
    p = Path(p)
    if not p.exists():
        raise SwapNotConfigured(f"swap configuration {p} not found")
    return json.loads(p.read_text())


def check_config(cfg: dict) -> list[str]:
    """Problems with a configuration (empty list = usable)."""
    problems = []
    backend = cfg.get("backend")
    if backend not in REQUIRED:
        problems.append(f"backend {backend!r} unknown (known: {', '.join(REQUIRED)})")
        return problems
    if cfg.get("licence_reviewed") is not True:
        problems.append("licence_reviewed is not true: the code and every model file must pass the licence audit "
                        "(docs/FACE_SWAP_LICENCES.md) before use")
    models = cfg.get("models") or {}
    for k in REQUIRED[backend]:
        if not models.get(k):
            problems.append(f"models.{k} not set")
        elif not Path(models[k]).exists():
            problems.append(f"models.{k}: {models[k]} not found")
    if not cfg.get("donor") or not Path(cfg["donor"]).exists():
        problems.append("donor face image not set or not found")
    if not cfg.get("donor_licence"):
        problems.append("donor_licence not set (donor faces must be synthetic or clearly licensed)")
    return problems


def make_swapper(config_path=None) -> FaceSwapper:
    cfg = load_config(config_path)
    problems = check_config(cfg)
    if problems:
        raise SwapNotConfigured("face swap not configured:\n  - " + "\n  - ".join(problems))
    cls = _REGISTRY.get(cfg["backend"])
    if cls is None:
        raise SwapNotConfigured(f"no implementation registered for backend {cfg['backend']!r}")
    return cls(cfg)


@register("ghost")
class GhostSwapper:
    """Placeholder for GHOST (AEI-Net generator + ArcFace-style identity encoder). Phase 2 wires the models
    in after approval; until then construction fails with SwapNotConfigured."""
    name = "ghost"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        raise SwapNotConfigured("the GHOST backend is a scaffold: model loading is not implemented yet "
                                "(waiting for licence approval, docs/FACE_SWAP_LICENCES.md)")

    def swap(self, frame, faces):
        raise SwapNotConfigured("GHOST backend not implemented")

    def close(self):
        pass


@register("warp")
def _warp(cfg: dict):
    """Landmark-driven warp of a synthetic donor (src/deid_warp.py); OpenCV and numpy only."""
    from src.deid_warp import WarpSwapper
    return WarpSwapper(cfg)
