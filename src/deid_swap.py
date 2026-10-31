"""Pluggable face swap (`--deid swap`; `ghost` is accepted as an alias of the variant name).

A swapper replaces every face in a frame with a donor face while keeping expression and head motion.
Swap output is REDUCED RISK, NOT DE-IDENTIFIED: the outline, hair, skin colour, pose and expression of the
subject remain and src/deid_check.py measures how much identity is left, so its sidecars carry
Deidentified: false. The face fill (`--deid fill`) stays the overall default, the shareable fallback and
the only mode treated as shareable.

Backends (`ghost` is the default swap backend: a configuration without "backend" uses it):
- `warp` (src/deid_warp.py): landmark-driven warp of a synthetic donor; OpenCV and numpy only.
- `ghost` (src/deid_ghost.py): GHOST AEI-Net generator, CPU port. RESEARCH USE ONLY.
- `liveportrait` (src/deid_liveportrait.py): LivePortrait reenactment of the donor. RESEARCH USE ONLY.
The research backends load third-party code and weights from a local location outside this repository
(never committed or redistributed) and need "research_use_acknowledged": true; licences and the pinned
sha256 of every weight file are in docs/FACE_SWAP_LICENCES.md.

Configuration is a JSON file (`--swap-config`, or SEMI_SIGNALS_SWAP_CONFIG):

    {"backend": "warp", "licence_reviewed": true,
     "models": {"face_mesh": "models/canonical_face_model.obj"},   # scripts/fetch_models.py --face-check
     "donor": "/path/donor.png", "donor_landmarks": "/path/donor_landmarks.npy",
     "donor_licence": "synthetic: rendered by src/deid_warp.render_donor (seed 2)",
     "geometry": "donor"}                   # or "subject"; see src/deid_warp.py

    {"backend": "ghost", "licence_reviewed": true, "research_use_acknowledged": true,   # default backend
     "models": {"code": "/path/third_party_local/ghost", "generator": "/path/G_unet_2blocks.pth",
                "id_embedder": "/path/backbone.pth"},
     "donor": ..., "donor_landmarks": ..., "donor_licence": ..., "device": "cpu"}

    {"backend": "liveportrait", "licence_reviewed": true, "research_use_acknowledged": true,
     "models": {"code": "/path/third_party_local/LivePortrait", "weights": "/path/weights/liveportrait"},
     "donor": ..., "donor_landmarks": ..., "donor_licence": ..., "device": "cpu"}

Third-party code, weights and donor images are not redistributed by this project: the user fetches them from
the official sources and accepts their terms (docs/FACE_SWAP_LICENCES.md).

Donor faces must be synthetic or clearly licensed, never a real person without a licence. Without a valid
configuration every entry point raises SwapNotConfigured with what is missing; nothing is downloaded
automatically.
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
        px) replaced by the donor face. How much of the input identity remains is measured with
        src/deid_check.py, not guaranteed by the interface."""
        ...

    def close(self) -> None:
        ...


REQUIRED = {"warp": ("face_mesh",), "ghost": ("code", "generator", "id_embedder"),
            "liveportrait": ("code", "weights")}
RESEARCH_ONLY = ("ghost", "liveportrait")
DEFAULT_BACKEND = "ghost"          # the default when swap mode is chosen; fill stays the shareable default
DEID_STATUS = "reduced risk, not de-identified"
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


def backend_of(cfg: dict) -> str:
    """The configured backend, or DEFAULT_BACKEND ("ghost") when the configuration does not name one."""
    return cfg.get("backend") or DEFAULT_BACKEND


def check_config(cfg: dict) -> list[str]:
    """Problems with a configuration (empty list = usable). A missing "backend" means DEFAULT_BACKEND."""
    problems = []
    backend = backend_of(cfg)
    if backend not in REQUIRED:
        problems.append(f"backend {backend!r} unknown (known: {', '.join(REQUIRED)})")
        return problems
    if cfg.get("licence_reviewed") is not True:
        problems.append("licence_reviewed is not true: the code and every model file must pass the licence audit "
                        "(docs/FACE_SWAP_LICENCES.md) before use")
    if backend in RESEARCH_ONLY and cfg.get("research_use_acknowledged") is not True:
        problems.append(f"{backend} uses research-only code or weights (docs/FACE_SWAP_LICENCES.md): set "
                        "research_use_acknowledged to true to use it locally; never commit or share its output")
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


def make_swapper(config_path=None, overrides: dict | None = None) -> FaceSwapper:
    """overrides: values set by the UI choice (src/deid_options.swap_overrides: backend and, for research
    backends, research_use_acknowledged from the UI acknowledgement)."""
    cfg = load_config(config_path)
    cfg = dict(cfg, **(overrides or {}))
    cfg = dict(cfg, backend=backend_of(cfg))
    problems = check_config(cfg)
    if problems:
        raise SwapNotConfigured("face swap not configured:\n  - " + "\n  - ".join(problems))
    cls = _REGISTRY.get(cfg["backend"])
    if cls is None:
        raise SwapNotConfigured(f"no implementation registered for backend {cfg['backend']!r}")
    return cls(cfg)


@register("ghost")
def _ghost(cfg: dict):
    """GHOST (ai-forever/ghost) CPU port; research use only (src/deid_ghost.py)."""
    from src.deid_ghost import GhostSwapper
    return GhostSwapper(cfg)


@register("liveportrait")
def _liveportrait(cfg: dict):
    """LivePortrait (KlingTeam/LivePortrait) reenactment of the donor; research use only."""
    from src.deid_liveportrait import LivePortraitSwapper
    return LivePortraitSwapper(cfg)


@register("warp")
def _warp(cfg: dict):
    """Landmark-driven warp of a synthetic donor (src/deid_warp.py); OpenCV and numpy only."""
    from src.deid_warp import WarpSwapper
    return WarpSwapper(cfg)
