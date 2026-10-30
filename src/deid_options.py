"""Face de-identification methods offered in the UI (launcher face mode and the annotation panel).

One table for every method, so the launcher, the annotation tool and the export command show the same
names, defaults and labels, and the export command (src/segment_export.py, the process the UI starts)
checks the same rules again server-side instead of trusting the window.

Selector (two levels):
- Method: fill (the default overall: the shareable fallback), mosaic8, mosaic16, gauss, or "swap".
- Swap backend (only when "swap" is chosen): ghost (the default swap, src/deid_swap.DEFAULT_BACKEND), warp,
  liveportrait.
Labels (also written into the export sidecars as DeidentificationStatus):
- fill: "shareable fallback (default)", the only method that may be exported for sharing.
- mosaic8, warp, ghost, liveportrait: "reduced risk, not de-identified".
- mosaic16: "not de-identifying" (provisional: still matched identity on 5 of 6 clips scored; final check
  from the phase 4 results in docs/FACE_DEID_TRADEOFF.md).
- gauss: "local review only"; written under --raw-root only, never into the export, never shareable.
ghost and liveportrait need the research-use acknowledgement, passed on as research_use_acknowledged.
Third-party code and weights are not redistributed (docs/FACE_SWAP_LICENCES.md); when they are missing,
missing_weights_message() says where to fetch them from.

No tkinter here (tested headless); the widgets are in src/deid_ui.py.
"""
from __future__ import annotations

from pathlib import Path

from src.deid_swap import DEFAULT_BACKEND, DEID_STATUS, RESEARCH_ONLY

ROOT = Path(__file__).resolve().parents[1]
SHAREABLE = "shareable fallback (default)"
NOT_DEID = "not de-identifying"
LOCAL_ONLY = "local review only"
MOSAIC16_NOTE = ("provisional: still matched identity on 5 of 6 clips scored so far; the final check comes "
                 "from the phase 4 results (docs/FACE_DEID_TRADEOFF.md)")

# key: (group, UI title, status label, shareable, local_only, research_only)
METHODS = {
    "fill": ("mask", "Fill (solid colour)", SHAREABLE, True, False, False),
    "mosaic8": ("mask", "Mosaic, 8 blocks across the face", DEID_STATUS, False, False, False),
    "mosaic16": ("mask", "Mosaic, 16 blocks across the face", NOT_DEID, False, True, False),
    "gauss": ("mask", "Gaussian blur (sigma = face width / 6)", LOCAL_ONLY, False, True, False),
    "ghost": ("swap", "GHOST face swap (default swap)", DEID_STATUS, False, False, True),
    "warp": ("swap", "Warp of a synthetic donor", DEID_STATUS, False, False, False),
    "liveportrait": ("swap", "LivePortrait reenactment", DEID_STATUS, False, False, True),
}
MASK_METHODS = tuple(k for k, v in METHODS.items() if v[0] == "mask")
SWAP_METHODS = tuple(k for k, v in METHODS.items() if v[0] == "swap")
DEFAULT_METHOD = "fill"                  # overall default: the shareable fallback
DEFAULT_SWAP = DEFAULT_BACKEND           # "ghost": the default once a swap is chosen
FAMILIES = MASK_METHODS + ("swap",)      # first selector level
assert DEFAULT_SWAP in SWAP_METHODS and set(RESEARCH_ONLY) == {k for k, v in METHODS.items() if v[5]}

FACE_MODEL_HINT = "python scripts/fetch_models.py --face --skip-data --skip-vsvig"
WARP_HINT = "python scripts/fetch_models.py --face-check --skip-data --skip-vsvig"
LICENCES_DOC = "docs/FACE_SWAP_LICENCES.md"
OFFICIAL = {
    "ghost": ("GHOST code: https://github.com/ai-forever/ghost (pinned commit in " + LICENCES_DOC + ", 'Phase 3' > "
              "'Third-party code'); weights with GHOST's own download_models.sh "
              "(https://github.com/ai-forever/ghost/blob/main/download_models.sh): G_unet_2blocks.pth and "
              "backbone.pth, sha256 pins in " + LICENCES_DOC + " 'Weights'"),
    "liveportrait": ("LivePortrait code: https://github.com/KlingTeam/LivePortrait (pinned commit in " + LICENCES_DOC
                     + "); weights: https://huggingface.co/KlingTeam/LivePortrait (sha256 pins in " + LICENCES_DOC
                     + " 'Weights')"),
    "warp": "the canonical face mesh (MediaPipe, Apache-2.0): " + WARP_HINT,
}


def family_of(method: str) -> str:
    return "swap" if method in SWAP_METHODS else method


def resolve(family: str, swap: str | None = None) -> str:
    """Selector state -> method. family "swap" without a backend gives DEFAULT_SWAP (ghost)."""
    if family == "swap":
        return swap if swap in SWAP_METHODS else DEFAULT_SWAP
    if family not in METHODS:
        raise ValueError(f"unknown de-identification method {family!r} ({', '.join(FAMILIES)})")
    return family


def title(method: str) -> str:
    return METHODS[method][1]


def label(method: str) -> str:
    """The status label shown next to the selector and written to the sidecar."""
    return METHODS[method][2]


def is_shareable(method: str) -> bool:
    return METHODS[method][3]


def is_local_only(method: str) -> bool:
    return METHODS[method][4]


def needs_ack(method: str) -> bool:
    return METHODS[method][5]


def label_text(method: str) -> str:
    """One line for the UI: the label plus what it implies."""
    s = label(method)
    if method == "fill":
        return s + ": the only method that may be exported for sharing"
    if method == "mosaic16":
        return f"{s} ({MOSAIC16_NOTE}); written to the local raw folder only"
    if method == "gauss":
        return s + ": written to the local raw folder only, never exported or shared"
    if needs_ack(method):
        return s + "; research use only: output stays on this computer"
    return s + "; not for sharing"


def missing_weights_message(method: str, problems=()) -> str:
    """Clear message for missing models: what is missing and the official place to fetch it from."""
    head = f"{title(method)}: models not available on this computer"
    lines = [head] + [f"  - {p}" for p in problems]
    if method in SWAP_METHODS:
        lines.append("This project does not redistribute third-party code or weights. Fetch them yourself from the "
                     f"official sources and accept their terms ({LICENCES_DOC}): {OFFICIAL[method]}.")
        lines.append("Then point a swap configuration JSON at them (format: src/deid_swap.py) and choose it as "
                     "'Swap config'.")
        if method == "ghost":
            lines.append("No GHOST files? Choose Fill (the shareable fallback) or the warp swap instead.")
    else:
        lines.append(f"Fetch the MediaPipe face model with: {FACE_MODEL_HINT}")
    return "\n".join(lines)


def face_model_problems(models_dir: Path | None = None) -> list[str]:
    p = Path(models_dir or ROOT / "models") / "face_landmarker.task"
    return [] if p.exists() else [f"{p} not found"]


def swap_overrides(method: str, research_ack: bool) -> dict:
    """Configuration values the UI choice sets: the backend, and the acknowledgement for research backends."""
    o = {"backend": method}
    if needs_ack(method):
        o["research_use_acknowledged"] = bool(research_ack)
    return o


def weights_problems(method: str, swap_config=None, models_dir: Path | None = None) -> list[str]:
    """Missing files for a method ([] = ready). Swaps: the configuration's model, code and donor files (the
    research gate and the backend name are checked separately by validate_request)."""
    if method not in SWAP_METHODS:
        return face_model_problems(models_dir)
    from src.deid_swap import SwapNotConfigured, check_config, load_config
    try:
        cfg = load_config(swap_config)
    except SwapNotConfigured:
        return ["no swap configuration chosen (Swap config)"]
    except (OSError, ValueError) as e:
        return [f"swap configuration unreadable: {e}"]
    cfg = dict(cfg, **swap_overrides(method, True))
    return [p for p in check_config(cfg) if "research_use_acknowledged" not in p]


def validate_request(method: str, *, research_ack: bool = False, shareable: bool = False,
                     swap_config_backend: str | None = None) -> list[str]:
    """Server-side rules for one export request ([] = allowed)."""
    if method not in METHODS:
        return [f"unknown de-identification method {method!r} (known: {', '.join(METHODS)})"]
    err = []
    if shareable and not is_shareable(method):
        err.append(f"{method} is '{label(method)}' and cannot be exported for sharing; only fill is shareable")
    if needs_ack(method) and not research_ack:
        err.append(f"{method} is research use only: tick the research-use acknowledgement "
                   "(--research-use-acknowledged) to run it locally; its output is never shared")
    if method in SWAP_METHODS and swap_config_backend and swap_config_backend != method:
        err.append(f"the swap configuration names backend {swap_config_backend!r} but {method!r} was chosen")
    return err


def export_variants(method: str) -> list[str]:
    """segment_export variants for a method: the image variant (swaps -> "swap") plus points."""
    return ["swap" if method in SWAP_METHODS else method, "points"]


def default_dirs() -> tuple[Path, Path]:
    """(de-identified export root, local raw root) used by the UI: outside the repository."""
    base = Path.home() / "semi-signals-local"
    return base / "export", base / "raw"


def export_cmd(video, method: str, *, python: str, annotations=None, out=None, raw_root=None, swap_config=None,
               research_ack: bool = False, shareable: bool = False, face_preprocess: str = "none",
               mode: str = "face") -> list[str]:
    """The segment_export command the UI starts for one method (validated again by segment_export)."""
    d_out, d_raw = default_dirs()
    cmd = [python, "-u", "-m", "src.segment_export", "--input", str(video), "--out", str(out or d_out),
           "--raw-root", str(raw_root or d_raw), "--deid-method", method, "--mode", mode,
           "--face-preprocess", face_preprocess]
    if annotations and Path(annotations).is_file():
        cmd += ["--annotations", str(annotations)]
    else:
        cmd.append("--whole")
    if method in SWAP_METHODS and swap_config:
        cmd += ["--swap-config", str(swap_config)]
    if needs_ack(method) and research_ack:
        cmd.append("--research-use-acknowledged")
    if shareable:
        cmd.append("--shareable")
    return cmd


# ── before/after preview (one frame) ─────────────────────────
def preview_image(frame, method: str, lm, others=(), swapper=None, width: int = 360):
    """Side-by-side BGR image: the frame before and after `method`, with the label under it."""
    import cv2
    import numpy as np
    from src.segment_export import mask_faces
    if method in SWAP_METHODS:
        faces = ([lm] if lm is not None else []) + list(others)
        after = swapper.swap(frame, faces) if (swapper is not None and faces) else None
        if after is None:
            after = mask_faces(frame, None, "frame", style="fill")
    else:
        after = mask_faces(frame, lm, "frame", others=others, style=method)
    h, w = frame.shape[:2]
    s = width / w
    size = (width, int(round(h * s)))
    a, b = cv2.resize(frame, size), cv2.resize(after, size)
    cap_h = 44
    out = np.full((size[1] + cap_h, 2 * width + 8, 3), 255, np.uint8)
    out[:size[1], :width], out[:size[1], width + 8:] = a, b
    col = (0, 120, 0) if is_shareable(method) else (0, 0, 190)
    cv2.putText(out, "before", (6, size[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(out, f"after: {method}", (width + 14, size[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1,
                cv2.LINE_AA)
    cv2.putText(out, label(method), (width + 14, size[1] + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return out


def preview_from_video(video, method: str, t: float | None = None, *, swap_config=None, research_ack=False,
                       models_dir: Path | None = None):
    """Preview one frame of `video` (middle frame by default). Raises ValueError with the reason (missing
    models, research gate) instead of running a method the export would refuse."""
    import cv2
    err = validate_request(method, research_ack=research_ack)
    if err:
        raise ValueError("; ".join(err))
    probs = face_model_problems(models_dir)
    if probs:
        raise ValueError(missing_weights_message(method, probs))
    swapper = None
    if method in SWAP_METHODS:
        probs = weights_problems(method, swap_config, models_dir)
        if probs:
            raise ValueError(missing_weights_message(method, probs))
        from src.deid_swap import make_swapper
        swapper = make_swapper(swap_config, swap_overrides(method, research_ack))
    cap = cv2.VideoCapture(str(video))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fi = int(t * fps) if t is not None else n // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(fi, n - 1)))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError(f"cannot read a frame from {video}")
    from src.face_estimator import FaceEstimator
    est = FaceEstimator(model_path=str(Path(models_dir or ROOT / "models") / "face_landmarker.task"))
    try:
        r = est.detect_image(frame)
    finally:
        est.close()
    lm = r["landmarks"] if r is not None else None
    try:
        return preview_image(frame, method, lm, swapper=swapper)
    finally:
        if swapper is not None:
            swapper.close()
