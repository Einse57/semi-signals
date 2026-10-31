"""Face de-identification choices offered in the UI (launcher face mode and the annotation panel).

One table, so the launcher, the annotation tool and the export command show the same names, defaults and
labels, and the export command (src/segment_export.py, the process the UI starts) checks the same rules
again instead of trusting the window.

The UI offers only:
- View: Original (the video as recorded; viewing and preview only, never exported from the selector).
- De-identify: Fill (the default and the only shareable export) or GHOST (face swap, the default swap
  backend src/deid_swap.DEFAULT_BACKEND).
Labels (also written into the export sidecars as DeidentificationStatus):
- fill: "shareable fallback (default)", the only method that may be exported for sharing.
- ghost: "reduced risk, not de-identified"; research use only, needs the research-use acknowledgement
  (passed on as research_use_acknowledged, never saved between sessions).
Third-party code and weights are not redistributed (docs/FACE_SWAP_LICENCES.md); when they are missing,
missing_weights_message() says how to fetch them (GHOST_FETCH_HINT: scripts/fetch_models.py --ghost).

Landmarker picker (next to the selector): MediaPipe (CPU, the default) or OpenVINO on a device OpenVINO
finds (CPU, GPU, NPU, plus AUTO); passed to the export as --face-backend / --face-device, saved in the
settings as face_backend / face_device, and written to the sidecars as FaceLandmarker (the device actually
used: a device that does not load falls back to CPU, src/face_estimator.open_face_estimator).

The other methods measured in phase 4 (mosaic8, mosaic16, Gaussian blur, warp, LivePortrait) stay in the
experiment and measurement code (src/segment_export.mask_faces styles, the swap backends in
src/deid_swap.py, the e2e scripts) and in docs/FACE_DEID_TRADEOFF.md; they are not offered in the UI and
the UI export path (segment_export --deid-method) refuses them.

No tkinter here (tested headless); the widgets are in src/deid_ui.py.
"""
from __future__ import annotations

from pathlib import Path

from src.deid_swap import DEFAULT_BACKEND, DEID_STATUS, RESEARCH_ONLY

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = "original"
SHAREABLE = "shareable fallback (default)"
ORIGINAL_LABEL = "original video, not de-identified: view and preview only, stays on this computer"

# key: (group, UI title, status label, shareable, research_only)
METHODS = {
    "fill": ("mask", "Fill (solid colour)", SHAREABLE, True, False),
    "ghost": ("swap", "GHOST face swap", DEID_STATUS, False, True),
}
MASK_METHODS = tuple(k for k, v in METHODS.items() if v[0] == "mask")      # ("fill",)
SWAP_METHODS = tuple(k for k, v in METHODS.items() if v[0] == "swap")      # ("ghost",)
DEFAULT_METHOD = "fill"                  # overall default: the shareable fallback
DEFAULT_SWAP = DEFAULT_BACKEND           # "ghost": the only swap offered in the UI
CHOICES = (ORIGINAL,) + tuple(METHODS)   # selector: View original | De-identify fill / ghost
assert DEFAULT_SWAP == "ghost" and SWAP_METHODS == (DEFAULT_SWAP,) and DEFAULT_SWAP in RESEARCH_ONLY

FACE_MODEL_HINT = "python scripts/fetch_models.py --face --skip-data --skip-vsvig"
LICENCES_DOC = "docs/FACE_SWAP_LICENCES.md"
GHOST_FETCH_HINT = "python scripts/fetch_models.py --ghost"
OFFICIAL = {
    "ghost": ("run " + GHOST_FETCH_HINT + " (clones https://github.com/ai-forever/ghost at the pinned commit and "
              "downloads its release files, sha256-checked, after you accept GHOST's licence terms; pins in "
              + LICENCES_DOC + "); it writes third_party_local/ghost_models.json with the local paths"),
}

# ── landmarker picker ────────────────────────────────────────
LANDMARKER_BACKENDS = {"mediapipe": "MediaPipe", "openvino": "OpenVINO"}
DEFAULT_LANDMARKER = ("mediapipe", "CPU")
OV_DEVICE_ORDER = ("CPU", "GPU", "NPU")


def openvino_devices(available=None) -> list[str]:
    """Devices the picker offers for OpenVINO: CPU / GPU / NPU that OpenVINO finds (GPU.0, GPU.1 -> GPU),
    plus AUTO when at least one was found. `available` is openvino.Core().available_devices (read here
    when None; [] when OpenVINO is not installed or cannot start)."""
    if available is None:
        try:
            import openvino as ov
            available = ov.Core().available_devices
        except Exception:  # noqa: BLE001
            available = []
    found = {str(d).split(".")[0].upper() for d in available}
    out = [d for d in OV_DEVICE_ORDER if d in found]
    return out + ["AUTO"] if out else []


def landmarker_devices(backend: str, ov_devices) -> list[str]:
    """Device choices for a backend: MediaPipe runs on CPU only."""
    return list(ov_devices) if backend == "openvino" else ["CPU"]


def landmarker_choice(settings: dict | None, ov_devices) -> tuple[str, str]:
    """(backend, device) from saved settings, checked against what this computer offers: an unknown
    backend, OpenVINO without devices, or a device not found here -> the default / CPU."""
    st = settings or {}
    b = st.get("face_backend", DEFAULT_LANDMARKER[0])
    d = str(st.get("face_device", DEFAULT_LANDMARKER[1])).upper()
    if b not in LANDMARKER_BACKENDS or (b == "openvino" and not ov_devices):
        return DEFAULT_LANDMARKER
    devs = landmarker_devices(b, ov_devices)
    return b, (d if d in devs else ("CPU" if "CPU" in devs else devs[0]))


def probe_landmarker(backend: str, device: str, models_dir: Path | None = None, opener=None) -> tuple[str, str]:
    """Load the landmarker once on (backend, device): returns (device actually used, note). A device that
    does not load or compile falls back to CPU with a note; without the model file nothing is checked."""
    if backend != "openvino" or device == "CPU":
        return "CPU", ""
    p = Path(models_dir or ROOT / "models") / "face_landmarker.task"
    if opener is None:
        if not p.exists():
            return device, ""
        from src.face_estimator import open_face_estimator as opener
    try:
        est, note = opener(model_path=str(p), backend=backend, device=device)
    except Exception as e:  # noqa: BLE001  (CPU failed too)
        return "CPU", f"OpenVINO did not load ({e}); export will try CPU"
    used = getattr(est, "device", device)
    try:
        est.close()
    except Exception:  # noqa: BLE001
        pass
    return used, note


def check_choice(choice: str) -> str:
    if choice not in CHOICES:
        raise ValueError(f"unknown face view / de-identification choice {choice!r} (UI offers: {', '.join(CHOICES)})")
    return choice


def title(method: str) -> str:
    return "Original" if method == ORIGINAL else METHODS[method][1]


def label(method: str) -> str:
    """The status label shown next to the selector and written to the sidecar."""
    return ORIGINAL_LABEL if method == ORIGINAL else METHODS[method][2]


def is_shareable(method: str) -> bool:
    return method in METHODS and METHODS[method][3]


def needs_ack(method: str) -> bool:
    return method in METHODS and METHODS[method][4]


def label_text(method: str) -> str:
    """One line for the UI: the label plus what it implies."""
    s = label(method)
    if method == ORIGINAL:
        return s
    if method == "fill":
        return (s + ": the only method that may be exported for sharing; every fill export is checked with YuNet "
                "and is not marked shareable if a face is still found in any frame")
    return s + "; research use only: output stays on this computer, never shared"


def missing_weights_message(method: str, problems=()) -> str:
    """Clear message for missing models: what is missing and the official place to fetch it from."""
    head = f"{title(method)}: models not available on this computer"
    lines = [head] + [f"  - {p}" for p in problems]
    if method in SWAP_METHODS:
        lines.append("This project does not redistribute third-party code or weights. Fetch them yourself from the "
                     f"official sources and accept their terms ({LICENCES_DOC}): {OFFICIAL[method]}.")
        lines.append("Then point a swap configuration JSON at them (format: src/deid_swap.py) and choose it as "
                     "'Swap config'.")
        lines.append("No GHOST files? Choose Fill (the shareable fallback) instead.")
    else:
        lines.append(f"Fetch the MediaPipe face model with: {FACE_MODEL_HINT}")
    return "\n".join(lines)


def face_model_problems(models_dir: Path | None = None) -> list[str]:
    p = Path(models_dir or ROOT / "models") / "face_landmarker.task"
    return [] if p.exists() else [f"{p} not found"]


def fill_check_problems(models_dir: Path | None = None) -> list[str]:
    """[] when the YuNet model for the fill fallback and unmasked-face check is present."""
    from src import fill_guard as G
    p = Path(models_dir or ROOT / "models") / G.MODEL_NAME
    return [] if p.exists() else [G.missing_message(p)]


def swap_overrides(method: str, research_ack: bool) -> dict:
    """Configuration values the UI choice sets: the backend and the research-use acknowledgement."""
    o = {"backend": method}
    if needs_ack(method):
        o["research_use_acknowledged"] = bool(research_ack)
    return o


def weights_problems(method: str, swap_config=None, models_dir: Path | None = None) -> list[str]:
    """Missing files for a choice ([] = ready). GHOST: the configuration's model, code and donor files (the
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
    """Rules for one export request from the UI ([] = allowed): only fill or ghost, only fill shareable,
    ghost only with the research-use acknowledgement and a ghost configuration."""
    if method == ORIGINAL:
        return ["Original is for viewing only and is not exported from here; choose Fill or GHOST to export"]
    if method not in METHODS:
        return [f"{method!r} is not offered for export from the UI (only: {', '.join(METHODS)})"]
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
    """segment_export variants for a method: the image variant (ghost -> "swap") plus points."""
    return ["swap" if method in SWAP_METHODS else method, "points"]


def default_dirs() -> tuple[Path, Path]:
    """(de-identified export root, local raw root) used by the UI: outside the repository."""
    base = Path.home() / "semi-signals-local"
    return base / "export", base / "raw"


def export_cmd(video, method: str, *, python: str, annotations=None, out=None, raw_root=None, swap_config=None,
               research_ack: bool = False, shareable: bool = False, face_preprocess: str = "none",
               mode: str = "face", face_backend: str = "mediapipe", face_device: str = "CPU") -> list[str]:
    """The segment_export command the UI starts for fill or ghost (validated again by segment_export)."""
    if method not in METHODS:
        raise ValueError(validate_request(method)[0])
    d_out, d_raw = default_dirs()
    cmd = [python, "-u", "-m", "src.segment_export", "--input", str(video), "--out", str(out or d_out),
           "--raw-root", str(raw_root or d_raw), "--deid-method", method, "--mode", mode,
           "--face-preprocess", face_preprocess, "--face-backend", face_backend,
           "--face-device", face_device if face_backend == "openvino" else "CPU"]
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
    check_choice(method)
    if method == ORIGINAL:
        after = frame.copy()
    elif method in SWAP_METHODS:
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
    cap = "not de-identified" if method == ORIGINAL else label(method)
    cv2.putText(out, cap, (width + 14, size[1] + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return out


def preview_from_video(video, method: str, t: float | None = None, *, swap_config=None, research_ack=False,
                       models_dir: Path | None = None, face_backend: str = "mediapipe", face_device: str = "CPU"):
    """Preview one frame of `video` (middle frame by default). Raises ValueError with the reason (not
    offered, missing models, research gate) instead of running something the export would refuse."""
    import cv2
    check_choice(method)
    if method != ORIGINAL:
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
    lm = None
    if method != ORIGINAL:
        from src.face_estimator import open_face_estimator
        est, _ = open_face_estimator(model_path=str(Path(models_dir or ROOT / "models") / "face_landmarker.task"),
                                     backend=face_backend, device=face_device)
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
