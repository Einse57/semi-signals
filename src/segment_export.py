"""Per-segment export: raw video (local only) + de-identified video + face landmarks + annotations.

  python -m src.segment_export --input clip.mp4 [--annotations clip.eaf] --out export/deid \\
         --raw-root ~/semi-signals-local/raw [--face-preprocess none|detect|coco-head] [--deid fill,points]

One segment per annotated event (window = clinical onset - --pre-s .. clinical end + --post-s; onset-only
events last --default-len-s), or the whole clip with --whole / without --annotations. Per segment
(file layout and format: src/segment_store.py, PROVISIONAL, not locked):

  raw (under --raw-root, local only, never in the export, gitignored in this repo):
      sub-<id>/beh/..._recording-<rec>_desc-raw_video.mp4 (+ .json), sub-<id>/sub-<id>_scans.tsv (real
      acq_time when --acq-time is given), and the private mapping CSV (segment -> source video, window,
      date shift).
      ..._desc-blur_video.mp4 only with --deid blur: blurred faces for LOCAL REVIEW, NOT FOR SHARING
      (a face detector still finds faces in blurred frames); it goes into the export only with
      --blur-in-export, and is then marked NotForSharing.
  de-identified (under --out):
      ..._desc-fill_video.mp4    (default) every face found painted with a solid colour (whole frame where
                                 the subject's face was not found); faces YuNet finds that the landmarker
                                 missed are filled as padded boxes, and YuNet checks every output frame
                                 (src/fill_guard.py): any face found there, or no YuNet model, means the
                                 sidecar says Shareable false and lists the frames (UnmaskedFaceCheck).
                                 A flagged frame first gets its flagged YuNet boxes filled and is checked
                                 once more; only frames still flagged block sharing
      ..._desc-points_video.mp4  (default) landmarks drawn on a blank canvas
      ..._desc-swap_video.mp4    face swap to a synthetic donor (--deid swap, src/deid_swap.py; the backend,
                                 ghost (default), warp or liveportrait, comes from --swap-config; "ghost" is also
                                 accepted as an alias of the variant). REDUCED RISK, NOT DE-IDENTIFIED: its
                                 sidecar says Deidentified false; ghost and liveportrait output is research
                                 use only (ResearchUseOnly true).
      ..._landmarks.nc (+ .json) movement-layout landmarks, ..._blendshapes.nc
      ..._events.tsv (+ .json)   from the .eaf, times relative to the segment start, no notes
      sub-<id>_scans.tsv         acq_time shifted (random per subject, year <= 1925) or n/a
De-identified here means reduced re-identification risk, not anonymous: faces neither the landmarker nor
YuNet finds are not masked, and landmarks can be biometric identifiers.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from src import segment_store as S
from src import deid_options as O
from src.deid_swap import DEID_STATUS, RESEARCH_ONLY
from src import fill_guard as G

REPO = Path(__file__).resolve().parent.parent
VARIANTS = ("fill", "points", "swap", "blur")       # --deid (CLI)
ALIASES = {"ghost": "swap"}
NOT_FOR_SHARING = ("blur",)
MASK_STYLES = ("fill", "blur")           # variants written with mask_faces (the mosaic / gauss styles are for
                                         # the phase 4 measurements only, not export variants)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Raw + de-identified segment export with face landmarks")
    p.add_argument("--input", required=True, help="video file")
    p.add_argument("--annotations", default=None, help="ELAN .eaf from the annotation tool (editable copy)")
    p.add_argument("--whole", action="store_true", help="one segment for the whole clip")
    p.add_argument("--out", required=True, help="de-identified export root")
    p.add_argument("--raw-root", required=True, help="local raw root (outside --out; gitignored if inside a repo)")
    p.add_argument("--mapping", default=None, help="private mapping CSV (default <raw-root>/private_mapping.csv)")
    p.add_argument("--face-preprocess", default="none", choices=["none", "detect", "coco-head"])
    p.add_argument("--face-model", default=None)
    from src.face_estimator import add_face_backend_args
    add_face_backend_args(p)
    p.add_argument("--deid", default="fill,points",
                   help="comma list of video variants: fill (default; faces painted solid), points (default; "
                        "landmarks only), swap (face swap to a donor; needs --swap-config; backend ghost unless the config names another; alias ghost), blur (LOCAL REVIEW ONLY, NOT FOR "
                        "SHARING: written under --raw-root unless --blur-in-export)")
    p.add_argument("--blur-in-export", action="store_true",
                   help="put the blur variant in the export too (marked NotForSharing in its sidecar)")
    p.add_argument("--swap-config", default=None, help="face-swap configuration JSON for --deid swap (src/deid_swap.py)")
    p.add_argument("--deid-method", default=None, choices=list(O.METHODS),
                   help="the de-identification chosen in the UI (src/deid_options.py): fill (default, the only "
                        "shareable export) or ghost (face swap, reduced risk, not de-identified, research use "
                        "only); replaces --deid with <method>,points and is validated here; nothing else is "
                        "accepted on this path")
    p.add_argument("--research-use-acknowledged", action="store_true",
                   help="with --deid-method ghost: the research-use acknowledgement from the UI; "
                        "passed on as research_use_acknowledged (required, overrides the configuration's value)")
    p.add_argument("--shareable", action="store_true",
                   help="export for sharing: refused unless the method is shareable (only fill)")
    p.add_argument("--mode", default="both", choices=["body", "face", "both"], help="events that make segments")
    p.add_argument("--pre-s", type=float, default=2.0)
    p.add_argument("--post-s", type=float, default=2.0)
    p.add_argument("--default-len-s", type=float, default=5.0)
    p.add_argument("--task", default="seizure", help="BIDS task label")
    p.add_argument("--recording", default="face", help="BIDS recording label (camera / view)")
    p.add_argument("--sub", default=None, help="pseudonymous subject label (default: random; reused from the mapping "
                                               "for the same source video)")
    p.add_argument("--acq-time", default=None, help="recording start, ISO 8601 (for EEG alignment via scans.tsv)")
    p.add_argument("--hed", action="store_true", help="add an (empty) HED column for HED-SCORE tags")
    p.add_argument("--missing-face", default="frame", choices=["frame", "none"],
                   help="fill/blur/swap frames without the subject's face: mask the whole frame (default) or leave them")
    p.add_argument("--no-bystanders", action="store_true",
                   help="mask only the subject's face (default: also every other face the landmarker finds)")
    p.add_argument("--device", default="cpu", help="for --face-preprocess coco-head (body pose)")
    p.add_argument("--yunet-model", default=str(REPO / "models" / G.MODEL_NAME),
                   help="YuNet face detector for the fill fallback and the unmasked-face check (fill only)")
    p.add_argument("--yunet-score", type=float, default=G.DEFAULT_SCORE, help="YuNet score threshold")
    p.add_argument("--fallback-pad", type=float, default=G.DEFAULT_PAD,
                   help="padding of YuNet fallback boxes, fraction of the box size on each side")
    return p.parse_args(argv)


# ── de-identified renders ────────────────────────────────────
def landmark_mask(shape, faces, grow=0.25):
    """uint8 mask of the grown convex hulls of the landmark faces."""
    mask = np.zeros(shape[:2], np.uint8)
    for f in faces:
        p = np.asarray(f, np.float32)[:, :2]
        c = p.mean(0)
        cv2.fillConvexPoly(mask, cv2.convexHull(((p - c) * (1 + grow) + c).astype(np.int32)), 255)
    return mask


def mask_faces(frame, lm, missing="frame", grow=0.25, others=(), style="fill", boxes=()):
    """Mask the convex hull (grown by `grow` x face size) of the subject's face landmarks `lm` and of
    every other face in `others` (bystanders): style "fill" paints it with its mean colour, "blur"
    pixelates and blurs it (local review only). No subject face: the whole frame (missing="frame").
    `boxes` (x0, y0, x1, y1): extra regions for style "fill" (YuNet fallback, src/fill_guard.py), painted
    together with the hulls."""
    boxes = list(boxes) if style == "fill" else []
    out = frame.copy()
    h, w = frame.shape[:2]
    k = max(31, (max(h, w) // 8) | 1)
    if lm is None or not np.isfinite(lm).all():
        if missing == "frame":
            return _whole(out, k, style)
        faces = list(others)
    else:
        faces = [lm] + list(others)
    if not faces and not boxes:
        return out
    mask = np.zeros((h, w), np.uint8)
    size = 8.0
    for f in faces:
        p = np.asarray(f, np.float32)[:, :2]
        c = p.mean(0)
        size = max(size, float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1]))))
        cv2.fillConvexPoly(mask, cv2.convexHull(((p - c) * (1 + grow) + c).astype(np.int32)), 255)
    for x0, y0, x1, y1 in boxes:
        mask[y0:y1, x0:x1] = 255
    if not mask.any():
        return _whole(out, k, style)
    if style == "fill":                  # solid mean colour of the face area
        out[mask > 0] = out[mask > 0].reshape(-1, 3).mean(0).astype(np.uint8)
        return out
    if style in BASELINE_STYLES:         # mosaic / Gaussian blur, sized per face (measured baselines, not shareable)
        src = frame.copy()
        for f in faces:
            m = np.zeros((h, w), np.uint8)
            p = np.asarray(f, np.float32)[:, :2]
            c = p.mean(0)
            cv2.fillConvexPoly(m, cv2.convexHull(((p - c) * (1 + grow) + c).astype(np.int32)), 255)
            if m.any():
                _baseline_region(out, src, m, max(8.0, float(np.ptp(p[:, 0]))), style)
        return out
    kk = max(15, int(size * 0.6) | 1)
    blurred = cv2.GaussianBlur(cv2.resize(cv2.resize(out, (max(1, w // 8), max(1, h // 8))), (w, h),
                                          interpolation=cv2.INTER_NEAREST), (kk, kk), 0)
    out[mask > 0] = blurred[mask > 0]
    return out


# Measured baselines (local measurements only; never the shareable default). Sizes are relative to the face
# bounding-box width W = horizontal extent of that face's landmarks (before the 25 % hull growth):
#   mosaic8  : square blocks of round(W / 8) px (about 8 blocks across the face, 10 across the grown hull)
#   mosaic16 : square blocks of round(W / 16) px (about 16 blocks across the face)
#   gauss    : Gaussian blur, sigma = W / 6 px, kernel 2 * ceil(3 * sigma) + 1 px (square)
# Blocks are anchored at the top-left of the grown hull's bounding box; only pixels inside the grown hull change.
# Frames without a subject face (missing="frame") use the whole frame as the region, W = frame width.
MOSAIC_BLOCKS = {"mosaic8": 8, "mosaic16": 16}
GAUSS_SIGMA_FRAC = 1 / 6
BASELINE_STYLES = ("mosaic8", "mosaic16", "gauss")


def baseline_params(face_w, style):
    """(block px, None) for mosaic styles, (None, (sigma px, kernel px)) for gauss."""
    if style in MOSAIC_BLOCKS:
        return max(2, int(round(face_w / MOSAIC_BLOCKS[style]))), None
    s = max(1.0, face_w * GAUSS_SIGMA_FRAC)
    return None, (s, 2 * int(np.ceil(3 * s)) + 1)


def _baseline_region(out, src, mask, face_w, style):
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    b, g = baseline_params(face_w, style)
    if b is not None:
        crop = src[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        nx, ny = -(-cw // b), -(-ch // b)
        pad = cv2.copyMakeBorder(crop, 0, ny * b - ch, 0, nx * b - cw, cv2.BORDER_REPLICATE)
        small = cv2.resize(pad, (nx, ny), interpolation=cv2.INTER_AREA)
        proc = cv2.resize(small, (nx * b, ny * b), interpolation=cv2.INTER_NEAREST)[:ch, :cw]
    else:
        s, k = g
        r = k // 2
        H, W = src.shape[:2]
        a0, a1, c0, c1 = max(0, y0 - r), min(H, y1 + r), max(0, x0 - r), min(W, x1 + r)
        proc = cv2.GaussianBlur(src[a0:a1, c0:c1], (k, k), s, borderType=cv2.BORDER_REFLECT)[y0 - a0:y1 - a0, x0 - c0:x1 - c0]
    sel = mask[y0:y1, x0:x1] > 0
    out[y0:y1, x0:x1][sel] = proc[sel]


def _whole(img, k, style):
    if style == "fill":
        return np.full_like(img, img.reshape(-1, 3).mean(0).astype(np.uint8))
    if style in BASELINE_STYLES:
        out = img.copy()
        _baseline_region(out, img, np.full(img.shape[:2], 255, np.uint8), float(img.shape[1]), style)
        return out
    return cv2.GaussianBlur(img, (k, k), 0)


def blur_face(frame, lm, missing="frame", grow=0.25, others=()):
    """Blurred faces (local review only, not for sharing)."""
    return mask_faces(frame, lm, missing, grow, others, style="blur")


def other_faces(multi, frame, subject_lm, max_tiles=True):
    """Faces other than the subject: the multi-face landmarker on the whole frame and on four
    overlapping tiles (small faces), duplicates and the subject removed (centre within half a face)."""
    h, w = frame.shape[:2]
    found = list(multi.all_faces(frame))
    if max_tiles:
        tw, th = int(w * 0.6), int(h * 0.6)
        for y0 in (0, h - th):
            for x0 in (0, w - tw):
                found += multi.all_faces(frame[y0:y0 + th, x0:x0 + tw], offset=(x0, y0))
    keep, centres = [], []
    if subject_lm is not None and np.isfinite(subject_lm).all():
        centres.append((subject_lm[:468, :2].mean(0), np.ptp(subject_lm[:468, 0])))
    for f in found:
        c, size = f[:468, :2].mean(0), np.ptp(f[:468, 0])
        if any(np.linalg.norm(c - c2) < 0.5 * max(size, s2) for c2, s2 in centres):
            continue
        centres.append((c, size))
        keep.append(f)
    return keep


def points_frame(shape, lm):
    """Landmarks on a blank canvas of the frame size (mesh + contours)."""
    from src.display import draw_face_mesh
    canvas = np.zeros(shape, np.uint8)
    if lm is not None and np.isfinite(lm).all():
        draw_face_mesh(canvas, lm, mesh=True)
    return canvas


# ── helpers ──────────────────────────────────────────────────
def check_raw_location(raw_root: Path, out: Path):
    S.check_outside(raw_root, out, "--raw-root")
    S.check_outside(out, raw_root, "--out")
    try:
        inside = REPO in raw_root.resolve().parents or raw_root.resolve() == REPO
    except OSError:
        inside = False
    if inside:
        probe = raw_root / "probe_video.mp4"
        r = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", str(probe)], capture_output=True)
        if r.returncode != 0:
            raise SystemExit(f"--raw-root {raw_root} is inside the repository and not gitignored; "
                             "use a folder outside the repo (or one listed in .gitignore)")


def segments(a, duration):
    if a.whole or not a.annotations:
        return [dict(start=0.0, end=duration, events=[])]
    from src.annotation import read_eaf
    evs = [e for e in read_eaf(a.annotations)["events"] if a.mode == "both" or e["mode"] == a.mode]
    out = []
    for e in sorted(evs, key=lambda e: e["clinical_onset_s"]):
        on = e["clinical_onset_s"]
        end = e.get("clinical_end_s")
        end = end if end is not None and end - on > S.ONSET_ONLY_S else on + a.default_len_s
        out.append(dict(start=max(0.0, on - a.pre_s), end=min(duration, end + a.post_s), events=[e]))
    return out


def load_mapping(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


MAP_COLS = ["sub", "run", "segment_start_s", "segment_end_s", "source_video", "source_sha256", "annotations",
            "acq_time", "date_shift_days", "raw_video", "exported"]


def append_mapping(path: Path, rows):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MAP_COLS)
        if new:
            w.writeheader()
        w.writerows(rows)


def face_backend_kwargs(a) -> dict:
    """--face-backend / --face-device as FaceEstimator keyword arguments (defaults when absent)."""
    return dict(backend=getattr(a, "face_backend", "mediapipe"), device=getattr(a, "face_device", "CPU"))


def landmarker_used(a, est) -> dict:
    """Sidecar FaceLandmarker: the backend and the device the landmarker actually ran on (after any CPU
    fallback), the device asked for, and the fallback note."""
    req = face_backend_kwargs(a)
    backend = getattr(est, "backend", req["backend"])
    dev = getattr(est, "device", req["device"] if backend == "openvino" else "CPU")
    d = dict(Backend=backend, Device=dev, DeviceRequested=req["device"] if backend == "openvino" else "CPU")
    if getattr(a, "face_note", ""):
        d["Note"] = a.face_note
    return d


def make_preprocessor(a):
    from src.face_preprocess import FacePreprocessor
    from src.face_estimator import open_face_estimator
    est, note = open_face_estimator(model_path=a.face_model, transform=True, **face_backend_kwargs(a))
    a.face_note = note
    if note:
        print("[face] " + note, flush=True)
    pose = det = None
    if a.face_preprocess == "coco-head":
        from src.detector import PersonDetector
        from src.pose_estimator import PoseEstimator
        det, pose = PersonDetector(device=a.device), PoseEstimator(device=a.device)
    return est, FacePreprocessor(est, a.face_preprocess), det, pose


def run_face(pre, det, pose, frame):
    if pre.mode != "coco-head":
        return pre.run(frame)
    boxes = sorted(det.detect(frame), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    if not boxes:
        return None
    best = pose.estimate(frame, boxes[:1])[0]          # largest person
    return pre.run(frame, best["keypoints"], best["scores"], best["bbox"])


def writer(path, fps, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


def resolve_variants(a):
    """(variants, swap overrides) from --deid or --deid-method, with the UI rules checked again here
    (src/deid_options.validate_request): known method, research-use acknowledgement, shareable only for fill,
    swap configuration backend. Exits with the reason otherwise."""
    if a.deid_method:
        m = a.deid_method
        backend = None
        if m in O.SWAP_METHODS and a.swap_config:
            from src.deid_swap import load_config
            try:
                backend = load_config(a.swap_config).get("backend")
            except (OSError, ValueError, RuntimeError):
                backend = None
        err = O.validate_request(m, research_ack=a.research_use_acknowledged, shareable=a.shareable,
                                 swap_config_backend=backend)
        if a.blur_in_export:
            err.append("--blur-in-export does not apply to --deid-method (only fill or ghost)")
        if err:
            raise SystemExit("--deid-method " + m + ": " + "; ".join(err))
        return O.export_variants(m), (O.swap_overrides(m, a.research_use_acknowledged) if m in O.SWAP_METHODS else None)
    variants = list(dict.fromkeys(ALIASES.get(v.strip(), v.strip()) for v in a.deid.split(",") if v.strip()))
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"--deid: unknown variant {v!r} ({', '.join(VARIANTS)})")
    if a.shareable:
        bad = [v for v in variants if v != "points" and not (v in O.METHODS and O.is_shareable(v))]
        if bad:
            raise SystemExit(f"--shareable: {', '.join(bad)} cannot be exported for sharing; only fill is shareable")
    if a.research_use_acknowledged:
        raise SystemExit("--research-use-acknowledged goes with --deid-method ghost")
    return variants, None


# ── main ─────────────────────────────────────────────────────
def main(argv=None):
    a = parse_args(argv)
    from src.annotation import sha256_file
    from src.face_estimator import BLENDSHAPE_NAMES
    video = Path(a.input).resolve()
    out, raw_root = Path(a.out).resolve(), Path(a.raw_root).expanduser().resolve()
    mapping = Path(a.mapping).resolve() if a.mapping else raw_root / "private_mapping.csv"
    check_raw_location(raw_root, out)
    S.check_outside(mapping, out, "--mapping")
    variants, overrides = resolve_variants(a)
    swapper = None
    if "swap" in variants:
        from src.deid_swap import SwapNotConfigured, make_swapper
        try:
            swapper = make_swapper(a.swap_config, overrides)
        except (SwapNotConfigured, ValueError, OSError) as e:
            if a.deid_method:
                raise SystemExit(O.missing_weights_message(a.deid_method, [str(e)]))
            raise SystemExit(f"--deid swap: {e}")
    elif a.deid_method and not a.face_model:
        probs = O.face_model_problems(REPO / "models")
        if probs:
            raise SystemExit(O.missing_weights_message(a.deid_method, probs))
    export_variants = [v for v in variants if not (v == "blur" and not a.blur_in_export)]
    local_variants = [v for v in variants if v not in export_variants]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n / fps
    sha = sha256_file(video)
    old = load_mapping(mapping)
    prev = [r for r in old if r["source_sha256"] == sha]
    sub = a.sub or (prev[0]["sub"] if prev else S.new_pseudonym(4))
    same_sub = [r for r in old if r["sub"] == sub]
    acq = _dt.datetime.fromisoformat(a.acq_time) if a.acq_time else None
    shift = None
    if acq is not None:
        known = [r["date_shift_days"] for r in same_sub if r.get("date_shift_days") not in (None, "", "n/a")]
        shift = int(known[0]) if known else S.random_shift_days(acq)
    run0 = max([int(r["run"]) for r in same_sub] or [0])
    est, pre, det, pose = make_preprocessor(a)
    lmk = landmarker_used(a, est)
    multi = None
    if any(v in variants for v in MASK_STYLES + ("swap",)) and not a.no_bystanders:
        from src.face_estimator import FaceEstimator
        multi = FaceEstimator(model_path=a.face_model, num_faces=8, blendshapes=False,
                              **dict(face_backend_kwargs(a), device=lmk["Device"]))   # same device as the subject
    guard = make_fill_guard(a) if "fill" in variants else None
    if guard is not None and not guard.available:
        print("[fill check] " + guard.problem, flush=True)
    blocked = []
    S.dataset_description(out, deidentified=True)
    S.dataset_description(raw_root, name="semi-signals segments (raw, local only)", deidentified=False)
    rows = []
    try:
        for k, seg in enumerate(segments(a, duration), 1):
            run = run0 + k
            P = S.SegmentPaths(out, sub, a.task, run, a.recording)
            R = S.SegmentPaths(raw_root, sub, a.task, run, a.recording)
            f0, f1 = int(round(seg["start"] * fps)), min(n, int(round(seg["end"] * fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            pre.reset()
            if swapper is not None and hasattr(swapper, "reset"):
                swapper.reset()                                  # per-segment face tracks (warp geometry)
            if guard is not None:
                guard.reset()
            ws = {"raw": writer(R.video("raw"), fps, (W, H))}
            for v in export_variants:
                ws[v] = writer(P.video(v), fps, (W, H))
            for v in local_variants:
                ws[v] = writer(R.video(v), fps, (W, H))
            T, LM, BS, SC, ROLL, n_other = [], [], [], [], [], []
            for fi in range(f0, f1):
                ok, frame = cap.read()
                if not ok:
                    break
                r = run_face(pre, det, pose, frame)
                lm = r["landmarks"] if r is not None else None
                T.append(fi / fps - seg["start"])
                LM.append(lm if lm is not None else np.full((478, 3), np.nan, np.float32))
                BS.append(r["blendshapes"] if r is not None else np.full(52, np.nan, np.float32))
                SC.append(float(r["score"]) if r is not None else np.nan)
                ROLL.append(float(r.get("roll_deg", np.nan)) if r is not None else np.nan)
                ws["raw"].write(frame)
                oth = None
                if multi is not None:
                    oth = other_faces(multi, frame, lm)
                    n_other.append(len(oth))
                for v in MASK_STYLES:
                    if v not in ws:
                        continue
                    if v == "fill" and guard is not None:
                        whole = (lm is None or not np.isfinite(lm).all()) and a.missing_face == "frame"
                        faces = [] if whole else ([lm] if lm is not None else []) + list(oth or [])
                        bx = [] if whole else guard.boxes(frame, landmark_mask(frame.shape, faces))
                        o = mask_faces(frame, lm, a.missing_face, others=oth or [], style=v, boxes=bx)
                        o, _ = guard.check_fix(o, fi - f0, lambda b2, bx=bx: mask_faces(
                            frame, lm, a.missing_face, others=oth or [], style=v, boxes=bx + b2))
                        ws[v].write(o)
                    else:
                        ws[v].write(mask_faces(frame, lm, a.missing_face, others=oth or [], style=v))
                if "swap" in ws:
                    faces = ([lm] if lm is not None else []) + list(oth or [])
                    g = swapper.swap(frame, faces) if (lm is not None or a.missing_face == "none") else None
                    ws["swap"].write(g if g is not None else mask_faces(frame, None, "frame", style="fill"))
                if "points" in ws:
                    ws["points"].write(points_frame(frame.shape, lm))
            for w_ in ws.values():
                w_.release()
            found = int(np.isfinite(SC).sum())
            common = dict(Deidentified=True, AnnotationMode=",".join(sorted({e["mode"] for e in seg["events"]})) or "none",
                          FacePreprocess=a.face_preprocess, FaceLandmarker=lmk, PrivateMappingRef=mapping.name,
                          Provisional=True)
            bys = "" if multi is None else (" and every other face the landmarker found (whole frame + 4 tiles); "
                                            "faces it misses are not masked, check before sharing")
            miss = ("; frames without the subject's face masked whole" if a.missing_face == "frame"
                    else "; frames without the subject's face NOT masked")
            fb = ("; faces YuNet found outside those areas filled as padded boxes; every output frame checked with "
                  "YuNet (UnmaskedFaceCheck)" if guard is not None and guard.available else
                  "; YuNet not available: no fallback boxes and no unmasked-face check")
            descs = dict(fill="subject face area (landmark hull grown 25%)" + bys + " painted with its mean colour" + miss
                         + (fb if guard is not None else ""),
                         blur="LOCAL REVIEW ONLY, NOT FOR SHARING: face areas pixelated and blurred; face detectors "
                              "still find faces in blurred frames" + miss,
                         swap="REDUCED RISK, NOT DE-IDENTIFIED: faces replaced by a donor face (face swap, "
                              + (swapper.name if swapper else "-") + "); outline, hair, skin colour, pose and "
                              "expression remain" + miss,
                         points="face landmarks drawn on a blank canvas; no image content")
            for v in variants:
                local = v in local_variants
                extra = dict(common, Variant=v, VariantDescription=descs[v], FramesWithFace=found)
                if v in NOT_FOR_SHARING:
                    extra.update(NotForSharing=True, Deidentified=False)
                if v in MASK_STYLES + ("swap",):
                    extra["BystanderFacesMasked"] = int(np.sum(n_other)) if multi is not None else None
                if v in O.METHODS:
                    extra["DeidentificationStatus"] = O.label(v)
                    extra["Shareable"] = O.is_shareable(v)
                    if not O.is_shareable(v):
                        extra["Deidentified"] = False
                if v == "fill" and guard is not None:
                    chk = guard.result()
                    extra["UnmaskedFaceCheck"] = chk
                    if not chk["Clean"]:
                        extra["Shareable"] = False
                        extra["ShareableBlockedReason"] = chk["Reason"]
                        if chk["Ran"]:
                            extra["Deidentified"] = False
                if v == "swap":
                    extra["Shareable"] = False
                if v == "swap":
                    extra.update(Deidentified=False, DeidentificationStatus=DEID_STATUS)
                if v == "swap" and swapper is not None:
                    extra["SwapBackend"] = swapper.name
                    if swapper.name in RESEARCH_ONLY:
                        extra["ResearchUseOnly"] = True
                    extra["SwapDonorLicence"] = getattr(swapper, "donor_licence", None)
                vp = R if local else P
                S.write_json(vp.video_json(v), S.video_sidecar(vp.video(v), extra))
            S.write_json(R.video_json("raw"), S.video_sidecar(R.video("raw"), dict(Deidentified=False, Variant="raw",
                                                                                  Provisional=True)))
            lm_path = S.write_landmarks(P.landmarks, T, np.stack(LM) if LM else np.zeros((0, 478, 3), np.float32),
                                        np.asarray(SC, np.float32), fps=fps,
                                        attrs=dict(face_preprocess=a.face_preprocess))
            bs_path = S.write_blendshapes(P.blendshapes, T, np.stack(BS) if BS else np.zeros((0, 52), np.float32),
                                          BLENDSHAPE_NAMES)
            S.write_json(P.landmarks_json, S.landmarks_sidecar(
                coords="image: origin top-left of the segment video frame, x right, y down; z = MediaPipe relative "
                       "depth (smaller = nearer the camera) on the x scale",
                units="pixels", deidentified=True, mode=common["AnnotationMode"], preprocess=a.face_preprocess,
                mapping_ref=mapping.name, blendshape_file=bs_path.name,
                extra=dict(FramesWithFace=found, Frames=len(T), FaceLandmarker=lmk,
                           RollRemovedDeg=None if a.face_preprocess != "detect" else
                           float(np.nanmedian(ROLL)) if np.isfinite(ROLL).any() else None)))
            if a.annotations:
                S.eaf_to_events(a.annotations, P.events, t0=seg["start"], t_end=seg["end"], hed=a.hed)
            else:
                S.write_events_tsv(P.events, [], hed=a.hed)
                S.write_json(P.events_json, S.events_sidecar(hed=a.hed))
            seg_acq = acq + _dt.timedelta(seconds=seg["start"]) if acq is not None else None
            files = [P.video(v) for v in export_variants] + [lm_path, bs_path]
            S.write_scans_tsv(P.scans, [dict(filename=P.rel(f), acq_time=S.shift_acq_time(seg_acq, shift)
                                             if seg_acq else S.NA) for f in files])
            S.write_scans_tsv(R.scans, [dict(filename=R.rel(R.video(v)),
                                             acq_time=seg_acq.isoformat(timespec="milliseconds") if seg_acq else S.NA)
                                        for v in ["raw"] + local_variants])
            rows.append(dict(sub=sub, run=run, segment_start_s=round(seg["start"], 3), segment_end_s=round(seg["end"], 3),
                             source_video=str(video), source_sha256=sha,
                             annotations=str(Path(a.annotations).resolve()) if a.annotations else "",
                             acq_time=a.acq_time or "", date_shift_days="" if shift is None else shift,
                             raw_video=str(R.video("raw")),
                             exported=_dt.datetime.now().astimezone().isoformat(timespec="seconds")))
            if guard is not None:
                print(guard.summary(f"sub-{sub} run-{run:02d}"), flush=True)
                if not guard.result()["Clean"]:
                    blocked.append(f"run-{run:02d}")
            print(f"[segment] sub-{sub} run-{run:02d}: {seg['start']:.2f}-{seg['end']:.2f} s, {len(T)} frames, "
                  f"face in {found}; {', '.join(export_variants)}" + (f"; local: raw, {', '.join(local_variants)}" if local_variants else "; local: raw"), flush=True)
    finally:
        cap.release()
        est.close()
        if multi is not None:
            multi.close()
        if swapper is not None:
            swapper.close()
    append_mapping(mapping, rows)
    print(f"[segment] {len(rows)} segment(s); de-identified: {out}; raw (local only): {raw_root}; mapping: {mapping}")
    if blocked:
        print(f"[fill check] NOT SHAREABLE: fill export of {', '.join(blocked)} is marked Shareable false "
              "(see UnmaskedFaceCheck in the fill sidecar)", flush=True)
        if a.shareable:
            return 3
    return 0


def make_fill_guard(a):
    """The fill fallback and unmasked-face check (src/fill_guard.py); replaced in tests."""
    return G.FillGuard(a.yunet_model, score=a.yunet_score, pad=a.fallback_pad)


if __name__ == "__main__":
    sys.exit(main())
