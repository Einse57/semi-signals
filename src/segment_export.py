"""Per-segment export: raw video (local only) + de-identified video + face landmarks + annotations.

  python -m src.segment_export --input clip.mp4 [--annotations clip.eaf] --out export/deid \\
         --raw-root ~/semi-signals-local/raw [--face-preprocess none|detect|coco-head] [--deid blur,points]

One segment per annotated event (window = clinical onset - --pre-s .. clinical end + --post-s; onset-only
events last --default-len-s), or the whole clip with --whole / without --annotations. Per segment
(file layout and format: src/segment_store.py, PROVISIONAL, not locked):

  raw (under --raw-root, local only, never in the export, gitignored in this repo):
      sub-<id>/beh/..._recording-<rec>_desc-raw_video.mp4 (+ .json), sub-<id>/sub-<id>_scans.tsv (real
      acq_time when --acq-time is given), and the private mapping CSV (segment -> source video, window,
      date shift).
  de-identified (under --out):
      ..._desc-blur_video.mp4    face area blurred (whole frame blurred where no face was found)
      ..._desc-points_video.mp4  landmarks drawn on a blank canvas
      ..._landmarks.nc (+ .json) movement-layout landmarks, ..._blendshapes.nc
      ..._events.tsv (+ .json)   from the .eaf, times relative to the segment start, no notes
      sub-<id>_scans.tsv         acq_time shifted (random per subject, year <= 1925) or n/a
De-identified here means reduced re-identification risk, not anonymous: blur can fail on frames where
the face is not found by the landmarker's detector in time, and landmarks can be biometric identifiers.
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

REPO = Path(__file__).resolve().parent.parent


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
    p.add_argument("--deid", default="blur,points", help="comma list of de-identified video variants: blur, points")
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
    p.add_argument("--blur-missing", default="frame", choices=["frame", "none"],
                   help="frames without a face: blur the whole frame (default) or leave them")
    p.add_argument("--blur-style", default="blur", choices=["blur", "fill"],
                   help="desc-blur variant: pixelate + Gaussian blur (default) or a solid fill of the face area")
    p.add_argument("--no-bystander-blur", action="store_true",
                   help="blur only the subject's face (default: also blur every other face the landmarker finds)")
    p.add_argument("--device", default="cpu", help="for --face-preprocess coco-head (body pose)")
    return p.parse_args(argv)


# ── de-identified renders ────────────────────────────────────
def blur_face(frame, lm, missing="frame", grow=0.25, others=(), style="blur"):
    """Blur the convex hull (grown by `grow` x face size) of the subject's face landmarks `lm` and of
    every other face in `others` (bystanders). No subject face: the whole frame (missing="frame")."""
    out = frame.copy()
    h, w = frame.shape[:2]
    k = max(31, (max(h, w) // 8) | 1)
    if lm is None or not np.isfinite(lm).all():
        if missing == "frame":
            return cv2.GaussianBlur(out, (k, k), 0)
        faces = list(others)
    else:
        faces = [lm] + list(others)
    if not faces:
        return out
    mask = np.zeros((h, w), np.uint8)
    size = 8.0
    for f in faces:
        p = np.asarray(f, np.float32)[:, :2]
        c = p.mean(0)
        size = max(size, float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1]))))
        cv2.fillConvexPoly(mask, cv2.convexHull(((p - c) * (1 + grow) + c).astype(np.int32)), 255)
    if not mask.any():
        return cv2.GaussianBlur(out, (k, k), 0)
    if style == "fill":                  # solid mean colour of the face area
        out[mask > 0] = out[mask > 0].reshape(-1, 3).mean(0).astype(np.uint8)
        return out
    kk = max(15, int(size * 0.6) | 1)
    blurred = cv2.GaussianBlur(cv2.resize(cv2.resize(out, (max(1, w // 8), max(1, h // 8))), (w, h),
                                          interpolation=cv2.INTER_NEAREST), (kk, kk), 0)
    out[mask > 0] = blurred[mask > 0]
    return out


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


def make_preprocessor(a):
    from src.face_estimator import FaceEstimator
    from src.face_preprocess import FacePreprocessor
    est = FaceEstimator(model_path=a.face_model, transform=True)
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
    variants = [v.strip() for v in a.deid.split(",") if v.strip()]
    for v in variants:
        if v not in ("blur", "points"):
            raise SystemExit(f"--deid: unknown variant {v!r} (blur, points)")
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
    multi = None
    if "blur" in variants and not a.no_bystander_blur:
        from src.face_estimator import FaceEstimator
        multi = FaceEstimator(model_path=a.face_model, num_faces=8, blendshapes=False)
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
            ws = {"raw": writer(R.video("raw"), fps, (W, H))}
            for v in variants:
                ws[v] = writer(P.video(v), fps, (W, H))
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
                if "blur" in ws:
                    oth = other_faces(multi, frame, lm) if multi is not None else []
                    n_other.append(len(oth))
                    ws["blur"].write(blur_face(frame, lm, a.blur_missing, others=oth, style=a.blur_style))
                if "points" in ws:
                    ws["points"].write(points_frame(frame.shape, lm))
            for w_ in ws.values():
                w_.release()
            found = int(np.isfinite(SC).sum())
            common = dict(Deidentified=True, AnnotationMode=",".join(sorted({e["mode"] for e in seg["events"]})) or "none",
                          FacePreprocess=a.face_preprocess, PrivateMappingRef=mapping.name, Provisional=True)
            for v in variants:
                desc = dict(blur=("subject face area blurred (convex hull of the landmarks, grown 25%)"
                                  + ("" if multi is None else " and every other face the landmarker found (whole frame "
                                     "+ 4 tiles); faces it misses are NOT blurred, check before sharing")
                                  + ("; frames without the subject's face blurred whole" if a.blur_missing == "frame"
                                     else "; frames without the subject's face NOT blurred")),
                            points="face landmarks drawn on a blank canvas; no image content")[v]
                extra = dict(common, Variant=v, VariantDescription=desc, FramesWithFace=found)
                if v == "blur":
                    extra["BlurStyle"] = a.blur_style
                    extra["BystanderFacesBlurred"] = int(np.sum(n_other)) if multi is not None else None
                S.write_json(P.video_json(v), S.video_sidecar(P.video(v), extra))
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
                extra=dict(FramesWithFace=found, Frames=len(T),
                           RollRemovedDeg=None if a.face_preprocess != "detect" else
                           float(np.nanmedian(ROLL)) if np.isfinite(ROLL).any() else None)))
            if a.annotations:
                S.eaf_to_events(a.annotations, P.events, t0=seg["start"], t_end=seg["end"], hed=a.hed)
            else:
                S.write_events_tsv(P.events, [], hed=a.hed)
                S.write_json(P.events_json, S.events_sidecar(hed=a.hed))
            seg_acq = acq + _dt.timedelta(seconds=seg["start"]) if acq is not None else None
            files = [P.video(v) for v in variants] + [lm_path, bs_path]
            S.write_scans_tsv(P.scans, [dict(filename=P.rel(f), acq_time=S.shift_acq_time(seg_acq, shift)
                                             if seg_acq else S.NA) for f in files])
            S.write_scans_tsv(R.scans, [dict(filename=R.rel(R.video("raw")),
                                             acq_time=seg_acq.isoformat(timespec="milliseconds") if seg_acq else S.NA)])
            rows.append(dict(sub=sub, run=run, segment_start_s=round(seg["start"], 3), segment_end_s=round(seg["end"], 3),
                             source_video=str(video), source_sha256=sha,
                             annotations=str(Path(a.annotations).resolve()) if a.annotations else "",
                             acq_time=a.acq_time or "", date_shift_days="" if shift is None else shift,
                             raw_video=str(R.video("raw")),
                             exported=_dt.datetime.now().astimezone().isoformat(timespec="seconds")))
            print(f"[segment] sub-{sub} run-{run:02d}: {seg['start']:.2f}-{seg['end']:.2f} s, {len(T)} frames, "
                  f"face in {found}; {', '.join(variants)} + raw", flush=True)
    finally:
        cap.release()
        est.close()
        if multi is not None:
            multi.close()
    append_mapping(mapping, rows)
    print(f"[segment] {len(rows)} segment(s); de-identified: {out}; raw (local only): {raw_root}; mapping: {mapping}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
