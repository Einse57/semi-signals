"""Points-only segment export for annotated events (reduced re-identification risk, NOT de-identification).

  python -m src.points_export --npz data/extract/clip.npz --annotations data/clip.annotation.json \\
         --out export/points --mapping private/points_mapping.csv [--mode face] [--drop-iris] [--canonical-face]

Input: a batch_extract .npz (with --face for face arrays; with --face-transform for --canonical-face) and the
annotation of the same video (<stem>.annotation.json or <stem>.eaf from the annotation tool).
Output, one file per annotated event: <out>/<segment_id>.npz with keypoints, scores, blendshapes and
timestamps relative to the marked clinical onset, for the selected person only. No frames, crops,
thumbnails, file names, paths, dates, annotator or patient fields, and no free-text notes (notes can hold
names; --include-notes adds them). The segment id is random (secrets.token_hex). The link back to the source
video goes into a separate private mapping CSV (--mapping), which must be outside --out.

Face options:
  --drop-iris        drop the 10 iris points (468..477): face_landmarks becomes (T, 468, 3)
  --canonical-face   remove head pose with MediaPipe's facial transformation matrix: image coordinates
                     (x, y down, z away) -> (x, -y, -z), centred on the mesh mean, rotated by R^T
                     (R = rotation part of the matrix) and divided by the outer-eye-corner distance
                     (landmarks 33-263). Needs face_transform in the .npz (batch_extract --face-transform).
                     Checked on rolled and warped copies of MediaPipe's test image: the remaining rotation
                     to MediaPipe's canonical face model was 3-10 degrees (30-54 degrees before).
  --keep-transform   also store the 4x4 matrices (they hold head pose and position)
Rigid normalisation removes pose and position, not face shape: dense landmarks can still be biometric
identifiers. See docs/FACE_LANDMARKS.md (Points-only export).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import secrets
import sys
from pathlib import Path

import numpy as np

from src import segment_store as S

SCHEMA = "semi-signals.points/1"          # provisional; files are written by src/segment_store.write_points
IRIS = slice(468, 478)
EYE_OUTER = (33, 263)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Points-only segment export (keypoints, scores, blendshapes)")
    p.add_argument("--npz", required=True, help="batch_extract output for the video")
    p.add_argument("--annotations", required=True, help="<stem>.annotation.json or <stem>.eaf")
    p.add_argument("--out", required=True, help="folder for the segment files")
    p.add_argument("--mapping", required=True, help="private CSV linking segment ids to the source (kept outside --out)")
    p.add_argument("--mode", default="both", choices=["body", "face", "both"], help="which annotated events to export")
    p.add_argument("--pre-s", type=float, default=10.0, help="seconds before the clinical onset")
    p.add_argument("--post-s", type=float, default=10.0, help="seconds after the clinical end")
    p.add_argument("--default-len-s", type=float, default=30.0, help="event length when no end was marked")
    p.add_argument("--drop-iris", action="store_true")
    p.add_argument("--canonical-face", action="store_true")
    p.add_argument("--keep-transform", action="store_true")
    p.add_argument("--no-face", action="store_true", help="body and hands only")
    p.add_argument("--include-notes", action="store_true", help="also export free-text notes (check them for names first)")
    return p.parse_args(argv)


def load_events(path) -> list[dict]:
    from src import annotation as A
    path = Path(path)
    if path.suffix.lower() == ".eaf":
        evs = A.read_eaf(path)["events"]
    else:
        evs = json.loads(path.read_text()).get("events", [])
    return [dict(e, mode=A.event_mode(e)) for e in evs if e.get("clinical_onset_s") is not None]


def canonical_face(lm: np.ndarray, M: np.ndarray) -> np.ndarray:
    """lm (N, 3) image px (z on the x scale), M (4, 4) facial transformation matrix -> pose-free (N, 3)
    in units of the outer-eye-corner distance. NaN in, NaN out."""
    if not (np.isfinite(lm).all() and np.isfinite(M).all()):
        return np.full(lm.shape, np.nan, np.float32)
    p = lm.astype(np.float64) * np.array([1.0, -1.0, -1.0])
    R = M[:3, :3] / np.linalg.norm(M[:3, 0])
    q = (p - p[:468].mean(0)) @ R                   # row vectors: q = R^T p
    iod = np.linalg.norm(q[EYE_OUTER[0]] - q[EYE_OUTER[1]])
    return (q / iod).astype(np.float32) if iod > 0 else np.full(lm.shape, np.nan, np.float32)


def _pick(arr, sel):
    """arr (F, P, ...) -> (F, ...) for the selected person per frame; NaN where no person was selected."""
    out = np.full((arr.shape[0],) + arr.shape[2:], np.nan, np.float32)
    ok = sel >= 0
    out[ok] = arr[np.nonzero(ok)[0], sel[ok]]
    return out


def export_segment(z, e: dict, a, seg_id: str) -> dict:
    t = z["t"].astype(np.float64)
    on = float(e["clinical_onset_s"])
    end = e.get("clinical_end_s")
    end = float(end) if end is not None and end > on + 0.05 else on + a.default_len_s
    w = (t >= on - a.pre_s) & (t <= end + a.post_s)
    sel = z["sel"].astype(np.int64)[w]
    arrs = dict(t=(t[w] - on).astype(np.float32),
                body_kpts=_pick(z["body_kpts"][w], sel), body_scores=_pick(z["body_scores"][w], sel),
                hand_kpts=z["hand_kpts"][w].astype(np.float32), hand_scores=z["hand_scores"][w].astype(np.float32))
    meta = json.loads(str(z["meta"]))
    face_info = None
    if not a.no_face and "face_landmarks" in z.files:
        fsel = z["face_person"].astype(np.int64)[w] if "face_person" in z.files else sel   # face slot per frame
        lm = _pick(z["face_landmarks"][w], fsel)
        coords = "image_px"
        if a.canonical_face or a.keep_transform:
            if "face_transform" not in z.files:
                raise SystemExit("--canonical-face / --keep-transform need face_transform in the .npz "
                                 "(run batch_extract with --face --face-transform)")
            tf = _pick(z["face_transform"][w], fsel)
        if a.canonical_face:
            lm = np.stack([canonical_face(l, m) for l, m in zip(lm, tf)]) if len(lm) else lm
            coords = "canonical_iod"
        if a.drop_iris:
            lm = lm[:, :468]
        arrs.update(face_landmarks=lm, face_blendshapes=_pick(z["face_blendshapes"][w], fsel),
                    face_scores=_pick(z["face_scores"][w], fsel))
        if a.keep_transform:
            arrs["face_transform"] = tf
        face_info = dict(coords=coords, iris_dropped=bool(a.drop_iris), n_landmarks=int(lm.shape[1]) if lm.ndim == 3 else 0,
                         transform_kept=bool(a.keep_transform),
                         blendshape_names=meta.get("face", {}).get("blendshape_names", []))
    rel = lambda x: None if x is None else round(float(x) - on, 3)
    ev = dict(mode=e["mode"], clinical_onset_rel_s=0.0, clinical_end_rel_s=rel(e.get("clinical_end_s")),
              eeg_onset_rel_s=rel(e.get("eeg_onset_s")))
    for k in ("event_class", "seizure_type", "side", "face_term"):
        if e.get(k):
            ev[k] = e[k]
    if a.include_notes and e.get("notes"):
        ev["notes"] = e["notes"]
    smeta = dict(schema=SCHEMA, segment_id=seg_id, event=ev, sample_fps=meta.get("sample_fps"),
                 frame_size=[meta.get("width"), meta.get("height")], body_layout="COCO-17 (x, y px)",
                 hand_layout="[left, right] x 21 (x, y px)", face=face_info,
                 note="points only; reduced re-identification risk, not de-identified")
    arrs["meta"] = np.array(json.dumps(smeta))
    return arrs


def main(argv=None):
    a = parse_args(argv)
    out, mapping = Path(a.out).resolve(), Path(a.mapping).resolve()
    if out == mapping.parent or out in mapping.parents:
        raise SystemExit("--mapping must be outside --out (the mapping links segments back to the video)")
    z = np.load(a.npz, allow_pickle=False)
    evs = [e for e in load_events(a.annotations) if a.mode == "both" or e["mode"] == a.mode]
    out.mkdir(parents=True, exist_ok=True)
    mapping.parent.mkdir(parents=True, exist_ok=True)
    new_map = not mapping.exists()
    rows = []
    for k, e in enumerate(sorted(evs, key=lambda e: e["clinical_onset_s"]), 1):
        seg = secrets.token_hex(8)
        arrs = export_segment(z, e, a, seg)
        if len(arrs["t"]) == 0:
            print(f"[points] event {k} ({e['mode']}, onset {e['clinical_onset_s']:.1f} s): no sampled frames, skipped")
            continue
        S.write_points(out / f"{seg}.npz", arrs)
        rows.append(dict(segment_id=seg, source_npz=str(Path(a.npz).resolve()), annotations=str(Path(a.annotations).resolve()),
                         event_mode=e["mode"], clinical_onset_s=e["clinical_onset_s"],
                         exported=_dt.datetime.now().astimezone().isoformat(timespec="seconds")))
        print(f"[points] {seg}.npz: {e['mode']} event, {len(arrs['t'])} frames")
    with open(mapping, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["segment_id"])
        if new_map and rows:
            w.writeheader()
        w.writerows(rows)
    print(f"[points] {len(rows)} segment(s) in {out}; private mapping: {mapping}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
