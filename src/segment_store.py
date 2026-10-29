"""PROVISIONAL on-disk format for annotated face/body segments. NOT LOCKED.

Every read and write of segment files goes through this module, so the format can be replaced
without touching the extraction, export or review code. FORMAT_VERSION is written into every
sidecar; readers check it. Expect breaking changes until the storage format is agreed.

Layout (one BIDS-style subject folder per pseudonymous subject, one run per segment):

  <root>/dataset_description.json
  <root>/sub-<id>/sub-<id>_scans.tsv                                   filename, acq_time
  <root>/sub-<id>/beh/sub-<id>_task-<task>_run-<NN>_recording-<rec>_desc-<d>_video.mp4  + _video.json
  <root>/sub-<id>/beh/sub-<id>_task-<task>_run-<NN>_recording-<rec>_landmarks.nc         + _landmarks.json
  <root>/sub-<id>/beh/sub-<id>_task-<task>_run-<NN>_recording-<rec>_blendshapes.nc
  <root>/sub-<id>/beh/sub-<id>_task-<task>_run-<NN>_events.tsv                          + _events.json

Video names follow BEP047 (bids-specification PR #2231: `beh/` datatype, `_video` suffix, optional
`recording-` entity). BEP047 is a proposal, not part of the released BIDS specification. The `desc-`
entity (BIDS derivatives) marks the variant: desc-raw (local only), desc-blur, desc-points.
`_landmarks` / `_blendshapes` are not BIDS suffixes (there is no released BIDS pose format).

Landmarks use the data layout of the `movement` toolbox (https://movement.neuroinformatics.dev,
BSD-3-Clause): an xarray Dataset with
  position   (time, space, keypoint, individual) float32, space = x, y, z
  confidence (time, keypoint, individual)        float32
  attrs ds_type="poses", source_software, fps, time_unit="seconds"
saved to netCDF (HDF5-based netCDF4 file, xarray.Dataset.to_netcdf). A 478-keypoint 3D dataset
was checked to save and reload unchanged with movement 0.17 / xarray (tests/test_segment_store.py).
movement itself is not needed to write or read; `xarray.open_dataset(path)` gives a dataset that
movement functions accept. If xarray or a netCDF backend is missing, the same arrays, dimension
names and attributes go into a .npz instead (write_landmarks(..., backend="npz")).
Blendshapes: separate (time, blendshape) array, blendshape = MediaPipe category names.
NWB / ndx-pose: not written yet (movement.io.save_poses.to_nwb_file is a candidate; see to_nwb()).

Annotations: the ELAN .eaf stays the editable copy. events.tsv is generated from it per recording
(columns onset, duration, trial_type, side, mode, eeg_onset[, HED]) with an events.json sidecar.
Free-text notes are never written to events.tsv. Alignment with EEG goes through acq_time in
scans.tsv (shifted dates in the de-identified copy; the shift is kept in the private mapping).
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import re
import secrets
from pathlib import Path

import numpy as np

FORMAT_VERSION = "semi-signals.segment/0.1-provisional"
PROVISIONAL = True
NA = "n/a"
EVENT_COLUMNS = ("onset", "duration", "trial_type", "side", "mode", "eeg_onset")
ONSET_ONLY_S = 0.05            # EAF stores an onset-only event as a 0.04 s span


# ── points-only per-event files (src/points_export.py) ───────
def write_points(path, arrs: dict) -> Path:
    """Points-only segment file (schema semi-signals.points/1, provisional): arrays + meta JSON string."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrs)
    return path


def read_points(path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


# ── naming ───────────────────────────────────────────────────
def bids_stem(sub, task, run=None, recording=None, desc=None, ses=None, acq=None) -> str:
    """Entities in BIDS order: sub, ses, task, acq, run, recording, desc. Labels must be alphanumeric."""
    for k, v in (("sub", sub), ("ses", ses), ("task", task), ("acq", acq), ("recording", recording), ("desc", desc)):
        if v is not None and not re.fullmatch(r"[A-Za-z0-9]+", str(v)):
            raise ValueError(f"BIDS {k} label must be alphanumeric, not {v!r}")
    parts = [f"sub-{sub}"]
    for k, v in (("ses", ses), ("task", task), ("acq", acq)):
        if v:
            parts.append(f"{k}-{v}")
    if run is not None:
        parts.append(f"run-{int(run):02d}")
    for k, v in (("recording", recording), ("desc", desc)):
        if v:
            parts.append(f"{k}-{v}")
    return "_".join(parts)


def new_pseudonym(n_hex: int = 8) -> str:
    """Random BIDS label (alphanumeric) for a subject or segment; never derived from the source."""
    return secrets.token_hex(n_hex)


class SegmentPaths:
    """File names for one segment (= one BIDS run) of one subject under root."""

    def __init__(self, root, sub, task="seizure", run=1, recording="face", ses=None):
        self.root, self.sub, self.task, self.run, self.recording, self.ses = Path(root), sub, task, run, recording, ses
        self.dir = self.root / f"sub-{sub}" / (f"ses-{ses}" if ses else "") / "beh"

    def _f(self, suffix, ext, desc=None, recording=True):
        stem = bids_stem(self.sub, self.task, self.run, self.recording if recording else None, desc, self.ses)
        return self.dir / f"{stem}_{suffix}{ext}"

    def video(self, desc):
        return self._f("video", ".mp4", desc)

    def video_json(self, desc):
        return self._f("video", ".json", desc)

    @property
    def landmarks(self):
        return self._f("landmarks", ".nc")

    @property
    def landmarks_json(self):
        return self._f("landmarks", ".json")

    @property
    def blendshapes(self):
        return self._f("blendshapes", ".nc")

    @property
    def events(self):
        return self._f("events", ".tsv", recording=False)

    @property
    def events_json(self):
        return self._f("events", ".json", recording=False)

    @property
    def scans(self):
        base = self.root / f"sub-{self.sub}" / (f"ses-{self.ses}" if self.ses else "")
        return base / (bids_stem(self.sub, None, ses=self.ses) + "_scans.tsv")

    def rel(self, p) -> str:
        """Path relative to the scans.tsv folder (BIDS scans.tsv filename column)."""
        return Path(p).relative_to(self.scans.parent).as_posix()


# ── JSON ─────────────────────────────────────────────────────
def write_json(path, d: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2, sort_keys=False) + "\n")
    return path


def read_json(path) -> dict:
    return json.loads(Path(path).read_text())


def check_version(d: dict, what="file"):
    v = d.get("SemiSignalsFormat")
    if v != FORMAT_VERSION:
        raise ValueError(f"{what}: format {v!r} is not {FORMAT_VERSION!r} (the provisional format changed; re-export)")


def dataset_description(root, name="semi-signals segments", deidentified=True) -> Path:
    return write_json(Path(root) / "dataset_description.json", dict(
        Name=name, BIDSVersion="1.10.0", DatasetType="derivative",
        GeneratedBy=[dict(Name="semi-signals segment_export", Description="provisional format, not locked")],
        SemiSignalsFormat=FORMAT_VERSION, SemiSignalsDeidentified=bool(deidentified),
        SemiSignalsNote=("Video names follow BEP047 (bids-specification PR #2231), which is not part of the "
                         "released BIDS specification. De-identified copies reduce re-identification risk; "
                         "they are not guaranteed anonymous (landmarks can be biometric identifiers).")))


# ── landmarks (movement layout) ──────────────────────────────
def keypoint_names(n: int = 478) -> list[str]:
    """MediaPipe face mesh index as the keypoint name: mp000 .. mp477."""
    return [f"mp{i:03d}" for i in range(n)]


def _have_netcdf() -> bool:
    try:
        import xarray  # noqa: F401
    except ImportError:
        return False
    for mod in ("netCDF4", "h5netcdf"):
        try:
            __import__(mod)
            return True
        except ImportError:
            pass
    return False


def landmarks_arrays(t, landmarks, confidence, names=None) -> dict:
    """(T, N, 3) landmarks + (T,) or (T, N) confidence -> movement-layout numpy arrays."""
    lm = np.asarray(landmarks, np.float32)
    T, N = lm.shape[:2]
    conf = np.asarray(confidence, np.float32)
    if conf.ndim == 1:
        conf = np.repeat(conf[:, None], N, 1)
    return dict(time=np.asarray(t, np.float64), space=np.array(["x", "y", "z"][: lm.shape[2]]),
                keypoint=np.array(names or keypoint_names(N)), individual=np.array(["subject"]),
                position=np.transpose(lm, (0, 2, 1))[..., None], confidence=conf[..., None])


def to_dataset(arrs: dict, attrs: dict):
    import xarray as xr
    return xr.Dataset(
        dict(position=(("time", "space", "keypoint", "individual"), arrs["position"]),
             confidence=(("time", "keypoint", "individual"), arrs["confidence"])),
        coords=dict(time=arrs["time"], space=arrs["space"], keypoint=arrs["keypoint"], individual=arrs["individual"]),
        attrs=attrs)


def _attrs(fps, source_software, extra):
    a = dict(ds_type="poses", source_software=source_software, time_unit="seconds",
             SemiSignalsFormat=FORMAT_VERSION)
    if fps:
        a["fps"] = float(fps)
    for k, v in (extra or {}).items():
        a[k] = v if isinstance(v, (str, int, float)) else json.dumps(v)
    return a


def write_landmarks(path, t, landmarks, confidence, fps=None, source_software="MediaPipe Face Landmarker",
                    attrs=None, backend="auto") -> Path:
    """Save (T, N, 3) landmarks in the movement layout. backend: "nc" (netCDF), "npz", or "auto"
    (netCDF when xarray and a netCDF library are installed). Returns the path written (suffix set
    by the backend)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = landmarks_arrays(t, landmarks, confidence)
    at = _attrs(fps, source_software, attrs)
    if backend == "auto":
        backend = "nc" if _have_netcdf() else "npz"
    if backend == "nc":
        out = path.with_suffix(".nc")
        to_dataset(arrs, at).to_netcdf(out)
        return out
    out = path.with_suffix(".npz")
    np.savez_compressed(out, **arrs, dims_position=np.array(["time", "space", "keypoint", "individual"]),
                        dims_confidence=np.array(["time", "keypoint", "individual"]), attrs=np.array(json.dumps(at)))
    return out


def read_landmarks(path) -> dict:
    """-> dict(t, landmarks (T, N, 3), confidence (T, N), keypoint_names, attrs, dataset (xarray or None))."""
    path = Path(path)
    if path.suffix == ".nc":
        import xarray as xr
        with xr.open_dataset(path) as d:
            ds = d.load()
        check_version(ds.attrs, str(path))
        pos = ds["position"].transpose("time", "space", "keypoint", "individual").values
        conf = ds["confidence"].transpose("time", "keypoint", "individual").values
        return dict(t=ds["time"].values, landmarks=np.transpose(pos[..., 0], (0, 2, 1)), confidence=conf[..., 0],
                    keypoint_names=[str(k) for k in ds["keypoint"].values], attrs=dict(ds.attrs), dataset=ds)
    z = np.load(path, allow_pickle=False)
    at = json.loads(str(z["attrs"]))
    check_version(at, str(path))
    assert list(z["dims_position"]) == ["time", "space", "keypoint", "individual"]
    return dict(t=z["time"], landmarks=np.transpose(z["position"][..., 0], (0, 2, 1)), confidence=z["confidence"][..., 0],
                keypoint_names=[str(k) for k in z["keypoint"]], attrs=at, dataset=None)


def write_blendshapes(path, t, blendshapes, names, backend="auto") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bs = np.asarray(blendshapes, np.float32)
    at = dict(SemiSignalsFormat=FORMAT_VERSION, source_software="MediaPipe Face Landmarker", time_unit="seconds")
    if backend == "auto":
        backend = "nc" if _have_netcdf() else "npz"
    if backend == "nc":
        import xarray as xr
        out = path.with_suffix(".nc")
        xr.Dataset(dict(blendshapes=(("time", "blendshape"), bs)),
                   coords=dict(time=np.asarray(t, np.float64), blendshape=np.array(names)), attrs=at).to_netcdf(out)
        return out
    out = path.with_suffix(".npz")
    np.savez_compressed(out, time=np.asarray(t, np.float64), blendshape=np.array(names), blendshapes=bs,
                        attrs=np.array(json.dumps(at)))
    return out


def read_blendshapes(path) -> dict:
    path = Path(path)
    if path.suffix == ".nc":
        import xarray as xr
        with xr.open_dataset(path) as d:
            ds = d.load()
        check_version(ds.attrs, str(path))
        return dict(t=ds["time"].values, blendshapes=ds["blendshapes"].values,
                    names=[str(n) for n in ds["blendshape"].values])
    z = np.load(path, allow_pickle=False)
    check_version(json.loads(str(z["attrs"])), str(path))
    return dict(t=z["time"], blendshapes=z["blendshapes"], names=[str(n) for n in z["blendshape"]])


def to_nwb(*_a, **_k):
    """Not implemented yet: NWB / ndx-pose export (movement.io.save_poses.to_nwb_file is a candidate)."""
    raise NotImplementedError("NWB / ndx-pose export is not implemented in the provisional format")


# ── events.tsv from the EAF ──────────────────────────────────
def _num(x):
    return NA if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{float(x):.3f}"


def events_rows(events: list[dict], t0: float = 0.0, t_end: float | None = None, hed: bool = False) -> list[dict]:
    """Annotation events (src/annotation.py dicts) -> events.tsv rows, times relative to t0 (segment start).
    One trial_type=clinical row per event (duration = clinical end - onset, n/a when onset only) and one
    trial_type=EEG row per marked EEG onset. Events entirely outside [t0, t_end] are left out.
    Notes are never included."""
    rows = []
    for e in events:
        on = e.get("clinical_onset_s")
        if on is None:
            continue
        end = e.get("clinical_end_s")
        end = end if end is not None and end - on > ONSET_ONLY_S else None
        if t_end is not None and on > t_end:
            continue
        if (end if end is not None else on) < t0:
            continue
        eeg = e.get("eeg_onset_s")
        mode = e.get("mode") or "body"
        side = e.get("side") or NA
        base = dict(side=side, mode=mode)
        if hed:
            base["HED"] = e.get("hed") or NA
        rows.append(dict(onset=_num(on - t0), duration=_num(None if end is None else end - on), trial_type="clinical",
                         eeg_onset=_num(None if eeg is None else eeg - t0), **base))
        if eeg is not None:
            rows.append(dict(onset=_num(eeg - t0), duration=_num(0.0), trial_type="EEG",
                             eeg_onset=_num(eeg - t0), **base))
    rows.sort(key=lambda r: (float(r["onset"]), r["trial_type"] != "clinical"))
    return rows


def write_events_tsv(path, rows: list[dict], hed: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(EVENT_COLUMNS) + (["HED"] if hed else [])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, NA) for c in cols})
    return path


def read_events_tsv(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def rows_to_events(rows: list[dict], t0: float = 0.0) -> list[dict]:
    """events.tsv rows -> event dicts (mode, clinical_onset_s, clinical_end_s, eeg_onset_s, side), absolute
    times (t0 added back). EEG rows are folded into their clinical event via eeg_onset."""
    f = lambda v: None if v in (NA, "", None) else float(v)
    out = []
    for r in rows:
        if r["trial_type"] != "clinical":
            continue
        on, dur, eeg = f(r["onset"]), f(r["duration"]), f(r["eeg_onset"])
        out.append(dict(mode=r["mode"], clinical_onset_s=round(on + t0, 3),
                        clinical_end_s=None if dur is None else round(on + dur + t0, 3),
                        eeg_onset_s=None if eeg is None else round(eeg + t0, 3),
                        side=None if r["side"] == NA else r["side"]))
    return out


def events_sidecar(hed: bool = False, side_levels=("L", "R", "both")) -> dict:
    d = dict(
        onset=dict(Description="Time of the event relative to the start of the video recording (this segment).",
                   Units="s"),
        duration=dict(Description="clinical rows: clinical end minus clinical onset; n/a when only the onset was "
                                  "marked. EEG rows: 0.", Units="s"),
        trial_type=dict(Description="What the row marks.",
                        Levels=dict(clinical="clinical (video) onset of an annotated event",
                                    EEG="EEG onset of the same event, as marked by the annotator")),
        side=dict(Description="Side of the subject the event involves (face mode); n/a when not set.",
                  Levels={k: dict(L="subject's left", R="subject's right", both="both sides").get(k, k)
                          for k in side_levels}),
        mode=dict(Description="Annotation mode the event was marked in.",
                  Levels=dict(body="body / hands annotation", face="face annotation")),
        eeg_onset=dict(Description="EEG onset of the event relative to the start of the video recording; align "
                                   "with the EEG through acq_time in scans.tsv. n/a when not marked.", Units="s"),
        SemiSignalsFormat=FORMAT_VERSION,
        SemiSignalsNote="Generated from the ELAN .eaf (the editable copy). Free-text notes are not exported.")
    if hed:
        d["HED"] = dict(Description="Optional HED annotation per row, intended for HED-SCORE "
                                    "(https://github.com/hed-standard/hed-schemas, library 'score'); n/a when not set.")
    return d


def eaf_to_events(eaf, out_tsv, t0: float = 0.0, t_end: float | None = None, hed: bool = False,
                  sidecar: bool = True) -> Path:
    """Generate <recording>_events.tsv (+ .json) from an ELAN .eaf written by src/annotation.py."""
    from src.annotation import read_eaf
    rows = events_rows(read_eaf(eaf)["events"], t0=t0, t_end=t_end, hed=hed)
    out = write_events_tsv(out_tsv, rows, hed=hed)
    if sidecar:
        write_json(Path(out_tsv).with_suffix(".json"), events_sidecar(hed=hed))
    return out


# ── scans.tsv ────────────────────────────────────────────────
def shift_acq_time(t: _dt.datetime, shift_days: int) -> str:
    """De-identified acq_time: date shifted by shift_days (keep the same shift for all of a subject's
    files; BIDS suggests shifted years of 1925 or earlier), time of day kept."""
    return (t + _dt.timedelta(days=shift_days)).replace(tzinfo=None).isoformat(timespec="milliseconds")


def random_shift_days(t: _dt.datetime) -> int:
    """A random negative day shift that moves t into 1900..1925."""
    target = _dt.datetime(1900 + secrets.randbelow(26), 1, 1) + _dt.timedelta(days=secrets.randbelow(365))
    return (target.date() - t.date()).days


def write_scans_tsv(path, rows: list[dict]) -> Path:
    """rows: dict(filename, acq_time) (+ extra columns). Rows for existing filenames are replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = read_scans_tsv(path) if path.exists() else []
    new = {r["filename"]: r for r in old}
    new.update({r["filename"]: r for r in rows})
    cols = ["filename", "acq_time"] + sorted({k for r in new.values() for k in r} - {"filename", "acq_time"})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for k in sorted(new):
            w.writerow({c: new[k].get(c, NA) or NA for c in cols})
    return path


def read_scans_tsv(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ── sidecars ─────────────────────────────────────────────────
def video_sidecar(video, extra: dict | None = None) -> dict:
    """BEP047 media fields (RecordingDuration, ImageWidth/Height, VideoCodec, VideoFrameRate, VideoFrameCount)
    read back from the written file, plus semi-signals fields."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    four = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    d = dict(RecordingDuration=round(n / fps, 3) if fps else None, ImageWidth=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             ImageHeight=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
             VideoCodec="".join(chr((four >> 8 * i) & 0xFF) for i in range(4)).strip("\x00") or None,
             VideoFrameRate=round(fps, 3), VideoFrameCount=n, SemiSignalsFormat=FORMAT_VERSION)
    cap.release()
    d.update(extra or {})
    return d


def landmarks_sidecar(*, coords: str, units: str, deidentified: bool, mode: str, preprocess: str,
                      mapping_ref: str | None, blendshape_file: str | None, extra: dict | None = None) -> dict:
    d = dict(SemiSignalsFormat=FORMAT_VERSION, Provisional=True,
             Layout="movement poses dataset: position (time, space, keypoint, individual), confidence "
                    "(time, keypoint, individual); keypoint = MediaPipe face mesh index (mp000..mp477)",
             CoordinateFrame=coords, Units=units, Deidentified=bool(deidentified), AnnotationMode=mode,
             FacePreprocess=preprocess, Confidence="per-frame face score repeated over keypoints (1.0 = face "
             "returned, NaN = no face); MediaPipe reports no per-landmark confidence",
             BlendshapesFile=blendshape_file, PrivateMappingRef=mapping_ref,
             Model="MediaPipe Face Landmarker (Google, Apache-2.0), face_landmarker.task float16 v1")
    d.update(extra or {})
    return d


def check_outside(path, root, what="file"):
    """Raise if path is inside root (the private mapping and raw video must never sit in the export)."""
    p, r = Path(path).resolve(), Path(root).resolve()
    if p == r or r in p.parents:
        raise ValueError(f"{what} {p} must be outside {r}")
