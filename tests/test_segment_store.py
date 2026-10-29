"""Provisional segment store (src/segment_store.py): landmark/blendshape save-reload, EAF -> events.tsv
round trip, naming, sidecars, scans.tsv. The format is not locked; these tests pin today's behaviour."""
import datetime as dt
import json

import numpy as np
import pytest

from src import annotation as A
from src import segment_store as S


def _lm(T=6, N=478, seed=0):
    rng = np.random.default_rng(seed)
    lm = rng.normal(100, 30, (T, N, 3)).astype(np.float32)
    lm[2] = np.nan                                   # a frame without a face
    conf = np.ones(T, np.float32)
    conf[2] = np.nan
    return np.arange(T) / 25.0, lm, conf


@pytest.mark.parametrize("backend", ["nc", "npz"])
def test_landmarks_roundtrip(tmp_path, backend):
    if backend == "nc" and not S._have_netcdf():
        pytest.skip("xarray/netCDF4 not installed")
    t, lm, conf = _lm()
    p = S.write_landmarks(tmp_path / "x_landmarks.nc", t, lm, conf, fps=25.0, backend=backend,
                          attrs=dict(face_preprocess="detect"))
    assert p.suffix == "." + backend
    d = S.read_landmarks(p)
    np.testing.assert_array_equal(d["landmarks"], lm)               # NaN in the same places
    np.testing.assert_allclose(d["t"], t)
    assert d["confidence"].shape == (6, 478) and np.isnan(d["confidence"][2]).all()
    assert d["keypoint_names"][:2] == ["mp000", "mp001"] and len(d["keypoint_names"]) == 478
    assert d["attrs"]["fps"] == 25.0 and d["attrs"]["face_preprocess"] == "detect"
    S.check_version(d["attrs"])


def test_landmarks_movement_layout(tmp_path):
    if not S._have_netcdf():
        pytest.skip("xarray/netCDF4 not installed")
    import xarray as xr
    t, lm, conf = _lm()
    ds = xr.open_dataset(S.write_landmarks(tmp_path / "a_landmarks.nc", t, lm, conf, fps=25.0))
    assert ds["position"].dims == ("time", "space", "keypoint", "individual")
    assert ds["confidence"].dims == ("time", "keypoint", "individual")
    assert list(ds["space"].values) == ["x", "y", "z"] and ds.attrs["ds_type"] == "poses"
    ds.close()


def test_blendshapes_roundtrip(tmp_path):
    from src.face_estimator import BLENDSHAPE_NAMES
    bs = np.random.default_rng(1).random((5, 52)).astype(np.float32)
    bs[1] = np.nan
    for backend in (["nc"] if S._have_netcdf() else []) + ["npz"]:
        p = S.write_blendshapes(tmp_path / f"b_{backend}_blendshapes.nc", np.arange(5) / 25, bs, BLENDSHAPE_NAMES,
                                backend=backend)
        d = S.read_blendshapes(p)
        np.testing.assert_array_equal(d["blendshapes"], bs)
        assert list(d["names"]) == list(BLENDSHAPE_NAMES)


def test_nwb_is_a_stub():
    with pytest.raises(NotImplementedError):
        S.to_nwb()


def _eaf(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    ann = A.new_annotation(video, 25.0, 60.0, annotator="XX")
    ann["events"] = [dict(mode="body", clinical_onset_s=5.0, clinical_end_s=9.0, eeg_onset_s=4.0,
                          event_class="seizure", seizure_type="", notes="private body note"),
                     dict(mode="face", clinical_onset_s=12.0, clinical_end_s=16.5, eeg_onset_s=11.5, side="L",
                          face_term="", notes="left mouth corner, patient name here"),
                     dict(mode="face", clinical_onset_s=20.0, clinical_end_s=None, eeg_onset_s=None, side=None,
                          face_term="", notes="")]
    return A.write_eaf(ann, video), ann


@pytest.mark.parametrize("t0", [0.0, 3.0])
def test_eaf_to_events_roundtrip_and_no_notes(tmp_path, t0):
    eaf, ann = _eaf(tmp_path)
    tsv = S.eaf_to_events(eaf, tmp_path / "sub-x_task-seizure_run-01_events.tsv", t0=t0, hed=True)
    text = tsv.read_text()
    assert "note" not in text.lower() and "mouth" not in text and "private" not in text
    rows = S.read_events_tsv(tsv)
    assert list(rows[0]) == list(S.EVENT_COLUMNS) + ["HED"]
    assert [r["trial_type"] for r in rows] == ["EEG", "clinical", "EEG", "clinical", "clinical"]
    clin = [r for r in rows if r["trial_type"] == "clinical"]
    assert clin[2]["duration"] == "n/a" and clin[1]["side"] == "L" and clin[0]["mode"] == "body"
    assert float(clin[0]["onset"]) == pytest.approx(5.0 - t0)
    back = S.rows_to_events(rows, t0=t0)
    want = [(e["mode"], e["clinical_onset_s"], e["clinical_end_s"], e["eeg_onset_s"], e.get("side"))
            for e in sorted(ann["events"], key=lambda e: e["clinical_onset_s"])]
    assert [(e["mode"], e["clinical_onset_s"], e["clinical_end_s"], e["eeg_onset_s"], e["side"]) for e in back] == want
    side = json.loads(tsv.with_suffix(".json").read_text())
    assert side["side"]["Levels"].keys() >= {"L", "R", "both"} and "HED" in side
    assert "eeg_onset" in side and "trial_type" in side


def test_events_window_and_empty(tmp_path):
    eaf, _ = _eaf(tmp_path)
    rows = S.events_rows(A.read_eaf(eaf)["events"], t0=10.0, t_end=17.0)
    assert {r["mode"] for r in rows} == {"face"} and float(rows[0]["onset"]) == pytest.approx(1.5)
    p = S.write_events_tsv(tmp_path / "e.tsv", [])
    assert p.read_text().strip().split("\t") == list(S.EVENT_COLUMNS)


def test_naming_bep047_style(tmp_path):
    P = S.SegmentPaths(tmp_path, "ab12", "seizure", 3, "face")
    assert P.video("blur").name == "sub-ab12_task-seizure_run-03_recording-face_desc-blur_video.mp4"
    assert P.landmarks.name == "sub-ab12_task-seizure_run-03_recording-face_landmarks.nc"
    assert P.events.name == "sub-ab12_task-seizure_run-03_events.tsv"
    assert P.video("raw").parent == tmp_path / "sub-ab12" / "beh"
    assert P.scans == tmp_path / "sub-ab12" / "sub-ab12_scans.tsv"
    assert P.rel(P.video("points")) == "beh/sub-ab12_task-seizure_run-03_recording-face_desc-points_video.mp4"
    with pytest.raises(ValueError):
        S.bids_stem("a_b", "seizure")


def test_scans_shift_and_merge(tmp_path):
    t = dt.datetime(2026, 3, 4, 10, 11, 12, 500000)
    d = S.random_shift_days(t)
    s = S.shift_acq_time(t, d)
    assert 1900 <= int(s[:4]) <= 1925 and s.endswith("10:11:12.500")
    p = tmp_path / "sub-x_scans.tsv"
    S.write_scans_tsv(p, [dict(filename="beh/a.mp4", acq_time=s)])
    S.write_scans_tsv(p, [dict(filename="beh/b.mp4", acq_time="n/a"), dict(filename="beh/a.mp4", acq_time=s)])
    assert [r["filename"] for r in S.read_scans_tsv(p)] == ["beh/a.mp4", "beh/b.mp4"]


def test_sidecars_and_guards(tmp_path):
    d = S.landmarks_sidecar(coords="image px", units="pixels", deidentified=True, mode="face", preprocess="none",
                            mapping_ref="private_mapping.csv", blendshape_file="b.nc")
    assert d["Provisional"] is True and d["Deidentified"] is True and d["PrivateMappingRef"] == "private_mapping.csv"
    S.check_version(d)
    with pytest.raises(ValueError):
        S.check_version(dict(SemiSignalsFormat="semi-signals.segment/9"))
    with pytest.raises(ValueError):
        S.check_outside(tmp_path / "out" / "map.csv", tmp_path / "out", "--mapping")
    S.check_outside(tmp_path / "raw" / "map.csv", tmp_path / "out", "--mapping")
    dd = json.loads(S.dataset_description(tmp_path / "out").read_text())
    assert "BEP047" in json.dumps(dd) and dd["DatasetType"] == "derivative"


def test_points_file_roundtrip(tmp_path):
    p = S.write_points(tmp_path / "e01.npz", dict(t=np.arange(3, dtype=np.float32), meta=json.dumps(dict(a=1))))
    d = S.read_points(p)
    assert d["meta"] == dict(a=1) and d["t"].tolist() == [0, 1, 2]
