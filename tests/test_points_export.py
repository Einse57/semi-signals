"""Points-only segment export: relative time, selected person, iris removal, canonical face, no
identifying fields in the segment files, and a separate private mapping."""
import csv
import json
import zipfile

import numpy as np
import pytest

from src import points_export as PE


def _rot(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def test_canonical_face_undoes_pose():
    rng = np.random.default_rng(0)
    c = rng.normal(size=(478, 3))
    R, s, t = _rot(0.6, -0.3, 0.4), 37.0, np.array([300.0, -200.0, 50.0])
    cam = (s * c @ R.T) + t                          # canonical -> camera (metric-like, y up, z to viewer)
    lm = (cam * np.array([1, -1, -1])).astype(np.float32)   # camera -> image convention (y down, z away)
    M = np.eye(4); M[:3, :3] = s * R; M[:3, 3] = t
    got = PE.canonical_face(lm, M)
    want = c - c[:468].mean(0)
    want /= np.linalg.norm(want[33] - want[263])
    assert np.allclose(got, want, atol=1e-4)
    assert np.isnan(PE.canonical_face(np.full((478, 3), np.nan, np.float32), M)).all()


def _npz(tmp_path, face=True, transform=True):
    F, P = 20, 3
    t = np.arange(F) / 2.0                                       # 2 samples/s, 0..9.5 s
    sel = np.zeros(F, np.int16); sel[7] = -1; sel[8] = 1
    body = np.zeros((F, P, 17, 2), np.float32); body[:, 1] = 7.0
    arrs = dict(t=t, frame_idx=np.arange(F), body_kpts=body, body_scores=np.ones((F, P, 17), np.float32),
                body_boxes=np.zeros((F, P, 5), np.float32), sel=sel,
                hand_kpts=np.zeros((F, 2, 21, 2), np.float32), hand_scores=np.zeros((F, 2, 21), np.float32),
                meta=np.array(json.dumps(dict(video="Jane_Doe_2026-01-02.mp4", sample_fps=2.0, width=640, height=480,
                                              face=dict(blendshape_names=[f"b{i}" for i in range(52)])))))
    if face:
        rng = np.random.default_rng(1)
        arrs.update(face_landmarks=rng.normal(size=(F, P, 478, 3)).astype(np.float32) * 10 + 100,
                    face_blendshapes=np.full((F, P, 52), 0.5, np.float32), face_scores=np.ones((F, P), np.float32),
                    face_boxes=np.zeros((F, P, 4), np.float32))
        if transform:
            arrs["face_transform"] = np.broadcast_to(np.eye(4, dtype=np.float32), (F, P, 4, 4)).copy()
    p = tmp_path / "Jane_Doe_2026-01-02.npz"
    np.savez_compressed(p, **arrs)
    return p


def _ann(tmp_path):
    p = tmp_path / "Jane_Doe.annotation.json"
    p.write_text(json.dumps(dict(annotator="AL", pat_id="Jane Doe", events=[
        dict(clinical_onset_s=4.0, clinical_end_s=5.0, eeg_onset_s=3.0, event_class="seizure", seizure_type="", notes="Jane said"),
        dict(mode="face", clinical_onset_s=6.0, clinical_end_s=None, side="R", face_term="", notes="")])))
    return p


def test_export_segments_and_private_mapping(tmp_path):
    out, mp = tmp_path / "export", tmp_path / "private" / "map.csv"
    PE.main(["--npz", str(_npz(tmp_path)), "--annotations", str(_ann(tmp_path)), "--out", str(out),
             "--mapping", str(mp), "--pre-s", "1", "--post-s", "1", "--default-len-s", "2", "--drop-iris"])
    files = sorted(out.glob("*.npz"))
    assert len(files) == 2
    rows = list(csv.DictReader(open(mp)))
    assert {r["segment_id"] + ".npz" for r in rows} == {f.name for f in files}
    by_mode = {r["event_mode"]: out / (r["segment_id"] + ".npz") for r in rows}
    z = np.load(by_mode["body"])
    assert np.allclose(z["t"], np.arange(-1.0, 2.01, 0.5))         # 3..6 s around onset 4 s, relative
    assert z["face_landmarks"].shape == (7, 468, 3)                  # iris dropped
    assert np.isnan(z["body_kpts"][1]).all() and (z["body_kpts"][2] == 7.0).all()   # sel -1 and sel 1
    m = json.loads(str(z["meta"]))
    assert m["event"] == dict(mode="body", clinical_onset_rel_s=0.0, clinical_end_rel_s=1.0, eeg_onset_rel_s=-1.0,
                              event_class="seizure")
    zf = np.load(by_mode["face"])
    assert json.loads(str(zf["meta"]))["event"]["side"] == "R" and np.isclose(zf["t"].max(), 3.0)
    for f in files:                                                   # nothing identifying inside
        with zipfile.ZipFile(f) as zz:
            blob = b"".join(zz.read(n) for n in zz.namelist())
            assert all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in zz.infolist())
        for word in (b"Jane", b"2026", b"AL", str(tmp_path).encode()):
            assert word not in blob
        assert not set(np.load(f).files) - {"t", "body_kpts", "body_scores", "hand_kpts", "hand_scores",
                                            "face_landmarks", "face_blendshapes", "face_scores", "meta"}


def test_export_options(tmp_path):
    npz, ann = _npz(tmp_path), _ann(tmp_path)
    out = tmp_path / "e"
    PE.main(["--npz", str(npz), "--annotations", str(ann), "--out", str(out), "--mapping", str(tmp_path / "m.csv"),
             "--mode", "face", "--canonical-face", "--keep-transform", "--include-notes"])
    (f,) = out.glob("*.npz")
    z = np.load(f)
    m = json.loads(str(z["meta"]))
    assert m["face"]["coords"] == "canonical_iod" and z["face_landmarks"].shape[1:] == (478, 3)
    assert "face_transform" in z.files and m["event"]["mode"] == "face"
    with pytest.raises(SystemExit):                                   # mapping inside the export folder
        PE.main(["--npz", str(npz), "--annotations", str(ann), "--out", str(out), "--mapping", str(out / "m.csv")])
    with pytest.raises(SystemExit):                                   # canonical needs the transform
        PE.main(["--npz", str(_npz(tmp_path, transform=False)), "--annotations", str(ann), "--out", str(tmp_path / "e2"),
                 "--mapping", str(tmp_path / "m2.csv"), "--canonical-face"])
    PE.main(["--npz", str(_npz(tmp_path, face=False)), "--annotations", str(ann), "--out", str(tmp_path / "e3"),
             "--mapping", str(tmp_path / "m3.csv")])
    assert all("face_landmarks" not in np.load(p).files for p in (tmp_path / "e3").glob("*.npz"))
