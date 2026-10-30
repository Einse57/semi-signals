"""Segment export (src/segment_export.py): blur masks, raw-location guard, and (with the model file and
MediaPipe's test image present) an end-to-end run on a short synthetic clip."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src import segment_export as X
from src import segment_store as S

REPO = Path(__file__).resolve().parent.parent


def _ring(cx, cy, r=20):
    a = np.linspace(0, 2 * np.pi, 478, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a), np.zeros(478)], 1).astype(np.float32)


def test_blur_covers_subject_and_bystanders():
    rng = np.random.default_rng(0)
    f = rng.integers(0, 255, (120, 200, 3), np.uint8)
    out = X.mask_faces(f, _ring(50, 60), others=[_ring(150, 60)])          # default style: fill
    assert not np.array_equal(out[60, 50], f[60, 50]) and not np.array_equal(out[60, 150], f[60, 150])
    assert np.array_equal(out[5, 100], f[5, 100])                     # outside both faces unchanged
    assert np.ptp(out[50:70, 40:60].reshape(-1, 3), 0).max() == 0      # solid
    assert np.ptp(out[50:70, 140:160].reshape(-1, 3), 0).max() == 0    # bystander solid too
    whole = X.mask_faces(f, None)                                      # no subject face: whole frame, one colour
    assert np.ptp(whole.reshape(-1, 3), 0).max() == 0
    assert np.array_equal(X.mask_faces(f, None, missing="none"), f)
    blur = X.blur_face(f, _ring(50, 60))
    assert not np.array_equal(blur[60, 50], f[60, 50]) and np.ptp(blur[50:70, 40:60].reshape(-1, 3), 0).max() > 0


def test_default_variants_are_shareable():
    a = X.parse_args(["--input", "x", "--out", "o", "--raw-root", "r"])
    assert a.deid == "fill,points" and not a.blur_in_export
    assert "blur" not in a.deid.split(",") and X.NOT_FOR_SHARING == ("blur",)


def test_ghost_without_config_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv("SEMI_SIGNALS_SWAP_CONFIG", raising=False)
    v = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    w.write(np.zeros((32, 32, 3), np.uint8))
    w.release()
    with pytest.raises(SystemExit, match="swap"):
        X.main(["--input", str(v), "--out", str(tmp_path / "o"), "--raw-root", str(tmp_path / "r"), "--deid", "ghost"])
    assert not (tmp_path / "o").exists()                               # nothing written


class _Multi:
    def __init__(self, faces):
        self.faces = faces

    def all_faces(self, img, offset=(0.0, 0.0)):
        h, w = img.shape[:2]
        return [f for f in self.faces if offset[0] <= f[:, 0].mean() < offset[0] + w
                and offset[1] <= f[:, 1].mean() < offset[1] + h]


def test_other_faces_dedupes_and_drops_subject():
    f = np.zeros((120, 200, 3), np.uint8)
    subj, other = _ring(50, 60), _ring(150, 60)
    got = X.other_faces(_Multi([subj, other]), f, subj)
    assert len(got) == 1 and np.allclose(got[0][:, 0].mean(), 150)


def test_raw_root_inside_repo_must_be_ignored(tmp_path):
    with pytest.raises(SystemExit):
        X.check_raw_location(REPO / "src" / "rawtest", tmp_path / "out")
    X.check_raw_location(REPO / "local" / "raw", tmp_path / "out")       # local/ is gitignored
    with pytest.raises(ValueError):
        X.check_raw_location(tmp_path / "out" / "raw", tmp_path / "out")


MODEL = REPO / "models" / "face_landmarker.task"
from tests.assets import TEST_IMAGE as IMG    # noqa: E402  (FACE_TEST_IMAGE overrides)


@pytest.mark.skipif(not (MODEL.exists() and IMG.exists()), reason="model file or test image not present")
def test_end_to_end_synthetic_clip(tmp_path):
    from src import annotation as A
    img = cv2.resize(cv2.imread(str(IMG)), (480, 340))
    video = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (480, 340))
    for i in range(30):
        R = cv2.getRotationMatrix2D((240, 170), 2 * np.sin(i / 3), 1.0)
        w.write(cv2.warpAffine(img, R, (480, 340), borderMode=cv2.BORDER_REPLICATE))
    w.release()
    ann = A.new_annotation(video, 10.0, 3.0, annotator="XX")
    ann["events"] = [dict(mode="face", clinical_onset_s=1.0, clinical_end_s=2.0, eeg_onset_s=0.8, side="R",
                          face_term="", notes="secret note")]
    eaf = A.write_eaf(ann, video)
    out, raw = tmp_path / "deid", tmp_path / "raw"
    assert X.main(["--input", str(video), "--annotations", str(eaf), "--out", str(out), "--raw-root", str(raw),
                   "--face-preprocess", "detect", "--pre-s", "0.5", "--post-s", "0.5",
                   "--acq-time", "2026-01-02T03:04:05"]) == 0
    m = list(csv.DictReader(open(raw / "private_mapping.csv")))
    assert len(m) == 1 and m[0]["source_video"] == str(video.resolve())
    P = S.SegmentPaths(out, m[0]["sub"], "seizure", 1, "face")
    Rp = S.SegmentPaths(raw, m[0]["sub"], "seizure", 1, "face")
    assert Rp.video("raw").exists() and not P.video("raw").exists()
    assert not list(out.rglob("*desc-raw*")) and not list(out.rglob("*mapping*"))
    for v in ("fill", "points"):
        j = S.read_json(P.video_json(v))
        assert j["VideoFrameCount"] == 20 and j["Deidentified"] is True and "NotForSharing" not in j
    assert not list(out.rglob("*desc-blur*")) and not list(raw.rglob("*desc-blur*"))   # blur not by default
    d = S.read_landmarks(P.landmarks)
    assert d["landmarks"].shape == (20, 478, 3) and np.isfinite(d["landmarks"]).all()
    assert S.read_blendshapes(P.blendshapes)["blendshapes"].shape == (20, 52)
    rows = S.read_events_tsv(P.events)
    assert [(r["trial_type"], r["onset"]) for r in rows] == [("EEG", "0.300"), ("clinical", "0.500")]
    assert "secret" not in "".join(p.read_text(errors="ignore") for p in out.rglob("*") if p.suffix in (".tsv", ".json"))
    sc = S.read_scans_tsv(P.scans)
    assert all(int(r["acq_time"][:4]) <= 1925 for r in sc)
    assert S.read_scans_tsv(Rp.scans)[0]["acq_time"].startswith("2026-01-02T03:04:05.500")
    assert json.loads((out / "dataset_description.json").read_text())["DatasetType"] == "derivative"


@pytest.mark.skipif(not (MODEL.exists() and IMG.exists()), reason="model file or test image not present")
def test_blur_is_local_unless_requested(tmp_path):
    img = cv2.resize(cv2.imread(str(IMG)), (320, 226))
    video = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 226))
    for _ in range(5):
        w.write(img)
    w.release()
    out, raw = tmp_path / "deid", tmp_path / "raw"
    X.main(["--input", str(video), "--whole", "--out", str(out), "--raw-root", str(raw), "--deid", "fill,blur"])
    assert list(raw.rglob("*desc-blur_video.mp4")) and not list(out.rglob("*desc-blur*"))
    j = S.read_json(next(raw.rglob("*desc-blur_video.json")))
    assert j["NotForSharing"] is True and j["Deidentified"] is False
    scans = (next(out.rglob("*_scans.tsv"))).read_text()
    assert "desc-fill" in scans and "desc-blur" not in scans
    out2, raw2 = tmp_path / "deid2", tmp_path / "raw2"
    X.main(["--input", str(video), "--whole", "--out", str(out2), "--raw-root", str(raw2), "--deid", "blur",
            "--blur-in-export"])
    j = S.read_json(next(out2.rglob("*desc-blur_video.json")))
    assert j["NotForSharing"] is True


MESH = REPO / "models" / "canonical_face_model.obj"


@pytest.mark.skipif(not (MODEL.exists() and IMG.exists() and MESH.exists()), reason="model files or test image not present")
def test_swap_variant_with_warp_backend(tmp_path):
    from src import deid_warp as DW
    dimg, dpts, info = DW.render_donor(MESH, seed=1, size=256)
    cv2.imwrite(str(tmp_path / "donor.png"), dimg)
    np.save(tmp_path / "donor.npy", dpts)
    cfg = tmp_path / "swap.json"
    cfg.write_text(json.dumps(dict(backend="warp", licence_reviewed=True, models=dict(face_mesh=str(MESH)),
                                   donor=str(tmp_path / "donor.png"), donor_landmarks=str(tmp_path / "donor.npy"),
                                   donor_licence="synthetic test donor")))
    img = cv2.resize(cv2.imread(str(IMG)), (320, 226))
    video = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 226))
    for _ in range(5):
        w.write(img)
    w.release()
    out, raw = tmp_path / "deid", tmp_path / "raw"
    for deid in ("swap", "ghost"):                                   # ghost is an alias of swap
        o = out / deid
        assert X.main(["--input", str(video), "--whole", "--out", str(o), "--raw-root", str(raw / deid),
                       "--deid", deid, "--swap-config", str(cfg)]) == 0
        j = S.read_json(next(o.rglob("*desc-swap_video.json")))
        assert j["SwapBackend"] == "warp" and j["SwapDonorLicence"] == "synthetic test donor"
        assert j["Deidentified"] is False and j["DeidentificationStatus"] == "reduced risk, not de-identified"
        assert "ResearchUseOnly" not in j and not list(o.rglob("*desc-ghost*"))
    cfg.write_text(json.dumps(dict(json.loads(cfg.read_text()), licence_reviewed=False)))
    with pytest.raises(SystemExit, match="swap"):
        X.main(["--input", str(video), "--whole", "--out", str(tmp_path / "x"), "--raw-root", str(raw / "x"),
                "--deid", "swap", "--swap-config", str(cfg)])


def test_mosaic_and_gauss_baselines():
    rng = np.random.default_rng(0)
    f = rng.integers(0, 255, (200, 240, 3), np.uint8)
    face = _ring(100, 100, 40)
    w = float(np.ptp(np.asarray(face)[:, 0]))
    for st, n in (("mosaic8", 8), ("mosaic16", 16)):
        b, _ = X.baseline_params(w, st)
        assert b == max(2, round(w / n))
        out = X.mask_faces(f, face, style=st)
        assert np.array_equal(out[0, 0], f[0, 0]) and not np.array_equal(out, f)
    _, (s, k) = X.baseline_params(w, "gauss")
    assert abs(s - w / 6) < 1e-6 and k % 2 == 1 and k >= 6 * s
    g = X.mask_faces(f, face, style="gauss")
    assert np.array_equal(g[0, 0], f[0, 0]) and g[90:110, 90:110].std() < 0.5 * f[90:110, 90:110].std()
    assert X.mask_faces(f, None, style="mosaic8").shape == f.shape      # no subject face: whole frame
    assert "mosaic8" not in X.VARIANTS and "gauss" not in X.VARIANTS     # not export variants
