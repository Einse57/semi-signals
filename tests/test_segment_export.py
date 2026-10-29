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
    out = X.blur_face(f, _ring(50, 60), others=[_ring(150, 60)])
    assert not np.array_equal(out[60, 50], f[60, 50]) and not np.array_equal(out[60, 150], f[60, 150])
    assert np.array_equal(out[5, 100], f[5, 100])                     # outside both faces unchanged
    fill = X.blur_face(f, _ring(50, 60), style="fill")
    assert np.ptp(fill[50:70, 40:60].reshape(-1, 3), 0).max() == 0     # solid
    whole = X.blur_face(f, None)                                       # no subject face: whole frame
    assert not np.array_equal(whole[5, 100], f[5, 100])
    assert np.array_equal(X.blur_face(f, None, missing="none"), f)


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
IMG = Path("/workspace/vsvig/facetest/business-person.png")


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
    for v in ("blur", "points"):
        j = S.read_json(P.video_json(v))
        assert j["VideoFrameCount"] == 20 and j["Deidentified"] is True
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
