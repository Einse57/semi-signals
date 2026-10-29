"""Optional face stage: head crop, NaN schema, motion alignment, flag-off parity. No MediaPipe needed."""
import json

import cv2
import numpy as np
import pytest

from src import batch_extract
from src.face_estimator import (BLENDSHAPE_NAMES, FACE_REGIONS, FACE_WIDTH_PAIR, N_BLEND, N_FACE, FaceMotion,
                                FaceStage, empty_face, face_region_speeds, head_crop_bbox, square_crop)


def _kpts(**pts):
    k = np.zeros((17, 2), np.float32)
    s = np.zeros(17, np.float32)
    idx = dict(nose=0, leye=1, reye=2, lear=3, rear=4, lsho=5, rsho=6)
    for n, xy in pts.items():
        k[idx[n]] = xy
        s[idx[n]] = 0.9
    return k, s


def test_crop_from_ears_is_square_and_centred():
    k, s = _kpts(nose=(200, 100), leye=(210, 90), reye=(190, 90), lear=(230, 95), rear=(170, 95))
    b = head_crop_bbox(k, s, 480, 640)
    assert b is not None and b.dtype == np.float32
    assert b[2] - b[0] == pytest.approx(b[3] - b[1])
    assert b[2] - b[0] == pytest.approx(2.0 * 60)          # ears 60 px apart = face width
    vis = np.array([(200, 100), (210, 90), (190, 90), (230, 95), (170, 95)], np.float32)
    assert ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) == pytest.approx(tuple(vis.mean(0)))


def test_crop_fallbacks_and_none():
    k, s = _kpts(nose=(100, 100), lsho=(150, 200), rsho=(50, 200))
    b = head_crop_bbox(k, s, 480, 640)
    assert b[2] - b[0] == pytest.approx(2.0 * 0.4 * 100)   # shoulder fallback
    k, s = _kpts(nose=(100, 100))
    b = head_crop_bbox(k, s, 480, 640, body_bbox=[0, 0, 400, 100, 0.9])
    assert b[2] - b[0] == pytest.approx(2.0 * 0.12 * 400)  # longer bbox side
    assert head_crop_bbox(k, s, 480, 640) is None          # one keypoint, no size
    k, s = _kpts()
    assert head_crop_bbox(k, s, 480, 640, body_bbox=[0, 0, 100, 100]) is None
    k, s = _kpts(nose=(201, 201), leye=(202, 200))
    assert head_crop_bbox(k, s, 480, 640)[2] - head_crop_bbox(k, s, 480, 640)[0] == 48  # min side
    k, s = _kpts(nose=(-500, -500), leye=(-490, -500))
    assert head_crop_bbox(k, s, 480, 640) is None          # entirely outside


def test_square_crop_pads():
    f = np.full((10, 10, 3), 7, np.uint8)
    c = square_crop(f, [-5, -5, 5, 5])
    assert c.shape == (10, 10, 3)
    assert c[:5].sum() == 0 and c[:, :5].sum() == 0 and (c[5:, 5:] == 7).all()


def test_empty_face_schema():
    e = empty_face(3)
    assert e["landmarks"].shape == (3, N_FACE, 3) and e["blendshapes"].shape == (3, N_BLEND)
    assert e["score"].shape == (3,) and e["bbox"].shape == (3, 4) and e["transform"].shape == (3, 4, 4)
    assert all(np.isnan(v).all() and v.dtype == np.float32 for v in e.values())
    assert len(BLENDSHAPE_NAMES) == 52 and BLENDSHAPE_NAMES[0] == "_neutral"
    assert N_FACE == 478 and all(max(v) < N_FACE for v in FACE_REGIONS.values())


def _face(rng):
    lm = rng.normal(0, 30, (N_FACE, 3)).astype(np.float32)
    lm[FACE_WIDTH_PAIR[0], :2] = (-50, 0)
    lm[FACE_WIDTH_PAIR[1], :2] = (50, 0)
    return lm


def test_region_speed_ignores_rigid_head_motion():
    rng = np.random.default_rng(0)
    a = _face(rng)
    th = 0.3
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    b = a.copy()
    b[:, :2] = 1.3 * a[:, :2] @ R.T + (40, -25)              # head turns, moves and comes closer
    sp = face_region_speeds(a, b, 0.1)
    assert set(sp) == set(FACE_REGIONS)
    assert all(v == pytest.approx(0, abs=1e-4) for v in sp.values())
    c = a.copy()
    c[FACE_REGIONS["mouth"], 1] += 10                       # the mouth moves on its own
    sp = face_region_speeds(a, c, 0.1)
    assert sp["mouth"] > 0.5 and sp["mouth"] > 3 * sp["l_eye"]
    m = FaceMotion()
    m.update(0.0, a); m.update(0.1, None); m.update(0.2, a)
    assert len(m.t) == 3 and all(np.isnan(m.speeds[k][1]) for k in FACE_REGIONS)


def test_face_stage_without_model_stays_off(tmp_path):
    st = FaceStage(model_path=tmp_path / "missing.task", on=True)
    assert st.on is False and st.error
    assert st.toggle() is False and st.on is False


# ── batch_extract: NaN schema with a stub face stage, and flag-off parity ──
class _Det:
    def detect(self, frame):
        return [np.array([100, 50, 220, 230, 0.9], np.float32)]


class _Pose:
    def estimate(self, frame, boxes):
        k, s = _kpts(nose=(160, 80), leye=(170, 70), reye=(150, 70), lear=(185, 75), rear=(135, 75),
                     lsho=(190, 120), rsho=(130, 120))
        return [dict(keypoints=k, scores=s, bbox=boxes[0])]


class _Face:
    model_path = "stub.task"
    _transform = True

    def __init__(self):
        self.n = 0

    def estimate(self, frame, k, s, body_bbox=None):
        self.n += 1
        if self.n % 2 == 0:
            return None
        e = empty_face()
        e["landmarks"][:] = 1.0; e["blendshapes"][:] = 0.5; e["score"] = np.float32(1.0)
        e["bbox"][:] = head_crop_bbox(k, s, 240, 320); e["transform"][:] = np.eye(4)
        return e


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "c.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(10):
        w.write(np.full((240, 320, 3), i * 20, np.uint8))
    w.release()
    return p


def test_batch_face_schema_and_flag_off_parity(clip, tmp_path):
    off, on = tmp_path / "off.npz", tmp_path / "on.npz"
    batch_extract.process_video(clip, off, _Det(), _Pose(), None, 5.0, False)
    batch_extract.process_video(clip, on, _Det(), _Pose(), None, 5.0, False, face=_Face())
    a, b = np.load(off), np.load(on)
    assert not any(k.startswith("face_") for k in a.files)
    for k in a.files:                                        # face on adds keys, changes nothing else
        if k != "meta":
            np.testing.assert_array_equal(a[k], b[k])
    ma, mb = json.loads(str(a["meta"])), json.loads(str(b["meta"]))
    assert "face" not in ma and mb["face"]["schema"] == "semi-signals.face/1"
    assert mb["face"]["blendshape_names"] == BLENDSHAPE_NAMES
    F, P = b["body_kpts"].shape[:2]
    assert b["face_landmarks"].shape == (F, P, 478, 3) and b["face_blendshapes"].shape == (F, P, 52)
    assert b["face_scores"].shape == (F, P) and b["face_boxes"].shape == (F, P, 4)
    assert b["face_transform"].shape == (F, P, 4, 4)
    assert b["face_scores"][0, 0] == 1.0 and np.isnan(b["face_scores"][1, 0])   # missed face -> NaN
    assert np.isnan(b["face_landmarks"][1, 0]).all() and np.isfinite(b["face_boxes"][1, 0]).all()
    assert np.isnan(b["face_landmarks"][:, 1:]).all() and np.isnan(b["face_scores"][:, 1:]).all()  # no person
