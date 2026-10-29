"""--face-preprocess: alignment geometry (no model) and, if the model file is present, a real-image check."""
import math
from pathlib import Path

import numpy as np
import pytest

from src import face_preprocess as fp

REPO = Path(__file__).resolve().parent.parent


def _mesh(cx=200.0, cy=150.0, half=40.0, roll=25.0):
    """Synthetic 478-point 'mesh': a ring, with 33/263 as the outer eye corners at angle `roll`."""
    a = np.linspace(0, 2 * np.pi, 478, endpoint=False)
    lm = np.stack([cx + half * np.cos(a), cy + half * np.sin(a), np.zeros(478)], 1).astype(np.float32)
    r = math.radians(roll)
    for i, s in ((33, -1), (263, 1)):
        lm[i, :2] = (cx + s * 0.6 * half * math.cos(r), cy + s * 0.6 * half * math.sin(r))
    return lm


@pytest.mark.parametrize("roll", [-30.0, 0.0, 12.0, 40.0])
def test_align_levels_eye_line(roll):
    lm = _mesh(roll=roll)
    assert fp.roll_deg(lm) == pytest.approx(roll, abs=1e-3)
    A = fp.align_affine(lm, 256)
    xy = lm[:, :2] @ A[:, :2].T + A[:, 2]
    crop = np.concatenate([xy, lm[:, 2:]], 1)
    assert fp.roll_deg(crop) == pytest.approx(0.0, abs=1e-3)          # eye line level in the crop
    assert np.allclose(xy[:468].mean(0), 128, atol=1e-3)              # centred
    assert crop[33, 0] < crop[263, 0]                                  # not flipped


def test_map_back_inverts():
    lm = _mesh(roll=17.0)
    lm[:, 2] = np.linspace(-5, 5, 478)
    A = fp.align_affine(lm, 256)
    s = math.hypot(A[0, 0], A[0, 1])
    crop = np.concatenate([lm[:, :2] @ A[:, :2].T + A[:, 2], lm[:, 2:] * s], 1)
    assert np.allclose(fp.map_back(crop, A), lm, atol=1e-3)
    box = fp.crop_box_of(A, 256)
    assert box[0] < lm[:, 0].min() and box[2] > lm[:, 0].max()


class _FakeEst:
    """Returns the synthetic mesh for any crop that contains it, in the coordinates asked for."""

    def __init__(self, lm):
        self.lm, self.calls = lm, []

    def estimate_crop(self, frame, box):
        self.calls.append(("crop", tuple(np.round(box, 1))))
        c = self.lm[:, :2].mean(0)
        if box[0] <= c[0] <= box[2] and box[1] <= c[1] <= box[3]:
            return dict(landmarks=self.lm.copy(), blendshapes=np.zeros(52, np.float32), score=np.float32(1),
                        transform=np.full((4, 4), np.nan, np.float32), bbox=np.asarray(box, np.float32))
        return None

    def detect_image(self, img, offset=(0.0, 0.0)):
        self.calls.append(("image", img.shape))
        return None                                    # aligned pass "fails": keeps the search result

    def estimate(self, frame, kp, sc, body_bbox=None):
        self.calls.append(("head",))
        return dict(landmarks=self.lm.copy())


def test_modes_route_calls():
    frame = np.zeros((300, 400, 3), np.uint8)
    lm = _mesh()
    est = _FakeEst(lm)
    assert fp.FacePreprocessor(est, "none").run(frame)["landmarks"] is not None
    assert est.calls[-1][0] == "crop"
    assert fp.FacePreprocessor(est, "coco-head").run(frame) is None    # no body keypoints
    pre = fp.FacePreprocessor(est, "detect")
    r = pre.run(frame)
    assert r is not None and pre._track is not None
    with pytest.raises(ValueError):
        fp.FacePreprocessor(est, "bogus")
    assert fp.DEFAULT_MODE == "none"


MODEL = REPO / "models" / "face_landmarker.task"
IMG = Path("/workspace/vsvig/facetest/business-person.png")


@pytest.mark.skipif(not (MODEL.exists() and IMG.exists()), reason="model file or test image not present")
def test_detect_mode_real_image_rotated():
    import cv2
    from src.face_estimator import FaceEstimator
    img = cv2.imread(str(IMG))
    h, w = img.shape[:2]
    R = cv2.getRotationMatrix2D((w / 2, h / 2), 25, 1.0)
    rot = cv2.warpAffine(img, R, (w, h))
    est = FaceEstimator(model_path=str(MODEL))
    try:
        r_none = fp.FacePreprocessor(est, "none").run(img)
        pre = fp.FacePreprocessor(est, "detect")
        r_det = pre.run(rot)
        assert r_none is not None and r_det is not None
        # the image was rotated 25 deg counter-clockwise -> the eye line tilts by about -25 deg
        assert fp.roll_deg(r_det["landmarks"]) == pytest.approx(fp.roll_deg(r_none["landmarks"]) - 25, abs=6)
        assert "affine" in r_det
        # landmarks mapped back into the rotated frame match the original ones rotated
        exp = r_none["landmarks"][:, :2] @ R[:, :2].T + R[:, 2]
        assert np.median(np.linalg.norm(r_det["landmarks"][:, :2] - exp, axis=1)) < 0.03 * max(h, w)
        r2 = pre.run(rot)                               # tracked frame
        assert r2 is not None and "affine" in r2
    finally:
        est.close()
