"""OpenVINO face backend (src/face_ov): anchors, decode, NMS and ROI math without any model, plus parity
checks that skip when the model bundle, OpenVINO, MediaPipe or a test clip is absent."""
import math
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.face_ov import geometry as G
from src.face_ov.bundle import MODEL_FILES, unpack_task

TASK = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def test_anchors_short_range():
    a = G.ssd_anchors(**G.SHORT_RANGE_ANCHORS)
    assert a.shape == (896, 4)
    # stride 8 layer: 16x16 cells x 2 anchors; then 8x8 cells x 6 anchors (three stride-16 layers merged)
    np.testing.assert_allclose(a[0], [0.5 / 16, 0.5 / 16, 1, 1])
    np.testing.assert_allclose(a[1], a[0])
    np.testing.assert_allclose(a[2], [1.5 / 16, 0.5 / 16, 1, 1])
    np.testing.assert_allclose(a[512], [0.5 / 8, 0.5 / 8, 1, 1])
    np.testing.assert_allclose(a[512:518], np.repeat(a[512:513], 6, 0))
    np.testing.assert_allclose(a[-1], [7.5 / 8, 7.5 / 8, 1, 1])


def test_decode_xywh_and_keypoints():
    anchors = np.array([[0.25, 0.5, 1, 1], [0.75, 0.75, 1, 1]], np.float32)
    raw = np.zeros((2, 16), np.float32)
    raw[0, :4] = [12.8, -6.4, 32.0, 64.0]            # dx = 0.1, dy = -0.05, w = 0.25, h = 0.5
    raw[0, 4:6] = [-12.8, 0.0]                       # first keypoint: x - 0.1
    scores = np.array([3.0, -3.0], np.float32)       # sigmoid 0.95 and 0.05
    s, b, k = G.decode_detections(raw, scores, anchors, min_score=0.5)
    assert s.shape == (1,) and s[0] == pytest.approx(1 / (1 + math.exp(-3)))
    np.testing.assert_allclose(b[0], [0.35 - 0.125, 0.45 - 0.25, 0.35 + 0.125, 0.45 + 0.25], atol=1e-6)
    np.testing.assert_allclose(k[0, 0], [0.15, 0.5], atol=1e-6)
    np.testing.assert_allclose(k[0, 1], [0.25, 0.5], atol=1e-6)


def test_decode_score_clipping_no_overflow():
    anchors = np.zeros((1, 4), np.float32)
    s, _, _ = G.decode_detections(np.zeros((1, 16)), np.array([1e6]), anchors, 0.5)
    assert np.isfinite(s).all() and s[0] == pytest.approx(1.0)


def test_weighted_nms_merges_overlaps():
    boxes = np.array([[0, 0, 1, 1], [0.1, 0, 1.1, 1], [5, 5, 6, 6]], np.float32)
    kps = np.zeros((3, 6, 2), np.float32)
    kps[1] += 1.0
    scores = np.array([0.9, 0.6, 0.8], np.float32)
    out = G.weighted_nms(scores, boxes, kps, iou_thr=0.5)
    assert len(out) == 2
    s0, b0, k0 = out[0]
    assert s0 == pytest.approx(0.9)                  # top score kept
    np.testing.assert_allclose(b0[0], 0.6 * 0.1 / 1.5, atol=1e-6)   # score-weighted xmin
    np.testing.assert_allclose(k0[0], [0.4, 0.4], atol=1e-6)
    assert out[1][0] == pytest.approx(0.8)
    assert len(G.weighted_nms(scores, boxes, kps, 0.5, max_out=1)) == 1


def test_letterbox_projection_roundtrip():
    w, h = 360, 288
    side, px, py = G.letterbox_params(w, h)
    assert side == 360 and px == 0 and py == pytest.approx((1 - 288 / 360) / 2)
    # a point at image pixel (90, 72) -> tensor-normalised -> back
    u, v = (90 / side) + px, (72 / side) + py
    b, k = G.project_detection(np.array([u, v, u, v]), np.array([[u, v]] * 6), w, h)
    np.testing.assert_allclose(k[0], [90 / w, 72 / h], atol=1e-6)


def test_rotation_and_rect_transform():
    # eyes level -> 0; right eye below left eye on screen (y down) -> positive (clockwise) rotation
    assert G.rotation_from_points((0.4, 0.5), (0.6, 0.5), 100, 100) == pytest.approx(0.0)
    assert G.rotation_from_points((0.4, 0.5), (0.6, 0.7), 100, 100) == pytest.approx(math.pi / 4)
    assert G.normalize_radians(3 * math.pi) == pytest.approx(-math.pi)
    r = G.transform_rect(G.Rect(0.5, 0.5, 0.2, 0.1, 0.0), 200, 100, 1.5, 1.5, square_long=True)
    # long side in px = max(40, 10) = 40 -> 0.2 of width, 0.4 of height, then x1.5
    assert (r.width, r.height) == (pytest.approx(0.3), pytest.approx(0.6))
    r = G.transform_rect(G.Rect(0.5, 0.5, 0.2, 0.2, 0.0), 100, 100, 1.0, 1.0, shift_x=0.5)
    assert r.x_center == pytest.approx(0.6)


@pytest.mark.parametrize("rot", [0.0, 0.3, -1.1])
def test_crop_and_unprojection_are_inverse(rot):
    w, h = 320, 240
    roi = G.Rect(0.45, 0.55, 0.3, 0.4, rot)
    M = G.roi_warp_matrix(roi, w, h, 256, 256)
    pts = np.array([[100.0, 120.0], [150.0, 140.0], [130.0, 90.0]])
    t = cv2.perspectiveTransform(pts[None].astype(np.float32), M)[0]
    raw = np.concatenate([t, np.full((3, 1), 25.6)], 1)
    lm = G.project_landmarks(raw, roi, w, h)
    np.testing.assert_allclose(lm[:, 0] * w, pts[:, 0], atol=1e-2)
    np.testing.assert_allclose(lm[:, 1] * h, pts[:, 1], atol=1e-2)
    # z: raw z / 256 scaled by the projected unit length (roi width in px, in image-width units)
    W = roi.width * w
    zs = math.hypot(W * math.cos(rot) / w, W * math.sin(rot) / h)
    np.testing.assert_allclose(lm[:, 2], 0.1 * zs, rtol=1e-5)


def test_landmarks_to_roi_square_and_rotation():
    lm = np.zeros((478, 3), np.float32)
    lm[:, 0] = np.linspace(0.4, 0.6, 478)
    lm[:, 1] = 0.5
    lm[:, 1] += np.linspace(-0.05, 0.05, 478)
    lm[33] = [0.45, 0.5, 0]
    lm[263] = [0.55, 0.5, 0]
    r = G.landmarks_to_roi(lm, 200, 100)
    assert r.rotation == pytest.approx(0.0)
    assert r.width * 200 == pytest.approx(r.height * 100)        # square in pixels
    assert r.width * 200 == pytest.approx(1.5 * 0.2 * 200, rel=1e-4)


def test_blendshape_input_subset_in_pixels():
    lm = np.random.default_rng(0).random((478, 3)).astype(np.float32)
    x = G.blendshape_input(lm, 640, 480)
    assert x.shape == (1, 146, 2)
    np.testing.assert_allclose(x[0, 0], lm[0, :2] * [640, 480], rtol=1e-6)
    np.testing.assert_allclose(x[0, -1], lm[477, :2] * [640, 480], rtol=1e-6)
    assert len(set(G.BLENDSHAPE_SUBSET.tolist())) == 146


def test_one_euro_first_frame_passthrough_and_static_smoothing():
    f = G.OneEuroLandmarks(**G.ONE_EURO)
    lm = np.random.default_rng(1).random((478, 3)).astype(np.float32)
    np.testing.assert_allclose(f.apply(lm, 0, 100, 100), lm, rtol=1e-6)
    jitter = lm + 1e-4
    out = f.apply(jitter, 33_333, 100, 100)
    assert np.all(np.abs(out - lm) < np.abs(jitter - lm))       # pulled towards the previous value
    assert np.array_equal(f.apply(jitter, 33_333, 100, 100), jitter)   # non-increasing timestamp: as is


def test_unpack_task_checks_sha(tmp_path):
    task = tmp_path / "x.task"
    with zipfile.ZipFile(task, "w") as z:
        for n in MODEL_FILES:
            z.writestr(n, b"tflite " + n.encode())
    paths = unpack_task(task, out_dir=tmp_path / "cache")
    assert all(p.read_bytes().startswith(b"tflite") for p in paths.values())
    with pytest.raises(ValueError):
        unpack_task(task, expected_sha256="0" * 64, out_dir=tmp_path / "cache")


def _need_model():
    if not TASK.is_file():
        pytest.skip("models/face_landmarker.task absent (scripts/fetch_models.py --face)")
    pytest.importorskip("openvino")


def test_parity_raw_tensors_vs_tflite():
    _need_model()
    litert = pytest.importorskip("ai_edge_litert.interpreter")
    from src.face_ov import FaceLandmarkerOV
    est = FaceLandmarkerOV(TASK, "CPU")
    rng = np.random.default_rng(0)
    cases = (("face_detector.tflite", rng.uniform(-1, 1, (1, 128, 128, 3)), est.run_detector),
             ("face_landmarks_detector.tflite", rng.uniform(0, 1, (1, 256, 256, 3)), est.run_landmarks))
    for name, x, fn in cases:
        it = litert.Interpreter(model_path=str(est.paths[name]))
        it.allocate_tensors()
        it.set_tensor(it.get_input_details()[0]["index"], x.astype(np.float32))
        it.invoke()
        ref = [it.get_tensor(o["index"]) for o in it.get_output_details()]
        got = fn(x.astype(np.float32))
        a = np.concatenate([np.ravel(r) for r in ref[:2]])
        b = np.concatenate([np.ravel(g) for g in got[:2]])
        assert np.allclose(a, b, atol=1e-2 * max(1.0, np.abs(a).max()))


def test_parity_pipeline_vs_mediapipe():
    _need_model()
    mp = pytest.importorskip("mediapipe")
    clip = os.environ.get("FACE_OV_CLIP")
    if not clip or not Path(clip).is_file():
        pytest.skip("set FACE_OV_CLIP to a short video with one face")
    from mediapipe.tasks.python import BaseOptions, vision
    from src.face_ov import FaceLandmarkerOV
    cap = cv2.VideoCapture(clip)
    frames = []
    while len(frames) < 20:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(TASK)),
                                        running_mode=vision.RunningMode.VIDEO, num_faces=1,
                                        output_face_blendshapes=True)
    est = FaceLandmarkerOV(TASK, "CPU", video=True)
    errs, bsd = [], []
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        for i, f in enumerate(frames):
            t = i * 40
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=f), t)
            o = est.process(f, t)
            assert (o is None) == (not r.face_landmarks)
            if o is None:
                continue
            h, w = f.shape[:2]
            ref = np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]])
            iod = np.linalg.norm(ref[468] - ref[473])
            errs.append(np.linalg.norm(ref - o.landmarks[:, :2] * [w, h], axis=1).mean() / iod)
            bs = np.array([c.score for c in sorted(r.face_blendshapes[0], key=lambda c: c.index)])
            bsd.append(np.abs(bs - o.blendshapes).mean())
    assert errs and np.mean(errs) < 0.01 and np.mean(bsd) < 0.01
