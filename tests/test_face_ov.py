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
from src.face_ov.bundle import BUNDLE_FILES, unpack_task

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
        for n in BUNDLE_FILES:
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


# ---- multi-face, blendshapes off, transformation matrix, backend selection (no model needed) ---------
def _rot(ax, ay, az):
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _angle_deg(a, b):
    return math.degrees(math.acos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))


def test_rect_iou():
    a = G.Rect(0.5, 0.5, 0.2, 0.2, 0.0)
    assert G.rect_iou(a, a) == pytest.approx(1.0)
    assert G.rect_iou(a, G.Rect(0.9, 0.9, 0.1, 0.1, 0.0)) == 0.0
    assert G.rect_iou(a, G.Rect(0.6, 0.5, 0.2, 0.2, 1.0)) == pytest.approx(1 / 3)   # rotation ignored


def test_weighted_orthogonal_procrustes_recovers_similarity():
    from src.face_ov.face_geometry import solve_weighted_orthogonal
    rng = np.random.default_rng(3)
    src = rng.normal(size=(3, 50))
    R, t, s = _rot(0.3, -0.5, 0.2), np.array([1.0, -2.0, 3.0]), 1.7
    m = solve_weighted_orthogonal(src, s * R @ src + t[:, None], rng.uniform(0.1, 1, 50))
    np.testing.assert_allclose(m[:3, :3], s * R, atol=1e-9)
    np.testing.assert_allclose(m[:3, 3], t, atol=1e-9)
    np.testing.assert_allclose(m[3], [0, 0, 0, 1])


def test_transform_matrix_shape_orthonormal_and_pose():
    from src.face_ov.face_geometry import FaceGeometry
    rng = np.random.default_rng(0)
    g = FaceGeometry.__new__(FaceGeometry)                 # synthetic canonical mesh, no model bundle
    g.canonical, g.weights, g.fov, g.near = rng.normal(0, 5, (3, 468)), np.full(468, 1 / 468), 63.0, 1.0
    w, h = 640, 480
    for ang in ((0, 0, 0), (0.17, 0.35, 0.09), (-0.44, 0.26, -0.17)):
        R = _rot(*ang)
        p = R @ g.canonical
        lm = np.stack([0.5 + p[0] / 60, 0.5 - p[1] / 60 * w / h, -p[2] / 60], 1)
        m = g.transform(lm, w, h)
        assert m.shape == (4, 4) and m.dtype == np.float32
        np.testing.assert_allclose(m[3], [0, 0, 0, 1])
        rn = m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0)
        np.testing.assert_allclose(rn.T @ rn, np.eye(3), atol=1e-5)
        assert np.linalg.det(rn) == pytest.approx(1.0, abs=1e-5)
        assert _angle_deg(rn, R) < 1.0
        assert m[2, 3] < 0                                 # face in front of the camera (-z)
    assert g.transform(np.full((478, 3), 0.5), w, h) is None   # degenerate landmarks


def _stub_landmarker(num_faces, video=True, blendshapes=False):
    """FaceLandmarkerOV with its three models replaced by stubs: landmarks fill the ROI."""
    from src.face_ov.landmarker import FaceLandmarkerOV
    est = object.__new__(FaceLandmarkerOV)
    est.video, est.num_faces, est.smooth, est.blendshapes = video, num_faces, False, blendshapes
    est.min_track, est.min_pres, est.geometry = 0.5, 0.5, None
    est.reset()
    grid = np.random.default_rng(0).uniform(48, 208, (478, 2))
    raw = np.concatenate([grid, np.zeros((478, 1))], 1)
    est.det_calls, est.dets = 0, []

    def detect_rois(rgb, max_faces=None):
        est.det_calls += 1
        return list(est.dets[:max_faces or est.num_faces])
    est.detect_rois = detect_rois
    est.landmark_input = lambda rgb, roi: roi
    est.run_landmarks = lambda roi: (raw, 5.0)
    est.run_blendshapes = lambda x: np.full(52, 0.25, np.float32)
    return est


def test_multi_face_detect_tracks_each_face_and_skips_detector():
    rgb = np.zeros((480, 640, 3), np.uint8)
    est = _stub_landmarker(num_faces=8)
    centres = [(0.15, 0.3), (0.5, 0.3), (0.85, 0.3), (0.3, 0.75), (0.7, 0.75)]
    est.dets = [(0.9, G.Rect(x, y, 0.15, 0.2, 0.0)) for x, y in centres]
    faces = est.detect(rgb, 0)
    assert len(faces) == 5 and est.det_calls == 1
    for f, (x, y) in zip(faces, centres):
        assert f.landmarks.shape == (478, 3) and f.blendshapes.shape == (52,)
        assert np.isnan(f.blendshapes).all() and f.transform is None and not f.tracked
        assert abs(f.landmarks[:, 0].mean() - x) < 0.03 and abs(f.landmarks[:, 1].mean() - y) < 0.03
    assert len(est._prev_rois) == 5
    # fewer tracked faces than num_faces: the detector runs; overlapping detections replace tracks
    faces = est.detect(rgb, 33)
    assert len(faces) == 5 and est.det_calls == 2 and not any(f.tracked for f in faces)
    # num_faces reached: the detector is skipped and every face comes from its own tracked rect
    est2 = _stub_landmarker(num_faces=2)
    est2.dets = est.dets[:2]
    est2.detect(rgb, 0)
    faces = est2.detect(rgb, 33)
    assert est2.det_calls == 1 and len(faces) == 2 and all(f.tracked for f in faces)
    # a new face away from the tracked one is added alongside it
    est3 = _stub_landmarker(num_faces=2)
    est3.dets = est.dets[:1]
    est3.detect(rgb, 0)
    est3.dets = est.dets[1:3]
    faces = est3.detect(rgb, 33)
    assert est3.det_calls == 2 and [f.tracked for f in faces] == [True, False]


def test_multi_face_clipped_and_image_mode_untracked():
    rgb = np.zeros((480, 640, 3), np.uint8)
    est = _stub_landmarker(num_faces=3, video=False, blendshapes=True)
    est.dets = [(0.9, G.Rect(0.1 + 0.2 * i, 0.5, 0.1, 0.1, 0.0)) for i in range(5)]
    for _ in range(2):
        faces = est.detect(rgb)
        assert len(faces) == 3 and not any(f.tracked for f in faces)
        assert all(np.allclose(f.blendshapes, 0.25) for f in faces)
    assert est.det_calls == 2 and est._prev_rois == []
    assert est.process(rgb).landmarks.shape == (478, 3)


def test_smoothing_with_several_faces_rejected():
    pytest.importorskip("openvino")
    from src.face_ov.landmarker import FaceLandmarkerOV
    with pytest.raises(ValueError):
        FaceLandmarkerOV("missing.task", num_faces=2, smooth=True)


def test_face_backend_selection_and_cli_flags():
    import argparse
    from src.face_estimator import FACE_BACKENDS, FaceEstimator, add_face_backend_args
    assert FACE_BACKENDS[0] == "mediapipe"
    with pytest.raises(ValueError):
        FaceEstimator(backend="onnx")
    ap = argparse.ArgumentParser()
    add_face_backend_args(ap)
    a = ap.parse_args([])
    assert (a.face_backend, a.face_device) == ("mediapipe", "CPU")
    a = ap.parse_args(["--face-backend", "openvino", "--face-device", "gpu"])
    assert (a.face_backend, a.face_device) == ("openvino", "GPU")
    with pytest.raises(SystemExit):
        ap.parse_args(["--face-device", "TPU"])
    from src import segment_export
    a = segment_export.parse_args(["--input", "x.mp4", "--out", "o", "--raw-root", "r", "--face-backend", "openvino",
                                   "--face-device", "npu"])
    assert segment_export.face_backend_kwargs(a) == dict(backend="openvino", device="NPU")
    assert segment_export.face_backend_kwargs(segment_export.parse_args(
        ["--input", "x.mp4", "--out", "o", "--raw-root", "r"])) == dict(backend="mediapipe", device="CPU")
    assert segment_export.face_backend_kwargs(argparse.Namespace()) == dict(backend="mediapipe", device="CPU")


def test_openvino_estimator_multi_face_blendshapes_off_and_matrix():
    _need_model()
    from src.face_estimator import FaceEstimator
    from src.face_ov import FaceLandmarkerOV
    ov = FaceLandmarkerOV(TASK, "CPU", video=False, num_faces=8, blendshapes=False, transform=True)
    assert "bs" not in ov._models and ov.geometry is not None
    assert ov.detect(np.full((240, 320, 3), 127, np.uint8)) == []
    ov.close()
    est = FaceEstimator(model_path=str(TASK), backend="openvino", num_faces=8, blendshapes=False)
    assert est.all_faces(np.full((240, 320, 3), 127, np.uint8)) == []
    est.close()
