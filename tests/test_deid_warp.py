"""Landmark-driven warp swapper (src/deid_warp.py), its registration behind the licence gate, and the
SFace/YuNet identity check. Tests needing models/canonical_face_model.obj or the OpenCV Zoo files
(scripts/fetch_models.py --face-check) skip when those are absent."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src import deid_check as C
from src import deid_swap as W
from src import deid_warp as DW

REPO = Path(__file__).resolve().parent.parent
MESH = REPO / "models" / "canonical_face_model.obj"
SFACE = REPO / "models" / "face_recognition_sface_2021dec.onnx"
YUNET = REPO / "models" / "face_detection_yunet_2023mar.onnx"
need_mesh = pytest.mark.skipif(not MESH.exists(), reason="run scripts/fetch_models.py --face-check")
need_zoo = pytest.mark.skipif(not (SFACE.exists() and YUNET.exists()), reason="run scripts/fetch_models.py --face-check")


def test_sim_recovers_similarity():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(50, 2))
    th, s, t = 0.4, 1.7, np.array([3.0, -2.0])
    A = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A2, t2 = DW._sim(src, src @ A.T + t)
    assert np.allclose(A2, A) and np.allclose(t2, t)


@need_mesh
def test_mesh_and_donor_render():
    v, tri = DW.load_mesh(MESH)
    assert v.shape == (468, 3) and tri.shape == (898, 3) and tri.max() < 468
    a, pa, ia = DW.render_donor(MESH, seed=3, size=256)
    b, pb, _ = DW.render_donor(MESH, seed=3, size=256)
    c, _, _ = DW.render_donor(MESH, seed=4, size=256)
    assert a.shape == (256, 256, 3) and pa.shape == (468, 2) and "iris_bgr" in ia
    assert np.array_equal(a, b) and np.array_equal(pa, pb) and not np.array_equal(a, c)


def _setup(tmp_path, **extra):
    img, pts, info = DW.render_donor(MESH, seed=2, size=256)
    cv2.imwrite(str(tmp_path / "donor.png"), img)
    np.save(tmp_path / "donor.npy", pts)
    cfg = dict(backend="warp", licence_reviewed=True, models=dict(face_mesh=str(MESH)),
               donor=str(tmp_path / "donor.png"), donor_landmarks=str(tmp_path / "donor.npy"),
               donor_licence="synthetic test donor", iris_bgr=list(info["iris_bgr"]), **extra)
    p = tmp_path / "swap.json"
    p.write_text(json.dumps(cfg))
    return p, pts


def _subject(pts, scale=0.6, off=(100, 60)):
    """Donor landmarks shrunk, moved and stretched sideways into a 320x240 frame; irises at the eye centres."""
    p = pts * scale * np.array([1.15, 1.0]) + np.array(off)
    lm = np.zeros((478, 3), np.float32)
    lm[:468, :2] = p
    for ir, (a, b) in ((DW.R_IRIS, (33, 133)), (DW.L_IRIS, (362, 263))):
        c, r = (p[a] + p[b]) / 2, 0.15 * np.linalg.norm(p[b] - p[a])
        lm[ir[0], :2] = c
        for k, ang in zip(ir[1:], np.linspace(0, 2 * np.pi, 4, endpoint=False)):
            lm[k, :2] = c + r * np.array([np.cos(ang), np.sin(ang)])
    return lm


@need_mesh
@pytest.mark.parametrize("geometry", ["donor", "subject"])
def test_warp_changes_face_only(tmp_path, geometry):
    cfg, pts = _setup(tmp_path, geometry=geometry)
    sw = W.make_swapper(cfg)
    assert sw.name == "warp" and sw.donor_licence == "synthetic test donor"
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (240, 320, 3), np.uint8)
    lm = _subject(pts)
    out = sw.swap(frame, [lm])
    x0, y0 = lm[:468, :2].min(0).astype(int)
    x1, y1 = lm[:468, :2].max(0).astype(int)
    assert out.shape == frame.shape
    inner = (slice(y0 + 15, y1 - 15), slice(x0 + 15, x1 - 15))
    assert np.abs(out[inner].astype(int) - frame[inner]).mean() > 20          # face replaced
    keep = np.ones(frame.shape[:2], bool)
    keep[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8] = False
    assert np.array_equal(out[keep], frame[keep])                             # outside the face untouched
    assert np.array_equal(sw.swap(frame, []), frame) and np.array_equal(sw.swap(frame, [None]), frame)
    sw.reset()
    sw.close()


@need_mesh
def test_donor_geometry_differs_from_subject_geometry(tmp_path):
    cfg, pts = _setup(tmp_path)
    sw = W.make_swapper(cfg)
    lm = _subject(pts)
    t = sw._target(lm)
    oval = sw.oval
    assert np.allclose(t[oval], lm[oval, :2], atol=1e-3)                     # outline kept
    inner = np.setdiff1d(np.arange(468), oval)
    assert np.abs(t[inner] - lm[inner, :2]).max() > 2                         # inner features moved to donor proportions
    with pytest.raises(ValueError):
        DW.WarpSwapper(dict(json.loads(cfg.read_text()), geometry="other"))


@need_mesh
def test_warp_behind_licence_gate(tmp_path):
    cfg, _ = _setup(tmp_path)
    c = json.loads(cfg.read_text())
    c.pop("licence_reviewed")
    cfg.write_text(json.dumps(c))
    with pytest.raises(W.SwapNotConfigured, match="licence"):
        W.make_swapper(cfg)
    c["licence_reviewed"] = True
    c["models"] = {}
    cfg.write_text(json.dumps(c))
    with pytest.raises(W.SwapNotConfigured, match="face_mesh"):
        W.make_swapper(cfg)


@need_zoo
def test_sface_embedder(tmp_path):
    p = tmp_path / "e.json"
    p.write_text(json.dumps(dict(backend="sface", licence_reviewed=True, model=str(SFACE), detector=str(YUNET))))
    e = C.make_embedder(p)
    assert e.same_identity_threshold == 0.363
    blank = np.full((240, 320, 3), 128, np.uint8)
    assert len(e.faces(blank)) == 0
    face = np.array([100, 60, 112, 112, 130, 100, 180, 100, 156, 130, 135, 150, 177, 150, 0.9], np.float32)
    v = e.embed(np.random.default_rng(0).integers(0, 255, (240, 320, 3), np.uint8), face)
    assert v.shape == (128,) and abs(np.linalg.norm(v) - 1) < 1e-5
    y = C.YuNetDetector(YUNET)
    assert y.count(blank) == 0


class _FakeEmb:
    name, same_identity_threshold = "fake", 0.5

    def faces(self, f):
        return [np.array([10, 10, 20, 20] + [0] * 10 + [1.0], np.float32)] if f.mean() > 1 else []

    def embed(self, f, face):
        v = f[10:30, 10:30].astype(np.float32).ravel()[:64] + 1
        return v / np.linalg.norm(v)


def test_identity_similarity_with_fake_embedder(tmp_path):
    rng = np.random.default_rng(0)
    fr = [rng.integers(1, 255, (40, 40, 3), np.uint8) for _ in range(4)]

    def vid(name, fs):
        w = cv2.VideoWriter(str(tmp_path / name), cv2.VideoWriter_fourcc(*"MJPG"), 10, (40, 40))
        for f in fs:
            w.write(f)
        w.release()
        return tmp_path / name
    a = vid("a.avi", fr)
    same = C.identity_similarity(a, a, _FakeEmb())
    assert same["detected"]["n"] == 4 and same["detected"]["mean"] > 0.999
    assert same["detected"]["frac_at_or_above_threshold"] == 1.0
    black = vid("b.avi", [np.zeros_like(f) for f in fr])
    r = C.identity_similarity(a, black, _FakeEmb())
    assert r["detected"]["n"] == 0 and r["same_location"]["n"] == 4
