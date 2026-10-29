"""Face-swap scaffold (src/deid_swap.py) and check harness (src/deid_check.py): no models involved."""
import json

import numpy as np
import pytest

from src import deid_check as C
from src import deid_swap as W


def test_swap_needs_config(monkeypatch):
    monkeypatch.delenv(W.ENV_CONFIG, raising=False)
    with pytest.raises(W.SwapNotConfigured, match="licence audit"):
        W.make_swapper(None)


def test_swap_config_problems_listed(tmp_path):
    cfg = dict(backend="ghost", models=dict(generator=str(tmp_path / "G.pth")))
    probs = W.check_config(cfg)
    assert any("licence_reviewed" in p for p in probs)
    assert any("models.code" in p for p in probs) and any("G.pth" in p for p in probs)
    assert any("research_use_acknowledged" in p for p in probs)
    assert any("donor_licence" in p for p in probs)
    assert W.check_config(dict(backend="other")) and "unknown" in W.check_config(dict(backend="other"))[0]


@pytest.mark.parametrize("backend", W.RESEARCH_ONLY)
def test_research_backends_need_acknowledgement(tmp_path, backend):
    files = {k: tmp_path / f"{k}.bin" for k in W.REQUIRED[backend]}
    for f in files.values():
        f.write_bytes(b"x")
    donor = tmp_path / "donor.png"
    donor.write_bytes(b"x")
    cfg = dict(backend=backend, licence_reviewed=True, models={k: str(v) for k, v in files.items()},
               donor=str(donor), donor_licence="synthetic, test")
    probs = W.check_config(cfg)
    assert len(probs) == 1 and "research_use_acknowledged" in probs[0]
    assert W.check_config(dict(cfg, research_use_acknowledged=True)) == []
    assert "warp" not in W.RESEARCH_ONLY and W.DEID_STATUS == "reduced risk, not de-identified"


def test_ghost_alignment_matches_insightface_template():
    from src import deid_ghost as G
    tpl = G._SRC112[2] * 2                                       # frontal template, 224 px
    A = np.array([[0.8, -0.3], [0.3, 0.8]])
    kps = (tpl @ A.T + [40, 25]).astype(np.float32)              # similarity transform of the template
    M = G.estimate_norm(kps)
    np.testing.assert_allclose(np.c_[kps, np.ones(5)] @ M.T, tpl, atol=1e-3)
    lm = np.zeros((478, 3), np.float32)
    lm[[33, 133, 362, 263, 1, 61, 291], :2] = [[0, 0], [2, 0], [8, 0], [10, 0], [5, 5], [2, 9], [8, 9]]
    np.testing.assert_allclose(G.five_points(lm), [[1, 0], [9, 0], [5, 5], [2, 9], [8, 9]])


def test_arcface_embedder_needs_files_and_acknowledgement(tmp_path):
    p = tmp_path / "e.json"
    p.write_text(json.dumps(dict(backend="arcface", licence_reviewed=True, model=str(tmp_path / "none.onnx"))))
    with pytest.raises(C.EmbedderNotConfigured, match="model"):
        C.make_embedder(p)
    for k in ("m.onnx", "d.onnx"):
        (tmp_path / k).write_bytes(b"x")
    p.write_text(json.dumps(dict(backend="arcface", licence_reviewed=True, model=str(tmp_path / "m.onnx"),
                                 detector=str(tmp_path / "d.onnx"))))
    with pytest.raises(C.EmbedderNotConfigured, match="research"):
        C.make_embedder(p)


def test_embedder_needs_config(tmp_path):
    with pytest.raises(C.EmbedderNotConfigured):
        C.make_embedder(None)
    p = tmp_path / "e.json"
    p.write_text(json.dumps(dict(backend="x", licence_reviewed=False)))
    with pytest.raises(C.EmbedderNotConfigured, match="licence"):
        C.make_embedder(p)


def test_pearson_and_cosine():
    a = np.arange(10.0)
    assert C.pearson(a, 2 * a + 1) == pytest.approx(1.0)
    assert C.pearson(a, -a) == pytest.approx(-1.0)
    b = a.copy()
    b[3] = np.nan
    assert C.pearson(a, b) == pytest.approx(1.0)
    assert C.pearson(np.ones(5), a[:5]) is None
    assert C.cosine(np.array([1, 0]), np.array([0, 1])) == pytest.approx(0.0)


def test_hit_rate_with_fake_detector(tmp_path):
    import cv2
    v = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    for i in range(6):
        w.write(np.full((32, 32, 3), 255 if i % 2 else 0, np.uint8))
    w.release()

    class Fake:
        def count(self, f):
            return int(f.mean() > 128)

    r = C.hit_rate(v, Fake())
    assert r["frames"] == 6 and r["frames_with_face"] == 3 and r["hit_rate"] == pytest.approx(0.5)


def test_region_motion_identical_sequences_correlate():
    rng = np.random.default_rng(0)
    base = rng.normal(200, 40, (478, 3)).astype(np.float32)
    seq = [base + rng.normal(0, 1 + (t % 3), (478, 3)).astype(np.float32) for t in range(12)]
    m = C.region_motion(seq)
    assert m.shape[0] == 12 and np.isnan(m[0]).all()
    r = C.pearson(m[1:, 0], m[1:, 0])
    assert r == pytest.approx(1.0)
