"""Replay/loop in the viewer: CausalScorer.reset() must give a second pass identical to the first.
Synthetic keypoints and a tiny LightGBM model; no video, pose models or window needed."""
import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")
from src.batch_extract import MAX_PERSONS                      # noqa: E402
from src.detector_a.features import feature_names              # noqa: E402
from src.run_seizure_demo import CausalScorer, draw_end_overlay  # noqa: E402

FPS = 6.0


def _booster():
    n = len(feature_names())
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, n)); y = (X[:, 0] + 0.5 * X[:, 5] > 0).astype(int)
    return lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 7, "seed": 1},
                     lgb.Dataset(X, y), num_boost_round=20)


def _samples(seconds=25):
    """One moving person; arms start shaking at 10 s."""
    rng = np.random.default_rng(1)
    base = np.array([[320 + 20 * (j % 5), 120 + 22 * j] for j in range(17)], np.float32)
    out = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        bk = np.full((MAX_PERSONS, 17, 2), np.nan, np.float32); bs = np.zeros((MAX_PERSONS, 17), np.float32)
        bb = np.full((MAX_PERSONS, 5), np.nan, np.float32)
        amp = 25.0 if t >= 10 else 2.0
        k = base + rng.normal(scale=1.0, size=base.shape)
        k[5:11, 0] += amp * np.sin(2 * np.pi * 2.0 * t)
        bk[0], bs[0], bb[0] = k, 0.9, [250, 100, 450, 500, 0.9]
        hk = np.full((2, 21, 2), np.nan, np.float32); hs = np.zeros((2, 21), np.float32)
        out.append(dict(t=t, body_kpts=bk, body_scores=bs, body_boxes=bb, sel=0, hand_kpts=hk, hand_scores=hs,
                        patches=np.zeros((15, 32, 32, 3), np.uint8), vkp=np.zeros((15, 3), np.float32)))
    return out


def _run(sc, samples):
    rows = []
    for s in samples:
        sc.add_sample(s)
        sc.score_due(s["t"], on_row=lambda r: rows.append((r["sec"], r["pa"], r["pv"], r["alarm"],
                                                           tuple(d["text"] for d in r["drivers"]))))
    return rows, {k: list(v) for k, v in sc.hist.items() if k in ("tA", "pA", "pAs", "alA")}, len(sc.mot_t)


def test_replay_after_reset_is_identical():
    sc = CausalScorer(_booster(), thr=0.6, fps=FPS, fnames=feature_names(), explain=True)
    samples = _samples()
    first = _run(sc, samples)
    assert len(first[0]) == 24 and first[2] > 0
    # without a reset the state carries over (next second is already past the clip)
    assert sc.next_second == 25 and len(sc.hist["pA"]) == 24
    sc.reset()
    assert sc.next_second == 1 and not sc.hist["pA"] and not sc.mot_t and not any(sc.buf.values())
    assert sc.drivers == [] and len(sc.explainer.hist) == 0
    second = _run(sc, samples)
    np.testing.assert_equal(first[1], second[1])
    assert first[0] == second[0] and first[2] == second[2]


def test_reset_with_start_offset():
    sc = CausalScorer(None, thr=0.95, fps=FPS, explain=False, start=12.4)
    assert sc.next_second == 13 and sc.next_sample == 12.4 and sc.explainer is None
    sc.reset(start=0.0)
    assert sc.next_second == 1 and sc.next_sample == 0.0


def test_end_overlay_keeps_frame_size():
    img = np.full((540, 1600, 3), 60, np.uint8)
    out = draw_end_overlay(img)
    assert out.shape == img.shape and not np.array_equal(out, img)
    assert np.array_equal(out[:100], img[:100])          # graphs above the banner untouched
