import numpy as np
import pytest
from src.fusion import align_per_second, noisy_or, LogisticFusion, fuse


def test_noisy_or_basic():
    assert noisy_or(np.array([0.5]), np.array([0.5]))[0] == pytest.approx(0.75)
    assert noisy_or(np.array([0.0]), np.array([0.0]))[0] == 0.0
    assert noisy_or(np.array([1.0]), np.array([0.2]))[0] == 1.0


def test_noisy_or_monotone_and_ge_inputs():
    rng = np.random.default_rng(0)
    a, b = rng.random(1000), rng.random(1000)
    f = noisy_or(a, b)
    assert np.all(f >= np.maximum(a, b) - 1e-12)
    assert np.all(noisy_or(a + 0.0, np.minimum(b + 0.1, 1)) >= f - 1e-12)


def test_video_missing_falls_back_to_ieeg():
    pe = np.array([0.1, 0.9, 0.3]); pv = np.array([np.nan, np.nan, 0.5])
    f = noisy_or(pe, pv)
    assert f[0] == pytest.approx(0.1) and f[1] == pytest.approx(0.9)
    assert f[2] == pytest.approx(1 - 0.7 * 0.5)


def test_ieeg_missing_uses_video_and_both_missing_nan():
    f = noisy_or(np.array([np.nan, np.nan]), np.array([0.4, np.nan]))
    assert f[0] == pytest.approx(0.4) and np.isnan(f[1])


def test_align_per_second_max_and_hold():
    t = np.array([0.1, 0.5, 0.9, 2.2, 6.0])
    p = np.array([0.1, 0.7, 0.2, 0.4, 0.9])
    a = align_per_second(t, p, t0=0.0, n_seconds=7, max_gap=2.0)
    assert a[0] == pytest.approx(0.7)       # max within second
    assert a[1] == pytest.approx(0.2)       # hold last sample (0.1 s gap)
    assert a[2] == pytest.approx(0.4)
    assert a[3] == pytest.approx(0.4)       # held, gap 0.8 s
    assert np.isnan(a[5])                   # gap > 2 s -> missing
    assert a[6] == pytest.approx(0.9)


def test_align_unsorted_input():
    a = align_per_second(np.array([1.5, 0.5]), np.array([0.3, 0.6]), 0.0, 2)
    assert list(a) == pytest.approx([0.6, 0.3])


def test_logistic_missing_video_and_identity():
    lf = LogisticFusion(b0=0, b1=1, b2=0, b3=0)
    pe = np.array([0.2, 0.8])
    assert lf(pe, np.array([np.nan, 0.9])) == pytest.approx(pe)
    assert np.isnan(LogisticFusion()(np.array([np.nan]), np.array([0.5]))[0])


def test_fuse_dispatch():
    assert fuse(np.array([0.5]), np.array([0.5]))[0] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        fuse(np.array([0.5]), np.array([0.5]), rule="bogus")
