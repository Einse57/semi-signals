"""Text helper: TrueType when available, OpenCV Hershey fallback otherwise, never a crash."""
import numpy as np
import pytest

from src import textdraw as T


def _clear():
    for f in (T.font_path, T._pil, T._font, T._mask, T._text_size, T._block):
        getattr(f, "cache_clear", lambda: None)()


@pytest.fixture
def no_ttf(monkeypatch):
    monkeypatch.delenv("SEMISIGNALS_FONT", raising=False)
    monkeypatch.delenv("SEMISIGNALS_FONT_BOLD", raising=False)
    monkeypatch.setattr(T, "CANDIDATES", {False: ["/nonexistent/a.ttf"], True: ["/nonexistent/b.ttf"]})
    _clear()
    yield
    _clear()


@pytest.fixture
def no_pillow(monkeypatch):
    _clear()
    monkeypatch.setattr(T, "_pil", lambda: None)
    yield
    _clear()


def _draws(img):
    return int((img.max(axis=2) > 40).sum())


@pytest.mark.parametrize("mode", ["no_ttf", "no_pillow"])
def test_fallback_draws_with_hershey(mode, request):
    request.getfixturevalue(mode)
    assert T.backend() == "hershey"
    img = np.zeros((40, 300, 3), np.uint8)
    x = T.put_text(img, "Detector A: 0.95", (5, 28), 14, (0, 255, 0))
    assert _draws(img) > 50 and x > 60
    w, asc, desc = T.text_size("Detector A: 0.95", 14)
    assert 60 < w < 300 and asc > 5
    T.put_text(img, "bold", (150, 20), 14, (255, 255, 255), bold=True, anchor="lm")
    assert T.fit_text("x" * 200, 80, 14).endswith("...")
    assert T.text_width(T.fit_text("x" * 200, 80, 14), 14) <= 80


def test_truetype_when_available():
    _clear()
    if T.font_path() is None or T._pil() is None:
        pytest.skip("no TrueType font or no Pillow here")
    assert T.backend().startswith("truetype")
    img = np.zeros((40, 300, 3), np.uint8)
    T.put_text(img, "Seizure probability", (5, 5), 14, (230, 230, 230), anchor="lt")
    assert _draws(img) > 50
    vals = np.unique(img[..., 0])
    assert len(vals) > 5                      # anti-aliased edges, not just on/off pixels
    assert T.fit_text("x" * 200, 80, 14).endswith("…")


def test_anchors_and_size():
    _clear()
    for size in (13, 14, 20):
        w, asc, desc = T.text_size("Hands", size)
        assert w > 0 and asc >= size * 0.6
    img = np.zeros((60, 400, 3), np.uint8)
    T.put_text(img, "right", (390, 30), 14, (255, 255, 255), anchor="rm")
    cols = np.where(img.max(axis=(0, 2)) > 40)[0]
    assert cols.max() <= 392 and cols.min() > 300


def test_clipped_at_image_edges():
    _clear()
    img = np.zeros((20, 50, 3), np.uint8)
    for org in [(-30, 10), (40, 10), (10, -5), (10, 30), (500, 500)]:
        T.put_text(img, "outside the image", org, 14, (255, 255, 255))
    T.put_block(img, [("block", (0, 255, 0))], (30, 10), 14)
    T.put_block(img, [("block", (0, 255, 0))], (60, 30), 14)
    T.put_text(img, "", (0, 0), 14, (255, 255, 255))


def test_block_dims_and_draws():
    _clear()
    img = np.full((80, 400, 3), 200, np.uint8)
    T.put_block(img, [("A: 0.95", (0, 255, 0)), ("B: --", (255, 200, 0))], (10, 10), 20)
    assert img[12, 12].max() < 120            # box dimmed
    assert img[70, 390].min() == 200          # outside untouched


def test_fit_to_screen_never_upscales():
    big = np.zeros((700, 2600, 3), np.uint8)
    out = T.fit_to_screen(big, area=(1920, 1040))
    assert out.shape[1] <= 1920 - 16 and out.shape[0] <= 1040 - 60
    small = np.zeros((574, 1600, 3), np.uint8)
    assert T.fit_to_screen(small, area=(1920, 1040)) is small


def test_dpi_awareness_is_safe():
    assert T.make_dpi_aware() in (True, False)
    assert T.make_dpi_aware() in (True, False)     # second call is a cached no-op
