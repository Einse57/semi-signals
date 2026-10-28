"""Anti-aliased TrueType text for OpenCV (BGR uint8) images.

put_text() draws with Pillow and a system font (Segoe UI on Windows, DejaVu Sans on Linux,
Arial/Helvetica on macOS). Each (text, size, weight) is rasterised once into a cached alpha
mask; drawing is then a small numpy blend into the region under the text, so a frame with a
few dozen labels costs well under a millisecond after the first frames.

Without Pillow or a TrueType font it falls back to the OpenCV Hershey font with LINE_AA,
scaled to roughly the same height. SEMISIGNALS_FONT / SEMISIGNALS_FONT_BOLD (paths to .ttf
files) override the font search.

Anchors (Pillow style, two letters): horizontal l/m/r (left, middle, right) and vertical
t/m/s/b (top, middle, baseline, bottom). The default "ls" matches cv2.putText, whose org is the
left end of the baseline.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

import cv2
import numpy as np

_WIN_FONTS = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
CANDIDATES = {
    False: [os.path.join(_WIN_FONTS, "segoeui.ttf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf"],
    True: [os.path.join(_WIN_FONTS, "segoeuib.ttf"),
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
           "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
           "/System/Library/Fonts/Supplemental/Arial Bold.ttf"],
}
_HERSHEY = cv2.FONT_HERSHEY_SIMPLEX
_HERSHEY_EM = 30.0          # Hershey simplex at scale 1: ~22 px cap height, i.e. roughly a 30 px font


@lru_cache(maxsize=None)
def font_path(bold: bool = False):
    """Path of the TrueType font used, or None (Hershey fallback). Bold falls back to regular."""
    env = os.environ.get("SEMISIGNALS_FONT_BOLD" if bold else "SEMISIGNALS_FONT")
    for p in ([env] if env else []) + CANDIDATES[bold]:
        if p and os.path.isfile(p):
            return p
    return font_path(False) if bold else None


@lru_cache(maxsize=None)
def _pil():
    try:
        from PIL import Image, ImageDraw, ImageFont
        return Image, ImageDraw, ImageFont
    except ImportError:
        return None


@lru_cache(maxsize=64)
def _font(size_px: int, bold: bool):
    if _pil() is None:
        return None
    p = font_path(bold)
    if p is None:
        return None
    try:
        return _pil()[2].truetype(p, size_px)
    except OSError:
        return None


def backend() -> str:
    """'truetype:<file>' or 'hershey' (for logs and tests)."""
    f = _font(14, False)
    return f"truetype:{os.path.basename(font_path(False))}" if f is not None else "hershey"


@lru_cache(maxsize=4096)
def _mask(text: str, size_px: int, bold: bool):
    """(alpha float32 HxW in 0..1, ascent, x_offset) for one label; ascent = rows above the baseline."""
    f = _font(size_px, bold)
    if f is None:
        return None
    Image, ImageDraw, _ = _pil()
    asc, desc = f.getmetrics()
    l, _, r, _ = f.getbbox(text, anchor="ls") if text else (0, 0, 0, 0)
    w, h = max(1, r - min(l, 0) + 2), asc + desc + 2
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((-min(l, 0) + 1, asc + 1), text, font=f, fill=255, anchor="ls")
    a = np.asarray(im, np.float32) / 255.0
    a.setflags(write=False)
    return a, asc + 1, -min(l, 0) + 1


def _hershey_scale(size_px):
    return max(0.25, size_px / _HERSHEY_EM)


def text_size(text: str, size_px: int, bold: bool = False):
    """(width, height above baseline, depth below baseline) in pixels."""
    return _text_size(text, int(round(size_px)), bool(bold))


@lru_cache(maxsize=4096)
def _text_size(text, size_px, bold):
    f = _font(size_px, bold)
    if f is not None:
        l, _, r, _ = f.getbbox(text, anchor="ls") if text else (0, 0, 0, 0)
        asc, desc = f.getmetrics()
        return int(r - min(l, 0)), int(asc), int(desc)
    sc = _hershey_scale(size_px)
    (w, h), base = cv2.getTextSize(text, _HERSHEY, sc, 2 if bold else 1)
    return int(w), int(h), int(base)


def text_width(text: str, size_px: int, bold: bool = False) -> int:
    return text_size(text, size_px, bold)[0]


def _origin(text, org, size_px, bold, anchor):
    """Left end of the baseline for the anchored position."""
    w, asc, desc = text_size(text, size_px, bold)
    x, y = org
    h_, v_ = (anchor + "ls")[:2] if len(anchor) == 2 else ("l", "s")
    x -= {"l": 0, "m": w // 2, "r": w}.get(h_, 0)
    y += {"t": asc, "m": (asc - desc) // 2, "s": 0, "b": -desc}.get(v_, 0)
    return int(x), int(y)


def put_text(img: np.ndarray, text: str, org, size_px: int, color_bgr, bold: bool = False, anchor: str = "ls"):
    """Draw text into a BGR uint8 image in place; returns the right x of the text.
    size_px is the font size in pixels (cap height is about 0.7 of it)."""
    if not text:
        return int(org[0])
    size_px = int(round(size_px))
    x, y = _origin(text, org, size_px, bold, anchor)
    m = _mask(text, size_px, bool(bold))
    if m is None:
        sc = _hershey_scale(size_px)
        cv2.putText(img, text, (x, y), _HERSHEY, sc, tuple(int(c) for c in color_bgr), 2 if bold else 1, cv2.LINE_AA)
        return x + text_width(text, size_px, bold)
    a, asc, xo = m
    x0, y0 = x - xo, y - asc
    H, W = img.shape[:2]
    ah, aw = a.shape
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + aw), min(H, y0 + ah)
    if sx1 <= sx0 or sy1 <= sy0:
        return x + aw
    al = a[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0, None]
    roi = img[sy0:sy1, sx0:sx1]
    col = np.asarray(color_bgr, np.float32)[:3]
    roi[:] = (roi * (1.0 - al) + col * al + 0.5).astype(np.uint8)
    return x + text_width(text, size_px, bold)


def fit_text(text: str, max_w: int, size_px: int, bold: bool = False) -> str:
    """Shorten text with an ellipsis so it fits max_w pixels."""
    if text_width(text, size_px, bold) <= max_w:
        return text
    ell = "…" if _font(int(size_px), bool(bold)) is not None else "..."
    while text and text_width(text + ell, size_px, bold) > max_w:
        text = text[:-1]
    return text.rstrip() + ell


# ── window helpers (Windows DPI, screen work area) ─────────────
_dpi_done = {"v": None}


def make_dpi_aware() -> bool:
    """Ask Windows not to bitmap-upscale our windows (OpenCV and tkinter look blurry otherwise).
    Call before any window is created. No-op elsewhere; returns True if it took effect."""
    if _dpi_done["v"] is not None:
        return _dpi_done["v"]
    ok = False
    if sys.platform == "win32":
        import ctypes
        try:
            ok = ctypes.windll.shcore.SetProcessDpiAwareness(1) == 0      # system DPI aware
        except Exception:
            try:
                ok = bool(ctypes.windll.user32.SetProcessDPIAware())
            except Exception:
                ok = False
    _dpi_done["v"] = ok
    return ok


def work_area(default=(1920, 1040)):
    """(width, height) of the primary screen without the taskbar, in physical pixels when DPI aware."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            r = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(r), 0):   # SPI_GETWORKAREA
                return r.right - r.left, r.bottom - r.top
        except Exception:
            pass
    env = os.environ.get("SEMISIGNALS_SCREEN")          # e.g. "1920x1040" (tests, unusual setups)
    if env and "x" in env:
        try:
            w, h = env.lower().split("x")
            return int(w), int(h)
        except ValueError:
            pass
    return default


def fit_to_screen(img: np.ndarray, margin=(16, 60), area=None) -> np.ndarray:
    """Downscale (INTER_AREA, never upscale) so the image fits the screen work area with a margin
    for the window frame and title bar."""
    W, H = area or work_area()
    s = min(1.0, (W - margin[0]) / img.shape[1], (H - margin[1]) / img.shape[0])
    if s >= 0.999:
        return img
    return cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)


# ── cached text blocks (several lines on a dimmed box, blended in one step per frame) ──
@lru_cache(maxsize=256)
def _block(lines, size_px, bold, pad, gap, dim):
    lh = int(size_px * 1.25) + gap
    w = max(text_width(t, size_px, bold) for t, _ in lines) + 2 * pad
    h = lh * len(lines) + pad
    C = np.zeros((h, w, 3), np.uint8)
    M = np.zeros((h, w, 3), np.uint8)
    for i, (t, col) in enumerate(lines):
        put_text(C, t, (pad, pad // 2 + 1 + i * lh), size_px, col, bold=bold, anchor="lt")
        put_text(M, t, (pad, pad // 2 + 1 + i * lh), size_px, (255, 255, 255), bold=bold, anchor="lt")
    a = M[..., :1].astype(np.float32) / 255.0
    A = (dim * (1.0 - a)).astype(np.float32)
    A.setflags(write=False); Cf = C.astype(np.float32); Cf.setflags(write=False)
    return A, Cf


def put_block(img, lines, org, size_px, bold=True, pad=8, gap=6, dim=0.4):
    """Text lines [(text, bgr), ...] on a translucent dark box at org (top-left). The box is
    rendered once per distinct content and then blended with one multiply-add per frame."""
    if not lines:
        return
    A, C = _block(tuple((str(t), tuple(int(c) for c in col)) for t, col in lines), int(size_px), bool(bold),
                  int(pad), int(gap), float(dim))
    x, y = int(org[0]), int(org[1])
    H, W = img.shape[:2]
    h, w = min(A.shape[0], H - y), min(A.shape[1], W - x)
    if h <= 0 or w <= 0 or x < 0 or y < 0:
        return
    roi = img[y:y + h, x:x + w]
    roi[:] = np.clip(roi * A[:h, :w] + C[:h, :w] + 0.5, 0, 255).astype(np.uint8)
