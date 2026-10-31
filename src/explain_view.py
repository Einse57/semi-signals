"""Side panel for the seizure demo: probability strip, top drivers, motion graphs, EEG lane.

All lanes share one time axis (same left/right margins), so a motion line
lines up with the probability above it.
"""
from __future__ import annotations

import numpy as np
import cv2

from src.config import REGION_COLORS
from src.textdraw import put_text, put_block, text_width, fit_text

FS, FS_TITLE = 13, 14              # text sizes (px); 13 px is the smallest label
L_M, R_M = 44, 86                 # shared plot margins (px)
HILITE = (0, 190, 255)            # amber (BGR)
GRID = (55, 55, 55)
TXT = (190, 190, 190)
MOTION_GROUPS = [("Hands", ["l_hand", "r_hand"]), ("Arms", ["l_arm", "r_arm"]), ("Head", ["head"]),
                 ("Legs", ["l_leg", "r_leg"]), ("Whole body", ["overall"])]
SHORT = {"l_hand": "L hand", "r_hand": "R hand", "l_arm": "L arm", "r_arm": "R arm", "head": "head",
         "l_leg": "L leg", "r_leg": "R leg", "overall": "all"}
# Face motion (optional --face): subject's left = green, right = blue, as for the body regions
FACE_GROUPS = [("Mouth", ["mouth"]), ("Eyes", ["l_eye", "r_eye"]), ("Face L/R", ["l_face", "r_face"])]
FACE_COLORS = {"mouth": (0, 165, 255), "l_eye": (0, 200, 0), "r_eye": (200, 0, 0),
               "l_face": (0, 200, 0), "r_face": (200, 0, 0)}
SHORT.update({"mouth": "mouth", "l_eye": "L eye", "r_eye": "R eye", "l_face": "L face", "r_face": "R face"})
FACE_HDR = "Face motion (head motion removed, face widths/s)"
DISCLAIMER = "Drivers explain the model score, not clinical cause. Research use only, not diagnostic."


class Axis:
    def __init__(self, w, now, span):
        self.now = max(now, span)
        self.t0 = self.now - span
        self.span = span
        self.pw = w - L_M - R_M

    def x(self, t):
        return int(L_M + (t - self.t0) / self.span * self.pw)


def _light(col, k=0.45):
    """Lighter version of a line colour for text (dark blues are unreadable on the dark panel)."""
    return tuple(int(c + (255 - c) * k) for c in col)


def dashed_hline(img, x0, x1, y, col, dash=8, gap=6):
    for x in range(x0, x1, dash + gap):
        cv2.line(img, (x, y), (min(x + dash, x1), y), col, 1)


def _polyline(img, ax, Y, ts, vs, col, th, lo_t=None):
    ts = np.asarray(ts, float); vs = np.asarray(vs, float)
    if not len(ts):
        return
    k = (ts >= ax.t0) & np.isfinite(vs)
    if lo_t is not None:
        k &= ts >= lo_t
    # break the line at gaps (NaN) by drawing finite runs separately
    runs, cur = [], []
    for ok, a, b in zip(k, ts, vs):
        if ok:
            cur.append((ax.x(a), Y(b)))
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    for r in runs:
        if len(r) > 1:
            cv2.polylines(img, [np.array(r, np.int32)], False, col, th, cv2.LINE_AA)


def draw_marks(img, ax, top, bot, marks, label=True):
    for name, tm, col in marks:
        if tm is not None and ax.t0 <= tm <= ax.now:
            cv2.line(img, (ax.x(tm), top), (ax.x(tm), bot), col, 1)
            if label:
                put_block(img, [(name, col)], (ax.x(tm) + 2, top + 1), FS, bold=False, pad=3, gap=0, dim=0.3)


def draw_prob_strip(w, h, now, span, tA, pA, pAs, alarm, thr, marks, extra=(), alarm_now=False):
    """tA/pA/pAs/alarm: per-second history (alarm = flagged, bool per second, covering [t-1, t))."""
    img = np.full((h, w, 3), 22, np.uint8)
    ax = Axis(w, now, span)
    T, B = 24, 20
    ph = h - T - B
    Y = lambda v: int(T + (1 - min(max(v, 0.0), 1.0)) * ph)
    # shading of flagged seconds
    for t, a in zip(tA, alarm):
        if a and t >= ax.t0:
            x0, x1 = max(ax.x(t - 1), L_M), ax.x(t)
            img[T:T + ph, x0:x1 + 1] = (img[T:T + ph, x0:x1 + 1] * 0.5 + np.array((20, 20, 150)) * 0.5).astype(np.uint8)
    for v in (0.0, 0.5, 1.0):
        cv2.line(img, (L_M, Y(v)), (w - R_M, Y(v)), GRID, 1)
        put_text(img, f"{v:.1f}", (L_M - 6, Y(v)), FS, TXT, anchor="rm")
    if thr is not None:
        dashed_hline(img, L_M, w - R_M, Y(thr), (80, 80, 255))
        put_text(img, f"{thr:.2f}", (w - R_M + 4, Y(thr)), FS, (80, 80, 255), anchor="lm")
    draw_marks(img, ax, T, T + ph, marks)
    for name, col, ts, vs in extra:
        _polyline(img, ax, Y, ts, vs, col, 1)
    _polyline(img, ax, Y, tA, pA, (0, 110, 0), 1)
    _polyline(img, ax, Y, tA, pAs, (0, 255, 0), 2)
    put_text(img, "Seizure probability (Detector A, 5 s smoothed)", (6, 4), FS_TITLE, (230, 230, 230), bold=True, anchor="lt")
    lx = L_M
    for name, col in [("raw", (0, 110, 0)), ("smoothed", (0, 255, 0))] + [(n, c) for n, c, _, _ in extra]:
        if lx + text_width(name, FS) > w - R_M:
            break
        cv2.line(img, (lx, h - 9), (lx + 12, h - 9), col, 2)
        lx = put_text(img, name, (lx + 16, h - 9), FS, col, anchor="lm") + 14
    # current value readout
    cur = next((v for v in reversed(pAs) if np.isfinite(v)), np.nan)
    col = (60, 60, 255) if alarm_now else (0, 255, 0)
    put_text(img, "now", (w - R_M + 6, T + 22), FS, TXT)
    put_text(img, "--" if not np.isfinite(cur) else f"{cur:.2f}", (w - R_M + 6, T + 54), 28, col, bold=True)
    put_text(img, f"t = {now:.0f} s", (w - R_M + 6, h - 9), FS, TXT, anchor="lm")
    return img


def draw_drivers(w, h, drivers, p_smooth, min_p=0.5):
    img = np.full((h, w, 3), 30, np.uint8)
    put_text(img, "Top drivers of the score (last 5 s)", (6, 4), FS_TITLE, (230, 230, 230), bold=True, anchor="lt")
    if not drivers:
        msg = (f"shown when the smoothed score is >= {min_p:.2f}" if not np.isfinite(p_smooth) or p_smooth < min_p
               else "no clear positive drivers")
        put_text(img, msg, (14, 40), FS, (140, 140, 140))
    y = 38
    for d in drivers[:3]:
        bw = int(110 * d["share"])
        cv2.rectangle(img, (12, y - 10), (12 + bw, y + 2), HILITE, -1)
        cv2.rectangle(img, (12, y - 10), (122, y + 2), (90, 90, 90), 1)
        put_text(img, fit_text(d["text"], w - 140, 14), (132, y), 14, HILITE)
        y += 18
    put_text(img, fit_text(DISCLAIMER, w - 12, FS), (6, h - 5), FS, (150, 150, 150))
    return img


def draw_motion(w, h, now, span, t, regions, highlight, marks, floor=0.15, groups=None, hdr=None, colors=None):
    """t: per-frame times; regions: {name: per-frame speed}; highlight: set of region names.
    groups/hdr/colors default to the body motion graphs (MOTION_GROUPS, REGION_COLORS)."""
    groups = MOTION_GROUPS if groups is None else groups
    cmap = REGION_COLORS if colors is None else colors
    n = len(groups)
    per = h // n
    ax = Axis(w, now, span)
    t = np.asarray(t, float)
    strips = []
    for title, keys in groups:
        img = np.full((per, w, 3), 14, np.uint8)
        T, B = 3, 3
        ph = per - T - B
        k = t >= ax.t0
        vals = [np.asarray(regions.get(r, []), float)[k] if len(t) else np.array([]) for r in keys]
        allv = np.concatenate([v[np.isfinite(v)] for v in vals]) if vals else np.array([])
        top = max(floor, float(np.percentile(allv, 99))) if len(allv) else floor
        Y = lambda v, top=top: int(T + (1 - min(max(v / top, 0.0), 1.0)) * ph)
        hl = any(r in highlight for r in keys)
        if hl:
            img[:, :] = (30, 38, 46)
        draw_marks(img, ax, 0, per, marks, label=False)
        tt = t[k]
        for r, v in zip(keys, vals):
            if r in highlight:
                _polyline(img, ax, Y, tt, v, HILITE, 4)
                _polyline(img, ax, Y, tt, v, cmap.get(r, TXT), 2)
            else:
                _polyline(img, ax, Y, tt, v, tuple(int(c * 0.6) for c in cmap.get(r, TXT)), 1)
        put_text(img, title, (4, 2), FS, HILITE if hl else TXT, bold=hl, anchor="lt")
        ly = 9
        for r in keys:
            col = HILITE if r in highlight else _light(cmap.get(r, TXT))
            cv2.line(img, (w - R_M + 4, ly), (w - R_M + 14, ly), cmap.get(r, TXT), 2)
            put_text(img, SHORT[r] + (" *" if r in highlight else ""), (w - R_M + 18, ly), FS, col, anchor="lm")
            ly += 16
        cv2.line(img, (0, per - 1), (w, per - 1), GRID, 1)
        strips.append(img)
    out = np.vstack(strips)
    if out.shape[0] < h:
        out = np.vstack([out, np.full((h - out.shape[0], w, 3), 14, np.uint8)])
    if hdr is None:
        hdr = "Motion (speed, body-size/s" + ("; * = top driver)" if highlight else ")")
    put_text(out, hdr, (w - R_M - 6, 2), FS, (130, 130, 130), anchor="rt")
    return out


def draw_eeg_lane(w, h, enabled=False, ts=None, vs=None, now=0.0, span=90.0):
    """Placeholder lane for a future paired EEG stream. Greyed out unless enabled."""
    img = np.full((h, w, 3), 26, np.uint8)
    if not enabled:
        for x in range(-h, w, 12):                       # hatch = not connected
            cv2.line(img, (x, h), (x + h, 0), (34, 34, 34), 1)
        put_text(img, "EEG lane: not connected (planned)", (L_M, h // 2), FS, (110, 110, 110), anchor="lm")
        return img
    ax = Axis(w, now, span)
    Y = lambda v: int(2 + (1 - min(max(v, 0.0), 1.0)) * (h - 4))
    _polyline(img, ax, Y, ts, vs, (255, 0, 255), 1)
    put_text(img, "EEG", (4, h // 2), FS, (255, 0, 255), anchor="lm")
    return img


def compose_panel(w, h, prob, drivers, motion, eeg, face=None):
    out = np.vstack([prob, drivers, motion] + ([face] if face is not None else []) + [eeg])
    if out.shape[0] < h:
        out = np.vstack([out, np.full((h - out.shape[0], w, 3), 14, np.uint8)])
    return out[:h]


PANEL_H = dict(prob=170, drivers=96, eeg=26)


def split_motion(hm, face_on):
    """Heights (body motion, face motion) inside the motion area; face motion 0 when off."""
    if not face_on:
        return hm, 0
    hf = max(90, int(hm * 0.4))
    return hm - hf, hf


def layout(h):
    """Heights for (prob, drivers, motion, eeg) in a panel of height h."""
    mh = max(150, h - PANEL_H["prob"] - PANEL_H["drivers"] - PANEL_H["eeg"])
    return PANEL_H["prob"], PANEL_H["drivers"], mh, PANEL_H["eeg"]
