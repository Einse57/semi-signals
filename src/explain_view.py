"""Side panel for the seizure demo: probability strip, top drivers, motion graphs, EEG lane.

All lanes share one time axis (same left/right margins), so a highlighted motion line
lines up with the probability above it.
"""
from __future__ import annotations

import numpy as np
import cv2

from src.config import REGION_COLORS

FONT = cv2.FONT_HERSHEY_SIMPLEX
L_M, R_M = 44, 86                 # shared plot margins (px)
HILITE = (0, 190, 255)            # amber (BGR)
GRID = (55, 55, 55)
TXT = (190, 190, 190)
MOTION_GROUPS = [("Hands", ["l_hand", "r_hand"]), ("Arms", ["l_arm", "r_arm"]), ("Head", ["head"]),
                 ("Legs", ["l_leg", "r_leg"]), ("Whole body", ["overall"])]
SHORT = {"l_hand": "L hand", "r_hand": "R hand", "l_arm": "L arm", "r_arm": "R arm", "head": "head",
         "l_leg": "L leg", "r_leg": "R leg", "overall": "all"}
DISCLAIMER = "Drivers explain the model score, not clinical cause. Research use only, not diagnostic."


class Axis:
    def __init__(self, w, now, span):
        self.now = max(now, span)
        self.t0 = self.now - span
        self.span = span
        self.pw = w - L_M - R_M

    def x(self, t):
        return int(L_M + (t - self.t0) / self.span * self.pw)


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
                cv2.putText(img, name, (ax.x(tm) + 3, top + 11), FONT, 0.36, col, 1)


def draw_prob_strip(w, h, now, span, tA, pA, pAs, alarm, thr, marks, extra=(), alarm_now=False):
    """tA/pA/pAs/alarm: per-second history (alarm = bool per second, covering [t-1, t))."""
    img = np.full((h, w, 3), 22, np.uint8)
    ax = Axis(w, now, span)
    T, B = 18, 16
    ph = h - T - B
    Y = lambda v: int(T + (1 - min(max(v, 0.0), 1.0)) * ph)
    # alarm shading
    for t, a in zip(tA, alarm):
        if a and t >= ax.t0:
            x0, x1 = max(ax.x(t - 1), L_M), ax.x(t)
            img[T:T + ph, x0:x1 + 1] = (img[T:T + ph, x0:x1 + 1] * 0.5 + np.array((20, 20, 150)) * 0.5).astype(np.uint8)
    for v in (0.0, 0.5, 1.0):
        cv2.line(img, (L_M, Y(v)), (w - R_M, Y(v)), GRID, 1)
        cv2.putText(img, f"{v:.1f}", (8, Y(v) + 4), FONT, 0.38, TXT, 1)
    if thr is not None:
        dashed_hline(img, L_M, w - R_M, Y(thr), (80, 80, 255))
        cv2.putText(img, f"alarm {thr:.2f}", (w - R_M + 4, Y(thr) + 4), FONT, 0.36, (80, 80, 255), 1)
    draw_marks(img, ax, T, T + ph, marks)
    for name, col, ts, vs in extra:
        _polyline(img, ax, Y, ts, vs, col, 1)
    _polyline(img, ax, Y, tA, pA, (0, 110, 0), 1)
    _polyline(img, ax, Y, tA, pAs, (0, 255, 0), 2)
    cv2.putText(img, "Seizure probability (Detector A, 5 s smoothed)", (6, 13), FONT, 0.42, (230, 230, 230), 1)
    lx = L_M
    for name, col in [("raw", (0, 110, 0)), ("smoothed", (0, 255, 0))] + [(n, c) for n, c, _, _ in extra]:
        cv2.putText(img, name, (lx, h - 4), FONT, 0.36, col, 1)
        lx += 12 + 7 * len(name)
    # current value readout
    cur = next((v for v in reversed(pAs) if np.isfinite(v)), np.nan)
    col = (60, 60, 255) if alarm_now else (0, 255, 0)
    cv2.putText(img, "now", (w - R_M + 6, T + 34), FONT, 0.4, TXT, 1)
    cv2.putText(img, "--" if not np.isfinite(cur) else f"{cur:.2f}", (w - R_M + 6, T + 64), FONT, 0.85, col, 2)
    if alarm_now:
        cv2.putText(img, "ALARM", (w - R_M + 6, T + 88), FONT, 0.55, (60, 60, 255), 2)
    cv2.putText(img, f"t={now:.0f}s", (w - R_M + 6, h - 4), FONT, 0.38, TXT, 1)
    if alarm_now:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 2)
    return img


def draw_drivers(w, h, drivers, p_smooth, min_p=0.5):
    img = np.full((h, w, 3), 30, np.uint8)
    cv2.putText(img, "Top drivers of the score (last 5 s)", (6, 15), FONT, 0.45, (230, 230, 230), 1)
    if not drivers:
        msg = (f"shown when the smoothed score is >= {min_p:.2f}" if not np.isfinite(p_smooth) or p_smooth < min_p
               else "no clear positive drivers")
        cv2.putText(img, msg, (14, 38), FONT, 0.42, (130, 130, 130), 1)
    y = 36
    for d in drivers[:3]:
        bw = int(110 * d["share"])
        cv2.rectangle(img, (12, y - 10), (12 + bw, y + 2), HILITE, -1)
        cv2.rectangle(img, (12, y - 10), (122, y + 2), (90, 90, 90), 1)
        cv2.putText(img, d["text"], (132, y), FONT, 0.5, HILITE, 1, cv2.LINE_AA)
        y += 19
    cv2.putText(img, DISCLAIMER, (6, h - 6), FONT, 0.36, (150, 150, 150), 1, cv2.LINE_AA)
    return img


def draw_motion(w, h, now, span, t, regions, highlight, marks, floor=0.15):
    """t: per-frame times; regions: {name: per-frame speed}; highlight: set of region names."""
    n = len(MOTION_GROUPS)
    per = h // n
    ax = Axis(w, now, span)
    t = np.asarray(t, float)
    strips = []
    for title, keys in MOTION_GROUPS:
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
                _polyline(img, ax, Y, tt, v, REGION_COLORS.get(r, TXT), 2)
            else:
                _polyline(img, ax, Y, tt, v, tuple(int(c * 0.6) for c in REGION_COLORS.get(r, TXT)), 1)
        cv2.putText(img, title, (4, 13), FONT, 0.38, HILITE if hl else TXT, 1)
        ly = 13
        for r in keys:
            col = HILITE if r in highlight else REGION_COLORS.get(r, TXT)
            cv2.line(img, (w - R_M + 4, ly - 4), (w - R_M + 14, ly - 4), REGION_COLORS.get(r, TXT), 2)
            cv2.putText(img, SHORT[r] + (" *" if r in highlight else ""), (w - R_M + 18, ly), FONT, 0.36, col, 1)
            ly += 13
        cv2.line(img, (0, per - 1), (w, per - 1), GRID, 1)
        strips.append(img)
    out = np.vstack(strips)
    if out.shape[0] < h:
        out = np.vstack([out, np.full((h - out.shape[0], w, 3), 14, np.uint8)])
    hdr = "Motion (speed, body-size/s; * = top driver)"
    cv2.putText(out, hdr, (L_M + 60, 13), FONT, 0.34, (120, 120, 120), 1)
    return out


def draw_eeg_lane(w, h, enabled=False, ts=None, vs=None, now=0.0, span=90.0):
    """Placeholder lane for a future paired EEG stream. Greyed out unless enabled."""
    img = np.full((h, w, 3), 26, np.uint8)
    if not enabled:
        for x in range(-h, w, 12):                       # hatch = not connected
            cv2.line(img, (x, h), (x + h, 0), (34, 34, 34), 1)
        cv2.putText(img, "EEG lane: not connected (planned)", (L_M, h // 2 + 5), FONT, 0.42, (95, 95, 95), 1)
        return img
    ax = Axis(w, now, span)
    Y = lambda v: int(2 + (1 - min(max(v, 0.0), 1.0)) * (h - 4))
    _polyline(img, ax, Y, ts, vs, (255, 0, 255), 1)
    cv2.putText(img, "EEG", (4, h // 2 + 5), FONT, 0.42, (255, 0, 255), 1)
    return img


def compose_panel(w, h, prob, drivers, motion, eeg):
    out = np.vstack([prob, drivers, motion, eeg])
    if out.shape[0] < h:
        out = np.vstack([out, np.full((h - out.shape[0], w, 3), 14, np.uint8)])
    return out[:h]


PANEL_H = dict(prob=170, drivers=96, eeg=26)


def layout(h):
    """Heights for (prob, drivers, motion, eeg) in a panel of height h."""
    mh = max(150, h - PANEL_H["prob"] - PANEL_H["drivers"] - PANEL_H["eeg"])
    return PANEL_H["prob"], PANEL_H["drivers"], mh, PANEL_H["eeg"]
