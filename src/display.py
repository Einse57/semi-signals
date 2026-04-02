"""Real-time display: skeleton overlay + multi-region motion chart."""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.config import (
    BODY_REGIONS,
    CHART_GROUPS,
    CHART_TIME_WINDOW,
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    HAND_SKELETON_EDGES,
    KEYPOINT_NAMES,
    REGION_COLORS,
    SKELETON_EDGES,
)


# Joint colour (BGR) per body region
_JOINT_COLORS: dict[int, tuple[int, int, int]] = {}
for _j in BODY_REGIONS.get("head", []):
    _JOINT_COLORS[_j] = REGION_COLORS.get("head", (0, 255, 255))
for _j in (*BODY_REGIONS.get("l_arm", []), *BODY_REGIONS.get("r_arm", [])):
    _JOINT_COLORS[_j] = (0, 200, 0)
for _j in (*BODY_REGIONS.get("l_hand", []), *BODY_REGIONS.get("r_hand", [])):
    _JOINT_COLORS[_j] = (0, 255, 128)
for _j in (*BODY_REGIONS.get("l_leg", []), *BODY_REGIONS.get("r_leg", [])):
    _JOINT_COLORS[_j] = (0, 128, 255)


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    score_thr: float = 0.3,
) -> np.ndarray:
    """Draw skeleton joints and limbs on *frame* (mutates in-place)."""
    for i, ((x, y), sc) in enumerate(zip(keypoints, scores)):
        if sc < score_thr:
            continue
        colour = _JOINT_COLORS.get(i, (255, 255, 255))
        cv2.circle(frame, (int(x), int(y)), 4, colour, -1)

    for i, j in SKELETON_EDGES:
        if scores[i] < score_thr or scores[j] < score_thr:
            continue
        pt1 = tuple(keypoints[i].astype(int))
        pt2 = tuple(keypoints[j].astype(int))
        cv2.line(frame, pt1, pt2, (200, 200, 200), 2)

    return frame


# Per-finger colour for hand skeleton (BGR)
_FINGER_COLORS = {
    "thumb":  (0, 200, 255),   # gold
    "index":  (0, 255, 0),     # green
    "middle": (255, 200, 0),   # teal
    "ring":   (255, 0, 128),   # magenta
    "pinky":  (255, 128, 255), # pink
    "wrist":  (200, 200, 200), # grey
}

def _finger_for_edge(i: int, j: int) -> str:
    joints = {i, j}
    if joints & {1, 2, 3, 4}:
        return "thumb"
    if joints & {5, 6, 7, 8}:
        return "index"
    if joints & {9, 10, 11, 12}:
        return "middle"
    if joints & {13, 14, 15, 16}:
        return "ring"
    if joints & {17, 18, 19, 20}:
        return "pinky"
    return "wrist"


def draw_hand_skeleton(
    frame: np.ndarray,
    hand_result: dict | None,
    score_thr: float = 0.3,
) -> np.ndarray:
    """Draw 21-keypoint hand skeleton on *frame* (mutates in-place)."""
    if hand_result is None:
        return frame

    kpts = hand_result["keypoints"]   # (21, 2)
    scores = hand_result["scores"]    # (21,)

    for i, j in HAND_SKELETON_EDGES:
        if scores[i] < score_thr or scores[j] < score_thr:
            continue
        color = _FINGER_COLORS.get(_finger_for_edge(i, j), (200, 200, 200))
        pt1 = tuple(kpts[i].astype(int))
        pt2 = tuple(kpts[j].astype(int))
        cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)

    for idx, ((x, y), sc) in enumerate(zip(kpts, scores)):
        if sc < score_thr:
            continue
        # Fingertips get slightly larger dots
        r = 3 if idx in {4, 8, 12, 16, 20} else 2
        cv2.circle(frame, (int(x), int(y)), r, (255, 255, 255), -1)

    return frame


def draw_motion_chart(
    intensity_history: list[float] | object,
    chart_w: int = DISPLAY_WIDTH,
    chart_h: int = 150,
) -> np.ndarray:
    """Single-region fallback (uses overall history only)."""
    region_hist = {"overall": intensity_history}
    return draw_region_chart(region_hist, chart_w=chart_w, chart_h=chart_h)


def _draw_single_chart(
    title: str,
    region_keys: list[str],
    region_histories: dict[str, list[float] | object],
    chart_w: int,
    chart_h: int,
    timestamps: list[float] | None = None,
) -> np.ndarray:
    """Render one mini-chart with overlaid region lines.

    When *timestamps* is provided, the x-axis represents a fixed
    ``CHART_TIME_WINDOW``-second window ending at the latest timestamp.
    Points are positioned by wall-clock time so the scroll rate is
    constant regardless of FPS.

    Returns a BGR image of size (chart_h, chart_w, 3).
    """
    canvas = np.zeros((chart_h, chart_w, 3), dtype=np.uint8)

    legend_w = 70
    plot_w = chart_w - legend_w
    title_h = 14
    plot_h = chart_h - title_h - 2
    plot_y0 = title_h

    # Determine time window
    use_time_axis = timestamps is not None and len(timestamps) >= 2
    if use_time_axis:
        t_now = timestamps[-1]
        t_start = t_now - CHART_TIME_WINDOW
    else:
        t_now = t_start = 0.0

    # Auto-scale: per-chart max (ignoring NaN)
    local_max = 1.0
    for key in region_keys:
        data = list(region_histories.get(key, []))
        finite = [v for v in data if not math.isnan(v)]
        if finite:
            local_max = max(local_max, max(finite))

    # Draw minute markers when using time axis
    if use_time_axis:
        for mins in range(1, int(CHART_TIME_WINDOW // 60) + 1):
            t_mark = t_now - mins * 60
            if t_mark < t_start:
                break
            frac = (t_mark - t_start) / CHART_TIME_WINDOW
            px = int(frac * plot_w)
            cv2.line(canvas, (px, plot_y0), (px, plot_y0 + plot_h),
                     (40, 40, 40), 1)
            cv2.putText(canvas, f"-{mins}m", (px + 2, plot_y0 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (80, 80, 80), 1)

    for key in region_keys:
        data = list(region_histories.get(key, []))
        if len(data) < 2:
            continue
        color = REGION_COLORS.get(key, (200, 200, 200))

        prev_pt = None
        for i, v in enumerate(data):
            if math.isnan(v):
                prev_pt = None
                continue

            if use_time_axis:
                t = timestamps[i]
                if t < t_start:
                    prev_pt = None
                    continue
                frac = (t - t_start) / CHART_TIME_WINDOW
                px = int(frac * plot_w)
            else:
                n_pts = len(data)
                px = int(i * plot_w / max(n_pts - 1, 1))

            py = plot_y0 + plot_h - 1 - int((v / local_max) * (plot_h - 2))
            py = max(plot_y0, min(plot_y0 + plot_h - 1, py))
            cur_pt = (px, py)
            if prev_pt is not None:
                cv2.line(canvas, prev_pt, cur_pt, color, 1, cv2.LINE_AA)
            prev_pt = cur_pt

    # Title (left)
    cv2.putText(canvas, title, (3, 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    # Legend (right)
    lx = plot_w + 4
    ly = 12
    for key in region_keys:
        color = REGION_COLORS.get(key, (200, 200, 200))
        label = key.replace("l_", "L ").replace("r_", "R ")
        cv2.line(canvas, (lx, ly), (lx + 10, ly), color, 2)
        cv2.putText(canvas, label, (lx + 13, ly + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)
        ly += 13

    # Separator line at bottom
    cv2.line(canvas, (0, chart_h - 1), (chart_w, chart_h - 1),
             (60, 60, 60), 1)

    return canvas


def draw_region_chart(
    region_histories: dict[str, list[float] | object],
    chart_w: int = DISPLAY_WIDTH,
    chart_h: int = 250,
    timestamps: list[float] | object | None = None,
) -> np.ndarray:
    """Render 4 stacked mini-charts based on CHART_GROUPS config.

    Each chart group gets an equal share of the total height, with
    its own auto-scale and legend.

    Returns a BGR image of size (chart_h, chart_w, 3).
    """
    ts = list(timestamps) if timestamps is not None else None
    n_groups = len(CHART_GROUPS)
    per_h = chart_h // n_groups
    strips: list[np.ndarray] = []

    for title, region_keys in CHART_GROUPS:
        strip = _draw_single_chart(
            title, region_keys, region_histories,
            chart_w, per_h, timestamps=ts,
        )
        strips.append(strip)

    # Stack and pad to exact chart_h if rounding differs
    result = np.vstack(strips)
    if result.shape[0] < chart_h:
        pad = np.zeros((chart_h - result.shape[0], chart_w, 3), dtype=np.uint8)
        result = np.vstack([result, pad])
    return result[:chart_h]


def draw_angles_overlay(
    frame: np.ndarray,
    angles: dict[str, float],
    origin: tuple[int, int] = (10, 30),
) -> np.ndarray:
    """Print joint angle values on frame."""
    y = origin[1]
    for name, deg in angles.items():
        text = f"{name}: {deg:.0f} deg"
        cv2.putText(
            frame, text, (origin[0], y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        y += 20
    return frame


def compose_display(
    video_frame: np.ndarray,
    chart_img: np.ndarray,
) -> np.ndarray:
    """Stack video frame (top) and chart (bottom) into a single window."""
    frame_resized = cv2.resize(video_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    return np.vstack([frame_resized, chart_img])
