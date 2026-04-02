"""Limb-motion quantification: velocity, angles, and body-region intensity."""

from __future__ import annotations

from collections import deque
import math
import time

import numpy as np

from src.config import (
    BODY_REGIONS,
    MOTION_HISTORY_LEN,
    KEYPOINT_NAMES,
)

# Sentinel for "region not visible this frame".
# NaN propagates through math naturally and the chart renderer can
# detect it to draw gaps instead of false-zero flat-lines.
_NOT_VISIBLE = float("nan")


class MotionQuantifier:
    """Tracks per-joint velocities and per-region motion intensity.

    All velocities are in *pixels per frame*.  Multiply by FPS to get
    pixels-per-second.
    """

    def __init__(
        self,
        history_len: int = MOTION_HISTORY_LEN,
        visibility_thr: float = 0.3,
    ) -> None:
        self._prev_keypoints: np.ndarray | None = None
        self._velocities: np.ndarray | None = None
        self._visibility_thr = visibility_thr

        # Hand tracking state (21 kpts per hand, in frame coords)
        self._prev_hand_kpts: dict[str, np.ndarray | None] = {
            "left": None, "right": None,
        }
        self._hand_velocity: float | None = None

        # Per-joint scrolling histories
        self.joint_velocity_history: dict[str, deque[float]] = {
            name: deque(maxlen=history_len) for name in KEYPOINT_NAMES
        }

        # Per-region scrolling histories (NaN = not visible)
        self.region_intensity_history: dict[str, deque[float]] = {
            region: deque(maxlen=history_len) for region in BODY_REGIONS
        }

        # Wall-clock timestamps aligned with history entries
        self.timestamps: deque[float] = deque(maxlen=history_len)

        # Legacy alias kept for back-compat
        self.total_intensity_history = self.region_intensity_history["overall"]

    # ── public API ────────────────────────────────────────────
    def update(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray | None = None,
        hand_results: dict[str, dict | None] | None = None,
    ) -> dict:
        """Accept (17,2) keypoints, optional (17,) scores, optional hand results.

        ``hand_results`` is the dict returned by ``HandEstimator.estimate()``,
        with keys ``"left"`` and ``"right"``, each ``None`` or a dict with
        ``"keypoints"`` (21,2) and ``"scores"`` (21,).

        When hand data is available the ``"l_hand"``/``"r_hand"`` region
        intensities are computed from 21 hand keypoints each instead of
        the single body-wrist joint.

        Returns dict with:
            velocities       – (17,) per-joint Euclidean velocity
            total_intensity  – scalar (all joints, NaN-safe)
            region_intensity – dict[str, float] per-region (NaN if not visible)
            joint_angles     – dict of named joint angles (degrees)
        """
        if self._prev_keypoints is not None:
            diff = keypoints - self._prev_keypoints
            self._velocities = np.linalg.norm(diff, axis=1)  # (17,)
        else:
            self._velocities = np.zeros(keypoints.shape[0], dtype=np.float32)

        self._prev_keypoints = keypoints.copy()

        # Record wall-clock time for this sample
        self.timestamps.append(time.monotonic())

        # Per-joint history
        for i, name in enumerate(KEYPOINT_NAMES):
            self.joint_velocity_history[name].append(float(self._velocities[i]))

        # ── Compute per-side hand velocities ──────────────────
        hand_vel = self._compute_hand_velocity_per_side(hand_results)

        # Per-region intensity with visibility gating
        region_intensity: dict[str, float] = {}
        for region, joint_ids in BODY_REGIONS.items():
            # Override l_hand / r_hand with detailed hand keypoints
            if region == "l_hand" and hand_vel.get("left") is not None:
                region_intensity["l_hand"] = hand_vel["left"]
                self.region_intensity_history["l_hand"].append(hand_vel["left"])
                continue
            if region == "r_hand" and hand_vel.get("right") is not None:
                region_intensity["r_hand"] = hand_vel["right"]
                self.region_intensity_history["r_hand"].append(hand_vel["right"])
                continue

            if scores is not None:
                visible_mask = scores[joint_ids] >= self._visibility_thr
                if not visible_mask.any():
                    region_intensity[region] = _NOT_VISIBLE
                    self.region_intensity_history[region].append(_NOT_VISIBLE)
                    continue
                val = float(self._velocities[joint_ids][visible_mask].mean())
            else:
                val = float(self._velocities[joint_ids].mean())

            region_intensity[region] = val
            self.region_intensity_history[region].append(val)

        # Blend finger velocities into overall: re-compute as weighted
        # mean of body-joint velocities + hand-keypoint velocities
        if "overall" in region_intensity and hand_vel:
            body_vals = []
            if scores is not None:
                vis = scores >= self._visibility_thr
                if vis.any():
                    body_vals = self._velocities[vis].tolist()
            else:
                body_vals = self._velocities.tolist()

            hand_vals = self._all_hand_velocities or []
            if hand_vals:
                all_vals = body_vals + hand_vals
                blended = float(np.mean(all_vals)) if all_vals else 0.0
                region_intensity["overall"] = blended
                self.region_intensity_history["overall"][-1] = blended

        total = region_intensity.get("overall", 0.0)

        angles = self._compute_angles(keypoints)

        return {
            "velocities": self._velocities,
            "total_intensity": total,
            "region_intensity": region_intensity,
            "joint_angles": angles,
        }

    def _compute_hand_velocity_per_side(
        self,
        hand_results: dict[str, dict | None] | None,
    ) -> dict[str, float | None]:
        """Compute per-side mean velocity over visible hand keypoints.

        Also populates ``self._all_hand_velocities`` with individual
        keypoint velocities so they can be blended into ``overall``.

        Returns {"left": float|None, "right": float|None}.
        """
        result: dict[str, float | None] = {"left": None, "right": None}
        self._all_hand_velocities: list[float] = []

        if hand_results is None:
            self._prev_hand_kpts = {"left": None, "right": None}
            return result

        for side in ("left", "right"):
            hr = hand_results.get(side)
            if hr is None:
                self._prev_hand_kpts[side] = None
                continue
            kpts = hr["keypoints"]           # (21, 2)
            sc = hr["scores"]                # (21,)
            vis = sc >= self._visibility_thr  # (21,)
            prev = self._prev_hand_kpts[side]

            if prev is not None and vis.any():
                diff = kpts - prev
                vels = np.linalg.norm(diff, axis=1)  # (21,)
                vis_vels = vels[vis]
                result[side] = float(vis_vels.mean())
                self._all_hand_velocities.extend(vis_vels.tolist())

            self._prev_hand_kpts[side] = kpts.copy()

        return result

    def reset(self) -> None:
        self._prev_keypoints = None
        self._velocities = None
        self.timestamps.clear()
        for d in self.joint_velocity_history.values():
            d.clear()
        for d in self.region_intensity_history.values():
            d.clear()

    # ── angle helpers ─────────────────────────────────────────
    @staticmethod
    def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Angle at vertex *b* formed by segments ba and bc (degrees)."""
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return math.degrees(math.acos(cos_angle))

    def _compute_angles(self, kpts: np.ndarray) -> dict[str, float]:
        """Compute clinically relevant joint angles."""
        idx = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
        angles: dict[str, float] = {}

        # Left elbow angle (shoulder – elbow – wrist)
        angles["left_elbow"] = self._angle_between(
            kpts[idx["left_shoulder"]],
            kpts[idx["left_elbow"]],
            kpts[idx["left_wrist"]],
        )
        # Right elbow
        angles["right_elbow"] = self._angle_between(
            kpts[idx["right_shoulder"]],
            kpts[idx["right_elbow"]],
            kpts[idx["right_wrist"]],
        )
        # Left knee
        angles["left_knee"] = self._angle_between(
            kpts[idx["left_hip"]],
            kpts[idx["left_knee"]],
            kpts[idx["left_ankle"]],
        )
        # Right knee
        angles["right_knee"] = self._angle_between(
            kpts[idx["right_hip"]],
            kpts[idx["right_knee"]],
            kpts[idx["right_ankle"]],
        )
        return angles
