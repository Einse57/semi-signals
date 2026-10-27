"""Per-second explanations for Detector A (LightGBM pred_contrib).

Feature contributions (raw-score / log-odds units) are summed into groups of
(body region, feature family), averaged over the same trailing window as the
score smoothing (5 s), and the top positive groups are reported as "drivers".
Drivers explain the model score, not the clinical cause.

Left/right follow COCO keypoints, i.e. the patient's left/right.
"""
from __future__ import annotations

import collections

import numpy as np

REGION_TEXT = {
    "head": "head", "l_arm": "left arm", "r_arm": "right arm", "l_hand": "left hand",
    "r_hand": "right hand", "l_leg": "left leg", "r_leg": "right leg", "overall": "whole-body",
}
# chart lines highlighted for each group region
GROUP_REGIONS = {
    "head": ["head"], "l_arm": ["l_arm"], "r_arm": ["r_arm"], "l_hand": ["l_hand"],
    "r_hand": ["r_hand"], "l_leg": ["l_leg"], "r_leg": ["r_leg"], "overall": ["overall"],
    "arms": ["l_arm", "r_arm"], "hands": ["l_hand", "r_hand"], "legs": ["l_leg", "r_leg"],
    "scene": [],
}


def feature_group(name: str) -> tuple[str, str]:
    """Map a Detector A feature name to (region, family)."""
    if name.startswith("asym_"):
        return {"arm": "arms", "hand": "hands", "leg": "legs"}[name[5:]], "asymmetry"
    if name in ("has_subject4", "npers4"):
        return "scene", "people"
    if name in ("lhand_fine4", "rhand_fine4"):
        return ("l_hand" if name[0] == "l" else "r_hand"), "fine"
    for reg in sorted(REGION_TEXT, key=len, reverse=True):
        if name.startswith(reg + "_"):
            rest = name[len(reg) + 1:]
            if rest.startswith("spd"):
                return reg, "speed"
            if rest.startswith("jerk"):
                return reg, "jerk"
            if rest == "bp_lo":
                return reg, "rhythm_lo"
            if rest.startswith("bp_hi"):
                return reg, "rhythm_hi"
            if rest.startswith("vis"):
                return reg, "visibility"
    return "scene", "other"


def group_text(region: str, family: str) -> str:
    r = REGION_TEXT.get(region, region)
    if family == "speed":
        return f"{r} speed"
    if family == "jerk":
        return f"jerky {r} motion"
    if family == "rhythm_lo":
        return f"rhythmic {r} 0.5-1.5 Hz"
    if family == "rhythm_hi":
        return f"rhythmic {r} 1.5-3 Hz"
    if family == "visibility":
        return f"{r} visibility"
    if family == "asymmetry":
        return f"left/right {region} asymmetry"
    if family == "fine":
        return f"{r} finger motion"
    if family == "people":
        return "people in view / tracking"
    return f"{r} {family}"


class Explainer:
    """Call update(X_row) once per second; drivers(p_smooth) returns the current top groups."""

    def __init__(self, booster, feature_names: list[str], window: int = 5, top_k: int = 3,
                 min_p: float = 0.5, keep_frac: float = 0.8):
        self.booster = booster
        self.groups = [feature_group(n) for n in feature_names]
        self.keys = sorted(set(self.groups))
        idx = {k: i for i, k in enumerate(self.keys)}
        self.M = np.zeros((len(feature_names), len(self.keys)))
        for f, g in enumerate(self.groups):
            self.M[f, idx[g]] = 1.0
        self.hist = collections.deque(maxlen=window)
        self.miss = collections.deque(maxlen=window)    # fraction of each group's features missing (NaN)
        self.n_per = self.M.sum(axis=0)
        self.top_k, self.min_p, self.keep_frac = top_k, min_p, keep_frac
        self.current: list[tuple[str, str]] = []

    def update(self, X: np.ndarray | None) -> np.ndarray:
        """X: (1, n_features). Returns per-group contributions for this second."""
        if X is None:
            g = np.full(len(self.keys), np.nan)
            self.miss.append(np.ones(len(self.keys)))
        else:
            self.miss.append((~np.isfinite(np.asarray(X, float)[0])).astype(float) @ self.M / self.n_per)
            c = np.asarray(self.booster.predict(X, pred_contrib=True, num_threads=1))[0]
            g = c[:-1] @ self.M        # last column is the expected value (bias)
        self.hist.append(g)
        return g

    def smoothed(self) -> np.ndarray:
        H = np.array(self.hist)
        if not len(H) or not np.isfinite(H).any():
            return np.zeros(len(self.keys))
        with np.errstate(all="ignore"):
            return np.nan_to_num(np.nanmean(H, axis=0))

    def drivers(self, p_smooth: float) -> list[dict]:
        """Top positive groups (with hysteresis so highlights do not flicker).
        Empty when the smoothed score is below min_p."""
        s = self.smoothed()
        if not np.isfinite(p_smooth) or p_smooth < self.min_p:
            self.current = []
            return []
        order = [i for i in np.argsort(-s) if s[i] > 0.05]
        if not order:
            self.current = []
            return []
        top = s[order[0]]
        cand = [i for i in order if s[i] >= 0.2 * top][: self.top_k]
        prev = [self.keys.index(k) for k in self.current]
        kth = s[cand[-1]] if len(cand) == self.top_k else 0.05
        for i in prev:                       # hysteresis: keep previous drivers that stay close
            if i in cand or s[i] <= 0.05 or s[i] < self.keep_frac * kth:
                continue
            newer = [j for j in cand if j not in prev]
            if len(cand) < self.top_k:
                cand.append(i)
            elif newer:
                cand.remove(min(newer, key=lambda j: s[j]))
                cand.append(i)
        cand.sort(key=lambda j: -s[j])
        self.current = [self.keys[i] for i in cand]
        out = []
        for i in cand:
            reg, fam = self.keys[i]
            miss = float(np.mean([m[i] for m in self.miss])) if self.miss else 0.0
            # LightGBM routes missing values down a learned branch, so a "driver" can be the
            # absence of tracking. Say so instead of naming the motion.
            text = group_text(reg, fam) if miss < 0.5 else f"{REGION_TEXT.get(reg, reg)} not tracked (pose lost)"
            out.append(dict(key=f"{reg}:{fam}", region=reg, family=fam, text=text, missing=miss,
                            value=float(s[i]), share=float(s[i] / s[cand].sum()),
                            regions=GROUP_REGIONS.get(reg, [])))
        return out
