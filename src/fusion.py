"""Late fusion of a per-second iEEG/EEG seizure probability (e.g. from a separate iEEG model) with a
per-second video probability (Detector A / VSViG).

STATUS: plumbing only. No public dataset pairs video with iEEG, so the fusion rule
below is fixed a priori and has NOT been validated on paired recordings.

Preregistered rule (default): noisy-OR
    p_fused = 1 - (1 - p_ieeg) * (1 - p_video)
with fallbacks:
    video missing  -> p_fused = p_ieeg
    iEEG missing   -> p_fused = p_video  (flagged)
    both missing   -> NaN
An optional logistic rule on [p_ieeg, p_video, video_missing] is provided for when
paired data exists; its coefficients must be fitted and preregistered then.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def align_per_second(t: np.ndarray, p: np.ndarray, t0: float, n_seconds: int,
                     max_gap: float = 2.0) -> np.ndarray:
    """Resample an irregular probability stream onto seconds [t0+s, t0+s+1).

    Value for second s = max of samples in that second (seizure-sensitive), or the
    most recent sample within ``max_gap`` seconds before it; else NaN (missing).
    """
    t = np.asarray(t, float); p = np.asarray(p, float)
    order = np.argsort(t); t, p = t[order], p[order]
    out = np.full(n_seconds, np.nan)
    for s in range(n_seconds):
        a, b = t0 + s, t0 + s + 1
        i, j = np.searchsorted(t, a, "left"), np.searchsorted(t, b, "left")
        seg = p[i:j]; seg = seg[np.isfinite(seg)]
        if len(seg):
            out[s] = seg.max()
        elif i > 0 and a - t[i - 1] <= max_gap and np.isfinite(p[i - 1]):
            out[s] = p[i - 1]
    return out


def noisy_or(p_ieeg: np.ndarray, p_video: np.ndarray) -> np.ndarray:
    pe = np.asarray(p_ieeg, float); pv = np.asarray(p_video, float)
    out = 1.0 - (1.0 - pe) * (1.0 - pv)
    out = np.where(np.isnan(pv), pe, out)
    out = np.where(np.isnan(pe), pv, out)
    return np.clip(out, 0.0, 1.0)


@dataclass
class LogisticFusion:
    """p = sigmoid(b0 + b1*logit(p_ieeg) + b2*logit(p_video)*(1-m) + b3*m), m = video missing.
    Coefficients must come from paired data (not available yet)."""
    b0: float = 0.0
    b1: float = 1.0
    b2: float = 1.0
    b3: float = 0.0

    def __call__(self, p_ieeg, p_video):
        eps = 1e-6
        pe = np.clip(np.asarray(p_ieeg, float), eps, 1 - eps)
        pv = np.asarray(p_video, float)
        m = np.isnan(pv).astype(float)
        pv = np.clip(np.nan_to_num(pv, nan=0.5), eps, 1 - eps)
        lg = lambda x: np.log(x / (1 - x))
        z = self.b0 + self.b1 * lg(pe) + self.b2 * lg(pv) * (1 - m) + self.b3 * m
        out = 1 / (1 + np.exp(-z))
        return np.where(np.isnan(np.asarray(p_ieeg, float)), np.nan, out)


def fuse(p_ieeg, p_video, rule: str = "noisy_or", **kw):
    if rule == "noisy_or":
        return noisy_or(p_ieeg, p_video)
    if rule == "logistic":
        return LogisticFusion(**kw)(p_ieeg, p_video)
    raise ValueError(rule)
