"""SzCORE-style event scoring (preregistered parameters)."""
from __future__ import annotations
import numpy as np
from scipy import stats

PRE_TOL, POST_TOL = 30.0, 60.0
MIN_DUR, MERGE_GAP, SMOOTH = 3, 30, 5


def smooth(p: np.ndarray, w: int = SMOOTH) -> np.ndarray:
    """Trailing mean (causal)."""
    c = np.cumsum(np.insert(np.nan_to_num(p), 0, 0.0))
    out = np.empty_like(p, dtype=float)
    for i in range(len(p)):
        lo = max(0, i + 1 - w)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def events_from_prob(p: np.ndarray, thr: float) -> list[tuple[int, int]]:
    """[start, end) seconds of runs with smoothed p >= thr, >= MIN_DUR long, merged if gap < MERGE_GAP."""
    on = smooth(p) >= thr
    ev = []
    i = 0
    n = len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            ev.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for e in ev:
        if merged and e[0] - merged[-1][1] < MERGE_GAP:
            merged[-1][1] = e[1]
        else:
            merged.append(e)
    return [tuple(e) for e in merged if e[1] - e[0] >= MIN_DUR]


def score_record(events, y: np.ndarray, clin: float | None, eeg: float | None):
    """Returns dict(tp, n_sz, fa, interictal_h, delay_clin, delay_eeg)."""
    inter_h = (y == 0).sum() / 3600.0
    res = dict(tp=0, n_sz=0, fa=0, interictal_h=inter_h, delay_clin=np.nan, delay_eeg=np.nan, events=events)
    ref = None
    if clin is not None:
        res["n_sz"] = 1
        end = min(clin + 120.0, len(y))
        ref = (clin - PRE_TOL, end + POST_TOL)
    for s, e in events:
        if ref is not None and s < ref[1] and e > ref[0]:
            if not res["tp"]:
                res["tp"] = 1
                res["delay_clin"] = s - clin
                res["delay_eeg"] = s - eeg
            continue
        seg = y[s:e]
        if (seg == 0).any():
            res["fa"] += 1
    return res


def wilson(k, n, a=0.05):
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - a / 2)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def poisson_rate_ci(k, T, a=0.05):
    lo = 0.0 if k == 0 else stats.chi2.ppf(a / 2, 2 * k) / 2
    hi = stats.chi2.ppf(1 - a / 2, 2 * k + 2) / 2
    return (lo / T, hi / T)
