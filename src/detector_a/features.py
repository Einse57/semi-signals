"""Causal per-second pose features for seizure detection (Detector A).

Input: per-frame arrays as written by src.batch_extract (or produced live):
  t (F,), body_kpts (F,P,17,2), body_scores (F,P,17), body_boxes (F,P,5), sel (F,),
  hand_kpts (F,2,21,2), hand_scores (F,2,21)
Output: feature matrix (n_seconds, n_features) where row s summarises the trailing
windows ending at time s+1 (so row s only uses data with t < s+1).
"""
from __future__ import annotations

import numpy as np

REGIONS = {
    "head": [0, 1, 2], "l_arm": [5, 7], "r_arm": [6, 8], "l_hand": [9], "r_hand": [10],
    "l_leg": [11, 13, 15], "r_leg": [12, 14, 16], "overall": list(range(17)),
}
SCORE_THR = 0.3
BANDS = [(0.5, 1.5), (1.5, 3.0)]


def subject_series(d: dict) -> dict:
    """Per-frame subject signals: region speeds (F,R), region visibility (F,R),
    hand speeds (F,2), n_persons (F,), has_subject (F,)."""
    t = np.asarray(d["t"], float)
    F = len(t)
    sel = np.asarray(d["sel"]).astype(int)
    P = d["body_kpts"].shape[1]
    idx = np.clip(sel, 0, P - 1)
    K = d["body_kpts"][np.arange(F), idx].astype(np.float64)
    S = d["body_scores"][np.arange(F), idx].astype(np.float64)
    B = d["body_boxes"][np.arange(F), idx].astype(np.float64)
    has = sel >= 0
    K[~has] = np.nan
    S[~has] = 0
    diag = np.hypot(B[:, 2] - B[:, 0], B[:, 3] - B[:, 1])
    diag[~has] = np.nan
    K[S < SCORE_THR] = np.nan
    dt = np.diff(t, prepend=np.nan)
    dt[dt <= 0] = np.nan
    disp = np.linalg.norm(np.diff(K, axis=0, prepend=np.nan), axis=-1)  # (F,17)
    ref = np.fmax(diag, np.roll(diag, 1))
    speed = disp / ref[:, None] / dt[:, None]
    # signed displacement (x,y) for spectral features
    dxy = np.diff(K, axis=0, prepend=np.nan) / ref[:, None, None]
    names = list(REGIONS)
    R = np.full((F, len(names)), np.nan)
    V = np.zeros((F, len(names)))
    DX = np.full((F, len(names)), np.nan)
    for i, n in enumerate(names):
        j = REGIONS[n]
        with np.errstate(all="ignore"):
            R[:, i] = np.nanmean(speed[:, j], axis=1) if np.isfinite(speed[:, j]).any() else np.nan
            # projected 1-D motion signal: mean x+y displacement of region joints
            DX[:, i] = np.nanmean(dxy[:, j, 0] + dxy[:, j, 1], axis=1)
        V[:, i] = (S[:, j] >= SCORE_THR).mean(axis=1)
    HK = d["hand_kpts"].astype(np.float64).copy()
    HS = d["hand_scores"].astype(np.float64)
    HK[HS < SCORE_THR] = np.nan
    hd = np.linalg.norm(np.diff(HK, axis=0, prepend=np.nan), axis=-1)  # (F,2,21)
    with np.errstate(all="ignore"):
        H = np.nanmean(hd, axis=2) / ref[:, None] / dt[:, None]
    npers = np.isfinite(d["body_boxes"][:, :, 0]).sum(axis=1).astype(float)
    return dict(t=t, R=R, V=V, DX=DX, H=H, npers=npers, has=has.astype(float), names=names)


def _stats(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return x.mean(), x.max(), x.std()


def _bandpower(x, fs):
    x = np.where(np.isfinite(x), x, 0.0)
    if len(x) < 8 or not np.any(x):
        return [np.nan] * (len(BANDS) + 1)
    x = x - x.mean()
    w = np.hanning(len(x))
    f = np.fft.rfftfreq(len(x), 1 / fs)
    p = np.abs(np.fft.rfft(x * w)) ** 2 / len(x)
    tot = p[(f >= 0.25)].sum() + 1e-12
    out = [np.log10(p[(f >= lo) & (f < hi)].sum() + 1e-9) for lo, hi in BANDS]
    out.append(p[(f >= BANDS[1][0]) & (f < BANDS[1][1])].sum() / tot)
    return out


def feature_names(names=list(REGIONS)) -> list[str]:
    fn = []
    for n in names:
        for w in (4, 10):
            fn += [f"{n}_spd{w}_mean", f"{n}_spd{w}_max", f"{n}_spd{w}_std"]
        fn += [f"{n}_jerk4", f"{n}_bp_lo", f"{n}_bp_hi", f"{n}_bp_hirel", f"{n}_vis4"]
    fn += ["asym_arm", "asym_hand", "asym_leg", "has_subject4", "npers4", "lhand_fine4", "rhand_fine4"]
    return fn


def compute_features(d: dict, fs: float = 6.0, n_seconds: int | None = None,
                     seconds: list[int] | None = None) -> np.ndarray:
    s = subject_series(d)
    t = s["t"]
    if seconds is None:
        if n_seconds is None:
            n_seconds = int(np.floor(t[-1])) + 1 if len(t) else 0
        seconds = range(n_seconds)
    names = s["names"]
    rows = []
    # index helpers
    for sec in seconds:
        end = sec + 1.0
        i1 = np.searchsorted(t, end, side="left")
        i4 = np.searchsorted(t, end - 4.0, side="left")
        i10 = np.searchsorted(t, end - 10.0, side="left")
        row = []
        for k, n in enumerate(names):
            r4 = s["R"][i4:i1, k]
            r10 = s["R"][i10:i1, k]
            row += list(_stats(r4)) + list(_stats(r10))
            jr = np.diff(r4, n=2) if len(r4) > 2 else np.array([np.nan])
            jr = jr[np.isfinite(jr)]
            row.append(np.abs(jr).mean() if len(jr) else np.nan)
            row += _bandpower(s["DX"][i10:i1, k], fs)
            v = s["V"][i4:i1, k]
            row.append(v.mean() if len(v) else 0.0)
        def asym(a, b):
            ia, ib = names.index(a), names.index(b)
            la = np.nanmean(s["R"][i4:i1, ia]) if np.isfinite(s["R"][i4:i1, ia]).any() else np.nan
            lb = np.nanmean(s["R"][i4:i1, ib]) if np.isfinite(s["R"][i4:i1, ib]).any() else np.nan
            return abs(la - lb) / (la + lb + 1e-6)
        row += [asym("l_arm", "r_arm"), asym("l_hand", "r_hand"), asym("l_leg", "r_leg")]
        h = s["has"][i4:i1]; npr = s["npers"][i4:i1]
        row += [h.mean() if len(h) else 0.0, npr.mean() if len(npr) else 0.0]
        for side in (0, 1):
            x = s["H"][i4:i1, side]
            x = x[np.isfinite(x)]
            row.append(x.mean() if len(x) else np.nan)
        rows.append(row)
    return np.array(rows, dtype=np.float32).reshape(len(rows), -1)
