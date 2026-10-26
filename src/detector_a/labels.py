"""Per-second labels for WU-SAHZU-EMU-Video (preregistered rules)."""
from __future__ import annotations
import numpy as np
import pandas as pd

ICTAL_LEN = 120.0


def _hms(x) -> float:
    if hasattr(x, "hour"):
        return x.hour * 3600 + x.minute * 60 + x.second
    h, m, s = str(x).split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def read_labels(xlsx: str) -> dict:
    """Returns {('pat01','Sz1'): (eeg_onset_s, clin_onset_s)}"""
    df = pd.read_excel(xlsx)
    df.columns = ["pat", "type", "sz", "eeg", "clin"]
    df["pat"] = df["pat"].ffill()
    out = {}
    for _, r in df.iterrows():
        out[(r.pat.lower(), r.sz)] = (_hms(r.eeg), _hms(r.clin))
    return out


def video_key(stem: str):
    """'Sz1PG' -> 'Sz1'; 'free' -> None"""
    if stem.startswith("Sz"):
        return stem[:3]
    return None


def second_labels(n_seconds: int, eeg: float | None, clin: float | None) -> np.ndarray:
    """1 ictal, 0 interictal, -1 excluded. Second s covers [s, s+1)."""
    y = np.zeros(n_seconds, np.int8)
    if clin is None:
        return y
    s = np.arange(n_seconds)
    y[(s + 1 > eeg) & (s < clin)] = -1
    y[(s >= np.floor(clin)) & (s < clin + ICTAL_LEN)] = 1
    y[s >= clin + ICTAL_LEN] = -1
    return y
