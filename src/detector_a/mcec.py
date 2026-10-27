"""MCEC_Seizure loader: MediaPipe-33 skeleton CSVs -> Detector A input dicts (COCO-17, 6 fps).
Mapping and rules preregistered in docs/mcec_crosssite_preregistration.md."""
from __future__ import annotations
import os, re, urllib.parse
import numpy as np

MP2COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
SRC_FPS, FPS = 25.0, 6.0


def _ts(s):
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def _read_files(folder):
    """-> list of (t_start, t_end, frames, kpts (n,33,2), vis (n,33)) sorted by filename start time."""
    out = []
    for f in os.listdir(folder):
        a_, b_ = urllib.parse.unquote(f)[:-4].split("_")[:2]
        a = np.genfromtxt(os.path.join(folder, f), delimiter=",", skip_header=1)
        a = a[np.lexsort((a[:, 1], a[:, 0]))]
        u = np.unique(a[:, 0])
        a = a.reshape(len(u), 33, 6)
        out.append((_ts(a_), _ts(b_), a[:, 0, 0].astype(int), a[:, :, 2:4], a[:, :, 5]))
    out.sort(key=lambda r: r[0])
    return out


def _read_folder(folder):
    """Seizure folders (one recording clock): sorted unique frame idx (N,), kpts (N,33,2), vis (N,33)."""
    fs = _read_files(folder)
    fr = [r[2] for r in fs]; K = [r[3] for r in fs]; V = [r[4] for r in fs]
    fr = np.concatenate(fr); K = np.concatenate(K); V = np.concatenate(V)
    o = np.argsort(fr, kind="stable"); fr, K, V = fr[o], K[o], V[o]
    _, ui = np.unique(fr, return_index=True)
    return fr[ui].astype(int), K[ui], V[ui]


def _to_dict(fr, K, V, x_scale=1.0):
    """One record spanning fr[0]..fr[-1]; nearest frame within 1/12 s at 6 fps, else missing."""
    t_src = (fr - fr[0]) / SRC_FPS
    tt = np.arange(0, t_src[-1] + 1e-9, 1 / FPS)
    j = np.clip(np.searchsorted(t_src, tt), 1, len(t_src) - 1) if len(t_src) > 1 else np.zeros(len(tt), int)
    if len(t_src) > 1:
        j = np.where(np.abs(t_src[j - 1] - tt) <= np.abs(t_src[j] - tt), j - 1, j)
    ok = np.abs(t_src[j] - tt) <= 1 / (2 * FPS) + 1e-9
    k = K[j][:, MP2COCO].astype(np.float32).copy()
    k[..., 0] *= x_scale
    s = V[j][:, MP2COCO].astype(np.float32)
    k[~ok] = np.nan; s[~ok] = 0
    F = len(tt)
    return dict(t=tt, body_kpts=k[:, None], body_scores=s[:, None],
                body_boxes=np.full((F, 1, 5), np.nan, np.float32),
                sel=np.where(ok, 0, -1).astype(np.int16))


def load_mcec(root, x_scale=1.0):
    """-> list of dict(name, pat, kind ('sz'|'norm'), d, n_src_frames)."""
    out = []
    for folder in sorted(os.listdir(root)):
        pat = folder.split("_")[0]
        if folder.endswith("_norm"):
            # runs of files contiguous in both filename time and frame index (excerpts from several videos)
            fs = _read_files(os.path.join(root, folder))
            runs = [[fs[0]]]
            for r in fs[1:]:
                p = runs[-1][-1]
                if r[2][0] == p[2][-1] + 1 and abs(r[0] - p[1] - 1 / SRC_FPS) < 0.1:
                    runs[-1].append(r)
                else:
                    runs.append([r])
            for i, run in enumerate(runs):
                fr = np.concatenate([r[2] for r in run]); K = np.concatenate([r[3] for r in run])
                V = np.concatenate([r[4] for r in run])
                out.append(dict(name=f"{folder}_r{i:02d}", pat=pat, kind="norm", d=_to_dict(fr, K, V, x_scale)))
        else:
            fr, K, V = _read_folder(os.path.join(root, folder))
            out.append(dict(name=folder, pat=pat, kind="sz", d=_to_dict(fr, K, V, x_scale)))
    return out
