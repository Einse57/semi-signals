"""Rebuild VSViG patches from the original videos at the frame indices and subject keypoints
of an existing extraction (slim npz: t, frame_idx, sel, K (F,15,2) VSViG order, S).
Writes <out>/<name>.patches.npy (F,15,32,32,3) uint8 (uncompressed, for mmap)."""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np
import cv2
from src.batch_extract import extract_vsvig_patches

cv2.setNumThreads(1)


def rebuild(slim, video, out):
    d = np.load(slim)
    fidx = d["frame_idx"]; sel = d["sel"]; K = d["K"]
    F = len(fidx)
    P = np.lib.format.open_memmap(out + ".tmp.npy", mode="w+", dtype=np.uint8, shape=(F, 15, 32, 32, 3))
    cap = cv2.VideoCapture(video)
    fi = -1; j = 0; t0 = time.time()
    while j < F:
        if not cap.grab():
            break
        fi += 1
        if fi < fidx[j]:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        if sel[j] >= 0:
            P[j] = extract_vsvig_patches(frame, K[j])
        j += 1
    cap.release()
    P.flush(); del P
    if j != F:
        raise RuntimeError(f"{video}: only {j}/{F} frames")
    os.replace(out + ".tmp.npy", out)
    return time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slim", required=True); ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.slim, "*.npz")))
    files = sorted(files, key=lambda f: -os.path.getsize(f))[a.shard::a.nshards]
    for f in files:
        name = os.path.basename(f)[:-4]
        pat, stem = name.split("_", 1)
        out = os.path.join(a.out, name + ".patches.npy")
        if os.path.exists(out):
            continue
        el = rebuild(f, os.path.join(a.videos, pat, stem + ".mp4"), out)
        print(f"{time.strftime('%H:%M:%S')} {name} done {el:.0f}s", flush=True)
    print("SHARDDONE", flush=True)
