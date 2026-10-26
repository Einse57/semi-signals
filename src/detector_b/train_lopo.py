"""VSViG-base LOPO retrain (preregistered: docs/vsvig_lopo_preregistration.md).
Outputs per-second probabilities for each held-out patient (outer) and inner-fold OOF probabilities."""
from __future__ import annotations
import argparse, glob, json, math, os, time
import numpy as np
import torch
from src.detector_a.labels import read_labels, video_key, second_labels

WIN = 30
EXCLUDE = ("pat04_no-Sz2P",)
EPOCHS, LR, WD, BS = 10, 3e-4, 1e-2, 32


def load(slim_dir, patch_dir, xlsx):
    lab = read_labels(xlsx); recs = []
    for f in sorted(glob.glob(os.path.join(slim_dir, "*.npz"))):
        name = os.path.basename(f)[:-4]
        if name in EXCLUDE:
            continue
        pat, stem = name.split("_", 1); k = video_key(stem)
        if k and (pat, k) not in lab:
            print("SKIP", name, flush=True); continue
        eeg, clin = lab[(pat, k)] if k else (None, None)
        d = np.load(f); t = d["t"]
        n = int(np.floor(t[-1])) + 1
        y = second_labels(n, eeg, clin)
        ends = np.searchsorted(t, np.arange(n) + 1.0)          # samples with t < s+1
        kp = np.concatenate([np.nan_to_num(d["K"]), d["S"].astype(np.float32)[..., None]], -1).astype(np.float32)
        kp[d["sel"] < 0] = 0
        recs.append(dict(name=name, pat=pat, y=y, ends=ends, kp=kp, P=os.path.join(patch_dir, name + ".patches.npy"),
                         eeg=eeg, clin=clin))
    return recs


_MM = {}


def _patches(path):
    # opened lazily per process (memmaps must not be pickled to DataLoader workers on Windows)
    if path not in _MM:
        _MM[path] = np.load(path, mmap_mode="r")
    return _MM[path]


def window(r, e):
    p = np.asarray(_patches(r["P"])[e - WIN:e], np.float32).transpose(0, 1, 4, 2, 3)   # (30,15,3,32,32)
    return p, r["kp"][e - WIN:e]


class DS(torch.utils.data.Dataset):
    """items: (rec_index, end, label); recs reduced to path + keypoints so workers pickle cheaply."""
    def __init__(self, items):
        recs = []; idx = {}
        for r, e, lab in items:
            if id(r) not in idx:
                idx[id(r)] = len(recs); recs.append(dict(P=r["P"], kp=r["kp"]))
        self.recs = recs; self.items = [(idx[id(r)], e, lab) for r, e, lab in items]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        ri, e, lab = self.items[i]; p, k = window(self.recs[ri], e)
        return torch.from_numpy(np.ascontiguousarray(p)), torch.from_numpy(np.ascontiguousarray(k)), torch.tensor(float(lab))


def train_items(recs):
    it = []
    for r in recs:
        for s, lab in enumerate(r["y"]):
            e = r["ends"][s]
            if lab < 0 or e < WIN:
                continue
            if lab == 0 and s % 2:
                continue
            it.append((r, int(e), int(lab)))
    return it


def make_model():
    from third_party.vsvig.VSViG import VSViG_base
    return VSViG_base()


def fit(recs, dev, log, workers=2):
    torch.manual_seed(0); np.random.seed(0)
    items = train_items(recs)
    npos = sum(1 for x in items if x[2] == 1); nneg = len(items) - npos
    m = make_model().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    g = torch.Generator(); g.manual_seed(0)
    dl = torch.utils.data.DataLoader(DS(items), batch_size=BS, shuffle=True, num_workers=workers, generator=g,
                                     drop_last=True, persistent_workers=workers > 0)
    steps = EPOCHS * len(dl); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    scaler = torch.amp.GradScaler("cuda")
    pw = torch.tensor(nneg / max(1, npos), device=dev)
    t0 = time.time()
    for ep in range(EPOCHS):
        m.train(); tot = 0.0
        for p, k, y in dl:
            p, k, y = p.to(dev, non_blocking=True), k.to(dev, non_blocking=True), y.to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                out = m(p, k).reshape(-1).float()
            out = out.clamp(1e-6, 1 - 1e-6)
            loss = -(pw * y * torch.log(out) + (1 - y) * torch.log(1 - out)).mean()
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        log(f"  epoch {ep + 1}/{EPOCHS} loss {tot / len(dl):.4f} ({len(items)} windows, pos {npos}) {time.time() - t0:.0f}s")
    return m


@torch.no_grad()
def predict(m, r, dev):
    m.eval(); n = len(r["y"]); prob = np.zeros(n, np.float32)
    ss = [s for s in range(n) if r["ends"][s] >= WIN]
    for i in range(0, len(ss), 64):
        b = ss[i:i + 64]; P, K = zip(*[window(r, int(r["ends"][s])) for s in b])
        with torch.autocast("cuda", dtype=torch.float16):
            o = m(torch.from_numpy(np.stack(P)).to(dev), torch.from_numpy(np.stack(K)).to(dev)).reshape(-1).float()
        prob[b] = o.cpu().numpy()
    return prob


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slim", required=True); ap.add_argument("--patches", required=True)
    ap.add_argument("--labels", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["outer", "inner"], default="outer")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, f"{a.stage}.log"), "a")
    def log(s):
        s = f"{time.strftime('%H:%M:%S')} {s}"; print(s, flush=True); logf.write(s + "\n"); logf.flush()
    dev = torch.device("cuda")
    recs = load(a.slim, a.patches, a.labels)
    pats = sorted({r["pat"] for r in recs})
    log(f"{len(recs)} records, {len(pats)} patients, stage {a.stage}")
    for p in pats:
        train = [r for r in recs if r["pat"] != p]
        if a.stage == "outer":
            f = os.path.join(a.out, f"outer_{p}.npz")
            if os.path.exists(f): continue
            log(f"outer fold {p}")
            m = fit(train, dev, log, a.workers)
            np.savez(f, **{r["name"]: predict(m, r, dev) for r in recs if r["pat"] == p})
            torch.save(m.state_dict(), os.path.join(a.out, f"outer_{p}.pth"))
        else:
            tp = sorted({r["pat"] for r in train}); groups = [tp[i::3] for i in range(3)]
            for gi, g in enumerate(groups):
                f = os.path.join(a.out, f"inner_{p}_{gi}.npz")
                if os.path.exists(f): continue
                log(f"inner fold {p} group {gi} {g}")
                m = fit([r for r in train if r["pat"] not in g], dev, log, a.workers)
                np.savez(f, **{r["name"]: predict(m, r, dev) for r in train if r["pat"] in g})
    log(f"STAGEDONE {a.stage}")
