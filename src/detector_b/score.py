"""Score the VSViG LOPO outputs exactly like Detector A (src.detector_a.lopo.summarize)."""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from sklearn.metrics import roc_auc_score
from src.detector_a import scoring
from src.detector_a.lopo import summarize, THR_GRID, FA_TARGET
from src.detector_a.labels import read_labels, video_key, second_labels


def recs_from_slim(slim, xlsx):
    lab = read_labels(xlsx); out = {}
    for f in sorted(glob.glob(os.path.join(slim, "*.npz"))):
        name = os.path.basename(f)[:-4]
        if name == "pat04_no-Sz2P": continue
        pat, stem = name.split("_", 1); k = video_key(stem)
        if k and (pat, k) not in lab: continue
        eeg, clin = lab[(pat, k)] if k else (None, None)
        t = np.load(f)["t"]; n = int(np.floor(t[-1])) + 1
        out[name] = dict(name=name, pat=pat, y=second_labels(n, eeg, clin), eeg=eeg, clin=clin)
    return out


def counts(items, thr):
    tp = n = fa = 0; h = 0.0
    for r, p in items:
        s = scoring.score_record(scoring.events_from_prob(p, thr), r["y"], r["clin"], r["eeg"])
        tp += s["tp"]; n += s["n_sz"]; fa += s["fa"]; h += s["interictal_h"]
    return tp, n, fa, h


def nested_thr(run, p, R):
    files = sorted(glob.glob(os.path.join(run, f"inner_{p}_*.npz")))
    if len(files) < 3: return None
    items = [(R[k], z[k]) for f in files for z in [np.load(f)] for k in z.files]
    for thr in THR_GRID:
        tp, n, fa, h = counts(items, thr)
        if h > 0 and fa / h <= FA_TARGET: return float(thr)
    return 0.95


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--slim", required=True)
    ap.add_argument("--labels", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = recs_from_slim(a.slim, a.labels)
    pats = sorted({r["pat"] for r in R.values()})
    rows, aucs, have_nested = [], {}, True
    for p in pats:
        z = np.load(os.path.join(a.run, f"outer_{p}.npz"))
        thr = nested_thr(a.run, p, R); have_nested &= thr is not None
        yy, pp = [], []
        for k in z.files:
            r, pr = R[k], z[k]
            for rule, t in (("nested", thr), ("fixed0.5", 0.5)):
                if t is None: continue
                s = scoring.score_record(scoring.events_from_prob(pr, t), r["y"], r["clin"], r["eeg"])
                rows.append(dict(rule=rule, thr=t, pat=p, name=k, tp=s["tp"], n_sz=s["n_sz"], fa=s["fa"],
                                 interictal_h=s["interictal_h"], delay_clin=s["delay_clin"], delay_eeg=s["delay_eeg"]))
            m = r["y"] >= 0; yy.append(r["y"][m]); pp.append(pr[m])
        yy = np.concatenate(yy); pp = np.concatenate(pp)
        aucs[p] = float(roc_auc_score(yy, pp)) if len(set(yy)) == 2 else None
    os.makedirs(a.out, exist_ok=True)
    if have_nested:
        summarize(rows, aucs, a.out)
    else:
        fixed = [r for r in rows if r["rule"] == "fixed0.5"]
        summarize(fixed + [dict(r, rule="nested") for r in fixed], aucs, a.out)
        txt = open(os.path.join(a.out, "RESULTS.md")).read().split("## rule=fixed0.5")[1]
        open(os.path.join(a.out, "RESULTS.md"), "w").write("(nested rule: inner folds not finished)\n\n## rule=fixed0.5" + txt)
    print(open(os.path.join(a.out, "RESULTS.md")).read())
