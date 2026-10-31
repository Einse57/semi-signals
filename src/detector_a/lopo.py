"""Leave-one-patient-out training/eval of Detector A (preregistered protocol)."""
from __future__ import annotations
import argparse, json, glob, os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from .features import compute_features, feature_names
from .labels import read_labels, video_key, second_labels
from . import scoring

PARAMS = dict(num_leaves=15, n_estimators=300, learning_rate=0.05, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0,
              n_jobs=4, verbose=-1)
THR_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
FA_TARGET = 1.0


def load_records(kp_dir, xlsx, cache=None, exclude=("pat04_no-Sz2P",)):
    lab = read_labels(xlsx)
    recs = []
    for f in sorted(glob.glob(os.path.join(kp_dir, "*.npz"))):
        name = os.path.basename(f)[:-4]
        if name in exclude:
            continue
        pat, stem = name.split("_", 1)
        k = video_key(stem)
        if k and (pat, k) not in lab:
            print(f"SKIP {name}: no row in Label.xlsx (unlabelled seizure video)", flush=True)
            continue
        eeg, clin = lab[(pat, k)] if k else (None, None)
        cf = os.path.join(cache, name + ".feat.npz") if cache else None
        if cf and os.path.exists(cf):
            z = np.load(cf); X = z["X"]
        else:
            d = dict(np.load(f))
            X = compute_features(d)
            if cf:
                os.makedirs(cache, exist_ok=True); np.savez(cf, X=X)
        y = second_labels(len(X), eeg, clin)
        recs.append(dict(name=name, pat=pat, X=X, y=y, eeg=eeg, clin=clin))
    return recs


def fit(recs):
    X = np.concatenate([r["X"][r["y"] >= 0] for r in recs])
    y = np.concatenate([r["y"][r["y"] >= 0] for r in recs])
    m = lgb.LGBMClassifier(**PARAMS, scale_pos_weight=(y == 0).sum() / max(1, (y == 1).sum()))
    m.fit(X, y)
    return m


def eval_thr(recs, probs, thr):
    tp = n = fa = 0; h = 0.0
    for r, p in zip(recs, probs):
        s = scoring.score_record(scoring.events_from_prob(p, thr), r["y"], r["clin"], r["eeg"])
        tp += s["tp"]; n += s["n_sz"]; fa += s["fa"]; h += s["interictal_h"]
    return tp, n, fa, h


def choose_thr(train):
    pats = sorted({r["pat"] for r in train})
    oof_recs, oof_p = [], []
    for p in pats:
        m = fit([r for r in train if r["pat"] != p])
        for r in train:
            if r["pat"] == p:
                oof_recs.append(r); oof_p.append(m.predict_proba(r["X"])[:, 1])
    for thr in THR_GRID:
        tp, n, fa, h = eval_thr(oof_recs, oof_p, thr)
        if h > 0 and fa / h <= FA_TARGET:
            return float(thr)
    return 0.95


def run(kp_dir, xlsx, out, cache=None, nested=True):
    recs = load_records(kp_dir, xlsx, cache)
    pats = sorted({r["pat"] for r in recs})
    rows = []; aucs = {}
    for p in pats:
        train = [r for r in recs if r["pat"] != p]
        test = [r for r in recs if r["pat"] == p]
        m = fit(train)
        thr = choose_thr(train) if nested else 0.5
        yy, pp = [], []
        for r in test:
            pr = m.predict_proba(r["X"])[:, 1]
            r["prob"] = pr
            for t_name, t in (("nested", thr), ("fixed0.5", 0.5)):
                s = scoring.score_record(scoring.events_from_prob(pr, t), r["y"], r["clin"], r["eeg"])
                rows.append(dict(rule=t_name, thr=t, pat=p, name=r["name"], tp=s["tp"], n_sz=s["n_sz"], fa=s["fa"],
                                 interictal_h=s["interictal_h"], delay_clin=s["delay_clin"], delay_eeg=s["delay_eeg"],
                                 events=s["events"]))
            k = r["y"] >= 0
            yy.append(r["y"][k]); pp.append(pr[k])
        yy = np.concatenate(yy); pp = np.concatenate(pp)
        aucs[p] = float(roc_auc_score(yy, pp)) if len(set(yy)) == 2 else None
        print(p, "thr", thr, "auc", aucs[p], flush=True)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "oof_probs.npz"), **{r["name"]: r["prob"] for r in recs})
    json.dump(dict(rows=rows, aucs=aucs), open(os.path.join(out, "lopo_rows.json"), "w"), default=str, indent=1)
    summarize(rows, aucs, out)
    # final model on all patients for the demo
    m = fit(recs)
    m.booster_.save_model(os.path.join(out, "detector_a_all.txt"))
    json.dump(dict(threshold=choose_thr(recs) if nested else 0.5, features=feature_names()),
              open(os.path.join(out, "detector_a_all.json"), "w"), indent=1)


def summarize(rows, aucs, out):
    lines = []
    for rule in ("nested", "fixed0.5"):
        R = [r for r in rows if r["rule"] == rule]
        tp = sum(r["tp"] for r in R); n = sum(r["n_sz"] for r in R)
        fa = sum(r["fa"] for r in R); h = sum(r["interictal_h"] for r in R)
        lo, hi = scoring.wilson(tp, n); flo, fhi = scoring.poisson_rate_ci(fa, h)
        # patient bootstrap
        pats = sorted({r["pat"] for r in R}); rng = np.random.default_rng(0)
        by = {p: [r for r in R if r["pat"] == p] for p in pats}
        bs_rec, bs_fa = [], []
        for _ in range(2000):
            sm = rng.choice(pats, len(pats))
            t_ = sum(r["tp"] for p in sm for r in by[p]); n_ = sum(r["n_sz"] for p in sm for r in by[p])
            f_ = sum(r["fa"] for p in sm for r in by[p]); h_ = sum(r["interictal_h"] for p in sm for r in by[p])
            if n_: bs_rec.append(t_ / n_)
            if h_: bs_fa.append(f_ / h_)
        dc = np.array([r["delay_clin"] for r in R if r["tp"]], float)
        de = np.array([r["delay_eeg"] for r in R if r["tp"]], float)
        lines.append(f"## rule={rule}\n")
        lines.append(f"- seizures detected: {tp}/{n} = {tp/n:.2f} (Wilson 95% {lo:.2f}-{hi:.2f}; patient bootstrap {np.percentile(bs_rec,2.5):.2f}-{np.percentile(bs_rec,97.5):.2f})")
        lines.append(f"- false-positive flags: {fa} in {h:.2f} interictal h = {fa/h:.2f}/h (Poisson 95% {flo:.2f}-{fhi:.2f}; patient bootstrap {np.percentile(bs_fa,2.5):.2f}-{np.percentile(bs_fa,97.5):.2f})")
        if len(dc):
            lines.append(f"- delay vs clinical onset: median {np.median(dc):.0f} s (IQR {np.percentile(dc,25):.0f} to {np.percentile(dc,75):.0f})")
            lines.append(f"- delay vs EEG onset: median {np.median(de):.0f} s (IQR {np.percentile(de,25):.0f} to {np.percentile(de,75):.0f})")
        lines.append("\n| patient | thr | detected | false-positive flags | interictal h |\n|---|---|---|---|---|")
        for p in pats:
            lines.append(f"| {p} | {by[p][0]['thr']} | {sum(r['tp'] for r in by[p])}/{sum(r['n_sz'] for r in by[p])} | {sum(r['fa'] for r in by[p])} | {sum(r['interictal_h'] for r in by[p]):.2f} |")
        lines.append("")
    lines.append("## per-second AUROC (held-out patient)\n")
    for p, a in aucs.items():
        lines.append(f"- {p}: {a if a is None else round(a,3)}")
    v = [a for a in aucs.values() if a is not None]
    lines.append(f"- mean {np.mean(v):.3f}")
    open(os.path.join(out, "RESULTS.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--cache")
    a = ap.parse_args()
    run(a.kp, a.labels, a.out, a.cache)
