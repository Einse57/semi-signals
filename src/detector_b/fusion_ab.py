"""Preregistered A+B noisy-OR fusion scoring (docs/fusion_ab_preregistration.md).

  --stage a_inner : retrain Detector A on Detector B's 3 patient-grouped inner folds (no B data used).
  --stage score   : fuse and score; needs B outer + inner npz files.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from sklearn.metrics import roc_auc_score
from src.detector_a import scoring
from src.detector_a.lopo import load_records, fit, summarize, THR_GRID, FA_TARGET
from src.fusion import noisy_or


def groups_for(recs, p):
    tp = sorted({r["pat"] for r in recs if r["pat"] != p})
    return [tp[i::3] for i in range(3)]


def a_inner(recs, out):
    os.makedirs(out, exist_ok=True)
    for p in sorted({r["pat"] for r in recs}):
        for gi, g in enumerate(groups_for(recs, p)):
            f = os.path.join(out, f"a_inner_{p}_{gi}.npz")
            if os.path.exists(f): continue
            m = fit([r for r in recs if r["pat"] != p and r["pat"] not in g])
            np.savez_compressed(f, **{r["name"]: m.predict_proba(r["X"])[:, 1] for r in recs if r["pat"] in g})
            print("a_inner", p, gi, g, flush=True)


def fuse(pa, pb):
    n = min(len(pa), len(pb)); return noisy_or(pa[:n], pb[:n])


def counts(items, thr):
    tp = n = fa = 0; h = 0.0
    for r, p in items:
        y = r["y"][:len(p)]
        s = scoring.score_record(scoring.events_from_prob(p, thr), y, r["clin"], r["eeg"])
        tp += s["tp"]; n += s["n_sz"]; fa += s["fa"]; h += s["interictal_h"]
    return tp, n, fa, h


def pick(items):
    for thr in THR_GRID:
        tp, n, fa, h = counts(items, thr)
        if h > 0 and fa / h <= FA_TARGET: return float(thr)
    return 0.95


def score(recs, a_oof, a_dir, b_run, out, a_rows):
    R = {r["name"]: r for r in recs}; A = np.load(a_oof)
    rows, aucs = [], {}
    for p in sorted({r["pat"] for r in recs}):
        items = []
        for gi in range(3):
            za = np.load(os.path.join(a_dir, f"a_inner_{p}_{gi}.npz"))
            zb = np.load(os.path.join(b_run, f"inner_{p}_{gi}.npz"))
            items += [(R[k], fuse(za[k], zb[k])) for k in za.files]
        thr = pick(items)
        zb = np.load(os.path.join(b_run, f"outer_{p}.npz")); yy, pp = [], []
        for k in zb.files:
            r = R[k]; pf = fuse(A[k], zb[k]); y = r["y"][:len(pf)]
            for rule, t in (("nested", thr), ("fixed0.5", 0.5)):
                s = scoring.score_record(scoring.events_from_prob(pf, t), y, r["clin"], r["eeg"])
                rows.append(dict(rule=rule, thr=t, pat=p, name=k, tp=s["tp"], n_sz=s["n_sz"], fa=s["fa"],
                                 interictal_h=s["interictal_h"], delay_clin=s["delay_clin"], delay_eeg=s["delay_eeg"]))
            m = y >= 0; yy.append(y[m]); pp.append(pf[m])
        yy = np.concatenate(yy); pp = np.concatenate(pp)
        aucs[p] = float(roc_auc_score(yy, pp)) if len(set(yy)) == 2 else None
    os.makedirs(out, exist_ok=True)
    json.dump(dict(rows=rows, aucs=aucs), open(os.path.join(out, "fusion_rows.json"), "w"), default=str, indent=1)
    summarize(rows, aucs, out)
    # preregistered decision + patient-bootstrap differences vs Detector A (nested rule)
    ar = [r for r in a_rows if r["rule"] == "nested"]; fr = [r for r in rows if r["rule"] == "nested"]
    pats = sorted({r["pat"] for r in fr})
    agg = lambda L, ps: (sum(r["tp"] for p in ps for r in L if r["pat"] == p), sum(r["n_sz"] for p in ps for r in L if r["pat"] == p),
                         sum(r["fa"] for p in ps for r in L if r["pat"] == p), sum(r["interictal_h"] for p in ps for r in L if r["pat"] == p))
    ta, na, fa_a, ha = agg(ar, pats); tf, nf, fa_f, hf = agg(fr, pats)
    helps = (tf > ta and fa_f / hf <= fa_a / ha) or (fa_f / hf < fa_a / ha and tf >= ta)
    rng = np.random.default_rng(0); dr, dfa = [], []
    for _ in range(2000):
        sm = rng.choice(pats, len(pats)); a_ = agg(ar, sm); f_ = agg(fr, sm)
        if a_[1]: dr.append(f_[0] / f_[1] - a_[0] / a_[1])
        if a_[3]: dfa.append(f_[2] / f_[3] - a_[2] / a_[3])
    txt = (f"\n## Preregistered decision (nested rule)\n- Detector A: {ta}/{na}, {fa_a} FA in {ha:.2f} h = {fa_a/ha:.2f}/h\n"
           f"- Fused A+B: {tf}/{nf}, {fa_f} FA in {hf:.2f} h = {fa_f/hf:.2f}/h\n"
           f"- recall difference (fused - A): {tf/nf - ta/na:+.3f} (patient bootstrap 95% {np.percentile(dr,2.5):+.3f} to {np.percentile(dr,97.5):+.3f})\n"
           f"- FA/h difference (fused - A): {fa_f/hf - fa_a/ha:+.2f} (patient bootstrap 95% {np.percentile(dfa,2.5):+.2f} to {np.percentile(dfa,97.5):+.2f})\n"
           f"- fusion helps (preregistered criterion): {'YES' if helps else 'NO'}\n")
    open(os.path.join(out, "RESULTS.md"), "a").write(txt); print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a_inner", "score"], required=True)
    ap.add_argument("--kp", required=True); ap.add_argument("--cache"); ap.add_argument("--labels", required=True)
    ap.add_argument("--a_dir", required=True); ap.add_argument("--a_oof"); ap.add_argument("--a_rows")
    ap.add_argument("--b_run"); ap.add_argument("--out")
    a = ap.parse_args()
    recs = load_records(a.kp, a.labels, a.cache)
    if a.stage == "a_inner": a_inner(recs, a.a_dir)
    else: score(recs, a.a_oof, a.a_dir, a.b_run, a.out, json.load(open(a.a_rows))["rows"])
