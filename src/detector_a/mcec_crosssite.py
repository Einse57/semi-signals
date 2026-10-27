"""MCEC_Seizure cross-site check for Detector A-body (docs/mcec_crosssite_preregistration.md).

Stages: `lopo` (A-body VSViG LOPO reference), `final` (all-14 model + threshold, frozen),
`mcec` (score MCEC with the frozen model), `fullA` (rescore saved full-Detector-A events with onset window).
"""
from __future__ import annotations
import argparse, json, os, pickle
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import scoring, lopo
from .body_only import load_vsvig, body_features, BODY_NAMES
from .labels import read_labels, video_key, second_labels
from .mcec import load_mcec

V = os.environ.get("SEMI_SIGNALS_WORK", "work")  # work dir with data/, results/, external/ (outside git)
KP, XLSX = f"{V}/data/kp", f"{V}/data/Label.xlsx"
CACHE = f"{V}/data/featcache_body"
OUT = f"{V}/results/mcec_crosssite"
MCEC = f"{V}/external/mcec/x"


def onset_score(events, y, onset):
    """Primary window: TP if an event overlaps [onset-30, onset+60]; FA if another event touches interictal."""
    tp, delay, fa = 0, np.nan, 0
    for s, e in events:
        if onset is not None and s < onset + 60 and e > onset - 30:
            if not tp:
                tp, delay = 1, s - onset
            continue
        if (y[s:e] == 0).any():
            fa += 1
    return tp, delay, fa


def summarize_rows(rows, key):
    tp = sum(r[key]["tp"] for r in rows); n = sum(r["n_sz"] for r in rows)
    fa = sum(r[key]["fa"] for r in rows); h = sum(r["interictal_h"] for r in rows)
    d = [r[key]["delay"] for r in rows if r[key]["tp"]]
    return dict(tp=tp, n=n, wilson=scoring.wilson(tp, n), fa=fa, h=h, fah=fa / h if h else np.nan,
                fah_ci=scoring.poisson_rate_ci(fa, h) if h else None,
                delay_median=float(np.median(d)) if d else None, delays=d)


def both_rules(events, y, clin, eeg):
    s = scoring.score_record(events, y, clin, eeg)
    tp, dl, fa = onset_score(events, y, clin)
    return dict(code=dict(tp=s["tp"], fa=s["fa"], delay=s["delay_clin"]), onset=dict(tp=tp, fa=fa, delay=dl)), s


def stage_lopo():
    recs = load_vsvig(KP, XLSX, CACHE)
    pats = sorted({r["pat"] for r in recs})
    rows, aucs, thrs, probs = [], {}, {}, {}
    for p in pats:
        train = [r for r in recs if r["pat"] != p]; test = [r for r in recs if r["pat"] == p]
        m = lopo.fit(train); thr = lopo.choose_thr(train); thrs[p] = thr
        yy, pp = [], []
        for r in test:
            pr = m.predict_proba(r["X"])[:, 1]; probs[r["name"]] = pr
            for rule, t in (("nested", thr), ("fixed0.5", 0.5)):
                ev = scoring.events_from_prob(pr, t)
                sc, s = both_rules(ev, r["y"], r["clin"], r["eeg"])
                rows.append(dict(rule=rule, thr=t, pat=p, name=r["name"], n_sz=s["n_sz"],
                                 interictal_h=s["interictal_h"], events=ev, **sc,
                                 delay_eeg=s["delay_eeg"]))
            k = r["y"] >= 0; yy.append(r["y"][k]); pp.append(pr[k])
        yy = np.concatenate(yy); pp = np.concatenate(pp)
        aucs[p] = float(roc_auc_score(yy, pp)) if len(set(yy)) == 2 else None
        print(p, "thr", thr, "auc", aucs[p], flush=True)
    np.savez_compressed(f"{OUT}/abody_lopo_oof.npz", **probs)
    res = dict(thr=thrs, aucs=aucs, rows=rows)
    for rule in ("nested", "fixed0.5"):
        rr = [r for r in rows if r["rule"] == rule]
        res[rule] = {k: summarize_rows(rr, k) for k in ("code", "onset")}
    vals = [a for a in aucs.values() if a is not None]
    res["auc_mean"] = float(np.mean(vals))
    json.dump(res, open(f"{OUT}/abody_lopo.json", "w"), default=float, indent=1)
    print(json.dumps({k: res[k] for k in ("nested", "fixed0.5", "auc_mean")}, default=float, indent=1))


def stage_final():
    recs = load_vsvig(KP, XLSX, CACHE)
    m = lopo.fit(recs); thr = lopo.choose_thr(recs)
    m.booster_.save_model(f"{OUT}/abody_all.txt")
    pickle.dump(m, open(f"{OUT}/abody_all.pkl", "wb"))
    json.dump(dict(threshold=thr, features=BODY_NAMES, n_records=len(recs),
                   patients=sorted({r["pat"] for r in recs})), open(f"{OUT}/abody_all.json", "w"), indent=1)
    print("final thr", thr, flush=True)


def _boot_auc(recs, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    sz = [r for r in recs if r["kind"] == "sz" and (r["y"] == 1).any()]
    nm = [r for r in recs if r["kind"] == "norm"]
    out = []
    for _ in range(B):
        pick = [sz[i] for i in rng.integers(0, len(sz), len(sz))] + [nm[i] for i in rng.integers(0, len(nm), len(nm))]
        y = np.concatenate([r["y"] for r in pick]); p = np.concatenate([r["p"] for r in pick])
        k = y >= 0
        out.append(roc_auc_score(y[k], p[k]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def _auc(recs):
    y = np.concatenate([r["y"] for r in recs]); p = np.concatenate([r["p"] for r in recs]); k = y >= 0
    return float(roc_auc_score(y[k], p[k])) if len(set(y[k])) == 2 else None


def stage_mcec():
    m = pickle.load(open(f"{OUT}/abody_all.pkl", "rb"))
    thr = json.load(open(f"{OUT}/abody_all.json"))["threshold"]
    res = dict(threshold=thr)
    for tag, xs in (("primary_1to1", 1.0), ("sens_x8of9", 8 / 9), ("sens_x2of3", 2 / 3)):
        recs = load_mcec(MCEC, x_scale=xs)
        for r in recs:
            X = body_features(r["d"]); n = len(X)
            r["y"] = second_labels(n, 0.0, 0.0) if r["kind"] == "sz" else np.zeros(n, int)
            r["p"] = m.predict_proba(X)[:, 1]
            r["dur_s"] = float(r["d"]["t"][-1]) if len(r["d"]["t"]) else 0.0
        out = dict()
        for rule, t in (("frozen", thr), ("fixed0.5", 0.5)):
            rows = []
            for r in recs:
                ev = scoring.events_from_prob(r["p"], t)
                onset = 0.0 if r["kind"] == "sz" else None
                sc, s = both_rules(ev, r["y"], onset, onset)
                rows.append(dict(name=r["name"], pat=r["pat"], kind=r["kind"], n_sz=s["n_sz"], n_s=len(r["y"]),
                                 interictal_h=s["interictal_h"], events=ev, pmax=float(r["p"].max()),
                                 psmax=float(scoring.smooth(r["p"]).max()), **sc))
            perpat = {}
            for p in sorted({r["pat"] for r in recs}):
                rr = [r for r in rows if r["pat"] == p]
                perpat[p] = dict(onset=summarize_rows(rr, "onset"), code=summarize_rows(rr, "code"),
                                 auc=_auc([r for r in recs if r["pat"] == p]))
            out[rule] = dict(onset=summarize_rows(rows, "onset"), code=summarize_rows(rows, "code"),
                             rows=rows, perpat=perpat)
        out["auc_pooled"] = _auc(recs)
        if tag == "primary_1to1":
            out["auc_pooled_ci"] = _boot_auc(recs)
            out["auc_excl4613"] = _auc([r for r in recs if r["name"] != "4613"])
            np.savez_compressed(f"{OUT}/mcec_probs.npz", **{r["name"]: r["p"] for r in recs})
        out["sz_mean_p"] = {r["name"]: float(r["p"][r["y"] == 1].mean()) for r in recs if r["kind"] == "sz"}
        out["norm_mean_p"] = float(np.concatenate([r["p"] for r in recs if r["kind"] == "norm"]).mean())
        out["norm_seconds"] = int(sum(len(r["y"]) for r in recs if r["kind"] == "norm"))
        res[tag] = out
        print(tag, json.dumps({k: out["frozen"][k] for k in ("onset", "code")}, default=float),
              "auc", out["auc_pooled"], flush=True)
    json.dump(res, open(f"{OUT}/mcec_scores.json", "w"), default=float, indent=1)


def stage_fullA():
    d = json.load(open(f"{V}/results/detector_a/lopo_rows.json"))
    oof = np.load(f"{V}/results/detector_a/oof_probs.npz")
    lab = read_labels(XLSX)
    out = {}
    for rule in ("nested", "fixed0.5"):
        rows = []
        for r in [r for r in d["rows"] if r["rule"] == rule]:
            pat, stem = r["name"].split("_", 1); k = video_key(stem)
            eeg, clin = lab[(pat, k)] if k else (None, None)
            y = second_labels(len(oof[r["name"]]), eeg, clin)
            ev = [tuple(e) for e in r["events"]]
            sc, s = both_rules(ev, y, clin, eeg)
            rows.append(dict(name=r["name"], n_sz=s["n_sz"], interictal_h=s["interictal_h"], **sc))
        out[rule] = {k: summarize_rows(rows, k) for k in ("code", "onset")}
    json.dump(out, open(f"{OUT}/fullA_rescored.json", "w"), default=float, indent=1)
    print(json.dumps(out, default=float, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["lopo", "final", "mcec", "fullA"])
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    globals()["stage_" + a.stage]()
