"""Seizure-detection demo: video + skeleton overlay + live scores.

  python -m src.run_seizure_demo --input pat10/Sz2P.mp4 --labels Label.xlsx \
      --detector-a models/detector_a_all.txt --vsvig models/vsvig_base.xml [--ieeg probs.csv]

Per sampled frame (6 fps): person detection + body pose + hands (OpenVINO iGPU when
available), subject tracking, VSViG patches.
Every second: Detector A (pose features + LightGBM) score.
Every 0.5 s: Detector B (VSViG-base, optional) score on the last 5 s.
Optional: per-second iEEG probabilities from a CSV (columns t,p) replayed alongside and
fused with noisy-OR. Such replays are UNPAIRED (different patients) unless stated.

Layout: video on the left; on the right a panel with the smoothed Detector A probability
(dashed alarm threshold, shaded alarm seconds, current value), the top drivers of the score
(LightGBM pred_contrib once per second, grouped by body region and feature family, averaged
over 5 s), per-region motion graphs with the driver regions highlighted, and a greyed-out
EEG lane reserved for a future paired EEG stream.

Keys: q quit, space pause, f fast-forward toggle (no real-time pacing).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

WIN_S = 12.0          # seconds of keypoint history kept for features
CHART_S = 90.0        # seconds shown on the chart
WIN_NAME = "semi-signals | seizure detection demo"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Video seizure detection demo")
    p.add_argument("--input", required=True, help="video file")
    p.add_argument("--detector-a", default="models/detector_a_all.txt",
                   help="LightGBM model (txt), or 'hf' to fetch the SHA-256-checked model from Hugging Face")
    p.add_argument("--detector-a-meta", default=None, help="json with threshold (default: <model>.json)")
    p.add_argument("--vsvig", default="models/vsvig_base.xml", help="VSViG OpenVINO IR / ONNX; '' to disable")
    p.add_argument("--labels", default=None, help="Label.xlsx to draw EEG/clinical onset markers")
    p.add_argument("--ieeg", default=None, help="CSV with per-second iEEG probability (t,p)")
    p.add_argument("--ieeg-offset", type=float, default=0.0, help="seconds added to iEEG times")
    p.add_argument("--ieeg-label", default="iEEG (unpaired replay)")
    p.add_argument("--device", default="auto", help="auto | gpu | cpu | cuda")
    p.add_argument("--fps", type=float, default=6.0, help="analysis sample rate")
    p.add_argument("--start", type=float, default=0.0, help="start time in video [s]")
    p.add_argument("--no-hands", action="store_true")
    p.add_argument("--fast", action="store_true", help="no real-time pacing")
    p.add_argument("--headless", action="store_true", help="no window (use with --save)")
    p.add_argument("--save", default=None, help="write the rendered demo to this mp4")
    p.add_argument("--log", default=None, help="write per-second scores to this csv")
    p.add_argument("--width", type=int, default=960, help="video width in the window")
    p.add_argument("--panel-width", type=int, default=640, help="width of the side panel")
    p.add_argument("--no-explain", action="store_true", help="skip pred_contrib drivers")
    p.add_argument("--end", type=float, default=None, help="stop at this video time [s]")
    p.add_argument("--timing", action="store_true", help="print a timing summary at the end")
    return p.parse_args(argv)


class VSViGModel:
    def __init__(self, path: str, device: str):
        self.kind = None
        if path.endswith(".xml"):
            import openvino as ov
            core = ov.Core()
            dev = "GPU" if device == "gpu" and "GPU" in core.available_devices else "CPU"
            cfg = {"INFERENCE_PRECISION_HINT": "f32"} if dev == "GPU" else {}
            self.c = core.compile_model(path, dev, cfg)
            self.kind, self.dev = "ov", dev
        else:
            import onnxruntime as ort
            self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.kind, self.dev = "ort", "CPU"

    def __call__(self, patches, kpts) -> float:
        if self.kind == "ov":
            return float(np.asarray(self.c([patches, kpts])[0]).reshape(-1)[0])
        return float(self.s.run(None, {"patches": patches, "kpts": kpts})[0].reshape(-1)[0])


def load_ieeg(path, offset):
    import csv
    t, p = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            t.append(float(r["t"]) + offset); p.append(float(r["p"]))
    return np.array(t), np.array(p)


def label_for_video(xlsx, video):
    from src.detector_a.labels import read_labels, video_key
    v = Path(video)
    stem, pat = v.stem, v.parent.name.lower()
    if "_" in stem and stem.lower().startswith("pat"):   # flattened name e.g. pat10_Sz2P
        pat, stem = stem.split("_", 1)[0].lower(), stem.split("_", 1)[1]
    k = video_key(stem)
    if not k:
        return None
    return read_labels(xlsx).get((pat, k))


def draw_chart(w, h, now, series, thr, marks, alarm, span=CHART_S):
    """series: list of (name, colour, times, values)."""
    CHART_S = span
    t_disp = now
    now = max(now, span)
    img = np.full((h, w, 3), 24, np.uint8)
    L, R, T, B = 60, 10, 10, 30
    pw, ph = w - L - R, h - T - B
    t0 = now - CHART_S
    X = lambda t: int(L + (t - t0) / CHART_S * pw)
    Y = lambda v: int(T + (1 - v) * ph)
    for v in (0, 0.5, 1):
        cv2.line(img, (L, Y(v)), (w - R, Y(v)), (60, 60, 60), 1)
        cv2.putText(img, f"{v:.1f}", (15, Y(v) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    if thr is not None:
        cv2.line(img, (L, Y(thr)), (w - R, Y(thr)), (0, 0, 160), 1)
    for name, tm, col in marks:
        if tm is not None and t0 <= tm <= now:
            cv2.line(img, (X(tm), T), (X(tm), T + ph), col, 2)
            cv2.putText(img, name, (X(tm) + 4, T + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    lx = L + 5
    for name, col, ts, vs in series:
        ts = np.asarray(ts); vs = np.asarray(vs, float)
        k = (ts >= t0) & np.isfinite(vs)
        pts = np.array([[X(a), Y(min(max(b, 0), 1))] for a, b in zip(ts[k], vs[k])], np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, col, 2)
        cv2.putText(img, name, (lx, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        lx += 18 + 9 * len(name)
    cv2.putText(img, f"t = {t_disp:6.1f} s", (w - 130, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if alarm:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
    return img


def main(argv=None):
    a = parse_args(argv)
    from src.hf_models import resolve_detector_a
    a.detector_a = resolve_detector_a(a.detector_a)
    from src import config  # applies the OpenVINO iGPU patch for rtmlib
    if a.device == "auto":
        device = config.INFERENCE_DEVICE
        try:
            import onnxruntime as ort
            if device == "cpu" and "CUDAExecutionProvider" in ort.get_available_providers():
                device = "cuda"
        except Exception:
            pass
    else:
        device = a.device
    backend = "openvino" if device == "gpu" else "onnxruntime"
    print(f"[demo] pose models: backend={backend} device={device}")
    from src.detector import PersonDetector
    from src.pose_estimator import PoseEstimator
    from src.hand_estimator import HandEstimator
    from src.batch_extract import SubjectTracker, extract_vsvig_patches, VSVIG_COCO_ORDER, MAX_PERSONS
    from src.detector_a.features import compute_features, subject_series, feature_names, REGIONS
    from src.explain import Explainer
    from src import explain_view as ev
    from src.detector_a.scoring import SMOOTH
    from src.display import draw_skeleton, draw_hand_skeleton
    from src.fusion import noisy_or

    det = PersonDetector(device=device, backend=backend, score_thr=0.3)
    pose = PoseEstimator(device=device, backend=backend)
    hand = None if a.no_hands else HandEstimator(device=device, backend=backend)

    booster, thr_a = None, 0.5
    if a.detector_a and os.path.exists(a.detector_a):
        import lightgbm as lgb
        booster = lgb.Booster(model_file=a.detector_a)
        mp = a.detector_a_meta or os.path.splitext(a.detector_a)[0] + ".json"
        meta = json.load(open(mp)) if os.path.exists(mp) else {}
        thr_a = meta.get("threshold", 0.5)
        print(f"[demo] detector A loaded, threshold {thr_a}")
    explainer = None
    if booster is not None and not a.no_explain:
        fnames = meta.get("features") or feature_names()
        assert len(fnames) == booster.num_feature(), "feature names do not match the model"
        explainer = Explainer(booster, fnames)
    vs = None
    if a.vsvig and os.path.exists(a.vsvig):
        vs = VSViGModel(a.vsvig, device if device == "gpu" else "cpu")
        print(f"[demo] VSViG loaded on {vs.dev}")
    lab = label_for_video(a.labels, a.input) if a.labels else None
    ieeg = load_ieeg(a.ieeg, a.ieeg_offset) if a.ieeg else None

    cap = cv2.VideoCapture(a.input)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if a.start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / src_fps
    chart_span = float(min(CHART_S, max(20.0, dur)))
    disp_w = a.width; disp_h = int(H * disp_w / W)
    pan_w = a.panel_width; out_h = max(disp_h, 540)
    hp, hd, hm, he = ev.layout(out_h)
    writer = None
    if a.save:
        writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (disp_w + pan_w, out_h))
    if not a.headless:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, disp_w + pan_w, out_h)

    maxlen = int(WIN_S * a.fps) + 2
    buf = {k: collections.deque(maxlen=maxlen) for k in
           ("t", "body_kpts", "body_scores", "body_boxes", "sel", "hand_kpts", "hand_scores", "patches", "vkp")}
    tracker = SubjectTracker()
    hist = dict(tA=[], pA=[], pAs=[], alA=[], drv=[], tB=[], pB=[], tF=[], pF=[])
    mot_t, mot = [], {r: [] for r in REGIONS}
    drivers, panel, new_data = [], None, True
    tm = collections.defaultdict(float); tn = collections.defaultdict(int)
    n_samples = 0; run0 = time.perf_counter()
    next_sample = a.start
    next_second = int(np.floor(a.start)) + 1
    last_b = -1e9
    fi = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    wall0 = time.perf_counter(); vid0 = a.start
    paused, fast = False, a.fast
    last_draw = None
    inf_times = collections.deque(maxlen=60)
    logf = open(a.log, "w") if a.log else None
    if logf:
        logf.write("t,p_video_a,p_video_a_smooth,p_video_b,p_ieeg,p_fused,alarm_a,drivers,driver_regions,driver_speed,"
                   + ",".join(f"speed_{r_}" for r_ in REGIONS) + "\n")
    sel_kp = sel_sc = None; hres = None
    while True:
        if paused:
            k = cv2.waitKey(30) & 0xFF
            if k == ord(" "):
                paused = False; wall0 = time.perf_counter(); vid0 = t
            elif k == ord("q"):
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / src_fps
        fi += 1
        if a.end is not None and t >= a.end:
            break
        if t + 1e-6 >= next_sample:
            next_sample += 1.0 / a.fps
            t_inf = time.perf_counter()
            boxes = det.detect(frame)
            boxes = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:MAX_PERSONS]
            bk = np.full((MAX_PERSONS, 17, 2), np.nan, np.float32); bs = np.zeros((MAX_PERSONS, 17), np.float32)
            bb = np.full((MAX_PERSONS, 5), np.nan, np.float32)
            hk = np.full((2, 21, 2), np.nan, np.float32); hs = np.zeros((2, 21), np.float32)
            sel = tracker.update(boxes)
            persons = pose.estimate(frame, boxes) if boxes else []
            for i, p in enumerate(persons):
                bk[i], bs[i] = p["keypoints"], p["scores"]; bb[i, :5] = np.asarray(p["bbox"], np.float32)[:5]
            hres = None
            if sel >= 0 and hand is not None:
                hres = hand.estimate(frame, bk[sel], bs[sel])
                for j, side in enumerate(("left", "right")):
                    if hres[side] is not None:
                        hk[j], hs[j] = hres[side]["keypoints"], hres[side]["scores"]
            if sel >= 0:
                patch = extract_vsvig_patches(frame, bk[sel][VSVIG_COCO_ORDER])
                vkp = np.concatenate([bk[sel][VSVIG_COCO_ORDER], bs[sel][VSVIG_COCO_ORDER, None]], -1)
                sel_kp, sel_sc = bk[sel].copy(), bs[sel].copy()
            else:
                patch = np.zeros((15, 32, 32, 3), np.uint8); vkp = np.zeros((15, 3), np.float32)
                sel_kp = sel_sc = None
            for k_, v_ in (("t", t), ("body_kpts", bk), ("body_scores", bs), ("body_boxes", bb), ("sel", sel),
                           ("hand_kpts", hk), ("hand_scores", hs), ("patches", patch), ("vkp", vkp)):
                buf[k_].append(v_)
            # Detector B every 0.5 s
            if vs is not None and len(buf["t"]) >= 30 and t - last_b >= 0.5:
                last_b = t
                P = np.ascontiguousarray(np.stack(list(buf["patches"])[-30:]).transpose(0, 1, 4, 2, 3)[None], np.float32)
                K = np.ascontiguousarray(np.stack(list(buf["vkp"])[-30:])[None], np.float32)
                hist["tB"].append(t); hist["pB"].append(vs(P, K))
            inf_times.append(time.perf_counter() - t_inf)
            n_samples += 1; new_data = True
        # Detector A once per completed second
        while t >= next_second:
            sec = next_second - 1
            pa = np.nan; X = None; n_mot0 = len(mot_t)
            t_a = time.perf_counter()
            if booster is not None and len(buf["t"]) > 3:
                d = {k_: np.array(buf[k_]) for k_ in ("t", "body_kpts", "body_scores", "body_boxes", "sel", "hand_kpts", "hand_scores")}
                X = compute_features(d, fs=a.fps, seconds=[sec])
                pa = float(booster.predict(X)[0])
                ss = subject_series(d)          # per-frame region speeds for the motion graphs
                kk = (ss["t"] >= sec) & (ss["t"] < sec + 1)
                for i_, tt_ in zip(np.where(kk)[0], ss["t"][kk]):
                    mot_t.append(float(tt_))
                    for j_, r_ in enumerate(ss["names"]):   # display: trailing 0.5 s mean of joint speed
                        w_ = ss["R"][max(0, i_ - 2):i_ + 1, j_]
                        mot[r_].append(float(np.nanmean(w_)) if np.isfinite(w_).any() else np.nan)
            tm["detA"] += time.perf_counter() - t_a; tn["detA"] += 1
            hist["tA"].append(sec + 1.0); hist["pA"].append(pa)
            recent = [x for x in hist["pA"][-SMOOTH:]]
            hist["pAs"].append(float(np.nanmean(recent)) if np.isfinite(recent).any() else np.nan)
            hist["alA"].append(len(hist["pAs"]) >= 3 and all(np.nan_to_num(hist["pAs"][-3:]) >= thr_a))
            if explainer is not None:
                t_e = time.perf_counter()
                explainer.update(X)
                drivers = explainer.drivers(hist["pAs"][-1])
                tm["explain"] += time.perf_counter() - t_e; tn["explain"] += 1
            hist["drv"].append(drivers)
            with np.errstate(all="ignore"):
                sec_speed = {r_: (float(np.nanmean(v_[n_mot0:])) if np.isfinite(v_[n_mot0:]).any() else np.nan)
                             for r_, v_ in mot.items()}
            hr_ = sorted({r_ for d_ in drivers for r_ in d_["regions"]})
            drv_speed = float(np.nanmean([sec_speed[r_] for r_ in hr_])) if hr_ and np.isfinite([sec_speed[r_] for r_ in hr_]).any() else np.nan
            new_data = True
            pb = hist["pB"][-1] if hist["pB"] and t - hist["tB"][-1] < 1.5 else np.nan
            pv = hist["pAs"][-1]
            pe = np.nan
            if ieeg is not None:
                k = (ieeg[0] >= sec) & (ieeg[0] < sec + 1)
                pe = float(ieeg[1][k].max()) if k.any() else np.nan
            pf = float(noisy_or(np.array([pe]), np.array([pv]))[0])
            hist["tF"].append(sec + 1.0); hist["pF"].append(pf)
            if logf:
                dtxt = "|".join(f"{d_['text']}={d_['value']:.2f}" for d_ in drivers)
                logf.write(f"{sec},{pa:.4f},{pv:.4f},{pb:.4f},{pe:.4f},{pf:.4f},{int(hist['alA'][-1])},{dtxt},{'|'.join(hr_)},{drv_speed:.4f},"
                           + ",".join(f"{sec_speed[r_]:.4f}" for r_ in REGIONS) + "\n")
            next_second += 1
        # draw
        t_d = time.perf_counter()
        vis = cv2.resize(frame, (disp_w, disp_h))
        sx = disp_w / W
        hl_regions = {r for d_ in drivers for r in d_["regions"]}
        if sel_kp is not None:
            draw_skeleton(vis, sel_kp * sx, sel_sc, score_thr=0.3)
            if hres is not None:
                for side in ("left", "right"):
                    if hres.get(side) is not None:
                        hd_ = dict(hres[side]); hd_["keypoints"] = hd_["keypoints"] * sx
                        draw_hand_skeleton(vis, hd_, score_thr=0.3)
            for r_ in hl_regions - {"overall"}:      # ring the joints of the driver regions
                for j_ in REGIONS[r_]:
                    if sel_sc[j_] >= 0.3:
                        cv2.circle(vis, tuple(int(v) for v in sel_kp[j_] * sx), 11, ev.HILITE, 2, cv2.LINE_AA)
        alarm_a = bool(hist["alA"] and hist["alA"][-1])
        alarm_b = len(hist["pB"]) >= 6 and np.mean(hist["pB"][-6:]) >= 0.5
        txt = f"Detector A (pose features): {np.nan_to_num(hist['pAs'][-1]) if hist['pAs'] else 0:.2f}"
        cv2.putText(vis, txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if alarm_a else (0, 255, 0), 2)
        if vs is not None:
            txt = f"Detector B (VSViG-base, demo only): {hist['pB'][-1] if hist['pB'] else 0:.2f}"
            cv2.putText(vis, txt, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if alarm_b else (255, 200, 0), 2)
        if alarm_a or alarm_b:
            cv2.putText(vis, "SEIZURE ALERT", (disp_w - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        if inf_times:
            cv2.putText(vis, f"analysis {1.0 / max(1e-3, np.mean(inf_times)):.0f} frames/s ({device})", (15, disp_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if panel is None or new_data:          # the panel only changes when new samples arrive
            extra = []
            if vs is not None:
                extra.append(("B VSViG", (255, 200, 0), hist["tB"], hist["pB"]))
            if ieeg is not None:
                extra.append((a.ieeg_label, (255, 0, 255), ieeg[0], ieeg[1]))
                extra.append(("fused", (255, 255, 255), hist["tF"], hist["pF"]))
            marks = []
            if lab:
                marks = [("EEG onset", lab[0], (0, 255, 255)), ("clinical onset", lab[1], (0, 0, 255))]
            p_now = hist["pAs"][-1] if hist["pAs"] else np.nan
            panel = ev.compose_panel(
                pan_w, out_h,
                ev.draw_prob_strip(pan_w, hp, t, chart_span, hist["tA"], hist["pA"], hist["pAs"], hist["alA"],
                                   thr_a, marks, extra, alarm_now=alarm_a),
                ev.draw_drivers(pan_w, hd, drivers if explainer is not None else [], p_now) if explainer is not None
                else np.full((hd, pan_w, 3), 30, np.uint8),
                ev.draw_motion(pan_w, hm, t, chart_span, mot_t, mot, hl_regions, marks),
                ev.draw_eeg_lane(pan_w, he, enabled=False))   # hook: pass a paired EEG stream here later
            new_data = False
        if disp_h < out_h:
            vis = np.vstack([vis, np.zeros((out_h - disp_h, disp_w, 3), np.uint8)])
        out = np.hstack([vis, panel])
        tm["draw"] += time.perf_counter() - t_d; tn["draw"] += 1
        if writer is not None:
            writer.write(out)
        if not a.headless:
            cv2.imshow(WIN_NAME, out)
            if not fast:
                lag = (t - vid0) - (time.perf_counter() - wall0)
                wait = max(1, int(lag * 1000))
            else:
                wait = 1
            k = cv2.waitKey(wait) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                paused = True
            if k == ord("f"):
                fast = not fast; wall0 = time.perf_counter(); vid0 = t
    cap.release()
    if writer is not None:
        writer.release()
    if logf:
        logf.close()
    if inf_times:
        print(f"[demo] mean analysis time per sampled frame: {np.mean(inf_times) * 1000:.1f} ms")
    if a.timing:
        wall = time.perf_counter() - run0
        print(f"[timing] {n_samples} sampled frames in {wall:.1f} s wall = {n_samples / max(wall, 1e-9):.2f} sampled frames/s")
        for k_ in ("detA", "explain", "draw"):
            if tn[k_]:
                print(f"[timing] {k_}: {tm[k_] / tn[k_] * 1000:.2f} ms per call x {tn[k_]} (total {tm[k_]:.2f} s)")
    if not a.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
