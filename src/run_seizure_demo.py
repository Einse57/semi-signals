"""Seizure-detection viewer: video + skeleton overlay + live scores.

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
over 5 s), plain per-region motion graphs, and a greyed-out
EEG lane reserved for a future paired EEG stream.

Keys: q quit, space pause, f fast-forward toggle (no real-time pacing), r replay from the start,
o open another video (file picker), m face tracking on/off, h show/hide the key legend (footer strip). When the video ends the window stays open on the last frame
(graphs and score strip visible): r replay, o open another video, q quit. --loop restarts by itself.
A replay resets all causal state (buffers, smoothing, alarms, explanations, graphs).

Optional face stage (--face, or the m key at run time; off by default): MediaPipe Face Landmarker on
a head crop of the tracked subject (src/face_estimator.py). It adds the face mesh to the video and
face motion graphs (mouth, eyes, left/right face) to the panel. It does not change Detector A or
Detector B inputs. The model loads the first time face tracking is switched on.
"""
from __future__ import annotations

import argparse
import collections
import functools
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from src.textdraw import put_text, put_block, text_width, make_dpi_aware, fit_to_screen

WIN_S = 12.0          # seconds of keypoint history kept for features
CHART_S = 90.0        # seconds shown on the chart
WIN_NAME = "semi-signals | seizure detection"
MIN_VALID_B = 0.5    # Detector B needs the tracked subject in at least half of its 5 s window, else no score
RENDER_VERSION = 2   # bump when the rendered overlay changes, so saved renders are redone (review_video)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Video seizure detection viewer")
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
    p.add_argument("--save", default=None, help="write the rendered output to this mp4 (first pass)")
    p.add_argument("--log", default=None, help="write per-second scores to this csv (first pass; use {pass} in the name for one file per pass)")
    p.add_argument("--width", type=int, default=960, help="video width in the window")
    p.add_argument("--panel-width", type=int, default=640, help="width of the side panel")
    p.add_argument("--no-explain", action="store_true", help="skip pred_contrib drivers")
    p.add_argument("--end", type=float, default=None, help="stop at this video time [s]")
    p.add_argument("--timing", action="store_true", help="print a timing summary at the end")
    p.add_argument("--loop", action="store_true", help="restart from the beginning when the video ends")
    p.add_argument("--max-loops", type=int, default=0, help="with --loop: stop after this many passes (0 = no limit)")
    p.add_argument("--exit-at-end", action="store_true", help="close the window when the video ends")
    p.add_argument("--end-frame", default=None, help="write the frozen last frame with the end-of-video overlay to this png")
    p.add_argument("--face", action="store_true",
                   help="start with face tracking on (MediaPipe Face Landmarker; toggle with m)")
    p.add_argument("--no-face-mesh", action="store_true", help="with face tracking: draw contours only, no mesh")
    p.add_argument("--face-model", default=None, help="face_landmarker.task (default models/face_landmarker.task)")
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


class CausalScorer:
    """All per-pass causal state: keypoint buffers, Detector A history and smoothing, alarm state,
    explanation history and the motion-graph series. reset() returns it to the state of a fresh
    start, so a replay of the same video gives identical scores."""

    BUF_KEYS = ("t", "body_kpts", "body_scores", "body_boxes", "sel", "hand_kpts", "hand_scores", "patches", "vkp")

    def __init__(self, booster, thr, fps, fnames=None, explain=True, start=0.0):
        self.booster, self.thr, self.fps = booster, thr, fps
        self.fnames, self.explain, self.start = fnames, explain, start
        self.reset()

    def reset(self, start=None):
        from src.batch_extract import SubjectTracker
        from src.detector_a.features import REGIONS
        if start is not None:
            self.start = start
        maxlen = int(WIN_S * self.fps) + 2
        self.buf = {k: collections.deque(maxlen=maxlen) for k in self.BUF_KEYS}
        self.tracker = SubjectTracker()
        self.hist = dict(tA=[], pA=[], pAs=[], alA=[], drv=[], tB=[], pB=[], tF=[], pF=[])
        self.mot_t, self.mot = [], {r: [] for r in REGIONS}
        self.drivers = []
        self.explainer = None
        if self.booster is not None and self.explain:
            from src.explain import Explainer
            self.explainer = Explainer(self.booster, self.fnames)
        self.next_sample = self.start
        self.next_second = int(np.floor(self.start)) + 1
        self.last_b = -1e9

    def add_sample(self, sample: dict):
        for k in self.BUF_KEYS:
            self.buf[k].append(sample[k])

    def score_due(self, t, ieeg=None, on_row=None):
        """Score every second completed by video time t. on_row(row) gets one dict per second."""
        from src.detector_a.features import compute_features, subject_series, REGIONS
        from src.detector_a.scoring import SMOOTH
        from src.fusion import noisy_or
        hist, mot, mot_t = self.hist, self.mot, self.mot_t
        n = 0
        while t >= self.next_second:
            sec = self.next_second - 1
            pa = np.nan; X = None; n_mot0 = len(mot_t)
            if self.booster is not None and len(self.buf["t"]) > 3:
                d = {k_: np.array(self.buf[k_]) for k_ in ("t", "body_kpts", "body_scores", "body_boxes", "sel", "hand_kpts", "hand_scores")}
                X = compute_features(d, fs=self.fps, seconds=[sec])
                pa = float(self.booster.predict(X)[0])
                ss = subject_series(d)          # per-frame region speeds for the motion graphs
                kk = (ss["t"] >= sec) & (ss["t"] < sec + 1)
                for i_, tt_ in zip(np.where(kk)[0], ss["t"][kk]):
                    mot_t.append(float(tt_))
                    for j_, r_ in enumerate(ss["names"]):   # display: trailing 0.5 s mean of joint speed
                        w_ = ss["R"][max(0, i_ - 2):i_ + 1, j_]
                        mot[r_].append(float(np.nanmean(w_)) if np.isfinite(w_).any() else np.nan)
            hist["tA"].append(sec + 1.0); hist["pA"].append(pa)
            recent = [x for x in hist["pA"][-SMOOTH:]]
            hist["pAs"].append(float(np.nanmean(recent)) if np.isfinite(recent).any() else np.nan)
            hist["alA"].append(len(hist["pAs"]) >= 3 and all(np.nan_to_num(hist["pAs"][-3:]) >= self.thr))
            if self.explainer is not None:
                self.explainer.update(X)
                self.drivers = self.explainer.drivers(hist["pAs"][-1])
            drivers = self.drivers
            hist["drv"].append(drivers)
            with np.errstate(all="ignore"):
                sec_speed = {r_: (float(np.nanmean(v_[n_mot0:])) if np.isfinite(v_[n_mot0:]).any() else np.nan)
                             for r_, v_ in mot.items()}
            hr_ = sorted({r_ for d_ in drivers for r_ in d_["regions"]})
            drv_speed = float(np.nanmean([sec_speed[r_] for r_ in hr_])) if hr_ and np.isfinite([sec_speed[r_] for r_ in hr_]).any() else np.nan
            pb = hist["pB"][-1] if hist["pB"] and t - hist["tB"][-1] < 1.5 else np.nan
            pv = hist["pAs"][-1]
            pe = np.nan
            if ieeg is not None:
                k = (ieeg[0] >= sec) & (ieeg[0] < sec + 1)
                pe = float(ieeg[1][k].max()) if k.any() else np.nan
            pf = float(noisy_or(np.array([pe]), np.array([pv]))[0])
            hist["tF"].append(sec + 1.0); hist["pF"].append(pf)
            if on_row is not None:
                on_row(dict(sec=sec, pa=pa, pv=pv, pb=pb, pe=pe, pf=pf, alarm=int(hist["alA"][-1]), drivers=drivers,
                            regions=hr_, drv_speed=drv_speed, speeds=[sec_speed[r_] for r_ in REGIONS]))
            self.next_second += 1
            n += 1
        return n


END_TEXT = "End of video:  R replay   O open another video   Q quit"


VIEWER_KEYS = [("q", "quit"), ("space", "pause / play"), ("f", "fast / real time"), ("r", "replay"),
               ("o", "open another video"), ("m", "face on/off"), ("h", "hide keys")]
LEGEND_H = 34
_legend = {"on": True, "mode": None}


def mode_label(face_on: bool) -> str:
    return "Mode: Body + Hands + Face" if face_on else "Mode: Body + Hands"


def draw_legend(w, items, h=LEGEND_H, note=None):
    """Footer strip with the key commands (below the video and graphs, never over them). Cached."""
    return _legend_img(int(w), tuple(tuple(i) for i in items), int(h), note)


@functools.lru_cache(maxsize=32)
def _legend_img(w, items, h, note):
    img = np.full((h, w, 3), 32, np.uint8)
    cv2.line(img, (0, 0), (w - 1, 0), (80, 80, 80), 1)
    x, y = 12, h // 2
    for key, what in items:
        kw = text_width(key, 14, True) if key else 0
        dw = text_width(what, 14)
        if x + kw + 12 + dw > w - 10:
            break
        if key:
            cv2.rectangle(img, (x - 6, y - 11), (x + kw + 6, y + 11), (90, 90, 90), -1)
            put_text(img, key, (x, y), 14, (255, 255, 255), bold=True, anchor="lm")
            x += kw + 14
        put_text(img, what, (x, y), 14, (210, 210, 210), anchor="lm")
        x += dw + 22
    if note:
        nw = text_width(note, 14, True)
        if x + nw < w - 10:
            put_text(img, note, (w - 12, y), 14, (0, 190, 255), bold=True, anchor="rm")
    img.setflags(write=False)
    return img


def with_legend(img, items=VIEWER_KEYS, note=None):
    if not _legend["on"]:
        return img
    if _legend.get("mode"):
        note = _legend["mode"] + (f"  |  {note}" if note else "")
    return np.vstack([img, draw_legend(img.shape[1], items, note=note)])


def draw_end_overlay(img, text=END_TEXT, sub=None):
    """Banner across the frozen last frame (graphs and score strip stay visible)."""
    out = img.copy()
    h, w = out.shape[:2]
    y0, y1 = h // 2 - 34, h // 2 + (40 if sub else 30)
    band = out[y0:y1].copy(); band[:] = (20, 20, 20)
    out[y0:y1] = cv2.addWeighted(out[y0:y1], 0.2, band, 0.8, 0)
    cv2.rectangle(out, (0, y0), (w - 1, y1), (0, 190, 255), 2)
    put_text(out, text, (w // 2, h // 2 - 4 if sub else h // 2), 26, (255, 255, 255), bold=True, anchor="mm")
    if sub:
        put_text(out, sub, (w // 2, h // 2 + 22), 15, (0, 190, 255), anchor="mm")
    return out


def show(img):
    """imshow at 1:1, shrunk (INTER_AREA) only if the frame is larger than the screen work area."""
    cv2.imshow(WIN_NAME, fit_to_screen(img))


def overlay_lines(img, lines, org, size_px, bold=True):
    """Text lines on a translucent dark box (readable on any video); lines = [(text, bgr), ...]."""
    put_block(img, lines, org, size_px, bold=bold)


def pick_video(initial_dir=None):
    """tkinter file picker; returns a path, '' if cancelled, or None if tkinter is unavailable."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    try:
        r = tk.Tk(); r.withdraw()
        try:
            r.attributes("-topmost", True)
        except Exception:
            pass
        f = filedialog.askopenfilename(parent=r, title="Open a video", initialdir=initial_dir or os.getcwd(),
                                       filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All files", "*.*")])
        r.destroy()
        return f or ""
    except Exception as e:
        print(f"[viewer] file picker failed: {e}")
        return None


def window_closed() -> bool:
    try:
        return cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main(argv=None):
    a = parse_args(argv)
    if not a.headless:
        make_dpi_aware()                 # crisp text on scaled Windows displays (before any window)
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
    from src.detector_a.features import feature_names

    det = PersonDetector(device=device, backend=backend, score_thr=0.3)
    pose = PoseEstimator(device=device, backend=backend)
    hand = None if a.no_hands else HandEstimator(device=device, backend=backend)

    booster, thr_a, meta = None, 0.5, {}
    if a.detector_a and os.path.exists(a.detector_a):
        import lightgbm as lgb
        booster = lgb.Booster(model_file=a.detector_a)
        mp = a.detector_a_meta or os.path.splitext(a.detector_a)[0] + ".json"
        meta = json.load(open(mp)) if os.path.exists(mp) else {}
        thr_a = meta.get("threshold", 0.5)
        print(f"[demo] detector A loaded, threshold {thr_a}")
    fnames = None
    if booster is not None and not a.no_explain:
        fnames = meta.get("features") or feature_names()
        assert len(fnames) == booster.num_feature(), "feature names do not match the model"
    vs = None
    if a.vsvig and os.path.exists(a.vsvig):
        vs = VSViGModel(a.vsvig, device if device == "gpu" else "cpu")
        print(f"[demo] VSViG loaded on {vs.dev}")
    ieeg = load_ieeg(a.ieeg, a.ieeg_offset) if a.ieeg else None
    from src.face_estimator import FaceStage
    scorer = CausalScorer(booster, thr_a, a.fps, fnames=fnames, explain=not a.no_explain, start=a.start)
    models = dict(det=det, pose=pose, hand=hand, vs=vs, thr_a=thr_a, device=device, ieeg=ieeg, scorer=scorer,
                  face=FaceStage(a.face_model, on=a.face))   # model loads on first enable

    video, labels, n_pass = a.input, a.labels, 0
    first_video = str(Path(a.input).resolve())
    stay_open = not (a.headless or a.exit_at_end)
    while True:
        n_pass += 1
        action, last = run_pass(a, models, video, labels, n_pass)
        if action == "quit":
            break
        if action == "end":
            if a.loop and (a.max_loops <= 0 or n_pass < a.max_loops):
                action = "replay"
            elif not stay_open or last is None:
                break
            else:
                action, picked = end_screen(last, video)
                if action == "open":
                    video = picked
        elif isinstance(action, tuple):           # ("open", path) during playback
            action, video = action
        if action == "quit":
            break
        if action == "open":
            # labels only apply to clips from the labelled dataset; label_for_video returns None otherwise
            labels = a.labels if a.labels else None
            print(f"[viewer] opening {video}")
        else:
            print(f"[viewer] replay {n_pass + 1}")
    if not a.headless:
        cv2.destroyAllWindows()


def end_screen(last, video):
    """Freeze on the last frame with the overlay; returns ('replay'|'quit'|'open', path)."""
    sub = None
    while True:
        show(with_legend(draw_end_overlay(last, sub=sub)))
        k = cv2.waitKey(50) & 0xFF
        if window_closed():
            return "quit", None
        if k == ord("h"):
            _legend["on"] = not _legend["on"]
        if k in (ord("q"), 27):
            return "quit", None
        if k in (ord("r"), ord(" ")):
            return "replay", None
        if k == ord("o"):
            f = pick_video(str(Path(video).parent))
            if f is None:
                sub = "File picker not available (tkinter missing): start the launcher or use --input"
            elif f:
                return "open", f


def run_pass(a, M, video, labels, n_pass):
    """Play one video from a.start with freshly reset causal state.
    Returns (action, last_frame): action is 'end', 'quit', 'replay' or ('open', path)."""
    from src.batch_extract import extract_vsvig_patches, VSVIG_COCO_ORDER, MAX_PERSONS
    from src.detector_a.features import REGIONS
    from src import explain_view as ev
    from src.display import draw_skeleton, draw_hand_skeleton
    det, pose, hand, vs, thr_a, device, ieeg, sc = (M[k] for k in ("det", "pose", "hand", "vs", "thr_a", "device", "ieeg", "scorer"))
    from src.face_estimator import FaceMotion, FaceStage
    from src.display import draw_face_mesh
    fst = M.get("face") or FaceStage()
    fmot = FaceMotion()
    face_lm = None
    _legend["mode"] = mode_label(fst.on) + (f" (face unavailable: {fst.error})" if fst.error else "")
    sc.reset(start=a.start)
    hist = sc.hist
    explainer_on = sc.explainer is not None
    lab = label_for_video(labels, video) if labels else None

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[viewer] cannot open {video}")
        return "quit", None
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
    if a.save and n_pass == 1:
        writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (disp_w + pan_w, out_h))
    if not a.headless:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)   # 1:1 pixels; show() shrinks to the screen if needed
        cv2.setWindowTitle(WIN_NAME, f"semi-signals | {Path(video).name}")

    tm = collections.defaultdict(float); tn = collections.defaultdict(int)
    n_samples = 0; run0 = time.perf_counter()
    fi = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    wall0 = time.perf_counter(); vid0 = a.start
    paused, fast = False, a.fast
    inf_times = collections.deque(maxlen=60)
    log_path = a.log.replace("{pass}", str(n_pass)) if a.log else None
    if a.log and "{pass}" not in a.log and n_pass > 1:
        log_path = None                         # keep the first pass's log
    logf = open(log_path, "w") if log_path else None
    if logf:
        logf.write("t,p_video_a,p_video_a_smooth,p_video_b,p_ieeg,p_fused,alarm_a,drivers,driver_regions,driver_speed,"
                   + ",".join(f"speed_{r_}" for r_ in REGIONS) + "\n")

    def on_row(r):
        if logf:
            dtxt = "|".join(f"{d_['text']}={d_['value']:.2f}" for d_ in r["drivers"])
            logf.write(f"{r['sec']},{r['pa']:.4f},{r['pv']:.4f},{r['pb']:.4f},{r['pe']:.4f},{r['pf']:.4f},{r['alarm']},{dtxt},"
                       f"{'|'.join(r['regions'])},{r['drv_speed']:.4f}," + ",".join(f"{v:.4f}" for v in r["speeds"]) + "\n")

    sel_kp = sel_sc = None; hres = None
    panel, new_data, out, t = None, True, None, a.start
    action = "end"
    last_prog = 0.0
    total = (min(dur, a.end) if a.end is not None else dur) - a.start
    while True:
        if paused:
            k = cv2.waitKey(30) & 0xFF
            if k == ord("h") and out is not None:
                _legend["on"] = not _legend["on"]
                show(with_legend(out, note="paused"))
            if k == ord(" "):
                paused = False; wall0 = time.perf_counter(); vid0 = t
            elif k == ord("q") or window_closed():
                action = "quit"; break
            elif k == ord("r"):
                action = "replay"; break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / src_fps
        fi += 1
        if a.end is not None and t >= a.end:
            break
        if (a.headless or a.fast) and time.perf_counter() - last_prog >= 2.0 and total > 0:
            last_prog = time.perf_counter()        # parsed by the launcher's progress bar
            print(f"[viewer] progress {min(100.0, 100.0 * (t - a.start) / total):.0f}% "
                  f"({t - a.start:.0f} of {total:.0f} s)", flush=True)
        if t + 1e-6 >= sc.next_sample:
            sc.next_sample += 1.0 / a.fps
            t_inf = time.perf_counter()
            boxes = det.detect(frame)
            boxes = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:MAX_PERSONS]
            bk = np.full((MAX_PERSONS, 17, 2), np.nan, np.float32); bs = np.zeros((MAX_PERSONS, 17), np.float32)
            bb = np.full((MAX_PERSONS, 5), np.nan, np.float32)
            hk = np.full((2, 21, 2), np.nan, np.float32); hs = np.zeros((2, 21), np.float32)
            sel = sc.tracker.update(boxes)
            persons = pose.estimate(frame, boxes) if boxes else []
            for i, p in enumerate(persons):
                bk[i], bs[i] = p["keypoints"], p["scores"]; bb[i, :5] = np.asarray(p["bbox"], np.float32)[:5]
            hres = None
            if sel >= 0 and hand is not None:
                hres = hand.estimate(frame, bk[sel], bs[sel])
                for j, side in enumerate(("left", "right")):
                    if hres[side] is not None:
                        hk[j], hs[j] = hres[side]["keypoints"], hres[side]["scores"]
            if fst.on:              # optional face stage: subject only, never feeds Detector A/B
                fr_ = fst.est.estimate(frame, bk[sel], bs[sel], body_bbox=bb[sel]) if sel >= 0 else None
                face_lm = fr_["landmarks"] if fr_ is not None else None
                fmot.update(t, face_lm)
            if sel >= 0:
                patch = extract_vsvig_patches(frame, bk[sel][VSVIG_COCO_ORDER])
                vkp = np.concatenate([bk[sel][VSVIG_COCO_ORDER], bs[sel][VSVIG_COCO_ORDER, None]], -1)
                sel_kp, sel_sc = bk[sel].copy(), bs[sel].copy()
            else:
                patch = np.zeros((15, 32, 32, 3), np.uint8); vkp = np.zeros((15, 3), np.float32)
                sel_kp = sel_sc = None
            sc.add_sample(dict(t=t, body_kpts=bk, body_scores=bs, body_boxes=bb, sel=sel,
                               hand_kpts=hk, hand_scores=hs, patches=patch, vkp=vkp))
            # Detector B every 0.5 s
            if vs is not None and len(sc.buf["t"]) >= 30 and t - sc.last_b >= 0.5:
                sc.last_b = t
                P = np.ascontiguousarray(np.stack(list(sc.buf["patches"])[-30:]).transpose(0, 1, 4, 2, 3)[None], np.float32)
                K = np.ascontiguousarray(np.stack(list(sc.buf["vkp"])[-30:])[None], np.float32)
                valid = float(np.mean(np.asarray(list(sc.buf["sel"])[-30:]) >= 0))
                # with no tracked subject the inputs are all zeros and VSViG returns ~0: show no score instead
                hist["tB"].append(t); hist["pB"].append(vs(P, K) if valid >= MIN_VALID_B else np.nan)
            inf_times.append(time.perf_counter() - t_inf)
            n_samples += 1; new_data = True
        # Detector A once per completed second
        t_a = time.perf_counter()
        ns = sc.score_due(t, ieeg=ieeg, on_row=on_row)
        if ns:
            tm["detA"] += time.perf_counter() - t_a; tn["detA"] += ns
            new_data = True
        drivers = sc.drivers
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
            if fst.on and face_lm is not None:
                draw_face_mesh(vis, face_lm * sx, mesh=not a.no_face_mesh)
            for r_ in hl_regions - {"overall"}:      # ring the joints of the driver regions
                for j_ in REGIONS[r_]:
                    if sel_sc[j_] >= 0.3:
                        cv2.circle(vis, tuple(int(v) for v in sel_kp[j_] * sx), 11, ev.HILITE, 2, cv2.LINE_AA)
        alarm_a = bool(hist["alA"] and hist["alA"][-1])
        pB6 = np.asarray(hist["pB"][-6:], float)
        alarm_b = len(pB6) >= 6 and np.isfinite(pB6).sum() >= 4 and np.nanmean(pB6) >= 0.5
        lines = [(f"Detector A (pose features): {np.nan_to_num(hist['pAs'][-1]) if hist['pAs'] else 0:.2f}",
                  (60, 60, 255) if alarm_a else (0, 255, 0))]
        if vs is not None:
            pb_ = hist["pB"][-1] if hist["pB"] else np.nan
            lines.append((f"Detector B (VSViG-base, reference): " + (f"{pb_:.2f}" if np.isfinite(pb_) else
                          ("-- (no subject tracked)" if hist["pB"] else "-- (warming up, 5 s)")),
                          (60, 60, 255) if alarm_b else (255, 200, 0)))
        if fst.on:
            lines.append((mode_label(True), (0, 220, 255)))
        overlay_lines(vis, lines, (12, 10), 20)
        if alarm_a or alarm_b:
            put_text(vis, "SEIZURE ALERT", (disp_w - 16, 14), 28, (0, 0, 255), bold=True, anchor="rt")
        if inf_times:
            overlay_lines(vis, [(f"analysis {1.0 / max(1e-3, np.mean(inf_times)):.0f} frames/s ({device})", (210, 210, 210))],
                          (12, disp_h - 30), 14, bold=False)
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
            hm_b, hm_f = ev.split_motion(hm, fst.on)
            face_strip = (ev.draw_motion(pan_w, hm_f, t, chart_span, fmot.t, fmot.speeds, set(), marks,
                                         groups=ev.FACE_GROUPS, hdr=ev.FACE_HDR, colors=ev.FACE_COLORS)
                          if fst.on else None)
            panel = ev.compose_panel(
                pan_w, out_h,
                ev.draw_prob_strip(pan_w, hp, t, chart_span, hist["tA"], hist["pA"], hist["pAs"], hist["alA"],
                                   thr_a, marks, extra, alarm_now=alarm_a),
                ev.draw_drivers(pan_w, hd, drivers if explainer_on else [], p_now) if explainer_on
                else np.full((hd, pan_w, 3), 30, np.uint8),
                ev.draw_motion(pan_w, hm_b, t, chart_span, sc.mot_t, sc.mot, set(), marks),
                ev.draw_eeg_lane(pan_w, he, enabled=False),   # hook: pass a paired EEG stream here later
                face=face_strip)
            new_data = False
        if disp_h < out_h:
            vis = np.vstack([vis, np.zeros((out_h - disp_h, disp_w, 3), np.uint8)])
        out = np.hstack([vis, panel])
        tm["draw"] += time.perf_counter() - t_d; tn["draw"] += 1
        if writer is not None:
            writer.write(out)
        if not a.headless:
            show(with_legend(out, note="fast" if fast else None))
            if not fast:
                lag = (t - vid0) - (time.perf_counter() - wall0)
                wait = max(1, int(lag * 1000))
            else:
                wait = 1
            k = cv2.waitKey(wait) & 0xFF
            if k == ord("q") or (k == 255 and n_samples % 30 == 0 and window_closed()):
                action = "quit"; break
            if k == ord(" "):
                paused = True
                show(with_legend(out, note="paused"))
            if k == ord("h"):
                _legend["on"] = not _legend["on"]
            if k == ord("m"):
                fst.toggle()
                _legend["mode"] = mode_label(fst.on) + (f" (face unavailable: {fst.error})" if fst.error else "")
                face_lm = None; fmot._prev = None; new_data = True   # no motion across an off gap
            if k == ord("f"):
                fast = not fast; wall0 = time.perf_counter(); vid0 = t
            if k == ord("r"):
                action = "replay"; break
            if k == ord("o"):
                f = pick_video(str(Path(video).parent))
                if f:
                    action = ("open", f); break
                wall0 = time.perf_counter(); vid0 = t
    cap.release()
    if writer is not None:
        writer.release()
    if logf:
        logf.close()
    if a.headless or a.fast:
        print("[viewer] progress 100%", flush=True)
    if inf_times:
        print(f"[demo] mean analysis time per sampled frame: {np.mean(inf_times) * 1000:.1f} ms")
    if a.timing:
        wall = time.perf_counter() - run0
        print(f"[timing] {n_samples} sampled frames in {wall:.1f} s wall = {n_samples / max(wall, 1e-9):.2f} sampled frames/s")
        for k_ in ("detA", "draw"):
            if tn[k_]:
                print(f"[timing] {k_}: {tm[k_] / tn[k_] * 1000:.2f} ms per call x {tn[k_]} (total {tm[k_]:.2f} s)")
    if a.end_frame and out is not None and n_pass == 1:
        cv2.imwrite(a.end_frame, with_legend(draw_end_overlay(out)))
    return action, out


if __name__ == "__main__":
    main()
