"""Load your own video, run Detector A over it, then play it back with the overlay and annotate.

  python -m src.review_video                       # opens a file picker
  python -m src.review_video --input my_clip.mp4 --annotator AL

Step 1 (process): runs the seizure demo over the file (pose + Detector A + drivers) and saves
  <stem>_semisignals.mp4 (rendered overlay) and <stem>_scores.csv next to the video.
  Skipped when both exist and are newer than the video (use --reprocess to redo).
Step 2 (review): plays the rendered video in sync, with a timeline of model alarms and your marks.
  Keys:  space pause | j / l  back / forward 5 s | o clinical onset (new event) | e clinical end
         g EEG onset (optional) | t seizure type | n note (attached to the event under the playhead, else a timed note)
         x exclude span start/stop | u undo last mark | s save | q quit (asks to save)
Saves next to the ORIGINAL video (see src/annotation.py): <stem>.eaf (ELAN), <stem>_signals.csv,
<stem>_tsconf.xml, <stem>.annotation.json.

Everything stays on this computer. Research demo, not diagnostic. Use self-recorded or public
clips; other people's videos need their consent (and likely IRB review) before any use.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np

from src import annotation as A

WIN = "semi-signals | review & annotate"
BAR_H = 74


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Review and annotate your own video")
    p.add_argument("--input", default=None, help="video file (default: file picker)")
    p.add_argument("--detector-a", default="models/detector_a_all.txt",
                   help="LightGBM model (txt), or 'hf' to fetch the SHA-256-checked model from Hugging Face")
    p.add_argument("--vsvig", default="", help="optional VSViG IR for Detector B (off by default)")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-hands", action="store_true")
    p.add_argument("--annotator", default=None)
    p.add_argument("--pat-id", default=None, help="patient id or pseudonym (never a real name)")
    p.add_argument("--consent", default="self", choices=["self", "public", "consented", "unknown"])
    p.add_argument("--reprocess", action="store_true")
    p.add_argument("--headless-process", action="store_true", help="no window while processing")
    p.add_argument("--relative-tsconf", action="store_true", help="relative CSV path in _tsconf.xml (portable folder)")
    p.add_argument("--keys", default=None, help="testing: scripted keys 't:key[:text];...' (no window)")
    return p.parse_args(argv)


# ── dialogs (tkinter when available, console otherwise) ───────
_tk = None


def _root():
    global _tk
    if _tk is None:
        try:
            import tkinter as tk
            _tk = tk.Tk(); _tk.withdraw()
        except Exception:
            _tk = False
    return _tk or None


def pick_file():
    r = _root()
    if r is None:
        return input("video file: ").strip().strip('"')
    from tkinter import filedialog
    return filedialog.askopenfilename(parent=r, title="Choose a video (stays on this computer)",
                                      filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All files", "*.*")])


def ask(title, prompt, initial=""):
    r = _root()
    if r is None:
        s = input(f"{prompt} [{initial}]: ").strip()
        return s or initial
    from tkinter import simpledialog
    return simpledialog.askstring(title, prompt, initialvalue=initial, parent=r)


def ask_yes(title, prompt):
    r = _root()
    if r is None:
        return input(f"{prompt} [y/N]: ").strip().lower().startswith("y")
    from tkinter import messagebox
    return messagebox.askyesno(title, prompt, parent=r)


# ── processing ───────────────────────────────────────────────
def outputs_for(video):
    v = Path(video)
    return v.with_name(v.stem + "_semisignals.mp4"), v.with_name(v.stem + "_scores.csv")


def process(a, video):
    rendered, scores = outputs_for(video)
    fresh = all(p.exists() and p.stat().st_mtime > Path(video).stat().st_mtime for p in (rendered, scores))
    if fresh and not a.reprocess:
        print(f"[review] using existing {rendered.name} and {scores.name}")
        return rendered, scores
    from src import run_seizure_demo
    argv = ["--input", str(video), "--detector-a", a.detector_a, "--vsvig", a.vsvig, "--device", a.device,
            "--fast", "--save", str(rendered), "--log", str(scores)]
    if a.no_hands:
        argv.append("--no-hands")
    if a.headless_process or a.keys:
        argv.append("--headless")
    print("[review] processing (pose + Detector A); this takes about the length of the video")
    run_seizure_demo.main(argv)
    return rendered, scores


def load_scores(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            fl = lambda k: float(r[k]) if r.get(k) not in (None, "", "nan") else float("nan")
            rows.append(dict(t=int(float(r["t"])), p=fl("p_video_a"), p_smooth=fl("p_video_a_smooth"),
                             alarm=int(r.get("alarm_a") or 0), drivers="; ".join(x.split("=")[0] for x in (r.get("drivers") or "").split("|") if x),
                             driver_speed=fl("driver_speed"),
                             speeds={g: fl(f"speed_{g}") for g in A.REGIONS}))
    return rows


def per_second(rows):
    out = []
    for r in rows:
        d = dict(t_s=float(r["t"] + 1), p_seizure=r["p"], p_smooth5s=r["p_smooth"], alarm=r["alarm"],
                 driver_speed=r["driver_speed"])
        d.update({f"speed_{g}": v for g, v in r["speeds"].items()})
        out.append(d)
    return out


def compare(events, alarms, pre=30.0, post=60.0):
    """Per human-marked event: first model alarm within [onset-30 s, onset+60 s] and its delay."""
    res = []
    for k, e in enumerate(events, 1):
        on = e.get("clinical_onset_s")
        hit = [al for al in alarms if on is not None and on - pre <= al["start_s"] <= on + post]
        res.append(dict(event=k, clinical_onset_s=on, detected=bool(hit),
                        alarm_start_s=hit[0]["start_s"] if hit else None,
                        delay_s=round(hit[0]["start_s"] - on, 1) if hit else None))
    used = {id(al) for al in alarms for e in events if e.get("clinical_onset_s") is not None
            and e["clinical_onset_s"] - pre <= al["start_s"] <= e["clinical_onset_s"] + post}
    return dict(window="alarm start within 30 s before to 60 s after marked clinical onset", events=res,
                alarms_outside_marked_events=sum(id(al) not in used for al in alarms))


# ── review UI ────────────────────────────────────────────────
def draw_bar(w, t, dur, ann, alarms, open_excl, status):
    img = np.full((BAR_H, w, 3), 18, np.uint8)
    L, R, y0, y1 = 10, 10, 8, 30
    X = lambda s: int(L + (s / max(dur, 1e-6)) * (w - L - R))
    cv2.rectangle(img, (L, y0), (w - R, y1), (45, 45, 45), -1)
    for al in alarms:
        cv2.rectangle(img, (X(al["start_s"]), y0), (max(X(al["end_s"]), X(al["start_s"]) + 2), y0 + 8), (40, 40, 220), -1)
    for e in ann["events"]:
        on, end = e.get("clinical_onset_s"), e.get("clinical_end_s")
        if on is None:
            continue
        cv2.rectangle(img, (X(on), y0 + 10), (max(X(end if end is not None else t), X(on) + 2), y1), (0, 160, 0), -1)
        cv2.line(img, (X(on), y0), (X(on), y1), (0, 255, 0), 2)
    for x in ann["excludes"]:
        cv2.rectangle(img, (X(x["start_s"]), y1 - 4), (X(x["end_s"]), y1), (128, 128, 128), -1)
    for n in ann["notes"]:
        cv2.circle(img, (X(n["start_s"]), y1 + 4), 3, (0, 220, 255), -1)
    cv2.line(img, (X(t), y0 - 4), (X(t), y1 + 4), (255, 255, 255), 2)
    ev = ann["events"][-1] if ann["events"] else None
    info = f"{A.hms(t)}  |  events {len(ann['events'])}"
    if ev:
        info += f"  last: onset {A.hms(ev['clinical_onset_s'])} end {A.hms(ev.get('clinical_end_s')) or '-'}"
        info += f" type '{ev.get('seizure_type') or '-'}'"
    if open_excl is not None:
        info += "  | exclude open"
    cv2.putText(img, info, (L, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(img, "space pause  j/l -5/+5 s  o onset  e end  g EEG  t type  n note  x exclude  u undo  s save  q quit"
                + f"   [{status}]", (L, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(img, "red: model alarms  green: your marks", (w - 290, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    return img


def main(argv=None):
    a = parse_args(argv)
    from src.hf_models import resolve_detector_a
    a.detector_a = resolve_detector_a(a.detector_a)
    video = a.input or pick_file()
    if not video:
        print("[review] no file chosen"); return
    video = Path(video).resolve()
    rendered, scores = process(a, video)
    rows = load_scores(scores)
    meta_thr = 0.95
    mj = os.path.splitext(a.detector_a)[0] + ".json"
    if os.path.exists(mj):
        import json
        meta_thr = json.load(open(mj)).get("threshold", 0.95)
    alarms = A.alarms_from_seconds(rows)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    cap.release()
    print("[review] hashing the video (sha256)")
    sha = A.sha256_file(video)
    annotator = a.annotator or ("" if a.keys else ask("Annotator", "Your initials (annotator):", os.environ.get("USERNAME", ""))) or ""
    pat_id = a.pat_id or ("" if a.keys else ask("Patient", "Patient id or pseudonym (no real names):", "self")) or ""
    ann = A.new_annotation(video, fps, dur, annotator=annotator, pat_id=pat_id, consent=a.consent, sha256=sha)
    eaf = A.paths_for(video)["eaf"]
    if eaf.exists():
        prev = A.read_eaf(eaf, video)          # raises ShaMismatch for a different video
        ann["events"] = [dict(e, eeg_onset_s=e.get("eeg_onset_s"), notes=e.get("notes", "")) for e in prev["events"]]
        ann["notes"] = [dict(start_s=r["start_s"], end_s=r["end_s"], text=r["value"])
                        for r in prev["tiers"].get("Notes", {}).get("rows", [])]
        ann["excludes"] = [dict(start_s=r["start_s"], end_s=r["end_s"], reason=r["value"])
                           for r in prev["tiers"].get("Exclude", {}).get("rows", [])]
        print(f"[review] loaded {len(ann['events'])} events from {eaf.name} (Model_* tiers are regenerated)")
    model_id = Path(a.detector_a).stem
    undo, open_excl, status = [], None, "unsaved"

    def save():
        ann["model"] = A.model_section(model_id, meta_thr, alarms, per_second(rows))
        ann["comparison"] = compare(ann["events"], alarms)
        out = A.save_all(ann, video, relative_tsconf=a.relative_tsconf)
        print("[review] saved: " + ", ".join(str(p) for p in out.values()))
        for c in ann["comparison"]["events"]:
            print(f"[review] event {c['event']}: onset {A.hms(c['clinical_onset_s'])} -> "
                  + (f"alarm at {A.hms(c['alarm_start_s'])} (delay {c['delay_s']:+.1f} s)" if c["detected"] else "no alarm in window"))
        return "saved"

    def handle(key, t, text=None):
        nonlocal open_excl, status
        if key == "o":
            ann["events"].append(dict(clinical_onset_s=round(t, 3), clinical_end_s=None, eeg_onset_s=None,
                                      event_class="seizure", seizure_type="", notes=""))
            undo.append(("event", None))
        elif key == "e" and ann["events"]:
            ev = ann["events"][-1]
            if t > ev["clinical_onset_s"]:
                undo.append(("end", ev.get("clinical_end_s"))); ev["clinical_end_s"] = round(t, 3)
        elif key == "g" and ann["events"]:      # EEG onset for the last event, if known (usually left empty)
            ev = ann["events"][-1]
            undo.append(("eeg", ev.get("eeg_onset_s"))); ev["eeg_onset_s"] = round(t, 3)
        elif key == "t" and ann["events"]:
            ev = ann["events"][-1]
            s = text if text is not None else ask("Seizure type", "Seizure type (e.g. P, PG, G, focal motor...):", ev.get("seizure_type", ""))
            if s is not None:
                undo.append(("type", ev.get("seizure_type"))); ev["seizure_type"] = s.strip()
        elif key == "n":
            s = text if text is not None else ask("Note", f"Note at {A.hms(t)}:")
            if s:
                inside = [e for e in ann["events"] if e["clinical_onset_s"] <= t <= (e.get("clinical_end_s") or 1e12)]
                if inside:
                    e = inside[-1]; undo.append(("evnote", (e, e.get("notes", ""))))
                    e["notes"] = (e.get("notes", "") + " | " if e.get("notes") else "") + s.strip()
                else:
                    ann["notes"].append(dict(start_s=round(t, 3), end_s=round(t + 1.0, 3), text=s.strip()))
                    undo.append(("note", None))
        elif key == "x":
            if open_excl is None:
                open_excl = t
            else:
                ann["excludes"].append(dict(start_s=round(min(open_excl, t), 3), end_s=round(max(open_excl, t), 3), reason="other"))
                undo.append(("excl", None)); open_excl = None
        elif key == "u" and undo:
            kind, val = undo.pop()
            if kind == "event": ann["events"].pop()
            elif kind == "end": ann["events"][-1]["clinical_end_s"] = val
            elif kind == "eeg": ann["events"][-1]["eeg_onset_s"] = val
            elif kind == "type": ann["events"][-1]["seizure_type"] = val
            elif kind == "evnote": val[0]["notes"] = val[1]
            elif kind == "note": ann["notes"].pop()
            elif kind == "excl": ann["excludes"].pop()
        elif key == "s":
            status = save(); return
        else:
            return
        status = "unsaved"

    if a.keys:   # scripted test run, no window
        for item in a.keys.split(";"):
            parts = item.split(":", 2)
            handle(parts[1], float(parts[0]), parts[2] if len(parts) > 2 else None)
        rcap = cv2.VideoCapture(str(rendered))
        rcap.set(cv2.CAP_PROP_POS_MSEC, float(a.keys.split(";")[-1].split(":")[0]) * 1000)
        ok, fr = rcap.read()
        if ok:
            cv2.imwrite(str(video.with_name(video.stem + "_review_frame.png")),
                        np.vstack([fr, draw_bar(fr.shape[1], float(a.keys.split(";")[-1].split(":")[0]), dur, ann, alarms, open_excl, status)]))
        if status != "saved":
            save()
        return

    cap = cv2.VideoCapture(str(rendered))
    rfps = cap.get(cv2.CAP_PROP_FPS) or fps
    n_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    paused, fi, frame = False, 0, None
    wall0, f0 = time.perf_counter(), 0
    while True:
        if not paused or frame is None:
            ok, fr = cap.read()
            if ok:
                frame = fr; fi += 1
            else:
                paused = True
        t = max(0, fi - 1) / rfps
        out = np.vstack([frame, draw_bar(frame.shape[1], t, dur, ann, alarms, open_excl, status)])
        cv2.imshow(WIN, out)
        lag = (fi - f0) / rfps - (time.perf_counter() - wall0)
        k = cv2.waitKey(30 if paused else max(1, int(lag * 1000))) & 0xFF
        if k == 255:
            continue
        c = chr(k)
        if c == "q":
            if status != "saved" and ask_yes("Save?", "Save annotations before quitting?"):
                save()
            break
        if c == " ":
            paused = not paused
        elif c in "jl":
            fi = int(np.clip(fi + (-5 if c == "j" else 5) * rfps, 0, max(0, n_fr - 1)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi); frame = None; paused = False
        else:
            handle(c, t)
        wall0, f0 = time.perf_counter(), fi
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
