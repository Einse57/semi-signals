"""Load your own video, run Detector A over it, then play it back with the overlay and annotate.

  python -m src.review_video                       # opens a file picker
  python -m src.review_video --input my_clip.mp4 --annotator AL

Step 1 (process): runs the seizure viewer over the file (pose + Detector A + drivers) and saves
  <stem>_semisignals.mp4 (rendered overlay), <stem>_scores.csv and <stem>_semisignals.json (what the
  render was made with) next to the video. Skipped when all three exist, are newer than the video and
  match the current settings and overlay version (use --reprocess to redo).
Step 2 (review): plays the rendered video in sync, with a timeline of model alarms and your marks.
  Keys:  space pause | j / l  back / forward 5 s | o clinical onset (new event) | e clinical end
         g EEG onset (optional) | t seizure type (pick list) | n note (attached to the event under the
         playhead, else a timed note) | x exclude span start/stop | u undo last mark | s save now
         r replay from the start | h show/hide the key legend | q quit
  A control panel next to the video has the same actions as buttons, plus Open folder and Open in ELAN.
  Every change is saved straight away (autosave, atomic writes); the status shows "saved hh:mm:ss".
  At the end of the video the window stays open on the last frame (r replay, q quit).
Saves next to the ORIGINAL video (see src/annotation.py): <stem>.eaf (ELAN), <stem>_signals.csv,
<stem>_tsconf.xml, <stem>.pfsx (ELAN viewer layout, only if absent), <stem>.annotation.json.

Everything stays on this computer. Research use only, not diagnostic. Use self-recorded or public
clips; other people's videos need their consent (and likely IRB review) before any use.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src import annotation as A
from src.run_seizure_demo import draw_end_overlay, draw_legend, LEGEND_H, RENDER_VERSION
from src.textdraw import put_text, text_width, fit_text, make_dpi_aware, work_area

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
    p.add_argument("--keys", default=None, help="testing: scripted keys 't:key[:text];...' (no window)")
    return p.parse_args(argv)


# ── dialogs (tkinter; a console prompt only when there is no tkinter and a terminal) ───
_tk = None
_parent = None          # the control panel, so dialogs open on top of it


def _root():
    global _tk
    if _tk is None:
        try:
            import tkinter as tk
            _tk = tk.Tk(); _tk.withdraw()
        except Exception:
            _tk = False
    return _tk or None


def _tty():
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def pick_file():
    r = _root()
    if r is None:
        return input("video file: ").strip().strip('"') if _tty() else ""
    from tkinter import filedialog
    return filedialog.askopenfilename(parent=r, title="Choose a video (stays on this computer)",
                                      filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All files", "*.*")])


def ask(title, prompt, initial=""):
    r = _root()
    if r is None:
        if not _tty():
            return initial
        s = input(f"{prompt} [{initial}]: ").strip()
        return s or initial
    from tkinter import simpledialog
    return simpledialog.askstring(title, prompt, initialvalue=initial, parent=_parent or r)


def ask_choice(title, prompt, options, initial=""):
    """Pick list (editable, so an unlisted type can still be typed). Returns the text, or None if cancelled."""
    r = _root()
    if r is None:
        if not _tty():
            return None
        for i, o in enumerate(options, 1):
            print(f"  {i:2d}  {o}")
        s = input(f"{prompt} (number or text) [{initial}]: ").strip()
        return options[int(s) - 1] if s.isdigit() and 0 < int(s) <= len(options) else (s or initial)
    import tkinter as tk
    from tkinter import ttk
    d = tk.Toplevel(_parent or r); d.title(title); d.resizable(False, False)
    try:
        d.attributes("-topmost", True)
    except tk.TclError:
        pass
    var, res = tk.StringVar(value=initial or options[0]), {"v": None}
    ttk.Label(d, text=prompt).pack(padx=12, pady=(12, 4), anchor="w")
    lb = tk.Listbox(d, height=min(12, len(options)), width=40, exportselection=False)
    for o in options:
        lb.insert("end", o)
    if initial in options:
        lb.selection_set(options.index(initial)); lb.see(options.index(initial))
    lb.pack(padx=12, fill="x")
    lb.bind("<<ListboxSelect>>", lambda e: var.set(lb.get(lb.curselection()[0])) if lb.curselection() else None)
    ttk.Label(d, text="or type:").pack(padx=12, pady=(6, 0), anchor="w")
    en = ttk.Entry(d, textvariable=var, width=42); en.pack(padx=12, fill="x")
    def done(ok):
        res["v"] = var.get().strip() if ok else None; d.destroy()
    lb.bind("<Double-Button-1>", lambda e: done(True))
    d.bind("<Return>", lambda e: done(True)); d.bind("<Escape>", lambda e: done(False))
    b = ttk.Frame(d); b.pack(pady=10)
    ttk.Button(b, text="OK", command=lambda: done(True)).pack(side="left", padx=4)
    ttk.Button(b, text="Cancel", command=lambda: done(False)).pack(side="left", padx=4)
    d.protocol("WM_DELETE_WINDOW", lambda: done(False))
    d.grab_set(); lb.focus_set(); d.wait_window()
    return res["v"]


def show_error(title, msg):
    print(f"[review] {title}: {msg}")
    r = _root()
    if r is not None:
        from tkinter import messagebox
        messagebox.showerror(title, msg, parent=_parent or r)


def open_path(path):
    path = str(path)
    if os.name == "nt":
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def find_elan():
    """ELAN.exe in the usual Windows install folders (newest version first), else None."""
    if os.name != "nt":
        return None
    roots = [os.environ.get(k) for k in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
    roots += [os.path.join(os.environ["LOCALAPPDATA"], "Programs")] if os.environ.get("LOCALAPPDATA") else []
    hits = []
    for r in filter(None, roots):
        hits += glob.glob(os.path.join(r, "ELAN*", "ELAN.exe")) + glob.glob(os.path.join(r, "ELAN*", "bin", "ELAN.exe"))
    return sorted(hits, reverse=True)[0] if hits else None


def open_in_elan(eaf):
    exe = find_elan()
    if exe:
        subprocess.Popen([exe, str(eaf)])
    else:
        open_path(eaf)            # whatever opens .eaf files (ELAN registers the extension)


# ── control panel (tk, driven from the OpenCV loop with update(): no second mainloop) ───
PANEL_BUTTONS = [("Onset", "o"), ("End", "e"), ("EEG onset", "g"), ("Type...", "t"), ("Note...", "n"),
                 ("Exclude start/stop", "x"), ("Undo", "u"), ("Back 5 s", "j"), ("Pause/Play", " "),
                 ("Fwd 5 s", "l"), ("Replay", "r"), ("Save now", "s"), ("Open folder", "folder"),
                 ("Open in ELAN", "elan"), ("Show/hide keys", "h"), ("Quit", "q")]


class ControlPanel:
    def __init__(self, title):
        global _parent
        self.pending, self.ok = [], False
        r = _root()
        if r is None:
            return
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.win = w = tk.Toplevel(r)
        w.title(title); w.resizable(False, False)
        try:
            w.attributes("-topmost", True)
        except tk.TclError:
            pass
        f = ttk.Frame(w, padding=8); f.pack(fill="both")
        self.status = tk.StringVar(value="")
        ttk.Label(f, text="Annotate", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        for i, (label, code) in enumerate(PANEL_BUTTONS):
            ttk.Button(f, text=label, width=17, command=lambda c=code: self.pending.append(c)).grid(
                row=1 + i // 2, column=i % 2, padx=2, pady=2, sticky="ew")
        try:                                       # tk scaling is pixels per point (1.333 at 96 dpi)
            ui = max(1.0, float(w.tk.call("tk", "scaling")) / (96 / 72))
        except (tk.TclError, ValueError):
            ui = 1.0
        ttk.Label(f, textvariable=self.status, foreground="#060", wraplength=int(260 * ui)).grid(
            row=2 + len(PANEL_BUTTONS) // 2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(f, text="Research use only, not diagnostic.", foreground="#a33").grid(
            row=3 + len(PANEL_BUTTONS) // 2, column=0, columnspan=2, sticky="w")
        w.protocol("WM_DELETE_WINDOW", lambda: self.pending.append("q"))
        _parent, self.ok = w, True
        self.update()

    def place(self, x, y):
        if self.ok:
            self.win.geometry(f"+{int(x)}+{int(y)}")

    def screen_width(self):
        return self.win.winfo_screenwidth() if self.ok else 1920

    def screen_height(self):
        return self.win.winfo_screenheight() if self.ok else 1080

    def width(self):
        if not self.ok:
            return 0
        self.win.update_idletasks()
        return max(self.win.winfo_reqwidth(), self.win.winfo_width())

    def set_status(self, text):
        if self.ok:
            self.status.set(text)

    def update(self):
        if self.ok:
            try:
                self.win.update()
            except self.tk.TclError:
                self.ok = False

    def next_code(self):
        return self.pending.pop(0) if self.pending else None

    def close(self):
        global _parent
        if self.ok:
            try:
                self.win.destroy()
            except self.tk.TclError:
                pass
        self.ok, _parent = False, None


# ── processing ───────────────────────────────────────────────
def outputs_for(video):
    v = Path(video)
    return v.with_name(v.stem + "_semisignals.mp4"), v.with_name(v.stem + "_scores.csv")


def stamp_path(video):
    v = Path(video)
    return v.with_name(v.stem + "_semisignals.json")


def render_settings(a):
    """What a saved render depends on; a different overlay version or setting means render again."""
    return dict(render_version=RENDER_VERSION, detector_a=Path(a.detector_a).name if a.detector_a else "",
                vsvig=bool(a.vsvig), hands=not a.no_hands)


def write_stamp(a, video):
    import json
    stamp_path(video).write_text(json.dumps(render_settings(a), indent=1))


def render_is_fresh(a, video):
    import json
    rendered, scores = outputs_for(video)
    st = stamp_path(video)
    if not all(p.exists() and p.stat().st_mtime > Path(video).stat().st_mtime for p in (rendered, scores, st)):
        return False
    try:
        return json.loads(st.read_text()) == render_settings(a)
    except (OSError, ValueError):
        return False


def process(a, video):
    rendered, scores = outputs_for(video)
    if render_is_fresh(a, video) and not a.reprocess:
        print(f"[review] using existing {rendered.name} and {scores.name}")
        return rendered, scores
    if rendered.exists() and not a.reprocess:
        print(f"[review] {rendered.name} was made with other settings or an older overlay; rendering again")
    from src import run_seizure_demo
    argv = ["--input", str(video), "--detector-a", a.detector_a, "--vsvig", a.vsvig, "--device", a.device,
            "--fast", "--save", str(rendered), "--log", str(scores), "--exit-at-end"]
    if a.no_hands:
        argv.append("--no-hands")
    if a.headless_process or a.keys:
        argv.append("--headless")
    print("[review] processing (pose + Detector A); this takes about the length of the video")
    run_seizure_demo.main(argv)
    write_stamp(a, video)
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
REVIEW_KEYS = [[("space", "pause"), ("j / l", "-5 s / +5 s"), ("o", "onset"), ("e", "end"), ("g", "EEG onset"),
                ("t", "type"), ("n", "note"), ("x", "exclude start/stop")],
               [("u", "undo"), ("s", "save now"), ("r", "replay"), ("h", "hide keys"), ("q", "quit"),
                ("", "autosave is on: every change is written immediately")]]


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
        info += "  |  exclude open"
    key = "red: model alarms   green: your marks"
    kw = text_width(key, 13)
    room = w - 2 * L - kw - 24
    if room > 200:
        put_text(img, key, (w - R, 48), 13, (150, 150, 150), anchor="rm")
    else:
        room = w - 2 * L
    put_text(img, fit_text(info, room, 14), (L, 48), 14, (225, 225, 225), anchor="lm")
    put_text(img, fit_text(f"[{status}]", w - 2 * L, 14), (L, 65), 14,
             (80, 200, 80) if "saved" in status else (0, 160, 255), anchor="lm")
    return img


def review_legend(w):
    return np.vstack([draw_legend(w, row) for row in REVIEW_KEYS])


def compose(frame, t, dur, ann, alarms, open_excl, status, legend=True, at_end=False):
    out = np.vstack([frame, draw_bar(frame.shape[1], t, dur, ann, alarms, open_excl, status)])
    if at_end:
        out = draw_end_overlay(out, "End of video:  R replay   Q quit", sub="annotation keys still work (j to step back)")
    if legend:
        out = np.vstack([out, review_legend(out.shape[1])])
    return out


def display_width(rw, rh, panel, area=None):
    """Width for the rendered video so the window plus the control panel fit the screen work area."""
    W, H = area or work_area(default=(panel.screen_width(), panel.screen_height() - 40))
    pw = panel.width() or 300
    by_w = W - pw - 50
    by_h = (H - 60 - BAR_H - 2 * LEGEND_H) * rw / max(1, rh)
    return int(max(480, min(rw, by_w, by_h)))


def main(argv=None):
    a = parse_args(argv)
    if not a.keys:
        make_dpi_aware()                  # crisp text and windows on scaled Windows displays (before any window)
    from src.hf_models import resolve_detector_a
    a.detector_a = resolve_detector_a(a.detector_a)
    video = a.input or pick_file()
    if not video:
        print("[review] no file chosen"); return
    video = Path(video).resolve()
    print("[review] step 1 of 2: processing (pose + Detector A)", flush=True)
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
    print("[review] hashing the video (sha256)", flush=True)
    sha = A.sha256_file(video)
    annotator = a.annotator or ("" if a.keys else ask("Annotator", "Your initials (annotator):", os.environ.get("USERNAME", ""))) or ""
    pat_id = a.pat_id or ("" if a.keys else ask("Patient", "Patient id or pseudonym (no real names):", "self")) or ""
    ann = A.new_annotation(video, fps, dur, annotator=annotator, pat_id=pat_id, consent=a.consent, sha256=sha)
    eaf = A.paths_for(video)["eaf"]
    status = "nothing marked yet"
    if eaf.exists():
        try:
            prev = A.read_eaf(eaf, video)          # raises ShaMismatch for a different video
        except A.ShaMismatch as e:
            if a.keys:
                raise
            show_error("Different video", f"{eaf.name} was made for a different video (sha256 does not match), "
                                          f"so it is not loaded or overwritten. Move or rename it first.\n\n{e}")
            raise SystemExit(2)
        ann["events"] = [dict(e, eeg_onset_s=e.get("eeg_onset_s"), notes=e.get("notes", "")) for e in prev["events"]]
        ann["notes"] = [dict(start_s=r["start_s"], end_s=r["end_s"], text=r["value"])
                        for r in prev["tiers"].get("Notes", {}).get("rows", [])]
        ann["excludes"] = [dict(start_s=r["start_s"], end_s=r["end_s"], reason=r["value"])
                           for r in prev["tiers"].get("Exclude", {}).get("rows", [])]
        print(f"[review] loaded {len(ann['events'])} events from {eaf.name} (Model_* tiers are regenerated)")
        status = f"loaded {len(ann['events'])} event(s) from {eaf.name}"
    model_id = Path(a.detector_a).stem
    undo, open_excl, n_saves = [], None, 0

    def save(quiet=False):
        nonlocal n_saves
        ann["model"] = A.model_section(model_id, meta_thr, alarms, per_second(rows))
        ann["comparison"] = compare(ann["events"], alarms)
        try:
            out = A.save_all(ann, video)
        except OSError as e:
            print(f"[review] SAVE FAILED: {e}", flush=True)
            return f"SAVE FAILED ({type(e).__name__}); press s to retry"
        n_saves += 1
        when = time.strftime("%H:%M:%S")
        if not quiet:
            print(f"[review] saved {when}: {out['eaf'].name} and sidecars", flush=True)
        return f"saved {when}"

    def report():
        for c in ann["comparison"]["events"] if ann.get("comparison") else []:
            print(f"[review] event {c['event']}: onset {A.hms(c['clinical_onset_s'])} -> "
                  + (f"alarm at {A.hms(c['alarm_start_s'])} (delay {c['delay_s']:+.1f} s)" if c["detected"] else "no alarm in window"))

    def handle(key, t, text=None):
        """Apply one mark; returns True if the annotation changed (then it is autosaved)."""
        nonlocal open_excl
        if key == "o":
            ann["events"].append(dict(clinical_onset_s=round(t, 3), clinical_end_s=None, eeg_onset_s=None,
                                      event_class="seizure", seizure_type="", notes=""))
            undo.append(("event", None))
        elif key == "e" and ann["events"]:
            ev = ann["events"][-1]
            if t <= ev["clinical_onset_s"]:
                return False
            undo.append(("end", ev.get("clinical_end_s"))); ev["clinical_end_s"] = round(t, 3)
        elif key == "g" and ann["events"]:      # EEG onset for the last event, if known (usually left empty)
            ev = ann["events"][-1]
            undo.append(("eeg", ev.get("eeg_onset_s"))); ev["eeg_onset_s"] = round(t, 3)
        elif key == "t" and ann["events"]:
            ev = ann["events"][-1]
            s = text if text is not None else ask_choice("Seizure type", f"Seizure type for the event at {A.hms(ev['clinical_onset_s'])}:",
                                                         list(A.SEIZURE_TYPES), ev.get("seizure_type", ""))
            if s is None or s.strip() == (ev.get("seizure_type") or ""):
                return False
            undo.append(("type", ev.get("seizure_type"))); ev["seizure_type"] = s.strip()
        elif key == "n":
            s = text if text is not None else ask("Note", f"Note at {A.hms(t)}:")
            if not s:
                return False
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
                return False
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
        else:
            return False
        return True

    if a.keys:   # scripted test run, no window: autosaves after every change, like the window
        last_t = 0.0
        for item in a.keys.split(";"):
            parts = item.split(":", 2)
            last_t = float(parts[0])
            if parts[1] == "s" or handle(parts[1], last_t, parts[2] if len(parts) > 2 else None):
                status = save()
        rcap = cv2.VideoCapture(str(rendered))
        rcap.set(cv2.CAP_PROP_POS_MSEC, last_t * 1000)
        ok, fr = rcap.read()
        if ok:
            cv2.imwrite(str(video.with_name(video.stem + "_review_frame.png")),
                        compose(fr, last_t, dur, ann, alarms, open_excl, status))
        if not n_saves:
            save()
        report()
        return

    cap = cv2.VideoCapture(str(rendered))
    rfps = cap.get(cv2.CAP_PROP_FPS) or fps
    n_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    rw, rh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1600), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 540)
    print("[review] step 2 of 2: review window open", flush=True)
    panel = ControlPanel(f"semi-signals | {video.name}")
    legend = True
    win_w = display_width(rw, rh, panel)
    disp_h = int(round(rh * win_w / rw))
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)      # 1:1 pixels: the frame is scaled here with INTER_AREA
    cv2.moveWindow(WIN, 10, 10)
    panel.place(10 + win_w + 24, 10)
    panel.set_status(status)
    paused, fi, frame, at_end = False, 0, None, False
    shown_src = shown = None
    wall0, f0 = time.perf_counter(), 0
    while True:
        due = frame is None or (not paused and (fi - f0) / rfps <= time.perf_counter() - wall0)
        if due:
            ok, fr = cap.read()
            if ok:
                frame = fr; fi += 1; at_end = False
            else:
                paused = True; at_end = True
                if frame is None and n_fr > 1:      # jumped past the end: show the last frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, n_fr - 2); ok, fr = cap.read()
                    if ok:
                        frame = fr; fi = n_fr - 1
        if frame is None:
            print(f"[review] cannot read {rendered}"); break
        t = max(0, fi - 1) / rfps
        if frame is not shown_src:          # scale each new frame once (bar and legend are drawn at this width)
            shown_src = frame
            shown = frame if win_w == rw else cv2.resize(frame, (win_w, disp_h), interpolation=cv2.INTER_AREA)
        cv2.imshow(WIN, compose(shown, t, dur, ann, alarms, open_excl,
                                "end of video  |  " + status if at_end else status, legend, at_end))
        lag = (fi - f0) / rfps - (time.perf_counter() - wall0)
        k = cv2.waitKey(30 if paused else max(1, min(15, int(lag * 1000)))) & 0xFF
        panel.update()                              # tk runs inside this loop (no second mainloop)
        try:
            closed = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            closed = True
        c = "q" if closed else (chr(k) if k != 255 else panel.next_code())
        if c is None:
            continue
        if c == "r":
            fi = 0; cap.set(cv2.CAP_PROP_POS_FRAMES, 0); frame = None; paused = False; at_end = False
        elif c in ("q", "\x1b"):
            break
        elif c == " ":
            paused = not paused
        elif c == "h":
            legend = not legend
        elif c in ("j", "l"):
            fi = int(np.clip(fi + (-5 if c == "j" else 5) * rfps, 0, max(0, n_fr - 1)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi); frame = None; paused = False
        elif c == "s":
            status = save()
        elif c == "folder":
            open_path(video.parent)
        elif c == "elan":
            status = save()
            open_in_elan(eaf)
        elif handle(c, t):
            status = save()
        panel.set_status(f"{status}  |  {len(ann['events'])} event(s)" + ("  |  exclude open" if open_excl is not None else ""))
        wall0, f0 = time.perf_counter(), fi
    if not n_saves or status.startswith("SAVE FAILED"):
        status = save()
    report()
    print(f"[review] {status}; files are next to {video.name} in {video.parent}", flush=True)
    panel.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
