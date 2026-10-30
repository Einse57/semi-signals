"""semi-signals launcher: a small window to pick a video and start the viewer, the annotation
tool or the model/data download. Standard library only (tkinter); the tools run as child
processes with the same Python, and their output goes to the log pane.

    python -m src.launcher            (or double-click start.cmd on Windows)

Research use only, not diagnostic.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_LINE = "Research use only, not diagnostic. Video stays on this computer."
SKIP_SUFFIXES = ("_semisignals.mp4", "_rendered.mp4")
CONSENT = {"self": "I recorded this video myself", "public": "Public dataset clip",
           "consented": "Recorded with documented consent", "unknown": "Not sure"}
VIEWER_KEYS_TEXT = ("Viewer:      q quit   space pause/play   f fast/real time   r replay   o open another video   "
                    "m face on/off   h hide keys")
ANNOT_MODES = {"body": "Body/Hands", "face": "Face", "both": "Both"}
REVIEW_KEYS_TEXT = ("Annotation:  o onset   e end   g EEG onset   t type   n note   x exclude start/stop   u undo\n"
                    "             Face/Both mode: d side (L/R/both); Both mode: o body onset, f face onset\n"
                    "             space pause   j / l  -5 s / +5 s   r replay   s save now   h hide keys   q quit\n"
                    "             (the same actions are buttons in the panel next to the video; every change autosaves)")
PROGRESS_RE = re.compile(r"progress (\d+)%(?: \((.*)\))?")


def parse_progress(line: str):
    """'[viewer] progress 42% (25 of 60 s)' -> (42, '25 of 60 s'); None for other lines."""
    m = PROGRESS_RE.search(line)
    return (int(m.group(1)), m.group(2) or "") if m else None


# ── helpers without tkinter (tested headless) ────────────────
def settings_path() -> Path:
    base = os.environ.get("APPDATA") if os.name == "nt" else None
    return (Path(base) / "semi-signals" if base else Path.home() / ".semi-signals") / "launcher.json"


def load_settings(p: Path | None = None) -> dict:
    p = p or settings_path()
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(d: dict, p: Path | None = None):
    p = p or settings_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def sample_clips(data_dir: Path) -> list[Path]:
    """data/pat*/*.mp4 as fetched by scripts/fetch_models.py (rendered outputs skipped)."""
    if not data_dir.is_dir():
        return []
    return sorted(p for p in data_dir.glob("**/pat*/*.mp4") if not p.name.endswith(SKIP_SUFFIXES))


def find_labels(video: Path | None, data_dir: Path) -> Path | None:
    """A Label.xlsx next to the video or up to three folders above it, else in data/ or one folder below."""
    cands = []
    if video:
        cands += [d / "Label.xlsx" for d in list(Path(video).parents)[:4]]
    cands += [data_dir / "Label.xlsx"] + (sorted(data_dir.glob("*/Label.xlsx")) if data_dir.is_dir() else [])
    return next((c for c in cands if c.is_file()), None)


def is_dataset_clip(video: Path, data_dir: Path) -> bool:
    """Label.xlsx only describes the sample dataset: data/patNN/<seizure>.mp4, also when the clips sit
    one or more folders deeper, e.g. data/dataset/patNN/<seizure>.mp4 from a manual download."""
    try:
        rel = video.resolve().relative_to(data_dir.resolve())
    except ValueError:
        return False
    return len(rel.parts) >= 2 and rel.parts[-2].lower().startswith("pat")


def models_status(models_dir: Path, data_dir: Path) -> str:
    man = {}
    mp = models_dir / "MANIFEST.json"
    try:
        man = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    det = "ok" if (models_dir / "detector_a_all.txt").exists() else "missing"
    vs = "ok" if (models_dir / "vsvig_base.xml").exists() else "not installed"
    n = len(sample_clips(data_dir))
    lab = "Label.xlsx ok" if (data_dir / "Label.xlsx").exists() else "no Label.xlsx"
    if man:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(mp.stat().st_mtime))
        src = f"MANIFEST.json {when}, {sum(1 for k in man if k != 'data')} model files"
    else:
        src = "no MANIFEST.json yet: click 'Fetch models & sample data'"
    return f"Models: Detector A {det} | VSViG {vs} | samples: {n} clip(s), {lab}  ({src})"


def viewer_cmd(s: dict, data_dir: Path, models_dir: Path, extra: list[str] = ()) -> list[str]:
    v = Path(s["video"])
    cmd = [sys.executable, "-u", "-m", "src.run_seizure_demo", "--input", str(v),
           "--device", s.get("device", "auto").lower(),
           "--detector-a", str(models_dir / "detector_a_all.txt"),
           "--vsvig", str(models_dir / "vsvig_base.xml") if s.get("vsvig") else ""]
    # markers are drawn only for clips listed in the labels file (patNN folder + SzN file name);
    # the viewer ignores the file for any other video, so it is always safe to pass
    lab = s.get("labels", "") or str(find_labels(v, data_dir) or "")
    if lab and Path(lab).is_file():
        cmd += ["--labels", lab]
    if s.get("loop"):
        cmd.append("--loop")
    if not s.get("hands", True):
        cmd.append("--no-hands")
    if not s.get("explain", True):
        cmd.append("--no-explain")
    if s.get("face"):
        cmd.append("--face")
    return cmd + list(extra)


def annotate_cmd(s: dict, data_dir: Path, models_dir: Path, extra: list[str] = ()) -> list[str]:
    v = Path(s["video"])
    cmd = [sys.executable, "-u", "-m", "src.review_video", "--input", str(v),
           "--device", s.get("device", "auto").lower(),
           "--detector-a", str(models_dir / "detector_a_all.txt"),
           "--consent", s.get("consent") or ("public" if is_dataset_clip(v, data_dir) else "self"),
           "--headless-process"]                 # progress goes to the launcher instead of a second window
    if s.get("vsvig"):
        cmd += ["--vsvig", str(models_dir / "vsvig_base.xml")]
    if s.get("annotator", "").strip():
        cmd += ["--annotator", s["annotator"].strip()]
    if s.get("pat_id", "").strip():
        cmd += ["--pat-id", s["pat_id"].strip()]
    if not s.get("hands", True):
        cmd.append("--no-hands")
    if s.get("annot_mode", "body") in ("face", "both"):
        cmd += ["--mode", s["annot_mode"]]
    return cmd + list(extra)


def deid_export_cmd(s: dict, d: dict, out=None, raw_root=None) -> list[str]:
    """segment_export for the face de-identification choice d (src/deid_ui.DeidSelector.state())."""
    from src import deid_options as O
    v = Path(s["video"])
    eaf = v.with_suffix(".eaf")
    return O.export_cmd(v, d["deid_method"], python=sys.executable, annotations=eaf if eaf.is_file() else None,
                        out=out, raw_root=raw_root, swap_config=d.get("swap_config") or None,
                        research_ack=d.get("research_ack", False), shareable=d.get("shareable", False),
                        mode=s.get("annot_mode", "face") if s.get("annot_mode") in ("face", "both") else "face",
                        face_backend=d.get("face_backend", "mediapipe"), face_device=d.get("face_device", "CPU"))


def _landmarker_keys(d: dict) -> dict:
    from src.deid_ui import landmarker_settings
    return landmarker_settings(d)


def has_torch() -> bool:
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def has_mediapipe() -> bool:
    import importlib.util
    return importlib.util.find_spec("mediapipe") is not None


def fetch_cmd(data_dir: Path, models_dir: Path, vsvig: bool | None = None, face: bool = False) -> list[str]:
    """VSViG is optional and its conversion needs torch; skip it when torch is not installed.
    face: also fetch the optional MediaPipe face model."""
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "fetch_models.py"),
           "--models-dir", str(models_dir), "--data-dir", str(data_dir)]
    if not (has_torch() if vsvig is None else vsvig):
        cmd.append("--skip-vsvig")
    if face:
        cmd.append("--face")
    return cmd


def child_env(proxy: str = "") -> dict:
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore::RuntimeWarning")
    if proxy.strip():
        for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            env[k] = proxy.strip()
    return env


def openvino_devices() -> list[str]:
    try:
        from openvino import Core
    except ImportError:
        try:
            from openvino.runtime import Core
        except ImportError:
            return []
    try:
        return list(Core().available_devices)
    except Exception:
        return []


def open_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(path))  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── the window ───────────────────────────────────────────────
class Launcher:
    def __init__(self, root, data_dir: Path, models_dir: Path, extra: list[str]):
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText
        self.tk, self.root = tk, root
        self.data_dir, self.models_dir, self.extra = data_dir, models_dir, extra
        self.proc: subprocess.Popen | None = None
        self.proc_name = ""
        self.q: queue.Queue = queue.Queue()
        st = load_settings()

        root.title("semi-signals: video seizure detection")
        root.minsize(760, 560)
        try:
            ttk.Style().theme_use("vista" if os.name == "nt" else "clam")
        except tk.TclError:
            pass
        pad = dict(padx=8, pady=4)
        m = ttk.Frame(root, padding=10); m.pack(fill="both", expand=True)
        m.columnconfigure(1, weight=1)

        ttk.Label(m, text="Video seizure detection",
                  font=("Segoe UI", 14, "bold") if os.name == "nt" else ("DejaVu Sans", 14, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(m, text=RESEARCH_LINE, foreground="#a33").grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 8))

        # video
        ttk.Label(m, text="Video").grid(row=2, column=0, sticky="w", **pad)
        self.video = tk.StringVar(value=st.get("video", ""))
        self.video_box = ttk.Combobox(m, textvariable=self.video)
        self.video_box.grid(row=2, column=1, sticky="ew", **pad)
        self.video_box.bind("<<ComboboxSelected>>", lambda e: self._video_changed())
        ttk.Button(m, text="Browse...", command=self.browse_video).grid(row=3 - 1, column=2, sticky="ew", **pad)
        ttk.Label(m, text="Pick a sample clip from data\\ or browse to any video file.",
                  foreground="#666").grid(row=3, column=1, sticky="w", padx=8)

        # labels
        ttk.Label(m, text="Labels (optional)").grid(row=4, column=0, sticky="w", **pad)
        lab_default = str(find_labels(None, data_dir) or "")
        self.labels = tk.StringVar(value=st.get("labels") or lab_default)
        ttk.Entry(m, textvariable=self.labels).grid(row=4, column=1, sticky="ew", **pad)
        ttk.Button(m, text="Browse...", command=self.browse_labels).grid(row=4, column=2, sticky="ew", **pad)
        ttk.Label(m, text="Onset markers show for clips listed in it (patNN folder, SzN file); found automatically if left empty.",
                  foreground="#666").grid(row=5, column=1, sticky="w", padx=8)

        # options
        opt = ttk.LabelFrame(m, text="Options", padding=8)
        opt.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        ttk.Label(opt, text="Device").grid(row=0, column=0, sticky="w")
        self.device = tk.StringVar(value=st.get("device", ""))
        self.dev_btn = {}
        for i, d in enumerate(("GPU", "CPU")):
            self.dev_btn[d] = ttk.Radiobutton(opt, text=d, value=d, variable=self.device)
            self.dev_btn[d].grid(row=0, column=1 + i, sticky="w", padx=6)
        self.dev_note = ttk.Label(opt, text="checking OpenVINO devices...", foreground="#666")
        self.dev_note.grid(row=0, column=3, columnspan=3, sticky="w", padx=6)
        self.loop = tk.BooleanVar(value=st.get("loop", False))
        self.hands = tk.BooleanVar(value=st.get("hands", True))
        self.explain = tk.BooleanVar(value=st.get("explain", True))
        self.vsvig = tk.BooleanVar(value=st.get("vsvig", False))
        ttk.Checkbutton(opt, text="Loop playback", variable=self.loop).grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Checkbutton(opt, text="Hands", variable=self.hands).grid(row=1, column=2, sticky="w", padx=6, pady=(6, 0))
        ttk.Checkbutton(opt, text="Explanations", variable=self.explain).grid(row=1, column=3, sticky="w", padx=6, pady=(6, 0))
        self.vsvig_cb = ttk.Checkbutton(opt, text="VSViG (reference model)", variable=self.vsvig)
        self.vsvig_cb.grid(row=1, column=4, sticky="w", padx=6, pady=(6, 0))
        self.face = tk.BooleanVar(value=st.get("face", False))
        self.face_cb = ttk.Checkbutton(opt, text="Face (MediaPipe)", variable=self.face)
        self.face_cb.grid(row=1, column=5, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(opt, text="Annotator initials").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.annotator = tk.StringVar(value=st.get("annotator", ""))
        ttk.Entry(opt, textvariable=self.annotator, width=8).grid(row=2, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(opt, text="Patient ID / pseudonym").grid(row=2, column=2, sticky="e", pady=(6, 0))
        self.pat_id = tk.StringVar(value=st.get("pat_id", ""))
        ttk.Entry(opt, textvariable=self.pat_id, width=14).grid(row=2, column=3, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(opt, text="Video source").grid(row=2, column=4, sticky="e", pady=(6, 0))
        self.consent = tk.StringVar(value=CONSENT.get(st.get("consent", ""), ""))
        ttk.Combobox(opt, textvariable=self.consent, values=list(CONSENT.values()), state="readonly", width=30).grid(
            row=2, column=5, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(opt, text="Annotation mode").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.annot_mode = tk.StringVar(value=st.get("annot_mode", "body") if st.get("annot_mode") in ANNOT_MODES else "body")
        for i, (k, lab) in enumerate(ANNOT_MODES.items()):
            ttk.Radiobutton(opt, text=lab, value=k, variable=self.annot_mode, command=self._mode_changed).grid(
                row=4, column=1 + i, sticky="w", padx=6, pady=(6, 0))
        from src.deid_ui import DeidSelector
        self.deid = DeidSelector(opt, st, models_dir)
        self.deid.frame.grid(row=5, column=0, columnspan=6, sticky="ew", pady=(8, 0))
        db = ttk.Frame(self.deid.frame); db.grid(row=8, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self.deid_btns = [ttk.Button(db, text="Preview (before/after)", command=self.deid_preview),
                          ttk.Button(db, text="Export de-identified segments", command=self.deid_export)]
        for i, w_ in enumerate(self.deid_btns):
            w_.grid(row=0, column=i, padx=(0, 6))
        self.deid_note = ttk.Label(db, text="", foreground="#666")
        self.deid_note.grid(row=0, column=2, sticky="w")
        self._mode_changed()
        ttk.Label(opt, text="labels: labels/body.json, labels/face.json", foreground="#666").grid(
            row=4, column=4, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(opt, text="Proxy (optional)").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.proxy = tk.StringVar(value=st.get("proxy", ""))
        ttk.Entry(opt, textvariable=self.proxy, width=34).grid(row=3, column=1, columnspan=3, sticky="w", padx=6, pady=(6, 0))
        env_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or ""
        ttk.Label(opt, text=("HTTPS_PROXY is set; leave empty to use it" if env_proxy else "e.g. http://proxy:8080, used for downloads"),
                  foreground="#666").grid(row=3, column=4, columnspan=2, sticky="w", pady=(6, 0))

        # buttons
        b = ttk.Frame(m); b.grid(row=7, column=0, columnspan=3, sticky="ew", pady=8)
        self.btns = {
            "play": ttk.Button(b, text="Play with live scores", command=self.play),
            "annotate": ttk.Button(b, text="Annotate this video", command=self.annotate),
            "folder": ttk.Button(b, text="Open output folder", command=self.open_output),
            "fetch": ttk.Button(b, text="Fetch models & sample data", command=self.fetch),
            "stop": ttk.Button(b, text="Stop", command=self.stop, state="disabled"),
        }
        for i, k in enumerate(("play", "annotate", "folder", "fetch", "stop")):
            self.btns[k].grid(row=0, column=i, padx=(0 if i == 0 else 6, 0), ipadx=4, ipady=2)

        pf = ttk.Frame(m); pf.grid(row=8, column=0, columnspan=3, sticky="ew")
        pf.columnconfigure(1, weight=1)
        self.prog_text = tk.StringVar(value="")
        ttk.Label(pf, textvariable=self.prog_text, width=44).grid(row=0, column=0, sticky="w")
        self.prog = ttk.Progressbar(pf, mode="determinate", maximum=100)
        self.prog.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.status = tk.StringVar()
        ttk.Label(m, textvariable=self.status, foreground="#333").grid(row=11, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.log = ScrolledText(m, height=10, wrap="word", font=("Consolas", 9) if os.name == "nt" else ("DejaVu Sans Mono", 9))
        self.log.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        self.log.configure(state="disabled")
        self.log.tag_configure("err", foreground="#b00")
        m.rowconfigure(9, weight=1)
        kf = ttk.LabelFrame(m, text="Keys", padding=6)
        kf.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        mono = ("Consolas", 9) if os.name == "nt" else ("DejaVu Sans Mono", 9)
        ttk.Label(kf, text=VIEWER_KEYS_TEXT + "\n" + REVIEW_KEYS_TEXT, font=mono, justify="left").pack(anchor="w")

        settings_path().parent.mkdir(parents=True, exist_ok=True)
        self.write(f"--- launcher started {time.strftime('%Y-%m-%d %H:%M')} (python {sys.version.split()[0]}) ---\n")
        self.refresh()
        threading.Thread(target=self._detect_devices, daemon=True).start()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self._pump)

    # state
    def settings(self) -> dict:
        return dict(video=self.video.get().strip(), labels=self.labels.get().strip(), device=self.device.get() or "GPU",
                    loop=self.loop.get(), hands=self.hands.get(), explain=self.explain.get(), face=self.face.get(),
                    vsvig=self.vsvig.get() and (self.models_dir / "vsvig_base.xml").exists(),
                    annotator=self.annotator.get(), pat_id=self.pat_id.get(), proxy=self.proxy.get(),
                    annot_mode=self.annot_mode.get() or "body",
                    deid_method=self.deid.method(), swap_config=self.deid.swap_config.get().strip(),
                    **_landmarker_keys(self.deid.state()),
                    consent=next((k for k, v in CONSENT.items() if v == self.consent.get()), ""))

    def refresh(self):
        clips = sample_clips(self.data_dir)
        self.video_box["values"] = [str(c) for c in clips]
        if not self.video.get() and clips:
            self.video.set(str(clips[0]))
        if not self.labels.get():
            f = find_labels(Path(self.video.get()) if self.video.get() else None, self.data_dir)
            if f:
                self.labels.set(str(f))
        if not self.consent.get() and self.video.get():
            self.consent.set(CONSENT["public" if is_dataset_clip(Path(self.video.get()), self.data_dir) else "self"])
        has_vsvig = (self.models_dir / "vsvig_base.xml").exists()
        self.vsvig_cb.configure(state="normal" if has_vsvig else "disabled")
        if not has_vsvig:
            self.vsvig.set(False)
        self.status.set(models_status(self.models_dir, self.data_dir))

    def _detect_devices(self):
        devs = openvino_devices()
        self.q.put(("devices", devs))

    def _set_devices(self, devs):
        gpu = any(d.upper().startswith("GPU") for d in devs)
        self.dev_btn["GPU"].configure(state="normal" if gpu else "disabled")
        if not self.device.get() or (self.device.get() == "GPU" and not gpu):
            self.device.set("GPU" if gpu else "CPU")
        self.dev_note.configure(text=("OpenVINO devices: " + ", ".join(devs)) if devs else
                                "OpenVINO not found or no devices; using CPU")

    # actions
    def browse_video(self):
        from tkinter import filedialog
        cur = Path(self.video.get()).parent if self.video.get() else self.data_dir
        p = filedialog.askopenfilename(title="Choose a video", initialdir=str(cur if cur.is_dir() else ROOT),
                                       filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv"), ("All files", "*.*")])
        if p:
            self.video.set(str(Path(p)))
            self._video_changed()

    def _video_changed(self):
        v = self.video.get().strip()
        if v:
            self.consent.set(CONSENT["public" if is_dataset_clip(Path(v), self.data_dir) else "self"])
            if not self.labels.get().strip():
                f = find_labels(Path(v), self.data_dir)
                if f:
                    self.labels.set(str(f))

    def browse_labels(self):
        from tkinter import filedialog
        p = filedialog.askopenfilename(title="Choose Label.xlsx", initialdir=str(self.data_dir if self.data_dir.is_dir() else ROOT),
                                       filetypes=[("Excel", "*.xlsx"), ("All files", "*.*")])
        if p:
            self.labels.set(str(Path(p)))

    def _need_video(self) -> dict | None:
        s = self.settings()
        if not s["video"]:
            self.write("Choose a video first (or click 'Fetch models & sample data').\n", "err"); return None
        if not Path(s["video"]).is_file():
            self.write(f"Video not found: {s['video']}\n", "err"); return None
        if not (self.models_dir / "detector_a_all.txt").exists():
            self.write("Detector A model missing: click 'Fetch models & sample data' first.\n", "err"); return None
        return s

    def play(self):
        s = self._need_video()
        if s:
            self.run("viewer", viewer_cmd(s, self.data_dir, self.models_dir, self.extra), s)

    def annotate(self):
        s = self._need_video()
        if s and not s["annotator"].strip():
            from tkinter import simpledialog
            ini = simpledialog.askstring("Annotator", "Your initials (annotator):", parent=self.root)
            if not ini:
                return
            self.annotator.set(ini.strip()); s["annotator"] = ini.strip()
        if s and not s["pat_id"].strip():
            from tkinter import simpledialog
            pid = simpledialog.askstring("Patient", "Patient ID or pseudonym (never a real name):", parent=self.root,
                                         initialvalue="self")
            if not pid:
                return
            self.pat_id.set(pid.strip()); s["pat_id"] = pid.strip()
        if s:
            if not s["consent"]:
                s["consent"] = "public" if is_dataset_clip(Path(s["video"]), self.data_dir) else "self"
                self.consent.set(CONSENT[s["consent"]])
            self.run("annotation", annotate_cmd(s, self.data_dir, self.models_dir), s)

    def _mode_changed(self):
        on = self.annot_mode.get() in ("face", "both")
        self.deid.set_enabled(on)
        for w_ in self.deid_btns:
            w_.configure(state="normal" if on else "disabled")
        self.deid_note.configure(text="" if on else "choose Annotation mode Face or Both to use these")

    def deid_preview(self):
        s = self._need_video()
        if s:
            from src.deid_ui import show_preview
            err = show_preview(self.root, s["video"], self.deid)
            if err:
                self.write(err + "\n", "err")

    def deid_export(self):
        s = self._need_video()
        if not s:
            return
        p = self.deid.problems()
        if p:
            self.write("\n".join(p) + "\n", "err"); return
        d = self.deid.state()
        self.run("export", deid_export_cmd(s, d), s)

    def fetch(self):
        if not has_torch():
            self.write("Note: VSViG (optional reference model) is skipped because torch is not installed.\n")
        self.run("fetch", fetch_cmd(self.data_dir, self.models_dir, face=self.face.get()), self.settings())

    def open_output(self):
        v = self.video.get().strip()
        folder = Path(v).parent if v and Path(v).parent.is_dir() else self.data_dir
        try:
            open_folder(folder)
            self.write(f"Opened {folder}\n")
        except OSError as e:
            self.write(f"Could not open {folder}: {e}\n", "err")

    def run(self, name, cmd, s):
        if self.proc and self.proc.poll() is None:
            self.write(f"The {self.proc_name} is still running; press Stop first.\n", "err"); return
        save_settings(s)
        self.write("\n> " + " ".join(shlex.quote(c) for c in cmd[1:]) + "\n")
        try:
            kw = {}
            if os.name == "nt":
                kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.proc = subprocess.Popen(cmd, cwd=str(ROOT), env=child_env(s.get("proxy", "")),
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         text=True, encoding="utf-8", errors="replace", bufsize=1, **kw)
        except OSError as e:
            self.write(f"Could not start the {name}: {e}\n", "err"); return
        self.proc_name = name
        self.prog.configure(value=0)
        self.prog_text.set({"viewer": "Viewer running", "annotation": "Step 1 of 2: processing (pose + scores)...",
                            "fetch": "Downloading...", "export": "Exporting de-identified segments..."}.get(name, ""))
        self.btns["stop"].configure(state="normal")
        threading.Thread(target=self._reader, args=(self.proc, name), daemon=True).start()

    def _reader(self, proc, name):
        for line in proc.stdout:
            self.q.put(("line", line))
        rc = proc.wait()
        self.q.put(("done", (name, rc)))

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.write(f"Stopping the {self.proc_name}...\n")

    def _pump(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "line":
                    pr = parse_progress(val)
                    if pr is not None:
                        self.prog.configure(value=pr[0])
                        if self.proc_name == "annotation":
                            self.prog_text.set(f"Step 1 of 2: processing {pr[0]}%" + (f" ({pr[1]})" if pr[1] else ""))
                        else:
                            self.prog_text.set(f"Progress {pr[0]}%")
                        continue                     # keep the log readable
                    if "step 2 of 2" in val:
                        self.prog.configure(value=100)
                        self.prog_text.set("Step 2 of 2: review window open (autosave on)")
                    elif "using existing" in val:
                        self.prog.configure(value=100)
                        self.prog_text.set("Step 1 of 2: already processed")
                    low = val.lower()
                    self.write(val, "err" if ("error" in low or "traceback" in low or "failed" in low
                                              or "not shareable" in low) else None)
                elif kind == "done":
                    name, rc = val
                    self.write(f"[{name} finished]\n" if rc == 0 else
                               "[export finished, NOT SHAREABLE: a face was still found in the fill output; see "
                               "the frame list above]\n" if (name == "export" and rc == 3) else
                               f"[{name} exited with code {rc}]\n", None if rc == 0 else "err")
                    self.btns["stop"].configure(state="disabled")
                    self.prog_text.set(f"{name} finished" if rc == 0 else f"{name} stopped (code {rc})")
                    self.refresh()
                elif kind == "devices":
                    self._set_devices(val)
        except queue.Empty:
            pass
        self.root.after(100, self._pump)

    def write(self, text, tag=None):
        try:                                     # also kept in launcher.log next to the settings
            with open(settings_path().with_name("launcher.log"), "a", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            pass
        self.log.configure(state="normal")
        self.log.insert("end", text, (tag,) if tag else ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def close(self):
        save_settings(self.settings())
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.root.destroy()


def main(argv=None):
    ap = argparse.ArgumentParser(description="semi-signals launcher window")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--viewer-args", default="", help="extra arguments for the viewer, e.g. \"--width 1280\"")
    ap.add_argument("--autorun", choices=["play", "annotate", "fetch"], default=None, help=argparse.SUPPRESS)
    ap.add_argument("--list", action="store_true", help="print the sample clips and model status, then exit")
    a = ap.parse_args(argv)
    data_dir, models_dir = Path(a.data_dir).resolve(), Path(a.models_dir).resolve()
    if a.list:
        for c in sample_clips(data_dir):
            print(c)
        print(models_status(models_dir, data_dir))
        return 0
    try:
        import tkinter as tk
    except ImportError:
        print("tkinter is not available in this Python. On Windows reinstall Python with 'tcl/tk and IDLE' "
              "ticked (pyenv-win installs include it); on Linux install python3-tk.\n"
              "You can still run the viewer directly:  python -m src.run_seizure_demo --input <video>")
        return 1
    from src.textdraw import make_dpi_aware
    make_dpi_aware()                 # Windows: sharp text instead of a bitmap-upscaled window (before tk.Tk)
    root = tk.Tk()
    app = Launcher(root, data_dir, models_dir, shlex.split(a.viewer_args, posix=os.name != "nt"))
    if a.autorun:
        root.after(1500, {"play": app.play, "annotate": app.annotate, "fetch": app.fetch}[a.autorun])
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
