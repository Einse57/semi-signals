"""Face view / de-identification selector (tkinter) for the launcher (face mode) and the annotation panel.

Offers only View: Original, and De-identify: Fill (default, the only shareable export) or GHOST.
Rules, names and labels come from src/deid_options.py; src/segment_export.py checks them again.
"""
from __future__ import annotations

import sys
from pathlib import Path

from src import deid_options as O

ACK_TEXT = ("I acknowledge that GHOST is research use only: I fetched its code and weights myself from the "
            "official sources under their terms, and the output stays on this computer.")
COLOURS = {O.SHAREABLE: "#060", O.ORIGINAL_LABEL: "#555"}


def label_colour(method: str) -> str:
    return COLOURS.get(O.label(method), "#a60")


class DeidSelector:
    """One choice: Original (view only), Fill (default) or GHOST."""

    def __init__(self, parent, settings: dict | None = None, models_dir: Path | None = None, on_change=None):
        import tkinter as tk
        from tkinter import ttk
        st = settings or {}
        self.models_dir, self.on_change, self.tk = models_dir, on_change, tk
        self.frame = f = ttk.LabelFrame(parent, text="Face view / de-identification (face mode)", padding=8)
        m0 = st.get("deid_method", O.DEFAULT_METHOD)
        m0 = m0 if m0 in O.METHODS else O.DEFAULT_METHOD        # older settings (mosaic, warp, ...) -> fill
        self.choice = tk.StringVar(value=m0)
        self.ack = tk.BooleanVar(value=False)              # never restored from settings: ticked per session
        self.share = tk.BooleanVar(value=False)
        self.swap_config = tk.StringVar(value=st.get("swap_config", ""))
        ttk.Label(f, text="View").grid(row=0, column=0, sticky="w")
        self.buttons = {}
        b = ttk.Radiobutton(f, text="Original", value=O.ORIGINAL, variable=self.choice, command=self.changed)
        b.grid(row=0, column=1, sticky="w", padx=(0, 10)); self.buttons[O.ORIGINAL] = b
        ttk.Label(f, text="De-identify").grid(row=1, column=0, sticky="w", pady=(4, 0))
        for i, k in enumerate(O.METHODS):
            txt = O.title(k) + (" (default)" if k == O.DEFAULT_METHOD else "")
            b = ttk.Radiobutton(f, text=txt, value=k, variable=self.choice, command=self.changed)
            b.grid(row=1, column=1 + i, sticky="w", padx=(0, 10), pady=(4, 0)); self.buttons[k] = b
        self.status = ttk.Label(f, text="", wraplength=720, justify="left")
        self.status.grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Label(f, text="Swap config").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.cfg_entry = ttk.Entry(f, textvariable=self.swap_config, width=52)
        self.cfg_entry.grid(row=3, column=1, columnspan=2, sticky="ew", pady=(4, 0))
        self.cfg_btn = ttk.Button(f, text="Browse...", command=self.browse_config)
        self.cfg_btn.grid(row=3, column=3, sticky="w", padx=6, pady=(4, 0))
        ack = ttk.Frame(f); ack.grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self.ack_cb = ttk.Checkbutton(ack, text="Research use acknowledged (GHOST)", variable=self.ack,
                                      command=self.changed)
        self.ack_cb.grid(row=0, column=0, sticky="nw")
        ttk.Label(ack, text=ACK_TEXT, foreground="#666", wraplength=520, justify="left").grid(
            row=0, column=1, sticky="w", padx=(8, 0))
        self.share_cb = ttk.Checkbutton(f, text="Export for sharing (fill only)", variable=self.share, command=self.changed)
        self.share_cb.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.weights = ttk.Label(f, text="", foreground="#a33", wraplength=720, justify="left")
        self.weights.grid(row=6, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self.widgets = list(self.buttons.values()) + [self.cfg_entry, self.cfg_btn, self.ack_cb, self.share_cb]
        self.swap_config.trace_add("write", lambda *_: self.changed())
        self.changed()

    # state
    def method(self) -> str:
        return O.check_choice(self.choice.get())

    def state(self) -> dict:
        m = self.method()
        return dict(deid_method=m, swap_config=self.swap_config.get().strip(),
                    research_ack=bool(self.ack.get()) and O.needs_ack(m),
                    shareable=bool(self.share.get()) and O.is_shareable(m))

    def problems(self) -> list[str]:
        """Why an export cannot start now ([] = ready)."""
        s = self.state()
        m = s["deid_method"]
        err = O.validate_request(m, research_ack=s["research_ack"], shareable=s["shareable"])
        if m == O.ORIGINAL:
            return err
        missing = O.weights_problems(m, s["swap_config"] or None, self.models_dir)
        if missing:
            err.append(O.missing_weights_message(m, missing))
        if s["shareable"]:
            err += ["Export for sharing: " + p for p in O.fill_check_problems(self.models_dir)]
        return err

    def notes(self) -> list[str]:
        """Warnings that do not stop an export (fill without YuNet: landmark-only, not shareable)."""
        s = self.state()
        if s["deid_method"] != "fill" or s["shareable"]:
            return []
        return O.fill_check_problems(self.models_dir)

    def changed(self):
        m = self.method()
        swap = m in O.SWAP_METHODS
        for w in (self.cfg_entry, self.cfg_btn):
            w.configure(state="normal" if swap else "disabled")
        self.ack_cb.configure(state="normal" if O.needs_ack(m) else "disabled")
        if not O.is_shareable(m):
            self.share.set(False)
        self.share_cb.configure(state="normal" if O.is_shareable(m) else "disabled")
        self.status.configure(text=f"{O.title(m)}: {O.label_text(m)}", foreground=label_colour(m))
        self.weights.configure(text="\n".join(p for p in self.problems() + self.notes() if m != O.ORIGINAL))
        if self.on_change:
            self.on_change()

    def set_enabled(self, on: bool):
        for w in self.widgets:
            w.configure(state="normal" if on else "disabled")
        if on:
            self.changed()

    def browse_config(self):
        from tkinter import filedialog
        p = filedialog.askopenfilename(title="Swap configuration JSON", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if p:
            self.swap_config.set(str(Path(p)))


def show_preview(parent, video, sel: DeidSelector, t: float | None = None):
    """Before/after of one frame in a small window; returns the error text instead when it cannot run."""
    import base64
    import tkinter as tk
    from tkinter import ttk
    import cv2
    s = sel.state()
    try:
        img = O.preview_from_video(video, s["deid_method"], t, swap_config=s["swap_config"] or None,
                                   research_ack=s["research_ack"], models_dir=sel.models_dir)
    except (ValueError, OSError, RuntimeError) as e:
        return str(e)
    ok, png = cv2.imencode(".png", img)
    w = tk.Toplevel(parent); w.title(f"Preview: {s['deid_method']} ({O.label(s['deid_method'])})")
    ph = tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode("ascii"))
    lab = ttk.Label(w, image=ph); lab.image = ph; lab.pack(padx=6, pady=6)
    ttk.Label(w, text="Preview only: nothing is saved.", foreground="#666").pack(pady=(0, 6))
    return None


def ask_export(parent, video, annotations=None, settings=None, models_dir=None, mode="face", face_preprocess="none"):
    """Modal dialog for the annotation panel: choose a method, then return the export command (or None)."""
    import tkinter as tk
    from tkinter import ttk
    d = tk.Toplevel(parent); d.title("De-identified export"); d.resizable(False, False)
    sel = DeidSelector(d, settings, models_dir)
    sel.frame.pack(padx=10, pady=10, fill="x")
    msg = ttk.Label(d, text="", foreground="#a33", wraplength=720); msg.pack(padx=10, anchor="w")
    res = {"cmd": None}

    def go():
        p = sel.problems()
        if p:
            msg.configure(text="\n".join(p)); return
        s = sel.state()
        res["cmd"] = O.export_cmd(video, s["deid_method"], python=sys.executable, annotations=annotations,
                                  swap_config=s["swap_config"] or None, research_ack=s["research_ack"],
                                  shareable=s["shareable"], mode=mode, face_preprocess=face_preprocess)
        d.destroy()

    def prev():
        e = show_preview(d, video, sel)
        msg.configure(text=e or "")

    b = ttk.Frame(d); b.pack(pady=10)
    ttk.Button(b, text="Preview (before/after)", command=prev).pack(side="left", padx=4)
    ttk.Button(b, text="Export segments", command=go).pack(side="left", padx=4)
    ttk.Button(b, text="Cancel", command=d.destroy).pack(side="left", padx=4)
    d.grab_set(); d.wait_window()
    return res["cmd"]
