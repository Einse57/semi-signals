"""Face de-identification in the UI (src/deid_options.py, src/deid_ui.py, launcher and annotation panel)
and the server-side checks in src/segment_export.py (the process the UI starts)."""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from src import deid_options as O
from src import deid_swap as W
from src import launcher as L
from src import review_video as R
from src import segment_export as X
from src import segment_store as S

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models" / "face_landmarker.task"
IMG = Path("/workspace/vsvig/facetest/business-person.png")


def _ring(cx, cy, r=20):
    a = np.linspace(0, 2 * np.pi, 478, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a), np.zeros(478)], 1).astype(np.float32)


def _video(tmp_path, frames=3, size=(64, 48)):
    v = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for _ in range(frames):
        w.write(np.full((size[1], size[0], 3), 90, np.uint8))
    w.release()
    return v


def _base(tmp_path, v):
    return ["--input", str(v), "--whole", "--out", str(tmp_path / "o"), "--raw-root", str(tmp_path / "r")]


def test_ui_offers_only_original_fill_ghost():
    assert set(O.METHODS) == {"fill", "ghost"} and O.CHOICES == ("original", "fill", "ghost")
    assert O.DEFAULT_METHOD == "fill" and O.DEFAULT_SWAP == "ghost" == W.DEFAULT_BACKEND
    assert O.MASK_METHODS == ("fill",) and O.SWAP_METHODS == ("ghost",)
    for gone in ("mosaic8", "mosaic16", "gauss", "blur", "warp", "liveportrait"):
        with pytest.raises(ValueError):
            O.check_choice(gone)
        assert "not offered" in O.validate_request(gone)[0]
        with pytest.raises(ValueError):
            O.export_cmd("v.mp4", gone, python="py")
    assert O.check_choice("original") == "original"
    # the phase 4 methods stay in the measurement code
    assert set(X.BASELINE_STYLES) == {"mosaic8", "mosaic16", "gauss"} and {"warp", "liveportrait"} <= set(W._REGISTRY)


def test_labels():
    assert O.label("ghost") == "reduced risk, not de-identified" == W.DEID_STATUS
    assert O.label("fill") == "shareable fallback (default)"
    assert "not de-identified" in O.label("original")
    assert [m for m in O.METHODS if O.is_shareable(m)] == ["fill"] and not O.is_shareable("original")
    assert {m for m in O.CHOICES if O.needs_ack(m)} == {"ghost"} and "ghost" in W.RESEARCH_ONLY
    assert "research use only" in O.label_text("ghost") and "only method" in O.label_text("fill")


def test_validate_request_gate_and_shareable():
    assert O.validate_request("fill", shareable=True) == [] and O.validate_request("fill") == []
    assert "only fill" in O.validate_request("ghost", research_ack=True, shareable=True)[0]
    assert "research use only" in O.validate_request("ghost")[0]
    assert O.validate_request("ghost", research_ack=True) == []
    assert O.validate_request("ghost", research_ack=True, swap_config_backend="warp")   # config / choice mismatch
    assert "viewing only" in O.validate_request("original")[0]
    assert "not offered" in O.validate_request("swirl")[0]
    assert O.swap_overrides("ghost", False) == {"backend": "ghost", "research_use_acknowledged": False}


def test_missing_weights_message_points_to_official_sources(tmp_path, monkeypatch):
    monkeypatch.delenv(W.ENV_CONFIG, raising=False)
    g = O.missing_weights_message("ghost", O.weights_problems("ghost", None))
    assert "no swap configuration" in g and O.GHOST_DOWNLOAD in g and "download_models.sh" in g
    assert "github.com/ai-forever/ghost" in g and "does not redistribute" in g and O.LICENCES_DOC in g
    assert "Fill" in g and "warp" not in g.lower() and "liveportrait" not in g.lower()
    assert O.FACE_MODEL_HINT in O.missing_weights_message("fill", O.face_model_problems(tmp_path))
    assert (REPO / "scripts/fetch_models.py").is_file() and (REPO / O.LICENCES_DOC).is_file()
    cfg = tmp_path / "swap.json"
    cfg.write_text(json.dumps(dict(backend="ghost", licence_reviewed=True,
                                   models=dict(code=str(tmp_path / "ghost"), generator=str(tmp_path / "G.pth"),
                                               id_embedder=str(tmp_path / "b.pth")))))
    probs = O.weights_problems("ghost", cfg)
    assert any("G.pth" in p for p in probs) and not any("research_use" in p for p in probs)


def test_export_cmd_parses_and_passes_the_gate(tmp_path):
    v = tmp_path / "clip.mp4"
    c = O.export_cmd(v, "ghost", python="py", swap_config="s.json", research_ack=True)
    a = X.parse_args(c[4:])
    assert a.deid_method == "ghost" and a.research_use_acknowledged and a.swap_config == "s.json" and a.whole
    assert not a.shareable and a.mode == "face"
    a = X.parse_args(O.export_cmd(v, "fill", python="py", shareable=True)[4:])
    assert a.deid_method == "fill" and a.shareable and a.swap_config is None
    a = X.parse_args(O.export_cmd(v, "fill", python="py", research_ack=True)[4:])
    assert not a.research_use_acknowledged                                     # gate flag only for ghost
    out, raw = O.default_dirs()
    assert REPO not in out.parents and REPO not in raw.parents
    s = dict(video=str(v), annot_mode="both")
    eaf = v.with_suffix(".eaf"); eaf.write_text("<x/>")
    a = X.parse_args(L.deid_export_cmd(s, dict(deid_method="ghost", research_ack=True, swap_config="c.json"))[4:])
    assert a.deid_method == "ghost" and a.annotations == str(eaf) and a.mode == "both" and a.research_use_acknowledged


def test_server_rejects_without_writing(tmp_path, monkeypatch):
    monkeypatch.delenv(W.ENV_CONFIG, raising=False)
    v = _video(tmp_path)
    for extra, msg in ((["--deid-method", "ghost"], "research use only"),
                       (["--deid-method", "ghost", "--research-use-acknowledged", "--shareable"], "only fill"),
                       (["--deid-method", "fill", "--blur-in-export"], "only fill or ghost"),
                       (["--deid", "fill,blur", "--shareable"], "only fill"),
                       (["--deid", "swap", "--research-use-acknowledged"], "goes with")):
        with pytest.raises(SystemExit, match=msg):
            X.main(_base(tmp_path, v) + extra)
        assert not (tmp_path / "o").exists()
    for gone in ("swirl", "original", "mosaic8", "mosaic16", "gauss", "blur", "warp", "liveportrait"):
        with pytest.raises(SystemExit):                                        # argparse: not on the UI path
            X.parse_args(_base(tmp_path, v) + ["--deid-method", gone])
        assert not (tmp_path / "o").exists()
    with pytest.raises(SystemExit, match="does not redistribute"):            # gate passed, weights missing
        X.main(_base(tmp_path, v) + ["--deid-method", "ghost", "--research-use-acknowledged"])
    cfg = tmp_path / "warp.json"
    cfg.write_text(json.dumps(dict(backend="warp")))
    with pytest.raises(SystemExit, match="names backend 'warp'"):
        X.main(_base(tmp_path, v) + ["--deid-method", "ghost", "--research-use-acknowledged", "--swap-config", str(cfg)])
    assert not (tmp_path / "o").exists()


def test_ui_ack_is_passed_to_backend(tmp_path, monkeypatch):
    files = {k: tmp_path / f"{k}.bin" for k in W.REQUIRED["ghost"]}
    for f in files.values():
        f.write_bytes(b"x")
    (tmp_path / "d.png").write_bytes(b"x")
    cfg = tmp_path / "swap.json"                  # the file says acknowledged, the UI choice decides
    cfg.write_text(json.dumps(dict(licence_reviewed=True, research_use_acknowledged=True,
                                   models={k: str(v) for k, v in files.items()}, donor=str(tmp_path / "d.png"),
                                   donor_licence="synthetic, test")))
    seen = {}
    monkeypatch.setitem(W._REGISTRY, "ghost", lambda c: seen.setdefault("cfg", c))
    W.make_swapper(str(cfg), O.swap_overrides("ghost", True))
    assert seen["cfg"]["research_use_acknowledged"] is True and seen["cfg"]["backend"] == "ghost"
    with pytest.raises(W.SwapNotConfigured, match="research_use_acknowledged"):
        W.make_swapper(str(cfg), O.swap_overrides("ghost", False))


def test_preview_image_before_after():
    rng = np.random.default_rng(1)
    f = rng.integers(0, 255, (120, 160, 3), np.uint8)
    img = O.preview_image(f, "fill", _ring(80, 60), width=160)
    assert img.shape == (120 + 44, 2 * 160 + 8, 3)
    assert np.array_equal(img[:120, :160], f) and not np.array_equal(img[:120, 168:], f)
    img = O.preview_image(f, "original", None, width=160)                     # original: after == before
    assert np.array_equal(img[:120, 168:], img[:120, :160])
    with pytest.raises(ValueError):
        O.preview_image(f, "mosaic8", _ring(80, 60), width=160)
    img = O.preview_image(f, "ghost", _ring(80, 60), swapper=None, width=160)   # no swapper: fill shown
    assert np.ptp(img[:120, 168:].reshape(-1, 3), 0).max() == 0
    with pytest.raises(ValueError, match="research use only"):
        O.preview_from_video("x.mp4", "ghost")


def test_annotation_panel_has_deid_button_in_face_modes():
    assert ("De-ID export...", "deid") in R.panel_buttons("face") and ("De-ID export...", "deid") in R.panel_buttons("both")
    assert ("De-ID export...", "deid") not in R.panel_buttons("body")


@pytest.mark.skipif(not (MODEL.exists() and IMG.exists()), reason="model file or test image not present")
def test_method_exports_labels_and_locations(tmp_path):
    img = cv2.resize(cv2.imread(str(IMG)), (320, 226))
    video = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 226))
    for _ in range(3):
        w.write(img)
    w.release()
    out, raw = tmp_path / "fill" / "deid", tmp_path / "fill" / "raw"
    assert X.main(["--input", str(video), "--whole", "--out", str(out), "--raw-root", str(raw),
                   "--deid-method", "fill", "--no-bystanders"]) == 0
    j = S.read_json(next(out.rglob("*desc-fill_video.json")))
    assert j["DeidentificationStatus"] == O.label("fill") and j["Shareable"] is True and j["Deidentified"] is True
    assert "NotForSharing" not in j and not list(raw.rglob("*desc-fill_video.mp4"))
    assert list(out.rglob("*desc-points_video.mp4"))
    assert X.main(["--input", str(video), "--whole", "--out", str(tmp_path / "s"), "--raw-root", str(tmp_path / "sr"),
                   "--deid-method", "fill", "--shareable", "--no-bystanders"]) == 0


def test_selector_widget_defaults_and_states():
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("no display")
    import tkinter as tk
    from src import deid_ui as U
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    try:
        sel = U.DeidSelector(root, {}, REPO / "models")
        assert set(sel.buttons) == {"original", "fill", "ghost"}                # nothing else in the UI
        assert sel.method() == "fill" and str(sel.share_cb.cget("state")) == "normal"
        assert str(sel.ack_cb.cget("state")) == "disabled" and str(sel.cfg_entry.cget("state")) == "disabled"
        assert "shareable fallback (default)" in sel.status.cget("text")
        sel.choice.set("ghost"); sel.changed()
        assert sel.method() == "ghost" and str(sel.ack_cb.cget("state")) == "normal"
        assert str(sel.share_cb.cget("state")) == "disabled" and not sel.share.get()
        assert "reduced risk, not de-identified" in sel.status.cget("text")
        assert any("research use only" in p for p in sel.problems())
        assert "download_models.sh" in sel.weights.cget("text")                # no swap config: where to fetch
        sel.ack.set(True); sel.changed()
        assert not any("research use only" in p for p in sel.problems()) and sel.state()["research_ack"]
        sel.choice.set("original"); sel.changed()
        assert not sel.state()["research_ack"] and not sel.state()["shareable"]
        assert any("viewing only" in p for p in sel.problems()) and sel.weights.cget("text") == ""
        for old in ("warp", "mosaic16", "liveportrait"):                          # older saved settings -> fill
            assert U.DeidSelector(root, {"deid_method": old}).method() == "fill"
        assert not U.DeidSelector(root, {"deid_method": "ghost"}).ack.get()       # gate never restored
    finally:
        root.destroy()
