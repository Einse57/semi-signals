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


def test_every_method_listed_with_defaults():
    assert set(O.METHODS) == {"fill", "mosaic8", "mosaic16", "gauss", "warp", "ghost", "liveportrait"}
    assert O.DEFAULT_METHOD == "fill" and O.DEFAULT_SWAP == "ghost" == W.DEFAULT_BACKEND
    assert O.MASK_METHODS == ("fill", "mosaic8", "mosaic16", "gauss")
    assert O.SWAP_METHODS[0] == "ghost" and set(O.SWAP_METHODS) == {"ghost", "warp", "liveportrait"}
    assert O.resolve("swap") == "ghost" and O.resolve("swap", None) == "ghost" and O.resolve("swap", "bogus") == "ghost"
    assert O.resolve("swap", "warp") == "warp" and O.resolve("fill") == "fill"
    with pytest.raises(ValueError):
        O.resolve("blurry")
    assert O.family_of("liveportrait") == "swap" and O.family_of("mosaic8") == "mosaic8"


def test_labels():
    for m in ("warp", "ghost", "liveportrait", "mosaic8"):
        assert O.label(m) == "reduced risk, not de-identified" == W.DEID_STATUS
    assert O.label("mosaic16") == "not de-identifying" and "provisional" in O.label_text("mosaic16")
    assert O.label("gauss") == "local review only"
    assert O.label("fill") == "shareable fallback (default)"
    assert [m for m in O.METHODS if O.is_shareable(m)] == ["fill"]
    assert O.is_local_only("gauss") and O.is_local_only("mosaic16") and not O.is_local_only("fill")
    assert {m for m in O.METHODS if O.needs_ack(m)} == set(W.RESEARCH_ONLY) == {"ghost", "liveportrait"}


def test_validate_request_gate_and_shareable():
    assert O.validate_request("fill", shareable=True) == []
    assert O.validate_request("gauss", shareable=True) and "only fill" in O.validate_request("gauss", shareable=True)[0]
    for m in ("mosaic8", "mosaic16", "warp", "ghost", "liveportrait"):
        assert O.validate_request(m, research_ack=True, shareable=True)
    assert "research use only" in O.validate_request("ghost")[0]
    assert O.validate_request("ghost", research_ack=True) == []
    assert O.validate_request("liveportrait") and O.validate_request("liveportrait", research_ack=True) == []
    assert O.validate_request("warp") == []                                    # no gate for warp
    assert O.validate_request("warp", swap_config_backend="ghost")             # config / choice mismatch
    assert "unknown" in O.validate_request("swirl")[0]
    assert O.swap_overrides("ghost", False) == {"backend": "ghost", "research_use_acknowledged": False}
    assert O.swap_overrides("warp", True) == {"backend": "warp"}


def test_missing_weights_message_points_to_official_sources(tmp_path, monkeypatch):
    monkeypatch.delenv(W.ENV_CONFIG, raising=False)
    g = O.missing_weights_message("ghost", O.weights_problems("ghost", None))
    assert "no swap configuration" in g and "download_models.sh" in g and "github.com/ai-forever/ghost" in g
    assert "does not redistribute" in g and O.LICENCES_DOC in g and "Fill" in g
    lp = O.missing_weights_message("liveportrait", ["x"])
    assert "huggingface.co/KlingTeam/LivePortrait" in lp and "github.com/KlingTeam/LivePortrait" in lp
    assert O.WARP_HINT in O.missing_weights_message("warp", ["x"])
    assert O.FACE_MODEL_HINT in O.missing_weights_message("fill", O.face_model_problems(tmp_path))
    for script in ("scripts/fetch_models.py",):                                # the in-repo script named exists
        assert (REPO / script).is_file()
    assert (REPO / O.LICENCES_DOC).is_file()
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
    a = X.parse_args(O.export_cmd(v, "warp", python="py", research_ack=True)[4:])
    assert not a.research_use_acknowledged                                     # gate flag only for ghost/LP
    out, raw = O.default_dirs()
    assert REPO not in out.parents and REPO not in raw.parents
    s = dict(video=str(v), annot_mode="both")
    eaf = v.with_suffix(".eaf"); eaf.write_text("<x/>")
    a = X.parse_args(L.deid_export_cmd(s, dict(deid_method="liveportrait", research_ack=True, swap_config="c.json"))[4:])
    assert a.deid_method == "liveportrait" and a.annotations == str(eaf) and a.mode == "both" and a.research_use_acknowledged


def test_server_rejects_without_writing(tmp_path, monkeypatch):
    monkeypatch.delenv(W.ENV_CONFIG, raising=False)
    v = _video(tmp_path)
    for extra, msg in ((["--deid-method", "ghost"], "research use only"),
                       (["--deid-method", "liveportrait"], "research use only"),
                       (["--deid-method", "gauss", "--shareable"], "only fill"),
                       (["--deid-method", "mosaic16", "--shareable"], "only fill"),
                       (["--deid-method", "ghost", "--research-use-acknowledged", "--shareable"], "only fill"),
                       (["--deid-method", "gauss", "--blur-in-export"], "local-only"),
                       (["--deid", "fill,blur", "--shareable"], "only fill"),
                       (["--deid", "swap", "--research-use-acknowledged"], "goes with")):
        with pytest.raises(SystemExit, match=msg):
            X.main(_base(tmp_path, v) + extra)
        assert not (tmp_path / "o").exists()
    with pytest.raises(SystemExit):                                            # argparse: unknown method
        X.parse_args(_base(tmp_path, v) + ["--deid-method", "swirl"])
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
    for m in O.MASK_METHODS:
        img = O.preview_image(f, m, _ring(80, 60), width=160)
        assert img.shape == (120 + 44, 2 * 160 + 8, 3)
        assert np.array_equal(img[:120, :160], f) and not np.array_equal(img[:120, 168:], f)
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
    for m in O.MASK_METHODS:
        out, raw = tmp_path / m / "deid", tmp_path / m / "raw"
        assert X.main(["--input", str(video), "--whole", "--out", str(out), "--raw-root", str(raw),
                       "--deid-method", m, "--no-bystanders"]) == 0
        where = raw if O.is_local_only(m) else out
        j = S.read_json(next(where.rglob(f"*desc-{m}_video.json")))
        assert j["DeidentificationStatus"] == O.label(m) and j["Shareable"] is O.is_shareable(m)
        assert j["Deidentified"] is (m == "fill")
        assert (j.get("NotForSharing") is True) == O.is_local_only(m)
        assert not list((out if O.is_local_only(m) else raw).rglob(f"*desc-{m}_video.mp4"))
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
        assert sel.method() == "fill" and str(sel.share_cb.cget("state")) == "normal"
        assert str(sel.ack_cb.cget("state")) == "disabled" and str(sel.swap_btns[0].cget("state")) == "disabled"
        assert "shareable fallback (default)" in sel.status.cget("text")
        sel.family.set("swap"); sel.changed()
        assert sel.method() == "ghost" and str(sel.ack_cb.cget("state")) == "normal"
        assert str(sel.share_cb.cget("state")) == "disabled" and not sel.share.get()
        assert "reduced risk, not de-identified" in sel.status.cget("text")
        assert any("research use only" in p for p in sel.problems())
        sel.ack.set(True); sel.changed()
        assert not any("research use only" in p for p in sel.problems()) and sel.state()["research_ack"]
        sel.family.set("gauss"); sel.changed()
        assert "local review only" in sel.status.cget("text") and not sel.state()["research_ack"]
        sel.family.set("mosaic16"); sel.changed()
        assert "not de-identifying" in sel.status.cget("text")
        assert U.DeidSelector(root, {"deid_method": "warp"}).method() == "warp"
    finally:
        root.destroy()
