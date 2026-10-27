"""Launcher helpers (no window): clip list, the command lines it builds, settings, proxy, progress."""
from pathlib import Path

from src import launcher as L


def _data(tmp_path):
    d = tmp_path / "data"
    for rel in ("pat10/Sz1P.mp4", "pat10/Sz2P.mp4", "pat10/Sz2P_semisignals.mp4", "other.mp4"):
        (d / rel).parent.mkdir(parents=True, exist_ok=True); (d / rel).write_bytes(b"x")
    (d / "Label.xlsx").write_bytes(b"x")
    return d


def test_sample_clips_and_dataset_check(tmp_path):
    d = _data(tmp_path)
    assert [p.name for p in L.sample_clips(d)] == ["Sz1P.mp4", "Sz2P.mp4"]
    assert L.is_dataset_clip(d / "pat10" / "Sz1P.mp4", d)
    assert L.is_dataset_clip(d / "dataset" / "pat10" / "Sz2P.mp4", d)
    assert not L.is_dataset_clip(d / "other.mp4", d)
    assert not L.is_dataset_clip(tmp_path / "mine.mp4", d)
    assert L.sample_clips(tmp_path / "nope") == []


def test_commands_parse_in_the_tools(tmp_path):
    from src import review_video, run_seizure_demo
    d, m = _data(tmp_path), tmp_path / "models"
    s = dict(video=str(d / "pat10" / "Sz1P.mp4"), labels=str(d / "Label.xlsx"), device="GPU", loop=True,
             hands=False, explain=True, vsvig=False, annotator="AL", pat_id="p10", consent="")
    v = L.viewer_cmd(s, d, m, ["--max-loops", "2"])
    assert v[1:4] == ["-u", "-m", "src.run_seizure_demo"]
    a = run_seizure_demo.parse_args(v[4:])
    assert a.loop and a.no_hands and a.device == "gpu" and a.labels and a.vsvig == "" and a.max_loops == 2
    own = dict(s, video=str(d / "other.mp4"))
    assert run_seizure_demo.parse_args(L.viewer_cmd(own, d, m)[4:]).labels is None     # labels only for dataset clips
    r = review_video.parse_args(L.annotate_cmd(s, d, m)[4:])
    assert r.annotator == "AL" and r.pat_id == "p10" and r.consent == "public" and r.headless_process
    assert review_video.parse_args(L.annotate_cmd(own, d, m)[4:]).consent == "self"
    assert "--skip-vsvig" in L.fetch_cmd(d, m, vsvig=False) and "--skip-vsvig" not in L.fetch_cmd(d, m, vsvig=True)


def test_settings_roundtrip_and_proxy(tmp_path):
    p = tmp_path / "s" / "launcher.json"
    assert L.load_settings(p) == {}
    L.save_settings({"annotator": "AL", "loop": True}, p)
    assert L.load_settings(p) == {"annotator": "AL", "loop": True}
    p.write_text("{broken")
    assert L.load_settings(p) == {}
    env = L.child_env("http://proxy:8080")
    assert env["HTTPS_PROXY"] == env["HTTP_PROXY"] == "http://proxy:8080" and env["PYTHONUNBUFFERED"] == "1"


def test_progress_lines_and_status(tmp_path):
    assert L.parse_progress("[viewer] progress 42% (25 of 60 s)") == (42, "25 of 60 s")
    assert L.parse_progress("[viewer] progress 100%") == (100, "")
    assert L.parse_progress("[review] saved") is None
    st = L.models_status(tmp_path / "models", _data(tmp_path))
    assert "Detector A missing" in st and "2 clip(s)" in st and "Label.xlsx ok" in st
