"""scripts/fetch_models.py --ghost: flag parsing, licence acknowledgement and sha256 handling (network mocked)."""
import hashlib, importlib.util, json, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("fetch_models_ghost", ROOT / "scripts" / "fetch_models.py")
fm = importlib.util.module_from_spec(spec); sys.modules["fetch_models_ghost"] = fm; spec.loader.exec_module(fm)

FAKE = {n: f"fake {n}".encode() for n in fm.GHOST_FILES}


@pytest.fixture
def mocked(monkeypatch):
    calls = {"download": [], "clone": [], "other": []}
    monkeypatch.setattr(fm, "GHOST_FILES", {n: (u, hashlib.sha256(FAKE[n]).hexdigest(), lic)
                                            for n, (u, _, lic) in fm.GHOST_FILES.items()})

    def download(url, dst):
        calls["download"].append(url); dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(calls.get("payload", {}).get(dst.name, FAKE[dst.name]))
    monkeypatch.setattr(fm, "download", download)
    monkeypatch.setattr(fm, "clone_ghost", lambda dst: calls["clone"].append(dst) or "cloned")
    for step in ("step_pose", "step_detector_a"):
        monkeypatch.setattr(fm, step, lambda *a, _s=step: (calls["other"].append(_s), a[-1].__setitem__(_s, "skipped")))
    return calls


def _run(tmp_path, *extra):
    return fm.main(["--models-dir", str(tmp_path / "models"), "--skip-vsvig", "--skip-data",
                    "--ghost-dir", str(tmp_path / "tp"), *extra])


def test_flags_parse():
    import argparse
    seen = {}
    real = argparse.ArgumentParser.parse_args
    def grab(self, argv=None, ns=None):
        a = real(self, argv, ns); seen.update(vars(a)); raise SystemExit(0)
    argparse.ArgumentParser.parse_args = grab
    try:
        with pytest.raises(SystemExit):
            fm.main(["--ghost", "--accept-ghost-licences", "--ghost-dir", "/x/y"])
    finally:
        argparse.ArgumentParser.parse_args = real
    assert seen["ghost"] and seen["accept_ghost_licences"] and seen["ghost_dir"] == "/x/y"
    with pytest.raises(SystemExit):
        real(argparse.ArgumentParser(), ["--nope"])


def test_urls_are_official_and_pins_full():
    assert fm.GHOST_REPO == "https://github.com/ai-forever/ghost" and len(fm.GHOST_COMMIT) == 40
    for n, (u, h, lic) in fm.GHOST_FILES.items():
        assert u.startswith("https://github.com/ai-forever/ghost/releases/download/") and u.endswith("/" + n)
        assert len(h) == 64 and lic


def test_refuses_without_acknowledgement(tmp_path, mocked, monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert _run(tmp_path, "--ghost") == 2
    assert mocked["download"] == mocked["clone"] == mocked["other"] == []
    assert not (tmp_path / "tp").exists() and not (tmp_path / "models").exists()
    out = capsys.readouterr()
    assert "non-commercial" in out.out and "Apache-2.0" in out.out and "nothing fetched" in out.err


def test_prompt_no_and_yes(tmp_path, mocked, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    a = type("A", (), {"accept_ghost_licences": False})()
    assert fm.ghost_acknowledged(a, ask=lambda _: "") is False
    assert fm.ghost_acknowledged(a, ask=lambda _: "n") is False
    assert fm.ghost_acknowledged(a, ask=lambda _: "y") is True
    monkeypatch.setattr("builtins.input", lambda _: "N")
    assert _run(tmp_path, "--ghost") == 2 and mocked["download"] == []


def test_fetch_ok_layout_and_paths(tmp_path, mocked):
    assert _run(tmp_path, "--ghost", "--accept-ghost-licences") == 0
    wd = tmp_path / "tp" / "weights" / "ghost"
    assert sorted(p.name for p in wd.iterdir()) == sorted(fm.GHOST_FILES)
    assert len(mocked["download"]) == 4 and mocked["clone"] == [(tmp_path / "tp" / "ghost").resolve()]
    cfg = json.loads((tmp_path / "tp" / "ghost_models.json").read_text())
    assert cfg["models"]["id_embedder"].endswith("backbone.pth") and cfg["arcface_model"].endswith("glintr100.onnx")
    assert _run(tmp_path, "--ghost", "--accept-ghost-licences") == 0 and len(mocked["download"]) == 4   # idempotent


def test_sha_mismatch_removed_and_fails(tmp_path, mocked):
    mocked["payload"] = {"backbone.pth": b"tampered"}
    assert _run(tmp_path, "--ghost", "--accept-ghost-licences") == 1
    wd = tmp_path / "tp" / "weights" / "ghost"
    assert not (wd / "backbone.pth").exists() and (wd / "G_unet_2blocks.pth").exists()
    assert not (tmp_path / "tp" / "ghost_models.json").exists()


def test_clone_rejects_other_commit(tmp_path, monkeypatch):
    d = tmp_path / "ghost"; (d / ".git").mkdir(parents=True)
    monkeypatch.setattr(fm.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "0" * 40})())
    with pytest.raises(RuntimeError, match="not 44e58aad8600"):
        fm.clone_ghost(d)
