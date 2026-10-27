"""scripts/fetch_models.py sample-data step, with the network mocked (no real downloads)."""
import hashlib, importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("fetch_models", ROOT / "scripts" / "fetch_models.py")
fm = importlib.util.module_from_spec(spec); sys.modules["fetch_models"] = fm; spec.loader.exec_module(fm)

FAKE = {"Label.xlsx": b"labels", "pat10/Sz2P.mp4": b"clip-a", "pat12/Sz1P.mp4": b"clip-b", "pat03/free.mp4": b"clip-c"}
H = {k: hashlib.sha256(v).hexdigest() for k, v in FAKE.items()}


def _mock(monkeypatch, calls):
    monkeypatch.setattr(fm, "DATA_SHA256", {k: H[k] for k in ("Label.xlsx", "pat10/Sz2P.mp4", "pat12/Sz1P.mp4")})

    def download(url, dst):
        assert url.startswith(fm.DATA_RESOLVE)
        rel = url[len(fm.DATA_RESOLVE):]
        calls.append(rel); dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(FAKE[rel])

    def fetch_json(url):
        assert url == fm.DATA_TREE_API
        return [{"type": "file", "path": "dataset/" + k, "size": len(v),
                 **({"lfs": {"oid": H[k]}} if k.endswith(".mp4") else {})} for k, v in FAKE.items()]
    monkeypatch.setattr(fm, "download", download)
    monkeypatch.setattr(fm, "fetch_json", fetch_json)


def _run(tmp_path, *extra):
    return fm.main(["--models-dir", str(tmp_path / "models"), "--data-dir", str(tmp_path / "data"),
                    "--skip-vsvig", "--detector-a", str(tmp_path / "missing.txt"), *extra])


def test_default_clips_paths_manifest_and_skip(tmp_path, monkeypatch):
    calls = []; _mock(monkeypatch, calls)
    monkeypatch.setattr(fm, "step_pose", lambda a, st: st.__setitem__("pose", "skipped"))
    monkeypatch.setattr(fm, "step_detector_a", lambda a, md, man, st: st.__setitem__("detector_a", "ok"))
    assert _run(tmp_path) == 0
    assert sorted(calls) == ["Label.xlsx", "pat10/Sz2P.mp4", "pat12/Sz1P.mp4"]
    assert (tmp_path / "data" / "Label.xlsx").read_bytes() == b"labels"          # no dataset/ prefix
    assert (tmp_path / "data" / "pat10" / "Sz2P.mp4").exists()
    man = json.loads((tmp_path / "models" / "MANIFEST.json").read_text())
    assert man["data"]["pat12/Sz1P.mp4"]["sha256"] == H["pat12/Sz1P.mp4"]
    assert fm.DATA_REVISION in man["data"]["Label.xlsx"]["source"]
    calls.clear(); _run(tmp_path)
    assert calls == []                                                            # hashes match: nothing fetched
    (tmp_path / "data" / "Label.xlsx").write_bytes(b"tampered")
    assert _run(tmp_path, "--offline") == 1                                       # offline verify flags it
    calls.clear(); _run(tmp_path)
    assert calls == ["Label.xlsx"]                                               # wrong hash: fetched again


def test_offline_and_skip_and_all(tmp_path, monkeypatch, capsys):
    calls = []; _mock(monkeypatch, calls)
    monkeypatch.setattr(fm, "step_pose", lambda a, st: st.__setitem__("pose", "ok"))
    monkeypatch.setattr(fm, "step_detector_a", lambda a, md, man, st: st.__setitem__("detector_a", "ok"))
    assert _run(tmp_path, "--offline") == 1 and calls == []                       # nothing there yet
    assert _run(tmp_path, "--skip-data") == 0 and calls == []
    assert _run(tmp_path, "--clips", "all") == 0
    assert sorted(calls) == sorted(FAKE)
    out = capsys.readouterr().out
    assert "--clips all is 3 videos" in out and "Apache License 2.0" in out
    assert _run(tmp_path, "--offline", "--clips", "all") == 0                     # pins from the manifest
    assert _run(tmp_path, "--clips", "pat99/Sz1P") == 1                           # not in the dataset


def test_clip_list():
    assert fm.clip_list("pat10/Sz2P, pat12\\Sz1P.mp4") == ["pat10/Sz2P.mp4", "pat12/Sz1P.mp4"]
    assert fm.clip_list("all") is None and fm.clip_list("none") == []
