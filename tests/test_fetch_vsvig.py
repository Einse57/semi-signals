"""scripts/fetch_models.py VSViG step: without torch the conversion is skipped, not failed."""
import importlib.util, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("fetch_models_vsvig", ROOT / "scripts" / "fetch_models.py")
fm = importlib.util.module_from_spec(spec); sys.modules["fetch_models_vsvig"] = fm; spec.loader.exec_module(fm)


def test_vsvig_skipped_without_torch(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "ok_file", lambda p, want: True)
    real = importlib.util.find_spec
    monkeypatch.setattr(fm.importlib.util, "find_spec", lambda name, *a: None if name == "torch" else real(name, *a))
    monkeypatch.setattr(fm.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no conversion")))
    a = type("A", (), {"force": False, "offline": False})()
    status = {}
    fm.step_vsvig(a, tmp_path, fm.Manifest(tmp_path / "MANIFEST.json"), status)
    assert status["vsvig"].startswith("skipped")
