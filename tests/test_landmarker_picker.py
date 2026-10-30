"""Landmarker picker in the de-ID UI: OpenVINO device list, default, pass-through to the export,
settings and sidecar round trip, CPU fallback with a note (src/deid_options.py, src/deid_ui.py,
src/face_estimator.open_face_estimator, src/segment_export.py)."""
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from src import deid_options as O
from src import face_estimator as F
from src import launcher as L
from src import segment_export as X
from src import segment_store as S

REPO = Path(__file__).resolve().parent.parent


def _fake_openvino(monkeypatch, devices):
    m = types.ModuleType("openvino")
    m.Core = lambda: types.SimpleNamespace(available_devices=list(devices))
    monkeypatch.setitem(sys.modules, "openvino", m)


@pytest.mark.parametrize("found, offered", [
    (["CPU", "GPU.0", "GPU.1", "NPU"], ["CPU", "GPU", "NPU", "AUTO"]),
    (["GPU", "CPU"], ["CPU", "GPU", "AUTO"]),
    (["CPU"], ["CPU", "AUTO"]),
    (["NPU"], ["NPU", "AUTO"]),
    (["HETERO", "GNA"], []),
    ([], []),
])
def test_device_list_is_what_openvino_finds(monkeypatch, found, offered):
    _fake_openvino(monkeypatch, found)
    assert O.openvino_devices() == offered
    assert O.openvino_devices(found) == offered


def test_device_list_without_openvino(monkeypatch):
    monkeypatch.setitem(sys.modules, "openvino", None)          # import fails
    assert O.openvino_devices() == []


def test_default_is_mediapipe_cpu():
    assert O.DEFAULT_LANDMARKER == ("mediapipe", "CPU")
    assert O.landmarker_choice({}, ["CPU", "GPU", "AUTO"]) == ("mediapipe", "CPU")
    assert O.landmarker_choice(None, []) == ("mediapipe", "CPU")
    c = O.export_cmd("v.mp4", "fill", python="py")
    a = X.parse_args(c[4:])
    assert a.face_backend == "mediapipe" and a.face_device == "CPU"
    # saved choices checked against this computer
    assert O.landmarker_choice(dict(face_backend="openvino", face_device="NPU"), ["CPU", "AUTO"]) == ("openvino", "CPU")
    assert O.landmarker_choice(dict(face_backend="openvino", face_device="GPU"), []) == ("mediapipe", "CPU")
    assert O.landmarker_choice(dict(face_backend="onnx"), ["CPU"]) == ("mediapipe", "CPU")
    assert O.landmarker_choice(dict(face_backend="mediapipe", face_device="GPU"), ["GPU"]) == ("mediapipe", "CPU")
    assert O.landmarker_devices("mediapipe", ["CPU", "GPU", "AUTO"]) == ["CPU"]


def test_choice_is_passed_to_the_export(tmp_path):
    v = tmp_path / "clip.mp4"
    a = X.parse_args(O.export_cmd(v, "fill", python="py", face_backend="openvino", face_device="GPU")[4:])
    assert a.face_backend == "openvino" and a.face_device == "GPU"
    assert X.face_backend_kwargs(a) == dict(backend="openvino", device="GPU")
    a = X.parse_args(O.export_cmd(v, "fill", python="py", face_backend="mediapipe", face_device="GPU")[4:])
    assert a.face_device == "CPU"                                  # MediaPipe runs on CPU only
    d = dict(deid_method="ghost", research_ack=True, swap_config="c.json", face_backend="openvino", face_device="AUTO")
    a = X.parse_args(L.deid_export_cmd(dict(video=str(v), annot_mode="face"), d)[4:])
    assert a.face_backend == "openvino" and a.face_device == "AUTO" and a.deid_method == "ghost"
    a = X.parse_args(L.deid_export_cmd(dict(video=str(v)), dict(deid_method="fill"))[4:])
    assert (a.face_backend, a.face_device) == ("mediapipe", "CPU")  # older state without the picker keys


def test_settings_round_trip(tmp_path):
    from src.deid_ui import landmarker_settings
    p = tmp_path / "launcher.json"
    st = dict(deid_method="fill", swap_config="", research_ack=False, shareable=False,
              face_backend="openvino", face_device="NPU", face_note="")
    L.save_settings(dict(dict(video="x.mp4", deid_method="fill"), **landmarker_settings(st)), p)
    back = L.load_settings(p)
    assert back["face_backend"] == "openvino" and back["face_device"] == "NPU" and back["deid_method"] == "fill"
    assert "face_note" not in back
    assert O.landmarker_choice(back, ["CPU", "GPU", "NPU", "AUTO"]) == ("openvino", "NPU")


class _FakeEst:
    """FaceEstimator stand-in: OpenVINO GPU fails to compile, everything else loads."""
    made = []

    def __init__(self, model_path=None, backend="mediapipe", device="CPU", **kw):
        if backend == "openvino" and device == "GPU":
            raise RuntimeError("Exception from src/inference/src/cpp/core.cpp: x\nmore")
        self.backend, self.device = backend, (device if backend == "openvino" else "CPU")
        _FakeEst.made.append((backend, self.device))

    def estimate_crop(self, frame, box):
        a = np.linspace(0, 2 * np.pi, 478, endpoint=False)
        lm = np.stack([32 + 10 * np.cos(a), 24 + 10 * np.sin(a), np.zeros(478)], 1).astype(np.float32)
        return dict(landmarks=lm, blendshapes=np.zeros(52, np.float32), score=np.float32(1.0),
                    transform=np.full((4, 4), np.nan, np.float32))

    def close(self):
        pass


def test_open_face_estimator_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(F, "FaceEstimator", _FakeEst)
    est, note = F.open_face_estimator(backend="openvino", device="GPU")
    assert est.device == "CPU" and "OpenVINO GPU did not load" in note and "using CPU" in note
    assert "\n" not in note and "more" in note and "core.cpp" not in note   # OpenVINO source lines skipped
    real = RuntimeError("Exception from src/inference/src/cpp/core.cpp:117:\nCheck '!m_device_map.empty()' failed\n"
                        "[GPU] Can't get PERFORMANCE_HINT property as no supported devices found\n\n")
    assert F.fallback_note("GPU", real) == ("OpenVINO GPU did not load ([GPU] Can't get PERFORMANCE_HINT property as "
                                            "no supported devices found); using CPU instead")
    est, note = F.open_face_estimator(backend="openvino", device="NPU")
    assert est.device == "NPU" and note == ""
    monkeypatch.setattr(F, "FaceEstimator", lambda **kw: (_ for _ in ()).throw(RuntimeError("cpu broken")))
    with pytest.raises(RuntimeError):                               # CPU itself failing is not hidden
        F.open_face_estimator(backend="openvino", device="CPU")

    def missing(**kw):
        raise FileNotFoundError("face_landmarker.task not found")
    monkeypatch.setattr(F, "FaceEstimator", missing)
    with pytest.raises(FileNotFoundError):                          # a missing model is not a device problem
        F.open_face_estimator(backend="openvino", device="GPU")


def test_probe_reports_cpu_and_note(monkeypatch):
    monkeypatch.setattr(F, "FaceEstimator", _FakeEst)
    assert O.probe_landmarker("mediapipe", "CPU") == ("CPU", "")
    used, note = O.probe_landmarker("openvino", "GPU", opener=F.open_face_estimator)
    assert used == "CPU" and "did not load" in note
    assert O.probe_landmarker("openvino", "NPU", opener=F.open_face_estimator) == ("NPU", "")


def test_sidecar_records_device_actually_used(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "FaceEstimator", _FakeEst)
    v = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(3):
        w.write(np.full((48, 64, 3), 90, np.uint8))
    w.release()
    base = ["--input", str(v), "--whole", "--raw-root", str(tmp_path / "r"), "--deid", "points"]
    assert X.main(base + ["--out", str(tmp_path / "o"), "--face-backend", "openvino", "--face-device", "GPU"]) == 0
    j = S.read_json(next((tmp_path / "o").rglob("*desc-points_video.json")))
    f = j["FaceLandmarker"]
    assert f["Backend"] == "openvino" and f["Device"] == "CPU" and f["DeviceRequested"] == "GPU"
    assert "did not load" in f["Note"]
    lj = S.read_json(next((tmp_path / "o").rglob("*_landmarks.json")))
    assert lj["FaceLandmarker"]["Device"] == "CPU"
    assert X.main(base + ["--out", str(tmp_path / "o2"), "--face-backend", "openvino", "--face-device", "NPU"]) == 0
    f = S.read_json(next((tmp_path / "o2").rglob("*desc-points_video.json")))["FaceLandmarker"]
    assert f == dict(Backend="openvino", Device="NPU", DeviceRequested="NPU")
    assert X.main(base + ["--out", str(tmp_path / "o3")]) == 0      # default
    f = S.read_json(next((tmp_path / "o3").rglob("*desc-points_video.json")))["FaceLandmarker"]
    assert f == dict(Backend="mediapipe", Device="CPU", DeviceRequested="CPU")


def test_ghost_hint_points_to_fetch_models():
    g = O.missing_weights_message("ghost", ["no swap configuration chosen (Swap config)"])
    assert "python scripts/fetch_models.py --ghost" in g and O.GHOST_FETCH_HINT in g
    assert "download_models.sh" not in g and not hasattr(O, "GHOST_DOWNLOAD")


def _root():
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("no display")
    import tkinter as tk
    try:
        return tk.Tk()
    except tk.TclError:
        pytest.skip("no display")


def test_picker_widget_devices_default_and_fallback(tmp_path):
    from src import deid_ui as U
    root = _root()
    try:
        sel = U.DeidSelector(root, {}, tmp_path, ov_devices=["CPU", "GPU.0", "NPU"],
                             probe=lambda b, d: ("CPU", f"OpenVINO {d} did not load (x); using CPU instead")
                             if d == "GPU" else (d, ""))
        s = sel.state()
        assert (s["face_backend"], s["face_device"]) == ("mediapipe", "CPU") and s["face_note"] == ""
        assert list(sel.backend_box.cget("values")) == ["MediaPipe", "OpenVINO"]
        assert str(sel.device_box.cget("state")) == "disabled"
        sel.backend.set("OpenVINO"); sel.landmarker_changed(check=False)
        assert list(sel.device_box.cget("values")) == ["CPU", "GPU", "NPU", "AUTO"]
        sel.device.set("NPU"); sel.landmarker_changed(check=False); sel.check_landmarker(wait=True)
        assert sel.state()["face_device"] == "NPU" and sel.state()["face_note"] == ""
        sel.device.set("GPU"); sel.landmarker_changed(check=False); sel.check_landmarker(wait=True)
        s = sel.state()
        assert s["face_device"] == "CPU" and "GPU did not load" in s["face_note"]
        assert "GPU did not load" in sel.lmk_label.cget("text")
        assert U.landmarker_settings(s) == dict(face_backend="openvino", face_device="CPU")
        # no OpenVINO device: OpenVINO is not offered; saved OpenVINO choice -> default
        sel2 = U.DeidSelector(root, dict(face_backend="openvino", face_device="GPU"), tmp_path, ov_devices=[])
        assert list(sel2.backend_box.cget("values")) == ["MediaPipe"]
        assert (sel2.state()["face_backend"], sel2.state()["face_device"]) == ("mediapipe", "CPU")
        sel3 = U.DeidSelector(root, dict(face_backend="openvino", face_device="NPU"), tmp_path,
                              ov_devices=["CPU", "NPU"], probe=lambda b, d: (d, ""))
        assert (sel3.state()["face_backend"], sel3.state()["face_device"]) == ("openvino", "NPU")
    finally:
        root.destroy()
