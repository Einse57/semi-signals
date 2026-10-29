"""Convert and compile the three Face Landmarker models with OpenVINO on each device; report status.

  python scripts/face_ov_devices.py --devices CPU GPU NPU --json out.json

Per model and device: read through the TFLite frontend, compile with the input shape as stored (the
landmark model has a dynamic batch) and with a static [1, ...] shape, each with the device default
precision and with f32 / f16 hints, plus a second compile from an IR cache. Each compiled model runs one
random input; the output is compared with CPU f32 (cosine, max abs).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from face_ov.bundle import unpack_task  # noqa: E402

SHAPES = {"face_detector.tflite": [1, 128, 128, 3], "face_landmarks_detector.tflite": [1, 256, 256, 3],
          "face_blendshapes.tflite": [1, 146, 2]}


def _run(cm, x):
    r = cm.create_infer_request().infer({0: x})
    return np.concatenate([np.ravel(np.asarray(r[o])) for o in cm.outputs]).astype(np.float64)


def _cmp(a, b):
    return {"cos": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)), "max_abs": float(np.abs(a - b).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=str(ROOT / "models" / "face_landmarker.task"))
    ap.add_argument("--devices", nargs="+", default=["CPU", "GPU", "NPU"])
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    import openvino as ov
    core = ov.Core()
    paths = unpack_task(a.task)
    rng = np.random.default_rng(0)
    out = {"openvino": ov.__version__, "available": core.available_devices,
           "names": {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices}, "models": {}}
    for name, shape in SHAPES.items():
        x = (rng.uniform(-1, 1, shape) if "detector" in name else rng.uniform(0, 1, shape) if "landmarks" in name
             else rng.uniform(50, 300, shape)).astype(np.float32)
        m0 = core.read_model(str(paths[name]))
        rep = {"inputs": [f"{i.get_any_name()} {i.get_partial_shape()}" for i in m0.inputs],
               "outputs": [f"{o.get_any_name()} {o.get_partial_shape()}" for o in m0.outputs],
               "ops": len(m0.get_ops())}
        ref = _run(core.compile_model(_static(core, paths[name], shape), "CPU", {"INFERENCE_PRECISION_HINT": "f32"}), x)
        for dev in a.devices:
            d = rep.setdefault(dev, {})
            for label, static, hint in (("as_stored", False, None), ("static", True, None),
                                        ("static_f32", True, "f32"), ("static_f16", True, "f16")):
                try:
                    m = _static(core, paths[name], shape) if static else core.read_model(str(paths[name]))
                    t0 = time.perf_counter()
                    cm = core.compile_model(m, dev, {"INFERENCE_PRECISION_HINT": hint} if hint else {})
                    ms = (time.perf_counter() - t0) * 1e3
                    y = _run(cm, x)
                    prec = str(cm.get_property("INFERENCE_PRECISION_HINT")) if dev != "NPU" else "n/a"
                    d[label] = {"ok": True, "compile_ms": round(ms, 1), "precision": prec, **_cmp(ref, y)}
                except Exception as e:     # noqa: BLE001 - record the failure and continue
                    d[label] = {"ok": False, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:240]}"}
            try:
                with tempfile.TemporaryDirectory() as td:
                    c2 = ov.Core()
                    c2.set_property({"CACHE_DIR": td})
                    t = []
                    for _ in range(2):
                        t0 = time.perf_counter()
                        c2.compile_model(_static(c2, paths[name], shape), dev)
                        t.append(round((time.perf_counter() - t0) * 1e3, 1))
                    d["ir_cache_ms"] = {"first": t[0], "cached": t[1]}
            except Exception as e:         # noqa: BLE001
                d["ir_cache_ms"] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:240]}"}
            print(name, dev, json.dumps(d)[:600], flush=True)
        out["models"][name] = rep
    Path(a.json).write_text(json.dumps(out, indent=1))


def _static(core, path, shape):
    m = core.read_model(str(path))
    m.reshape({m.inputs[0]: shape})
    return m


if __name__ == "__main__":
    main()
