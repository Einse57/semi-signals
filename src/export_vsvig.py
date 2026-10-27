"""Export VSViG-base weights to ONNX (and OpenVINO IR if openvino is installed).

  python -m src.export_vsvig --weights VSViG-base.pth --out models/vsvig_base
"""
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True, help="output path stem, e.g. models/vsvig_base")
    ap.add_argument("--batch", type=int, default=1)
    a = ap.parse_args()
    import torch
    from src.vsvig_runner import load_torch_model
    m = load_torch_model(a.weights)
    x = torch.rand(a.batch, 30, 15, 3, 32, 32) * 255
    k = torch.rand(a.batch, 30, 15, 3) * torch.tensor([1920, 1080, 1.0])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = out.with_suffix(".onnx")
    torch.onnx.export(m, (x, k), str(onnx_path), input_names=["patches", "kpts"], output_names=["prob"],
                      opset_version=17, dynamo=False)
    print("wrote", onnx_path)
    with torch.no_grad():
        ref = m(x, k).numpy()
    import onnxruntime as ort
    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = s.run(None, {"patches": x.numpy(), "kpts": k.numpy()})[0]
    print("onnx max abs diff vs torch:", float(abs(got.reshape(-1) - ref.reshape(-1)).max()))
    try:
        import openvino as ov
        ovm = ov.convert_model(str(onnx_path))
        ov.save_model(ovm, str(out.with_suffix(".xml")))
        c = ov.Core().compile_model(str(out.with_suffix(".xml")), "CPU")
        g = c([x.numpy(), k.numpy()])[0]
        print("wrote", out.with_suffix(".xml"), "openvino max abs diff:", float(abs(g.reshape(-1) - ref.reshape(-1)).max()))
    except ImportError:
        print("openvino not installed; skipped IR")


if __name__ == "__main__":
    main()
