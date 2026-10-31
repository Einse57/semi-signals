"""Fetch, verify and convert the models and sample data the seizure detector uses (idempotent, sha256-checked).

  python scripts/fetch_models.py                       # models into ./models, labels + 2 sample clips into ./data
  python scripts/fetch_models.py --offline             # verify only, no network
  python scripts/fetch_models.py --detector-a D:/models/detector_a_all.txt
  python scripts/fetch_models.py --skip-vsvig --force
  python scripts/fetch_models.py --clips pat12/Sz1P,pat12/Sz2P   # other sample clips (or --clips all, ~27 GB)
  python scripts/fetch_models.py --skip-data           # models only

What it does:
1. Pose models (YOLOX-tiny, RTMPose-m body7 and hand5): downloads them into rtmlib's own cache
   (the same files `src/config.py` points at), so the live viewer starts without a download.
2. VSViG-base: downloads `VSViG-base.pth` and `dy_point_order.pt` from the official repository
   (github.com/xuyankun/VSViG, pinned commit), checks sha256, then converts to ONNX and OpenVINO IR
   with `src/export_vsvig.py` only if the IR is missing or was built from different weights.
   Conversion needs torch (CPU build is enough); the live viewer itself does not. Without torch the
   conversion is reported as skipped.
3. Detector A: downloads from the Hugging Face repo `Einse57/semi-signals-detector-a` with the pinned
   SHA-256s in `src/hf_models.py`,
   or copies a local model given with `--detector-a PATH` (its `.json` sidecar is taken from next to it).
4. Sample data: `Label.xlsx` and a few clips from the Hugging Face dataset `xuyankun/WU-SAHZU-EMU-Video`
   (Apache-2.0, pinned commit) into --data-dir, without the dataset's `dataset/` prefix
   (e.g. `data/Label.xlsx`, `data/pat10/Sz2P.mp4`).

Hashes of everything in --models-dir are recorded in `<models-dir>/MANIFEST.json` (sample data under "data").
Works on Windows, macOS and Linux. Model files are not part of the repository.
"""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, os, shutil, subprocess, sys, tempfile, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VSVIG_COMMIT = "1026e7e7f2287b96f3cc375830f2836ffdf4588e"
VSVIG_RAW = f"https://raw.githubusercontent.com/xuyankun/VSViG/{VSVIG_COMMIT}/"
VSVIG_FILES = {
    "VSViG-base.pth": "0a95e3e0093730dc535e21a84f04943c73549a2b6a19a85e33aebc7c64f965a9",
    "dy_point_order.pt": "9aa778ca1221f47b6afe0242787becfdfaae3f2424b54dd15a48c585ef0a0ced",
}
VSVIG_IR_STEM = "vsvig_base"

# sha256 of the extracted .onnx files in rtmlib's cache
POSE_SHA256 = {
    "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx":
        "ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250",
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx":
        "5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c",
    "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.onnx":
        "39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29",
}

# Sample data: WU-SAHZU-EMU-Video (VSViG authors), Hugging Face dataset, pinned commit.
DATA_REPO = "xuyankun/WU-SAHZU-EMU-Video"
DATA_REVISION = "d26859e666ddfeb00eede0fadde03642d9e8ef1e"
DATA_PREFIX = "dataset/"
DATA_RESOLVE = f"https://huggingface.co/datasets/{DATA_REPO}/resolve/{DATA_REVISION}/{DATA_PREFIX}"
DATA_TREE_API = f"https://huggingface.co/api/datasets/{DATA_REPO}/tree/{DATA_REVISION}/{DATA_PREFIX.rstrip('/')}?recursive=true"
DATA_LABELS = "Label.xlsx"
DEFAULT_CLIPS = "pat10/Sz2P,pat10/Sz1P"
# sha256 at DATA_REVISION (clips: the LFS oid). Other clips get their sha256 from the tree API at the same commit.
DATA_SHA256 = {
    "Label.xlsx": "488e6babeb33e78fd05803e2eb094942c43b1212649310b5b174ef0112fbab26",
    "pat10/Sz1P.mp4": "1105f64e01e9a0ef5c2446e130d49f7955986ff3698b034aa0ed325b0d7702ce",
    "pat10/Sz2P.mp4": "41644d9940a5e15e0b49ecedc3277adc78eabb6c32fcfdb5802694aa107c36ce",
    "pat12/Sz1P.mp4": "092ca70fe46c02c3398cb2e2d1ddb084b0d09706d44ade61fba7eea68e28cc07",
    "pat12/Sz2P.mp4": "fbaafd24e7050acc95822c5957a295036281875a28656f6d1b85adaaa116da1d",
}
DATA_CREDIT = """\
  Sample data: WU-SAHZU-EMU-Video, https://huggingface.co/datasets/xuyankun/WU-SAHZU-EMU-Video
    (Yankun Xu, Mohamad Sawan et al., VSViG, ECCV 2024) - Apache License 2.0. Patient faces are
    masked by the dataset authors."""

sys.path.insert(0, str(ROOT))
from src import hf_models  # noqa: E402  (Detector A repo, revision and pinned SHA-256s live there)

ATTRIBUTION = """\
Attribution
  VSViG code and VSViG-base weights: Yankun Xu and Mohamad Sawan, "VSViG: Real-time Video-based
    Seizure Detection via Skeleton-based Spatiotemporal ViG", ECCV 2024.
    https://github.com/xuyankun/VSViG - Apache License 2.0.
  rtmlib, RTMPose, YOLOX: Apache-2.0. Note: the YOLOX-tiny checkpoint was trained with Human-Art
    (non-commercial licence) and the RTMPose body7/hand5 checkpoints include research-only datasets.
  Research use only, not a medical device."""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, suffix=".part")
    os.close(fd)
    try:
        print(f"    downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "semi-signals-fetch-models"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.d = json.loads(path.read_text()) if path.exists() else {}

    def set(self, name, **kw):
        self.d[name] = kw

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.d, indent=2, sort_keys=True) + "\n")


def ok_file(p: Path, want: str | None) -> bool:
    return p.exists() and (want is None or sha256(p) == want)


# ── 1. pose models via rtmlib ────────────────────────────────
def pose_urls() -> list[str]:
    tree = ast.parse((ROOT / "src" / "config.py").read_text(encoding="utf-8"))
    urls = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id.endswith("_ONNX_URL") for t in n.targets):
            urls.append(ast.literal_eval(n.value))
    return urls


def step_pose(a, status):
    print("[1/4] pose models (rtmlib cache)")
    try:
        from rtmlib.tools.file import download_checkpoint, _get_rtmhub_dir
    except ImportError:
        status["pose"] = "FAILED: rtmlib not installed (pip install -r requirements-demo.txt)"
        print("    " + status["pose"]); return
    cache = Path(_get_rtmhub_dir()) / "checkpoints"
    bad = []
    for url in pose_urls():
        onnx = cache / (Path(url).name.split(".")[0] + ".onnx")
        want = POSE_SHA256.get(onnx.name)
        if a.force and onnx.exists() and not a.offline:
            onnx.unlink()
        if not ok_file(onnx, want):
            if a.offline:
                bad.append(f"{onnx.name} missing or hash mismatch"); continue
            if onnx.exists():
                onnx.unlink()  # wrong hash: fetch again
            onnx = Path(download_checkpoint(url))
            if not ok_file(onnx, want):
                bad.append(f"{onnx.name} sha256 mismatch after download"); continue
        print(f"    ok  {onnx.name}  sha256 {sha256(onnx)[:12]}...")
    status["pose"] = "ok" if not bad else "FAILED: " + "; ".join(bad)
    print(f"    cache: {cache}")


# ── 2. VSViG-base weights + OpenVINO IR ──────────────────────
def step_vsvig(a, md: Path, man: Manifest, status):
    print("[2/4] VSViG-base (github.com/xuyankun/VSViG @ %s)" % VSVIG_COMMIT[:7])
    for name, want in VSVIG_FILES.items():
        p = md / name
        if ok_file(p, want) and not (a.force and not a.offline):
            print(f"    ok  {name}")
        elif a.offline:
            status["vsvig"] = f"FAILED: {name} missing or hash mismatch"; print("    " + status["vsvig"]); return
        else:
            download(VSVIG_RAW + name, p)
            if not ok_file(p, want):
                status["vsvig"] = f"FAILED: {name} sha256 mismatch ({sha256(p)})"; print("    " + status["vsvig"]); return
            print(f"    ok  {name} (downloaded)")
        man.set(name, sha256=want, source=VSVIG_RAW + name, license="Apache-2.0")
    vendored = ROOT / "third_party" / "vsvig" / "dy_point_order.pt"
    if vendored.exists() and sha256(vendored) != VSVIG_FILES["dy_point_order.pt"]:
        print("    warning: third_party/vsvig/dy_point_order.pt differs from the official file")

    xml, bin_, onnx = (md / (VSVIG_IR_STEM + s) for s in (".xml", ".bin", ".onnx"))
    rec = man.d.get(xml.name, {})
    fresh = (xml.exists() and bin_.exists() and rec.get("from_sha256") == VSVIG_FILES["VSViG-base.pth"]
             and rec.get("sha256") == sha256(xml) and man.d.get(bin_.name, {}).get("sha256") == sha256(bin_))
    if fresh and not a.force:
        print(f"    ok  {xml.name} / {bin_.name} (built from the pinned weights)")
        status["vsvig"] = "ok"; return
    if a.offline:
        status["vsvig"] = "FAILED: OpenVINO IR missing or stale (run without --offline)"; print("    " + status["vsvig"]); return
    if importlib.util.find_spec("torch") is None:
        status["vsvig"] = ("skipped: torch not installed (only needed to convert VSViG once; e.g. "
                           "pip install torch --index-url https://download.pytorch.org/whl/cpu)")
        print("    " + status["vsvig"]); return
    print("    converting to ONNX + OpenVINO IR with src/export_vsvig.py")
    r = subprocess.run([sys.executable, "-m", "src.export_vsvig", "--weights", str(md / "VSViG-base.pth"),
                        "--out", str(md / VSVIG_IR_STEM)], cwd=ROOT)
    if r.returncode != 0 or not (xml.exists() and bin_.exists()):
        status["vsvig"] = ("FAILED: conversion failed (needs torch, onnx, onnxruntime, openvino; e.g. "
                           "pip install torch --index-url https://download.pytorch.org/whl/cpu)")
        print("    " + status["vsvig"]); return
    for p in (onnx, bin_):
        if p.exists():
            man.set(p.name, sha256=sha256(p), from_sha256=VSVIG_FILES["VSViG-base.pth"], built_by="src/export_vsvig.py")
    man.set(xml.name, sha256=sha256(xml), from_sha256=VSVIG_FILES["VSViG-base.pth"], built_by="src/export_vsvig.py")
    print(f"    ok  {xml.name} / {bin_.name} (converted)")
    status["vsvig"] = "ok"


# ── 3. Detector A ────────────────────────────────────────────
def step_detector_a(a, md: Path, man: Manifest, status):
    if a.detector_a:
        src = Path(a.detector_a).expanduser().resolve()
        print(f"[3/4] Detector A (local file {src})")
        side = src.with_suffix(".json")
        if not src.exists():
            status["detector_a"] = f"FAILED: {src} not found"; print("    " + status["detector_a"]); return
        for p in [src] + ([side] if side.exists() else []):
            dst = md / p.name
            if p != dst.resolve():
                if not (dst.exists() and sha256(dst) == sha256(p)):
                    shutil.copy2(p, dst)
            h = sha256(dst)
            pinned = hf_models.EXPECTED_SHA256.get(dst.name)
            note = "" if pinned is None else (" (matches pinned release)" if pinned == h else " (differs from pinned release)")
            man.set(dst.name, sha256=h, source=f"local:{p.name}")
            print(f"    ok  {dst.name}  sha256 {h[:12]}...{note}")
        if not side.exists():
            print(f"    warning: no {side.name} next to the model; the live viewer needs it for the threshold")
        status["detector_a"] = "ok (local)"; return

    print(f"[3/4] Detector A (huggingface.co/{hf_models.HF_REPO}, via src/hf_models.py)")
    pins = hf_models.EXPECTED_SHA256
    need = [n for n, h in pins.items() if a.force or not ok_file(md / n, h)]
    for n in pins:
        if n not in need:
            man.set(n, sha256=pins[n], source=f"hf:{hf_models.HF_REPO}")
    if not need:
        print("    ok  " + ", ".join(pins) + " present and verified"); status["detector_a"] = "ok"; return
    if a.offline:
        status["detector_a"] = "FAILED: missing or unverified: " + ", ".join(need); print("    " + status["detector_a"]); return
    try:
        from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError, HfHubHTTPError
    except ImportError:
        status["detector_a"] = "FAILED: huggingface_hub not installed (pip install huggingface_hub)"; print("    " + status["detector_a"]); return
    try:
        for n in need:
            if a.force and (md / n).exists():
                (md / n).unlink()
            hf_models._fetch(n, local_dir=str(md))  # downloads and refuses a SHA-256 mismatch
            man.set(n, sha256=pins[n], source=f"hf:{hf_models.HF_REPO}")
            print(f"    ok  {n} (downloaded, sha256 verified)")
    except (RepositoryNotFoundError, EntryNotFoundError):
        status["detector_a"] = (f"NOT AVAILABLE: {hf_models.HF_REPO} (or a file in it) is not published on Hugging Face "
                                "yet. Use --detector-a PATH to point at a local detector_a_*.txt (with its .json).")
        print("    " + status["detector_a"]); return
    except SystemExit as e:  # hf_models refuses a hash mismatch this way
        status["detector_a"] = f"FAILED: {e}"; print("    " + status["detector_a"]); return
    except (HfHubHTTPError, OSError) as e:
        status["detector_a"] = f"FAILED: could not reach Hugging Face ({type(e).__name__}). Retry, or use --detector-a PATH."
        print("    " + status["detector_a"]); return
    status["detector_a"] = "ok"


# ── 4. sample data (labels + clips) ─────────────────────────
def fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "semi-signals-fetch-models"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def data_tree() -> dict[str, tuple[str | None, int]]:
    """{relative path without 'dataset/': (sha256 or None, size)} at DATA_REVISION."""
    out = {}
    for e in fetch_json(DATA_TREE_API):
        if e.get("type") == "file" and e["path"].startswith(DATA_PREFIX):
            out[e["path"][len(DATA_PREFIX):]] = ((e.get("lfs") or {}).get("oid"), int(e.get("size", 0)))
    return out


def clip_list(spec: str) -> list[str] | None:
    """'pat10/Sz2P,pat10/Sz1P' -> ['pat10/Sz2P.mp4', ...]; 'all' -> None; '' or 'none' -> []."""
    spec = spec.strip()
    if spec.lower() == "all":
        return None
    if spec.lower() in ("", "none"):
        return []
    return [c.strip().replace("\\", "/") + ("" if c.strip().endswith(".mp4") else ".mp4") for c in spec.split(",") if c.strip()]


def step_data(a, man: Manifest, status):
    dd = Path(a.data_dir).expanduser().resolve()
    print(f"[4/4] sample data (huggingface.co/datasets/{DATA_REPO} @ {DATA_REVISION[:7]}) -> {dd}")
    clips = clip_list(a.clips)
    rec = man.d.setdefault("data", {})
    pins = dict(DATA_SHA256)
    tree = None
    need_tree = clips is None or any(c not in pins for c in clips)
    if need_tree and not a.offline:
        try:
            tree = data_tree()
        except OSError as e:
            status["data"] = f"FAILED: could not list the dataset ({type(e).__name__}). Retry, or use --skip-data."
            print("    " + status["data"]); return
        for k, (h, _) in tree.items():
            if h and k not in pins:
                pins[k] = h
    if clips is None:
        clips = sorted(k for k in tree if k.endswith(".mp4")) if tree else sorted(k for k in rec if k.endswith(".mp4"))
        if tree and not a.offline:
            gb = sum(tree[k][1] for k in clips) / 1e9
            print(f"    warning: --clips all is {len(clips)} videos, about {gb:.1f} GB")
    wanted = [DATA_LABELS] + clips
    bad = []
    for rel in wanted:
        want = pins.get(rel) or (rec.get(rel) or {}).get("sha256")
        if want is None and a.offline:
            bad.append(f"{rel}: no pinned sha256 (run once online)"); continue
        if tree is not None and rel not in tree:
            bad.append(f"{rel}: not in the dataset"); continue
        dst = dd / Path(rel)
        if ok_file(dst, want) and want is not None and not (a.force and not a.offline):
            print(f"    ok  {rel}")
        elif a.offline:
            bad.append(f"{rel} missing or hash mismatch"); continue
        else:
            try:
                download(DATA_RESOLVE + rel, dst)
            except OSError as e:
                bad.append(f"{rel}: download failed ({type(e).__name__})"); continue
            h = sha256(dst)
            if want is not None and h != want:
                dst.unlink()
                bad.append(f"{rel}: sha256 mismatch ({h[:12]}...)"); continue
            want = h
            print(f"    ok  {rel} (downloaded, {dst.stat().st_size / 1e6:.1f} MB, sha256 verified)")
        rec[rel] = {"sha256": want, "path": str(dst), "source": f"hf-dataset:{DATA_REPO}@{DATA_REVISION}",
                    "license": "Apache-2.0"}
    status["data"] = "ok" if not bad else "FAILED: " + "; ".join(bad)
    if bad:
        print("    " + status["data"])
    print(DATA_CREDIT)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--detector-a", default=None, metavar="PATH", help="use a local Detector A model (.txt, with .json next to it)")
    ap.add_argument("--skip-vsvig", action="store_true", help="skip VSViG download and conversion")
    ap.add_argument("--force", action="store_true", help="download and convert again even if files verify")
    ap.add_argument("--offline", action="store_true", help="no network: only verify what is already there")
    ap.add_argument("--skip-data", action="store_true", help="skip Label.xlsx and the sample clips")
    ap.add_argument("--clips", default=DEFAULT_CLIPS,
                    help=f"sample clips, comma-separated patNN/SzNP (default {DEFAULT_CLIPS}); 'all' for every video (~27 GB)")
    ap.add_argument("--data-dir", default=str(ROOT / "data"), help="where Label.xlsx and the clips go (default ./data)")
    a = ap.parse_args(argv)
    md = Path(a.models_dir).expanduser().resolve(); md.mkdir(parents=True, exist_ok=True)
    man = Manifest(md / "MANIFEST.json")
    status = {}
    step_pose(a, status)
    if a.skip_vsvig:
        status["vsvig"] = "skipped"
    else:
        step_vsvig(a, md, man, status)
    step_detector_a(a, md, man, status)
    if a.skip_data:
        status["data"] = "skipped"
    else:
        step_data(a, man, status)
    man.save()
    print("\nSummary (models dir: %s)" % md)
    for k, v in status.items():
        print(f"  {k:11s} {v}")
    print("\n" + ATTRIBUTION)
    return 0 if all(v.startswith(("ok", "skipped")) for v in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
