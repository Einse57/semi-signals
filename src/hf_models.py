"""Fetch Detector A from the Hugging Face Hub, with a pinned SHA-256 check.

Usage: pass `--detector-a hf` (or `hf:detector_a_all.txt`) to run_seizure_demo / review_video,
or leave the default `models/detector_a_all.txt`: if that file is missing, it is fetched from the Hub.
The expected hashes live here, in the code, so a changed file on the Hub is refused, not trusted.
"""
from __future__ import annotations

import hashlib
import os

HF_REPO = "Einse57/semi-signals-detector-a"
HF_REVISION = "a52676cbb7ab8f5cd57c02cc06ed392442b477ba"  # Hub commit of the v1 upload
EXPECTED_SHA256 = {
    "detector_a_all.txt": "737fb6a103be81cc9a4984a2139b952a7e07800a07e877761539f85a6a0cb87e",
    "detector_a_all.json": "fa409d62dba5a3459d05e217dc173860e2574e85d764646d1519aed09c5a0c65",
    "detector_a_body.txt": "d6faf006733b43a1dc443206658c7094df5e5d65cc80febe9eed1201bd821747",
    "detector_a_body.json": "09fc4f0e1fc80451abfda9e168dd04761660c9960c596874e6b81911930add00",
}
DEFAULT_LOCAL = os.path.join("models", "detector_a_all.txt")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(filename: str, local_dir: str | None = None) -> str:
    """Download one pinned file (into the HF cache, or into local_dir) and verify its SHA-256.
    Hub errors (e.g. repository not found) are raised to the caller unchanged."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit("pip install huggingface_hub to download Detector A, "
                         "or pass --detector-a <local .txt>") from e
    if filename not in EXPECTED_SHA256:
        raise SystemExit(f"{filename}: no pinned hash; only {sorted(EXPECTED_SHA256)} are supported here")
    path = hf_hub_download(HF_REPO, filename, revision=HF_REVISION, local_dir=local_dir)
    got = sha256(path)
    if got != EXPECTED_SHA256[filename]:
        raise SystemExit(f"{filename}: SHA-256 mismatch (got {got}); refusing to load")
    return path


def resolve_detector_a(spec: str) -> str:
    """Return a local path to a verified Detector A model (its .json sits next to it)."""
    if not spec:
        return spec
    if spec == "hf" or spec.startswith("hf:"):
        name = spec[3:] or "detector_a_all.txt"
    elif not os.path.exists(spec) and os.path.normpath(spec) == DEFAULT_LOCAL:
        name = "detector_a_all.txt"
        print(f"[models] {spec} not found; fetching {name} from {HF_REPO}")
    else:
        return spec
    model = _fetch(name)
    _fetch(os.path.splitext(name)[0] + ".json")  # same snapshot folder as the model
    return model
