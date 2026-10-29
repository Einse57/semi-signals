"""Unpack the MediaPipe face_landmarker.task bundle (a zip of three .tflite models) at run time.

The bundle is not redistributed here: it is downloaded from Google's model storage and checked against
its SHA-256 before use. Models: Apache-2.0 (see the MediaPipe Face Landmarker model card).
"""
from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

MODEL_FILES = ("face_detector.tflite", "face_landmarks_detector.tflite", "face_blendshapes.tflite")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_cache_dir() -> Path:
    base = os.environ.get("FACE_OV_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")), "face_ov")
    return Path(base)


def unpack_task(task_path, expected_sha256: str | None = None, out_dir=None) -> dict[str, Path]:
    """Extract the three models into out_dir/<sha256[:16]>/ (once) and return {name: path}.
    Raises ValueError when expected_sha256 is given and does not match."""
    task_path = Path(task_path)
    if not task_path.is_file():
        raise FileNotFoundError(f"face model bundle not found: {task_path}")
    digest = sha256_file(task_path)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError(f"SHA-256 mismatch for {task_path}: {digest} != {expected_sha256}")
    out = Path(out_dir or default_cache_dir()) / digest[:16]
    paths = {n: out / n for n in MODEL_FILES}
    if not all(p.is_file() for p in paths.values()):
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(task_path) as z:
            names = set(z.namelist())
            missing = [n for n in MODEL_FILES if n not in names]
            if missing:
                raise ValueError(f"{task_path} lacks {missing}")
            for n in MODEL_FILES:
                tmp = out / (n + ".part")
                tmp.write_bytes(z.read(n))
                tmp.replace(out / n)
    return paths
