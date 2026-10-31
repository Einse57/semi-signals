"""Label sets for the annotation tool, one file per mode: labels/<mode>.json (or .yaml / .yml).

A label set says which fields an event of that mode has. Both modes use the same timing fields
(clinical onset, clinical end, EEG onset) and an optional free-text note. On top of that:
  event_class  values for the event tier (body: Seizure tier; empty for face)
  term         an optional pick list (body: ILAE 2017 seizure type). An empty "choices" list switches
               the term key off; filling it in (plus "source") switches it on, with no code change.
  side         allowed side values (face: L / R / both); empty means no side field.

A .yaml/.yml file is used instead of the .json one when it exists and PyYAML is installed (optional;
JSON needs no extra package). SEMI_SIGNALS_LABELS=<dir> points to another folder of label files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

SCHEMA = "semi-signals.labels/1"
MODES = ("body", "face")
TIMING_FIELDS = ("clinical_onset_s", "clinical_end_s", "eeg_onset_s")
LABEL_DIR = Path(__file__).resolve().parent.parent / "labels"


def label_dir() -> Path:
    return Path(os.environ.get("SEMI_SIGNALS_LABELS") or LABEL_DIR)


def _read(path: Path) -> dict:
    if path.suffix in (".yaml", ".yml"):
        import yaml                                    # optional dependency, only for .yaml files
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def label_file(mode: str, directory=None) -> Path:
    d = Path(directory) if directory else label_dir()
    for ext in (".yaml", ".yml"):
        p = d / f"{mode}{ext}"
        if p.exists():
            try:
                import yaml  # noqa: F401
                return p
            except ImportError:
                pass
    return d / f"{mode}.json"


def validate(cfg: dict, mode: str) -> dict:
    """Check one label set and fill defaults. Raises ValueError on a malformed file."""
    if cfg.get("schema") != SCHEMA:
        raise ValueError(f"label set '{mode}': schema must be {SCHEMA!r}, got {cfg.get('schema')!r}")
    if cfg.get("mode") != mode:
        raise ValueError(f"label set '{mode}': mode field is {cfg.get('mode')!r}")
    if tuple(cfg.get("timing_fields", TIMING_FIELDS)) != TIMING_FIELDS:
        raise ValueError(f"label set '{mode}': timing_fields must be {list(TIMING_FIELDS)} (shared by all modes)")
    term = dict(field=None, tier=None, prompt="Term", source="", choices=[])
    term.update(cfg.get("term") or {})
    if term["choices"] and not (term["field"] and term["tier"]):
        raise ValueError(f"label set '{mode}': term needs 'field' and 'tier' when it has choices")
    for k in ("event_class", "side"):
        if not isinstance(cfg.get(k, []), list):
            raise ValueError(f"label set '{mode}': {k} must be a list")
    return dict(cfg, timing_fields=list(TIMING_FIELDS), event_class=list(cfg.get("event_class", [])),
                term=term, side=list(cfg.get("side", [])), note=bool(cfg.get("note", True)),
                title=cfg.get("title", mode))


def load(mode: str, directory=None) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown label mode {mode!r} (expected one of {MODES})")
    p = label_file(mode, directory)
    cfg = validate(_read(p), mode)
    cfg["path"] = str(p)
    return cfg


def load_all(directory=None) -> dict:
    return {m: load(m, directory) for m in MODES}
