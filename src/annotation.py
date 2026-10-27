"""Annotation files for semi-signals videos (ELAN-compatible). Self-contained on purpose.

Per video <stem>.<ext>, written next to it:
  <stem>.eaf              ELAN EAF 3.0 (ms). Human tiers: Seizure (+ ILAE2017_type child), EEG_onset,
                          Notes, Exclude. Model tiers: Model_alarm (+ Model_drivers child).
                          Provenance in HEADER PROPERTY elements.
  <stem>_signals.csv      per-second model probability, alarm state and region speeds
  <stem>_tsconf.xml       ELAN Timeseries Viewer track setup for the CSV
  <stem>.pfsx             ELAN per-document preferences: which tracks go in which viewer panel
                          (only written if absent; ELAN keeps the user's layout there)
  <stem>.annotation.json  the same content as plain JSON (the app's own copy)

Rules: model output never goes into human tiers; on import, a sha256 mismatch is refused and
Model_* tiers are never read back as human labels. ELAN itself (GPL-3.0) is not bundled;
only its documented XML format is written (stdlib xml.etree, no extra dependency).
Research use only, not diagnostic.
"""
from __future__ import annotations

import contextlib
import csv
import datetime as _dt
import hashlib
import json
import os
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

SCHEMA = "semi-signals.annotation/1"
EAF_XSD = "http://www.mpi.nl/tools/elan/EAFv3.0.xsd"
HUMAN_TIERS = ("Seizure", "ILAE2017_type", "EEG_onset", "Notes", "Exclude")
MODEL_TIERS = ("Model_alarm", "Model_drivers")
SEIZURE_CV = ("seizure", "possible_seizure", "non_epileptic_event")
# Pick list for the ILAE2017_type tier (ILAE 2017 operational classification, short forms).
# The sample dataset's Label.xlsx uses P (focal) and PG (focal to bilateral). Free text is still allowed.
SEIZURE_TYPES = ("focal aware motor", "focal impaired awareness motor", "focal non-motor",
                 "focal to bilateral tonic-clonic", "generalized tonic-clonic", "generalized motor (other)",
                 "generalized non-motor (absence)", "unknown onset motor", "unknown onset non-motor",
                 "unclassified")
REGIONS = ("head", "l_arm", "r_arm", "l_hand", "r_hand", "l_leg", "r_leg", "overall")


# ── small helpers ─────────────────────────────────────────────
@contextlib.contextmanager
def atomic_write(out):
    """Yield a temp path next to `out`; on success it replaces `out` in one step (os.replace), so a crash
    or power cut leaves either the old file or the new one, never a half-written file."""
    out = Path(out)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    try:
        yield tmp
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def hms(s: float | None) -> str | None:
    if s is None:
        return None
    s = max(0.0, float(s))
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:06.3f}"


def paths_for(video) -> dict:
    v = Path(video)
    return dict(eaf=v.with_suffix(".eaf"), signals=v.with_name(v.stem + "_signals.csv"),
                tsconf=v.with_name(v.stem + "_tsconf.xml"), json=v.with_name(v.stem + ".annotation.json"))


def new_annotation(video, fps: float, duration_s: float, annotator: str = "", pat_id: str = "",
                   consent: str = "unknown", sha256: str | None = None) -> dict:
    """events: list of {clinical_onset_s, clinical_end_s, eeg_onset_s, event_class, seizure_type, notes}"""
    v = Path(video)
    return dict(schema=SCHEMA, video=dict(file=v.name, sha256=sha256 or sha256_file(v), fps=float(fps),
                                          duration_s=float(duration_s)),
                annotator=annotator, pat_id=pat_id, consent=consent, research_demo_not_diagnostic=True,
                events=[], notes=[], excludes=[], model=None)


def model_section(model_id: str, threshold: float, alarms: list[dict], per_second: list[dict]) -> dict:
    """alarms: [{start_s, end_s, peak_p, drivers}], per_second: rows for the signals CSV."""
    return dict(detector="A (pose features + LightGBM)", model_id=model_id, threshold=threshold,
                alarm_rule="smooth 5 s; alarm when >= threshold for >= 3 s; merge alarms < 30 s apart",
                alarms=alarms, per_second=per_second)


def alarms_from_seconds(rows: list[dict], merge_gap: float = 30.0) -> list[dict]:
    """rows: per-second dicts with t (second index), p_smooth, alarm (0/1), drivers (str).
    Returns merged alarm spans (alarm seconds covering [t, t+1))."""
    spans = []
    for r in rows:
        if not r.get("alarm"):
            continue
        t = float(r["t"])
        if spans and t - spans[-1]["end_s"] < merge_gap:
            sp = spans[-1]; sp["end_s"] = t + 1.0
        else:
            sp = dict(start_s=t, end_s=t + 1.0, peak_p=0.0, drivers="")
            spans.append(sp)
        if r["p_smooth"] >= sp["peak_p"]:
            sp["peak_p"] = float(r["p_smooth"]); sp["drivers"] = r.get("drivers", "")
    return spans


# ── EAF 3.0 writer ────────────────────────────────────────────
def write_eaf(ann: dict, video, out=None) -> Path:
    v = Path(video).resolve()
    p = paths_for(v)
    out = Path(out) if out else p["eaf"]
    XSI = "http://www.w3.org/2001/XMLSchema-instance"
    ET.register_namespace("xsi", XSI)
    now = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    root = ET.Element("ANNOTATION_DOCUMENT", {"AUTHOR": ann.get("annotator", ""), "DATE": now, "FORMAT": "3.0",
                                              "VERSION": "3.0", f"{{{XSI}}}noNamespaceSchemaLocation": EAF_XSD})
    hdr = ET.SubElement(root, "HEADER", MEDIA_FILE="", TIME_UNITS="milliseconds")
    ET.SubElement(hdr, "MEDIA_DESCRIPTOR", MEDIA_URL=v.as_uri(), MIME_TYPE="video/mp4",
                  RELATIVE_MEDIA_URL="./" + v.name)
    ET.SubElement(hdr, "LINKED_FILE_DESCRIPTOR", LINK_URL=link_url(p["signals"]),
                  RELATIVE_LINK_URL="./" + p["signals"].name, MIME_TYPE="text/plain", TIME_ORIGIN="0",
                  ASSOCIATED_WITH=v.as_uri())
    ET.SubElement(hdr, "LINKED_FILE_DESCRIPTOR", LINK_URL=link_url(p["tsconf"]),
                  RELATIVE_LINK_URL="./" + p["tsconf"].name, MIME_TYPE="text/xml")
    m = ann.get("model") or {}
    props = [("URN", f"urn:nl-mpi-tools-elan-eaf:{uuid.uuid4()}"), ("semisignals_schema", SCHEMA),
             ("video_sha256", ann["video"]["sha256"]), ("video_fps", f"{ann['video']['fps']:.6g}"),
             ("annotator", ann.get("annotator", "")), ("pat_id", ann.get("pat_id", "")),
             ("consent", ann.get("consent", "unknown")), ("research_demo_not_diagnostic", "true"),
             ("model_id", m.get("model_id", "")), ("model_threshold", str(m.get("threshold", ""))),
             ("alarm_rule", m.get("alarm_rule", ""))]
    slots, tiers = [], {}

    def slot(sec):
        slots.append(int(round(float(sec) * 1000)))
        return f"ts{len(slots)}"

    aid = [0]

    def ann_id():
        aid[0] += 1
        return f"a{aid[0]}"

    def tier(tid, lt, parent=None, human=True):
        attrs = {"TIER_ID": tid, "LINGUISTIC_TYPE_REF": lt}
        if parent:
            attrs["PARENT_REF"] = parent
        if human and ann.get("annotator"):
            attrs["ANNOTATOR"] = ann["annotator"]
        if ann.get("pat_id"):
            attrs["PARTICIPANT"] = ann["pat_id"]
        tiers[tid] = attrs, []
        return tiers[tid][1]

    def span(lst, t0, t1, val):
        i = ann_id()
        lst.append(("align", i, slot(t0), slot(max(t1, t0 + 0.001)), val))
        return i

    sz, ty = tier("Seizure", "seizure_event"), tier("ILAE2017_type", "symbolic", "Seizure")
    eeg, notes, excl = tier("EEG_onset", "default"), tier("Notes", "default"), tier("Exclude", "default")
    mal, mdr = tier("Model_alarm", "default", human=False), tier("Model_drivers", "symbolic", "Model_alarm", human=False)
    for k, e in enumerate(sorted(ann.get("events", []), key=lambda e: e.get("clinical_onset_s") or 0.0), 1):
        if e.get("clinical_onset_s") is None:
            continue
        end = e.get("clinical_end_s")
        end = end if end is not None else e["clinical_onset_s"] + 0.04   # onset only: one-frame span
        i = span(sz, e["clinical_onset_s"], end, e.get("event_class") or "seizure")
        if e.get("seizure_type"):
            ty.append(("ref", ann_id(), i, e["seizure_type"]))
        if e.get("eeg_onset_s") is not None:
            span(eeg, e["eeg_onset_s"], e["eeg_onset_s"] + 0.04, f"sz{k:02d}")
        if e.get("notes"):
            span(notes, e["clinical_onset_s"], end, e["notes"])
    for n in ann.get("notes", []):
        span(notes, n["start_s"], n.get("end_s") or n["start_s"] + 1.0, n["text"])
    for x in ann.get("excludes", []):
        span(excl, x["start_s"], x["end_s"], x.get("reason", "other"))
    for al in m.get("alarms", []):
        i = span(mal, al["start_s"], al["end_s"], f"{al['peak_p']:.2f}")
        if al.get("drivers"):
            mdr.append(("ref", ann_id(), i, al["drivers"]))
    props.append(("lastUsedAnnotationId", str(aid[0])))
    for k_, v_ in props:
        ET.SubElement(hdr, "PROPERTY", NAME=k_).text = v_
    to = ET.SubElement(root, "TIME_ORDER")
    for i, ms in enumerate(slots, 1):
        ET.SubElement(to, "TIME_SLOT", TIME_SLOT_ID=f"ts{i}", TIME_VALUE=str(ms))
    for tid, (attrs, lst) in tiers.items():
        t = ET.SubElement(root, "TIER", attrs)
        for a in lst:
            an = ET.SubElement(t, "ANNOTATION")
            if a[0] == "align":
                el = ET.SubElement(an, "ALIGNABLE_ANNOTATION", ANNOTATION_ID=a[1], TIME_SLOT_REF1=a[2], TIME_SLOT_REF2=a[3])
                ET.SubElement(el, "ANNOTATION_VALUE").text = a[4]
            else:
                el = ET.SubElement(an, "REF_ANNOTATION", ANNOTATION_ID=a[1], ANNOTATION_REF=a[2])
                ET.SubElement(el, "ANNOTATION_VALUE").text = a[3]
    ET.SubElement(root, "LINGUISTIC_TYPE", LINGUISTIC_TYPE_ID="default", TIME_ALIGNABLE="true", GRAPHIC_REFERENCES="false")
    ET.SubElement(root, "LINGUISTIC_TYPE", LINGUISTIC_TYPE_ID="seizure_event", TIME_ALIGNABLE="true",
                  GRAPHIC_REFERENCES="false", CONTROLLED_VOCABULARY_REF="seizure_event")
    ET.SubElement(root, "LINGUISTIC_TYPE", LINGUISTIC_TYPE_ID="symbolic", TIME_ALIGNABLE="false",
                  CONSTRAINTS="Symbolic_Association", GRAPHIC_REFERENCES="false")
    ET.SubElement(root, "LANGUAGE", LANG_ID="eng", LANG_DEF="http://cdb.iso.org/lg/CDB-00138502-001",
                  LANG_LABEL="English (eng)")
    for st, d in (("Time_Subdivision", "Time subdivision of parent annotation's time interval, no time gaps allowed within this interval"),
                  ("Symbolic_Subdivision", "Symbolic subdivision of a parent annotation. Annotations refering to the same parent are ordered"),
                  ("Symbolic_Association", "1-1 association with a parent annotation"),
                  ("Included_In", "Time alignable annotations within the parent annotation's time interval, gaps are allowed")):
        ET.SubElement(root, "CONSTRAINT", STEREOTYPE=st, DESCRIPTION=d)
    cv = ET.SubElement(root, "CONTROLLED_VOCABULARY", CV_ID="seizure_event")
    ET.SubElement(cv, "DESCRIPTION", LANG_REF="eng").text = "Clinical event class (onset to end)"
    for i, val in enumerate(SEIZURE_CV):
        e = ET.SubElement(cv, "CV_ENTRY_ML", CVE_ID=f"cveid_{val}")
        ET.SubElement(e, "CVE_VALUE", LANG_REF="eng").text = val
    ET.indent(root)
    with atomic_write(out) as tmp:
        ET.ElementTree(root).write(tmp, encoding="UTF-8", xml_declaration=True)
    return out


# ── EAF reader (human tiers only) ─────────────────────────────
class ShaMismatch(ValueError):
    pass


def read_eaf(eaf, video=None) -> dict:
    """Read human annotations back. Refuses a video_sha256 mismatch when *video* is given.
    Model_* tiers are ignored (they are regenerated, never treated as labels)."""
    root = ET.parse(eaf).getroot()
    props = {p.get("NAME"): (p.text or "") for p in root.iter("PROPERTY")}
    if video is not None:
        got = sha256_file(video)
        if props.get("video_sha256") and props["video_sha256"] != got:
            raise ShaMismatch(f"{eaf}: video_sha256 {props['video_sha256'][:12]}... != {got[:12]}... for {video}")
    ts = {s.get("TIME_SLOT_ID"): int(s.get("TIME_VALUE")) / 1000.0 for s in root.iter("TIME_SLOT")}
    tiers = {}
    for t in root.iter("TIER"):
        tid = t.get("TIER_ID")
        if tid.startswith("Model_"):
            continue
        rows = []
        for a in t.iter("ALIGNABLE_ANNOTATION"):
            rows.append(dict(id=a.get("ANNOTATION_ID"), start_s=ts[a.get("TIME_SLOT_REF1")],
                             end_s=ts[a.get("TIME_SLOT_REF2")], value=a.findtext("ANNOTATION_VALUE") or ""))
        for a in t.iter("REF_ANNOTATION"):
            rows.append(dict(id=a.get("ANNOTATION_ID"), ref=a.get("ANNOTATION_REF"),
                             value=a.findtext("ANNOTATION_VALUE") or ""))
        tiers[tid] = dict(annotator=t.get("ANNOTATOR"), rows=rows)
    types = {r["ref"]: r["value"] for r in tiers.get("ILAE2017_type", {}).get("rows", [])}
    sz = sorted(tiers.get("Seizure", {}).get("rows", []), key=lambda r: r["start_s"])
    eeg = {r["value"]: r["start_s"] for r in tiers.get("EEG_onset", {}).get("rows", [])}
    events = [dict(clinical_onset_s=r["start_s"], clinical_end_s=r["end_s"], event_class=r["value"],
                   seizure_type=types.get(r["id"], ""), eeg_onset_s=eeg.get(f"sz{k:02d}"))
              for k, r in enumerate(sz, 1)]
    return dict(properties=props, tiers=tiers, events=sorted(events, key=lambda e: e["clinical_onset_s"]))


# ── signals CSV + ELAN timeseries config ─────────────────────
# Format notes, checked against the ELAN 7.0 source (GPL-3.0; read only, nothing copied or bundled):
#  * CSVReader skips empty lines and lines starting with "#", so the header row is written as a "#" line.
#    A plain header row would be read as a data sample.
#  * Column indices in a _tsconf.xml are 0-based (the Configure Tracks dialog shows 1-based numbers and
#    subtracts 1 before storing them).
#  * The time column is parsed as seconds when the values contain a "." (e.g. 12.0000), else as ms.
#  * An unparseable cell becomes NaN and is drawn as a gap; "NaN" parses cleanly, a blank cell logs a
#    warning, and a blank last cell is dropped by the split and read as 0, so missing values are "NaN".
#  * The tsconf is only read through the .eaf's LINKED_FILE_DESCRIPTOR (name must end in _tsconf.xml).
#    Its tracksource source-url must be the exact string of the CSV's LINK_URL in the .eaf: ELAN relocates
#    moved linked files (same folder as the .eaf, then RELATIVE_LINK_URL) and rewrites a source-url only
#    when it equals an old LINK_URL. A relative source-url never matches, so no tracks are created.
#  * Loaded tracks are only drawn in a panel listed in the per-document preferences (<eafstem>.pfsx next
#    to the .eaf: TimeSeriesViewer.NumberOfPanels and TimeSeriesViewer.Panel-<n>); without it the viewer
#    shows one empty panel with a 0-100 ruler.
SIGNAL_COLS = ["t_s", "p_seizure", "p_smooth5s", "alarm", "driver_speed"] + [f"speed_{r}" for r in REGIONS]
TS_PROVIDER = "mpi.eudico.client.annotator.timeseries.csv.CSVServiceProvider"


def link_url(path) -> str:
    """The file URL used for LINK_URL in the .eaf and for source-url in the _tsconf.xml (must be identical)."""
    return Path(path).resolve().as_uri()


def write_signals(rows: list[dict], video) -> Path:
    """rows: per-second dicts with keys of SIGNAL_COLS (t_s = end of the 1 s window, seconds).
    The header line starts with "#" (ELAN skips it); missing values are written as NaN.
    Read back with e.g. pandas.read_csv(path) and strip the "#" from the first column name."""
    out = paths_for(video)["signals"]
    with atomic_write(out) as tmp, open(tmp, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["#" + SIGNAL_COLS[0]] + SIGNAL_COLS[1:])
        for r in rows:
            vals = []
            for c in SIGNAL_COLS:
                x = r.get(c)
                if x is None or x == "" or (isinstance(x, float) and x != x):
                    vals.append("NaN")
                elif c == "t_s":
                    vals.append(f"{float(x):.4f}")      # always has a "." so ELAN reads seconds
                else:
                    vals.append(f"{x:.4f}" if isinstance(x, float) else str(x))
            w.writerow(vals)
    return out


SPEED_MAX = 1.5   # fixed y range for speed lanes (ELAN's CSV loader ignores detect-range); clips above it
_TRACKS = [("p_smooth5s", "Detector A 5 s smoothed probability (alarm threshold in header)", "p", 0, 1, "0,220,0"),
           ("p_seizure", "Detector A raw per-second probability", "p", 0, 1, "0,110,0"),
           ("alarm", "Detector A alarm state (0/1)", "", 0, 1, "255,40,40"),
           ("driver_speed", "mean speed of the top-driver regions (gap when none)", "body/s", 0, SPEED_MAX, "255,190,0")] + \
          [(f"speed_{r}", f"{r} speed", "body/s", 0, SPEED_MAX, c) for r, c in
           zip(REGIONS, ("255,255,0", "0,200,0", "0,0,200", "128,255,0", "100,100,255", "0,200,0", "0,0,200", "180,180,180"))]
# Timeseries Viewer panels (track names per panel), written to <stem>.pfsx
TS_PANELS = [["p_smooth5s", "p_seizure", "alarm"], ["driver_speed"], [f"speed_{r}" for r in REGIONS]]


def write_tsconf(video, signals_path=None) -> Path:
    """ELAN Timeseries Viewer config (<stem>_tsconf.xml): one CSV source, 0-based columns, time column 0
    (t_s, seconds), sample type "Discontinuous Rate" (explicit time stamps)."""
    p = paths_for(video)
    sig = Path(signals_path or p["signals"])
    root = ET.Element("timeseries", date=_dt.datetime.now().astimezone().isoformat(timespec="seconds"), version="1.0")
    src = ET.SubElement(root, "tracksource", {"sample-type": "Discontinuous Rate", "source-url": link_url(sig),
                                               "time-column": str(SIGNAL_COLS.index("t_s"))})
    ET.SubElement(src, "property", key="provider", value=TS_PROVIDER)
    for name, desc, units, lo, hi, col in _TRACKS:
        tr = ET.SubElement(src, "track", derivative="0", name=name)
        ET.SubElement(tr, "property", key="detect-range", value="false")
        sp = ET.SubElement(tr, "sample-position")
        ET.SubElement(sp, "pos", col=str(SIGNAL_COLS.index(name)), row="0")
        ET.SubElement(tr, "description").text = desc
        ET.SubElement(tr, "units").text = units
        ET.SubElement(tr, "range", max=f"{hi:.1f}", min=f"{lo:.1f}")
        ET.SubElement(tr, "color").text = col
    ET.indent(root)
    with atomic_write(p["tsconf"]) as tmp:
        ET.ElementTree(root).write(tmp, encoding="UTF-8", xml_declaration=True)
    return p["tsconf"]


def write_pfsx(video, overwrite: bool = False) -> Path | None:
    """Per-document ELAN preferences (<stem>.pfsx next to the .eaf) that put the CSV tracks into
    Timeseries Viewer panels. ELAN rewrites this file with the user's own layout, so an existing one is
    kept unless overwrite=True. Returns None when kept."""
    out = paths_for(video)["eaf"].with_suffix(".pfsx")
    if out.exists() and not overwrite:
        return None
    XSI = "http://www.w3.org/2001/XMLSchema-instance"
    ET.register_namespace("xsi", XSI)
    root = ET.Element("preferences", {"version": "1.1",
                                      f"{{{XSI}}}noNamespaceSchemaLocation": "http://www.mpi.nl/tools/elan/Prefs_v1.1.xsd"})
    pref = ET.SubElement(root, "pref", key="TimeSeriesViewer.NumberOfPanels")
    ET.SubElement(pref, "Int").text = str(len(TS_PANELS))
    for i, names in enumerate(TS_PANELS, 1):
        lst = ET.SubElement(root, "prefList", key=f"TimeSeriesViewer.Panel-{i}")
        for n in names:
            ET.SubElement(lst, "String").text = n
    ET.indent(root)
    with atomic_write(out) as tmp:
        ET.ElementTree(root).write(tmp, encoding="UTF-8", xml_declaration=True)
    return out


def write_json(ann: dict, video) -> Path:
    out = paths_for(video)["json"]
    a = dict(ann)
    a["events"] = [dict(e, clinical_onset_hms=hms(e.get("clinical_onset_s")), clinical_end_hms=hms(e.get("clinical_end_s")),
                        eeg_onset_s=e.get("eeg_onset_s"), eeg_onset_hms=hms(e.get("eeg_onset_s")))
                   for e in ann.get("events", [])]
    a["saved"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    if a.get("model"):
        a["model"] = {k: v for k, v in a["model"].items() if k != "per_second"}   # per-second data: see _signals.csv
    with atomic_write(out) as tmp, open(tmp, "w") as f:
        json.dump(a, f, indent=1)
    return out


def save_all(ann: dict, video) -> dict:
    """Write .eaf, _signals.csv, _tsconf.xml, .pfsx (only if absent) and .annotation.json next to the video."""
    out = {}
    if ann.get("model") and ann["model"].get("per_second"):
        out["signals"] = write_signals(ann["model"]["per_second"], video)
        out["tsconf"] = write_tsconf(video)
        pf = write_pfsx(video)
        if pf:
            out["pfsx"] = pf
    out["eaf"] = write_eaf(ann, video)
    out["json"] = write_json(ann, video)
    return out
