"""ELAN Timeseries Viewer files (_signals.csv, _tsconf.xml, .pfsx): format rules from the ELAN 7.0
CSV reader / tsconf loader (see the notes in src/annotation.py)."""
import math
import xml.etree.ElementTree as ET

from src import annotation as A


def _bundle(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    ann = A.new_annotation(video, fps=30.0, duration_s=4.0, annotator="test", pat_id="p0", consent="self")
    ann["events"] = [dict(clinical_onset_s=1.0, clinical_end_s=3.0, event_class="seizure", seizure_type="",
                          eeg_onset_s=None, notes="")]
    rows = []
    for t in range(1, 5):
        r = {c: 0.1 * t for c in A.SIGNAL_COLS}
        r["t_s"] = float(t); r["alarm"] = int(t >= 3)
        r["driver_speed"] = None if t < 3 else 0.5          # blank cells in the middle
        r["speed_overall"] = float("nan") if t == 4 else 0.2  # missing last cell
        rows.append(r)
    alarms = [dict(start_s=3.0, end_s=5.0, peak_p=0.97, drivers="right hand speed")]
    ann["model"] = A.model_section("m", 0.95, alarms, rows)
    out = A.save_all(ann, video)
    return video, rows, out


def test_signals_csv_header_is_comment_and_no_blank_cells(tmp_path):
    _, rows, out = _bundle(tmp_path)
    lines = out["signals"].read_text().splitlines()
    assert lines[0] == "#" + ",".join(A.SIGNAL_COLS)
    data = [ln.split(",") for ln in lines[1:] if ln and not ln.startswith("#")]
    assert len(data) == len(rows)
    for cells in data:
        assert len(cells) == len(A.SIGNAL_COLS)             # String.split would drop a blank last cell
        assert all(c != "" for c in cells)
        assert "." in cells[0]                               # seconds, not ms
        float(cells[0])
    assert data[0][A.SIGNAL_COLS.index("driver_speed")] == "NaN"
    assert data[-1][A.SIGNAL_COLS.index("speed_overall")] == "NaN"


def test_tsconf_matches_eaf_link_and_uses_0_based_columns(tmp_path):
    _, rows, out = _bundle(tmp_path)
    eaf = ET.parse(out["eaf"]).getroot()
    links = {lfd.get("RELATIVE_LINK_URL"): lfd.get("LINK_URL") for lfd in eaf.iter("LINKED_FILE_DESCRIPTOR")}
    assert links["./clip_tsconf.xml"].endswith("_tsconf.xml")      # ELAN only loads a tsconf by this suffix
    root = ET.parse(out["tsconf"]).getroot()
    assert root.tag == "timeseries"
    srcs = root.findall("tracksource")
    assert len(srcs) == 1
    src = srcs[0]
    assert src.get("source-url") == links["./clip_signals.csv"]   # exact string, not relative
    assert src.get("source-url").startswith("file:///")
    assert src.get("sample-type") == "Discontinuous Rate"
    assert src.get("time-column") == "0"
    assert src.find("property[@key='provider']").get("value") == A.TS_PROVIDER
    header = out["signals"].read_text().splitlines()[0].lstrip("#").split(",")
    data = [ln.split(",") for ln in out["signals"].read_text().splitlines() if ln and not ln.startswith("#")]
    names = []
    for tr in src.findall("track"):
        name = tr.get("name"); names.append(name)
        pos = tr.findall("sample-position/pos")
        assert len(pos) == 1 and pos[0].get("row") == "0"
        col = int(pos[0].get("col"))
        assert header[col] == name                                 # 0-based index into the row
        rng = tr.find("range")
        assert float(rng.get("min")) < float(rng.get("max"))
        # read the column the way ELAN does: time from column 0 (seconds), value parsed as float
        for r, cells in zip(rows, data):
            assert float(cells[0]) * 1000 == r["t_s"] * 1000
            v, want = float(cells[col]), r[name]
            if want is None or (isinstance(want, float) and math.isnan(want)):
                assert math.isnan(v)
            else:
                assert abs(v - want) < 1e-4
    assert len(set(names)) == len(names)


def test_pfsx_puts_every_track_in_a_panel_and_is_not_overwritten(tmp_path):
    video, _, out = _bundle(tmp_path)
    pf = out["pfsx"]
    assert pf.name == "clip.pfsx" and pf.parent == out["eaf"].parent
    root = ET.parse(pf).getroot()
    assert root.tag == "preferences" and root.get("version") == "1.1"
    n = int(root.find("pref[@key='TimeSeriesViewer.NumberOfPanels']/Int").text)
    panels = [[s.text for s in root.find(f"prefList[@key='TimeSeriesViewer.Panel-{i}']").findall("String")]
              for i in range(1, n + 1)]
    tracks = {tr.get("name") for tr in ET.parse(out["tsconf"]).getroot().iter("track")}
    assert sorted(sum(panels, [])) == sorted(tracks)
    pf.write_text("<preferences version='1.1'/>")                  # ELAN's own saved layout
    assert A.write_pfsx(video) is None
    assert pf.read_text() == "<preferences version='1.1'/>"
