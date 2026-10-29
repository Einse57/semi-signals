"""Annotation modes (body / face / both), config-driven label sets, and backward compatibility with
annotation files written before modes existed (fixture written by the 4d4445f annotation.write_eaf)."""
import json
import re
from pathlib import Path

import pytest

from src import annotation as A
from src import labels as LB
from src import review_video as R
from tests.test_review_autosave import _clip

FIXTURE = Path(__file__).parent / "fixtures" / "annotation_v1_body.eaf"
FIXTURE_VIDEO = "/tmp/semi_signals_fixture/clip.mp4"      # path the fixture was written for


def _fixture_ann(video):
    """The annotation the fixture was made from (see the PR description for the generator)."""
    ann = A.new_annotation(video, 25.0, 60.0, annotator="XX", pat_id="p00", consent="public",
                           sha256=A.sha256_file(video))
    ann["events"] = [dict(clinical_onset_s=5.0, clinical_end_s=20.5, eeg_onset_s=3.2, event_class="seizure",
                          seizure_type="focal aware motor", notes="left arm"),
                     dict(clinical_onset_s=40.0, clinical_end_s=None, eeg_onset_s=None, event_class="possible_seizure",
                          seizure_type="", notes="")]
    ann["notes"] = [dict(start_s=30.0, end_s=31.0, text="nurse in view")]
    ann["excludes"] = [dict(start_s=50.0, end_s=55.0, reason="other")]
    ann["model"] = A.model_section("detector_a_all", 0.95, [dict(start_s=6.0, end_s=12.0, peak_p=0.98, drivers="left arm")], [])
    return ann


def _norm(xml: str) -> str:
    xml = re.sub(r'DATE="[^"]*"', 'DATE=""', xml)
    return re.sub(r"urn:nl-mpi-tools-elan-eaf:[0-9a-f-]+", "urn:x", xml)


# ── label sets ───────────────────────────────────────────────
def test_label_sets_body_matches_existing_lists_and_face_is_minimal():
    c = LB.load_all()
    assert tuple(c["body"]["term"]["choices"]) == A.SEIZURE_TYPES
    assert tuple(c["body"]["event_class"]) == A.SEIZURE_CV
    assert c["face"]["timing_fields"] == c["body"]["timing_fields"] == list(LB.TIMING_FIELDS)
    assert c["face"]["term"]["choices"] == [] and c["face"]["event_class"] == []     # no face terms until vetted
    assert c["face"]["side"] == ["L", "R", "both"] and c["face"]["note"] and c["body"]["side"] == []


def test_label_set_drop_in_and_validation(tmp_path):
    for m in LB.MODES:
        (tmp_path / f"{m}.json").write_text(Path(LB.LABEL_DIR / f"{m}.json").read_text())
    face = json.loads((tmp_path / "face.json").read_text())
    face["term"].update(source="test list", choices=["term A", "term B"])
    (tmp_path / "face.json").write_text(json.dumps(face))
    assert LB.load("face", tmp_path)["term"]["choices"] == ["term A", "term B"]
    face["timing_fields"] = ["clinical_onset_s"]
    (tmp_path / "face.json").write_text(json.dumps(face))
    with pytest.raises(ValueError):
        LB.load("face", tmp_path)
    with pytest.raises(ValueError):
        LB.load("hands", tmp_path)


# ── backward compatibility ───────────────────────────────────
def test_old_body_eaf_reads_as_body_events():
    d = A.read_eaf(FIXTURE)
    assert [e["mode"] for e in d["events"]] == ["body", "body"]
    e0, e1 = d["events"]
    assert (e0["clinical_onset_s"], e0["clinical_end_s"], e0["eeg_onset_s"], e0["event_class"], e0["seizure_type"]) == \
        (5.0, 20.5, 3.2, "seizure", "focal aware motor")
    assert (e1["clinical_onset_s"], e1["event_class"], e1["eeg_onset_s"]) == (40.0, "possible_seizure", None)
    assert not any(t in d["tiers"] for t in A.FACE_TIERS)


def test_body_only_eaf_is_unchanged_by_modes():
    v = Path(FIXTURE_VIDEO)
    v.parent.mkdir(parents=True, exist_ok=True)
    v.write_bytes(b"fixture video bytes")
    for mode in (None, "body"):                 # events without a mode (old) or explicitly body
        ann = _fixture_ann(v)
        if mode:
            ann["events"] = [dict(e, mode=mode) for e in ann["events"]]
        out = A.write_eaf(ann, v, out=v.with_name(f"new_{mode}.eaf"))
        assert _norm(out.read_text()) == _norm(FIXTURE.read_text())


def test_events_without_mode_default_to_body():
    assert A.event_mode({}) == "body" and A.event_mode({"mode": "face"}) == "face"
    evs = [dict(clinical_onset_s=1.0), dict(mode="face", clinical_onset_s=2.0)]
    assert len(A.events_of(evs, "body")) == 1 and len(A.events_of(evs, "face")) == 1


# ── face events in EAF and JSON ──────────────────────────────
def test_face_and_body_events_roundtrip_separately(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    ann = A.new_annotation(video, 25.0, 60.0, annotator="XX")
    ann["events"] = [dict(mode="body", clinical_onset_s=5.0, clinical_end_s=9.0, eeg_onset_s=4.0, event_class="seizure",
                          seizure_type="", notes=""),
                     dict(mode="face", clinical_onset_s=2.0, clinical_end_s=6.0, eeg_onset_s=1.5, side="L",
                          face_term="", notes="left mouth corner"),
                     dict(mode="face", clinical_onset_s=7.0, clinical_end_s=None, eeg_onset_s=None, side=None,
                          face_term="", notes="")]
    A.write_eaf(ann, video)
    d = A.read_eaf(A.paths_for(video)["eaf"], video)
    body, face = A.events_of(d, "body"), A.events_of(d, "face")
    assert [(e["clinical_onset_s"], e["eeg_onset_s"]) for e in body] == [(5.0, 4.0)]     # sz01 counts body only
    assert [(e["clinical_onset_s"], e["side"], e["eeg_onset_s"]) for e in face] == [(2.0, "L", 1.5), (7.0, None, None)]
    assert "Face_event" in d["tiers"] and "Face_term" not in d["tiers"]                  # no term tier without terms
    assert [r["value"] for r in d["tiers"]["Seizure"]["rows"]] == ["seizure"]           # face never in Seizure
    j = json.loads(A.write_json(ann, video).read_text())
    assert [e["mode"] for e in j["events"]] == ["body", "face", "face"]


def test_compare_uses_body_events_only():
    alarms = [dict(start_s=10.0, end_s=20.0, peak_p=0.99, drivers="")]
    c = R.compare([dict(mode="face", clinical_onset_s=10.0), dict(clinical_onset_s=12.0)], alarms)
    assert len(c["events"]) == 1 and c["events"][0]["clinical_onset_s"] == 12.0


# ── review tool modes (scripted keys, no window) ─────────────
def _clip_for_mode(tmp_path, mode):
    video = _clip(tmp_path)
    R.write_stamp(R.parse_args(["--detector-a", "none.txt", "--mode", mode]), video)
    import os, time
    later = time.time() + 5
    os.utime(R.stamp_path(video), (later, later))
    return video


def test_render_settings_unchanged_for_body_and_face_adds_flag():
    assert "face" not in R.render_settings(R.parse_args(["--detector-a", "x.txt"]))
    assert R.render_settings(R.parse_args(["--detector-a", "x.txt", "--mode", "both"]))["face"] is True


def test_face_mode_keys(tmp_path):
    video = _clip_for_mode(tmp_path, "face")
    keys = "2:o;4:e;4.5:d:R;5:t:anything;5.5:n:mouth"
    R.main(["--input", str(video), "--detector-a", str(tmp_path / "none.txt"), "--mode", "face", "--keys", keys])
    d = A.read_eaf(A.paths_for(video)["eaf"], video)
    assert [(e["mode"], e["clinical_onset_s"], e["clinical_end_s"], e.get("side"), e.get("face_term")) for e in d["events"]] == \
        [("face", 2.0, 4.0, "R", "")]          # t is off while labels/face.json has no term list
    assert "Seizure" not in d["tiers"] or not d["tiers"]["Seizure"]["rows"]
    assert (video.with_name(video.stem + "_review_frame.png")).exists()


def test_both_mode_keys_keep_body_and_face_apart(tmp_path):
    video = _clip_for_mode(tmp_path, "both")
    keys = "1:o;3:e;3.2:t:focal aware motor;5:f;6:d:both;7:e;8:d:sideways"
    R.main(["--input", str(video), "--detector-a", str(tmp_path / "none.txt"), "--mode", "both", "--keys", keys])
    d = A.read_eaf(A.paths_for(video)["eaf"], video)
    assert [(e["mode"], e["clinical_onset_s"], e["clinical_end_s"]) for e in d["events"]] == \
        [("body", 1.0, 3.0), ("face", 5.0, 7.0)]
    assert A.events_of(d, "body")[0]["seizure_type"] == "focal aware motor"
    assert A.events_of(d, "face")[0]["side"] == "both"                 # an invalid side is refused
    j = json.loads(A.paths_for(video)["json"].read_text())
    assert [e["mode"] for e in j["events"]] == ["body", "face"]


def test_body_mode_ignores_face_keys(tmp_path):
    video = _clip(tmp_path)
    R.main(["--input", str(video), "--detector-a", str(tmp_path / "none.txt"), "--keys", "1:f;2:d:L;3:o"])
    d = A.read_eaf(A.paths_for(video)["eaf"], video)
    assert [(e["mode"], e["clinical_onset_s"]) for e in d["events"]] == [("body", 3.0)]


def test_launcher_passes_annotation_mode(tmp_path):
    from src import launcher as L
    s = dict(video=str(tmp_path / "v.mp4"), device="CPU", annotator="AL")
    assert "--mode" not in L.annotate_cmd(s, tmp_path, tmp_path)                       # body: as before
    for m in ("face", "both"):
        assert R.parse_args(L.annotate_cmd(dict(s, annot_mode=m), tmp_path, tmp_path)[4:]).mode == m


def test_face_type_key_hidden_until_terms_exist():
    assert ("Type...", "t") not in R.panel_buttons("face", face_terms=False)
    assert ("Type...", "t") in R.panel_buttons("face", face_terms=True)
    assert ("Type...", "t") in R.panel_buttons("both") and ("Side...", "d") in R.panel_buttons("both")
    assert R.panel_buttons("body") == R.PANEL_BUTTONS                          # body panel unchanged
    assert R.review_keys("body") == [list(r) for r in R.REVIEW_KEYS]
    assert all(k != "t" for k, _ in R.review_keys("face")[0])
