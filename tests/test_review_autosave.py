"""Annotation tool: every change is written to the .eaf straight away (scripted keys, no window),
writes are atomic, and an .eaf made for a different video is refused."""
import os
import time

import cv2
import numpy as np
import pytest

from src import annotation as A
from src import review_video as R


def _clip(tmp_path, seconds=12, fps=10):
    video = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(seconds * fps):
        w.write(np.full((48, 64, 3), (i * 2) % 255, np.uint8))
    w.release()
    rendered, scores = R.outputs_for(video)
    rendered.write_bytes(video.read_bytes())          # stand-in for the rendered overlay video
    cols = ["t", "p_video_a", "p_video_a_smooth", "alarm_a", "drivers", "driver_speed"] + [f"speed_{g}" for g in A.REGIONS]
    with open(scores, "w") as f:
        f.write(",".join(cols) + "\n")
        for t in range(seconds):
            p = 0.97 if 4 <= t <= 7 else 0.1
            f.write(",".join([str(t), f"{p}", f"{p}", str(int(p > 0.95)), "right hand speed=0.20|", "0.3"]
                             + ["0.1"] * len(A.REGIONS)) + "\n")
    R.write_stamp(R.parse_args(["--detector-a", "none.txt"]), video)    # made with the settings the tests use
    later = time.time() + 5
    for p in (rendered, scores, R.stamp_path(video)):                  # newer than the video: no reprocessing
        os.utime(p, (later, later))
    return video


def test_autosave_after_every_change(tmp_path, monkeypatch):
    video = _clip(tmp_path)
    eaf = A.paths_for(video)["eaf"]
    snaps = []
    real = A.save_all

    def spy(ann, v):
        out = real(ann, v)
        d = A.read_eaf(eaf, video)
        snaps.append([(e["clinical_onset_s"], e.get("clinical_end_s"), e.get("seizure_type", "")) for e in d["events"]])
        return out
    monkeypatch.setattr(A, "save_all", spy)
    keys = "3:o;8.5:e;9:t:focal aware motor;9.5:n:arm jerks;10:x;11:x;11.5:u;11.6:o;11.7:u"
    R.main(["--input", str(video), "--detector-a", str(tmp_path / "none.txt"), "--keys", keys])
    # x (open) does not change anything; every other key does and is saved at once
    assert len(snaps) == 8
    # an event without an end yet is written with a short placeholder span, so compare onsets there
    assert [(on, ty) for on, _, ty in snaps[0]] == [(3.0, "")]
    assert snaps[1] == [(3.0, 8.5, "")]
    assert snaps[2] == [(3.0, 8.5, "focal aware motor")]
    assert [(on, ty) for on, _, ty in snaps[-2]] == [(3.0, "focal aware motor"), (11.6, "")]
    assert snaps[-1] == [(3.0, 8.5, "focal aware motor")]
    assert not list(tmp_path.glob(".*.tmp"))              # no temp files left behind
    sig = A.paths_for(video)["signals"]
    assert sig.exists() and A.paths_for(video)["json"].exists()
    import json
    j = json.loads(A.paths_for(video)["json"].read_text())
    assert [n["text"] for n in j["notes"]] == ["arm jerks"] and j["excludes"] == []   # after the event: timed note; u removed the span


def test_atomic_write_keeps_old_file_on_crash(tmp_path):
    f = tmp_path / "x.eaf"
    f.write_text("old")
    with pytest.raises(RuntimeError):
        with A.atomic_write(f) as tmp:
            tmp.write_text("half writ")
            raise RuntimeError("crash while writing")
    assert f.read_text() == "old"
    assert list(tmp_path.iterdir()) == [f]
    with A.atomic_write(f) as tmp:
        tmp.write_text("new")
    assert f.read_text() == "new" and list(tmp_path.iterdir()) == [f]


def test_reload_refuses_eaf_of_another_video(tmp_path):
    video = _clip(tmp_path)
    R.main(["--input", str(video), "--detector-a", str(tmp_path / "none.txt"), "--keys", "2:o"])
    other = tmp_path / "other"; other.mkdir()
    v2 = _clip(other, seconds=6)
    os.replace(A.paths_for(video)["eaf"], A.paths_for(v2)["eaf"])   # .eaf next to a different video
    with pytest.raises(A.ShaMismatch):
        R.main(["--input", str(v2), "--detector-a", str(tmp_path / "none.txt"), "--keys", "1:o"])


def test_seizure_type_pick_list_without_tk(monkeypatch):
    monkeypatch.setattr(R, "_root", lambda: None)
    monkeypatch.setattr(R, "_tty", lambda: False)
    assert R.ask_choice("t", "p", list(A.SEIZURE_TYPES)) is None      # no terminal prompt when launched from the GUI
    assert R.ask("t", "p", "AL") == "AL"


def test_stale_render_is_redone(tmp_path):
    """A render made with other settings or an older overlay version is not reused."""
    video = _clip(tmp_path)
    a = R.parse_args(["--detector-a", "none.txt"])
    assert R.render_is_fresh(a, video)
    assert not R.render_is_fresh(R.parse_args(["--detector-a", "none.txt", "--vsvig", "m.xml"]), video)
    assert not R.render_is_fresh(R.parse_args(["--detector-a", "none.txt", "--no-hands"]), video)
    R.stamp_path(video).write_text('{"render_version": 1}')
    later = time.time() + 6
    os.utime(R.stamp_path(video), (later, later))
    assert not R.render_is_fresh(a, video)
    R.stamp_path(video).unlink()                      # renders from before the stamp existed
    assert not R.render_is_fresh(a, video)


def test_display_width_fits_screen():
    class P:
        def width(self): return 420
        def screen_width(self): return 1920
        def screen_height(self): return 1080
    rw, rh = 1600, 540
    w = R.display_width(rw, rh, P(), area=(1920, 1032))      # 1920x1080 at 150 %, DPI aware, taskbar
    assert w + 420 + 24 <= 1920 and w <= rw
    assert rh * w / rw + R.BAR_H + 2 * R.LEGEND_H <= 1032
    assert R.display_width(rw, rh, P(), area=(3840, 2100)) == rw   # never upscaled
