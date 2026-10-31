"""Fill fallback (YuNet boxes for faces the landmarker missed) and the unmasked-face check that decides
whether a fill export is marked shareable (src/fill_guard.py, src/segment_export.py). Synthetic frames and
mocked detectors only."""
import numpy as np
import cv2
import pytest

from src import fill_guard as G
from src import segment_export as X
from src import segment_store as S
from src import deid_options as O


def _ring(cx, cy, r=12):
    a = np.linspace(0, 2 * np.pi, 478, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a), np.zeros(478)], 1).astype(np.float32)


def _row(x, y, w, h, score=0.9):
    return np.array([x, y, w, h] + [0.0] * 10 + [score], np.float32)


class _Fixed:
    """Mock YuNet: the same rows for every frame."""
    def __init__(self, rows):
        self.rows = rows

    def detect(self, frame):
        return list(self.rows)


class _Textured:
    """Mock YuNet: a face at each box whose pixels are not flat (fill makes them flat)."""
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, frame):
        out = []
        for x, y, w, h in self.boxes:
            if frame[y:y + h, x:x + w].reshape(-1, 3).std(0).max() > 8:
                out.append(_row(x, y, w, h))
        return out


class _Bright:
    """Mock YuNet: one face on bright frames only (survives the fill)."""
    def detect(self, frame):
        return [_row(2, 2, 10, 10)] if frame.mean() > 150 else []


def test_pad_box_clips_to_frame():
    assert G.pad_box(_row(10, 10, 20, 10), (100, 100), pad=0.2) == (6, 8, 34, 22)
    assert G.pad_box(_row(-5, 90, 20, 20), (100, 100), pad=0.2) == (0, 86, 19, 100)


def test_fallback_fills_uncovered_yunet_faces_only():
    rng = np.random.default_rng(1)
    f = rng.integers(0, 255, (120, 200, 3), np.uint8)
    lm = _ring(40, 60)
    rows = [_row(30, 50, 20, 20), _row(140, 30, 30, 30), _row(150, 90, 20, 20)]   # 1 covered, 2 missed faces
    g = G.FillGuard(detector=_Fixed(rows), pad=0.2)
    bx = g.boxes(f, X.landmark_mask(f.shape, [lm]))
    assert bx == [(134, 24, 176, 66), (146, 86, 174, 114)] and g.n_boxes == 2 and g.box_frames == 1
    out = X.mask_faces(f, lm, others=[], style="fill", boxes=bx)
    colour = out[60, 40]
    for x0, y0, x1, y1 in bx:                                         # same solid colour as the landmark fill
        assert (out[y0:y1, x0:x1].reshape(-1, 3) == colour).all()
    assert np.array_equal(out[5, 100], f[5, 100])                     # elsewhere unchanged
    assert np.array_equal(X.mask_faces(f, lm, style="blur", boxes=bx)[30, 150], X.blur_face(f, lm)[30, 150])
    only = X.mask_faces(f, None, missing="none", boxes=bx[:1])        # no landmark face at all: boxes still filled
    assert np.ptp(only[24:66, 134:176].reshape(-1, 3), 0).max() == 0 and np.array_equal(only[5, 5], f[5, 5])


def test_check_lists_unmasked_frames_and_blocks_shareable():
    g = G.FillGuard(detector=_Bright())
    for i in range(6):
        g.check(np.full((20, 20, 3), 200 if i in (1, 4) else 50, np.uint8), i)
    r = g.result()
    assert r["Ran"] and not r["Clean"] and r["UnmaskedFrames"] == [1, 4] and r["FramesChecked"] == 6
    assert "NOT SHAREABLE" in g.summary("x") and "1, 4" in g.summary("x")
    g.reset()
    g.check(np.zeros((20, 20, 3), np.uint8), 0)
    assert g.result()["Clean"] and "shareable" in g.summary()


def test_missing_model_runs_but_is_not_shareable(tmp_path):
    g = G.FillGuard(tmp_path / "none.onnx")
    assert not g.available and "fetch_models.py --face-check" in g.problem
    assert g.boxes(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), np.uint8)) == []
    r = g.result()
    assert not r["Ran"] and not r["Clean"] and "landmarks only" in r["Reason"]


# ── through segment_export.main with a mocked landmarker ─────
class _Pre:
    mode = "none"

    def __init__(self, lm):
        self.lm = lm

    def run(self, frame):
        return dict(landmarks=self.lm, blendshapes=np.zeros(52, np.float32), score=0.9)

    def reset(self):
        pass


class _Est:
    def close(self):
        pass


def _clip(tmp_path, bright=(), face=(100, 30, 40, 40)):
    x, y, fw, fh = face
    v = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 96))
    rng = np.random.default_rng(0)
    for i in range(8):
        f = np.full((96, 160, 3), 210 if i in bright else 60, np.uint8)
        f[y:y + fh, x:x + fw] = rng.integers(0, 255, (fh, fw, 3), np.uint8)   # a "face" for the mock YuNet
        w.write(f)
    w.release()
    return v


def _run(tmp_path, monkeypatch, det, extra=(), model=None):
    monkeypatch.setattr(X, "make_preprocessor", lambda a: (_Est(), _Pre(_ring(40, 48)), None, None))
    if det is not None:
        monkeypatch.setattr(X, "make_fill_guard", lambda a: G.FillGuard(detector=det))
    args = ["--input", str(tmp_path / "v.mp4"), "--whole", "--out", str(tmp_path / "o"), "--raw-root",
            str(tmp_path / "r"), "--deid", "fill", "--no-bystanders"] + list(extra)
    if model is not None:
        args += ["--yunet-model", str(model)]
    rc = X.main(args)
    j = S.read_json(next((tmp_path / "o").rglob("*desc-fill_video.json")))
    return rc, j


def test_export_clean_is_shareable(tmp_path, monkeypatch, capsys):
    _clip(tmp_path)
    rc, j = _run(tmp_path, monkeypatch, _Textured([(100, 30, 40, 40)]), ["--shareable"])
    c = j["UnmaskedFaceCheck"]
    assert rc == 0 and j["Shareable"] is True and j["Deidentified"] is True
    assert c["Clean"] and c["UnmaskedFrames"] == [] and c["FallbackBoxesFilled"] == 8 and c["FramesChecked"] == 8
    assert c["EncodedOutputCheck"]["Clean"] and c["EncodedOutputCheck"]["FramesChecked"] == 8
    out = capsys.readouterr().out
    assert "no unmasked face: shareable" in out and "encoded file re-check: 8 frames, 0 flagged" in out


def test_export_with_unmasked_frames_is_not_shareable(tmp_path, monkeypatch, capsys):
    _clip(tmp_path, bright=(2, 5))
    rc, j = _run(tmp_path, monkeypatch, _Bright(), ["--shareable"])
    c = j["UnmaskedFaceCheck"]
    assert rc == 3 and j["Shareable"] is False and j["Deidentified"] is False
    assert c["UnmaskedFrames"] == [2, 5] and "2 fill output frame" in j["ShareableBlockedReason"]
    out = capsys.readouterr().out
    assert "NOT SHAREABLE" in out and "2, 5" in out
    rc2, j2 = _run(tmp_path, monkeypatch, _Bright())                  # not requested for sharing: rc 0, still flagged
    assert rc2 == 0 and j2["Shareable"] is False


def test_export_without_yunet_model(tmp_path, monkeypatch, capsys):
    _clip(tmp_path)
    rc, j = _run(tmp_path, monkeypatch, None, ["--shareable"], model=tmp_path / "missing.onnx")
    assert rc == 3 and j["Shareable"] is False and not j["UnmaskedFaceCheck"]["Ran"]
    assert "fetch_models.py --face-check" in capsys.readouterr().out


def test_ui_flags_missing_yunet(tmp_path):
    assert O.fill_check_problems(tmp_path) and "cannot be marked shareable" in O.fill_check_problems(tmp_path)[0]
    (tmp_path / G.MODEL_NAME).write_bytes(b"x")
    assert O.fill_check_problems(tmp_path) == []
    assert "checked with YuNet" in O.label_text("fill")


def test_second_pass_unit():
    g = G.FillGuard(detector=_Textured([(4, 4, 10, 10)]))
    rng = np.random.default_rng(2)
    f = rng.integers(0, 255, (30, 30, 3), np.uint8)
    flat = lambda bx: X.mask_faces(f, None, missing="none", boxes=bx)
    o, ok = g.check_fix(f, 7, flat)                                   # second pass clears it
    assert ok and np.ptp(o[4:14, 4:14].reshape(-1, 3), 0).max() == 0
    o, ok = g.check_fix(f, 9, lambda bx: f)                           # refill that leaves the face: blocked
    r = g.result()
    assert not ok and r["FirstPassUnmaskedFrames"] == [7, 9] and r["UnmaskedFrames"] == [9]
    assert r["SecondPassBoxesFilled"] == 2 and r["FramesWithSecondPassBoxes"] == 2 and not r["Clean"]


def test_second_pass_clears_face_centred_inside_hull(tmp_path, monkeypatch, capsys):
    # face box centred on the landmark hull (so the fallback skips it) but larger than the hull
    _clip(tmp_path, face=(20, 28, 40, 40))
    rc, j = _run(tmp_path, monkeypatch, _Textured([(20, 28, 40, 40)]), ["--shareable"])
    c = j["UnmaskedFaceCheck"]
    assert c["FallbackBoxesFilled"] == 0
    assert c["FirstPassUnmaskedFrames"] == list(range(8)) and c["FramesWithSecondPassBoxes"] == 8
    assert c["UnmaskedFrames"] == [] and c["Clean"] and j["Shareable"] is True and rc == 0
    assert "second pass on 8 frame(s)" in capsys.readouterr().out


def test_second_pass_recheck_still_flagged_blocks(tmp_path, monkeypatch, capsys):
    _clip(tmp_path, bright=(3, 6))
    rc, j = _run(tmp_path, monkeypatch, _Bright(), ["--shareable"])
    c = j["UnmaskedFaceCheck"]
    assert c["FirstPassUnmaskedFrames"] == [3, 6] and c["SecondPassBoxesFilled"] == 2
    assert c["UnmaskedFrames"] == [3, 6] and j["Shareable"] is False and rc == 3
    assert "NOT SHAREABLE" in capsys.readouterr().out


# ── check of the written (encoded) file ──────────────────────
def test_encoded_check_reads_back_the_written_file(tmp_path):
    v = tmp_path / "fill.mp4"
    w = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    for i in range(6):
        w.write(np.full((32, 32, 3), 200 if i in (2, 5) else 50, np.uint8))
    w.release()
    g = G.FillGuard(detector=_Bright())
    e = g.check_encoded(v, 6)
    assert e["Ran"] and e["FramesChecked"] == 6 and e["UnmaskedFrames"] == [2, 5] and not e["Clean"]
    r = g.result()
    assert not r["Clean"] and r["UnmaskedFrames"] == [] and "encoded fill video" in r["Reason"]
    assert r["EncodedOutputCheck"]["UnmaskedFrameCount"] == 2
    assert "NOT SHAREABLE" in g.summary() and "encoded fill video: 2, 5" in g.summary()
    assert not g.check_encoded(v, 8)["Clean"] and "6 of 8 frames" in g.result()["Reason"]   # short read-back
    assert not g.check_encoded(tmp_path / "none.mp4")["Clean"]
    g.reset()
    assert g.result()["EncodedOutputCheck"] is None


class _Late:
    """Mock YuNet that only sees a face once switched on (stands in for a face that reappears after encoding)."""
    def __init__(self):
        self.on = False

    def detect(self, frame):
        return [_row(2, 2, 10, 10)] if self.on else []


def test_export_blocked_when_encoded_file_shows_a_face(tmp_path, monkeypatch, capsys):
    _clip(tmp_path)
    det = _Late()
    g = G.FillGuard(detector=det)
    orig = g.check_encoded

    def late(path, expected=None):
        det.on = True
        return orig(path, expected)
    monkeypatch.setattr(g, "check_encoded", late)
    monkeypatch.setattr(X, "make_preprocessor", lambda a: (_Est(), _Pre(_ring(40, 48)), None, None))
    monkeypatch.setattr(X, "make_fill_guard", lambda a: g)
    rc = X.main(["--input", str(tmp_path / "v.mp4"), "--whole", "--out", str(tmp_path / "o"), "--raw-root",
                 str(tmp_path / "r"), "--deid", "fill", "--no-bystanders", "--shareable"])
    j = S.read_json(next((tmp_path / "o").rglob("*desc-fill_video.json")))
    c = j["UnmaskedFaceCheck"]
    assert rc == 3 and j["Shareable"] is False and j["Deidentified"] is False
    assert c["UnmaskedFrames"] == [] and c["EncodedOutputCheck"]["UnmaskedFrames"] == list(range(8))
    assert c["EncodedOutputCheck"]["FramesChecked"] == 8 and "encoded fill video" in j["ShareableBlockedReason"]
    out = capsys.readouterr().out
    rf = c["EncodedRefill"]                                           # a face that never goes away: bounded passes
    assert rf["Passes"] == rf["MaxPasses"] == len(G.REFILL_PADS) and rf["RefilledFrames"] == list(range(8))
    assert rf["FirstCheckUnmaskedFrames"] == list(range(8)) and [r["BoxPad"] for r in rf["PerPass"]] == list(G.REFILL_PADS)
    assert "re-filled 8 frame(s) in 2 pass(es), final check: 8 frames, 8 flagged" in out and "NOT SHAREABLE" in out


class _LateTextured(_Textured):
    """Mock YuNet that misses the face before encoding and finds it (while unfilled) in the written file."""
    on = False

    def detect(self, frame):
        return super().detect(frame) if self.on else []


def _late_guard(monkeypatch, det):
    g = G.FillGuard(detector=det)
    orig = g.check_encoded

    def late(path, expected=None):
        det.on = True
        return orig(path, expected)
    monkeypatch.setattr(g, "check_encoded", late)
    return g


def test_refill_boxes_unit():
    g = G.FillGuard(detector=_Fixed([]))
    g.encoded_rows = {3: ([_row(10, 10, 20, 20)], (100, 100, 3)), 5: ([_row(90, 90, 20, 20)], (100, 100, 3))}
    assert g.refill_boxes(1) == {3: [(4, 4, 36, 36)], 5: [(84, 84, 100, 100)]}
    assert g.refills == [dict(Pass=1, BoxPad=G.REFILL_PADS[0], Frames=[3, 5], BoxesFilled=2)]
    assert g.refill_boxes(2)[3] == [(0, 0, 40, 40)]


def test_refill_makes_encoded_file_clean(tmp_path, monkeypatch, capsys):
    _clip(tmp_path)
    g = _late_guard(monkeypatch, _LateTextured([(100, 30, 40, 40)]))
    monkeypatch.setattr(X, "make_preprocessor", lambda a: (_Est(), _Pre(_ring(40, 48)), None, None))
    monkeypatch.setattr(X, "make_fill_guard", lambda a: g)
    rc = X.main(["--input", str(tmp_path / "v.mp4"), "--whole", "--out", str(tmp_path / "o"), "--raw-root",
                 str(tmp_path / "r"), "--deid", "fill", "--no-bystanders", "--shareable"])
    vid = next((tmp_path / "o").rglob("*desc-fill_video.mp4"))
    j = S.read_json(vid.with_suffix(".json"))
    c = j["UnmaskedFaceCheck"]
    rf = c["EncodedRefill"]
    assert rc == 0 and j["Shareable"] is True and c["Clean"] and c["EncodedOutputCheck"]["UnmaskedFrames"] == []
    assert rf["FirstCheckUnmaskedFrames"] == list(range(8)) and rf["Passes"] == 1 and rf["RefilledFrames"] == list(range(8))
    assert not list(vid.parent.glob("*.refill.*"))
    cap = cv2.VideoCapture(str(vid))
    n, ok = 0, True
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        assert f[30:70, 100:140].reshape(-1, 3).std(0).max() <= 8      # face box painted flat
    assert n == 8
    out = capsys.readouterr().out
    assert "re-filled 8 frame(s) in 1 pass(es), final check: 8 frames, 0 flagged" in out and "shareable" in out
