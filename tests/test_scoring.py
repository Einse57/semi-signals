import numpy as np
from src.detector_a import scoring
from src.detector_a.labels import second_labels


def test_labels():
    y = second_labels(400, eeg=100, clin=110)
    assert (y[:100] == 0).all() and (y[100:110] == -1).all()
    assert (y[110:230] == 1).all() and (y[230:] == -1).all()


def test_events_and_scoring():
    p = np.zeros(600); p[120:160] = 1.0; p[20:30] = 1.0
    ev = scoring.events_from_prob(p, 0.5)
    assert len(ev) == 2
    y = second_labels(600, eeg=100, clin=110)
    s = scoring.score_record(ev, y, clin=110, eeg=100)
    assert s["tp"] == 1 and s["fa"] == 1
    assert s["delay_clin"] == ev[1][0] - 110


def test_short_runs_dropped_and_merge():
    p = np.zeros(300); p[10:11] = 1; p[100:110] = 1; p[120:130] = 1
    ev = scoring.events_from_prob(p, 0.5)
    assert len(ev) == 1 and ev[0][0] <= 102 and ev[0][1] >= 130


def test_ci():
    lo, hi = scoring.wilson(10, 20)
    assert lo < 0.5 < hi
    lo, hi = scoring.poisson_rate_ci(0, 2.0)
    assert lo == 0 and hi > 1
