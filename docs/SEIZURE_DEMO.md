# Seizure-detection demo (video, optional iEEG replay)

Status: research demo. Not a medical device. Detector A results come from leave-one-patient-out
evaluation on 14 patients (see `docs/detector_a_results.md`); Detector B uses the released VSViG-base
weights, which were trained on the same 14 patients, so its scores on those videos are not an honest
estimate. The video + iEEG fusion is plumbing only: no public dataset pairs the two.

## What it shows
- The video, with the tracked patient's body skeleton and hands.
- **Detector A** (pose features + LightGBM, trained here): score every second, alert when the
  5 s smoothed score stays at or above its threshold (0.95) for 3 s.
- **Detector B** (VSViG-base, Apache-2.0 weights, OpenVINO IR): score every 0.5 s on the last 5 s.
- Markers for the EEG onset and clinical onset from `Label.xlsx`.
- **Side panel** (all lanes share one time axis):
  - probability strip: Detector A raw and 5 s smoothed score (0-1), dashed alarm threshold (0.95),
    seconds in alarm shaded red, current value readout (and Detector B / iEEG lines when enabled);
  - **top drivers**: once per second, LightGBM `pred_contrib` splits Detector A's score into per-feature
    contributions; these are summed by body region and feature family (speed, jerk, 0.5-1.5 Hz and 1.5-3 Hz
    rhythm, visibility, left/right asymmetry, finger motion, people in view) and averaged over the same 5 s
    window as the score. The top 2-3 positive groups are listed (with a little hysteresis so they do not
    flicker) once the smoothed score reaches 0.5. When a driver's features are mostly missing, it is labelled
    "<region> not tracked (pose lost)": the model treats lost tracking as evidence, and the caption says so;
  - motion graphs (hands, arms, head, legs, whole body; speed in body sizes per second): driver regions get a
    thick amber line and an amber background; the same joints are ringed on the video;
  - an EEG lane, greyed out: a layout hook (`explain_view.draw_eeg_lane`) for a future paired EEG stream.
  - Caption on screen: "Drivers explain the model score, not clinical cause. Research demo, not diagnostic."
  - Left/right are the patient's left/right (COCO keypoints), not the viewer's.
- Explanation cost: about 1 ms per second of video on CPU (`pred_contrib` once per second), and the
  panel is redrawn only when new samples arrive. Use `--no-explain` to turn it off, `--timing` for measurements.

## Review your own video (basic, local only)
```bat
python -m src.review_video                 :: opens a file picker
python -m src.review_video --input my_clip.mp4 --annotator AL --pat-id self01
```
1. Processes the chosen file once (pose + Detector A + drivers; about real time) and saves
   `<stem>_semisignals.mp4` and `<stem>_scores.csv` next to it.
2. Plays the rendered video back in sync, with a timeline bar (red: model alarms, green: your marks).
   Keys: `space` pause, `j`/`l` back/forward 5 s, `o` clinical onset (new event), `e` clinical end,
   `t` seizure type, `n` note (attached to the event under the playhead, else a timed note),
   `x` start/stop an exclude span, `u` undo, `s` save, `q` quit.
3. Saves next to the original video (format code isolated in `src/annotation.py`):
   - `<stem>.eaf`: ELAN EAF 3.0 (ms). Tiers `Seizure` (onset-to-end, value from the `seizure_event` CV)
     with child `ILAE2017_type`, `EEG_onset` (empty for now), `Notes`, `Exclude`, and the model tiers
     `Model_alarm` (value = peak probability) with child `Model_drivers`. Header properties: video sha256, fps,
     annotator, patient id/pseudonym, consent, model id, threshold, alarm rule, research-demo flag.
   - `<stem>_signals.csv` (per second: probability, smoothed probability, alarm state, driver speed, region
     speeds) and `<stem>_tsconf.xml`, so ELAN's Timeseries Viewer can show the lanes.
   - `<stem>.annotation.json`: the same content plus a comparison of your marks with the model's alarms
     (alarm start within 30 s before to 60 s after your marked onset).
   Re-opening a video reloads its `.eaf`; a sha256 mismatch is refused, and `Model_*` tiers are regenerated,
   never read back as labels. ELAN (GPL-3.0) is not bundled; only its XML format is written.
Only use self-recorded or public clips. Other people's videos need their consent, and likely IRB review.

## Roadmap
- Review app: nicer UI (in-window notes box, a progress bar while processing), several annotators per video.
- Annotation vocabularies: full ILAE 2025 seizure types and ILAE 2022 semiology CVs, `Semiology`/`Laterality`
  and `Consciousness` tiers, a shared `.etf` ELAN template, and an `ilae2025 -> ilae2017 -> Label.xlsx` map.
- Exports: BIDS `events.tsv` + `events.json`, and `Label.xlsx` rows for the training loader.
- Check on ELAN 7.0: tsconf column indexing and time units, relative paths after moving a folder.
- EEG integration: fill the `EEG_onset` tier / `eeg_onset_s` field and the greyed EEG lane from paired EEG
  (EDF+ annotations), once a dataset with paired video and EEG is available.
- Optional: an iEEG probability stream (CSV `t,p`, one row per second) replayed alongside, plus the
  noisy-OR fused score. A replay from another patient is unpaired and is labelled that way on screen.

## Windows setup (iGPU via OpenVINO)
1. Install Python 3.11 or 3.12 (64-bit) and the current graphics driver.
2. In a terminal, from the repo root:
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements-demo.txt
   pip uninstall -y opencv-contrib-python opencv-python-headless
   pip install --force-reinstall --no-deps opencv-python
   ```
   (rtmlib pulls in opencv-contrib-python; two OpenCV wheels in one venv can break `cv2`.)
   With pyenv-win, call the venv's `python.exe` by full path in `.cmd` scripts: calling the `python`
   shim (a `.bat`) without `call` ends the script after that line.
3. Fetch and verify the models (while online; safe to re-run, it skips files whose sha256 already matches):
   ```bat
   pip install torch --index-url https://download.pytorch.org/whl/cpu   :: only needed to convert VSViG once
   python scripts\fetch_models.py
   ```
   This fills rtmlib's cache with the pose models (`%USERPROFILE%\.cache\rtmlib`), downloads VSViG-base from
   github.com/xuyankun/VSViG and converts it to `models\vsvig_base.xml/.bin` (only if missing or stale), and
   downloads Detector A (`models\detector_a_all.txt/.json`) from Hugging Face, all sha256-checked; hashes go in
   `models\MANIFEST.json`. Until the Hugging Face repo is published, pass a local copy:
   `python scripts\fetch_models.py --detector-a D:\path\detector_a_all.txt`. Other flags: `--offline`
   (verify only), `--force`, `--skip-vsvig`, `--models-dir`.
   The held-out models for the runbook (`detector_a_wo_pat10/12`) are copied the same way with `--detector-a`.
   Videos: put `data\Label.xlsx` and some videos from https://huggingface.co/datasets/xuyankun/WU-SAHZU-EMU-Video
   (Apache-2.0) in `data\`, e.g. `data\pat10\Sz2P.mp4` (not in git).
4. Check the iGPU is visible:
   ```bat
   python -c "import openvino as ov; print(ov.Core().available_devices)"
   ```
   It should list `GPU`.
5. Run:
   ```bat
   python -m src.run_seizure_demo --input data\pat10\Sz2P.mp4 --labels data\Label.xlsx ^
       --detector-a models\detector_a_all.txt --vsvig models\vsvig_base.xml
   ```
   Keys: `q` quit, `space` pause, `f` toggle real-time pacing.
   Options: `--device cpu` to force CPU, `--no-hands`, `--start 3500` to jump into a long video,
   `--save out.mp4` to record the rendered demo, `--log scores.csv` for per-second scores,
   `--ieeg probs.csv --ieeg-offset 0` for an iEEG replay.

## Demo runbook

**The day before (online):** do the setup above, then run each clip once with `--headless --fast` so the
pose models download and OpenVINO compiles its GPU kernels. Copy `demo_recordings\` onto the laptop.

**Models for the clips:** use the *held-out* Detector A models, so the clip's patient was not in training:
`models\detector_a_wo_pat10.txt` for pat10 and `models\detector_a_wo_pat12.txt` for pat12 (same threshold, 0.95).

```bat
python -m src.run_seizure_demo --input data\pat12_Sz1P.mp4 --labels data\Label.xlsx ^
    --detector-a models\detector_a_wo_pat12.txt --vsvig models\vsvig_base.xml --device gpu
```

| order | clip | what happens (Detector A, held-out) | point to make |
|---|---|---|---|
| 1 | `pat12_Sz1P` (first 90 s) | alert about 5 s after clinical onset | a clear motor seizure caught quickly |
| 2 | `pat10_Sz1P` (35 s) | alert about 11 s after clinical onset | detection on a patient the model never saw |
| 3 | `pat10_Sz2P` (32 s) | score peaks at 0.94, just under 0.95, **no alert** | honest miss; the threshold trades misses for fewer false alarms |

On an iGPU (OpenVINO) the demo gives the same alerts as the CPU runs, at the same seconds (scores differ by a few
hundredths because the GPU runs pose at lower precision). It keeps up with real time with hands on.

**Talking points (honest):**
- Detector A, leave-one-patient-out on 14 patients: 29/33 seizures detected (95% CI 0.73-0.95) with the
  scoring code's window (clinical onset - 30 s to min(onset + 120 s, recording end) + 60 s), median 18 s after
  clinical onset; with the stricter onset-only window (onset - 30 s to + 60 s) 26/33 (0.62-0.89), median 13 s.
  4.3 false alarms per hour (95% CI 2.5-7.0) either way. Preregistered before scoring.
- The false-alarm estimate rests on only 3.7 h of non-seizure video, mostly from three patients; real monitoring
  would need many more hours. Videos start close to the seizure, which flatters recall.
- Detector B (VSViG-base) curves are shown for comparison; the released weights saw all 14 patients, so they
  are not evidence. A preregistered leave-one-patient-out retrain (`docs/vsvig_lopo_preregistration.md`) detected 27/33 at
  10.7 false alarms/h, so it does not beat Detector A.
- iEEG fusion is plumbing: there is no public dataset with paired video and iEEG, so any iEEG overlay is an
  unpaired replay and is labelled on screen.
- Research prototype, not a medical device. Video stays on the laptop; nothing is uploaded.

**If something breaks:** `--device cpu` (slower but works), `--no-hands` to lighten the load, or play the
pre-rendered recordings below.

## Fallback
`demo_recordings\` (not in git) holds pre-rendered mp4s of the same runs in case the laptop misbehaves:
`pat12_Sz1P_heldout_90s.mp4`, `pat10_Sz1P_heldout.mp4`, `pat10_Sz2P_heldout.mp4` (held-out Detector A models), plus
`pat10_Sz1P_igpu.mp4` and `pat10_Sz2P_igpu.mp4` recorded on an iGPU.

## Licences of what the demo loads
- semi-signals: MIT. VSViG code and weights, WU-SAHZU-EMU-Video: Apache-2.0.
- rtmlib, RTMPose, YOLOX: Apache-2.0. Flag: the YOLOX-tiny detector was trained with Human-Art
  (non-commercial licence) and the body7 / hand5 RTMPose checkpoints include research-only datasets.
  Swap these checkpoints before any commercial use.
