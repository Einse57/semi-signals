# Seizure detection (video, optional iEEG replay)

Status: research prototype. Not a medical device. Detector A results come from leave-one-patient-out
evaluation on 14 patients (see `docs/detector_a_results.md`); Detector B uses the released VSViG-base
weights, which were trained on the same 14 patients, so its scores on those videos are not an honest
estimate. The video + iEEG fusion is plumbing only: no public dataset pairs the two.

## Live viewer (`src.run_seizure_demo`)
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
     speeds; the header line starts with `#`, missing values are `NaN`), `<stem>_tsconf.xml` (track setup)
     and `<stem>.pfsx` (ELAN viewer layout, written only if absent), so ELAN's Timeseries Viewer shows the
     lanes on open. Keep the files together; ELAN finds them again after the folder is moved or unzipped.
   - `<stem>.annotation.json`: the same content plus a comparison of your marks with the model's alarms
     (alarm start within 30 s before to 60 s after your marked onset).
   Re-opening a video reloads its `.eaf`; a sha256 mismatch is refused, and `Model_*` tiers are regenerated,
   never read back as labels. ELAN (GPL-3.0) is not bundled; only its XML format is written.
If the ELAN Timeseries Viewer shows no curves (checked on ELAN 7.0):
- Tracks loaded but no panel shows them: right-click in the Timeseries Viewer, then
  `TrackPanel` > `Add All Tracks` (or `Add TrackPanel For Each Track`).
- No tracks at all: right-click, `Configure Tracks...`, pick `<stem>_signals.csv` under
  `Select a source to configure`, `Select`. On the `Add` tab set `Time Column Index` = 1, leave
  `Continuous Rate` unticked, and add one track per lane: `Track Name`, `Column` (1-based: p_seizure 2,
  p_smooth5s 3, alarm 4, driver_speed 5, speed_head ... speed_overall 6-13), `(Preferred) Range` >
  `Manual Setting` (0-1 for probabilities, 0-1.5 for speeds), `Add`. `Close`, then add them to a panel as above.
- The CSV is not linked: `Edit` > `Linked Files...`, `Linked Secondary Files`, `Add...`, pick the CSV, `Apply`.

Only use self-recorded or public clips. Other people's videos need their consent, and likely IRB review.

## Future work
- EEG integration: fill the `EEG_onset` tier / `eeg_onset_s` field and the greyed EEG lane from paired EEG
  (EDF+ annotations), once a dataset with paired video and EEG is available.

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
3. Fetch and verify the models, labels and sample clips (while online; safe to re-run, it skips files whose
   sha256 already matches):
   ```bat
   pip install torch --index-url https://download.pytorch.org/whl/cpu   :: only needed to convert VSViG once
   python scripts\fetch_models.py
   ```
   This fills rtmlib's cache with the pose models (`%USERPROFILE%\.cache\rtmlib`), downloads VSViG-base from
   github.com/xuyankun/VSViG and converts it to `models\vsvig_base.xml/.bin` (only if missing or stale), and
   downloads Detector A (`models\detector_a_all.txt/.json`, plus the body-only `detector_a_body.txt/.json`) from
   https://huggingface.co/Einse57/semi-signals-detector-a at a pinned commit, all sha256-checked; hashes go in
   `models\MANIFEST.json`. To use a local copy instead:
   `python scripts\fetch_models.py --detector-a D:\path\detector_a_all.txt`. Other flags: `--offline`
   (verify only, models and data), `--force`, `--skip-vsvig`, `--models-dir`.
   It also downloads `Label.xlsx` and two sample clips from the WU-SAHZU-EMU-Video dataset
   (https://huggingface.co/datasets/xuyankun/WU-SAHZU-EMU-Video, Apache-2.0, pinned commit) into `data\`:
   `data\Label.xlsx`, `data\pat10\Sz2P.mp4` (36 MB) and `data\pat10\Sz1P.mp4` (43 MB). `data\` is not in git.
   Data flags: `--clips pat12/Sz1P,pat12/Sz2P` for other clips (484 MB and 70 MB), `--clips all` for every video (about 27 GB),
   `--skip-data`, `--data-dir`. Fallback without the script: download `dataset/Label.xlsx` and the clips by hand
   from the dataset page and put them in `data\` without the `dataset/` prefix.
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
   `--save out.mp4` to record the rendered output, `--log scores.csv` for per-second scores,
   `--ieeg probs.csv --ieeg-offset 0` for an iEEG replay.

## Licences of loaded code, models and data
- semi-signals: MIT. VSViG code and weights, WU-SAHZU-EMU-Video: Apache-2.0.
- rtmlib, RTMPose, YOLOX: Apache-2.0. Flag: the YOLOX-tiny detector was trained with Human-Art
  (non-commercial licence) and the body7 / hand5 RTMPose checkpoints include research-only datasets.
  Swap these checkpoints before any commercial use.
