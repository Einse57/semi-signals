# Seizure detection (video, optional iEEG replay)

Status: research prototype. Not a medical device. Detector A results come from leave-one-patient-out
evaluation on 14 patients (see `docs/detector_a_results.md`); Detector B uses the released VSViG-base
weights, which were trained on the same 14 patients, so its scores on those videos are not an honest
estimate. The video + iEEG fusion is plumbing only: no public dataset pairs the two.

## Quick start: double-click start.cmd (or run .\start.ps1)
On Windows, from the repo folder:
1. Double-click `start.cmd` (or run `.\start.ps1` in PowerShell). The first time, if there is no `.venv`, it
   offers to create one and install the requirements (the same steps as [Windows setup](#windows-setup-igpu-via-openvino),
   including the OpenCV fix). Behind a proxy, set `HTTPS_PROXY` first, or answer `n` to see the manual steps.
   It looks for Python in this order: `SEMI_PYTHON`, pyenv-win, the `py` launcher, `python` on PATH. To pick
   one yourself: `$env:SEMI_PYTHON = "C:\path\to\python.exe"` in PowerShell (or `set SEMI_PYTHON=...` in cmd)
   before running it.
2. In the launcher window, click **Fetch models & sample data** once (output in the log pane; the optional
   **Proxy** field sets `HTTPS_PROXY` for the download). VSViG is skipped when torch is not installed.
3. Pick a sample clip from the **Video** list (`data\pat*\*.mp4`) or **Browse...** to any video.
   `Label.xlsx` is filled in automatically and only used for clips from the sample dataset.
4. Choose **Device** (GPU is selected when OpenVINO lists it, else CPU) and the options: **Loop playback**,
   **Hands**, **Explanations**, **VSViG** (enabled once `models\vsvig_base.xml` exists).
5. **Play with live scores** opens the viewer. When the video ends the window stays open on the last frame
   (graphs visible): `R` replay, `O` open another video, `Q` quit. With **Loop playback** it restarts by itself.
6. **Annotate this video**: fill in **Annotator initials**, **Patient ID / pseudonym** (never a real name) and
   **Video source** (remembered between sessions). The launcher shows the processing progress (pose and
   scoring pass), then the review window opens with a control panel next to it (see
   [Review your own video](#review-your-own-video-basic-local-only)). Every change is saved immediately.
7. **Open output folder** opens the folder of the selected video, where the annotation files are written.

The launcher stays open while the viewer or the annotation tool runs; their output and any errors appear in
its log pane (also kept in `%APPDATA%\semi-signals\launcher.log`). Settings are remembered in
`%APPDATA%\semi-signals\launcher.json` (`~/.semi-signals/` on Linux and macOS). On Linux or macOS run
`python -m src.launcher` from the repo folder (needs tkinter, e.g. the `python3-tk` package).

Keys (also shown in a footer strip in each window; `h` hides or shows it):
- Viewer: `q` quit, `space` pause/play, `f` fast / real-time pacing, `r` replay from the start,
  `o` open another video, `h` show/hide the key legend.
- Annotation: `o` clinical onset, `e` clinical end, `g` EEG onset, `t` seizure type, `n` note,
  `x` exclude start/stop, `u` undo, `space` pause, `j` / `l` back / forward 5 s, `r` replay, `s` save now,
  `h` show/hide the key legend, `q` quit.

## Live viewer (`src.run_seizure_demo`)
- The video, with the tracked patient's body skeleton and hands.
- **Detector A** (pose features + LightGBM, trained here): score every second, alert when the
  5 s smoothed score stays at or above its threshold (0.95) for 3 s.
- **Detector B** (VSViG-base, Apache-2.0 weights, OpenVINO IR): score every 0.5 s on the last 5 s.
  It is scored only when the tracked patient is in at least half of that window; otherwise the readout
  shows "-- (no subject tracked)" and the line has a gap (with no person, VSViG's inputs are all zeros and
  it returns about 0, which is not a meaningful score).
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
  - motion graphs (hands, arms, head, legs, whole body; speed in body sizes per second), drawn plain; the
    joints of the driver regions are ringed in amber on the video;
  - an EEG lane, greyed out: a layout hook (`explain_view.draw_eeg_lane`) for a future paired EEG stream.
  - Caption on screen: "Drivers explain the model score, not clinical cause. Research use only, not diagnostic."
  - Left/right are the patient's left/right (COCO keypoints), not the viewer's.
- End of video: the window stays open on the last frame with the graphs and the score strip, and shows
  "End of video: R replay, O open another video, Q quit" (`O` opens a file picker). `--loop` restarts by
  itself (`--max-loops N` to stop after N passes), `--exit-at-end` closes the window instead. A replay resets
  all causal state (buffers, smoothing, alarms, explanations, graphs), so the scores repeat exactly.
- Explanation cost: about 1 ms per second of video on CPU (`pred_contrib` once per second), and the
  panel is redrawn only when new samples arrive. Use `--no-explain` to turn it off, `--timing` for measurements.
- Text is drawn anti-aliased with a TrueType font (`src/textdraw.py`: Segoe UI on Windows, DejaVu Sans on
  Linux; set `SEMISIGNALS_FONT` / `SEMISIGNALS_FONT_BOLD` to a `.ttf` to override). Without Pillow or a
  TrueType font it falls back to OpenCV's Hershey font; `start.cmd` installs Pillow into an existing `.venv`
  if it is missing. On Windows the viewer, the review window and the launcher declare themselves DPI aware,
  so display scaling does not bitmap-upscale (blur) them; the windows show the frame 1:1 and shrink it
  (INTER_AREA) only when it is larger than the screen work area.

## Review your own video (basic, local only)
Start it from the launcher (**Annotate this video**), or from a terminal:
```bat
python -m src.review_video                 :: opens a file picker
python -m src.review_video --input my_clip.mp4 --annotator AL --pat-id self01
```
Missing details are asked for in dialogs, not in the terminal.
1. Processes the chosen file once (pose + Detector A + drivers; about real time) and saves
   `<stem>_semisignals.mp4`, `<stem>_scores.csv` and `<stem>_semisignals.json` (the settings and overlay
   version the render was made with) next to it. The render is reused only when it is newer than the video
   and matches the current settings (VSViG on/off, hands, Detector A model) and overlay version; otherwise
   it is made again. `--reprocess` forces it.
2. Plays the rendered video back in sync, with a timeline bar (red: model alarms, green: your marks).
   Keys: `space` pause, `j`/`l` back/forward 5 s, `o` clinical onset (new event), `e` clinical end,
   `g` EEG onset, `t` seizure type (pick list of ILAE 2017 types, or type your own),
   `n` note (attached to the event under the playhead, else a timed note), `x` start/stop an exclude span,
   `u` undo, `s` save now, `r` replay from the start, `h` show/hide the key legend, `q` quit.
   A control panel next to the video has the same actions as buttons (Onset, End, EEG onset, Type...,
   Note..., Exclude start/stop, Undo, Back 5 s, Pause/Play, Fwd 5 s, Replay, Save now), plus **Open folder**
   and **Open in ELAN** (ELAN from its usual Windows install folder, else whatever opens `.eaf` files).
   Autosave: every change is written straight away (to a temporary file, then swapped in, so a crash never
   leaves a half-written `.eaf`); the status shows `saved hh:mm:ss`. Quitting needs no save prompt.
   At the end of the video the window stays on the last frame under an "End of video" banner (amber
   outline across the middle): `r` replay, `q` quit; the annotation keys still work.
3. Saves next to the original video (format code isolated in `src/annotation.py`):
   - `<stem>.eaf`: ELAN EAF 3.0 (ms). Tiers `Seizure` (onset-to-end, value from the `seizure_event` CV)
     with child `ILAE2017_type`, `EEG_onset` (empty for now), `Notes`, `Exclude`, and the model tiers
     `Model_alarm` (value = peak probability) with child `Model_drivers`. Header properties: video sha256, fps,
     annotator, patient id/pseudonym, consent, model id, threshold, alarm rule, research-demo flag.
     In Face or Both mode (`--mode`, launcher **Annotation mode**), face events go to a `Face_event` tier
     (value = side L / R / both, or empty), their EEG onsets to `EEG_onset` as `fe01`, `fe02`, ... (body
     events keep `sz01`, ... numbered over body events only), and terms, once `labels/face.json` has a list,
     to a `Face_term` child tier. These tiers are written only when face events exist, so body-only files
     are unchanged. See the README section "Annotation modes and label sets".
   - `<stem>_signals.csv` (per second: probability, smoothed probability, alarm state, driver speed, region
     speeds; the header line starts with `#`, missing values are `NaN`), `<stem>_tsconf.xml` (track setup)
     and `<stem>.pfsx` (ELAN viewer layout, written only if absent), so ELAN's Timeseries Viewer shows the
     lanes on open. Keep the files together; ELAN finds them again after the folder is moved or unzipped.
   - `<stem>.annotation.json`: the same content (each event has `mode`: `body` or `face`) plus a comparison
     of your body marks with the model's alarms (alarm start within 30 s before to 60 s after your marked onset).
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
1. Install Python 3.10, 3.11 or 3.12 (64-bit, with tcl/tk for the launcher) and the current graphics driver.
   `start.cmd` does steps 2 and 5 for you; the steps below are the manual route.
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
   (The conversion also needs `onnx`, which `requirements-demo.txt` installs.) Without torch, VSViG is
   reported as skipped and the rest still installs; add `--skip-vsvig` for Detector A only.
   This fills rtmlib's cache with the pose models (`%USERPROFILE%\.cachetmlib`), downloads VSViG-base from
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
   Keys: `q` quit, `space` pause, `f` toggle real-time pacing, `r` replay, `o` open another video,
   `h` show/hide the key legend.
   Options: `--loop` to repeat the video, `--device cpu` to force CPU, `--no-hands`, `--start 3500` to jump into a long video,
   `--save out.mp4` to record the rendered output, `--log scores.csv` for per-second scores,
   `--headless --fast` to run without a window and without real-time pacing (with `--save` or `--log`),
   `--ieeg probs.csv --ieeg-offset 0` for an iEEG replay.

## Licences of loaded code, models and data
- semi-signals: MIT. VSViG code and weights, WU-SAHZU-EMU-Video: Apache-2.0.
- rtmlib, RTMPose, YOLOX: Apache-2.0. Flag: the YOLOX-tiny detector was trained with Human-Art
  (non-commercial licence) and the body7 / hand5 RTMPose checkpoints include research-only datasets.
  Swap these checkpoints before any commercial use.
