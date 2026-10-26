# Seizure-detection demo (video, optional iEEG replay)

Status: research demo. Not a medical device. Detector A results come from leave-one-patient-out
evaluation on 14 patients (see `docs/detector_a_results.md`); Detector B uses the released VSViG-base
weights, which were trained on the same 14 patients, so its scores on those videos are not an honest
estimate. The video + iEEG fusion is plumbing only: no public dataset pairs the two.

## What it shows
- The video, with the tracked patient's body skeleton and hands.
- **Detector A** (pose features + LightGBM, trained here): score every second, alert when the
  5 s smoothed score stays above its threshold for 3 s.
- **Detector B** (VSViG-base, Apache-2.0 weights, OpenVINO IR): score every 0.5 s on the last 5 s.
- Markers for the EEG onset and clinical onset from `Label.xlsx`.
- Optional: an iEEG probability stream (CSV `t,p`, one row per second) replayed alongside, plus the
  noisy-OR fused score. A replay from another patient is unpaired and is labelled that way on screen.

## Windows setup (Intel Core Ultra laptop, iGPU via OpenVINO)
1. Install Python 3.11 or 3.12 (64-bit) and the current Intel graphics driver.
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
3. Put the files in `models\` and `data\` (not in git):
   - `models\detector_a_all.txt` and `models\detector_a_all.json` (Detector A, trained on all 14 patients)
   - `models\vsvig_base.xml` and `models\vsvig_base.bin` (VSViG-base converted to OpenVINO IR with
     `python -m src.export_vsvig --weights VSViG-base.pth --out models\vsvig_base`; needs torch)
   - `data\Label.xlsx` and some videos from https://huggingface.co/datasets/xuyankun/WU-SAHZU-EMU-Video
     (Apache-2.0), e.g. `data\pat10\Sz2P.mp4`
   The pose ONNX models download automatically on first run (rtmlib cache in `%USERPROFILE%\.cache\rtmlib`).
   Run once while online before the demo.
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

## Demo runbook (Andrew's MTL laptop; rehearsed on the ARL, Arc 140T)

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

The ARL (Arc iGPU, OpenVINO) gives the same alerts as the CPU runs, at the same seconds (scores differ by a few
hundredths because the GPU runs pose at lower precision). It keeps up with real time with hands on.

**Talking points (honest):**
- Detector A, leave-one-patient-out on 14 patients: 29/33 seizures detected (95% CI 0.73-0.95), 4.3 false
  alarms per hour (95% CI 2.5-7.0), median 18 s after clinical onset. Preregistered before scoring.
- The false-alarm estimate rests on only 3.7 h of non-seizure video, mostly from three patients; real monitoring
  would need many more hours. Videos start close to the seizure, which flatters recall.
- Detector B (VSViG-base) curves are shown for comparison; the released weights saw all 14 patients, so they
  are not evidence. A leave-one-patient-out retrain is in progress (`docs/vsvig_lopo_preregistration.md`).
- iEEG fusion is plumbing: there is no public dataset with paired video and iEEG, so any iEEG overlay is an
  unpaired replay and is labelled on screen.
- Research prototype, not a medical device. Video stays on the laptop; nothing is uploaded.

**If something breaks:** `--device cpu` (slower but works), `--no-hands` to lighten the load, or play the
pre-rendered recordings below.

## Fallback
`demo_recordings\` (not in git) holds pre-rendered mp4s of the same runs in case the laptop misbehaves:
`pat12_Sz1P_heldout_90s.mp4`, `pat10_Sz1P_heldout.mp4`, `pat10_Sz2P_heldout.mp4` (held-out Detector A models), plus
`pat10_Sz1P_arl.mp4` and `pat10_Sz2P_arl.mp4` recorded on the ARL's Arc iGPU.

## Licences of what the demo loads
- semi-signals: MIT. VSViG code and weights, WU-SAHZU-EMU-Video: Apache-2.0.
- rtmlib, RTMPose, YOLOX: Apache-2.0. Flag: the YOLOX-tiny detector was trained with Human-Art
  (non-commercial licence) and the body7 / hand5 RTMPose checkpoints include research-only datasets.
  Swap these checkpoints before any commercial use.
