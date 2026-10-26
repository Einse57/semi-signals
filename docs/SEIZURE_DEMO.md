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
   ```
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

## Fallback
`demo_recordings\` (not in git) holds pre-rendered mp4s of the same runs in case the laptop misbehaves.

## Licences of what the demo loads
- semi-signals: MIT. VSViG code and weights, WU-SAHZU-EMU-Video: Apache-2.0.
- rtmlib, RTMPose, YOLOX: Apache-2.0. Flag: the YOLOX-tiny detector was trained with Human-Art
  (non-commercial licence) and the body7 / hand5 RTMPose checkpoints include research-only datasets.
  Swap these checkpoints before any commercial use.
