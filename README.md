# semi-signals

Real-time skeleton-based motion analysis — live body + hand tracking with per-region L/R motion quantification, accelerated via OpenVINO on Intel iGPU.

Inspired by [VSViG (Xu et al., ECCV 2024)](https://arxiv.org/abs/2311.14775) but extends its input pipeline with 21-keypoint hand tracking and finger-level motion. See [JOINT_MAPPING.md](src/JOINT_MAPPING.md) for a detailed comparison.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m src.run_live
```

### CLI Options

```bash
python -m src.run_live --cam 1                  # specific webcam
python -m src.run_live --input video.mp4        # video file
python -m src.run_live --list-cams              # probe available cameras
python -m src.run_live --no-hands               # body only (~40 FPS vs ~25-30)
python -m src.run_live --det-score 0.6          # detection confidence threshold
python -m src.run_live --det-interval 10        # detect every N frames
```

**Runtime keys:** `q` quit · `h` toggle hands · `c` cycle camera

## Models

All ONNX checkpoints are auto-downloaded by [rtmlib](https://github.com/Tau-J/rtmlib) from [OpenMMLab](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose).

| Model | Task | Input | Output |
|-------|------|-------|--------|
| [YOLOX-tiny](https://arxiv.org/abs/2107.08430) | Person detection | 416×416 | Bounding boxes |
| [RTMPose-m](https://arxiv.org/abs/2303.07399) body | Body pose | 256×192 crop | 17 keypoints (COCO) |
| [RTMPose-m](https://arxiv.org/abs/2303.07399) hand | Hand pose | 256×256 crop | 21 keypoints × 2 hands |

## Performance

Tested on Intel Core Ultra 7 165U (Meteor Lake) — Xe-LPG iGPU, AI Boost NPU 3720.

| Configuration | FPS |
|---------------|-----|
| CPU (ONNX Runtime) | ~6 |
| iGPU (OpenVINO) — body only | ~40 |
| iGPU (OpenVINO) — body + hands | ~25–30 |

OpenVINO targets the iGPU via a monkey-patch in `config.py` (rtmlib's built-in OpenVINO backend hardcodes CPU). The NPU compiles all models but is slower — YOLOX ~9× slower, RTMPose ~1.2× slower — as it's optimised for INT8/always-on workloads, not FP32 real-time vision.

## Project Layout

```
semi-signals/
├── requirements.txt
├── src/
│   ├── config.py            # Model URLs, device config, OpenVINO GPU patch
│   ├── detector.py          # Person detection (YOLOX-tiny)
│   ├── pose_estimator.py    # Body skeleton (RTMPose-m, COCO-17)
│   ├── hand_estimator.py    # Hand pose (RTMPose-m, 21 kpts per hand)
│   ├── motion.py            # Per-region velocity + finger blending
│   ├── display.py           # Skeleton overlay + stacked motion charts
│   ├── run_live.py          # Entry-point
│   └── JOINT_MAPPING.md     # VSViG joint mapping comparison
├── tests/
│   └── smoke_stage1.py
└── README.md
```

## References

- Xu et al., *"VSViG: Real-time Video-based Seizure Detection via Skeleton-based Spatiotemporal ViG"*, ECCV 2024. [arXiv:2311.14775](https://arxiv.org/abs/2311.14775)
- Jiang et al., *"RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose"*, 2023. [arXiv:2303.07399](https://arxiv.org/abs/2303.07399)
- Ge et al., *"YOLOX: Exceeding YOLO Series in 2021"*. [arXiv:2107.08430](https://arxiv.org/abs/2107.08430)

**What this project can do today:**
- Rule-based alerts on motion thresholds (e.g., sustained bilateral tremor)
- L/R asymmetry detection as an indicator
- Motion time-series recording for offline analysis
- Clip export on motion spikes for review
