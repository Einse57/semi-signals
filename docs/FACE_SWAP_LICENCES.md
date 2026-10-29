# Face swap: licence audit (phase 1)

Audit of GHOST (<https://github.com/ai-forever/ghost>, commit `44e58aa`, 2025-02-25) and everything it
needs, done before any weights were downloaded or run. Read from the source repository, its release
pages, the upstream projects and PyPI on 2026-10-28. This is a reading of published terms, not legal
advice.

Verdict rules: no licence or unknown = **off-limits**; non-commercial = **excluded**; GPL, research-only
or conflicting statements = **flagged**; Apache-2.0 / MIT / BSD / CC-BY = **OK**. Only OK items may be
downloaded or run. `src/deid_swap.py` refuses a configuration without `"licence_reviewed": true`.

## GHOST code

| Item | Source | Licence text or statement | Verdict |
|---|---|---|---|
| GHOST repository code (AEI-Net generator, `network/`, `utils/inference/` except below, `inference.py`) | [LICENSE](https://github.com/ai-forever/ghost/blob/main/LICENSE) | "Apache License Version 2.0, January 2004" (GitHub: Apache-2.0) | OK |
| Super-resolution code `models/pix2pix_model.py`, `models/networks/*.py`, `utils/inference/util.py` (from NVIDIA SPADE) | [models/pix2pix_model.py](https://github.com/ai-forever/ghost/blob/main/models/pix2pix_model.py), [SPADE LICENSE.md](https://github.com/NVlabs/SPADE/blob/master/LICENSE.md) | "Copyright (C) 2019 NVIDIA Corporation. All rights reserved. Licensed under the CC BY-NC-SA 4.0 license" | **excluded**. Used only with `--use_sr`, but `inference.py` line 16 imports it unconditionally; an integration must not import it |
| `models/networks/sync_batchnorm` | [vacancy/Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) | MIT | OK (only used by the excluded super-resolution code) |
| `apex/` (vendored NVIDIA apex) | [apex/LICENSE](https://github.com/ai-forever/ghost/blob/main/apex/LICENSE) | BSD 3-Clause ("Redistribution and use in source and binary forms, with or without modification, are permitted…") | OK (training only) |
| `AdaptiveWingLoss` submodule (eye loss, training only) | [NastyaMittseva/AdaptiveWingLoss](https://github.com/NastyaMittseva/AdaptiveWingLoss) (fork of [protossw512/AdaptiveWingLoss](https://github.com/protossw512/AdaptiveWingLoss), Apache-2.0) | the fork has no LICENSE file | **off-limits** as checked out; not needed for inference |
| `insightface_func/`, `coordinate_reg/image_infer.py` (wrappers around the `insightface` package) | GHOST repo | covered by the GHOST Apache-2.0 licence; call InsightFace models (below) | OK as code; the models they load are excluded |

## GHOST pretrained weights

No release page states a licence for its assets. The repository's Apache-2.0 text covers the source
tree; it does not name the release binaries. The README trains the swap model on VGGFace2.

| File | Source | Licence text or statement | Verdict |
|---|---|---|---|
| `G_unet_{1block,2blocks,3blocks}.pth` (generator; 2 blocks is the main model) | [release sber-swap-v2.0](https://github.com/ai-forever/ghost/releases/tag/sber-swap-v2.0) | release note: "Generator model with unet backbone and Discriminator model for 1, 2, 3 blocks in AAD ResBlock"; no licence. Training data VGGFace2: "non-commercial research purposes only" ([VGG Face licence](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/licence.txt), images CC BY-NC 4.0); identity loss uses the InsightFace ArcFace model (below) | **off-limits** (no licence); training data and teacher model non-commercial |
| `D_unet_{1block,2blocks,3blocks}.pth` (discriminator, training only) | same release | no licence | **off-limits** |
| `G_0_035000_init_arch_arcface2.pth` (v1 generator) | [release sber-swap-v1.0](https://github.com/ai-forever/ghost/releases/tag/sber-swap-v1.0) | no licence | **off-limits** |
| `backbone.pth` (ArcFace iResNet-100 identity encoder) + `iresnet.py` | [release arcface](https://github.com/ai-forever/ghost/releases/tag/arcface) | release note: "InsightFace arcface model". InsightFace: "The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only." | **excluded** |
| `scrfd_10g_bnkps.onnx` (face detector) | [release antelope](https://github.com/ai-forever/ghost/releases/tag/antelope) | InsightFace model zoo: "ALL models are available for non-commercial research purposes only." | **excluded** |
| `glintr100.onnx` (recognition model of the antelope pack) | same release | as above | **excluded** |
| `coordinate_reg/model/2d106det-*` (106-point landmarks, MXNet) | committed in the GHOST repo | InsightFace alignment model; same non-commercial statement | **excluded** |
| `coordinate_reg/model/SCRFD/model_25GF.pth` | committed in the GHOST repo | InsightFace SCRFD detector; same statement | **excluded** |
| `WFLW_4HG.pth` (AWL eye landmarks, training only) | [release awl_detector](https://github.com/ai-forever/ghost/releases/tag/awl_detector) | no licence | **off-limits** |
| `10_net_G.pth` (super resolution) | [release super-res](https://github.com/ai-forever/ghost/releases/tag/super-res) | no licence; its code is CC BY-NC-SA 4.0 | **excluded** |

InsightFace statements, verbatim: [README](https://github.com/deepinsight/insightface#license) "The code of
InsightFace is released under the MIT License. There is no limitation for both academic and commercial
usage. The training data containing the annotation (and the models trained with these data) are available
for non-commercial research purposes only. Both manual-downloading models from our github repo and
auto-downloading models with our python-library follow the above license policy (which is for
non-commercial research purposes only)." [python-package README](https://github.com/deepinsight/insightface/tree/master/python-package):
"The pretrained models provided with this library are for non-commercial research only, whether downloaded
automatically or manually." The README also directs licensing of the open recognition models
(buffalo_l) and inswapper models to InsightFace by email.

## Python dependencies (GHOST `requirements.txt`)

| Package (pinned) | Licence (PyPI metadata or repo) | Verdict |
|---|---|---|
| torch 1.6.0+cu101, torchvision 0.7.0+cu101 | BSD-3-Clause (torch now lists Apache-2.0 AND BSD-3-Clause AND others) | OK; the cu101 wheels bundle the NVIDIA CUDA runtime under NVIDIA's EULA |
| opencv-python | Apache-2.0 | OK |
| onnx 1.9.0 | Apache-2.0 | OK |
| onnxruntime-gpu 1.4.0 | MIT | OK |
| mxnet-cu101mkl | Apache-2.0; [apache/mxnet](https://github.com/apache/mxnet) is archived (retired project) | **flagged**: unmaintained; bundles CUDA and MKL binaries under their own terms |
| scikit-image | BSD-3-Clause | OK |
| insightface 0.2.1 | MIT (code only; models excluded above) | OK as code |
| requests 2.25.1 | Apache-2.0 | OK |
| kornia 0.5.4 | Apache-2.0 | OK |
| dill | BSD-3-Clause | OK |
| wandb (training logging) | MIT | OK; not needed for inference |
| tqdm, scipy, matplotlib, Pillow (imported) | MPL-2.0 AND MIT; BSD-3-Clause; Matplotlib (PSF-based); MIT-CMU | OK (MPL-2.0 is file-level copyleft, unmodified use is fine) |

## Summary for GHOST

The code is usable. No GHOST weight file is usable: the generator and discriminator have no licence and
were trained on non-commercial data, and every detector, landmark and identity model it loads is an
InsightFace non-commercial model. Using GHOST would need a generator retrained on licensed data with a
permissively licensed identity encoder, or written permission from the rights holders.

## Substitutes

### Face detectors

| Candidate | Source | Licence | Verdict |
|---|---|---|---|
| MediaPipe Face Landmarker / BlazeFace (already used) | [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe), [model cards](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md) | Apache-2.0 | OK |
| YuNet (OpenCV Zoo) | [models/face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | "All files in this directory are licensed under MIT License." | OK |

### Identity embedders (check harness and any swap identity encoder)

| Candidate | Source | Licence | Verdict |
|---|---|---|---|
| SFace (OpenCV Zoo, MobileFaceNet, ONNX) | [models/face_recognition_sface](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | "All files in this directory are licensed under Apache 2.0 License." | OK (training-data provenance is not stated in the directory; note it) |
| AdaFace | [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace) | code MIT; weights on Google Drive with no separate statement, trained on CASIA-WebFace / MS1MV2 / WebFace4M/12M | **flagged** (weights' terms unstated; training sets are research-only) |
| EdgeFace | [otroshi/edgeface](https://github.com/otroshi/edgeface), [HF Idiap/EdgeFace-Base](https://huggingface.co/Idiap/EdgeFace-Base) | repo BSD-3-Clause (checkpoints in repo); Hugging Face card `cc-by-nc-sa-4.0` | **flagged** (conflicting statements) |
| facenet-pytorch | [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch) | code MIT; weights trained on VGGFace2 / CASIA-WebFace, no separate statement | **flagged** |
| InsightFace ArcFace (any pack) | as above | non-commercial | **excluded** |

Proposed: YuNet or MediaPipe for detection, SFace for the identity-similarity pass.

### Swap or reenactment approaches

| Candidate | Source | Licence | Verdict |
|---|---|---|---|
| Landmark-driven warp of a synthetic donor face: Delaunay triangulation of the 478 MediaPipe landmarks, piecewise-affine warp of the donor texture onto the subject's landmarks per frame, colour matched and blended (OpenCV only) | no weights beyond MediaPipe | Apache-2.0 (MediaPipe), Apache-2.0 (OpenCV) | OK. Keeps expression and head motion through the landmarks; keeps the subject's face geometry (landmark positions), which is itself identifying |
| LivePortrait (reenactment: animate a donor image with the subject's motion) | [KlingAIResearch/LivePortrait](https://github.com/KlingAIResearch/LivePortrait), [HF KwaiVGI/LivePortrait](https://huggingface.co/KwaiVGI/LivePortrait) | code MIT; HF weights `license:mit`. README: "The models of InsightFace are for non-commercial research purposes only. If you want to use the LivePortrait project for commercial purposes, you should remove and replace InsightFace's detection models" | **flagged** until checked further: OK only with the InsightFace detector and 106-point model replaced (MediaPipe/YuNet); provenance of `landmark.onnx` and of the training data not yet checked |
| SimSwap | [neuralchen/SimSwap](https://github.com/neuralchen/SimSwap) | "For academic and non-commercial use only. The whole project is under the CC-BY-NC 4.0 license." | **excluded** |
| FaceFusion | [facefusion/facefusion](https://github.com/facefusion/facefusion) | OpenRAIL-AS (use restrictions); default models include InsightFace inswapper | **excluded** (models) / **flagged** (licence) |
| InsightFace inswapper | InsightFace | licensing by email only | **excluded** |

Proposed order: (1) landmark-driven warp, no new weights; (2) LivePortrait with InsightFace parts
replaced, after its remaining files are checked.

### Donor faces

| Candidate | Source | Licence | Verdict |
|---|---|---|---|
| Faces generated locally with FLUX.1 [schnell] | [HF black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | `license:apache-2.0` | OK (outputs; check the model card's use policy before use) |
| SFHQ synthetic faces | [SelfishGene/SFHQ-dataset](https://github.com/SelfishGene/SFHQ-dataset) | repo MIT; images made with StyleGAN2 (NVIDIA Source Code License-NC) and Stable Diffusion 2.1 (OpenRAIL++) | **flagged** |
| Microsoft DigiFace-1M | [microsoft/DigiFace1M](https://github.com/microsoft/DigiFace1M) | "can be used for non-commercial research" | **excluded** |
| Microsoft Face Synthetics | [microsoft/FaceSynthetics](https://github.com/microsoft/FaceSynthetics) | "can be used for non-commercial research" | **excluded** |
| Photos of real people | - | - | not used without a licence and consent |

## GHOST runtime and hardware (from reading the code)

- Pins: torch 1.6.0+cu101, torchvision 0.7.0+cu101, mxnet-cu101mkl, onnxruntime-gpu 1.4.0 (CUDA 10.1,
  Python 3.6-3.8 era wheels).
- CUDA-only as written: `inference.py` calls `G.cuda()`, `G.half()`, `netArc.cuda()`;
  `utils/inference/core.py`, `video_processing.py` and `image_processing.py` move tensors with `.cuda()`;
  InsightFace and the MXNet landmark model are prepared with `ctx_id=0` (GPU 0). Only `models/config.py`
  (super resolution) falls back to CPU.
- CPU: plausible with code changes: a device variable instead of `.cuda()`, float32 instead of `.half()`,
  `ctx_id=-1` for InsightFace/MXNet, CPU onnxruntime. The README already says to try `onnxruntime` when
  the GPU build cannot be installed.
- Intel XPU: would need a port to a current PyTorch with the XPU backend (torch 1.6 has none) and removal
  of MXNet, which has no XPU backend and is retired. The generator and ArcFace are plain PyTorch modules and
  kornia is device-agnostic, so the port is code work rather than model work.
- None of this matters until the weights question above is resolved.

## What is in this repository

- `src/deid_swap.py`: `FaceSwapper` protocol, backend registry, configuration checks (licence flag, model
  files, donor and donor licence), `GhostSwapper` placeholder that refuses to load.
- `src/deid_check.py`: check harness. Detector hit rate (default MediaPipe), identity similarity
  (pluggable `Embedder`, none configured), landmark-region and blendshape motion correlation.
- No weights, donor images or downloads.
