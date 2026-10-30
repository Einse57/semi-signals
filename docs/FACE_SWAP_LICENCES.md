# Face swap: licence audit (phases 1 to 4)

Audit of GHOST (<https://github.com/ai-forever/ghost>, commit `44e58aa`, 2025-02-25) and everything it
needs, done before any weights were downloaded or run. Read from the source repository, its release
pages, the upstream projects and PyPI on 2026-10-28. This is a reading of published terms, not legal
advice.

Verdict rules: no licence or unknown = **off-limits**; non-commercial = **excluded**; GPL, research-only
or conflicting statements = **flagged**; Apache-2.0 / MIT / BSD / CC-BY = **OK**. Only OK items may be
downloaded or run. `src/deid_swap.py` refuses a configuration without `"licence_reviewed": true`.

Phase 3 (decision of 2026-10-28): research-use-only and non-commercial models may be used for face
de-identification measurements in this project, locally only. See "Phase 3" below. The verdicts in the
phase 1 and 2 sections are unchanged as statements of the published terms.

## Licence evidence and use (checked 2026-10-29)

Project use: non-commercial research (project owner, 2026-10-29). Each licence below was fetched from the primary
source on 2026-10-29 at 17:22 PDT (Stable Diffusion 2.1: attempted 19:06 PDT). "Use" is one of: local inference,
download only, committed artifact, derived image in repo. Outputs and third-party weights stay local and
non-commercial; this repository holds only its own code, docs and numbers. This is a reading of published
terms, not legal advice.

Verdicts: **PERMITTED** = the quoted terms allow the use listed, including any public output listed;
**NOT CLEAR** = no licence, a licence that could not be fetched, or terms that do not settle the use. Outputs
of a NOT CLEAR component are not published until it is resolved. No row is NOT CLEAR (2026-10-29 19:50 PDT).

No redistribution (owner decision, 2026-10-29 19:50 PDT): this project redistributes no third-party code,
weights, datasets or images. Each user fetches them from the official source named in the row and accepts
those terms themselves. SFHQ comes from Kaggle only (the user signs in and accepts the dataset terms there;
this repository holds no Kaggle credentials or download code). No SFHQ (StyleGAN2 or Stable Diffusion
2.1 lineage) images or swaps are in this repository or in public posts; public figures use the mesh donor,
fill or mosaic. Aggregate measurement numbers with SFHQ donors are published.

| # | Component | Primary source | Licence clause (quoted) | Use here | Public outputs | Verdict and reason |
|---|---|---|---|---|---|---|
| 1 | GHOST code (AEI-Net, `network/`) | [ai-forever/ghost LICENSE](https://github.com/ai-forever/ghost/blob/main/LICENSE) | "Apache License Version 2.0, January 2004" | not redistributed, fetched by the user from <https://github.com/ai-forever/ghost> under Apache-2.0; local inference from a git-ignored checkout; no GHOST code committed (adapter `src/deid_ghost.py` is this project's code) | none from the code itself | **PERMITTED**: Apache-2.0 sec. 2 grants use; sec. 4 conditions apply to redistribution, which does not happen. No NC or share-alike term |
| 2 | GHOST generator weights `G_unet_2blocks.pth` | [ai-forever/ghost LICENSE](https://raw.githubusercontent.com/ai-forever/ghost/main/LICENSE), [download_models.sh](https://github.com/ai-forever/ghost/blob/main/download_models.sh), [README](https://github.com/ai-forever/ghost/blob/main/README.md) line 100, [release sber-swap-v2.0](https://github.com/sberbank-ai/sber-swap/releases/tag/sber-swap-v2.0) (sber-swap is the former name of ai-forever/ghost); fetched 2026-10-29 19:47 PDT | LICENSE: "Apache License Version 2.0, January 2004"; download_models.sh line 11: "wget -P ./weights https://github.com/sberbank-ai/sber-swap/releases/download/sber-swap-v2.0/G_unet_2blocks.pth"; README line 100: "you can use our pretrain weights for generator and discriminator that are located in `weights` folder"; release note: "Generator model with unet backbone and Discriminator model for 1, 2, 3 blocks in AAD ResBlock" (no separate licence text in the release) | not redistributed, fetched by the user from <https://github.com/sberbank-ai/sber-swap/releases/tag/sber-swap-v2.0> (GHOST release assets, via `https://github.com/ai-forever/ghost/blob/main/download_models.sh`) under Apache-2.0; local inference, pinned sha256, weights not committed | GHOST rows in tables; GHOST stills only with the mesh donor (no SFHQ-donor stills, rows 15-16) | **PERMITTED**: the weights are the project's own release assets, fetched by its own script and described as its pretrain weights; the release carries no separate licence text, and the repository's Apache-2.0 licence is taken to cover the project's own released weights (owner decision, 2026-10-29). Training data (VGGFace2) is non-commercial research; use here is local non-commercial research |
| 3 | InsightFace ArcFace `backbone.pth` (GHOST source identity; fetched by GHOST `download_models.sh` line 2 from release `arcface`) | [insightface README](https://github.com/deepinsight/insightface#license) | "The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only." | not redistributed, fetched by the user from <https://github.com/sberbank-ai/sber-swap/releases/tag/arcface> (GHOST release `arcface`, InsightFace model) under InsightFace non-commercial research terms; local inference, never committed | via GHOST only (see row 2) | **PERMITTED** for local non-commercial research; not redistributed. No attribution or share-alike term stated |
| 4 | InsightFace `glintr100.onnx` (antelope / antelopev2 recognition) | [model_zoo README](https://github.com/deepinsight/insightface/tree/master/model_zoo) | "ALL models are available for non-commercial research purposes only." | not redistributed, fetched by the user from <https://github.com/sberbank-ai/sber-swap/releases/tag/antelope> (GHOST release `antelope`, InsightFace model zoo <https://github.com/deepinsight/insightface/tree/master/model_zoo>) under InsightFace non-commercial research terms; local inference as the ArcFace scorer only, never committed | similarity numbers in tables | **PERMITTED**: non-commercial research use; numbers are research results, the model is not redistributed |
| 4a | InsightFace `scrfd_10g_bnkps.onnx` (antelope detector, fetched by GHOST `download_models.sh` line 7 from release `antelope`) | [model_zoo README](https://github.com/deepinsight/insightface/tree/master/model_zoo) | "ALL models are available for non-commercial research purposes only." | not redistributed, fetched by the user from <https://github.com/sberbank-ai/sber-swap/releases/tag/antelope> under InsightFace non-commercial research terms if ever used; not fetched or loaded here | none | **PERMITTED** for non-commercial research if used; not used and not redistributed |
| 5 | InsightFace buffalo_l `det_10g.onnx`, `2d106det.onnx` (bundled with LivePortrait) | [insightface README](https://github.com/deepinsight/insightface#license) | "Both manual-downloading models from our github repo and auto-downloading models with our python-library follow the above license policy(which is for non-commercial research purposes only)." | not redistributed, fetched by the user from <https://huggingface.co/KlingTeam/LivePortrait> (bundled with the LivePortrait weights) under InsightFace non-commercial research terms; download only (fetched with the LivePortrait weights, never loaded) | none | **PERMITTED**: download for non-commercial research; not redistributed |
| 6 | InsightFace inswapper | [insightface README](https://github.com/deepinsight/insightface#license) | "For inswapper series face swap models ... please contact contact@insightface.ai for licensing" | not redistributed; not used | none | not used |
| 7 | NVIDIA SPADE (CC BY-NC-SA 4.0): GHOST super-resolution code; LivePortrait SPADE decoder (acknowledged derivation) | [NVlabs/SPADE LICENSE.md](https://github.com/NVlabs/SPADE/blob/master/LICENSE.md) | "reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only"; "produce, reproduce, and Share Adapted Material for NonCommercial purposes only" | not redistributed, fetched by the user from <https://huggingface.co/KlingTeam/LivePortrait> (decoder weights) and <https://github.com/KlingTeam/LivePortrait> (code) under CC BY-NC-SA 4.0 (SPADE-derived parts); GHOST SPADE code not imported; LivePortrait decoder: local inference; nothing committed | via LivePortrait (rows 8-9) | **PERMITTED**: non-commercial use is granted; share-alike applies to shared Adapted Material, and none is shared |
| 8 | LivePortrait code | [KlingTeam/LivePortrait LICENSE](https://github.com/KlingTeam/LivePortrait/blob/main/LICENSE) | "MIT License ... Permission is hereby granted, free of charge ... to deal in the Software without restriction"; same file: "If you want to use the LivePortrait project for commercial purposes, you should remove and replace InsightFace's detection models" | not redistributed, fetched by the user from <https://github.com/KlingTeam/LivePortrait> under MIT; local inference from a git-ignored checkout; adapter `src/deid_liveportrait.py` is this project's code | none from the code itself | **PERMITTED**: MIT; InsightFace models not loaded (row 5) |
| 9 | LivePortrait weights (appearance, motion, warping, SPADE generator, stitching; `landmark.onnx` download only) | [HF KlingTeam/LivePortrait](https://huggingface.co/KlingTeam/LivePortrait) | model card metadata: "license: mit" | not redistributed, fetched by the user from <https://huggingface.co/KlingTeam/LivePortrait> under MIT; local inference, pinned sha256, never committed | LivePortrait stills (mesh donor) and table rows | **PERMITTED** under the stated MIT licence. Note: the paper lists NC and private training sets (see Phase 2 above); the licence statement itself is MIT |
| 10 | SFace (OpenCV Zoo) | [face_recognition_sface](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | "All files in this directory are licensed under Apache 2.0 License." | not redistributed, fetched by the user from <https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface> (OpenCV Zoo, by `scripts/fetch_models.py`) under Apache-2.0; download (URL + sha256 in `scripts/fetch_models.py`) and local inference | similarity numbers | **PERMITTED**: Apache-2.0; file not redistributed |
| 11 | YuNet (OpenCV Zoo) | [face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | "All files in this directory are licensed under MIT License." | not redistributed, fetched by the user from <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet> (OpenCV Zoo, by `scripts/fetch_models.py`) under MIT; download (URL + sha256) and local inference | hit-rate numbers | **PERMITTED**: MIT |
| 12 | MediaPipe Face Landmarker (Face Mesh V2, blendshapes), `canonical_face_model.obj` | [mediapipe LICENSE](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE); Face Mesh V2 model card | model card: "LICENSED UNDER Apache License, Version 2.0" | not redistributed, fetched by the user from <https://storage.googleapis.com/mediapipe-models/face_landmarker/> and <https://github.com/google-ai-edge/mediapipe> (MediaPipe, by `scripts/fetch_models.py`) under Apache-2.0; download and local inference; mesh donor renders derived locally (not committed) | landmark numbers; mesh-donor stills | **PERMITTED**: Apache-2.0 |
| 13 | SFHQ dataset, parts 3 and 4 (D. Beniaguev) | [Kaggle part 3](https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-3), [part 4](https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-4), [SelfishGene/SFHQ-dataset](https://github.com/SelfishGene/SFHQ-dataset) | Kaggle licence field: "CC0: Public Domain" (both parts); GitHub LICENSE: "MIT License, Copyright (c) 2022 David Beniaguev"; README: "no privacy issues or license issues surrounding these images" | not redistributed, fetched by the user from <https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-3> and <https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-4> (Kaggle; the user signs in to Kaggle and accepts its terms; no credentials or download code in this repository) under CC0 1.0; 32 donor images, local only (git-ignored `data/donors_synthetic/`), never committed | no SFHQ images or SFHQ-donor swaps in this repository or public posts (rows 15-16); aggregate measurement numbers only | **PERMITTED**: CC0 1.0 places no conditions on use; local use only, nothing redistributed |
| 14 | SFHQ download source | Kaggle only (row 13); no other download source is used by or referenced in this repository | Kaggle licence field: "CC0: Public Domain" (row 13) | not redistributed; users download from Kaggle (row 13) under CC0 1.0 and accept its terms themselves; no download code, mirror or credentials in this repository | none (images); aggregate measurement numbers only | **PERMITTED**: resolved by owner decision (2026-10-29 19:50 PDT): third-party assets are not redistributed; users download SFHQ from Kaggle |
| 15 | StyleGAN2 / FFHQ lineage of SFHQ (part 3 donors incl. the table donor 00001482) | [stylegan2 LICENSE.txt](https://github.com/NVlabs/stylegan2/blob/master/LICENSE.txt), [stylegan2-ada-pytorch LICENSE.txt](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt), [ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) | StyleGAN2 sec. 3.3: "The Work and any derivative works thereof only may be used or intended for use non-commercially ... "non-commercially" means for research or evaluation purposes only."; FFHQ: "Creative Commons BY-NC-SA 4.0 ... distribute any derivative works under the same license" | not redistributed; not used directly; inherited by the SFHQ donor images (fetched by the user from Kaggle, row 13) | none: no images of SFHQ donors or SFHQ-donor swaps; aggregate measurement numbers only | **PERMITTED** as limited: resolved by owner decision (2026-10-29 19:50 PDT): no SFHQ (StyleGAN2/SD2.1-lineage) images or swaps in the repository or public posts; public figures use the mesh donor, fill or mosaic; aggregate measurement numbers are fine. Evidence kept: neither NVIDIA text says whether generated images are derivative works and the Kaggle CC0 label does not settle it, so no such image is published; local non-commercial research use fits the strictest reading |
| 16 | Stable Diffusion 2.1 lineage of SFHQ part 4 (sweep donors only) | Stability-AI/stablediffusion LICENSE-MODEL, HF stabilityai/stable-diffusion-2-1-base | could not be fetched (404 and 401 on 2026-10-29) | not redistributed; not used directly; part 4 donors (fetched by the user from Kaggle, row 13) used locally in the donor sweep | none: no images of SFHQ part 4 donors or their swaps; aggregate measurement numbers only | **PERMITTED** as limited: resolved by owner decision (2026-10-29 19:50 PDT): no SFHQ (StyleGAN2/SD2.1-lineage) images or swaps in the repository or public posts; public figures use the mesh donor, fill or mosaic; aggregate measurement numbers are fine. Evidence kept: licence text not obtainable from the primary source (404 and 401 on 2026-10-29), so no such image is published |
| 17 | FLUX.1 [schnell] | [HF black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | API metadata: "license: apache-2.0" (gated repository) | not redistributed; not used: no weights downloaded, no code, no images | none | not used |
| 18 | GRID audio-visual speech corpus (Cooke, Barker, Cunningham, Shao) | [Zenodo 3625687](https://zenodo.org/records/3625687) | record licence: "cc-by-4.0"; CC BY 4.0 sec. 3(a): "If You Share the Licensed Material (including in modified form), You must: ... retain ... identification of the creator(s) ... indicate if You modified the Licensed Material" | not redistributed, fetched by the user from <https://zenodo.org/records/3625687> under CC BY 4.0; 4 clips local; derived numbers in docs; one derived figure in the repository (row 18a), other stills outside the repository | numbers; stills or article figures only with attribution and a modification note | **PERMITTED** with attribution: NOTICE credits the corpus; the only committed frames are the row 18a figure |
| 18a | Figure `docs/figures/face_deid_grid_lwwy7p_fill_mosaic_blur.png`: GRID clip `lwwy7p` (speaker s14) frames 20 and 45, panels original, `fill`, `mosaic8`, `mosaic16`, Gaussian blur; no donor, no swap | [Zenodo 3625687](https://zenodo.org/records/3625687) (row 18) | as row 18: "cc-by-4.0"; CC BY 4.0 sec. 3(a) attribution and modification notice | derived image in repo: a modified derivative (face filled, pixelated or blurred; panels arranged and captioned) committed with attribution and a modification note in NOTICE, `docs/figures/README.md` and the legend in FACE_DEID_TRADEOFF.md; licensed CC BY 4.0, not MIT; owner approval to commit, 2026-10-29 20:49 PDT | the figure itself | **PERMITTED**: CC BY 4.0 sec. 2 allows sharing adapted material; sec. 3(a) attribution, licence link and modification note are given |
| 19 | Intel IoT DevKit sample videos | [intel-iot-devkit/sample-videos LICENSE](https://github.com/intel-iot-devkit/sample-videos) | "Attribution 4.0 International" (CC BY 4.0), same sec. 3(a) | not redistributed, fetched by the user from <https://github.com/intel-iot-devkit/sample-videos> under CC BY 4.0; 2 clips local; derived numbers in docs | as row 18 | **PERMITTED** with attribution: NOTICE credits the videos; no frames are committed |

Stills: all phase 2-4 stills are kept outside this repository, except the row 18a GRID figure (original, `fill`,
mosaic and blur; no donor or swap). Stills that show an SFHQ donor or an SFHQ-donor
swap (rows 13-16) are not published. Stills of `fill`, `mosaic` and blur on GRID or Intel frames may be
published with the row 18/19 attribution and a note that the frames were modified.

Repository check (2026-10-29): no model weights, video, donor images or stills in any commit of this branch other
than the row 18a GRID figure (one PNG, 362 KB)
(blobs over 1 MB or with weight, media or array extensions: only `third_party/vsvig/dy_point_order.pt`, 38 KB,
VSViG point order, Apache-2.0, already on `main`, and the row 18a PNG); no Git LFS entries in `.gitattributes`; no access tokens or
keys; every commit not yet on GitHub is authored and committed as the noreply address; `LICENSE` is MIT
(c) Andrew F Lamkin and covers this project's code only.

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
| `G_unet_{1block,2blocks,3blocks}.pth` (generator; 2 blocks is the main model) | [release sber-swap-v2.0](https://github.com/sberbank-ai/sber-swap/releases/tag/sber-swap-v2.0), fetched by `download_models.sh` | README: "our pretrain weights"; release has no separate licence text; repository Apache-2.0. Training data VGGFace2: "non-commercial research purposes only" ([VGG Face licence](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/licence.txt), images CC BY-NC 4.0); identity loss uses the InsightFace ArcFace model (below) | **OK** for local non-commercial use: covered by the repository Apache-2.0 licence as the project's own released weights (owner decision, 2026-10-29); only `G_unet_2blocks.pth` is used |
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

The code is usable. The generator and discriminator are the project's own release assets; they carry no
separate licence text and are treated as covered by the repository Apache-2.0 licence (owner decision,
2026-10-29). They were trained on non-commercial data, and every detector, landmark and identity model it loads is an
InsightFace non-commercial model. Using GHOST would need a generator retrained on licensed data with a
permissively licensed identity encoder, or written permission from the rights holders. Phase 3 uses the
generator and identity encoder locally under the research-use-only decision; see "Phase 3".

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
| Landmark-driven warp of a synthetic donor face (implemented in phase 2 as backend `warp`, `src/deid_warp.py`): piecewise-affine warp over the 898 triangles of the MediaPipe canonical face mesh, colour matched and blended (OpenCV and numpy only) | MediaPipe Face Landmarker; `canonical_face_model.obj` (mesh topology) | Apache-2.0 (MediaPipe, mesh), Apache-2.0 (OpenCV) | OK. See "Phase 2" below for what carries over from the subject |
| LivePortrait (reenactment: animate a donor image with the subject's motion) | [KlingTeam/LivePortrait](https://github.com/KlingTeam/LivePortrait) (was KlingAIResearch / KwaiVGI), [HF KlingTeam/LivePortrait](https://huggingface.co/KlingTeam/LivePortrait) | code MIT with NC-derived parts; bundled InsightFace models non-commercial; training data includes NC datasets | **excluded** as distributed; core weights **flagged**. Details in "Phase 2: LivePortrait" below |
| SimSwap | [neuralchen/SimSwap](https://github.com/neuralchen/SimSwap) | "For academic and non-commercial use only. The whole project is under the CC-BY-NC 4.0 license." | **excluded** |
| FaceFusion | [facefusion/facefusion](https://github.com/facefusion/facefusion) | OpenRAIL-AS (use restrictions); default models include InsightFace inswapper | **excluded** (models) / **flagged** (licence) |
| InsightFace inswapper | InsightFace | licensing by email only | **excluded** |

Phase 2 outcome: (1) landmark-driven warp implemented; (2) LivePortrait not used (verdict below). Phase 3: GHOST and
LivePortrait run locally, research use only.

### Donor faces

| Candidate | Source | Licence | Verdict |
|---|---|---|---|
| Faces generated locally with FLUX.1 [schnell] | [HF black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | `license:apache-2.0` | Licence OK, but not used: see "Phase 2: FLUX.1 [schnell]" below |
| Procedural donors rendered from the MediaPipe canonical face mesh (`src/deid_warp.render_donor`) | `canonical_face_model.obj` (Apache-2.0) | no photograph, no image model | OK. Used in phase 2 (stylised, not photographic) |
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

## Phase 2: FLUX.1 [schnell] (checked 2026-10-28)

- Model card ([HF](https://huggingface.co/black-forest-labs/FLUX.1-schnell)): Apache-2.0; "can be used for
  personal, scientific, and commercial purposes". The card's out-of-scope list (unlawful use, harm to
  minors, disinformation, personal information used to harm, harassment, non-consensual nudity, automated
  adverse decisions) does not cover synthetic donor faces.
- The Hugging Face repository is gated: access requires an account that has accepted the terms and an
  access token. None is configured here, so per the phase 2 instruction generation was stopped and no
  weights were downloaded.
- Size: about 12 B parameters. The working machine (8 vCPU, 15 GB RAM, about 3 GB free, no swap, no GPU)
  cannot run it at useful speed in any case.
- Fallback used: procedural donors rendered from the Apache-2.0 canonical face mesh (no photograph, no
  image model), stored outside the repository with a provenance note. Photographic donors need either
  FLUX access (token with accepted terms, a machine with enough memory) or another Apache/MIT generator;
  no NC model is proposed.

## Phase 2: LivePortrait (checked 2026-10-28; nothing cloned, downloaded or run)

| Part | Source | Licence / provenance | Verdict |
|---|---|---|---|
| Code (top level) | [LICENSE](https://github.com/KlingTeam/LivePortrait/blob/main/LICENSE) | MIT, (c) 2024 Kuaishou Visual Generation and Interaction Center; the same file adds "The models of InsightFace are for non-commercial research purposes only" | OK on its own |
| Code: `SPADE`, `SPADEResnetBlock` in `src/modules/util.py`, SPADE decoder | acknowledged [NVlabs/SPADE](https://github.com/NVlabs/SPADE) | NVlabs/SPADE is CC BY-NC-SA 4.0; whether LivePortrait's classes are copied or rewritten is not stated | **flagged** |
| Code: keypoint/warping modules (`util.py` Hourglass, `kp2gaussian`, dense motion) | acknowledged FOMM and [Open Facevid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis) | FOMM MIT; Open Facevid2vid CC BY-NC 4.0 (the paper says face vid2vid was reproduced from that implementation) | **flagged** |
| Code: X-Pose (animals mode, `src/utils/dependencies/XPose`) | [IDEA-Research/X-Pose](https://github.com/IDEA-Research/X-Pose) | "License for Non-commercial Scientific Research Purposes" | **excluded** (not needed for humans) |
| Weights `insightface/models/buffalo_l/det_10g.onnx`, `2d106det.onnx` | InsightFace buffalo_l | InsightFace README: models "available for non-commercial research purposes only"; buffalo_l licensing by email | **excluded** |
| Weights `liveportrait/base_models/*.pth` (appearance, motion, warping, SPADE generator) and `retargeting_models/stitching_retargeting_module.pth` | HF repo, tag `license:mit`, not gated | MIT tag; trained on the data below | **flagged** (training data) |
| Weights `liveportrait/landmark.onnx` | HF repo | no documented origin, architecture or training data | **flagged** (unknown provenance) |
| Weights `liveportrait_animals/*`, `xpose.pth` | HF repo | X-Pose weights under the IDEA NC licence | **excluded** |
| Motion extractor backbone | ConvNeXt-V2-Tiny | ConvNeXt-V2 code MIT, its released ImageNet weights CC BY-NC 4.0; the paper says the base model was trained from scratch | note only |
| Training data ([arXiv 2407.03168](https://arxiv.org/abs/2407.03168) sec. 3.2) | VoxCeleb, MEAD, RAVDESS, AAHQ, plus private 4K portrait videos, 200 h of talking-head video, the private LightStage set and styled portraits; face-id loss with ArcFace | RAVDESS CC BY-NC-SA 4.0; AAHQ "academic research purpose only" (metadata CC BY-NC-SA 4.0); MEAD research-only; VoxCeleb is YouTube URLs (no longer distributed); private sets: terms unknown | **flagged** (NC and unknown sources) |

Verdict against the project rules: LivePortrait as distributed is **excluded** (bundled InsightFace
detection and landmark models are non-commercial). With those replaced by MediaPipe/YuNet, the remaining
human-mode weights are still **flagged** (NC and private training data, undocumented `landmark.onnx`) and
parts of the code are **flagged** (SPADE and face vid2vid derivations). Not used in phase 2; used locally in
phase 3 under the research-use-only decision.

## Phase 2: warp backend and measurements (2026-10-28)

`warp` (src/deid_warp.py) warps a procedural donor over the MediaPipe mesh triangles, redraws the irises at
the subject's iris positions, matches colour (Lab mean and spread) and blends with `cv2.seamlessClone`.
Geometry `donor` (default) uses the donor's proportions for eyes, brows, nose and mouth plus the subject's
expression (deviation from the subject's running mean shape); geometry `subject` uses the subject's own
landmark positions. Carried over from the subject in both: face outline, expression, head pose, gaze,
skin colour (mean and spread), hair, ears, neck, body and background; with `subject` also the face's
proportions.

Measured on the working machine (shared VM, 8 vCPU "Intel(R) Xeon(R) Processor", CPU only) on GRID clips
bbad9n, bbae1p, bbae2a, lwwy7p (300 frames) and the two Intel sample clips (2810 frames). Detector hit =
frames with at least one face (MediaPipe / YuNet). SFace = cosine similarity, original face vs the face
YuNet finds at the same place in the output (n = face pairs), threshold 0.363. Landmark r = mean Pearson r
of five face-region speeds; blendshape r = median r over 51 blendshapes (excluding `_neutral`), both on
frames with a face in original and output. ms/frame = the method only; the landmark stage it needs costs
another 41 (GRID) / 47 (Intel) ms/frame. Blur is a local review variant, listed for reference.

| Method | Clips | MediaPipe hit | YuNet hit | SFace mean (n; share >= 0.363) | Landmark r | Blendshape r | ms/frame |
|---|---|---|---|---|---|---|---|
| original | GRID | 100 % | 100 % | - | - | - | - |
| fill | GRID | 0 % | 0 % | no face (n 0) | undefined | undefined | 1.3 |
| blur | GRID | 49 % | 100 % | 0.07 (300; 0 %) | 0.03 | -0.02 | 5.2 |
| warp, donor geometry | GRID | 100 % | 100 % | 0.31 (300; 9 %) | 0.78 | 0.68 | 12.0 |
| warp, subject geometry | GRID | 100 % | 100 % | 0.46 (300; 100 %) | 0.82 | 0.74 | 10.3 |
| original | Intel | 73 % | 84 % | - | - | - | - |
| fill | Intel | 0.2 % | 0.7 % | 0.75 (19; 79 %): faces the landmarker missed | (19 frames) | (19 frames) | 3.3 |
| blur | Intel | 20 % | 51 % | -0.01 (1970; 1 %) | 0.16 | -0.06 | 7.4 |
| warp, donor geometry | Intel | 73 % | 73 % | 0.22 (3866; 6 %) | 0.50 | 0.29 | 14.6 |
| warp, subject geometry | Intel | 73 % | 73 % | 0.29 (3909; 15 %) | 0.68 | 0.40 | 12.9 |

Notes: warp with subject geometry is above the SFace threshold in every GRID frame, so it does not
de-identify by this measure; donor geometry lowers similarity but 9 % (GRID) and 6 % (Intel) of face pairs
are still at or above the threshold. Similarity of the aligned crop at the original face position with the
face filled is 0.30 on average (GRID), so values near 0.3 also reflect hair, outline and background. On
Intel, faces the landmarker misses are left untouched by every method (19 fill frames). Hardware for an
Intel iGPU run is still pending. Phase 3 adds ArcFace for these methods (see "Phase 3").

## Phase 2: identity check models

- YuNet `face_detection_yunet_2023mar.onnx` (OpenCV Zoo, MIT) and SFace `face_recognition_sface_2021dec.onnx`
  (OpenCV Zoo, Apache-2.0), opencv_zoo commit 48b10fa505aeb51fbd127fc60a5c660040e823b7, pinned by sha256 in
  `scripts/fetch_models.py --face-check`.
- `canonical_face_model.obj` (MediaPipe, Apache-2.0), commit a908d668c730da128dfa8d9f6bd25d519d006692, same step.
- OpenCV's SFace example treats cosine similarity >= 0.363 (or normalised L2 <= 1.128) as the same identity.

## Phase 3: research-use-only face swap (2026-10-28)

Decision (project owner, 2026-10-28): research-only and non-commercial models may be used for semi-signals
face de-identification only, under these conditions:

- local use only; weights are never committed or redistributed;
- every weight is fetched with a pinned sha256;
- third-party code stays outside the repository (git-ignored `third_party_local/`); the repository holds
  only adapter code behind the existing `FaceSwapper` interface and configuration gate;
- the GHOST generator weights carry no separate licence text; they are covered by the repository Apache-2.0
  licence as the project's own released weights (owner decision, 2026-10-29) and stay local.

**Swap mode note: `warp`, `ghost` and `liveportrait` outputs are "reduced risk, not de-identified".**
`ghost` and `liveportrait` are **research use only**: their configuration must set
`"research_use_acknowledged": true`, their sidecars carry `ResearchUseOnly: true`, and their outputs stay
local. `fill` stays the default and the only mode treated as shareable. `ghost` is the default swap
backend (a swap configuration without `"backend"` uses it; owner decision, 2026-10-29); its output stays
"reduced risk, not de-identified".

### Third-party code (local checkouts, not in this repository)

| Code | Commit | Licence | Parts used |
|---|---|---|---|
| [ai-forever/ghost](https://github.com/ai-forever/ghost) | `44e58aad8600ee83ad4c8213aaa9698ccaf66c1c` | Apache-2.0 | `network/AEI_Net.py`, `network/AADLayer.py`, `network/resnet.py`. Not imported: SPADE super-resolution code (CC BY-NC-SA 4.0), MXNet landmark code, InsightFace detector wrappers, `inference.py` |
| [KlingTeam/LivePortrait](https://github.com/KlingTeam/LivePortrait) | `9b294b3d0536135442ea73cb01e6cb3ca7029dd3` | MIT with NC-derived parts (flagged above: SPADE decoder, face vid2vid modules) | `live_portrait_wrapper.py`, `modules/` (including the flagged SPADE decoder and warping modules), `config/`, `utils/crop.py`, `utils/camera.py`. Not imported: InsightFace face analysis, `landmark.onnx` runner, X-Pose, animal mode |
| insightface 0.2.1 (PyPI wheel, extracted) | - | MIT (code) | `utils/face_align.py` read for the 5-point template; the adapter carries its own copy of the template constants and a test checks the alignment maps the template onto itself |

### Weights (local, pinned sha256, never committed)

GitHub release assets publish no digest, so the GHOST pins were recorded on first download (2026-10-28)
and are checked on every re-fetch. The LivePortrait pins are the Hugging Face LFS sha256 at revision
`82a4fa6735ca58432b6ce39301b4b9ee066dea47`.

| File | Source | sha256 | Licence | Status |
|---|---|---|---|---|
| `G_unet_2blocks.pth` (GHOST generator) | release `sber-swap-v2.0` | `85f130421c65722927c47502ea1b28b62be4893ca9180fec735b67be1e5edd8d` | Apache-2.0 (repository licence, covering the project's own released weights; owner decision 2026-10-29); trained on VGGFace2 (non-commercial research) | used, local, research use only |
| `backbone.pth` + `iresnet.py` (ArcFace iResNet-100, GHOST source identity) | release `arcface` | `a566a62357f0c55b679d9ff2f022a294486568be0c00665d39029d0e46a8109b`, `990006bfcd6583bdc9775045dfe02cbb10904018c0c34825f05d9cdd834fd388` | InsightFace: non-commercial research only | used, research use only |
| `glintr100.onnx` (InsightFace antelope recognition model) | release `antelope` | `4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf` | InsightFace: non-commercial research only | used as the ArcFace scorer in the check harness only, research use only |
| `liveportrait/base_models/appearance_feature_extractor.pth` | HF KlingTeam/LivePortrait | `5279bb8654293dbdf327030b397f107237dd9212fb11dd75b83dfb635211ceb5` | `license:mit` tag; training data flagged (NC and private sets) | used, research use only |
| `liveportrait/base_models/motion_extractor.pth` | same | `251e6a94ad667a1d0c69526d292677165110ef7f0cf0f6d199f0e414e8aa0ca5` | same | used, research use only |
| `liveportrait/base_models/spade_generator.pth` | same | `4780afc7909a9f84e24c01d73b31a555ef651521a1fe3b2429bd04534d992aee` | same | used, research use only |
| `liveportrait/base_models/warping_module.pth` | same | `2f61a6f265fe344f14132364859a78bdbbc2068577170693da57fb96d636e282` | same | used, research use only |
| `liveportrait/retargeting_models/stitching_retargeting_module.pth` | same | `3652d5a3f95099141a56986aaddec92fadf0a73c87a20fac9a2c07c32b28b611` | same | used, research use only |
| `liveportrait/landmark.onnx` | same | `31d22a5041326c31f19b78886939a634a5aedcaa5ab8b9b951a1167595d147db` | unknown provenance (flagged) | fetched, not loaded |
| `insightface/models/buffalo_l/det_10g.onnx`, `2d106det.onnx` | same | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`, `f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf` | InsightFace: non-commercial research only | fetched, not loaded |

Not fetched: GHOST discriminators, v1 generator, SCRFD detectors, MXNet `2d106det`, AWL eye landmarks,
super-resolution `10_net_G.pth`; LivePortrait animal and X-Pose weights. Detection and landmarks in both
adapters come from MediaPipe (Apache-2.0).

### Adapters

- `ghost` (`src/deid_ghost.py`): MediaPipe 5 points (eye centres, nose tip, mouth corners) aligned to the
  InsightFace 0.2.1 template at 224 px; AEI-Net (U-Net, 2 AAD blocks, float32, CPU) at 256 px; source
  identity from `backbone.pth` on the donor; result warped back and blended over the MediaPipe face hull
  (eroded and feathered). No super resolution.
- `liveportrait` (`src/deid_liveportrait.py`): the donor image is the source; the subject's video drives
  it. Crops use LivePortrait's own `crop_image` with MediaPipe 5 points. Motion is relative to the first
  frame of each face track (rotation, expression, scale, translation), then stitching and decoding at
  512 px. The animated face is aligned back to the subject's landmarks (Umeyama) and blended over the
  MediaPipe face hull. Frames where the animated face has no landmarks reuse the last alignment.
- Donors: the procedural mesh donors (`src/deid_warp.render_donor`, Apache-2.0 mesh, no photograph). No
  real person's photo or unlicensed face image is used. FLUX donors are on hold.

### ArcFace scorer and threshold

`deid_check.ArcFaceEmbedder` (`backend: "arcface"`) runs `glintr100.onnx` on CPU on YuNet detections aligned
to the ArcFace 112 px template. InsightFace publishes no single same-identity threshold; its model guide
puts 1:1 verification thresholds for its recognition packs at cosine 0.30-0.45 (false-match rate 1e-4 to
1e-5). The harness uses 0.30, the low end, so more pairs count as the same person, and also reports the
share at or above 0.45. Checks on this machine: the same GRID speaker (s14) in two clips 0.89; s14 against
the two Intel clips' people 0.01 and 0.03. The SFace threshold stays 0.363.

### Measurements

| Method | Clips | MediaPipe hit | YuNet hit | SFace mean (n; share >= 0.363) | ArcFace mean (n; share >= 0.30; share >= 0.45) | Landmark r | Blendshape r | ms/frame |
|---|---|---|---|---|---|---|---|---|
| original (landmarks: ms/frame) | GRID (4 clips, 300 frames) | 100 % | 100 % | - | - | - | - | 42.8 |
| fill | GRID | 0 % | 0 % | no face (n 0) | no face (n 0) | (0 frames) | (0 frames) | 1.3 |
| blur | GRID | 49 % | 100 % | 0.07 (300; 0 %) | 0.18 (300; 0 %; 0 %) | 0.03 | -0.02 | 5.2 |
| warp, donor geometry | GRID | 100 % | 100 % | 0.31 (300; 8.7 %) | 0.18 (300; 0 %; 0 %) | 0.78 | 0.68 | 12.0 |
| warp, subject geometry | GRID | 100 % | 100 % | 0.46 (300; 100 %) | 0.32 (300; 82 %; 0 %) | 0.82 | 0.74 | 10.3 |
| ghost (mesh donor) | GRID | 100 % | 100 % | 0.36 (300; 43 %) | 0.13 (300; 0 %; 0 %) | 0.78 | 0.68 | 356.8 |
| liveportrait (mesh donor) | GRID | 100 % | 100 % | 0.27 (300; 1.7 %) | 0.13 (300; 0 %; 0 %) | 0.14 | 0.02 | 1798.2 |
| original (landmarks: ms/frame) | Intel (2 clips, 2810 frames) | 73 % | 84 % | - | - | - | - | 55.8 |
| fill | Intel | 0.2 % | 0.7 % | 0.75 (19; 79 %) | 0.73 (19; 79 %; 79 %) | (19 frames) | (19 frames) | 3.3 |
| blur | Intel | 20 % | 51 % | -0.01 (1970; 0.8 %) | 0.07 (1970; 1.0 %; 0.8 %) | 0.16 | -0.05 | 7.4 |
| warp, donor geometry | Intel | 73 % | 73 % | 0.22 (3866; 5.7 %) | 0.11 (3866; 0.8 %; 0.4 %) | 0.50 | 0.29 | 14.6 |
| warp, subject geometry | Intel | 73 % | 73 % | 0.29 (3909; 15 %) | 0.18 (3909; 4.5 %; 0.4 %) | 0.68 | 0.40 | 12.9 |
| ghost (mesh donor) | Intel | 73 % | 73 % | 0.31 (3920; 31 %) | 0.13 (3920; 1.0 %; 0.4 %) | 0.55 | 0.41 | 1176.1 |
| liveportrait (mesh donor) | Intel | 71 % | 70 % | 0.22 (3624; 2.2 %) | 0.10 (3624; 0.5 %; 0.4 %) | 0.18 | -0.01 | 3780.3 |

Measured on the working machine (shared VM, 8 vCPU "Intel(R) Xeon(R) Processor", 15.6 GB RAM, no GPU,
torch 2.15.0+cpu with 8 threads, onnxruntime 1.31.1, OpenCV 5.0.0) on GRID s14 clips bbad9n, bbae1p, bbae2a,
lwwy7p and the Intel sample clips listed in the Clips column. Donor: procedural mesh donor, seed 2. Columns
as in the phase 2 table; fill, blur and warp rows are the phase 2 renders of the same clips. ArcFace =
`glintr100.onnx` cosine similarity, original face vs the face YuNet finds at the same place in the output
(n = face pairs), thresholds 0.30 and 0.45. ms/frame = the method only; the landmark stage is listed on the
"original" row. Frames where the subject's face is not found are filled whole-frame by every method.

Intel: both clips scored (face-demographics-walking-and-pause, head-pose-face-detection-female-and-male).

Observations:

- `ghost` with the mesh donor: on GRID 43 % of face pairs are at or above the SFace threshold and none at
  or above the ArcFace 0.30 threshold. In the stills the output keeps the subject's face shape, hair line and
  skin, with the eyes rendered closed or blank. The mesh donor is not photographic, so the identity code
  GHOST takes from it carries little face detail. This is reported as a donor failure for `ghost`.
- `liveportrait` with the mesh donor: the output shows the stylised donor face; the mouth moves. Landmark r
  0.14 and blendshape r 0.02 (GRID): the animated face's motion follows the subject's only weakly.
- SFace and ArcFace give different shares for the same outputs (for example warp with subject geometry on
  GRID: SFace 100 % at or above 0.363, ArcFace 82 % at or above 0.30 and 0 % at or above 0.45).
- All `warp`, `ghost` and `liveportrait` outputs are **reduced risk, not de-identified**.

## Phase 4: synthetic photo donors (2026-10-29)

Phase 3 used a procedural mesh render as the donor. GHOST takes its identity code from the donor, and LivePortrait
animates the donor image, so both need a photographic donor face. FLUX.1 [schnell] stays on hold (gated
repository). Decision (project owner, 2026-10-29): donors come from an existing synthetic face dataset only; no
photographs of real people.

### Donor dataset

| Dataset | Version / source | Licence | Subjects and consent | Realism | Status |
|---|---|---|---|---|---|
| SFHQ, Synthetic Faces High Quality (D. Beniaguev) | parts 3 and 4 on Kaggle (DOI 10.34740/kaggle/dsv/4737549), code at github.com/SelfishGene/SFHQ-dataset (MIT); not redistributed: users download it from Kaggle and accept its terms themselves | images CC0 1.0 (Kaggle) | no real subjects: every image is generated (StyleGAN2 with e4e encoding; part 3 from StyleGAN2 samples, part 4 from Stable Diffusion 2.1 portraits), so there is no consent to record. Flag: StyleGAN2 was trained on FFHQ (Flickr photos, no subject consent; StyleGAN2 weights under the NVIDIA Source Code License, non-commercial), so a generated face can resemble a real person by chance | photographic 1024 px faces; some stylised hair colours and painted looks (removed by hand) | **used**, 32 donors, local only, never committed; no SFHQ images or SFHQ-donor swaps in this repository or public posts (aggregate numbers only) |
| DigiFace-1M (Microsoft) | github.com/microsoft/DigiFace1M | Research Use of Data Agreement v1.0 (non-commercial research) | no real subjects (rendered) | 112 x 112 px rendered faces, not photographic | not used |

FLUX.1 [schnell] (black-forest-labs, Apache-2.0): on hold; the gated repository was not downloaded.

Donor selection (`e2e/phase4/select_donors.py`, outside this repository): 1450 SFHQ images resized to 512 px;
YuNet finds exactly one face (score >= 0.85); frontal (yaw proxy <= 0.08, roll <= 6 degrees); neutral
(MediaPipe blendshapes mouthSmile <= 0.3, jawOpen <= 0.15, eyeBlink <= 0.5); even light (face luma 80-200,
left/right ratio >= 0.8); SFace < 0.363 and ArcFace < 0.30 against every face found in every 5th frame of the 6
clips. 153 passed; a manual pass removed children, glasses, sunglasses, a grayscale image and stylised faces;
32 donors kept. The donor images, landmarks, hashes and provenance stay in the local, git-ignored
`data/donors_synthetic/` (PROVENANCE.md there). Donor for the table: SFHQ part 3 image 00001482.

**Every `warp`, `ghost` and `liveportrait` output, with either donor, is "reduced risk, not de-identified".**
`ghost` and `liveportrait` stay research use only and local. `fill` stays the default and the only mode
treated as shareable. `ghost` is the default swap backend (owner decision, 2026-10-29). Public figures use the
mesh donor, `fill` or mosaic; no SFHQ images or SFHQ-donor swaps are published.

Measured baselines (no donor, one row each; `src/segment_export.py`, `mask_faces` styles `mosaic8`,
`mosaic16`, `gauss`): the same region as `fill` (each face's landmark hull grown by 25 %, subject and bystanders;
the whole frame when there is no subject face). W = the face box width (horizontal extent of that face's
landmarks; frame width for whole-frame frames).

- `mosaic8`: square blocks of round(W / 8) px, about 8 blocks across the face box (W = 108 px on GRID lwwy7p
  frame 20 gives 13 px blocks). **Reduced risk, not de-identified.**
- `mosaic16`: square blocks of round(W / 16) px, about 16 blocks across (7 px blocks at W = 108 px). **Not de-identifying.**
  (rule and per-clip results in [FACE_DEID_TRADEOFF.md](FACE_DEID_TRADEOFF.md)).
- `gauss`: Gaussian blur, sigma = W / 6 px, square kernel 2 * ceil(3 * sigma) + 1 px (sigma 18 px, kernel 109 px
  at W = 108 px). **Local review only.**

Blocks start at the top-left corner of the grown hull's bounding box (area-averaged, nearest-neighbour upscale); only
pixels inside the hull change.

### Measurements: baselines, mesh donor and SFHQ donor

| Method | Donor | Clips (frames) | MediaPipe hit | YuNet hit | SFace mean (n; share >= 0.363) | ArcFace mean (n; share >= 0.30; share >= 0.45) | Landmark r | Blendshape r | ms/frame (box CPU) |
|---|---|---|---|---|---|---|---|---|---|
| fill (shareable default) | - | GRID 4/4 (300) | 0 % | 0 % | no face (n 0) | no face (n 0) | (0 frames) | (0 frames) | 1.1 |
| mosaic, 8 blocks across face | - | GRID 4/4 (300) | 11 % | 50 % | 0.19 (151; 0 %) | 0.10 (151; 0 %; 0 %) | (38 frames) | (38 frames) | 1.1 |
| mosaic, 16 blocks across face | - | GRID 4/4 (300) | 100 % | 100 % | 0.57 (300; 99 %) | 0.58 (300; 100 %; 100 %) | 0.25 | 0.32 | 1.1 |
| Gaussian blur, sigma face width / 6 (local review only) | - | GRID 4/4 (300) | 1.3 % | 0 % | no face (n 0) | no face (n 0) | (4 frames) | (4 frames) | 8.1 |
| warp, donor geometry | mesh render (seed 2) | GRID 4/4 (300) | 100 % | 100 % | 0.31 (300; 8.7 %) | 0.18 (300; 0 %; 0 %) | 0.78 | 0.68 | 11.2 |
| warp, donor geometry | SFHQ synthetic photo | GRID 4/4 (300) | 100 % | 100 % | 0.48 (300; 100 %) | 0.29 (300; 41 %; 0 %) | 0.69 | 0.54 | 11.5 |
| ghost | mesh render (seed 2) | GRID 4/4 (300) | 100 % | 100 % | 0.36 (300; 43 %) | 0.13 (300; 0 %; 0 %) | 0.78 | 0.68 | 384.2 |
| ghost | SFHQ synthetic photo | GRID 4/4 (300) | 100 % | 100 % | 0.38 (300; 64 %) | 0.22 (300; 4.0 %; 0 %) | 0.82 | 0.74 | 372.8 |
| liveportrait | mesh render (seed 2) | GRID 4/4 (300) | 100 % | 100 % | 0.27 (300; 1.7 %) | 0.13 (300; 0 %; 0 %) | 0.14 | 0.02 | 1950.6 |
| liveportrait | SFHQ synthetic photo | GRID 4/4 (300) | 100 % | 100 % | 0.35 (300; 39 %) | 0.17 (300; 0 %; 0 %) | 0.60 | 0.39 | 1886.7 |
| fill (shareable default) | - | Intel 2/2 (2810) | 0.2 % | 0.7 % | 0.75 (19; 79 %) | 0.73 (19; 79 %; 79 %) | (19 frames) | (19 frames) | 1.1 |
| mosaic, 8 blocks across face | - | Intel 2/2 (2810) | 13 % | 51 % | 0.19 (1764; 5.4 %) | 0.07 (1764; 2.0 %; 0.9 %) | 0.16 | 0.01 | 1.1 |
| mosaic, 16 blocks across face | - | Intel 2/2 (2810) | 68 % | 73 % | 0.54 (3849; 89 %) | 0.55 (3849; 92 %; 86 %) | 0.25 | 0.09 | 1.1 |
| Gaussian blur, sigma face width / 6 (local review only) | - | Intel 2/2 (2810) | 0.5 % | 4.2 % | 0.27 (124; 12 %) | 0.14 (124; 11 %; 11 %) | (16 frames) | (16 frames) | 8.1 |
| warp, donor geometry | mesh render (seed 2) | Intel 2/2 (2810) | 73 % | 73 % | 0.22 (3866; 5.7 %) | 0.11 (3866; 0.8 %; 0.4 %) | 0.50 | 0.29 | 11.2 |
| warp, donor geometry | SFHQ synthetic photo | Intel 2/2 (2810) | 73 % | 73 % | 0.27 (3917; 26 %) | 0.12 (3917; 1.0 %; 0.4 %) | 0.47 | 0.27 | 11.5 |
| ghost | mesh render (seed 2) | Intel 2/2 (2810) | 73 % | 73 % | 0.31 (3920; 31 %) | 0.13 (3920; 1.0 %; 0.4 %) | 0.55 | 0.41 | 384.2 |
| ghost | SFHQ synthetic photo | Intel 2/2 (2810) | 73 % | 73 % | 0.26 (3928; 17 %) | 0.10 (3928; 1.2 %; 0.5 %) | 0.53 | 0.44 | 372.8 |
| liveportrait | mesh render (seed 2) | Intel 2/2 (2810) | 71 % | 70 % | 0.22 (3624; 2.2 %) | 0.10 (3624; 0.5 %; 0.4 %) | 0.18 | -0.01 | 1950.6 |
| liveportrait | SFHQ synthetic photo | Intel 2/2 (2810) | 68 % | 72 % | 0.10 (3800; 2.3 %) | 0.04 (3800; 0.4 %; 0.4 %) | 0.32 | 0.20 | 1886.7 |

Clips: GRID s14 bbad9n, bbae1p, bbae2a, lwwy7p (300 frames) and the Intel sample clips
face-demographics-walking-and-pause and head-pose-face-detection-female-and-male (2810 frames). Columns as in the
phase 3 table. ms/frame = the method only, on the working machine (shared VM, 8 vCPU "Intel(R) Xeon(R)
Processor", 15.6 GB RAM, no GPU, torch 2.15.0+cpu). ms/frame are render-only on GRID lwwy7p (75 frames) with the box otherwise idle (8 vCPU Xeon, 8 threads), landmarks excluded. No donor, ms/frame: fill 1.1, mosaic8 1.1, mosaic16 1.1, gauss 8.1. Swaps, ms/frame: SFHQ donor warp 11, ghost 373, liveportrait 1887; mesh donor warp 11, ghost 384, liveportrait 1951. Part of the phase 4 GHOST and LivePortrait rendering and the donor sweep ran on a separate workstation to finish sooner; those timings are not reported here.

Donor spread (warp and ghost on GRID lwwy7p, 75 frames, each of the 32 SFHQ donors):

| Method | Donors | YuNet hit (min-max) | SFace share >= 0.363: median (min-max) over donors | ArcFace share >= 0.30: median (min-max) | ArcFace share >= 0.45: median (min-max) |
|---|---|---|---|---|---|
| warp, donor geometry | 32 SFHQ | 100 %-100 % | 55 % (0 %-100 %) | 0 % (0 %-100 %) | 0 % (0 %-15 %) |
| ghost | 32 SFHQ | 100 %-100 % | 31 % (0 %-100 %) | 0 % (0 %-12 %) | 0 % (0 %-0 %) |

Observations:

- GRID, `warp`: SFace share >= 0.363 8.7 % (mesh) vs 100.0 % (SFHQ); ArcFace share >= 0.30 0.0 % vs 41.0 %, >= 0.45 0.0 % vs 0.0 %; YuNet hit 100.0 % vs 100.0 %; landmark r 0.78 vs 0.69; blendshape r 0.68 vs 0.54.
- GRID, `ghost`: SFace share >= 0.363 42.7 % (mesh) vs 64.3 % (SFHQ); ArcFace share >= 0.30 0.0 % vs 4.0 %, >= 0.45 0.0 % vs 0.0 %; YuNet hit 100.0 % vs 100.0 %; landmark r 0.78 vs 0.82; blendshape r 0.68 vs 0.74.
- GRID, `liveportrait`: SFace share >= 0.363 1.7 % (mesh) vs 38.7 % (SFHQ); ArcFace share >= 0.30 0.0 % vs 0.0 %, >= 0.45 0.0 % vs 0.0 %; YuNet hit 100.0 % vs 100.0 %; landmark r 0.14 vs 0.60; blendshape r 0.02 vs 0.39.
- Intel, `warp`: SFace share >= 0.363 5.7 % (mesh) vs 25.7 % (SFHQ); ArcFace share >= 0.30 0.8 % vs 1.0 %, >= 0.45 0.4 % vs 0.4 %; YuNet hit 73.0 % vs 73.0 %; landmark r 0.50 vs 0.47; blendshape r 0.29 vs 0.27.
- Intel, `ghost`: SFace share >= 0.363 31.1 % (mesh) vs 16.6 % (SFHQ); ArcFace share >= 0.30 1.0 % vs 1.2 %, >= 0.45 0.4 % vs 0.5 %; YuNet hit 73.0 % vs 73.0 %; landmark r 0.55 vs 0.53; blendshape r 0.41 vs 0.44.
- Intel, `liveportrait`: SFace share >= 0.363 2.2 % (mesh) vs 2.3 % (SFHQ); ArcFace share >= 0.30 0.5 % vs 0.4 %, >= 0.45 0.4 % vs 0.4 %; YuNet hit 70.4 % vs 72.2 %; landmark r 0.18 vs 0.32; blendshape r -0.01 vs 0.20.
- GRID, `mosaic8` (mosaic, 8 blocks, no donor): MediaPipe hit 10.7 %, YuNet hit 50.3 %; SFace share >= 0.363 0.0 % (n 151); ArcFace share >= 0.30 0.0 %, >= 0.45 0.0 %; landmark r -0.07; blendshape r 0.01.
- GRID, `mosaic16` (mosaic, 16 blocks, no donor): MediaPipe hit 100.0 %, YuNet hit 100.0 %; SFace share >= 0.363 99.0 % (n 300); ArcFace share >= 0.30 100.0 %, >= 0.45 99.7 %; landmark r 0.25; blendshape r 0.32.
- GRID, `gauss` (Gaussian blur, no donor): MediaPipe hit 1.3 %, YuNet hit 0.0 %; SFace share >= 0.363 - (n 0); ArcFace share >= 0.30 -, >= 0.45 -; landmark r -; blendshape r 0.07.
- Intel, `mosaic8` (mosaic, 8 blocks, no donor): MediaPipe hit 12.7 %, YuNet hit 50.6 %; SFace share >= 0.363 5.4 % (n 1764); ArcFace share >= 0.30 2.0 %, >= 0.45 0.9 %; landmark r 0.16; blendshape r 0.01.
- Intel, `mosaic16` (mosaic, 16 blocks, no donor): MediaPipe hit 68.3 %, YuNet hit 72.8 %; SFace share >= 0.363 88.8 % (n 3849); ArcFace share >= 0.30 92.1 %, >= 0.45 86.2 %; landmark r 0.25; blendshape r 0.09.
- Intel, `gauss` (Gaussian blur, no donor): MediaPipe hit 0.5 %, YuNet hit 4.2 %; SFace share >= 0.363 12.1 % (n 124); ArcFace share >= 0.30 11.3 %, >= 0.45 11.3 %; landmark r 0.99; blendshape r -0.05.
- `mosaic8` is **reduced risk, not de-identified**; `mosaic16` is **not de-identifying**; `gauss` is **local review only**.
- All `warp`, `ghost` and `liveportrait` outputs, with either donor, are **reduced risk, not de-identified**; `fill` stays the only shareable default.

Motion preservation vs identity for every method and donor: [FACE_DEID_TRADEOFF.md](FACE_DEID_TRADEOFF.md).

## What is in this repository

- `src/deid_swap.py`: `FaceSwapper` protocol, backend registry, configuration checks (licence flag, model
  files, donor and donor licence, `research_use_acknowledged` for `ghost` and `liveportrait`), lazy loaders
  for `warp`, `ghost` and `liveportrait`, `ghost` as the default swap backend (`DEFAULT_BACKEND`), and the
  status string "reduced risk, not de-identified".
- `src/deid_warp.py`: mesh loader, procedural donor renderer, `WarpSwapper`.
- `src/deid_ghost.py`: `GhostSwapper`, a CPU adapter that loads GHOST's code and weights from a local
  checkout outside the repository (research use only).
- `src/deid_liveportrait.py`: `LivePortraitSwapper`, a CPU adapter that loads LivePortrait's code and
  weights from a local checkout outside the repository (research use only).
- `src/deid_check.py`: check harness. Detector hit rate (MediaPipe; YuNet with `--yunet`), identity
  similarity (SFace on YuNet detections; ArcFace with a research-use-only InsightFace model), landmark-region
  and blendshape motion correlation.
- `src/segment_export.py`: swap sidecars carry `Deidentified: false`, `DeidentificationStatus: "reduced
  risk, not de-identified"` and, for `ghost` and `liveportrait`, `ResearchUseOnly: true`. `fill` stays the
  default and the only mode treated as shareable.
- No third-party code, weights, donor images or downloads. `scripts/fetch_models.py --face-check` fetches the
  phase 2 files above; the phase 3 code and weights live in a git-ignored `third_party_local/` directory
  outside the repository.
