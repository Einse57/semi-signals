# Face de-identification: third-party licences

What the face de-identification code uses, where it comes from and its licence. This repository
redistributes no third-party code, weights, datasets or images: you fetch each one from the source below and
accept its terms yourself. Weights and swap outputs stay local and are for non-commercial research. This is a
reading of published terms, not legal advice.

No SFHQ images or SFHQ-donor swaps are in this repository. The only media in it is Figure 1 (row 18a).

## Components

| # | Component | Source | Licence | In this repository |
|---|---|---|---|---|
| 1 | GHOST code (AEI-Net, `network/`) | <https://github.com/ai-forever/ghost> | Apache-2.0 | not redistributed / fetched by user (`--ghost`); adapter `src/deid_ghost.py` only |
| 2 | GHOST generator `G_unet_2blocks.pth` | <https://github.com/ai-forever/ghost/releases/tag/sber-swap-v2.0> | Apache-2.0 (repository licence; release has no separate text); trained on VGGFace2 (non-commercial research) | not redistributed / fetched by user (`--ghost`) |
| 3 | InsightFace ArcFace `backbone.pth` + `iresnet.py` (GHOST source identity) | <https://github.com/ai-forever/ghost/releases/tag/arcface>; terms: <https://github.com/deepinsight/insightface#license> | model: non-commercial research only; `iresnet.py`: MIT | not redistributed / fetched by user (`--ghost`) |
| 4 | InsightFace `glintr100.onnx` (ArcFace scorer in `src/deid_check.py`) | <https://github.com/ai-forever/ghost/releases/tag/antelope>; terms: <https://github.com/deepinsight/insightface/tree/master/model_zoo> | non-commercial research only | not redistributed / fetched by user (`--ghost`) |
| 5 | InsightFace buffalo_l `det_10g.onnx`, `2d106det.onnx` (bundled with LivePortrait, not loaded) | <https://huggingface.co/KlingTeam/LivePortrait> | non-commercial research only | not redistributed / fetched by user |
| 7 | SPADE-derived parts (LivePortrait decoder) | <https://github.com/NVlabs/SPADE> | CC BY-NC-SA 4.0 | not redistributed / fetched by user |
| 8 | LivePortrait code | <https://github.com/KlingTeam/LivePortrait> | MIT (with SPADE / face vid2vid derived parts) | not redistributed / fetched by user; adapter `src/deid_liveportrait.py` only |
| 9 | LivePortrait weights | <https://huggingface.co/KlingTeam/LivePortrait> | MIT (model card); training data includes non-commercial and private sets | not redistributed / fetched by user |
| 10 | SFace (OpenCV Zoo) | <https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface> | Apache-2.0 | not redistributed / fetched by user (`scripts/fetch_models.py --face-check`) |
| 11 | YuNet (OpenCV Zoo; fill fallback and unmasked-face check) | <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet> | MIT | not redistributed / fetched by user (`scripts/fetch_models.py --face-check`) |
| 12 | MediaPipe Face Landmarker, `canonical_face_model.obj` | <https://github.com/google-ai-edge/mediapipe> | Apache-2.0 | not redistributed / fetched by user (`scripts/fetch_models.py`) |
| 12a | MediaPipe graph and calculators re-implemented for the OpenVINO face backend (`src/face_ov/geometry.py`, `landmarker.py`, `face_geometry.py`); the model bundle is converted locally at load time | <https://github.com/google-ai-edge/mediapipe> | Apache-2.0, modified | yes, with Apache-2.0 file headers, `LICENSES/Apache-2.0.txt` and NOTICE; no weights or converted files |
| 13 | SFHQ synthetic faces, parts 3 and 4 (D. Beniaguev) | <https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-3>, <https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-4> | CC0 1.0 (Kaggle); code MIT | not redistributed / fetched by user from Kaggle; no images, no download code |
| 15 | StyleGAN2 / FFHQ lineage of SFHQ part 3 | <https://github.com/NVlabs/stylegan2/blob/master/LICENSE.txt>, <https://github.com/NVlabs/ffhq-dataset> | non-commercial; FFHQ CC BY-NC-SA 4.0 | not used directly; no SFHQ images or swaps published |
| 16 | Stable Diffusion 2.1 lineage of SFHQ part 4 | <https://huggingface.co/stabilityai/stable-diffusion-2-1-base> | licence text not obtainable (404 / 401) | not used directly; no SFHQ images or swaps published |
| 18 | GRID audio-visual speech corpus (Cooke, Barker, Cunningham, Shao) | <https://zenodo.org/records/3625687> | CC BY 4.0 | not redistributed / fetched by user; numbers only, plus Figure 1 (row 18a) |
| 18a | Figure `docs/figures/face_deid_grid_lwwy7p_fill_mosaic_blur.png` (GRID `lwwy7p` frames 20 and 45: original, fill, mosaic8, mosaic16, blur; no donor, no swap) | <https://zenodo.org/records/3625687> | CC BY 4.0 (modified derivative), not MIT | committed with attribution in NOTICE, `docs/figures/README.md` and FACE_DEID_TRADEOFF.md |
| 19 | Intel IoT DevKit sample videos | <https://github.com/intel-iot-devkit/sample-videos> | CC BY 4.0 | not redistributed / fetched by user; numbers only |
| 19a | Screenshot `docs/img/annotation_face_fill.png` (Fill export of `head-pose-face-detection-female-and-male.mp4` in the annotation tool) | <https://github.com/intel-iot-devkit/sample-videos> | CC BY 4.0 (modified derivative), not MIT | committed with attribution in NOTICE and `docs/img/README.md` |

Not used: InsightFace inswapper (licence by email), FLUX.1 [schnell] (gated), SCRFD detectors, GHOST
discriminators, v1 generator, super-resolution code and weights, AdaptiveWingLoss, MXNet landmarks, X-Pose and
LivePortrait animal weights, SimSwap (CC BY-NC 4.0), FaceFusion, DigiFace-1M, Face Synthetics, photos of real
people.

## Weights (local, sha256)

| File | Source | sha256 |
|---|---|---|
| `G_unet_2blocks.pth` | GHOST release `sber-swap-v2.0` | `85f130421c65722927c47502ea1b28b62be4893ca9180fec735b67be1e5edd8d` |
| `backbone.pth` | GHOST release `arcface` | `a566a62357f0c55b679d9ff2f022a294486568be0c00665d39029d0e46a8109b` |
| `iresnet.py` | GHOST release `arcface` | `990006bfcd6583bdc9775045dfe02cbb10904018c0c34825f05d9cdd834fd388` |
| `glintr100.onnx` | GHOST release `antelope` | `4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf` |
| `liveportrait/base_models/appearance_feature_extractor.pth` | HF KlingTeam/LivePortrait @ `82a4fa6` | `5279bb8654293dbdf327030b397f107237dd9212fb11dd75b83dfb635211ceb5` |
| `liveportrait/base_models/motion_extractor.pth` | same | `251e6a94ad667a1d0c69526d292677165110ef7f0cf0f6d199f0e414e8aa0ca5` |
| `liveportrait/base_models/spade_generator.pth` | same | `4780afc7909a9f84e24c01d73b31a555ef651521a1fe3b2429bd04534d992aee` |
| `liveportrait/base_models/warping_module.pth` | same | `2f61a6f265fe344f14132364859a78bdbbc2068577170693da57fb96d636e282` |
| `liveportrait/retargeting_models/stitching_retargeting_module.pth` | same | `3652d5a3f95099141a56986aaddec92fadf0a73c87a20fac9a2c07c32b28b611` |
| `liveportrait/landmark.onnx` (not loaded) | same | `31d22a5041326c31f19b78886939a634a5aedcaa5ab8b9b951a1167595d147db` |
| `insightface/models/buffalo_l/det_10g.onnx` (not loaded) | same | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |
| `insightface/models/buffalo_l/2d106det.onnx` (not loaded) | same | `f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf` |

Third-party checkouts: GHOST `44e58aad8600ee83ad4c8213aaa9698ccaf66c1c`, LivePortrait
`9b294b3d0536135442ea73cb01e6cb3ca7029dd3`, both in the git-ignored `third_party_local/`. Detection and
landmarks in both adapters come from MediaPipe.

## Fetching GHOST

`python scripts/fetch_models.py --ghost --accept-ghost-licences --skip-data --skip-vsvig` clones GHOST at the
commit above into `third_party_local/ghost` and downloads `G_unet_2blocks.pth`, `backbone.pth`, `iresnet.py`
and `glintr100.onnx` from <https://github.com/ai-forever/ghost/releases> into `third_party_local/weights/ghost`,
each checked against the sha256 above (a mismatch is deleted and reported). `iresnet.py` is InsightFace code
(MIT). Without `--accept-ghost-licences` the script lists each file's source and licence and asks; any answer
other than `y`, or no terminal, fetches nothing and exits with code 2. `--ghost-dir` picks another location;
`ghost_models.json` there holds the paths for the `ghost` and `arcface` configurations. Donor images are not
fetched.

## Run GHOST

1. Fetch GHOST (above) and the face model (`python scripts/fetch_models.py --face --skip-data --skip-vsvig`).
   Pick a donor image: a synthetic face, or one licensed for this use, never a real person without a licence.
2. Write `swap.json`. Copy the three `models` paths from `third_party_local/ghost_models.json` (relative paths
   are read from the folder you run in):
   ```json
   {"backend": "ghost", "licence_reviewed": true, "research_use_acknowledged": true,
    "models": {"code": "C:/semi-signals/third_party_local/ghost",
               "generator": "C:/semi-signals/third_party_local/weights/ghost/G_unet_2blocks.pth",
               "id_embedder": "C:/semi-signals/third_party_local/weights/ghost/backbone.pth"},
    "donor": "C:/donors/donor.png", "donor_licence": "synthetic, generated by ...", "device": "cpu"}
   ```
   `"donor_landmarks"` is optional: a `.npy` with the donor's face-mesh landmarks in image pixels (468 or 478
   rows, x and y first). Without it the donor face is found with the Face Landmarker.
3. Run it:
   ```bat
   python -m src.segment_export --input clip.mp4 --whole --out export --raw-root raw_local --mode face ^
       --deid-method ghost --research-use-acknowledged --swap-config swap.json
   ```
   In the UI, pick GHOST, set **Swap config** to `swap.json` and tick **Research use acknowledged (GHOST)**.
   The swap video's sidecar has `Shareable: false` and `ResearchUseOnly: true`: keep it local.

## Labels

- `fill`: shareable default. Gated by the unmasked-face check (see [FACE_DEID_TRADEOFF.md](FACE_DEID_TRADEOFF.md)).
- `warp`, `ghost`, `liveportrait`: **reduced risk, not de-identified**. `ghost` and `liveportrait` are research
  use only: the configuration must set `"research_use_acknowledged": true`, sidecars carry
  `ResearchUseOnly: true`, outputs stay local. `ghost` is the default swap backend.
- `mosaic8`: reduced risk, not de-identified. `mosaic16`: **not de-identifying**. `gauss` / blur: **local review only**.
- `src/deid_swap.py` refuses a configuration without `"licence_reviewed": true`.

Measurements for every method: [FACE_DEID_TRADEOFF.md](FACE_DEID_TRADEOFF.md).
