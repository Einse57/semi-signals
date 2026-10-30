# Segment storage (PROVISIONAL)

> **Provisional, not locked.** Format version `semi-signals.segment/0.1-provisional`. File names, array
> layouts and sidecar fields may change. All reading and writing goes through one module,
> `src/segment_store.py`, so a later change touches one place. Every JSON sidecar carries
> `SemiSignalsFormat`; `segment_store.check_version` refuses other versions.

`src/segment_export.py` cuts a video into segments (one per annotated event, or the whole clip) and writes
two trees:

- **raw** (`--raw-root`, local only): the unmodified clip of each segment, its `scans.tsv` with the real
  acquisition time, and the private mapping. It must be outside the export; inside this repository it must
  be a gitignored folder (`local/`, `raw_segments/`, `segments/`, `export/` and `*_desc-raw_video.*` are in
  `.gitignore`). Never commit or upload it.
- **de-identified** (`--out`): face-filled and landmarks-only video (by default), landmarks, blendshapes, events, shifted
  `scans.tsv`. "De-identified" here means reduced re-identification risk, not anonymous (see Limits).

```
python -m src.segment_export --input clip.mp4 --annotations clip.eaf \
       --out export/deid --raw-root ~/semi-signals-local/raw \
       --face-preprocess detect --pre-s 2 --post-s 2 --acq-time 2026-03-04T10:11:12 [--hed] \
       [--deid fill,points] [--missing-face frame|none] [--no-bystanders] \
       [--deid ...,swap --swap-config swap.json] [--deid ...,blur [--blur-in-export]]
```

## Layout

```
<out>/dataset_description.json                      DatasetType "derivative", BEP047 and privacy notes
<out>/sub-<id>/sub-<id>_scans.tsv                   filename, acq_time (date-shifted, year <= 1925, or n/a)
<out>/sub-<id>/beh/
    sub-<id>_task-<task>_run-<NN>_recording-<rec>_desc-fill_video.mp4   (+ .json)
    sub-<id>_task-<task>_run-<NN>_recording-<rec>_desc-points_video.mp4 (+ .json)
    sub-<id>_task-<task>_run-<NN>_recording-<rec>_landmarks.nc          (+ .json)
    sub-<id>_task-<task>_run-<NN>_recording-<rec>_blendshapes.nc
    sub-<id>_task-<task>_run-<NN>_events.tsv                            (+ .json)
<raw-root>/dataset_description.json
<raw-root>/private_mapping.csv
<raw-root>/sub-<id>/sub-<id>_scans.tsv              real acq_time
<raw-root>/sub-<id>/beh/..._recording-<rec>_desc-raw_video.mp4          (+ .json)
<raw-root>/sub-<id>/beh/..._recording-<rec>_desc-blur_video.mp4         (+ .json; only with --deid blur)
```

- `sub-<id>`: random pseudonym (8 hex characters), reused for the same source video (same SHA-256) through
  the mapping, or set with `--sub`. One run per segment, numbered on from the subject's earlier runs.
- Entity order follows BIDS (sub, ses, task, acq, run, recording, desc); labels must be alphanumeric.
- **BEP047.** Video names and the `beh/` folder follow the BIDS Extension Proposal 047 for audio and video
  ([bids-specification PR #2231](https://github.com/bids-standard/bids-specification/pull/2231): `_video`
  suffix, `recording-` entity for the camera, media sidecar fields `RecordingDuration`, `ImageWidth`,
  `ImageHeight`, `VideoCodec`, `VideoFrameRate`, `VideoFrameCount`). **BEP047 is a proposal and not part of
  the BIDS specification**; it may change before or instead of being merged. `desc-` is a derivatives
  entity, which is why the export is labelled a derivative dataset. `_landmarks` and `_blendshapes` are not
  BIDS suffixes.

## Landmarks and blendshapes

`*_landmarks.nc` is an xarray dataset in the layout of [`movement`](https://movement.neuroinformatics.dev)
pose datasets (`ds_type = "poses"`):

| Variable | Dims | Content |
|---|---|---|
| `position` | (time, space = x/y/z, keypoint = 478, individual = 1) | MediaPipe landmarks in pixels of the segment video; z = MediaPipe relative depth on the x scale (smaller = nearer the camera). NaN = no face |
| `confidence` | (time, keypoint, individual) | per-frame face score repeated over the keypoints (MediaPipe reports no per-landmark confidence) |

Keypoints are named `mp000` ... `mp477` by MediaPipe face-mesh index (468 mesh points + 10 iris points).
`time` is seconds from the segment start. Attributes: `fps`, `time_unit`, `source_software`, format version,
face preprocessing mode.

`*_blendshapes.nc`: `blendshapes` (time, blendshape = 52), named by MediaPipe category (`_neutral`,
`browDownLeft`, ...).

**Why netCDF.** A 478-keypoint 3D `movement` dataset was saved with `xarray.Dataset.to_netcdf` and
reloaded with `xarray.open_dataset`; values, NaN positions, coordinates and attributes came back identical
with the netCDF4, h5netcdf and scipy engines, and `movement.kinematics.compute_speed` ran on the reloaded
data (movement 0.17.0). `movement.io.load` has no netCDF reader, so files are opened with xarray. When
xarray or netCDF4 is missing, `segment_store` writes `.npz` with the same arrays and dimension names
(`dims_position`, `dims_confidence`). semi-signals itself needs only xarray + netCDF4; `movement` is
optional (install it in a separate environment: one of its dependencies, `sleap-io`, installs a top-level
`tests` package that shadows a project's own `tests` folder).

**NWB.** An NWB / `ndx-pose` export is not implemented (`segment_store.to_nwb` raises
`NotImplementedError`). The `movement` layout maps onto `ndx-pose` `PoseEstimationSeries` per keypoint if
it is added later.

## Events

The ELAN `.eaf` written by the annotation tool stays the editable copy. `events.tsv` is generated from it
(`segment_store.eaf_to_events`), with times relative to the segment start:

| Column | Content |
|---|---|
| `onset` | s from the segment start |
| `duration` | clinical end - onset; `n/a` for onset-only events; 0 for EEG rows |
| `trial_type` | `clinical` (clinical onset) or `EEG` (EEG onset of the same event) |
| `side` | `L`, `R`, `both` or `n/a` (face events) |
| `mode` | `body` or `face` |
| `eeg_onset` | EEG onset of the event, s from the segment start, or `n/a` |
| `HED` | only with `--hed`: empty (`n/a`) column for HED tags from the HED-SCORE library (`score`) |

Free-text notes are never written to `events.tsv`. `events.json` describes the columns and levels.
EEG alignment: the segment start is `acq_time` in `scans.tsv` (BIDS convention); the de-identified copy
shifts the date by a per-subject number of days into 1900-1925 (time of day kept) and the shift is kept
in the private mapping.

## Video variants

| `desc` | Tree | Content |
|---|---|---|
| `raw` | raw only | the source frames of the segment |
| `fill` | export (default) | face areas painted with their mean colour: the subject's landmark hull grown by 25 %, plus every other face the landmarker finds (whole frame + four tiles; `--no-bystanders` turns this off). Frames without the subject's face are filled whole (`--missing-face frame`, default) |
| `swap` | export (opt-in) | REDUCED RISK, NOT DE-IDENTIFIED: face swap to a donor face (`--deid swap`; `ghost` is accepted as an alias). Needs `--swap-config` with a licence-reviewed backend and a licensed donor, and exits with an error otherwise. Backends: `ghost` (the default swap backend, used when the configuration names none) and `liveportrait` (research use only; need `research_use_acknowledged`; code and weights outside the repository). `warp`: landmark-driven warp of a synthetic donor. The subject's face outline, hair, skin colour, head pose and expression remain, and an identity check still finds partial similarity (see [FACE_SWAP_LICENCES.md](FACE_SWAP_LICENCES.md)). Sidecar: `Deidentified: false`, `DeidentificationStatus: "reduced risk, not de-identified"`, `SwapBackend`, `SwapDonorLicence`, and `ResearchUseOnly: true` for ghost and liveportrait. Not a shareable default |
| `blur` | raw only (opt-in) | **local review only, not for sharing.** Face areas pixelated and blurred. Written under `--raw-root`; `--blur-in-export` puts it in the export, with `NotForSharing: true` and `Deidentified: false` in its sidecar |
| `points` | export | the subject's landmarks drawn on a black canvas, no image content |
| `mosaic8` | export (opt-in, `--deid-method mosaic8`) | REDUCED RISK, NOT DE-IDENTIFIED: face areas pixelated, square blocks of face width / 8 px. Sidecar `Deidentified: false`, `Shareable: false` |
| `mosaic16` | raw only (`--deid-method mosaic16`) | **not de-identifying** (provisional: identity still matched on 5 of 6 clips scored; the final check is in [FACE_DEID_TRADEOFF.md](FACE_DEID_TRADEOFF.md)). Blocks of face width / 16 px. Always under `--raw-root`, `NotForSharing: true` |
| `gauss` | raw only (`--deid-method gauss`) | **local review only.** Gaussian blur, sigma = face width / 6 px. Always under `--raw-root`, `NotForSharing: true`; never exported or shareable |

Each video sidecar records the variant, the number of frames with a face, the number of bystander faces
masked (`BystanderFacesMasked`), the preprocessing mode and `Deidentified`; the blur variant also has
`NotForSharing`.

Variants chosen with `--deid-method` (the face de-identification selector in the UI, below) also carry
`DeidentificationStatus` (the UI label) and `Shareable` (true only for `fill`).

## Face de-identification in the UI (face mode)

The launcher (Annotation mode Face or Both) and the annotation panel ("De-ID export...", face and both
modes) have the same selector (`src/deid_ui.py`; names, labels and rules in `src/deid_options.py`):

- **Method**: Fill (default), Mosaic 8, Mosaic 16, Gaussian blur, or Face swap. Fill is the default
  overall because it is the shareable fallback.
- **Swap backend** (only with Face swap): GHOST (the default swap), warp, LivePortrait.
- **Label** next to the choice: fill "shareable fallback (default)"; mosaic8, warp, GHOST, LivePortrait
  "reduced risk, not de-identified"; mosaic16 "not de-identifying" (provisional); Gaussian blur "local
  review only".
- **Export for sharing** can only be ticked with fill.
- **Research-use acknowledgement** for GHOST and LivePortrait: without it the export and preview buttons
  refuse; it is passed to the backend as `research_use_acknowledged` (`--research-use-acknowledged`), and
  the UI choice replaces the configuration file's value. It is not saved between sessions.
- **Missing models**: the selector lists what is missing and where to fetch it from the official sources
  (GHOST `download_models.sh`, Hugging Face KlingTeam/LivePortrait, `scripts/fetch_models.py --face` /
  `--face-check`); nothing is redistributed or downloaded automatically ([FACE_SWAP_LICENCES.md](FACE_SWAP_LICENCES.md)).
- **Preview (before/after)**: one frame of the video (middle frame) before and after the chosen method,
  nothing saved.

Export runs `python -m src.segment_export ... --deid-method <method>` into `~/semi-signals-local/export`
(raw and local-only variants into `~/semi-signals-local/raw`). segment_export checks the method, the
shareable rule, the acknowledgement and the configuration's backend again and exits with the reason before
writing anything.

## Sidecar fields (custom)

`*_landmarks.json`: `SemiSignalsFormat`, `Provisional`, `Layout`, `CoordinateFrame`, `Units`,
`Deidentified`, `AnnotationMode`, `FacePreprocess`, `Confidence`, `BlendshapesFile`, `PrivateMappingRef`
(file name only; the mapping itself is never in the export), `Model`, `FramesWithFace`, `Frames`.

## Private mapping

`<raw-root>/private_mapping.csv`: `sub`, `run`, segment start and end in the source (s), source video path
and SHA-256, annotation file, real `acq_time`, date shift (days), raw clip path, export time. It must be
outside `--out` (checked). Keep it with the source videos.

## Limits

- Blur does not de-identify: on the test clips the landmarker still detected a face in 9-60 % of blurred
  frames, so blur is for local review only. On the same clips it detected no face in filled frames.
- Fill and blur only cover faces the landmarker finds; faces it never finds (small, turned away, occluded)
  are not masked. Check every de-identified clip before sharing.
- Landmarks, blendshapes and landmark-only video can be biometric identifiers (HIPAA, GDPR, state
  biometric laws).
- Body, clothing, voice (if audio is added), room and timing can identify a person.
- Face swapping (`--deid swap`) is reduced risk, not de-identification, with every backend (`warp`, and the
  research-only `ghost` and `liveportrait`). The licence audit and measurements are in
  [FACE_SWAP_LICENCES.md](FACE_SWAP_LICENCES.md). The check harness `python -m src.deid_check` reports the
  detector hit rate (MediaPipe, YuNet with `--yunet`), identity similarity (SFace; ArcFace when a local
  research-only model is configured) and landmark and blendshape motion correlation between original and
  de-identified clips.

## Test data used during development (not in this repository)

- GRID audio-visual sentence corpus (Cooke, Barker, Cunningham, Shao, 2006), talker s14,
  [zenodo.org/records/3625687](https://zenodo.org/records/3625687), CC BY 4.0.
- Intel IoT DevKit sample videos (`head-pose-face-detection-female-and-male.mp4`,
  `face-demographics-walking-and-pause.mp4`),
  [github.com/intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos), CC BY 4.0.
- MediaPipe's sample image `business-person.png` (used by two tests when present; skipped otherwise).
