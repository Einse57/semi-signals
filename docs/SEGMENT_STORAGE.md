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

netCDF round-trips through xarray (`xarray.open_dataset`; `movement.io.load` has no netCDF reader). Without
xarray / netCDF4, `segment_store` writes `.npz` with the same arrays and dimension names. `movement` is
optional; install it in a separate environment (its dependency `sleap-io` installs a top-level `tests` package).

**NWB.** Not implemented (`segment_store.to_nwb` raises `NotImplementedError`).

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
| `fill` | export (default) | face areas painted with their mean colour: the subject's landmark hull grown by 25 %, every other face the landmarker finds (`--no-bystanders` turns this off), and every YuNet face no landmark mask covers (box padded 20 %, `--fallback-pad`). Frames without the subject's face are filled whole (`--missing-face frame`, default). Then YuNet checks every output frame (score >= 0.6, `--yunet-score`): a flagged frame gets its flagged YuNet boxes filled and is checked once more; a face still found marks the export not shareable and lists the frames in `UnmaskedFaceCheck`. The written file is decoded and checked again the same way (`EncodedOutputCheck`); flagged frames get the flagged boxes filled with a wider margin and the file is written and checked again, up to two passes (`EncodedRefill`). The export is shareable only if the final check of the written file is clean. Without the YuNet model (`--yunet-model`), fill runs on landmarks only and is not marked shareable |
| `swap` | export (opt-in) | **reduced risk, not de-identified**: face swap to a donor (`--deid swap --swap-config swap.json`). Backends `ghost` (default) and `liveportrait` (research use only, need `research_use_acknowledged`) and `warp`. Sidecar: `Deidentified: false`, `DeidentificationStatus`, `SwapBackend`, `SwapDonorLicence`, `ResearchUseOnly` |
| `blur` | raw only (opt-in) | **local review only, not for sharing.** Written under `--raw-root`; `--blur-in-export` puts it in the export with `NotForSharing: true`, `Deidentified: false` |
| `points` | export | the subject's landmarks drawn on a black canvas, no image content |

Each video sidecar records the variant, frames with a face, `BystanderFacesMasked`, the preprocessing mode
and `Deidentified`. With `--deid-method fill|ghost` it also has `DeidentificationStatus` and `Shareable`
(true only for a fill export that passed the unmasked-face check; otherwise `ShareableBlockedReason`).
`--shareable` exits with code 3 when the check blocks it.

## Face de-identification in the UI (face mode)

The launcher (Annotation mode Face or Both) and the annotation panel ("De-ID export...") offer:

- **View: Original**: not de-identified; viewing and preview only.
- **De-identify: Fill** (default): "shareable fallback (default)"; the only choice that can be exported for
  sharing. The log shows the unmasked-face check result, and a NOT SHAREABLE line with the frame list when
  it fails. Without the YuNet model the selector says so and sharing is blocked.
- **De-identify: GHOST**: "reduced risk, not de-identified"; research use only, behind a research-use
  acknowledgement (not saved). Missing weights point to `python scripts/fetch_models.py --ghost`.
- **Landmarker**: MediaPipe (CPU, default) or OpenVINO on a device OpenVINO finds (CPU, GPU, NPU, AUTO).
  A device that doesn't load falls back to CPU with a note; sidecars record it as `FaceLandmarker`.
- **Preview (before/after)**: the middle frame, nothing saved.

Export runs `python -m src.segment_export ... --deid-method fill|ghost` into `~/semi-signals-local/export`
(raw into `~/semi-signals-local/raw`) and checks the same rules again before writing.

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
- Fill covers faces the landmarker or YuNet finds; the unmasked-face check uses the same detector, so a face
  both miss is not caught. Check every clip before sharing.
- Landmarks, blendshapes and landmark-only video can be biometric identifiers (HIPAA, GDPR, state
  biometric laws).
- Body, clothing, voice (if audio is added), room and timing can identify a person.
- Face swapping is reduced risk, not de-identification, with every backend. `python -m src.deid_check`
  reports detector hit rate, identity similarity and motion correlation between original and output.
  Licences: [FACE_SWAP_LICENCES.md](FACE_SWAP_LICENCES.md).

## Test data used during development (not in this repository)

- GRID audio-visual sentence corpus (Cooke, Barker, Cunningham, Shao, 2006), talker s14,
  [zenodo.org/records/3625687](https://zenodo.org/records/3625687), CC BY 4.0.
- Intel IoT DevKit sample videos (`head-pose-face-detection-female-and-male.mp4`,
  `face-demographics-walking-and-pause.mp4`),
  [github.com/intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos), CC BY 4.0.
- MediaPipe's sample image `business-person.png` (used by five tests when present, see the README Tests
  section; skipped otherwise).
