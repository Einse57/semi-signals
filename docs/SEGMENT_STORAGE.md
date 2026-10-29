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
- **de-identified** (`--out`): blurred and landmarks-only video, landmarks, blendshapes, events, shifted
  `scans.tsv`. "De-identified" here means reduced re-identification risk, not anonymous (see Limits).

```
python -m src.segment_export --input clip.mp4 --annotations clip.eaf \
       --out export/deid --raw-root ~/semi-signals-local/raw \
       --face-preprocess detect --pre-s 2 --post-s 2 --acq-time 2026-03-04T10:11:12 [--hed] \
       [--deid blur,points] [--blur-style blur|fill] [--blur-missing frame|none] [--no-bystander-blur]
```

## Layout

```
<out>/dataset_description.json                      DatasetType "derivative", BEP047 and privacy notes
<out>/sub-<id>/sub-<id>_scans.tsv                   filename, acq_time (date-shifted, year <= 1925, or n/a)
<out>/sub-<id>/beh/
    sub-<id>_task-<task>_run-<NN>_recording-<rec>_desc-blur_video.mp4   (+ .json)
    sub-<id>_task-<task>_run-<NN>_recording-<rec>_desc-points_video.mp4 (+ .json)
    sub-<id>_task-<task>_run-<NN>_recording-<rec>_landmarks.nc          (+ .json)
    sub-<id>_task-<task>_run-<NN>_recording-<rec>_blendshapes.nc
    sub-<id>_task-<task>_run-<NN>_events.tsv                            (+ .json)
<raw-root>/dataset_description.json
<raw-root>/private_mapping.csv
<raw-root>/sub-<id>/sub-<id>_scans.tsv              real acq_time
<raw-root>/sub-<id>/beh/..._recording-<rec>_desc-raw_video.mp4          (+ .json)
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
| `blur` | export | face areas blurred: the subject's landmark hull grown by 25 %, plus every other face the landmarker finds (whole frame + four tiles; `--no-bystander-blur` turns this off). `--blur-style fill` paints the areas with their mean colour instead. Frames without the subject's face are blurred whole (`--blur-missing frame`, default) |
| `points` | export | the subject's landmarks drawn on a black canvas, no image content |

Each video sidecar records the variant, the number of frames with a face, the number of bystander faces
blurred, the preprocessing mode and `Deidentified`.

## Sidecar fields (custom)

`*_landmarks.json`: `SemiSignalsFormat`, `Provisional`, `Layout`, `CoordinateFrame`, `Units`,
`Deidentified`, `AnnotationMode`, `FacePreprocess`, `Confidence`, `BlendshapesFile`, `PrivateMappingRef`
(file name only; the mapping itself is never in the export), `Model`, `FramesWithFace`, `Frames`.

## Private mapping

`<raw-root>/private_mapping.csv`: `sub`, `run`, segment start and end in the source (s), source video path
and SHA-256, annotation file, real `acq_time`, date shift (days), raw clip path, export time. It must be
outside `--out` (checked). Keep it with the source videos.

## Limits

- Blur is only as good as face detection. On the test clips the landmarker still detected a face shape in
  9-60 % of blurred frames (no detections with `--blur-style fill`); faces it never finds (small, turned
  away, occluded) are not blurred. Check every de-identified clip before sharing.
- Landmarks, blendshapes and landmark-only video can be biometric identifiers (HIPAA, GDPR, state
  biometric laws).
- Body, clothing, voice (if audio is added), room and timing can identify a person.
- Face swapping (for example GHOST) as a further de-identification step is future work and not implemented.

## Test data used during development (not in this repository)

- GRID audio-visual sentence corpus (Cooke, Barker, Cunningham, Shao, 2006), talker s14,
  [zenodo.org/records/3625687](https://zenodo.org/records/3625687), CC BY 4.0.
- Intel IoT DevKit sample videos (`head-pose-face-detection-female-and-male.mp4`,
  `face-demographics-walking-and-pause.mp4`),
  [github.com/intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos), CC BY 4.0.
- MediaPipe's sample image `business-person.png` (used by two tests when present; skipped otherwise).
