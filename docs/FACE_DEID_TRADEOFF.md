# Face de-identification: landmarks before and after

For each method: how much of the face the landmarker can still track in the de-identified video, how much face
motion that keeps, how steady the tracking is, and how much identity remains. Licences:
[FACE_SWAP_LICENCES.md](FACE_SWAP_LICENCES.md). Interpreting the motion is for clinical experts.

**Where the landmarks come from.** On the shareable `fill` path, all 478 landmarks and 52 blendshapes are
taken from the raw video before the face is filled and exported next to the filled clip (`_landmarks.nc`,
`_blendshapes.nc`; [SEGMENT_STORAGE.md](SEGMENT_STORAGE.md)). The filled video itself has no trackable face.
Warp and the face swaps keep a trackable face in the video, but only part of the original motion. Saved
landmarks can still carry some identity ([FACE_LANDMARKS.md](FACE_LANDMARKS.md#privacy)).

## Methods

`fill`: grown landmark hull painted one flat colour, plus YuNet boxes (the shareable default). `mosaic8` /
`mosaic16`: about 8 / 16 blocks across the face. `gauss`: Gaussian blur, sigma = face width / 6. `warp`:
landmark-driven warp of a synthetic donor. `ghost`: GHOST face swap. `liveportrait`: donor animated by the
subject's motion. GHOST and LivePortrait are research use only, local.

## Results

Each cell: GRID / Intel. Every face found has the full 478 landmarks. Motion and flicker need over 50 frames
with a face (otherwise the frame count or "-").

| Method | Donor | Face found after | Landmark r, blendshape r | Lost or regained per 100 frames | Flicker (% face width) | Identity kept: SFace, ArcFace | Label |
|---|---|---|---|---|---|---|---|
| original | - | 100 % / 73 % | reference | 0 / 0.71 | 0.15 / 0.22 | - | - |
| fill | - | 0 % / 0.2 % | none in video; 478 + 52 saved from the raw video | 0 / 0.71 | - | no face / 79 %, 79 % (19 pairs, before the fill check) | shareable fallback (default) |
| mosaic, 8 blocks | - | 11 % / 13 % | (38 frames) / 0.16, 0.01 | 11.7 / 4.9 | - / 0.97 | 0 %, 0 % / 5.4 %, 2.0 % | reduced risk, not de-identified |
| mosaic, 16 blocks | - | 100 % / 68 % | 0.25, 0.32 / 0.25, 0.09 | 0 / 1.00 | 0.43 / 0.49 | 99 %, 100 % / 89 %, 92 % | not de-identifying |
| Gaussian blur | - | 1.3 % / 0.5 % | (4 frames) / (16 frames) | 2.7 / 0.78 | - | no face / 12 %, 11 % | local review only |
| warp | mesh | 100 % / 73 % | 0.78, 0.68 / 0.50, 0.29 | 0 / 0.57 | 0.18 / 0.25 | 8.7 %, 0 % / 5.7 %, 0.8 % | reduced risk, not de-identified |
| warp | SFHQ | 100 % / 73 % | 0.69, 0.54 / 0.47, 0.27 | 0 / 0.57 | 0.16 / 0.21 | 100 %, 41 % / 26 %, 1.0 % | reduced risk, not de-identified |
| GHOST | mesh | 100 % / 73 % | 0.78, 0.68 / 0.55, 0.41 | 0 / 0.57 | 0.16 / 0.23 | 43 %, 0 % / 31 %, 1.0 % | reduced risk, not de-identified; research use only, local |
| GHOST | SFHQ | 100 % / 73 % | 0.82, 0.74 / 0.53, 0.44 | 0 / 0.57 | 0.17 / 0.19 | 64 %, 4.0 % / 17 %, 1.2 % | reduced risk, not de-identified; research use only, local |
| LivePortrait | mesh | 100 % / 71 % | 0.14, 0.02 / 0.18, -0.01 | 0 / 1.07 | 0.25 / 0.47 | 1.7 %, 0 % / 2.2 %, 0.5 % | reduced risk, not de-identified; research use only, local |
| LivePortrait | SFHQ | 100 % / 68 % | 0.60, 0.39 / 0.32, 0.20 | 0 / 1.78 | 0.17 / 0.46 | 39 %, 0 % / 2.3 %, 0.4 % | reduced risk, not de-identified; research use only, local |

In short: fill keeps no face in the video and all landmarks in the export. Warp and GHOST keep a face wherever
the original has one, track it about as steadily as the original and keep most of the region motion on GRID
and about half on the Intel clips. LivePortrait keeps the face but less of the motion, with more flicker and
more dropouts on the Intel clips. Mosaic either hides the face from the landmarker (8 blocks) or keeps the
identity (16 blocks); blur hides it.

Landmark r by face region (subject's left / right; GRID / Intel):

| Method | Donor | Mouth | Eyes L / R | Face side L / R |
|---|---|---|---|---|
| mosaic, 16 blocks | - | 0.50 / 0.24 | 0.13, 0.12 / 0.18, 0.19 | 0.23, 0.26 / 0.30, 0.33 |
| warp | mesh | 0.86 / 0.43 | 0.76, 0.78 / 0.49, 0.53 | 0.75, 0.76 / 0.54, 0.50 |
| warp | SFHQ | 0.74 / 0.46 | 0.57, 0.65 / 0.41, 0.49 | 0.74, 0.77 / 0.50, 0.50 |
| GHOST | mesh | 0.86 / 0.51 | 0.73, 0.74 / 0.51, 0.54 | 0.80, 0.79 / 0.60, 0.61 |
| GHOST | SFHQ | 0.87 / 0.40 | 0.81, 0.75 / 0.52, 0.59 | 0.84, 0.81 / 0.55, 0.56 |
| LivePortrait | mesh | 0.21 / 0.19 | 0.11, 0.05 / 0.17, 0.16 | 0.17, 0.17 / 0.21, 0.18 |
| LivePortrait | SFHQ | 0.68 / 0.24 | 0.50, 0.43 / 0.35, 0.38 | 0.71, 0.67 / 0.31, 0.31 |

**Fill check.** Fill also paints YuNet faces the landmarker missed (box padded 20 %), then YuNet checks every
output frame and the written file (score >= 0.6); flagged frames are filled again with wider boxes, and a face
still found in the final file blocks sharing (sidecar: `UnmaskedFaceCheck`, `EncodedOutputCheck`,
`EncodedRefill`; details in [SEGMENT_STORAGE.md](SEGMENT_STORAGE.md)). Without the YuNet model the export
can't be marked shareable (`python scripts/fetch_models.py --face-check`). The Intel fill row is before this
check: with it, no face was found in any of the 6 clips.

**Mosaic per clip.** A clip is an identity match with at least 20 face pairs and at least 50 % at or above
SFace 0.363 or ArcFace 0.30. `mosaic16` matched on 6 of 6 clips, `mosaic8` on none (one GRID clip had too few
faces). Blur left no face on the GRID clips, too few on one Intel clip and no match on the other.

## How measured

- Clips: GRID s14 (4 clips, 300 frames), Intel sample videos (2 clips, 2810 frames). Donors: mesh = procedural
  mesh render; SFHQ = one synthetic photo (SFHQ part 3, image 00001482; no SFHQ images are in this repository).
- Face found = share of frames where the MediaPipe Face Landmarker finds a face in the output.
- Landmark r = mean Pearson r of five face-region speeds (head motion removed, regions as in
  [FACE_LANDMARKS.md](FACE_LANDMARKS.md)); blendshape r = median r over 51 blendshapes; original vs output,
  frames with a face in both.
- Tracking stability (`scripts/face_deid_stability.py`, frame by frame, no smoothing): lost or regained =
  face appearing or disappearing between frames; longest gap without a face on the Intel clips: original 192
  frames, warp and GHOST 198, LivePortrait 200-202, mosaic16 202. Flicker = median distance of each landmark
  from a centred 5-frame average after removing head motion, in % of face width.
- Identity kept = share of face pairs (original vs output, same place) at or above SFace 0.363 / ArcFace
  `glintr100` 0.30.
- Render time per frame, box CPU (8 vCPU Xeon VM), landmarks excluded: fill 1.1 ms (6-15 ms with the fill
  check), mosaic 1.1 ms, blur 8.1 ms, warp 11 ms, GHOST 373-384 ms, LivePortrait 1887-1951 ms.

## Figure 1: fill and blur on a GRID frame

![Figure 1: GRID lwwy7p frames 20 and 45, left fill, right Gaussian blur](figures/face_deid_grid_lwwy7p_fill_blur.png)

Top row frame 20, bottom row frame 45 of GRID clip `lwwy7p` (speaker s14); W = face box width (108 px).
Left: **fill** (shareable default). Right: **Gaussian blur** (sigma W / 6; local review only).

Credit: frames from the GRID audio-visual speech corpus, M. Cooke, J. Barker, S. Cunningham and X. Shao
(2006), <https://doi.org/10.5281/zenodo.3625687>, CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>).
Modified (face filled or blurred; panels arranged and captioned). The figure is CC BY 4.0, not MIT.
