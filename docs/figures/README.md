# Figures

## `face_deid_grid_lwwy7p_fill_mosaic_blur.png`

Original, `fill`, `mosaic8`, `mosaic16` and Gaussian blur on two frames of one GRID clip (no donor, no
swap). Used in [FACE_DEID_TRADEOFF.md](../FACE_DEID_TRADEOFF.md).

Panels, left to right; top row frame 20, bottom row frame 45 of GRID clip `lwwy7p` (speaker s14).
W = the face box width (W = 108 px on frame 20).

1. **original**: the unmodified GRID frame.
2. **fill**: the grown face hull filled with a flat colour; the shareable fallback and the default.
3. **mosaic, 8 blocks** (`mosaic8`): square blocks of round(W / 8) px, about 8 blocks across the face;
   **reduced risk, not de-identified**.
4. **mosaic, 16 blocks** (`mosaic16`): square blocks of round(W / 16) px, about 16 blocks across the face;
   **not de-identifying** (identity matched on 6 of 6 clips, see "Mosaic and blur per clip"). The caption
   line inside the image reads "mosaic: reduced risk, not de-identified" for both mosaic panels; for
   `mosaic16` the label in this legend applies.
5. **Gaussian blur** (`gauss`): sigma = W / 6 px; **local review only**.

Credit: frames from the GRID audio-visual speech corpus, M. Cooke, J. Barker, S. Cunningham and X. Shao
(2006), speaker s14, clip `lwwy7p`, <https://doi.org/10.5281/zenodo.3625687>, licensed CC BY 4.0
(<https://creativecommons.org/licenses/by/4.0/>). This image is a modified derivative of those frames (face
filled, pixelated or blurred in panels 2 to 5; panels arranged and captioned) and is licensed CC BY 4.0, not
under this repository's MIT licence. No synthetic (SFHQ) donor face or face swap appears in it.

Licence: CC BY 4.0 (derivative of GRID corpus frames), not MIT. See [NOTICE](../../NOTICE) and row 18a of
[FACE_SWAP_LICENCES.md](../FACE_SWAP_LICENCES.md).
