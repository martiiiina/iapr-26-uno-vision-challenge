# UNO Vision Challenge — IAPR 2026

Project for the EE-451 Image Analysis and Pattern Recognition course at EPFL (Spring 2026).

Given overhead photos of a 4-player UNO game, the system identifies the center discard card, the active player, and each player's hand.

---

## Task

Each image shows a UNO table from above with up to 5 card regions:

| Region | Location (3×3 grid) |
|--------|---------------------|
| Center pile | center |
| Player 1 | bottom |
| Player 2 | right |
| Player 3 | top |
| Player 4 | left |

A **turn token** (colored disk) indicates whose turn it is. The output CSV lists `center_card`, `active_player`, and `player_N_cards` for each image.

---

## Pipeline

```
Image
  │
  ▼
1. HSV Segmentation          — label pixels as red / yellow / green / blue / black
  │
  ▼
2. Region Filtering           — keep only regions with a 90° corner (cards)
   (corner detection)           or high circularity (turn token); discard noise
  │
  ▼
3. Turn Token Detection       — separate the token from card regions;
                                determine active player side (top/bottom/left/right);
                                zero out any card region overlapping the token box
  │
  ▼
4. Card Region Filtering      — direction-aware merge of nearby blobs → bounding boxes;
   (merge + size + dedup)       discard boxes below MIN_CARD_AREA;
                                remove one of the two symmetric halves each UNO card
                                produces (split by the white oval)
  │
  ▼
5. Corner Patch Extraction    — find the best 90° corner on each card region;
                                rotate + crop a fixed-size patch
  │
  ▼
6. Number Classification      — template-match each patch against reference patches
   (template matching)          with 4×90° rotation invariance
```

### Step 1 — HSV Segmentation

Pixels are labeled in HSV space using fixed hue bands (OpenCV H range 0–179):

| Color | Hue range |
|-------|-----------|
| Red | 178–4 (wraps around 0) |
| Yellow | 23–27 |
| Green | 47–70 |
| Blue | 95–105 |
| Black | V < threshold, S < threshold |

Background type detection distinguishes **noisy** (yellow-green textured) from **clean** backgrounds and adjusts black thresholds accordingly to avoid false positives.

### Step 2 — Region Filtering

Connected components smaller than `MIN_REGION_SIZE` are dropped. Surviving regions are tested for card-like shape: morphological closing + hole removal → Douglas-Peucker polygon → check for at least one ~90° vertex (`CORNER_TOL`). Circular regions (circularity ≥ 0.8) are kept as turn-token candidates.

### Step 3 — Turn Token Detection

- **Noisy background**: the yellow region with the highest circularity score is the token.
- **Clean background**: morphological closing then opening on all black regions isolates the compact token disk (largest surviving component).

The token centroid is mapped to the nearest table edge (top/bottom/left/right) to identify the active player. After detection, any card region whose bounding box overlaps the token bounding box is zeroed out to prevent double-counting.

### Step 4 — Card Region Filtering

Three sub-steps clean up the segmented card regions:

1. **Direction-aware merge** (`merge_nearby_boxes_oriented`, distance = 60 px): bounding boxes are merged if their gap is within the threshold, using only the horizontal gap for left/right players and only the vertical gap for top/bottom players. This ensures all visible portions of a card in the same row/column merge into one box even when inter-card spacing is small.

2. **Size filter** (`MIN_CARD_AREA`): merged boxes smaller than the threshold are discarded as noise.

3. **Symmetric region deduplication**: UNO cards are segmented into two colored halves split by the central white oval. For each color region, if another region of the same color (or any color for the center cell) in the same grid cell has a bounding-box center within `PROXIMITY_THRESH` pixels in the relevant axis, only the larger region (by pixel count) is kept. The relevant axis is:
   - **X proximity** for Players 1 & 3 (top/bottom) — the two halves share the same horizontal position
   - **Y proximity** for Players 2 & 4 (left/right) — the two halves share the same vertical position
   - **Any axis** for the center cell — only the single largest region is kept
   - **Black (wild) cards** skip the same-color constraint and remove all smaller same-cell regions

### Step 5 — Corner Patch Extraction

For each card region the best 90° polygon vertex (longest combined adjacent edges, within `CORNER_TOL` degrees of 90°) is found. The image is rotated so that the card edge is horizontal, then a `PATCH_SIZE × PATCH_SIZE` crop is extracted from the corner, inset by `CORNER_MARGIN` pixels so the card border is flush with the patch boundary.

The card's centroid (pixel mean) is used for grid-cell location assignment rather than the corner position, which can point toward the table centre.

### Step 6 — Number Classification (Template Matching)

Reference patches are created manually from `reference_images/` using an interactive widget (`CREATE_TEMPLATE_PATCHES = True`). At inference time each extracted patch is matched against all templates using `cv2.matchTemplate` (TM_CCOEFF_NORMED), trying all four 90° rotations. The label with the highest score is assigned.

---

## Repository Structure

```
├── project_notebook.ipynb   # Main development notebook (full pipeline)
├── segmentation.py          # Standalone segmentation module (block-based)
├── main.py                  # End-to-end inference script (kNN baseline)
├── utils.py                 # Shared preprocessing helpers
├── explore_hsv.py           # HSV colour exploration tools
│
├── template_patches/        # Saved reference patches for template matching
├── reference_images/        # Reference card images — 4 figures (gitignored)
├── train_images/            # Training images (gitignored)
├── test_images/             # Test images (gitignored)
│
├── train.csv                # Ground-truth labels for training images
├── sample_submission.csv    # Expected output format
└── labs/                    # IAPR lab utilities
```

---

## Key Hyperparameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `MIN_SATURATION` | 90 | Discard low-saturation (background) pixels |
| `MIN_VALUE` | 40 | Discard very dark pixels |
| `MIN_REGION_SIZE` | 8 000 px | Drop small noise regions after segmentation |
| `MIN_CARD_AREA` | 150 000 px² | Minimum merged-box area to count as a card |
| `CLOSING_DISK` | 3 | Morphological closing disk for corner extraction |
| `EPSILON_FACTOR` | 0.05 | Douglas-Peucker tolerance for polygon simplification |
| `PATCH_SIZE` | 90 px | Side length of extracted corner patch |
| `CORNER_TOL` | 15° | Max deviation from 90° to accept a polygon vertex as a corner |
| `CORNER_MARGIN` | 5 px | Inset so card border is flush with patch edge |
| `PROXIMITY_THRESH` | 300 px | Max bbox-center distance to treat two regions as the same card |

---

## Output Format

```csv
image_id,center_card,active_player,player_1_cards,player_2_cards,player_3_cards,player_4_cards
```

- `center_card`: e.g. `r_5`, `b_skip`, `wild`
- `active_player`: `p1` / `p2` / `p3` / `p4`
- `player_N_cards`: semicolon-separated list of card labels, e.g. `r_3;g_7;b_skip`

Card labels follow the format `{color}_{value}` where color ∈ {r, y, g, b} and value ∈ {0–9, skip, reverse, draw_2, wild, draw_4}.