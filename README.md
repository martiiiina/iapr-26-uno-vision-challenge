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
                                determine active player side (top/bottom/left/right)
  │
  ▼
4. Card Location Assignment   — merge nearby color blobs → bounding boxes;
   (3×3 grid)                   map each box to a table position
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

Connected components smaller than `MIN_REGION_SIZE` are dropped. Surviving regions are tested for card-like shape: morphological closing + hole removal → Douglas-Peucker polygon → check for at least one ~90° vertex. Circular regions (circularity ≥ 0.8) are kept as turn-token candidates.

### Step 3 — Turn Token Detection

- **Noisy background**: the yellow region with the highest circularity score is the token.
- **Clean background**: morphological closing then opening on all black regions isolates the compact token disk (largest surviving component).

The token centroid is mapped to the nearest table edge (top/bottom/left/right) to identify the active player.

### Step 4 — Card Location Assignment

Color blobs belonging to the same physical card are merged by iteratively unioning bounding boxes within `MERGE_DISTANCE` pixels. Merged boxes below `MIN_CARD_AREA` are discarded. Each surviving box is assigned a location (`top`, `bottom`, `left`, `right`, `center`) via a 3×3 grid.

### Step 5 — Corner Patch Extraction

For each card region the best 90° polygon vertex (longest combined adjacent edges) is found. The image is rotated so that the card edge is horizontal, then a `PATCH_SIZE × PATCH_SIZE` crop is extracted from the corner, inset by `CORNER_MARGIN` pixels so the card border is flush with the patch boundary.

### Step 6 — Number Classification (Template Matching)

Reference patches are created manually from `reference_images/` using an interactive widget (`CREATE_TEMPLATE_PATCHES = True`). At inference time each extracted patch is matched against all templates using `cv2.matchTemplate` (TM_CCOEFF_NORMED), trying all four 90° rotations. The label with the highest score above `MATCH_THRESHOLD` is assigned; otherwise the card is marked `"?"`.

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

| Parameter | Default | Effect |
|-----------|---------|--------|
| `MIN_SATURATION` | 90 | Discard low-saturation (background) pixels |
| `MIN_VALUE` | 40 | Discard very dark pixels |
| `MIN_REGION_SIZE` | 8000 px | Drop small noise regions after segmentation |
| `MERGE_DISTANCE` | 30 px | Max gap between boxes to merge into one card |
| `MIN_CARD_AREA` | 250 000 px² | Minimum bounding-box area to count as a card |
| `PATCH_SIZE` | 120 px | Side length of extracted corner patch |
| `CORNER_MARGIN` | 20 px | Inset so card border is flush with patch edge |
| `EPSILON_FACTOR` | 0.05 | Douglas-Peucker tolerance for polygon simplification |
| `MATCH_THRESHOLD` | 0.5 | Minimum template-match score to assign a label |

---

## Output Format

```csv
image_id,center_card,active_player,player_1_cards,player_2_cards,player_3_cards,player_4_cards
```

- `center_card`: e.g. `r_5`, `b_skip`, `wild`
- `active_player`: `p1` / `p2` / `p3` / `p4`
- `player_N_cards`: space-separated list of card labels, e.g. `r_3 g_7 b_skip`

Card labels follow the format `{color}_{value}` where color ∈ {r, y, g, b} and value ∈ {0–9, skip, reverse, draw_2, wild, draw_4}.

---

## Status

| Stage | Status |
|-------|--------|
| Data loading | Done |
| HSV segmentation | Done |
| Corner-based region filtering | Done |
| Turn token detection | Done |
| Card location assignment (3×3 grid) | Done |
| Corner patch extraction | Done |
| Template matching classifier | Ready to run |
| End-to-end inference pipeline | In progress |
