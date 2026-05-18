# UNO Vision Challenge — IAPR 2026

A computer-vision pipeline that classifies the full game state from a top-down photograph of an UNO table: card colours, card values, player hands, and whose turn it is.

---

## Task

Given a single RGB image of a 4-player UNO table, predict:

| Field | Description |
|---|---|
| `center_card` | The face-up card on the discard pile (e.g. `r_5`, `b_skip`) |
| `active_player` | Which player holds the turn token (`p1`–`p4`) |
| `player_N_cards` | Semicolon-separated list of cards in each player's hand |

---

## Pipeline Overview

```
Image
  │
  ├─ 1. Segmentation          HSV thresholding → per-colour region labels
  │                           Adaptive black thresholds (noisy vs. clean background)
  │
  ├─ 2. Region Filtering      Keep regions with ≥1 right-angle corner OR high circularity
  │                           (cards vs. turn token vs. background noise)
  │
  ├─ 3. Token Detection       Noisy BG → most circular yellow region
  │                           Clean BG → closing+opening on black regions → largest blob
  │
  ├─ 4. Location Assignment   3×3 grid → top / bottom / left / right / center
  │
  ├─ 5. Corner Patch Extraction   Rotate-align each card region; crop a PATCH_SIZE² patch
  │                               from the card corner containing the number/symbol
  │
  └─ 6. Template Matching     Chamfer-distance comparison against reference patches
                              (4 rotations × all translations); lowest distance wins
```

---

## Repository Structure

```
.
├── project_notebook.ipynb   # End-to-end walkthrough with visualizations
├── utils.py                 # All core functions (segmentation → classification)
├── template_patches/        # Reference corner patches (one PNG per card type)
├── train_images/            # Training photographs
├── train_masks/             # Ground-truth masks (not loaded into images_train)
├── test_images/             # Test photographs for submission
├── reference_images/        # Clean reference cards used to create templates
└── submission.csv           # Output: one row per test image
```

---

## Key Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `MIN_REGION_SIZE` | 8 000 px | Minimum connected-component area to keep after segmentation |
| `MERGE_DISTANCE` | 80 px | Max gap between bounding boxes before merging them |
| `MIN_CARD_AREA` | 150 000 px² | Minimum merged-box area to count as a card location |
| `PATCH_SIZE` | 90 px | Side length of the extracted corner patch |
| `CORNER_MARGIN` | 5 px | Inset from patch edge to card corner (≈ white border width) |
| `EPSILON_FACTOR` | 0.05 | Douglas-Peucker tolerance for polygon approximation |
| `SHIFT_STEP` | 1 px | Translation grid step during template matching |
| `SYMBOL_MIN_V` | 160 | Minimum HSV value to classify a pixel as white (symbol) |
| `SYMBOL_MAX_S` | 80 | Maximum HSV saturation for white symbol pixels |

---

## Running the Pipeline

### Interactive exploration (notebook)

Open `project_notebook.ipynb` and run cells sequentially.  
Set `TEST_REFERENCES = True` to visualize segmentation on reference images.  
Set `CREATE_TEMPLATE_PATCHES = True` to launch the interactive patch selector.

### Generating `submission.csv`

In the **Test set classification** cell, set:

```python
RUN_CLASSIFICATION = True
```

Then run the cell. Results are saved to `submission.csv`.

---

## Output Format

`submission.csv` columns:

```
image_id, center_card, active_player, player_1_cards, player_2_cards, player_3_cards, player_4_cards
```

- Card labels follow the pattern `{color}_{value}`, e.g. `r_5`, `g_skip`, `b_draw_2`.
- Wild cards: `wild`, `draw_4` (no colour prefix).
- Multiple cards per player are joined with `;`.
- `EMPTY` when a field cannot be determined.

Player positions correspond to image quadrants:

```
           player_3 (top)
player_4 (left)  ·  player_2 (right)
           player_1 (bottom)
```

---

## Dependencies

```
numpy
opencv-python
scikit-image
scipy
matplotlib
Pillow
ipywidgets
pandas
```

Install with:

```bash
pip install numpy opencv-python scikit-image scipy matplotlib Pillow ipywidgets pandas
```
