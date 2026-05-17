import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import disk, closing, opening, remove_small_holes, remove_small_objects
import io
import math
import ipywidgets as widgets
from IPython.display import display
from PIL import Image, ImageDraw
import skimage.measure

### SEGMENT AND VISUALIZE

def _is_noisy_background(img_rgb, params):
    """Return True if the image contains a significant yellow-green noisy background."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S = hsv[:, :, 0], hsv[:, :, 1]
    bg_mask = (H >= params["BG_HUE_LO"]) & (H <= params["BG_HUE_HI"]) & (S >= params["BG_MIN_SAT"])
    ratio = bg_mask.sum() / (img_rgb.shape[0] * img_rgb.shape[1])
    return ratio > params["BG_NOISY_RATIO"]


def apply_color_mask(img_rgb, black_max_v, black_max_s, params):
    """This function takes an RGB image and creates a pixel-wise color label map based on HSV color ranges. 
    It identifies known colors (from COLOR_RANGES) and separately detects black regions."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    labels = np.zeros(img_rgb.shape[:2], dtype=np.int32)

    chromatic_mask = (S > params["MIN_SATURATION"]) & (V > params["MIN_VALUE"])
    known_color_hue = np.zeros(img_rgb.shape[:2], dtype=bool)
    for i, (name, (lo, hi)) in enumerate(params["COLOR_RANGES"].items(), start=1):
        hue_mask = (H >= lo) | (H <= hi) if lo > hi else (H >= lo) & (H <= hi)
        labels[hue_mask & chromatic_mask] = i
        known_color_hue |= hue_mask          # track all known-color hues

    # Only label black if dark AND not a known color hue
    black_mask = (V < black_max_v) & (S < black_max_s) & ~known_color_hue
    labels[black_mask] = 5
    return labels


def segment(img_rgb, min_region_size, params):
    """
    Threshold -> connected components -> drop regions < min_region_size.
    Returns label map and list of (x, y, w, h, color_name).
    Automatically selects tighter black thresholds when a noisy background is detected.
    """
    if _is_noisy_background(img_rgb, params=params):
        bv, bs = params["BLACK_MAX_V_NOISY"], params["BLACK_MAX_S_NOISY"]
    else:
        bv, bs = params["BLACK_MAX_V_CLEAN"], params["BLACK_MAX_S_CLEAN"]

    raw   = apply_color_mask(img_rgb, bv, bs, params=params)
    H, W  = img_rgb.shape[:2]
    final = np.zeros((H, W), dtype=np.int32)
    boxes = []

    for clbl in range(1, len(params["COLOR_NAMES"]) + 1):
        cc, n = ndimage.label(raw == clbl)  # regions and number of regions
        sizes = np.bincount(cc.ravel())     # dimension of detected region
        for i in range(1, n + 1):
            if sizes[i] < min_region_size:
                continue
            region = cc == i                # i-th region that survived dimension thresholding
            final[region] = clbl            # final is the image with color labels according to the dict
            ys, xs = np.where(region)
            boxes.append((int(xs.min()), int(ys.min()),
                          int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1),
                          params["COLOR_NAMES"][clbl - 1]))
    return final, boxes


def visualize(img_rgb, labels, params, title=""):
    """Visualize image, result of segmentation and hue histogram"""
    pixel_map = np.zeros((*img_rgb.shape[:2], 3), dtype=np.uint8)
    for clbl in range(1, len(params["COLOR_NAMES"]) + 1):
        pixel_map[labels == clbl] = params["OVERLAY_RGB"][clbl-1].astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    axes[0].imshow(img_rgb);    axes[0].set_title("Original");             axes[0].axis("off")
    axes[1].imshow(pixel_map);  axes[1].set_title("Pixel mask");           axes[1].axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()

    # Hue histogram (chromatic pixels only) 
    noisy = _is_noisy_background(img_rgb, params=params)
    bv = params["BLACK_MAX_V_NOISY"] if noisy else params["BLACK_MAX_V_CLEAN"]
    bs = params["BLACK_MAX_S_NOISY"] if noisy else params["BLACK_MAX_S_CLEAN"]

    hsv   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hmask = (hsv[:, :, 1] > params["MIN_SATURATION"]) & (hsv[:, :, 2] > params["MIN_VALUE"]) & (hsv[:, :, 2] < 240)
    hues  = hsv[:, :, 0][hmask].flatten()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(hues, bins=200, range=(0, 180), color="steelblue", edgecolor="none", alpha=0.7)
    for name, (lo, hi) in params["COLOR_RANGES"].items():
        c = params["_MPL_COLORS"][name]
        if lo > hi:
            ax.axvspan(lo, 180, alpha=0.25, color=c)
            ax.axvspan(0,  hi,  alpha=0.25, color=c, label=f"{name} [{lo}-{hi}]")
        else:
            ax.axvspan(lo, hi, alpha=0.25, color=c, label=f"{name} [{lo}-{hi}]")
    ax.axvspan(params["BG_HUE_LO"], params["BG_HUE_HI"], alpha=0.15, color="yellowgreen",
               label=f"BG band [{params['BG_HUE_LO']}-{params['BG_HUE_HI']}]")
    ax.set_xlabel("Hue (OpenCV 0-179)")
    ax.set_ylabel("Pixel count")
    ax.set_xlim(0, 179)
    ax.set_title(f"Hue histogram — {'NOISY' if noisy else 'CLEAN'} background "
                 f"→ black thresholds: V<{bv}, S<{bs}")
    ax.legend(fontsize=8, ncol=5)
    plt.tight_layout()
    plt.show()

### FIRST FILTERING 90°

def apply_closing(img_th, disk_size):
    return closing(img_th, disk(disk_size))

def apply_opening(img_th, disk_size):
    return opening(img_th, disk(disk_size))

def remove_holes(img_th, size):
    return remove_small_holes(img_th.astype(bool), area_threshold=size)


def _crop_region(region_mask, pad):
    """Crop the full-image mask to the region's bounding box + padding."""
    ys, xs = np.where(region_mask)
    H, W = region_mask.shape
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(H, int(ys.max()) + pad + 1)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(W, int(xs.max()) + pad + 1)
    return region_mask[y1:y2, x1:x2]


def _polygon_from_mask(crop, disk_size, epsilon_factor, hole_size):
    """Fill holes, apply closing, find contour, return simplified polygon.
    Epsilon defines the detail of polygonisation.
    """
    mask_no_holes = remove_holes(crop, hole_size)
    mask_closed   = apply_closing(mask_no_holes, disk_size)
    mask_u8 = mask_closed.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask_closed
    contour = max(contours, key=cv2.contourArea)
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True)
    return polygon, mask_closed


def _angles_at_vertices(polygon):
    """Returns the interior angle (degrees) at every vertex of the polygon."""
    pts = polygon[:, 0, :].astype(float)
    n = len(pts)
    angles = []
    for i in range(n):
        A = pts[(i - 1) % n];  B = pts[i];  C = pts[(i + 1) % n]
        BA = A - B;  BC = C - B
        nBA = np.linalg.norm(BA);  nBC = np.linalg.norm(BC)
        if nBA == 0 or nBC == 0:
            angles.append(None)
            continue
        cos_a = np.clip(np.dot(BA, BC) / (nBA * nBC), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_a)))
    return angles


def _circularity(mask_closed):
    """
    Compute circularity = 4π·area / perimeter² from the closed mask.
    Returns 1.0 for a perfect circle, lower values for elongated/irregular shapes.
    """
    mask_u8 = mask_closed.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour   = max(contours, key=cv2.contourArea)
    area      = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)


def has_right_angle_corner(region_mask, disk_size=5, epsilon_factor=0.02,
                            angle_tolerance=10, hole_size=500, circularity_threshold=0.8):
    """
    Returns True if the region looks like a card OR a circular turn token.

    Keep condition:
      - has at least one ~90° corner (card), OR
      - circularity score >= circularity_threshold (turn token)

    Parameters
    ----------
    angle_tolerance       : accepted deviation from 90° in degrees
    circularity_threshold : min score to be considered circular (0-1, 1 = perfect circle)
    """
    crop = _crop_region(region_mask, pad=disk_size + 2)
    polygon, mask_closed = _polygon_from_mask(crop, disk_size, epsilon_factor, hole_size)

    # Check 1: card corner
    if polygon is not None and len(polygon) >= 3:
        for angle in _angles_at_vertices(polygon):
            if angle is not None and abs(angle - 90) <= angle_tolerance:
                return True

    # Check 2: circular turn token
    if _circularity(mask_closed) >= circularity_threshold:
        return True

    return False


def visualize_corner_detection(region_mask, disk_size=5, epsilon_factor=0.02,
                                angle_tolerance=10, hole_size=500,
                                circularity_threshold=0.8, title=""):
    """
    Show the steps of corner detection for one region:
      1. Original binary mask (cropped)
      2. Mask after hole removal + closing 
      3. Simplified polygon — green dot = ~90° corner, red dot = other
    """
    crop = _crop_region(region_mask, pad=disk_size + 2)
    polygon, mask_closed = _polygon_from_mask(crop, disk_size, epsilon_factor, hole_size)
    circ = _circularity(mask_closed)
    is_circular = circ >= circularity_threshold

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(crop, cmap="gray")
    axes[0].set_title("Original mask"); axes[0].axis("off")

    circ_color = "lime" if is_circular else "white"
    axes[1].imshow(mask_closed, cmap="gray")
    axes[1].set_title(f"After holes + closing\ncircularity = {circ:.2f}  "
                      f"(threshold = {circularity_threshold})", color=circ_color)
    axes[1].axis("off")

    axes[2].imshow(mask_closed, cmap="gray")
    axes[2].set_title("Simplified polygon + corners"); axes[2].axis("off")

    if polygon is not None and len(polygon) >= 3:
        pts    = polygon[:, 0, :].astype(float)
        n      = len(pts)
        angles = _angles_at_vertices(polygon)

        for i in range(n):
            p1, p2 = pts[i], pts[(i + 1) % n]
            axes[2].plot([p1[0], p2[0]], [p1[1], p2[1]], "b-", linewidth=1.5)

        for i, angle in enumerate(angles):
            if angle is None:
                continue
            B     = pts[i]
            color = "lime" if abs(angle - 90) <= angle_tolerance else "red"
            axes[2].plot(B[0], B[1], "o", color=color, markersize=8)
            axes[2].text(B[0] + 4, B[1] - 4, f"{angle:.0f}°", color=color, fontsize=7, fontweight="bold")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.show()


def filter_cards_by_corners(labels, params, disk_size=5, epsilon_factor=0.02,
                             angle_tolerance=10, hole_size=500, circularity_threshold=0.8):
    """
    Keep regions that have a ~90° corner (card) OR are approximately circular (turn token).
    Removes everything else (leaves, noise).
    """
    filtered_labels = labels.copy()
    filtered_boxes  = []

    for clbl in range(1, len(params["COLOR_NAMES"]) + 1):
        cc, n_regions = ndimage.label(labels == clbl)

        for region_id in range(1, n_regions + 1):
            region_mask = (cc == region_id)

            if has_right_angle_corner(region_mask, disk_size, epsilon_factor,
                                      angle_tolerance, hole_size, circularity_threshold):
                ys, xs = np.where(region_mask)
                filtered_boxes.append((
                    int(xs.min()), int(ys.min()),
                    int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1),
                    params["COLOR_NAMES"][clbl - 1],
                ))
            else:
                filtered_labels[region_mask] = 0

    return filtered_labels, filtered_boxes

### TURN TOKEN DETECTION

def _nearest_player(cx, cy, img_shape):
    """Return the nearest player side ('top'/'bottom'/'left'/'right') to point (cx, cy)."""
    H, W = img_shape[:2]
    centers = {
        "top":    (W / 2,     H / 6),
        "bottom": (W / 2,     H * 5 / 6),
        "left":   (W / 6,     H / 2),
        "right":  (W * 5 / 6, H / 2),
    }
    return min(centers, key=lambda s: (cx - centers[s][0])**2 + (cy - centers[s][1])**2)


def _region_circularity(region_mask, params):
    """Return circularity = 4π·area/perimeter² after a small closing to smooth the boundary."""
    closed  = apply_closing(region_mask, params["TOKEN_CLOSING_DISK"])
    mask_u8 = closed.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    a = cv2.contourArea(c)
    p = cv2.arcLength(c, True)
    return 4 * np.pi * a / p**2 if p > 0 else 0.0


def _token_result(region_mask, color_name, filtered_labels, filtered_boxes, img_shape, params):
    """Package a confirmed token region into the standard return tuple."""
    ys, xs = np.where(region_mask)
    cx, cy = float(xs.mean()), float(ys.mean())
    x1, y1 = int(xs.min()), int(ys.min())
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    token_box      = (x1, y1, w, h, color_name)
    token_location = _nearest_player(cx, cy, img_shape)
    card_labels    = filtered_labels.copy()
    card_labels[region_mask] = 0
    # Remove same-color fragments whose bounding box overlaps the token box
    # (handles cases where the token is fragmented into multiple connected components)
    clbl = params["COLOR_NAMES"].index(color_name) + 1
    cc_tok, n_tok = ndimage.label(card_labels == clbl)
    for rid in range(1, n_tok + 1):
        frag = cc_tok == rid
        ys_f, xs_f = np.where(frag)
        if (int(xs_f.min()) < x1 + w and int(xs_f.max()) > x1 and
                int(ys_f.min()) < y1 + h and int(ys_f.max()) > y1):
            card_labels[frag] = 0
    card_boxes = [b for b in filtered_boxes
                  if not (b[4] == color_name
                          and b[0] < x1 + w and b[0] + b[2] > x1
                          and b[1] < y1 + h and b[1] + b[3] > y1)]
    return token_box, token_location, card_labels, card_boxes


def _black_token_morph(filtered_labels, params):
    """
    Isolate the black token from all black regions using closing then opening.

    Closing merges nearby black fragments; opening then removes thin/small
    regions (card borders, numbers), leaving only compact solid objects.
    Returns the largest surviving connected component — the token.

    Also returns intermediate masks for visualization.
    """
    clbl       = params["COLOR_NAMES"].index("black") + 1
    black_mask = (filtered_labels == clbl)
    closed     = apply_closing(black_mask, params["TOKEN_MORPH_DISK"])
    opened     = apply_opening(closed,     params["TOKEN_MORPH_DISK"])

    cc, n_regions = ndimage.label(opened)
    if n_regions == 0:
        return None, black_mask, closed, opened

    sizes   = np.bincount(cc.ravel()); sizes[0] = 0
    best_id = int(sizes.argmax())
    return (cc == best_id), black_mask, closed, opened


def visualize_black_token_morph(img_rgb, black_mask, closed, opened, token_mask, params):
    """4-panel: raw black regions → after closing → after opening → selected token."""
    steps = [
        (black_mask, "Black regions (raw)"),
        (closed,     f"After closing  (disk={params['TOKEN_MORPH_DISK']})"),
        (opened,     f"After opening  (disk={params['TOKEN_MORPH_DISK']})"),
        (token_mask, "Selected token  (largest region)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (mask, title) in zip(axes, steps):
        ax.imshow(img_rgb)
        if mask is not None:
            overlay = np.zeros((*img_rgb.shape[:2], 4), dtype=np.uint8)
            overlay[mask] = [255, 220, 0, 160]      # Overlays a yellow mask if the token is found
            ax.imshow(overlay)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.suptitle("Black token isolation — closing + opening", fontsize=11)
    plt.tight_layout()
    plt.show()


def detect_token(filtered_labels, filtered_boxes, img_shape, params, noisy=False, visualize_morph=False):
    """
    Find the turn token among filtered regions and return it separately.

    Strategy depends on background type:
      - noisy background : yellow region with the highest circularity (round disk)
      - clean background : apply closing+opening to all black regions, pick the largest

    Returns
    -------
    token_box      : (x, y, w, h, color) or None
    token_location : 'top'|'bottom'|'left'|'right' or None
    card_labels    : filtered_labels with the token region zeroed out
    card_boxes     : filtered_boxes with the token entry removed
    """
    if noisy:
        best_circ, best_mask = -1.0, None
        clbl = params["COLOR_NAMES"].index("yellow") + 1
        cc, n_regions = ndimage.label(filtered_labels == clbl)
        for region_id in range(1, n_regions + 1):
            region_mask = (cc == region_id)
            circ = _region_circularity(region_mask, params)
            if circ > best_circ:
                best_circ, best_mask = circ, region_mask
        if best_mask is not None:
            closed_mask = apply_closing(best_mask, params["TOKEN_CLOSING_DISK"])
            return _token_result(closed_mask, "yellow", filtered_labels, filtered_boxes, img_shape, params)

    else:
        token_mask, black_mask, closed, opened = _black_token_morph(filtered_labels, params)
        if visualize_morph:
            visualize_black_token_morph(
                np.zeros((*filtered_labels.shape, 3), dtype=np.uint8),  # placeholder — pass img_rgb at call site
                black_mask, closed, opened, token_mask, params)
        if token_mask is not None:
            return _token_result(token_mask, "black", filtered_labels, filtered_boxes, img_shape, params)

    return None, None, filtered_labels, list(filtered_boxes)


def visualize_token(img_rgb, token_box, token_location, params):
    """Show the image with the 3x3 grid, token bounding box, and player arrow."""
    H, W = img_rgb.shape[:2]
    player_centers = {
        "top":    (W / 2,     H / 6),
        "bottom": (W / 2,     H * 5 / 6),
        "left":   (W / 6,     H / 2),
        "right":  (W * 5 / 6, H / 2),
    }

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img_rgb); ax.axis("off")

    for frac in [1/3, 2/3]:
        ax.axvline(W * frac, color="white", linewidth=1, linestyle="--", alpha=0.5)
        ax.axhline(H * frac, color="white", linewidth=1, linestyle="--", alpha=0.5)

    if token_box is not None:
        x, y, w, h, color_name = token_box
        ec = params["EDGE_COLORS"].get(color_name, "white")
        ax.add_patch(plt.Rectangle((x, y), w, h, lw=3, edgecolor=ec, facecolor="none"))
        ax.text(x + w / 2, y + h / 2, f"token\n({color_name})", color="white",
                fontsize=9, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6))

        if token_location in player_centers:
            tx, ty = x + w / 2, y + h / 2
            px, py = player_centers[token_location]
            ax.annotate("", xy=(px, py), xytext=(tx, ty),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2))
            ax.text(px, py, token_location, color="white", fontsize=11, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.4", fc="black", alpha=0.6))
    else:
        ax.set_title("No token detected")

    ax.set_title(f"Token detection — location: {token_location}")
    plt.tight_layout()
    plt.show()

### CARD DETECTION PIPELINE 1: REGION MERGING AND LOCATION ASSIGNMENT

def _box_gap(b1, b2):
    """Chebyshev gap (pixels) between two bounding boxes. 0 if they overlap."""
    x1, y1, w1, h1 = b1[:4]; x2, y2, w2, h2 = b2[:4]
    dx = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
    dy = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
    return max(dx, dy)


def _union_box(b1, b2):
    """Smallest (x, y, w, h) covering both boxes."""
    x1, y1, w1, h1 = b1[:4]; x2, y2, w2, h2 = b2[:4]
    x  = min(x1, x2); y  = min(y1, y2)
    x2r = max(x1 + w1, x2 + w2); y2r = max(y1 + h1, y2 + h2)
    return (x, y, x2r - x, y2r - y)


def merge_nearby_boxes(boxes, merge_distance):
    """
    Greedily merge bounding boxes within merge_distance pixels of each other.
    Repeats until stable. Returns list of (x, y, w, h).
    """
    rects = [b[:4] for b in boxes]
    changed = True
    while changed:
        changed = False
        merged = []; used = [False] * len(rects)
        for i in range(len(rects)):
            if used[i]:
                continue
            current = rects[i]
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                if _box_gap(current, rects[j]) <= merge_distance:
                    current = _union_box(current, rects[j])
                    used[j] = True
                    changed = True
            merged.append(current)
            used[i] = True
        rects = merged
    return rects

def _filter_labels_by_boxes(card_labels, valid_boxes, params, noisy=False):
    """For noisy images, zero out label regions whose centroid is closer to an
    image border than any valid-box centroid is to that same border.
    Clean images are returned unchanged."""
    if not noisy or not valid_boxes:
        return card_labels.copy()
    vcxs = [vx + vw / 2 for vx, vy, vw, vh in valid_boxes]
    vcys = [vy + vh / 2 for vx, vy, vw, vh in valid_boxes]
    min_cx, max_cx = min(vcxs), max(vcxs)
    min_cy, max_cy = min(vcys), max(vcys)
    result = card_labels.copy()
    for clbl in range(1, len(params["COLOR_NAMES"]) + 1):
        cc, n = ndimage.label(card_labels == clbl)
        for region_id in range(1, n + 1):
            mask = cc == region_id
            ys, xs = np.where(mask)
            rcx, rcy = xs.mean(), ys.mean()
            if rcx < min_cx or rcx > max_cx or rcy < min_cy or rcy > max_cy:
                result[mask] = 0
    return result

def filter_by_size(rects, min_area):
    """Keep only boxes whose area (w x h) is at least min_area."""
    return [r for r in rects if r[2] * r[3] >= min_area]


# 3x3 grid layout (row=vertical third, col=horizontal third):
#
#   col:  0        1        2
#   row 0 [corner | top    | corner]
#   row 1 [left   | center | right ]
#   row 2 [corner | bottom | corner]
#
# Corner cells → None (discarded)
_GRID_LABELS = {
    (0, 1): "top",
    (2, 1): "bottom",
    (1, 0): "left",
    (1, 2): "right",
    (1, 1): "center",
}

def assign_location(rect, img_shape):
    """
    Divide the image into a 3×3 grid and return the centroid's cell label.
    Returns 'top', 'bottom', 'left', 'right', 'center', or None for corner cells.
    """
    H, W = img_shape[:2]
    cx = rect[0] + rect[2] / 2
    cy = rect[1] + rect[3] / 2
    col = min(int(cx / W * 3), 2)   # 0=left third, 1=center third, 2=right third
    row = min(int(cy / H * 3), 2)   # 0=top third,  1=center third, 2=bottom third
    return _GRID_LABELS.get((row, col), None)


def visualize_merge_steps(img_rgb, filtered_boxes, merged_boxes, card_boxes):
    """3-panel: raw filtered regions → merged boxes → size-filtered card boxes."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    titles   = [f"Filtered regions ({len(filtered_boxes)})",
                f"After merge ({len(merged_boxes)})",
                f"Card locations ({len(card_boxes)})"]
    box_sets = [filtered_boxes, merged_boxes, card_boxes]
    for ax, boxes_to_draw, title in zip(axes, box_sets, titles):
        ax.imshow(img_rgb); ax.axis("off"); ax.set_title(title)
        for b in boxes_to_draw:
            x, y, w, h = b[:4]
            ax.add_patch(plt.Rectangle((x, y), w, h, lw=2, edgecolor="lime", facecolor="none"))
    plt.tight_layout(); plt.show()


def visualize_locations(img_rgb, card_boxes):
    """Image with rectangles and location label; dashed lines show the 3x3 grid."""
    H, W = img_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img_rgb); ax.axis("off")

    # Draw the 3x3 grid for reference
    for frac in [1/3, 2/3]:
        ax.axvline(W * frac, color="white", linewidth=1, linestyle="--", alpha=0.5)
        ax.axhline(H * frac, color="white", linewidth=1, linestyle="--", alpha=0.5)

    for b in card_boxes:
        x, y, w, h = b[:4]
        loc = assign_location(b, img_rgb.shape)
        ax.add_patch(plt.Rectangle((x, y), w, h, lw=2.5, edgecolor="white", facecolor="none"))
        ax.text(x + w / 2, y + h / 2, loc or "?", color="white", fontsize=11, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.55))
    ax.set_title(f"Card locations — {len(card_boxes)} cards")
    plt.tight_layout(); plt.show()

### CARD DETECTION PIPELINE 2: CORNER PATCH EXTRACTION

def _grid_cell(x, y, img_shape):
    """Returns location in the image (e.g. top, left ecc)"""
    H, W = img_shape[:2]
    col = min(int(x / W * 3), 2)
    row = min(int(y / H * 3), 2)
    return row, col


def _find_best_corner(polygon, angle_tol):
    """
    Return the index of the ~90° vertex with the longest combined adjacent edges.
    Keeps only one vertex per region.
    Returns None if no vertex qualifies.
    """
    pts = polygon[:, 0, :].astype(float)
    n = len(pts)
    best_idx, best_edge_sum = None, -1.0
    for i in range(n):
        A = pts[(i - 1) % n]; B = pts[i]; C = pts[(i + 1) % n]
        BA = A - B; BC = C - B
        nBA = np.linalg.norm(BA); nBC = np.linalg.norm(BC)
        if nBA == 0 or nBC == 0:
            continue
        cos_a = np.clip(np.dot(BA, BC) / (nBA * nBC), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_a))
        if abs(angle - 90) <= angle_tol:
            edge_sum = nBA + nBC
            if edge_sum > best_edge_sum:
                best_edge_sum, best_idx = edge_sum, i
    return best_idx


def _extract_patch(img_rgb, corner_full, edge_vec, other_vec, patch_size, corner_margin):
    """
    Rotate img_rgb so edge_vec points right (+x), then crop a patch_size x patch_size square
    whose corner coincides with the card corner, inset by corner_margin pixels.

    Quadrant logic (OpenCV rotation matrix for angle θ maps direction (ox,oy) as):
        new_x =  cos(θ)·ox + sin(θ)·oy
        new_y = -sin(θ)·ox + cos(θ)·oy          ← note the minus on sin for y

    After rotation, edge_vec → +x. The sign of other_vec's rotated y tells us
    whether the card interior is below (+y) or above (-y) the corner in the rotated frame:
        other_rot_y > 0  →  card below  →  corner at top-left   of patch
        other_rot_y < 0  →  card above  →  corner at bottom-left of patch

    corner_margin pulls the crop start back by that many pixels so the detected
    card-corner vertex sits slightly inside the patch (not right at the very edge pixel).
    """
    cx, cy = float(corner_full[0]), float(corner_full[1])
    angle_deg = np.degrees(np.arctan2(edge_vec[1], edge_vec[0]))
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    H, W = img_rgb.shape[:2]
    rotated = cv2.warpAffine(img_rgb, M, (W, H))

    # OpenCV rotation: new_y = -sin(θ)·ox + cos(θ)·oy
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    other_rot_y = -sin_a * other_vec[0] + cos_a * other_vec[1]

    # Card extends to the right (along rotated edge_vec) → corner at left edge, offset by margin.
    # Card extends down (other_rot_y ≥ 0) → corner near top;  up → corner near bottom.
    x1 = int(cx) - corner_margin
    if other_rot_y >= 0:
        y1 = int(cy) - corner_margin          # corner near top-left
    else:
        y1 = int(cy) - patch_size + corner_margin  # corner near bottom-left
    x2, y2 = x1 + patch_size, y1 + patch_size

    pad_top    = max(0, -y1);    pad_bottom = max(0, y2 - H)
    pad_left   = max(0, -x1);   pad_right  = max(0, x2 - W)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, W), min(y2, H)
    crop = rotated[y1:y2, x1:x2]
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return crop


def extract_corner_patches(img_rgb, labels, params,
                            closing_disk, epsilon_factor, patch_size,
                            proximity_thresh, angle_tol, corner_margin):
    results = []

    for clbl in range(1, len(params["COLOR_NAMES"]) + 1):
        cc, n_regions = ndimage.label(labels == clbl)
        color_name = params["COLOR_NAMES"][clbl - 1]

        for region_id in range(1, n_regions + 1):
            region_mask = (cc == region_id)
            area = int(region_mask.sum())

            ys, xs = np.where(region_mask)
            rcx, rcy = int(xs.mean()), int(ys.mean())
            bbox_cx = (int(xs.min()) + int(xs.max())) / 2
            bbox_cy = (int(ys.min()) + int(ys.max())) / 2
            pad = closing_disk + 2
            H, W = region_mask.shape
            y1c = max(0, int(ys.min()) - pad); y2c = min(H, int(ys.max()) + pad + 1)
            x1c = max(0, int(xs.min()) - pad); x2c = min(W, int(xs.max()) + pad + 1)
            crop_mask = region_mask[y1c:y2c, x1c:x2c]

            closed = apply_closing(crop_mask, closing_disk)
            mask_u8 = closed.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)
            if polygon is None or len(polygon) < 3:
                continue

            best_idx = _find_best_corner(polygon, angle_tol)
            if best_idx is None:
                continue

            pts = polygon[:, 0, :].astype(float)
            corner_full = (pts[best_idx][0] + x1c, pts[best_idx][1] + y1c)

            n = len(pts)
            A = pts[(best_idx - 1) % n]; C = pts[(best_idx + 1) % n]; B = pts[best_idx]
            eA = A - B; eC = C - B
            if np.linalg.norm(eA) >= np.linalg.norm(eC):
                edge_vec, other_vec = eA, eC
            else:
                edge_vec, other_vec = eC, eA

            cx, cy = corner_full
            cell = _grid_cell(bbox_cx, bbox_cy, img_rgb.shape)
            row, col = cell
            duplicate = False
            if color_name == "black":
                to_remove = []
                for i, existing in enumerate(results):
                    if _grid_cell(*existing["bbox_center"], img_rgb.shape) != cell:
                        continue
                    ebx, eby = existing["bbox_center"]
                    if cell != (1, 1):
                        if row in (0, 2):
                            if abs(bbox_cx - ebx) >= proximity_thresh:
                                continue
                        elif col in (0, 2):
                            if abs(bbox_cy - eby) >= proximity_thresh:
                                continue
                    if area >= existing["region_area"]:
                        to_remove.append(i)
                    else:
                        duplicate = True
                        break
                if not duplicate:
                    for i in reversed(to_remove):
                        results.pop(i)
            else:
                for idx, existing in enumerate(results):
                    if existing["color"] != color_name and cell != (1, 1):
                        continue
                    if _grid_cell(*existing["bbox_center"], img_rgb.shape) != cell:
                        continue
                    ebx, eby = existing["bbox_center"]
                    if cell != (1, 1):
                        if row in (0, 2):
                            if abs(bbox_cx - ebx) >= proximity_thresh:
                                continue
                        elif col in (0, 2):
                            if abs(bbox_cy - eby) >= proximity_thresh:
                                continue
                    keep_current = (area >= existing["region_area"])
                    if keep_current:
                        results.pop(idx)
                    else:
                        duplicate = True
                    break
            if duplicate:
                continue

            patch = _extract_patch(img_rgb, corner_full, edge_vec, other_vec,
                                   patch_size, corner_margin)
            results.append({
                "patch":        patch,
                "color":        color_name,
                "corner_pos":   (int(cx), int(cy)),
                "region_center": (rcx, rcy),
                "bbox_center":  (bbox_cx, bbox_cy),
                "region_area":  area,
            })

    return results


def visualize_morph_steps(labels, params, closing_disk, epsilon_factor, angle_tol):
    rows = []
    for clbl in range(1, len(params["COLOR_NAMES"]) + 1):
        cc, n_regions = ndimage.label(labels == clbl)
        color_name = params["COLOR_NAMES"][clbl - 1]
        for region_id in range(1, n_regions + 1):
            region_mask = (cc == region_id)
            area = int(region_mask.sum())
            ys, xs = np.where(region_mask)
            pad = closing_disk + 2
            H, W = region_mask.shape
            y1c = max(0, int(ys.min()) - pad); y2c = min(H, int(ys.max()) + pad + 1)
            x1c = max(0, int(xs.min()) - pad); x2c = min(W, int(xs.max()) + pad + 1)
            crop = region_mask[y1c:y2c, x1c:x2c]
            closed = apply_closing(crop, closing_disk)
            mask_u8 = closed.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygon, best_idx = None, None
            if contours:
                contour = max(contours, key=cv2.contourArea)
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                polygon = cv2.approxPolyDP(contour, epsilon, True)
                if polygon is not None and len(polygon) >= 3:
                    best_idx = _find_best_corner(polygon, angle_tol)
            rows.append((color_name, area, crop, closed, polygon, best_idx))

    n = len(rows)
    if n == 0:
        print("No regions to display.")
        return
    fig, axes = plt.subplots(n, 2, figsize=(6, n * 3))
    axes = np.array(axes).reshape(n, 2)
    axes[0, 0].set_title("Original crop", fontsize=9, fontweight="bold")
    axes[0, 1].set_title(f"After closing + polygon  (ε={epsilon_factor})", fontsize=9, fontweight="bold")
    for row, (color_name, area, crop, closed, polygon, best_idx) in enumerate(rows):
        passed = best_idx is not None
        row_color = "white" if passed else "red"
        axes[row, 0].imshow(crop, cmap="gray"); axes[row, 0].axis("off")
        axes[row, 1].imshow(closed, cmap="gray"); axes[row, 1].axis("off")
        axes[row, 0].set_ylabel(
            f"{color_name}\narea={area:,}\n{'✓ corner found' if passed else '✗ no corner'}",
            fontsize=7, rotation=0, labelpad=60, va="center", color=row_color,
        )
        if polygon is not None and len(polygon) >= 3:
            pts = polygon[:, 0, :].astype(float)
            n_pts = len(pts)
            for i in range(n_pts):
                p1, p2 = pts[i], pts[(i + 1) % n_pts]
                axes[row, 1].plot([p1[0], p2[0]], [p1[1], p2[1]], "b-", linewidth=1.5)
            for i, pt in enumerate(pts):
                color = "lime" if i == best_idx else "red"
                axes[row, 1].plot(pt[0], pt[1], "o", color=color, markersize=6)
        else:
            axes[row, 1].text(0.5, 0.5, "no polygon", ha="center", va="center",
                              transform=axes[row, 1].transAxes, color="red", fontsize=8)
    plt.tight_layout()
    plt.show()


def visualize_corner_positions(img_rgb, results):
    dot_colors = {
        "red": (220, 50, 50), "yellow": (220, 180, 0),
        "green": (40, 160, 60), "blue": (30, 100, 220), "black": (180, 180, 180),
    }
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img_rgb); ax.axis("off")
    for r in results:
        cx, cy = r["corner_pos"]
        c = [v / 255 for v in dot_colors.get(r["color"], (255, 255, 255))]
        ax.plot(cx, cy, "o", color=c, markersize=10, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(cx + 10, cy, r["color"], color=c, fontsize=8, fontweight="bold")
    ax.set_title(f"Corner positions — {len(results)} patches extracted")
    plt.tight_layout(); plt.show()


def visualize_patches(results):
    n = len(results)
    if n == 0:
        print("No patches to display.")
        return
    ncols = min(n, 6)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.5))
    axes = np.array(axes).reshape(nrows, ncols)
    for i, r in enumerate(results):
        row, col = divmod(i, ncols)
        axes[row, col].imshow(r["patch"])
        axes[row, col].set_title(f"{r['color']}\narea={r['region_area']:,}", fontsize=7)
        axes[row, col].axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    plt.suptitle("Extracted corner patches"); plt.tight_layout(); plt.show()

### CARD DETECTION PIPELINE 3: NUMBER CLASSIFICATION VIA TEMPLATE MATCHING

def _rotated_corners(cx, cy, half, angle_deg):
    """4 corners of a square centred at (cx,cy) rotated by angle_deg."""
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    offsets = [(-half, -half), (half, -half), (half, half), (-half, half)]
    return [(cx + cos_a * dx - sin_a * dy,
             cy + sin_a * dx + cos_a * dy) for dx, dy in offsets]


def _run_patch_selector(img_rgb, _PATCH_SIZE, _OUTPUT_DIR):
    H, W = img_rgb.shape[:2]
    scale = min(1.0, 900 / max(H, W))
    dH, dW = int(H * scale), int(W * scale)

    pil_orig    = Image.fromarray(img_rgb.astype(np.uint8))
    pil_display = pil_orig.resize((dW, dH), Image.LANCZOS)

    def _png(pil):
        buf = io.BytesIO(); pil.save(buf, format='PNG'); return buf.getvalue()

    sl_layout = widgets.Layout(width='700px')
    x_sl  = widgets.IntSlider(0, 0, W - _PATCH_SIZE, description='X:',
                              continuous_update=True, layout=sl_layout)
    y_sl  = widgets.IntSlider(0, 0, H - _PATCH_SIZE, description='Y:',
                              continuous_update=True, layout=sl_layout)
    rot_sl = widgets.IntSlider(0, -180, 180, description='Rotation:',
                               continuous_update=True, layout=sl_layout)

    img_w   = widgets.Image(format='png', layout=widgets.Layout(max_width='900px'))
    patch_w = widgets.Image(format='png', layout=widgets.Layout(width='150px', height='150px'))
    save_b  = widgets.Button(description='Save', button_style='success', icon='check',
                             layout=widgets.Layout(width='130px', height='36px'))
    retry_b = widgets.Button(description='Retry', button_style='warning', icon='refresh',
                             layout=widgets.Layout(width='130px', height='36px'))
    status  = widgets.Label(f'Slide to position and rotate the {{{_PATCH_SIZE}×{_PATCH_SIZE}}} box, then click Save.')
    count   = [0]

    def redraw(x, y, angle):
        # ── display image with rotated box overlay ────────────────────────────
        d    = pil_display.copy()
        draw = ImageDraw.Draw(d)
        half = _PATCH_SIZE * scale / 2
        dcx  = (x + _PATCH_SIZE / 2) * scale
        dcy  = (y + _PATCH_SIZE / 2) * scale
        corners = _rotated_corners(dcx, dcy, half, angle)
        for i in range(4):
            draw.line([corners[i], corners[(i + 1) % 4]], fill=(255, 0, 0), width=3)
        img_w.value = _png(d)

        # ── patch: rotate full image around box centre, then crop ─────────────
        cx_orig = x + _PATCH_SIZE / 2
        cy_orig = y + _PATCH_SIZE / 2
        rotated = pil_orig.rotate(angle, resample=Image.BICUBIC,
                                  center=(cx_orig, cy_orig), expand=False)
        patch_w.value = _png(rotated.crop((x, y, x + _PATCH_SIZE, y + _PATCH_SIZE)))

    x_sl.observe(lambda _:   redraw(x_sl.value, y_sl.value, rot_sl.value), names='value')
    y_sl.observe(lambda _:   redraw(x_sl.value, y_sl.value, rot_sl.value), names='value')
    rot_sl.observe(lambda _: redraw(x_sl.value, y_sl.value, rot_sl.value), names='value')

    def on_save(_):
        x, y, angle = x_sl.value, y_sl.value, rot_sl.value
        cx_orig = x + _PATCH_SIZE / 2
        cy_orig = y + _PATCH_SIZE / 2
        rotated = pil_orig.rotate(angle, resample=Image.BICUBIC,
                                  center=(cx_orig, cy_orig), expand=False)
        patch = rotated.crop((x, y, x + _PATCH_SIZE, y + _PATCH_SIZE))
        count[0] += 1
        path = _OUTPUT_DIR / f"patch_{count[0]:03d}.png"
        patch.save(str(path))
        status.value = f'Saved → {path}  (total: {count[0]})'

    def on_retry(_):
        status.value = 'Ready — reposition the box and click Save again.'

    save_b.on_click(on_save)
    retry_b.on_click(on_retry)

    redraw(0, 0, 0)
    display(widgets.VBox([
        widgets.HTML(
            f'<b>Image {H}c{W} — select a {_PATCH_SIZE}×{_PATCH_SIZE} patch and save to '
            f'<code>{_OUTPUT_DIR}/</code></b>'
        ),
        x_sl, y_sl, rot_sl,
        widgets.HBox([save_b, retry_b, widgets.Label('  '), status]),
        widgets.HBox([
            img_w,
            widgets.VBox([widgets.Label(f'Patch preview ({_PATCH_SIZE}×{_PATCH_SIZE}):'), patch_w],
                         layout=widgets.Layout(margin='0 0 0 20px')),
        ]),
    ]))

def load_templates(template_dir, patch_size):
    """Load all PNG templates from template_dir as RGB, resize to patch_size x patch_size."""
    templates = {}
    for p in sorted(template_dir.glob("*.png")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        templates[p.stem] = cv2.resize(img, (patch_size, patch_size))
    print(f"Loaded {len(templates)} templates: {list(templates.keys())}")
    return templates


def visualize_classification(img_rgb, results, predictions, score_label="dist"):
    """
    Image with prediction overlays + patch grid with label/score.
    score_label: 'dist'  for chamfer distance (lower=better),
                 'score' for cross-correlation (higher=better).
    """
    n = len(results)
    if n == 0:
        print("No patches to classify.")
        return

    ncols = min(n, 6)
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(ncols * 2, nrows * 2 + 4))
    ax_img = fig.add_subplot(nrows + 1, 1, 1)
    ax_img.imshow(img_rgb); ax_img.axis("off")
    for r, (label, score) in zip(results, predictions):
        cx, cy = r["corner_pos"]
        ax_img.plot(cx, cy, "wo", markersize=6)
        ax_img.text(cx + 8, cy, f"{label} ({score_label}={score:.1f})", color="white",
                    fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))
    ax_img.set_title("Classification results")

    for i, (r, (label, score)) in enumerate(zip(results, predictions)):
        ax = fig.add_subplot(nrows + 1, ncols, ncols + i + 1)
        ax.imshow(r["patch"])
        ax.set_title(f"{r['color']}\n{label}\n{score_label}={score:.1f}", fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def compute_distance_map(pattern: np.ndarray) -> np.ndarray:
    """
    Chamfer distance map: each pixel value is its distance to the nearest
    contour point of the binarised pattern. Uses the fast integer-approximation
    pass (+3 for orthogonal, +4 for diagonal neighbours).
    """
    threshold = 0.6 * np.max(pattern)
    binary = pattern > threshold

    cnt_list = skimage.measure.find_contours(binary, fully_connected='high',
                                             positive_orientation='low')
    distance_map = np.full(binary.shape, np.inf)
    if cnt_list:
        cnt = np.concatenate(cnt_list, axis=0)  # all regions
        contours = cnt[:, [1, 0]]   # (row, col) → (x, y)
        xs = contours[:, 0].astype(int)
        ys = contours[:, 1].astype(int)
        distance_map[ys, xs] = 0

    n_rows, n_cols = binary.shape

    # Direct pass: top-left → bottom-right
    for i in range(1, n_rows):
        for j in range(1, n_cols):
            neighbors = [
                distance_map[i-1, j-1] + 4,
                distance_map[i-1, j]   + 3,
                distance_map[i-1, j+1] + 4 if j + 1 < n_cols else np.inf,
                distance_map[i,   j-1] + 3,
                distance_map[i,   j],
            ]
            distance_map[i, j] = min(neighbors)

    # Inverse pass: bottom-right → top-left
    for i in range(n_rows - 2, -1, -1):
        for j in range(n_cols - 2, -1, -1):
            neighbors = [
                distance_map[i,   j+1] + 3,
                distance_map[i+1, j-1] + 4 if j - 1 >= 0 else np.inf,
                distance_map[i+1, j]   + 3,
                distance_map[i+1, j+1] + 4,
                distance_map[i,   j],
            ]
            distance_map[i, j] = min(neighbors)

    return distance_map


def compute_distance(imgs: np.ndarray, d_map: np.ndarray) -> np.ndarray:
    """
    Average chamfer distance from each image contour to the reference d_map.

    imgs : (N, H, W) float array — contours are extracted directly from each image
    d_map: (H, W) distance map from compute_distance_map()
    Returns: (N,) array of mean distances (np.inf when no contour is found)
    """
    dist = np.full(len(imgs), np.inf)
    h, w = d_map.shape
    for i in range(len(imgs)):
        try:
            cnt_list = skimage.measure.find_contours(imgs[i], fully_connected='high',
                                                     positive_orientation='low')
            if not cnt_list:
                continue
            cnt = np.concatenate(cnt_list, axis=0)  # all regions
            contour = cnt[:, [1, 0]]   # (row, col) → (x, y)
            x = np.clip(contour[:, 0].astype(int), 0, w - 1)
            y = np.clip(contour[:, 1].astype(int), 0, h - 1)
            dist[i] = np.mean(d_map[y, x])
        except Exception:
            pass
    return dist

def segment_patch_symbol(patch_rgb, min_v, max_s, border, open_disk, min_size):
    """
    Isolate the white number/symbol from a card corner patch.

    1. HSV threshold: V >= SYMBOL_MIN_V and S <= SYMBOL_MAX_S (white pixels).
    2. Zero-out the outer SYMBOL_BORDER strip to exclude the card frame.
    3. Morphological opening (disk=SYMBOL_OPEN_DISK) to remove isolated noise.
    Fallback: Otsu on grayscale for black/wild cards.
    """
    hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
    S_ch, V_ch = hsv[:, :, 1], hsv[:, :, 2]

    white_mask = (V_ch >= min_v) & (S_ch <= max_s)

    H, W = patch_rgb.shape[:2]
    m = border
    interior = np.zeros((H, W), dtype=bool)
    if H > 2 * m and W > 2 * m:
        interior[m:H - m, m:W - m] = True

    symbol = white_mask & interior

    if symbol.sum() == 0:
        # Fallback for black/wild cards: Otsu, still restricted to the interior
        gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        symbol = binary.astype(bool) & interior

    # Opening: erode then dilate to remove small isolated noise blobs
    symbol = apply_opening(symbol, open_disk)
    # Remove small connected components
    symbol = remove_small_objects(symbol, min_size=min_size)
    return symbol.astype(float)

def build_template_distance_maps(templates: dict, min_v: int = None, max_s: int = None, border: int = None, 
                         open_disk: int = None, min_size: int = None) -> dict:
    """
    templates : dict {label: RGB np.ndarray (H, W, 3)} from load_templates()
    Returns   : dict {label: distance_map (H, W)}
    """
    return {label: compute_distance_map(segment_patch_symbol(tmpl, min_v, max_s, border, open_disk, min_size))
            for label, tmpl in templates.items()}

def match_patch_distance(patch_rgb: np.ndarray, card_color: str,
                         template_dmaps: dict, shift_step: int = None, black_only: int = None,
                         min_v: int = None, max_s: int = None, border: int = None, 
                         open_disk: int = None, min_size: int = None) -> tuple:
    """
    Match a patch via chamfer distance against template distance maps.
    Returns (best_label, best_distance) — lower distance = better match.

    Tries all 4x90° rotations of the segmented patch symbol.  For each
    rotation the contour is also shifted by every (dx, dy) on a grid with
    step=shift_step.  The shift range is derived from the contour's own
    bounding box so that every candidate position still fits entirely within
    the template distance map — no hard-coded limit, no coordinate clipping.
    The minimum score over all (rotation, translation) pairs is returned.
    """
    symbol = segment_patch_symbol(patch_rgb, min_v, max_s, border, open_disk, min_size)

    if card_color == "black":
        relevant = {k: v for k, v in template_dmaps.items() if k in black_only}
    else:
        relevant = {k: v for k, v in template_dmaps.items() if k not in black_only}

    best_label, best_dist = "?", np.inf
    for label, d_map in relevant.items():
        h, w = d_map.shape
        for k in range(4):
            rotated = np.rot90(symbol, k)
            # Extract contour once per rotation, reuse across all shifts
            try:
                cnt_list = skimage.measure.find_contours(
                    rotated, fully_connected='high', positive_orientation='low')
            except Exception:
                continue
            if not cnt_list:
                continue
            cnt = np.concatenate(cnt_list, axis=0)
            cx = cnt[:, 1].astype(int)   # col → x
            cy = cnt[:, 0].astype(int)   # row → y

            # Valid shift range: contour bbox must stay inside [0, w) × [0, h)
            dx_min, dx_max = -int(cx.min()), (w - 1) - int(cx.max())
            dy_min, dy_max = -int(cy.min()), (h - 1) - int(cy.max())

            for dy in range(dy_min, dy_max + 1, shift_step):
                for dx in range(dx_min, dx_max + 1, shift_step):
                    d = float(np.mean(d_map[cy + dy, cx + dx]))
                    if d < best_dist:
                        best_dist, best_label = d, label

    return best_label, best_dist


def classify_all_patches_distance(results: list, template_dmaps: dict, shift_step: int = None, black_only: int = None,
                                  min_v: int = None, max_s: int = None, border: int = None, open_disk: int = None, 
                                  min_size: int = None) -> list:
    """
    results : list of dicts with 'patch' (RGB) and 'color' keys
    Returns : list of (label, distance) tuples — lower distance = better match
    """
    return [
        match_patch_distance(r["patch"], r["color"], template_dmaps, shift_step, black_only,
                             min_v, max_s, border, open_disk, min_size)
        for r in results
    ]

def visualize_template_dmaps(templates: dict, template_dmaps: dict, template_symbols: dict = None, 
                             min_v: int = None, max_s: int = None, border: int = None, 
                             open_disk: int = None, min_size: int = None):
    """
    For every template: show (1) RGB patch, (2) HSV symbol mask, (3) distance map.
    Pass template_symbols to show the exact masks used to build the distance maps.
    """
    n = len(templates)
    if n == 0:
        print("No templates loaded.")
        return

    fig, axes = plt.subplots(n, 3, figsize=(9, n * 3))
    axes = np.array(axes).reshape(n, 3)
    axes[0, 0].set_title("Template (RGB)",  fontsize=9, fontweight="bold")
    axes[0, 1].set_title("Symbol mask (HSV)", fontsize=9, fontweight="bold")
    axes[0, 2].set_title("Distance map",    fontsize=9, fontweight="bold")

    for row, (label, tmpl_rgb) in enumerate(templates.items()):
        if template_symbols is not None and label in template_symbols:
            symbol = template_symbols[label]
        else:
            symbol = segment_patch_symbol(tmpl_rgb, min_v, max_s, border, open_disk, min_size)
        dmap   = template_dmaps[label]
        finite = dmap[np.isfinite(dmap)]
        dmap_vis = np.where(np.isfinite(dmap), dmap, finite.max() if len(finite) else 0)

        axes[row, 0].imshow(tmpl_rgb)
        axes[row, 0].set_ylabel(label, fontsize=8, rotation=0, labelpad=70, va="center")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(symbol, cmap="gray")
        axes[row, 1].axis("off")

        im = axes[row, 2].imshow(dmap_vis, cmap="hot")
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.show()

def visualize_patch_matching(results: list, template_dmaps: dict, top_k: int = 5, shift_step: int = None, black_only: int = None,
                             min_v: int = None, max_s: int = None, border: int = None, open_disk: int = None, min_size: int = None):
    """
    For every extracted patch show:
      Col 0: original patch (RGB)
      Col 1: HSV symbol mask (dominant card colour removed)
      Cols 2…: distance map of the top_k closest templates, with the patch
               contour overlaid in cyan — so you can see where the mismatch is.
    """
    if not results:
        print("No patches to display.")
        return

    ncols = 2 + top_k
    n     = len(results)

    fig, axes = plt.subplots(n, ncols, figsize=(ncols * 2.5, n * 2.8))
    axes = np.array(axes).reshape(n, ncols)

    for row, r in enumerate(results):
        patch_rgb  = r["patch"]
        color_name = r["color"]
        symbol     = segment_patch_symbol(patch_rgb, min_v, max_s, border, open_disk, min_size)

        if color_name == "black":
            relevant = {k: v for k, v in template_dmaps.items() if k in black_only}
        else:
            relevant = {k: v for k, v in template_dmaps.items() if k not in black_only}

        scores = {}
        best_rot = {}
        best_shift = {}
        for label, dmap in relevant.items():
            h, w = dmap.shape
            best_d, best_k, best_dx, best_dy = np.inf, 0, 0, 0
            for k in range(4):
                rotated = np.rot90(symbol, k)
                try:
                    cnt_list = skimage.measure.find_contours(
                        rotated, fully_connected='high', positive_orientation='low')
                except Exception:
                    continue
                if not cnt_list:
                    continue
                cnt = np.concatenate(cnt_list, axis=0)
                cx = cnt[:, 1].astype(int)
                cy = cnt[:, 0].astype(int)
                dx_min, dx_max = -int(cx.min()), (w - 1) - int(cx.max())
                dy_min, dy_max = -int(cy.min()), (h - 1) - int(cy.max())
                for dy in range(dy_min, dy_max + 1, shift_step):
                    for dx in range(dx_min, dx_max + 1, shift_step):
                        d = float(np.mean(dmap[cy + dy, cx + dx]))
                        if d < best_d:
                            best_d, best_k, best_dx, best_dy = d, k, dx, dy
            scores[label]     = best_d
            best_rot[label]   = best_k
            best_shift[label] = (best_dx, best_dy)

        ranked = sorted(scores.items(), key=lambda x: x[1])

        axes[row, 0].imshow(patch_rgb)
        axes[row, 0].set_title(f"{color_name}\ncorner={r['corner_pos']}", fontsize=7)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(symbol, cmap="gray")
        axes[row, 1].set_title("symbol mask", fontsize=7)
        axes[row, 1].axis("off")

        for col_offset, (label, dist) in enumerate(ranked[:top_k]):
            ax   = axes[row, 2 + col_offset]
            dmap = template_dmaps[label]
            rotated_sym = np.rot90(symbol, best_rot[label])

            finite = dmap[np.isfinite(dmap)]
            dmap_vis = np.where(np.isfinite(dmap), dmap, finite.max() if len(finite) else 0)
            ax.imshow(dmap_vis, cmap="hot")

            try:
                cnts = skimage.measure.find_contours(rotated_sym, fully_connected='high',
                                                     positive_orientation='low')
                bdx, bdy = best_shift[label]
                for cnt in cnts:
                    ax.plot(cnt[:, 1] + bdx, cnt[:, 0] + bdy, "c-", linewidth=0.8, alpha=0.8)
            except Exception:
                pass

            winner_str = " ★" if col_offset == 0 else ""
            ax.set_title(f"{label}{winner_str}\ndist={dist:.1f}", fontsize=6,
                         color="lime" if col_offset == 0 else "white",
                         fontweight="bold" if col_offset == 0 else "normal")
            ax.axis("off")

        for col_offset in range(len(ranked), top_k):
            axes[row, 2 + col_offset].axis("off")

    axes[0, 0].set_title("Patch (RGB)",    fontsize=8, fontweight="bold")
    axes[0, 1].set_title("Symbol mask",    fontsize=8, fontweight="bold")
    for k in range(top_k):
        axes[0, 2 + k].set_title(f"Template rank {k+1}\n(dist map + contour)",
                                  fontsize=7, fontweight="bold")

    plt.suptitle("Per-patch matching diagnostics  (cyan = patch contour on template dist map)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.show()

### TEST SET CLASSIFICATION

def _to_card_label(tmpl_label, color, params):
    if tmpl_label not in params["_TMPL_TO_VALUE"]:
        return None
    value = params["_TMPL_TO_VALUE"][tmpl_label]
    if value is None:
        return 'draw_4' if tmpl_label == 'template_plus4' else 'wild'
    return f"{params['_COLOR_ABBREV'].get(color, color)}_{value}"


def classify_image(img_rgb, tmpl_dmaps, params):
    noisy  = _is_noisy_background(img_rgb)
    H, W   = img_rgb.shape[:2]

    labels, _          = segment(img_rgb, min_region_size=params["MIN_REGION_SIZE"])
    f_labels, f_boxes  = filter_cards_by_corners(labels, params)
    token_box, token_loc, card_labels, card_boxes = detect_token(
        f_labels, f_boxes, img_rgb.shape, noisy=noisy)

    merged_boxes = merge_nearby_boxes(card_boxes, params["MERGE_DISTANCE"])
    valid_boxes  = filter_by_size(merged_boxes, params["MIN_CARD_AREA"])
    card_labels  = _filter_labels_by_boxes(card_labels, valid_boxes, params, noisy=noisy)

    active  = params["_LOC_TO_PLAYER"].get(token_loc, 'EMPTY')
    patches = extract_corner_patches(
                  img_rgb, card_labels,
                  closing_disk=params["CLOSING_DISK"], epsilon_factor=params["EPSILON_FACTOR"],
                  patch_size=params["PATCH_SIZE"], proximity_thresh=params["PROXIMITY_THRESH"],
                  angle_tol=params["CORNER_TOL"], corner_margin=params["CORNER_MARGIN"])
    preds   = classify_all_patches_distance(patches, tmpl_dmaps, params["SHIFT_STEP"])

    # Zone anchors: midpoint of each player edge + image centre
    anchors = {
        'center': (W / 2, H / 2),
        'top'   : (W / 2, 0),
        'bottom': (W / 2, H),
        'left'  : (0,     H / 2),
        'right' : (W,     H / 2),
    }

    loc_cards = {loc: [] for loc in anchors}
    for r, (tmpl_label, _) in zip(patches, preds):
        rcx, rcy = r['region_center']
        loc = min(anchors, key=lambda k: (rcx - anchors[k][0])**2 + (rcy - anchors[k][1])**2)
        lbl = _to_card_label(tmpl_label, r['color'], params)
        if lbl:
            loc_cards[loc].append(lbl)

    def fmt(cards):
        return ';'.join(dict.fromkeys(cards)) if cards else 'EMPTY'

    return {
        'center_card'   : fmt(loc_cards['center']),
        'active_player' : active,
        'player_1_cards': fmt(loc_cards['bottom']),
        'player_2_cards': fmt(loc_cards['right']),
        'player_3_cards': fmt(loc_cards['top']),
        'player_4_cards': fmt(loc_cards['left']),
    }
