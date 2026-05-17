import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import disk, closing, opening, remove_small_holes, remove_small_objects

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


