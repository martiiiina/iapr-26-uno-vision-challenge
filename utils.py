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

