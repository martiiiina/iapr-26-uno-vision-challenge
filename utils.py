import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy import ndimage
from skimage import morphology, measure
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Paths 

ROOT = Path(__file__).parent
TRAIN_DIR = ROOT / "train_images"
TEST_DIR  = ROOT / "test_images"
REF_DIR   = ROOT / "reference_images"
TRAIN_CSV = ROOT / "train.csv"
SAMPLE_CSV = ROOT / "sample_submission.csv"
OUTPUT_CSV = ROOT / "submission.csv"

# Card taxonomy 

COLORS = ["r", "y", "g", "b"]
NUMBERS = [str(i) for i in range(10)]
ACTIONS = ["skip", "reverse", "draw_2"]
WILD_CARDS = ["wild", "draw_4"]

ALL_CARDS = (
    [f"{c}_{n}" for c in COLORS for n in NUMBERS]
    + [f"{c}_{a}" for c in COLORS for a in ACTIONS]
    + WILD_CARDS
    + ["EMPTY"]
)
## NOTE : VISUALIZE SAMPLE IMAGES TO GET THE COLORS
# HSV ranges for UNO card colors (H in [0,179], S/V in [0,255])
COLOR_HSV_RANGES = {
    "r": [(0, 100, 100), (10, 255, 255), (160, 100, 100), (179, 255, 255)],  # two red ranges
    "y": [(10, 100, 100), (35, 255, 255)],
    "g": [(36, 80, 80),  (85, 255, 255)],
    "b": [(86, 80, 80),  (130, 255, 255)],
}


# ─── 1. Segmentation ──────────────────────────────────────────────────────────

def preprocess(img_bgr):
    """Resize to a standard width while keeping aspect ratio."""
    h, w = img_bgr.shape[:2]
    target_w = 1200
    scale = target_w / w
    return cv2.resize(img_bgr, (target_w, int(h * scale)))


def segment_cards(img_bgr):
    """
    Detect card-shaped rectangular regions via edge detection + contour filtering.
    Returns list of (x, y, w, h) bounding boxes sorted left-to-right, top-to-bottom.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Morphological closing to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.003 or area > img_area * 0.4:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h
        # Cards are roughly 0.6–1.8 aspect ratio
        if 0.5 < aspect < 2.0:
            boxes.append((x, y, w, h))

    # Merge heavily overlapping boxes (NMS-style)
    boxes = _nms(boxes, iou_threshold=0.3)
    return sorted(boxes, key=lambda b: (b[1], b[0]))  # top-to-bottom, left-to-right


def _nms(boxes, iou_threshold=0.3):
    """Simple greedy NMS."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for box in boxes:
        x1, y1, w1, h1 = box
        suppress = False
        for kx, ky, kw, kh in kept:
            ix = max(0, min(x1+w1, kx+kw) - max(x1, kx))
            iy = max(0, min(y1+h1, ky+kh) - max(y1, ky))
            inter = ix * iy
            union = w1*h1 + kw*kh - inter
            if union > 0 and inter/union > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(box)
    return kept


# ─── 2. Spatial layout → roles ────────────────────────────────────────────────

def assign_regions(img_bgr, boxes):
    """
    Split bounding boxes into spatial roles based on image quadrants.
    Returns dict: {role: [box, ...]}
    Roles: 'center', 'top' (p3), 'bottom' (p1), 'left' (p4), 'right' (p2)
    Convention matches typical UNO table layout.
    """
    H, W = img_bgr.shape[:2]
    cx, cy = W / 2, H / 2
    margin = 0.25  # center zone ±25% of each axis

    roles = {"center": [], "top": [], "bottom": [], "left": [], "right": []}
    for (x, y, w, h) in boxes:
        bx, by = x + w/2, y + h/2
        in_center_x = abs(bx - cx) < W * margin
        in_center_y = abs(by - cy) < H * margin
        if in_center_x and in_center_y:
            roles["center"].append((x, y, w, h))
        elif by < cy - H * margin:
            roles["top"].append((x, y, w, h))
        elif by > cy + H * margin:
            roles["bottom"].append((x, y, w, h))
        elif bx < cx - W * margin:
            roles["left"].append((x, y, w, h))
        else:
            roles["right"].append((x, y, w, h))
    return roles


# ─── 3. Feature extraction ────────────────────────────────────────────────────

def extract_card_color(img_bgr, box):
    """Classify card background color using HSV histogram."""
    x, y, w, h = box
    roi = img_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return "wild"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    scores = {}
    for color, ranges in COLOR_HSV_RANGES.items():
        if color == "r":
            m1 = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
            m2 = cv2.inRange(hsv, np.array(ranges[2]), np.array(ranges[3]))
            mask = cv2.bitwise_or(m1, m2)
        else:
            mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
        scores[color] = np.sum(mask) / mask.size
    best = max(scores, key=scores.get)
    return best if scores[best] > 0.05 else "wild"


def extract_features(img_bgr, box):
    """
    Extract a feature vector for one card region:
    - Color histogram in HSV (hue channel, 32 bins)
    - Saturation histogram (16 bins)
    - Dominant color class (one-hot, 5 values: r/y/g/b/wild)
    - Aspect ratio
    Total: 32 + 16 + 5 + 1 = 54 dims
    """
    x, y, w, h = box
    roi = img_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return np.zeros(54)

    roi_resized = cv2.resize(roi, (64, 96))
    hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    h_hist = h_hist / (h_hist.sum() + 1e-6)
    s_hist = s_hist / (s_hist.sum() + 1e-6)

    color_label = extract_card_color(img_bgr, box)
    color_onehot = np.zeros(5)
    color_map = {"r": 0, "y": 1, "g": 2, "b": 3, "wild": 4}
    color_onehot[color_map[color_label]] = 1.0

    aspect = np.array([w / (h + 1e-6)])

    return np.concatenate([h_hist, s_hist, color_onehot, aspect])


def extract_number_features(img_bgr, box):
    """
    Coarse digit/symbol features using HOG-like gradient histogram on grayscale center crop.
    Returns 32-dim vector.
    """
    x, y, w, h = box
    roi = img_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return np.zeros(32)

    # Focus on center 60% where the number/symbol lives
    cy, cx = h // 2, w // 2
    crop_h, crop_w = int(h * 0.6), int(w * 0.6)
    y0 = max(0, cy - crop_h // 2)
    x0 = max(0, cx - crop_w // 2)
    center = roi[y0:y0+crop_h, x0:x0+crop_w]
    if center.size == 0:
        return np.zeros(32)

    gray = cv2.cvtColor(cv2.resize(center, (32, 48)), cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    ang = np.arctan2(sobely, sobelx) % np.pi

    hist, _ = np.histogram(ang.flatten(), bins=32, range=(0, np.pi), weights=mag.flatten())
    return hist / (hist.sum() + 1e-6)


def full_feature(img_bgr, box):
    """Concatenate color + gradient features."""
    return np.concatenate([extract_features(img_bgr, box),
                           extract_number_features(img_bgr, box)])


# ─── 4. Build reference dataset from training images ─────────────────────────

def build_training_data(df):
    """
    Extract (feature_vector, card_label) pairs from all training images.
    Uses the CSV labels to associate detected card regions with their ground-truth labels.

    This is approximate: we match detected boxes to labeled cards by spatial role.
    """
    X, y = [], []
    for _, row in df.iterrows():
        img_path = TRAIN_DIR / f"{row['image_id']}.jpg"
        if not img_path.exists():
            continue
        img = preprocess(cv2.imread(str(img_path)))
        if img is None:
            continue

        boxes = segment_cards(img)
        roles = assign_regions(img, boxes)

        # Map spatial role → CSV column → expected cards
        role_map = {
            "bottom": ("player_1_cards", "p1"),
            "right":  ("player_2_cards", "p2"),
            "top":    ("player_3_cards", "p3"),
            "left":   ("player_4_cards", "p4"),
        }

        # Center pile: grab the largest box in center role
        if roles["center"]:
            best_box = max(roles["center"], key=lambda b: b[2]*b[3])
            feat = full_feature(img, best_box)
            X.append(feat)
            y.append(row["center_card"])

        # Player hands: distribute detected boxes evenly among expected cards
        for role, (col, _) in role_map.items():
            cards_str = row[col]
            if cards_str == "EMPTY" or pd.isna(cards_str):
                continue
            expected = cards_str.split(";")
            detected = roles[role]
            for i, card_label in enumerate(expected):
                if i < len(detected):
                    feat = full_feature(img, detected[i])
                else:
                    # No matching detected region: use mean of available
                    if detected:
                        feat = np.mean([full_feature(img, b) for b in detected], axis=0)
                    else:
                        feat = np.zeros(86)
                X.append(feat)
                y.append(card_label)

    return np.array(X), np.array(y)


# ─── 5. Active player detection ───────────────────────────────────────────────

def detect_active_player(img_bgr, roles):
    """
    Heuristic: the active player's hand tends to have the most / largest cards visible,
    or there may be a highlighted border. Use card count per region as proxy.
    """
    region_to_player = {"bottom": "p1", "right": "p2", "top": "p3", "left": "p4"}
    counts = {role: len(boxes) for role, boxes in roles.items() if role != "center"}
    best_role = max(counts, key=counts.get, default="bottom")
    return region_to_player.get(best_role, "p1")


# ─── 6. Inference on a single image ──────────────────────────────────────────

def predict_image(img_bgr, clf, le):
    """
    Run full pipeline on one image.
    Returns dict with center_card, active_player, player_X_cards.
    """
    boxes = segment_cards(img_bgr)
    roles = assign_regions(img_bgr, boxes)

    def classify_boxes(box_list):
        if not box_list:
            return "EMPTY"
        preds = []
        for box in box_list:
            feat = full_feature(img_bgr, box).reshape(1, -1)
            pred_idx = clf.predict(feat)[0]
            preds.append(le.inverse_transform([pred_idx])[0])
        # Filter out EMPTY predictions from individual boxes
        preds = [p for p in preds if p != "EMPTY"] or ["EMPTY"]
        return ";".join(preds)

    center = "EMPTY"
    if roles["center"]:
        best = max(roles["center"], key=lambda b: b[2]*b[3])
        feat = full_feature(img_bgr, best).reshape(1, -1)
        center = le.inverse_transform(clf.predict(feat))[0]

    active = detect_active_player(img_bgr, roles)

    return {
        "center_card":    center,
        "active_player":  active,
        "player_1_cards": classify_boxes(roles["bottom"]),
        "player_2_cards": classify_boxes(roles["right"]),
        "player_3_cards": classify_boxes(roles["top"]),
        "player_4_cards": classify_boxes(roles["left"]),
    }


# ─── 7. Visualisation helper ─────────────────────────────────────────────────

def visualize(img_bgr, boxes, roles, title=""):
    role_colors = {
        "center": (0, 255, 255),
        "top":    (255, 0, 0),
        "bottom": (0, 255, 0),
        "left":   (255, 0, 255),
        "right":  (0, 128, 255),
    }
    vis = img_bgr.copy()
    for role, box_list in roles.items():
        color = role_colors.get(role, (200, 200, 200))
        for (x, y, w, h) in box_list:
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
            cv2.putText(vis, role, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    plt.figure(figsize=(14, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
