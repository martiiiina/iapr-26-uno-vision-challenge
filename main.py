"""
UNO Vision Challenge
Pipeline: Segmentation → Feature Extraction → Classification
Approach mirrors the IAPR labs: color-based segmentation, shape/color descriptors, kNN/SVM.
"""

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
from utils import preprocess, build_training_data, predict_image
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


# Main 

def main():
    df_train = pd.read_csv(TRAIN_CSV)
    df_sample = pd.read_csv(SAMPLE_CSV)

    print(f"Training on {len(df_train)} images …")
    X_train, y_train = build_training_data(df_train)
    print(f"  → {len(X_train)} card samples, {len(set(y_train))} unique labels")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    # kNN classifier (Lab 3 style)
    clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    clf.fit(X_train, y_enc)

    # Quick sanity check on training set
    y_pred = clf.predict(X_train)
    print(f"  Train accuracy (card-level): {accuracy_score(y_enc, y_pred):.2%}")

    # Predict test images
    print(f"\nPredicting {len(df_sample)} test images …")
    results = []
    for _, row in df_sample.iterrows():
        img_path = TEST_DIR / f"{row['image_id']}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            results.append({
                "image_id": row["image_id"],
                "center_card": "EMPTY",
                "active_player": "p1",
                "player_1_cards": "EMPTY",
                "player_2_cards": "EMPTY",
                "player_3_cards": "EMPTY",
                "player_4_cards": "EMPTY",
            })
            continue
        img = preprocess(img)
        pred = predict_image(img, clf, le)
        pred["image_id"] = row["image_id"]
        results.append(pred)

    df_out = pd.DataFrame(results, columns=["image_id", "center_card", "active_player",
                                             "player_1_cards", "player_2_cards",
                                             "player_3_cards", "player_4_cards"])
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
