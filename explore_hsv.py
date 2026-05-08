"""
Empirical HSV color range extraction from UNO reference images.
This script is to justify / tune COLOR_HSV_RANGES in main.py.

Usage:
    python explore_hsv.py

Controls (OpenCV window):
  R / Y / G / B  — select which color you are sampling next
  Left click      — sample that pixel and record its HSV under the active color
  P               — print current min/max ranges for all collected samples
  Q               — close current image and move to the next
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

REF_DIR = Path(__file__).parent / "reference_images"
ref_images = sorted(REF_DIR.glob("*.jpg"))

if not ref_images:
    raise FileNotFoundError(f"No .jpg files found in {REF_DIR}")

# BGR colors used to draw crosshairs per UNO color
MARKER_BGR = {
    "r": (0,   0,   220),
    "y": (0,   215, 255),
    "g": (0,   180,  0),
    "b": (200,  80,   0),
}

LABEL = {"r": "Red", "y": "Yellow", "g": "Green", "b": "Blue"}


# ─── Interactive picker ───────────────────────────────────────────────────────

def hsv_picker(img_bgr, title, samples: dict):
    """
    Interactive OpenCV window.
    Keys R/Y/G/B switch the active color.
    Left-click records HSV of that pixel under the active color.
    P prints a live summary. Q closes the window.

    samples: shared dict {color: [(h,s,v), ...]} updated in-place.
    """
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    display = img_bgr.copy()
    state = {"active": "r"}

    def redraw_overlay():
        """Redraw color-selector bar at the top of the display."""
        bar_h = 40
        cv2.rectangle(display, (0, 0), (display.shape[1], bar_h), (30, 30, 30), -1)
        seg_w = display.shape[1] // 4
        for i, c in enumerate(["r", "y", "g", "b"]):
            x0, x1 = i * seg_w, (i + 1) * seg_w
            bgr = MARKER_BGR[c]
            cv2.rectangle(display, (x0, 0), (x1, bar_h), bgr, -1)
            text = f"{LABEL[c]} [{c.upper()}]"
            if c == state["active"]:
                cv2.rectangle(display, (x0, 0), (x1, bar_h), (255, 255, 255), 3)
                text = f">>> {text} <<<"
            cv2.putText(display, text, (x0 + 5, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(display, text, (x0 + 5, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(title, display)

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or y < 40:
            return
        h, s, v = hsv_img[y, x]
        c = state["active"]
        samples[c].append((int(h), int(s), int(v)))
        print(f"  [{LABEL[c]}]  ({x:4d},{y:4d})  H={h:3d}  S={s:3d}  V={v:3d}  "
              f"(total {len(samples[c])} samples)")
        cv2.drawMarker(display, (x, y), MARKER_BGR[c], cv2.MARKER_CROSS, 18, 2)
        cv2.imshow(title, display)

    cv2.imshow(title, display)
    cv2.setMouseCallback(title, on_click)
    redraw_overlay()

    print(f"\n── {title} ──")
    print("  Press R/Y/G/B to select color, click to sample, P to print ranges, Q for next image.")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("r"), ord("y"), ord("g"), ord("b")):
            state["active"] = chr(key)
            redraw_overlay()
            print(f"  Active color → {LABEL[state['active']]}")
        elif key == ord("p"):
            print_ranges(samples)

    cv2.destroyAllWindows()


# Summary helpers 

def print_ranges(samples: dict):
    print("\n  ── Current HSV ranges (min / max across all samples) ──")
    print(f"  {'Color':<8}  {'H min–max':>12}  {'S min–max':>12}  {'V min–max':>12}  n")
    any_data = False
    for c in ["r", "y", "g", "b"]:
        pts = samples.get(c, [])
        if not pts:
            print(f"  {LABEL[c]:<8}  (no samples)")
            continue
        any_data = True
        hs = [p[0] for p in pts]
        ss = [p[1] for p in pts]
        vs = [p[2] for p in pts]
        print(f"  {LABEL[c]:<8}  {min(hs):>4} – {max(hs):<4}    "
              f"{min(ss):>4} – {max(ss):<4}    "
              f"{min(vs):>4} – {max(vs):<4}    {len(pts)}")
    if any_data:
        print()
        print("  Suggested COLOR_HSV_RANGES snippet:")
        for c in ["r", "y", "g", "b"]:
            pts = samples.get(c, [])
            if not pts:
                continue
            hs = [p[0] for p in pts]
            ss = [p[1] for p in pts]
            vs = [p[2] for p in pts]
            h_lo, h_hi = max(0, min(hs) - 5), min(179, max(hs) + 5)
            s_lo = max(0, min(ss) - 20)
            v_lo = max(0, min(vs) - 20)
            if c == "r" and h_hi > 90:
                # red wraps around 0/179 — split into two ranges
                print(f'    "{c}": [({h_lo}, {s_lo}, {v_lo}), (10, 255, 255), '
                      f'(150, {s_lo}, {v_lo}), ({h_hi}, 255, 255)],')
            else:
                print(f'    "{c}": [({h_lo}, {s_lo}, {v_lo}), ({h_hi}, 255, 255)],')


def plot_hsv_histograms(images_bgr, titles, samples=None):
    """H, S, V histograms per image with sampled points overlaid as vertical lines."""
    n = len(images_bgr)
    _, axes = plt.subplots(4, n, figsize=(5 * n, 14))
    if n == 1:
        axes = np.array(axes).reshape(4, 1)

    color_bands = {
        "r": [(0, 10), (160, 179)],
        "y": [(20, 35)],
        "g": [(36, 85)],
        "b": [(86, 130)],
    }
    mpl_color = {"r": "red", "y": "gold", "g": "green", "b": "royalblue"}

    for i, (img, title) in enumerate(zip(images_bgr, titles)):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat_mask = hsv[:, :, 1] > 60
        val_mask = (hsv[:, :, 2] > 40) & (hsv[:, :, 2] < 240)
        mask = sat_mask & val_mask

        # Row 0: image
        axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(title, fontsize=9)
        axes[0, i].axis("off")

        channel_cfg = [
            (1, hsv[:, :, 0][mask].flatten(), 90,  (0, 180),  "Hue (0–179)",        [0,15,30,60,90,120,150,179]),
            (2, hsv[:, :, 1][mask].flatten(), 64,  (0, 256),  "Saturation (0–255)", [0,64,128,192,255]),
            (3, hsv[:, :, 2][mask].flatten(), 64,  (0, 256),  "Value (0–255)",      [0,64,128,192,255]),
        ]

        for row, vals, bins, rng, xlabel, xticks in channel_cfg:
            ax = axes[row, i]
            ax.hist(vals, bins=bins, range=rng, color="steelblue", edgecolor="none")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Pixel count")
            ax.set_xticks(xticks)
            ax.grid(axis="x", alpha=0.3)

            # Shade hue bands only on the H histogram
            if row == 1:
                for color_name, bands in color_bands.items():
                    for lo, hi in bands:
                        ax.axvspan(lo, hi, alpha=0.12, color=mpl_color[color_name],
                                   label=LABEL[color_name])

            # Overlay sampled values as vertical lines
            if samples is not None:
                for c, pts in samples.items():
                    for (h, s, v) in pts:
                        val = {1: h, 2: s, 3: v}[row]
                        ax.axvline(val, color=mpl_color[c], alpha=0.7, linewidth=1.2)

            if row == 1:
                handles, labels = ax.get_legend_handles_labels()
                seen = {}
                for h_handle, l in zip(handles, labels):
                    seen.setdefault(l, h_handle)
                ax.legend(seen.values(), seen.keys(), fontsize=7, loc="upper right")

    plt.suptitle("HSV histograms — vertical lines = your sampled pixels\n"
                 "(shaded bands on H = expected color ranges)", fontsize=11)
    plt.tight_layout()
    plt.show()


# Main 

def main():
    images, titles = [], []
    for p in ref_images:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Could not read {p}")
            continue
        h, w = img.shape[:2]
        scale = min(1.0, 1200 / w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        images.append(img)
        titles.append(p.name)

    if not images:
        raise RuntimeError("No images could be loaded.")

    # Shared sample store across all images
    samples = defaultdict(list)

    for img, title in zip(images, titles):
        hsv_picker(img, title=f"HSV picker — {title}", samples=samples)

    # Final summary
    print("\n══ Final summary ══")
    print_ranges(samples)

    # Histogram with sampled points overlaid
    plot_hsv_histograms(images, titles, samples)


if __name__ == "__main__":
    main()