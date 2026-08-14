"""
Figure for the green case study.

Left  : the model's green/yellow decision boundary, swept on flat patches.
Right : where the 25 library greens sit in hue, sized by how often the sampler
        actually draws them. The contested band below 135 degrees holds 2 of 25
        colors and 4.1% of the draws, which is why the model's boundary sits
        at ~135 degrees rather than at the library's 117.7.

Usage
-----
  python scripts/make_green_figure.py
"""
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.color_utils import ColorLibrary
from utils.instrumented_generator import COLOR_CLASSES
from scripts.predict import TorchBackend, softmax

CSV = os.path.join(PROJECT_ROOT, "datasets", "categorized_colors_normalized.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "documentation", "figures")
SWEATER = (57.6, -16.1, 27.1)


def synth(hexcode, bg=(160, 160, 160)):
    h = hexcode.lstrip("#")
    bgr = tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))
    img = np.full((224, 224, 3), bg, np.uint8)
    img[56:168, 56:168] = bgr
    return img


def predict(backend, comp):
    rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
    return softmax(backend(((chw - mean) / std)[None].astype(np.float32))[0])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    backend = TorchBackend(os.path.join(PROJECT_ROOT, "checkpoints",
                                        "distilB_student_int8.pth"))
    lib = ColorLibrary.from_categorized_csv(CSV)
    gi, yi = COLOR_CLASSES.index("green"), COLOR_CLASSES.index("yellow")

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── left: the learned decision boundary ──────────────────────────────────
    a_vals = np.arange(-64, -6, 2.0)
    g, y, hues, cols = [], [], [], []
    for a in a_vals:
        hx = ColorLibrary._lab_to_hex(60.0, float(a), 28.0)[1]
        p = predict(backend, synth(hx))
        g.append(p[gi]); y.append(p[yi])
        hues.append(np.degrees(np.arctan2(28.0, a)) % 360)
        cols.append(ColorLibrary._lab_to_hex(60.0, float(a), 28.0)[0])
    ax1.plot(hues, g, "-o", ms=3.5, lw=2, color="#2b8a3e", label="p(green)")
    ax1.plot(hues, y, "-o", ms=3.5, lw=2, color="#d9a300", label="p(yellow)")
    for h, c in zip(hues, cols):
        ax1.scatter([h], [-0.06], c=[c], s=70, marker="s", edgecolors="none",
                    clip_on=False)
    sw_hue = np.degrees(np.arctan2(SWEATER[2], SWEATER[1])) % 360
    ax1.axvline(sw_hue, color="crimson", ls="--", lw=1.5)
    ax1.annotate("the sweater\n117.8°", (sw_hue, 0.55), color="crimson",
                 fontsize=8, ha="left", xytext=(sw_hue + 3, 0.55))
    ax1.axvline(117.7, color="#555", ls=":", lw=1.2)
    ax1.annotate("library says\ngreen starts here", (117.7, 0.9), fontsize=7.5,
                 color="#555", ha="right", xytext=(116, 0.86))
    ax1.set_xlabel("CIELAB hue angle (degrees).  Flat patches, L*=60, b*=28")
    ax1.set_ylabel("predicted probability")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title("The model's green/yellow boundary sits at ~135°,\n"
                  "not at the library's 117.7°", fontsize=10)
    ax1.legend(fontsize=8, loc="center left")

    # ── right: library greens by hue, sized by draw probability ──────────────
    d = lib.get_category_mahalanobis_distances("green")
    w = np.array([(1.0 / (x + 1e-6)) ** 2 for _, x in d])
    p = w / w.sum()
    labs = [ColorLibrary._hex_to_lab(h) for h, _ in d]
    hue = np.array([np.degrees(np.arctan2(b, a)) % 360 for _, a, b in labs])
    srgb = [ColorLibrary._lab_to_hex(*l)[0] for l in labs]

    ax2.scatter(hue, p * 100, c=srgb, s=260, edgecolors="black", linewidths=0.6, zorder=3)
    ax2.axvspan(110, 135, color="#d9a300", alpha=0.13, zorder=0)
    ax2.axvline(135, color="crimson", ls="--", lw=1.5, zorder=2)
    ax2.annotate("model classifies\neverything left of\nhere as yellow",
                 (134, 8.4), fontsize=8, color="crimson", ha="right")
    ax2.axvline(sw_hue, color="#333", ls=":", lw=1.2, zorder=2)
    ax2.annotate("sweater", (sw_hue, 0.4), fontsize=8, rotation=90,
                 ha="right", va="bottom", color="#333")
    ax2.set_xlabel("CIELAB hue angle of each library green (degrees)")
    ax2.set_ylabel("probability the sampler draws it (%)")
    ax2.set_xlim(108, 205)
    ax2.set_title("Only 2 of 25 library greens sit below 135°,\n"
                  "and they take 4.1% of all green draws", fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "green_boundary.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  {out}")


if __name__ == "__main__":
    main()
