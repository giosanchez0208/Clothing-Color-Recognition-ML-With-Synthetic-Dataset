"""
Visualize the 13 color categories as regions in CIELAB space.

Two panels:
  left   all 325 library colors in one 3-D CIELAB scatter, each point drawn
         in its own sRGB. The categories are visibly separate volumes, not
         points
  right  a small 3-D view per category, so each region's shape and spread is
         legible on its own

CIELAB rather than RGB because the axes are perceptual: equal distances
correspond roughly to equal perceived differences, which is the whole premise
of asking "which color would a person call this?"

Usage
-----
  python scripts/make_cielab_figure.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers the 3-D projection

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.color_utils import ColorLibrary
from utils.instrumented_generator import COLOR_CLASSES

CSV = os.path.join(PROJECT_ROOT, "datasets", "categorized_colors_normalized.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "assets", "figures")


def srgb_of(lib, lab):
    return ColorLibrary._lab_to_hex(*lab)[0]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lib = ColorLibrary.from_categorized_csv(CSV)
    cats = [c for c in COLOR_CLASSES if c in lib.categories]

    data = {c: np.array(lib.get_category_colors(c)) for c in cats}
    total = sum(len(v) for v in data.values())

    fig = plt.figure(figsize=(17, 8.5))

    # ── left: everything at once ─────────────────────────────────────────────
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for c in cats:
        pts = data[c]
        cols = [srgb_of(lib, p) for p in pts]
        ax.scatter(pts[:, 1], pts[:, 2], pts[:, 0], c=cols, s=55,
                   edgecolors="none", depthshade=True)
    ax.set_xlabel("a*  (green – red)", fontsize=9)
    ax.set_ylabel("b*  (blue – yellow)", fontsize=9)
    ax.set_zlabel("L*  (lightness)", fontsize=9)
    ax.set_xlim(-90, 90); ax.set_ylim(-90, 90); ax.set_zlim(0, 100)
    ax.view_init(elev=18, azim=42)
    ax.set_title(f"All {total} library colors in CIELAB\n"
                 "each category occupies a volume, not a point", fontsize=11)

    # ── right: one small view per category ───────────────────────────────────
    ncol, nrow = 5, 3
    for i, c in enumerate(cats):
        sub = fig.add_subplot(nrow, ncol * 2, (i // ncol) * ncol * 2 + ncol + (i % ncol) + 1,
                              projection="3d")
        pts = data[c]
        sub.scatter(pts[:, 1], pts[:, 2], pts[:, 0],
                    c=[srgb_of(lib, p) for p in pts], s=18, edgecolors="none")
        sub.set_xlim(-90, 90); sub.set_ylim(-90, 90); sub.set_zlim(0, 100)
        sub.view_init(elev=18, azim=42)
        sub.set_xticklabels([]); sub.set_yticklabels([]); sub.set_zticklabels([])
        sub.set_title(f"{c}  ({len(pts)})", fontsize=8, pad=-2)
        sub.grid(True)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "cielab_categories.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  {out}")

    # ── hue-angle view: why a hue rotation is dangerous here ─────────────────
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for c in cats:
        pts = data[c]
        hue = np.degrees(np.arctan2(pts[:, 2], pts[:, 1])) % 360
        chroma = np.hypot(pts[:, 1], pts[:, 2])
        keep = chroma > 12                     # skip near-neutrals
        ax.scatter(hue[keep], pts[keep, 0],
                   c=[srgb_of(lib, p) for p in pts[keep]],
                   s=45 + chroma[keep], edgecolors="none", alpha=0.9)
        if keep.sum():
            ax.annotate(c, (hue[keep].mean(), pts[keep, 0].mean()), fontsize=8,
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.65,
                                  ec="none"))
    ax.set_xlabel("CIELAB hue angle  h_ab  (degrees)")
    ax.set_ylabel("L*  (lightness)")
    ax.set_xlim(0, 360); ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.set_title("Chromatic categories by hue angle. Adjacent centers are a "
                 "median 13.2° apart,\nwhich is why a ±25° hue shift can move a "
                 "color into a different category", fontsize=10)
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "cielab_hue_angles.png")
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  {out2}")


if __name__ == "__main__":
    main()
