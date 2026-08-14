"""
Build presentation figures for the synthetic data engine.

These are deliberately separate from the audit figures in reports/figures/.
The audit answers "is this dataset valid?"; these answer "what does this
generator actually do?". The second question is the one worth showing a
reader who has thirty seconds.

Outputs (reports/showcase/):
  augmentation_axes.png     each axis swept in isolation, one row per axis,
                            identical base image throughout
  patterns.png              the seven garment pattern types
  augmentation_stacking.png how effects compound at realistic densities
  pipeline_stages.png       background -> patch -> folds -> composite -> augmented

Usage
-----
  python scripts/make_showcase.py
  python scripts/make_showcase.py --seed 7
"""
import argparse
import os
import random
import sys

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.instrumented_augment import AXES, apply_single_axis, probe_values
from utils.instrumented_generator import (
    COLOR_CLASSES, InstrumentedGenerator, split_background_pool,
)
from utils.color_utils_extended import PATTERN_TYPES_V2

DEFAULT_BGS = os.path.join(PROJECT_ROOT, "datasets", "indoorCVPR_09_modified")
DEFAULT_CSV = os.path.join(PROJECT_ROOT, "datasets", "categorized_colors_normalized.csv")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "reports", "showcase")

# Units for axis labels, so a reader knows what the number means
AXIS_UNIT = {
    "brightness": "x", "temperature": " (cool<->warm)", "hue": "deg",
    "saturation": "x", "shadow": "", "blur": " sigma",
    "noise": " density", "jpeg": " quality", "specular": "", "vignette": "",
}


def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def fmt(axis, v):
    u = AXIS_UNIT[axis]
    return f"{v:.0f}{u}" if axis in ("hue", "jpeg") else f"{v:.3g}{u}"


# ── Figure 1: every axis swept in isolation ──────────────────────────────────
def fig_axes(gen, out_dir, seed, n_levels=7):
    """One row per axis, identical base image, parameter increasing left to right.

    This is the visual statement of the probe set's design: only one thing
    changes along a row, so any perceived difference is attributable.
    """
    base, _, _ = gen.generate(index=101, base_seed=seed, probe=True,
                              axis=None, value=None)   # neutral composite

    n_rows = len(AXES)
    fig, axes = plt.subplots(n_rows, n_levels + 1,
                             figsize=(1.55 * (n_levels + 1), 1.62 * n_rows))

    for r, axis in enumerate(AXES):
        # Column 0 is the untouched control for that row
        axes[r][0].imshow(rgb(base))
        axes[r][0].set_ylabel(axis, fontsize=9, rotation=0, ha="right",
                              va="center", labelpad=42, fontweight="bold")
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for s in axes[r][0].spines.values():
            s.set_edgecolor("#2b8a3e"); s.set_linewidth(2.5)
        if r == 0:
            axes[r][0].set_title("neutral", fontsize=8, color="#2b8a3e",
                                 fontweight="bold")

        for c, v in enumerate(probe_values(axis, n_levels)):
            random.seed(seed + r * 100 + c)      # stabilize shadow direction etc.
            np.random.seed(seed + r * 100 + c)
            out, _ = apply_single_axis(base, axis, v)
            ax = axes[r][c + 1]
            ax.imshow(rgb(out))
            ax.set_title(fmt(axis, v), fontsize=7.5)
            ax.axis("off")

    fig.suptitle("Augmentation axes swept in isolation\n"
                 "identical base image; exactly one parameter varies per row",
                 fontsize=13, y=0.997)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    p = os.path.join(out_dir, "augmentation_axes.png")
    plt.savefig(p, dpi=110, bbox_inches="tight")
    plt.close()
    return p


# ── Figure 2: the seven pattern types ────────────────────────────────────────
def fig_patterns(gen, out_dir, seed, per_pattern=5):
    """Collect real examples of each pattern by generating until each bucket fills."""
    buckets = {p: [] for p in PATTERN_TYPES_V2}
    i = 0
    while any(len(v) < per_pattern for v in buckets.values()) and i < 4000:
        img, vec, meta = gen.generate(index=5000 + i, base_seed=seed,
                                      probe=True, axis=None, value=None)
        pat = meta["pattern"]
        if pat in buckets and len(buckets[pat]) < per_pattern:
            top = sorted(zip(COLOR_CLASSES, vec), key=lambda x: -x[1])[:2]
            buckets[pat].append((img, meta, top))
        i += 1

    n_rows = len(PATTERN_TYPES_V2)
    fig, axes = plt.subplots(n_rows, per_pattern,
                             figsize=(2.05 * per_pattern, 2.25 * n_rows))
    for r, pat in enumerate(PATTERN_TYPES_V2):
        for c in range(per_pattern):
            ax = axes[r][c]
            if c < len(buckets[pat]):
                img, meta, top = buckets[pat][c]
                ax.imshow(rgb(img))
                lbl = "  ".join(f"{n} {v:.0%}" for n, v in top if v > 0.04)
                ax.set_title(f"{lbl}\nH={meta['label_entropy']:.2f}", fontsize=7)
            ax.axis("off")
        axes[r][0].axis("on")
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        axes[r][0].set_ylabel(pat, fontsize=10, rotation=0, ha="right",
                              va="center", labelpad=52, fontweight="bold")

    fig.suptitle("Garment pattern types\n"
                 "labels are pixel-composition derived; H is normalized label entropy",
                 fontsize=13, y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.986])
    p = os.path.join(out_dir, "patterns.png")
    plt.savefig(p, dpi=110, bbox_inches="tight")
    plt.close()
    return p


# ── Figure 3: how effects compound ───────────────────────────────────────────
def fig_stacking(gen, out_dir, seed, per_row=6):
    """Group real training samples by how many effects fired.

    Isolated axes (figure 1) are how you diagnose. This is what the model
    actually trains on: roughly 4.4 effects compounding at once.
    """
    targets = [0, 2, 4, 6, 8]
    buckets = {t: [] for t in targets}
    i = 0
    while any(len(v) < per_row for v in buckets.values()) and i < 12000:
        img, _, meta = gen.generate(index=20000 + i, base_seed=seed)
        n = int(meta["n_aug"])
        if n in buckets and len(buckets[n]) < per_row:
            buckets[n].append((img, meta))
        i += 1

    fig, axes = plt.subplots(len(targets), per_row,
                             figsize=(2.05 * per_row, 2.3 * len(targets)))
    for r, t in enumerate(targets):
        for c in range(per_row):
            ax = axes[r][c]
            if c < len(buckets[t]):
                img, meta = buckets[t][c]
                ax.imshow(rgb(img))
                fired = [k.replace("aug_", "") for k in meta
                         if k.startswith("aug_") and k != "n_aug"
                         and not (isinstance(meta[k], float) and np.isnan(meta[k]))]
                ax.set_title(", ".join(fired) or "none", fontsize=6.2)
            ax.axis("off")
        axes[r][0].axis("on")
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        axes[r][0].set_ylabel(f"{t} effects", fontsize=10, rotation=0, ha="right",
                              va="center", labelpad=46, fontweight="bold")

    fig.suptitle("Augmentations compounding\n"
                 "the training distribution averages 4.4 simultaneous effects; "
                 "captions list which fired",
                 fontsize=13, y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.984])
    p = os.path.join(out_dir, "augmentation_stacking.png")
    plt.savefig(p, dpi=110, bbox_inches="tight")
    plt.close()
    return p


# ── Figure 4: pipeline stages ────────────────────────────────────────────────
def fig_pipeline(gen, out_dir, seed, n_examples=4):
    """background -> patch -> folded patch -> composite -> augmented."""
    from utils.instrumented_augment import apply_lighting

    cols = ["1. background", "2. color + pattern", "3. + fold texture",
            "4. composited", "5. + augmentation"]
    fig, axes = plt.subplots(n_examples, len(cols),
                             figsize=(2.15 * len(cols), 2.3 * n_examples))

    for r in range(n_examples):
        s = seed + r * 977
        random.seed(s); np.random.seed(s)
        bg, _ = gen._background()

        random.seed(s); np.random.seed(s)
        patch, label_pcts = gen.inner.compose_random_color()
        random.seed(s); np.random.seed(s)
        folded = gen.inner.apply_synthetic_fold_texture(patch)

        comp = bg.copy()
        ih, iw = folded.shape[:2]
        top, left = (comp.shape[0] - ih) // 2, (comp.shape[1] - iw) // 2
        comp[top:top + ih, left:left + iw] = folded

        random.seed(s); np.random.seed(s)
        auged, meta = apply_lighting(comp.copy())

        for c, im in enumerate([bg, patch, folded, comp, auged]):
            ax = axes[r][c]
            ax.imshow(rgb(im))
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c], fontsize=9, fontweight="bold")
        top2 = sorted(label_pcts.items(), key=lambda x: -x[1])[:2]
        axes[r][4].set_xlabel(", ".join(f"{k} {v:.0f}%" for k, v in top2), fontsize=7)

    fig.suptitle("Generation pipeline, stage by stage\n"
                 "the label is computed from stage 2's pixel composition, "
                 "never from human judgement",
                 fontsize=13, y=0.999)
    plt.tight_layout(rect=[0, 0, 1, 0.982])
    p = os.path.join(out_dir, "pipeline_stages.png")
    plt.savefig(p, dpi=110, bbox_inches="tight")
    plt.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--bgs", default=DEFAULT_BGS)
    ap.add_argument("--colors", default=DEFAULT_CSV)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # Showcase figures use the probe background pool so they never display an
    # image built from a training background.
    pools = split_background_pool(args.bgs, seed=args.seed)
    gen = InstrumentedGenerator(csv_path=args.colors, path_to_bgs=args.bgs,
                                bg_pool=pools["probe"])

    for name, fn in (("axis sweep", fig_axes), ("patterns", fig_patterns),
                     ("stacking", fig_stacking), ("pipeline", fig_pipeline)):
        print(f"  building {name} ...", flush=True)
        p = fn(gen, args.out, args.seed)
        print(f"    -> {p}  ({os.path.getsize(p)/1024:.0f} KB)")

    print(f"\nShowcase figures in {args.out}")


if __name__ == "__main__":
    main()
