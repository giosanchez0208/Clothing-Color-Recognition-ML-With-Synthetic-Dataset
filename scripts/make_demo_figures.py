"""
Build the README demo figures from real photographs.

  documentation/figures/pipeline_demo.png   one photo, three panels:
                                            input -> what the model actually
                                            sees -> prediction
  documentation/figures/real_results.png    a row per sample, same treatment,
                                            with the full 13-way distribution

The middle panel matters: the model never sees the photograph. It sees a
112x112 torso crop pasted onto a wider context crop, which reproduces the
layout the synthetic training data was built in. Showing the photo alone would
misrepresent what was classified.

Usage
-----
  python scripts/make_demo_figures.py
  python scripts/make_demo_figures.py --ckpt checkpoints/distilB_student_int8.pth
"""
import argparse
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

from utils.instrumented_generator import COLOR_CLASSES
from scripts.predict import (BGR, Pose, TorchBackend, annotate, center_box,
                             compose_input, report, softmax)

IMG_DIR = os.path.join(PROJECT_ROOT, "documentation", "images")
OUT_DIR = os.path.join(PROJECT_ROOT, "documentation", "figures")

# Ground truth as a person would describe it, used only for honest captions.
TRUTH = {
    "sample0.png": "heather grey tee",
    "sample1.jpg": "teal + white paisley",
    "sample2.jpg": "navy + white stripes",
    "sample3.jpg": "red varsity, white sleeves, grey hood",
    "sample4.jpg": "pale green sweater",
}

# matplotlib swatch colors for the probability bars
SWATCH = {c: tuple(v / 255 for v in BGR[c][::-1]) for c in COLOR_CLASSES}


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run(path, backend, pose):
    frame = cv2.imread(path)
    boxes = pose(frame) if pose else []
    person, torso = boxes[0] if boxes else (center_box(frame),) * 2
    batch, comp = compose_input(frame, torso)
    probs = softmax(backend(batch)[0])
    picks, ambiguous = report(probs)
    annotated = annotate(frame.copy(), person, picks, ambiguous)
    return frame, comp, annotated, probs, picks, bool(boxes)


def bars(ax, probs, picks):
    """Full 13-way distribution, each bar drawn in the color it represents."""
    order = np.argsort(probs)[::-1]
    names = [COLOR_CLASSES[i] for i in order]
    vals = [probs[i] for i in order]
    reported = {c for c, _ in picks}
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=[SWATCH[n] for n in names],
            edgecolor=["black" if n in reported else "#bbbbbb" for n in names],
            linewidth=[1.4 if n in reported else 0.5 for n in names])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  {v:.0%}" if n in reported else n
                        for n, v in zip(names, vals)], fontsize=7)
    for lbl, n in zip(ax.get_yticklabels(), names):
        lbl.set_fontweight("bold" if n in reported else "normal")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50%", "100%"], fontsize=7)
    ax.grid(axis="x", alpha=0.3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(PROJECT_ROOT, "checkpoints",
                                                   "distilB_student_int8.pth"))
    ap.add_argument("--no-pose", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    backend = TorchBackend(args.ckpt)
    pose = None if args.no_pose else Pose()
    print(f"  model: {backend.arch}")

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

    # ── Figure 1: the pipeline, on one photo ─────────────────────────────────
    p0 = os.path.join(IMG_DIR, "sample0.png")
    frame, comp, annotated, probs, picks, found = run(p0, backend, pose)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5))
    axes[0].imshow(rgb(frame)); axes[0].set_title("1. input photograph", fontsize=10)
    axes[1].imshow(rgb(comp))
    axes[1].set_title("2. what the model actually sees\n"
                      "torso crop on wider context, 224x224", fontsize=10)
    axes[2].imshow(rgb(annotated)); axes[2].set_title("3. prediction", fontsize=10)
    for a in axes:
        a.axis("off")
    top = ", ".join(f"{c} {v:.0%}" for c, v in picks)
    fig.suptitle(f"{TRUTH['sample0.png']}   ->   {top}", fontsize=11, y=0.03)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out1 = os.path.join(OUT_DIR, "pipeline_demo.png")
    plt.savefig(out1, bbox_inches="tight"); plt.close()
    print(f"  {out1}")

    # ── Figure 2: real-world results ─────────────────────────────────────────
    samples = ["sample1.jpg", "sample2.jpg", "sample3.jpg", "sample4.jpg"]
    fig, axes = plt.subplots(len(samples), 3, figsize=(11, 4.1 * len(samples)),
                             gridspec_kw={"width_ratios": [1.15, 1, 1.35]})
    for r, name in enumerate(samples):
        path = os.path.join(IMG_DIR, name)
        if not os.path.exists(path):
            continue
        frame, comp, annotated, probs, picks, found = run(path, backend, pose)
        axes[r][0].imshow(rgb(frame)); axes[r][0].axis("off")
        axes[r][0].set_title(TRUTH.get(name, name), fontsize=9, loc="left")
        axes[r][1].imshow(rgb(comp)); axes[r][1].axis("off")
        axes[r][1].set_title("model input" + ("" if found else "  (no pose, center crop)"),
                             fontsize=8)
        bars(axes[r][2], probs, picks)
        axes[r][2].set_title("predicted distribution", fontsize=9)
    fig.suptitle("Real photographs. Trained on 100% synthetic data, "
                 "never saw a real garment", fontsize=12, y=0.997)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out2 = os.path.join(OUT_DIR, "real_results.png")
    plt.savefig(out2, bbox_inches="tight"); plt.close()
    print(f"  {out2}")


if __name__ == "__main__":
    main()
