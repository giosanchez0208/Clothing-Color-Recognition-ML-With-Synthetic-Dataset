"""
Plot a training run from reports/training/<tag>_history.json.

Figures
-------
  <tag>_curves.png    train/val loss + probe control over epochs
  <tag>_probe.png     per-axis response curves at the final evaluation --
                      loss as a function of each augmentation parameter,
                      measured against the neutral control baseline
  <tag>_axis_trend.png how each axis's mean loss evolved during training

The probe figure is the point of the whole exercise: a flat curve means the
model handles that axis across its range, a rising curve locates *where* it
breaks, and a cliff says the training distribution never covered that region.

Usage
-----
  python scripts/plot_training.py --tag runA
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.instrumented_augment import AXES

DEFAULT_DIR = os.path.join(PROJECT_ROOT, "reports", "training")
UNIT = {"brightness": "multiplier", "temperature": "cool <-> warm", "hue": "degrees",
        "saturation": "multiplier", "shadow": "strength", "blur": "sigma",
        "noise": "density", "jpeg": "quality (lower = harsher)",
        "specular": "strength", "vignette": "strength"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="runA")
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args()

    hist_path = os.path.join(args.dir, f"{args.tag}_history.json")
    if not os.path.exists(hist_path):
        sys.exit(f"ERROR: {hist_path} not found")
    with open(hist_path, encoding="utf-8") as fh:
        h = json.load(fh)

    train, val = h["train"], h["val"]
    probe = h.get("probe", {})
    epochs = np.arange(1, len(train) + 1)
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.3})

    # ── 1. Loss curves ───────────────────────────────────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
    a1.plot(epochs, train, label="train (smoothed+weighted targets)", lw=1.6)
    a1.plot(epochs, val, label="val (raw labels)", lw=2)
    best_i = int(np.argmin(val))
    a1.scatter([best_i + 1], [val[best_i]], color="crimson", zorder=5,
               label=f"best {val[best_i]:.4f} @ epoch {best_i+1}")
    a1.set_xlabel("epoch"); a1.set_ylabel("KL divergence")
    a1.set_title("Training and validation loss")
    a1.legend(fontsize=8)
    a1.text(0.98, 0.95, "train < val is a scale artifact:\ntrain uses "
            "label-smoothed, class-weighted targets",
            transform=a1.transAxes, ha="right", va="top", fontsize=6.5,
            style="italic", color="#555")

    if probe:
        ep = sorted(probe, key=int)
        ctrl = [probe[e]["_control"] for e in ep]
        a2.plot([int(e) for e in ep], ctrl, marker="o", ms=3, lw=2,
                color="#2b8a3e", label="probe control (no augmentation)")
        a2.plot(epochs, val, lw=1.2, alpha=0.6, label="val (full distribution)")
        a2.set_xlabel("epoch"); a2.set_ylabel("KL divergence")
        a2.set_title("Clean images vs the full augmented distribution")
        a2.legend(fontsize=8)
    plt.tight_layout()
    p1 = os.path.join(args.dir, f"{args.tag}_curves.png")
    plt.savefig(p1); plt.close()
    print(f"  {p1}")

    if not probe:
        print("  (no probe evaluations recorded)")
        return

    last = sorted(probe, key=int)[-1]
    pr = probe[last]
    control = pr["_control"]

    # ── 2. Response curve per axis ───────────────────────────────────────────
    # Paired runs report deltas vs each image's own control (zero baseline);
    # older unpaired runs report absolute loss vs a shared control group.
    paired = bool(pr.get("_paired"))
    baseline = 0.0 if paired else control
    present = [a for a in AXES if a in pr and not a.endswith("__abs")]
    ncol = 5
    nrow = -(-len(present) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.7 * nrow),
                             squeeze=False)
    flat = np.ravel(axes)
    axis_mean = {a: float(np.mean(list(pr[a].values()))) for a in present}
    worst = max(axis_mean, key=axis_mean.get)

    for ax, name in zip(flat, present):
        xs = sorted(float(k) for k in pr[name])
        ys = [pr[name][str(x)] if str(x) in pr[name] else pr[name][repr(x)] for x in xs]
        is_worst = name == worst
        ax.plot(xs, ys, marker="o", ms=4, lw=2,
                color="crimson" if is_worst else "steelblue")
        ax.axhline(baseline, ls="--", lw=1, color="#2b8a3e")
        ax.set_title(f"{name}{'  (weakest)' if is_worst else ''}",
                     fontsize=9, fontweight="bold" if is_worst else "normal",
                     color="crimson" if is_worst else "black")
        ax.set_xlabel(UNIT.get(name, ""), fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in flat[len(present):]:
        ax.axis("off")

    fig.suptitle(
        f"Probe response curves, epoch {last}"
        + ("  [PAIRED]" if paired else "  [unpaired]") + "\n"
        + ("each point is loss MINUS that same image's own control "
           "(green = 0), so garment difficulty cancels exactly"
           if paired else
           f"green dashed = shared control group ({control:.3f}), drawn from different "
           f"garments, so this baseline carries a content bias"),
        fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    p2 = os.path.join(args.dir, f"{args.tag}_probe.png")
    plt.savefig(p2); plt.close()
    print(f"  {p2}")

    # ── 3. How each axis evolved ─────────────────────────────────────────────
    ep_sorted = sorted(probe, key=int)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in present:
        series = [float(np.mean(list(probe[e][name].values())))
                  for e in ep_sorted if name in probe[e]]
        xs = [int(e) for e in ep_sorted if name in probe[e]]
        ax.plot(xs, series, marker="o", ms=2.5, lw=1.8 if name == worst else 1.0,
                alpha=1.0 if name == worst else 0.55,
                label=name + (" (weakest)" if name == worst else ""))
    ax.plot([int(e) for e in ep_sorted], [probe[e]["_control"] for e in ep_sorted],
            ls="--", lw=2, color="#2b8a3e", label="control")
    ax.set_xlabel("epoch"); ax.set_ylabel("mean KL over the axis sweep")
    ax.set_title("Per-axis difficulty over training")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    p3 = os.path.join(args.dir, f"{args.tag}_axis_trend.png")
    plt.savefig(p3); plt.close()
    print(f"  {p3}")

    # ── Console summary ──────────────────────────────────────────────────────
    print(f"\n  best val      : {min(val):.4f} @ epoch {int(np.argmin(val))+1}")
    print(f"  probe control : {control:.4f}  (epoch {last})")
    kind = ("paired, vs each image's own control" if paired
            else "UNPAIRED, vs a shared control group, content-biased")
    print(f"\n  axes ranked by cost above baseline  ({kind}):")
    for name, m in sorted(axis_mean.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<12} {m - baseline:+.4f}")
    if not paired:
        n_neg = sum(1 for m in axis_mean.values() if m - baseline < 0)
        if n_neg:
            print(f"\n  WARNING: {n_neg} axes sit below baseline, which is "
                  f"impossible for a genuine control.\n"
                  f"  This run used the unpaired probe. Regenerate with the "
                  f"paired design for a trustworthy reading.")

    test_path = os.path.join(args.dir, f"{args.tag}_test.json")
    if os.path.exists(test_path):
        with open(test_path, encoding="utf-8") as fh:
            t = json.load(fh)
        print(f"\n  HELD-OUT TEST: KL {t['test_kl']:.4f} | top-1 {t['test_top1']:.1%} "
              f"| MAE {t['test_mae']:.4f}  (n={t['n_test']:,})")
    else:
        print("\n  (test set not evaluated yet, written when training finishes)")


if __name__ == "__main__":
    main()
