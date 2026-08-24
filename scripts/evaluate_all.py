"""
Cross-version evaluation: every trained model on identical data.

Consistency guarantees
----------------------
* Every model is scored on the **complete** held-out test split (all 2,000
  images) -- no subsampling, so there is no sample-selection difference between
  versions to explain away.
* One shared `labels.csv` parse, one dataset object, `shuffle=False`, so every
  model sees the same images in the same order.
* The paired probe (all 4,860 renders) is likewise run in full.
* Where a figure needs a subsample, it is drawn with a fixed seed.

Ordering note: quantized models are CPU-only, and running CPU inference while a
CUDA context is live faults on Windows (see docs/results.md). GPU models are
therefore scored first, the GPU is released, and CPU models follow.

Outputs
-------
  reports/eval/comparison.json      every metric, machine-readable
  reports/eval/comparison.md        the tables
  reports/eval/per_class_mae.png
  reports/eval/confusion_<tag>.png
  reports/eval/probe_axes.png

Usage
-----
  python scripts/evaluate_all.py
  python scripts/evaluate_all.py --skip-latency     # faster
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.instrumented_augment import AXES
from utils.instrumented_generator import COLOR_CLASSES
from scripts.train import ColorDataset, build_transforms, create_model, evaluate_probe
from scripts.distill import create_student

NUM_CLASSES = len(COLOR_CLASSES)
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "datasets", "generated_v3")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "eval")

# (tag, checkpoint, human label). Order defines table/figure order.
REGISTRY = [
    ("runA",    "runA_best.pth",             "Run A: ablation arm (no live aug, LR x0.001)"),
    ("runA2",   "runA2_best.pth",            "Run A2: live aug restored (stopped @15, NOT converged)"),
    ("runB",    "runB_best.pth",             "Run B: flat LR, smoothing 0"),
    ("student", "distilB_student_fp32.pth",  "Student FP32: distilled from Run B"),
    ("int8",    "distilB_student_int8.pth",  "Student INT8: deployable"),
    # V4: colors sampled from the Gaussian mixture, labels are the posterior.
    ("runV4",     "runV4_best.pth",            "Run V4: mixture labels, teacher"),
    ("studentV4", "distilV4_student_fp32.pth", "Student V4 FP32: distilled from Run V4"),
    ("int8V4",    "distilV4_student_int8.pth", "Student V4 INT8: deployable"),
]


def load_model(path):
    """Rebuild whatever the checkpoint declares. Returns (model, arch, cpu_only)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("architecture", "resnet50")
    quantized = "int8" in arch
    if "mobilenet" in arch:
        m = create_student().eval()
        if quantized:
            m = torch.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
    else:
        m = create_model()
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, arch, quantized


@torch.no_grad()
def score(model, loader, device):
    """Full-split scoring. Returns metrics plus raw preds/labels for later plots."""
    model.to(device).eval()
    tot, n, preds, labs = 0.0, 0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        lp = F.log_softmax(model(imgs).float(), dim=1)
        tot += F.kl_div(lp, labels, reduction="batchmean").item() * imgs.size(0)
        n += imgs.size(0)
        preds.append(lp.exp().cpu())
        labs.append(labels.cpu())
    p, l = torch.cat(preds), torch.cat(labs)
    return {
        "kl": tot / max(n, 1),
        "top1": (p.argmax(1) == l.argmax(1)).float().mean().item(),
        "mae": (p - l).abs().mean().item(),
        "per_class_mae": (p - l).abs().mean(0).tolist(),
        "n": int(n),
    }, p, l


@torch.no_grad()
def latency(model, n=40, warmup=8):
    model.cpu().eval()
    x = torch.randn(1, 3, 224, 224)
    for _ in range(warmup):
        model(x)
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    return (time.perf_counter() - t0) / n * 1000


def size_mb(model):
    tmp = os.path.join(OUT_DIR, "_t.pth")
    torch.save(model.state_dict(), tmp)
    mb = os.path.getsize(tmp) / 1024 / 1024
    os.remove(tmp)
    return mb


def soft_confusion(preds, labels):
    """Row = true class, column = mean predicted probability."""
    true = labels.argmax(1)
    C = torch.zeros(NUM_CLASSES, NUM_CLASSES)
    for c in range(NUM_CLASSES):
        m = true == c
        if m.sum():
            C[c] = preds[m].mean(0)
    return C.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    csv_path = os.path.join(args.data, "labels.csv")
    img_dir = os.path.join(args.data, "images")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # One parse, one dataset, fixed order, shared by every model.
    _, eval_tf = build_transforms(None)
    df = pd.read_csv(csv_path, low_memory=False)
    test_ds = ColorDataset(csv_path, img_dir, "test", eval_tf, df=df)
    probe_ds = ColorDataset(csv_path, img_dir, "probe", eval_tf, df=df)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)

    print(f"  test split : {len(test_ds):,} images (FULL, no subsampling)")
    print(f"  probe split: {len(probe_ds):,} images")
    print(f"  device     : {device}\n")

    available = [(t, os.path.join(CKPT_DIR, f), d) for t, f, d in REGISTRY
                 if os.path.exists(os.path.join(CKPT_DIR, f))]
    for t, f, _ in REGISTRY:
        if not os.path.exists(os.path.join(CKPT_DIR, f)):
            print(f"  (skipping {t}: {f} not found)")

    results, preds_by_tag = {}, {}

    # GPU-capable models first, then release CUDA before touching CPU-only ones.
    for cpu_only_phase in (False, True):
        if cpu_only_phase and device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        for tag, path, desc in available:
            model, arch, quantized = load_model(path)
            if quantized != cpu_only_phase:
                continue
            dev = torch.device("cpu") if quantized else device
            t0 = time.perf_counter()
            m, p, l = score(model, test_loader, dev)
            m["arch"] = arch
            m["desc"] = desc
            m["size_mb"] = size_mb(model)
            # Probe BEFORE latency: latency() moves the model to CPU, which would
            # leave a CUDA-bound probe pass with a CPU model and mismatched types.
            if not args.skip_probe:
                pr = evaluate_probe(model, probe_ds, dev, False, args.batch_size, 0)
                m["probe"] = {a: float(np.mean(list(pr[a].values())))
                              for a in AXES if a in pr}
                m["probe_control"] = pr["_control"]
            m["cpu_ms"] = None if args.skip_latency else latency(model)
            results[tag] = m
            preds_by_tag[tag] = (p, l)
            print(f"  {tag:<9} KL {m['kl']:.4f}  top1 {m['top1']:6.1%}  "
                  f"MAE {m['mae']:.4f}  {m['size_mb']:5.1f} MB"
                  + (f"  {m['cpu_ms']:5.1f} ms" if m["cpu_ms"] else "")
                  + f"   [{time.perf_counter()-t0:.0f}s]")
            del model

    order = [t for t, _, _ in REGISTRY if t in results]

    # ── Report ───────────────────────────────────────────────────────────────
    L = []
    L.append("# Cross-Version Evaluation\n")
    L.append(f"All models scored on the **same {len(test_ds):,}-image held-out test "
             f"split**, complete rather than subsampled, in identical order.\n")
    L.append("## Headline\n")
    L.append("| Model | KL | top-1 | MAE | Size | CPU ms |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for t in order:
        r = results[t]
        L.append(f"| {r['desc']} | {r['kl']:.4f} | {r['top1']:.1%} | {r['mae']:.4f} "
                 f"| {r['size_mb']:.1f} MB | "
                 f"{r['cpu_ms']:.1f} |" if r["cpu_ms"] else
                 f"| {r['desc']} | {r['kl']:.4f} | {r['top1']:.1%} | {r['mae']:.4f} "
                 f"| {r['size_mb']:.1f} MB | n/a |")

    L.append("\n## Per-class MAE\n")
    L.append("| Color | " + " | ".join(order) + " |")
    L.append("|---" * (len(order) + 1) + "|")
    for i, c in enumerate(COLOR_CLASSES):
        L.append(f"| {c} | " + " | ".join(f"{results[t]['per_class_mae'][i]:.4f}"
                                          for t in order) + " |")

    if not args.skip_probe:
        L.append("\n## Probe: paired cost per axis\n")
        L.append("| Axis | " + " | ".join(order) + " |")
        L.append("|---" * (len(order) + 1) + "|")
        for a in AXES:
            L.append(f"| {a} | " + " | ".join(
                f"{results[t]['probe'].get(a, float('nan')):+.4f}" for t in order) + " |")

    with open(os.path.join(args.out, "comparison.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(args.out, "comparison.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)

    # ── Figures ──────────────────────────────────────────────────────────────
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.3})

    x = np.arange(NUM_CLASSES)
    w = 0.8 / len(order)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for i, t in enumerate(order):
        ax.bar(x + i * w - 0.4 + w / 2, results[t]["per_class_mae"], w, label=t)
    ax.set_xticks(x); ax.set_xticklabels(COLOR_CLASSES, rotation=45, ha="right")
    ax.set_ylabel("MAE"); ax.set_title("Per-class MAE on an identical test split")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "per_class_mae.png")); plt.close()

    if not args.skip_probe:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        xa = np.arange(len(AXES))
        for i, t in enumerate(order):
            ax.bar(xa + i * w - 0.4 + w / 2,
                   [results[t]["probe"].get(a, 0) for a in AXES], w, label=t)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(xa); ax.set_xticklabels(AXES, rotation=45, ha="right")
        ax.set_ylabel("paired cost vs own control")
        ax.set_title("Per-axis robustness on an identical probe set")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "probe_axes.png")); plt.close()

    for t in order:
        p, l = preds_by_tag[t]
        C = soft_confusion(p, l)
        fig, ax = plt.subplots(figsize=(7.5, 6))
        im = ax.imshow(C, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(COLOR_CLASSES, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(COLOR_CLASSES, fontsize=7)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if C[i, j] > 0.06:
                    ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                            fontsize=5.5, color="black" if C[i, j] < 0.6 else "white")
        ax.set_xlabel("predicted probability →"); ax.set_ylabel("true class ↓")
        ax.set_title(f"Soft confusion, {t}  (KL {results[t]['kl']:.4f})")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"confusion_{t}.png")); plt.close()

    print(f"\n  wrote {args.out}")
    best = min(order, key=lambda t: results[t]["kl"])
    print(f"  best KL: {best} ({results[best]['kl']:.4f}, "
          f"top-1 {results[best]['top1']:.1%})")


if __name__ == "__main__":
    main()
