"""
Distil the ResNet-50 teacher into MobileNetV3-Small, then quantise to INT8.

Produces the deployable artifact: ~4 MB, CPU-real-time, exported to ONNX.

Why this script is also the label-smoothing experiment
-----------------------------------------------------
Muller, Kornblith & Hinton (NeurIPS 2019, arXiv:1906.02629) found that a
teacher trained WITH label smoothing distils worse -- smoothing tightens
same-class clusters and erases the inter-class logit structure a student needs.

Critically, they also found smoothing does NOT hurt the teacher's own
generalisation or calibration. So comparing teacher val loss across smoothing
settings shows nothing; the effect is only visible in the STUDENT. That makes
this script, not train.py, the place where the smoothing decision gets settled.

Lukasik et al. (ICML 2020, arXiv:2003.02819) supply the counterweight: under
label noise, smoothing the teacher *helps* distillation. Whether that applies
here depends on how much the generative-vs-perceptual gap counts as noise, so
the comparison is run rather than argued.

Loss
----
    L = alpha * T^2 * KL(student/T || teacher/T)  +  (1 - alpha) * KL(student || y)

The T^2 factor restores the gradient magnitude that dividing logits by T
removes (Hinton et al., arXiv:1503.02531).

Usage
-----
  python scripts/distill.py --teacher checkpoints/runA2_best.pth
  python scripts/distill.py --teacher ... --epochs 60 --tag distilA2
  python scripts/distill.py --teacher ... --smoke
  python scripts/distill.py --tag distilA2 --eval-only     # quantise + benchmark
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
from torchvision import models

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.controlled_augment import DEFAULT_STRENGTHS
from utils.instrumented_generator import COLOR_CLASSES
from scripts.train import (ColorDataset, Tee, build_transforms, create_model,
                           evaluate, worker_init)

NUM_CLASSES = len(COLOR_CLASSES)
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "datasets", "generated_v3")
DEFAULT_CKPT = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_LOG = os.path.join(PROJECT_ROOT, "reports", "training")


def create_student(num_classes=NUM_CLASSES, dropout=0.2):
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    for i, layer in enumerate(m.classifier):
        if isinstance(layer, nn.Dropout):
            m.classifier[i] = nn.Dropout(p=dropout)
    return m


def distillation_loss(s_logits, t_logits, labels, T, alpha):
    soft = F.kl_div(F.log_softmax(s_logits / T, dim=1),
                    F.softmax(t_logits / T, dim=1),
                    reduction="batchmean") * (T * T)
    hard = F.kl_div(F.log_softmax(s_logits, dim=1), labels, reduction="batchmean")
    return alpha * soft + (1 - alpha) * hard, soft.item(), hard.item()


def model_size_mb(model, tmp):
    torch.save(model.state_dict(), tmp)
    mb = os.path.getsize(tmp) / 1024 / 1024
    os.remove(tmp)
    return mb


@torch.no_grad()
def benchmark(model, device="cpu", n=60, warmup=10):
    model.eval().to(device)
    x = torch.randn(1, 3, 224, 224, device=device)
    for _ in range(warmup):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


@torch.no_grad()
def evaluate_cpu(model, loader):
    """Evaluate on CPU — required for dynamically quantised models."""
    model.eval()
    tot, n, preds, labs = 0.0, 0, [], []
    for imgs, labels in loader:
        log_probs = F.log_softmax(model(imgs).float(), dim=1)
        tot += F.kl_div(log_probs, labels, reduction="batchmean").item() * imgs.size(0)
        n += imgs.size(0)
        preds.append(log_probs.exp())
        labs.append(labels)
    p, l = torch.cat(preds), torch.cat(labs)
    return (tot / max(n, 1),
            (p.argmax(1) == l.argmax(1)).float().mean().item(),
            (p - l).abs().mean().item())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default=os.path.join(DEFAULT_CKPT, "runA2_best.pth"))
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--tag", default="distil")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--label-smooth", type=float, default=0.0,
                    help="applied to the HARD-target term only; 0 keeps the "
                         "measured label distribution intact")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--early-stop", type=int, default=12)
    ap.add_argument("--no-live-aug", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; quantise + benchmark an existing student")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(DEFAULT_CKPT, exist_ok=True)
    os.makedirs(DEFAULT_LOG, exist_ok=True)
    log = Tee(os.path.join(DEFAULT_LOG, f"{args.tag}.log"))

    csv_path = os.path.join(args.data, "labels.csv")
    img_dir = os.path.join(args.data, "images")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"

    log("=" * 68)
    log(f"  DISTIL {args.tag}   {'[SMOKE]' if args.smoke else ''}")
    log("=" * 68)
    log(f"  teacher     : {os.path.basename(args.teacher)}")
    log(f"  device      : {device}   AMP {amp}")
    log(f"  T={args.temperature}  alpha={args.alpha}  label_smooth={args.label_smooth}")

    live = None if args.no_live_aug else dict(DEFAULT_STRENGTHS)
    train_tf, eval_tf = build_transforms(live)
    labels_df = pd.read_csv(csv_path, low_memory=False)
    train_ds = ColorDataset(csv_path, img_dir, "train", train_tf, df=labels_df)
    val_ds = ColorDataset(csv_path, img_dir, "val", eval_tf, df=labels_df)
    test_ds = ColorDataset(csv_path, img_dir, "test", eval_tf, df=labels_df)

    if args.smoke:
        for ds in (train_ds, val_ds, test_ds):
            ds.df = ds.df.head(200).reset_index(drop=True)
            ds.files, ds.labels = ds.files[:200], ds.labels[:200]
        args.epochs = 2

    dl = dict(num_workers=args.workers, pin_memory=(device.type == "cuda"),
              persistent_workers=args.workers > 0,
              worker_init_fn=worker_init if args.workers > 0 else None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **dl)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dl)

    # ── Teacher (frozen) ─────────────────────────────────────────────────────
    if not os.path.exists(args.teacher):
        log(f"ERROR: teacher checkpoint not found: {args.teacher}")
        sys.exit(1)
    t_ck = torch.load(args.teacher, map_location=device, weights_only=False)
    teacher = create_model()
    teacher.load_state_dict(t_ck["model_state_dict"])
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    log(f"  teacher val : {t_ck.get('best_val_loss', float('nan')):.4f} "
        f"(epoch {t_ck.get('epoch', -1) + 1})")

    student = create_student().to(device)
    n_t = sum(p.numel() for p in teacher.parameters())
    n_s = sum(p.numel() for p in student.parameters())
    log(f"  params      : teacher {n_t:,} | student {n_s:,} "
        f"({n_t / n_s:.1f}x compression)")
    log(f"  train/val   : {len(train_ds):,} / {len(val_ds):,}   test {len(test_ds):,}")

    best_path = os.path.join(DEFAULT_CKPT, f"{args.tag}_student_best.pth")
    last_path = os.path.join(DEFAULT_CKPT, f"{args.tag}_student_latest.pth")

    if not args.eval_only:
        opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                           eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        start, best_val, no_imp = 0, float("inf"), 0
        history = {"train": [], "val": [], "soft": [], "hard": []}

        if args.resume and os.path.exists(last_path):
            ck = torch.load(last_path, map_location=device, weights_only=False)
            student.load_state_dict(ck["model_state_dict"])
            opt.load_state_dict(ck["optimizer_state_dict"])
            sched.load_state_dict(ck["scheduler_state_dict"])
            start, best_val = ck["epoch"] + 1, ck["best_val_loss"]
            history = ck.get("history", history)
            log(f"  resumed from epoch {start}, best {best_val:.4f}")

        uniform = 1.0 / NUM_CLASSES
        log("")
        for epoch in range(start, args.epochs):
            student.train()
            run, run_s, run_h, seen = 0.0, 0.0, 0.0, 0
            t0 = time.perf_counter()
            for imgs, labels in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if args.label_smooth > 0:
                    labels = (1 - args.label_smooth) * labels + args.label_smooth * uniform

                with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
                    t_logits = teacher(imgs)
                with torch.amp.autocast("cuda", enabled=amp):
                    s_logits = student(imgs)

                loss, sv, hv = distillation_loss(
                    s_logits.float(), t_logits.float(), labels,
                    args.temperature, args.alpha)
                if not torch.isfinite(loss):
                    log("    WARNING: non-finite loss, skipping batch")
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                run += loss.item() * imgs.size(0)
                run_s += sv * imgs.size(0)
                run_h += hv * imgs.size(0)
                seen += imgs.size(0)

            sched.step()
            tr = run / max(seen, 1)
            val_loss, _, _ = evaluate(student, val_loader, device, amp)
            is_best = val_loss < best_val
            best_val = min(best_val, val_loss)
            no_imp = 0 if is_best else no_imp + 1
            history["train"].append(tr)
            history["val"].append(val_loss)
            history["soft"].append(run_s / max(seen, 1))
            history["hard"].append(run_h / max(seen, 1))

            log(f"  {epoch+1:>2}/{args.epochs} | train {tr:.4f} "
                f"(soft {run_s/max(seen,1):.4f} hard {run_h/max(seen,1):.4f}) "
                f"| val {val_loss:.4f} | lr {opt.param_groups[0]['lr']:.1e} "
                f"| {(time.perf_counter()-t0)/60:.1f}m"
                f"{' *BEST*' if is_best else f'  no-improve {no_imp}'}")

            ck = {"epoch": epoch, "model_state_dict": student.state_dict(),
                  "optimizer_state_dict": opt.state_dict(),
                  "scheduler_state_dict": sched.state_dict(),
                  "best_val_loss": best_val, "history": history,
                  "architecture": "mobilenet_v3_small",
                  "color_classes": COLOR_CLASSES,
                  "teacher": os.path.basename(args.teacher),
                  "label_smooth": args.label_smooth}
            torch.save(ck, last_path)
            if is_best:
                torch.save(ck, best_path)
            with open(os.path.join(DEFAULT_LOG, f"{args.tag}_history.json"),
                      "w", encoding="utf-8") as fh:
                json.dump(history, fh, indent=1)

            if no_imp >= args.early_stop:
                log(f"\n  early stop at epoch {epoch+1}")
                break
        log(f"\n  best student val: {best_val:.4f}")

    # ── Quantise, benchmark, and read the held-out test ONCE ─────────────────
    log("\n" + "=" * 68)
    log("  COMPRESSION + HELD-OUT TEST")
    log("=" * 68)

    # Free the GPU before doing CPU inference. Training leaves the teacher and
    # student resident on CUDA; holding that context while running CPU forwards
    # through spawned dataloader workers is the one path train.py never takes,
    # and it faults on Windows.
    try:
        del teacher, student
    except NameError:
        pass
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    student_cpu = create_student()
    if os.path.exists(best_path):
        student_cpu.load_state_dict(torch.load(best_path, map_location="cpu",
                                               weights_only=False)["model_state_dict"])
    student_cpu.eval()
    # nn.Linear only: dynamic quantisation has never supported Conv2d, so
    # including it was a no-op (verified -- identical 5.95 -> 4.23 MB).
    int8 = torch.quantization.quantize_dynamic(
        student_cpu, {nn.Linear}, dtype=torch.qint8)

    teacher_cpu = create_model()
    teacher_cpu.load_state_dict(t_ck["model_state_dict"])
    teacher_cpu.eval()

    tmp = os.path.join(DEFAULT_CKPT, "_tmp.pth")
    sizes = {"teacher": model_size_mb(teacher_cpu, tmp),
             "student_fp32": model_size_mb(student_cpu, tmp),
             "student_int8": model_size_mb(int8, tmp)}

    # Artifacts first: a fault in the CPU benchmark must not cost the models.
    for name, m in (("student_fp32", student_cpu), ("student_int8", int8)):
        torch.save({"model_state_dict": m.state_dict(),
                    "architecture": ("mobilenet_v3_small_int8" if "int8" in name
                                     else "mobilenet_v3_small"),
                    "color_classes": COLOR_CLASSES,
                    "teacher": os.path.basename(args.teacher)},
                   os.path.join(DEFAULT_CKPT, f"{args.tag}_{name}.pth"))
    onnx_path = os.path.join(DEFAULT_CKPT, f"{args.tag}_student.onnx")
    torch.onnx.export(student_cpu, torch.randn(1, 3, 224, 224), onnx_path,
                      input_names=["image"], output_names=["color_logits"],
                      dynamic_axes={"image": {0: "batch"},
                                    "color_logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    log(f"  exported {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")

    # num_workers=0: no subprocess spawn while a CUDA context has just been torn down
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)
    log("  evaluating on held-out test (CPU) ...")
    res = {}
    for name, m in (("teacher", teacher_cpu), ("student_fp32", student_cpu),
                    ("student_int8", int8)):
        try:
            kl, top1, mae = evaluate_cpu(m, test_loader)
            ms = benchmark(m)
            res[name] = {"kl": kl, "top1": top1, "mae": mae,
                         "size_mb": sizes[name], "cpu_ms": ms}
        except Exception as exc:
            log(f"  {name}: FAILED ({type(exc).__name__}: {exc})")
            res[name] = {"kl": float("nan"), "top1": float("nan"),
                         "mae": float("nan"), "size_mb": sizes[name],
                         "cpu_ms": float("nan")}

    log(f"\n  {'model':<16}{'KL':>9}{'top-1':>9}{'MAE':>9}{'MB':>8}{'ms':>8}")
    log("  " + "-" * 58)
    for name, r in res.items():
        log(f"  {name:<16}{r['kl']:>9.4f}{r['top1']:>8.1%}{r['mae']:>9.4f}"
            f"{r['size_mb']:>8.1f}{r['cpu_ms']:>8.1f}")
    log(f"\n  size reduction : {sizes['teacher']/sizes['student_int8']:.1f}x")
    log(f"  speedup (CPU)  : {res['teacher']['cpu_ms']/res['student_int8']['cpu_ms']:.1f}x")

    out = {"teacher_ckpt": os.path.basename(args.teacher),
           "label_smooth": args.label_smooth, "results": res}
    with open(os.path.join(DEFAULT_LOG, f"{args.tag}_test.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    log("\nDone.")
    log.close()


if __name__ == "__main__":
    main()
