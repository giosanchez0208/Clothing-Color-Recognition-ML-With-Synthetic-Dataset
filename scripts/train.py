"""
Train the ResNet-50 teacher locally on the V3 instrumented dataset.

Replaces the Colab-bound notebook: no Drive auth, no gdown, no upload worker.
Reads datasets/generated_v3 directly and checkpoints to checkpoints/.

Key differences from the V2 notebook, and why
---------------------------------------------
AMP + gradient accumulation
    A 6 GB laptop GPU cannot hold ResNet-50 at batch 64 / 224px in fp32
    (~10-12 GB of activations). Batch 32 under autocast fits in ~4 GB, and
    accumulating 2 steps keeps the EFFECTIVE batch at 64 so the optimization
    trajectory stays comparable to the original run.

Trainer-controlled photometric augmentation
    V2 applied ColorJitter + RandomColorTemperature at load time on top of the
    augmentation already baked into each JPEG. Measured, that live layer
    carried ~70% of the photometric variance while only the baked layer is
    recorded in labels.csv (corr recorded-vs-actual = 0.55), so any curriculum
    steering by metadata was steering with ~30% of the wheel.

    Run A is the ablation arm with that layer removed, which quantifies its
    regularization contribution: the train/val gap widens monotonically from
    0.048 (epoch 10) to 0.264 (epoch 80).

    Both properties are wanted, and neither configuration supplies both. The
    resolution is utils/controlled_augment.ControlledPhotometric: the same live
    augmentation, but the TRAINER owns its per-axis strength rather than
    torchvision. The loop knows exactly what was applied because it set it, so
    nothing needs logging -- and the probe's weak axis maps directly onto a
    strength dial. --no-live-aug reproduces the ablation arm.

Starts from ImageNet, not V1
    The original V2 run initialized from V1's finetune_best.pth. That
    checkpoint no longer exists, so this starts from ImageNet weights. Run A's
    numbers are therefore NOT directly comparable to V2's published 0.5942 --
    which was already true, since V2's exact 22k images are unreproducible.
    Run A is the new baseline.

Test set is read exactly once
    Only at the very end, on the best checkpoint. Nothing in the loop sees it.

Usage
-----
  python scripts/train.py --smoke          # 2 epochs, verifies it fits + runs
  python scripts/train.py                  # full run
  python scripts/train.py --resume         # continue from latest checkpoint
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torchvision.transforms.functional as TF

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.controlled_augment import ControlledPhotometric, DEFAULT_STRENGTHS
from utils.instrumented_generator import COLOR_CLASSES

NUM_CLASSES = len(COLOR_CLASSES)
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "datasets", "generated_v3")
DEFAULT_CKPT = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_LOG = os.path.join(PROJECT_ROOT, "reports", "training")


# ── Logging ──────────────────────────────────────────────────────────────────
class Tee:
    """stdout + logfile, so an unattended run leaves a record."""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8", buffering=1)

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.f.write(line + "\n")

    def close(self):
        self.f.close()


# ── Data ─────────────────────────────────────────────────────────────────────
def worker_init(_):
    """Pin each dataloader worker to one OpenCV thread.

    cv2 spawns its own thread pool per process; with N workers that becomes
    N*M threads contending for 16 cores, which is slower than N*1.
    """
    import cv2
    cv2.setNumThreads(1)


class ColorDataset(Dataset):
    def __init__(self, csv_path, img_dir, split, transform=None, df=None):
        # `df` lets callers read the 28k-row CSV once and share it across the
        # four splits instead of re-parsing it per dataset.
        if df is None:
            df = pd.read_csv(csv_path, low_memory=False)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        # Labels pulled once as a contiguous array: per-row .iloc in __getitem__
        # is a measurable chunk of dataloader time at 20k images/epoch.
        self.labels = self.df[COLOR_CLASSES].to_numpy(dtype=np.float32)
        self.files = self.df["filename"].tolist()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import cv2
        img = cv2.imread(os.path.join(self.img_dir, self.files[idx]))
        if img is None:
            raise FileNotFoundError(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, torch.from_numpy(self.labels[idx])


class RandomRotate90:
    """Rotate by a random multiple of 90 degrees.

    A module-level class rather than a lambda inside build_transforms(): on
    Windows the DataLoader spawns workers, which pickles the transform, and a
    locally-defined lambda is not picklable ("Can't get local object").
    """

    def __call__(self, img):
        return TF.rotate(img, random.choice([0, 90, 180, 270]))


def build_transforms(live_strengths=None, live_p=0.5):
    """Geometric augmentation + trainer-controlled photometric augmentation.

    The photometric layer runs FIRST, on the raw RGB array, because the
    trainer sets its per-axis strengths and therefore knows exactly what was
    applied. Passing live_strengths=None disables it entirely (Run A's
    behavior), which is measurably under-regularized.
    """
    stages = []
    if live_strengths is not None:
        stages.append(ControlledPhotometric(live_strengths, p=live_p))
    train_tf = transforms.Compose(stages + [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        RandomRotate90(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def create_model(dropout=0.4):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(m.fc.in_features, NUM_CLASSES))
    return m


def get_param_groups(model, base_lr, weight_decay, log, mults=None):
    """Discriminative LRs.

    The original (0.001, 0.01, 0.1, 1.0) spread is a *fine-tuning* profile: it
    assumes the backbone is already adapted and only needs nudging. Applied to
    an ImageNet cold start it pins backbone_early at base_lr/1000 -- 5e-8 at
    the default base -- so the early layers never move at all for the whole
    run. A cold start needs a flatter profile.
    """
    m = mults or (0.001, 0.01, 0.1, 1.0)
    buckets = {
        "backbone_early": {"params": [], "lr": base_lr * m[0]},
        "layer3": {"params": [], "lr": base_lr * m[1]},
        "layer4": {"params": [], "lr": base_lr * m[2]},
        "fc": {"params": [], "lr": base_lr * m[3]},
    }
    for name, p in model.named_parameters():
        p.requires_grad = True
        key = ("fc" if "fc" in name else
               "layer4" if name.startswith("layer4") else
               "layer3" if name.startswith("layer3") else "backbone_early")
        buckets[key]["params"].append(p)
    groups = []
    for bname, b in buckets.items():
        groups.append({"params": b["params"], "lr": b["lr"], "weight_decay": weight_decay})
        log(f"    {bname:>16s}: {sum(p.numel() for p in b['params']):>12,} params  lr={b['lr']:.1e}")
    return groups


# ── Evaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, amp):
    """Mean per-sample KL against the true (unsmoothed) labels."""
    model.eval()
    total, n = 0.0, 0
    preds, labs = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(imgs)
        log_probs = F.log_softmax(logits.float(), dim=1)
        total += F.kl_div(log_probs, labels, reduction="batchmean").item() * imgs.size(0)
        n += imgs.size(0)
        preds.append(log_probs.exp().cpu())
        labs.append(labels.cpu())
    return total / max(n, 1), torch.cat(preds), torch.cat(labs)


@torch.no_grad()
def evaluate_probe(model, dataset, device, amp, batch_size, workers):
    """Per-axis mean loss on the one-factor-at-a-time probe set.

    Returns {axis: {value: mean_loss}} plus the neutral control baseline. This
    is the diagnostic that says WHICH photometric axis is weak and WHERE on it.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True,
                        worker_init_fn=worker_init if workers > 0 else None)
    losses = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(imgs)
        log_probs = F.log_softmax(logits.float(), dim=1)
        per = F.kl_div(log_probs, labels, reduction="none").sum(dim=1)
        losses.append(per.cpu())
    per_sample = torch.cat(losses).numpy()

    meta = dataset.df.copy()
    meta["_loss"] = per_sample
    ctrl_mask = meta["probe_axis"].isna() | (meta["probe_axis"].astype(str) == "")

    # PAIRED: every render sharing a seed depicts the same garment on the same
    # background, so subtracting that base's own control cancels content
    # exactly. The previous unpaired version compared against a global control
    # drawn from *different* garments, which let six genuinely-free axes finish
    # below baseline purely because the control group happened to be harder.
    base_ctrl = (meta[ctrl_mask].groupby("seed")["_loss"].mean()
                 if ctrl_mask.any() else None)
    paired = base_ctrl is not None and len(base_ctrl) > 1
    if paired:
        meta["_delta"] = meta["_loss"] - meta["seed"].map(base_ctrl)
    else:
        meta["_delta"] = meta["_loss"] - (float(per_sample[ctrl_mask.to_numpy()].mean())
                                          if ctrl_mask.any() else 0.0)

    out = {"_control": float(meta.loc[ctrl_mask, "_loss"].mean()) if ctrl_mask.any() else None,
           "_paired": bool(paired), "_n_base": int(base_ctrl.size) if paired else 0}
    for axis in sorted(set(meta.loc[~ctrl_mask, "probe_axis"].dropna())):
        sub = meta[meta["probe_axis"] == axis]
        # Values keyed as strings; the delta is the reportable quantity, but
        # absolute loss is kept so curves remain readable on their own scale.
        out[axis] = {str(v): float(g["_delta"].mean())
                     for v, g in sub.groupby("probe_value")}
        out[axis + "__abs"] = {str(v): float(g["_loss"].mean())
                               for v, g in sub.groupby("probe_value")}
    return out


class Adaptive:
    """Per-class loss weighting from validation MAE.

    Label smoothing is deliberately NOT adaptive here. V2 escalated it on
    stagnation, on the reasoning "plateau => overfitting => regularize harder".
    Three problems with that:

      * It never fired. Across Run A's 73 adjustments: 0 increases, 7
        decreases, and s sat pinned at the 0.02 floor for 90% of training.
      * No result in the literature supports degrading targets on a plateau.
        Plateau response belongs on the LR and on the inputs.
      * Smoothing is the only common regularizer that alters ground truth
        rather than the model or the input -- and these targets are measured
        pixel compositions, not estimates (Geng 2016; Singh et al. 2025).

    Smoothing is now a fixed hyperparameter, defaulting to 0. Muller,
    Kornblith & Hinton (NeurIPS 2019) additionally show a smoothed teacher
    distils worse, and that the damage is invisible in the teacher's own
    metrics -- so this default matters most downstream, in distill.py.
    """

    def __init__(self, device, label_smooth=0.0):
        self.label_smooth = label_smooth
        self.class_weights = torch.ones(NUM_CLASSES, device=device)
        self.device = device

    def recalibrate(self, val_loss, best_val_loss, epochs_no_improve, preds, labels, log):
        mae = (preds - labels).abs().mean(dim=0)
        self.class_weights = (mae / mae.mean()).clamp(0.5, 3.0).to(self.device)
        worst = torch.topk(mae, 3)
        log("    [adapt] hardest: " + ", ".join(
            f"{COLOR_CLASSES[i]}={mae[i]:.4f}" for i in worst.indices))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    ap.add_argument("--log-dir", default=DEFAULT_LOG)
    ap.add_argument("--tag", default="runA")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=2, help="effective batch = batch_size * accum")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=3,
                    help="3 matches the GPU (~230 img/s) and fits 8 GB RAM; "
                         "4+ risked ERROR_COMMITMENT_LIMIT on this machine")
    ap.add_argument("--early-stop", type=int, default=15)
    ap.add_argument("--probe-every", type=int, default=5)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-live-aug", action="store_true",
                    help="disable trainer-controlled photometric augmentation "
                         "(reproduces Run A, which under-regularized)")
    ap.add_argument("--live-p", type=float, default=0.5)
    ap.add_argument("--label-smooth", type=float, default=0.0,
                    help="fixed; 0 keeps the measured label distribution intact")
    ap.add_argument("--lr-mults", default="0.05,0.2,0.5,1.0",
                    help="per-group LR multipliers backbone_early,layer3,layer4,fc. "
                         "V2 used 0.001,0.01,0.1,1.0 -- a fine-tuning profile that "
                         "freezes early layers on a cold start")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on a small subset, to verify it runs and fits")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log = Tee(os.path.join(args.log_dir, f"{args.tag}.log"))

    csv_path = os.path.join(args.data, "labels.csv")
    img_dir = os.path.join(args.data, "images")
    for p in (csv_path, img_dir):
        if not os.path.exists(p):
            log(f"ERROR: missing {p}")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"
    log("=" * 68)
    log(f"  RUN {args.tag}   {'[SMOKE TEST]' if args.smoke else ''}")
    log("=" * 68)
    log(f"  device      : {device}" +
        (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        log(f"  VRAM        : {free/1024**3:.2f} free / {total/1024**3:.2f} GiB")
    log(f"  AMP         : {amp}")
    log(f"  batch       : {args.batch_size} x {args.accum} accum "
        f"= {args.batch_size * args.accum} effective")
    log(f"  label_smooth: {args.label_smooth}  (fixed; escalation removed)")
    log(f"  lr          : {args.lr:.1e} base, mults {args.lr_mults}")

    live = None if args.no_live_aug else dict(DEFAULT_STRENGTHS)
    train_tf, eval_tf = build_transforms(live, live_p=args.live_p)
    log(f"  live aug    : " + ("off" if live is None
        else ControlledPhotometric(live, args.live_p).describe() + f"  p={args.live_p}"))
    labels_df = pd.read_csv(csv_path, low_memory=False)
    train_ds = ColorDataset(csv_path, img_dir, "train", train_tf, df=labels_df)
    val_ds = ColorDataset(csv_path, img_dir, "val", eval_tf, df=labels_df)
    probe_ds = ColorDataset(csv_path, img_dir, "probe", eval_tf, df=labels_df)

    if args.smoke:
        keep = 400
        for ds in (train_ds, val_ds):
            ds.df = ds.df.head(keep).reset_index(drop=True)
            ds.files = ds.files[:keep]
            ds.labels = ds.labels[:keep]
        args.epochs, args.probe_every = 2, 1

    log(f"  train/val   : {len(train_ds):,} / {len(val_ds):,}   probe {len(probe_ds):,}")

    dl = dict(num_workers=args.workers, pin_memory=(device.type == "cuda"),
              persistent_workers=args.workers > 0,
              worker_init_fn=worker_init if args.workers > 0 else None,
              prefetch_factor=2 if args.workers > 0 else None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **dl)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dl)

    model = create_model().to(device)
    log(f"  params      : {sum(p.numel() for p in model.parameters()):,}")
    mults = tuple(float(x) for x in args.lr_mults.split(","))
    groups = get_param_groups(model, args.lr, args.weight_decay, log, mults)
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           patience=4, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    adapt = Adaptive(device, label_smooth=args.label_smooth)
    criterion = nn.KLDivLoss(reduction="none")

    best_path = os.path.join(args.ckpt_dir, f"{args.tag}_best.pth")
    last_path = os.path.join(args.ckpt_dir, f"{args.tag}_latest.pth")
    hist_path = os.path.join(args.log_dir, f"{args.tag}_history.json")

    start_epoch, best_val, no_improve = 0, float("inf"), 0
    history = {"train": [], "val": [], "probe": {}}
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        if ck.get("scaler_state_dict"):
            scaler.load_state_dict(ck["scaler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val = ck["best_val_loss"]
        history = ck.get("history", history)
        adapt.label_smooth = ck.get("label_smooth", adapt.label_smooth)
        log(f"  resumed from epoch {start_epoch}, best val {best_val:.4f}")

    uniform = 1.0 / NUM_CLASSES
    log("")
    t_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()

        for step, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            smoothed = (1 - adapt.label_smooth) * labels + adapt.label_smooth * uniform

            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(imgs)
            log_probs = F.log_softmax(logits.float(), dim=1)
            per_sample = criterion(log_probs, smoothed).sum(dim=1)
            w = (smoothed * adapt.class_weights.unsqueeze(0)).sum(dim=1)
            loss = (per_sample * w).mean()

            if not torch.isfinite(loss):
                log(f"    WARNING: non-finite loss at step {step}, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss / args.accum).backward()
            if (step + 1) % args.accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * imgs.size(0)
            seen += imgs.size(0)

        train_loss = running / max(seen, 1)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, device, amp)
        scheduler.step(val_loss)

        is_best = val_loss < best_val
        best_val = min(best_val, val_loss)
        no_improve = 0 if is_best else no_improve + 1
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        dt = time.perf_counter() - t0
        vram = (f"  vram {torch.cuda.max_memory_allocated()/1024**3:.2f}G"
                if device.type == "cuda" else "")
        log(f"  {epoch+1:>3}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f}"
            f" | lr {optimizer.param_groups[-1]['lr']:.1e} | {dt/60:.1f}m{vram}"
            f" | no-improve {no_improve}{' *BEST*' if is_best else ''}")

        adapt.recalibrate(val_loss, best_val, no_improve, val_preds, val_labels, log)

        if (epoch + 1) % args.probe_every == 0 or epoch == args.epochs - 1:
            pr = evaluate_probe(model, probe_ds, device, amp, args.batch_size, 0)
            history["probe"][str(epoch + 1)] = pr
            axis_means = {a: float(np.mean(list(v.values())))
                          for a, v in pr.items()
                          if isinstance(v, dict) and not a.endswith("__abs")}
            worst = sorted(axis_means.items(), key=lambda kv: -kv[1])[:3]
            tag_p = "paired" if pr.get("_paired") else "UNPAIRED"
            log(f"    [probe] control {pr['_control']:.4f} ({tag_p}) | "
                f"worst delta: " + ", ".join(f"{a}={v:+.4f}" for a, v in worst))

        ck = {"epoch": epoch, "model_state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "scheduler_state_dict": scheduler.state_dict(),
              "scaler_state_dict": scaler.state_dict() if amp else None,
              "best_val_loss": best_val, "history": history,
              "label_smooth": adapt.label_smooth,
              "architecture": "resnet50", "color_classes": COLOR_CLASSES}
        torch.save(ck, last_path)
        if is_best:
            torch.save(ck, best_path)
        with open(hist_path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=1)

        if no_improve >= args.early_stop:
            log(f"\n  early stop at epoch {epoch+1} ({args.early_stop} without improvement)")
            break

    log(f"\n  training done in {(time.perf_counter()-t_start)/60:.1f} min")
    log(f"  best val loss: {best_val:.4f}")

    # ── Test set: read exactly once, on the best checkpoint ──────────────────
    log("\n" + "=" * 68)
    log("  HELD-OUT TEST  (first and only read)")
    log("=" * 68)
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device,
                                         weights_only=False)["model_state_dict"])
    test_ds = ColorDataset(csv_path, img_dir, "test", eval_tf, df=labels_df)
    if args.smoke:
        test_ds.df = test_ds.df.head(200).reset_index(drop=True)
        test_ds.files, test_ds.labels = test_ds.files[:200], test_ds.labels[:200]
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **dl)
    test_loss, tp, tl = evaluate(model, test_loader, device, amp)
    top1 = (tp.argmax(1) == tl.argmax(1)).float().mean().item()
    mae = (tp - tl).abs().mean().item()

    log(f"  n            : {len(test_ds):,}")
    log(f"  KL loss      : {test_loss:.4f}")
    log(f"  top-1        : {top1:.1%}")
    log(f"  MAE          : {mae:.4f}")
    log(f"  (val was {best_val:.4f}. Val is optimized against, test is not)")

    with open(os.path.join(args.log_dir, f"{args.tag}_test.json"), "w", encoding="utf-8") as fh:
        json.dump({"test_kl": test_loss, "test_top1": top1, "test_mae": mae,
                   "best_val_kl": best_val, "n_test": len(test_ds)}, fh, indent=1)
    log("\nDone.")
    log.close()


if __name__ == "__main__":
    main()
