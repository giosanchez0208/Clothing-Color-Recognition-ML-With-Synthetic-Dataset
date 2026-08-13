"""
Generate the instrumented synthetic dataset (train + val + probe).

Three outputs, all under datasets/generated_v3/:

  images/        the JPEGs
  labels.csv     one row per image: soft label + every generation parameter
  probe/         one-factor-at-a-time diagnostic set (see --probe-only)

Why the metadata matters
------------------------
The soft label says what the model *should* predict. The metadata says what
was *done to the image* to make it hard. Only the second one lets you ask
"which photometric axis is the model failing on" — the question that drives
the whole curriculum. See docs/curriculum-design.md.

Determinism
-----------
Every image is seeded from (base_seed, global_index), so output is identical
whether generated on 1 worker or 16, and any single image can be regenerated
in isolation from its `seed` column. V2 was unseeded, which is why its
original 22k images are unrecoverable.

Usage
-----
  python scripts/generate_dataset.py                      # train+val+probe
  python scripts/generate_dataset.py --n-train 2000 --n-val 200
  python scripts/generate_dataset.py --probe-only
  python scripts/generate_dataset.py --workers 4
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.instrumented_augment import AXES, META_COLUMNS
from utils.instrumented_generator import (
    COLOR_CLASSES, InstrumentedGenerator, STRUCTURE_COLUMNS, probe_plan,
    split_background_pool,
)

DEFAULT_OUT = os.path.join(PROJECT_ROOT, "datasets", "generated_v3")
DEFAULT_BGS = os.path.join(PROJECT_ROOT, "datasets", "indoorCVPR_09_modified")
DEFAULT_CSV = os.path.join(PROJECT_ROOT, "datasets", "categorized_colors_normalized.csv")

CSV_COLUMNS = (
    ["filename", "split"] + list(COLOR_CLASSES)
    + ["seed", "probe_axis", "probe_value"]
    + list(STRUCTURE_COLUMNS) + list(META_COLUMNS)
)

_GENS = {}           # split -> generator, each with its own background pool
_IMAGE_DIR = None
_BASE_SEED = None
_JPEG_QUALITY = 95


def _init_worker(csv_path, bg_dir, image_dir, base_seed, jpeg_quality, pools):
    """Build the generators once per worker process.

    One generator per split, each restricted to its own DISJOINT background
    pool, so no background is ever seen by two splits. Sampling with
    replacement from a shared pool guarantees overlap, and since the model sees
    the whole frame rather than just the garment patch, a shared background is
    a memorisation path between splits.

    Windows uses spawn, so this runs fresh in each child. cv2's thread pool is
    disabled: N worker processes each spawning M threads gives N*M runnable
    threads contending for 16 cores, which is slower than N*1.
    """
    global _GENS, _IMAGE_DIR, _BASE_SEED, _JPEG_QUALITY
    cv2.setNumThreads(1)
    _GENS = {
        split: InstrumentedGenerator(csv_path=csv_path, path_to_bgs=bg_dir,
                                     bg_pool=pool)
        for split, pool in pools.items()
    }
    _IMAGE_DIR = image_dir
    _BASE_SEED = base_seed
    _JPEG_QUALITY = jpeg_quality


def _make_one(task):
    """Generate + write one image. Returns only its metadata row.

    The worker writes the JPEG itself; shipping 224x224x3 arrays back to the
    parent would make IPC the bottleneck.
    """
    index, filename, split, probe, axis, value = task
    gen = _GENS[split]
    img, vec, meta = gen.generate(
        index=index, base_seed=_BASE_SEED, probe=probe, axis=axis, value=value,
    )
    cv2.imwrite(
        os.path.join(_IMAGE_DIR, filename), img,
        [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
    )
    row = {"filename": filename, "split": split}
    row.update({c: f"{v:.6f}" for c, v in zip(COLOR_CLASSES, vec)})
    for col in ["seed", "probe_axis", "probe_value"] + list(STRUCTURE_COLUMNS) + list(META_COLUMNS):
        row[col] = meta.get(col, "")
    return row


def build_tasks(n_train, n_val, n_test, include_probe, probe_levels,
                probe_per_cell, probe_control, existing):
    """Assemble the full work list.

    Global index is unique and stable across splits, so adding a probe or test
    set never perturbs the seeds of the train/val images.
    """
    tasks, idx = [], 0
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        for i in range(count):
            fn = f"{split}_{i:06d}.jpg"
            if fn not in existing:
                tasks.append((idx, fn, split, False, None, None))
            idx += 1

    if include_probe:
        plan = probe_plan(axes=AXES, n_levels=probe_levels,
                          per_cell=probe_per_cell, n_control=probe_control)
        for i, (axis, value) in enumerate(plan):
            fn = f"probe_{i:06d}.jpg"
            if fn not in existing:
                tasks.append((idx, fn, "probe", True, axis, value))
            idx += 1
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000,
                    help="held-out split; nothing in the training loop may touch it")
    ap.add_argument("--probe-levels", type=int, default=8)
    ap.add_argument("--probe-per-cell", type=int, default=50)
    ap.add_argument("--probe-control", type=int, default=200)
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--jpeg-quality", type=int, default=95)

    ap.add_argument("--workers", type=int,
                    default=max(1, min(8, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--bgs", default=DEFAULT_BGS)
    ap.add_argument("--colors", default=DEFAULT_CSV)
    args = ap.parse_args()

    image_dir = os.path.join(args.out, "images")
    labels_csv = os.path.join(args.out, "labels.csv")
    os.makedirs(image_dir, exist_ok=True)

    for path, what in ((args.bgs, "backgrounds"), (args.colors, "colour library")):
        if not os.path.exists(path):
            sys.exit(f"ERROR: {what} not found at {path}")

    n_train = 0 if args.probe_only else args.n_train
    n_val = 0 if args.probe_only else args.n_val
    n_test = 0 if args.probe_only else args.n_test
    include_probe = not args.no_probe

    # Resume: skip anything already recorded in labels.csv
    existing = set()
    if os.path.exists(labels_csv):
        with open(labels_csv, newline="", encoding="utf-8") as fh:
            existing = {r["filename"] for r in csv.DictReader(fh)}
        print(f"Resuming — {len(existing):,} images already recorded")

    tasks = build_tasks(n_train, n_val, n_test, include_probe,
                        args.probe_levels, args.probe_per_cell,
                        args.probe_control, existing)
    if not tasks:
        print("Nothing to do — dataset already complete.")
        return

    pools = split_background_pool(args.bgs, seed=args.seed)

    print(f"  output   : {args.out}")
    print(f"  seed     : {args.seed}")
    print(f"  workers  : {args.workers}")
    print(f"  to build : {len(tasks):,} images")
    print("  bg pools : " + "  ".join(f"{k}={len(v):,}" for k, v in pools.items())
          + "   (mutually disjoint)")
    print()

    write_header = not existing
    t0 = time.perf_counter()
    done = 0

    with open(labels_csv, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.colors, args.bgs, image_dir, args.seed, args.jpeg_quality,
                      pools),
        ) as pool:
            for row in pool.imap_unordered(_make_one, tasks, chunksize=32):
                writer.writerow(row)
                done += 1
                if done % 250 == 0 or done == len(tasks):
                    el = time.perf_counter() - t0
                    rate = done / el
                    eta = (len(tasks) - done) / rate if rate else 0
                    pct = 100 * done / len(tasks)
                    print(f"\r  {done:>6,}/{len(tasks):,} ({pct:5.1f}%)  "
                          f"{rate:6.1f} img/s  ETA {eta/60:5.1f} min",
                          end="", flush=True)
                    fh.flush()

    el = time.perf_counter() - t0
    print(f"\n\nDone in {el/60:.1f} min ({done/el:.1f} img/s)")
    print(f"  labels : {labels_csv}")
    print(f"  images : {image_dir}")


if __name__ == "__main__":
    main()
