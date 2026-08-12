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
)

DEFAULT_OUT = os.path.join(PROJECT_ROOT, "datasets", "generated_v3")
DEFAULT_BGS = os.path.join(PROJECT_ROOT, "datasets", "indoorCVPR_09_modified")
DEFAULT_CSV = os.path.join(PROJECT_ROOT, "datasets", "categorized_colors_normalized.csv")

CSV_COLUMNS = (
    ["filename", "split"] + list(COLOR_CLASSES)
    + ["seed", "probe_axis", "probe_value"]
    + list(STRUCTURE_COLUMNS) + list(META_COLUMNS)
)

_GEN = None          # per-worker generator (built once in the initializer)
_IMAGE_DIR = None
_BASE_SEED = None
_JPEG_QUALITY = 95


def _init_worker(csv_path, bg_dir, image_dir, base_seed, jpeg_quality):
    """Build one generator per worker process.

    Windows uses spawn, so this runs fresh in each child. cv2's internal thread
    pool is disabled: with N worker processes each spawning M threads you get
    N*M runnable threads fighting over 16 cores, which is slower than N*1.
    """
    global _GEN, _IMAGE_DIR, _BASE_SEED, _JPEG_QUALITY
    cv2.setNumThreads(1)
    _GEN = InstrumentedGenerator(csv_path=csv_path, path_to_bgs=bg_dir)
    _IMAGE_DIR = image_dir
    _BASE_SEED = base_seed
    _JPEG_QUALITY = jpeg_quality


def _make_one(task):
    """Generate + write one image. Returns only its metadata row.

    The worker writes the JPEG itself; shipping 224x224x3 arrays back to the
    parent would make IPC the bottleneck.
    """
    index, filename, split, probe, axis, value = task
    img, vec, meta = _GEN.generate(
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


def build_tasks(n_train, n_val, include_probe, probe_levels, probe_per_cell,
                probe_control, existing):
    """Assemble the full work list.

    Global index is unique and stable across splits, so adding a probe set
    never perturbs the seeds of the train/val images.
    """
    tasks, idx = [], 0
    for i in range(n_train):
        fn = f"train_{i:06d}.jpg"
        if fn not in existing:
            tasks.append((idx, fn, "train", False, None, None))
        idx += 1
    for i in range(n_val):
        fn = f"val_{i:06d}.jpg"
        if fn not in existing:
            tasks.append((idx, fn, "val", False, None, None))
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
    ap.add_argument("--probe-levels", type=int, default=8)
    ap.add_argument("--probe-per-cell", type=int, default=25)
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
    include_probe = not args.no_probe

    # Resume: skip anything already recorded in labels.csv
    existing = set()
    if os.path.exists(labels_csv):
        with open(labels_csv, newline="", encoding="utf-8") as fh:
            existing = {r["filename"] for r in csv.DictReader(fh)}
        print(f"Resuming — {len(existing):,} images already recorded")

    tasks = build_tasks(n_train, n_val, include_probe, args.probe_levels,
                        args.probe_per_cell, args.probe_control, existing)
    if not tasks:
        print("Nothing to do — dataset already complete.")
        return

    print(f"  output   : {args.out}")
    print(f"  seed     : {args.seed}")
    print(f"  workers  : {args.workers}")
    print(f"  to build : {len(tasks):,} images")
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
            initargs=(args.colors, args.bgs, image_dir, args.seed, args.jpeg_quality),
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
