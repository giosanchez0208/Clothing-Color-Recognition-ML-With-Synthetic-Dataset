"""
Flatten and colour-normalise the IndoorCVPR_09 backgrounds.

  datasets/indoorCVPR_09/Images/<category>/*.jpg
      -> datasets/indoorCVPR_09_modified/*.jpg      (flat, normalised)

Normalisation (utils.dataset_utils.color_balance_image):
  linearise -> Shades-of-Gray white balance (p=6) -> LAB exposure -> CLAHE

Why normalise at all: the garment patch is composited onto these backgrounds,
and the whole point of the dataset is that colour is the signal. A background
carrying a strong colour cast pushes the composite's apparent colour around,
so the label stops matching what the image looks like. Normalising first gives
a daylight-neutral canvas, and the augmentation stage then re-applies colour
casts *with known parameters* — which is exactly the difference between noise
and a measurable variable.

Differences from the original notebook:
  * parallel (the notebook looped serially over ~15.6k images)
  * writes to a separate output dir instead of overwriting in place, so a
    failed run cannot destroy the source images
  * skips already-processed files, so it is resumable

Usage
-----
  python scripts/prepare_backgrounds.py
  python scripts/prepare_backgrounds.py --workers 4 --limit 500
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_SRC = os.path.join(PROJECT_ROOT, "datasets", "indoorCVPR_09", "Images")
DEFAULT_DST = os.path.join(PROJECT_ROOT, "datasets", "indoorCVPR_09_modified")
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

MIN_SIDE = 224  # composite target; anything smaller cannot be cropped to size


def _init_worker():
    cv2.setNumThreads(1)


def _process(task):
    """Normalise one image. Returns (status, name)."""
    src, dst = task
    try:
        img = cv2.imread(src)
        if img is None:
            return "unreadable", os.path.basename(src)
        h, w = img.shape[:2]
        if min(h, w) < MIN_SIDE:
            return "too_small", os.path.basename(src)

        from utils.dataset_utils import color_balance_image
        out = color_balance_image(img)

        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        if not cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            return "write_failed", os.path.basename(src)
        return "ok", os.path.basename(src)
    except Exception as exc:
        return f"error:{type(exc).__name__}", os.path.basename(src)


def build_tasks(src_root, dst_root, limit=None):
    """Flatten category subdirs into unique output names.

    Filenames repeat across categories, so collisions are disambiguated with
    the category prefix — matching the original notebook's behaviour.
    """
    os.makedirs(dst_root, exist_ok=True)
    tasks, seen = [], set()
    for category in sorted(os.listdir(src_root)):
        cat_dir = os.path.join(src_root, category)
        if not os.path.isdir(cat_dir):
            continue
        for fn in sorted(os.listdir(cat_dir)):
            if not fn.lower().endswith(VALID_EXT):
                continue
            name = fn if fn not in seen else f"{category}_{fn}"
            if name in seen:
                continue
            seen.add(name)
            out_name = os.path.splitext(name)[0] + ".jpg"
            dst = os.path.join(dst_root, out_name)
            if not os.path.exists(dst):
                tasks.append((os.path.join(cat_dir, fn), dst))
            if limit and len(tasks) >= limit:
                return tasks
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int,
                    default=max(1, min(8, (os.cpu_count() or 4) - 2)))
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        sys.exit(f"ERROR: source not found: {args.src}")

    tasks = build_tasks(args.src, args.dst, args.limit)
    already = len([f for f in os.listdir(args.dst)]) if os.path.isdir(args.dst) else 0
    if not tasks:
        print(f"Nothing to do — {already:,} backgrounds already prepared.")
        return

    print(f"  src     : {args.src}")
    print(f"  dst     : {args.dst}")
    print(f"  workers : {args.workers}")
    print(f"  to do   : {len(tasks):,}" + (f"  (already have {already:,})" if already else ""))
    print()

    counts, t0, done = {}, time.perf_counter(), 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=_init_worker) as pool:
        for status, _name in pool.imap_unordered(_process, tasks, chunksize=16):
            counts[status] = counts.get(status, 0) + 1
            done += 1
            if done % 200 == 0 or done == len(tasks):
                el = time.perf_counter() - t0
                rate = done / el
                eta = (len(tasks) - done) / rate if rate else 0
                print(f"\r  {done:>6,}/{len(tasks):,} ({100*done/len(tasks):5.1f}%)  "
                      f"{rate:6.1f} img/s  ETA {eta/60:5.1f} min", end="", flush=True)

    el = time.perf_counter() - t0
    print(f"\n\nDone in {el/60:.1f} min")
    for status in sorted(counts, key=lambda k: -counts[k]):
        print(f"  {status:<16} {counts[status]:>7,}")

    final = len(os.listdir(args.dst))
    print(f"\n  usable backgrounds: {final:,}")
    if final < 5000:
        print("  WARNING: fewer backgrounds than expected — check the source tree.")


if __name__ == "__main__":
    main()
