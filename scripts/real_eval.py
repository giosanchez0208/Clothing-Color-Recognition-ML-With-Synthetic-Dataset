"""
Sim2real evaluation: score the model on real photographs you annotate yourself.

Two stages.

  --prepare   ingest photos, run the real inference pipeline (pose -> torso ->
              compose_input), save the exact 224x224 composite the model will
              see, and write a blank annotation CSV
  --score     read the completed CSV, run the model, report metrics

Why annotate the CROP and not the photo
---------------------------------------
The model never sees your photograph. It sees a 112x112 torso crop pasted onto
a wider context crop. A black polo with a white undershirt at the collar may be
95% black in that crop or 60% black, depending entirely on where the torso box
landed. Annotating the original photo would score the model against something it
was never shown. --prepare saves the composite so ground truth and input match.

Why not annotate proportions
----------------------------
The synthetic labels are exact pixel fractions because the generator computed
them. A human cannot produce that, and asking for "70/30" invents precision that
does not exist. So the schema asks only for things a person can judge reliably:

  dominant         the single most-covering colour   (one of the 13)
  colors_present   every colour covering >= ~15% of the visible garment
  freeform         what you would naturally call it -- "cyan", "navy", "teal"

`freeform` exists to separate two very different failures: the model being
wrong, versus the 13-category taxonomy being unable to express the answer at
all. A cyan-and-orange Hawaiian shirt has no cyan category; forcing it to "blue"
and then scoring the model as correct or incorrect conflates those. Recording
both lets you report them separately.

Anti-anchoring
--------------
--prepare deliberately does NOT write the model's prediction into the annotation
file. Seeing the model's answer before annotating biases the label toward it.
Predictions are only joined at --score time.

Usage
-----
  python scripts/real_eval.py --prepare --photos my_photos/
  # ... fill in datasets/real/annotations.csv by hand ...
  python scripts/real_eval.py --score --ckpt checkpoints/distilB_student_int8.pth
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.instrumented_generator import COLOR_CLASSES
from scripts.predict import (Pose, TorchBackend, centre_box, compose_input,
                             report, softmax)

DEFAULT_DIR = os.path.join(PROJECT_ROOT, "datasets", "real")
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

COLUMNS = ["filename", "crop_file", "pose_found",
           "dominant", "colors_present", "freeform", "notes"]


def prepare(photo_dir, out_dir, use_pose=True):
    crops = os.path.join(out_dir, "crops")
    os.makedirs(crops, exist_ok=True)
    photos = sorted(p for p in glob.glob(os.path.join(photo_dir, "*"))
                    if p.lower().endswith(VALID_EXT))
    if not photos:
        sys.exit(f"ERROR: no images in {photo_dir}")

    pose = Pose() if use_pose else None
    rows, n_pose = [], 0
    for i, p in enumerate(photos):
        frame = cv2.imread(p)
        if frame is None:
            print(f"  unreadable, skipped: {p}")
            continue
        boxes = pose(frame) if pose else []
        found = bool(boxes)
        n_pose += found
        torso = boxes[0][1] if found else centre_box(frame)
        _, comp = compose_input(frame, torso)

        name = f"{i:03d}_{os.path.splitext(os.path.basename(p))[0]}.jpg"
        cv2.imwrite(os.path.join(crops, name), comp)
        rows.append({"filename": os.path.basename(p), "crop_file": name,
                     "pose_found": int(found), "dominant": "",
                     "colors_present": "", "freeform": "", "notes": ""})

    csv_path = os.path.join(out_dir, "annotations.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n  photos ingested : {len(rows)}")
    print(f"  torso found     : {n_pose}/{len(rows)}"
          + ("  (rest fell back to a centre crop)" if n_pose < len(rows) else ""))
    print(f"  crops           : {crops}")
    print(f"  annotate here   : {csv_path}")
    print(f"\n  Annotate the CROPS, not the originals — that is what the model sees.")
    print(f"  Valid colours   : {', '.join(COLOR_CLASSES)}")
    print("""
  dominant       one colour, the one covering the most garment area
  colors_present semicolon-separated, every colour covering >= ~15%
                 e.g.  black;white
  freeform       what you'd naturally say, even if it is not in the 13
                 e.g.  cyan   (this is how taxonomy gaps get measured)
  notes          anything odd: heavy shadow, motion blur, tiny torso""")


def parse_set(s):
    return {c.strip().lower() for c in str(s).split(";") if c.strip()}


def score(out_dir, ckpt, use_pose=True, topk=3):
    csv_path = os.path.join(out_dir, "annotations.csv")
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: {csv_path} not found — run --prepare first")
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)]

    done = [r for r in rows if r["dominant"].strip()]
    if not done:
        sys.exit("ERROR: no annotated rows — fill in `dominant` at minimum")
    print(f"  annotated: {len(done)}/{len(rows)}")

    valid = set(COLOR_CLASSES)
    bad = {r["dominant"].strip().lower() for r in done} - valid
    if bad:
        sys.exit(f"ERROR: `dominant` values outside the 13 categories: {sorted(bad)}")

    backend = TorchBackend(ckpt)
    print(f"  model    : {backend.arch}\n")

    crops = os.path.join(out_dir, "crops")
    n_top1 = n_exact = 0
    inter = union = pred_n = true_n = 0
    cover_hits = cover_tot = 0
    taxonomy_gap = 0
    detail = []

    for r in done:
        img = cv2.imread(os.path.join(crops, r["crop_file"]))
        if img is None:
            continue
        # The crop is already composed; feed it straight through.
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
        batch = ((chw - mean) / std)[None].astype(np.float32)

        probs = softmax(backend(batch)[0])
        picks, ambiguous = report(probs)
        pred_set = {c for c, _ in picks}
        pred_top1 = COLOR_CLASSES[int(np.argmax(probs))]
        topk_set = {COLOR_CLASSES[i] for i in np.argsort(probs)[::-1][:topk]}

        true_top1 = r["dominant"].strip().lower()
        true_set = parse_set(r["colors_present"]) or {true_top1}
        free = r["freeform"].strip().lower()
        if free and free not in valid:
            taxonomy_gap += 1

        n_top1 += pred_top1 == true_top1
        n_exact += pred_set == true_set
        inter += len(pred_set & true_set)
        union += len(pred_set | true_set)
        pred_n += len(pred_set)
        true_n += len(true_set)
        cover_hits += len(true_set & topk_set)
        cover_tot += len(true_set)

        detail.append({
            "crop": r["crop_file"], "true_top1": true_top1,
            "true_set": ";".join(sorted(true_set)), "pred_top1": pred_top1,
            "pred_set": ";".join(sorted(pred_set)),
            "top1_ok": pred_top1 == true_top1, "set_ok": pred_set == true_set,
            "ambiguous": ambiguous, "freeform": free,
            "pose_found": r.get("pose_found", ""),
        })

    n = len(detail)
    prec = inter / max(pred_n, 1)
    rec = inter / max(true_n, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    se = (n_top1 / n * (1 - n_top1 / n) / n) ** 0.5

    print("=" * 62)
    print(f"  SIM2REAL — n = {n}")
    print("=" * 62)
    print(f"  top-1 (dominant colour)   {n_top1/n:6.1%}   +/- {1.96*se:.1%} (95% CI)")
    print(f"  exact set match           {n_exact/n:6.1%}")
    print(f"  set Jaccard               {inter/max(union,1):6.1%}")
    print(f"  set precision / recall    {prec:6.1%} / {rec:.1%}   F1 {f1:.1%}")
    print(f"  coverage @top-{topk}          {cover_hits/max(cover_tot,1):6.1%}"
          f"   (human colours appearing in the model's top {topk})")
    if taxonomy_gap:
        print(f"\n  taxonomy gaps             {taxonomy_gap}/{n} "
              f"({taxonomy_gap/n:.0%}) had a freeform name outside the 13 categories")
    print(f"\n  For reference, synthetic held-out top-1 is 67.4%.")
    if n < 100:
        print(f"  NOTE: n={n} gives a +/-{1.96*se:.0%} confidence interval. This "
              f"distinguishes\n        'roughly working' from 'broken'; it cannot "
              f"resolve small differences.")

    print("\n  failures:")
    for d in detail:
        if not d["top1_ok"]:
            print(f"    {d['crop']:<34} true {d['true_set']:<20} "
                  f"pred {d['pred_set']:<24}"
                  + ("  [uncertain]" if d["ambiguous"] else ""))

    out = os.path.join(out_dir, "real_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(detail[0].keys()))
        w.writeheader()
        w.writerows(detail)
    print(f"\n  per-image results: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--photos", default=os.path.join(DEFAULT_DIR, "photos"))
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--ckpt", default=os.path.join(PROJECT_ROOT, "checkpoints",
                                                   "distilB_student_int8.pth"))
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    if args.prepare:
        prepare(args.photos, args.dir, use_pose=not args.no_pose)
    elif args.score:
        score(args.dir, args.ckpt, use_pose=not args.no_pose, topk=args.topk)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
