"""
Barebones tester for the clothing-colour pipeline.

Runs the full inference path on a real image or a live camera feed, with
either the PyTorch checkpoint or an exported ONNX model, so both backends
can be sanity-checked against each other.

Pipeline (matches how the training data was built)
--------------------------------------------------
  1. locate a torso            YOLO pose keypoints, or a centre crop with --no-pose
  2. compose_input             112x112 tight torso crop pasted onto a 224x224
                               wider context crop -- this reproduces the
                               inner-square-on-background layout the model was
                               trained on. Feeding a raw photo instead would be
                               a distribution mismatch.
  3. predict                   13-way softmax
  4. report                    every colour above max(0.08, top * 0.35), so
                               patterned garments can return several

Usage
-----
  python scripts/predict.py --export-onnx
  python scripts/predict.py --image photo.jpg
  python scripts/predict.py --image photo.jpg --backend onnx
  python scripts/predict.py --image photo.jpg --no-pose        # skip ultralytics
  python scripts/predict.py --camera
  python scripts/predict.py --compare photo.jpg               # torch vs onnx
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.instrumented_generator import COLOR_CLASSES

NUM_CLASSES = len(COLOR_CLASSES)
IMG_SIZE, INNER = 224, 112
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

DEFAULT_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "runA_best.pth")
DEFAULT_ONNX = os.path.join(PROJECT_ROOT, "checkpoints", "runA_best.onnx")

SIG_RATIO, SIG_FLOOR, AMBIGUOUS = 0.35, 0.08, 0.25

BGR = {"red": (0, 0, 255), "orange": (0, 165, 255), "yellow": (0, 255, 255),
       "green": (0, 180, 0), "blue": (255, 0, 0), "violet": (226, 43, 138),
       "purple": (128, 0, 128), "white": (255, 255, 255), "gray": (150, 150, 150),
       "black": (40, 40, 40), "pink": (180, 105, 255), "brown": (19, 69, 139),
       "olive": (0, 128, 128)}


# ── Preprocessing ────────────────────────────────────────────────────────────
def _clamp_square(cx, cy, half, w, h):
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    side = min(x2 - x1, y2 - y1)
    x1 = max(0, min(x1, w - side)); y1 = max(0, min(y1, h - side))
    return int(x1), int(y1), int(x1 + side), int(y1 + side)


def compose_input(frame_bgr, box):
    """Rebuild the training layout: tight torso crop centred on wider context."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    bw, bh = x2 - x1, y2 - y1

    inner_side = max(min(bw, bh), 10)
    ix1, iy1, ix2, iy2 = _clamp_square(cx, cy, inner_side // 2, w, h)
    inner = cv2.resize(frame_bgr[iy1:iy2, ix1:ix2], (INNER, INNER),
                       interpolation=cv2.INTER_AREA)

    outer_side = max(int(max(bw, bh) * 2.0), inner_side + 20)
    ox1, oy1, ox2, oy2 = _clamp_square(cx, cy, outer_side // 2, w, h)
    comp = cv2.resize(frame_bgr[oy1:oy2, ox1:ox2], (IMG_SIZE, IMG_SIZE),
                      interpolation=cv2.INTER_AREA)

    off = (IMG_SIZE - INNER) // 2
    comp[off:off + INNER, off:off + INNER] = inner

    rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return ((chw - MEAN) / STD)[None, ...].astype(np.float32), comp


def centre_box(frame):
    """Fallback torso guess when pose detection is unavailable."""
    h, w = frame.shape[:2]
    s = int(min(h, w) * 0.5)
    return (w // 2 - s // 2, h // 2 - s // 2, w // 2 + s // 2, h // 2 + s // 2)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def report(probs):
    order = np.argsort(probs)[::-1]
    top = probs[order[0]]
    thresh = max(SIG_FLOOR, top * SIG_RATIO)
    picks = [(COLOR_CLASSES[i], float(probs[i])) for i in order if probs[i] >= thresh]
    return picks, bool(top < AMBIGUOUS)


# ── Backends ─────────────────────────────────────────────────────────────────
class TorchBackend:
    name = "torch"

    def __init__(self, ckpt_path, device=None):
        import torch
        from scripts.train import create_model
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model = create_model()
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval().to(self.device)
        self.epoch = ck.get("epoch", -1) + 1
        self.val = ck.get("best_val_loss")

    def __call__(self, batch):
        t = self.torch.from_numpy(batch).to(self.device)
        with self.torch.inference_mode():
            return self.model(t).float().cpu().numpy()


class OnnxBackend:
    name = "onnx"

    def __init__(self, onnx_path):
        import onnxruntime as ort
        if not os.path.exists(onnx_path):
            sys.exit(f"ERROR: {onnx_path} not found — run --export-onnx first")
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.epoch = self.val = None

    def __call__(self, batch):
        return self.sess.run(None, {self.input: batch})[0]


def export_onnx(ckpt_path, out_path):
    import torch
    from scripts.train import create_model
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = create_model()
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(model, dummy, out_path,
                      input_names=["image"], output_names=["color_logits"],
                      dynamic_axes={"image": {0: "batch"}, "color_logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  exported {out_path}  ({mb:.1f} MB)")

    # An export that loads but disagrees with the source model is worse than a
    # failed export, so verify numerically rather than trusting the file.
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    x = np.random.randn(2, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    with torch.inference_mode():
        ref = model(torch.from_numpy(x)).numpy()
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    d = float(np.abs(ref - got).max())
    print(f"  max |torch - onnx| on random input: {d:.3e}  "
          f"{'OK' if d < 1e-3 else 'MISMATCH'}")
    return d < 1e-3


# ── Pose ─────────────────────────────────────────────────────────────────────
class Pose:
    """YOLO pose -> torso box. Optional; --no-pose falls back to a centre crop.

    Kept optional deliberately: ultralytics is AGPL-3.0 and is the only
    copyleft dependency in the project (see NOTICE.md).
    """

    def __init__(self, conf=0.5, imgsz=384):
        from ultralytics import YOLO
        self.m = YOLO("yolo11n-pose.pt")
        self.conf, self.imgsz = conf, imgsz

    def __call__(self, frame):
        r = self.m(frame, conf=self.conf, verbose=False, imgsz=self.imgsz)
        if not r or r[0].boxes is None or len(r[0].boxes) == 0:
            return []
        res, kp = r[0], r[0].keypoints
        out = []
        for i in range(len(res.boxes)):
            b = res.boxes.xyxy[i]
            px1, py1, px2, py2 = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            pts = []
            if kp is not None and kp.xy is not None and len(kp.xy) > i:
                conf = kp.conf[i] if kp.conf is not None else None
                for idx in (5, 6, 11, 12):          # shoulders, hips (COCO)
                    if conf is not None and float(conf[idx]) < 0.4:
                        continue
                    x, y = float(kp.xy[i][idx][0]), float(kp.xy[i][idx][1])
                    if x or y:
                        pts.append((x, y))
            if len(pts) >= 2:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                tb = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            else:
                ph, pw = py2 - py1, px2 - px1
                tb = (px1 + pw // 10, int(py1 + 0.15 * ph),
                      px2 - pw // 10, int(py1 + 0.65 * ph))
            if tb[2] - tb[0] > 10 and tb[3] - tb[1] > 10:
                out.append(((px1, py1, px2, py2), tb))
        return out


def annotate(frame, person, picks, ambiguous):
    x1, y1, x2, y2 = person
    col = (120, 120, 120) if (ambiguous or not picks) else BGR.get(picks[0][0], (0, 255, 0))
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
    text = ("uncertain" if ambiguous else
            " / ".join(f"{c} {p:.0%}" for c, p in picks[:3]))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty = max(y1 - 8, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + base), col, cv2.FILLED)
    lum = 0.299 * col[2] + 0.587 * col[1] + 0.114 * col[0]
    cv2.putText(frame, text, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0) if lum > 128 else (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def run_frame(frame, backend, pose):
    boxes = pose(frame) if pose else []
    if not boxes:
        cb = centre_box(frame)
        boxes = [(cb, cb)]
    results = []
    for person, torso in boxes:
        batch, comp = compose_input(frame, torso)
        t0 = time.perf_counter()
        logits = backend(batch)
        ms = (time.perf_counter() - t0) * 1000
        probs = softmax(logits[0])
        picks, amb = report(probs)
        annotate(frame, person, picks, amb)
        results.append((picks, amb, probs, ms, comp))
    return frame, results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image")
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--compare", metavar="IMAGE", help="run torch and onnx on one image")
    ap.add_argument("--export-onnx", action="store_true")
    ap.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    ap.add_argument("--no-pose", action="store_true", help="skip ultralytics (AGPL)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--out", default=None, help="where to write the annotated image")
    args = ap.parse_args()

    if args.export_onnx:
        sys.exit(0 if export_onnx(args.ckpt, args.onnx) else 1)

    if not any([args.image, args.camera, args.compare]):
        ap.print_help()
        sys.exit(0)

    def make(name):
        return TorchBackend(args.ckpt) if name == "torch" else OnnxBackend(args.onnx)

    # ── compare mode ─────────────────────────────────────────────────────────
    if args.compare:
        frame = cv2.imread(args.compare)
        if frame is None:
            sys.exit(f"ERROR: cannot read {args.compare}")
        pose = None if args.no_pose else Pose()
        print(f"\n{'':<8}{'top-3':<44}{'ms':>8}")
        store = {}
        for name in ("torch", "onnx"):
            b = make(name)
            _, res = run_frame(frame.copy(), b, pose)
            picks, amb, probs, ms, _ = res[0]
            store[name] = probs
            top3 = ", ".join(f"{c} {p:.1%}" for c, p in picks[:3]) or "-"
            print(f"{name:<8}{top3:<44}{ms:>8.1f}")
        d = float(np.abs(store["torch"] - store["onnx"]).max())
        same_top1 = int(np.argmax(store["torch"])) == int(np.argmax(store["onnx"]))
        # Tolerance is in PROBABILITY space, where fp32 accumulation differences
        # get amplified by softmax. Logit-level agreement is ~1e-6; anything
        # under ~5e-3 here is numerically identical for practical purposes.
        print(f"\n  max |torch - onnx| probability: {d:.3e}")
        print(f"  same top-1 prediction         : {same_top1}")
        print(f"  verdict                       : "
              f"{'AGREE' if (d < 5e-3 and same_top1) else 'DISAGREE'}")
        return

    backend = make(args.backend)
    pose = None if args.no_pose else Pose()
    if getattr(backend, "epoch", None):
        print(f"  checkpoint: epoch {backend.epoch}, best val {backend.val:.4f}")
    print(f"  backend   : {backend.name}   pose: {'off (centre crop)' if pose is None else 'yolo11n'}")

    # ── single image ─────────────────────────────────────────────────────────
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"ERROR: cannot read {args.image}")
        out, results = run_frame(frame, backend, pose)
        for i, (picks, amb, probs, ms, comp) in enumerate(results):
            print(f"\n  person {i+1}  ({ms:.1f} ms){'  [uncertain]' if amb else ''}")
            for c, p in picks:
                print(f"    {c:<8} {p:6.1%}  {'#' * int(p * 40)}")
            rest = sorted(zip(COLOR_CLASSES, probs), key=lambda x: -x[1])[len(picks):len(picks)+3]
            print("    " + "  ".join(f"{c} {p:.1%}" for c, p in rest) + "   (below threshold)")
        dest = args.out or os.path.splitext(args.image)[0] + "_pred.jpg"
        cv2.imwrite(dest, out)
        print(f"\n  annotated -> {dest}")
        return

    # ── camera ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open camera {args.camera_index}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("  press q to quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame, _ = run_frame(frame, backend, pose)
            cv2.imshow("clothing colour", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty("clothing colour", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
