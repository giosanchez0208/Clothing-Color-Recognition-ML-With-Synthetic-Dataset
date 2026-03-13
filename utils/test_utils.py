import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from ultralytics import YOLO
import os

# ── Constants ─────────────────────────────────────────────────────────────────
COLOR_CLASSES = [
    "red", "orange", "yellow", "green", "blue",
    "violet", "purple", "white", "gray", "black",
    "pink", "brown", "olive",
]
NUM_CLASSES = len(COLOR_CLASSES)
IMG_SIZE = 224
INNER_SIZE = IMG_SIZE // 2  # 112

# Pre-computed normalisation tensors (avoids re-creating every frame)
_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_NORM_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

# YOLO pose keypoint indices (COCO format)
_LEFT_SHOULDER  = 5
_RIGHT_SHOULDER = 6
_LEFT_HIP       = 11
_RIGHT_HIP      = 12
_TORSO_INDICES  = (_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP)
_KP_CONF_THRESH = 0.4


# ── Model helpers ─────────────────────────────────────────────────────────────
def _create_resnet50(num_classes=NUM_CLASSES):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def _create_mobilenetv3_small(num_classes=NUM_CLASSES, dropout=0.2):
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model.classifier[0] = nn.Linear(
        model.classifier[0].in_features,
        model.classifier[0].out_features,
    )
    for i, layer in enumerate(model.classifier):
        if isinstance(layer, nn.Dropout):
            model.classifier[i] = nn.Dropout(p=dropout)
    return model


def load_model(checkpoint_path, device=None):
    """Load a trained model from a checkpoint file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt.get("architecture", "resnet50")

    if "mobilenet" in arch:
        if "int8" in arch:
            # Quantized model: build FP32 first, then quantize, then load
            model = _create_mobilenetv3_small()
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8,
            )
        else:
            model = _create_mobilenetv3_small()
    else:
        model = _create_resnet50()

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model = model.to(device)
    return model, device


def find_best_checkpoint(models_dir):
    """Return path to the best available checkpoint in models_dir."""
    for name in ("student_int8.pth", "student_fp32.pth", "finetune_best.pth", "best.pth"):
        path = os.path.join(models_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No checkpoint found in {models_dir}")


# ── Torso estimation ─────────────────────────────────────────────────────────
def _estimate_torso(ls, rs, lh, rh, person_box):
    px1, py1, px2, py2 = person_box
    ph = py2 - py1
    pw = px2 - px1

    shoulder_pts = [p for p in (ls, rs) if p is not None]
    hip_pts      = [p for p in (lh, rh) if p is not None]
    all_pts      = shoulder_pts + hip_pts

    if len(all_pts) >= 2:
        xs = [p[0] for p in all_pts]
        tx1, tx2 = int(min(xs)), int(max(xs))
    else:
        inset = int(pw * 0.10)
        tx1, tx2 = px1 + inset, px2 - inset

    if shoulder_pts:
        ty1 = int(min(p[1] for p in shoulder_pts))
    elif hip_pts:
        ty1 = int(min(p[1] for p in hip_pts) - 0.40 * ph)
    else:
        ty1 = int(py1 + 0.15 * ph)

    if hip_pts:
        ty2 = int(max(p[1] for p in hip_pts))
    elif shoulder_pts:
        ty2 = int(max(p[1] for p in shoulder_pts) + 0.40 * ph)
    else:
        ty2 = int(py1 + 0.65 * ph)

    tx1, tx2 = max(tx1, px1), min(tx2, px2)
    ty1, ty2 = max(ty1, py1), min(ty2, py2)

    if tx2 - tx1 < 10 or ty2 - ty1 < 10:
        inset = int(pw * 0.10)
        return (px1 + inset, int(py1 + 0.15 * ph), px2 - inset, int(py1 + 0.65 * ph))

    return tx1, ty1, tx2, ty2


# ── Pose detection ────────────────────────────────────────────────────────────
class PoseDetector:
    def __init__(self, confidence=0.5, imgsz=384):
        self._model = YOLO("yolo11n-pose.pt")
        self._conf = confidence
        self._imgsz = imgsz

    def detect(self, frame_bgr):
        """
        Run YOLO pose on a BGR frame.

        Returns
        -------
        detections : list[tuple]  Each element is (person_box, torso_box)
        """
        results = self._model(
            frame_bgr, conf=self._conf, verbose=False, imgsz=self._imgsz,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        r = results[0]
        boxes_xyxy = r.boxes.xyxy
        has_kpts = (r.keypoints is not None and r.keypoints.xy is not None)
        kpts_xy   = r.keypoints.xy   if has_kpts else None
        kpts_conf = r.keypoints.conf if (has_kpts and r.keypoints.conf is not None) else None

        detections = []
        for i in range(len(r.boxes)):
            bxy = boxes_xyxy[i]
            person_box = (int(bxy[0]), int(bxy[1]), int(bxy[2]), int(bxy[3]))

            ls = rs = lh = rh = None
            if kpts_xy is not None and len(kpts_xy) > i:
                xy = kpts_xy[i]
                conf = kpts_conf[i] if kpts_conf is not None else None
                for kp_idx, slot in zip(_TORSO_INDICES, range(4)):
                    if conf is not None and float(conf[kp_idx]) < _KP_CONF_THRESH:
                        continue
                    x, y = float(xy[kp_idx][0]), float(xy[kp_idx][1])
                    if x == 0.0 and y == 0.0:
                        continue
                    pt = (x, y)
                    if slot == 0:   ls = pt
                    elif slot == 1: rs = pt
                    elif slot == 2: lh = pt
                    else:           rh = pt

            torso_box = _estimate_torso(ls, rs, lh, rh, person_box)
            detections.append((person_box, torso_box))

        return detections

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Cropping & compositing ───────────────────────────────────────────────────
def _clamp_square(cx, cy, half, img_w, img_h):
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img_w, cx + half)
    y2 = min(img_h, cy + half)
    side = min(x2 - x1, y2 - y1)
    if side < 2 * half:
        x1 = max(0, min(x1, img_w - side))
        y1 = max(0, min(y1, img_h - side))
        x2 = x1 + side
        y2 = y1 + side
    return int(x1), int(y1), int(x2), int(y2)


def compose_input(frame_bgr, torso_box):
    """
    Build a 224x224 composite image (RGB tensor) from a BGR frame.
    Skips PIL entirely — does resize in OpenCV, then converts the small
    224x224 patch to a normalised float32 tensor directly.
    """
    h, w = frame_bgr.shape[:2]
    tx1, ty1, tx2, ty2 = torso_box
    tcx = (tx1 + tx2) // 2
    tcy = (ty1 + ty2) // 2
    torso_w = tx2 - tx1
    torso_h = ty2 - ty1

    # Inner crop
    inner_side = max(min(torso_w, torso_h), 10)
    ix1, iy1, ix2, iy2 = _clamp_square(tcx, tcy, inner_side // 2, w, h)
    inner_resized = cv2.resize(
        frame_bgr[iy1:iy2, ix1:ix2], (INNER_SIZE, INNER_SIZE), interpolation=cv2.INTER_AREA,
    )

    # Outer crop
    outer_side = max(int(max(torso_w, torso_h) * 2.0), inner_side + 20)
    ox1, oy1, ox2, oy2 = _clamp_square(tcx, tcy, outer_side // 2, w, h)
    composite = cv2.resize(
        frame_bgr[oy1:oy2, ox1:ox2], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA,
    )

    # Paste inner
    offset = (IMG_SIZE - INNER_SIZE) // 2
    composite[offset:offset + INNER_SIZE, offset:offset + INNER_SIZE] = inner_resized

    # BGR uint8 -> RGB float32 tensor, normalised (no PIL round-trip)
    rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    tensor.sub_(_NORM_MEAN).div_(_NORM_STD)
    return tensor


# ── Inference ─────────────────────────────────────────────────────────────────
SIGNIFICANT_RATIO = 0.35
SIGNIFICANT_FLOOR = 0.08
AMBIGUITY_THRESHOLD = 0.25


def predict_color(model, device, input_tensor):
    """
    Run inference on a pre-processed tensor.

    Returns
    -------
    colors    : list[tuple[str, float]]
    probs     : np.ndarray (13,)
    ambiguous : bool
    """
    batch = input_tensor.unsqueeze(0).to(device)
    with torch.inference_mode():
        probs = F.softmax(model(batch), dim=1).squeeze(0)

    if device.type != "cpu":
        probs = probs.cpu()
    probs_np = probs.numpy()

    sorted_idx = probs_np.argsort()[::-1]
    top_prob = probs_np[sorted_idx[0]]
    ambiguous = top_prob < AMBIGUITY_THRESHOLD

    threshold = max(SIGNIFICANT_FLOOR, top_prob * SIGNIFICANT_RATIO)
    colors = [
        (COLOR_CLASSES[idx], float(probs_np[idx]))
        for idx in sorted_idx
        if probs_np[idx] >= threshold
    ]

    return colors, probs_np, ambiguous


# ── Annotation ────────────────────────────────────────────────────────────────
# Pre-computed BGR tuples so we never convert at runtime
_LABEL_COLORS_BGR = {
    "red": (0, 0, 255), "orange": (0, 165, 255), "yellow": (0, 255, 255),
    "green": (0, 180, 0), "blue": (255, 0, 0), "violet": (226, 43, 138),
    "purple": (128, 0, 128), "white": (255, 255, 255), "gray": (150, 150, 150),
    "black": (40, 40, 40), "pink": (180, 105, 255), "brown": (19, 69, 139),
    "olive": (0, 128, 128),
}
_DEFAULT_BOX_BGR = (120, 120, 120)


def annotate_frame(frame_bgr, person_box, colors, ambiguous):
    x1, y1, x2, y2 = person_box
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2

    if ambiguous or not colors:
        box_bgr = _DEFAULT_BOX_BGR
    else:
        box_bgr = _LABEL_COLORS_BGR.get(colors[0][0], (0, 255, 0))

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), box_bgr, 2)

    if ambiguous:
        text = "uncertain"
    elif len(colors) == 1:
        lbl, conf = colors[0]
        text = f"{lbl} ({conf:.0%})"
    else:
        parts = [c[0] for c in colors]
        total_conf = sum(c[1] for c in colors)
        text = f"{' / '.join(parts)} ({total_conf:.0%})"

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    txt_y = max(y1 - 8, th + 4)

    cv2.rectangle(frame_bgr,
                  (x1, txt_y - th - 4),
                  (x1 + tw + 6, txt_y + baseline),
                  box_bgr, cv2.FILLED)

    b_ch, g_ch, r_ch = box_bgr
    luminance = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    text_bgr = (0, 0, 0) if luminance > 128 else (255, 255, 255)

    cv2.putText(frame_bgr, text, (x1 + 3, txt_y), font, scale, text_bgr, thickness, cv2.LINE_AA)
    return frame_bgr


# ── High-level pipeline ──────────────────────────────────────────────────────
class ShirtColorDetector:
    def __init__(self, models_dir=None, checkpoint_path=None, device=None):
        if checkpoint_path is None:
            if models_dir is None:
                models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            checkpoint_path = find_best_checkpoint(models_dir)
        self.model, self.device = load_model(checkpoint_path, device)
        self.pose = PoseDetector()

    def process_frame(self, frame_bgr):
        """
        Detect all persons + torsos, predict colour(s), annotate frame in-place.

        Returns
        -------
        annotated : np.ndarray  BGR frame with annotations
        results   : list[dict]  One dict per person.
        """
        detections = self.pose.detect(frame_bgr)
        if not detections:
            return frame_bgr, []

        results = []
        for person_box, torso_box in detections:
            input_tensor = compose_input(frame_bgr, torso_box)
            colors, probs, ambiguous = predict_color(self.model, self.device, input_tensor)
            annotate_frame(frame_bgr, person_box, colors, ambiguous)
            results.append({"colors": colors, "ambiguous": ambiguous, "probs": probs})

        return frame_bgr, results

    def close(self):
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
