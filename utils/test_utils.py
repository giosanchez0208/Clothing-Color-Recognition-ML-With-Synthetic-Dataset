import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
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

_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
_NORMALIZE_STD = [0.229, 0.224, 0.225]

_VAL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(_NORMALIZE_MEAN, _NORMALIZE_STD),
])

# YOLO pose keypoint indices (COCO format)
_LEFT_SHOULDER  = 5
_RIGHT_SHOULDER = 6
_LEFT_HIP       = 11
_RIGHT_HIP      = 12


# ── Model helpers ─────────────────────────────────────────────────────────────
def _create_model():
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def load_model(checkpoint_path, device=None):
    """Load a trained model from a checkpoint file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _create_model().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device


def find_best_checkpoint(models_dir):
    """Return path to the best available checkpoint in models_dir."""
    for name in ("finetune_best.pth", "best.pth"):
        path = os.path.join(models_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No checkpoint found in {models_dir}")


# ── Torso estimation ─────────────────────────────────────────────────────────
def _estimate_torso(ls, rs, lh, rh, person_box):
    """
    Estimate the torso bounding box robustly from any combination of visible
    shoulder/hip keypoints, falling back to proportional estimates when some
    are missing or low-confidence.

    Parameters
    ----------
    ls, rs, lh, rh : (x, y) float tuple or None
    person_box     : (px1, py1, px2, py2) full person bounding box
    """
    px1, py1, px2, py2 = person_box
    ph = py2 - py1
    pw = px2 - px1

    shoulder_pts = [p for p in (ls, rs) if p is not None]
    hip_pts      = [p for p in (lh, rh) if p is not None]
    all_pts      = shoulder_pts + hip_pts

    # X bounds — use keypoint spread if we have ≥2 pts, else inset person bbox
    if len(all_pts) >= 2:
        xs = [p[0] for p in all_pts]
        tx1, tx2 = int(min(xs)), int(max(xs))
    else:
        inset = int(pw * 0.10)
        tx1, tx2 = px1 + inset, px2 - inset

    # Y top — shoulder line or estimated from hips / person bbox
    if shoulder_pts:
        ty1 = int(min(p[1] for p in shoulder_pts))
    elif hip_pts:
        ty1 = int(min(p[1] for p in hip_pts) - 0.40 * ph)
    else:
        ty1 = int(py1 + 0.15 * ph)

    # Y bottom — hip line or estimated from shoulders / person bbox
    if hip_pts:
        ty2 = int(max(p[1] for p in hip_pts))
    elif shoulder_pts:
        ty2 = int(max(p[1] for p in shoulder_pts) + 0.40 * ph)
    else:
        ty2 = int(py1 + 0.65 * ph)

    # Clamp to person bbox
    tx1, tx2 = max(tx1, px1), min(tx2, px2)
    ty1, ty2 = max(ty1, py1), min(ty2, py2)

    # Safety: fall back to proportional slice if box is degenerate
    if tx2 - tx1 < 10 or ty2 - ty1 < 10:
        inset = int(pw * 0.10)
        return (px1 + inset, int(py1 + 0.15 * ph), px2 - inset, int(py1 + 0.65 * ph))

    return tx1, ty1, tx2, ty2


# ── Pose detection ────────────────────────────────────────────────────────────
class PoseDetector:
    def __init__(self, confidence=0.5):
        self._model = YOLO("yolo11n-pose.pt")
        self._conf = confidence

    def detect(self, frame_rgb):
        """
        Run YOLO pose on an RGB frame.

        Returns
        -------
        detections : list[tuple]  Each element is (person_box, torso_box) where
                                  person_box : (x1, y1, x2, y2) full-body bounding box
                                  torso_box  : (x1, y1, x2, y2) estimated torso region
        """
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = self._model(frame_bgr, conf=self._conf, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        kpts  = results[0].keypoints
        detections = []

        for i in range(len(boxes)):
            bxy        = boxes.xyxy[i]
            person_box = (int(bxy[0]), int(bxy[1]), int(bxy[2]), int(bxy[3]))

            # Extract the four torso keypoints (best-effort)
            ls = rs = lh = rh = None
            if kpts is not None and kpts.xy is not None and len(kpts.xy) > i:
                xy   = kpts.xy[i]
                conf = kpts.conf[i] if kpts.conf is not None else None

                def _get(idx, _xy=xy, _conf=conf):
                    if _conf is not None and float(_conf[idx]) < 0.4:
                        return None
                    x, y = float(_xy[idx][0]), float(_xy[idx][1])
                    return None if (x == 0.0 and y == 0.0) else (x, y)

                ls = _get(_LEFT_SHOULDER)
                rs = _get(_RIGHT_SHOULDER)
                lh = _get(_LEFT_HIP)
                rh = _get(_RIGHT_HIP)

            torso_box = _estimate_torso(ls, rs, lh, rh, person_box)
            detections.append((person_box, torso_box))

        return detections

    def close(self):
        pass  # YOLO requires no explicit cleanup

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Cropping & compositing ───────────────────────────────────────────────────
def _clamp_square(cx, cy, half, img_w, img_h):
    """Return (x1, y1, x2, y2) for a square centred at (cx, cy) clamped to image bounds."""
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img_w, cx + half)
    y2 = min(img_h, cy + half)
    # Adjust to keep square if clamped
    side = min(x2 - x1, y2 - y1)
    if side < 2 * half:
        # Re-centre within available space
        x1 = max(0, min(x1, img_w - side))
        y1 = max(0, min(y1, img_h - side))
        x2 = x1 + side
        y2 = y1 + side
    return int(x1), int(y1), int(x2), int(y2)


def compose_input(frame_rgb, torso_box):
    """
    Build a 224×224 composite image that matches the training data format:
    - Outer 224×224: a larger crop of the environment around the torso
    - Inner 112×112: a crop from inside the torso, pasted in the centre

    Parameters
    ----------
    frame_rgb : np.ndarray  (H, W, 3) RGB image
    torso_box : tuple       (x1, y1, x2, y2) torso bounding box

    Returns
    -------
    composite : np.ndarray  (224, 224, 3) RGB image ready for the model
    """
    h, w = frame_rgb.shape[:2]
    tx1, ty1, tx2, ty2 = torso_box
    tcx = (tx1 + tx2) // 2
    tcy = (ty1 + ty2) // 2
    torso_w = tx2 - tx1
    torso_h = ty2 - ty1

    # ── Inner crop: square from inside the torso ──────────────────────────
    inner_side = min(torso_w, torso_h)
    inner_side = max(inner_side, 10)  # safety floor
    ix1, iy1, ix2, iy2 = _clamp_square(tcx, tcy, inner_side // 2, w, h)
    inner_crop = frame_rgb[iy1:iy2, ix1:ix2]
    inner_resized = cv2.resize(inner_crop, (INNER_SIZE, INNER_SIZE), interpolation=cv2.INTER_AREA)

    # ── Outer crop: larger square around the torso ────────────────────────
    outer_side = int(max(torso_w, torso_h) * 2.0)
    outer_side = max(outer_side, inner_side + 20)  # at least a bit larger
    ox1, oy1, ox2, oy2 = _clamp_square(tcx, tcy, outer_side // 2, w, h)
    outer_crop = frame_rgb[oy1:oy2, ox1:ox2]
    outer_resized = cv2.resize(outer_crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # ── Composite: paste inner centred on outer ───────────────────────────
    composite = outer_resized.copy()
    offset = (IMG_SIZE - INNER_SIZE) // 2  # 56
    composite[offset:offset + INNER_SIZE, offset:offset + INNER_SIZE] = inner_resized

    return composite


# ── Inference ─────────────────────────────────────────────────────────────────
# A colour is "significant" if it holds at least this fraction of the top
# prediction's probability.  e.g. top=0.50, second=0.30 → 0.30/0.50 = 0.60 ≥ 0.35 → shown.
SIGNIFICANT_RATIO = 0.35
# Absolute floor — ignore anything below this regardless of ratio.
SIGNIFICANT_FLOOR = 0.08
# If even the top colour is below this, the prediction is ambiguous.
AMBIGUITY_THRESHOLD = 0.25


def predict_color(model, device, composite_rgb):
    """
    Run inference on a composite image.

    Returns
    -------
    colors : list[tuple[str, float]]  Significant (label, probability) pairs, sorted descending.
    probs  : np.ndarray               Full probability vector (13,).
    ambiguous : bool                   True when the model is not confident.
    """
    tensor = _VAL_TRANSFORM(composite_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().squeeze(0)

    probs_np = probs.numpy()
    sorted_idx = probs_np.argsort()[::-1]
    top_prob = probs_np[sorted_idx[0]]

    ambiguous = top_prob < AMBIGUITY_THRESHOLD

    # Collect every colour that is both above the absolute floor and
    # within SIGNIFICANT_RATIO of the top prediction.
    colors = []
    for idx in sorted_idx:
        p = float(probs_np[idx])
        if p < SIGNIFICANT_FLOOR:
            break
        if p >= top_prob * SIGNIFICANT_RATIO:
            colors.append((COLOR_CLASSES[idx], p))

    return colors, probs_np, ambiguous


# ── Annotation ────────────────────────────────────────────────────────────────
_LABEL_COLORS = {
    "red": (255, 0, 0), "orange": (255, 165, 0), "yellow": (255, 255, 0),
    "green": (0, 180, 0), "blue": (0, 0, 255), "violet": (138, 43, 226),
    "purple": (128, 0, 128), "white": (255, 255, 255), "gray": (150, 150, 150),
    "black": (40, 40, 40), "pink": (255, 105, 180), "brown": (139, 69, 19),
    "olive": (128, 128, 0),
}


def _rgb_to_bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


def annotate_frame(frame_bgr, person_box, colors, ambiguous):
    """
    Draw a bounding box around the full person and label with predicted
    shirt colour(s).  Handles multi-colour and ambiguous predictions.

    The label background is filled with the predicted colour; text colour
    is black or white chosen by luminance for guaranteed high contrast.

    Parameters
    ----------
    frame_bgr  : np.ndarray
    person_box : tuple (x1, y1, x2, y2)  full-body bounding box
    colors     : list[tuple[str, float]]  Significant colours sorted by probability.
    ambiguous  : bool
    """
    x1, y1, x2, y2 = person_box
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2

    # Box colour = top predicted colour (or gray if ambiguous)
    if ambiguous or not colors:
        box_bgr = (120, 120, 120)
    else:
        box_bgr = _rgb_to_bgr(_LABEL_COLORS.get(colors[0][0], (0, 255, 0)))

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), box_bgr, 2)

    # Build label text
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

    # Fill label background with the box colour
    cv2.rectangle(frame_bgr,
                  (x1, txt_y - th - 4),
                  (x1 + tw + 6, txt_y + baseline),
                  box_bgr, cv2.FILLED)

    # High-contrast text: white on dark, black on light (ITU-R BT.601 luminance)
    b_ch, g_ch, r_ch = box_bgr
    luminance = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    text_bgr = (0, 0, 0) if luminance > 128 else (255, 255, 255)

    cv2.putText(frame_bgr, text, (x1 + 3, txt_y), font, scale, text_bgr, thickness, cv2.LINE_AA)

    return frame_bgr


# ── High-level pipeline ──────────────────────────────────────────────────────
class ShirtColorDetector:
    """
    End-to-end detector: given a BGR frame, detects the person's torso,
    predicts the shirt colour, and annotates the frame.
    """

    def __init__(self, models_dir=None, checkpoint_path=None, device=None):
        if checkpoint_path is None:
            if models_dir is None:
                models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            checkpoint_path = find_best_checkpoint(models_dir)
        self.model, self.device = load_model(checkpoint_path, device)
        self.pose = PoseDetector()

    def process_frame(self, frame_bgr):
        """
        Detect all persons + torsos, predict colour(s), annotate frame.

        The bounding box drawn on screen tracks each full person.
        Colour inference uses the estimated torso region for each person.

        Returns
        -------
        annotated : np.ndarray  BGR frame with annotations for all detected persons
        results   : list[dict]  One dict per person: {'colors', 'ambiguous', 'probs'}.
                                Empty list if no person is found.
        """
        frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = self.pose.detect(frame_rgb)
        if not detections:
            return frame_bgr, []

        annotated = frame_bgr.copy()
        results   = []
        for person_box, torso_box in detections:
            composite = compose_input(frame_rgb, torso_box)
            colors, probs, ambiguous = predict_color(self.model, self.device, composite)
            annotated = annotate_frame(annotated, person_box, colors, ambiguous)
            results.append({"colors": colors, "ambiguous": ambiguous, "probs": probs})

        return annotated, results

    def close(self):
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
