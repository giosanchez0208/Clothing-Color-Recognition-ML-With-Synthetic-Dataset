# =============================================================================
# instrumented_generator.py — dataset generation that records its own decisions
# =============================================================================
# Differences from DatasetGeneratorV2:
#
#   1. METADATA. Every image emits the parameters that produced it (pattern,
#      colour count, label entropy, and every augmentation value). Without this
#      a failure-mode diagnosis is impossible: you can see that some images are
#      harder, never which axis made them hard.
#
#   2. DETERMINISM. Each image is seeded from (base_seed, index), so the whole
#      dataset is reproducible and any single image can be regenerated in
#      isolation from the `seed` column. V2 used unseeded globals, which is why
#      the original 22k set can never be recovered exactly.
#
#   3. CACHED BACKGROUND LISTING. V2's OuterSquareGenerator called os.listdir()
#      inside generate() — once per image, over ~15.6k files. Hoisting it out
#      removes ~343M redundant filename operations across a 22k run.
#
#   4. PARALLEL. Generation is embarrassingly parallel; per-image seeding keeps
#      it deterministic regardless of how work is distributed across workers.
# =============================================================================

import math
import os
import random

import cv2
import numpy as np

from .color_utils_extended import (
    PATTERN_TYPES_V2,
    PATTERN_WEIGHTS_V2,
    InnerSquareGeneratorV2,
)
from .dataset_utils import DEFAULT_DIMENSIONS, _compute_label_percentages
from .instrumented_augment import AXES, META_COLUMNS, apply_lighting, apply_single_axis

COLOR_CLASSES = [
    "red", "orange", "yellow", "green", "blue",
    "violet", "purple", "white", "gray", "black",
    "pink", "brown", "olive",
]
NUM_CLASSES = len(COLOR_CLASSES)

# Columns written to labels.csv, in order.
STRUCTURE_COLUMNS = ("pattern", "n_colors", "label_entropy", "fold_blend", "fold_alpha")
EXTRA_COLUMNS = ("split", "seed", "probe_axis", "probe_value") + STRUCTURE_COLUMNS + META_COLUMNS


def seed_for(base_seed, index):
    """Deterministic per-image seed. Order-independent, so parallel == serial."""
    return (base_seed * 1_000_003 + index * 2_654_435_761) % (2 ** 32)


def label_entropy(label_vec):
    """Shannon entropy of the soft label, normalised to [0, 1].

    A free difficulty signal: a solid shirt is one-hot (0.0), a four-colour
    plaid spreads mass across categories (towards 1.0). Orthogonal to the
    photometric axes, which is exactly why it is worth recording separately.
    """
    total = sum(label_vec)
    if total <= 0:
        return 0.0
    h = -sum((p / total) * math.log(p / total) for p in label_vec if p > 0)
    return max(0.0, h / math.log(NUM_CLASSES))  # clamp float noise at the one-hot end


class _RecordingInnerGenerator(InnerSquareGeneratorV2):
    """InnerSquareGeneratorV2 that remembers which pattern/blend it chose."""

    def __init__(self, csv_path, dimensions=DEFAULT_DIMENSIONS):
        super().__init__(csv_path=csv_path, dimensions=dimensions)
        self.last = {}

    def add_pattern(self, inner_square, labels, label_map, used_hexes):
        pattern_type = random.choices(PATTERN_TYPES_V2, weights=PATTERN_WEIGHTS_V2, k=1)[0]
        self.last["pattern"] = pattern_type

        # Re-dispatch through the parent by forcing the choice. Parent re-rolls
        # its own pattern, so instead we inline the parent's branch structure
        # using the already-decided type.
        from .color_utils_extended import (
            _apply_chevron_pattern, _apply_gradient_pattern, _apply_plaid_pattern,
        )
        from .dataset_utils import (
            _apply_color_blocking_pattern, _apply_polka_dot_pattern,
            _apply_solid_gradient_pattern, _apply_stripes_pattern,
            _sample_distinct_random_colors,
        )

        if pattern_type == "solid":
            patterned = _apply_solid_gradient_pattern(inner_square, max_opacity=0.2)
            self.last["n_colors"] = 1
            return patterned, labels, label_map

        extra_counts = {
            "stripes": (1, 3), "color_blocking": (1, 2), "polka_dot": (2, 5),
            "plaid": (2, 3), "gradient": (1, 2), "chevron": (1, 3),
        }
        lo, hi = extra_counts.get(pattern_type, (1, 3))
        extra_pairs = _sample_distinct_random_colors(
            self.generate_random_color, used_hexes, random.randint(lo, hi),
        )
        extra_labels = [cat for cat, _ in extra_pairs]
        extra_bgr = [self._hex_to_bgr(hx) for _, hx in extra_pairs]
        label_indices = list(range(len(labels), len(labels) + len(extra_labels)))
        self.last["n_colors"] = 1 + len(extra_labels)

        dispatch = {
            "stripes": _apply_stripes_pattern,
            "color_blocking": _apply_color_blocking_pattern,
            "polka_dot": _apply_polka_dot_pattern,
            "plaid": _apply_plaid_pattern,
            "gradient": _apply_gradient_pattern,
            "chevron": _apply_chevron_pattern,
        }
        patterned, label_map = dispatch[pattern_type](
            inner_square, label_map, extra_bgr, label_indices,
        )
        return patterned, labels + extra_labels, label_map

    def apply_synthetic_fold_texture(self, inner_square):
        from .dataset_utils import _sample_texture_opacity, generate_synthetic_clothing_folds

        # Seed drawn from the (already-seeded) global RNG so the fold texture
        # is reproducible. Passing seed=None here silently breaks determinism —
        # _fbm_perlin's default_rng(None) ignores np.random.seed().
        texture = generate_synthetic_clothing_folds(
            seed=int(np.random.randint(0, 2 ** 31 - 1)),
        )
        target_h, target_w = inner_square.shape[:2]
        scale = random.uniform(1.0, 3.0)
        new_h, new_w = int(target_h * scale), int(target_w * scale)
        texture = cv2.resize(texture, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        sy = random.randint(0, new_h - target_h)
        sx = random.randint(0, new_w - target_w)
        texture = texture[sy:sy + target_h, sx:sx + target_w]

        base = inner_square.astype(np.float32) / 255.0
        tex3 = np.repeat((texture.astype(np.float32) / 255.0)[:, :, None], 3, axis=2)

        blend_mode = random.choice(["multiply", "overlay"])
        if blend_mode == "overlay":
            blended_raw = np.where(base < 0.5,
                                   2.0 * base * tex3,
                                   1.0 - 2.0 * (1.0 - base) * (1.0 - tex3))
        else:
            blended_raw = base * tex3

        alpha = _sample_texture_opacity()
        self.last["fold_blend"] = blend_mode
        self.last["fold_alpha"] = alpha
        blended = base * (1.0 - alpha) + blended_raw * alpha
        return np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    def generate(self):
        self.last = {"pattern": None, "n_colors": 0,
                     "fold_blend": "none", "fold_alpha": 0.0}
        inner, label_pcts = self.compose_random_color()
        if random.random() < 0.85:
            inner = self.apply_synthetic_fold_texture(inner)
        return inner, label_pcts


class InstrumentedGenerator:
    """Produces (image, label_vector, metadata) triples."""

    def __init__(self, csv_path, path_to_bgs, dimensions=DEFAULT_DIMENSIONS):
        self.dimensions = dimensions
        self.inner = _RecordingInnerGenerator(csv_path=csv_path, dimensions=dimensions)
        # Hoisted out of the hot loop (V2 re-listed this directory per image)
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        self.bg_files = [
            os.path.join(path_to_bgs, f)
            for f in os.listdir(path_to_bgs)
            if f.lower().endswith(exts)
        ]
        if not self.bg_files:
            raise ValueError(f"No background images found in {path_to_bgs}")

    def _background(self):
        img = cv2.imread(random.choice(self.bg_files))
        if img is None:
            return np.full((*self.dimensions, 3), 128, dtype=np.uint8)
        th, tw = self.dimensions
        h, w = img.shape[:2]
        top, left = max(0, (h - th) // 2), max(0, (w - tw) // 2)
        crop = img[top:top + th, left:left + tw]
        if crop.shape[:2] != (th, tw):
            crop = cv2.resize(crop, (tw, th))
        return crop

    def generate(self, index, base_seed, probe=False, axis=None, value=None):
        """Generate one sample.

        probe=False -> stochastic augmentation (the training distribution)
        probe=True  -> exactly one axis at `value`, every other axis neutral.
                       axis=None with probe=True is the neutral control group,
                       which must be distinguishable from "no axis requested" —
                       hence an explicit flag rather than inferring from axis.
        """
        s = seed_for(base_seed, index)
        random.seed(s)
        np.random.seed(s)

        background = self._background()
        inner, label_pcts = self.inner.generate()

        composed = background.copy()
        bh, bw = composed.shape[:2]
        ih, iw = inner.shape[:2]
        top, left = (bh - ih) // 2, (bw - iw) // 2
        composed[top:top + ih, left:left + iw] = inner

        if probe:
            composed, aug_meta = apply_single_axis(composed, axis, value)
        else:
            composed, aug_meta = apply_lighting(composed)

        vec = [label_pcts.get(c, 0.0) for c in COLOR_CLASSES]
        total = sum(vec)
        vec = [v / total for v in vec] if total > 0 else [1.0 / NUM_CLASSES] * NUM_CLASSES

        meta = dict(aug_meta)
        meta.update({
            "seed": s,
            "probe_axis": axis if axis is not None else "",
            "probe_value": value if value is not None else float("nan"),
            "pattern": self.inner.last.get("pattern") or "solid",
            "n_colors": self.inner.last.get("n_colors", 1),
            "label_entropy": label_entropy(vec),
            "fold_blend": self.inner.last.get("fold_blend", "none"),
            "fold_alpha": self.inner.last.get("fold_alpha", 0.0),
        })
        return composed, vec, meta


def probe_plan(axes=AXES, n_levels=8, per_cell=25, n_control=200):
    """Build the (axis, value) work list for a one-factor-at-a-time probe set.

    Returns a list of (axis, value) tuples, one entry per image to generate.
    `axis=None` entries are the neutral control group — the baseline every
    response curve is measured against.
    """
    from .instrumented_augment import probe_values

    plan = [(None, None)] * n_control
    for axis in axes:
        for value in probe_values(axis, n_levels):
            plan.extend([(axis, value)] * per_cell)
    return plan
