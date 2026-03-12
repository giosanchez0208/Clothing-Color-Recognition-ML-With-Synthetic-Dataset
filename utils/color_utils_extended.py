# =============================================================================
# color_utils_extended.py — V2 dataset generation extensions
# =============================================================================
# New features over v1 (dataset_utils.py):
#   • Three additional clothing patterns: plaid, gradient/ombre, chevron
#   • Three additional augmentations: perspective warp, specular highlight,
#     vignette
#   • InnerSquareGeneratorV2  — configurable CSV path, expanded pattern set,
#                               overlay blend mode for fold textures
#   • DatasetGeneratorV2      — composes with v2 augmentation pipeline
# =============================================================================

import numpy as np
import cv2
import os
import random

from .color_utils import ColorLibrary
from .dataset_utils import (
    DEFAULT_DIMENSIONS,
    OuterSquareGenerator,
    generate_synthetic_clothing_folds,
    _compute_label_percentages,
    _sample_distinct_random_colors,
    _apply_solid_gradient_pattern,
    _apply_stripes_pattern,
    _apply_color_blocking_pattern,
    _apply_polka_dot_pattern,
    _apply_global_lighting,
    _sample_texture_opacity,
)

# ── V2 Pattern Configuration ────────────────────────────────────────────────

PATTERN_TYPES_V2 = [
    'solid', 'stripes', 'color_blocking', 'polka_dot',
    'plaid', 'gradient', 'chevron',
]
PATTERN_WEIGHTS_V2 = [0.25, 0.15, 0.10, 0.08, 0.17, 0.13, 0.12]

# ═══════════════════════════════════════════════════════════════════════════════
#  NEW PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_plaid_pattern(img, label_map, colors_bgr, label_indices):
    """Overlapping horizontal + vertical stripes with alpha blending."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    map_out = label_map.copy()
    n_colors = len(colors_bgr)

    stripe_w = random.randint(max(4, min(h, w) // 14), max(10, min(h, w) // 5))
    gap = random.randint(stripe_w, stripe_w * 3)
    alpha_h = random.uniform(0.6, 0.9)
    alpha_v = random.uniform(0.3, 0.6)

    # Horizontal stripes
    h_mask = np.zeros((h, w), dtype=bool)
    y, idx = random.randint(0, gap // 2), 0
    while y < h:
        end_y = min(h, y + stripe_w)
        color = np.array(colors_bgr[idx % n_colors], dtype=np.float32)
        out[y:end_y] = out[y:end_y] * (1.0 - alpha_h) + color * alpha_h
        map_out[y:end_y, :] = label_indices[idx % n_colors]
        h_mask[y:end_y, :] = True
        y += stripe_w + gap
        idx += 1

    # Vertical stripes (thinner, lower opacity, offset color index)
    v_stripe_w = max(2, stripe_w * 2 // 3)
    x, idx = random.randint(0, gap // 2), 0
    while x < w:
        end_x = min(w, x + v_stripe_w)
        color = np.array(colors_bgr[(idx + 1) % n_colors], dtype=np.float32)
        out[:, x:end_x] = out[:, x:end_x] * (1.0 - alpha_v) + color * alpha_v
        # Label: only where no horizontal stripe already dominates
        no_h = ~h_mask[:, x:end_x]
        map_out[:, x:end_x][no_h] = label_indices[(idx + 1) % n_colors]
        x += v_stripe_w + gap
        idx += 1

    return np.clip(out, 0, 255).astype(np.uint8), map_out


def _apply_gradient_pattern(img, label_map, colors_bgr, label_indices):
    """Smooth ombre/gradient transition between colors."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    map_out = label_map.copy()

    direction = random.choice(['horizontal', 'vertical', 'diagonal'])
    n_colors = min(len(colors_bgr), 3)

    # Build 0→1 transition map
    if direction == 'horizontal':
        t = np.linspace(0, 1, w, dtype=np.float32)[None, :]
        t = np.broadcast_to(t, (h, w)).copy()
    elif direction == 'vertical':
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        t = np.broadcast_to(t, (h, w)).copy()
    else:
        xv = np.linspace(0, 1, w, dtype=np.float32)
        yv = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        t = np.clip((xv + yv) * 0.5, 0, 1)

    base = out.copy()
    c1 = np.array(colors_bgr[0], dtype=np.float32)

    if n_colors >= 2:
        c2 = np.array(colors_bgr[1], dtype=np.float32)
        # Two-segment gradient: base → c1 (at t=0.5) → c2 (at t=1.0)
        f = t[:, :, None]
        low_mask = f < 0.5
        f1 = f * 2.0
        f2 = (f - 0.5) * 2.0
        gradient = np.where(low_mask,
                            base * (1.0 - f1) + c1 * f1,
                            c1 * (1.0 - f2) + c2 * f2)
        map_out[t >= 0.33] = label_indices[0]
        map_out[t >= 0.67] = label_indices[1]
    else:
        f = t[:, :, None]
        gradient = base * (1.0 - f) + c1 * f
        map_out[t >= 0.5] = label_indices[0]

    alpha = random.uniform(0.7, 1.0)
    out = out * (1.0 - alpha) + gradient * alpha

    return np.clip(out, 0, 255).astype(np.uint8), map_out


def _apply_chevron_pattern(img, label_map, colors_bgr, label_indices):
    """V-shaped zigzag stripe pattern."""
    h, w = img.shape[:2]
    out = img.copy()
    map_out = label_map.copy()

    n_colors = len(colors_bgr)
    band_h = random.randint(max(8, h // 10), max(16, h // 4))
    cx = w / 2.0
    slope = random.uniform(0.3, 1.5)

    yy, xx = np.mgrid[0:h, 0:w]
    phase = yy + (np.abs(xx - cx) * slope)
    band = (phase / band_h).astype(np.int32) % (n_colors + 1)

    for i in range(n_colors):
        mask = band == (i + 1)
        out[mask] = colors_bgr[i]
        map_out[mask] = label_indices[i]

    return out, map_out


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW AUGMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _augment_specular_highlight(img_float):
    """Add a bright elliptical highlight to simulate fabric sheen."""
    h, w = img_float.shape[:2]
    cx = random.randint(w // 4, 3 * w // 4)
    cy = random.randint(h // 4, 3 * h // 4)
    rx = random.randint(w // 8, w // 3)
    ry = random.randint(h // 8, h // 3)
    strength = random.uniform(0.15, 0.4)

    yy, xx = np.mgrid[0:h, 0:w]
    dist = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2
    mask = np.clip(1.0 - dist, 0, 1) ** 2  # soft falloff
    mask = mask[:, :, None] * strength

    return np.clip(img_float + mask, 0, 1)


def _augment_vignette(img_float):
    """Darken edges to simulate lens vignetting."""
    h, w = img_float.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((xx - cx) / max(cx, 1)) ** 2 + ((yy - cy) / max(cy, 1)) ** 2)
    strength = random.uniform(0.2, 0.5)
    mask = 1.0 - np.clip(dist * strength, 0, 1)
    mask = mask[:, :, None]

    return img_float * mask


def _apply_global_lighting_v2(img, p=0.5, **kwargs):
    """V1 augmentation pipeline + perspective warp, specular highlight, vignette."""
    out = _apply_global_lighting(img, p=p, **kwargs)

    out_f = out.astype(np.float32) / 255.0
    if random.random() < p * 0.4:
        out_f = _augment_specular_highlight(out_f)
    if random.random() < p * 0.4:
        out_f = _augment_vignette(out_f)
    out = (np.clip(out_f, 0, 1) * 255).astype(np.uint8)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  V2 GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

class InnerSquareGeneratorV2:
    """Like InnerSquareGenerator but with configurable CSV path and more patterns.

    Key differences from v1:
      • Accepts csv_path parameter (for normalized CSV)
      • 7 pattern types (v1 had 4): adds plaid, gradient, chevron
      • Fold texture uses randomly chosen blend mode (multiply or overlay)
    """

    def __init__(self, csv_path, dimensions=DEFAULT_DIMENSIONS):
        self.dimensions = dimensions
        self.color_library = ColorLibrary.from_categorized_csv(csv_path)

    def generate_random_color(self):
        categories = self.color_library.categories
        category = random.choice(categories)
        color_distances = self.color_library.get_category_mahalanobis_distances(category)
        if not color_distances:
            raise ValueError(f'No colors found for category {category}.')

        eps = 1e-6
        hex_colors = [hx for hx, _ in color_distances]
        weights = [(1.0 / (dist + eps)) ** 2 for _, dist in color_distances]
        selected_hex = random.choices(hex_colors, weights=weights, k=1)[0]
        return category, selected_hex

    def compose_random_color(self):
        category, color_hex = self.generate_random_color()
        labels = [category]
        square_height = int(self.dimensions[0] / 2)
        square_width = int(self.dimensions[1] / 2)
        color_bgr = self._hex_to_bgr(color_hex)
        inner_square = np.full((square_height, square_width, 3), color_bgr, dtype=np.uint8)
        label_map = np.zeros((square_height, square_width), dtype=np.uint16)
        inner_square, labels, label_map = self.add_pattern(
            inner_square, labels, label_map, [color_hex],
        )
        label_percentages = _compute_label_percentages(labels, label_map)
        return inner_square, label_percentages

    def add_pattern(self, inner_square, labels, label_map, used_hexes):
        pattern_type = random.choices(
            PATTERN_TYPES_V2, weights=PATTERN_WEIGHTS_V2, k=1,
        )[0]

        if pattern_type == 'solid':
            patterned = _apply_solid_gradient_pattern(inner_square, max_opacity=0.2)
            return patterned, labels, label_map

        # Extra colors needed per pattern
        extra_counts = {
            'stripes': (1, 3), 'color_blocking': (1, 2), 'polka_dot': (2, 5),
            'plaid': (2, 3), 'gradient': (1, 2), 'chevron': (1, 3),
        }
        lo, hi = extra_counts.get(pattern_type, (1, 3))
        extra_count = random.randint(lo, hi)

        extra_pairs = _sample_distinct_random_colors(
            self.generate_random_color, used_hexes, extra_count,
        )
        extra_labels = [cat for cat, _ in extra_pairs]
        extra_hexes = [hx for _, hx in extra_pairs]
        extra_bgr = [self._hex_to_bgr(hx) for hx in extra_hexes]
        label_indices = list(range(len(labels), len(labels) + len(extra_labels)))
        labels_extended = labels + extra_labels

        dispatch = {
            'stripes':        _apply_stripes_pattern,
            'color_blocking': _apply_color_blocking_pattern,
            'polka_dot':      _apply_polka_dot_pattern,
            'plaid':          _apply_plaid_pattern,
            'gradient':       _apply_gradient_pattern,
            'chevron':        _apply_chevron_pattern,
        }
        patterned, label_map = dispatch[pattern_type](
            inner_square, label_map, extra_bgr, label_indices,
        )
        return patterned, labels_extended, label_map

    def apply_synthetic_fold_texture(self, inner_square):
        texture = generate_synthetic_clothing_folds()
        target_h, target_w = inner_square.shape[:2]

        scale = random.uniform(1.0, 3.0)
        new_h, new_w = int(target_h * scale), int(target_w * scale)
        texture_resized = cv2.resize(texture, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        max_y = new_h - target_h
        max_x = new_w - target_w
        start_y = random.randint(0, max_y)
        start_x = random.randint(0, max_x)
        texture_cropped = texture_resized[start_y:start_y + target_h,
                                          start_x:start_x + target_w]

        base = inner_square.astype(np.float32) / 255.0
        texture_norm = texture_cropped.astype(np.float32) / 255.0
        texture_3ch = np.repeat(texture_norm[:, :, None], 3, axis=2)

        # V2: randomly choose blend mode
        blend_mode = random.choice(['multiply', 'overlay'])
        if blend_mode == 'overlay':
            low = 2.0 * base * texture_3ch
            high = 1.0 - 2.0 * (1.0 - base) * (1.0 - texture_3ch)
            blended_raw = np.where(base < 0.5, low, high)
        else:
            blended_raw = base * texture_3ch

        alpha = _sample_texture_opacity()
        blended = base * (1.0 - alpha) + blended_raw * alpha
        return np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    def generate(self):
        inner_square, label_percentages = self.compose_random_color()
        if random.random() < 0.85:
            textured = self.apply_synthetic_fold_texture(inner_square)
        else:
            textured = inner_square
        return textured, label_percentages

    @staticmethod
    def _hex_to_bgr(color_hex):
        color_hex = color_hex.lstrip('#')
        return tuple(int(color_hex[i:i + 2], 16) for i in (4, 2, 0))


class DatasetGeneratorV2:
    """Composes inner + outer squares with V2 augmentation pipeline."""

    def __init__(self, csv_path, dimensions=DEFAULT_DIMENSIONS, path_to_bgs=None):
        self.outer_gen = OuterSquareGenerator(dimensions=dimensions, path_to_bgs=path_to_bgs)
        self.inner_gen = InnerSquareGeneratorV2(csv_path=csv_path, dimensions=dimensions)

    def generate(self):
        background = self.outer_gen.generate()
        inner_square, label_percentages = self.inner_gen.generate()

        composed_image = background.copy()
        bg_height, bg_width = composed_image.shape[:2]
        square_height, square_width = inner_square.shape[:2]
        top = (bg_height - square_height) // 2
        left = (bg_width - square_width) // 2

        composed_image[top:top + square_height, left:left + square_width] = inner_square
        composed_image = _apply_global_lighting_v2(composed_image)
        return composed_image, label_percentages
