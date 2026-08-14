# =============================================================================
# instrumented_augment.py: augmentations that report what they did
# =============================================================================
# The V1/V2 pipeline samples a parameter for every augmentation and then throws
# it away. That discarded number is exactly what a failure-mode diagnosis needs:
# without it you can observe *that* the model is worse on some images, but never
# *which photometric axis* is responsible.
#
# This module re-implements the V2 lighting pipeline so that every effect
# returns the value it sampled. Distributions are unchanged from V2, so the
# data is statistically identical, with one deliberate exception documented
# below.
#
# DELIBERATE CHANGE vs V2: hue and saturation are decoupled.
#   In V2 both live inside `_augment_color_jitter` behind ONE p=0.5 gate, so
#   they always fire together. Perfectly correlated regressors cannot be
#   attributed: "the model struggles with hue" is unfalsifiable if hue never
#   varies independently of saturation. They now have independent gates.
#
# Two modes:
#   apply_lighting(...)      stochastic, matches the V2 training distribution
#   apply_single_axis(...)   forces exactly ONE axis to a chosen value, leaving
#                            every other axis neutral. Used for probe sets,
#                            where clean attribution requires that only one
#                            thing varies at a time.
# =============================================================================

import random

import cv2
import numpy as np

from .dataset_utils import (
    _augment_brightness_contrast,
    _augment_gaussian_blur,
    _augment_jpeg_compression,
    _augment_salt_and_pepper,
    _augment_shadow,
    _sample_truncated_normal,
)
from .color_utils_extended import (
    _augment_specular_highlight,
    _augment_vignette,
)

# ── Axis registry ────────────────────────────────────────────────────────────
# Every axis: the metadata column it writes, its neutral (no-effect) value, and
# the range it is swept over when building a probe set.
AXES = (
    "brightness", "temperature", "hue", "saturation", "shadow",
    "blur", "noise", "jpeg", "specular", "vignette",
)

AXIS_RANGE = {
    "brightness":  (0.3, 1.8),      # multiplier;  neutral 1.0
    "temperature": (-1.0, 1.0),     # signed warm/cool;  neutral 0.0
    "hue":         (-25.0, 25.0),   # degrees;  neutral 0.0
    "saturation":  (0.3, 2.0),      # multiplier;  neutral 1.0
    "shadow":      (0.1, 0.55),     # ramp strength;  neutral 0.0
    "blur":        (0.1, 1.5),      # gaussian sigma;  neutral 0.0
    "noise":       (0.001, 0.005),  # salt-pepper density;  neutral 0.0
    "jpeg":        (40, 85),        # quality (LOWER is harder);  neutral 100
    "specular":    (0.15, 0.4),     # highlight strength;  neutral 0.0
    "vignette":    (0.2, 0.5),      # edge falloff;  neutral 0.0
}

AXIS_NEUTRAL = {
    "brightness": 1.0, "temperature": 0.0, "hue": 0.0, "saturation": 1.0,
    "shadow": 0.0, "blur": 0.0, "noise": 0.0, "jpeg": 100,
    "specular": 0.0, "vignette": 0.0,
}

# Metadata columns emitted per image. NaN means "this effect did not fire".
META_COLUMNS = (
    "aug_brightness", "aug_temp_r", "aug_temp_b", "aug_hue", "aug_sat",
    "aug_shadow", "aug_blur", "aug_noise", "aug_jpeg",
    "aug_specular", "aug_vignette", "n_aug",
)

TEMP_HALF_RANGE = 0.25  # r,b scales live in 1.0 +/- this  (matches V2's U(0.75, 1.25))


def probe_values(axis, n_levels=8):
    """Concrete parameter values sweeping `axis` from easiest to hardest.

    Used to build one-factor-at-a-time probe sets. JPEG is reversed because
    lower quality means more degradation.
    """
    lo, hi = AXIS_RANGE[axis]
    vals = np.linspace(lo, hi, n_levels)
    if axis == "jpeg":
        return [int(round(v)) for v in vals]
    return [float(v) for v in vals]


# ── Instrumented primitives ──────────────────────────────────────────────────
# Each returns (image, sampled_parameter). Distributions match V2 exactly.

def _temp_scales_from_signed(t):
    """Map a signed warm/cool scalar in [-1, 1] to (r_scale, b_scale).

    t > 0 warms (boost red, cut blue); t < 0 cools. Keeping this symmetric
    means the derived axis log(r/b) is monotone in t, which is what makes a
    response curve over temperature interpretable.
    """
    return 1.0 + TEMP_HALF_RANGE * t, 1.0 - TEMP_HALF_RANGE * t


def _apply_temperature(img_float, r_scale, b_scale):
    out = img_float.copy()
    out[:, :, 2] = np.clip(out[:, :, 2] * r_scale, 0.0, 1.0)  # R (BGR index 2)
    out[:, :, 0] = np.clip(out[:, :, 0] * b_scale, 0.0, 1.0)  # B (BGR index 0)
    return out


def _apply_hue(img_float, hue_shift_deg):
    img_uint8 = (np.clip(img_float, 0.0, 1.0) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift_deg) % 180.0
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0


def _apply_saturation(img_float, sat_scale):
    img_uint8 = (np.clip(img_float, 0.0, 1.0) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0


def _apply_blur_sigma(img_float, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img_float, sigma=[sigma, sigma, 0])


def _apply_noise_density(img_float, density):
    out = img_float.copy()
    n = int(density * out.shape[0] * out.shape[1])
    rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
    out[rng.integers(0, out.shape[0], n), rng.integers(0, out.shape[1], n)] = 1.0
    out[rng.integers(0, out.shape[0], n), rng.integers(0, out.shape[1], n)] = 0.0
    return out


def _apply_shadow_strength(img_float, strength):
    """Directional shadow at a *fixed* strength (V2 samples it internally)."""
    h, w = img_float.shape[:2]
    direction = random.choice(["left", "right", "top", "bottom"])
    if direction == "left":
        ramp = np.tile(np.linspace(strength, 0.0, w, dtype=np.float32), (h, 1))
    elif direction == "right":
        ramp = np.tile(np.linspace(0.0, strength, w, dtype=np.float32), (h, 1))
    elif direction == "top":
        ramp = np.tile(np.linspace(strength, 0.0, h, dtype=np.float32)[:, None], (1, w))
    else:
        ramp = np.tile(np.linspace(0.0, strength, h, dtype=np.float32)[:, None], (1, w))
    out = img_float * (1.0 - ramp[:, :, None])
    out[:, :, 0] = np.clip(out[:, :, 0] + ramp * 0.12, 0.0, 1.0)  # cool the shadow
    return np.clip(out, 0.0, 1.0)


def _apply_specular_strength(img_float, strength):
    h, w = img_float.shape[:2]
    cx, cy = random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)
    rx, ry = random.randint(w // 8, w // 3), random.randint(h // 8, h // 3)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2
    mask = np.clip(1.0 - dist, 0, 1) ** 2
    return np.clip(img_float + mask[:, :, None] * strength, 0, 1)


def _apply_vignette_strength(img_float, strength):
    h, w = img_float.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((xx - cx) / max(cx, 1)) ** 2 + ((yy - cy) / max(cy, 1)) ** 2)
    return img_float * (1.0 - np.clip(dist * strength, 0, 1))[:, :, None]


# ── Mode 1: stochastic (training distribution) ───────────────────────────────
def apply_lighting(img, p=0.5):
    """V2's augmentation pipeline, instrumented.

    Returns
    -------
    out  : np.ndarray  uint8 BGR
    meta : dict        sampled parameter per axis; NaN where the effect
                       did not fire. Order of effects matches V2.
    """
    meta = {c: float("nan") for c in META_COLUMNS}
    n_aug = 0
    out = img.astype(np.float32) / 255.0

    if random.random() < p:
        factor = _sample_truncated_normal(1.0, 0.35, *AXIS_RANGE["brightness"])
        out = out * factor
        meta["aug_brightness"] = factor
        n_aug += 1

    if random.random() < p:
        r_scale = random.uniform(1 - TEMP_HALF_RANGE, 1 + TEMP_HALF_RANGE)
        b_scale = random.uniform(1 - TEMP_HALF_RANGE, 1 + TEMP_HALF_RANGE)
        out = _apply_temperature(out, r_scale, b_scale)
        meta["aug_temp_r"], meta["aug_temp_b"] = r_scale, b_scale
        n_aug += 1

    # DECOUPLED from saturation (see module header)
    if random.random() < p:
        hue_shift = random.uniform(-25.0, 25.0)
        out = _apply_hue(out, hue_shift)
        meta["aug_hue"] = hue_shift
        n_aug += 1

    if random.random() < p:
        sat_scale = _sample_truncated_normal(1.0, 0.4, *AXIS_RANGE["saturation"])
        out = _apply_saturation(out, sat_scale)
        meta["aug_sat"] = sat_scale
        n_aug += 1

    if random.random() < p:
        strength = random.uniform(*AXIS_RANGE["shadow"])
        out = _apply_shadow_strength(out, strength)
        meta["aug_shadow"] = strength
        n_aug += 1

    if random.random() < p:
        sigma = float(np.clip(abs(np.random.normal(0.0, 0.5)), *AXIS_RANGE["blur"]))
        out = _apply_blur_sigma(out, sigma)
        meta["aug_blur"] = sigma
        n_aug += 1

    if random.random() < p:
        density = random.uniform(*AXIS_RANGE["noise"])
        out = _apply_noise_density(out, density)
        meta["aug_noise"] = density
        n_aug += 1

    # V2 applies specular/vignette at p*0.4 on the float image
    if random.random() < p * 0.4:
        strength = random.uniform(*AXIS_RANGE["specular"])
        out = _apply_specular_strength(out, strength)
        meta["aug_specular"] = strength
        n_aug += 1

    if random.random() < p * 0.4:
        strength = random.uniform(*AXIS_RANGE["vignette"])
        out = _apply_vignette_strength(out, strength)
        meta["aug_vignette"] = strength
        n_aug += 1

    out = (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)

    if random.random() < p:
        quality = random.randint(*AXIS_RANGE["jpeg"])
        out = _augment_jpeg_compression(out, quality_range=(quality, quality))
        meta["aug_jpeg"] = quality
        n_aug += 1

    meta["n_aug"] = n_aug
    return out, meta


# ── Mode 2: single axis (probe sets) ─────────────────────────────────────────
_META_KEY = {
    "brightness": "aug_brightness", "hue": "aug_hue", "saturation": "aug_sat",
    "shadow": "aug_shadow", "blur": "aug_blur", "noise": "aug_noise",
    "jpeg": "aug_jpeg", "specular": "aug_specular", "vignette": "aug_vignette",
}


def apply_single_axis(img, axis, value):
    """Apply exactly ONE axis at `value`; every other axis stays neutral.

    This is what makes a probe set diagnostic. In the training distribution
    ~5 effects compound per image, so a slice like "dark images" is also a
    slice of blurred, shadowed, hue-shifted images and attribution is
    impossible. Varying one axis alone gives a clean response curve.
    """
    if axis is not None and axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}; expected one of {AXES}")

    meta = {c: float("nan") for c in META_COLUMNS}
    meta["n_aug"] = 0 if axis is None else 1
    out = img.astype(np.float32) / 255.0

    if axis is None:                       # neutral control group
        return (np.clip(out, 0, 1) * 255).astype(np.uint8), meta

    if axis == "brightness":
        out = out * value
    elif axis == "temperature":
        r_scale, b_scale = _temp_scales_from_signed(value)
        out = _apply_temperature(out, r_scale, b_scale)
        meta["aug_temp_r"], meta["aug_temp_b"] = r_scale, b_scale
    elif axis == "hue":
        out = _apply_hue(out, value)
    elif axis == "saturation":
        out = _apply_saturation(out, value)
    elif axis == "shadow":
        out = _apply_shadow_strength(out, value)
    elif axis == "blur":
        out = _apply_blur_sigma(out, value)
    elif axis == "noise":
        out = _apply_noise_density(out, value)
    elif axis == "specular":
        out = _apply_specular_strength(out, value)
    elif axis == "vignette":
        out = _apply_vignette_strength(out, value)

    out = (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)

    if axis == "jpeg":
        out = _augment_jpeg_compression(out, quality_range=(int(value), int(value)))

    if axis != "temperature":
        meta[_META_KEY[axis]] = float(value)
    return out, meta
