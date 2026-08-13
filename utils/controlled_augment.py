# =============================================================================
# controlled_augment.py — live photometric augmentation the trainer can steer
# =============================================================================
# The problem this solves
# -----------------------
# V2 applied torchvision ColorJitter + RandomColorTemperature at load time.
# Measured on the real data, that live layer carried ~70% of the photometric
# variance the model actually saw, while only the *baked* layer is recorded in
# labels.csv -- correlation between recorded and actual brightness was 0.55.
# So the metadata described under a third of reality, and any curriculum that
# steered by it would be steering with a mostly-disconnected wheel.
#
# Run A is the ablation arm with the live layer removed. It confirms the
# metadata claim (recorded parameters then describe everything the model sees)
# and quantifies what the layer was contributing: the train/val gap widens
# monotonically from 0.048 at epoch 10 to 0.264 at epoch 80 -- substantial
# regularisation.
#
# The resolution
# --------------
# Keep the live layer, but make the TRAINER own its intensity instead of
# torchvision. Strength per axis is set by the training loop, so the loop knows
# exactly what was applied -- no logging required, because the controller *is*
# the record. That turns a blind spot into the curriculum's primary lever:
# "the probe says hue is weak" becomes "raise hue strength", directly, rather
# than the indirect and diluted trick of oversampling images that happen to
# carry a large baked hue shift.
#
# Axes deliberately mirror the probe axes, so the diagnostic and the lever
# speak the same language.
# =============================================================================

import random

import cv2
import numpy as np

# Full-strength range per axis (strength 1.0 reproduces V2's magnitudes:
# ColorJitter(brightness=0.6, saturation=0.6, hue=0.07~=25deg) and
# RandomColorTemperature(0.25)).
AXIS_MAX = {
    "brightness": 0.6,    # factor = 1 +/- 0.6
    "saturation": 0.6,    # factor = 1 +/- 0.6
    "hue": 25.0,          # degrees
    "temperature": 1.0,   # signed; maps to r/b scales of 1 +/- 0.25
}
TEMP_HALF = 0.25

DEFAULT_STRENGTHS = {k: 1.0 for k in AXIS_MAX}


class ControlledPhotometric:
    """Per-axis photometric augmentation on an RGB uint8 array.

    strengths : {axis: 0.0-1.0}. 0 disables that axis entirely; 1.0 matches
                the V2 live-augmentation magnitude.
    p         : probability each axis fires.

    Deliberately a module-level class, not a closure: Windows DataLoader
    workers use spawn and pickle the transform, and a local lambda is not
    picklable.
    """

    def __init__(self, strengths=None, p=0.5):
        self.strengths = dict(DEFAULT_STRENGTHS)
        if strengths:
            self.strengths.update(strengths)
        self.p = p

    def describe(self):
        return " ".join(f"{k[:4]}={v:.2f}" for k, v in sorted(self.strengths.items()))

    def __call__(self, rgb_uint8):
        out = rgb_uint8.astype(np.float32) / 255.0

        s = self.strengths.get("brightness", 0.0)
        if s > 0 and random.random() < self.p:
            out = out * (1.0 + random.uniform(-1, 1) * AXIS_MAX["brightness"] * s)

        s = self.strengths.get("temperature", 0.0)
        if s > 0 and random.random() < self.p:
            t = random.uniform(-1, 1) * s
            out[:, :, 0] = np.clip(out[:, :, 0] * (1.0 + TEMP_HALF * t), 0, 1)  # R
            out[:, :, 2] = np.clip(out[:, :, 2] * (1.0 - TEMP_HALF * t), 0, 1)  # B

        # Hue and saturation stay behind SEPARATE gates. Sharing one gate is
        # exactly the V1/V2 defect that made the two axes mutually
        # unattributable (correlation +1.0).
        do_hue = self.strengths.get("hue", 0.0) > 0 and random.random() < self.p
        do_sat = self.strengths.get("saturation", 0.0) > 0 and random.random() < self.p
        if do_hue or do_sat:
            u8 = (np.clip(out, 0, 1) * 255).astype(np.uint8)
            hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
            if do_hue:
                shift = random.uniform(-1, 1) * AXIS_MAX["hue"] * self.strengths["hue"]
                hsv[:, :, 0] = (hsv[:, :, 0] + shift * 0.5) % 180.0  # OpenCV H is 0-180
            if do_sat:
                f = 1.0 + random.uniform(-1, 1) * AXIS_MAX["saturation"] * self.strengths["saturation"]
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * f, 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        return (np.clip(out, 0, 1) * 255).astype(np.uint8)
