"""
V4 generator: colors drawn from the mixture, labels are the mixture posterior.

What changes from V3
--------------------
V3 drew a hex from the 325-color library, weighting by inverse Mahalanobis
distance squared, and labeled the result one-hot by whichever category the hex
came from. Two problems compounded:

  1. The weighting concentrated draws at category centroids, so only 1.8% of
     draws landed on a color a person would find ambiguous.
  2. Those rare boundary colors were still labeled with full certainty, which
     taught the model that a color halfway between green and yellow is
     definitively green.

The model's green/yellow decision boundary duly landed near 135 degrees of hue
while the library's sat at 117.7, and a library green fed back in was classified
yellow at 96%.

V4 samples CIELAB continuously from each fitted Gaussian and labels by the
mixture posterior. A prototypical green still gets green ~1.0. A boundary color
gets something like green 0.56, yellow 0.43, which is what it is. Measured over
4,000 simulated garments, mean categorical entropy rises from 0.0164 to 0.0417
and the share of ambiguous draws from 1.8% to about 10%.

Note that continuous sampling and soft labels are one coupled change rather than
two independent ones. Continuous sampling has no meaning under hard labels,
because there would be no category to assign to a point between two components.
This run therefore cannot attribute its result to one half or the other.

The Voronoi rejection that produced the 325-color library is not used here. It
existed to keep labels unambiguous, and labels are now allowed to be ambiguous.
The library still defines the Gaussians, via utils/color_mixture.py.
"""
import os
import random

import cv2
import numpy as np

from utils.color_mixture import CategoryMixture
from utils.color_utils import ColorLibrary
from utils.instrumented_augment import apply_lighting, apply_single_axis
from utils.instrumented_generator import (COLOR_CLASSES, NUM_CLASSES,
                                          InstrumentedGenerator, _RecordingInnerGenerator,
                                          label_entropy, seed_for)

DEFAULT_DIMENSIONS = (224, 224)


class _MixtureInnerGenerator(_RecordingInnerGenerator):
    """Inner-square generator whose colors are points, not library entries.

    generate_random_color keeps the parent's (key, hex) contract so the pattern
    painters and the distinct-color sampler work unchanged. The key is now the
    CIELAB point rather than a category name, which means the parent's
    label_map bookkeeping accumulates pixel fractions per COLOR instead of per
    category. Converting those fractions into a label is then one weighted sum
    of posteriors.
    """

    def __init__(self, csv_path, mixture, dimensions=DEFAULT_DIMENSIONS):
        super().__init__(csv_path=csv_path, dimensions=dimensions)
        self.mixture = mixture
        self.rng = np.random.default_rng(0)   # replaced per image by the caller

    def generate_random_color(self):
        category = random.choice(COLOR_CLASSES)
        lab, hexcode = self.mixture.sample(self.rng, category)
        return tuple(float(v) for v in lab), hexcode

    def soft_label(self, labs_by_index, label_map):
        """Pixel-fraction-weighted mixture of per-color posteriors.

        Spatial mixture and categorical ambiguity compose linearly: a shirt that
        is 60% of a color reading green 0.9 / yellow 0.1 and 40% of a solid
        white contributes 0.54 green, 0.06 yellow, 0.40 white.
        """
        total = float(label_map.size)
        idx, counts = np.unique(label_map, return_counts=True)
        vec = np.zeros(NUM_CLASSES, dtype=np.float64)
        cat_H = 0.0
        for i, n in zip(idx.tolist(), counts.tolist()):
            lab = labs_by_index[int(i)]
            p = self.mixture.posterior(np.asarray(lab, float))
            frac = n / total
            vec += frac * p
            cat_H += frac * self.mixture.entropy(p)
        s = vec.sum()
        vec = vec / s if s > 0 else np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        return vec, float(cat_H)

    def generate(self):
        self.last = {"pattern": None, "n_colors": 0,
                     "fold_blend": "none", "fold_alpha": 0.0}

        lab, color_hex = self.generate_random_color()
        labs = [lab]
        h = int(self.dimensions[0] / 2)
        w = int(self.dimensions[1] / 2)
        inner = np.full((h, w, 3), self._hex_to_bgr(color_hex), dtype=np.uint8)
        label_map = np.zeros((h, w), dtype=np.uint16)
        inner, labs, label_map = self.add_pattern(inner, labs, label_map, [color_hex])

        vec, cat_H = self.soft_label(labs, label_map)
        self.last["categorical_entropy"] = cat_H
        if random.random() < 0.85:
            inner = self.apply_synthetic_fold_texture(inner)
        return inner, vec


class MixtureGenerator(InstrumentedGenerator):
    """InstrumentedGenerator with the V4 color model. Same images otherwise."""

    def __init__(self, csv_path, path_to_bgs, mixture, dimensions=DEFAULT_DIMENSIONS,
                 bg_pool=None):
        super().__init__(csv_path=csv_path, path_to_bgs=path_to_bgs,
                         dimensions=dimensions, bg_pool=bg_pool)
        self.mixture = mixture
        self.inner = _MixtureInnerGenerator(csv_path=csv_path, mixture=mixture,
                                            dimensions=dimensions)

    def generate(self, index, base_seed, probe=False, axis=None, value=None):
        s = seed_for(base_seed, index)
        random.seed(s)
        np.random.seed(s)
        # Explicit Generator seeded from the same value. default_rng(None) would
        # draw from OS entropy and ignore the seeds above, which is exactly the
        # defect that made the V2 fold texture irreproducible.
        self.inner.rng = np.random.default_rng(s)

        background, bg_name = self._background()
        inner, vec = self.inner.generate()

        composed = background.copy()
        bh, bw = composed.shape[:2]
        ih, iw = inner.shape[:2]
        top, left = (bh - ih) // 2, (bw - iw) // 2
        composed[top:top + ih, left:left + iw] = inner

        if probe:
            composed, aug_meta = apply_single_axis(composed, axis, value)
        else:
            composed, aug_meta = apply_lighting(composed)

        vec = [float(v) for v in vec]
        meta = dict(aug_meta)
        meta.update({
            "background": bg_name,
            "seed": s,
            "probe_axis": axis if axis is not None else "",
            "probe_value": value if value is not None else float("nan"),
            "pattern": self.inner.last.get("pattern") or "solid",
            "n_colors": self.inner.last.get("n_colors", 1),
            # Total label entropy, spatial mixing and color ambiguity combined.
            "label_entropy": label_entropy(vec),
            # Color ambiguity alone, so the two sources stay separable in
            # analysis. A solid garment of a boundary color has near-zero
            # spatial entropy and high categorical entropy.
            "categorical_entropy": self.inner.last.get("categorical_entropy", 0.0),
            "fold_blend": self.inner.last.get("fold_blend", "none"),
            "fold_alpha": self.inner.last.get("fold_alpha", 0.0),
        })
        return composed, vec, meta
