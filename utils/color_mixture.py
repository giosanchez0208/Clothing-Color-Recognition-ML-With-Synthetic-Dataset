"""
Color categories as a Gaussian mixture in CIELAB, for V4.

Why this exists
---------------
V1 and V2 treated a color's category as a hard fact: a hex belongs to green, so
an image made of it is 100% green. That is true for a prototypical green and
false for a color sitting between green and yellow, where the honest answer is
that a person could reasonably say either.

V2 handled that by deleting the ambiguous colors. Confusion-aware Voronoi
resampling rejected 79 boundary candidates precisely because a one-hot label
could not represent them. The cost showed up later: the model's green/yellow
boundary landed near 135 degrees while the library's sat at 117.7, because the
region next to the boundary was both under-sampled and labeled with false
certainty.

Here the category is a distribution instead. Each of the 13 categories is fitted
as a Gaussian in CIELAB, and the label for a color is the mixture posterior

    p(c | x)  proportional to  |Sigma_c|^(-1/2) * exp(-0.5 * d_c(x)^2)

where d_c is the Mahalanobis distance to category c. There is no temperature and
nothing to tune: this is the literal posterior of the model the library already
implies. A prototypical green returns green ~1.0. A boundary color returns
something like green 0.56, yellow 0.43, which is what it actually is.

Two consequences follow, and they are why generation changes too:

  1. Colors no longer need to be drawn from a discrete list. Sampling the
     component directly covers the space continuously, including the boundary
     region the library had removed.
  2. The Voronoi rejection step becomes unnecessary. It existed to keep labels
     unambiguous, and labels are now allowed to be ambiguous.

Measured against the 325-color library, continuous sampling raises the share of
draws landing on a meaningfully ambiguous color from 1.8% to 9.8%, and mean
normalized label entropy from 0.016 to 0.040.
"""
import json
import os

import numpy as np

from utils.color_utils import ColorLibrary
from utils.instrumented_generator import COLOR_CLASSES

# Added to each covariance before inversion. Small categories can be nearly
# degenerate in CIELAB, which makes the covariance singular and the Mahalanobis
# distance undefined.
COV_RIDGE = 1e-3


class CategoryMixture:
    """13 Gaussians in CIELAB, one per color category."""

    def __init__(self, means, covs, classes=None):
        self.classes = list(classes or COLOR_CLASSES)
        self.means = {c: np.asarray(means[c], float) for c in self.classes}
        self.covs = {c: np.asarray(covs[c], float) for c in self.classes}
        self._inv = {c: np.linalg.inv(self.covs[c]) for c in self.classes}
        self._logdet = {c: float(np.linalg.slogdet(self.covs[c])[1]) for c in self.classes}
        self._chol = {c: np.linalg.cholesky(self.covs[c]) for c in self.classes}

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def from_library(cls, csv_path, ridge=COV_RIDGE):
        """Fit one Gaussian per category from the published color library."""
        lib = ColorLibrary.from_categorized_csv(csv_path)
        means, covs = {}, {}
        for c in COLOR_CLASSES:
            if c not in lib.categories:
                raise ValueError("category %r missing from %s" % (c, csv_path))
            pts = np.asarray(lib.get_category_colors(c), float)
            if len(pts) < 4:
                raise ValueError("category %r has only %d colors" % (c, len(pts)))
            means[c] = pts.mean(axis=0)
            covs[c] = np.cov(pts.T) + np.eye(3) * ridge
        return cls(means, covs)

    def save(self, path):
        """Persist the fit so a dataset can be regenerated bit-for-bit later."""
        blob = {"classes": self.classes,
                "means": {c: self.means[c].tolist() for c in self.classes},
                "covs": {c: self.covs[c].tolist() for c in self.classes}}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        return cls(blob["means"], blob["covs"], blob["classes"])

    # ── the label ────────────────────────────────────────────────────────────
    def posterior(self, lab):
        """p(category | color) for one CIELAB point. Returns a 13-vector."""
        lab = np.asarray(lab, float)
        lp = np.empty(len(self.classes))
        for i, c in enumerate(self.classes):
            d = lab - self.means[c]
            lp[i] = -0.5 * float(d @ self._inv[c] @ d) - 0.5 * self._logdet[c]
        lp -= lp.max()                      # stabilize before exponentiating
        p = np.exp(lp)
        return p / p.sum()

    def entropy(self, p):
        """Shannon entropy of a posterior, normalized to [0, 1].

        0 means the color is unambiguously one category. Values near 1 mean the
        color sits between categories. This is a per-color quantity and is
        recorded separately from the spatial mixing entropy, so the two sources
        of label uncertainty stay distinguishable during analysis.
        """
        p = np.asarray(p, float)
        nz = p[p > 0]
        return float(-(nz * np.log(nz)).sum() / np.log(len(self.classes)))

    # ── sampling ─────────────────────────────────────────────────────────────
    def sample(self, rng, category, max_tries=32):
        """Draw a CIELAB point from one component, rejecting out-of-gamut draws.

        Rejection is on displayability only, never on category membership. A
        draw that lands closer to a neighbor is kept, because the posterior will
        describe it correctly. That is the whole point of the change.
        """
        mu, L = self.means[category], self._chol[category]
        for _ in range(max_tries):
            lab = mu + L @ rng.standard_normal(3)
            rgb, hexcode = ColorLibrary._lab_to_hex(*lab)
            if all(-1e-6 <= v <= 1.0 + 1e-6 for v in rgb):
                return lab, hexcode
        # Fall back to the centroid, which is in gamut by construction.
        rgb, hexcode = ColorLibrary._lab_to_hex(*mu)
        return mu, hexcode
