"""
Figure for the V4 soft-label change.

Left  : the a*/b* plane of CIELAB with each category drawn as a 1-sigma
        confidence ellipse of its fitted Gaussian, and the example colors
        marked where they actually sit. Overlapping ellipses are the whole
        point: category regions are not disjoint, so a color landing in an
        overlap genuinely belongs to more than one.
Right : the mixture posterior for each example. A prototypical color returns
        one category near 1.0; a color in an overlap returns a split.

Under V3 every one of these colors would have been labeled 100% of a single
category, because the label came from whichever library bin the hex was drawn
from rather than from where the color sits.

Usage
-----
  python scripts/make_mixture_figure.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.color_mixture import CategoryMixture
from utils.color_utils import ColorLibrary
from utils.instrumented_generator import COLOR_CLASSES

# a*/b* offsets in points, for labels that would otherwise collide.
NEUTRAL_OFFSET = {"white": (-34, 18), "gray": (-42, 0), "black": (-30, -18)}
# Examples 3 and 4 are greens a few units apart, so one gets a leader line.
MARKER_OFFSET = {3: (26, 16)}

MIX = os.path.join(PROJECT_ROOT, "datasets", "category_mixture_v4.json")
CSV = os.path.join(PROJECT_ROOT, "datasets", "categorized_colors_normalized.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "assets", "figures")

# Chosen to span the range: unambiguous, mild, two-way, and a three-way tie.
EXAMPLES = [
    ("#6fac95", "prototypical green",     "a color near a category center"),
    ("#77572d", "warm brown",             "mildly ambiguous"),
    ("#7f915a", "the sweater",            "the real photo that started this"),
    ("#8b9a5f", "boundary green",         "a library green the V3 model called yellow"),
    ("#783d41", "muted red",              "two-way split"),
    ("#d6c5ba", "pale warm neutral",      "three-way tie"),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mix = CategoryMixture.load(MIX) if os.path.exists(MIX) else CategoryMixture.from_library(CSV)
    swatch = {c: ColorLibrary._lab_to_hex(*mix.means[c])[1] for c in COLOR_CLASSES}

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(len(EXAMPLES), 2, width_ratios=[1.05, 1.0],
                          wspace=0.18, hspace=0.55)

    # ── left: the a*/b* plane with 1-sigma ellipses ──────────────────────────
    ax = fig.add_subplot(gs[:, 0])
    for c in COLOR_CLASSES:
        mu, S = mix.means[c], mix.covs[c]
        sub = S[np.ix_([1, 2], [1, 2])]            # a*, b* block
        vals, vecs = np.linalg.eigh(sub)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        w, h = 2 * np.sqrt(np.maximum(vals, 1e-9))
        ax.add_patch(Ellipse((mu[1], mu[2]), w, h, angle=ang,
                             facecolor=swatch[c], alpha=0.30,
                             edgecolor=swatch[c], linewidth=1.6))
        # white, gray and black share the a*/b* origin and separate only in
        # L*, which this projection does not show. Fan the labels out.
        off = NEUTRAL_OFFSET.get(c)
        if off is None:
            ax.annotate(c, (mu[1], mu[2]), fontsize=8, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7,
                                  ec="none"))
        else:
            ax.annotate(c, (mu[1], mu[2]), xytext=off, textcoords="offset points",
                        fontsize=8, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.9,
                                  ec="none"),
                        arrowprops=dict(arrowstyle="-", lw=0.7, color="#888"))

    for i, (hx, name, _) in enumerate(EXAMPLES):
        L, a, b = ColorLibrary._hex_to_lab(hx)
        ax.scatter([a], [b], s=210, c=[hx], edgecolors="black", linewidths=1.6, zorder=5)
        # A pale swatch with a white numeral is unreadable, so contrast
        # follows the swatch lightness.
        dx, dy = MARKER_OFFSET.get(i, (0, 0))
        if dx or dy:
            ax.annotate(str(i + 1), (a, b), xytext=(dx, dy),
                        textcoords="offset points", fontsize=9, fontweight="bold",
                        color="black", ha="center", va="center", zorder=6,
                        bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec="black",
                                  lw=1.0),
                        arrowprops=dict(arrowstyle="-", lw=0.9, color="black"))
        else:
            ax.annotate(str(i + 1), (a, b), fontsize=9, fontweight="bold",
                        color=("black" if L > 55 else "white"),
                        ha="center", va="center", zorder=6)

    ax.set_xlabel("a*   (green to red)")
    ax.set_ylabel("b*   (blue to yellow)")
    ax.set_title(chr(10).join([
        "Category regions overlap. Each ellipse is one standard deviation",
        "of a fitted Gaussian, projected onto a* and b*.",
        "White, gray and black separate in lightness, not shown here.",
    ]), fontsize=10)
    ax.set_xlim(-70, 80)
    ax.set_ylim(-70, 90)

    # ── right: one posterior per example ─────────────────────────────────────
    order = list(COLOR_CLASSES)
    for i, (hx, name, note) in enumerate(EXAMPLES):
        axr = fig.add_subplot(gs[i, 1])
        p = mix.posterior(ColorLibrary._hex_to_lab(hx))
        H = mix.entropy(p)
        axr.bar(range(len(order)), [p[COLOR_CLASSES.index(c)] for c in order],
                color=[swatch[c] for c in order], edgecolor="#333", linewidth=0.5)
        axr.set_ylim(0, 1.0)
        axr.set_xticks(range(len(order)))
        axr.set_xticklabels(order if i == len(EXAMPLES) - 1 else [],
                            rotation=45, ha="right", fontsize=7.5)
        axr.set_yticks([0, 0.5, 1.0])
        axr.set_yticklabels(["0", "50%", "100%"], fontsize=7.5)
        axr.tick_params(axis="y", labelsize=7.5)

        top = sorted(zip(COLOR_CLASSES, p), key=lambda kv: -kv[1])[:3]
        txt = ", ".join("%s %.0f%%" % (c, v * 100) for c, v in top if v >= 0.02)
        axr.set_title("%d.  %s   %s        %s        entropy %.2f"
                      % (i + 1, name, hx, txt, H), fontsize=8.5, loc="left")
        # The swatch itself, so the reader can see the color being described.
        axr.add_patch(plt.Rectangle((1.012, 0), 0.055, 1.0, transform=axr.transAxes,
                                    clip_on=False, facecolor=hx, edgecolor="black",
                                    linewidth=0.8))

    fig.suptitle("V4 labels a color by where it sits, not by which bin it came from",
                 fontsize=12, y=0.975)
    fig.text(0.5, 0.012,
             "Under V3 every color above was labeled 100% of a single category. "
             "The V4 label is the Gaussian mixture posterior, which has no free parameters.",
             ha="center", fontsize=9, color="#444")

    out = os.path.join(OUT_DIR, "mixture_labels.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print("  %s" % out)


if __name__ == "__main__":
    main()
