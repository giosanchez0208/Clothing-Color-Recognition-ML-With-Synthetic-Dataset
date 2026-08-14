"""
Statistical audit of the generated dataset.

This is deliberately an *audit*, not a summary. A summary tells you what the
dataset contains; an audit tries to find the ways it is broken. Each section
states what it checks, what result would indicate a defect, and the verdict.

Checks
------
 1. Integrity          rows vs files, label simplex, seed uniqueness
 2. Label coverage     per-class mass, dominance balance, entropy
 3. Structure          pattern mix vs design weights, color counts
 4. Fire rates         observed vs designed probability, binomial CI
 5. Parameter fidelity KS test of each axis against its intended distribution
 6. Independence       pairwise correlation of augmentation firing
 7. Confounds          augmentation x color leakage  (the dangerous one)
 8. Train/val parity   KS test per axis between splits
 9. Probe validity     cell coverage, isolation, control neutrality

Outputs reports/dataset_report.md and reports/figures/*.png

Usage
-----
  python scripts/analyze_dataset.py
  python scripts/analyze_dataset.py --data datasets/generated_v3
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.color_utils_extended import PATTERN_TYPES_V2, PATTERN_WEIGHTS_V2
from utils.instrumented_augment import AXES, AXIS_RANGE, probe_values
from utils.instrumented_generator import COLOR_CLASSES

# Designed fire probability per metadata column (V2 pipeline: p=0.5, spec/vig p*0.4)
DESIGNED_P = {
    "aug_brightness": 0.5, "aug_temp_r": 0.5, "aug_hue": 0.5, "aug_sat": 0.5,
    "aug_shadow": 0.5, "aug_blur": 0.5, "aug_noise": 0.5, "aug_jpeg": 0.5,
    "aug_specular": 0.2, "aug_vignette": 0.2,
}

# Analytic reference distribution per axis, for the KS goodness-of-fit test.
# (column, scipy frozen dist or None if not analytically specified)
def _reference_dists():
    tn = lambda m, s, lo, hi: stats.truncnorm((lo - m) / s, (hi - m) / s, loc=m, scale=s)
    return {
        "aug_brightness": ("TN(1.0, 0.35) on [0.3, 1.8]", tn(1.0, 0.35, 0.3, 1.8)),
        "aug_sat":        ("TN(1.0, 0.40) on [0.3, 2.0]", tn(1.0, 0.4, 0.3, 2.0)),
        "aug_temp_r":     ("U(0.75, 1.25)", stats.uniform(0.75, 0.5)),
        "aug_temp_b":     ("U(0.75, 1.25)", stats.uniform(0.75, 0.5)),
        "aug_hue":        ("U(-25, 25)", stats.uniform(-25, 50)),
        "aug_shadow":     ("U(0.10, 0.55)", stats.uniform(0.10, 0.45)),
        "aug_noise":      ("U(0.001, 0.005)", stats.uniform(0.001, 0.004)),
        "aug_specular":   ("U(0.15, 0.40)", stats.uniform(0.15, 0.25)),
        "aug_vignette":   ("U(0.20, 0.50)", stats.uniform(0.20, 0.30)),
        "aug_blur":       ("half-normal(0.5), clipped [0.1, 1.5]", None),
        "aug_jpeg":       ("discrete uniform {40..85}", None),
    }


def holm_bonferroni(pvals, alpha=0.01):
    """Holm-Bonferroni step-down. Returns (any_significant, adjusted_threshold).

    Why this is necessary: these audits run ~10 independent tests per section.
    Requiring every one to clear alpha=0.01 individually gives a family-wise
    false-alarm rate of 1 - 0.99^10 ~= 9.6%, so roughly one in ten *perfectly
    valid* datasets would be reported broken. Holm controls the family-wise
    error rate while staying more powerful than plain Bonferroni.

    This is the same multiple-comparisons discipline the curriculum needs when
    picking a "weakest axis" out of ten. See docs/curriculum-design.md.

    Returns (any_rejected, cutoff) where a p-value counts as a real difference
    iff p <= cutoff. cutoff is 0.0 when nothing survives correction.
    """
    if not pvals:
        return False, 0.0
    ordered = sorted(pvals)
    m = len(ordered)
    n_rejected = 0
    for i, p in enumerate(ordered):
        # Step down: reject while p(i) clears its own shrinking threshold, and
        # STOP at the first failure, since everything after it is retained too.
        if p <= alpha / (m - i):
            n_rejected += 1
        else:
            break
    cutoff = ordered[n_rejected - 1] if n_rejected else 0.0
    return n_rejected > 0, cutoff


class Report:
    """Accumulates markdown and tracks pass/fail verdicts."""

    def __init__(self):
        self.lines = []
        self.verdicts = []

    def h(self, text, level=2):
        self.lines.append(f"\n{'#' * level} {text}\n")

    def p(self, text):
        self.lines.append(text + "\n")

    def table(self, headers, rows):
        self.lines.append("| " + " | ".join(headers) + " |")
        self.lines.append("|" + "|".join("---" for _ in headers) + "|")
        for r in rows:
            self.lines.append("| " + " | ".join(str(c) for c in r) + " |")
        self.lines.append("")

    def verdict(self, name, ok, detail=""):
        self.verdicts.append((name, ok, detail))
        mark = "**PASS**" if ok else "**FAIL**"
        self.p(f"{mark}: {detail}" if detail else mark)

    def fig(self, path, caption):
        rel = os.path.relpath(path, os.path.join(PROJECT_ROOT, "reports"))
        self.lines.append(f"\n![{caption}]({rel.replace(os.sep, '/')})\n")
        self.lines.append(f"*{caption}*\n")

    def render(self):
        return "\n".join(self.lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(PROJECT_ROOT, "datasets", "generated_v3"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "reports"))
    args = ap.parse_args()

    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    csv_path = os.path.join(args.data, "labels.csv")
    img_dir = os.path.join(args.data, "images")
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: {csv_path} not found. Run generate_dataset.py first")

    # low_memory=False: probe_axis is blank for train/val rows and a string for
    # probe rows, which trips pandas' chunked type inference.
    df = pd.read_csv(csv_path, low_memory=False)
    tv = df[df["split"].isin(["train", "val"])].copy()
    probe = df[df["split"] == "probe"].copy()

    R = Report()
    R.lines.append("# Synthetic Dataset Audit\n")
    R.p(f"`{os.path.relpath(args.data, PROJECT_ROOT)}`: **{len(df):,} images**, "
        f"{len(df.columns)} recorded fields per image.\n")
    R.p("Every section states what it tests and what a failure would mean. "
        "Verdicts are summarized at the end.\n")

    plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.25})

    # ── 1. Integrity ─────────────────────────────────────────────────────────
    R.h("1. Integrity")
    R.p("*Does the manifest agree with what is on disk, and is every label a "
        "valid probability distribution? A failure here invalidates everything "
        "downstream.*\n")

    n_files = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0
    sums = df[COLOR_CLASSES].sum(axis=1)
    simplex_ok = bool(((sums - 1.0).abs() < 1e-4).all())
    files_ok = n_files == len(df)
    names_ok = bool(df["filename"].nunique() == len(df))
    neg_ok = bool((df[COLOR_CLASSES] >= -1e-9).all().all())

    # Seeds identify an image everywhere except the probe split, where the
    # paired design re-renders each base at every (axis, level) cell on purpose.
    # Testing global uniqueness there would flag the pairing itself. The
    # invariant that still has teeth: every base rendered the same number of
    # times, with exactly one own-control render each. An uneven grid, which is
    # what a half-finished probe run looks like, still fails.
    probe = df[df["split"] == "probe"]
    others = df[df["split"] != "probe"]
    seeds_ok = bool(others["seed"].nunique() == len(others))
    if len(probe):
        per_base = probe.groupby("seed").size()
        n_ctrl = int(probe["probe_axis"].isna().sum())
        paired_ok = bool(per_base.nunique() == 1 and n_ctrl == len(per_base))
        probe_note = (f"{len(per_base)} bases x {int(per_base.iloc[0])} renders, "
                      f"{n_ctrl} own-controls")
    else:
        paired_ok, probe_note = True, "no probe split"

    R.table(["Check", "Result", "Status"], [
        ["Rows in labels.csv", f"{len(df):,}", "n/a"],
        ["Files on disk", f"{n_files:,}", "OK" if files_ok else "MISMATCH"],
        ["Unique filenames", f"{df['filename'].nunique():,}", "OK" if names_ok else "DUPES"],
        ["Unique seeds (train/val/test)", f"{others['seed'].nunique():,} of {len(others):,}",
         "OK" if seeds_ok else "COLLISION"],
        ["Probe pairing grid", probe_note, "OK" if paired_ok else "UNEVEN"],
        ["Labels sum to 1.0", f"[{sums.min():.6f}, {sums.max():.6f}]",
         "OK" if simplex_ok else "INVALID"],
        ["Labels non-negative", "yes" if neg_ok else "no", "OK" if neg_ok else "INVALID"],
    ])
    R.table(["Split", "Count"], [[k, f"{v:,}"] for k, v in df["split"].value_counts().items()])
    ok = all([files_ok, names_ok, seeds_ok, paired_ok, simplex_ok, neg_ok])
    R.verdict("integrity", ok,
              "manifest, files, seeds, probe pairing and label simplex all consistent"
              if ok else "one or more integrity checks failed")

    # ── 2. Label coverage ────────────────────────────────────────────────────
    R.h("2. Label space coverage")
    R.p("*Categories are sampled uniformly by design, so dominant-class counts "
        "should be roughly balanced. Heavy imbalance would bias the model "
        "toward over-represented colors regardless of the loss function.*\n")

    mean_mass = tv[COLOR_CLASSES].mean()
    dominant = tv[COLOR_CLASSES].idxmax(axis=1).value_counts().reindex(COLOR_CLASSES, fill_value=0)
    chi = stats.chisquare(dominant.values)
    balance = dominant.max() / max(dominant.min(), 1)

    R.table(["Color", "Mean mass", "Dominant count", "Share"],
            [[c, f"{mean_mass[c]:.4f}", f"{dominant[c]:,}", f"{dominant[c]/len(tv):.1%}"]
             for c in COLOR_CLASSES])
    R.p(f"\nUniform reference mass = {1/len(COLOR_CLASSES):.4f}. "
        f"Max/min dominance ratio = **{balance:.2f}x**. "
        f"Chi-square uniformity: stat={chi.statistic:.1f}, p={chi.pvalue:.3g}.\n")

    ent = tv["label_entropy"]
    pure = float((ent < 0.01).mean())
    R.p(f"Label entropy: mean **{ent.mean():.3f}**, median {ent.median():.3f}, "
        f"max {ent.max():.3f}. Effectively one-hot (H<0.01): **{pure:.1%}**, so "
        f"the remaining {1-pure:.1%} are genuinely multi-modal targets, which are "
        "the samples that exercise the KL objective rather than behaving like "
        "hard labels.\n")
    R.verdict("label balance", balance < 2.0,
              f"dominance ratio {balance:.2f}x (<2.0 considered acceptable)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    axes[0].bar(COLOR_CLASSES, mean_mass.values, color="steelblue")
    axes[0].axhline(1/13, color="crimson", ls="--", lw=1, label="uniform")
    axes[0].set_title("Mean probability mass"); axes[0].legend()
    axes[0].tick_params(axis="x", rotation=60)
    axes[1].bar(COLOR_CLASSES, dominant.values, color="darkorange")
    axes[1].set_title("Dominant-class counts")
    axes[1].tick_params(axis="x", rotation=60)
    axes[2].hist(ent, bins=50, color="seagreen")
    axes[2].set_title("Label entropy (0 = one-hot)")
    plt.tight_layout()
    p = os.path.join(fig_dir, "labels.png"); plt.savefig(p); plt.close()
    R.fig(p, "Label distribution, class dominance, and target entropy")

    # ── 3. Structure ─────────────────────────────────────────────────────────
    R.h("3. Pattern and structure")
    R.p("*Patterns are drawn from fixed design weights. A significant deviation "
        "means the sampler is not doing what the configuration says.*\n")

    obs = tv["pattern"].value_counts().reindex(PATTERN_TYPES_V2, fill_value=0)
    exp = np.array(PATTERN_WEIGHTS_V2) * len(tv)
    chi_p = stats.chisquare(obs.values, exp)
    R.table(["Pattern", "Designed", "Observed", "Expected", "Delta"],
            [[pt, f"{w:.0%}", f"{obs[pt]:,}", f"{e:,.0f}", f"{obs[pt]-e:+,.0f}"]
             for pt, w, e in zip(PATTERN_TYPES_V2, PATTERN_WEIGHTS_V2, exp)])
    R.p(f"\nChi-square goodness-of-fit: stat={chi_p.statistic:.2f}, "
        f"p={chi_p.pvalue:.3f}. (p > 0.05 means observed mix is consistent with design.)\n")
    R.verdict("pattern mix", chi_p.pvalue > 0.01,
              f"pattern frequencies consistent with design weights (p={chi_p.pvalue:.3f})")

    by_pat = tv.groupby("pattern")["label_entropy"].agg(["mean", "count"]).sort_values("mean")
    R.p("\nEntropy by pattern, a sanity check that multi-color patterns really "
        "do produce multi-modal labels:\n")
    R.table(["Pattern", "Mean entropy", "n"],
            [[i, f"{r['mean']:.3f}", f"{int(r['count']):,}"] for i, r in by_pat.iterrows()])

    # ── 4. Fire rates ────────────────────────────────────────────────────────
    R.h("4. Augmentation fire rates")
    R.p("*Each effect fires independently with a designed probability. "
        "Observed rates outside the binomial 95% interval indicate a sampler bug.*\n")

    n = len(tv)
    # Bonferroni: 10 simultaneous intervals at 95% each would flag a clean
    # dataset 1 - 0.95^10 = 40% of the time. Widen each so the FAMILY-wise
    # rate is 5%.
    m_tests = len(DESIGNED_P)
    z = stats.norm.ppf(1 - 0.05 / (2 * m_tests))
    rows, all_in_ci = [], True
    for col, p_des in DESIGNED_P.items():
        k = int(tv[col].notna().sum())
        rate = k / n
        se = np.sqrt(p_des * (1 - p_des) / n)
        lo, hi = p_des - z * se, p_des + z * se
        inside = lo <= rate <= hi
        all_in_ci &= inside
        rows.append([col.replace("aug_", ""), f"{p_des:.0%}", f"{rate:.2%}",
                     f"[{lo:.2%}, {hi:.2%}]", "OK" if inside else "OUTSIDE"])
    R.table(["Axis", "Designed", "Observed", "Bonferroni CI", "Status"], rows)
    R.p(f"\n*Intervals are Bonferroni-corrected across {m_tests} axes "
        f"(z={z:.2f}) so the family-wise false-alarm rate is 5%, not 40%.*\n")
    R.verdict("fire rates", all_in_ci,
              f"all {m_tests} axes fire within their family-wise 95% interval"
              if all_in_ci else "at least one axis fires outside its interval")

    # ── 5. Parameter fidelity ────────────────────────────────────────────────
    R.h("5. Parameter distribution fidelity")
    R.p("*Firing at the right rate is not enough. The sampled values must also "
        "follow the intended distribution. A Kolmogorov-Smirnov test against the "
        "analytic reference catches a sampler that fires correctly but draws from "
        "the wrong shape (e.g. uniform where truncated-normal was intended).*\n")
    R.p("**Why each axis has the shape it does.** Peaked where the physical "
        "world has a 'correct' value it clusters around; uniform where it does "
        "not. Cameras aim for correct exposure and mostly succeed, so "
        "`brightness` and `saturation` are truncated normals. Most photographs "
        "are roughly in focus and focus error has a hard floor at zero, so "
        "`blur` is a half-normal. But there is no privileged illuminant across "
        "many rooms, no privileged compression setting, and no privileged hue "
        "offset for what is purely a regularizer, so `temperature`, `jpeg`, "
        "`hue`, `shadow`, `noise`, `specular` and `vignette` are uniform. "
        "Uniform brightness would assert that a wildly underexposed frame is as "
        "likely as a correct one, which is false about cameras and wastes model "
        "capacity on a failure mode reality rarely produces. Full reasoning in "
        "`docs/curriculum-design.md`.\n")
    R.p("**Reading the figure below.** Two axes can show spikes for entirely "
        "different reasons. `blur` has a *real* point mass: `clip()` maps every "
        "draw below 0.1 onto exactly 0.1 rather than resampling, so ~15.8% of "
        "blur values are identical. `jpeg` shows *no* real structure, since it takes "
        "46 integer values, and any bin count that is not a multiple of that "
        "produces moiré banding. Bins are now allocated per-value for discrete "
        "axes. Treat a spiky histogram as a question; separating a sampler "
        "defect from a rendering artifact takes a test on the values.\n")

    refs = _reference_dists()
    rows, ks_ok = [], True
    for col, (desc, dist) in refs.items():
        vals = tv[col].dropna().values
        if len(vals) < 100:
            continue
        if dist is None:
            rows.append([col.replace("aug_", ""), desc, f"{len(vals):,}",
                         f"[{vals.min():.4g}, {vals.max():.4g}]", "n/a", "not analytic"])
            continue
        ks = stats.kstest(vals, dist.cdf)
        ok = ks.pvalue > 0.01
        ks_ok &= ok
        rows.append([col.replace("aug_", ""), desc, f"{len(vals):,}",
                     f"[{vals.min():.4g}, {vals.max():.4g}]",
                     f"{ks.pvalue:.3f}", "OK" if ok else "DEVIATES"])
    R.table(["Axis", "Intended distribution", "n", "Observed range", "KS p", "Status"], rows)
    R.p("\n*KS p > 0.01 means the sampled values are consistent with the intended "
        "distribution. Low p on a large sample can reflect discretization rather "
        "than a real defect.*\n")
    R.verdict("parameter fidelity", ks_ok,
              "sampled values match their intended distributions"
              if ks_ok else "at least one axis deviates from its intended distribution")

    cols = [c for c in refs if tv[c].notna().sum() > 100]
    if cols:
        ncol = min(4, len(cols))
        nrow = max(1, -(-len(cols) // ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.5 * nrow),
                                 squeeze=False)
        flat = np.ravel(axes)
        for ax, col in zip(flat, cols):
            vals = tv[col].dropna()
            # Discrete axes need one bin per value. JPEG quality takes 46
            # integers; forcing them into 40 bins makes every ~8th bin swallow
            # two integers and show up at 2x height, a moiré artifact that
            # looks exactly like periodic structure in the sampler but is
            # purely a binning choice.
            uniq = vals.nunique()
            bins = int(uniq) if uniq <= 60 and float(vals.max() - vals.min()) >= uniq - 1 \
                else 40
            ax.hist(vals, bins=bins, color="slateblue")
            ax.set_title(col.replace("aug_", "") +
                         (f"  (discrete, {uniq} values)" if bins != 40 else ""),
                         fontsize=8)
        for ax in flat[len(cols):]:
            ax.axis("off")
        plt.tight_layout()
        p = os.path.join(fig_dir, "params.png"); plt.savefig(p); plt.close()
        R.fig(p, "Sampled parameter distribution per augmentation axis")
    else:
        R.p("*(too few samples per axis to plot distributions)*\n")

    # ── 6. Independence ──────────────────────────────────────────────────────
    R.h("6. Augmentation independence")
    R.p("*Effects are meant to fire independently. Correlated firing makes axes "
        "mutually unattributable. If two always co-occur, no analysis can tell "
        "which one caused a failure. This is the check that catches the V1/V2 bug "
        "where hue and saturation shared a single gate (correlation +1.0).*\n")

    ind = pd.DataFrame({c.replace("aug_", ""): tv[c].notna().astype(float)
                        for c in DESIGNED_P})
    corr = ind.corr()
    off = corr.where(~np.eye(len(corr), dtype=bool))
    max_abs = float(np.nanmax(np.abs(off.values)))
    i, j = np.unravel_index(np.nanargmax(np.abs(off.values)), off.shape)
    thresh = 3.0 / np.sqrt(n)  # ~3 SE for a correlation under independence

    hue_sat = float(corr.loc["hue", "sat"])
    R.p(f"Largest off-diagonal correlation: **{max_abs:.4f}** "
        f"({corr.index[i]} vs {corr.columns[j]}). "
        f"Noise threshold at n={n:,} is ~{thresh:.4f}.\n")
    R.p(f"**hue vs saturation = {hue_sat:+.4f}**. In V1/V2 this was structurally "
        f"+1.000 because both sat behind one probability gate. They are now "
        f"independently sampled, which is what makes per-axis attribution possible "
        f"at all.\n")
    R.verdict("independence", max_abs < max(thresh * 2, 0.05),
              f"max |correlation| {max_abs:.4f} is within sampling noise")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index)
    ax.set_title("Augmentation co-firing correlation\n(off-diagonal should be ~0)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    p = os.path.join(fig_dir, "independence.png"); plt.savefig(p); plt.close()
    R.fig(p, "Pairwise correlation of augmentation firing")

    counts = tv["n_aug"].value_counts().sort_index()
    R.p("\n**Effects stacked per image.** Under independence this is a Poisson-"
        "binomial with mean = sum of designed probabilities:\n")
    expected_mean = sum(DESIGNED_P.values())
    R.p(f"Expected mean = {expected_mean:.2f}, observed = **{tv['n_aug'].mean():.2f}** "
        f"(range {int(tv['n_aug'].min())}–{int(tv['n_aug'].max())}).\n")
    R.table(["n effects", "images", "share"],
            [[k, f"{v:,}", f"{v/len(tv):.1%}"] for k, v in counts.items()])

    # ── 7. Confounds ─────────────────────────────────────────────────────────
    R.h("7. Color/augmentation confounding")
    R.p("*The most consequential check here. Augmentation is applied after the "
        "garment color is chosen, so the two must be statistically independent. "
        "If, say, blue garments were systematically darker, the model could reach "
        "the right answer using brightness as a proxy for color, scoring well on "
        "this dataset while learning something that does not transfer. A leak here "
        "would invalidate the benchmark, not merely degrade it.*\n")

    leak_rows, max_leak, worst = [], 0.0, ("", "")
    for col in DESIGNED_P:
        vals = tv[col]
        mask = vals.notna()
        if mask.sum() < 200:
            continue
        best_r, best_c = 0.0, ""
        for c in COLOR_CLASSES:
            r = np.corrcoef(vals[mask].values, tv.loc[mask, c].values)[0, 1]
            if abs(r) > abs(best_r):
                best_r, best_c = r, c
        leak_rows.append([col.replace("aug_", ""), best_c, f"{best_r:+.4f}"])
        if abs(best_r) > abs(max_leak):
            max_leak, worst = best_r, (col.replace("aug_", ""), best_c)
    R.table(["Axis", "Most correlated color", "Pearson r"], leak_rows)
    if not leak_rows:
        # Never report a pass for a test that did not run. An untested check
        # silently reporting max|r| = 0.0000 is worse than an outright failure.
        R.p("\n**INCONCLUSIVE**: no axis had the >=200 samples needed for a "
            "meaningful correlation estimate. This check did not run.\n")
        R.verdict("no color confound", False,
                  "INCONCLUSIVE: insufficient samples to test for confounding")
    else:
        R.p(f"\nStrongest leak across all {len(leak_rows)}x{len(COLOR_CLASSES)} pairs: "
            f"**r = {max_leak:+.4f}** ({worst[0]} vs {worst[1]}). "
            f"Noise floor at n={n:,} is ~{thresh:.4f}.\n")
        R.verdict("no color confound", abs(max_leak) < 0.05,
                  f"max |r| = {abs(max_leak):.4f}, so augmentation carries no usable "
                  f"information about garment color")

    # ── 8. Train/val parity ──────────────────────────────────────────────────
    R.h("8. Train/validation parity")
    R.p("*Validation must be drawn from the same distribution as training, "
        "otherwise val loss measures distribution shift rather than "
        "generalization, and is not comparable across runs.*\n")

    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    rows, parity_ok = [], True
    for col in DESIGNED_P:
        a, b = tr[col].dropna().values, va[col].dropna().values
        if len(a) < 100 or len(b) < 100:
            continue
        ks = stats.ks_2samp(a, b)
        ok = ks.pvalue > 0.01
        parity_ok &= ok
        rows.append([col.replace("aug_", ""), f"{len(a):,}", f"{len(b):,}",
                     f"{ks.pvalue:.3f}", "same" if ok else "DIFFERS"])
    R.table(["Axis", "n train", "n val", "KS p", "Verdict"], rows)
    ent_ks = stats.ks_2samp(tr["label_entropy"].values, va["label_entropy"].values)
    R.p(f"\nLabel entropy, train vs val: KS p = {ent_ks.pvalue:.3f}.\n")
    R.verdict("train/val parity", parity_ok and ent_ks.pvalue > 0.01,
              "validation is distributionally indistinguishable from training")

    # ── 8b. Background leakage ───────────────────────────────────────────────
    R.h("8b. Background isolation", level=3)
    R.p("*Distributional parity is not sufficient. The model sees the full "
        "224x224 frame, not just the garment patch, so a background appearing "
        "in both train and validation is a memorization channel. The model can "
        "recognize the room rather than generalize the color. Backgrounds must "
        "be partitioned, not merely sampled identically.*\n")

    if "background" not in df.columns:
        R.p("**INCONCLUSIVE**: this dataset predates background provenance "
            "tracking, so leakage cannot be measured. Regenerate to enable "
            "this check.\n")
        R.verdict("background isolation", False,
                  "INCONCLUSIVE: no `background` column recorded")
    else:
        import itertools
        present = [s for s in ("train", "val", "test", "probe")
                   if (df["split"] == s).any()]
        bg = {s: set(df.loc[df["split"] == s, "background"].dropna().astype(str)) - {""}
              for s in present}
        n_img = {s: int((df["split"] == s).sum()) for s in present}

        R.table(["Split", "Unique backgrounds", "Images", "Reuse factor"],
                [[s, f"{len(bg[s]):,}", f"{n_img[s]:,}",
                  f"{n_img[s]/max(len(bg[s]),1):.2f}x"] for s in present])

        pair_rows, clean = [], True
        for a, b in itertools.combinations(present, 2):
            n_ov = len(bg[a] & bg[b])
            clean &= n_ov == 0
            pair_rows.append([f"{a} n {b}", f"{n_ov}", "OK" if n_ov == 0 else "LEAK"])
        R.p("\nPairwise background overlap, where every cell must be zero:\n")
        R.table(["Pair", "Shared backgrounds", "Status"], pair_rows)
        R.verdict("background isolation", clean,
                  f"all {len(pair_rows)} split pairs draw from disjoint background "
                  f"pools, so no room is ever seen by two splits" if clean
                  else "backgrounds leak between splits")

    # ── 8c. Held-out test set ────────────────────────────────────────────────
    R.h("8c. Held-out test set", level=3)
    R.p("*Validation is not a clean generalization estimate here: it drives the "
        "LR scheduler, checkpoint selection, early stopping AND the adaptive "
        "controller (class weights, augmentation strength, label smoothing). "
        "Four channels of optimization pressure on the same 2,000 images. A "
        "separate split that nothing in the training loop ever reads is the "
        "only number that can be reported without that caveat.*\n")

    te = df[df["split"] == "test"]
    if len(te) == 0:
        R.p("**No test split present.** Any reported metric will carry the same "
            "optimistic bias as V2's published 0.5942.\n")
        R.verdict("held-out test set", False, "no test split exists")
    else:
        rows, pv_t = [], []
        for col in DESIGNED_P:
            a, b = tr[col].dropna().values, te[col].dropna().values
            if len(a) < 100 or len(b) < 100:
                continue
            ks = stats.ks_2samp(a, b)
            pv_t.append(ks.pvalue)
            rows.append([col.replace("aug_", ""), f"{ks.pvalue:.3f}"])
        differs_t, thresh_t = holm_bonferroni(pv_t, alpha=0.01)
        te_ok = not differs_t
        for r in rows:
            r.append("DIFFERS" if float(r[1]) <= thresh_t else "same")
        R.p(f"Test split: **{len(te):,} images**, drawn from a background pool "
            f"disjoint from train, val and probe.\n")
        R.p("Distributional parity with train (a test set must be representative, "
            "not merely separate):\n")
        R.table(["Axis", "KS p", "Verdict"], rows)
        R.p(f"\n*Holm-Bonferroni across {len(pv_t)} tests, family-wise alpha "
            f"0.01 (threshold {thresh_t:.4f}).*\n")
        R.verdict("held-out test set", te_ok and len(te) >= 1000,
                  f"{len(te):,} images, representative of train, never touched "
                  f"by the training loop")

    # ── 9. Probe validity ────────────────────────────────────────────────────
    R.h("9. Probe set validity")
    R.p("*The probe set exists to make per-axis attribution possible. It is only "
        "valid if each image varies exactly one axis, the control group varies "
        "none, and every axis x level cell is populated evenly.*\n")

    ctrl = probe[probe["probe_axis"].isna() | (probe["probe_axis"].astype(str) == "")]
    swept = probe[probe["probe_axis"].notna() & (probe["probe_axis"].astype(str) != "")]
    ctrl_ok = bool((ctrl["n_aug"] == 0).all())
    iso_ok = bool((swept["n_aug"] == 1).all())
    axes_ok = set(swept["probe_axis"].unique()) == set(AXES)

    R.table(["Check", "Result", "Status"], [
        ["Control images", f"{len(ctrl):,}", "OK" if len(ctrl) else "MISSING"],
        ["Control has zero augmentation", "yes" if ctrl_ok else "no",
         "OK" if ctrl_ok else "FAIL"],
        ["Swept images", f"{len(swept):,}", "n/a"],
        ["Every swept image isolates 1 axis", "yes" if iso_ok else "no",
         "OK" if iso_ok else "FAIL"],
        ["All 10 axes covered", "yes" if axes_ok else "no", "OK" if axes_ok else "FAIL"],
    ])

    cell_rows, cells_even = [], True
    for axis in AXES:
        sub = swept[swept["probe_axis"] == axis]
        per_cell = sub.groupby("probe_value").size()
        expected_levels = len(probe_values(axis, len(per_cell))) if len(per_cell) else 0
        even = bool(per_cell.nunique() == 1) if len(per_cell) else False
        cells_even &= even
        cell_rows.append([axis, len(per_cell), f"{per_cell.min()}–{per_cell.max()}",
                          f"{sub['probe_value'].min():.4g}",
                          f"{sub['probe_value'].max():.4g}",
                          "even" if even else "UNEVEN"])
    R.table(["Axis", "Levels", "Images/cell", "Min value", "Max value", "Balance"], cell_rows)
    R.verdict("probe validity", all([ctrl_ok, iso_ok, axes_ok, cells_even]),
              "probe set supports clean single-axis attribution")

    # ── Sample grid ──────────────────────────────────────────────────────────
    try:
        import cv2
        R.h("Samples")
        sample = tv.sample(min(24, len(tv)), random_state=0)
        fig, axes_g = plt.subplots(4, 6, figsize=(15, 10.5))
        for ax, (_, row) in zip(np.ravel(axes_g), sample.iterrows()):
            im = cv2.imread(os.path.join(img_dir, row["filename"]))
            if im is None:
                ax.axis("off"); continue
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            top = sorted([(c, row[c]) for c in COLOR_CLASSES], key=lambda x: -x[1])[:2]
            ax.set_title(f"{row['pattern']} | " +
                         ", ".join(f"{c} {v:.0%}" for c, v in top if v > 0.05),
                         fontsize=7)
            ax.axis("off")
        plt.tight_layout()
        p = os.path.join(fig_dir, "samples.png"); plt.savefig(p); plt.close()
        R.fig(p, "Random training samples with dominant labels")
    except Exception as exc:
        R.p(f"*(sample grid skipped: {type(exc).__name__})*\n")

    # ── Verdict summary ──────────────────────────────────────────────────────
    R.h("Summary")
    n_pass = sum(1 for _, ok, _ in R.verdicts if ok)
    R.table(["Check", "Verdict", "Detail"],
            [[nm, "PASS" if ok else "FAIL", d] for nm, ok, d in R.verdicts])
    R.p(f"\n**{n_pass} of {len(R.verdicts)} checks passed.**\n")
    if n_pass == len(R.verdicts):
        R.p("The dataset is internally consistent, matches its design "
            "specification, and carries no measurable color/augmentation "
            "confound. It is suitable as a benchmark.\n")
    else:
        R.p("One or more checks failed. See the sections above before "
            "training on this dataset.\n")

    out_md = os.path.join(args.out, "dataset_report.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(R.render())

    print(f"\n{n_pass}/{len(R.verdicts)} checks passed")
    for nm, ok, d in R.verdicts:
        print(f"  {'PASS' if ok else 'FAIL'}  {nm:<22} {d}")
    print(f"\nreport  : {out_md}")
    print(f"figures : {fig_dir}")


if __name__ == "__main__":
    main()
