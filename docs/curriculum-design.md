# Curriculum Design: Diagnosing and Targeting Failure Modes

*Design notes for V3. Written to explain the reasoning, not just the code.*

---

## The one-sentence version

If the generator records **what it did to every image**, then validation loss stops
being a single number and becomes a **map of where the model is weak**. Anything
you can map, you can target.

---

## Where this idea comes from

This is V2's method pointed at a second axis.

**V2 (already built).** V1's confusion matrix showed the model mixing up orange and
yellow. Rather than accept that, the pipeline measured the confusion, then reshaped
the *color library* to widen the gap exactly where the model was failing, using
pair-specific Voronoi margins derived from real measured confusion. Result:
0.7474 → 0.5942, from data alone.

**V3 (this document).** The same loop, applied to **photometric space** instead of
**color space**:

| | V2 | V3 |
|---|---|---|
| Measure | confusion matrix over color categories | loss curve over augmentation axes |
| Diagnose | "orange/yellow boundary is weak" | "the model is weak under warm light" |
| Act | widen Voronoi margin for that pair | oversample / regenerate along that axis |
| Domain | which colors exist in the data | what is done to the image |

The symmetry is what collapses two ad-hoc fixes into one method: **measure where
the model fails, reshape the data to attack exactly there, repeat.** V2 applies it
to color space. V3 applies it to photometric space.

---

## The enabling insight: stop throwing the parameters away

The V1/V2 generator does this, ten times per image:

```python
factor = _sample_truncated_normal(1.0, 0.35, 0.3, 1.8)
return img_float * factor          # factor is now gone forever
```

Every augmentation samples a number describing exactly how hard it made the image,
uses it, and discards it. Once it's gone, the only thing left is the image and its
label, so the only answerable questions are *"is the model good?"* and *"on which
color?"* You cannot ask *"under which conditions?"*, because the conditions were
never written down.

Keeping that number is a plumbing change. What it buys is a different class of question:

- Which axis is the model weakest on?
- *Where* on that axis does it break: gradually, or past a cliff?
- Is the weakness real, or noise?
- Did the last intervention actually fix it?

None of these are answerable without the metadata. All of them are trivial with it.
Near-zero cost for a whole new category of question is why this is worth doing
even if the curriculum itself turns out not to help.

---

## Why one dataset, not several

The intuitive design is a ladder of datasets: easy, medium, hard, brutal. Train
through them in order. That design has three problems.

**It bakes the curriculum in at generation time.** The schedule becomes a property of
bytes on disk. Changing your mind about pacing means regenerating everything.

**It multiplies cost for nothing.** N datasets is N× disk and N× generation.

**It cannot express partial ordering.** Real difficulty is not a single ladder. An
image can be dark-but-clean or bright-but-blurred. Which rung is that?

The alternative: **one dataset, every sample carrying its parameters, and the
curriculum expressed as a sampling policy over it.**

The dataset becomes a fixed, reusable asset. The curriculum becomes a few lines of
sampler logic. Changing the schedule costs a rerun of the loop rather than a
regeneration of 22,000 images, which puts ablations inside the compute budget
instead of outside it.

**Principle: store raw parameters, derive difficulty at analysis time.** Do not bake
a `difficulty` scalar into the CSV. The moment you want to reweight the axes, and
you will, a stored scalar forces a regeneration while raw parameters cost a
recompute.

---

## Why a separate probe set

Here is the problem that makes the naive approach fail.

In the training distribution, each augmentation fires independently at p≈0.5, so a
typical image has **about five effects stacked on it**. Now try to answer "is the
model bad at low brightness?" by filtering to dark images. That slice is *also* a
slice of blurred, shadowed, hue-shifted, JPEG-crushed images. Every confound rides
along. You measure a correlation and learn nothing causal.

Slicing harder does not help. The information is absent from the data itself,
because the axes were never varied independently.

So: a small **one-factor-at-a-time probe set**. Sweep one axis across its range,
hold every other axis at neutral, ~25 images per cell, plus a neutral control group.

- ~2,200 images (10 axes × 8 levels × 25, + 200 control)
- Roughly one minute to generate
- Because only one thing varies, differences are **attributable**

What comes out is a **response curve per axis**: loss as a function of brightness,
of hue shift, of JPEG quality, each measured against the same neutral baseline.
That gives *which* axis is weak, *where on it*, and *how sharply*. A gentle slope
and a cliff at one end call for completely different fixes.

---

## Why validation does not change

Tempting: make val easier early so the numbers look better.

Do not. **Validation is the only fixed reference point in the whole system.** If val
gets easier as the curriculum advances, loss slopes downward for free and measures
nothing. You lose the ability to compare against V2, against Run A, against
yesterday.

The split of responsibilities:

| Set | Distribution | Job |
|---|---|---|
| `train` | full stochastic | learning; the curriculum reweights *sampling*, not content |
| `val` | full stochastic, **frozen** | the one comparable headline number |
| `probe` | one-factor-at-a-time, **frozen** | diagnosis: which axis, where, how badly |

The curriculum changes *what the model sees*, never *what it is graded on*. Without
that separation there is nothing left to compare a run against.

---

## Two bugs this exposed

Instrumenting the pipeline surfaced two latent defects. Both had been silently
present since V1.

### Hue and saturation were entangled

In V1/V2 both live inside `_augment_color_jitter`, behind a **single** `p=0.5` gate.
They always fired together, correlation exactly +1.0.

Two perfectly correlated variables cannot be told apart, by any amount of analysis.
"The model struggles with hue" was **structurally unfalsifiable** given how the data
was made.

Fixed by giving them independent gates. Measured correlation dropped from an implied
+1.000 to −0.040, which is the threshold at which the two axes become separately
attributable.

### The fold texture was unseedable

`generate_synthetic_clothing_folds()` → `_fbm_perlin(seed=None)` → `np.random.default_rng(None)`.

`default_rng(None)` draws from OS entropy. It does **not** inherit `np.random.seed()`.
So even pinning every global RNG left the Perlin fold texture non-deterministic, and
the fold texture covers most of the garment patch.

This is the concrete reason the original 22,000 V2 images can never be reproduced,
even with the color library recovered bit-for-bit. Fixed by threading a seed through.
Any single image can now be regenerated on its own from the row that describes it,
which makes debugging one bad sample tractable. For `train`, `val` and `test` the
`seed` column is sufficient and unique. For `probe` rows it takes `seed` plus
`probe_axis` and `probe_value`, because the paired design renders one base seed at
81 different cells on purpose; `split` also matters, since each split draws
backgrounds from its own disjoint pool.

Both defects share a shape. They were invisible while the pipeline only produced
images, and obvious the moment it had to report what it had done.

---

## The controller

### Something already owns augmentation intensity, and it disagrees

The V2 training loop contains this:

```python
if val_delta > 0.005:      aug_strength += 0.05   # got worse -> MORE aug
elif val_delta < -0.005:   aug_strength -= 0.03   # got better -> LESS aug
```

That is an **anti-overfitting** controller: rising val loss reads as memorization, so
it turns augmentation up as a regularizer.

A curriculum reads the same signal in the opposite direction: improving → advance →
harder.

On the "val improved" branch these directly contradict: one turns intensity down
while the other turns it up. Left in place together, they oscillate and produce a
training curve nobody can explain.

**Resolution: the curriculum owns intensity.** The old controller's *signal* is
retained, but it feeds the advancement decision instead of steering augmentation
directly, leaving exactly one owner per knob.

### Advancing, carefully

The natural rule, *advance when performance is good or when it stalls*, is half
right. The competence half is sound. The stagnation half has a failure mode worth
respecting: **stagnation is ambiguous.** The model may be stuck because the current
difficulty is *already too hard*. Advancing then makes it strictly worse, and the run
spirals with no obvious cause in the logs.

So on stagnation, escalate in order, cheapest first:

1. **Reduce LR** (`ReduceLROnPlateau` already does this)
2. Still stuck after the LR drop? **Then** advance difficulty
3. **Competence floor:** never advance while val loss sits above the previous stage's plateau

The floor is what makes this safe. Without it, a run that is failing looks identical
to a run that is ready.

### Acting on a diagnosis

**Within a run.** `WeightedRandomSampler` over the train split, upweighting samples
loaded on the weak axis. All metadata is already in `labels.csv`; no regeneration.
Nearly free, so it is the right first experiment.

**Across runs.** Widen that axis's sampling range or raise its fire probability, and
regenerate. This is precisely the V1→V2 move, one level down. Expensive, so only
worth it once the cheap version shows the signal is real.

---

## The statistical trap

**With ten axes, one of them is always the worst.** In a perfectly balanced model,
one axis still comes last by pure sampling noise.

React to that every round and you will chase ghosts, generate a great deal of
activity, and confidently report progress that is indistinguishable from a random
walk.

Two cheap guards, both worth having:

1. **Effect size over noise.** The weak axis must exceed the across-axis mean by more
   than its bootstrap confidence interval. With 25 images per cell the CI is wide
   enough that this rejects most spurious winners.
2. **Persistence.** It must rank weakest on **two consecutive** evaluations before
   anything is acted on.

Neither is expensive, and together they separate acting on a measured effect from
acting on rank order alone.

---

## How we would know it worked

The comparison is one dataset, two runs, one variable:

| | Data | Loop |
|---|---|---|
| **Run A** (baseline) | V3 dataset, seeded | existing V2 adaptive loop |
| **Run B** (curriculum) | *same* dataset, *same* seed | curriculum controller |

Success is not only "B beats A on val loss." Also informative:

- **B's probe curves are flatter.** Even at equal mean loss, more uniform competence
  across axes is a real improvement and shows up nowhere in a scalar metric
- **B reaches A's final loss in fewer epochs.** A convergence-speed win is still a win
- **B is no better.** A genuine null result, and worth reporting

That last one deserves emphasis. Curriculum learning has a **mixed empirical record**.
This model starts from ImageNet weights that already carry strong low-level color
features, so the headroom may be small.

The README already reports the V1 fine-tuning null result, every trick in the book
moving val loss by 0.0005. "I built the diagnostic, ran the experiment, and it
bought me nothing; here are the curves" is a reportable outcome with a reproducible
figure behind it. The probe set holds its value regardless of which way the run
lands, because it converts an untestable intuition into a measurement.

---

## Corrections after Run A

Run A trained to convergence (80 epochs, val 0.9278, held-out test KL 0.8957 /
top-1 53.4%) and exposed two defects in the design above. Both are recorded
here rather than quietly patched, because the *way* they surfaced is the
argument for having built the instrument at all.

### The probe was unpaired, and its baseline was biased

**Symptom.** At convergence, **six of ten axes scored *below* the control
baseline**, meaning an augmentation apparently made the model *better*
than leaving the image alone. That is impossible for a genuine control.

**Cause.** `probe_plan` gave every cell its own randomly generated garment, so
the control group and the swept groups contained entirely different images
(zero shared seeds). The control group happened to draw harder content:

| | control | swept |
|---|---:|---:|
| label entropy | 0.2978 | 0.2743 |
| solid garments | 20.5% | 24.4% |
| colors per garment | 2.81 | 2.62 |

So "loss above control" was measuring *augmentation effect + content
difference*, and for the near-free axes the content term dominated.

**Fix.** Pair the design: render the same base image at every axis and level.
Reusing the generator `index` produces an identical garment on an identical
background (measured difference 0.33, purely the JPEG re-encode) with only the
augmentation varying, versus 70.97 for a different index. The reported
quantity becomes a within-image difference, `loss(image, axis, level) −
loss(image, own control)`, so content cancels exactly instead of averaging out
slowly. After the fix, control and swept content match to **−0.00000** entropy.

**What it changed.** Re-scoring the *same trained model*:

| axis | unpaired | paired | shift |
|---|---:|---:|---:|
| hue | +0.2269 | **+0.3317** | +0.105 |
| temperature | +0.1893 | **+0.2901** | +0.101 |
| saturation | +0.1138 | **+0.2014** | +0.088 |
| brightness | +0.1207 | +0.1535 | +0.033 |
| blur | −0.1285 | +0.0185 | +0.147 |
| jpeg | −0.1430 | −0.0562 | +0.087 |

Every cost was understated, and **saturation and brightness swapped rank**: the
unpaired reading had them in the wrong order. A curriculum built on it
would have targeted the wrong axis.

The paired curves also self-validate: brightness at ×1.0, saturation at ×1.0,
hue at 0° and temperature at 0 all return **exactly 0.000**, because the
identity value of an axis *is* the control. An unpaired design cannot produce
that guarantee.

Two axes (`jpeg` −0.056, `noise` −0.059) remain slightly negative. Pairing rules
out content bias, which leaves a distribution effect: half of all training images
carry an extra JPEG pass or noise, so a perfectly clean image is *rarer* than a
mildly degraded one and sits marginally out-of-distribution.

## Ablation: how much regularization was the live augmentation providing?

V1 and V2 applied `ColorJitter` + `RandomColorTemperature` at load time, on top
of the augmentation already baked into each JPEG. Two properties of that layer
were unknown and worth separating:

1. **How much of the photometric variance does it carry?** If most, then the
   recorded metadata describes only a fraction of what the model sees, and any
   curriculum steering by metadata is steering a disconnected wheel.
2. **How much regularization is it providing?** Unknown, because it had never
   been run without.

Question 1 is answerable analytically. Baked brightness fires at p=0.5 from
TN(1.0, 0.35); live `ColorJitter(brightness=0.6)` fires *every* time from
U(0.4, 1.6). Decomposing the variance in log space:

| component | variance | share |
|---|---:|---:|
| baked (**recorded**) | 0.0619 | **30%** |
| live (**unrecorded**) | 0.1461 | **70%** |

Correlation between the recorded value and what the model actually sees:
**r = 0.55, r² = 0.30**. So metadata-driven control would have operated on
under a third of the real photometric variation.

Question 2 required an ablation, which Run A provided: the same pipeline with
the live layer removed and nothing else changed.

**Result.** Train 0.6728 vs val 0.9365, with the train/val gap widening
monotonically from 0.048 (epoch 10) to 0.264 (epoch 80) and val flat from about
epoch 60. That is under-regularization, and it quantifies the live layer's
contribution: without it the model overfits progressively.

Both properties are wanted. Neither configuration provides both.

**Resolution.** `utils/controlled_augment.ControlledPhotometric` keeps the live
layer but moves ownership of its intensity from torchvision to the *trainer*.
The loop sets a strength per axis, so it knows exactly what was applied. The
controller **is** the record, and nothing needs logging.

This is also a better curriculum lever than the original plan. Weighted
sampling would nudge the distribution indirectly, by oversampling images that
happen to carry a large baked shift; a strength dial acts on the axis directly.
"The probe says hue is weak" becomes "raise hue strength", and the axes
deliberately mirror the probe axes so diagnosis and lever use the same names.

Hue and saturation keep **separate probability gates**, preserving the
independence the original pipeline destroyed.

The follow-up run (A2) restores the layer under trainer control with everything
else held fixed, isolating the regularization effect from every other change.

---

## Why each distribution has the shape it has

Every axis declares an intended distribution, and that declaration is what makes
the sampler *testable*. Section 5 of the dataset
audit runs a Kolmogorov-Smirnov test of each axis against its declared shape,
so a sampler that fires at the right rate but draws from the wrong shape gets
caught. An axis with no declared intent cannot be wrong, which also means it
cannot be verified.

### The governing principle

**Use a peaked distribution where the physical world has a "correct" value it
clusters around. Use uniform where it does not.**

That single rule explains every choice below.

| Axis | Shape | Why |
|---|---|---|
| `brightness` | truncated normal | Real cameras aim for correct exposure and mostly succeed. Failures are occasional and symmetric-ish. Mass belongs near 1.0. |
| `saturation` | truncated normal | Same logic: scenes cluster near their natural saturation; heavy wash-out and heavy boost are both unusual. |
| `blur` | half-normal | Most photographs are roughly in focus. Focus error has a hard floor at zero and a decaying tail, so the density is half-normal. |
| `temperature` | uniform | There is no single "correct" illuminant across a dataset of many rooms. Tungsten, fluorescent and daylight are all common. No value is privileged. |
| `hue` | uniform | This one simulates nothing physical. It is a deliberate regularizer stopping the model from over-fitting exact hue values. With no physical process to mimic, no offset deserves more mass than another. |
| `shadow` | uniform | Shadow strength in real rooms varies continuously with no modal value. |
| `noise` | uniform | Sensor noise scales with ISO, which varies arbitrarily across sources. Uniform over a narrow band is an honest simplification. |
| `jpeg` | discrete uniform | Compression quality is a pipeline setting, not a physical quantity. Arbitrary across sources. |
| `specular`, `vignette` | uniform | Fabric sheen and lens falloff vary by material and optics; no modal value to center on. |

### Why not uniform everywhere

It would be simpler, and it would be wrong for the peaked axes. Uniform
brightness asserts that a wildly underexposed frame is exactly as likely as a
correctly exposed one. That is false about cameras, and a model trained on it
spends capacity on a failure mode the real world rarely produces while
under-sampling the near-neutral region where most real inputs actually live.

Truncated normal concentrates mass where reality concentrates it, while still
generating hard examples in the tails. This reasoning is inherited from V1/V2
and is unchanged in V3.

### Two shapes worth reading carefully

**Saturation is asymmetrically truncated.** TN(1.0, 0.4) on [0.3, 2.0] puts the
lower bound at **-1.75σ** and the upper at **+2.5σ**. The left side is clipped
much closer to the mean, so the desaturated end terminates against a hard wall
with real density piled at it, while the oversaturated end tapers smoothly to
zero.

The asymmetry is intended, and it carries a consequence into the analysis:
desaturation drives colors toward gray, which is precisely the white/gray
confusion boundary the model already struggles with. So the one axis with an
asymmetric hard edge piles its density against a known weak spot.

**Blur has a point mass at its easy end.** The sampler is
`clip(|N(0, 0.5)|, 0.1, 1.5)`, and `clip` maps *everything* below 0.1 onto
exactly 0.1 rather than resampling. Measured consequences:

- **15.8%** of blur draws are exactly σ = 0.1 (theory: 15.85%)
- **45%** of blur draws are σ < 0.3, which is sub-pixel on a 224px image and
  therefore visually nothing

So blur nominally fires half the time, but roughly half of those are
imperceptible and a sixth are literally the same number. The axis is weaker
than its declared 0.1–1.5 range suggests, and its easy end is a spike where the
declared range implies a gradient.

Impact is bounded: the probe set *forces* specific σ values, so blur response
curves are unaffected and the diagnostic instrument stays clean. Only the
training distribution is lumpy, which matters if you weight-sample toward low
blur and get a clump of identical images.

The fix is one line, a genuine truncated half-normal instead of a clip:

```python
sigma = float(stats.truncnorm.rvs(0.2, 3.0, loc=0.0, scale=0.5))
```

Deliberately not applied yet, because it changes the augmentation distribution
relative to V1/V2 and the degeneracy does not corrupt the diagnosis.

### How to read the parameter figure without being fooled

`reports/figures/params.png` contains one trap that matters more than the plots
themselves.

Two axes render with prominent spikes. **They are completely different
phenomena, and the picture alone gets both backwards:**

| | `blur` | `jpeg` |
|---|---|---|
| What you see | tall spike at the low end | regular spikes every ~9 units |
| What it is | **a real point mass**: 15.8% of draws share one value | **a histogram artifact**: the data is perfectly uniform |
| Evidence | matches analytic P(\|N(0,0.5)\| < 0.1) = 0.1585 | 46 distinct integers, chi-square p = 0.455 |
| Fix | change the sampler | change the bin count |

The JPEG artifact arises because quality takes 46 integer values and the plot
originally used 40 bins. At a bin width of 1.125 value-units, every ~8th bin
swallows two integers and renders at double height, which is textbook moiré.
Max/min bar ratio is **2.07** at 40 bins and **1.04** at 46.

The audit now auto-detects discrete axes and allocates one bin per value, so
this no longer appears. The general lesson is worth keeping: **a spiky histogram
is a question.** Separating a sampler defect from a rendering artifact takes a
test on the values, which is something the picture cannot supply.

---

## Reference: the metadata schema

One row per image in `labels.csv`. `NaN` means the effect did not fire.

**Identity and label**

| Column | Meaning |
|---|---|
| `filename` | image file |
| `split` | `train` / `val` / `probe` |
| 13 color columns | soft label, sums to 1.0 |
| `seed` | generator seed. Unique per row in `train`/`val`/`test`; in `probe`, shared by the 81 renders of one base image |

**Structure** (content difficulty, independent of photometrics)

| Column | Meaning |
|---|---|
| `pattern` | solid / stripes / plaid / gradient / chevron / color_blocking / polka_dot |
| `n_colors` | distinct colors in the garment patch |
| `label_entropy` | Shannon entropy of the label, normalized to [0,1]. Free difficulty signal: solid = 0, four-color plaid → 1 |
| `fold_blend`, `fold_alpha` | fold texture blend mode and opacity |

**Photometrics** (one column per axis; the diagnosis surface)

| Column | Range | Neutral |
|---|---|---|
| `aug_brightness` | 0.3 – 1.8 | 1.0 |
| `aug_temp_r`, `aug_temp_b` | 0.75 – 1.25 each | 1.0 |
| `aug_hue` | −25° – +25° | 0 |
| `aug_sat` | 0.3 – 2.0 | 1.0 |
| `aug_shadow` | 0.1 – 0.55 | 0 |
| `aug_blur` | 0.1 – 1.5 | 0 |
| `aug_noise` | 0.001 – 0.005 | 0 |
| `aug_jpeg` | 40 – 85 | 100 |
| `aug_specular` | 0.15 – 0.4 | 0 |
| `aug_vignette` | 0.2 – 0.5 | 0 |
| `n_aug` | count of effects fired | 0 |

**Probe bookkeeping**

| Column | Meaning |
|---|---|
| `probe_axis` | which axis was swept (blank = control group) |
| `probe_value` | the value it was held at |

### One modeling note on temperature

Temperature is stored as **two** raw scales, not one derived number, and that is
deliberate. The meaningful warm/cool axis is `log(r_scale / b_scale)`:

- both scales up together → that is **brightness**, not temperature
- `r > b` → warm; `r < b` → cool

Collapsing them at write time would silently merge a brightness effect into the
temperature axis and corrupt both response curves. Storing raw and deriving at
analysis time costs nothing and keeps the axes clean, on the same principle as not
baking in a difficulty scalar.

---

## Code map

| File | Role |
|---|---|
| `utils/instrumented_augment.py` | augmentations that report their sampled parameter; probe mode (`apply_single_axis`) |
| `utils/instrumented_generator.py` | seeded, metadata-emitting generator; `probe_plan()` |
| `scripts/generate_dataset.py` | parallel driver; train + val + probe; resumable |
| `docs/curriculum-design.md` | this document |

### An incidental speedup

`OuterSquareGenerator.generate()` called `os.listdir()` **inside the per-image path**,
over ~15,600 background files, once per generated image. Across a 22,000-image run
that is roughly 343 million redundant filename operations, and it dominated
generation time. Hoisting the listing into the constructor is a one-line change that
turns dataset generation from a coffee break into about a minute.
