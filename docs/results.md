# Experimental Record

Running log of what has been built, what has been run, and what it produced.
Design reasoning lives in [`curriculum-design.md`](curriculum-design.md); this
file is the ledger.

---

## Dataset: V3

**28,860 images, 36 recorded fields each**, fully seeded and reproducible.

| Split | Images | Backgrounds | Purpose |
|---|---:|---:|---|
| `train` | 20,000 | 8,946 | learning |
| `val` | 2,000 | 1,102 | LR schedule, checkpointing, early stop, adaptive controller |
| `test` | 2,000 | 1,075 | read **once**, at the end |
| `probe` | 4,860 | 1,403 | one-factor-at-a-time diagnostic (60 base images × 81 renders) |

Background pools are **mutually disjoint across all six split pairs**. The model
sees the whole 224×224 frame rather than the garment alone, so a shared room is a
memorization channel between splits.

Color library recovered bit-for-bit from upstream: 325 colors, 25/category,
verified against the published V2 statistics (violet 1-of-11 seeds surviving,
79 Voronoi rejections).

Audit: **11/11 checks pass** (`scripts/analyze_dataset.py`). Strongest
color↔augmentation confound across all 10×13 pairs is r = 0.037; largest
augmentation co-firing correlation is 0.018 against a 0.020 noise floor.

---

## Run A: baseline

**Configuration**

| | |
|---|---|
| Backbone | ResNet-50, ImageNet `IMAGENET1K_V2` init |
| Head | `Dropout(0.4) → Linear(2048, 13)` |
| Loss | KL divergence against soft labels |
| Batch | 32 × 2 gradient accumulation = **64 effective** |
| Precision | AMP (fp16 autocast + GradScaler) |
| LR | 5e-5 base, discriminative ×0.001 / ×0.01 / ×0.1 / ×1.0 |
| Schedule | `ReduceLROnPlateau`, patience 4 |
| Live augmentation | **ablated** (geometric only); see below |
| Adaptive | per-class weights from val MAE, label smoothing 0.02–0.15 |
| Epochs | 80, best at 77 |
| Hardware | RTX 4050 Laptop 6 GB; peak **1.78 GB**, 1.5 min/epoch, **136 min total** |

**Results**

| Metric | Value |
|---|---:|
| Best val KL | 0.9278 (epoch 77) |
| **Held-out test KL** | **0.8957** |
| **Held-out test top-1** | **53.4%** |
| Held-out test MAE | 0.0720 |
| Final train KL | 0.6728 |

Test came in **better than val** (0.8957 vs 0.9278). Despite val driving four
separate optimization channels, the untouched split generalized fine, so the
adaptive loop did not meaningfully overfit it.

**Comparison to the published V2 numbers**

V2 reported teacher val 0.5942, student 0.4798, top-1 68.8%. Run A is worse,
for three identified reasons:

1. **Cold start.** V2 initialized from V1's `finetune_best.pth`; that
   checkpoint no longer exists. Run A starts from ImageNet, and with
   `backbone_early` at ×0.001 → 5e-8 the early layers were effectively frozen
   throughout. That LR scheme was designed for fine-tuning a backbone that had
   already been adapted, and a cold start is a different problem.
2. **Live photometric augmentation ablated.** Run A is the ablation arm that
   measures that layer's regularization contribution; see below.
3. **Backgrounds are now partitioned.** V2's validation shared rooms with
   training. Run A's number measures something strictly harder and more honest,
   so the two are not directly comparable in the first place.

**Diagnosis of the gap.** Train 0.6728 vs val 0.9365 with val flat from epoch
~60 is the signature of under-regularization, which makes cause 2 the primary
lever. Run A2 tests it directly: identical configuration with the live layer
restored under trainer control, nothing else changed.

---

## Probe findings (paired, at convergence)

Cost of each axis, measured as loss minus **that same image's own control**:

| Axis | Cost | Shape |
|---|---:|---|
| **hue** | **+0.3317** | symmetric V, flat within ±10°, +0.70/+0.80 at ∓25° |
| **temperature** | **+0.2901** | symmetric V, slightly worse warm (+0.73 vs +0.61) |
| **saturation** | +0.2014 | strongly asymmetric: desaturation +0.81, oversaturation +0.23 |
| brightness | +0.1535 | U-shaped, overexposure (+0.48) worse than under (+0.20) |
| blur | +0.0185 | ~flat |
| specular | +0.0086 | ~flat |
| shadow | −0.0014 | ~flat |
| vignette | −0.0119 | ~flat |
| jpeg | −0.0562 | slightly negative |
| noise | −0.0587 | slightly negative |

**The chromatic axes dominate.** Hue, temperature and saturation, the three axes
that actually move color, carry nearly all the cost. Axes that change
luminance, texture or compression artifacts without shifting hue are close to
free, which is the expected behavior for a color model and a good sign the
task is being learned rather than memorized.

**Saturation's asymmetry was predicted before training.** `TN(1.0, 0.4)` on
`[0.3, 2.0]` truncates at −1.75σ versus +2.5σ, putting a hard wall at the
desaturated end; desaturation drives colors toward gray, which is the known
white/gray confusion boundary. The measured 3.5× asymmetry confirms it.

**Blur completely reversed.** Worst axis at epoch 1 (1.862 raw), essentially
free at epoch 80 (+0.0185 paired). Early-training weakness did not survive to
convergence, which is why no curriculum was committed to on the epoch-5 reading.

**`jpeg` and `noise` stay mildly negative for a real reason.** Half of
all training images carry an extra JPEG pass or salt-and-pepper noise, so a
perfectly clean image is *rarer* than a mildly degraded one and sits marginally
out-of-distribution.

---

## Run B: research-informed

Three changes from Run A, each with a stated basis:

| Change | Run A | Run B | Basis |
|---|---|---|---|
| Label smoothing | adaptive 0.02–0.15 | **fixed 0** | Müller et al. 2019 (teacher distillability); Geng 2016, Singh et al. 2025 (the targets here are measurements) |
| Smoothing escalation | on stagnation | **removed** | never fired in 80 epochs; no result supports degrading targets on a plateau |
| LR multipliers | 0.001 / 0.01 / 0.1 / 1.0 | **0.05 / 0.2 / 0.5 / 1.0**, base 1e-4 | Run A's profile is a *fine-tuning* scheme; applied to an ImageNet cold start it pinned `backbone_early` at 5e-8 and the early layers never moved |

**Results.** Early stopped at epoch 40, 64 minutes.

| Metric | Run A | **Run B** | Change |
|---|---:|---:|---:|
| Best val KL | 0.9278 | **0.6096** | −34% |
| **Held-out test KL** | 0.8957 | **0.6225** | −30% |
| **Held-out test top-1** | 53.4% | **63.2%** | +9.8 pts |
| Test MAE | 0.0720 | **0.0499** | −31% |
| Epochs to best | 77 | **25** | 3× faster |

Run B passed Run A's *final, 80-epoch* best by **epoch 4**. Against V2's published
teacher (val 0.5942, top-1 68.8%), Run B is within 0.015 on val while measuring
a harder problem, since V2's validation shared backgrounds with its training set.

Attribution note: three variables changed together, so this falls short of a clean
single-variable ablation. The magnitude and timing point overwhelmingly at the
LR profile. Run A2 had already shown that restoring live augmentation fixes the
overfitting gap *without* improving val, and Müller et al. predict smoothing has
little effect on teacher metrics. Confirming it cleanly would take one more run.

### Probe: the diagnosis collapsed to a single axis

| Axis | Run A | **Run B** |
|---|---:|---:|
| **hue** | +0.3317 | **+0.2163** |
| saturation | +0.2014 | +0.0486 |
| brightness | +0.1535 | +0.0363 |
| **temperature** | +0.2901 | **+0.0141** |

Temperature fell 20×, saturation and brightness roughly 4×. Hue is now **4.5×**
the next axis.

### Why the remaining hue cost should NOT be curriculum-targeted

The obvious next move, raising hue augmentation to attack the weakest axis, is
wrong here, and the color library says why.

Measuring CIELAB hue angles of the 325-color library:

| | |
|---|---:|
| median angular gap between adjacent category centers | **13.2°** |
| hue augmentation range | **±25°** |
| a full-strength shift travels | **190%** of the distance to the next category |

The label is computed from pixel composition **before** augmentation. So at
large hue shifts the probe is asking the model to name the *original* color
after the color has genuinely been changed into another category's territory.
At that point the request is ill-posed, and the cost stops measuring robustness.

Training harder on it would teach **hue invariance**, which is precisely the
wrong inductive bias for a color namer. Hue is the signal here, not a nuisance
variable.

Corroborating evidence: `temperature`, the axis that models *physically real*
illumination change via R/B channel scaling, dropped to **+0.0141**. The model
is already robust to realistic lighting shifts. `hue` is a global rotation that
simulates nothing physical; it was introduced as a regularizer, and the residual
cost is largely correct behavior.

**So the instrument's most useful output here was to block an intervention that
the raw ranking would have recommended.**

---

## Distillation: ResNet-50 → MobileNetV3-Small

**Configuration.** Teacher = Run B. `T = 4.0`, `alpha = 0.7`, 60 epochs, AdamW
`lr 1e-3` with cosine annealing to 1e-6, **label smoothing 0** on the hard-target
term. 47 minutes.

$$\mathcal{L} = 0.7 \cdot T^2 \cdot D_{KL}\!\left(\sigma(z_s/T) \,\|\, \sigma(z_t/T)\right) + 0.3 \cdot D_{KL}\!\left(\sigma(z_s) \,\|\, y\right)$$

**Held-out test, read once, backgrounds disjoint from train and val.**

| Model | KL | top-1 | MAE | Size | CPU |
|---|---:|---:|---:|---:|---:|
| Teacher (ResNet-50) | 0.6226 | 63.2% | 0.0499 | 90.1 MB | 45.4 ms |
| Student FP32 (MobileNetV3-S) | **0.4799** | **67.3%** | 0.0423 | 6.0 MB | 6.2 ms |
| **Student INT8** | **0.4800** | **67.4%** | 0.0423 | **4.2 MB** | **6.9 ms** |

**21.3× smaller, 6.6× faster, and better than the teacher on every metric.**

### The student beat the teacher, on an honest split this time

V2 reported the same inversion (student 0.4798 vs teacher 0.5942) but measured
it with backgrounds shared between train and validation. Here the test split
draws from a background pool disjoint from every other split, and the inversion
holds: **−23% KL and +4.2 points top-1** relative to the teacher.

That makes the V2 observation considerably more credible than it was. The
likeliest mechanism is unchanged: MobileNetV3's depthwise-separable convolutions
and inverted residuals impose a stronger inductive bias than ResNet-50's deeper
parameterisation, which regularizes better on a 13-class problem where the
teacher is mildly overfitting the synthetic distribution.

Worth noting for scale: Run B's teacher (0.6226) sits close to V2's published
teacher (0.5942), and this student (0.4800) lands within 0.0002 of V2's
published student (0.4798). The four-decimal agreement is luck. Landing in the
same region on a harder split is the part that carries information.

### INT8 quantization is free

KL 0.4799 → 0.4800 (**+0.0001**), top-1 67.3% → 67.4%. Identical to V2's
reported delta. Dynamic quantization costs nothing measurable here.

### Correction: `quantize_dynamic` never touched the convolutions

V1/V2 called `quantize_dynamic(model, {nn.Linear, nn.Conv2d})`. **Dynamic
quantization has never supported `nn.Conv2d`**. It applies to `Linear`, `LSTM`,
`GRU`, `RNN`, `Embedding` and `EmbeddingBag` only, so the `Conv2d` entry was a
silent no-op.

Verified directly: `{Linear, Conv2d}` and `{Linear}` both produce **5.95 MB →
4.23 MB**, byte-identical. The convolutional stack was never quantized at all;
the reported 21× compression came entirely from the two classifier `Linear`
layers (~600k of the student's 1.53M parameters). The spec now reads
`{nn.Linear}`.

Genuine convolution quantization would need static (calibrated) quantization or
`torchao`, and would shrink the model further. Not attempted.

### Engineering note: a Windows access violation

The CPU evaluation stage crashed with exit code `-1073741819` (`ACCESS_VIOLATION`).
Every component passed in isolation (quantized forward, teacher forward,
`evaluate_cpu` at `num_workers` 0 and 2, and `benchmark()`), so the fault
required the full context. The distinguishing factor: `train.py` evaluates on
CUDA and never runs CPU inference after CUDA training, whereas `distill.py` does.

Resolved by releasing the GPU (`del teacher, student`, `empty_cache()`,
`synchronize()`) before switching to CPU, and using `num_workers=0` for the CPU
loaders so no subprocess spawns against a just-torn-down CUDA context.

The script now also **exports the models and ONNX before benchmarking**, and
wraps each model's evaluation in `try/except`, so a fault in the measurement
stage cannot cost the artifacts.

---

## Cross-version evaluation

Produced by `scripts/evaluate_all.py`. Every model scored on the **complete
2,000-image held-out test split**: no subsampling, one shared `labels.csv`
parse, one dataset object, `shuffle=False`, so all versions see identical
images in identical order. The paired probe (4,860 renders) is likewise run in
full for every model.

| Model | KL | top-1 | MAE | Size | CPU |
|---|---:|---:|---:|---:|---:|
| Run A, ablation arm (no live aug, LR ×0.001) | 0.8957 | 53.5% | 0.0720 | 90.1 MB | 67.6 ms |
| Run A2, live aug restored *(stopped @15, not converged)* | 1.1511 | 42.4% | 0.0907 | 90.1 MB | 68.1 ms |
| Run B, flat LR + smoothing 0 | 0.6225 | 63.2% | 0.0499 | 90.1 MB | 70.3 ms |
| Student FP32, distilled from Run B | **0.4799** | 67.3% | 0.0423 | 6.0 MB | 10.8 ms |
| **Student INT8, deployable** | 0.4800 | **67.4%** | **0.0423** | **4.2 MB** | 11.5 ms |

Run A2 is included for completeness but was **stopped at epoch 15 of 30** once
it had answered its question; its numbers are a partial run, not a converged
result, and should not be read as a version comparison.

### Per-axis robustness across versions

Paired probe cost, loss minus each image's own control:

| Axis | Run A | Run B | INT8 student |
|---|---:|---:|---:|
| **hue** | +0.3323 | +0.2132 | **+0.1472** |
| temperature | +0.2903 | +0.0324 | +0.0327 |
| saturation | +0.2021 | +0.0702 | +0.0651 |
| brightness | +0.1530 | +0.0330 | +0.0100 |
| everything else | ±0.06 | ±0.04 | ±0.03 |

Two things are visible here that a scalar metric hides.

**Temperature went from a cost to a rounding error.** +0.2903 → +0.0327, a 9×
drop. The model went from meaningfully degraded by realistic illumination shifts
to essentially indifferent to them.

**Hue remains the only substantial residual, and it survived distillation.**
+0.3323 → +0.2132 → +0.1472. It shrinks but never approaches zero, consistent
with the geometric argument above: a ±25° rotation is 190% of the median
distance between adjacent category centers, so at the extremes the request is
ill-posed. This is the axis that should *not* be curriculum-targeted.

### Per-class MAE

Distillation improved **every one of the 13 categories** over its teacher:

| Color | Run A | Run B | INT8 | Δ vs teacher |
|---|---:|---:|---:|---:|
| red | 0.0716 | 0.0455 | 0.0404 | −0.0051 |
| orange | 0.0640 | 0.0489 | 0.0421 | −0.0068 |
| yellow | 0.0764 | 0.0542 | 0.0468 | −0.0073 |
| green | 0.0562 | 0.0366 | 0.0301 | −0.0065 |
| blue | 0.0582 | 0.0372 | 0.0319 | −0.0052 |
| violet | 0.0731 | 0.0547 | 0.0458 | −0.0089 |
| purple | 0.0864 | 0.0638 | 0.0556 | −0.0082 |
| white | 0.0706 | 0.0455 | 0.0423 | −0.0032 |
| gray | 0.0886 | 0.0633 | 0.0556 | −0.0077 |
| black | 0.0534 | 0.0306 | 0.0233 | −0.0073 |
| pink | 0.0781 | 0.0580 | 0.0460 | −0.0120 |
| brown | 0.0864 | 0.0624 | 0.0509 | −0.0115 |
| olive | 0.0727 | 0.0484 | 0.0395 | −0.0089 |

Uniform improvement across all 13 makes the student-beats-teacher result harder
to dismiss as noise on a favourable subset. The hardest categories remain
**purple, gray and brown** in every version, the same three V2 reported. That is
independent evidence the pipeline behaves consistently despite a rebuilt dataset,
a different initialization and a different LR schedule.

### Do not read the CPU column as a latency benchmark

`evaluate_all.py` was run twice on identical checkpoints. The accuracy and probe
columns reproduced to the last digit. The latency column did not:

| | Run A | Run A2 | Run B | Student FP32 | Student INT8 |
|---|---:|---:|---:|---:|---:|
| first pass | 49.0 ms | 51.2 ms | 49.0 ms | 5.7 ms | 12.0 ms |
| second pass | 67.6 ms | 68.1 ms | 70.3 ms | 10.8 ms | 11.5 ms |

The first pass had INT8 apparently *slower* than FP32 (12.0 vs 5.7); the second
put them 0.7 ms apart. Two effects are at work. `evaluate_all.py` benchmarks INT8
last, after four other models have churned through the CPU caches, and quantized
ops are more sensitive to background load and threading state than FP32 ones. The
second pass also ran with more competing on the machine, which inflated
everything on the FP32 side.

`distill.py` benchmarks the student pair in isolation and measured **6.2 ms FP32
vs 6.9 ms INT8**, which is the figure to quote. The accuracy columns in this
report are unaffected either way, since those come from full-split scoring rather
than timing. Both passes are shown here because a number that moves 90% between
identical runs is worth showing rather than quietly replacing.

---

## Case study: why a green sweater scored green at 2.3%

A real photograph of an unambiguously pale-green sweater returned
`yellow 52%, olive 45%, green 2.3%`. Tracing it produced the clearest single
finding in the project.

### Ruled out: the label space

The sweater's median color is `#7f915a`, CIELAB `L*=57.6 a*=-16.1 b*=27.1`.
Its nearest category centroid **is green** (dE 23.3, versus olive 26.9 and
yellow 31.0). The taxonomy has the right answer.

### Ruled out: missing training data

The library contains `#8b9a5f` at `L*=61.2 a*=-15.4 b*=29.2`, **dE 4.2** from
the sweater. A near-exact match exists and is labeled green.

### The cause: where that color sits in the sampling distribution

Sorting the 25 library greens by CIELAB hue angle, and weighting by the
generator's actual draw probability:

| | |
|---|---:|
| green library hue span | 117.7° – 198.2° |
| green **centroid** hue | ~156° |
| library greens below 135° | **2 of 25** |
| share of green draws landing below 135° | **4.1%** |

`generate_random_color` samples within a category with weights
`(1 / mahalanobis_distance)²`, so prototypical colors dominate and colors at
the category edge are rare. The three most central greens alone take **24.8%**
of all draws; `#8b9a5f` takes **1.91%**.

### The consequence, measured

Sweeping a* on flat patches at `L*=60, b*=28`:

| a* | p(green) | p(yellow) |
|---:|---:|---:|
| −44 | 99.1% | 0.5% |
| −36 | **93.2%** | 6.2% |
| −28 | 9.0% | **90.2%** |
| −20 | 1.1% | 97.2% |

The model's learned green/yellow boundary sits near **135°** of hue, well above
the library's 117.7°. The 117–135° band is labeled green in training and
classified yellow at inference. Fed `#8b9a5f` as a flat patch, a color taken
from its own green training set, the model answers **yellow 96%**.

The model contradicts its own labels in that band, and does so because it saw
that band in roughly 4% of green examples while the central cluster dominated
the rest. **The decision boundary contracted toward the centroid.**

### Why V2's library fix did not reach the model

V2's confusion-aware Voronoi resampling was built to widen the gaps between
confusable categories, and it worked: green and yellow now overlap over a
**0.5° band containing one color each**. The *library* has a crisp edge.

But a crisp edge in the library does not produce a crisp edge in the model,
because the centroid-weighted sampler then under-represents the region right up
against that edge. Clean separation plus prototype-biased sampling yields a
model whose boundary is both fuzzy and displaced inward.

**Future work.** The sampler optimizes for prototypical examples, while
boundaries are where the task is actually hard. Sampling uniformly within a
category, or deliberately oversampling near boundaries, is the obvious
counter-experiment, and it stays on the data side of the pipeline like the rest
of this project's interventions.

---

## Color constancy ends up in the weights

Training backgrounds are color-normalized (linearise → Shades-of-Gray white
balance → LAB exposure → CLAHE). Camera frames are not. The obvious hypothesis
was that applying the same normalization at inference would close part of the
sim2real gap.

Tested on the five real photographs, raw versus white-balanced input:

| | raw | white-balanced |
|---|---:|---:|
| top-1 | 3/5 | 3/5 |
| p(red) on the varsity jacket | 27.8% | 36.6% |
| p(gray) on the heather tee | 45.2% | 50.9% |
| p(blue) on the teal paisley | 23.2% | 9.8% |

No improvement. Confident cases sharpen slightly, ambiguous ones degrade.

Read as a **positive result for the training pipeline**. The
augmentation pipeline exposes the model to brightness, color-temperature,
saturation and shadow variation during training, and the probe confirms the
effect: temperature cost fell from +0.2903 to **+0.0327** between Run A and
Run B, a 9× reduction. The model absorbed the color-constancy adjustment into
its weights instead of requiring it as a preprocessing stage.

Practically: inference is a single **6.9 ms** forward pass on CPU with no
white-balance estimation, no CLAHE, no LAB round-trip in the hot path. A
classical pipeline would need all of those per frame, and would still need a
rule for where to draw category boundaries.

---

## Defects found and corrected

Recorded in full in [`curriculum-design.md`](curriculum-design.md#corrections-after-run-a).

| # | Defect | How it surfaced | Fix |
|---|---|---|---|
| 1 | Hue and saturation shared one probability gate (correlation +1.0) | code read during instrumentation | independent gates; measured correlation now −0.006 |
| 2 | Perlin fold texture used `default_rng(None)`, ignoring global seeds | determinism test failed | seed threaded through; dataset now fully reproducible |
| 3 | `os.listdir()` called per generated image over 15.6k files | profiling | hoisted; ~343M redundant operations removed |
| 4 | Case-insensitive filename collisions on NTFS | 20 missing backgrounds | dedupe on lowercased output name |
| 5 | Train/val background leakage | audit reported INCONCLUSIVE | 4-way disjoint partition |
| 6 | Audit ran 10 tests at α=0.01 each, a 10% family-wise false-alarm rate | test-parity check failed on `temp_r` noise | Holm-Bonferroni / Bonferroni-widened intervals |
| 7 | **Probe was unpaired**; control group drew harder content | six axes scored below control, which is impossible | paired design; content bias now −0.00000 |
| 8 | Integrity check demanded globally unique seeds, which the paired probe breaks by design | audit reported FAIL on a correct dataset | uniqueness tested per split; probe tested for an even 60 × 81 grid with one own-control per base |

Defect 7 was found *by* the instrumentation, after it had already been
validated, which is the argument for building the instrument first.

Defect 8 is the mirror image and worth keeping for that reason. Fixing the probe
(defect 7) invalidated an assumption the audit had been written against, and the
audit went on asserting the old invariant until the dataset was re-audited. A
check that encodes a stale design constraint fails loudly on correct data, which
is the good failure mode; the same mistake pointing the other way would have
passed silently. The replacement tests what the paired design actually promises:
seeds unique within `train`/`val`/`test`, and every probe base rendered the same
number of times with exactly one own-control. A half-finished probe run still
fails it.

### The live augmentation layer, which is an ablation rather than a defect

Run A deliberately omits the live photometric layer. Two properties of that layer
were in tension:

- **Metadata fidelity.** Variance decomposition shows the live layer carried
  **70%** of the photometric variance while only the baked layer is recorded
  (r = 0.55 between recorded and actual). Curriculum control via metadata would
  have operated on under a third of the real variation.
- **Regularization.** Unquantified, because the pipeline had never been run
  without it.

Run A supplies the second number: the train/val gap widens monotonically from
0.048 (epoch 10) to 0.264 (epoch 80). The layer was providing substantial
regularization.

`ControlledPhotometric` resolves the tension by moving ownership of the
intensity from torchvision to the trainer. The loop sets it, so the loop knows
it, and the same dial doubles as the curriculum's lever. **Run A2** restores it
with everything else fixed.

---

## Built

| Component | File |
|---|---|
| Instrumented augmentation + probe mode | `utils/instrumented_augment.py` |
| Seeded metadata-emitting generator | `utils/instrumented_generator.py` |
| Trainer-controlled live augmentation | `utils/controlled_augment.py` |
| Background preparation (parallel) | `scripts/prepare_backgrounds.py` |
| Dataset generation (parallel, resumable) | `scripts/generate_dataset.py` |
| Statistical audit, 11 checks | `scripts/analyze_dataset.py` |
| Training (AMP, accumulation, probe eval, held-out test) | `scripts/train.py` |
| Training figures | `scripts/plot_training.py` |
| Showcase figures | `scripts/make_showcase.py` |
| Inference tester (image / camera / ONNX) | `scripts/predict.py` |

ONNX export verified: 89.7 MB, max logit deviation from PyTorch **1.55e-06**,
identical top-1, 75 ms CPU inference.

---

## Open questions

1. ~~Run A2 before Run B?~~ **Done.** A2 (epochs 1–15) showed restoring live
   augmentation flips the train/val gap from +0.05 to −0.05 but does not improve
   val, which is what redirected attention to the LR profile.
2. ~~Fix the cold-start LR scheme?~~ **Done in Run B.** It was the dominant
   effect: −30% test KL, +9.8 points top-1.
3. ~~Curriculum target.~~ **Answered, negatively.** The residual hue cost should
   not be trained against; see "Why the remaining hue cost should NOT be
   curriculum-targeted" above.
4. **Distillation.** MobileNetV3-Small + INT8 remains unbuilt for V3.
   `scripts/distill.py` is written; training and quantization legs verified,
   CPU benchmark leg untested. This is also where the label-smoothing question
   gets settled, since Müller et al. show the effect is invisible in teacher
   metrics and only appears in the student.
5. **No real-data validation yet.** Every number in this document is measured on
   synthetic data. The project's central claim, that a model trained purely on
   procedurally generated images transfers to photographs, is currently
   **untested**. Candidate benchmark: the Clothing Attributes Dataset
   (Chen, Gallagher & Girod, ECCV 2012; 1,856 real photographs, 11 binary color
   attributes, torso-centered, Stanford Digital Repository). Note the metric would
   be top-1 against hard human labels rather than KL, and the label sets
   intersect only partially.
6. **Clean single-variable confirmation of the LR result**, if the attribution
   needs to be airtight rather than strongly-inferred.

---

## Label smoothing: what the literature says

### What actually ran

The escalation branch **never fired** in Run A. Across 73 adjustments: 0
increases, 7 decreases, and `s` sat pinned at the 0.02 floor for **90% of
training**. At s=0.02 that is 1.6% of target mass on zero-evidence categories,
well short of the 12% the top of the range would imply. The practical question is
only whether the floor should be 0.02 or 0.

### The relevant findings

**[Szegedy et al. 2016](https://arxiv.org/abs/1512.00567)** introduced label
smoothing to fix a pathology of **one-hot** targets: they demand an infinite
logit gap. 72.9% of this dataset's labels are already multi-modal, so for most
of the data that pathology never existed.

**[Müller, Kornblith & Hinton, NeurIPS 2019](https://arxiv.org/abs/1906.02629)**
is the decisive result here. Smoothing improves generalization and calibration,
but *"results in loss of information in the logits about resemblances between
instances of different classes, which is necessary for distillation."* A teacher
trained with label smoothing distils **worse**.

This project intends to distil ResNet-50 → MobileNetV3-Small, so the smoothing
setting propagates all the way into the shipped model.

**The experimental-design consequence is easy to miss:** Müller et al. found
smoothing does *not* hurt the teacher's own generalization or calibration. Only
its distillability. So comparing teacher val loss between s=0 and s=0.02 would
show nothing, and one would wrongly conclude smoothing is harmless. **The
comparison has to be made on the student.**

**[Lukasik et al., ICML 2020](https://arxiv.org/abs/2003.02819)** is the
counterweight: under label noise, smoothing is competitive with loss-correction,
and smoothing the teacher **helps** distillation from noisy data, explicitly
reversing Müller et al. for the noisy regime.

So the question reduces to: **are these labels noisy?**

| Sense | Present here? |
|---|---|
| Annotation noise (annotator disagreement, mislabeling) | **None.** Labels are computed exactly from the generator's label map. |
| Generative-vs-perceptual gap | **Real.** A blue region under a 0.55-strength shadow reads near-black but is still labeled blue. |

The second is genuine noise from the learner's point of view, but it is
*structured and asymmetric*: blue drifts toward black and orange toward brown,
along perceptual adjacencies. Lukasik et al. analyze symmetric noise, and uniform
smoothing applies a symmetric correction, so the mechanism does not match the
noise this dataset actually has.

**[Geng 2016](https://palm.seu.edu.cn/xgeng/files/tkde16.pdf)** names what this
project is doing: **Label Distribution Learning**, where the target is a
distribution and KL divergence is the established measure for it.

**[Singh et al. 2025](https://arxiv.org/abs/2511.14117)** argues that when the
distribution is genuine rather than a noisy estimate, it *is* the correct
target. These labels are stronger than that paper's case: computed from pixel
composition, with zero annotator disagreement.

### Conclusion

1. **Drop the escalation.** No result supports degrading targets on a plateau,
   and it never fired anyway. Plateau response belongs on LR and inputs.
2. **Default to s = 0** for any teacher intended for distillation
   (Müller et al.), and because the targets are measured rather than estimated
   (Singh et al., Geng).
3. **But test it**, because the generative-perceptual gap is real and Lukasik
   et al. give a mechanism by which a small floor could help.
4. **Judge it on the student, not the teacher.** That is the specific trap
   Müller et al. identifies.
