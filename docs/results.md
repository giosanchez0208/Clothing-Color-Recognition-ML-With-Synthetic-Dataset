# Experimental Record

Running log of what has been built, what has been run, and what it produced.
Design reasoning lives in [`curriculum-design.md`](curriculum-design.md); this
file is the ledger.

---

## Dataset — V3

**28,860 images, 36 recorded fields each**, fully seeded and reproducible.

| Split | Images | Backgrounds | Purpose |
|---|---:|---:|---|
| `train` | 20,000 | 8,946 | learning |
| `val` | 2,000 | 1,102 | LR schedule, checkpointing, early stop, adaptive controller |
| `test` | 2,000 | 1,075 | read **once**, at the end |
| `probe` | 4,860 | 1,403 | one-factor-at-a-time diagnostic (60 base images × 81 renders) |

Background pools are **mutually disjoint across all six split pairs** — the
model sees the whole 224×224 frame, not just the garment, so a shared room is a
memorisation channel between splits.

Colour library recovered bit-for-bit from upstream: 325 colours, 25/category,
verified against the published V2 statistics (violet 1-of-11 seeds surviving,
79 Voronoi rejections).

Audit: **11/11 checks pass** (`scripts/analyze_dataset.py`). Strongest
colour↔augmentation confound across all 10×13 pairs is r = 0.037; largest
augmentation co-firing correlation is 0.018 against a 0.020 noise floor.

---

## Run A — baseline

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
| Live augmentation | **ablated** (geometric only) — see below |
| Adaptive | per-class weights from val MAE, label smoothing 0.02–0.15 |
| Epochs | 80, best at 77 |
| Hardware | RTX 4050 Laptop 6 GB — peak **1.78 GB**, 1.5 min/epoch, **136 min total** |

**Results**

| Metric | Value |
|---|---:|
| Best val KL | 0.9278 (epoch 77) |
| **Held-out test KL** | **0.8957** |
| **Held-out test top-1** | **53.4%** |
| Held-out test MAE | 0.0720 |
| Final train KL | 0.6728 |

Test came in **better than val** (0.8957 vs 0.9278). Despite val driving four
separate optimisation channels, the untouched split generalised fine — the
adaptive loop did not meaningfully overfit it.

**Comparison to the published V2 numbers**

V2 reported teacher val 0.5942, student 0.4798, top-1 68.8%. Run A is worse,
for three identified reasons:

1. **Cold start.** V2 initialised from V1's `finetune_best.pth`; that
   checkpoint no longer exists. Run A starts from ImageNet — and with
   `backbone_early` at ×0.001 → 5e-8, the early layers were effectively frozen
   throughout. That LR scheme was designed for fine-tuning something already
   adapted, not for a cold start.
2. **Live photometric augmentation ablated.** Run A is the ablation arm that
   measures that layer's regularisation contribution; see below.
3. **Backgrounds are now partitioned.** V2's validation shared rooms with
   training. Run A's number measures something strictly harder and more honest,
   so the two are not directly comparable in the first place.

**Diagnosis of the gap.** Train 0.6728 vs val 0.9365 with val flat from epoch
~60 is under-regularisation, not under-fitting, so cause 2 is the primary
lever. Run A2 tests it directly: identical configuration with the live layer
restored under trainer control, nothing else changed.

---

## Probe findings (paired, at convergence)

Cost of each axis, measured as loss minus **that same image's own control**:

| Axis | Cost | Shape |
|---|---:|---|
| **hue** | **+0.3317** | symmetric V, flat within ±10°, +0.70/+0.80 at ∓25° |
| **temperature** | **+0.2901** | symmetric V, slightly worse warm (+0.73 vs +0.61) |
| **saturation** | +0.2014 | strongly asymmetric — desaturation +0.81, oversaturation +0.23 |
| brightness | +0.1535 | U-shaped, overexposure (+0.48) worse than under (+0.20) |
| blur | +0.0185 | ~flat |
| specular | +0.0086 | ~flat |
| shadow | −0.0014 | ~flat |
| vignette | −0.0119 | ~flat |
| jpeg | −0.0562 | slightly negative |
| noise | −0.0587 | slightly negative |

**The chromatic axes dominate.** Hue, temperature and saturation — the three
that actually move colour — carry nearly all the cost. Axes that change
luminance, texture or compression artifacts without shifting hue are close to
free, which is the expected behaviour for a colour model and a good sign the
task is being learned rather than memorised.

**Saturation's asymmetry was predicted before training.** `TN(1.0, 0.4)` on
`[0.3, 2.0]` truncates at −1.75σ versus +2.5σ, putting a hard wall at the
desaturated end; desaturation drives colours toward gray, which is the known
white/gray confusion boundary. The measured 3.5× asymmetry confirms it.

**Blur completely reversed.** Worst axis at epoch 1 (1.862 raw), essentially
free at epoch 80 (+0.0185 paired). Early-training weakness did not survive to
convergence — the reason no curriculum was committed to on the epoch-5 reading.

**`jpeg` and `noise` staying mildly negative is real, not an artifact.** Half of
all training images carry an extra JPEG pass or salt-and-pepper noise, so a
perfectly clean image is *rarer* than a mildly degraded one and sits marginally
out-of-distribution.

---

## Run B — research-informed

Three changes from Run A, each with a stated basis:

| Change | Run A | Run B | Basis |
|---|---|---|---|
| Label smoothing | adaptive 0.02–0.15 | **fixed 0** | Müller et al. 2019 (teacher distillability); Geng 2016, Singh et al. 2025 (targets are measured, not estimated) |
| Smoothing escalation | on stagnation | **removed** | never fired in 80 epochs; no result supports degrading targets on a plateau |
| LR multipliers | 0.001 / 0.01 / 0.1 / 1.0 | **0.05 / 0.2 / 0.5 / 1.0**, base 1e-4 | Run A's profile is a *fine-tuning* scheme; applied to an ImageNet cold start it pinned `backbone_early` at 5e-8 and the early layers never moved |

**Results** — early stopped at epoch 40, 64 minutes.

| Metric | Run A | **Run B** | Change |
|---|---:|---:|---:|
| Best val KL | 0.9278 | **0.6096** | −34% |
| **Held-out test KL** | 0.8957 | **0.6225** | −30% |
| **Held-out test top-1** | 53.4% | **63.2%** | +9.8 pts |
| Test MAE | 0.0720 | **0.0499** | −31% |
| Epochs to best | 77 | **25** | 3× faster |

Run B passed Run A's *final, 80-epoch* best by **epoch 4**. Against V2's published
teacher (val 0.5942, top-1 68.8%), Run B is within 0.015 on val while measuring
a harder problem — V2's validation shared backgrounds with its training set.

Attribution note: three variables changed together, so this is not a clean
single-variable ablation. The magnitude and timing point overwhelmingly at the
LR profile — Run A2 had already shown that restoring live augmentation fixes the
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

The obvious next move — raise hue augmentation to attack the weakest axis — is
wrong here, and the colour library says why.

Measuring CIELAB hue angles of the 325-colour library:

| | |
|---|---:|
| median angular gap between adjacent category centres | **13.2°** |
| hue augmentation range | **±25°** |
| a full-strength shift travels | **190%** of the distance to the next category |

The label is computed from pixel composition **before** augmentation. So at
large hue shifts the probe is asking the model to name the *original* colour
after the colour has genuinely been changed into another category's territory.
That is not a robustness failure; the request is ill-posed.

Training harder on it would teach **hue invariance** — precisely the wrong
inductive bias for a colour namer, where hue is the signal rather than a
nuisance variable.

Corroborating evidence: `temperature` — the axis that models *physically real*
illumination change via R/B channel scaling — dropped to **+0.0141**. The model
is already robust to realistic lighting shifts. `hue` is a global rotation that
simulates nothing physical; it was introduced as a regulariser, and the residual
cost is largely correct behaviour.

**The instrument's most useful output here was to prevent an intervention, not
to motivate one.**

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
| 6 | Audit ran 10 tests at α=0.01 each — 10% family-wise false-alarm rate | test-parity check failed on `temp_r` noise | Holm-Bonferroni / Bonferroni-widened intervals |
| 7 | **Probe was unpaired**; control group drew harder content | six axes scored below control, which is impossible | paired design; content bias now −0.00000 |

Defect 7 was found *by* the instrumentation, after it had already been
validated — which is the argument for building the instrument first.

### Ablation, not a defect: the live augmentation layer

Run A deliberately omits the live photometric layer, and that omission is an
experiment rather than an oversight. Two properties were in tension:

- **Metadata fidelity.** Variance decomposition shows the live layer carried
  **70%** of the photometric variance while only the baked layer is recorded
  (r = 0.55 between recorded and actual). Curriculum control via metadata would
  have operated on under a third of the real variation.
- **Regularisation.** Unquantified, because the pipeline had never been run
  without it.

Run A supplies the second number: the train/val gap widens monotonically from
0.048 (epoch 10) to 0.264 (epoch 80). The layer was providing substantial
regularisation.

`ControlledPhotometric` resolves the tension by moving ownership of the
intensity from torchvision to the trainer — the loop sets it, so the loop knows
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
   val — which is what redirected attention to the LR profile.
2. ~~Fix the cold-start LR scheme?~~ **Done in Run B.** It was the dominant
   effect: −30% test KL, +9.8 points top-1.
3. ~~Curriculum target.~~ **Answered, negatively.** The residual hue cost should
   not be trained against — see "Why the remaining hue cost should NOT be
   curriculum-targeted" above.
4. **Distillation.** MobileNetV3-Small + INT8 remains unbuilt for V3.
   `scripts/distill.py` is written; training and quantisation legs verified,
   CPU benchmark leg untested. This is also where the label-smoothing question
   gets settled, since Müller et al. show the effect is invisible in teacher
   metrics and only appears in the student.
5. **No real-data validation yet.** Every number in this document is measured on
   synthetic data. The project's central claim — that a model trained purely on
   procedurally generated images transfers to photographs — is currently
   **untested**. Candidate benchmark: the Clothing Attributes Dataset
   (Chen, Gallagher & Girod, ECCV 2012; 1,856 real photographs, 11 binary colour
   attributes, torso-centred, Stanford Digital Repository). Note the metric would
   be top-1 against hard human labels rather than KL, and the label sets
   intersect only partially.
6. **Clean single-variable confirmation of the LR result**, if the attribution
   needs to be airtight rather than strongly-inferred.

---

## Label smoothing: what the literature says

### What actually ran

The escalation branch **never fired** in Run A. Across 73 adjustments: 0
increases, 7 decreases, and `s` sat pinned at the 0.02 floor for **90% of
training**. At s=0.02 that is 1.6% of target mass on zero-evidence categories —
not the 12% the top of the range would imply. The practical question is only
whether the floor should be 0.02 or 0.

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

This project intends to distil ResNet-50 → MobileNetV3-Small. So the smoothing
setting is not a minor knob — it propagates into the shipped model.

**The experimental-design consequence is easy to miss:** Müller et al. found
smoothing does *not* hurt the teacher's own generalization or calibration. Only
its distillability. So comparing teacher val loss between s=0 and s=0.02 would
show nothing, and one would wrongly conclude smoothing is harmless. **The
comparison has to be made on the student.**

**[Lukasik et al., ICML 2020](https://arxiv.org/abs/2003.02819)** is the
counterweight: under label noise, smoothing is competitive with loss-correction,
and smoothing the teacher **helps** distillation from noisy data — explicitly
reversing Müller et al. for the noisy regime.

So the question reduces to: **are these labels noisy?**

| Sense | Present here? |
|---|---|
| Annotation noise (annotator disagreement, mislabelling) | **None.** Labels are computed exactly from the generator's label map. |
| Generative-vs-perceptual gap | **Real.** A blue region under a 0.55-strength shadow reads near-black but is still labelled blue. |

The second is genuine noise from the learner's point of view. But it is
*structured and asymmetric* — blue drifts toward black, orange toward brown,
along perceptual adjacencies — whereas Lukasik et al. analyse symmetric noise
and uniform smoothing applies a symmetric correction. Right idea, wrong shape.

**[Geng 2016](https://palm.seu.edu.cn/xgeng/files/tkde16.pdf)** names what this
project is doing: **Label Distribution Learning**, where the target is a
distribution and KL divergence is an established measure. Not an ad hoc choice.

**[Singh et al. 2025](https://arxiv.org/abs/2511.14117)** argues that when the
distribution is genuine rather than a noisy estimate, it *is* the correct
target. These labels are stronger than that paper's case — computed from pixel
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
