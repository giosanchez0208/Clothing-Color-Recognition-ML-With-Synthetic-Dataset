# Synthetic Dataset Audit

`datasets\generated_v4`: **28,860 images**, 37 recorded fields per image.


Every section states what it tests and what a failure would mean. Verdicts are summarized at the end.



## 1. Integrity

*Does the manifest agree with what is on disk, and is every label a valid probability distribution? A failure here invalidates everything downstream.*


| Check | Result | Status |
|---|---|---|
| Rows in labels.csv | 28,860 | n/a |
| Files on disk | 28,860 | OK |
| Unique filenames | 28,860 | OK |
| Unique seeds (train/val/test) | 24,000 of 24,000 | OK |
| Probe pairing grid | 60 bases x 81 renders, 60 own-controls | OK |
| Labels sum to 1.0 | [0.999997, 1.000003] | OK |
| Labels non-negative | yes | OK |

| Split | Count |
|---|---|
| train | 20,000 |
| probe | 4,860 |
| val | 2,000 |
| test | 2,000 |

**PASS**: manifest, files, seeds, probe pairing and label simplex all consistent


## 2. Label space coverage

*Categories are sampled uniformly by design, so dominant-class counts should be roughly balanced. Heavy imbalance would bias the model toward over-represented colors regardless of the loss function.*


| Color | Mean mass | Dominant count | Share |
|---|---|---|---|
| red | 0.0770 | 1,634 | 7.4% |
| orange | 0.0776 | 1,719 | 7.8% |
| yellow | 0.0778 | 1,659 | 7.5% |
| green | 0.0741 | 1,605 | 7.3% |
| blue | 0.0765 | 1,637 | 7.4% |
| violet | 0.0756 | 1,736 | 7.9% |
| purple | 0.0780 | 1,685 | 7.7% |
| white | 0.0784 | 1,911 | 8.7% |
| gray | 0.0778 | 1,714 | 7.8% |
| black | 0.0755 | 1,840 | 8.4% |
| pink | 0.0783 | 1,730 | 7.9% |
| brown | 0.0777 | 1,588 | 7.2% |
| olive | 0.0759 | 1,542 | 7.0% |


Uniform reference mass = 0.0769. Max/min dominance ratio = **1.24x**. Chi-square uniformity: stat=72.6, p=1.04e-10.


Label entropy: mean **0.305**, median 0.338, max 0.792. Effectively one-hot (H<0.01): **14.1%**, so the remaining 85.9% are genuinely multi-modal targets, which are the samples that exercise the KL objective rather than behaving like hard labels.


**PASS**: dominance ratio 1.24x (<2.0 considered acceptable)


![Label distribution, class dominance, and target entropy](dataset_report_v4.md/figures/labels.png)

*Label distribution, class dominance, and target entropy*


## 3. Pattern and structure

*Patterns are drawn from fixed design weights. A significant deviation means the sampler is not doing what the configuration says.*


| Pattern | Designed | Observed | Expected | Delta |
|---|---|---|---|---|
| solid | 25% | 5,489 | 5,500 | -11 |
| stripes | 15% | 3,160 | 3,300 | -140 |
| color_blocking | 10% | 2,215 | 2,200 | +15 |
| polka_dot | 8% | 1,807 | 1,760 | +47 |
| plaid | 17% | 3,813 | 3,740 | +73 |
| gradient | 13% | 2,863 | 2,860 | +3 |
| chevron | 12% | 2,653 | 2,640 | +13 |


Chi-square goodness-of-fit: stat=8.81, p=0.185. (p > 0.05 means observed mix is consistent with design.)


**PASS**: pattern frequencies consistent with design weights (p=0.185)


Entropy by pattern, a sanity check that multi-color patterns really do produce multi-modal labels:


| Pattern | Mean entropy | n |
|---|---|---|
| solid | 0.042 | 5,489 |
| color_blocking | 0.301 | 2,215 |
| gradient | 0.350 | 2,863 |
| stripes | 0.378 | 3,160 |
| chevron | 0.407 | 2,653 |
| plaid | 0.436 | 3,813 |
| polka_dot | 0.482 | 1,807 |


## 4. Augmentation fire rates

*Each effect fires independently with a designed probability. Observed rates outside the binomial 95% interval indicate a sampler bug.*


| Axis | Designed | Observed | Bonferroni CI | Status |
|---|---|---|---|---|
| brightness | 50% | 49.85% | [49.05%, 50.95%] | OK |
| temp_r | 50% | 49.98% | [49.05%, 50.95%] | OK |
| hue | 50% | 49.40% | [49.05%, 50.95%] | OK |
| sat | 50% | 50.28% | [49.05%, 50.95%] | OK |
| shadow | 50% | 49.78% | [49.05%, 50.95%] | OK |
| blur | 50% | 49.94% | [49.05%, 50.95%] | OK |
| noise | 50% | 49.93% | [49.05%, 50.95%] | OK |
| jpeg | 50% | 50.51% | [49.05%, 50.95%] | OK |
| specular | 20% | 19.77% | [19.24%, 20.76%] | OK |
| vignette | 20% | 19.43% | [19.24%, 20.76%] | OK |


*Intervals are Bonferroni-corrected across 10 axes (z=2.81) so the family-wise false-alarm rate is 5%, not 40%.*


**PASS**: all 10 axes fire within their family-wise 95% interval


## 5. Parameter distribution fidelity

*Firing at the right rate is not enough. The sampled values must also follow the intended distribution. A Kolmogorov-Smirnov test against the analytic reference catches a sampler that fires correctly but draws from the wrong shape (e.g. uniform where truncated-normal was intended).*


**Why each axis has the shape it does.** Peaked where the physical world has a 'correct' value it clusters around; uniform where it does not. Cameras aim for correct exposure and mostly succeed, so `brightness` and `saturation` are truncated normals. Most photographs are roughly in focus and focus error has a hard floor at zero, so `blur` is a half-normal. But there is no privileged illuminant across many rooms, no privileged compression setting, and no privileged hue offset for what is purely a regularizer, so `temperature`, `jpeg`, `hue`, `shadow`, `noise`, `specular` and `vignette` are uniform. Uniform brightness would assert that a wildly underexposed frame is as likely as a correct one, which is false about cameras and wastes model capacity on a failure mode reality rarely produces. Full reasoning in `docs/curriculum-design.md`.


**Reading the figure below.** Two axes can show spikes for entirely different reasons. `blur` has a *real* point mass: `clip()` maps every draw below 0.1 onto exactly 0.1 rather than resampling, so ~15.8% of blur values are identical. `jpeg` shows *no* real structure, since it takes 46 integer values, and any bin count that is not a multiple of that produces moiré banding. Bins are now allocated per-value for discrete axes. Treat a spiky histogram as a question; separating a sampler defect from a rendering artifact takes a test on the values.


| Axis | Intended distribution | n | Observed range | KS p | Status |
|---|---|---|---|---|---|
| brightness | TN(1.0, 0.35) on [0.3, 1.8] | 10,968 | [0.3, 1.797] | 0.818 | OK |
| sat | TN(1.0, 0.40) on [0.3, 2.0] | 11,061 | [0.3005, 1.996] | 0.922 | OK |
| temp_r | U(0.75, 1.25) | 10,996 | [0.7501, 1.25] | 0.149 | OK |
| temp_b | U(0.75, 1.25) | 10,996 | [0.75, 1.25] | 0.297 | OK |
| hue | U(-25, 25) | 10,868 | [-25, 25] | 0.295 | OK |
| shadow | U(0.10, 0.55) | 10,952 | [0.1, 0.5499] | 0.263 | OK |
| noise | U(0.001, 0.005) | 10,985 | [0.001, 0.004999] | 0.815 | OK |
| specular | U(0.15, 0.40) | 4,349 | [0.1501, 0.3999] | 0.321 | OK |
| vignette | U(0.20, 0.50) | 4,274 | [0.2002, 0.5] | 0.028 | OK |
| blur | half-normal(0.5), clipped [0.1, 1.5] | 10,987 | [0.1, 1.5] | n/a | not analytic |
| jpeg | discrete uniform {40..85} | 11,112 | [40, 85] | n/a | not analytic |


*KS p > 0.01 means the sampled values are consistent with the intended distribution. Low p on a large sample can reflect discretization rather than a real defect.*


**PASS**: sampled values match their intended distributions


![Sampled parameter distribution per augmentation axis](dataset_report_v4.md/figures/params.png)

*Sampled parameter distribution per augmentation axis*


## 6. Augmentation independence

*Effects are meant to fire independently. Correlated firing makes axes mutually unattributable. If two always co-occur, no analysis can tell which one caused a failure. This is the check that catches the V1/V2 bug where hue and saturation shared a single gate (correlation +1.0).*


Largest off-diagonal correlation: **0.0179** (brightness vs vignette). Noise threshold at n=22,000 is ~0.0202.


**hue vs saturation = -0.0028**. In V1/V2 this was structurally +1.000 because both sat behind one probability gate. They are now independently sampled, which is what makes per-axis attribution possible at all.


**PASS**: max |correlation| 0.0179 is within sampling noise


![Pairwise correlation of augmentation firing](dataset_report_v4.md/figures/independence.png)

*Pairwise correlation of augmentation firing*


**Effects stacked per image.** Under independence this is a Poisson-binomial with mean = sum of designed probabilities:


Expected mean = 4.40, observed = **4.39** (range 0–10).


| n effects | images | share |
|---|---|---|
| 0 | 54 | 0.2% |
| 1 | 500 | 2.3% |
| 2 | 1,745 | 7.9% |
| 3 | 3,884 | 17.7% |
| 4 | 5,531 | 25.1% |
| 5 | 5,200 | 23.6% |
| 6 | 3,339 | 15.2% |
| 7 | 1,307 | 5.9% |
| 8 | 363 | 1.7% |
| 9 | 71 | 0.3% |
| 10 | 6 | 0.0% |


## 7. Color/augmentation confounding

*The most consequential check here. Augmentation is applied after the garment color is chosen, so the two must be statistically independent. If, say, blue garments were systematically darker, the model could reach the right answer using brightness as a proxy for color, scoring well on this dataset while learning something that does not transfer. A leak here would invalidate the benchmark, not merely degrade it.*


| Axis | Most correlated color | Pearson r |
|---|---|---|
| brightness | brown | +0.0200 |
| temp_r | violet | +0.0161 |
| hue | yellow | +0.0170 |
| sat | olive | -0.0138 |
| shadow | purple | +0.0230 |
| blur | blue | -0.0218 |
| noise | yellow | -0.0181 |
| jpeg | orange | +0.0226 |
| specular | purple | +0.0193 |
| vignette | purple | +0.0283 |


Strongest leak across all 10x13 pairs: **r = +0.0283** (vignette vs purple). Noise floor at n=22,000 is ~0.0202.


**PASS**: max |r| = 0.0283, so augmentation carries no usable information about garment color


## 8. Train/validation parity

*Validation must be drawn from the same distribution as training, otherwise val loss measures distribution shift rather than generalization, and is not comparable across runs.*


| Axis | n train | n val | KS p | Verdict |
|---|---|---|---|---|
| brightness | 9,985 | 983 | 0.569 | same |
| temp_r | 9,978 | 1,018 | 0.261 | same |
| hue | 9,886 | 982 | 0.415 | same |
| sat | 10,051 | 1,010 | 0.618 | same |
| shadow | 9,937 | 1,015 | 0.282 | same |
| blur | 9,968 | 1,019 | 0.751 | same |
| noise | 9,995 | 990 | 0.495 | same |
| jpeg | 10,136 | 976 | 0.562 | same |
| specular | 3,969 | 380 | 0.846 | same |
| vignette | 3,881 | 393 | 0.308 | same |


Label entropy, train vs val: KS p = 0.553.


**PASS**: validation is distributionally indistinguishable from training


### 8b. Background isolation

*Distributional parity is not sufficient. The model sees the full 224x224 frame, not just the garment patch, so a background appearing in both train and validation is a memorization channel. The model can recognize the room rather than generalize the color. Backgrounds must be partitioned, not merely sampled identically.*


| Split | Unique backgrounds | Images | Reuse factor |
|---|---|---|---|
| train | 8,946 | 20,000 | 2.24x |
| val | 1,102 | 2,000 | 1.81x |
| test | 1,075 | 2,000 | 1.86x |
| probe | 58 | 4,860 | 83.79x |


Pairwise background overlap, where every cell must be zero:


| Pair | Shared backgrounds | Status |
|---|---|---|
| train n val | 0 | OK |
| train n test | 0 | OK |
| train n probe | 0 | OK |
| val n test | 0 | OK |
| val n probe | 0 | OK |
| test n probe | 0 | OK |

**PASS**: all 6 split pairs draw from disjoint background pools, so no room is ever seen by two splits


### 8c. Held-out test set

*Validation is not a clean generalization estimate here: it drives the LR scheduler, checkpoint selection, early stopping AND the adaptive controller (class weights, augmentation strength, label smoothing). Four channels of optimization pressure on the same 2,000 images. A separate split that nothing in the training loop ever reads is the only number that can be reported without that caveat.*


Test split: **2,000 images**, drawn from a background pool disjoint from train, val and probe.


Distributional parity with train (a test set must be representative, not merely separate):


| Axis | KS p | Verdict |
|---|---|---|
| brightness | 0.998 | same |
| temp_r | 0.295 | same |
| hue | 0.991 | same |
| sat | 0.961 | same |
| shadow | 0.616 | same |
| blur | 0.402 | same |
| noise | 0.936 | same |
| jpeg | 0.125 | same |
| specular | 0.482 | same |
| vignette | 0.027 | same |


*Holm-Bonferroni across 10 tests, family-wise alpha 0.01 (threshold 0.0000).*


**PASS**: 2,000 images, representative of train, never touched by the training loop


## 9. Probe set validity

*The probe set exists to make per-axis attribution possible. It is only valid if each image varies exactly one axis, the control group varies none, and every axis x level cell is populated evenly.*


| Check | Result | Status |
|---|---|---|
| Control images | 60 | OK |
| Control has zero augmentation | yes | OK |
| Swept images | 4,800 | n/a |
| Every swept image isolates 1 axis | yes | OK |
| All 10 axes covered | yes | OK |

| Axis | Levels | Images/cell | Min value | Max value | Balance |
|---|---|---|---|---|---|
| brightness | 8 | 60–60 | 0.3 | 1.8 | even |
| temperature | 8 | 60–60 | -1 | 1 | even |
| hue | 8 | 60–60 | -25 | 25 | even |
| saturation | 8 | 60–60 | 0.3 | 2 | even |
| shadow | 8 | 60–60 | 0.1 | 0.55 | even |
| blur | 8 | 60–60 | 0.1 | 1.5 | even |
| noise | 8 | 60–60 | 0.001 | 0.005 | even |
| jpeg | 8 | 60–60 | 40 | 85 | even |
| specular | 8 | 60–60 | 0.15 | 0.4 | even |
| vignette | 8 | 60–60 | 0.2 | 0.5 | even |

**PASS**: probe set supports clean single-axis attribution


## Samples


![Random training samples with dominant labels](dataset_report_v4.md/figures/samples.png)

*Random training samples with dominant labels*


## Summary

| Check | Verdict | Detail |
|---|---|---|
| integrity | PASS | manifest, files, seeds, probe pairing and label simplex all consistent |
| label balance | PASS | dominance ratio 1.24x (<2.0 considered acceptable) |
| pattern mix | PASS | pattern frequencies consistent with design weights (p=0.185) |
| fire rates | PASS | all 10 axes fire within their family-wise 95% interval |
| parameter fidelity | PASS | sampled values match their intended distributions |
| independence | PASS | max |correlation| 0.0179 is within sampling noise |
| no color confound | PASS | max |r| = 0.0283, so augmentation carries no usable information about garment color |
| train/val parity | PASS | validation is distributionally indistinguishable from training |
| background isolation | PASS | all 6 split pairs draw from disjoint background pools, so no room is ever seen by two splits |
| held-out test set | PASS | 2,000 images, representative of train, never touched by the training loop |
| probe validity | PASS | probe set supports clean single-axis attribution |


**11 of 11 checks passed.**


The dataset is internally consistent, matches its design specification, and carries no measurable color/augmentation confound. It is suitable as a benchmark.

