# Clothing Color Recognition with Synthetic Data

Most clothing color classifiers pick one label and commit. A shirt is yellow or orange—the model has no way to say "both, sort of." That works fine when the shirt is clearly one color under clean lighting. It falls apart the moment a garment sits on a perceptual boundary, catches a shadow, or contains a pattern with three colors in it.

This project treats color prediction as distribution estimation. The model outputs a 13-way probability vector, not a single class. A prediction like `yellow: 0.55, orange: 0.35, brown: 0.10` captures the ambiguity that a hard classifier is structurally unable to express. The training objective is KL divergence against soft labels derived from actual pixel composition in procedurally generated garment images.

No real clothing photographs were labeled for this project. The entire training set is synthetic—procedurally generated garment patches composited onto real indoor backgrounds, with soft labels computed directly from pixel content. This was a deliberate choice: assigning a 13-dimensional probability distribution to a real photograph by hand would require photometric measurement, not human judgment. Procedural generation sidesteps the annotation problem entirely.

The final deployed model is a MobileNetV3-Small distilled from a ResNet-50 teacher, then dynamically quantized to INT8. It is 4.2 MB, runs inference in 8.6ms on CPU, and actually outperforms the 90 MB teacher on validation KL divergence (0.4798 vs 0.5942). The inference pipeline uses YOLO pose estimation to locate torsos in webcam or video feeds, crops the relevant region, and predicts the color distribution in real time.

---

## The Problem

Color is continuous. Category boundaries are not.

A perceptual color category like violet occupies a region in CIELAB space, not a point. Two shirts with different hue angles can both sit inside it. A third shirt, positioned between violet and blue, falls where neither label is uniquely correct. Shadows reduce saturation toward gray. Highlights push warm tones toward yellow. JPEG compression at low bitrate destroys fine hue information entirely.

A classifier trained to output a single hard label is forced to make an arbitrary commitment at every category boundary. We reframe the task: target labels are valid probability distributions that allow mass to spread across adjacent categories proportional to the pixel evidence.

### Label Space

The 13 categories come from the ISCC-NBS Level 3 color system: `red`, `orange`, `yellow`, `green`, `blue`, `violet`, `purple`, `white`, `gray`, `black`, `pink`, `brown`, `olive`. We built the taxonomy by tokenizing all ISCC-NBS compound color names (e.g., "dark purplish red") into subwords, discarding modifiers (`dark`, `light`, `grayish`, etc.), and mapping the remaining base hue words to 13 simplified categories. The details live in `color_categorization.ipynb`. The adjacent pairs—yellow/orange, blue/violet, violet/purple, white/gray—are where all the interesting failures happen, as the confusion analysis below shows.

---

## Synthetic Dataset Engine

The annotation cost of soft labels made real-data collection impractical, so we built a procedural generator instead. Every training image is constructed from scratch: a garment patch with a known color composition, composited onto a real indoor background, then degraded with stochastic lighting and compression artifacts. The label is derived from what the generator actually drew, not from any human judgment call.

### Color Library

V1 used 286 colors extracted from the ISCC-NBS dictionary, distributed unevenly across categories—green had 44 entries, white had 6. Cross-category duplicates and ambiguous boundary colors contaminated the training signal.

V2 replaced this with a confusion-aware Voronoi resampling pipeline. Rather than pruning, the preparation stage resamples clean colors from scratch. Original colors survive only if they pass an asymmetric Voronoi check: the distance to the assigned centroid must be less than the distance to any other centroid, scaled by a pair-specific margin derived from V1's confusion matrix. The most confused pair (orange/yellow, confusion 0.173) gets a margin of 1.4—a color must be 40% closer to its own centroid than to the confused neighbor. Non-confused pairs use standard Voronoi (margin 1.0). Categories that fall short are filled by sampling from a multivariate Gaussian fit to the surviving seeds, with every sample re-checked against the same margin criteria.

The result: 325 colors, exactly 25 per category, with zero cross-category overlap. Violet had the hardest time—only 1 of 11 original colors survived the Voronoi check, and 79 Gaussian samples were rejected before 24 filled.

### Garment Patch Patterns

V1 supported 4 pattern types. V2 adds 3 more and rebalances the sampling weights:

| Pattern | V1 | V2 | Notes |
|---|---:|---:|---|
| `solid` | 50% | 25% | Low-opacity gradient overlay, not flat fill |
| `plaid` | — | 17% | Overlapping horizontal + vertical stripes with alpha blending |
| `stripes` | 25% | 15% | Random orientation, width, spacing |
| `gradient` | — | 13% | Smooth ombre transitions along configurable axes |
| `chevron` | — | 12% | V-shaped zigzag bands with variable slope |
| `color_blocking` | 15% | 10% | Large-area dominant shapes |
| `polka_dot` | 10% | 8% | Circles, ellipses, rounded rectangles |

Multi-color patterns (stripes, plaid, color blocking) produce multi-peaked label distributions that directly exercise the KL divergence loss. A striped shirt with 60% blue pixels and 40% white pixels gets a label that says exactly that.

### Fold Texture

Synthetic clothing folds are generated from fractal Brownian motion Perlin noise (5 octaves, persistence 0.5), then Gaussian-blurred, horizontally stretched, and center-cropped back to target resolution. Before blending, the generator applies random scaling and random cropping to the fold source—this prevents the network from learning to recognize tiled texture repetition as a proxy for cloth appearance. V1 blended folds using multiply mode only. V2 randomly selects between multiply and overlay, which adds contrast variation that better matches real fabric sheen under directional light.

### Augmentation Pipeline

After compositing the garment patch onto an indoor background from MIT IndoorCVPR_09 (which itself passes through a normalization pipeline: linearization, Shades of Gray white balance at $p = 6$, LAB exposure correction, and CLAHE), the generator applies stochastic lighting distortions:

- Brightness scaling from $TN(1.0,\ 0.35)$ within $[0.3,\ 1.8]$
- Independent R/B channel shifts for color temperature
- Hue jitter $\pm25°$ and saturation from $TN(1.0,\ 0.4)$
- Directional shadow overlays with blue tint
- Gaussian blur, salt-and-pepper noise, JPEG compression down to quality 40

Both brightness and saturation use truncated normal sampling rather than uniform. Uniform sampling assigns equal probability to extreme underexposure and extreme overexposure, which does not reflect real camera distributions. Truncated normal concentrates mass near typical conditions while still generating hard examples at the tails.

V2 adds two more augmentations: specular highlights (bright elliptical hotspots simulating fabric sheen) and vignette (edge darkening simulating lens falloff).

### Dataset Scale

20,000 training images, 2,000 validation images. All 224×224 pixels, JPEG quality 95. Labels stored in `labels.csv`—one row per image containing the filename, split assignment, and the 13-dimensional soft label vector summing to 1.0.

---

## Architecture

### Backbone and Head

The teacher model is ResNet-50 pretrained on ImageNet, with the standard 1000-class head replaced by `Dropout(p=0.4) → Linear(2048, 13)`. The 40% dropout acts as the primary per-layer regularizer. During training and inference, logits pass through Softmax to produce a valid distribution.

### Why KL Divergence

MSE treats the output as a point in Euclidean space. Predicting `yellow: 0.5, orange: 0.5` and `yellow: 0.5, red: 0.5` can produce equal MSE if the squared vector differences happen to match—but perceptually these predictions are not equivalently wrong. KL divergence measures information loss:

$$D_{KL}(P \parallel Q) = \sum_{i=1}^{C} P_i \log \frac{P_i}{Q_i}$$

where $P$ is the soft label and $Q$ is the model's Softmax output. This penalizes probability mass placed on perceptually distant categories more severely than MSE does.

---

## Training

### V1: First Pass

V1 training (`notebooks/training.ipynb`) used AdamW with lr `1e-4`, weight decay `5e-2`, batch size 64, and ReduceLROnPlateau scheduling. Training augmentations: horizontal flips, 90° rotational sampling, ColorJitter, a custom warm/cool temperature transform, random erasing, and ImageNet normalization. Best V1 val loss: **0.7474** at epoch 25.

V1 fine-tuning (`notebooks/finetuning.ipynb`) introduced an adaptive loop: per-class MAE drives class weights, val loss trend drives augmentation intensity, stagnation triggers label smoothing increases. Discriminative learning rates gave earlier backbone layers 1000× smaller updates than the classifier head. The result was underwhelming—best fine-tuned val loss of **0.7469**, a marginal improvement. Early stopping triggered after 15 epochs as val loss drifted upward. The confusion structure barely changed. Orange/yellow remained the worst pair at 0.173 average leakage.

This told us something useful: the bottleneck was upstream of the optimizer. No amount of learning rate scheduling or loss weighting would fix a synthetic distribution that didn't accurately represent real boundary colors. Version 2 needed to attack the data, not the training loop.

### V2: Better Data, Same Model

V2 training (`notebooks_v2/training.ipynb`) started from the V1 fine-tuned checkpoint and trained on the V2 dataset with the resampled color library, 7 pattern types, and expanded augmentations. All 23.5M parameters were trainable from the start with discriminative learning rates:

| Layer Group | Params | LR Multiplier |
|---|---:|---|
| backbone_early | 1,444,928 | ×0.001 |
| layer3 | 7,098,368 | ×0.01 |
| layer4 | 14,964,736 | ×0.1 |
| fc | 26,637 | ×1.0 |

Base LR: `5e-5`. The same adaptive loop from V1 fine-tuning carried over: per-class MAE → class weights (0.5–3.0), val loss trend → augmentation strength (0.3–0.8), stagnation → label smoothing (0.02–0.15).

Best V2 val loss: **0.5942** at epoch 71 — a 20% improvement over V1's 0.7474. The training-validation gap narrowed substantially, confirming the hypothesis that V1's generalization failure was a data problem, not a model capacity problem.

The hardest classes at the end of V2 training: purple (MAE 0.0665), brown (0.0613), gray (0.0609). The easiest: black (0.0342). Every confused pair involves categories sharing a perceptual boundary—violet/purple, orange/brown, white/gray.

---

## Distillation and Quantization

A 90 MB ResNet-50 running at 152ms per frame on CPU is not deployable for real-time inference. We distilled the V2 teacher into a MobileNetV3-Small student (1.5M params, initialized from ImageNet weights) using a combined loss:

$$\mathcal{L} = 0.7 \cdot D_{KL}\left(\sigma(z_s / T)\ \|\ \sigma(z_t / T)\right) \cdot T^2 + 0.3 \cdot D_{KL}\left(\sigma(z_s)\ \|\ y\right)$$

Temperature $T = 4.0$, 60 epochs with cosine annealing LR from `1e-3` to `1e-6`.

The student outperformed its teacher. Best student val loss: **0.4798** vs. teacher's 0.5942. Per-class MAE improved on every single category—purple dropped from 0.0665 to 0.0574, violet from 0.0534 to 0.0446. The smaller model's inductive bias (depthwise separable convolutions, inverted residuals) apparently regularizes better than the ResNet's deeper parameterization for this specific 13-class problem. We did not expect this result.

Dynamic INT8 quantization (on `nn.Linear` and `nn.Conv2d`) was essentially lossless:

| | Teacher | Student FP32 | Student INT8 |
|---|---:|---:|---:|
| Val Loss (KL) | 0.5942 | 0.4798 | 0.4799 |
| Top-1 Accuracy | — | 68.8% | 68.9% |
| Size | 90.1 MB | 6.0 MB | 4.2 MB |
| CPU Latency | 152.2 ms | 8.9 ms | 8.6 ms |

21× smaller, 18× faster, better accuracy. The INT8 model ships as `student_int8.pth`.

---

## Inference Pipeline

`camera_test.ipynb` and `utils/test_utils.py` run end-to-end on webcam or video input:

1. YOLOv11n-pose detects people and extracts shoulder/hip keypoints (runs at 384px input for CPU efficiency)
2. `_estimate_torso` derives a torso bounding box from whatever keypoints are visible, falling back to proportional estimates when shoulders or hips are occluded
3. `compose_input` builds a 224×224 composite matching the training data layout: a tight torso crop (112×112) pasted over a wider context crop, mimicking the inner-square-on-background structure the model was trained on
4. The MobileNetV3-Small INT8 model predicts a 13-way distribution. Colors above `max(0.08, top_prob × 0.35)` are reported—allowing multi-color predictions for patterned clothing. If even the top color falls below 0.25, the prediction is marked uncertain
5. Bounding boxes and labels are drawn on the frame, with label background color matching the predicted color and text color chosen by luminance contrast

The entire pipeline avoids PIL and unnecessary copies. Normalization happens directly in numpy/torch on the 224×224 crop. On a laptop CPU, the bottleneck is YOLO pose detection, not the color model.

---

## Notebook Reference

| Notebook | What it does |
|---|---|
| `notebooks/color_categorization.ipynb` | Builds 13-category taxonomy from ISCC-NBS, exports categorized colors |
| `notebooks/preview_color_categories.ipynb` | Visualizes category centroids and Mahalanobis-ranked member colors in CIELAB |
| `notebooks/dataset_preparation.ipynb` | Normalizes IndoorCVPR_09 backgrounds (white balance, CLAHE) |
| `notebooks/dataset_generation.ipynb` | V1 synthetic dataset generation (4 patterns, 286 colors) |
| `notebooks/training.ipynb` | V1 ResNet-50 training |
| `notebooks/finetuning.ipynb` | V1 adaptive fine-tuning with discriminative LRs |
| `notebooks/camera_test.ipynb` | Live webcam and video inference |
| `notebooks_v2/dataset_preparation.ipynb` | Confusion-aware Voronoi resampling of color library (325 colors, 25/category) |
| `notebooks_v2/dataset_generation.ipynb` | V2 synthetic dataset generation (7 patterns, new augmentations) |
| `notebooks_v2/training.ipynb` | V2 adaptive training on cleaned dataset |
| `notebooks_v2/distill_and_quantize.ipynb` | Knowledge distillation to MobileNetV3-Small + INT8 quantization |

---

## What I Learned

**The data bottleneck was real.** V1's fine-tuning loop had every trick—adaptive class weights, dynamic augmentation, label smoothing, discriminative learning rates—and moved val loss from 0.7474 to 0.7469. Three thousandths. The same training loop on V2's cleaned data moved it to 0.5942 without any algorithmic changes. The lesson is old but apparently needs re-learning: when the optimizer can't push validation loss down, the problem is almost always in the data.

**Confusion-aware Voronoi resampling worked.** The idea was simple: if V1's model confused orange and yellow, then the color library should not contain ambiguous colors near that boundary. Widening the dead zone between confused categories with pair-specific margins (derived from the actual confusion matrix) produced a cleaner training signal. Violet was the extreme case—10 of 11 original colors were rejected, and the stage had to generate 79 Gaussian samples before 24 passed. The resulting library has zero cross-category contamination.

**The student beat the teacher.** The MobileNetV3-Small distilled from the ResNet-50 achieved lower val loss on every metric. This was not the plan. The expected outcome was a modest accuracy trade for a large speed gain. Instead, the student's lighter inductive bias—depthwise separable convolutions and inverted residuals—apparently regularizes better than ResNet for a 13-class problem where the teacher was already overfitting slightly to the synthetic distribution. INT8 quantization on top was free—val loss changed by 0.0001.

**Synthetic data can work, but the failure modes are specific.** The model never confuses red and green. It never puts mass on black when the shirt is yellow. The coarse geometry of color space is well-learned. Every failure concentrates at adjacent category boundaries—violet/purple, orange/brown, white/gray—where real-world lighting shifts can push a color across the boundary. The synthetic augmentation pipeline handles many of these shifts, but the remaining gap is the difference between simulated and photographed light transport.

---

## Status

The project ships a working end-to-end pipeline: taxonomy construction, synthetic data generation with soft labels, a two-version training progression, knowledge distillation, INT8 quantization, and live inference on webcam and video. The final model is 4.2 MB and runs at 8.6ms per inference on CPU. The pair-level confusion structure is well-characterized and directly informs where future work should focus—tighter boundary treatment for violet/purple, orange/brown, and white/gray, potentially incorporating a small number of real labeled examples to calibrate the synthetic distribution.
