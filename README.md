# Clothing Color Recognition with Synthetic Data

Standard clothing color classifiers force a hard label onto a garment. A shirt is yellow or orange — never both, never uncertain. That design choice is convenient but wrong when the thing being classified is color, which is continuous, illumination-dependent, and camera-sensitive. A garment straddling a perceptual boundary, sitting under uneven lighting, or containing multiple visible hues will expose a hard classifier's failure immediately.

This project frames the problem as distribution estimation: the output is a 13-way probability vector, not a single predicted class. A model predicting `yellow: 0.55, orange: 0.35, brown: 0.10` is both more informative and more honest than one forced to commit to `yellow`. The training objective is KL divergence against soft labels derived from actual pixel composition.

The system runs end-to-end: a procedural synthetic data engine, a ResNet-50 backbone, an adaptive fine-tuning stage, and a YOLO-pose-based inference pipeline for live webcam and video testing. This repository documents the first major version—including where it works and exactly where it breaks.

---

## Problem Statement

### The Soft Boundary Problem

A perceptual color category like violet is a region in CIELAB space, not a point. Two shirts with different hue angles can both clearly sit inside it. A third shirt, positioned between violet and blue, falls in a region where neither label is uniquely correct.

Shadows reduce saturation toward gray. Highlights push warm colors toward yellow. JPEG compression at insufficient bitrate destroys fine hue information. Pixel-level color evidence is fragile under real camera conditions.

Any classifier trained to output a single hard label is forced to make an arbitrary commitment at category boundaries—not because the model is wrong, but because the task is physically underspecified. We treat this as an estimation problem: target labels are valid probability distributions that allow probability mass to spread across adjacent categories, where the evidence genuinely supports multiple interpretations.

### Label Space

The 13-category label space is derived from the ISCC-NBS Level 3 color system: `red`, `orange`, `yellow`, `green`, `blue`, `violet`, `purple`, `white`, `gray`, `black`, `pink`, `brown`, `olive`. The adjacent pairs—yellow/orange, blue/violet, white/gray—are exactly where the model's confusion concentrates, as the validation error analysis below shows.

---

## Architecture

### Backbone and Head

The model uses a ResNet-50 backbone pre-trained on ImageNet, with the standard 1000-class head replaced by:

```
Dropout(p=0.4) → Linear(2048, 13)
```

This maps the 2048-dimensional pooled feature vector from ResNet's final block directly to 13 logits. The 40% dropout acts as the primary per-layer regularizer; the KL divergence loss already constrains the output space by requiring a valid distribution, which limits some degenerate prediction behaviors independently.

During training and inference, logits pass through Softmax to produce a distribution summing to 1.0.

### Why KL Divergence Instead of MSE

MSE treats the output vector as a point in Euclidean space. The difference between predicting `yellow: 0.5, orange: 0.5` versus `yellow: 0.5, red: 0.5` is not equal in any perceptual sense, but MSE assigns the same loss in both cases if the squared vector differences happen to match.

KL divergence measures information loss from the target distribution $P$ to the predicted distribution $Q$:

$$
D_{KL}(P \parallel Q) = \sum_{i=1}^{C} P_i \log \frac{P_i}{Q_i}
$$

where $P$ is the soft label vector derived from pixel composition and $Q$ is the Softmax output. This penalizes mass placed on perceptually distant alternatives more severely than MSE does, and aligns naturally with the information-theoretic framing of distribution matching.

---

## Color Taxonomy

Category membership is not defined by manual hue-range rules. We built the 13-category taxonomy from the ISCC-NBS color dictionary by grouping subword tokens, then characterized each category's distribution in CIELAB space (see `color_categorization.ipynb`).

Category spread is measured using Mahalanobis distance rather than Euclidean distance. For a color vector $x$ against category centroid $\mu$ and covariance matrix $S$:

$$
d_M(x, \mu) = \sqrt{(x - \mu)^\top S^{-1}(x - \mu)}
$$

Euclidean distance treats each CIELAB dimension—$L^*$, $a^*$, $b^*$—as equally scaled. Mahalanobis distance normalizes by within-category variance, which captures the fact that some categories extend further along the lightness axis than along the hue plane. This directly governs how the dataset generator samples colors near or on category boundaries.

---

## Synthetic Dataset Engine

The synthetic pipeline (implemented in `utils/dataset_utils.py` across `InnerSquareGenerator`, `OuterSquareGenerator`, and `DatasetGenerator`) generates garment images without any real labeled clothing photographs. Each sample is constructed procedurally and composited onto a real indoor background.

This design was forced by the annotation cost of soft labels. A 13-dimensional distribution target cannot be assigned to a photographed garment by a human annotator without a photometric measurement process. Procedural generation lets the label be *derived* from image content automatically.

### Garment Patch Generation

`InnerSquareGenerator` builds the synthetic garment region from a color sampled from the perceptual library. It constructs one of four appearance patterns:

| Pattern | Sampling Weight |
|---|---:|
| `solid` | 50% |
| `stripes` | 25% |
| `color_blocking` | 15% |
| `polka_dot` | 10% |

Implementations are varied to stress the model across local structure types:

- `solid` applies a low-opacity gradient instead of a flat fill, so the network does not equate "solid color" with perfect pixel uniformity
- `stripes` randomize orientation (vertical, horizontal, angled), width, and inter-stripe spacing
- `color_blocking` generates large-area dominant shapes rather than thin decorative overlays
- `polka_dot` places circles, ellipses, and rounded rectangles at variable density

### Soft Labels from Pixel Composition

After pattern generation, the generator counts pixels by color label and normalizes the result into a 13-dimensional vector. The label is derived from actual image content, not inferred from sampling parameters. Patches with two or three colors produce multi-peaked distributions that directly exercise the KL divergence loss.

### Fold and Texture Simulation

`generate_synthetic_clothing_folds()` produces grayscale fabric-like texture from fractal Brownian motion Perlin noise. The map passes through Gaussian blur, horizontal stretching, and center-cropping back to target resolution, then gets brightness- and contrast-amplified before blending onto the garment patch.

Before blending, the generator applies random scaling and random cropping to the fold source. This prevents the network from keying on tiled texture repetition as a proxy for cloth appearance—the same fold source appears at different scales and offsets across training samples.

### Battle-Hardened Augmentation Environment

After compositing the garment patch onto a background, `_apply_global_lighting()` applies stochastic lighting distortions:

- brightness scaling and color temperature shifts
- saturation scaling and hue jitter
- directional shadow overlays
- Gaussian blur
- salt-and-pepper noise
- JPEG compression artifacts

Brightness draws from $TN(1.0,\ 0.35)$ within a configurable bound; saturation from $TN(1.0,\ 0.4)$. Both use truncated normal sampling rather than uniform sampling, producing a middle-heavy profile with bounded extremes. Uniform sampling would assign equal probability to extreme-underexposure and extreme-overexposure events, which does not reflect real indoor camera distributions.

### Background Preparation

Source backgrounds come from MIT IndoorCVPR_09. Each background is processed through a color-normalization pipeline before the compositing step:

1. Linearization and black-level subtraction
2. Shades of Gray white balance at $p = 6$
3. LAB-space exposure correction
4. CLAHE local contrast enhancement

Without this step, garment colors inherit the ambient hue of the scene, making the synthesized soft labels inconsistent with the visual content.

### Dataset Scale

Training set: 20,000 images. Validation set: 2,000 images. All images are 224×224 pixels. Labels are stored in `labels.csv` alongside the images, with one row per image containing the filename, split assignment, and the 13-dimensional soft label vector.

---

## Training

### First Stage

Run `training.ipynb`. Configuration:

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `5e-2` |
| Scheduler | `ReduceLROnPlateau` |
| Batch size | `64` |
| Early stopping patience | `15` epochs |
| Label smoothing | `0.1` |

Training augmentations: horizontal flips, 90-degree rotational sampling, strong color jitter, custom warm/cool temperature transform, random erasing, ImageNet normalization.

### Adaptive Fine-Tuning

`finetuning.ipynb` implements a second-stage loop that adjusts training parameters based on observed failure structure rather than following a fixed schedule. Per-epoch, the loop recalibrates:

- per-class loss weights from current per-class MAE
- augmentation intensity from validation-loss trend
- label smoothing factor from stagnation detection

Backbone layers receive discriminative learning rates: earlier layers get substantially smaller updates than the classifier head. This prevents low-level ImageNet features from being overwritten by the distribution shift the synthetic data introduces.

---

## Results

### Convergence Behavior

Training loss dropped to approximately `0.11` during the most productive phase, with a transient dip lower before later instability. Validation KL divergence plateaued between `0.75` and `0.81`; the best checkpoint landed at approximately `0.75`. The training-validation gap is large and unmistakable.

The model learns the synthetic task efficiently. It does not generalize equivalently. This is the central unresolved problem in version 1.

### Per-Class MAE

Validation set per-class mean absolute error, sorted by difficulty:

| Color | MAE |
|---|---:|
| black | 0.0410 |
| white | 0.0421 |
| green | 0.0463 |
| blue | 0.0582 |
| red | 0.0648 |
| yellow | 0.0660 |
| orange | 0.0680 |
| violet | 0.0740 |
| purple | 0.0854 |

The three most accurate categories—black, white, green—are perceptually well-separated from all neighbors. The three hardest—violet, purple, orange—each sit adjacent to one or more other categories across both hue and lightness axes.

### Confusion Pairs

Top confusion pairs from validation (mean predicted probability assigned to the wrong class):

| Pair | Confusion |
|---|---:|
| yellow → orange | 0.176 |
| orange → yellow | 0.159 |
| purple → violet | 0.155 |
| violet → purple | 0.140 |
| blue → violet | 0.140 |
| olive → brown | 0.105 |
| white → gray | 0.080 |
| pink → brown | 0.073 |
| brown → pink | 0.073 |
| blue → gray | 0.067 |

Every confused pair shares a perceptual boundary that compression, saturation loss, or shadow gradients can cross. The model is not misclassifying randomly; it is failing precisely at the boundaries where the current synthetic augmentation pipeline cannot yet produce sufficient boundary-crossing examples. The model has learned the coarse geometry of color space—red and green are never confused at scale—but the fine-grained separation between adjacent families remains unresolved.

### Fine-Tuning Outcome

The adaptive fine-tuning run did not close the generalization gap. Training loss continued downward; validation loss drifted flat-to-upward; confusion structure stayed broadly constant.

This outcome is informative. If optimizer changes cannot reduce validation error, the bottleneck sits upstream of the training loop—in the fidelity of the synthetic distribution, in the representational overlap between adjacent categories, or both. Version 2 needs to address those causes directly.

---

## Inference Pipeline

`camera_test.ipynb` and `utils/test_utils.py` run an end-to-end inference pipeline on webcam or video input:

1. Detect people with YOLO pose estimation
2. Localize the torso region from shoulder and hip keypoints
3. Crop and resize to 224×224
4. Predict a 13-way color distribution with the trained ResNet model
5. Annotate the frame with the inferred distribution

This gives the project a direct path from synthetic training to real video testing without a separate deployment step.

---

## Notebook Reference

| Notebook | Role |
|---|---|
| `color_categorization.ipynb` | Builds the 13-category taxonomy from ISCC-NBS and exports categorized color data |
| `preview_color_categories.ipynb` | Visualizes category centroids and Mahalanobis-ranked member colors |
| `dataset_preparation.ipynb` | Normalizes IndoorCVPR_09 backgrounds |
| `dataset_generation.ipynb` | Generates synthetic training and validation images with soft labels |
| `training.ipynb` | Runs first-stage ResNet-50 training |
| `finetuning.ipynb` | Implements adaptive fine-tuning and runs per-class error analysis |
| `camera_test.ipynb` | Live webcam and video inference |

---

## Engineering Lessons from Version 1

1. Distribution prediction via KL divergence is a better-specified target than hard classification for garments that sit on category boundaries.
2. Soft labels derived from pixel composition—not from sampling parameters—ground model feedback in actual image content.
3. Truncated-normal lighting sampling produces a more realistic distortion distribution than uniform sampling; uniform sampling over-represents extreme brightness and saturation events that rarely occur in real footage.
4. Fold texture randomization through variable scale and random crop prevents the network from keying on procedural texture repetition.
5. The training-validation gap in version 1 cannot be closed with optimizer tuning alone. Version 2 needs stronger synthetic boundary realism, specifically at the warm adjacent pairs (yellow/orange) and cool adjacent pairs (violet/blue/purple), and likely some form of real-data calibration.

---

## Status

The project has a working taxonomy, a functioning synthetic engine, a trained model with characterized per-class and per-pair error structure, and a live inference pipeline. Version 2 work will target synthetic-to-real domain alignment and finer-grained boundary treatment for the high-confusion pairs.