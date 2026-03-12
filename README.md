# Clothing Color Recognition with Synthetic Data

This project explores a harder version of clothing color recognition than standard single-label classifiers usually attempt. The goal is not to force every garment into one rigid class, but to predict a probability distribution over color categories that reflects real-world ambiguity.

The current system is built around a synthetic dataset engine, a ResNet-50 backbone trained with KL divergence on soft labels, and a local inference pipeline for webcam and video testing. It has produced meaningful progress, but it has also exposed the limits of the current synthetic setup, especially around ambiguous neighboring hues.

This repository represents the first major version of that work. A version 2 is planned.

## Problem Statement

### Color Is Not a Binary Category

Most clothing color models are framed as hard classification: a shirt is `yellow` or `orange`, `blue` or `violet`, `white` or `gray`. That framing breaks down quickly in practice.

Color is a spectrum, not a set of isolated buckets. A garment can sit on a boundary, contain multiple visible hues, or shift appearance under shadows, highlights, camera white balance, and low-quality compression. In other words, the model is often asked to make a hard decision in a space where the evidence is physically soft.

This project is built around a different objective:

- predict the color composition of a garment as a probability distribution
- preserve uncertainty instead of hiding it
- treat ambiguity as signal, not failure

If a shirt reads as mostly yellow with a meaningful orange component, the model should be able to express that. If violet and blue are both plausible because lighting has shifted the observed pixels, the output should show that uncertainty directly.

### The Hue Overlap Problem

One of the main challenges here is what I refer to as the hue overlap problem. Lighting, folds, shadow ramps, saturation shifts, and camera artifacts can move observed pixels across category boundaries. A physically orange garment may drift toward yellow under highlights. A violet garment may collapse toward blue in lower saturation regions. White and gray can easily blur together once local lighting becomes uneven.

Instead of trying to eliminate that ambiguity entirely, this project is designed to navigate it intelligently. The training target is a valid probability distribution, and the model is optimized to learn relative color composition rather than just the single winning class.

## Project Goals

- Build a soft-label clothing color recognizer that outputs a calibrated 13-way color distribution.
- Use procedural synthetic data to scale training without hand-labeling thousands of real garments.
- Simulate realistic lighting, folds, texture, and pattern variation so the model sees more than flat color patches.
- Analyze where the model fails, especially on adjacent hue families, to guide a stronger version 2.

## Current Pipeline

The current workflow is:

1. Build a perceptual color taxonomy from the ISCC-NBS color system.
2. Clean and neutralize indoor scene backgrounds from MIT IndoorCVPR_09.
3. Generate synthetic clothing samples by compositing an inner garment patch onto real backgrounds.
4. Train a ResNet-50 model on soft labels using KL divergence.
5. Fine-tune with adaptive augmentation, smoothing, and class weighting.
6. Test locally on webcam and video through a pose-based inference pipeline.

## Color Taxonomy and Label Space

The project uses 13 color categories:

- red
- orange
- yellow
- green
- blue
- violet
- purple
- white
- gray
- black
- pink
- brown
- olive

These categories are built from the ISCC-NBS Level 3 color system using subword-based grouping in `color_categorization.ipynb`. Colors are analyzed in CIELAB space, and category spread is measured with Mahalanobis distance rather than plain Euclidean distance.

That matters because a category is not spherical in perceptual space. A color family can vary more in lightness than in hue, or vice versa. Mahalanobis distance accounts for the actual covariance of each category cloud, which makes it a better basis for sampling and centroid-relative analysis.

For a color vector $x$ with category centroid $\mu$ and covariance matrix $S$, the distance is:

$$
d_M(x, \mu) = \sqrt{(x - \mu)^\top S^{-1}(x - \mu)}
$$

This gives the dataset generator a principled way to sample colors near or far from a category center while still respecting the shape of the underlying color distribution.

## Synthetic Dataset Engine

This is the strongest part of the current system.

The synthetic pipeline is implemented primarily in `utils/dataset_utils.py`, centered around `InnerSquareGenerator`, `OuterSquareGenerator`, and `DatasetGenerator`.

### InnerSquareGenerator

`InnerSquareGenerator` creates the garment patch that acts as the synthetic clothing region. It does not just paint a flat square. It generates a structured sample with:

- a primary color sampled from a perceptual color library
- optional secondary colors
- procedural patterns
- synthetic fold texture
- global lighting distortions after compositing

The output label is not a single class. It is a soft vector describing the proportion of each color present in the generated sample.

### Procedural Generation

The generator currently supports four garment appearance modes:

- solid
- stripes
- color blocking
- polka dot

These are sampled with the following weights:

- `solid`: 50%
- `stripes`: 25%
- `color_blocking`: 15%
- `polka_dot`: 10%

Pattern generation is deliberately varied:

- `solid` applies a low-opacity gradient instead of a perfectly uniform patch
- `stripes` can be vertical, horizontal, or angled, with random stripe widths and spacing
- `color_blocking` produces large dominant shapes instead of thin decorative fragments
- `polka_dot` uses circles, ellipses, and rounded rectangles across a random density range

This matters because a training image should not teach the model that "shirt color" means one perfectly flat region with no local structure.

### Soft Labels from Pixel Composition

Each generated sample tracks a label map internally. After pattern generation, the system computes the percentage of pixels occupied by each color label and normalizes the result into a 13-dimensional distribution.

That means the label reflects actual image composition rather than a guessed dominant class. This is the core reason the training problem can be formulated as distribution prediction instead of hard classification.

### Battle-Hardened Augmentation Environment

After compositing onto a real background, the synthetic sample is exposed to a deliberately harsh augmentation pipeline in `_apply_global_lighting()`.

The current pipeline includes stochastic combinations of:

- brightness scaling
- color temperature shifts
- saturation scaling
- hue jitter
- directional shadows
- Gaussian blur
- salt-and-pepper noise
- JPEG compression

The important detail is that this is not purely uniform random noise.

Brightness and saturation use truncated normal sampling:

- brightness uses $TN(1.0, 0.35)$ within a configurable range
- saturation uses $TN(1.0, 0.4)$ within a configurable range

This gives the generator a more realistic middle-heavy distribution with bounded extremes, which is a better fit for real camera footage than equally likely low and high distortions. In practice, this creates a more battle-hardened training environment for gym-style, indoor, and inconsistent lighting.

### Synthetic Folds and Texture

The fold texture generator is another strong component.

`generate_synthetic_clothing_folds()` builds grayscale fabric-like texture from fractal Brownian motion Perlin noise, then applies:

- Gaussian blur
- horizontal stretching
- center cropping back to target size
- brightness amplification
- contrast amplification

When this texture is applied to the garment patch, the generator also introduces random scaling and random cropping before blending. That prevents the model from memorizing a few repeated texture stamps. The same fold source can be seen at different scales, offsets, and local appearances across training samples.

This is a small but important design choice. Without it, the network could partially overfit to a procedural artifact instead of learning color under textured cloth-like variation.

### Background Preparation

The outer scene comes from the MIT IndoorCVPR_09 dataset. Backgrounds are flattened into one directory and then processed with a color-balancing pipeline before being used in generation.

The background cleaning steps are:

1. linearization and black-level subtraction
2. Shades of Gray white balance with $p = 6$
3. LAB-space exposure handling
4. CLAHE-based local contrast enhancement

The objective is to create a daylight-neutral canvas so that synthetic garment colors remain interpretable and reproducible instead of being immediately corrupted by background color cast.

### Current Synthetic Dataset Size

The current dataset generation notebook creates:

- 20,000 training images
- 2,000 validation images

All images are generated at `224 x 224` resolution and stored alongside `labels.csv`, where each row contains the filename, split, and 13-way soft label vector.

## Model Architecture

The model uses a ResNet-50 backbone pre-trained on ImageNet.

The standard ImageNet classifier is replaced with a lightweight 13-output head:

- dropout with `p = 0.4`
- linear projection from 2048 features to 13 color classes

The network outputs logits, and those logits are converted through Softmax during training and inference to produce a valid probability distribution over the 13 color classes.

This is important because the project is explicitly not solving a hard one-class-only problem. The output must sum to $1.0$, and every component must be interpretable as probability mass assigned to one of the color families.

## Why KL Divergence Instead of MSE

The training target is a probability distribution, so the loss should compare distributions properly.

This project uses KL divergence rather than mean squared error:

$$
D_{KL}(P \parallel Q) = \sum_{i=1}^{C} P_i \log \frac{P_i}{Q_i}
$$

where:

- $P$ is the target soft label distribution
- $Q$ is the predicted distribution after Softmax

Why this matters:

- MSE treats the output as just another vector in Euclidean space.
- KL divergence measures information loss between distributions.
- KL divergence respects the hierarchy of the target distribution more naturally.

If the true target is something like `yellow 0.55 / orange 0.35 / brown 0.10`, then predicting mass on nearby alternatives is very different from predicting mass on unrelated colors. KL divergence captures that structure better than MSE, which is one of the main reasons it was chosen here.

This is one of the central technical ideas in the project: the task is distribution matching, not point regression and not hard classification.

## Training Setup

### Initial Training

The main training run in `training.ipynb` uses:

- optimizer: AdamW
- learning rate: `1e-4`
- weight decay: `5e-2`
- scheduler: `ReduceLROnPlateau`
- batch size: `64`
- early stopping patience: `15`
- label smoothing: `0.1`

Training augmentations include:

- horizontal flips
- 90-degree rotational sampling
- strong color jitter
- custom warm/cool color temperature augmentation
- random erasing
- ImageNet normalization

### Adaptive Fine-Tuning

The second-stage notebook explores a more ambitious fine-tuning strategy.

Instead of simple continued training, the fine-tuning loop dynamically recalibrates:

- class weights from per-class MAE
- augmentation strength from validation-loss trend
- label smoothing from stagnation signals

It also uses discriminative learning rates, where earlier backbone layers receive far smaller updates than later layers and the classifier head.

In practical terms, this fine-tuning notebook is an attempt to make the training loop react to actual failure structure instead of treating all classes and all epochs the same way.

## Current Model State

This project has real progress, but the model is not finished.

### Convergence Plateau

The initial training run improved quickly and then hit a plateau. Validation KL divergence settled around the mid-`0.7x` range, with the training curve dropping much further than the validation curve.

From the saved plots and notebook outputs:

- validation loss plateaued at roughly `0.75` to `0.81`
- the best reported validation point sits around `0.75`
- training loss fell much lower, reaching roughly `0.11` during the stronger phase of learning, with an even lower transient dip before later instability

That is the main headline: the model learns the synthetic task well enough to reduce training error sharply, but generalization saturates early.

### Generalization Gap

The current project shows a clear train/validation gap.

At a high level, the model learns synthetic structure faster than it learns robust color semantics that transfer across the validation distribution. Framed more bluntly: the network is good at fitting the world the generator creates, but the generator still does not capture all the ambiguity the model needs to survive.

This is one of the most useful outcomes of the first version. The gap is not just a performance number. It is evidence of feature mismatch and incomplete regularization between the synthetic universe and the real recognition problem.

## Performance Analysis

### Per-Class MAE

The per-class error analysis on the validation set shows a clear hierarchy of difficulty.

Representative values from the saved analysis are:

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

The hardest classes are not random. They are exactly the families where perceptual boundaries are weakest under lighting variation.

### Top Confused Pairs

The confusion matrix is one of the most informative outputs in the project because it shows where the current synthetic setup still runs into the limit of ambiguity.

Top confused pairs from the validation analysis include:

- `yellow -> orange`: `0.176`
- `orange -> yellow`: `0.159`
- `purple -> violet`: `0.155`
- `violet -> purple`: `0.140`
- `blue -> violet`: `0.140`
- `olive -> brown`: `0.105`
- `white -> gray`: `0.080`
- `pink -> brown`: `0.073`
- `brown -> pink`: `0.073`
- `blue -> gray`: `0.067`

These are not just mistakes. They are a map of the system's ambiguity frontier.

Warm overlaps such as orange versus yellow, cool overlaps such as purple versus violet, and neutral overlaps such as white versus gray are exactly the regions where pixel evidence is most vulnerable to illumination and texture effects.

### What the Confusion Matrix Is Saying

The model is already learning the coarse geometry of color space. It is not confusing red with green at scale. The failures are concentrated around adjacent perceptual neighborhoods.

That is a meaningful success and a meaningful limitation at the same time.

The success is that the model has learned a structured color manifold rather than a random classifier.

The limitation is that the current generator still does not provide enough boundary realism for the model to consistently separate neighboring hue families under realistic distortions.

## Fine-Tuning Outcome

The adaptive fine-tuning stage was a useful experiment, but it did not solve the generalization problem.

The fine-tuning plots show:

- training loss continuing downward
- validation loss staying high and slightly drifting upward
- confusion structure remaining broadly similar after fine-tuning

That is important. It suggests the bottleneck is not only optimizer tuning. The bigger issue is likely the fidelity of the synthetic-to-real mismatch and the unresolved ambiguity around neighboring categories.

In other words, the project did not just hit an optimization plateau. It hit a representation plateau.

## Inference Pipeline

The repository also contains a local test pipeline in `camera_test.ipynb` and `utils/test_utils.py`.

The current inference stack:

- detects people using YOLO pose
- estimates torso regions from shoulder and hip keypoints
- builds a composite torso-centered input
- predicts shirt-color distributions with the trained ResNet model
- annotates webcam or video frames with the inferred colors

This makes the project more than a pure offline experiment. There is already an end-to-end path from synthetic training to real local video testing.

## Notebook Guide

The notebooks are organized as an experimental workflow:

| Notebook | Purpose |
|---|---|
| `color_categorization.ipynb` | Builds the 13-category taxonomy from ISCC-NBS colors and exports categorized color data. |
| `preview_color_categories.ipynb` | Visualizes category centroids and Mahalanobis-ranked members. |
| `dataset_preparation.ipynb` | Prepares IndoorCVPR backgrounds and neutralizes their color response. |
| `dataset_generation.ipynb` | Generates the synthetic dataset with soft labels. |
| `training.ipynb` | Runs first-stage training on synthetic data. |
| `finetuning.ipynb` | Performs adaptive fine-tuning and deep error analysis. |
| `camera_test.ipynb` | Runs local webcam and video-based inference. |

## Main Lessons So Far

This version of the project produced several useful engineering lessons:

1. Soft-label learning is the right framing for ambiguous garment color recognition.
2. KL divergence is a better fit than MSE for comparing color distributions.
3. Synthetic data can push the model surprisingly far, but only if lighting, texture, and pattern variance are treated seriously.
4. The hardest errors are not random classification misses. They cluster around neighboring hue families and neutral boundaries.
5. Better optimizer tricks alone are not enough. Version 2 will likely need a stronger synthetic engine, better domain alignment, or more real-data calibration.

## Status

This project is already past the idea stage. It has:

- a principled color taxonomy
- a custom synthetic dataset engine
- a soft-label training setup with KL divergence
- measurable strengths and measurable failure modes
- a working inference path for local video experiments

The current model is not the final answer, but it is a strong technical first version and a useful base for a more realistic version 2.