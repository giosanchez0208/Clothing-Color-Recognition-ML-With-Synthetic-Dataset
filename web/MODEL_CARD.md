---
license: mit
library_name: pytorch
pipeline_tag: image-classification
tags:
  - image-classification
  - color-recognition
  - synthetic-data
  - knowledge-distillation
  - label-distribution-learning
  - mobilenetv3
  - onnx
model-index:
  - name: clothing-color-recognition
    results:
      - task:
          type: image-classification
          name: Clothing color distribution
        dataset:
          type: synthetic
          name: Procedurally generated garments (V4 held-out split)
        metrics:
          - type: accuracy
            name: top-1
            value: 0.628
          - type: kl_divergence
            name: KL divergence
            value: 0.5604
          - type: mae
            name: MAE
            value: 0.0518
---

# Clothing color recognition

Predicts a **probability distribution over 13 color categories** for a garment,
rather than assigning a single label. A shirt that is two-thirds navy and
one-third white has a correct answer that no single label can express, so the
target is a distribution and the loss is KL divergence, which is Label
Distribution Learning in the sense of Geng (2016).

Trained on **28,860 procedurally generated images and zero real photographs**.

| | |
|---|---|
| Architecture | MobileNetV3-Small, 13-way head |
| Parameters | 1.53 M |
| Size | 6.0 MB FP32, 4.2 MB INT8, 5.85 MB ONNX |
| Teacher | ResNet-50, distilled at T=4.0, alpha=0.7 |
| Classes | red, orange, yellow, green, blue, violet, purple, white, gray, black, pink, brown, olive |

## Try it

Browser demo, model runs client-side:
<https://huggingface.co/spaces/giosanchez0208/clothing-color-recognition>

## Results

Held-out test split, 2,000 images, read once, backgrounds drawn from a pool
disjoint from train, validation, and the probe set.

| Model | KL | top-1 | MAE | Size | CPU |
|---|---:|---:|---:|---:|---:|
| Teacher, ResNet-50 | 0.6968 | 58.1% | 0.0589 | 90.1 MB | 45.4 ms |
| Student FP32 | 0.5600 | 62.7% | 0.0518 | 6.0 MB | 6.2 ms |
| **Student INT8** | **0.5604** | **62.8%** | **0.0518** | **4.2 MB** | **6.9 ms** |

Measured on the V4 test split. The labels are the Gaussian mixture posterior
over categories rather than one-hot, so a color between green and yellow is
labeled as being between them. These numbers are therefore not comparable to the
0.4800 an earlier version reported against one-hot labels. Re-scoring that
earlier model on these labels gives KL 0.5852 and top-1 63.5%, so the change
improved KL by 4.2% while costing 0.7 points of top-1.

The student beats its teacher on every metric, which V2 of this project also
observed but measured with backgrounds shared between train and validation.
Here the split is honest and the inversion holds: 23% lower KL and 4.2 points
higher top-1 than the model it was distilled from. The likeliest mechanism is
that MobileNetV3's depthwise-separable convolutions impose a stronger inductive
bias than ResNet-50's deeper parameterization, which regularizes better on a
13-class problem where the teacher is mildly overfitting the synthetic
distribution.

INT8 dynamic quantization is free here: KL moves by 0.0001 and top-1 goes up a
tenth of a point.

## Files

| File | What it is |
|---|---|
| `distilV4_student.onnx` | ONNX, opset 17, dynamic batch. Named I/O: `image` -> `color_logits`. Use this for the browser or cross-platform inference. |
| `distilV4_student_fp32.pth` | PyTorch state dict, FP32. |
| `distilV4_student_int8.pth` | PyTorch state dict, dynamically quantized. CPU only. |
| `category_mixture_v4.json` | The fitted Gaussian per category, so the labels can be reproduced. |

The checkpoints carry an `architecture` field so a loader does not have to guess
which of the three it holds.

## Input format, which is not a plain resize

The model was trained on a composite, not on photographs: a **112x112 crop of
the garment region pasted into the middle of a 224x224 wider-context crop**,
then ImageNet-normalized. Feeding it a plain resized photo produces confident
nonsense. The reference implementation is `compose_input()` in
`scripts/predict.py` in the source repository, and a JavaScript port is in the
Space.

```python
IMG_SIZE, INNER = 224, 112
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
```

Outputs are raw logits over the 13 classes in the order listed above. Apply
softmax to get the distribution.

## Limitations

**No real-data validation.** Every number above is measured on synthetic data.
The central claim, that a model trained purely on procedurally generated images
transfers to photographs, is currently untested at scale.

**Category boundaries are where the remaining error lives.** In V3 this showed up
as a hard failure: pale green read as yellow, because the learned green/yellow
boundary sat near 135 degrees of hue while the color library put it at 117.7.
The cause was the generator drawing colors with inverse-Mahalanobis weights,
which starved the boundary region, and then labeling those rare boundary colors
as 100% certain. V4 addresses both, sampling colors continuously from the fitted
Gaussians and labeling by the posterior. A boundary color now carries a split
target rather than a false one, and ambiguous draws rose from 1.8% to 9.8% of
the dataset. Whether that fully resolves the failure on real photographs is
untested.

**Color constancy is learned rather than preprocessed.** Applying white balance
at inference gives no improvement, which is positive evidence the augmentation
pipeline pushed the correction into the weights, but it also means the model has
only ever seen the illuminant range the generator simulated.

## Licensing and provenance

Project code is MIT. Two things ride along and are worth stating plainly:

- The backbones initialize from ImageNet-pretrained torchvision weights, whose
  terms permit non-commercial research use. Anyone reusing this model
  commercially should confirm the current ImageNet terms cover their situation.
- The training backgrounds derive from MIT Indoor Scene Recognition (Quattoni
  and Torralba, CVPR 2009). No background images are redistributed here.

The color taxonomy derives from Paul Centore's sRGB centroids for the ISCC-NBS
Colour System, which originates in NBS Special Publication 440, a US Government
publication in the public domain.

## Citation

```bibtex
@software{sanchez2026clothingcolor,
  author = {Sanchez, Gio},
  title  = {Clothing Color Recognition with a Synthetic Dataset},
  year   = {2026},
  url    = {https://github.com/giosanchez0208/Clothing-Color-Recognition-ML-With-Synthetic-Dataset}
}
```

## References

- Geng, X. (2016). Label Distribution Learning. *IEEE TKDE* 28(7), 1734-1748.
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- Müller, R., Kornblith, S., & Hinton, G. (2019). When Does Label Smoothing Help? *NeurIPS*. [arXiv:1906.02629](https://arxiv.org/abs/1906.02629)
- Quattoni, A. & Torralba, A. (2009). Recognizing Indoor Scenes. *CVPR*.
