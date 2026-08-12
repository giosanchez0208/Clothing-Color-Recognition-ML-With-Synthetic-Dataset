# Third-Party Notices

This project's own source code is MIT licensed (see [LICENSE](LICENSE)). It
depends on, and derives data from, third-party works with their own terms.

---

## The AGPL boundary — read this first

**`ultralytics` is licensed AGPL-3.0**, and AGPL-3.0 is a strong copyleft
licence whose defining feature is that *network use counts as distribution*:
if users interact with the software over a network, they are entitled to the
source of the combined work.

**Where it is used:** only the optional inference demo.

| Component | Uses `ultralytics`? |
|---|---|
| Colour taxonomy (`01`, `02`) | No |
| Background preparation (`03`) | No |
| Synthetic dataset generation (`04`, `08`) | No |
| Colour library resampling (`07`) | No |
| Training (`05`, `06`, `09`) | No |
| Distillation + quantization (`10`) | No |
| **Live webcam / video inference (`11`)** | **Yes** — YOLOv11n-pose for torso detection |

The trained model and the entire pipeline that produces it are free of any
copyleft dependency. `ultralytics` is used solely to locate a person's torso in
a camera frame before the colour model runs; it is not part of the model, the
data, or the training process.

**Practical consequences**

- *This repository.* `ultralytics` is a pip dependency users install
  themselves. No AGPL code is vendored or redistributed here, so MIT licensing
  of this project's own source is appropriate and conventional.
- *A hosted demo (e.g. HuggingFace Spaces).* If a deployed, network-accessible
  demo imports `ultralytics`, the AGPL network clause is engaged for that
  deployment. Two ways to stay clean:
  1. Publish the demo's full source (this repository is already public, which
     substantially satisfies the obligation), **or**
  2. Build the demo without pose detection — accept an uploaded image and let
     the user select the garment region. The colour model itself needs no YOLO,
     so this path is entirely AGPL-free.

---

## Software dependencies

| Package | Licence | Role |
|---|---|---|
| `ultralytics` | **AGPL-3.0** | YOLOv11n-pose; optional inference demo only |
| `torch`, `torchvision` | BSD-style (BSD-3-Clause) | training, model definitions, quantization |
| `opencv-python` | Apache-2.0 | image processing throughout |
| `numpy`, `scipy`, `pandas` | BSD-3-Clause | numerics, Perlin noise, tabular I/O |
| `matplotlib`, `seaborn` | matplotlib licence (BSD-style) / BSD-3-Clause | figures and analysis |
| `pillow` | MIT-CMU | image I/O in the training transforms |
| `onnx` | Apache-2.0 | cross-platform model export |
| `huggingface-hub` | Apache-2.0 | fetching background images |

Exact versions are pinned in [requirements.txt](requirements.txt). To audit the
full transitive set in your own environment:

```bash
pip install pip-licenses && pip-licenses --format=markdown
```

---

## Pretrained weights

Both backbones initialise from ImageNet-pretrained torchvision weights:

- **ResNet-50** (`ResNet50_Weights.IMAGENET1K_V2`) — teacher
- **MobileNetV3-Small** (`MobileNet_V3_Small_Weights.IMAGENET1K_V1`) — student

The torchvision code is BSD-licensed. The weights derive from ImageNet
(ILSVRC), whose terms permit non-commercial research use. Anyone reusing the
distilled model commercially should confirm the current ImageNet terms apply to
their situation.

---

## Data sources

### ISCC-NBS colour centroids

`datasets/isccnbs_color_categories.csv` is derived from **"sRGB Centroids for
the ISCC-NBS Colour System"** by Paul Centore (2016), and its companion data
file `sRGBcentroidsForISCCNBS.txt`.

- Paper: <https://munsellcolorscienceforpainters.com/ColourSciencePapers/sRGBCentroidsForTheISCCNBSColourSystem.pdf>
- Data: <https://www.munsellcolorscienceforpainters.com/ISCCNBS/ISCCNBSSystem.html>

No explicit licence is stated on the source pages. The content is colorimetric
reference data for the ISCC-NBS system, which originates in **NBS Special
Publication 440**, *Color: Universal Language and Dictionary of Names* (Kelly &
Judd) — a US Government publication in the public domain. The file included
here is a reformatting of the published Level-3 sRGB centroid values into CSV,
with attribution. If you are the rights holder and would prefer it removed,
please open an issue.

`categorized_colors.csv` and `categorized_colors_normalized.csv` are derived
works computed by this project's own code (`notebooks/01_taxonomy.ipynb` and
`notebooks/07_color_library_v2.ipynb`).

### Indoor background images

Backgrounds come from the **MIT Indoor Scene Recognition (CVPR 2009)** dataset —
67 categories, 15,620 images.

> A. Quattoni and A. Torralba. *Recognizing Indoor Scenes.* CVPR 2009.
> <https://web.mit.edu/torralba/www/indoor.html>

**The original download is no longer available.** As of August 2026,
`indoorCVPR_09.tar` at `groups.csail.mit.edu` returns HTTP 404, though the
project page still advertises it. This project therefore fetches the dataset
from a HuggingFace mirror that preserves the original `Images/<category>/`
layout:

- <https://huggingface.co/datasets/u5753411/MIT-Indoor-Scenes> (declared MIT licence)

Background images are **not redistributed** in this repository. They are
downloaded by `scripts/prepare_backgrounds.py` and remain gitignored. Only
locally generated composites derive from them.

---

## Generated dataset

Images under `datasets/generated_v3/` are produced entirely by this project's
code from the colour library and the background images above. They are
gitignored here; if published separately (e.g. as a HuggingFace dataset), that
publication inherits the background dataset's terms for the background regions.
