---
title: Clothing Color Recognition
emoji: ▪
colorFrom: gray
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: 5.85 MB color model trained on synthetic data, in-browser
---

# Clothing color recognition

A MobileNetV3-Small that predicts a **distribution over 13 color categories**
rather than committing to one label. 5.85 MB, distilled from a ResNet-50,
trained on 28,860 procedurally generated images and zero real photographs.

The model runs in your browser through ONNX Runtime Web. Nothing you point the
camera at, and nothing you upload, leaves your machine.

- **Model card:** <https://huggingface.co/giosanchez0208/clothing-color-recognition>
- **Source and write-up:** <https://github.com/giosanchez0208/Clothing-Color-Recognition-ML-With-Synthetic-Dataset>

## What is actually running

Two models, both client-side.

| | |
|---|---|
| Color model | MobileNetV3-Small, 13-way head, `model.onnx`, opset 17, 5.85 MB |
| Torso detection | MediaPipe Pose Landmarker lite, Apache-2.0, 5.5 MB |
| Runtime | onnxruntime-web 1.27.0 and tasks-vision 1.0.1, both WASM |
| Input | 224x224 composite, ImageNet normalization |
| Held-out test top-1 | 67.4% |

## Why MediaPipe and not the pipeline's own detector

The Python pipeline uses YOLOv11n-pose. That model is AGPL-3.0, and AGPL treats
network use as distribution, so a hosted demo importing it would pull the whole
deployment into copyleft. MediaPipe Pose is Apache-2.0 and does the same job
here, so the demo keeps torso detection without the copyleft. The boxes differ
slightly between the two detectors, which is expected and is the only place this
demo departs from the Python reference. Given the same box, the composite and
the prediction match Python exactly, verified to 0.00 percentage points across
the top 5 classes.

Detection runs in IMAGE mode on every frame rather than VIDEO mode. VIDEO mode
reuses tracking between frames, which sounds right for a camera and measured
far worse here: see below.

The AGPL reasoning is worked through in `NOTICE.md` in the source repository.

## A square guide box would have broken the composite silently

`compose_input` sizes the inner crop from the **short** side of the torso box
and the outer context from the **long** side, so the inner patch's magnification
relative to the ring around it is exactly the box's aspect ratio.

An earlier version of this demo had no detector and used a fixed square guide
box. A square gives 1.00x. The inner patch was drawn at scale 0.2286 and the
outer ring at 0.2286, identical, so the composite collapsed into an ordinary
crop and the panel showing "what the model sees" taught nothing. It still looked
plausible, which is what made it worth writing down. Real torso boxes run 1.5x
to 1.8x.

## The preprocessing is not a resize

The model never sees a photograph. It sees a 112x112 crop of the torso pasted
into the middle of a 224x224 wider-context crop, which is the layout every
training image was built in. `app.js` reproduces that, including a hand-written
port of `cv2.INTER_AREA`.

That last part is not incidental. Canvas `drawImage` resamples on its own
sub-pixel grid, and at the reduction this pipeline needs, that grid is offset
from the box average OpenCV computes. Composite means still agree, because a
phase shift preserves a mean, but the fine detail does not. Letting canvas do
the downscale moved the white logit by 0.19 and the gray logit by 0.32, turning
58.4% white into 51.5% white, on precisely the category pair this model is
weakest at.

## Never hand the pose model a video element

This is the bug that took the longest to find, and it was worth the trouble
because it was invisible from the outside: the box tracked the person, moved
with them, and sat beside their torso instead of on it.

Two faults, both in how the frame reached the models.

**MediaPipe could not reliably read an `HTMLVideoElement`.** Fed the live video
it found a pose in 3 or 4 frames out of 12. The same frames drawn to a canvas
first detect essentially every time. Detection was therefore intermittent, and
whatever survived was what the demo displayed.

**The two models were reading different frames.** Pose ran against the video
element, then the crop read the video again after an `await` for inference. On a
moving subject those are different moments, so the box was computed from where
the person had been and applied to where they now were.

Both are fixed by taking one canvas snapshot per frame and running pose and the
crop against that single buffer. Scored against `scripts/predict.py` on a real
clip, sampling the demo's own loop:

| | before | after |
|---|---|---|
| frames with a detection | 3-4 of 12 | 7 of 10 |
| median IoU vs the YOLO torso box | not measurable | **0.863** |

On the frames where playback timing lines up cleanly, IoU reaches 0.94 and the
top-1 color matches the Python pipeline. Where the two still disagree, the top
two classes are within a couple of points of each other, so the ordering is a
coin flip rather than a real difference.

A correction worth recording: an earlier version of this file claimed MediaPipe's
hips were off by 127 to 191 px and that the demo was only good facing the
camera. That came from two unusual photographs. Measured on real video the same
landmarks land within 3 to 15 px on shoulders and 13 to 53 px on hips, and the
shoulder spans agree to within a few pixels. The landmarks were never the
problem.

## The line endings of index.html are load-bearing

A Space injects a `window.huggingface = {...}` script into `index.html` when it
serves it. The injector computes where to put it as though the file used CRLF
line endings. Given an LF file it lands early by exactly the number of newlines
before `<head>`, so with two of them the tag comes back as

```
<hea<script>window.huggingface={...};</script>d>
```

The browser cannot make a head out of that, so it abandons it and renders the
script source as text at the top of the page, followed by a stray `d>`.

`scripts/publish_hf.py` converts HTML to CRLF as it uploads, which keeps the fix
out of the working tree where an editor writing LF would undo it.

## One more thing that does not mean what it looks like

MediaPipe reports a `visibility` score per landmark, which reads like a
detection confidence and is not one. It answers "is this joint occluded, given
that a person was found". Fed a matplotlib figure, this model returned torso
visibilities of 0.92 to 0.99, indistinguishable from a real person at 0.998, and
happily produced a torso box for a bar chart.

Rejecting non-people is the detector's own threshold, set at construction.
Swept on two photographs and one plot: at the 0.5 default the plot is detected
as a person, 0.7 through 0.9 keeps both people and rejects the plot, and 0.95
starts losing real detections. This demo uses 0.8.

## Known limits

Trained on synthetic data only, so it has never seen a real garment. It is
weakest where two categories meet: pale green against yellow, gray against
white. The write-up traces the green case to centroid-weighted sampling leaving
the category boundary under-represented, rather than to a gap in the data.
