/* Browser inference for the clothing color model.
 *
 * Two models run here, both client-side:
 *
 *   pose_landmarker_lite.task   MediaPipe, Apache-2.0, finds a torso
 *   model.onnx                  the color model, 5.85 MB
 *
 * The pose step is not decoration. compose_input() sizes the inner crop from
 * the SHORT side of the torso box and the outer context from the LONG side,
 * which means the inner patch's magnification is exactly the box's aspect
 * ratio. A square box gives 1.00x, i.e. no composite at all, just a crop. The
 * Python pipeline builds its box from shoulder and hip keypoints, so it is
 * never square; a fixed square guide box silently collapses the whole design.
 *
 * The other delicate part is compose(). The model was trained on a composite,
 * not on photographs, and canvas is not allowed to do the downscale. See
 * areaResample() for why.
 *
 * Ultralytics is AGPL and stays out of this. MediaPipe is Apache-2.0, and its
 * model file is served from this Space rather than from Google, so no frame and
 * no request leaves the visitor's machine.
 */
import { FilesetResolver, PoseLandmarker }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const CLASSES = ["red", "orange", "yellow", "green", "blue", "violet", "purple",
                 "white", "gray", "black", "pink", "brown", "olive"];

// Same palette as scripts/predict.py, converted from its BGR tuples.
const SWATCH = {
  red: "#ff0000", orange: "#ffa500", yellow: "#ffd400", green: "#00b400",
  blue: "#0000ff", violet: "#8a2be2", purple: "#800080", white: "#e8e8e8",
  gray: "#969696", black: "#282828", pink: "#ff69b4", brown: "#8b4513",
  olive: "#808000",
};

const SIZE = 224, INNER = 112, OFF = (SIZE - INNER) / 2;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// report() thresholds, matching scripts/predict.py.
const SIG_RATIO = 0.35, SIG_FLOOR = 0.08, AMBIGUOUS = 0.25;

// MediaPipe BlazePose indices. The Python pipeline uses COCO 5/6/11/12 for the
// same four joints; the landmark set differs, the intent does not.
const L_SHOULDER = 11, R_SHOULDER = 12, L_HIP = 23, R_HIP = 24;

// `visibility` answers "is this joint occluded, GIVEN that a person was found".
// It is not a detector score and cannot reject a false positive: fed a
// matplotlib figure, this model returned torso visibilities of 0.92 to 0.99,
// indistinguishable from a real person at 0.998. So visibility is used only for
// its actual purpose, dropping joints that are out of frame.
const MIN_VIS = 0.4;

// Rejecting non-people is the detector's job, and it only takes the threshold
// at construction. Swept on two photographs and one plot: at the 0.5 default
// the plot is detected as a person, 0.7 through 0.9 keeps both people and
// rejects the plot, and 0.95 starts losing real detections. 0.8 sits in the
// middle of the usable window.
const MIN_DETECTION = 0.8;

const ORT_VERSION = "1.27.0";
const MP_VERSION = "1.0.1";
const PLANE = SIZE * SIZE;

const $ = (id) => document.getElementById(id);
const el = {
  status: $("status"), tabCamera: $("tab-camera"), tabImage: $("tab-image"),
  file: $("file"), video: $("video"), photo: $("photo"), overlay: $("overlay"),
  composite: $("composite"), bars: $("bars"), verdict: $("verdict"),
  latency: $("m-latency"), backend: $("m-backend"), empty: $("empty"),
  compEmpty: $("comp-empty"), debug: $("debug"),
};

let session = null;
let pose = null;
let mode = "idle";                         // "idle" | "image" | "camera"
let stream = null;
let running = false;
const inputBuffer = new Float32Array(3 * PLANE);

const cctx = el.composite.getContext("2d", { willReadFrequently: true });

/* ── preprocessing ─────────────────────────────────────────────────────── */

// Port of _clamp_square. Returns [x, y, side] rather than a box, since every
// caller wants the side length anyway.
function clampSquare(cx, cy, half, w, h) {
  let x1 = Math.max(0, cx - half);
  let y1 = Math.max(0, cy - half);
  const x2 = Math.min(w, cx + half);
  const y2 = Math.min(h, cy + half);
  const side = Math.min(x2 - x1, y2 - y1);
  x1 = Math.max(0, Math.min(x1, w - side));
  y1 = Math.max(0, Math.min(y1, h - side));
  return [Math.trunc(x1), Math.trunc(y1), Math.trunc(side)];
}

// Canvas is not allowed to do the downscale.
//
// drawImage resamples on its own sub-pixel grid, and at the ~4x reduction this
// pipeline needs, that grid is offset from the box average cv2.INTER_AREA
// computes. Composite means still agree, because a phase shift preserves a
// mean, so the error hides from every summary statistic. Measured against
// Python it moved the white logit by 0.19 and gray by 0.32, which is 58.4% ->
// 51.5% on white, on exactly the pair this model is weakest at. With the area
// average below, all 13 classes match Python to within 0.1 percentage point.
const srcPad = document.createElement("canvas");
const sctx = srcPad.getContext("2d", { willReadFrequently: true });
const compBuf = new Uint8ClampedArray(SIZE * SIZE * 4);
const compImage = new ImageData(compBuf, SIZE, SIZE);

function areaResample(src, sw, sh, sx0, sy0, sSide, dSide, dx0, dy0) {
  const scale = sSide / dSide;
  for (let dy = 0; dy < dSide; dy++) {
    const fy1 = sy0 + dy * scale, fy2 = fy1 + scale;
    const y1 = Math.max(0, Math.floor(fy1)), y2 = Math.min(sh, Math.ceil(fy2));
    for (let dx = 0; dx < dSide; dx++) {
      const fx1 = sx0 + dx * scale, fx2 = fx1 + scale;
      const x1 = Math.max(0, Math.floor(fx1)), x2 = Math.min(sw, Math.ceil(fx2));
      let r = 0, g = 0, b = 0, wsum = 0;
      for (let y = y1; y < y2; y++) {
        const wy = Math.min(y + 1, fy2) - Math.max(y, fy1);
        if (wy <= 0) continue;
        const row = y * sw;
        for (let x = x1; x < x2; x++) {
          const wx = Math.min(x + 1, fx2) - Math.max(x, fx1);
          if (wx <= 0) continue;
          const w = wx * wy, p = (row + x) * 4;
          r += src[p] * w; g += src[p + 1] * w; b += src[p + 2] * w; wsum += w;
        }
      }
      const q = ((dy0 + dy) * SIZE + (dx0 + dx)) * 4;
      compBuf[q] = Math.round(r / wsum);
      compBuf[q + 1] = Math.round(g / wsum);
      compBuf[q + 2] = Math.round(b / wsum);
      compBuf[q + 3] = 255;
    }
  }
}

// One snapshot of the source into srcPad, returning its pixels.
//
// Everything downstream reads this, never the live element. Two reasons, both
// measured. First, MediaPipe is unreliable reading an HTMLVideoElement here:
// fed the video directly it detected a pose in 3 or 4 frames out of 12, while
// the same frames drawn to a canvas detect essentially always. Second, pose ran
// on the video, then compose() read the video again after an await, so the
// landmarks and the cropped pixels could come from different frames, which on a
// moving subject puts the box beside the person rather than on them.
function snapshot(src, w, h) {
  if (srcPad.width !== w || srcPad.height !== h) { srcPad.width = w; srcPad.height = h; }
  sctx.drawImage(src, 0, 0, w, h);
  return sctx.getImageData(0, 0, w, h).data;
}

// Port of compose_input. `pixels` comes from snapshot(); `box` is
// [x1, y1, x2, y2] in source pixels.
function compose(pixels, w, h, box) {
  const [x1, y1, x2, y2] = box;
  const cx = Math.floor((x1 + x2) / 2);
  const cy = Math.floor((y1 + y2) / 2);
  const bw = x2 - x1, bh = y2 - y1;

  const innerSide = Math.max(Math.min(bw, bh), 10);
  const [ix, iy, iSide] = clampSquare(cx, cy, Math.floor(innerSide / 2), w, h);

  const outerSide = Math.max(Math.trunc(Math.max(bw, bh) * 2.0), innerSide + 20);
  const [ox, oy, oSide] = clampSquare(cx, cy, Math.floor(outerSide / 2), w, h);

  areaResample(pixels, w, h, ox, oy, oSide, SIZE, 0, 0);
  areaResample(pixels, w, h, ix, iy, iSide, INNER, OFF, OFF);

  cctx.putImageData(compImage, 0, 0);
  return {
    inner: [ix, iy, iSide], outer: [ox, oy, oSide],
    magnification: (INNER / iSide) / (SIZE / oSide),
  };
}

function toTensor() {
  const d = compBuf;
  const f = inputBuffer;
  for (let i = 0, p = 0; i < PLANE; i++, p += 4) {
    f[i] = (d[p] / 255 - MEAN[0]) / STD[0];
    f[PLANE + i] = (d[p + 1] / 255 - MEAN[1]) / STD[1];
    f[2 * PLANE + i] = (d[p + 2] / 255 - MEAN[2]) / STD[2];
  }
  return new ort.Tensor("float32", f, [1, 3, SIZE, SIZE]);
}

function softmax(logits) {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const out = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - max); sum += out[i]; }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

function report(probs) {
  const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]);
  const top = probs[order[0]];
  const thresh = Math.max(SIG_FLOOR, top * SIG_RATIO);
  const picks = order.filter((i) => probs[i] >= thresh)
                     .map((i) => [CLASSES[i], probs[i]]);
  return { picks, ambiguous: top < AMBIGUOUS, order };
}

/* ── pose ──────────────────────────────────────────────────────────────── */

// Torso box from shoulders and hips.
//
// The obvious construction, and the one the Python Pose class uses, is the
// bounding box of the four joints. That is correct only when the subject faces
// the camera square on. Turn them, and the shoulders and hips form a diagonal,
// so the bounding box balloons sideways to span it: measured on one photograph
// it came out 2.4x wider than the shoulders with its center pulled 47 px off
// them, and only 42% of the resulting crop was actually chest. On a half-body
// framing it was 4.8x wider and 21% chest. The model was mostly classifying
// background while looking like it was tracking the person.
//
// So build the box from the torso midline instead: width from the shoulder
// span, height from the shoulder line down to the hip line, centered between the
// two midpoints. For a front-facing subject this is arithmetically identical to
// the bounding box, so nothing changes in the case the Python pipeline was
// tuned on; it only diverges where the bounding box was wrong.
function torsoFrom(landmarks, w, h) {
  if (!landmarks || !landmarks.length) return null;
  const lm = landmarks[0];

  // MediaPipe extrapolates joints past the frame edge rather than dropping
  // them, and reports high visibility while doing it, so clamp into the image.
  const at = (i) => {
    const p = lm[i];
    if (!p) return null;
    if (p.visibility !== undefined && p.visibility < MIN_VIS) return null;
    return [Math.min(Math.max(p.x, 0), 1) * w, Math.min(Math.max(p.y, 0), 1) * h];
  };

  const ls = at(L_SHOULDER), rs = at(R_SHOULDER);
  if (!ls || !rs) return null;              // shoulders set the garment width
  const shMid = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2];
  const shW = Math.abs(ls[0] - rs[0]);
  if (shW < 12) return null;                // too small, or too side-on to read

  // Take the hips' vertical position but not their horizontal one.
  //
  // Checked against the YOLO keypoints the Python pipeline uses, on the same
  // photograph: MediaPipe's shoulders agree to 18-31 px, its hips are off by
  // 127-191 px, and that holds for the lite, full and heavy models alike, so a
  // bigger download does not fix it. Most of that error is lateral, and lateral
  // hip error is the damaging kind: a bounding box over all four joints stretches
  // sideways to span the shoulder-to-hip diagonal, which measured 2.4x wider than
  // the shoulders on one photo and 4.8x on another, leaving 42% and 21% of the
  // crop actually on the garment. Vertical hip position survives well enough to
  // set the torso's length.
  //
  // Measured on the two reference photographs, IoU against the YOLO torso box:
  //   four-joint bounding box   0.13 turned / 0.86 front-on
  //   shoulders only            0.13 turned / 0.86 front-on
  //   this split                0.34 turned / 0.85 front-on
  const lh = at(L_HIP), rh = at(R_HIP);
  let hipY = null;
  if (lh && rh) hipY = (lh[1] + rh[1]) / 2;
  else if (lh || rh) hipY = (lh || rh)[1];
  if (hipY !== null && hipY - shMid[1] < 0.4 * shW) hipY = null;   // implausible
  if (hipY === null) hipY = shMid[1] + 1.5 * shW;                  // hips gone

  const bw = shW;
  // The box's aspect ratio IS the inner patch's magnification, because the
  // inner crop is sized from the short side and the outer from the long one.
  // A turned subject foreshortens the shoulder span without shortening the
  // torso, so the ratio runs away: one half-body framing produced a 39 px span
  // against a 190 px drop, magnifying a 39 px sliver 5x. Floor it so a collapsed
  // height cannot invert the composite, and cap it so a narrow one cannot
  // explode it.
  const bh = Math.min(Math.max(hipY - shMid[1], 0.8 * bw), 2.5 * bw);

  const box = [Math.round(shMid[0] - bw / 2), Math.round(shMid[1]),
               Math.round(shMid[0] + bw / 2), Math.round(shMid[1] + bh)];
  if (box[2] - box[0] <= 10 || box[3] - box[1] <= 10) return null;
  return box;
}

/* ── rendering ─────────────────────────────────────────────────────────── */

let barRows = null;

function buildBars() {
  el.bars.innerHTML = "";
  barRows = CLASSES.map((name) => {
    const nameEl = document.createElement("div");
    nameEl.className = "name";
    nameEl.textContent = name;

    const track = document.createElement("div");
    track.className = "track";
    const fill = document.createElement("div");
    fill.className = "fill";
    fill.style.background = SWATCH[name];
    fill.style.width = "0%";
    track.appendChild(fill);

    const pct = document.createElement("div");
    pct.className = "pct";
    pct.textContent = "0.0%";

    el.bars.append(nameEl, track, pct);
    return { name: nameEl, fill, pct };
  });
}

function render(probs) {
  const { picks, ambiguous, order } = report(probs);
  const hits = new Set(picks.map(([c]) => c));

  for (const idx of order) {
    const name = CLASSES[idx];
    const row = barRows[idx];
    el.bars.append(row.name, row.fill.parentElement, row.pct);
    row.fill.style.width = (probs[idx] * 100).toFixed(1) + "%";
    row.pct.textContent = (probs[idx] * 100).toFixed(1) + "%";
    row.name.classList.toggle("hit", hits.has(name));
    row.pct.classList.toggle("hit", hits.has(name));
  }

  const text = picks.map(([c, p]) => `${c} ${(p * 100).toFixed(0)}%`).join(", ");
  el.verdict.innerHTML = ambiguous
    ? `<span class="amb">uncertain</span>, ${text}`
    : text;
}

// No torso: the model has not been shown anything, so show nothing.
function clearPrediction(message) {
  cctx.clearRect(0, 0, SIZE, SIZE);
  el.compEmpty.hidden = false;
  el.composite.classList.add("blank");
  for (const row of barRows) {
    row.fill.style.width = "0%";
    row.pct.textContent = "0.0%";
    row.name.classList.remove("hit");
    row.pct.classList.remove("hit");
  }
  el.verdict.textContent = "";
  el.latency.textContent = "not run";
  if (message) setStatus(message);
}

// Draws the torso box. The wider context rect and the landmarks it was built
// from are diagnostics rather than part of the demo, so they appear only under
// ?debug=1: if the dots sit on the subject and the box does not, the arithmetic
// here is wrong; if the dots are off the subject, the pose model is wrong and no
// box construction on top of it can recover.
function drawOverlay(rects, dispW, dispH, srcW, srcH, landmarks) {
  const c = el.overlay;
  if (c.width !== dispW || c.height !== dispH) { c.width = dispW; c.height = dispH; }
  const g = c.getContext("2d");
  g.clearRect(0, 0, dispW, dispH);
  const sx = dispW / srcW, sy = dispH / srcH;

  if (rects) {
    if (DEBUG) {
      const [ox, oy, oSide] = rects.outer;
      g.strokeStyle = "rgba(255,255,255,0.45)";
      g.lineWidth = 1;
      g.setLineDash([4, 4]);
      g.strokeRect(ox * sx, oy * sy, oSide * sx, oSide * sy);
    }

    const [ix, iy, iSide] = rects.inner;
    g.setLineDash([]);
    g.lineWidth = 2;
    g.strokeStyle = "rgba(0,0,0,0.85)";
    g.strokeRect(ix * sx - 1, iy * sy - 1, iSide * sx + 2, iSide * sy + 2);
    g.strokeStyle = "#ffffff";
    g.strokeRect(ix * sx, iy * sy, iSide * sx, iSide * sy);
  }

  if (!DEBUG || !landmarks || !landmarks.length) return;
  const lm = landmarks[0];
  const pt = (i) => [Math.min(Math.max(lm[i].x, 0), 1) * srcW * sx,
                     Math.min(Math.max(lm[i].y, 0), 1) * srcH * sy];

  g.setLineDash([]);
  for (const [pair, color] of [[[L_SHOULDER, R_SHOULDER], "#00e0ff"],
                               [[L_HIP, R_HIP], "#ffb000"]]) {
    const a = pt(pair[0]), b = pt(pair[1]);
    g.strokeStyle = color;
    g.lineWidth = 2;
    g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
  }
  for (const [i, color] of [[L_SHOULDER, "#00e0ff"], [R_SHOULDER, "#00e0ff"],
                            [L_HIP, "#ffb000"], [R_HIP, "#ffb000"]]) {
    const [x, y] = pt(i);
    g.beginPath(); g.arc(x, y, 5, 0, Math.PI * 2);
    g.fillStyle = color; g.fill();
    g.lineWidth = 1.5; g.strokeStyle = "#000"; g.stroke();
  }
}

// Numeric dump for when the dots alone do not settle it. Enable with ?debug=1.
const DEBUG = new URLSearchParams(location.search).has("debug");

function debugDump(srcW, srcH, dispW, dispH, landmarks, box, rects) {
  if (!DEBUG || !el.debug) return;
  const lm = landmarks && landmarks[0];
  const px = (i) => lm ? `(${Math.round(lm[i].x * srcW)}, ${Math.round(lm[i].y * srcH)}) vis=${lm[i].visibility.toFixed(2)}`
                       : "n/a";
  el.debug.hidden = false;
  el.debug.textContent = [
    `source      ${srcW} x ${srcH}      displayed ${dispW} x ${dispH}`,
    `scale       x ${(dispW / srcW).toFixed(4)}   y ${(dispH / srcH).toFixed(4)}`,
    `L_shoulder  ${px(L_SHOULDER)}`,
    `R_shoulder  ${px(R_SHOULDER)}`,
    `L_hip       ${px(L_HIP)}`,
    `R_hip       ${px(R_HIP)}`,
    `shoulder span ${lm ? Math.round(Math.abs(lm[L_SHOULDER].x - lm[R_SHOULDER].x) * srcW) : "n/a"} px`,
    `torso box   ${box ? JSON.stringify(box) : "null"}`,
    rects ? `inner crop  [${rects.inner}]   outer crop  [${rects.outer}]` : "no crop",
    rects ? `magnification ${rects.magnification.toFixed(3)}x` : "",
  ].join("\n");
}

/* ── inference ─────────────────────────────────────────────────────────── */

let smoothed = null;

async function infer(pixels, srcW, srcH, dispW, dispH, box, smooth, landmarks) {
  if (!session || !srcW || !srcH) return;
  const rects = compose(pixels, srcW, srcH, box);
  drawOverlay(rects, dispW, dispH, srcW, srcH, landmarks);
  debugDump(srcW, srcH, dispW, dispH, landmarks, box, rects);
  el.compEmpty.hidden = true;
  el.composite.classList.remove("blank");

  const t0 = performance.now();
  const out = await session.run({ image: toTensor() });
  const ms = performance.now() - t0;

  let probs = softmax(out.color_logits.data);
  if (smooth) {
    if (!smoothed || smoothed.length !== probs.length) smoothed = Float64Array.from(probs);
    else for (let i = 0; i < probs.length; i++) smoothed[i] += 0.35 * (probs[i] - smoothed[i]);
    probs = smoothed;
  } else {
    smoothed = null;
  }

  render(probs);
  el.latency.textContent = ms.toFixed(1) + " ms";
  return rects;
}

let missStreak = 0;

async function loop() {
  if (!running || mode !== "camera") return;
  const v = el.video;
  if (v.readyState >= 2) {
    const w = v.videoWidth, h = v.videoHeight;
    // Freeze the frame first, then run both models against that one snapshot.
    const pixels = snapshot(v, w, h);
    const res = pose.detect(srcPad);
    const box = torsoFrom(res.landmarks, w, h);
    if (box) {
      missStreak = 0;
      const r = await infer(pixels, w, h, v.clientWidth, v.clientHeight, box, true,
                            res.landmarks);
      setStatus(`torso found, ${r.magnification.toFixed(2)}x inner magnification`);
    } else {
      // Blank the prediction but keep drawing whatever landmarks came back, so
      // a bad pose is visible rather than silently discarded.
      drawOverlay(null, v.clientWidth, v.clientHeight, w, h, res.landmarks);
      debugDump(w, h, v.clientWidth, v.clientHeight, res.landmarks, null, null);
      if (++missStreak === 1) clearPrediction("no person detected");
    }
  }
  requestAnimationFrame(loop);
}

/* ── modes ─────────────────────────────────────────────────────────────── */

function setStatus(msg, isError) {
  el.status.textContent = msg;
  el.status.classList.toggle("err", !!isError);
}

function stopCamera() {
  running = false;
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  el.video.hidden = true;
  el.video.srcObject = null;
}

async function startCamera() {
  mode = "camera";
  el.tabCamera.setAttribute("aria-pressed", "true");
  el.tabImage.setAttribute("aria-pressed", "false");
  el.photo.hidden = true;

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("this browser exposes no camera API", true);
    return;
  }
  try {
    setStatus("requesting camera…");
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 960 } }, audio: false,
    });
  } catch (e) {
    setStatus("camera unavailable: " + e.name + ". the image tab still works.", true);
    return;
  }
  el.video.srcObject = stream;
  el.video.hidden = false;
  el.empty.hidden = true;
  await el.video.play();
  clearPrediction("looking for a torso…");
  running = true;
  missStreak = 0;
  loop();
}

function showImage(url) {
  mode = "image";
  stopCamera();
  el.tabCamera.setAttribute("aria-pressed", "false");
  el.tabImage.setAttribute("aria-pressed", "true");
  el.photo.hidden = false;
  el.empty.hidden = true;
  el.photo.onload = async () => {
    const w = el.photo.naturalWidth, h = el.photo.naturalHeight;
    const pixels = snapshot(el.photo, w, h);
    const res = pose.detect(srcPad);
    const box = torsoFrom(res.landmarks, w, h);
    if (!box) {
      drawOverlay(null, el.photo.clientWidth, el.photo.clientHeight, w, h, res.landmarks);
      debugDump(w, h, el.photo.clientWidth, el.photo.clientHeight, res.landmarks, null, null);
      clearPrediction("no person detected in this image");
      return;
    }
    const r = await infer(pixels, w, h, el.photo.clientWidth, el.photo.clientHeight,
                          box, false, res.landmarks);
    setStatus(`torso found, ${r.magnification.toFixed(2)}x inner magnification`);
  };
  el.photo.src = url;
}

/* ── boot ──────────────────────────────────────────────────────────────── */

async function boot() {
  buildBars();
  clearPrediction();

  ort.env.wasm.wasmPaths =
    `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

  // Multithreaded WASM needs cross-origin isolation, which a static host does
  // not set. Asking for threads without it makes ORT spawn workers that fail,
  // so ask only when the browser says isolation is actually on.
  const threads = self.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort.env.wasm.numThreads = threads;

  try {
    setStatus("loading color model (5.85 MB)…");
    session = await ort.InferenceSession.create("model.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  } catch (e) {
    setStatus("could not load the color model: " + e.message, true);
    return;
  }

  try {
    setStatus("loading pose model (5.5 MB)…");
    const fileset = await FilesetResolver.forVisionTasks(
      `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`);
    // IMAGE mode throughout, including the camera loop. VIDEO mode exists to
    // reuse tracking between frames, but measured on a real clip it detected a
    // pose in 3 or 4 frames out of 12 where a per-frame IMAGE detect on the
    // same content succeeded almost every time. Full detection every frame
    // costs more and is worth it.
    pose = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "pose_landmarker_lite.task" },
      runningMode: "IMAGE",
      numPoses: 1,
      minPoseDetectionConfidence: MIN_DETECTION,
      minPosePresenceConfidence: MIN_DETECTION,
    });
  } catch (e) {
    setStatus("could not load the pose model: " + e.message, true);
    return;
  }

  el.backend.textContent = `wasm, ${threads} thread${threads === 1 ? "" : "s"}`;
  setStatus("both models ready. start the camera, or upload an image");

  el.tabCamera.onclick = startCamera;
  el.tabImage.onclick = () => el.file.click();
  el.file.onchange = () => {
    const f = el.file.files && el.file.files[0];
    if (f) showImage(URL.createObjectURL(f));
  };

  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) showImage(URL.createObjectURL(f));
  });

  // Exposed so the pipeline can be exercised from the console against the
  // Python reference, which is how the resampler bug was caught.
  window.__demo = { compose, infer, showImage, toTensor, softmax, report,
                    torsoFrom, snapshot, srcPad,
                    get session() { return session; },
                    get pose() { return pose; }, compBuf };
}

boot();
