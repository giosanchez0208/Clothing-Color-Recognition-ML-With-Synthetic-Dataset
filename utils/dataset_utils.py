import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.stats import truncnorm
import cv2
import os
import random
from .color_utils import ColorLibrary

DEFAULT_DIMENSIONS = (224, 224)
PATTERN_TYPES = ['solid', 'stripes', 'color_blocking', 'polka_dot']
PATTERN_WEIGHTS = [0.5, 0.25, 0.15, 0.1]
    
class InnerSquareGenerator:
    def __init__(self, dimensions=DEFAULT_DIMENSIONS):
        self.dimensions = dimensions
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'categorized_colors.csv')
        self.color_library = ColorLibrary.from_categorized_csv(csv_path)
        
    def generate_random_color(self):
        categories = self.color_library.categories

        # Uniform category choice
        category = random.choice(categories)
        color_distances = self.color_library.get_category_mahalanobis_distances(category)
        if not color_distances:
            raise ValueError(f'No colors found for category {category}.')

        # Larger chance near centroid: use inverse Mahalanobis distance.
        eps = 1e-6
        hex_colors = [hx for hx, _ in color_distances]
        weights = [(1.0 / (dist + eps))**2 for _, dist in color_distances]
        selected_hex = random.choices(hex_colors, weights=weights, k=1)[0]
        return category, selected_hex

    def compose_random_color(self):
        category, color_hex = self.generate_random_color()
        labels = [category]
        square_height = int(self.dimensions[0] / 2)
        square_width = int(self.dimensions[1] / 2)
        color_bgr = self._hex_to_bgr(color_hex)
        inner_square = np.full((square_height, square_width, 3), color_bgr, dtype=np.uint8)
        label_map = np.zeros((square_height, square_width), dtype=np.uint16)
        inner_square, labels, label_map = self.add_pattern(inner_square, labels, label_map, [color_hex])
        label_percentages = _compute_label_percentages(labels, label_map)
        return inner_square, label_percentages
    
    def add_pattern(self, inner_square, labels, label_map, used_hexes):
        pattern_type = random.choices(PATTERN_TYPES, weights=PATTERN_WEIGHTS, k=1)[0]

        if pattern_type == 'solid':
            patterned = _apply_solid_gradient_pattern(inner_square, max_opacity=0.2)
            return patterned, labels, label_map

        if pattern_type == 'polka_dot':
            extra_count = random.randint(2, 5)
        elif pattern_type == 'color_blocking':
            extra_count = random.randint(1, 2)
        else:
            extra_count = random.randint(1, 3)
        extra_pairs = _sample_distinct_random_colors(self.generate_random_color, used_hexes, extra_count)
        extra_labels = [cat for cat, _ in extra_pairs]
        extra_hexes = [hx for _, hx in extra_pairs]
        extra_bgr = [self._hex_to_bgr(hx) for hx in extra_hexes]
        label_indices = list(range(len(labels), len(labels) + len(extra_labels)))
        labels_extended = labels + extra_labels

        if pattern_type == 'stripes':
            patterned, label_map = _apply_stripes_pattern(inner_square, label_map, extra_bgr, label_indices)
        elif pattern_type == 'color_blocking':
            patterned, label_map = _apply_color_blocking_pattern(inner_square, label_map, extra_bgr, label_indices)
        else:
            patterned, label_map = _apply_polka_dot_pattern(inner_square, label_map, extra_bgr, label_indices)

        return patterned, labels_extended, label_map

    def apply_synthetic_fold_texture(self, inner_square):
        texture = generate_synthetic_clothing_folds()
        target_h, target_w = inner_square.shape[:2]

        # 1. Random Scaling (from 1x to 3x)
        scale = random.uniform(1.0, 3.0)
        new_h, new_w = int(target_h * scale), int(target_w * scale)
        texture_resized = cv2.resize(texture, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 2. Random Cropping
        # Calculate max possible start indices for the crop
        max_y = new_h - target_h
        max_x = new_w - target_w
        start_y = random.randint(0, max_y)
        start_x = random.randint(0, max_x)

        # Slice the resized texture to the original target size
        texture_cropped = texture_resized[start_y:start_y + target_h, start_x:start_x + target_w]

        # 3. Normalization and Blending (same as before)
        base = inner_square.astype(np.float32) / 255.0
        texture_norm = texture_cropped.astype(np.float32) / 255.0
        texture_3ch = np.repeat(texture_norm[:, :, None], 3, axis=2)

        multiplied = base * texture_3ch
        alpha = _sample_texture_opacity()
        blended = base * (1.0 - alpha) + multiplied * alpha
        
        return np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    def generate(self):
        inner_square, label_percentages = self.compose_random_color()
        textured_inner_square = self.apply_synthetic_fold_texture(inner_square) if random.random() < 0.85 else inner_square
        return textured_inner_square, label_percentages
    
    
    @staticmethod
    def _hex_to_bgr(color_hex):
        color_hex = color_hex.lstrip('#')
        return tuple(int(color_hex[i:i + 2], 16) for i in (4, 2, 0))

class OuterSquareGenerator:
    
    def __init__(self, dimensions=DEFAULT_DIMENSIONS, path_to_bgs=None):
        self.dimensions = dimensions
        self.path_to_bgs = path_to_bgs

    def generate(self):
        if self.path_to_bgs is None:
            raise ValueError("path_to_bgs is not set")
        
        # Get all image files from path_to_bgs
        image_files = [f for f in os.listdir(self.path_to_bgs) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not image_files:
            raise ValueError(f"No image files found in {self.path_to_bgs}")
        
        # Select a random image
        random_image = random.choice(image_files)
        image_path = os.path.join(self.path_to_bgs, random_image)
        
        # Load the image
        img = cv2.imread(image_path)
        
        # Crop to dimensions
        h, w = img.shape[:2]
        target_h, target_w = self.dimensions
        
        # Calculate crop coordinates (center crop)
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        bottom = top + target_h
        right = left + target_w
        
        # Ensure bounds are valid
        top = max(0, top)
        left = max(0, left)
        bottom = min(h, bottom)
        right = min(w, right)
        
        cropped_img = img[top:bottom, left:right]
        
        # Resize if needed to match exact dimensions
        if cropped_img.shape[:2] != (target_h, target_w):
            cropped_img = cv2.resize(cropped_img, (target_w, target_h))
        
        return cropped_img

class DatasetGenerator:
    
    def __init__(self, dimensions=DEFAULT_DIMENSIONS, path_to_bgs=None):
        self.outer_gen = OuterSquareGenerator(dimensions=dimensions, path_to_bgs=path_to_bgs)
        self.inner_gen = InnerSquareGenerator(dimensions=dimensions)
    
    def generate(self):
        background = self.outer_gen.generate()
        inner_square, label_percentages = self.inner_gen.generate()

        composed_image = background.copy()
        bg_height, bg_width = composed_image.shape[:2]
        square_height, square_width = inner_square.shape[:2]
        top = (bg_height - square_height) // 2
        left = (bg_width - square_width) // 2
        bottom = top + square_height
        right = left + square_width

        composed_image[top:bottom, left:right] = inner_square
        composed_image = _apply_global_lighting(composed_image)
        return composed_image, label_percentages


def generate_synthetic_clothing_folds(octaves = 5, persistence = 0.5):
    # Generate a 224x224 random Perlin noise image
    img = _fbm_perlin(shape=DEFAULT_DIMENSIONS, base_res=(7, 7), octaves=octaves, persistence=persistence)

    # FACTORS:
    blur_factor = 1.5
    brightness_factor = 2
    stretch_factor = 100
    contrast_factor = 2

    # Start from the existing img
    work = img.astype(np.float32)

    # (1) Gaussian blur
    work = gaussian_filter(work, sigma=blur_factor)

    # (2) Stretch horizontally by x%, then center-crop back to original width
    #     so final image keeps original aspect ratio (224x224)
    work = zoom(work, (1.0, 1.0 + stretch_factor / 100), order=1)
    h, w = img.shape
    x0 = (work.shape[1] - w) // 2
    work = work[:, x0:x0 + w]

    # (3) Increase brightness
    work = work * brightness_factor

    # (4) Bump contrast
    m = work.mean()
    work = (work - m) * contrast_factor + m

    img = np.clip(work, 0, 255).astype(np.uint8)
    return img

def color_balance_image(img):
    # 1. Prepare: Convert to float for math and Linearize
    img_float = img.astype(np.float32) / 255.0
    img_linear = _linearization_and_black_level_sub(img_float)

    # 2. White Balance: Estimate and correct illuminant (Shades of Gray)
    img_wb = _chromatic_adaptation_shades_of_gray(img_linear, p=6)

    # 3. Enhance: Local contrast and exposure correction
    # Note: Contrast is best handled in non-linear space for human viewing
    img_gamma = np.power(np.clip(img_wb, 0, 1), 1/2.2)
    img_final = _exposure_and_local_contrast(img_gamma)

    return (img_final * 255).astype(np.uint8)


def _sample_distinct_random_colors(color_sampler, existing_hexes, count, max_attempts=500):
    results = []
    used = set(existing_hexes)
    attempts = 0

    while len(results) < count and attempts < max_attempts:
        category, color_hex = color_sampler()
        attempts += 1
        if color_hex in used:
            continue
        used.add(color_hex)
        results.append((category, color_hex))

    if len(results) < count:
        raise ValueError('Unable to sample enough distinct colors from the color library.')
    return results


def _blend_with_alpha(base_img, overlay_img, alpha_mask):
    base = base_img.astype(np.float32)
    overlay = overlay_img.astype(np.float32)
    alpha = alpha_mask.astype(np.float32)[..., None]
    blended = base * (1.0 - alpha) + overlay * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _apply_solid_gradient_pattern(img, max_opacity=0.2):
    h, w = img.shape[:2]
    direction = random.choice(['horizontal', 'vertical', 'diag_down', 'diag_up'])

    if direction == 'horizontal':
        t = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))
    elif direction == 'vertical':
        t = np.tile(np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None], (1, w))
    elif direction == 'diag_down':
        xv = np.linspace(0.0, 1.0, w, dtype=np.float32)
        yv = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        t = (xv + yv) * 0.5
    else:
        xv = np.linspace(1.0, 0.0, w, dtype=np.float32)
        yv = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        t = (xv + yv) * 0.5

    opacity = random.uniform(0.0, max_opacity)
    alpha = t * opacity
    white_overlay = np.full_like(img, 255, dtype=np.uint8)
    return _blend_with_alpha(img, white_overlay, alpha)


def _apply_stripes_pattern(img, label_map, stripe_colors_bgr, label_indices):
    h, w = img.shape[:2]
    out = img.copy()
    map_out = label_map.copy()
    orientation = random.choices(['vertical', 'horizontal', 'angled'], weights=[0.25, 0.25, 0.5], k=1)[0]
    stripe_idx = 0

    if orientation == 'vertical':
        x = random.randint(0, 10)
        while x < w:
            stripe_w = random.randint(max(4, w // 20), max(8, w // 8))
            gap = random.randint(max(3, w // 30), max(7, w // 10))
            color = stripe_colors_bgr[stripe_idx % len(stripe_colors_bgr)]
            label_idx = label_indices[stripe_idx % len(label_indices)]
            cv2.rectangle(out, (x, 0), (min(w, x + stripe_w), h), color, -1)
            cv2.rectangle(map_out, (x, 0), (min(w, x + stripe_w), h), int(label_idx), -1)
            x += stripe_w + gap
            stripe_idx += 1
    elif orientation == 'horizontal':
        y = random.randint(0, 10)
        while y < h:
            stripe_h = random.randint(max(4, h // 20), max(8, h // 8))
            gap = random.randint(max(3, h // 30), max(7, h // 10))
            color = stripe_colors_bgr[stripe_idx % len(stripe_colors_bgr)]
            label_idx = label_indices[stripe_idx % len(label_indices)]
            cv2.rectangle(out, (0, y), (w, min(h, y + stripe_h)), color, -1)
            cv2.rectangle(map_out, (0, y), (w, min(h, y + stripe_h)), int(label_idx), -1)
            y += stripe_h + gap
            stripe_idx += 1
    else:
        min_dim = min(h, w)
        spacing = random.randint(max(8, min_dim // 16), max(20, min_dim // 6))
        thickness = random.randint(max(4, min_dim // 22), max(8, min_dim // 10))
        angle_deg = random.uniform(15.0, 165.0)
        if abs(angle_deg - 90.0) < 8.0:
            angle_deg += 12.0
        theta = np.deg2rad(angle_deg)
        d = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        n = np.array([-d[1], d[0]], dtype=np.float32)
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        diag = int(np.hypot(h, w))

        stripe_idx = 0
        for t in range(-diag, diag + spacing, spacing):
            color = stripe_colors_bgr[stripe_idx % len(stripe_colors_bgr)]
            label_idx = label_indices[stripe_idx % len(label_indices)]
            stripe_center = center + n * float(t)
            p1 = stripe_center - d * float(diag)
            p2 = stripe_center + d * float(diag)
            cv2.line(
                out,
                (int(round(p1[0])), int(round(p1[1]))),
                (int(round(p2[0])), int(round(p2[1]))),
                color=color,
                thickness=thickness,
            )
            cv2.line(
                map_out,
                (int(round(p1[0])), int(round(p1[1]))),
                (int(round(p2[0])), int(round(p2[1]))),
                color=int(label_idx),
                thickness=thickness,
            )
            stripe_idx += 1

    return out, map_out


def _apply_color_blocking_pattern(img, label_map, block_colors_bgr, label_indices):
    h, w = img.shape[:2]
    out = img.copy()
    map_out = label_map.copy()
    image_area = h * w
    n_blocks = len(block_colors_bgr)
    centers = []
    diag = float(np.hypot(h, w))
    min_center_dist = 0.5 * diag

    # Favor large, dominant shapes. Each block is intentionally big.
    target_total_fraction = random.uniform(0.6, 0.9)
    area_per_block = int((target_total_fraction * image_area) / n_blocks)

    for i, color in enumerate(block_colors_bgr):
        label_idx = label_indices[i]
        # Keep aspect closer to square so blocks read as "big shapes", not thin bands.
        aspect = random.uniform(0.8, 1.25)
        bh = int(np.sqrt(area_per_block / max(aspect, 1e-6)))
        bw = int(area_per_block / max(bh, 1))
        bh = max(int(0.55 * h), min(int(1.35 * h), bh))
        bw = max(int(0.55 * w), min(int(1.35 * w), bw))

        placed = False
        for _ in range(60):
            place_mode = random.choice(['centerish', 'off_side'])
            if place_mode == 'centerish':
                cx = random.randint(int(0.3 * w), int(0.7 * w))
                cy = random.randint(int(0.3 * h), int(0.7 * h))
            else:
                side = random.choice(['left', 'right', 'top', 'bottom'])
                if side == 'left':
                    cx = random.randint(-bw // 3, bw // 3)
                    cy = random.randint(0, h - 1)
                elif side == 'right':
                    cx = random.randint(w - bw // 3, w + bw // 3)
                    cy = random.randint(0, h - 1)
                elif side == 'top':
                    cx = random.randint(0, w - 1)
                    cy = random.randint(-bh // 3, bh // 3)
                else:
                    cx = random.randint(0, w - 1)
                    cy = random.randint(h - bh // 3, h + bh // 3)

            if all(np.hypot(cx - px, cy - py) >= min_center_dist for px, py in centers):
                x1 = cx - bw // 2
                y1 = cy - bh // 2
                x2 = cx + bw // 2
                y2 = cy + bh // 2
                cv2.rectangle(out, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(map_out, (x1, y1), (x2, y2), int(label_idx), -1)
                centers.append((cx, cy))
                placed = True
                break

        if not placed:
            x1 = random.randint(-bw // 4, max(-bw // 4, w - bw // 2))
            y1 = random.randint(-bh // 4, max(-bh // 4, h - bh // 2))
            x2 = x1 + bw
            y2 = y1 + bh
            cv2.rectangle(out, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(map_out, (x1, y1), (x2, y2), int(label_idx), -1)

    return out, map_out


def _apply_polka_dot_pattern(img, label_map, dot_colors_bgr, label_indices):
    h, w = img.shape[:2]
    out = img.copy()
    map_out = label_map.copy()
    min_dim = min(h, w)
    n_dots = random.randint(30, 90)

    for i in range(n_dots):
        color = dot_colors_bgr[i % len(dot_colors_bgr)]
        label_idx = label_indices[i % len(label_indices)]
        shape = random.choice(['circle', 'ellipse', 'rounded_rect'])
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)

        if shape == 'circle':
            radius = random.randint(max(3, min_dim // 25), max(8, min_dim // 9))
            cv2.circle(out, (cx, cy), radius, color, -1)
            cv2.circle(map_out, (cx, cy), radius, int(label_idx), -1)
        elif shape == 'ellipse':
            axis1 = random.randint(max(4, min_dim // 30), max(9, min_dim // 10))
            axis2 = random.randint(max(4, min_dim // 30), max(9, min_dim // 10))
            angle = random.randint(0, 179)
            cv2.ellipse(out, (cx, cy), (axis1, axis2), angle, 0, 360, color, -1)
            cv2.ellipse(map_out, (cx, cy), (axis1, axis2), angle, 0, 360, int(label_idx), -1)
        else:
            rw = random.randint(max(6, min_dim // 20), max(12, min_dim // 7))
            rh = random.randint(max(6, min_dim // 20), max(12, min_dim // 7))
            x1 = cx - rw // 2
            y1 = cy - rh // 2
            x2 = x1 + rw
            y2 = y1 + rh
            corner = max(2, min(rw, rh) // 4)
            cv2.rectangle(out, (x1 + corner, y1), (x2 - corner, y2), color, -1)
            cv2.rectangle(out, (x1, y1 + corner), (x2, y2 - corner), color, -1)
            cv2.circle(out, (x1 + corner, y1 + corner), corner, color, -1)
            cv2.circle(out, (x2 - corner, y1 + corner), corner, color, -1)
            cv2.circle(out, (x1 + corner, y2 - corner), corner, color, -1)
            cv2.circle(out, (x2 - corner, y2 - corner), corner, color, -1)

            cv2.rectangle(map_out, (x1 + corner, y1), (x2 - corner, y2), int(label_idx), -1)
            cv2.rectangle(map_out, (x1, y1 + corner), (x2, y2 - corner), int(label_idx), -1)
            cv2.circle(map_out, (x1 + corner, y1 + corner), corner, int(label_idx), -1)
            cv2.circle(map_out, (x2 - corner, y1 + corner), corner, int(label_idx), -1)
            cv2.circle(map_out, (x1 + corner, y2 - corner), corner, int(label_idx), -1)
            cv2.circle(map_out, (x2 - corner, y2 - corner), corner, int(label_idx), -1)

    return out, map_out


def _compute_label_percentages(labels, label_map):
    total = float(label_map.size)
    if total <= 0:
        return {}

    unique_indices, counts = np.unique(label_map, return_counts=True)
    percentages = {}
    for idx, count in zip(unique_indices.tolist(), counts.tolist()):
        label = labels[int(idx)]
        pct = (count / total) * 100.0
        percentages[label] = percentages.get(label, 0.0) + pct

    return {k: round(v, 2) for k, v in percentages.items()}


def _sample_texture_opacity(min_alpha=0.3, max_alpha=1.0):
    return float(np.random.uniform(min_alpha, max_alpha))

# =================================================
#               HELPER FUNCTIONS
# =================================================

def _perlin_2d(shape, res, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # shape must be divisible by res
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:shape[0]*1j, 0:res[1]:shape[1]*1j].transpose(1, 2, 0) % 1

    angles = 2 * np.pi * rng.random((res[0] + 1, res[1] + 1))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))

    g00 = gradients[:-1, :-1].repeat(d[0], axis=0).repeat(d[1], axis=1)
    g10 = gradients[1:,  :-1].repeat(d[0], axis=0).repeat(d[1], axis=1)
    g01 = gradients[:-1, 1: ].repeat(d[0], axis=0).repeat(d[1], axis=1)
    g11 = gradients[1:,  1: ].repeat(d[0], axis=0).repeat(d[1], axis=1)

    n00 = (grid * g00).sum(axis=2)
    n10 = (np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10).sum(axis=2)
    n01 = (np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01).sum(axis=2)
    n11 = (np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11).sum(axis=2)

    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    t = fade(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

def _fbm_perlin(shape=(224, 224), base_res=(7, 7), octaves=5, persistence=0.5, seed=None):
    rng = np.random.default_rng(seed)
    noise = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0

    for i in range(octaves):
        res = (base_res[0] * (2 ** i), base_res[1] * (2 ** i))
        noise += amplitude * _perlin_2d(shape, res, rng)
        total_amp += amplitude
        amplitude *= persistence

    noise /= total_amp
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return (noise * 255).astype(np.uint8)

def _linearization_and_black_level_sub(img, black_level=0.0):
    """Subtracts base sensor noise and undoes gamma compression."""
    # Subtract black level (clipped to 0)
    img = np.maximum(img - black_level, 0)
    # De-gamma (Standard sRGB approx is 2.2)
    return np.power(img, 2.2)

def _chromatic_adaptation_shades_of_gray(img, p=6):
    """Estimates the illuminant using the Lp-norm (Shades of Gray)."""
    # Calculate the p-th power of the image
    img_p = np.power(img, p)
    
    # Calculate the p-th mean for each channel (B, G, R)
    # We use the Minkowski norm: (1/N * sum(x^p))^(1/p)
    white_b = np.power(np.mean(img_p[:, :, 0]), 1/p)
    white_g = np.power(np.mean(img_p[:, :, 1]), 1/p)
    white_r = np.power(np.mean(img_p[:, :, 2]), 1/p)
    
    # Scale to prevent dimming (preserve G or average brightness)
    avg_white = (white_b + white_g + white_r) / 3.0
    
    img[:, :, 0] *= (avg_white / (white_b + 1e-8))
    img[:, :, 1] *= (avg_white / (white_g + 1e-8))
    img[:, :, 2] *= (avg_white / (white_r + 1e-8))
    
    return img

def _exposure_and_local_contrast(img):
    """Uses CLAHE in LAB space to fix exposure without shifting hues."""
    # Convert to LAB for Luminance-based processing
    img_uint8 = (img * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge and convert back
    limg = cv2.merge((cl, a, b))
    final_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final_bgr.astype(np.float32) / 255.0


def _sample_truncated_normal(mean, std, low, high):
    a = (low - mean) / std
    b = (high - mean) / std
    return float(truncnorm.rvs(a, b, loc=mean, scale=std))


def _apply_global_lighting(
    img,
    p=0.5,
    brightness_range=(0.3, 1.8),
    temp_range=(0.75, 1.25),
    sat_range=(0.3, 2.0),
    hue_shift_deg=25.0,
    jpeg_quality_range=(40, 85),
):
    """Apply stochastic global lighting augmentations to a uint8 BGR image.

    Args:
        p: Probability each effect is applied.
        brightness_range: (low, high) multiplier for overall brightness.
        temp_range: (low, high) uniform scale applied independently to R and B channels.
        sat_range: (low, high) multiplier for HSV saturation channel.
        hue_shift_deg: Max hue rotation in degrees (applied as ±hue_shift_deg).
        jpeg_quality_range: (min, max) JPEG quality for compression simulation.
    """
    out = img.astype(np.float32) / 255.0
    if random.random() < p:
        out = _augment_brightness_contrast(out, brightness_range=brightness_range)
    if random.random() < p:
        out = _augment_color_temperature(out, temp_range=temp_range)
    if random.random() < p:
        out = _augment_color_jitter(out, sat_range=sat_range, hue_shift_deg=hue_shift_deg)
    if random.random() < p:
        out = _augment_shadow(out)
    if random.random() < p:
        out = _augment_gaussian_blur(out)
    if random.random() < p:
        out = _augment_salt_and_pepper(out)
    out = (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)
    if random.random() < p:
        out = _augment_jpeg_compression(out, quality_range=jpeg_quality_range)
    return out


def _augment_brightness_contrast(img_float, brightness_range=(0.3, 1.8)):
    """Scale overall brightness using Truncated Normal ~ TN(1.0, 0.35, brightness_range)."""
    low, high = brightness_range
    factor = _sample_truncated_normal(mean=1.0, std=0.35, low=low, high=high)
    return img_float * factor


def _augment_color_temperature(img_float, temp_range=(0.75, 1.25)):
    """Simulate warm/cool lighting by independently scaling the R and B channels."""
    low, high = temp_range
    r_scale = random.uniform(low, high)
    b_scale = random.uniform(low, high)
    out = img_float.copy()
    out[:, :, 2] = np.clip(out[:, :, 2] * r_scale, 0.0, 1.0)  # R channel (BGR index 2)
    out[:, :, 0] = np.clip(out[:, :, 0] * b_scale, 0.0, 1.0)  # B channel (BGR index 0)
    return out


def _augment_color_jitter(img_float, sat_range=(0.3, 2.0), hue_shift_deg=25.0):
    """Shift hue by ±hue_shift_deg and scale saturation via TN(1.0, 0.4, sat_range)."""
    img_uint8 = (np.clip(img_float, 0.0, 1.0) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
    # OpenCV H is in [0, 180]; shift by a random amount within ±hue_shift_deg
    hue_shift = random.uniform(-hue_shift_deg, hue_shift_deg)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180.0
    # Saturation scaling: Δs ~ TN(1.0, 0.4, sat_range)
    low, high = sat_range
    sat_scale = _sample_truncated_normal(mean=1.0, std=0.4, low=low, high=high)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0.0, 255.0)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result.astype(np.float32) / 255.0


def _augment_gaussian_blur(img_float):
    """Blur with sigma drawn from a half-normal, clipped to [0.1, 1.5]."""
    sigma = float(np.clip(abs(np.random.normal(0.0, 0.5)), 0.1, 1.5))
    return gaussian_filter(img_float, sigma=[sigma, sigma, 0])


def _augment_shadow(img_float, max_strength=0.55):
    """Overlay a directional gradient shadow with a subtle cool (blue) tint in the dark region.

    A linear ramp runs from 0 (no shadow) at one image edge to `strength` (full shadow)
    at the opposite edge, oriented in a random direction. The shadow darkens luminance
    and slightly boosts the blue channel to mimic ambient skylight.
    """
    h, w = img_float.shape[:2]
    strength = random.uniform(0.1, max_strength)

    direction = random.choice(['left', 'right', 'top', 'bottom'])
    if direction == 'left':
        ramp = np.tile(np.linspace(strength, 0.0, w, dtype=np.float32), (h, 1))
    elif direction == 'right':
        ramp = np.tile(np.linspace(0.0, strength, w, dtype=np.float32), (h, 1))
    elif direction == 'top':
        ramp = np.tile(np.linspace(strength, 0.0, h, dtype=np.float32)[:, None], (1, w))
    else:
        ramp = np.tile(np.linspace(0.0, strength, h, dtype=np.float32)[:, None], (1, w))

    out = img_float.copy()
    # Darken all channels
    out = out * (1.0 - ramp[:, :, None])
    # Cool the shadow: lift blue slightly proportional to the ramp
    cool_amount = ramp * 0.12
    out[:, :, 0] = np.clip(out[:, :, 0] + cool_amount, 0.0, 1.0)  # B channel
    return np.clip(out, 0.0, 1.0)


def _augment_jpeg_compression(img_uint8, quality_range=(40, 85)):
    """Simulate JPEG camera compression artifacts by encoding and decoding at a random quality."""
    quality = random.randint(*quality_range)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode('.jpg', img_uint8, encode_params)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _augment_salt_and_pepper(img_float):
    """Add salt-and-pepper noise with a uniformly sampled density in [0.001, 0.005]."""
    density = random.uniform(0.001, 0.005)
    out = img_float.copy()
    n_noise = int(density * out.shape[0] * out.shape[1])
    rng = np.random.default_rng()
    salt_y = rng.integers(0, out.shape[0], n_noise)
    salt_x = rng.integers(0, out.shape[1], n_noise)
    pepper_y = rng.integers(0, out.shape[0], n_noise)
    pepper_x = rng.integers(0, out.shape[1], n_noise)
    out[salt_y, salt_x] = 1.0
    out[pepper_y, pepper_x] = 0.0
    return out