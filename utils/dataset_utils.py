import numpy as np
from scipy.ndimage import gaussian_filter, zoom

DEFAULT_DIMENSIONS = (224, 224)

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
