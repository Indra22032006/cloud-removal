"""
cloud_mask.py
Generate realistic synthetic cloud masks using fractal (multi-octave) noise,
built entirely with numpy/scipy (no external noise library needed).

Used to create (cloudy_input, clean_target) training pairs from clean patches.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import matplotlib.pyplot as plt
from pathlib import Path

PATCH_DIR = Path(__file__).parent.parent / "data" / "patches"
PAIRS_DIR = Path(__file__).parent.parent / "data" / "training_pairs"
PAIRS_DIR.mkdir(parents=True, exist_ok=True)
(PAIRS_DIR / "input").mkdir(exist_ok=True)
(PAIRS_DIR / "target").mkdir(exist_ok=True)
(PAIRS_DIR / "mask").mkdir(exist_ok=True)


def generate_fractal_noise(shape, octaves=5, persistence=0.5, base_scale=8, seed=None):
    """
    Generate fractal noise by summing multiple octaves of smoothed random noise
    at different scales. This mimics Perlin noise's soft, blobby characteristic
    without needing an external library.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    noise = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    total_amplitude = 0.0

    for octave in range(octaves):
        scale = base_scale * (2 ** octave)
        small_h = max(2, h // scale)
        small_w = max(2, w // scale)

        small_noise = rng.random((small_h, small_w)).astype(np.float32)
        zoom_factors = (h / small_h, w / small_w)
        layer = zoom(small_noise, zoom_factors, order=3)

        layer = layer[:h, :w]
        if layer.shape != (h, w):
            padded = np.zeros((h, w), dtype=np.float32)
            ph, pw = layer.shape
            padded[:ph, :pw] = layer
            layer = padded

        noise += amplitude * layer
        total_amplitude += amplitude
        amplitude *= persistence

    noise /= total_amplitude
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


def generate_cloud_mask(shape, coverage=0.3, softness=4, seed=None):
    """
    Generate a soft cloud mask (values 0-1, where 1 = fully cloud-covered).
    Produces distinct, clumped blob clusters (like real cloud fields) rather
    than a smooth continuous gradient.
    """
    noise = generate_fractal_noise(shape, octaves=4, persistence=0.6, base_scale=10, seed=seed)

    threshold = np.percentile(noise, 100 * (1 - coverage))
    mask = (noise > threshold).astype(np.float32)

    mask = gaussian_filter(mask, sigma=softness)
    mask = np.clip((mask - 0.3) / 0.4, 0, 1)

    if mask.max() > 0:
        mask = mask / mask.max()

    return mask


def apply_cloud_to_patch(patch, mask, cloud_brightness=600, opacity_scale=1.0):
    """
    Blend a cloud appearance into a clean patch using the mask.
    """
    mask_3ch = mask[:, :, np.newaxis] * opacity_scale
    mask_3ch = np.clip(mask_3ch, 0, 1)

    cloud_layer = np.full_like(patch, cloud_brightness, dtype=np.float32)
    cloudy_patch = patch.astype(np.float32) * (1 - mask_3ch) + cloud_layer * mask_3ch

    return cloudy_patch.astype(patch.dtype)


def process_clean_patches(coverage_range=(0.15, 0.45), seed_base=0):
    """
    Take all clean patches, apply a synthetic cloud mask to each,
    and save (input, target, mask) triplets for training.
    """
    clean_files = sorted((PATCH_DIR / "clean").glob("*.npy"))
    print(f"Found {len(clean_files)} clean patches to process.\n")

    rng = np.random.default_rng(seed_base)

    for i, path in enumerate(clean_files):
        patch = np.load(path)

        coverage = rng.uniform(*coverage_range)
        mask = generate_cloud_mask(patch.shape[:2], coverage=coverage, softness=4, seed=seed_base + i)
        cloudy_patch = apply_cloud_to_patch(patch, mask, cloud_brightness=600)

        stem = path.stem
        np.save(PAIRS_DIR / "input" / f"{stem}.npy", cloudy_patch)
        np.save(PAIRS_DIR / "target" / f"{stem}.npy", patch)
        np.save(PAIRS_DIR / "mask" / f"{stem}.npy", mask)

        if i < 3:
            save_preview(patch, cloudy_patch, mask, PAIRS_DIR / f"{stem}_preview.png")

    print(f"Saved {len(clean_files)} training pairs to {PAIRS_DIR}")
    print("Check the *_preview.png files in data/training_pairs/ to confirm cloud realism.")


def save_preview(clean, cloudy, mask, out_path):
    def false_color(p):
        fc = p[:, :, [2, 1, 0]].astype(np.float32)
        for i in range(3):
            ch = fc[:, :, i]
            lo, hi = np.percentile(ch, [2, 98])
            fc[:, :, i] = np.clip((ch - lo) / (hi - lo + 1e-6), 0, 1)
        return fc

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(false_color(clean))
    axes[0].set_title("Clean (target)")
    axes[0].axis("off")

    axes[1].imshow(false_color(cloudy))
    axes[1].set_title("Synthetic cloudy (input)")
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Cloud mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


if __name__ == "__main__":
    process_clean_patches()