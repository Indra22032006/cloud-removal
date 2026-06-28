"""
extract_patches.py
Split the cropped AOI into smaller patches, and classify each patch
as 'clean' (cloud-free) or 'cloudy' using a brightness-based heuristic.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CROPPED_DIR = Path(__file__).parent.parent / "data" / "cropped"
PATCH_DIR = Path(__file__).parent.parent / "data" / "patches"
PATCH_DIR.mkdir(parents=True, exist_ok=True)
(PATCH_DIR / "clean").mkdir(exist_ok=True)
(PATCH_DIR / "cloudy").mkdir(exist_ok=True)

BAND_PATHS = {
    "green": CROPPED_DIR / "BAND2_cropped.tif",
    "red": CROPPED_DIR / "BAND3_cropped.tif",
    "nir": CROPPED_DIR / "BAND4_cropped.tif",
}

PATCH_SIZE = 256  # pixels (~1.28km x 1.28km at 5m/px)
STRIDE = 256      # no overlap; set smaller than PATCH_SIZE for overlapping patches

# Cloud detection heuristic: clouds are very bright across all bands.
# LISS-IV is 10-bit (0-1023). Clouds typically saturate well above land/vegetation.
CLOUD_BRIGHTNESS_THRESHOLD = 270   # mean pixel value above this -> likely cloud
CLOUD_PIXEL_FRACTION_THRESHOLD = 0.02  # if >5% of patch pixels are "bright", call it cloudy


def load_stacked():
    bands = []
    for name in ["green", "red", "nir"]:
        with rasterio.open(BAND_PATHS[name]) as src:
            bands.append(src.read(1))
    return np.stack(bands, axis=-1)  # (H, W, 3), uint16


def is_cloudy(patch):
    """
    Classify a patch as cloudy if a meaningful fraction of its pixels
    are bright across all three bands (clouds are bright in visible + NIR,
    unlike vegetation which is bright mainly in NIR, or water which is dark).
    """
    brightness = patch.mean(axis=-1)  # average across green, red, nir
    bright_pixel_fraction = np.mean(brightness > CLOUD_BRIGHTNESS_THRESHOLD)
    return bright_pixel_fraction > CLOUD_PIXEL_FRACTION_THRESHOLD, bright_pixel_fraction


def extract_patches(stacked, patch_size=PATCH_SIZE, stride=STRIDE):
    h, w, _ = stacked.shape
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = stacked[y:y + patch_size, x:x + patch_size, :]
            patches.append((x, y, patch))
    return patches


def save_patch_preview(patch, path):
    """Save a quick PNG preview (false color) of a patch for visual inspection."""
    false_color = patch[:, :, [2, 1, 0]].astype(np.float32)
    for i in range(3):
        ch = false_color[:, :, i]
        lo, hi = np.percentile(ch, [2, 98])
        false_color[:, :, i] = np.clip((ch - lo) / (hi - lo + 1e-6), 0, 1)
    plt.imsave(path, false_color)


def main():
    print("Loading cropped AOI...")
    stacked = load_stacked()
    print(f"Shape: {stacked.shape}")

    print(f"\nExtracting {PATCH_SIZE}x{PATCH_SIZE} patches (stride={STRIDE})...")
    patches = extract_patches(stacked)
    print(f"Total patches: {len(patches)}")

    clean_count = 0
    cloudy_count = 0

    for x, y, patch in patches:
        cloudy, frac = is_cloudy(patch)
        label = "cloudy" if cloudy else "clean"

        # Save as .npy for training use (preserves full uint16 precision)
        np.save(PATCH_DIR / label / f"patch_{y}_{x}.npy", patch)

        # Save a quick PNG preview for visual sanity-checking
        save_patch_preview(patch, PATCH_DIR / label / f"patch_{y}_{x}_preview.png")

        if cloudy:
            cloudy_count += 1
        else:
            clean_count += 1

    print(f"\nClean patches: {clean_count}")
    print(f"Cloudy patches: {cloudy_count}")
    print(f"\nSaved to: {PATCH_DIR}")
    print("Check the *_preview.png files in data/patches/clean and data/patches/cloudy")
    print("to confirm the classification looks visually correct.")


if __name__ == "__main__":
    main()