"""
debug_cloud_threshold.py
Inspect actual brightness values to figure out why cloud detection failed.
"""

import rasterio
import numpy as np
from pathlib import Path

CROPPED_DIR = Path(__file__).parent.parent / "data" / "cropped"
BAND_PATHS = {
    "green": CROPPED_DIR / "BAND2_cropped.tif",
    "red": CROPPED_DIR / "BAND3_cropped.tif",
    "nir": CROPPED_DIR / "BAND4_cropped.tif",
}


def load_stacked():
    bands = []
    for name in ["green", "red", "nir"]:
        with rasterio.open(BAND_PATHS[name]) as src:
            bands.append(src.read(1))
    return np.stack(bands, axis=-1)


def main():
    stacked = load_stacked()
    brightness = stacked.mean(axis=-1)

    print(f"Overall brightness stats across whole cropped AOI:")
    print(f"  Min: {brightness.min():.1f}")
    print(f"  Max: {brightness.max():.1f}")
    print(f"  Mean: {brightness.mean():.1f}")
    print(f"  Median: {np.median(brightness):.1f}")
    print(f"  95th percentile: {np.percentile(brightness, 95):.1f}")
    print(f"  99th percentile: {np.percentile(brightness, 99):.1f}")
    print(f"  99.9th percentile: {np.percentile(brightness, 99.9):.1f}")

    print(f"\nPixel counts above various thresholds:")
    for t in [300, 400, 500, 600, 700, 800, 900]:
        frac = np.mean(brightness > t)
        print(f"  > {t}: {100*frac:.2f}% of pixels")


if __name__ == "__main__":
    main()