"""
data_utils.py
Load and inspect LISS-IV band TIFs for the cloud removal project.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths to the three single-band TIFs
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
BAND_PATHS = {
    "green": DATA_DIR / "BAND2.tif",
    "red": DATA_DIR / "BAND3.tif",
    "nir": DATA_DIR / "BAND4.tif",
}


def inspect_band(path):
    """Print key metadata for a single band file."""
    with rasterio.open(path) as src:
        print(f"\n--- {path.name} ---")
        print(f"Dimensions: {src.width} x {src.height}")
        print(f"CRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        print(f"Data type: {src.dtypes[0]}")
        print(f"NoData value: {src.nodata}")
        band = src.read(1)
        print(f"Value range: min={band.min()}, max={band.max()}")
        return band


def load_stacked(band_paths=BAND_PATHS):
    """Load all three bands and stack into a single (H, W, 3) array."""
    bands = []
    for name in ["green", "red", "nir"]:
        with rasterio.open(band_paths[name]) as src:
            bands.append(src.read(1))
    stacked = np.stack(bands, axis=-1)  # shape: (H, W, 3)
    print(f"\nStacked array shape: {stacked.shape}")
    return stacked


def show_false_color(stacked, downsample=4):
    """Quick false-color visualization (NIR-Red-Green), downsampled for speed."""
    small = stacked[::downsample, ::downsample, :]
    false_color = small[:, :, [2, 1, 0]].astype(np.float32)

    for i in range(3):
        ch = false_color[:, :, i]
        lo, hi = np.percentile(ch, [2, 98])
        false_color[:, :, i] = np.clip((ch - lo) / (hi - lo + 1e-6), 0, 1)

    plt.figure(figsize=(10, 10))
    plt.imshow(false_color)
    plt.title("False Color Composite (NIR-Red-Green)")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    print("Inspecting individual bands...")
    for name, path in BAND_PATHS.items():
        inspect_band(path)

    print("\nLoading and stacking bands...")
    stacked = load_stacked()

    print("\nDisplaying false-color composite...")
    show_false_color(stacked)