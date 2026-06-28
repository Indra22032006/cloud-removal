"""
crop_aoi.py
Crop the full LISS-IV scene to a specific AOI defined in km offsets
from the scene's top-left origin (matching the grid overlay from pick_aoi.py).
"""

import rasterio
from rasterio.windows import Window
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "cropped"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BAND_PATHS = {
    "green": DATA_DIR / "BAND2.tif",
    "red": DATA_DIR / "BAND3.tif",
    "nir": DATA_DIR / "BAND4.tif",
}

# AOI definition: km offsets from top-left of the scene (matching pick_aoi.py grid)
EAST_START_KM = 40
EAST_END_KM = 50
SOUTH_START_KM = 30
SOUTH_END_KM = 40


def crop_band(path, transform, east_start_m, east_end_m, south_start_m, south_end_m):
    """Crop a single band file to the given offset window (in meters from top-left)."""
    px_w = abs(transform.a)
    px_h = abs(transform.e)

    col_start = int(east_start_m / px_w)
    col_end = int(east_end_m / px_w)
    row_start = int(south_start_m / px_h)
    row_end = int(south_end_m / px_h)

    window = Window(col_start, row_start, col_end - col_start, row_end - row_start)

    with rasterio.open(path) as src:
        data = src.read(1, window=window)
        new_transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update({
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": new_transform,
        })
    return data, profile


def main():
    east_start_m = EAST_START_KM * 1000
    east_end_m = EAST_END_KM * 1000
    south_start_m = SOUTH_START_KM * 1000
    south_end_m = SOUTH_END_KM * 1000

    print(f"Cropping AOI: {EAST_START_KM}-{EAST_END_KM}km east, {SOUTH_START_KM}-{SOUTH_END_KM}km south")
    print(f"Box size: {EAST_END_KM-EAST_START_KM}km x {SOUTH_END_KM-SOUTH_START_KM}km\n")

    # Get the transform from one file first (all bands share the same transform)
    with rasterio.open(BAND_PATHS["green"]) as src:
        full_transform = src.transform

    cropped_bands = {}
    for name, path in BAND_PATHS.items():
        data, profile = crop_band(path, full_transform, east_start_m, east_end_m, south_start_m, south_end_m)
        cropped_bands[name] = data
        out_path = OUT_DIR / f"{path.stem}_cropped.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
        print(f"Saved: {out_path}  (shape: {data.shape})")

    # Quick visual sanity check
    stacked = np.stack([cropped_bands["green"], cropped_bands["red"], cropped_bands["nir"]], axis=-1)
    false_color = stacked[:, :, [2, 1, 0]].astype(np.float32)
    for i in range(3):
        ch = false_color[:, :, i]
        lo, hi = np.percentile(ch, [2, 98])
        false_color[:, :, i] = np.clip((ch - lo) / (hi - lo + 1e-6), 0, 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(false_color)
    plt.title(f"Cropped AOI: {EAST_END_KM-EAST_START_KM}km x {SOUTH_END_KM-SOUTH_START_KM}km")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"\nCropped percentage of nodata (black) pixels: {100*np.mean(stacked == 0):.2f}%")
    print("(if this is high, your box may be hitting the scene's rotated black edges)")


if __name__ == "__main__":
    main()