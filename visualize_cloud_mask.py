"""
visualize_cloud_mask.py
Overlay detected 'cloud' pixels on the false-color image to visually verify
the brightness threshold actually catches the real clouds.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CROPPED_DIR = Path(__file__).parent.parent / "data" / "cropped"
BAND_PATHS = {
    "green": CROPPED_DIR / "BAND2_cropped.tif",
    "red": CROPPED_DIR / "BAND3_cropped.tif",
    "nir": CROPPED_DIR / "BAND4_cropped.tif",
}

THRESHOLD = 270  # try adjusting this and re-running to tune


def load_stacked():
    bands = []
    for name in ["green", "red", "nir"]:
        with rasterio.open(BAND_PATHS[name]) as src:
            bands.append(src.read(1))
    return np.stack(bands, axis=-1)


def main():
    stacked = load_stacked()
    brightness = stacked.mean(axis=-1)
    cloud_mask = brightness > THRESHOLD

    false_color = stacked[:, :, [2, 1, 0]].astype(np.float32)
    for i in range(3):
        ch = false_color[:, :, i]
        lo, hi = np.percentile(ch, [2, 98])
        false_color[:, :, i] = np.clip((ch - lo) / (hi - lo + 1e-6), 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(false_color)
    axes[0].set_title("False color")
    axes[0].axis("off")

    overlay = false_color.copy()
    overlay[cloud_mask] = [1, 0, 1]  # magenta highlight on detected cloud pixels
    axes[1].imshow(overlay)
    axes[1].set_title(f"Detected clouds (threshold={THRESHOLD}) in magenta")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Cloud pixel fraction at threshold {THRESHOLD}: {100*np.mean(cloud_mask):.2f}%")


if __name__ == "__main__":
    main()