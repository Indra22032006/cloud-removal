"""
pick_aoi.py
Visualize the full scene with a km grid overlay, and/or auto-pick a 15x15km AOI.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
BAND_PATHS = {
    "green": DATA_DIR / "BAND2.tif",
    "red": DATA_DIR / "BAND3.tif",
    "nir": DATA_DIR / "BAND4.tif",
}

AOI_SIZE_M = 15000  # 15km box


def load_stacked_with_transform():
    bands = []
    transform = None
    crs = None
    for name in ["green", "red", "nir"]:
        with rasterio.open(BAND_PATHS[name]) as src:
            bands.append(src.read(1))
            transform = src.transform
            crs = src.crs
    stacked = np.stack(bands, axis=-1)
    return stacked, transform, crs


def make_false_color(stacked):
    false_color = stacked[:, :, [2, 1, 0]].astype(np.float32)
    for i in range(3):
        ch = false_color[:, :, i]
        lo, hi = np.percentile(ch, [2, 98])
        false_color[:, :, i] = np.clip((ch - lo) / (hi - lo + 1e-6), 0, 1)
    return false_color


def pixel_size_m(transform):
    """Get pixel size in meters from the affine transform."""
    return abs(transform.a), abs(transform.e)


def show_with_grid(stacked, transform, downsample=4, grid_km=10):
    false_color = make_false_color(stacked)
    small = false_color[::downsample, ::downsample, :]

    px_w, px_h = pixel_size_m(transform)
    eff_px_w = px_w * downsample
    eff_px_h = px_h * downsample

    h, w = small.shape[:2]

    fig, ax = plt.subplots(figsize=(14, 14 * h / w))
    ax.imshow(small)

    # Draw grid lines every grid_km kilometers
    grid_m = grid_km * 1000
    x_step_px = grid_m / eff_px_w
    y_step_px = grid_m / eff_px_h

    x = 0
    km_x = 0
    while x < w:
        ax.axvline(x, color="yellow", linewidth=0.5, alpha=0.6)
        ax.text(x, -10, f"{km_x}km", color="yellow", fontsize=7, ha="center")
        x += x_step_px
        km_x += grid_km

    y = 0
    km_y = 0
    while y < h:
        ax.axhline(y, color="yellow", linewidth=0.5, alpha=0.6)
        ax.text(-25, y, f"{km_y}km", color="yellow", fontsize=7, va="center")
        y += y_step_px
        km_y += grid_km

    ax.set_title("Full scene with km grid (use this to pick your 15x15km box)")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def auto_pick_aoi(stacked, transform, size_m=AOI_SIZE_M, seed=None):
    """
    Randomly pick a size_m x size_m box, but biased toward land (avoiding
    large uniform regions like open water) using a simple variance check.
    """
    rng = np.random.default_rng(seed)
    px_w, px_h = pixel_size_m(transform)
    box_px_w = int(size_m / px_w)
    box_px_h = int(size_m / px_h)

    h, w = stacked.shape[:2]

    best_box = None
    best_score = -1

    # Try several random candidates, score by pixel variance (texture richness)
    # Higher variance = more terrain diversity = less likely to be flat water/uniform
    for _ in range(40):
        y0 = rng.integers(0, h - box_px_h)
        x0 = rng.integers(0, w - box_px_w)
        patch = stacked[y0:y0 + box_px_h, x0:x0 + box_px_w, :]

        # Skip if patch touches nodata/black borders (rotated scene edges)
        if np.mean(patch == 0) > 0.05:
            continue

        score = np.var(patch.astype(np.float32))
        if score > best_score:
            best_score = score
            best_box = (x0, y0, box_px_w, box_px_h)

    return best_box


def show_candidate_box(stacked, transform, box, downsample=4):
    false_color = make_false_color(stacked)
    small = false_color[::downsample, ::downsample, :]

    x0, y0, bw, bh = box
    x0_d, y0_d, bw_d, bh_d = x0 / downsample, y0 / downsample, bw / downsample, bh / downsample

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(small)
    rect = patches.Rectangle((x0_d, y0_d), bw_d, bh_d, linewidth=2, edgecolor="red", facecolor="none")
    ax.add_patch(rect)
    ax.set_title("Auto-picked 15x15km AOI candidate (red box)")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"\nPixel coordinates of this box (in full-resolution image):")
    print(f"  x0={x0}, y0={y0}, width={bw}px, height={bh}px")
    print(f"  (i.e. rows {y0}:{y0+bh}, cols {x0}:{x0+bw})")


if __name__ == "__main__":
    print("Loading data...")
    stacked, transform, crs = load_stacked_with_transform()

    print("\n--- Option A: showing full scene with grid overlay ---")
    show_with_grid(stacked, transform, downsample=4, grid_km=10)

    print("\n--- Option B: auto-picking a good 15x15km candidate box ---")
    box = auto_pick_aoi(stacked, transform, seed=42)
    if box:
        show_candidate_box(stacked, transform, box, downsample=4)
    else:
        print("Couldn't find a good candidate — try adjusting parameters.")