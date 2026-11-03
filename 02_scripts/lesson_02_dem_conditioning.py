"""
DUWE DISASTER INTELLIGENCE
DUWE AI FLOOD INTELLIGENCE ENGINE

Lesson 2: DEM Conditioning (Flat Resolution Only) - Fixed NoData
Module: Terrain Conditioning Module
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pysheds.grid import Grid
from datetime import datetime
import os

# -------------------------------------------------
# PATHS
# -------------------------------------------------
base_dir = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"

dem_path = os.path.join(base_dir, r"01_data\raw\filled_ayaya.tif")
conditioned_dem_path = os.path.join(base_dir, r"01_data\processed\dem_conditioned_duwe.tif")
figure_path = os.path.join(base_dir, r"03_outputs\figures\lesson_02_dem_conditioning.png")
temp_dem_path = os.path.join(base_dir, r"01_data\processed\temp_dem_for_pysheds.tif")

print("=" * 70)
print("DUWE DISASTER INTELLIGENCE")
print("DUWE AI FLOOD INTELLIGENCE ENGINE")
print("=" * 70)
print("LESSON 2: DEM CONDITIONING (Flat Resolution Only)")
print("Module: Terrain Conditioning Module")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("Note: DEM is already sink-filled")
print("=" * 70)

# -------------------------------------------------
# 1. LOAD ORIGINAL DEM
# -------------------------------------------------
print("\n[1] Loading original DEM...")

with rasterio.open(dem_path) as src:
    dem = src.read(1).astype("float64")
    profile = src.profile.copy()
    nodata = src.nodata
    print(f"    Original NoData value : {nodata}")
    print(f"    Shape                 : {dem.shape}")
    print(f"    Data type             : {dem.dtype}")

# -------------------------------------------------
# 2. STRONG NODATA DETECTION
# -------------------------------------------------
print("\n[2] Detecting NoData pixels...")

print(f"    Minimum value in DEM  : {np.min(dem)}")
print(f"    Maximum value in DEM  : {np.max(dem)}")

# Very strong NoData rule for this file
mask_nodata = dem < -1000          # Catches -99999 safely

dem_masked = dem.copy()
dem_masked[mask_nodata] = np.nan

valid_pixels = np.sum(np.isfinite(dem_masked))
nodata_pixels = np.sum(np.isnan(dem_masked))

print(f"    Valid pixels           : {valid_pixels:,}")
print(f"    NoData pixels          : {nodata_pixels:,}")

# -------------------------------------------------
# 3. SAVE TEMPORARY DEM FOR PYSHEDS
# -------------------------------------------------
print("\n[3] Preparing temporary DEM for PySheds...")

profile.update({
    "dtype": "float32",
    "nodata": -9999.0,
    "compress": "lzw"
})

dem_temp = np.where(np.isnan(dem_masked), -9999.0, dem_masked)

with rasterio.open(temp_dem_path, "w", **profile) as dst:
    dst.write(dem_temp.astype("float32"), 1)

print("    Temporary DEM saved.")

# -------------------------------------------------
# 4. RESOLVE FLATS ONLY
# -------------------------------------------------
print("\n[4] Resolving flat areas...")

grid = Grid.from_raster(temp_dem_path)
dem_grid = grid.read_raster(temp_dem_path, nodata=-9999.0)

dem_conditioned = grid.resolve_flats(dem_grid)

print("    Flat resolution completed.")

# -------------------------------------------------
# 5. SAVE CONDITIONED DEM
# -------------------------------------------------
print("\n[5] Saving conditioned DEM...")

cond_array = np.asarray(dem_conditioned, dtype="float64")
cond_array[mask_nodata] = -9999.0

with rasterio.open(conditioned_dem_path, "w", **profile) as dst:
    dst.write(cond_array.astype("float32"), 1)

print(f"    Saved to:\n    {conditioned_dem_path}")

# -------------------------------------------------
# 6. DIAGNOSTICS
# -------------------------------------------------
print("\n[6] CONDITIONING DIAGNOSTICS")
print("-" * 50)

valid_after = np.sum(cond_array != -9999.0)
print(f"    Valid pixels after     : {valid_after:,}")

original_valid = dem_masked[np.isfinite(dem_masked)]
conditioned_valid = cond_array[np.isfinite(dem_masked)]

diff = conditioned_valid - original_valid
modified = np.sum(np.abs(diff) > 0.001)

print(f"    Pixels modified        : {modified:,}")
print(f"    Maximum change         : {np.max(np.abs(diff)):.3f} m")
print(f"    Mean change            : {np.mean(diff):.4f} m")

# -------------------------------------------------
# 7. VISUAL COMPARISON
# -------------------------------------------------
print("\n[7] Generating comparison figure...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

im1 = axes[0].imshow(dem_masked, cmap="terrain")
axes[0].set_title("Original DEM (Already Sink-Filled)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Column")
axes[0].set_ylabel("Row")
plt.colorbar(im1, ax=axes[0], shrink=0.7, label="Elevation (m)")

cond_plot = np.where(cond_array == -9999.0, np.nan, cond_array)
im2 = axes[1].imshow(cond_plot, cmap="terrain")
axes[1].set_title("After Flat Resolution", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Column")
axes[1].set_ylabel("Row")
plt.colorbar(im2, ax=axes[1], shrink=0.7, label="Elevation (m)")

fig.suptitle("DUWE AI FLOOD INTELLIGENCE ENGINE\nLesson 2 – Terrain Conditioning Module",
             fontsize=15, fontweight="bold", y=1.02)

plt.figtext(0.5, 0.01, "DUWE DISASTER INTELLIGENCE  |  Terrain Conditioning Module",
            ha="center", fontsize=9, style="italic", color="gray")

plt.tight_layout()
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n    Figure saved to:\n    {figure_path}")

print("\n" + "=" * 70)
print("LESSON 2 COMPLETED SUCCESSFULLY")
print("Conditioned DEM is ready for Lesson 3")
print("=" * 70)
