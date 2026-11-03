"""
DUWE DISASTER INTELLIGENCE
DUWE AI FLOOD INTELLIGENCE ENGINE

Lesson 5: Catchment Delineation
Module: Watershed Intelligence Module
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

dem_path = os.path.join(base_dir, r"01_data\processed\dem_conditioned_duwe.tif")
flow_dir_path = os.path.join(base_dir, r"01_data\processed\flow_direction_duwe.tif")
catchment_path = os.path.join(base_dir, r"01_data\processed\catchment_duwe.tif")
figure_path = os.path.join(base_dir, r"03_outputs\figures\lesson_05_catchment.png")

# -------------------------------------------------
# OUTLET LOCATION (from previous lessons)
# -------------------------------------------------
outlet_row = 2447
outlet_col = 2972

print("=" * 70)
print("DUWE DISASTER INTELLIGENCE")
print("DUWE AI FLOOD INTELLIGENCE ENGINE")
print("=" * 70)
print("LESSON 5: CATCHMENT DELINEATION")
print("Module: Watershed Intelligence Module")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Outlet (Row, Col): ({outlet_row}, {outlet_col})")
print("=" * 70)

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------
print("\n[1] Loading conditioned DEM and Flow Direction...")

with rasterio.open(dem_path) as src:
    dem = src.read(1).astype("float64")
    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs
    print(f"    DEM shape     : {dem.shape}")
    print(f"    CRS           : {crs}")

with rasterio.open(flow_dir_path) as src:
    fdir = src.read(1)

# -------------------------------------------------
# 2. CREATE MASK
# -------------------------------------------------
print("\n[2] Creating valid data mask...")

mask_nodata = dem < -1000
dem_masked = dem.copy()
dem_masked[mask_nodata] = np.nan

valid_pixels = np.sum(np.isfinite(dem_masked))
print(f"    Valid pixels  : {valid_pixels:,}")

# -------------------------------------------------
# 3. INITIALIZE PYSHEDS AND DELINEATE CATCHMENT
# -------------------------------------------------
print("\n[3] Delineating catchment from outlet...")

grid = Grid.from_raster(dem_path)
dem_grid = grid.read_raster(dem_path, nodata=-9999.0)
fdir_grid = grid.read_raster(flow_dir_path, nodata=-9999.0)

# Delineate catchment
catch = grid.catchment(
    x=outlet_col,
    y=outlet_row,
    fdir=fdir_grid,
    routing="d8",
    xytype="index"
)

catch_array = np.asarray(catch, dtype="float32")
print("    Catchment delineation completed.")

# -------------------------------------------------
# 4. SAVE CATCHMENT RASTER
# -------------------------------------------------
print("\n[4] Saving catchment raster...")

# 1 = inside catchment, 0 = outside, -9999 = NoData
catch_save = np.where(catch_array == 1, 1.0, 0.0)
catch_save[mask_nodata] = -9999.0

profile.update({
    "dtype": "float32",
    "nodata": -9999.0,
    "compress": "lzw"
})

with rasterio.open(catchment_path, "w", **profile) as dst:
    dst.write(catch_save, 1)

print(f"    Saved to:\n    {catchment_path}")

# -------------------------------------------------
# 5. CATCHMENT STATISTICS
# -------------------------------------------------
print("\n[5] CATCHMENT STATISTICS")
print("-" * 50)

catchment_pixels = np.sum(catch_array == 1)
pixel_area_km2 = (30 * 30) / 1_000_000          # 30m resolution
catchment_area_km2 = catchment_pixels * pixel_area_km2

print(f"    Catchment pixels          : {catchment_pixels:,}")
print(f"    Catchment area            : {catchment_area_km2:.2f} km²")
print(f"    Percentage of total terrain: {(catchment_pixels / valid_pixels) * 100:.2f}%")

# Elevation statistics inside catchment
dem_catch = dem_masked.copy()
dem_catch[catch_array != 1] = np.nan

print(f"\n    Elevation (inside catchment):")
print(f"      Minimum   : {np.nanmin(dem_catch):.1f} m")
print(f"      Maximum   : {np.nanmax(dem_catch):.1f} m")
print(f"      Mean      : {np.nanmean(dem_catch):.1f} m")
print(f"      Median    : {np.nanmedian(dem_catch):.1f} m")

# -------------------------------------------------
# 6. VISUALISATION
# -------------------------------------------------
print("\n[6] Generating catchment map...")

fig, ax = plt.subplots(figsize=(11, 9))

# Background DEM
im = ax.imshow(dem_masked, cmap="terrain", alpha=0.9)

# Catchment boundary / area
catch_plot = np.where(catch_array == 1, 1, np.nan)
ax.imshow(catch_plot, cmap="Reds", alpha=0.35, vmin=0, vmax=1)

# Outlet point
ax.plot(outlet_col, outlet_row, "b*", markersize=15, label="Outlet")

ax.set_title(
    "DUWE AI FLOOD INTELLIGENCE ENGINE\nLesson 5 – Delineated Catchment",
    fontsize=14, fontweight="bold", pad=12
)
ax.set_xlabel("Column")
ax.set_ylabel("Row")
ax.legend(loc="upper right")

plt.colorbar(im, ax=ax, shrink=0.75, label="Elevation (m)")

plt.figtext(0.5, 0.01, "DUWE DISASTER INTELLIGENCE  |  Watershed Intelligence Module",
            ha="center", fontsize=9, style="italic", color="gray")

plt.tight_layout()
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n    Figure saved to:\n    {figure_path}")

print("\n" + "=" * 70)
print("LESSON 5 COMPLETED SUCCESSFULLY")
print(f"Catchment Area: {catchment_area_km2:.2f} km²")
print("=" * 70)
