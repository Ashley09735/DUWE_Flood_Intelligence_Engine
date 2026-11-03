"""
DUWE DISASTER INTELLIGENCE
DUWE AI FLOOD INTELLIGENCE ENGINE

Lesson 7: Sub-catchment Delineation (Updated with better pour points)
Module: Watershed Intelligence Module – Advanced
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
flow_acc_path = os.path.join(base_dir, r"01_data\processed\flow_accumulation_duwe.tif")
subcatch_path = os.path.join(base_dir, r"01_data\processed\subcatchments_duwe.tif")
figure_path = os.path.join(base_dir, r"03_outputs\figures\lesson_07_subcatchments.png")

print("=" * 70)
print("DUWE DISASTER INTELLIGENCE")
print("DUWE AI FLOOD INTELLIGENCE ENGINE")
print("=" * 70)
print("LESSON 7: SUB-CATCHMENT DELINEATION (Updated)")
print("Module: Watershed Intelligence Module – Advanced")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

# -------------------------------------------------
# POUR POINTS (Selected from improved suggestions)
# -------------------------------------------------
pour_points = [
    (2447, 2972, "Main Outlet"),
    (2387, 2770, "Sub-catchment 2"),
    (2188, 2702, "Sub-catchment 4"),
    (2047, 2565, "Sub-catchment 6"),
    (1973, 2497, "Sub-catchment 7"),
]

print("\n[1] Pour points to be used:")
for i, (row, col, name) in enumerate(pour_points, 1):
    print(f"    {i}. {name:<20} → Row {row}, Col {col}")

# -------------------------------------------------
# 2. LOAD DATA
# -------------------------------------------------
print("\n[2] Loading rasters...")

with rasterio.open(dem_path) as src:
    dem = src.read(1).astype("float64")
    profile = src.profile.copy()

with rasterio.open(flow_dir_path) as src:
    fdir = src.read(1)

with rasterio.open(flow_acc_path) as src:
    acc = src.read(1).astype("float64")

print("    Rasters loaded successfully.")

# -------------------------------------------------
# 3. INITIALIZE PYSHEDS
# -------------------------------------------------
print("\n[3] Initializing PySheds...")

grid = Grid.from_raster(dem_path)
fdir_grid = grid.read_raster(flow_dir_path, nodata=-9999.0)

# -------------------------------------------------
# 4. DELINEATE SUB-CATCHMENTS
# -------------------------------------------------
print("\n[4] Delineating sub-catchments...")

subcatch = np.zeros(dem.shape, dtype="float32")

for idx, (row, col, name) in enumerate(pour_points, 1):
    print(f"    → Delineating {name}...")
    
    catch = grid.catchment(
        x=col,
        y=row,
        fdir=fdir_grid,
        routing="d8",
        xytype="index"
    )
    
    catch_array = np.asarray(catch)
    
    # Assign ID only to cells not yet assigned
    mask = (catch_array == 1) & (subcatch == 0)
    subcatch[mask] = idx

print("    Sub-catchment delineation completed.")

# -------------------------------------------------
# 5. SAVE SUB-CATCHMENT RASTER
# -------------------------------------------------
print("\n[5] Saving sub-catchment raster...")

mask_nodata = dem < -1000
subcatch[mask_nodata] = -9999.0

profile.update({
    "dtype": "float32",
    "nodata": -9999.0,
    "compress": "lzw"
})

with rasterio.open(subcatch_path, "w", **profile) as dst:
    dst.write(subcatch, 1)

print(f"    Saved to:\n    {subcatch_path}")

# -------------------------------------------------
# 6. SUB-CATCHMENT STATISTICS
# -------------------------------------------------
print("\n[6] SUB-CATCHMENT STATISTICS")
print("-" * 75)
print(f"{'ID':<4} {'Name':<20} {'Pixels':>12} {'Area (km²)':>12} {'Max Acc':>14}")
print("-" * 75)

pixel_area_km2 = (30 * 30) / 1_000_000

for idx, (row, col, name) in enumerate(pour_points, 1):
    pixels = np.sum(subcatch == idx)
    area = pixels * pixel_area_km2
    max_acc = np.nanmax(acc[subcatch == idx]) if pixels > 0 else 0
    
    print(f"{idx:<4} {name:<20} {pixels:>12,} {area:>12.2f} {max_acc:>14,.0f}")

total_sub_pixels = np.sum(subcatch > 0)
print("-" * 75)
print(f"{'Total assigned':<25} {total_sub_pixels:>12,}")

# -------------------------------------------------
# 7. VISUALISATION
# -------------------------------------------------
print("\n[7] Generating sub-catchment map...")

fig, ax = plt.subplots(figsize=(12, 9))

subcatch_plot = np.where(subcatch > 0, subcatch, np.nan)

im = ax.imshow(subcatch_plot, cmap="tab10", vmin=1, vmax=len(pour_points))

# Mark pour points
for row, col, name in pour_points:
    ax.plot(col, row, "k*", markersize=11)

ax.set_title(
    "DUWE AI FLOOD INTELLIGENCE ENGINE\nLesson 7 – Sub-catchment Delineation",
    fontsize=14, fontweight="bold", pad=12
)
ax.set_xlabel("Column")
ax.set_ylabel("Row")

cbar = plt.colorbar(im, ax=ax, shrink=0.75)
cbar.set_label("Sub-catchment ID")

plt.figtext(0.5, 0.01, "DUWE DISASTER INTELLIGENCE  |  Watershed Intelligence Module",
            ha="center", fontsize=9, style="italic", color="gray")

plt.tight_layout()
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n    Figure saved to:\n    {figure_path}")

print("\n" + "=" * 70)
print("LESSON 7 COMPLETED SUCCESSFULLY")
print("Sub-catchments have been delineated and saved")
print("=" * 70)
