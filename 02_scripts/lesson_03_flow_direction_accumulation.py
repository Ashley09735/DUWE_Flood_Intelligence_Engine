"""
DUWE DISASTER INTELLIGENCE
DUWE AI FLOOD INTELLIGENCE ENGINE

Lesson 3: Flow Direction & Flow Accumulation
Module: Hydrological Routing Engine
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
figure_path = os.path.join(base_dir, r"03_outputs\figures\lesson_03_flow_accumulation.png")

print("=" * 70)
print("DUWE DISASTER INTELLIGENCE")
print("DUWE AI FLOOD INTELLIGENCE ENGINE")
print("=" * 70)
print("LESSON 3: FLOW DIRECTION & FLOW ACCUMULATION")
print("Module: Hydrological Routing Engine")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

# -------------------------------------------------
# 1. LOAD CONDITIONED DEM
# -------------------------------------------------
print("\n[1] Loading conditioned DEM...")

with rasterio.open(dem_path) as src:
    dem = src.read(1).astype("float64")
    profile = src.profile.copy()
    nodata = src.nodata
    print(f"    NoData value : {nodata}")
    print(f"    Shape        : {dem.shape}")

# -------------------------------------------------
# 2. CREATE MASK
# -------------------------------------------------
print("\n[2] Creating valid data mask...")

mask_nodata = dem < -1000
dem_masked = dem.copy()
dem_masked[mask_nodata] = np.nan

valid_pixels = np.sum(np.isfinite(dem_masked))
print(f"    Valid pixels : {valid_pixels:,}")

# -------------------------------------------------
# 3. INITIALIZE PYSHEDS GRID
# -------------------------------------------------
print("\n[3] Initializing PySheds Grid...")

grid = Grid.from_raster(dem_path)
dem_grid = grid.read_raster(dem_path, nodata=-9999.0)

# -------------------------------------------------
# 4. CALCULATE FLOW DIRECTION (D8)
# -------------------------------------------------
print("\n[4] Calculating D8 Flow Direction...")

fdir = grid.flowdir(dem_grid, routing="d8")
fdir_array = np.asarray(fdir)

print("    Flow Direction completed.")

# -------------------------------------------------
# 5. FLOW DIRECTION DIAGNOSTICS
# -------------------------------------------------
print("\n[5] FLOW DIRECTION DIAGNOSTICS")
print("-" * 50)

values, counts = np.unique(fdir_array, return_counts=True)

for value, count in zip(values, counts):
    print(f"    Value {value:>4} : {count:>10,} cells")

# -------------------------------------------------
# 6. CALCULATE FLOW ACCUMULATION
# -------------------------------------------------
print("\n[6] Calculating Flow Accumulation...")

acc = grid.accumulation(fdir, routing="d8")
acc_array = np.asarray(acc, dtype="float64")

print(f"    Maximum accumulation : {np.nanmax(acc_array):,.0f}")

# -------------------------------------------------
# 7. SAVE RESULTS
# -------------------------------------------------
print("\n[7] Saving results...")

# Save Flow Direction
profile.update(dtype="float32", nodata=-9999.0)

fdir_save = fdir_array.astype("float32")
fdir_save[mask_nodata] = -9999.0

with rasterio.open(flow_dir_path, "w", **profile) as dst:
    dst.write(fdir_save, 1)

print(f"    Flow Direction saved to:\n    {flow_dir_path}")

# Save Flow Accumulation
acc_save = acc_array.copy()
acc_save[mask_nodata] = -9999.0

with rasterio.open(flow_acc_path, "w", **profile) as dst:
    dst.write(acc_save.astype("float32"), 1)

print(f"    Flow Accumulation saved to:\n    {flow_acc_path}")

# -------------------------------------------------
# 8. OUTLET CHECK
# -------------------------------------------------
print("\n[8] OUTLET CHECK")
print("-" * 50)

outlet_row = 2447
outlet_col = 2972

print(f"    Outlet Row          : {outlet_row}")
print(f"    Outlet Column       : {outlet_col}")
print(f"    Accumulation at outlet : {acc_array[outlet_row, outlet_col]:,.0f}")

# -------------------------------------------------
# 9. VISUALISATION
# -------------------------------------------------
print("\n[9] Generating Flow Accumulation map...")

# Prepare accumulation for display
acc_plot = acc_array.copy()
acc_plot[mask_nodata] = np.nan
acc_plot[acc_plot < 100] = np.nan          # hide very low accumulation

fig, ax = plt.subplots(figsize=(11, 9))

im = ax.imshow(
    np.log1p(acc_plot),
    cmap="Blues",
    vmin=np.log1p(100),
    vmax=np.log1p(np.nanpercentile(acc_array, 99.5))
)

cbar = plt.colorbar(im, ax=ax, shrink=0.75)
cbar.set_label("Log(Flow Accumulation + 1)", fontsize=11)

# Mark outlet
ax.plot(outlet_col, outlet_row, "r*", markersize=14, label="Outlet")
ax.legend(loc="upper right")

ax.set_title("DUWE AI FLOOD INTELLIGENCE ENGINE\nLesson 3 – Flow Accumulation (D8)",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Column")
ax.set_ylabel("Row")

plt.figtext(0.5, 0.01, "DUWE DISASTER INTELLIGENCE  |  Hydrological Routing Engine",
            ha="center", fontsize=9, style="italic", color="gray")

plt.tight_layout()
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n    Figure saved to:\n    {figure_path}")

print("\n" + "=" * 70)
print("LESSON 3 COMPLETED SUCCESSFULLY")
print("Flow Direction and Flow Accumulation are ready")
print("=" * 70)
