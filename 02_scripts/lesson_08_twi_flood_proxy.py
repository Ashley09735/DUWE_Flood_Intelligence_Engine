"""
DUWE DISASTER INTELLIGENCE
DUWE AI FLOOD INTELLIGENCE ENGINE

Lesson 8: Topographic Wetness Index & Flood Proxy Mapping (Corrected)
Module: Flood Susceptibility Module
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from datetime import datetime
import os

# -------------------------------------------------
# PATHS
# -------------------------------------------------
base_dir = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"

dem_path = os.path.join(base_dir, r"01_data\processed\dem_conditioned_duwe.tif")
flow_acc_path = os.path.join(base_dir, r"01_data\processed\flow_accumulation_duwe.tif")
catchment_path = os.path.join(base_dir, r"01_data\processed\catchment_duwe.tif")

slope_path = os.path.join(base_dir, r"01_data\processed\slope_duwe.tif")
twi_path = os.path.join(base_dir, r"01_data\processed\twi_duwe.tif")
flood_proxy_path = os.path.join(base_dir, r"01_data\processed\flood_proxy_duwe.tif")
figure_path = os.path.join(base_dir, r"03_outputs\figures\lesson_08_twi_flood_proxy.png")

print("=" * 70)
print("DUWE DISASTER INTELLIGENCE")
print("DUWE AI FLOOD INTELLIGENCE ENGINE")
print("=" * 70)
print("LESSON 8: TOPOGRAPHIC WETNESS INDEX & FLOOD PROXY (Corrected)")
print("Module: Flood Susceptibility Module")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------
print("\n[1] Loading rasters...")

with rasterio.open(dem_path) as src:
    dem = src.read(1).astype("float64")
    profile = src.profile.copy()
    transform = src.transform
    cellsize = abs(transform[0])      # 30 m

with rasterio.open(flow_acc_path) as src:
    acc = src.read(1).astype("float64")

with rasterio.open(catchment_path) as src:
    catchment = src.read(1)

print("    Rasters loaded successfully.")

# -------------------------------------------------
# 2. CREATE MASK
# -------------------------------------------------
print("\n[2] Creating catchment mask...")

mask = (catchment == 1) & (dem > -1000)
dem_masked = np.where(mask, dem, np.nan)
acc_masked = np.where(mask, acc, np.nan)

print(f"    Valid catchment pixels : {np.sum(mask):,}")

# -------------------------------------------------
# 3. CALCULATE SLOPE (using NumPy gradient)
# -------------------------------------------------
print("\n[3] Calculating Slope...")

# Calculate gradients
dy, dx = np.gradient(dem_masked, cellsize)

# Slope in radians then convert to degrees
slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
slope_deg = np.degrees(slope_rad)

# Avoid zero slope for TWI calculation
slope_rad_safe = np.where(slope_rad < 0.001, 0.001, slope_rad)

print("    Slope calculated successfully.")

# -------------------------------------------------
# 4. CALCULATE TOPOGRAPHIC WETNESS INDEX (TWI)
# -------------------------------------------------
print("\n[4] Calculating Topographic Wetness Index (TWI)...")

# Specific catchment area
specific_catchment_area = (acc_masked + 1) * cellsize

twi = np.log(specific_catchment_area / np.tan(slope_rad_safe))
twi = np.where(mask, twi, np.nan)

print(f"    TWI range : {np.nanmin(twi):.2f} to {np.nanmax(twi):.2f}")

# -------------------------------------------------
# 5. CREATE FLOOD PROXY
# -------------------------------------------------
print("\n[5] Creating Flood Proxy map...")

def normalize(arr):
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    return (arr - arr_min) / (arr_max - arr_min + 1e-6)

norm_acc = normalize(np.log1p(acc_masked))
norm_twi = normalize(twi)
norm_slope = 1 - normalize(slope_deg)     # Low slope = higher susceptibility

flood_proxy = (0.45 * norm_acc) + (0.40 * norm_twi) + (0.15 * norm_slope)
flood_proxy = np.where(mask, flood_proxy, np.nan)

print("    Flood Proxy created.")

# -------------------------------------------------
# 6. SAVE RESULTS
# -------------------------------------------------
print("\n[6] Saving results...")

profile.update(dtype="float32", nodata=-9999.0, compress="lzw")

# Slope
slope_save = np.where(mask, slope_deg, -9999.0)
with rasterio.open(slope_path, "w", **profile) as dst:
    dst.write(slope_save.astype("float32"), 1)

# TWI
twi_save = np.where(mask, twi, -9999.0)
with rasterio.open(twi_path, "w", **profile) as dst:
    dst.write(twi_save.astype("float32"), 1)

# Flood Proxy
proxy_save = np.where(mask, flood_proxy, -9999.0)
with rasterio.open(flood_proxy_path, "w", **profile) as dst:
    dst.write(proxy_save.astype("float32"), 1)

print(f"    Slope saved to       : {slope_path}")
print(f"    TWI saved to         : {twi_path}")
print(f"    Flood Proxy saved to : {flood_proxy_path}")

# -------------------------------------------------
# 7. VISUALISATION
# -------------------------------------------------
print("\n[7] Generating maps...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

im1 = axes[0, 0].imshow(np.where(mask, slope_deg, np.nan), cmap="YlOrBr")
axes[0, 0].set_title("Slope (degrees)", fontsize=13, fontweight="bold")
plt.colorbar(im1, ax=axes[0, 0], shrink=0.7)

im2 = axes[0, 1].imshow(twi, cmap="Blues")
axes[0, 1].set_title("Topographic Wetness Index (TWI)", fontsize=13, fontweight="bold")
plt.colorbar(im2, ax=axes[0, 1], shrink=0.7)

im3 = axes[1, 0].imshow(flood_proxy, cmap="RdYlBu_r")
axes[1, 0].set_title("Flood Proxy / Susceptibility", fontsize=13, fontweight="bold")
plt.colorbar(im3, ax=axes[1, 0], shrink=0.7)

im4 = axes[1, 1].imshow(np.log1p(acc_masked), cmap="Blues")
axes[1, 1].set_title("Flow Accumulation (Log)", fontsize=13, fontweight="bold")
plt.colorbar(im4, ax=axes[1, 1], shrink=0.7)

for ax in axes.flat:
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

fig.suptitle("DUWE AI FLOOD INTELLIGENCE ENGINE\nLesson 8 – TWI & Flood Proxy Mapping",
             fontsize=16, fontweight="bold", y=0.98)

plt.figtext(0.5, 0.01, "DUWE DISASTER INTELLIGENCE  |  Flood Susceptibility Module",
            ha="center", fontsize=10, style="italic", color="gray")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n    Figure saved to:\n    {figure_path}")

print("\n" + "=" * 70)
print("LESSON 8 COMPLETED SUCCESSFULLY")
print("Slope, TWI and Flood Proxy maps have been generated")
print("=" * 70)
