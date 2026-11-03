"""
DUWE DISASTER INTELLIGENCE
DUWE AI FLOOD INTELLIGENCE ENGINE

Lesson 13: Real Rainfall Integration (CHIRPS, newest rainy season available)
Corrected Version - safe log1p handling, nodata handling, automatic file choice
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from datetime import datetime
import os

from chirps_utils import latest_chirps_file   # lives in 02_scripts\chirps_utils.py

# -------------------------------------------------
# PATHS
# -------------------------------------------------
base_dir = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
raw_dir = os.path.join(base_dir, r"01_data\raw")

# Newest clipped CHIRPS season file (chosen by the season in the file name).
# Use clipped=False if you want the un-clipped file instead.
chirps_path = latest_chirps_file(raw_dir, clipped=True)

# Season label for titles, e.g. "2023-2024", taken from the file name
_years = re.findall(r"(?<!\d)(20\d{2})(?!\d)", os.path.basename(chirps_path))
season_label = f"{_years[0]}-{_years[-1]}" if len(_years) >= 2 else "unknown season"

dem_path = os.path.join(base_dir, r"01_data\processed\dem_conditioned_duwe.tif")
catchment_path = os.path.join(base_dir, r"01_data\processed\catchment_duwe.tif")
acc_path = os.path.join(base_dir, r"01_data\processed\flow_accumulation_duwe.tif")
flood_proxy_path = os.path.join(base_dir, r"01_data\processed\flood_proxy_duwe.tif")

rainfall_aligned_path = os.path.join(base_dir, r"01_data\processed\chirps_rainfall_aligned_duwe.tif")
rain_hazard_path = os.path.join(base_dir, r"01_data\processed\rainfall_enhanced_hazard_duwe.tif")
figure_path = os.path.join(base_dir, r"03_outputs\figures\lesson_13_chirps_integration.png")

os.makedirs(os.path.dirname(figure_path), exist_ok=True)

print("=" * 70)
print("DUWE DISASTER INTELLIGENCE")
print("DUWE AI FLOOD INTELLIGENCE ENGINE")
print("=" * 70)
print("LESSON 13: REAL RAINFALL INTEGRATION (CHIRPS) - CORRECTED")
print(f"Rainy Season {season_label}")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

# -------------------------------------------------
# 1. LOAD REFERENCE LAYERS
# -------------------------------------------------
print("\n[1] Loading reference layers...")

with rasterio.open(dem_path) as src:
    dem = src.read(1).astype("float64")
    dem_meta = src.meta.copy()
    dem_transform = src.transform
    dem_crs = src.crs
    dem_shape = src.shape

with rasterio.open(catchment_path) as src:
    catchment = src.read(1)

mask_valid = (catchment == 1) & (dem > -1000)
print(f"    Target shape     : {dem_shape}")
print(f"    Target CRS       : {dem_crs}")
print(f"    Valid pixels     : {np.sum(mask_valid):,}")

# -------------------------------------------------
# 2. LOAD CHIRPS RAINFALL
# -------------------------------------------------
print("\n[2] Loading CHIRPS rainfall...")

with rasterio.open(chirps_path) as src:
    chirps = src.read(1).astype("float64")
    chirps_nodata = src.nodata
    chirps_crs = src.crs
    chirps_transform = src.transform
    chirps_shape = src.shape

# Treat the file's nodata value (and any non-finite value) as missing, not as rainfall
if chirps_nodata is not None:
    chirps[chirps == chirps_nodata] = np.nan
chirps[~np.isfinite(chirps)] = np.nan

print(f"    CHIRPS file      : {os.path.basename(chirps_path)}")
print(f"    CHIRPS shape     : {chirps_shape}")
print(f"    CHIRPS CRS       : {chirps_crs}")
print(f"    Nodata value     : {chirps_nodata}")
print(f"    Exact-zero cells : {np.sum(chirps == 0):,} of {chirps.size:,}")
print(f"    Rainfall range   : {np.nanmin(chirps):.1f} - {np.nanmax(chirps):.1f} mm")

# -------------------------------------------------
# 3. REPROJECT + RESAMPLE CHIRPS TO MATCH DEM
# -------------------------------------------------
print("\n[3] Reprojecting and resampling CHIRPS to 30 m DEM grid...")

rainfall_aligned = np.full(dem_shape, np.nan, dtype="float64")

reproject(
    source=chirps,
    destination=rainfall_aligned,
    src_transform=chirps_transform,
    src_crs=chirps_crs,
    src_nodata=np.nan,
    dst_transform=dem_transform,
    dst_crs=dem_crs,
    dst_nodata=np.nan,
    resampling=Resampling.bilinear
)

rainfall_aligned = np.where(mask_valid, rainfall_aligned, np.nan)

n_zero = int(np.sum(rainfall_aligned[mask_valid] == 0))
n_missing = int(np.sum(np.isnan(rainfall_aligned[mask_valid])))
print(f"    Aligned rainfall range : {np.nanmin(rainfall_aligned):.1f} - {np.nanmax(rainfall_aligned):.1f} mm")
print(f"    Mean rainfall          : {np.nanmean(rainfall_aligned):.1f} mm")
print(f"    Pixels with exactly 0 mm        : {n_zero:,}")
print(f"    Valid pixels with no rainfall data: {n_missing:,}")
if n_missing > 0:
    print("    WARNING: some catchment pixels are not covered by the CHIRPS file.")

# -------------------------------------------------
# 4. SAVE ALIGNED RAINFALL
# -------------------------------------------------
print("\n[4] Saving aligned rainfall raster...")

dem_meta.update(dtype="float32", nodata=-9999.0, compress="lzw")

rain_save = np.where(mask_valid & np.isfinite(rainfall_aligned), rainfall_aligned, -9999.0)
with rasterio.open(rainfall_aligned_path, "w", **dem_meta) as dst:
    dst.write(rain_save.astype("float32"), 1)

print(f"    Saved to: {rainfall_aligned_path}")

# -------------------------------------------------
# 5. CREATE RAINFALL-ENHANCED HAZARD
# -------------------------------------------------
print("\n[5] Creating Rainfall-Enhanced Hazard...")

with rasterio.open(acc_path) as src:
    acc = src.read(1).astype("float64")

with rasterio.open(flood_proxy_path) as src:
    proxy = src.read(1).astype("float64")


def normalize(arr):
    arr = np.where(mask_valid, arr, np.nan)
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    return (arr - amin) / (amax - amin + 1e-6)


norm_rain = normalize(rainfall_aligned)
norm_proxy = normalize(proxy)

# ----- SAFE handling of accumulation (fixes the warning) -----
acc_safe = np.where(mask_valid, acc, np.nan)
acc_safe = np.where(acc_safe < 0, 0, acc_safe)          # no negative values
norm_acc = normalize(np.log1p(acc_safe))
# ---------------------------------------------------------------

# Weighted combination
rain_hazard = (0.40 * norm_rain) + (0.35 * norm_proxy) + (0.25 * norm_acc)
rain_hazard = np.where(mask_valid, rain_hazard, np.nan)

print(f"    Rainfall-Enhanced Hazard range: {np.nanmin(rain_hazard):.3f} - {np.nanmax(rain_hazard):.3f}")

# -------------------------------------------------
# 6. SAVE RAINFALL-ENHANCED HAZARD
# -------------------------------------------------
print("\n[6] Saving Rainfall-Enhanced Hazard...")

hazard_save = np.where(mask_valid & np.isfinite(rain_hazard), rain_hazard, -9999.0)
with rasterio.open(rain_hazard_path, "w", **dem_meta) as dst:
    dst.write(hazard_save.astype("float32"), 1)

print(f"    Saved to: {rain_hazard_path}")

# -------------------------------------------------
# 7. STATISTICS
# -------------------------------------------------
print(f"\n[7] RAINFALL STATISTICS (Rainy Season {season_label})")
print("-" * 50)
print(f"    Minimum rainfall     : {np.nanmin(rainfall_aligned):.1f} mm")
print(f"    Maximum rainfall     : {np.nanmax(rainfall_aligned):.1f} mm")
print(f"    Mean rainfall        : {np.nanmean(rainfall_aligned):.1f} mm")
print(f"    Median rainfall      : {np.nanmedian(rainfall_aligned):.1f} mm")

p80 = np.nanpercentile(rainfall_aligned, 80)
high_rain_area = np.sum(rainfall_aligned >= p80) * (30 * 30) / 1e6
print(f"\n    Area with rainfall >= {p80:.0f} mm (80th percentile): {high_rain_area:.1f} km2")

# -------------------------------------------------
# 8. VISUALISATION
# -------------------------------------------------
print("\n[8] Generating maps...")

fig, axes = plt.subplots(2, 2, figsize=(15, 13))

im0 = axes[0, 0].imshow(np.where(mask_valid, rainfall_aligned, np.nan), cmap="Blues")
axes[0, 0].set_title(f"CHIRPS Rainfall (Rainy Season {season_label})", fontsize=12, fontweight="bold")
plt.colorbar(im0, ax=axes[0, 0], shrink=0.7, label="Rainfall (mm)")

im1 = axes[0, 1].imshow(np.where(mask_valid, proxy, np.nan), cmap="YlOrRd")
axes[0, 1].set_title("Original Flood Proxy", fontsize=12, fontweight="bold")
plt.colorbar(im1, ax=axes[0, 1], shrink=0.7)

im2 = axes[1, 0].imshow(np.where(mask_valid, rain_hazard, np.nan), cmap="RdYlBu_r")
axes[1, 0].set_title("Rainfall-Enhanced Hazard", fontsize=12, fontweight="bold")
plt.colorbar(im2, ax=axes[1, 0], shrink=0.7)

# Stretch the colours between the 2nd and 98th percentile so channels are visible
log_acc = np.log1p(np.where(mask_valid, acc_safe, np.nan))
acc_vmin, acc_vmax = np.nanpercentile(log_acc, [2, 98])
im3 = axes[1, 1].imshow(log_acc, cmap="Blues", vmin=acc_vmin, vmax=acc_vmax)
axes[1, 1].set_title("Flow Accumulation (Log)", fontsize=12, fontweight="bold")
plt.colorbar(im3, ax=axes[1, 1], shrink=0.7)

for ax in axes.flat:
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

fig.suptitle("DUWE AI FLOOD INTELLIGENCE ENGINE\nLesson 13 - CHIRPS Rainfall Integration (Corrected)",
             fontsize=14, fontweight="bold", y=0.98)

plt.figtext(0.5, 0.01, "DUWE DISASTER INTELLIGENCE  |  Hydro-Meteorological Intelligence Module",
            ha="center", fontsize=9, style="italic", color="gray")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"\n    Figure saved to:\n    {figure_path}")

print("\n" + "=" * 70)
print("LESSON 13 COMPLETED SUCCESSFULLY")
print(f"Real CHIRPS rainfall ({season_label}) has been integrated")
print("=" * 70)
