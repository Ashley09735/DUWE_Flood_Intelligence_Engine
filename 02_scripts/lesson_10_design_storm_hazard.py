"""
DUWE DISASTER INTELLIGENCE
DUWE AI FLOOD INTELLIGENCE ENGINE

Lesson 10: Design Storm Runoff & Enhanced Flood Hazard
Module: Hydro-Meteorological Intelligence Module
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

flood_proxy_path = os.path.join(base_dir, r"01_data\processed\flood_proxy_duwe.tif")
flow_acc_path = os.path.join(base_dir, r"01_data\processed\flow_accumulation_duwe.tif")
catchment_path = os.path.join(base_dir, r"01_data\processed\catchment_duwe.tif")

hazard_50_path = os.path.join(base_dir, r"01_data\processed\hazard_50mm_duwe.tif")
hazard_100_path = os.path.join(base_dir, r"01_data\processed\hazard_100mm_duwe.tif")
hazard_150_path = os.path.join(base_dir, r"01_data\processed\hazard_150mm_duwe.tif")
figure_path = os.path.join(base_dir, r"03_outputs\figures\lesson_10_design_storm_hazard.png")

print("=" * 70)
print("DUWE DISASTER INTELLIGENCE")
print("DUWE AI FLOOD INTELLIGENCE ENGINE")
print("=" * 70)
print("LESSON 10: DESIGN STORM RUNOFF & ENHANCED FLOOD HAZARD")
print("Module: Hydro-Meteorological Intelligence Module")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

# -------------------------------------------------
# DESIGN STORMS (mm in 24 hours)
# -------------------------------------------------
design_storms = {
    "50mm": 50.0,
    "100mm": 100.0,
    "150mm": 150.0
}

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------
print("\n[1] Loading base layers...")

with rasterio.open(flood_proxy_path) as src:
    proxy = src.read(1).astype("float64")
    profile = src.profile.copy()

with rasterio.open(flow_acc_path) as src:
    acc = src.read(1).astype("float64")

with rasterio.open(catchment_path) as src:
    catchment = src.read(1)

mask = (catchment == 1) & (proxy > -1000)
proxy_masked = np.where(mask, proxy, np.nan)
acc_masked = np.where(mask, acc, np.nan)

print(f"    Valid catchment pixels : {np.sum(mask):,}")

# -------------------------------------------------
# 2. NORMALISATION HELPER
# -------------------------------------------------
def normalize(arr):
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    return (arr - arr_min) / (arr_max - arr_min + 1e-6)

norm_proxy = normalize(proxy_masked)
norm_acc = normalize(np.log1p(acc_masked))

# -------------------------------------------------
# 3. CALCULATE HAZARD FOR EACH DESIGN STORM
# -------------------------------------------------
print("\n[2] Calculating Rainfall-Enhanced Flood Hazard...")

hazard_results = {}
pixel_area_km2 = (30 * 30) / 1_000_000

for name, rainfall_mm in design_storms.items():
    print(f"    → Processing {name} design storm...")

    # Simple rainfall weighting (higher rainfall increases hazard)
    # We scale rainfall influence between 0.6 and 1.4
    rain_factor = 0.6 + (rainfall_mm / 150.0) * 0.8

    # Enhanced hazard = weighted combination
    hazard = (0.55 * norm_proxy) + (0.30 * norm_acc) + (0.15 * rain_factor)
    hazard = np.where(mask, hazard, np.nan)

    hazard_results[name] = hazard

    # Quick statistics
    mean_haz = np.nanmean(hazard)
    p90 = np.nanpercentile(hazard, 90)
    print(f"       Mean hazard: {mean_haz:.3f} | 90th percentile: {p90:.3f}")

# -------------------------------------------------
# 4. SAVE HAZARD RASTERS
# -------------------------------------------------
print("\n[3] Saving hazard rasters...")

profile.update(dtype="float32", nodata=-9999.0, compress="lzw")

paths = {
    "50mm": hazard_50_path,
    "100mm": hazard_100_path,
    "150mm": hazard_150_path
}

for name, path in paths.items():
    save_arr = np.where(mask, hazard_results[name], -9999.0)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(save_arr.astype("float32"), 1)
    print(f"    Saved: {name} → {path}")

# -------------------------------------------------
# 5. AREA STATISTICS UNDER HIGH HAZARD
# -------------------------------------------------
print("\n[4] HIGH HAZARD AREA COMPARISON (above 90th percentile of 50mm scenario)")
print("-" * 65)

# Use 90th percentile of the 50mm scenario as common threshold
threshold = np.nanpercentile(hazard_results["50mm"], 90)
print(f"Common High-Hazard Threshold: {threshold:.3f}\n")

print(f"{'Scenario':<12} {'High Hazard Area (km²)':>22} {'% of Catchment':>15}")
print("-" * 65)

total_area = np.sum(mask) * pixel_area_km2

for name in ["50mm", "100mm", "150mm"]:
    high_mask = hazard_results[name] >= threshold
    high_area = np.sum(high_mask) * pixel_area_km2
    pct = (high_area / total_area) * 100
    print(f"{name:<12} {high_area:>22.2f} {pct:>14.1f}%")

print("-" * 65)

# -------------------------------------------------
# 6. VISUALISATION
# -------------------------------------------------
print("\n[5] Generating comparison maps...")

fig, axes = plt.subplots(2, 2, figsize=(15, 13))

# 50mm
im0 = axes[0, 0].imshow(hazard_results["50mm"], cmap="RdYlBu_r", vmin=0.2, vmax=0.9)
axes[0, 0].set_title("50 mm Design Storm", fontsize=12, fontweight="bold")
plt.colorbar(im0, ax=axes[0, 0], shrink=0.7)

# 100mm
im1 = axes[0, 1].imshow(hazard_results["100mm"], cmap="RdYlBu_r", vmin=0.2, vmax=0.9)
axes[0, 1].set_title("100 mm Design Storm", fontsize=12, fontweight="bold")
plt.colorbar(im1, ax=axes[0, 1], shrink=0.7)

# 150mm
im2 = axes[1, 0].imshow(hazard_results["150mm"], cmap="RdYlBu_r", vmin=0.2, vmax=0.9)
axes[1, 0].set_title("150 mm Design Storm", fontsize=12, fontweight="bold")
plt.colorbar(im2, ax=axes[1, 0], shrink=0.7)

# Difference 150mm - 50mm
diff = hazard_results["150mm"] - hazard_results["50mm"]
im3 = axes[1, 1].imshow(diff, cmap="YlOrRd", vmin=0, vmax=0.25)
axes[1, 1].set_title("Increase (150mm – 50mm)", fontsize=12, fontweight="bold")
plt.colorbar(im3, ax=axes[1, 1], shrink=0.7)

for ax in axes.flat:
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

fig.suptitle("DUWE AI FLOOD INTELLIGENCE ENGINE\nLesson 10 – Design Storm Enhanced Flood Hazard",
             fontsize=15, fontweight="bold", y=0.98)

plt.figtext(0.5, 0.01, "DUWE DISASTER INTELLIGENCE  |  Hydro-Meteorological Intelligence Module",
            ha="center", fontsize=9, style="italic", color="gray")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n    Figure saved to:\n    {figure_path}")

print("\n" + "=" * 70)
print("LESSON 10 COMPLETED SUCCESSFULLY")
print("Design storm hazard layers have been generated")
print("=" * 70)
