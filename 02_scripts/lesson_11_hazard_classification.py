"""
DUWE DISASTER INTELLIGENCE
DUWE AI FLOOD INTELLIGENCE ENGINE

Lesson 11: Design Storm Hazard Classification (Corrected - Fixed Thresholds)
Module: Hydro-Meteorological Intelligence Module
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import ListedColormap
from datetime import datetime
import os

# -------------------------------------------------
# PATHS
# -------------------------------------------------
base_dir = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"

hazard_files = {
    "50mm":  os.path.join(base_dir, r"01_data\processed\hazard_50mm_duwe.tif"),
    "100mm": os.path.join(base_dir, r"01_data\processed\hazard_100mm_duwe.tif"),
    "150mm": os.path.join(base_dir, r"01_data\processed\hazard_150mm_duwe.tif"),
}

catchment_path = os.path.join(base_dir, r"01_data\processed\catchment_duwe.tif")
figure_path = os.path.join(base_dir, r"03_outputs\figures\lesson_11_hazard_classification.png")

output_classified = {
    "50mm":  os.path.join(base_dir, r"01_data\processed\hazard_50mm_classified_duwe.tif"),
    "100mm": os.path.join(base_dir, r"01_data\processed\hazard_100mm_classified_duwe.tif"),
    "150mm": os.path.join(base_dir, r"01_data\processed\hazard_150mm_classified_duwe.tif"),
}

print("=" * 70)
print("DUWE DISASTER INTELLIGENCE")
print("DUWE AI FLOOD INTELLIGENCE ENGINE")
print("=" * 70)
print("LESSON 11: DESIGN STORM HAZARD CLASSIFICATION (Corrected)")
print("Module: Hydro-Meteorological Intelligence Module")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("Using FIXED thresholds from 50mm scenario")
print("=" * 70)

# -------------------------------------------------
# 1. LOAD CATCHMENT + 50mm HAZARD (to set thresholds)
# -------------------------------------------------
print("\n[1] Loading data and calculating fixed thresholds...")

with rasterio.open(catchment_path) as src:
    catchment = src.read(1)
    profile = src.profile.copy()

mask = catchment == 1
print(f"    Valid catchment pixels : {np.sum(mask):,}")

# Load 50mm hazard to define fixed thresholds
with rasterio.open(hazard_files["50mm"]) as src:
    hazard_50 = src.read(1).astype("float64")

hazard_50_masked = np.where(mask, hazard_50, np.nan)

# Fixed thresholds based on 50mm scenario
t_low_med = np.nanpercentile(hazard_50_masked, 30)
t_med_high = np.nanpercentile(hazard_50_masked, 60)
t_high_vhigh = np.nanpercentile(hazard_50_masked, 85)

print(f"    Fixed Thresholds (from 50mm):")
print(f"      Low / Medium     : {t_low_med:.4f}")
print(f"      Medium / High    : {t_med_high:.4f}")
print(f"      High / Very High : {t_high_vhigh:.4f}")

# -------------------------------------------------
# 2. CLASSIFY ALL SCENARIOS WITH THE SAME THRESHOLDS
# -------------------------------------------------
print("\n[2] Classifying all scenarios with fixed thresholds...")

pixel_area_km2 = (30 * 30) / 1_000_000
total_area = np.sum(mask) * pixel_area_km2

classified_maps = {}
stats = {}

for name, path in hazard_files.items():
    print(f"    → Processing {name}...")

    with rasterio.open(path) as src:
        hazard = src.read(1).astype("float64")

    hazard_masked = np.where(mask, hazard, np.nan)

    classified = np.zeros_like(hazard_masked)
    classified[hazard_masked <= t_low_med] = 1
    classified[(hazard_masked > t_low_med) & (hazard_masked <= t_med_high)] = 2
    classified[(hazard_masked > t_med_high) & (hazard_masked <= t_high_vhigh)] = 3
    classified[hazard_masked > t_high_vhigh] = 4

    classified = np.where(mask, classified, np.nan)
    classified_maps[name] = classified

    # Statistics
    stats[name] = {}
    for cls, label in [(1, "Low"), (2, "Medium"), (3, "High"), (4, "Very High")]:
        pixels = np.sum(classified == cls)
        area = pixels * pixel_area_km2
        pct = (area / total_area) * 100
        stats[name][label] = {"area": area, "pct": pct}

# -------------------------------------------------
# 3. PRINT COMPARISON TABLE
# -------------------------------------------------
print("\n[3] FLOOD HAZARD AREA COMPARISON BY SCENARIO (Fixed Thresholds)")
print("=" * 78)
print(f"{'Risk Class':<12} {'50 mm':>20} {'100 mm':>20} {'150 mm':>20}")
print(f"{'':<12} {'km²       %':>20} {'km²       %':>20} {'km²       %':>20}")
print("-" * 78)

for label in ["Low", "Medium", "High", "Very High"]:
    row = f"{label:<12}"
    for name in ["50mm", "100mm", "150mm"]:
        area = stats[name][label]["area"]
        pct = stats[name][label]["pct"]
        row += f" {area:8.1f} {pct:6.1f}%"
    print(row)

print("-" * 78)
print(f"{'Total':<12} {total_area:8.1f}  100.0% {total_area:8.1f}  100.0% {total_area:8.1f}  100.0%")
print("=" * 78)

# High + Very High summary
print("\n[4] HIGH + VERY HIGH RISK SUMMARY")
print("-" * 55)
for name in ["50mm", "100mm", "150mm"]:
    high_area = stats[name]["High"]["area"] + stats[name]["Very High"]["area"]
    high_pct = stats[name]["High"]["pct"] + stats[name]["Very High"]["pct"]
    print(f"    {name:<8} → {high_area:7.1f} km²   ({high_pct:5.1f}%)")

# -------------------------------------------------
# 5. SAVE CLASSIFIED RASTERS
# -------------------------------------------------
print("\n[5] Saving classified rasters...")

profile.update(dtype="float32", nodata=-9999.0, compress="lzw")

for name, classified in classified_maps.items():
    save_arr = np.where(mask, classified, -9999.0)
    with rasterio.open(output_classified[name], "w", **profile) as dst:
        dst.write(save_arr.astype("float32"), 1)
    print(f"    Saved: {name}")

# -------------------------------------------------
# 6. VISUALISATION
# -------------------------------------------------
print("\n[6] Generating classification maps...")

cmap = ListedColormap(["#2ecc71", "#f1c40f", "#e67e22", "#c0392b"])
labels = ["Low", "Medium", "High", "Very High"]

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

for ax, name in zip(axes, ["50mm", "100mm", "150mm"]):
    im = ax.imshow(classified_maps[name], cmap=cmap, vmin=1, vmax=4)
    ax.set_title(f"{name} Design Storm", fontsize=13, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
cbar = fig.colorbar(im, cax=cbar_ax, ticks=[1, 2, 3, 4])
cbar.ax.set_yticklabels(labels)
cbar.set_label("Flood Hazard Class")

fig.suptitle("DUWE AI FLOOD INTELLIGENCE ENGINE\nLesson 11 – Design Storm Hazard Classification (Fixed Thresholds)",
             fontsize=14, fontweight="bold", y=1.02)

plt.figtext(0.5, 0.02, "DUWE DISASTER INTELLIGENCE  |  Hydro-Meteorological Intelligence Module",
            ha="center", fontsize=9, style="italic", color="gray")

plt.tight_layout(rect=[0, 0.05, 0.90, 0.95])
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n    Figure saved to:\n    {figure_path}")

print("\n" + "=" * 70)
print("LESSON 11 COMPLETED SUCCESSFULLY")
print("Fixed thresholds applied – hazard expansion with rainfall is now visible")
print("=" * 70)
