"""
DUWE DISASTER INTELLIGENCE
DUWE AI FLOOD INTELLIGENCE ENGINE

Lesson 14: Classify Rainfall-Enhanced Hazard + Update Risk
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

rain_hazard_path = os.path.join(base_dir, r"01_data\processed\rainfall_enhanced_hazard_duwe.tif")
exposure_path = os.path.join(base_dir, r"01_data\processed\exposure_proxy_duwe.tif")
catchment_path = os.path.join(base_dir, r"01_data\processed\catchment_duwe.tif")

rain_hazard_class_path = os.path.join(base_dir, r"01_data\processed\rainfall_hazard_classified_duwe.tif")
updated_risk_path = os.path.join(base_dir, r"01_data\processed\risk_updated_duwe.tif")
updated_risk_class_path = os.path.join(base_dir, r"01_data\processed\risk_updated_classified_duwe.tif")
figure_path = os.path.join(base_dir, r"03_outputs\figures\lesson_14_rain_hazard_risk.png")

print("=" * 70)
print("DUWE DISASTER INTELLIGENCE")
print("DUWE AI FLOOD INTELLIGENCE ENGINE")
print("=" * 70)
print("LESSON 14: RAINFALL-ENHANCED HAZARD CLASSIFICATION + RISK UPDATE")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------
print("\n[1] Loading layers...")

with rasterio.open(rain_hazard_path) as src:
    rain_hazard = src.read(1).astype("float64")
    profile = src.profile.copy()

with rasterio.open(exposure_path) as src:
    exposure = src.read(1).astype("float64")

with rasterio.open(catchment_path) as src:
    catchment = src.read(1)

mask = (catchment == 1) & (rain_hazard > -1000)
print(f"    Valid pixels : {np.sum(mask):,}")

# -------------------------------------------------
# 2. CLASSIFY RAINFALL-ENHANCED HAZARD
# -------------------------------------------------
print("\n[2] Classifying Rainfall-Enhanced Hazard...")

p30 = np.nanpercentile(rain_hazard[mask], 30)
p60 = np.nanpercentile(rain_hazard[mask], 60)
p85 = np.nanpercentile(rain_hazard[mask], 85)

print(f"    Thresholds → Low/Med: {p30:.3f} | Med/High: {p60:.3f} | High/V.High: {p85:.3f}")

hazard_class = np.zeros_like(rain_hazard)
hazard_class[rain_hazard <= p30] = 1
hazard_class[(rain_hazard > p30) & (rain_hazard <= p60)] = 2
hazard_class[(rain_hazard > p60) & (rain_hazard <= p85)] = 3
hazard_class[rain_hazard > p85] = 4
hazard_class = np.where(mask, hazard_class, np.nan)

# -------------------------------------------------
# 3. UPDATE RISK = Hazard × Exposure
# -------------------------------------------------
print("\n[3] Updating Risk layer (Hazard × Exposure)...")

def normalize(arr):
    arr = np.where(mask, arr, np.nan)
    return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + 1e-6)

norm_hazard = normalize(rain_hazard)
norm_exposure = normalize(exposure)

risk = norm_hazard * norm_exposure
risk = np.where(mask, risk, np.nan)

# Classify updated risk
r30 = np.nanpercentile(risk[mask], 30)
r60 = np.nanpercentile(risk[mask], 60)
r85 = np.nanpercentile(risk[mask], 85)

risk_class = np.zeros_like(risk)
risk_class[risk <= r30] = 1
risk_class[(risk > r30) & (risk <= r60)] = 2
risk_class[(risk > r60) & (risk <= r85)] = 3
risk_class[risk > r85] = 4
risk_class = np.where(mask, risk_class, np.nan)

# -------------------------------------------------
# 4. AREA STATISTICS
# -------------------------------------------------
print("\n[4] AREA STATISTICS")
print("-" * 60)

pixel_area = (30 * 30) / 1_000_000
total_area = np.sum(mask) * pixel_area

labels = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}

print("\nRainfall-Enhanced Hazard Classification:")
print(f"{'Class':<12} {'Area (km²)':>12} {'Percentage':>12}")
print("-" * 40)
for cls, name in labels.items():
    area = np.sum(hazard_class == cls) * pixel_area
    pct = (area / total_area) * 100
    print(f"{name:<12} {area:>12.1f} {pct:>11.1f}%")

print("\nUpdated Risk Classification (Hazard × Exposure):")
print(f"{'Class':<12} {'Area (km²)':>12} {'Percentage':>12}")
print("-" * 40)
for cls, name in labels.items():
    area = np.sum(risk_class == cls) * pixel_area
    pct = (area / total_area) * 100
    print(f"{name:<12} {area:>12.1f} {pct:>11.1f}%")

# -------------------------------------------------
# 5. SAVE RESULTS
# -------------------------------------------------
print("\n[5] Saving results...")

profile.update(dtype="float32", nodata=-9999.0, compress="lzw")

for path, arr in [
    (rain_hazard_class_path, hazard_class),
    (updated_risk_path, risk),
    (updated_risk_class_path, risk_class)
]:
    save_arr = np.where(mask, arr, -9999.0)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(save_arr.astype("float32"), 1)

print("    All outputs saved successfully.")

# -------------------------------------------------
# 6. VISUALISATION
# -------------------------------------------------
print("\n[6] Generating maps...")

cmap = ListedColormap(["#2ecc71", "#f1c40f", "#e67e22", "#c0392b"])

fig, axes = plt.subplots(2, 2, figsize=(15, 13))

im0 = axes[0, 0].imshow(np.where(mask, rain_hazard, np.nan), cmap="RdYlBu_r")
axes[0, 0].set_title("Rainfall-Enhanced Hazard", fontsize=12, fontweight="bold")
plt.colorbar(im0, ax=axes[0, 0], shrink=0.7)

im1 = axes[0, 1].imshow(hazard_class, cmap=cmap, vmin=1, vmax=4)
axes[0, 1].set_title("Classified Rainfall Hazard", fontsize=12, fontweight="bold")
plt.colorbar(im1, ax=axes[0, 1], shrink=0.7, ticks=[1,2,3,4])

im2 = axes[1, 0].imshow(np.where(mask, risk, np.nan), cmap="RdYlBu_r")
axes[1, 0].set_title("Updated Risk (Hazard × Exposure)", fontsize=12, fontweight="bold")
plt.colorbar(im2, ax=axes[1, 0], shrink=0.7)

im3 = axes[1, 1].imshow(risk_class, cmap=cmap, vmin=1, vmax=4)
axes[1, 1].set_title("Classified Updated Risk", fontsize=12, fontweight="bold")
cbar = plt.colorbar(im3, ax=axes[1, 1], shrink=0.7, ticks=[1,2,3,4])
cbar.ax.set_yticklabels(["Low", "Medium", "High", "Very High"])

for ax in axes.flat:
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

fig.suptitle("DUWE AI FLOOD INTELLIGENCE ENGINE\nLesson 14 – Rainfall Hazard Classification & Updated Risk",
             fontsize=14, fontweight="bold", y=0.98)

plt.figtext(0.5, 0.01, "DUWE DISASTER INTELLIGENCE  |  Hydro-Meteorological + Exposure Modules",
            ha="center", fontsize=9, style="italic", color="gray")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n    Figure saved to:\n    {figure_path}")

print("\n" + "=" * 70)
print("LESSON 14 COMPLETED SUCCESSFULLY")
print("=" * 70)
