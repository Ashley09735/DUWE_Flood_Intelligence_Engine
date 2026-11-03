"""
DUWE DISASTER INTELLIGENCE
DUWE AI FLOOD INTELLIGENCE ENGINE

Lesson 12: Exposure Proxy & Preliminary Risk Mapping
Module: Exposure & Vulnerability Module (Level 1)
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

dem_path = os.path.join(base_dir, r"01_data\processed\dem_conditioned_duwe.tif")
slope_path = os.path.join(base_dir, r"01_data\processed\slope_duwe.tif")
acc_path = os.path.join(base_dir, r"01_data\processed\flow_accumulation_duwe.tif")
stream_path = os.path.join(base_dir, r"01_data\processed\stream_network_duwe.tif")
hazard_path = os.path.join(base_dir, r"01_data\processed\hazard_100mm_duwe.tif")  # using 100mm as reference
catchment_path = os.path.join(base_dir, r"01_data\processed\catchment_duwe.tif")

exposure_path = os.path.join(base_dir, r"01_data\processed\exposure_proxy_duwe.tif")
risk_path = os.path.join(base_dir, r"01_data\processed\risk_proxy_duwe.tif")
risk_class_path = os.path.join(base_dir, r"01_data\processed\risk_proxy_classified_duwe.tif")
figure_path = os.path.join(base_dir, r"03_outputs\figures\lesson_12_exposure_risk.png")

print("=" * 70)
print("DUWE DISASTER INTELLIGENCE")
print("DUWE AI FLOOD INTELLIGENCE ENGINE")
print("=" * 70)
print("LESSON 12: EXPOSURE PROXY & PRELIMINARY RISK MAPPING")
print("Module: Exposure & Vulnerability Module (Level 1)")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------
print("\n[1] Loading required layers...")

with rasterio.open(dem_path) as src:
    dem = src.read(1).astype("float64")
    profile = src.profile.copy()
    cellsize = abs(src.transform[0])

with rasterio.open(slope_path) as src:
    slope = src.read(1).astype("float64")

with rasterio.open(acc_path) as src:
    acc = src.read(1).astype("float64")

with rasterio.open(stream_path) as src:
    streams = src.read(1)

with rasterio.open(hazard_path) as src:
    hazard = src.read(1).astype("float64")

with rasterio.open(catchment_path) as src:
    catchment = src.read(1)

mask = (catchment == 1) & (dem > -1000)
print(f"    Valid catchment pixels : {np.sum(mask):,}")

# -------------------------------------------------
# 2. BUILD EXPOSURE PROXY
# -------------------------------------------------
print("\n[2] Building Exposure Proxy...")

# Components of Exposure Proxy (higher value = higher potential exposure):
# 1. Low elevation (people & assets tend to concentrate in valleys)
# 2. Proximity to streams (settlements often near water)
# 3. Gentle slopes (easier for settlement and agriculture)
# 4. Higher flow accumulation (indicates larger valleys / floodplains)

def normalize(arr, invert=False):
    arr = np.where(mask, arr, np.nan)
    amin, amax = np.nanmin(arr), np.nanmax(arr)
    norm = (arr - amin) / (amax - amin + 1e-6)
    if invert:
        norm = 1 - norm
    return norm

# Elevation – lower = higher exposure
norm_elev = normalize(dem, invert=True)

# Slope – gentler = higher exposure
norm_slope = normalize(slope, invert=True)

# Flow accumulation – higher = higher exposure potential
norm_acc = normalize(np.log1p(np.where(mask, acc, np.nan)))

# Distance to stream (approximate)
# Simple approach: cells on or near streams get higher value
stream_prox = np.where(streams == 1, 1.0, 0.3)   # basic proximity weight
stream_prox = np.where(mask, stream_prox, np.nan)

# Weighted Exposure Proxy
exposure = (
    0.30 * norm_elev +
    0.25 * norm_slope +
    0.25 * norm_acc +
    0.20 * stream_prox
)
exposure = np.where(mask, exposure, np.nan)

print(f"    Exposure range : {np.nanmin(exposure):.3f} – {np.nanmax(exposure):.3f}")

# -------------------------------------------------
# 3. CALCULATE PRELIMINARY RISK PROXY
# -------------------------------------------------
print("\n[3] Calculating preliminary Risk Proxy (Hazard × Exposure)...")

# Normalize hazard
norm_hazard = normalize(hazard)

# Risk = Hazard × Exposure (classic simple form)
risk = norm_hazard * exposure
risk = np.where(mask, risk, np.nan)

print(f"    Risk range : {np.nanmin(risk):.3f} – {np.nanmax(risk):.3f}")

# -------------------------------------------------
# 4. CLASSIFY RISK
# -------------------------------------------------
print("\n[4] Classifying Risk Proxy...")

p30 = np.nanpercentile(risk, 30)
p60 = np.nanpercentile(risk, 60)
p85 = np.nanpercentile(risk, 85)

risk_class = np.zeros_like(risk)
risk_class[risk <= p30] = 1
risk_class[(risk > p30) & (risk <= p60)] = 2
risk_class[(risk > p60) & (risk <= p85)] = 3
risk_class[risk > p85] = 4
risk_class = np.where(mask, risk_class, np.nan)

# -------------------------------------------------
# 5. AREA STATISTICS
# -------------------------------------------------
print("\n[5] RISK PROXY AREA STATISTICS")
print("-" * 55)

pixel_area_km2 = (cellsize * cellsize) / 1_000_000
total_area = np.sum(mask) * pixel_area_km2

labels = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}

print(f"{'Class':<12} {'Area (km²)':>12} {'Percentage':>12}")
print("-" * 55)

for cls, name in labels.items():
    area = np.sum(risk_class == cls) * pixel_area_km2
    pct = (area / total_area) * 100
    print(f"{name:<12} {area:>12.1f} {pct:>11.1f}%")

print("-" * 55)
print(f"{'Total':<12} {total_area:>12.1f} {100.0:>11.1f}%")

# -------------------------------------------------
# 6. SAVE RESULTS
# -------------------------------------------------
print("\n[6] Saving results...")

profile.update(dtype="float32", nodata=-9999.0, compress="lzw")

for path, arr in [
    (exposure_path, exposure),
    (risk_path, risk),
    (risk_class_path, risk_class)
]:
    save_arr = np.where(mask, arr, -9999.0)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(save_arr.astype("float32"), 1)

print(f"    Exposure Proxy saved")
print(f"    Risk Proxy saved")
print(f"    Classified Risk saved")

# -------------------------------------------------
# 7. VISUALISATION
# -------------------------------------------------
print("\n[7] Generating maps...")

fig, axes = plt.subplots(2, 2, figsize=(15, 13))

im0 = axes[0, 0].imshow(np.where(mask, exposure, np.nan), cmap="YlOrRd")
axes[0, 0].set_title("Exposure Proxy", fontsize=12, fontweight="bold")
plt.colorbar(im0, ax=axes[0, 0], shrink=0.7)

im1 = axes[0, 1].imshow(np.where(mask, hazard, np.nan), cmap="Blues")
axes[0, 1].set_title("Flood Hazard (100 mm)", fontsize=12, fontweight="bold")
plt.colorbar(im1, ax=axes[0, 1], shrink=0.7)

im2 = axes[1, 0].imshow(np.where(mask, risk, np.nan), cmap="RdYlBu_r")
axes[1, 0].set_title("Risk Proxy (Hazard × Exposure)", fontsize=12, fontweight="bold")
plt.colorbar(im2, ax=axes[1, 0], shrink=0.7)

cmap = ListedColormap(["#2ecc71", "#f1c40f", "#e67e22", "#c0392b"])
im3 = axes[1, 1].imshow(risk_class, cmap=cmap, vmin=1, vmax=4)
axes[1, 1].set_title("Classified Risk", fontsize=12, fontweight="bold")
cbar = plt.colorbar(im3, ax=axes[1, 1], shrink=0.7, ticks=[1, 2, 3, 4])
cbar.ax.set_yticklabels(["Low", "Medium", "High", "Very High"])

for ax in axes.flat:
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

fig.suptitle("DUWE AI FLOOD INTELLIGENCE ENGINE\nLesson 12 – Exposure Proxy & Preliminary Risk",
             fontsize=15, fontweight="bold", y=0.98)

plt.figtext(0.5, 0.01, "DUWE DISASTER INTELLIGENCE  |  Exposure & Vulnerability Module (Level 1)",
            ha="center", fontsize=9, style="italic", color="gray")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n    Figure saved to:\n    {figure_path}")

print("\n" + "=" * 70)
print("LESSON 12 COMPLETED SUCCESSFULLY")
print("Exposure Proxy and preliminary Risk maps have been generated")
print("=" * 70)
