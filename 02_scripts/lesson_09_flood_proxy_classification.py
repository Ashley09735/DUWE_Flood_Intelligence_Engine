"""
DUWE DISASTER INTELLIGENCE
DUWE AI FLOOD INTELLIGENCE ENGINE

Lesson 9: Flood Proxy Classification & Area Statistics
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

flood_proxy_path = os.path.join(base_dir, r"01_data\processed\flood_proxy_duwe.tif")
catchment_path = os.path.join(base_dir, r"01_data\processed\catchment_duwe.tif")
classified_path = os.path.join(base_dir, r"01_data\processed\flood_proxy_classified_duwe.tif")
figure_path = os.path.join(base_dir, r"03_outputs\figures\lesson_09_flood_classification.png")

print("=" * 70)
print("DUWE DISASTER INTELLIGENCE")
print("DUWE AI FLOOD INTELLIGENCE ENGINE")
print("=" * 70)
print("LESSON 9: FLOOD PROXY CLASSIFICATION")
print("Module: Flood Susceptibility Module")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------
print("\n[1] Loading Flood Proxy...")

with rasterio.open(flood_proxy_path) as src:
    proxy = src.read(1).astype("float64")
    profile = src.profile.copy()

with rasterio.open(catchment_path) as src:
    catchment = src.read(1)

mask = (catchment == 1) & (proxy > -1000)
proxy_masked = np.where(mask, proxy, np.nan)

print(f"    Valid pixels : {np.sum(mask):,}")

# -------------------------------------------------
# 2. CLASSIFY INTO RISK CATEGORIES
# -------------------------------------------------
print("\n[2] Classifying Flood Proxy...")

# Classification thresholds (using percentiles for balance)
p30 = np.nanpercentile(proxy_masked, 30)
p60 = np.nanpercentile(proxy_masked, 60)
p85 = np.nanpercentile(proxy_masked, 85)

print(f"    Thresholds → Low/Med: {p30:.3f} | Med/High: {p60:.3f} | High/V.High: {p85:.3f}")

# Create classified map
# 1 = Low, 2 = Medium, 3 = High, 4 = Very High
classified = np.zeros_like(proxy_masked)

classified[proxy_masked <= p30] = 1
classified[(proxy_masked > p30) & (proxy_masked <= p60)] = 2
classified[(proxy_masked > p60) & (proxy_masked <= p85)] = 3
classified[proxy_masked > p85] = 4

classified = np.where(mask, classified, np.nan)

# -------------------------------------------------
# 3. AREA STATISTICS
# -------------------------------------------------
print("\n[3] FLOOD RISK AREA STATISTICS")
print("-" * 60)

pixel_area_km2 = (30 * 30) / 1_000_000
total_area = np.sum(mask) * pixel_area_km2

classes = {
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Very High"
}

print(f"{'Class':<12} {'Pixels':>12} {'Area (km²)':>12} {'Percentage':>12}")
print("-" * 60)

for cls_id, cls_name in classes.items():
    pixels = np.sum(classified == cls_id)
    area = pixels * pixel_area_km2
    pct = (area / total_area) * 100
    print(f"{cls_name:<12} {pixels:>12,} {area:>12.2f} {pct:>11.1f}%")

print("-" * 60)
print(f"{'Total':<12} {np.sum(mask):>12,} {total_area:>12.2f} {100.0:>11.1f}%")

# -------------------------------------------------
# 4. SAVE CLASSIFIED RASTER
# -------------------------------------------------
print("\n[4] Saving classified raster...")

profile.update(dtype="float32", nodata=-9999.0, compress="lzw")
classified_save = np.where(mask, classified, -9999.0)

with rasterio.open(classified_path, "w", **profile) as dst:
    dst.write(classified_save.astype("float32"), 1)

print(f"    Saved to: {classified_path}")

# -------------------------------------------------
# 5. VISUALISATION
# -------------------------------------------------
print("\n[5] Generating classification map...")

from matplotlib.colors import ListedColormap

cmap = ListedColormap(["#2ecc71", "#f1c40f", "#e67e22", "#c0392b"])
labels = ["Low", "Medium", "High", "Very High"]

fig, ax = plt.subplots(figsize=(11, 9))

im = ax.imshow(classified, cmap=cmap, vmin=1, vmax=4)

cbar = plt.colorbar(im, ax=ax, shrink=0.75, ticks=[1, 2, 3, 4])
cbar.ax.set_yticklabels(labels)
cbar.set_label("Flood Risk Class")

ax.set_title("DUWE AI FLOOD INTELLIGENCE ENGINE\nLesson 9 – Flood Proxy Classification",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Column")
ax.set_ylabel("Row")

plt.figtext(0.5, 0.01, "DUWE DISASTER INTELLIGENCE  |  Flood Susceptibility Module",
            ha="center", fontsize=9, style="italic", color="gray")

plt.tight_layout()
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n    Figure saved to:\n    {figure_path}")

print("\n" + "=" * 70)
print("LESSON 9 COMPLETED SUCCESSFULLY")
print("=" * 70)
