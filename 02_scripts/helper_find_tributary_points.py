"""
DUWE DISASTER INTELLIGENCE
DUWE AI FLOOD INTELLIGENCE ENGINE

Helper Script: Find Good Tributary Pour Points
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

flow_acc_path = os.path.join(base_dir, r"01_data\processed\flow_accumulation_duwe.tif")
catchment_path = os.path.join(base_dir, r"01_data\processed\catchment_duwe.tif")
stream_path = os.path.join(base_dir, r"01_data\processed\stream_network_duwe.tif")
flow_dir_path = os.path.join(base_dir, r"01_data\processed\flow_direction_duwe.tif")
figure_path = os.path.join(base_dir, r"03_outputs\figures\helper_tributary_points.png")

MAIN_OUTLET_ROW = 2447
MAIN_OUTLET_COL = 2972

print("=" * 70)
print("DUWE DISASTER INTELLIGENCE")
print("DUWE AI FLOOD INTELLIGENCE ENGINE")
print("=" * 70)
print("HELPER: TRIBUTARY POUR POINT FINDER")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------
print("\n[1] Loading rasters...")

with rasterio.open(flow_acc_path) as src:
    acc = src.read(1).astype("float64")

with rasterio.open(catchment_path) as src:
    catchment = src.read(1)

with rasterio.open(stream_path) as src:
    streams = src.read(1)

print("    Rasters loaded.")

# -------------------------------------------------
# 2. IDENTIFY MAIN STEM (very high accumulation)
# -------------------------------------------------
print("\n[2] Identifying main stem...")

# Cells with very high accumulation are likely on the main stem
main_stem_threshold = np.nanpercentile(acc[catchment == 1], 99.5)
main_stem_mask = (acc >= main_stem_threshold) & (catchment == 1)

print(f"    Main stem threshold : {main_stem_threshold:,.0f}")
print(f"    Main stem cells     : {np.sum(main_stem_mask):,}")

# -------------------------------------------------
# 3. FIND TRIBUTARY CANDIDATES
# -------------------------------------------------
print("\n[3] Searching for tributary candidates...")

# Good tributary characteristics:
# - On stream network
# - Inside catchment
# - Accumulation high enough to be significant
# - NOT on the main stem
# - Reasonable distance from main outlet

min_tributary_acc = 30000
min_dist_from_outlet = 150

rows = np.arange(acc.shape[0])[:, None]
cols = np.arange(acc.shape[1])[None, :]
dist_from_outlet = np.sqrt((rows - MAIN_OUTLET_ROW)**2 + (cols - MAIN_OUTLET_COL)**2)

tributary_mask = (
    (catchment == 1) &
    (streams == 1) &
    (acc >= min_tributary_acc) &
    (acc < main_stem_threshold) &          # exclude main stem
    (dist_from_outlet >= min_dist_from_outlet)
)

acc_trib = acc.copy()
acc_trib[np.logical_not(tributary_mask)] = 0

print(f"    Tributary candidate cells : {np.sum(tributary_mask):,}")

# -------------------------------------------------
# 4. SELECT WELL-SPACED TRIBUTARY POINTS
# -------------------------------------------------
print("\n[4] Selecting well-spaced tributary points...")

flat_indices = np.argsort(acc_trib.ravel())[::-1]

suggested_points = []
min_distance_between = 90

for idx in flat_indices:
    acc_val = acc_trib.ravel()[idx]
    if acc_val < min_tributary_acc:
        break

    row = idx // acc.shape[1]
    col = idx % acc.shape[1]

    too_close = False
    for r, c, _ in suggested_points:
        dist = np.sqrt((row - r)**2 + (col - c)**2)
        if dist < min_distance_between:
            too_close = True
            break

    if not too_close:
        suggested_points.append((row, col, acc_val))

    if len(suggested_points) >= 8:
        break

# -------------------------------------------------
# 5. DISPLAY RESULTS
# -------------------------------------------------
print("\n[5] SUGGESTED TRIBUTARY POUR POINTS")
print("-" * 70)
print(f"{'No.':<5} {'Row':>8} {'Col':>8} {'Accumulation':>15} {'Dist from Outlet':>18}")
print("-" * 70)

for i, (row, col, acc_val) in enumerate(suggested_points, 1):
    dist = np.sqrt((row - MAIN_OUTLET_ROW)**2 + (col - MAIN_OUTLET_COL)**2)
    print(f"{i:<5} {row:>8} {col:>8} {acc_val:>15,.0f} {dist:>18.1f}")

print("-" * 70)

# -------------------------------------------------
# 6. VISUALISATION
# -------------------------------------------------
print("\n[6] Generating map...")

fig, ax = plt.subplots(figsize=(12, 9))

acc_plot = acc.copy()
acc_plot[catchment != 1] = np.nan
acc_plot[acc_plot < 80] = np.nan

im = ax.imshow(np.log1p(acc_plot), cmap="Blues", alpha=0.85)

# Main outlet
ax.plot(MAIN_OUTLET_COL, MAIN_OUTLET_ROW, "ks", markersize=10, label="Main Outlet")

# Suggested tributary points
for i, (row, col, acc_val) in enumerate(suggested_points, 1):
    ax.plot(col, row, "r*", markersize=13)
    ax.text(col + 18, row - 12, str(i), color="red", fontsize=10, fontweight="bold")

ax.set_title(
    "DUWE AI FLOOD INTELLIGENCE ENGINE\nSuggested Tributary Pour Points",
    fontsize=14, fontweight="bold", pad=12
)
ax.set_xlabel("Column")
ax.set_ylabel("Row")
ax.legend(loc="upper right")

plt.colorbar(im, ax=ax, shrink=0.75, label="Log(Flow Accumulation + 1)")

plt.figtext(0.5, 0.01, "DUWE DISASTER INTELLIGENCE  |  Tributary Pour Point Finder",
            ha="center", fontsize=9, style="italic", color="gray")

plt.tight_layout()
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n    Figure saved to:\n    {figure_path}")

print("\n" + "=" * 70)
print("TRIBUTARY SUGGESTION COMPLETED")
print("These points should give much better sub-catchments")
print("=" * 70)
