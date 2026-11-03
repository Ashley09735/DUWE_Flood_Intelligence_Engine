"""
DUWE DISASTER INTELLIGENCE
DUWE AI FLOOD INTELLIGENCE ENGINE

Helper Script: Improved Pour Point Suggestion (v2)
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
figure_path = os.path.join(base_dir, r"03_outputs\figures\helper_suggested_pour_points_v2.png")

# Main outlet (we will avoid points too close to it)
MAIN_OUTLET_ROW = 2447
MAIN_OUTLET_COL = 2972

print("=" * 70)
print("DUWE DISASTER INTELLIGENCE")
print("DUWE AI FLOOD INTELLIGENCE ENGINE")
print("=" * 70)
print("HELPER: IMPROVED POUR POINT SUGGESTION (v2)")
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
# 2. CREATE CANDIDATE MASK
# -------------------------------------------------
print("\n[2] Creating candidate mask...")

# Conditions for a good candidate:
# - Inside main catchment
# - On stream network
# - Accumulation above a reasonable threshold
# - Not too close to the main outlet

min_acc_threshold = 80000          # Minimum accumulation to consider
min_dist_from_outlet = 120         # Minimum distance from main outlet (pixels)
min_distance_between = 100         # Minimum distance between suggested points

# Distance from main outlet
rows = np.arange(acc.shape[0])[:, None]
cols = np.arange(acc.shape[1])[None, :]
dist_from_outlet = np.sqrt((rows - MAIN_OUTLET_ROW)**2 + (cols - MAIN_OUTLET_COL)**2)

valid_mask = (
    (catchment == 1) &
    (streams == 1) &
    (acc >= min_acc_threshold) &
    (dist_from_outlet >= min_dist_from_outlet)
)

acc_valid = acc.copy()
acc_valid[np.logical_not(valid_mask)] = 0

print(f"    Candidate cells found : {np.sum(valid_mask):,}")

# -------------------------------------------------
# 3. SELECT WELL-SPACED HIGH ACCUMULATION POINTS
# -------------------------------------------------
print("\n[3] Selecting well-spaced pour points...")

flat_indices = np.argsort(acc_valid.ravel())[::-1]   # highest accumulation first

suggested_points = []

for idx in flat_indices:
    acc_val = acc_valid.ravel()[idx]
    if acc_val < min_acc_threshold:
        break

    row = idx // acc.shape[1]
    col = idx % acc.shape[1]

    # Check distance from already selected points
    too_close = False
    for r, c, _ in suggested_points:
        dist = np.sqrt((row - r)**2 + (col - c)**2)
        if dist < min_distance_between:
            too_close = True
            break

    if not too_close:
        suggested_points.append((row, col, acc_val))

    if len(suggested_points) >= 7:
        break

# -------------------------------------------------
# 4. DISPLAY RESULTS
# -------------------------------------------------
print("\n[4] SUGGESTED POUR POINTS (Improved)")
print("-" * 65)
print(f"{'No.':<5} {'Row':>8} {'Col':>8} {'Accumulation':>15} {'Dist from Outlet':>18}")
print("-" * 65)

for i, (row, col, acc_val) in enumerate(suggested_points, 1):
    dist = np.sqrt((row - MAIN_OUTLET_ROW)**2 + (col - MAIN_OUTLET_COL)**2)
    print(f"{i:<5} {row:>8} {col:>8} {acc_val:>15,.0f} {dist:>18.1f}")

print("-" * 65)

# -------------------------------------------------
# 5. VISUALISATION
# -------------------------------------------------
print("\n[5] Generating map...")

fig, ax = plt.subplots(figsize=(12, 9))

# Background
acc_plot = acc.copy()
acc_plot[catchment != 1] = np.nan
acc_plot[acc_plot < 100] = np.nan

im = ax.imshow(np.log1p(acc_plot), cmap="Blues", alpha=0.85)

# Main outlet
ax.plot(MAIN_OUTLET_COL, MAIN_OUTLET_ROW, "ks", markersize=10, label="Main Outlet")

# Suggested points
for i, (row, col, acc_val) in enumerate(suggested_points, 1):
    ax.plot(col, row, "r*", markersize=14)
    ax.text(col + 20, row - 15, str(i), color="red", fontsize=11, fontweight="bold")

ax.set_title(
    "DUWE AI FLOOD INTELLIGENCE ENGINE\nImproved Suggested Pour Points for Sub-catchments",
    fontsize=14, fontweight="bold", pad=12
)
ax.set_xlabel("Column")
ax.set_ylabel("Row")
ax.legend(loc="upper right")

plt.colorbar(im, ax=ax, shrink=0.75, label="Log(Flow Accumulation + 1)")

plt.figtext(0.5, 0.01, "DUWE DISASTER INTELLIGENCE  |  Pour Point Suggestion Tool v2",
            ha="center", fontsize=9, style="italic", color="gray")

plt.tight_layout()
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n    Figure saved to:\n    {figure_path}")

print("\n" + "=" * 70)
print("IMPROVED SUGGESTION COMPLETED")
print("Copy the best Row and Col values into Lesson 7")
print("=" * 70)
