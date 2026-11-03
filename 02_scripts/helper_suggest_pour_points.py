"""
DUWE DISASTER INTELLIGENCE
DUWE AI FLOOD INTELLIGENCE ENGINE

Helper Script: Automatic Pour Point Suggestion
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
figure_path = os.path.join(base_dir, r"03_outputs\figures\helper_suggested_pour_points.png")

print("=" * 70)
print("DUWE DISASTER INTELLIGENCE")
print("DUWE AI FLOOD INTELLIGENCE ENGINE")
print("=" * 70)
print("HELPER: AUTOMATIC POUR POINT SUGGESTION")
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
# 2. FOCUS ONLY ON MAIN CATCHMENT + STREAMS
# -------------------------------------------------
print("\n[2] Filtering to main catchment and stream network...")

# Only consider cells that are:
# - Inside the main catchment
# - Part of the stream network
# - Have significant accumulation

valid_mask = (catchment == 1) & (streams == 1) & (acc > 10000)

acc_valid = acc.copy()
acc_valid[np.logical_not(valid_mask)] = 0

print(f"    Candidate stream cells : {np.sum(valid_mask):,}")

# -------------------------------------------------
# 3. FIND TOP CANDIDATE POINTS
# -------------------------------------------------
print("\n[3] Finding top candidate pour points...")

# Get indices of highest accumulation cells
flat_indices = np.argsort(acc_valid.ravel())[::-1]  # descending order

# We will pick well-separated points
suggested_points = []
min_distance = 80          # minimum distance between points (in pixels)

for idx in flat_indices:
    if acc_valid.ravel()[idx] < 15000:
        break

    row = idx // acc.shape[1]
    col = idx % acc.shape[1]

    # Check distance from already selected points
    too_close = False
    for r, c, _ in suggested_points:
        dist = np.sqrt((row - r)**2 + (col - c)**2)
        if dist < min_distance:
            too_close = True
            break

    if not too_close:
        suggested_points.append((row, col, acc_valid[row, col]))

    if len(suggested_points) >= 8:          # suggest up to 8 points
        break

# -------------------------------------------------
# 4. DISPLAY SUGGESTIONS
# -------------------------------------------------
print("\n[4] SUGGESTED POUR POINTS")
print("-" * 60)
print(f"{'No.':<5} {'Row':>8} {'Col':>8} {'Accumulation':>15}")
print("-" * 60)

for i, (row, col, acc_val) in enumerate(suggested_points, 1):
    print(f"{i:<5} {row:>8} {col:>8} {acc_val:>15,.0f}")

print("-" * 60)
print("Note: Point No. 1 is usually the main outlet or very close to it.")

# -------------------------------------------------
# 5. VISUALISATION
# -------------------------------------------------
print("\n[5] Generating suggestion map...")

fig, ax = plt.subplots(figsize=(12, 9))

# Background: log accumulation inside catchment
acc_plot = acc.copy()
acc_plot[catchment != 1] = np.nan
acc_plot[acc_plot < 100] = np.nan

im = ax.imshow(np.log1p(acc_plot), cmap="Blues", alpha=0.85)

# Plot suggested points
for i, (row, col, acc_val) in enumerate(suggested_points, 1):
    ax.plot(col, row, "r*", markersize=14)
    ax.text(col + 25, row, str(i), color="red", fontsize=11, fontweight="bold")

ax.set_title(
    "DUWE AI FLOOD INTELLIGENCE ENGINE\nSuggested Pour Points for Sub-catchments",
    fontsize=14, fontweight="bold", pad=12
)
ax.set_xlabel("Column")
ax.set_ylabel("Row")

plt.colorbar(im, ax=ax, shrink=0.75, label="Log(Flow Accumulation + 1)")

plt.figtext(0.5, 0.01, "DUWE DISASTER INTELLIGENCE  |  Pour Point Suggestion Tool",
            ha="center", fontsize=9, style="italic", color="gray")

plt.tight_layout()
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n    Figure saved to:\n    {figure_path}")

print("\n" + "=" * 70)
print("SUGGESTION COMPLETED")
print("Copy the Row and Col values you want to use into Lesson 7")
print("=" * 70)
