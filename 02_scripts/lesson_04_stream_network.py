"""
DUWE DISASTER INTELLIGENCE
DUWE AI FLOOD INTELLIGENCE ENGINE

Lesson 4: Stream Network Extraction
Module: Drainage Network Module
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
stream_raster_path = os.path.join(base_dir, r"01_data\processed\stream_network_duwe.tif")
figure_path = os.path.join(base_dir, r"03_outputs\figures\lesson_04_stream_network.png")

# -------------------------------------------------
# STREAM EXTRACTION THRESHOLD
# -------------------------------------------------
# You can change this value later
# Higher threshold = fewer (only major) streams
# Lower threshold  = more detailed network
ACCUMULATION_THRESHOLD = 5000

print("=" * 70)
print("DUWE DISASTER INTELLIGENCE")
print("DUWE AI FLOOD INTELLIGENCE ENGINE")
print("=" * 70)
print("LESSON 4: STREAM NETWORK EXTRACTION")
print("Module: Drainage Network Module")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Accumulation Threshold: {ACCUMULATION_THRESHOLD:,} cells")
print("=" * 70)

# -------------------------------------------------
# 1. LOAD FLOW ACCUMULATION
# -------------------------------------------------
print("\n[1] Loading Flow Accumulation...")

with rasterio.open(flow_acc_path) as src:
    acc = src.read(1).astype("float64")
    profile = src.profile.copy()
    nodata = src.nodata
    print(f"    NoData value : {nodata}")
    print(f"    Shape        : {acc.shape}")

# -------------------------------------------------
# 2. CREATE MASK
# -------------------------------------------------
print("\n[2] Creating valid data mask...")

mask_nodata = acc < -1000
acc_masked = acc.copy()
acc_masked[mask_nodata] = np.nan

valid_pixels = np.sum(np.isfinite(acc_masked))
print(f"    Valid pixels : {valid_pixels:,}")

# -------------------------------------------------
# 3. EXTRACT STREAM NETWORK
# -------------------------------------------------
print("\n[3] Extracting stream network...")

# Create binary stream network
streams = np.where(acc_masked >= ACCUMULATION_THRESHOLD, 1, 0).astype("float32")
streams[mask_nodata] = -9999.0

stream_pixels = np.sum(streams == 1)
print(f"    Stream pixels extracted : {stream_pixels:,}")
print(f"    Threshold used          : {ACCUMULATION_THRESHOLD:,}")

# -------------------------------------------------
# 4. SAVE STREAM NETWORK RASTER
# -------------------------------------------------
print("\n[4] Saving stream network raster...")

profile.update({
    "dtype": "float32",
    "nodata": -9999.0,
    "compress": "lzw"
})

with rasterio.open(stream_raster_path, "w", **profile) as dst:
    dst.write(streams, 1)

print(f"    Saved to:\n    {stream_raster_path}")

# -------------------------------------------------
# 5. DIAGNOSTICS
# -------------------------------------------------
print("\n[5] STREAM NETWORK DIAGNOSTICS")
print("-" * 50)
print(f"    Total valid terrain pixels     : {valid_pixels:,}")
print(f"    Stream pixels                  : {stream_pixels:,}")
print(f"    Percentage of terrain as stream: {(stream_pixels / valid_pixels) * 100:.3f}%")

# -------------------------------------------------
# 6. VISUALISATION
# -------------------------------------------------
print("\n[6] Generating stream network map...")

# Prepare background (log accumulation)
acc_plot = acc_masked.copy()
acc_plot[acc_plot < 50] = np.nan

fig, ax = plt.subplots(figsize=(11, 9))

# Background: Flow accumulation
im = ax.imshow(
    np.log1p(acc_plot),
    cmap="Blues",
    alpha=0.7,
    vmin=np.log1p(50),
    vmax=np.log1p(np.nanpercentile(acc_masked, 99.5))
)

# Overlay: Stream network
streams_plot = np.where(streams == 1, 1, np.nan)
ax.imshow(streams_plot, cmap="autumn", alpha=0.9, vmin=0, vmax=1)

ax.set_title(
    f"DUWE AI FLOOD INTELLIGENCE ENGINE\nLesson 4 – Stream Network (Threshold = {ACCUMULATION_THRESHOLD:,})",
    fontsize=14, fontweight="bold", pad=12
)
ax.set_xlabel("Column")
ax.set_ylabel("Row")

plt.colorbar(im, ax=ax, shrink=0.75, label="Log(Flow Accumulation + 1)")

plt.figtext(0.5, 0.01, "DUWE DISASTER INTELLIGENCE  |  Drainage Network Module",
            ha="center", fontsize=9, style="italic", color="gray")

plt.tight_layout()
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n    Figure saved to:\n    {figure_path}")

print("\n" + "=" * 70)
print("LESSON 4 COMPLETED SUCCESSFULLY")
print("Stream Network has been extracted and saved")
print("=" * 70)
print("\nNote: You can change the ACCUMULATION_THRESHOLD value")
print("at the top of the script and run again to adjust network density.")
