"""
DUWE DISASTER INTELLIGENCE
DUWE AI FLOOD INTELLIGENCE ENGINE

Lesson 6: Catchment Morphometric Analysis
Module: Terrain Intelligence Layer
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from datetime import datetime
import os
from math import pi, sqrt

# -------------------------------------------------
# PATHS
# -------------------------------------------------
base_dir = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"

dem_path = os.path.join(base_dir, r"01_data\processed\dem_conditioned_duwe.tif")
catchment_path = os.path.join(base_dir, r"01_data\processed\catchment_duwe.tif")
stream_path = os.path.join(base_dir, r"01_data\processed\stream_network_duwe.tif")
figure_path = os.path.join(base_dir, r"03_outputs\figures\lesson_06_morphometric.png")

print("=" * 70)
print("DUWE DISASTER INTELLIGENCE")
print("DUWE AI FLOOD INTELLIGENCE ENGINE")
print("=" * 70)
print("LESSON 6: CATCHMENT MORPHOMETRIC ANALYSIS")
print("Module: Terrain Intelligence Layer")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------
print("\n[1] Loading required rasters...")

with rasterio.open(dem_path) as src:
    dem = src.read(1).astype("float64")
    transform = src.transform
    pixel_size = transform[0]          # 30 metres

with rasterio.open(catchment_path) as src:
    catchment = src.read(1)

with rasterio.open(stream_path) as src:
    streams = src.read(1)

print("    All rasters loaded successfully.")

# -------------------------------------------------
# 2. BASIC CATCHMENT PROPERTIES
# -------------------------------------------------
print("\n[2] Calculating basic catchment properties...")

# Catchment mask
catch_mask = catchment == 1

# Area
catchment_pixels = np.sum(catch_mask)
area_m2 = catchment_pixels * (pixel_size ** 2)
area_km2 = area_m2 / 1_000_000

# Elevation inside catchment
dem_catch = dem.copy()
dem_catch[catchment != 1] = np.nan
dem_catch[dem_catch < -1000] = np.nan

h_min = np.nanmin(dem_catch)
h_max = np.nanmax(dem_catch)
h_mean = np.nanmean(dem_catch)
basin_relief = h_max - h_min

print(f"    Catchment Area          : {area_km2:.2f} km²")
print(f"    Minimum Elevation       : {h_min:.1f} m")
print(f"    Maximum Elevation       : {h_max:.1f} m")
print(f"    Mean Elevation          : {h_mean:.1f} m")
print(f"    Basin Relief            : {basin_relief:.1f} m")

# -------------------------------------------------
# 3. STREAM NETWORK PROPERTIES
# -------------------------------------------------
print("\n[3] Calculating stream network properties...")

# Streams inside catchment only
streams_in_catch = np.logical_and(streams == 1, catch_mask)
stream_pixels = np.sum(streams_in_catch)

# Approximate total stream length (each pixel ≈ pixel_size)
# Note: This is a simple approximation (better methods exist with vectorization)
stream_length_m = stream_pixels * pixel_size
stream_length_km = stream_length_m / 1000

print(f"    Stream pixels           : {stream_pixels:,}")
print(f"    Approx. Stream Length   : {stream_length_km:.2f} km")

# -------------------------------------------------
# 4. MORPHOMETRIC PARAMETERS
# -------------------------------------------------
print("\n[4] MORPHOMETRIC PARAMETERS")
print("-" * 55)

# 1. Drainage Density (Dd) = Stream Length / Area
drainage_density = stream_length_km / area_km2

# 2. Stream Frequency (Fs) = Number of streams / Area
# (Using stream pixels as proxy for now)
stream_frequency = stream_pixels / area_km2

# 3. Relief Ratio (Rr) = Basin Relief / Basin Length
# Approximate basin length as diameter of equivalent circle
basin_length_km = sqrt(area_km2 / pi) * 2
relief_ratio = (basin_relief / 1000) / basin_length_km   # convert relief to km

# 4. Elongation Ratio (Re)
# Re = Diameter of circle with same area / Basin length
elongation_ratio = (2 * sqrt(area_km2 / pi)) / basin_length_km

# 5. Circularity Ratio (Rc)
# Rc = 4 * pi * Area / Perimeter²
# Approximate perimeter using a simple method
from scipy import ndimage
# Binary catchment for perimeter estimation
catch_binary = catch_mask.astype(int)
# Simple perimeter approximation
eroded = ndimage.binary_erosion(catch_binary)
perimeter_pixels = np.sum(catch_binary) - np.sum(eroded)
perimeter_km = (perimeter_pixels * pixel_size) / 1000

circularity_ratio = (4 * pi * area_km2) / (perimeter_km ** 2) if perimeter_km > 0 else 0

# 6. Form Factor (Ff)
form_factor = area_km2 / (basin_length_km ** 2)

# 7. Ruggedness Number (Rn)
ruggedness_number = drainage_density * (basin_relief / 1000)

print(f"    Drainage Density (Dd)      : {drainage_density:.3f} km/km²")
print(f"    Stream Frequency (Fs)      : {stream_frequency:.1f} /km²")
print(f"    Basin Length (approx)      : {basin_length_km:.2f} km")
print(f"    Relief Ratio (Rr)          : {relief_ratio:.4f}")
print(f"    Elongation Ratio (Re)      : {elongation_ratio:.3f}")
print(f"    Circularity Ratio (Rc)     : {circularity_ratio:.3f}")
print(f"    Form Factor (Ff)           : {form_factor:.3f}")
print(f"    Ruggedness Number (Rn)     : {ruggedness_number:.3f}")

# -------------------------------------------------
# 5. INTERPRETATION
# -------------------------------------------------
print("\n[5] QUICK INTERPRETATION")
print("-" * 55)

if elongation_ratio < 0.7:
    shape_interp = "Elongated basin → slower flood response"
elif elongation_ratio > 0.9:
    shape_interp = "Circular basin → faster flood response"
else:
    shape_interp = "Moderately elongated basin"

if drainage_density < 1.0:
    dd_interp = "Low drainage density → more infiltration / storage"
elif drainage_density > 2.0:
    dd_interp = "High drainage density → faster runoff"
else:
    dd_interp = "Moderate drainage density"

print(f"    Shape characteristics   : {shape_interp}")
print(f"    Drainage characteristics: {dd_interp}")

# -------------------------------------------------
# 6. VISUAL SUMMARY
# -------------------------------------------------
print("\n[6] Generating summary figure...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Catchment + Streams
axes[0].imshow(dem_catch, cmap="terrain")
streams_plot = np.where(streams_in_catch, 1, np.nan)
axes[0].imshow(streams_plot, cmap="Reds", alpha=0.8)
axes[0].set_title("Catchment + Stream Network", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Column")
axes[0].set_ylabel("Row")

# Right: Text summary
axes[1].axis("off")
summary_text = (
    f"CATCHMENT MORPHOMETRIC SUMMARY\n"
    f"{'='*40}\n\n"
    f"Area                 : {area_km2:.2f} km²\n"
    f"Basin Relief         : {basin_relief:.1f} m\n"
    f"Drainage Density     : {drainage_density:.3f} km/km²\n"
    f"Elongation Ratio     : {elongation_ratio:.3f}\n"
    f"Circularity Ratio    : {circularity_ratio:.3f}\n"
    f"Form Factor          : {form_factor:.3f}\n"
    f"Ruggedness Number    : {ruggedness_number:.3f}\n\n"
    f"Interpretation:\n"
    f"- {shape_interp}\n"
    f"- {dd_interp}"
)
axes[1].text(0.05, 0.95, summary_text, transform=axes[1].transAxes,
             fontsize=12, verticalalignment="top", fontfamily="monospace")

fig.suptitle("DUWE AI FLOOD INTELLIGENCE ENGINE\nLesson 6 – Catchment Morphometric Analysis",
             fontsize=15, fontweight="bold", y=1.02)

plt.figtext(0.5, 0.01, "DUWE DISASTER INTELLIGENCE  |  Terrain Intelligence Layer",
            ha="center", fontsize=9, style="italic", color="gray")

plt.tight_layout()
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n    Figure saved to:\n    {figure_path}")

print("\n" + "=" * 70)
print("LESSON 6 COMPLETED SUCCESSFULLY")
print("Morphometric analysis of the catchment is ready")
print("=" * 70)
