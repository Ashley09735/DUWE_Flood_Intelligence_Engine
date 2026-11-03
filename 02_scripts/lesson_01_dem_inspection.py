"""
DUWE DISASTER INTELLIGENCE
DUWE AI FLOOD INTELLIGENCE ENGINE

Lesson 1: DEM Inspection & Understanding
Module: Data Ingestion Layer
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os
from datetime import datetime

# -------------------------------------------------
# PATHS
# -------------------------------------------------
dem_path = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE\01_data\raw\filled_ayaya.tif"

print("=" * 70)
print("DUWE DISASTER INTELLIGENCE")
print("DUWE AI FLOOD INTELLIGENCE ENGINE")
print("=" * 70)
print("LESSON 1: DEM INSPECTION & UNDERSTANDING")
print("Module: Data Ingestion Layer")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

# -------------------------------------------------
# 1. LOAD DEM
# -------------------------------------------------
print("\n[1] Loading DEM...")

with rasterio.open(dem_path) as src:
    dem = src.read(1).astype("float64")
    profile = src.profile
    transform = src.transform
    crs = src.crs
    nodata = src.nodata
    bounds = src.bounds
    res = src.res
    width = src.width
    height = src.height

    print("    DEM loaded successfully.")

    print("\n[2] BASIC RASTER INFORMATION")
    print("-" * 50)
    print(f"    File name          : {os.path.basename(dem_path)}")
    print(f"    Driver             : {src.driver}")
    print(f"    CRS                : {crs}")
    print(f"    EPSG Code          : {crs.to_epsg() if crs else 'None'}")
    print(f"    Dimensions         : {width} columns x {height} rows")
    print(f"    Resolution         : {res[0]} m x {res[1]} m")
    print(f"    NoData value       : {nodata}")
    print(f"    Number of bands    : {src.count}")

    print("\n[3] GEOGRAPHIC EXTENT")
    print("-" * 50)
    print(f"    West  (Left)       : {bounds.left:,.2f}")
    print(f"    South (Bottom)     : {bounds.bottom:,.2f}")
    print(f"    East  (Right)      : {bounds.right:,.2f}")
    print(f"    North (Top)        : {bounds.top:,.2f}")

# -------------------------------------------------
# 2. ELEVATION STATISTICS
# -------------------------------------------------
if nodata is not None:
    dem_masked = np.ma.masked_equal(dem, nodata)
else:
    dem_masked = np.ma.masked_invalid(dem)

print("\n[4] ELEVATION STATISTICS")
print("-" * 50)
print(f"    Minimum elevation  : {dem_masked.min():.2f} m")
print(f"    Maximum elevation  : {dem_masked.max():.2f} m")
print(f"    Mean elevation     : {dem_masked.mean():.2f} m")
print(f"    Median elevation   : {np.ma.median(dem_masked):.2f} m")
print(f"    Std. Deviation     : {dem_masked.std():.2f} m")

valid_pixels = dem_masked.count()
total_pixels = dem.size
nodata_pixels = total_pixels - valid_pixels

print(f"\n    Valid pixels       : {valid_pixels:,}")
print(f"    NoData pixels      : {nodata_pixels:,}")
print(f"    Total pixels       : {total_pixels:,}")

# -------------------------------------------------
# 3. VISUALISATION
# -------------------------------------------------
print("\n[5] Generating DEM visualisation...")

fig, ax = plt.subplots(figsize=(11, 9))

im = ax.imshow(
    dem_masked,
    cmap="terrain",
    vmin=dem_masked.min(),
    vmax=dem_masked.max()
)

cbar = plt.colorbar(im, ax=ax, shrink=0.75)
cbar.set_label("Elevation (metres)", fontsize=11)

ax.set_title("DUWE AI FLOOD INTELLIGENCE ENGINE\nLesson 1 – Chikwawa DEM", 
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Column", fontsize=11)
ax.set_ylabel("Row", fontsize=11)

# Footer branding
plt.figtext(0.5, 0.01, "DUWE DISASTER INTELLIGENCE  |  Data Ingestion Layer", 
            ha="center", fontsize=9, style="italic", color="gray")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()

print("\n" + "=" * 70)
print("LESSON 1 COMPLETED SUCCESSFULLY")
print("DUWE AI FLOOD INTELLIGENCE ENGINE – Data Ingestion Layer ready")
print("=" * 70)
