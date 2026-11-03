
"""
DUWE DISASTER INTELLIGENCE
MODULE 11: TERRAIN AND RIVER DATA INSPECTOR

Purpose:
1. Inspect the Chikwawa DEM.
2. Inspect the catchment raster.
3. Check whether both rasters align.
4. Identify available river and watercourse datasets.
5. Prepare for pilot 2D hydraulic modelling.
"""

from pathlib import Path
import rasterio
import numpy as np

# --------------------------------------------------
# 1. PROJECT DIRECTORIES
# --------------------------------------------------

ROOT = Path(
    r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
)

RAW = ROOT / "01_data" / "raw"
PROCESSED = ROOT / "01_data" / "processed"

DEM = PROCESSED / "dem_conditioned_duwe.tif"
CATCHMENT = PROCESSED / "catchment_duwe.tif"

# --------------------------------------------------
# 2. RASTER INSPECTION FUNCTION
# --------------------------------------------------

def inspect_raster(path):

    print("\n" + "=" * 65)
    print("RASTER:", path.name)
    print("=" * 65)

    if not path.exists():

        print("FILE NOT FOUND")
        return None

    with rasterio.open(path) as src:

        print("CRS:", src.crs)
        print("Dimensions:", src.width, "x", src.height)
        print("Resolution:", src.res)
        print("Bounds:", src.bounds)
        print("NoData:", src.nodata)
        print("Data type:", src.dtypes[0])

        valid_count = 0
        minimum = float("inf")
        maximum = float("-inf")

        # Read one raster block at a time
        for _, window in src.block_windows(1):

            data = src.read(1, window=window, masked=True)

            values = data.compressed()
            values = values[np.isfinite(values)]

            if values.size > 0:

                valid_count += values.size
                minimum = min(minimum, float(values.min()))
                maximum = max(maximum, float(values.max()))

        print("Valid cells:", valid_count)

        if valid_count:

            print("Minimum:", round(minimum, 3))
            print("Maximum:", round(maximum, 3))

        print(
            "Pixel area:",
            abs(
                src.transform.a * src.transform.e
                - src.transform.b * src.transform.d
            ),
            "square map units"
        )

        return {
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "transform": src.transform,
            "bounds": src.bounds,
            "valid_cells": valid_count,
            "minimum": minimum,
            "maximum": maximum
        }

# --------------------------------------------------
# 3. INSPECT BOTH RASTERS
# --------------------------------------------------

print("\nDUWE TERRAIN INTELLIGENCE")
print("CHIKWAWA HYDRAULIC MODEL PREPARATION")

dem_info = inspect_raster(DEM)
catchment_info = inspect_raster(CATCHMENT)

# --------------------------------------------------
# 4. CHECK RASTER ALIGNMENT
# --------------------------------------------------

print("\n" + "=" * 65)
print("RASTER ALIGNMENT CHECK")
print("=" * 65)

if dem_info and catchment_info:

    same_crs = dem_info["crs"] == catchment_info["crs"]

    same_dimensions = (
        dem_info["width"] == catchment_info["width"]
        and dem_info["height"] == catchment_info["height"]
    )

    same_transform = (
        dem_info["transform"] == catchment_info["transform"]
    )

    print("Same CRS:", same_crs)
    print("Same dimensions:", same_dimensions)
    print("Same transform:", same_transform)

    if same_crs and same_dimensions and same_transform:

        print("RESULT: RASTERS ARE ALIGNED")

    else:

        print("RESULT: RASTERS REQUIRE ALIGNMENT REVIEW")

# --------------------------------------------------
# 5. SEARCH FOR RIVER DATA
# --------------------------------------------------

print("\n" + "=" * 65)
print("SEARCHING FOR RIVER AND WATERCOURSE DATA")
print("=" * 65)

extensions = [
    ".shp",
    ".gpkg",
    ".geojson",
    ".gdb",
    ".kml"
]

for folder in [RAW, PROCESSED]:

    if not folder.exists():
        continue

    print("\nSEARCHING:", folder)

    for file in folder.rglob("*"):

        if file.is_file() and file.suffix.lower() in extensions:

            print("FOUND:", file)

# --------------------------------------------------
# 6. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 65)
print("DUWE TERRAIN INSPECTION COMPLETED")
print("=" * 65)
