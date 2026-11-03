
"""
DUWE DISASTER INTELLIGENCE
MODULE 14: SHIRE RIVER ELEVATION PROFILE

Target OSM ID: 24896284
"""

import os
import csv
import fiona
import rasterio
import numpy as np

from shapely.geometry import shape, LineString, MultiLineString

BASE = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"

RIVER_FILE = os.path.join(
    BASE, "01_data", "raw", "mw_rivers.gpkg"
)

DEM_FILE = os.path.join(
    BASE, "01_data", "processed",
    "dem_conditioned_duwe.tif"
)

OUTPUT_DIR = os.path.join(
    BASE, "01_data", "processed", "forecasting"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_ID = "24896284"
INTERVAL = 30  # metres

print("\nDUWE DISASTER INTELLIGENCE")
print("MODULE 14: SHIRE RIVER ELEVATION PROFILE")
print("=" * 60)

# --------------------------------------------------
# 1. FIND THE SELECTED RIVER FEATURE
# --------------------------------------------------

river_geometry = None

with fiona.open(RIVER_FILE) as source:

    for feature in source:

        properties = feature["properties"]

        if str(properties.get("osm_id")) == TARGET_ID:

            river_geometry = shape(feature["geometry"])
            break

if river_geometry is None:
    raise ValueError("Selected Shire River feature was not found.")

print("Selected OSM ID:", TARGET_ID)
print("Geometry type:", river_geometry.geom_type)
print("Total geometry length (km):",
      round(river_geometry.length / 1000, 3))

# --------------------------------------------------
# 2. GET THE LONGEST LINE COMPONENT
# --------------------------------------------------

if isinstance(river_geometry, LineString):

    main_line = river_geometry

elif isinstance(river_geometry, MultiLineString):

    main_line = max(
        river_geometry.geoms,
        key=lambda line: line.length
    )

else:
    raise TypeError("Unexpected river geometry type.")

print("Longest component length (km):",
      round(main_line.length / 1000, 3))

# --------------------------------------------------
# 3. SAMPLE THE DEM ALONG THE RIVER
# --------------------------------------------------

records = []

with rasterio.open(DEM_FILE) as dem:

    if dem.crs.to_epsg() != 32736:
        raise ValueError("DEM CRS is not EPSG:32736.")

    number_of_samples = int(
        main_line.length / INTERVAL
    ) + 1

    for i in range(number_of_samples + 1):

        distance = min(
            i * INTERVAL,
            main_line.length
        )

        point = main_line.interpolate(distance)

        elevation = next(
            dem.sample([(point.x, point.y)])
        )[0]

        if (
            not np.isfinite(elevation)
            or elevation == dem.nodata
        ):
            elevation_value = None
        else:
            elevation_value = float(elevation)

        records.append({
            "distance_m": round(distance, 2),
            "easting": round(point.x, 2),
            "northing": round(point.y, 2),
            "elevation_m": elevation_value
        })

# --------------------------------------------------
# 4. SAVE PROFILE AS CSV
# --------------------------------------------------

output_file = os.path.join(
    OUTPUT_DIR,
    "shire_river_elevation_profile.csv"
)

with open(output_file, "w", newline="",
          encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "distance_m",
            "easting",
            "northing",
            "elevation_m"
        ]
    )

    writer.writeheader()
    writer.writerows(records)

# --------------------------------------------------
# 5. REPORT STATISTICS
# --------------------------------------------------

valid = [
    row["elevation_m"]
    for row in records
    if row["elevation_m"] is not None
]

print("\nPROFILE RESULTS")
print("Total samples:", len(records))
print("Valid elevation samples:", len(valid))
print("Missing elevation samples:", len(records) - len(valid))

if valid:

    print("Minimum DEM elevation:",
          round(min(valid), 3), "m")

    print("Maximum DEM elevation:",
          round(max(valid), 3), "m")

    print("Elevation difference:",
          round(max(valid) - min(valid), 3), "m")

else:
    print("No valid elevation samples were found.")

print("\nCSV saved:", output_file)
print("\nMODULE 14 COMPLETED")
