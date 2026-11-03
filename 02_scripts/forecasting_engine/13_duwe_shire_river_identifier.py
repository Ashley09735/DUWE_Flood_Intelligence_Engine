
"""
DUWE DISASTER INTELLIGENCE
MODULE 13: SHIRE RIVER IDENTIFICATION

Purpose:
1. Find named Shire River features.
2. Calculate their lengths.
3. Inspect their geographic locations.
4. Sample the catchment raster along the river.
"""

import os
import json
import fiona
import rasterio

from shapely.geometry import shape

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"

RIVER_FILE = os.path.join(
    BASE_DIR, "01_data", "raw", "mw_rivers.gpkg"
)

CATCHMENT_FILE = os.path.join(
    BASE_DIR, "01_data", "processed", "catchment_duwe.tif"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "01_data", "processed", "forecasting"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. IDENTIFY SHIRE RIVER FEATURES
# --------------------------------------------------

print("\nDUWE DISASTER INTELLIGENCE")
print("MODULE 13: SHIRE RIVER IDENTIFICATION")
print("=" * 65)

results = []

with rasterio.open(CATCHMENT_FILE) as raster:

    with fiona.open(RIVER_FILE) as rivers:

        for feature in rivers:

            properties = feature["properties"]

            river_name = properties.get("name")

            if not river_name:
                continue

            if "shire" not in str(river_name).lower():
                continue

            geometry = shape(feature["geometry"])

            # Calculate river length in kilometres.
            # CRS is projected in metres (EPSG:32736).
            length_km = geometry.length / 1000

            # Sample points along the river.
            number_of_samples = max(
                2, int(geometry.length / 100) + 1
            )

            catchment_counts = {
                "inside": 0,
                "outside": 0,
                "nodata": 0
            }

            for i in range(number_of_samples):

                distance = (
                    geometry.length * i /
                    (number_of_samples - 1)
                )

                point = geometry.interpolate(distance)

                value = next(
                    raster.sample([(point.x, point.y)])
                )[0]

                if value == raster.nodata:
                    catchment_counts["nodata"] += 1

                elif value == 1:
                    catchment_counts["inside"] += 1

                elif value == 0:
                    catchment_counts["outside"] += 1

            record = {
                "name": str(river_name),
                "osm_id": str(properties.get("osm_id")),
                "feature_class": properties.get("fclass"),
                "length_km": round(length_km, 3),
                "bounds": list(geometry.bounds),
                "catchment_samples": catchment_counts
            }

            results.append(record)

# --------------------------------------------------
# 3. DISPLAY RESULTS
# --------------------------------------------------

print("\nSHIRE RIVER FEATURES FOUND:", len(results))

for index, item in enumerate(results, start=1):

    print("\nFEATURE:", index)
    print("Name:", item["name"])
    print("OSM ID:", item["osm_id"])
    print("Class:", item["feature_class"])
    print("Length (km):", item["length_km"])
    print("Bounds:", item["bounds"])
    print("Catchment samples:", item["catchment_samples"])

# --------------------------------------------------
# 4. SAVE REPORT
# --------------------------------------------------

output_file = os.path.join(
    OUTPUT_DIR, "shire_river_identification.json"
)

with open(output_file, "w", encoding="utf-8") as file:

    json.dump(results, file, indent=4)

print("\n" + "=" * 65)
print("IDENTIFICATION COMPLETED")
print("Report saved:", output_file)
