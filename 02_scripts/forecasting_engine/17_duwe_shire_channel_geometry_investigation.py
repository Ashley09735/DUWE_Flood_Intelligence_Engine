
"""
DUWE DISASTER INTELLIGENCE
MODULE 17: SHIRE RIVER CHANNEL GEOMETRY INVESTIGATION

Purpose:
1. Extract the selected Shire River centreline.
2. Inspect available river width information.
3. Generate perpendicular terrain investigation transects.
4. Sample DEM elevations along each transect.
5. Export preliminary cross-section investigation datasets.

IMPORTANT:
These are terrain investigation transects.
They are NOT surveyed river-channel cross-sections.
"""

import os
import csv
import math

import fiona
import rasterio
import numpy as np

from shapely.geometry import shape, LineString, MultiLineString


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"

RIVER_FILE = os.path.join(
    BASE_DIR,
    "01_data",
    "raw",
    "mw_rivers.gpkg"
)

DEM_FILE = os.path.join(
    BASE_DIR,
    "01_data",
    "processed",
    "dem_conditioned_duwe.tif"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "01_data",
    "processed",
    "forecasting"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. INVESTIGATION SETTINGS
# --------------------------------------------------

TARGET_OSM_ID = "24896284"

CROSS_SECTION_SPACING_M = 1000

TRANSECT_HALF_WIDTH_M = 500

SAMPLE_INTERVAL_M = 30


# --------------------------------------------------
# 3. DISPLAY MODULE INFORMATION
# --------------------------------------------------

print("\nDUWE DISASTER INTELLIGENCE")
print("MODULE 17: SHIRE RIVER CHANNEL GEOMETRY INVESTIGATION")
print("=" * 60)


# --------------------------------------------------
# 4. FIND THE SELECTED RIVER FEATURE
# --------------------------------------------------

print("\nSearching for selected Shire River feature...")

river_geometry = None
river_properties = None
selected_layer = None

layers = fiona.listlayers(RIVER_FILE)

for layer_name in layers:

    with fiona.open(RIVER_FILE, layer=layer_name) as source:

        for feature in source:

            properties = dict(feature["properties"])

            osm_value = properties.get("osm_id")

            if osm_value is not None and str(osm_value) == TARGET_OSM_ID:

                river_geometry = shape(feature["geometry"])
                river_properties = properties
                selected_layer = layer_name

                break

    if river_geometry is not None:
        break


if river_geometry is None:
    raise ValueError(
        f"River feature OSM ID {TARGET_OSM_ID} was not found."
    )


# --------------------------------------------------
# 5. SELECT THE LONGEST RIVER COMPONENT
# --------------------------------------------------

if isinstance(river_geometry, MultiLineString):

    river_line = max(
        river_geometry.geoms,
        key=lambda component: component.length
    )

elif isinstance(river_geometry, LineString):

    river_line = river_geometry

else:

    raise TypeError(
        "Selected river geometry is not a LineString or MultiLineString."
    )


river_length_m = river_line.length

river_width = river_properties.get("width")

river_name = river_properties.get("name")

river_class = river_properties.get("fclass")

print("\nRIVER FEATURE INFORMATION")
print("-" * 60)

print(f"OSM ID: {TARGET_OSM_ID}")
print(f"Layer: {selected_layer}")
print(f"River name: {river_name}")
print(f"River class: {river_class}")
print(f"Mapped width attribute: {river_width}")
print(f"Selected component length: {river_length_m / 1000:.3f} km")


# --------------------------------------------------
# 6. PREPARE OUTPUT FILES
# --------------------------------------------------

samples_file = os.path.join(
    OUTPUT_DIR,
    "shire_river_cross_section_dem_samples.csv"
)

transects_file = os.path.join(
    OUTPUT_DIR,
    "shire_river_cross_section_candidates.csv"
)

report_file = os.path.join(
    OUTPUT_DIR,
    "shire_river_channel_geometry_report.txt"
)


# --------------------------------------------------
# 7. GENERATE CROSS-SECTION CANDIDATES
# --------------------------------------------------

print("\nGenerating terrain investigation transects...")

sample_records = []
transect_records = []

chainages = np.arange(
    0,
    river_length_m,
    CROSS_SECTION_SPACING_M
)

# Include the final endpoint if it is not already included.
if len(chainages) == 0 or chainages[-1] < river_length_m:
    chainages = np.append(chainages, river_length_m)

with rasterio.open(DEM_FILE) as dem:

    print("\nDEM INFORMATION")
    print("-" * 60)

    print(f"CRS: {dem.crs}")
    print(f"Resolution: {dem.res}")
    print(f"Width: {dem.width}")
    print(f"Height: {dem.height}")
    print(f"NoData: {dem.nodata}")

    for section_number, chainage in enumerate(chainages, start=1):

        # Locate the centre of the cross-section.
        centre = river_line.interpolate(float(chainage))

        # Find two nearby points to estimate the centreline direction.
        before_distance = max(
            0,
            float(chainage) - 15
        )

        after_distance = min(
            river_length_m,
            float(chainage) + 15
        )

        before = river_line.interpolate(before_distance)
        after = river_line.interpolate(after_distance)

        dx = after.x - before.x
        dy = after.y - before.y

        tangent_length = math.sqrt(dx ** 2 + dy ** 2)

        if tangent_length == 0:
            print(
                f"Skipping section {section_number}: "
                "cannot calculate centreline direction."
            )
            continue

        # Calculate perpendicular unit vector.
        normal_x = -dy / tangent_length
        normal_y = dx / tangent_length

        section_id = f"XS_{section_number:03d}"

        section_points = []

        offsets = np.arange(
            -TRANSECT_HALF_WIDTH_M,
            TRANSECT_HALF_WIDTH_M + SAMPLE_INTERVAL_M,
            SAMPLE_INTERVAL_M
        )

        for offset in offsets:

            x = centre.x + normal_x * offset
            y = centre.y + normal_y * offset

            section_points.append((x, y))

        # Sample the DEM at every transect point.
        elevation_values = list(
            dem.sample(section_points)
        )

        valid_count = 0
        missing_count = 0

        elevations = []

        for point_index, (point, value) in enumerate(
            zip(section_points, elevation_values)
        ):

            x, y = point

            elevation = float(value[0])

            is_valid = (
                np.isfinite(elevation)
                and elevation != dem.nodata
            )

            if is_valid:
                valid_count += 1
                elevations.append(elevation)
            else:
                missing_count += 1
                elevation = None

            sample_records.append({
                "section_id": section_id,
                "chainage_m": round(float(chainage), 2),
                "offset_m": round(float(offsets[point_index]), 2),
                "easting": round(x, 3),
                "northing": round(y, 3),
                "elevation_m": elevation,
                "valid_elevation": is_valid
            })

        start_point = section_points[0]
        end_point = section_points[-1]

        transect_records.append({
            "section_id": section_id,
            "chainage_m": round(float(chainage), 2),
            "chainage_km": round(float(chainage) / 1000, 3),
            "centre_easting": round(centre.x, 3),
            "centre_northing": round(centre.y, 3),
            "start_easting": round(start_point[0], 3),
            "start_northing": round(start_point[1], 3),
            "end_easting": round(end_point[0], 3),
            "end_northing": round(end_point[1], 3),
            "transect_length_m": round(
                math.dist(start_point, end_point), 2
            ),
            "valid_dem_samples": valid_count,
            "missing_dem_samples": missing_count,
            "minimum_dem_elevation_m": (
                min(elevations) if elevations else None
            ),
            "maximum_dem_elevation_m": (
                max(elevations) if elevations else None
            )
        })

        print(
            f"{section_id}: "
            f"Chainage {chainage / 1000:.2f} km | "
            f"Valid samples: {valid_count} | "
            f"Missing: {missing_count}"
        )


# --------------------------------------------------
# 8. SAVE DEM SAMPLE DATA
# --------------------------------------------------

if sample_records:

    with open(
        samples_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(sample_records[0].keys())
        )

        writer.writeheader()
        writer.writerows(sample_records)


# --------------------------------------------------
# 9. SAVE CROSS-SECTION CANDIDATES
# --------------------------------------------------

if transect_records:

    with open(
        transects_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(transect_records[0].keys())
        )

        writer.writeheader()
        writer.writerows(transect_records)


# --------------------------------------------------
# 10. CALCULATE SUMMARY STATISTICS
# --------------------------------------------------

total_sections = len(transect_records)

total_samples = len(sample_records)

valid_samples = sum(
    record["valid_elevation"]
    for record in sample_records
)

missing_samples = total_samples - valid_samples

valid_percentage = (
    valid_samples / total_samples * 100
    if total_samples else 0
)


# --------------------------------------------------
# 11. SAVE INVESTIGATION REPORT
# --------------------------------------------------

with open(
    report_file,
    "w",
    encoding="utf-8"
) as report:

    report.write("DUWE DISASTER INTELLIGENCE\n")
    report.write("MODULE 17: SHIRE RIVER CHANNEL GEOMETRY INVESTIGATION\n")
    report.write("=" * 60 + "\n\n")

    report.write(f"OSM ID: {TARGET_OSM_ID}\n")
    report.write(f"River name: {river_name}\n")
    report.write(f"River class: {river_class}\n")
    report.write(f"Mapped width attribute: {river_width}\n")
    report.write(f"Selected component length: {river_length_m / 1000:.3f} km\n\n")

    report.write(f"Cross-section spacing: {CROSS_SECTION_SPACING_M} m\n")
    report.write(
        f"Investigation corridor width: "
        f"{TRANSECT_HALF_WIDTH_M * 2} m\n"
    )
    report.write(f"DEM sample interval: {SAMPLE_INTERVAL_M} m\n\n")

    report.write(f"Candidate cross-sections: {total_sections}\n")
    report.write(f"Total DEM sample locations: {total_samples}\n")
    report.write(f"Valid DEM samples: {valid_samples}\n")
    report.write(f"Missing DEM samples: {missing_samples}\n")
    report.write(f"Valid sample percentage: {valid_percentage:.2f}%\n\n")

    report.write(
        "IMPORTANT: These transects are preliminary terrain "
        "investigation lines. They do not represent surveyed "
        "channel cross-sections or verified river widths.\n"
    )


# --------------------------------------------------
# 12. DISPLAY FINAL RESULTS
# --------------------------------------------------

print("\nMODULE 17 RESULTS")
print("=" * 60)

print(f"River length: {river_length_m / 1000:.3f} km")
print(f"Candidate cross-sections: {total_sections}")
print(f"Total DEM sample locations: {total_samples}")
print(f"Valid DEM samples: {valid_samples}")
print(f"Missing DEM samples: {missing_samples}")
print(f"Valid sample percentage: {valid_percentage:.2f}%")

print("\nOUTPUT FILES")
print("-" * 60)

print(samples_file)
print(transects_file)
print(report_file)

print("\nMODULE 17 COMPLETED")
