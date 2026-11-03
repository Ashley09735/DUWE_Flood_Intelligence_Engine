
"""
DUWE DISASTER INTELLIGENCE
MODULE 12: RIVER GEOMETRY INSPECTOR

Purpose:
1. Inspect Shire River and other river datasets.
2. Identify river attribute fields.
3. Inspect geometry types and coordinate systems.
4. Inspect catchment raster values.
5. Prepare for pilot hydraulic geometry selection.
"""

from pathlib import Path
from collections import Counter
import json

import fiona
import rasterio
import numpy as np

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(
    r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
)

RAW = ROOT / "01_data" / "raw"
PROCESSED = ROOT / "01_data" / "processed"

CATCHMENT = PROCESSED / "catchment_duwe.tif"

OUTPUT = PROCESSED / "forecasting"
OUTPUT.mkdir(parents=True, exist_ok=True)

REPORT = OUTPUT / "river_geometry_inspection.json"

# --------------------------------------------------
# 2. INSPECT VECTOR DATASETS
# --------------------------------------------------

river_files = [
    RAW / "ck_rivers.shp",
    RAW / "mw_rivers.gpkg",
    RAW / "mw_water_bodies.gpkg",
    RAW / "mw_outlets.gpkg"
]

report = {
    "river_datasets": [],
    "catchment_raster": {}
}

print("=" * 70)
print("DUWE RIVER GEOMETRY INSPECTION")
print("=" * 70)

for path in river_files:

    if not path.exists():

        print("\nMISSING:", path.name)
        continue

    print("\n" + "-" * 70)
    print("DATASET:", path.name)

    try:

        layers = (
            fiona.listlayers(path)
            if path.suffix.lower() == ".gpkg"
            else [None]
        )

        for layer in layers:

            with fiona.open(path, layer=layer) as src:

                print("\nLayer:", layer or "Default")
                print("Feature count:", len(src))
                print("Geometry type:", src.schema["geometry"])
                print("CRS:", src.crs)

                print("Bounds:", src.bounds)

                print("Attribute fields:")

                fields = list(src.schema["properties"].keys())

                for field in fields:
                    print("  ", field)

                samples = []

                for index, feature in enumerate(src):

                    if index >= 5:
                        break

                    properties = dict(feature["properties"])

                    samples.append(properties)

                    print(
                        "Sample feature",
                        index + 1,
                        ":",
                        properties
                    )

                report["river_datasets"].append({
                    "file": str(path),
                    "layer": layer,
                    "feature_count": len(src),
                    "geometry_type": src.schema["geometry"],
                    "crs": str(src.crs),
                    "bounds": list(src.bounds),
                    "attribute_fields": fields,
                    "sample_attributes": samples
                })

    except Exception as error:

        print("INSPECTION ERROR:", error)

# --------------------------------------------------
# 3. INSPECT CATCHMENT RASTER VALUES
# --------------------------------------------------

print("\n" + "=" * 70)
print("CATCHMENT RASTER VALUE INSPECTION")
print("=" * 70)

counts = Counter()

with rasterio.open(CATCHMENT) as src:

    print("CRS:", src.crs)
    print("NoData:", src.nodata)
    print("Dimensions:", src.width, "x", src.height)

    for _, window in src.block_windows(1):

        data = src.read(1, window=window, masked=True)

        values, frequencies = np.unique(
            data.compressed(),
            return_counts=True
        )

        for value, frequency in zip(values, frequencies):

            counts[str(float(value))] += int(frequency)

for value, count in sorted(counts.items()):

    print("Value:", value, "| Cells:", count)

report["catchment_raster"] = {
    "value_counts": dict(counts),
    "interpretation": (
        "Inspect raster values to determine which value represents "
        "the actual catchment."
    )
}

# --------------------------------------------------
# 4. IDENTIFY POSSIBLE SHIRE RIVER ATTRIBUTES
# --------------------------------------------------

print("\n" + "=" * 70)
print("POSSIBLE RIVER IDENTIFICATION FIELDS")
print("=" * 70)

keywords = [
    "name",
    "river",
    "water",
    "stream",
    "type",
    "class"
]

for dataset in report["river_datasets"]:

    matching = [
        field
        for field in dataset["attribute_fields"]
        if any(word in field.lower() for word in keywords)
    ]

    print("\nFile:", Path(dataset["file"]).name)
    print("Potential identification fields:", matching)

# --------------------------------------------------
# 5. SAVE REPORT
# --------------------------------------------------

REPORT.write_text(
    json.dumps(report, indent=4, default=str),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("REPORT SAVED:", REPORT)
print("=" * 70)
