"""
DUWE FLOOD INTELLIGENCE ENGINE
MODULE 6: TERRAIN FLOOD SCREENING

This is NOT a hydraulic inundation simulation.
"""

import json
from pathlib import Path

import numpy as np
import rasterio


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "01_data" / "processed"

FORECAST_DIR = DATA_DIR / "forecasting"

DEM_FILE = DATA_DIR / "dem_conditioned_duwe.tif"

CATCHMENT_FILE = DATA_DIR / "catchment_duwe.tif"

OUTPUT_RASTER = (
    FORECAST_DIR / "terrain_flood_screening.tif"
)

OUTPUT_JSON = (
    FORECAST_DIR / "terrain_screening_summary.json"
)


def main():

    FORECAST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with rasterio.open(DEM_FILE) as dem_src:

        dem = dem_src.read(1, masked=True)

        profile = dem_src.profile.copy()

    with rasterio.open(CATCHMENT_FILE) as catchment_src:

        catchment = catchment_src.read(
            1,
            masked=True
        )

        if (
            dem_src.crs != catchment_src.crs
            or dem_src.transform != catchment_src.transform
            or dem_src.shape != catchment_src.shape
        ):
            raise ValueError(
                "DEM and catchment must have matching grids."
            )

    valid = (
        ~np.ma.getmaskarray(dem)
        & ~np.ma.getmaskarray(catchment)
        & np.isfinite(dem.data)
        & (catchment.filled(0) != 0)
    )

    values = dem.data[valid]

    if values.size == 0:
        raise ValueError("No valid overlapping terrain cells.")

    p10, p25, p50 = np.percentile(
        values,
        [10, 25, 50]
    )

    output = np.zeros(
        dem.shape,
        dtype="uint8"
    )

    output[valid & (dem.data <= p10)] = 1

    output[
        valid & (dem.data > p10) & (dem.data <= p25)
    ] = 2

    output[
        valid & (dem.data > p25) & (dem.data <= p50)
    ] = 3

    output[valid & (dem.data > p50)] = 4

    profile.update(
        dtype="uint8",
        count=1,
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(
        OUTPUT_RASTER,
        "w",
        **profile
    ) as dst:

        dst.write(output, 1)

    result = {

        "terrain_screening_raster": str(OUTPUT_RASTER),

        "classes": {

            "1": "Lowest 10 percent elevation",

            "2": "10th to 25th percentile",

            "3": "25th to 50th percentile",

            "4": "Above 50th percentile",

            "0": "Outside catchment or NoData"

        },

        "status": "RELATIVE_TERRAIN_SCREENING_ONLY",

        "warning": (
            "This is not a predicted flood extent or flood depth map."
        )

    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:

        json.dump(result, file, indent=4)

    print("Terrain screening completed.")

    print("Output:", OUTPUT_RASTER)


if __name__ == "__main__":
    main()