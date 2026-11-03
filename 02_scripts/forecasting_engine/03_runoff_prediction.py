"""
DUWE FLOOD INTELLIGENCE ENGINE
MODULE 3: RUNOFF PREDICTION
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import rasterio


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "01_data" / "processed"

FORECAST_DIR = DATA_DIR / "forecasting"

CATCHMENT = DATA_DIR / "catchment_duwe.tif"

INPUT_FILE = FORECAST_DIR / "rainfall_analysis_latest.json"

OUTPUT_FILE = FORECAST_DIR / "runoff_latest.json"


# PLACEHOLDER VALUE.
# Replace with a catchment-specific calibrated Curve Number.
CURVE_NUMBER = 75.0


def get_catchment_area():

    with rasterio.open(CATCHMENT) as src:

        raster = src.read(1, masked=True)

        valid_cells = raster.count()

        if not src.crs or not src.crs.is_projected:
            raise ValueError(
                "Catchment raster must have a projected CRS."
            )

        pixel_area = abs(
            src.transform.a * src.transform.e
        )

        area_km2 = (
            valid_cells * pixel_area
        ) / 1_000_000

    return area_km2


def calculate_runoff(rainfall):

    S = (25400 / CURVE_NUMBER) - 254

    initial_abstraction = 0.2 * S

    if rainfall <= initial_abstraction:

        return 0.0

    runoff = (
        (rainfall - initial_abstraction) ** 2
    ) / (
        rainfall - initial_abstraction + S
    )

    return runoff


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        rainfall = json.load(file)

    area = get_catchment_area()

    observed = rainfall[
        "observed_rainfall_mm"
    ]["past_24h"]

    forecast = rainfall[
        "forecast_rainfall_mm"
    ]["next_24h"]

    observed_runoff = calculate_runoff(observed)

    forecast_runoff = calculate_runoff(forecast)

    observed_volume = (
        observed_runoff / 1000
    ) * area * 1_000_000

    forecast_volume = (
        forecast_runoff / 1000
    ) * area * 1_000_000

    result = {

        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "catchment_area_km2": round(area, 3),

        "curve_number": CURVE_NUMBER,

        "observed_runoff_mm": round(
            observed_runoff, 2
        ),

        "forecast_runoff_mm": round(
            forecast_runoff, 2
        ),

        "observed_runoff_volume_m3": round(
            observed_volume, 2
        ),

        "forecast_runoff_volume_m3": round(
            forecast_volume, 2
        ),

        "status": "PRELIMINARY_UNCALIBRATED_ESTIMATE"

    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(result, file, indent=4)

    print("Runoff estimation completed.")

    print("Catchment area:", round(area, 3), "km2")

    print("Forecast runoff:", round(forecast_runoff, 2), "mm")


if __name__ == "__main__":
    main()