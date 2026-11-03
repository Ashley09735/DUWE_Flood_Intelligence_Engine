"""
DUWE FLOOD INTELLIGENCE ENGINE
MODULE 4: DISCHARGE INDICATOR
"""

import json
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

FORECAST_DIR = (
    PROJECT_ROOT / "01_data" / "processed" / "forecasting"
)

INPUT_FILE = FORECAST_DIR / "runoff_latest.json"

OUTPUT_FILE = (
    FORECAST_DIR / "discharge_indicator_latest.json"
)

# Placeholder assumption, not measured catchment response time.
RUNOFF_DURATION_HOURS = 12


def calculate_flow(volume):

    duration_seconds = (
        RUNOFF_DURATION_HOURS * 3600
    )

    return volume / duration_seconds


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        runoff = json.load(file)

    observed_volume = runoff[
        "observed_runoff_volume_m3"
    ]

    forecast_volume = runoff[
        "forecast_runoff_volume_m3"
    ]

    observed_flow = calculate_flow(
        observed_volume
    )

    forecast_flow = calculate_flow(
        forecast_volume
    )

    result = {

        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "observed_flow_indicator_m3s": round(
            observed_flow, 3
        ),

        "forecast_flow_indicator_m3s": round(
            forecast_flow, 3
        ),

        "assumed_duration_hours": RUNOFF_DURATION_HOURS,

        "status": "NON_OPERATIONAL_FLOW_INDICATOR",

        "warning": (
            "These are catchment-average runoff indicators, "
            "not measured or predicted Shire River discharge."
        )

    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(result, file, indent=4)

    print("Discharge indicators generated.")


if __name__ == "__main__":
    main()