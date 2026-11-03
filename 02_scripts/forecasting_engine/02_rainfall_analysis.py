"""
DUWE FLOOD INTELLIGENCE ENGINE
MODULE 2: RAINFALL ANALYSIS
"""

import json
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

FORECAST_DIR = (
    PROJECT_ROOT / "01_data" / "processed" / "forecasting"
)

INPUT_FILE = FORECAST_DIR / "monitoring_latest.json"
OUTPUT_FILE = FORECAST_DIR / "rainfall_analysis_latest.json"

# Provisional screening thresholds, not official warning thresholds.
THRESHOLDS = {
    "high_24h": 50.0,
    "very_high_24h": 100.0
}


def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def classify_rainfall(observed, forecast):

    if observed >= THRESHOLDS["very_high_24h"]:
        return "VERY_HIGH_SCREENING"

    if forecast >= THRESHOLDS["very_high_24h"]:
        return "VERY_HIGH_SCREENING"

    if observed >= THRESHOLDS["high_24h"]:
        return "HIGH_SCREENING"

    if forecast >= THRESHOLDS["high_24h"]:
        return "HIGH_SCREENING"

    if observed > 0 or forecast > 0:
        return "RAINFALL_PRESENT"

    return "LOW_RAINFALL"


def main():

    data = load_data()

    rainfall = data["rainfall"]

    observed_24h = float(rainfall["past_24h_mm"])
    observed_72h = float(rainfall["past_72h_mm"])

    forecast_24h = float(
        rainfall["forecast_next_24h_mm"]
    )

    forecast_48h = float(
        rainfall["forecast_next_48h_mm"]
    )

    status = classify_rainfall(
        observed_24h,
        forecast_24h
    )

    result = {

        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "catchment": "Chikwawa",

        "observed_rainfall_mm": {

            "past_24h": observed_24h,

            "past_72h": observed_72h

        },

        "forecast_rainfall_mm": {

            "next_24h": forecast_24h,

            "next_48h": forecast_48h

        },

        "rainfall_status": status,

        "thresholds": THRESHOLDS,

        "limitations": (
            "Provisional rainfall screening. "
            "Thresholds require local validation."
        )

    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(result, file, indent=4)

    print("Rainfall analysis completed.")
    print("Rainfall status:", status)


if __name__ == "__main__":
    main()