"""
DUWE FLOOD INTELLIGENCE ENGINE
MODULE 8: EARLY WARNING SCREENING
"""

import json
from pathlib import Path
from datetime import datetime, timezone


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

FORECAST_DIR = (
    PROJECT_ROOT / "01_data" / "processed" / "forecasting"
)

RAIN_FILE = (
    FORECAST_DIR / "rainfall_analysis_latest.json"
)

TIMING_FILE = (
    FORECAST_DIR / "flood_timing_latest.json"
)

ASSET_FILE = (
    FORECAST_DIR / "infrastructure_screening_summary.json"
)

OUTPUT_FILE = (
    FORECAST_DIR / "early_warning_latest.json"
)


def read_json(path):

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:

        return json.load(file)


def main():

    rainfall = read_json(RAIN_FILE)

    timing = read_json(TIMING_FILE)

    assets = read_json(ASSET_FILE)

    rain_status = rainfall.get(
        "rainfall_status",
        "UNKNOWN"
    )

    level = "INFORMATION"

    reasons = []

    if rain_status == "VERY_HIGH_SCREENING":

        level = "ELEVATED_SCREENING"

        reasons.append(
            "Very high rainfall screening threshold reached."
        )

    elif rain_status == "HIGH_SCREENING":

        level = "WATCH_SCREENING"

        reasons.append(
            "High rainfall screening threshold reached."
        )

    elif rain_status == "RAINFALL_PRESENT":

        level = "MONITOR"

        reasons.append(
            "Rainfall is present or forecast."
        )

    if timing.get("status") == "OBSERVATION_BASED_ESTIMATE":

        reasons.append(
            "Recent river gauge observations are available."
        )

        if timing.get("message", "").lower().find(
            "at or above"
        ) >= 0:

            level = "URGENT_REVIEW"

            reasons.append(
                "River level is at or above a configured threshold."
            )

    result = {

        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "catchment": "Chikwawa",

        "screening_level": level,

        "reasons": reasons,

        "rainfall_status": rain_status,

        "gauge_status": timing.get(
            "status",
            "UNAVAILABLE"
        ),

        "high_risk_assets": assets.get(
            "high_or_very_high_assets"
        ),

        "public_warning_authorised": False,

        "message": (
            "Technical decision-support screening only. "
            "Verify observations and follow official DoDMA "
            "and DCCMS warning procedures."
        )

    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(result, file, indent=4)

    print("Early warning screening completed.")

    print("Screening level:", level)


if __name__ == "__main__":
    main()