"""
DUWE FLOOD INTELLIGENCE ENGINE
MODULE 5: FLOOD TIMING ASSESSMENT
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

FORECAST_DIR = (
    PROJECT_ROOT / "01_data" / "processed" / "forecasting"
)

GAUGE_FILE = (
    PROJECT_ROOT
    / "01_data"
    / "raw"
    / "river_gauge"
    / "river_gauge_latest.csv"
)

CONFIG_FILE = FORECAST_DIR / "gauge_thresholds.json"

OUTPUT_FILE = FORECAST_DIR / "flood_timing_latest.json"


def main():

    result = {

        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "status": "UNAVAILABLE",

        "estimated_time_to_threshold_hours": None

    }

    if not GAUGE_FILE.exists() or not CONFIG_FILE.exists():

        result["message"] = (
            "Verified gauge observations and approved thresholds required."
        )

    else:

        with open(CONFIG_FILE, "r", encoding="utf-8") as file:

            config = json.load(file)

        df = pd.read_csv(GAUGE_FILE)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce"
        )

        df["water_level_m"] = pd.to_numeric(
            df["water_level_m"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["timestamp", "water_level_m"]
        ).sort_values("timestamp")

        if len(df) >= 3:

            latest = df.iloc[-1]

            previous = df.iloc[-3]

            elapsed = (
                latest["timestamp"] -
                previous["timestamp"]
            ).total_seconds() / 3600

            age = (
                pd.Timestamp.now(tz="UTC") -
                latest["timestamp"]
            ).total_seconds() / 3600

            if elapsed > 0 and 0 <= age <= 6:

                rate = (
                    latest["water_level_m"] -
                    previous["water_level_m"]
                ) / elapsed

                if rate > 0:

                    thresholds = {

                        "watch": config["watch_level_m"],

                        "alert": config["alert_level_m"],

                        "emergency": config["emergency_level_m"]

                    }

                    future = [

                        (name, (level - latest["water_level_m"]) / rate)

                        for name, level in thresholds.items()

                        if level > latest["water_level_m"]

                    ]

                    if future:

                        name, hours = min(
                            future,
                            key=lambda item: item[1]
                        )

                        result.update({

                            "status": "OBSERVATION_BASED_ESTIMATE",

                            "station": config.get(
                                "station_name", "Unknown"
                            ),

                            "water_level_m": float(
                                latest["water_level_m"]
                            ),

                            "level_change_m_per_hour": round(
                                rate, 4
                            ),

                            "next_threshold": name,

                            "estimated_time_to_threshold_hours": round(
                                hours, 2
                            )

                        })

                        result["message"] = (
                            "Linear extrapolation only; not a calibrated forecast."
                        )

                    else:

                        result["message"] = (
                            "Current level is at or above configured thresholds."
                        )

                else:

                    result["message"] = (
                        "No rising trend in the selected gauge observations."
                    )

            else:

                result["message"] = (
                    "Gauge data is stale or observation interval is invalid."
                )

        else:

            result["message"] = (
                "At least three valid gauge observations are required."
            )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(result, file, indent=4)

    print("Flood timing assessment completed.")

    print(result["message"])


if __name__ == "__main__":
    main()