"""
DUWE DISASTER INTELLIGENCE
DUWE FLOOD INTELLIGENCE ENGINE

MODULE 1: AUTOMATED DATA MONITORING

Functions:
1. Catchment location detection
2. Live rainfall acquisition
3. Rainfall analysis
4. River gauge monitoring
5. Automated JSON reporting
6. Ten-minute monitoring cycle
"""

import json
import time
import logging

from pathlib import Path
from datetime import datetime, timedelta
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import rasterio

from rasterio.warp import transform


# =====================================================
# 1. PROJECT PATHS
# =====================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parents[1]

CATCHMENT_PATH = (
    PROJECT_ROOT
    / "01_data"
    / "processed"
    / "catchment_duwe.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "01_data"
    / "processed"
    / "forecasting"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "monitoring_latest.json"

GAUGE_CSV = (
    PROJECT_ROOT
    / "01_data"
    / "raw"
    / "river_gauge"
    / "river_gauge_latest.csv"
)

LOG_FILE = OUTPUT_DIR / "monitoring.log"


# =====================================================
# 2. MONITORING SETTINGS
# =====================================================

TIMEZONE = "Africa/Blantyre"

UPDATE_INTERVAL = 600

FORECAST_DAYS = 3

PAST_DAYS = 3

API_URL = "https://api.open-meteo.com/v1/forecast"


# =====================================================
# 3. LOGGING
# =====================================================

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# =====================================================
# 4. GET CATCHMENT LOCATION
# =====================================================

def get_catchment_location():

    if not CATCHMENT_PATH.exists():

        raise FileNotFoundError(
            f"Catchment raster not found: {CATCHMENT_PATH}"
        )

    with rasterio.open(CATCHMENT_PATH) as src:

        bounds = src.bounds

        center_x = (bounds.left + bounds.right) / 2

        center_y = (bounds.bottom + bounds.top) / 2

        if src.crs is None:

            raise ValueError(
                "Catchment raster has no coordinate reference system."
            )

        lon, lat = transform(
            src.crs,
            "EPSG:4326",
            [center_x],
            [center_y]
        )

    return float(lat[0]), float(lon[0])


# =====================================================
# 5. DOWNLOAD LIVE RAINFALL DATA
# =====================================================

def download_rainfall(latitude, longitude, retries=4):

    parameters = {

        "latitude": latitude,

        "longitude": longitude,

        "current": "precipitation",

        "hourly": "precipitation",

        "past_days": PAST_DAYS,

        "forecast_days": FORECAST_DAYS,

        "timezone": TIMEZONE

    }

    url = API_URL + "?" + urlencode(parameters)

    request = Request(
        url,
        headers={"User-Agent": "DUWE-Flood-Intelligence/1.0"}
    )

    last_error = None

    for attempt in range(1, retries + 1):

        try:

            with urlopen(request, timeout=30) as response:

                return json.loads(response.read().decode("utf-8"))

        except HTTPError as error:

            # Retry only on temporary server errors and rate limits
            if error.code not in (429, 500, 502, 503, 504):
                raise

            last_error = error

        except (URLError, TimeoutError) as error:

            last_error = error

        wait = 10 * attempt

        logging.warning(
            f"Download attempt {attempt}/{retries} failed "
            f"({last_error}). Retrying in {wait}s."
        )

        print(f"Download failed ({last_error}). Retrying in {wait}s...")

        time.sleep(wait)

    raise last_error


# =====================================================
# 6. ANALYSE RAINFALL
# =====================================================

def analyse_rainfall(data):

    tz = ZoneInfo(TIMEZONE)

    now = datetime.now(tz)

    current_rain = data["current"]["precipitation"]

    hourly_times = data["hourly"]["time"]

    hourly_rain = data["hourly"]["precipitation"]

    records = []

    for timestamp, rainfall in zip(
        hourly_times,
        hourly_rain
    ):

        if rainfall is None:
            continue

        # Open-Meteo returns local times with no offset,
        # so attach the same timezone used in the request.
        dt = datetime.fromisoformat(timestamp).replace(tzinfo=tz)

        records.append((dt, float(rainfall)))

    past_24h = sum(
        rain for dt, rain in records
        if now - timedelta(hours=24) <= dt <= now
    )

    past_72h = sum(
        rain for dt, rain in records
        if now - timedelta(hours=72) <= dt <= now
    )

    next_24h = sum(
        rain for dt, rain in records
        if now < dt <= now + timedelta(hours=24)
    )

    next_48h = sum(
        rain for dt, rain in records
        if now < dt <= now + timedelta(hours=48)
    )

    return {

        "current_rainfall_mm": float(current_rain or 0),

        "past_24h_mm": round(past_24h, 2),

        "past_72h_mm": round(past_72h, 2),

        "forecast_next_24h_mm": round(next_24h, 2),

        "forecast_next_48h_mm": round(next_48h, 2)

    }


# =====================================================
# 7. READ RIVER GAUGE OBSERVATIONS
# =====================================================

def read_river_gauge():

    if not GAUGE_CSV.exists():

        return {

            "available": False,

            "message": "River gauge data not configured."

        }

    import pandas as pd

    df = pd.read_csv(GAUGE_CSV)

    required_columns = [

        "timestamp",

        "station",

        "water_level_m",

        "discharge_m3s"

    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:

        return {

            "available": False,

            "message": f"Missing gauge columns: {missing}"

        }

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True
    )

    df = df.dropna(subset=["timestamp"])

    if df.empty:

        return {

            "available": False,

            "message": "No valid gauge observations."

        }

    latest = df.sort_values(
        "timestamp"
    ).iloc[-1]

    return {

        "available": True,

        "station": str(latest["station"]),

        "timestamp": latest["timestamp"].isoformat(),

        "water_level_m": float(latest["water_level_m"]),

        "discharge_m3s": float(latest["discharge_m3s"])

    }


# =====================================================
# 8. SAVE MONITORING RESULTS
# =====================================================

def save_results(results):

    temporary_file = OUTPUT_JSON.with_suffix(".tmp")

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    temporary_file.replace(OUTPUT_JSON)


# =====================================================
# 9. RUN ONE MONITORING CYCLE
# =====================================================

def run_monitoring():

    print("\nDUWE FLOOD INTELLIGENCE ENGINE")

    print("AUTOMATED DATA MONITORING")

    print("--------------------------------")

    try:

        latitude, longitude = get_catchment_location()

        print("Catchment location detected.")

        print("Latitude:", round(latitude, 5))

        print("Longitude:", round(longitude, 5))

        rainfall_data = download_rainfall(
            latitude,
            longitude
        )

        rainfall_summary = analyse_rainfall(
            rainfall_data
        )

        gauge_data = read_river_gauge()

        results = {

            "system": "DUWE FLOOD INTELLIGENCE ENGINE",

            "monitoring_timestamp": datetime.now().astimezone().isoformat(),

            "catchment": "Chikwawa",

            "location": {

                "latitude": latitude,

                "longitude": longitude

            },

            "rainfall": rainfall_summary,

            "river_gauge": gauge_data,

            "monitoring_status": "SUCCESS"

        }

        save_results(results)

        logging.info(
            "Monitoring cycle completed successfully."
        )

        print("\nRAINFALL MONITORING")

        for key, value in rainfall_summary.items():

            print(key, ":", value)

        print("\nRIVER GAUGE")

        print(gauge_data)

        print("\nMonitoring results saved.")

        print(OUTPUT_JSON)

        return True

    except Exception as error:

        logging.exception(
            "Monitoring cycle failed."
        )

        print("\nMONITORING ERROR:")

        print(error)

        return False


# =====================================================
# 10. AUTOMATIC MONITORING CONTROLLER
# =====================================================

def main():

    print("\nStarting DUWE automated monitoring...")

    print("Update interval: 10 minutes")

    while True:

        start_time = time.time()

        run_monitoring()

        elapsed = time.time() - start_time

        sleep_time = max(
            0,
            UPDATE_INTERVAL - elapsed
        )

        print(
            f"\nNext monitoring cycle in "
            f"{sleep_time / 60:.1f} minutes."
        )

        time.sleep(sleep_time)


# =====================================================
# 11. PROGRAM ENTRY POINT
# =====================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one monitoring cycle only."
    )

    args = parser.parse_args()

    if args.once:

        run_monitoring()

    else:

        main()
