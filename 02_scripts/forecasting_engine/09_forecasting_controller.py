"""
DUWE FLOOD INTELLIGENCE ENGINE
MODULE 9: FORECASTING CONTROLLER

Runs Modules 2 to 8.
Monitoring Module 1 must also be running separately.
"""

import time
import logging
import subprocess
import sys

from pathlib import Path
from datetime import datetime


SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parents[1]

FORECAST_DIR = (
    PROJECT_ROOT / "01_data" / "processed" / "forecasting"
)

FORECAST_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_FILE = FORECAST_DIR / "controller.log"

UPDATE_INTERVAL = 600


MODULES = [

    "02_rainfall_analysis.py",

    "03_runoff_prediction.py",

    "04_discharge_forecasting.py",

    "05_flood_timing.py",

    "06_flood_inundation.py",

    "07_infrastructure_prediction.py",

    "08_early_warning.py"

]


logging.basicConfig(

    filename=str(LOG_FILE),

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)


def run_module(filename):

    script = SCRIPT_DIR / filename

    if not script.exists():

        raise FileNotFoundError(
            f"Module not found: {script}"
        )

    print("\nRunning:", filename)

    result = subprocess.run(

        [sys.executable, str(script)],

        capture_output=True,

        text=True

    )

    if result.stdout:

        print(result.stdout)

    if result.stderr:

        print(result.stderr)

    if result.returncode != 0:

        logging.error(
            "%s failed: %s",
            filename,
            result.stderr
        )

        raise RuntimeError(
            f"{filename} failed."
        )

    logging.info(
        "%s completed successfully",
        filename
    )


def run_forecasting_cycle():

    monitoring_file = (
        FORECAST_DIR / "monitoring_latest.json"
    )

    if not monitoring_file.exists():

        raise FileNotFoundError(
            "Run Module 1 first. monitoring_latest.json is missing."
        )

    print("\nDUWE FLOOD FORECASTING ENGINE")

    print(
        "Cycle started:",
        datetime.now().isoformat(timespec="seconds")
    )

    for module in MODULES:

        run_module(module)

    print("\nForecasting cycle completed.")


def main():

    print("DUWE AUTOMATED FORECASTING CONTROLLER")

    print("Update interval: 10 minutes")

    while True:

        start = time.time()

        try:

            run_forecasting_cycle()

        except Exception as error:

            logging.exception(
                "Forecasting cycle failed."
            )

            print("\nERROR:", error)

        elapsed = time.time() - start

        remaining = max(
            0,
            UPDATE_INTERVAL - elapsed
        )

        print(
            f"\nNext cycle in {remaining / 60:.1f} minutes."
        )

        time.sleep(remaining)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--once",
        action="store_true"
    )

    args = parser.parse_args()

    if args.once:

        run_forecasting_cycle()

    else:

        main()