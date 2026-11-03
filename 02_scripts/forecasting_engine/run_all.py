"""
DUWE FLOOD INTELLIGENCE ENGINE
RUN ALL: Module 1 (data) -> controller (modules 2 to 8), every 10 minutes.

Put this file in the same folder as 01_data_monitoring.py and
09_forecasting_controller.py, then run:

    python run_all.py          (continuous, every 10 minutes)
    python run_all.py --once   (one cycle only)
"""

import sys
import json
import time
import logging
import subprocess

from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

FORECAST_DIR = PROJECT_ROOT / "01_data" / "processed" / "forecasting"
FORECAST_DIR.mkdir(parents=True, exist_ok=True)

MONITORING_JSON = FORECAST_DIR / "monitoring_latest.json"

CYCLE_SECONDS = 600
MAX_DATA_AGE_MINUTES = 30
STEP_TIMEOUT_SECONDS = 300

logging.basicConfig(
    filename=str(FORECAST_DIR / "run_all.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def run_step(script_name):

    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script_name), "--once"],
        capture_output=True,
        text=True,
        timeout=STEP_TIMEOUT_SECONDS
    )

    logging.info("%s exit=%s\n%s", script_name, result.returncode, result.stdout)

    if result.returncode != 0:
        logging.error("%s failed:\n%s", script_name, result.stderr)

    return result.returncode == 0


def monitoring_is_fresh():

    """Module 1 exits normally even when a download fails, so check
    that monitoring_latest.json was actually updated recently."""

    if not MONITORING_JSON.exists():
        return False

    with open(MONITORING_JSON, "r", encoding="utf-8") as file:
        stamp = json.load(file).get("monitoring_timestamp")

    if not stamp:
        return False

    age = datetime.now(timezone.utc) - datetime.fromisoformat(stamp)

    return age.total_seconds() / 60 <= MAX_DATA_AGE_MINUTES


def run_cycle():

    print(f"\nCycle started: {datetime.now():%Y-%m-%d %H:%M:%S}")

    run_step("01_data_monitoring.py")

    if not monitoring_is_fresh():
        message = "Monitoring data is missing or stale. Forecasting skipped."
        print(message)
        logging.warning(message)
        return

    if run_step("09_forecasting_controller.py"):
        print("Cycle completed.")
    else:
        print("Forecasting controller failed. See run_all.log.")


def main():

    once = "--once" in sys.argv

    while True:

        start = time.time()

        try:
            run_cycle()
        except Exception:
            logging.exception("Cycle crashed.")
            print("Cycle crashed. See run_all.log.")

        if once:
            break

        time.sleep(max(0, CYCLE_SECONDS - (time.time() - start)))


if __name__ == "__main__":
    main()
