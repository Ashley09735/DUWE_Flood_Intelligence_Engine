"""
DUWE AI FLOOD INTELLIGENCE ENGINE
Stage 1 - CHIRPS Data Check Module (data-date aware)

Judges how current the rainfall data is by the DATA IT COVERS, read from the
file name, not by the date the file was last saved. Works offline.

Updated: file choice and date reading now come from chirps_utils.py, so this
check agrees with the alignment script and the dashboard.
  * A full date in the name (chirps_chikwawa_total_2026-11-02) is used as the
    data end date.
  * A season name (..._2023_2024) still means the season ended 30 April 2024.
"""

import os
import sys
import logging
from datetime import datetime, date

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
RAW_DIR = os.path.join(BASE_DIR, r"01_data\raw")
LOG_DIR = os.path.join(BASE_DIR, r"02_scripts\automation\logs")

# Malawi rainy season runs roughly November-April, so a season "ends" on 30 April
SEASON_END_MONTH, SEASON_END_DAY = 4, 30

# Find chirps_utils.py whether this script sits in 02_scripts or in a sub-folder
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE), os.path.join(BASE_DIR, "02_scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from chirps_utils import latest_chirps_file, data_end_date
except ImportError as exc:
    raise SystemExit(
        f"Cannot import chirps_utils ({exc}). Save the updated chirps_utils.py "
        f"in {os.path.join(BASE_DIR, '02_scripts')}."
    )

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Keep console output safe for any character when run from the pipeline
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

log_file = os.path.join(LOG_DIR, f"download_chirps_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    filename=log_file,
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

STATUS_FILE = os.path.join(LOG_DIR, "chirps_status.txt")


def latest_completed_season_end(today):
    """End date of the most recent rainy season that has already finished."""
    this_year_end = date(today.year, SEASON_END_MONTH, SEASON_END_DAY)
    return this_year_end if today > this_year_end else date(today.year - 1, SEASON_END_MONTH, SEASON_END_DAY)


def write_status(status, file="", data_end="", data_age_days="", seasons_behind=""):
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        f.write(f"status={status}\n")
        f.write(f"file={file}\n")
        f.write(f"data_end={data_end}\n")
        f.write(f"age_days={data_age_days}\n")
        f.write(f"seasons_behind={seasons_behind}\n")
        f.write(f"checked={datetime.now().isoformat()}\n")


def main():
    """Return exit code: 0 = check completed (even if data is old), 1 = real failure."""
    logging.info("Starting CHIRPS data check...")
    today = date.today()

    try:
        latest = latest_chirps_file(RAW_DIR, clipped=None)
    except FileNotFoundError:
        logging.warning("No CHIRPS files found in raw folder.")
        print("\nSTATUS: NO CHIRPS DATA FOUND")
        print("Place a CHIRPS rainfall file here:")
        print(RAW_DIR)
        print("(Google Earth Engine or ClimateSERV can export a season total for Chikwawa.)")
        write_status("NO_DATA")
        return 1

    saved = datetime.fromtimestamp(os.path.getmtime(latest))
    end = data_end_date(latest)

    print("\nLatest CHIRPS file found:")
    print(f"  File        : {os.path.basename(latest)}")
    print(f"  File saved  : {saved.strftime('%Y-%m-%d %H:%M')}  (not the data date)")

    if end is None:
        print("  Data period : could not read a year from the file name")
        print("\nSTATUS: UNKNOWN_DATE")
        print("Include the date or season in the file name, e.g. chirps_chikwawa_total_2026-11-02.tif")
        logging.warning(f"Cannot determine data period for {latest}")
        write_status("UNKNOWN_DATE", latest)
        return 0

    age_days = (today - end).days
    expected_end = latest_completed_season_end(today)
    seasons_behind = max(0, expected_end.year - end.year)

    print(f"  Data ends   : {end.isoformat()}  ({age_days} days ago)")
    print(f"  Newest completed season ended: {expected_end.isoformat()}")

    if seasons_behind == 0:
        status = "UP_TO_DATE"
        print("\nSTATUS: UP_TO_DATE - covers the latest completed rainy season")
        logging.info("CHIRPS data covers the latest completed season.")
    elif seasons_behind == 1:
        status = "ONE_SEASON_BEHIND"
        print("\nSTATUS: ONE_SEASON_BEHIND - a newer completed season exists")
        logging.warning("CHIRPS data is one season behind.")
    else:
        status = "OUTDATED"
        print(f"\nSTATUS: OUTDATED - {seasons_behind} seasons behind")
        print("Download a more recent rainy-season total.")
        logging.warning(f"CHIRPS data is {seasons_behind} seasons behind (ends {end}).")

    write_status(status, latest, end.isoformat(), age_days, seasons_behind)
    print(f"\nStatus file written: {STATUS_FILE}")
    logging.info(f"CHIRPS check finished with status: {status}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logging.error(f"CHIRPS check failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)
