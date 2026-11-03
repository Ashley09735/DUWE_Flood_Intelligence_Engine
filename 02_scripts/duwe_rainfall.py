# DUWE AI FLOOD INTELLIGENCE ENGINE - Automated CHIRPS Download (updated)
#
# Changes from the first version:
#   1. The rainfall window is now ROLLING (last WINDOW_DAYS days of data).
#      The old version had the start date fixed at 2025-11-01, so it would
#      keep adding months and never move on to a new season.
#   2. The date in the file name is the LAST DAY THAT HAS DATA, not the day
#      the script ran. CHIRPS arrives a few days late, so the old name could
#      claim data that did not exist yet. The dashboard reads this date.
#   3. The script waits for the export and reports success or failure,
#      instead of printing "success" as soon as the task is queued.
#
# The file is still exported to Google Drive. Copy or sync it from
# Drive folder DUWE_Rainfall into 01_data\raw before running the alignment.

import datetime
import time

import ee

# Your verified Google Cloud Project ID
PROJECT_ID = 'capable-country-434808-n6'

WINDOW_DAYS = 365          # length of the rainfall total, in days
WAIT_FOR_EXPORT = True     # poll Earth Engine until the export finishes
MAX_WAIT_SECONDS = 900     # stop waiting after 15 minutes (the task keeps running)

try:
    ee.Initialize(project=PROJECT_ID)
except Exception as e:
    print(f"Initialization fallback triggered: {e}")
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

# Approximate bounding box for Chikwawa catchment
chikwawa = ee.Geometry.Rectangle([34.5, -16.5, 35.3, -15.7])

COLLECTION_ID = "UCSB-CHC/CHIRPS/V3/DAILY_SAT"


def chirps_between(start, end_inclusive):
    """Daily CHIRPS from start to end_inclusive (Earth Engine end is exclusive)."""
    return (
        ee.ImageCollection(COLLECTION_ID)
        .filterDate(start.isoformat(),
                    (end_inclusive + datetime.timedelta(days=1)).isoformat())
        .filterBounds(chikwawa)
        .select("precipitation")
    )


# 1. Find the last day that actually has data (CHIRPS lags by a few days)
today = datetime.date.today()
data_end = today
try:
    recent = chirps_between(today - datetime.timedelta(days=45), today)
    last_ms = recent.aggregate_max("system:time_start").getInfo()
    if last_ms:
        data_end = datetime.datetime.fromtimestamp(
            last_ms / 1000, tz=datetime.timezone.utc).date()
    else:
        print("No recent CHIRPS images found; falling back to today's date.")
except Exception as e:
    print(f"Could not read the latest data date ({e}); using today's date.")

data_start = data_end - datetime.timedelta(days=WINDOW_DAYS - 1)
end_str = data_end.isoformat()
start_str = data_start.isoformat()

# 2. Total rainfall for the window
chirps = chirps_between(data_start, data_end)
totalRain = chirps.sum().clip(chikwawa)

# 3. Export to Google Drive. The date in the file name is the data end date.
export_description = f"CHIRPS_Chikwawa_Latest_Run_{today.strftime('%Y_%m_%d')}"
file_prefix = f"chirps_chikwawa_total_{end_str}"

task = ee.batch.Export.image.toDrive(
    image=totalRain,
    description=export_description,
    folder="DUWE_Rainfall",
    fileNamePrefix=file_prefix,
    region=chikwawa,
    scale=5000,  # 5 km resolution
    crs="EPSG:4326",
    maxPixels=1e13,
)

task.start()
print(f"Export queued. Rainfall total {start_str} to {end_str} "
      f"({WINDOW_DAYS} days).")
print(f"Expected Drive file: DUWE_Rainfall/{file_prefix}.tif")

# 4. Wait and report the real outcome
if WAIT_FOR_EXPORT:
    waited = 0
    state = "UNKNOWN"
    while waited < MAX_WAIT_SECONDS:
        status = task.status()
        state = status.get("state", "UNKNOWN")
        if state in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(20)
        waited += 20
    if state == "COMPLETED":
        print("Export COMPLETED. Copy the file from Drive to 01_data\\raw.")
    elif state in ("FAILED", "CANCELLED"):
        print(f"Export {state}: {task.status().get('error_message', 'no message')}")
    else:
        print(f"Still {state} after {MAX_WAIT_SECONDS} s. The task keeps running "
              "in Earth Engine; check the Tasks tab or your Drive folder.")
