"""
DUWE AI FLOOD INTELLIGENCE ENGINE
Stage 1 - Master Daily Pipeline (with CHIRPS data check)
"""

import os
import sys
import logging
from datetime import datetime
import subprocess

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
SCRIPTS_DIR = os.path.join(BASE_DIR, r"02_scripts")
LOG_DIR = os.path.join(BASE_DIR, r"02_scripts\automation\logs")
STATUS_FILE = os.path.join(LOG_DIR, "chirps_status.txt")

# Step 1: data check (path is relative to SCRIPTS_DIR)
CHECK_SCRIPT = r"automation\01_download_chirps.py"

# Steps that need rainfall data
RAINFALL_SCRIPTS = [
    "lesson_13_chirps_rainfall_integration.py",
    "lesson_14_rain_hazard_risk_update.py",
]

# Infrastructure (schools, roads, villages, buildings...) from OpenStreetMap.
# The download is cached for 30 days, so most daily runs skip it and finish quickly.
# If the network is down, the last good files are kept.
INFRA_DOWNLOAD = r"automation\03_download_infrastructure.py"
ASSET_EXPOSURE = r"automation\04_asset_exposure.py"   # counts assets per risk class (after Lesson 14)

# Set to True ONLY after the lesson scripts save their maps with plt.savefig(...).
# True  = no map windows open (needed for the unattended 6:00 AM run)
# False = map windows open and you must close them (fine when you run by hand)
SUPPRESS_PLOT_WINDOWS = False

os.makedirs(LOG_DIR, exist_ok=True)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

log_file = os.path.join(LOG_DIR, f"daily_pipeline_{datetime.now().strftime('%Y%m%d_%H%M')}.log")

logging.basicConfig(
    filename=log_file,
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)


def run_script(script_name, timeout=2400):
    """Run a Python script and return True if successful"""
    script_path = os.path.join(SCRIPTS_DIR, script_name)

    if not os.path.exists(script_path):
        logging.error(f"Script not found: {script_path}")
        return False

    logging.info(f"Starting: {script_name}")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if SUPPRESS_PLOT_WINDOWS:
        env["MPLBACKEND"] = "Agg"

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=SCRIPTS_DIR,
            timeout=timeout
        )

        if result.returncode == 0:
            logging.info(f"SUCCESS: {script_name}")
            return True
        else:
            logging.error(f"FAILED: {script_name}")
            logging.error(result.stderr[-2000:] if result.stderr else "No error output")
            return False

    except subprocess.TimeoutExpired:
        logging.error(f"TIMEOUT: {script_name}")
        return False
    except Exception as e:
        logging.error(f"EXCEPTION in {script_name}: {str(e)}")
        return False


def read_chirps_status():
    """Read key=value lines written by the data check. Returns a dict."""
    info = {}
    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    info[key] = value
    except Exception as e:
        logging.warning(f"Could not read status file: {e}")
    return info


def main():
    """Run the pipeline. Returns True only if every step succeeded."""
    start_time = datetime.now()
    logging.info("=" * 70)
    logging.info("DUWE AI FLOOD INTELLIGENCE ENGINE")
    logging.info("DAILY PIPELINE STARTED")
    logging.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 70)

    all_steps = [CHECK_SCRIPT, INFRA_DOWNLOAD] + RAINFALL_SCRIPTS + [ASSET_EXPOSURE]
    results = {}

    # Step 1: check rainfall data
    logging.info("Step 1: Checking CHIRPS rainfall data...")
    results[CHECK_SCRIPT] = run_script(CHECK_SCRIPT)

    status_info = read_chirps_status()
    data_status = status_info.get("status", "UNKNOWN")
    logging.info(
        f"Rainfall data status: {data_status} | data ends: {status_info.get('data_end', '?')} "
        f"| seasons behind: {status_info.get('seasons_behind', '?')}"
    )

    # Step 2: infrastructure download (independent of rainfall; cached, offline-safe)
    logging.info("Step 2: Checking infrastructure data (OpenStreetMap)...")
    results[INFRA_DOWNLOAD] = run_script(INFRA_DOWNLOAD, timeout=5400)

    # Step 3: run rainfall-dependent steps (skip if there is no data at all)
    if data_status == "NO_DATA":
        logging.error("No CHIRPS data found - skipping rainfall-dependent steps.")
    else:
        if data_status in ("OUTDATED", "ONE_SEASON_BEHIND"):
            logging.warning(
                f"Rainfall data is {data_status}. Results will use the newest file available, "
                "which may not reflect current conditions."
            )
        for script in RAINFALL_SCRIPTS:
            results[script] = run_script(script)

        # Step 4: how many assets sit in each risk class (uses the Lesson 14 risk map)
        logging.info("Step 4: Calculating asset exposure...")
        results[ASSET_EXPOSURE] = run_script(ASSET_EXPOSURE)

    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    success_count = sum(1 for ok in results.values() if ok)
    total = len(all_steps)
    all_ok = success_count == total

    logging.info("-" * 70)
    logging.info("Pipeline finished.")
    logging.info(f"Steps succeeded: {success_count} / {total}")
    logging.info(f"Rainfall data status: {data_status}")
    logging.info(f"Duration: {duration}")
    logging.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("DAILY PIPELINE COMPLETED" if all_ok else "DAILY PIPELINE FINISHED WITH ERRORS")
    logging.info("=" * 70)

    if all_ok:
        print("\nDaily pipeline completed successfully.")
    else:
        print("\nDaily pipeline completed with some issues. Check the log file.")
    if data_status in ("OUTDATED", "ONE_SEASON_BEHIND"):
        print(f"NOTE: rainfall data status is {data_status} - add a newer CHIRPS season file.")

    return all_ok


if __name__ == "__main__":
    # Exit code 0 = success, 1 = failure, so Task Scheduler shows the truth
    sys.exit(0 if main() else 1)
