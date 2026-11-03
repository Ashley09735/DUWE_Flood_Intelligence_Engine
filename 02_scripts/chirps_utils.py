"""
DUWE AI FLOOD INTELLIGENCE ENGINE
Shared helper: find the newest CHIRPS rainfall file in the raw folder.

Save this file in:  02_scripts\\chirps_utils.py
Use it in a lesson script:

    from chirps_utils import latest_chirps_file
    CHIRPS_PATH = latest_chirps_file(RAW_DIR, clipped=True)

What changed from the first version
-----------------------------------
1. Files named like  chirps_chikwawa_total_2026-11-02  are now accepted
   when clipped=True. They are already specific to the study area
   ("chikwawa"), but they do not contain the word "clipped", so the old
   version silently skipped them and kept using the 2023_2024 file.
2. A full date in the name (2026-11-02, 2026_11_02, 20261102) is now used
   as the data end date. The old version always turned the year into
   30 April of that year, so a November 2026 download looked like April.
3. Matching is case-insensitive on every system, and files with .tif,
   .tiff or no extension are accepted (Windows can hide extensions).
"""

import os
import re
from datetime import date

# Words that mark a file as already limited to the study area.
CLIPPED_WORDS = ("clipped", "chikwawa")

_FULL_DATE = re.compile(
    r"(?<!\d)(20\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])(?!\d)"
)


def data_end_date(path):
    """
    Last day covered by the file, or None.

    1. A full date in the name (2026-11-02) is used as it is.
    2. Otherwise the season end: 30 April of the latest 20xx year in the name
       (RainySeason_2023_2024 -> 2024-04-30).
    """
    name = os.path.basename(path)
    m = _FULL_DATE.search(name)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    years = [int(y) for y in re.findall(r"(?<!\d)(20\d{2})(?!\d)", name)]
    return date(max(years), 4, 30) if years else None


def _is_raster_name(name):
    ext = os.path.splitext(name)[1].lower()
    return ext in (".tif", ".tiff", "")


def _is_clipped(name):
    low = name.lower()
    return any(word in low for word in CLIPPED_WORDS)


def latest_chirps_file(raw_dir, clipped=None):
    """
    Return the path of the newest CHIRPS raster in raw_dir.

    clipped=True  -> only files already limited to the study area
                     (name contains 'clipped' or 'chikwawa')
    clipped=False -> only files that are NOT limited to the study area
    clipped=None  -> any CHIRPS file

    Newest = latest data end date; save time only breaks ties.
    """
    files = []
    if os.path.isdir(raw_dir):
        for name in os.listdir(raw_dir):
            full = os.path.join(raw_dir, name)
            if not os.path.isfile(full):
                continue
            if "chirps" not in name.lower() or not _is_raster_name(name):
                continue
            if clipped is True and not _is_clipped(name):
                continue
            if clipped is False and _is_clipped(name):
                continue
            files.append(full)

    if not files:
        raise FileNotFoundError(
            f"No CHIRPS raster found in {raw_dir} (clipped={clipped}). "
            "Check the folder and the file names."
        )

    return max(
        files,
        key=lambda f: (data_end_date(f) or date.min, os.path.getmtime(f)),
    )


if __name__ == "__main__":
    # Quick test:  python chirps_utils.py
    raw = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE\01_data\raw"
    for flag in (True, False, None):
        try:
            path = latest_chirps_file(raw, clipped=flag)
            print(f"clipped={flag}: {path}  (data end {data_end_date(path)})")
        except FileNotFoundError as e:
            print(f"clipped={flag}: {e}")
