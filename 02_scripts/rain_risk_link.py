"""
DUWE AI FLOOD INTELLIGENCE ENGINE
Link between live rainfall / river discharge and high-risk infrastructure.

Save in 02_scripts, next to duwe_flood_dashboard.py.
"""

import os
import pandas as pd

BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
HIGH_RISK_CSV = os.path.join(BASE_DIR, r"01_data\processed\assets_high_risk.csv")

# A warning is shown only for at least MODERATE rain somewhere in the catchment, so a light
# drizzle does not raise an alarm. 2.5 mm/h matches LIGHT_MAX_MM_H in live_rainfall.py;
# use 7.5 if you want the message for heavy rain only.
MODERATE_RAIN_MM_H = 2.5
HEAVY_24H_MM = 50

# Possible names for the asset-type column in assets_high_risk.csv (first match is used).
CATEGORY_COLUMNS = ["category", "layer", "type", "asset_type", "asset_category"]

LABELS = {
    "schools": "Schools",
    "health": "Health facilities",
    "villages": "Villages / settlements",
    "water_points": "Water points",
    "worship": "Places of worship",
    "power": "Power sites",
    "markets": "Markets",
}


def _category_column(df):
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for name in CATEGORY_COLUMNS:
        if name in lowered:
            return lowered[name]
    return None


def get_high_risk_summary():
    """Return counts of high-risk assets by category, or None if the file is missing/unreadable."""
    if not os.path.exists(HIGH_RISK_CSV):
        return None
    try:
        df = pd.read_csv(HIGH_RISK_CSV, encoding="utf-8-sig")
    except Exception:
        return None

    if df.empty:
        return {"total": 0, "by_category": {}, "dataframe": df}

    col = _category_column(df)
    counts = df[col].astype(str).str.strip().str.lower().value_counts().to_dict() if col else {}
    return {"total": len(df), "by_category": counts, "dataframe": df}


def build_rain_risk_message(rain_status, high_risk, discharge_level=None):
    """
    Create a clear message linking rain / river discharge with high-risk assets.

    rain_status: the dict from catchment_status() in live_rainfall.py
    high_risk: the dict from get_high_risk_summary()
    discharge_level: "Normal", "Elevated", "Watch" or "Alert" from live_discharge.worst_status(), or None
    Returns a message string, or None when no warning is needed.
    """
    if not high_risk or high_risk.get("total", 0) == 0:
        return None

    river_high = discharge_level in ("Watch", "Alert")
    raining_now = rain_status.get("max_rate", 0) >= MODERATE_RAIN_MM_H
    heavy_next = rain_status.get("max_next24", 0) >= HEAVY_24H_MM
    heavy_past = rain_status.get("max_past24", 0) >= HEAVY_24H_MM

    if not (river_high or raining_now or heavy_next or heavy_past):
        return None

    # Category summary in priority order
    priority = ["health", "villages", "schools", "power", "worship", "water_points", "markets"]
    parts = []
    for cat in priority:
        count = high_risk["by_category"].get(cat, 0)
        if count > 0:
            parts.append(f"{count} {LABELS.get(cat, cat)}")
    assets_text = ", ".join(parts) if parts else f"{high_risk['total']} assets"
    tail = (f"**{high_risk['total']} named assets** are in High or Very High risk zones "
            f"({assets_text}).")

    if river_high:
        return f"Modelled river discharge is at the {discharge_level} level. {tail}"
    if raining_now:
        return f"Moderate or heavy rain is falling over parts of the catchment. {tail}"
    if heavy_next:
        return f"Heavy rain is forecast in the next 24 hours. {tail}"
    return f"Heavy rain fell in the last 24 hours. {tail}"
