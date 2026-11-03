"""
DUWE FLOOD INTELLIGENCE ENGINE
Link between live rainfall and high-risk infrastructure
Stage 1 - Improved thresholds and clearer alerts
"""

import os
import pandas as pd

BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
HIGH_RISK_CSV = os.path.join(BASE_DIR, r"01_data\processed\assets_high_risk.csv")

# -------------------------------------------------
# RAINFALL THRESHOLDS (adjustable)
# -------------------------------------------------
RAINING_NOW_MM_H = 0.5          # Rain is falling
MODERATE_RAIN_MM_H = 5.0        # Moderate intensity
HEAVY_RAIN_MM_H = 10.0          # Heavy intensity
HEAVY_24H_MM = 50               # Significant rain in last 24 hours
EXTREME_24H_MM = 100            # Very heavy rain in last 24 hours
HEAVY_FORECAST_24H_MM = 50      # Heavy rain expected in next 24 hours

LABELS = {
    "schools": "Schools",
    "health": "Health facilities",
    "villages": "Villages / settlements",
    "water_points": "Water points",
    "worship": "Places of worship",
    "power": "Power sites",
    "markets": "Markets",
}

PRIORITY_ORDER = ["health", "villages", "schools", "power", "worship", "water_points"]


def get_high_risk_summary():
    """Return summary of assets in High / Very High risk zones."""
    if not os.path.exists(HIGH_RISK_CSV):
        return None

    try:
        df = pd.read_csv(HIGH_RISK_CSV, encoding="utf-8-sig")
        if df.empty:
            return {"total": 0, "by_category": {}, "dataframe": df}

        counts = df["category"].value_counts().to_dict()
        return {
            "total": len(df),
            "by_category": counts,
            "dataframe": df
        }
    except Exception:
        return None


def get_rain_level(rain_status: dict) -> str:
    """
    Determine the overall rainfall concern level.
    Returns: "none", "raining", "moderate", "heavy", "extreme"
    """
    max_rate = rain_status.get("max_rate", 0)
    mean_rate = rain_status.get("mean_rate", 0)
    max_past24 = rain_status.get("max_past24", 0)
    max_next24 = rain_status.get("max_next24", 0)
    raining_points = rain_status.get("raining", 0)

    if max_past24 >= EXTREME_24H_MM or max_next24 >= EXTREME_24H_MM:
        return "extreme"
    if max_rate >= HEAVY_RAIN_MM_H or max_past24 >= HEAVY_24H_MM or max_next24 >= HEAVY_FORECAST_24H_MM:
        return "heavy"
    if max_rate >= MODERATE_RAIN_MM_H or mean_rate >= MODERATE_RAIN_MM_H:
        return "moderate"
    if raining_points > 0 or max_rate >= RAINING_NOW_MM_H:
        return "raining"
    return "none"


def build_assets_text(high_risk: dict) -> str:
    """Create a short readable list of priority assets at risk."""
    if not high_risk or high_risk["total"] == 0:
        return "no named high-risk assets"

    parts = []
    for cat in PRIORITY_ORDER:
        count = high_risk["by_category"].get(cat, 0)
        if count > 0:
            label = LABELS.get(cat, cat)
            parts.append(f"{count} {label.lower()}")

    if not parts:
        return f"{high_risk['total']} high-risk assets"

    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def build_rain_risk_message(rain_status: dict, high_risk: dict) -> str | None:
    """
    Create a clear alert message linking rainfall conditions
    with high-risk infrastructure.
    Returns None if no alert is needed.
    """
    if high_risk is None or high_risk.get("total", 0) == 0:
        return None

    level = get_rain_level(rain_status)
    if level == "none":
        return None

    assets_text = build_assets_text(high_risk)
    total = high_risk["total"]

    if level == "extreme":
        return (
            f"EXTREME RAINFALL CONCERN: Very heavy rain has occurred or is forecast "
            f"(≥ {EXTREME_24H_MM} mm). There are **{total} named assets** in High or Very High "
            f"risk zones ({assets_text}). Close monitoring is advised."
        )

    if level == "heavy":
        return (
            f"HEAVY RAIN ALERT: Significant rainfall is occurring or expected. "
            f"**{total} named assets** are located in High or Very High risk zones "
            f"({assets_text})."
        )

    if level == "moderate":
        return (
            f"Moderate rain is falling over parts of the catchment. "
            f"**{total} named assets** are in High or Very High risk zones "
            f"({assets_text})."
        )

    # level == "raining"
    return (
        f"Rain is currently falling over parts of the catchment. "
        f"There are **{total} named assets** in High or Very High risk zones "
        f"({assets_text})."
    )


def get_priority_assets(high_risk: dict, limit: int = 15) -> list:
    """
    Return a short list of priority high-risk assets
    (health facilities and villages first).
    """
    if not high_risk or high_risk.get("total", 0) == 0:
        return []

    df = high_risk.get("dataframe")
    if df is None or df.empty:
        return []

    # Priority order
    priority_cats = ["health", "villages", "schools", "power"]
    selected = []

    for cat in priority_cats:
        subset = df[df["category"] == cat]
        for _, row in subset.iterrows():
            selected.append({
                "category": cat,
                "name": row.get("name", "Unnamed"),
                "type": row.get("type", ""),
                "risk_class": row.get("risk_class", "")
            })
            if len(selected) >= limit:
                return selected

    return selected
