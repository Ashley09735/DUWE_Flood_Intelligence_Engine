"""
DUWE FLOOD INTELLIGENCE ENGINE
MODULE 7: INFRASTRUCTURE EXPOSURE SCREENING
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "01_data" / "processed"

FORECAST_DIR = DATA_DIR / "forecasting"

ASSET_FILE = DATA_DIR / "assets_high_risk.csv"

OUTPUT_CSV = (
    FORECAST_DIR / "infrastructure_screening_latest.csv"
)

OUTPUT_JSON = (
    FORECAST_DIR / "infrastructure_screening_summary.json"
)


def main():

    FORECAST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not ASSET_FILE.exists():

        raise FileNotFoundError(
            f"Infrastructure file not found: {ASSET_FILE}"
        )

    df = pd.read_csv(ASSET_FILE)

    if "risk_class" not in df.columns:

        raise ValueError(
            "Missing risk_class column."
        )

    risk = (
        df["risk_class"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    high_risk = df[
        risk.isin([
            "high",
            "very high",
            "very_high"
        ])
    ].copy()

    high_risk.to_csv(
        OUTPUT_CSV,
        index=False
    )

    category_column = next(
        (
            column
            for column in [
                "category",
                "layer",
                "type",
                "asset_type",
                "asset_category"
            ]
            if column in high_risk.columns
        ),
        None
    )

    counts = {}

    if category_column:

        counts = (
            high_risk[category_column]
            .fillna("Unspecified")
            .astype(str)
            .value_counts()
            .to_dict()
        )

    result = {

        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "high_or_very_high_assets": len(high_risk),

        "category_counts": counts,

        "output_csv": str(OUTPUT_CSV),

        "status": "EXISTING_EXPOSURE_SCREENING",

        "warning": (
            "These assets are already classified as high risk. "
            "This does not establish that they will be flooded."
        )

    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:

        json.dump(result, file, indent=4)

    print("Infrastructure screening completed.")

    print("High-risk assets:", len(high_risk))


if __name__ == "__main__":
    main()