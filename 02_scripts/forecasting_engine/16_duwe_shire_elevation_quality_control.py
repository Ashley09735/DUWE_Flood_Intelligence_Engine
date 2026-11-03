
"""
DUWE DISASTER INTELLIGENCE
MODULE 16: SHIRE RIVER ELEVATION QUALITY CONTROL

Purpose:
Inspect the DEM elevation profile, identify suspicious
elevation changes and generate diagnostic outputs.

Original elevation measurements are preserved.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "01_data",
    "processed",
    "forecasting",
    "shire_river_elevation_profile.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "01_data",
    "processed",
    "forecasting"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. QUALITY CONTROL SETTINGS
# --------------------------------------------------

ELEVATION_CHANGE_LIMIT = 5.0
GRADIENT_LIMIT = 10.0
ROLLING_WINDOW = 11
RESIDUAL_LIMIT = 3.0


# --------------------------------------------------
# 3. LOAD ELEVATION DATA
# --------------------------------------------------

print("\nDUWE DISASTER INTELLIGENCE")
print("MODULE 16: SHIRE RIVER ELEVATION QUALITY CONTROL")
print("=" * 55)

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "distance_m",
    "easting",
    "northing",
    "elevation_m"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

for column in required_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=required_columns)
df = df.sort_values("distance_m").reset_index(drop=True)

if len(df) < ROLLING_WINDOW:
    raise ValueError("Not enough valid samples for quality control.")

print(f"\nValid elevation samples: {len(df)}")


# --------------------------------------------------
# 4. CALCULATE ELEVATION CHANGES
# --------------------------------------------------

df["distance_change_m"] = df["distance_m"].diff()

df["elevation_change_m"] = df["elevation_m"].diff()

df["local_gradient_percent"] = (
    df["elevation_change_m"] /
    df["distance_change_m"]
) * 100

df["distance_midpoint_km"] = (
    df["distance_m"] -
    (df["distance_change_m"] / 2)
) / 1000


# --------------------------------------------------
# 5. IDENTIFY SUSPICIOUS SEGMENTS
# --------------------------------------------------

df["large_elevation_change"] = (
    df["elevation_change_m"].abs() >
    ELEVATION_CHANGE_LIMIT
)

df["steep_gradient"] = (
    df["local_gradient_percent"].abs() >
    GRADIENT_LIMIT
)

df["suspicious_segment"] = (
    df["large_elevation_change"] |
    df["steep_gradient"]
)

df["segment_flag_reason"] = ""

df.loc[
    df["large_elevation_change"],
    "segment_flag_reason"
] += "Elevation change above 5 m; "

df.loc[
    df["steep_gradient"],
    "segment_flag_reason"
] += "Absolute gradient above 5%; "


# --------------------------------------------------
# 6. IDENTIFY POSSIBLE ISOLATED ELEVATION ANOMALIES
# --------------------------------------------------

df["rolling_median_m"] = (
    df["elevation_m"]
    .rolling(
        window=ROLLING_WINDOW,
        center=True,
        min_periods=5
    )
    .median()
)

df["median_residual_m"] = (
    df["elevation_m"] -
    df["rolling_median_m"]
)

df["possible_isolated_anomaly"] = (
    df["median_residual_m"].abs() >
    RESIDUAL_LIMIT
)


# --------------------------------------------------
# 7. EXTRACT FLAGGED RESULTS
# --------------------------------------------------

flagged_segments = df[
    df["suspicious_segment"]
].copy()

flagged_points = df[
    df["possible_isolated_anomaly"]
].copy()

segment_output = os.path.join(
    OUTPUT_DIR,
    "shire_river_flagged_segments.csv"
)

point_output = os.path.join(
    OUTPUT_DIR,
    "shire_river_possible_elevation_anomalies.csv"
)

flagged_segments.to_csv(
    segment_output,
    index=False
)

flagged_points.to_csv(
    point_output,
    index=False
)


# --------------------------------------------------
# 8. CALCULATE QUALITY CONTROL STATISTICS
# --------------------------------------------------

total_segments = len(df) - 1

flagged_segment_count = len(flagged_segments)

flagged_point_count = len(flagged_points)

flagged_percentage = (
    flagged_segment_count / total_segments
) * 100 if total_segments > 0 else 0

mean_absolute_gradient = (
    df["local_gradient_percent"].abs().mean()
)

maximum_absolute_gradient = (
    df["local_gradient_percent"].abs().max()
)

first_elevation = df["elevation_m"].iloc[0]

last_elevation = df["elevation_m"].iloc[-1]

overall_length_km = (
    df["distance_m"].iloc[-1] -
    df["distance_m"].iloc[0]
) / 1000

overall_gradient = (
    (last_elevation - first_elevation) /
    (overall_length_km * 1000)
) * 100


# --------------------------------------------------
# 9. CREATE QUALITY CONTROL GRAPH
# --------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    df["distance_m"] / 1000,
    df["elevation_m"],
    label="Original DEM elevation",
    linewidth=1
)

plt.plot(
    df["distance_m"] / 1000,
    df["rolling_median_m"],
    label="Rolling median (diagnostic only)",
    linewidth=2
)

plt.scatter(
    flagged_segments["distance_m"] / 1000,
    flagged_segments["elevation_m"],
    label="Flagged segment endpoints",
    s=18
)

plt.scatter(
    flagged_points["distance_m"] / 1000,
    flagged_points["elevation_m"],
    label="Possible isolated anomalies",
    s=30,
    marker="x"
)

plt.title(
    "DUWE DISASTER INTELLIGENCE\n"
    "Shire River Elevation Quality Control"
)

plt.xlabel("Distance along digitised river line (km)")
plt.ylabel("DEM elevation (m)")

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

graph_output = os.path.join(
    OUTPUT_DIR,
    "shire_river_elevation_quality_control.png"
)

plt.savefig(
    graph_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 10. SAVE QUALITY CONTROL REPORT
# --------------------------------------------------

report_output = os.path.join(
    OUTPUT_DIR,
    "shire_river_quality_control_report.txt"
)

with open(report_output, "w", encoding="utf-8") as report:

    report.write("DUWE DISASTER INTELLIGENCE\n")
    report.write("MODULE 16: SHIRE RIVER ELEVATION QUALITY CONTROL\n")
    report.write("=" * 55 + "\n\n")

    report.write(f"Valid samples: {len(df)}\n")
    report.write(f"Total segments: {total_segments}\n")
    report.write(f"Flagged segments: {flagged_segment_count}\n")
    report.write(f"Flagged segment percentage: {flagged_percentage:.2f}%\n")
    report.write(f"Possible isolated anomalies: {flagged_point_count}\n\n")

    report.write(f"First elevation: {first_elevation:.3f} m\n")
    report.write(f"Last elevation: {last_elevation:.3f} m\n")
    report.write(f"Profile length: {overall_length_km:.3f} km\n")
    report.write(f"Overall signed terrain gradient: {overall_gradient:.5f}%\n")
    report.write(
        f"Mean absolute local gradient: "
        f"{mean_absolute_gradient:.5f}%\n"
    )
    report.write(
        f"Maximum absolute local gradient: "
        f"{maximum_absolute_gradient:.5f}%\n"
    )

    report.write("\nQUALITY CONTROL THRESHOLDS\n")
    report.write(f"Elevation change: > {ELEVATION_CHANGE_LIMIT} m\n")
    report.write(f"Absolute gradient: > {GRADIENT_LIMIT}%\n")
    report.write(f"Rolling median window: {ROLLING_WINDOW} samples\n")
    report.write(f"Median residual: > {RESIDUAL_LIMIT} m\n")

    report.write(
        "\nIMPORTANT: Flagged measurements are not automatically "
        "errors. They require investigation. The rolling median "
        "is diagnostic and does not replace the original elevations.\n"
    )


# --------------------------------------------------
# 11. DISPLAY RESULTS
# --------------------------------------------------

print("\nQUALITY CONTROL RESULTS")
print("-" * 55)

print(f"Total segments: {total_segments}")
print(f"Flagged segments: {flagged_segment_count}")
print(f"Flagged segment percentage: {flagged_percentage:.2f}%")
print(f"Possible isolated anomalies: {flagged_point_count}")

print(f"\nOverall signed terrain gradient: {overall_gradient:.5f}%")
print(f"Mean absolute local gradient: {mean_absolute_gradient:.5f}%")
print(f"Maximum absolute local gradient: {maximum_absolute_gradient:.5f}%")

print("\nOUTPUT FILES")
print("-" * 55)

print(segment_output)
print(point_output)
print(graph_output)
print(report_output)

print("\nMODULE 16 COMPLETED")
