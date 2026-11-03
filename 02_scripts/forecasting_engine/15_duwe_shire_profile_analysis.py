
"""
DUWE DISASTER INTELLIGENCE
MODULE 15: SHIRE RIVER PROFILE ANALYSIS
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

BASE = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"

INPUT_FILE = os.path.join(
    BASE, "01_data", "processed",
    "forecasting", "shire_river_elevation_profile.csv"
)

OUTPUT_DIR = os.path.join(
    BASE, "01_data", "processed", "forecasting"
)

# --------------------------------------------------
# 1. LOAD PROFILE DATA
# --------------------------------------------------

distances = []
elevations = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:

        if row["elevation_m"] == "":
            continue

        distances.append(float(row["distance_m"]))
        elevations.append(float(row["elevation_m"]))

distance = np.array(distances)
elevation = np.array(elevations)

# --------------------------------------------------
# 2. CHECK PROFILE DIRECTION
# --------------------------------------------------

upstream_elevation = elevation[0]
downstream_elevation = elevation[-1]

total_length = distance[-1]

elevation_difference = (
    upstream_elevation - downstream_elevation
)

# Positive means elevation decreases along digitised line.
# Negative means elevation increases along digitised line.

gradient = (
    elevation_difference / total_length
) * 100

# --------------------------------------------------
# 3. CALCULATE LOCAL GRADIENTS
# --------------------------------------------------

distance_change = np.diff(distance)
elevation_change = np.diff(elevation)

valid_steps = distance_change > 0

local_gradient = (
    elevation_change[valid_steps] /
    distance_change[valid_steps]
) * 100

# --------------------------------------------------
# 4. IDENTIFY LARGE ELEVATION CHANGES
# --------------------------------------------------

large_changes = np.where(
    np.abs(elevation_change) > 5
)[0]

# --------------------------------------------------
# 5. PRINT RESULTS
# --------------------------------------------------

print("\nDUWE DISASTER INTELLIGENCE")
print("MODULE 15: SHIRE RIVER PROFILE ANALYSIS")
print("=" * 60)

print("Total length:", round(total_length / 1000, 3), "km")

print("First sampled elevation:",
      round(upstream_elevation, 3), "m")

print("Last sampled elevation:",
      round(downstream_elevation, 3), "m")

print("Elevation difference:",
      round(elevation_difference, 3), "m")

print("Overall terrain gradient:",
      round(gradient, 5), "%")

print("Mean local gradient:",
      round(float(np.mean(local_gradient)), 5), "%")

print("Maximum local gradient:",
      round(float(np.max(local_gradient)), 5), "%")

print("Minimum local gradient:",
      round(float(np.min(local_gradient)), 5), "%")

print("Elevation changes greater than 5 m:",
      len(large_changes))

# --------------------------------------------------
# 6. CREATE LONGITUDINAL PROFILE
# --------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    distance / 1000,
    elevation,
    linewidth=1.2
)

plt.xlabel("Distance along digitised river line (km)")
plt.ylabel("DEM elevation (m)")

plt.title(
    "DUWE DISASTER INTELLIGENCE\n"
    "Shire River Terrain Elevation Profile"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

output_graph = os.path.join(
    OUTPUT_DIR,
    "shire_river_longitudinal_profile.png"
)

plt.savefig(output_graph, dpi=300)

plt.close()

print("\nProfile graph saved:", output_graph)

print("\nMODULE 15 COMPLETED")
