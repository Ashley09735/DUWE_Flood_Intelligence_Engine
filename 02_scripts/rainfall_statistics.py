"""
Rainfall statistics for Chikwawa (CHIRPS rainy season 2023-2024).
Reads the aligned 30 m raster, masks nodata, and reports area-weighted
statistics (every 30 m pixel has equal area, so pixel stats = area stats).

Run:  python rainfall_statistics.py
Needs: numpy, rasterio
"""
import csv
from pathlib import Path

import numpy as np
import rasterio

BASE = Path(r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE")
RAIN = BASE / "01_data" / "processed" / "chirps_rainfall_aligned_duwe.tif"
OUT_CSV = BASE / "01_data" / "processed" / "rainfall_statistics_chikwawa.csv"

PERCENTILES = [5, 10, 25, 50, 75, 80, 90, 95]
THRESHOLDS_MM = [300, 400, 500, 560, 600, 700]


def summarize(x, px_km2, label):
    rows = [(f"[{label}] valid pixels", x.size),
            (f"[{label}] area (km2)", x.size * px_km2),
            (f"[{label}] min (mm)", x.min()),
            (f"[{label}] max (mm)", x.max()),
            (f"[{label}] mean (mm)", x.mean()),
            (f"[{label}] median (mm)", np.median(x)),
            (f"[{label}] std dev (mm)", x.std()),
            (f"[{label}] coeff. of variation (%)", 100 * x.std() / x.mean())]
    for p in PERCENTILES:
        rows.append((f"[{label}] P{p} (mm)", np.percentile(x, p)))
    for t in THRESHOLDS_MM:
        n = (x >= t).sum()
        rows.append((f"[{label}] area >= {t} mm (km2)", n * px_km2))
        rows.append((f"[{label}] share >= {t} mm (%)", 100 * n / x.size))
    return rows


with rasterio.open(RAIN) as src:
    arr = src.read(1, masked=True)          # nodata is masked automatically
    px_km2 = abs(src.transform.a * src.transform.e) / 1e6
    print(f"Raster: {RAIN.name} | CRS: {src.crs} | pixel area: {px_km2:.6f} km2")

data = arr.compressed().astype("float64")
data = data[np.isfinite(data)]

n_zero = int((data == 0).sum())
print(f"Valid pixels: {data.size:,} | exactly 0.0 mm: {n_zero:,} "
      f"({100 * n_zero / data.size:.2f}%)")

results = summarize(data, px_km2, "all valid")
if n_zero:
    results += summarize(data[data > 0], px_km2, "excluding zeros")

# Distinct values ~ number of native CHIRPS cells covering the district
results.append(("distinct rainfall values (~CHIRPS cells)", np.unique(np.round(data, 3)).size))

print("\nRAINFALL STATISTICS")
print("-" * 60)
for name, val in results:
    print(f"{name:<48}: {val:,.2f}")

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["statistic", "value"])
    for name, val in results:
        w.writerow([name, round(float(val), 4)])
print(f"\nSaved: {OUT_CSV}")
