"""
DUWE AI FLOOD INTELLIGENCE ENGINE
Stage 1 - Asset Exposure

Counts how many schools, health facilities, villages, buildings, etc. fall in each
risk class (Low / Medium / High / Very High), and how many km of road and bridges.
Reads the files made by 03_download_infrastructure.py and the classified risk raster
made by Lesson 14. Works offline.

Outputs (in 01_data\\processed):
    asset_exposure_summary.csv   one row per category
    assets_high_risk.csv         named assets in High / Very High zones
"""

import os
import sys
import csv
import json
import logging
from datetime import datetime

import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio.transform import rowcol

BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
RISK_RASTER = os.path.join(BASE_DIR, r"01_data\processed\risk_updated_classified_duwe.tif")
INFRA_DIR = os.path.join(BASE_DIR, r"01_data\raw\infrastructure")
OUT_DIR = os.path.join(BASE_DIR, r"01_data\processed")
LOG_DIR = os.path.join(BASE_DIR, r"02_scripts\automation\logs")

CLASS_NAMES = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
POINT_CATEGORIES = ["schools", "health", "villages", "water_points", "markets",
                    "worship", "power", "buildings"]
LINE_CATEGORIES = ["roads", "bridges"]
LISTED_CATEGORIES = ["schools", "health", "villages", "water_points", "markets", "worship", "power"]


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    log_file = os.path.join(LOG_DIR, f"asset_exposure_{datetime.now().strftime('%Y%m%d')}.log")
    logging.basicConfig(filename=log_file, encoding="utf-8", level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)


def load_features(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("features", [])


class RiskGrid:
    """The classified risk raster, with a fast way to look up the class at coordinates."""

    def __init__(self, path):
        with rasterio.open(path) as src:
            self.cls = src.read(1)
            self.transform = src.transform
            self.crs = src.crs
        self.h, self.w = self.cls.shape

    def to_grid_crs(self, lons, lats):
        xs, ys = transform("EPSG:4326", self.crs, list(lons), list(lats))
        return np.asarray(xs, dtype="float64"), np.asarray(ys, dtype="float64")

    def classes_at(self, xs, ys):
        """Risk class 1-4 at each x,y (raster CRS). 0 = outside the study area / no data."""
        rows, cols = rowcol(self.transform, list(xs), list(ys))
        rows, cols = np.asarray(rows), np.asarray(cols)
        out = np.zeros(len(rows), dtype=int)
        inside = (rows >= 0) & (rows < self.h) & (cols >= 0) & (cols < self.w)
        vals = self.cls[rows[inside], cols[inside]]
        vals = np.where((vals >= 1) & (vals <= 4), vals, 0)
        out[inside] = vals.astype(int)
        return out


def summary_row(category, unit, per_class):
    """per_class: array of 5 values, index 0 = outside area, 1-4 = classes."""
    inside_total = float(np.sum(per_class[1:5]))
    high = float(per_class[3] + per_class[4])
    return {
        "category": category,
        "unit": unit,
        "total_in_area": round(inside_total, 2),
        "Low": round(float(per_class[1]), 2),
        "Medium": round(float(per_class[2]), 2),
        "High": round(float(per_class[3]), 2),
        "Very High": round(float(per_class[4]), 2),
        "outside_area": round(float(per_class[0]), 2),
        "pct_high_or_very_high": round(100 * high / inside_total, 1) if inside_total else 0.0,
    }


def analyse(raster_path, infra_dir):
    """Return (summary_rows, high_risk_rows)."""
    grid = RiskGrid(raster_path)
    summary, high_risk = [], []

    for cat in POINT_CATEGORIES:
        feats = load_features(os.path.join(infra_dir, f"{cat}.geojson"))
        if feats is None:
            logging.warning(f"{cat}: file not found - skipped")
            continue
        pts = [f for f in feats if f["geometry"]["type"] == "Point"]
        if not pts:
            summary.append(summary_row(cat, "count", np.zeros(5)))
            continue
        lons = [f["geometry"]["coordinates"][0] for f in pts]
        lats = [f["geometry"]["coordinates"][1] for f in pts]
        xs, ys = grid.to_grid_crs(lons, lats)
        classes = grid.classes_at(xs, ys)
        summary.append(summary_row(cat, "count", np.bincount(classes, minlength=5).astype(float)))
        logging.info(f"{cat}: {len(pts):,} features analysed")

        if cat in LISTED_CATEGORIES:
            for f, lon, lat, c in zip(pts, lons, lats, classes):
                if c >= 3:
                    p = f["properties"]
                    high_risk.append({
                        "category": cat,
                        "name": p.get("name", "(unnamed)"),
                        "type": p.get("amenity") or p.get("healthcare") or p.get("place")
                                or p.get("man_made") or p.get("power") or "",
                        "risk_class": CLASS_NAMES[int(c)],
                        "risk_rank": int(c),
                        "lat": round(lat, 6),
                        "lon": round(lon, 6),
                        "osm_id": p.get("osm_id", ""),
                    })

    for cat in LINE_CATEGORIES:
        feats = load_features(os.path.join(infra_dir, f"{cat}.geojson"))
        if feats is None:
            logging.warning(f"{cat}: file not found - skipped")
            continue
        lines = [f["geometry"]["coordinates"] for f in feats if f["geometry"]["type"] == "LineString"]
        if not lines:
            summary.append(summary_row(cat, "km", np.zeros(5)))
            continue
        lens = [len(c) for c in lines]
        all_lon = [p[0] for c in lines for p in c]
        all_lat = [p[1] for c in lines for p in c]
        xs, ys = grid.to_grid_crs(all_lon, all_lat)       # metres (UTM)
        bounds = np.cumsum([0] + lens)
        starts = np.concatenate([np.arange(a, b - 1) for a, b in zip(bounds[:-1], bounds[1:])])
        x0, x1, y0, y1 = xs[starts], xs[starts + 1], ys[starts], ys[starts + 1]
        seg_m = np.hypot(x1 - x0, y1 - y0)
        mid_cls = grid.classes_at((x0 + x1) / 2, (y0 + y1) / 2)
        km = np.bincount(mid_cls, weights=seg_m, minlength=5) / 1000.0
        summary.append(summary_row(cat, "km", km))
        logging.info(f"{cat}: {len(lines):,} lines, {seg_m.sum() / 1000:,.0f} km analysed")

    high_risk.sort(key=lambda r: (-r["risk_rank"], r["category"], r["name"]))
    return summary, high_risk


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    setup_logging()
    print("=" * 65)
    print("DUWE AI FLOOD INTELLIGENCE ENGINE")
    print("Asset Exposure")
    print("=" * 65)

    if not os.path.exists(RISK_RASTER):
        logging.error(f"Risk raster not found: {RISK_RASTER} (run Lesson 14 first)")
        return 1
    if not os.path.isdir(INFRA_DIR) or not any(n.endswith(".geojson") for n in os.listdir(INFRA_DIR)):
        logging.error(f"No infrastructure files in {INFRA_DIR} (run 03_download_infrastructure.py first)")
        return 1

    summary, high_risk = analyse(RISK_RASTER, INFRA_DIR)

    s_fields = ["category", "unit", "total_in_area", "Low", "Medium", "High", "Very High",
                "outside_area", "pct_high_or_very_high"]
    h_fields = ["category", "name", "type", "risk_class", "risk_rank", "lat", "lon", "osm_id"]
    s_path = os.path.join(OUT_DIR, "asset_exposure_summary.csv")
    h_path = os.path.join(OUT_DIR, "assets_high_risk.csv")
    write_csv(s_path, summary, s_fields)
    write_csv(h_path, high_risk, h_fields)

    print(f"\n{'Category':<14}{'Unit':<7}{'In area':>10}{'Low':>10}{'Medium':>10}{'High':>10}{'V.High':>10}{'%High+':>8}")
    print("-" * 79)
    for r in summary:
        print(f"{r['category']:<14}{r['unit']:<7}{r['total_in_area']:>10,.1f}{r['Low']:>10,.1f}"
              f"{r['Medium']:>10,.1f}{r['High']:>10,.1f}{r['Very High']:>10,.1f}"
              f"{r['pct_high_or_very_high']:>7.1f}%")
    print(f"\nHigh-risk named assets: {len(high_risk):,}")
    print(f"Saved: {s_path}")
    print(f"Saved: {h_path}")
    logging.info("Asset exposure finished.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as ex:
        logging.error(f"Asset exposure failed: {ex}")
        print(f"Error: {ex}")
        sys.exit(1)
