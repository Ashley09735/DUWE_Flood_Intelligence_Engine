"""
DUWE AI FLOOD INTELLIGENCE ENGINE
Stage 2 - Heavy hydraulic simulation (2D flood inundation, rain on the whole catchment)

Runs the local-inertial shallow-water solver in hydraulic_engine.py over your 30 m elevation model
and saves maps of maximum flood depth, speed, hazard, arrival time and wet duration, plus statistics
on how many schools, clinics, villages, buildings and kilometres of road are flooded.

Usage (from 02_scripts\\automation):
    python 05_hydraulic_simulation.py --list
    python 05_hydraulic_simulation.py --scenario storm_100mm_24h --dry-run
    python 05_hydraulic_simulation.py --scenario storm_100mm_24h --hours 2 --resolution 120   (quick test)
    python 05_hydraulic_simulation.py --scenario storm_100mm_24h                              (full run)
    python 05_hydraulic_simulation.py --live                                                  (latest rain + forecast)
    python 05_hydraulic_simulation.py --rain-mm 150 --rain-hours 12 --shape burst --name my_event

IMPORTANT: this is a screening-level scenario model. It is not a forecast and it has not been
calibrated or validated against recorded floods. See the notes printed at the end of each run.
"""

import os
import sys
import csv
import json
import time
import argparse
import logging
from datetime import datetime, timedelta

import numpy as np
import rasterio
from affine import Affine
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 02_scripts
import hydraulic_engine as he

BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
DEM_PATH = os.path.join(BASE_DIR, r"01_data\processed\dem_conditioned_duwe.tif")
ACC_PATH = os.path.join(BASE_DIR, r"01_data\processed\flow_accumulation_duwe.tif")
CATCHMENT_PATH = os.path.join(BASE_DIR, r"01_data\processed\catchment_duwe.tif")
CHIRPS_PATH = os.path.join(BASE_DIR, r"01_data\processed\chirps_rainfall_aligned_duwe.tif")
LIVE_JSON = os.path.join(BASE_DIR, r"01_data\processed\live_rainfall_latest.json")
INFRA_DIR = os.path.join(BASE_DIR, r"01_data\raw\infrastructure")
OUT_ROOT = os.path.join(BASE_DIR, r"01_data\processed\hydraulic")
LOG_DIR = os.path.join(BASE_DIR, r"02_scripts\automation\logs")

# Design scenarios. They are NOT tied to return periods: replace the depths with the rainfall
# depth-duration-frequency values from DCCMS when you have them.
SCENARIOS = {
    "storm_50mm_24h":     {"rain_mm": 50,  "rain_hours": 24, "shape": "uniform",    "hours": 36},
    "storm_100mm_24h":    {"rain_mm": 100, "rain_hours": 24, "shape": "burst",      "hours": 36},
    "storm_200mm_24h":    {"rain_mm": 200, "rain_hours": 24, "shape": "burst",      "hours": 36},
    "cyclone_300mm_72h":  {"rain_mm": 300, "rain_hours": 72, "shape": "triangular", "hours": 96},
}
DEFAULT_SCENARIO = "storm_100mm_24h"

# Placeholder hydrology and roughness: calibrate these against recorded floods.
DEFAULTS = {"n": 0.05, "initial_loss_mm": 10.0, "infil_mm_h": 5.0, "soil_storage_mm": 100.0}

DEPTH_CLASSES = [("0.1-0.5 m", 0.1, 0.5), ("0.5-1 m", 0.5, 1.0), ("1-2 m", 1.0, 2.0), (">2 m", 2.0, 1e9)]
POINT_ASSETS = ["schools", "health", "villages", "water_points", "markets", "worship", "power", "buildings"]
LISTED_ASSETS = ["schools", "health", "villages", "water_points", "markets", "worship", "power"]
LINE_ASSETS = ["roads", "bridges"]


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(filename=os.path.join(LOG_DIR, f"hydraulic_{datetime.now().strftime('%Y%m%d')}.log"),
                        encoding="utf-8", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)


# -------------------------------------------------
# RAINFALL SCENARIOS
# -------------------------------------------------
def design_hyetograph(total_mm, hours, shape="uniform", step_min=15):
    """Return (edges in seconds, rates in m/s) for a design storm of the given depth and duration."""
    n = max(1, int(round(hours * 60 / step_min)))
    x = (np.arange(n) + 0.5) / n
    if shape == "triangular":
        w = np.where(x < 0.4, x / 0.4, (1 - x) / 0.6)
    elif shape == "burst":
        w = np.exp(-0.5 * ((x - 0.3) / 0.15) ** 2) + 0.05
    else:
        w = np.ones(n)
    mm = total_mm * w / w.sum()
    edges = np.arange(n + 1) * step_min * 60.0
    return edges, mm / 1000.0 / (step_min * 60.0)


def hyetograph_from_csv(path):
    """CSV with two columns: hour (start of interval), mm_per_hour."""
    hrs, rate = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            keys = {k.strip().lower(): v for k, v in row.items()}
            hrs.append(float(keys["hour"]))
            rate.append(float(keys["mm_per_hour"]))
    hrs = np.array(hrs)
    step = np.median(np.diff(hrs)) if len(hrs) > 1 else 1.0
    edges = np.append(hrs, hrs[-1] + step) * 3600.0
    return edges - edges[0], np.array(rate) / 1000.0 / 3600.0


def hyetograph_from_live(path, hours_before=24, hours_after=48):
    """Catchment-average hourly rain from the saved Open-Meteo reading: last 24 h plus the forecast."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    series, now = [], None
    for loc in payload.get("locations", []):
        h = loc.get("hourly") or {}
        if not h.get("time"):
            continue
        series.append(np.array([0.0 if v is None else float(v) for v in h["precipitation"]]))
        times = [datetime.fromisoformat(t) for t in h["time"]]
        now = now or datetime.fromisoformat((loc.get("current") or {}).get("time", h["time"][-1]))
    if not series:
        raise ValueError("The live rainfall file holds no hourly data.")
    mean = np.mean(series, axis=0)
    start, end = now - timedelta(hours=hours_before), now + timedelta(hours=hours_after)
    keep = [k for k, t in enumerate(times) if start < t <= end]
    mm = mean[keep]
    edges = np.arange(len(mm) + 1) * 3600.0
    return edges, mm / 1000.0 / 3600.0, f"live rain from {times[keep[0]]:%Y-%m-%d %H:%M} to {times[keep[-1]]:%Y-%m-%d %H:%M}"


# -------------------------------------------------
# GRIDS
# -------------------------------------------------
def coarsen(arr, valid, f, reducer="mean"):
    """Aggregate f x f blocks. Blocks with fewer than half valid cells become NaN."""
    h, w = arr.shape
    bh, bw = h // f, w // f
    a = arr[:bh * f, :bw * f]
    v = valid[:bh * f, :bw * f]
    cnt = v.reshape(bh, f, bw, f).sum(axis=(1, 3))
    if reducer == "max":
        out = np.where(v, a, -np.inf).reshape(bh, f, bw, f).max(axis=(1, 3))
        return np.where(cnt >= (f * f) / 2, out, np.nan)
    s = np.where(v, a, 0.0).reshape(bh, f, bw, f).sum(axis=(1, 3))
    return np.where(cnt >= (f * f) / 2, s / np.maximum(cnt, 1), np.nan)


def find_outlets(active, acc_block, fraction):
    """Boundary cells where a large drainage area leaves the model (the river exits here)."""
    p = np.pad(active, 1, constant_values=False)
    interior = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    boundary = active & ~interior
    acc = np.where(boundary, np.nan_to_num(acc_block, nan=0.0), 0.0)
    if acc.max() <= 0:
        return boundary
    outlet = acc >= fraction * acc.max()
    return outlet & boundary


def save_tif(path, arr, transform, crs):
    out = np.where(np.isfinite(arr), arr, -9999.0).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=out.shape[0], width=out.shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=-9999.0, compress="lzw") as d:
        d.write(out, 1)


# -------------------------------------------------
# FLOODED ASSETS
# -------------------------------------------------
def _sample(grid, transform, crs, lons, lats):
    xs, ys = warp_transform("EPSG:4326", crs, list(lons), list(lats))
    rows, cols = rowcol(transform, list(xs), list(ys))
    rows, cols = np.asarray(rows), np.asarray(cols)
    ok = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
    out = np.full(len(rows), np.nan)
    out[ok] = grid[rows[ok], cols[ok]]
    return out, np.asarray(xs), np.asarray(ys)


def asset_flooding(infra_dir, maxd, arrival, transform, crs):
    summary, listed = {}, []
    if not os.path.isdir(infra_dir):
        return summary, listed

    def load(name):
        path = os.path.join(infra_dir, f"{name}.geojson")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("features", [])

    for cat in POINT_ASSETS:
        feats = load(cat)
        if feats is None:
            continue
        pts = [f for f in feats if f["geometry"]["type"] == "Point"]
        if not pts:
            summary[cat] = {"unit": "count", "in_model": 0, "depth_ge_0.1m": 0, "depth_ge_0.5m": 0, "depth_ge_1m": 0}
            continue
        lons = [f["geometry"]["coordinates"][0] for f in pts]
        lats = [f["geometry"]["coordinates"][1] for f in pts]
        d, _, _ = _sample(maxd, transform, crs, lons, lats)
        a, _, _ = _sample(arrival, transform, crs, lons, lats)
        inside = np.isfinite(d)
        summary[cat] = {"unit": "count", "in_model": int(inside.sum()),
                        "depth_ge_0.1m": int(np.nansum(d >= 0.1)), "depth_ge_0.5m": int(np.nansum(d >= 0.5)),
                        "depth_ge_1m": int(np.nansum(d >= 1.0))}
        if cat in LISTED_ASSETS:
            for f, lon, lat, depth, arr in zip(pts, lons, lats, d, a):
                if np.isfinite(depth) and depth >= 0.1:
                    p = f["properties"]
                    listed.append({"category": cat, "name": str(p.get("name", "(unnamed)")).replace("\n", " ")[:80],
                                   "type": p.get("amenity") or p.get("healthcare") or p.get("place") or "",
                                   "max_depth_m": round(float(depth), 2),
                                   "arrival_h": round(float(arr), 1) if np.isfinite(arr) else "",
                                   "lat": round(lat, 6), "lon": round(lon, 6)})

    for cat in LINE_ASSETS:
        feats = load(cat)
        if feats is None:
            continue
        lines = [f["geometry"]["coordinates"] for f in feats if f["geometry"]["type"] == "LineString"]
        if not lines:
            summary[cat] = {"unit": "km", "in_model": 0.0, "depth_ge_0.1m": 0.0, "depth_ge_0.3m": 0.0, "depth_ge_0.5m": 0.0}
            continue
        counts = [len(c) for c in lines]
        lon = np.array([p[0] for c in lines for p in c])
        lat = np.array([p[1] for c in lines for p in c])
        xs, ys = warp_transform("EPSG:4326", crs, list(lon), list(lat))
        xs, ys = np.asarray(xs), np.asarray(ys)
        bounds = np.cumsum([0] + counts)
        starts = np.concatenate([np.arange(a, b - 1) for a, b in zip(bounds[:-1], bounds[1:])])
        x0, x1, y0, y1 = xs[starts], xs[starts + 1], ys[starts], ys[starts + 1]
        seg = np.hypot(x1 - x0, y1 - y0)
        rows, cols = rowcol(transform, list((x0 + x1) / 2), list((y0 + y1) / 2))
        rows, cols = np.asarray(rows), np.asarray(cols)
        ok = (rows >= 0) & (rows < maxd.shape[0]) & (cols >= 0) & (cols < maxd.shape[1])
        d = np.full(len(seg), np.nan)
        d[ok] = maxd[rows[ok], cols[ok]]
        inside = np.isfinite(d)
        summary[cat] = {"unit": "km", "in_model": round(float(seg[inside].sum()) / 1000, 1),
                        "depth_ge_0.1m": round(float(seg[d >= 0.1].sum()) / 1000, 1),
                        "depth_ge_0.3m": round(float(seg[d >= 0.3].sum()) / 1000, 1),
                        "depth_ge_0.5m": round(float(seg[d >= 0.5].sum()) / 1000, 1)}
    listed.sort(key=lambda r: (-r["max_depth_m"], r["category"], r["name"]))
    return summary, listed


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="2D hydraulic flood simulation")
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--name", default=None, help="output folder name")
    ap.add_argument("--resolution", type=int, default=60, help="cell size in metres: 30, 60, 90 or 120")
    ap.add_argument("--hours", type=float, default=None, help="simulated duration in hours")
    ap.add_argument("--rain-mm", type=float, default=None)
    ap.add_argument("--rain-hours", type=float, default=None)
    ap.add_argument("--shape", choices=["uniform", "triangular", "burst"], default=None)
    ap.add_argument("--hyetograph", default=None, help="CSV with columns hour,mm_per_hour")
    ap.add_argument("--live", action="store_true", help="use the saved live rainfall (last 24 h + forecast)")
    ap.add_argument("--uniform-rain", action="store_true", help="ignore the CHIRPS spatial pattern")
    ap.add_argument("--n", type=float, default=DEFAULTS["n"], help="Manning roughness")
    ap.add_argument("--initial-loss-mm", type=float, default=DEFAULTS["initial_loss_mm"])
    ap.add_argument("--infil-mm-h", type=float, default=DEFAULTS["infil_mm_h"])
    ap.add_argument("--soil-storage-mm", type=float, default=DEFAULTS["soil_storage_mm"])
    ap.add_argument("--outlet-fraction", type=float, default=0.01, help="boundary cells with at least this share of the maximum drainage area are outlets")
    ap.add_argument("--inflow-file", default=None, help="JSON list of {lon, lat, csv} inflow hydrographs (hour,m3s)")
    ap.add_argument("--snapshot-hours", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true", help="show the grid size and memory need, then stop")
    args = ap.parse_args()

    if args.list:
        for k, v in SCENARIOS.items():
            print(f"{k:<22} {v['rain_mm']} mm over {v['rain_hours']} h ({v['shape']}), simulated for {v['hours']} h")
        return 0

    setup_logging()
    print("=" * 70)
    print("DUWE AI FLOOD INTELLIGENCE ENGINE")
    print("Hydraulic simulation (2D, rain on the whole catchment)")
    print("=" * 70)

    if args.resolution % 30 != 0 or args.resolution < 30:
        print("Resolution must be a multiple of 30 m (30, 60, 90, 120).")
        return 1
    f = args.resolution // 30

    # ---- rainfall scenario ----
    label = ""
    if args.live:
        edges, rates, label = hyetograph_from_live(LIVE_JSON)
        scen = {"hours": args.hours or float(edges[-1] / 3600)}
        name = args.name or "live_latest"
    elif args.hyetograph:
        edges, rates = hyetograph_from_csv(args.hyetograph)
        scen = {"hours": args.hours or float(edges[-1] / 3600) + 12}
        name = args.name or os.path.splitext(os.path.basename(args.hyetograph))[0]
        label = f"hyetograph from {os.path.basename(args.hyetograph)}"
    else:
        key = args.scenario or (None if args.rain_mm else DEFAULT_SCENARIO)
        if key and key not in SCENARIOS:
            print(f"Unknown scenario '{key}'. Use --list.")
            return 1
        base = dict(SCENARIOS.get(key, {"rain_mm": 100, "rain_hours": 24, "shape": "uniform", "hours": 36}))
        if args.rain_mm is not None:
            base["rain_mm"] = args.rain_mm
        if args.rain_hours is not None:
            base["rain_hours"] = args.rain_hours
        if args.shape:
            base["shape"] = args.shape
        if args.hours is None and args.rain_hours is not None:
            base["hours"] = base["rain_hours"] + 12
        edges, rates = design_hyetograph(base["rain_mm"], base["rain_hours"], base["shape"])
        scen = base
        name = args.name or key or f"storm_{base['rain_mm']:.0f}mm_{base['rain_hours']:.0f}h"
        label = f"{base['rain_mm']} mm over {base['rain_hours']} h ({base['shape']})"
    hours = args.hours or scen["hours"]
    total_mm = float(np.sum(rates * np.diff(edges)) * 1000)

    # ---- terrain ----
    for p in (DEM_PATH, ACC_PATH, CATCHMENT_PATH):
        if not os.path.exists(p):
            logging.error(f"Required map missing: {p}")
            return 1
    with rasterio.open(DEM_PATH) as s:
        dem = s.read(1).astype("float32"); crs = s.crs; tr = s.transform
    with rasterio.open(ACC_PATH) as s:
        acc = s.read(1).astype("float32")
    with rasterio.open(CATCHMENT_PATH) as s:
        catch = s.read(1)
    valid = (catch == 1) & np.isfinite(dem) & (dem > -1000)
    z = coarsen(dem, valid, f) if f > 1 else np.where(valid, dem, np.nan)
    acc_b = coarsen(np.where(np.isfinite(acc), acc, 0), valid, f, "max") if f > 1 else np.where(valid, acc, np.nan)
    active = np.isfinite(z)
    ny, nx = z.shape
    dx = float(args.resolution)
    tr_sim = tr * Affine.scale(f, f)
    outlet = find_outlets(active, acc_b, args.outlet_fraction)

    print(f"Scenario        : {name}  ({label})")
    print(f"Rain total      : {total_mm:.0f} mm | simulated time {hours:.0f} h")
    print(f"Grid            : {ny} x {nx} cells at {args.resolution} m | active cells {active.sum():,} | outlet cells {int(outlet.sum())}")
    print(f"Memory need     : about {he.estimate_memory_mb(ny, nx):,.0f} MB")
    print(f"Losses          : initial {args.initial_loss_mm} mm, infiltration {args.infil_mm_h} mm/h, soil storage {args.soil_storage_mm} mm | Manning n {args.n}")
    if args.dry_run:
        print("\nDry run only. Nothing was simulated.")
        return 0

    # ---- spatial rain pattern from CHIRPS (keeps the total, redistributes it) ----
    rain_factor = None
    if not args.uniform_rain and os.path.exists(CHIRPS_PATH):
        try:
            with rasterio.open(CHIRPS_PATH) as s:
                ch = s.read(1).astype("float32")
            ok = valid & np.isfinite(ch) & (ch > 0)
            c = coarsen(np.where(ok, ch, 0), ok, f) if f > 1 else np.where(ok, ch, np.nan)
            fac = c / np.nanmean(c[active])
            fac = np.clip(np.where(np.isfinite(fac), fac, 1.0), 0.6, 1.4)
            rain_factor = (fac / fac[active].mean()).astype("float32")
            print(f"Rain pattern    : CHIRPS seasonal pattern (range {rain_factor[active].min():.2f}-{rain_factor[active].max():.2f}, total volume kept)")
        except Exception as ex:
            logging.warning(f"Could not use the CHIRPS pattern: {ex}")
    if rain_factor is None:
        print("Rain pattern    : uniform")

    # ---- optional inflows ----
    inflows = []
    if args.inflow_file:
        with open(args.inflow_file, "r", encoding="utf-8") as fh:
            for item in json.load(fh):
                x, y = warp_transform("EPSG:4326", crs, [item["lon"]], [item["lat"]])
                r, c = rowcol(tr_sim, x[0], y[0])
                if 0 <= r < ny and 0 <= c < nx and active[r, c]:
                    t_h, q = [], []
                    with open(item["csv"], newline="", encoding="utf-8-sig") as cf:
                        for row in csv.DictReader(cf):
                            k = {a.strip().lower(): b for a, b in row.items()}
                            t_h.append(float(k["hour"])); q.append(float(k["m3s"]))
                    inflows.append({"i": r, "j": c, "t_s": np.array(t_h) * 3600.0, "q_m3s": np.array(q)})
                else:
                    logging.warning(f"Inflow at ({item['lon']}, {item['lat']}) is outside the model and was ignored.")
        print(f"Inflows         : {len(inflows)}")

    # ---- run ----
    pool = int(np.ceil(max(ny, nx) / 450))
    snaps, snap_t = [], []

    def on_snap(t, depth):
        by, bx = ny // pool, nx // pool
        a = np.where(np.isfinite(depth), depth, -1.0)[:by * pool, :bx * pool].reshape(by, pool, bx, pool).max(axis=(1, 3))
        snaps.append(np.where(a < 0, np.nan, a).astype("float16"))
        snap_t.append(t / 3600.0)

    def on_prog(t, T, steps, hmax, wall):
        done = t / T
        eta = wall / done * (1 - done) if done > 0 else 0
        print(f"  simulated {t / 3600:5.1f} / {T / 3600:.0f} h ({done * 100:3.0f}%) | steps {steps:,} | deepest cell {hmax:.2f} m | elapsed {wall / 60:.1f} min | about {eta / 60:.1f} min left", flush=True)

    print("\nRunning (the first start also compiles the solver, which takes a little while)...")
    res = he.simulate(
        z, dx, hours, edges, rates, outlet, n_manning=args.n, rain_factor=rain_factor,
        init_abstraction_m=args.initial_loss_mm / 1000.0, infil_rate_ms=args.infil_mm_h / 1000.0 / 3600.0,
        soil_storage_m=args.soil_storage_mm / 1000.0, inflows=inflows,
        snapshot_interval_s=args.snapshot_hours * 3600.0, on_snapshot=on_snap, on_progress=on_prog)
    print(f"\nFinished in {res['runtime_s'] / 60:.1f} min | {res['steps']:,} time steps | median step {res['dt_median_s']:.1f} s")

    # ---- save ----
    out = os.path.join(OUT_ROOT, name)
    os.makedirs(out, exist_ok=True)
    for key, fname in [("max_depth", "max_depth.tif"), ("max_speed", "max_speed.tif"), ("max_hazard", "max_hazard.tif"),
                       ("arrival_h", "arrival_h.tif"), ("wet_hours", "wet_hours.tif")]:
        save_tif(os.path.join(out, fname), res[key], tr_sim, crs)
    if snaps:
        np.savez_compressed(os.path.join(out, "snapshots.npz"), depth=np.stack(snaps), hours=np.array(snap_t), pool=pool)

    cell_km2 = (dx * dx) / 1e6
    maxd = res["max_depth"]
    flooded = {lab: round(float(np.nansum((maxd >= lo) & (maxd < hi))) * cell_km2, 2) for lab, lo, hi in DEPTH_CLASSES}
    area_total = float(active.sum()) * cell_km2
    assets, listed = asset_flooding(INFRA_DIR, maxd, res["arrival_h"], tr_sim, crs)
    if listed:
        with open(os.path.join(out, "asset_flooding.csv"), "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(listed[0].keys()))
            w.writeheader(); w.writerows(listed)

    mb = res["mass_balance"]
    summary = {
        "name": name, "created": datetime.now().isoformat(timespec="seconds"), "scenario": label,
        "rain_total_mm": round(total_mm, 1), "simulated_hours": hours, "resolution_m": args.resolution,
        "grid": [ny, nx], "active_km2": round(area_total, 1), "outlet_cells": int(outlet.sum()),
        "parameters": {"manning_n": args.n, "initial_loss_mm": args.initial_loss_mm, "infiltration_mm_h": args.infil_mm_h,
                       "soil_storage_mm": args.soil_storage_mm, "chirps_pattern": rain_factor is not None},
        "flooded_area_km2": flooded,
        "flooded_ge_0.1m_km2": round(float(np.nansum(maxd >= 0.1)) * cell_km2, 2),
        "flooded_ge_0.1m_percent": round(100 * float(np.nansum(maxd >= 0.1)) * cell_km2 / area_total, 2),
        "max_depth_m": round(float(np.nanmax(maxd)), 2),
        "p99_depth_m_where_flooded": round(float(np.nanpercentile(maxd[maxd >= 0.1], 99)), 2) if np.any(maxd >= 0.1) else 0.0,
        "max_speed_ms": round(float(np.nanmax(res["max_speed"])), 2),
        "assets": assets, "snapshot_pool": pool,
        "mass_balance_m3": {k: round(v, 1) for k, v in mb.items()},
        "runtime_min": round(res["runtime_s"] / 60, 1), "steps": res["steps"],
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("\nRESULTS")
    print("-" * 70)
    print(f"Flooded area (max depth >= 0.1 m): {summary['flooded_ge_0.1m_km2']:,.1f} km2 ({summary['flooded_ge_0.1m_percent']:.1f}% of the model)")
    for lab, v in flooded.items():
        print(f"   {lab:<10} {v:>9,.2f} km2")
    print(f"Deepest water: {summary['max_depth_m']} m | fastest flow: {summary['max_speed_ms']} m/s")
    for cat, s in assets.items():
        if s["unit"] == "count":
            print(f"   {cat:<13} {s['depth_ge_0.1m']:>6,} of {s['in_model']:>7,} flooded (>=0.1 m) | {s['depth_ge_0.5m']:,} at >=0.5 m")
        else:
            print(f"   {cat:<13} {s['depth_ge_0.3m']:>6,.1f} km of {s['in_model']:,.1f} km under >=0.3 m of water")
    print(f"Water balance error: {mb['balance_error_percent']:.4f}%")
    print(f"\nSaved in: {out}")
    print("\nHOW TO READ THIS: a screening-level scenario, not a forecast. Rain-on-grid only (no river inflow unless")
    print("you supplied one), no channel bathymetry, placeholder losses and roughness, not yet calibrated or validated.")
    logging.info(f"Hydraulic run '{name}' finished: {summary['flooded_ge_0.1m_km2']} km2 flooded, {res['runtime_s']/60:.1f} min")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as ex:
        logging.error(f"Hydraulic simulation failed: {ex}")
        print(f"Error: {ex}")
        sys.exit(1)
