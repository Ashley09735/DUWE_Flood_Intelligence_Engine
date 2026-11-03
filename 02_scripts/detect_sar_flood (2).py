"""
DUWE SENTINEL-1 FLOOD DETECTION (detect_sar_flood.py)

Put this file in:
    <BASE_DIR>\\02_scripts\\flood_detection\\detect_sar_flood.py

What it does
------------
1. Takes the study area from your risk raster (or a bounding box you give).
2. Finds Sentinel-1 radar images from the last N days (Google Earth Engine).
3. Compares them with a one-year reference of normal backscatter.
4. Marks pixels that became dark (open water) and were not water before.
5. Removes permanent water, steep slopes and high-ground cells.
6. Downloads the result and saves, in <BASE_DIR>\\01_data\\flood_observed\\:
       flood_extent_YYYY-MM-DD.tif   1 = flooded, 0 = dry, 255 = no satellite data
       flood_extent_YYYY-MM-DD.json  sidecar with start_date / end_date, settings, stats
   The date in the name is the date of the NEWEST radar image used, not today.

One-time setup
--------------
    pip install earthengine-api rasterio numpy
    earthengine authenticate
    (your Google Cloud project must be registered for Earth Engine:
     https://code.earthengine.google.com/register)

Run
---
    python detect_sar_flood.py --days 12
    python detect_sar_flood.py --days 24 --end-date 2026-03-15   (test on a past event)
    (the project capable-country-434808-n6 is built in; override with --project
     or the EE_PROJECT environment variable)

Limits you should know about
----------------------------
* Radar sees open water. It can MISS flooding under dense vegetation or among
  buildings, so the extent is a minimum.
* Sentinel-1 revisits roughly every 6-12 days over Malawi. The map shows water
  at the time of the overpass, not live conditions.
* Thresholds are generic. Check results against a known event and tune
  --water-db / --diff-db before relying on them.
* Decision support only. Not an official warning.
"""
import argparse
import io
import json
import math
import os
import sys
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

DEFAULT_PROJECT = "capable-country-434808-n6"   # Google Cloud project registered for Earth Engine
RISK_FILE = r"01_data\processed\risk_updated_classified_duwe.tif"
OUT_SUBDIR = r"01_data\flood_observed"
MAX_DOWNLOAD_PIXELS = 25_000_000   # Earth Engine direct-download size limit


def parse_args():
    here = Path(__file__).resolve()
    default_base = here.parents[2] if len(here.parents) > 2 else Path.cwd()
    p = argparse.ArgumentParser(description="Detect floods with Sentinel-1 (Earth Engine).")
    p.add_argument("--days", type=int, default=12, help="Look-back window in days (default 12).")
    p.add_argument("--end-date", help="Last day of the window, YYYY-MM-DD (default: today). "
                   "Use it to test on a past flood.")
    p.add_argument("--project", default=os.environ.get("EE_PROJECT", DEFAULT_PROJECT),
                   help="Google Cloud project registered for Earth Engine.")
    p.add_argument("--base-dir", default=str(default_base), help="DUWE project folder.")
    p.add_argument("--bbox", help="minlon,minlat,maxlon,maxlat (overrides the risk raster).")
    p.add_argument("--scale", type=float, default=30.0, help="Output pixel size in metres.")
    p.add_argument("--ref-days", type=int, default=365, help="Length of the reference period.")
    p.add_argument("--water-db", type=float, default=-15.0, help="VV below this = water.")
    p.add_argument("--diff-db", type=float, default=-3.0, help="Drop vs reference needed.")
    p.add_argument("--perm-water", type=float, default=25.0,
                   help="Exclude cells where JRC water occurrence (%%) is above this.")
    p.add_argument("--max-slope", type=float, default=5.0, help="Exclude slopes above (deg).")
    p.add_argument("--max-hand", type=float, default=15.0,
                   help="Exclude cells higher than this above drainage (m).")
    p.add_argument("--tile", type=int, default=800,
                   help="Download tile size in pixels (smaller = safer, default 800).")
    p.add_argument("--min-pixels", type=int, default=4, help="Drop flood patches smaller than this.")
    return p.parse_args()


def study_bbox(args):
    if args.bbox:
        a = [float(v) for v in args.bbox.split(",")]
        if len(a) != 4:
            raise SystemExit("--bbox needs four numbers: minlon,minlat,maxlon,maxlat")
        return a
    risk = os.path.join(args.base_dir, RISK_FILE)
    if not os.path.exists(risk):
        raise SystemExit(f"Risk raster not found: {risk}\nUse --bbox or --base-dir.")
    with rasterio.open(risk) as src:
        if src.crs is None:
            raise SystemExit("The risk raster has no CRS. Use --bbox.")
        return list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def init_ee(project):
    try:
        import ee
    except ImportError:
        raise SystemExit("Install the Earth Engine library first:  pip install earthengine-api")

    def _init():
        ee.Initialize(project=project) if project else ee.Initialize()

    try:
        _init()
    except Exception as exc:
        if not sys.stdin or not sys.stdin.isatty():
            raise SystemExit(
                "Earth Engine is not signed in (or no project was given).\n"
                "Run once in a terminal:  earthengine authenticate\n"
                "and pass --project YOUR_GCP_PROJECT (or set EE_PROJECT).\n"
                f"Details: {exc}")
        ee.Authenticate()
        _init()
    return ee


def build_flood_image(ee, region, start, end, ref_days, a):
    """Detect flooding separately for every Sentinel-1 orbit, then combine them."""
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filterBounds(region))

    recent_all = s1.filterDate(start, end)
    n = recent_all.size().getInfo()
    if n == 0:
        raise SystemExit(f"No Sentinel-1 images over the study area between "
                         f"{start:%Y-%m-%d} and {end:%Y-%m-%d}. Try a longer --days.")
    orbits = sorted(set(recent_all.aggregate_array("relativeOrbitNumber_start").getInfo()))
    print(f"{n} image(s) found on relative orbit(s): {', '.join(str(o) for o in orbits)}")

    smooth = lambda img: img.focal_median(50, "circle", "meters")
    perm = (ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
            .unmask(0).gt(a.perm_water))
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    hnd = ee.Image("MERIT/Hydro/v1_0_1").select("hnd")
    keep = perm.Not().And(slope.lt(a.max_slope)).And(hnd.lt(a.max_hand))

    floods, valids, all_dates, used = [], [], [], []
    ref_window = s1.filterDate(start - timedelta(days=ref_days), start)
    for orb in orbits:
        rec = recent_all.filter(ee.Filter.eq("relativeOrbitNumber_start", orb))
        ref = ref_window.filter(ee.Filter.eq("relativeOrbitNumber_start", orb))
        n_ref = ref.size().getInfo()
        times = sorted(rec.aggregate_array("system:time_start").getInfo())
        if n_ref < 3:
            print(f"  orbit {orb}: only {n_ref} reference images, skipped.")
            continue
        print(f"  orbit {orb}: {len(times)} recent image(s), {n_ref} reference images.")
        after = smooth(rec.select("VV").mean())
        before = smooth(ref.select("VV").median())
        fl = after.lt(a.water_db).And(after.subtract(before).lt(a.diff_db)).And(keep)
        floods.append(fl.unmask(0))
        valids.append(after.mask().And(before.mask()).unmask(0))
        all_dates += [datetime.fromtimestamp(t / 1000, tz=timezone.utc).date() for t in times]
        used.append(orb)

    if not floods:
        raise SystemExit("No orbit had enough reference images. Increase --ref-days.")

    flood = ee.ImageCollection(floods).max().selfMask()
    valid = ee.ImageCollection(valids).max().gt(0)
    cc = flood.connectedPixelCount(100, True)
    flood = flood.updateMask(cc.gte(a.min_pixels)).unmask(0)

    out = ee.Image(255).where(valid, 0).where(flood.And(valid), 1).toUint8().rename("flood")
    info = {"images": len(all_dates), "reference_images": len(used),
            "orbit_pass": "relative orbits " + ", ".join(str(o) for o in used),
            "first_image": min(all_dates), "last_image": max(all_dates)}
    return out, info


def _fetch(url):
    """Download one tile; surface Earth Engine's real error message."""
    try:
        with urllib.request.urlopen(url, timeout=900) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")[:600]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code} from Earth Engine: {body or exc.reason}")
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = next(n for n in z.namelist() if n.lower().endswith((".tif", ".tiff")))
            data = z.read(name)
    return data


def download_tiles(ee, image, bbox, scale, tile_px):
    """Download the flood image in small tiles and stitch them on one grid."""
    from rasterio.io import MemoryFile

    minlon, minlat, maxlon, maxlat = bbox
    crs = utm_epsg((minlon + maxlon) / 2, (minlat + maxlat) / 2)
    xmin, ymin, xmax, ymax = transform_bounds("EPSG:4326", crs, *bbox)
    x0 = math.floor(xmin / scale) * scale
    y1 = math.ceil(ymax / scale) * scale
    W = int(math.ceil((xmax - x0) / scale))
    H = int(math.ceil((y1 - ymin) / scale))
    arr = np.full((H, W), 255, dtype="uint8")

    nx, ny = math.ceil(W / tile_px), math.ceil(H / tile_px)
    print(f"Grid {W} x {H} px at {scale:.0f} m in {crs}; downloading {nx * ny} tile(s) ...")
    done = 0
    for j in range(ny):
        for i in range(nx):
            c0, c1 = i * tile_px, min(W, (i + 1) * tile_px)
            r0, r1 = j * tile_px, min(H, (j + 1) * tile_px)
            rect = [x0 + c0 * scale, y1 - r1 * scale, x0 + c1 * scale, y1 - r0 * scale]
            region = ee.Geometry.Rectangle(rect, proj=crs, geodesic=False)
            url = None
            for attempt in (1, 2):
                try:
                    url = image.getDownloadURL({"name": "flood", "scale": scale, "crs": crs,
                                                "region": region, "format": "GEO_TIFF"})
                    data = _fetch(url)
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise RuntimeError(
                            f"Tile {j * nx + i + 1}/{nx * ny} failed: {exc}\n"
                            "Try a smaller --tile (e.g. 500) or a shorter --ref-days (e.g. 180).")
                    print(f"Tile {j * nx + i + 1} failed once ({exc}); retrying ...")
            with MemoryFile(data) as mf, mf.open() as ds:
                t = ds.read(1)
                col = int(round((ds.transform.c - x0) / scale))
                row = int(round((y1 - ds.transform.f) / scale))
            h, w = t.shape
            rr0, cc0 = max(row, 0), max(col, 0)
            rr1, cc1 = min(row + h, H), min(col + w, W)
            if rr1 > rr0 and cc1 > cc0:
                arr[rr0:rr1, cc0:cc1] = t[rr0 - row:rr1 - row, cc0 - col:cc1 - col]
            done += 1
            print(f"  tile {done}/{nx * ny} ok")
    return arr, Affine(scale, 0, x0, 0, -scale, y1), crs, scale


def main():
    a = parse_args()
    bbox = study_bbox(a)
    print(f"Study area (lon/lat): {bbox[0]:.4f}, {bbox[1]:.4f} to {bbox[2]:.4f}, {bbox[3]:.4f}")

    ee = init_ee(a.project)
    region = ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)
    if a.end_date:
        end = datetime.strptime(a.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    else:
        end = datetime.now(timezone.utc) + timedelta(days=1)
    start = end - timedelta(days=a.days + 1)

    image, info = build_flood_image(ee, region, start, end, a.ref_days, a)
    print(f"Sentinel-1: {info['images']} image(s) from {info['first_image']} to "
          f"{info['last_image']} ({info['orbit_pass']}).")

    arr, transform, crs, scale = download_tiles(ee, image, bbox, a.scale, a.tile)

    out_dir = os.path.join(a.base_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    tif = os.path.join(out_dir, f"flood_extent_{info['last_image'].isoformat()}.tif")
    with rasterio.open(tif, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="uint8", crs=crs, transform=transform,
                       nodata=255, compress="lzw") as dst:
        dst.write(arr, 1)
    px_km2 = abs(transform.a * transform.e) / 1e6
    valid = arr != 255
    flood_km2 = float((arr == 1).sum()) * px_km2
    cover = 100.0 * valid.sum() / arr.size if arr.size else 0.0

    meta = {
        "source": "Sentinel-1 GRD (VV) change detection, Google Earth Engine",
        "start_date": info["first_image"].isoformat(),
        "end_date": info["last_image"].isoformat(),
        "images_used": info["images"],
        "orbit_pass": info["orbit_pass"],
        "pixel_size_m": round(scale, 1),
        "flood_km2": round(flood_km2, 2),
        "coverage_pct": round(cover, 1),
        "thresholds": {"water_db": a.water_db, "diff_db": a.diff_db,
                       "perm_water_pct": a.perm_water, "max_slope_deg": a.max_slope,
                       "max_hand_m": a.max_hand, "min_pixels": a.min_pixels},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.splitext(tif)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {tif}")
    print(f"Flooded area: {flood_km2:,.1f} km2 | satellite coverage: {cover:.0f}%")
    if cover < 90:
        print("Warning: coverage is below 90%, so the true extent may be larger.")
        print("Try a longer window, e.g. --days 24, so more orbits are included.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Detection failed: {exc}")
        sys.exit(1)
