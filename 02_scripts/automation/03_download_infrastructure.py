"""
DUWE AI FLOOD INTELLIGENCE ENGINE
Stage 1 - Infrastructure & Vulnerable Assets Download (OpenStreetMap)

Downloads schools, health facilities, villages, roads, bridges, buildings, water
points, markets, places of worship and power sites for the study area, using the
free OpenStreetMap Overpass API. Needs internet, but only for the download:
files are cached, and if the network is down the last good files are kept.

Usage (from 02_scripts\\automation):
    python 03_download_infrastructure.py                  # all categories (skips fresh ones)
    python 03_download_infrastructure.py --only schools,villages
    python 03_download_infrastructure.py --force          # re-download even if fresh
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime

import requests
import rasterio
from rasterio.warp import transform_bounds

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
CATCHMENT_PATH = os.path.join(BASE_DIR, r"01_data\processed\catchment_duwe.tif")
OUT_DIR = os.path.join(BASE_DIR, r"01_data\raw\infrastructure")
LOG_DIR = os.path.join(BASE_DIR, r"02_scripts\automation\logs")
STATUS_PATH = os.path.join(OUT_DIR, "infrastructure_status.json")

CACHE_DAYS = 30            # infrastructure changes slowly; re-download after this many days
PAUSE_BETWEEN_TILES = 2    # seconds, to be polite to the free public servers
BBOX_MARGIN_DEG = 0.01

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
USER_AGENT = "DUWE-Flood-Intelligence/1.0 (disaster risk research, Chikwawa, Malawi)"

ROAD_TYPES = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|track"

# geom "point" -> one point per feature (centre of buildings/areas)
# geom "line"  -> full line geometry (roads, bridges)
# tiles        -> the study area is split into tiles x tiles requests (bigger = safer for heavy data)
CATEGORIES = {
    "schools":      {"geom": "point", "tiles": 2,
                     "filters": ['nwr["amenity"~"^(school|kindergarten|college|university)$"]']},
    "health":       {"geom": "point", "tiles": 2,
                     "filters": ['nwr["amenity"~"^(hospital|clinic|doctors|pharmacy)$"]', 'nwr["healthcare"]']},
    "villages":     {"geom": "point", "tiles": 2,
                     "filters": ['nwr["place"~"^(city|town|village|hamlet|suburb|neighbourhood|isolated_dwelling)$"]']},
    "water_points": {"geom": "point", "tiles": 2,
                     "filters": ['nwr["amenity"="drinking_water"]',
                                 'nwr["man_made"~"^(water_well|water_tower|water_works)$"]']},
    "markets":      {"geom": "point", "tiles": 2,
                     "filters": ['nwr["amenity"="marketplace"]']},
    "worship":      {"geom": "point", "tiles": 2,
                     "filters": ['nwr["amenity"="place_of_worship"]']},
    "power":        {"geom": "point", "tiles": 2,
                     "filters": ['nwr["power"~"^(substation|plant|generator)$"]']},
    "bridges":      {"geom": "line", "tiles": 3,
                     "filters": ['way["bridge"="yes"]["highway"]']},
    "roads":        {"geom": "line", "tiles": 4,
                     "filters": [f'way["highway"~"^({ROAD_TYPES})$"]']},
    "buildings":    {"geom": "point", "tiles": 6,
                     "filters": ['nwr["building"]']},
}

KEEP_TAGS = ("name", "amenity", "healthcare", "highway", "surface", "bridge", "building",
             "place", "population", "operator", "man_made", "power", "isced:level", "emergency")


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    log_file = os.path.join(LOG_DIR, f"download_infrastructure_{datetime.now().strftime('%Y%m%d')}.log")
    logging.basicConfig(filename=log_file, encoding="utf-8", level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)


def catchment_bbox(path=CATCHMENT_PATH):
    """(south, west, north, east) in degrees, from the catchment raster."""
    with rasterio.open(path) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    m = BBOX_MARGIN_DEG
    return (south - m, west - m, north + m, east + m)


def make_tiles(bbox, n):
    south, west, north, east = bbox
    dy, dx = (north - south) / n, (east - west) / n
    for i in range(n):
        for j in range(n):
            yield (south + i * dy, west + j * dx, south + (i + 1) * dy, west + (j + 1) * dx)


def build_query(filters, tile, geom):
    s, w, n, e = tile
    body = "".join(f"{f}({s:.5f},{w:.5f},{n:.5f},{e:.5f});" for f in filters)
    out = "out geom tags;" if geom == "line" else "out center tags;"
    return f"[out:json][timeout:180];({body});{out}"


def overpass(query):
    """POST a query, trying each server and retrying with back-off. Returns the element list."""
    last_error = "unknown error"
    for attempt in range(3):
        for url in ENDPOINTS:
            try:
                r = requests.post(url, data={"data": query},
                                  headers={"User-Agent": USER_AGENT}, timeout=240)
                if r.status_code == 200:
                    return r.json().get("elements", [])
                last_error = f"HTTP {r.status_code} from {url}"
                if r.status_code in (429, 504):
                    time.sleep(15 * (attempt + 1))
            except (requests.RequestException, ValueError) as ex:
                last_error = f"{type(ex).__name__}: {ex}"
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(last_error)


def to_feature(el, geom):
    """Convert one Overpass element into a GeoJSON feature (or None)."""
    tags = {k: v for k, v in el.get("tags", {}).items() if k in KEEP_TAGS}
    tags["osm_id"] = f'{el.get("type")}/{el.get("id")}'
    if geom == "line":
        pts = el.get("geometry")
        if not pts or len(pts) < 2:
            return None
        geometry = {"type": "LineString", "coordinates": [[p["lon"], p["lat"]] for p in pts]}
    else:
        if el.get("type") == "node":
            lon, lat = el.get("lon"), el.get("lat")
        else:
            c = el.get("center") or {}
            lon, lat = c.get("lon"), c.get("lat")
        if lon is None or lat is None:
            return None
        geometry = {"type": "Point", "coordinates": [lon, lat]}
    return {"type": "Feature", "properties": tags, "geometry": geometry}


def download_category(name, spec, bbox):
    """Download one category tile by tile. Raises if any tile fails (keeps old data safe)."""
    found = {}
    tiles = list(make_tiles(bbox, spec["tiles"]))
    for k, tile in enumerate(tiles, 1):
        logging.info(f"  {name}: tile {k}/{len(tiles)}")
        for el in overpass(build_query(spec["filters"], tile, spec["geom"])):
            key = f'{el.get("type")}/{el.get("id")}'
            if key not in found:
                feat = to_feature(el, spec["geom"])
                if feat:
                    found[key] = feat
        time.sleep(PAUSE_BETWEEN_TILES)
    return list(found.values())


def save_geojson(path, features):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False)
    os.replace(tmp, path)      # never leaves a half-written file behind


def is_fresh(path):
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < CACHE_DAYS * 86400


def load_status():
    try:
        with open(STATUS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-separated categories")
    parser.add_argument("--force", action="store_true", help="ignore the cache")
    args = parser.parse_args()

    setup_logging()
    print("=" * 65)
    print("DUWE AI FLOOD INTELLIGENCE ENGINE")
    print("Infrastructure & Vulnerable Assets Download (OpenStreetMap)")
    print("=" * 65)

    chosen = [c.strip() for c in args.only.split(",") if c.strip()] or list(CATEGORIES)
    unknown = [c for c in chosen if c not in CATEGORIES]
    if unknown:
        print(f"Unknown categories: {unknown}. Choose from: {list(CATEGORIES)}")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    bbox = catchment_bbox()
    logging.info(f"Study area bbox (S,W,N,E): {tuple(round(v, 4) for v in bbox)}")

    status = load_status()
    missing = 0

    for name in chosen:
        spec = CATEGORIES[name]
        path = os.path.join(OUT_DIR, f"{name}.geojson")
        entry = status.get(name, {})

        if not args.force and is_fresh(path):
            entry["status"] = "CACHED"
            logging.info(f"{name}: cached file is under {CACHE_DAYS} days old - skipped")
        else:
            logging.info(f"{name}: downloading...")
            try:
                features = download_category(name, spec, bbox)
                save_geojson(path, features)
                entry = {"status": "DOWNLOADED", "count": len(features),
                         "updated": datetime.now().isoformat(timespec="seconds")}
                logging.info(f"{name}: {len(features):,} features saved")
            except Exception as ex:
                entry["error"] = str(ex)[:200]
                if os.path.exists(path):
                    entry["status"] = "FAILED_USING_OLD_FILE"
                    logging.warning(f"{name}: download failed ({ex}); keeping the previous file")
                else:
                    entry["status"] = "FAILED_NO_DATA"
                    missing += 1
                    logging.error(f"{name}: download failed ({ex}) and no earlier file exists")
        status[name] = entry

    status["_meta"] = {"source": "OpenStreetMap contributors (Overpass API)",
                       "bbox_swne": [round(v, 5) for v in bbox],
                       "checked": datetime.now().isoformat(timespec="seconds")}
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)

    print("\nSUMMARY")
    print("-" * 65)
    for name in chosen:
        e = status[name]
        print(f"  {name:<14} {e.get('status', '?'):<24} {e.get('count', '')}")
    print(f"\nStatus file: {STATUS_PATH}")
    return 1 if missing else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as ex:
        print(f"Error: {ex}")
        sys.exit(1)
