"""
DUWE FLOOD DETECTION MODULE  (flood_detection.py)

Put this file NEXT TO your dashboard script.

What it does
------------
1. Reads a satellite-derived flood map (the "observed" flood).
2. Shows the flooded area on the map.
3. Finds which infrastructure assets were hit by the real flood.
4. Compares the observed flood with the susceptibility model (risk classes).
5. Raises an alert banner at the top of the dashboard.

Where the satellite flood maps live
-----------------------------------
    <BASE_DIR>\\01_data\\flood_observed\\

Accepted files (any CRS, any resolution; they are re-gridded to the risk raster):
    .tif / .tiff   1 = flooded, 0 = dry, 255 or nodata = no satellite data
    .geojson       flood polygons in EPSG:4326
    .shp / .gpkg   flood polygons (needs geopandas)

Put the acquisition date in the file name (flood_extent_2026-11-02.tif) or in
a sidecar .json file with an "end_date" key. detect_sar_flood.py writes both.

Sources of flood maps: the Sentinel-1 script that goes with this module,
Copernicus Global Flood Monitoring (GFM), Copernicus EMS / UNOSAT rapid
mapping products, NASA Disasters, or your own Earth Engine exports.

Note: satellite flood maps show water at the moment of the overpass. They are
decision support, not an official warning.
"""
import io
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine, rowcol
from rasterio.warp import Resampling, reproject, transform as warp_transform, transform_geom

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
FLOOD_EXTS = (".tif", ".tiff", ".geojson", ".shp", ".gpkg")

ALERT_MIN_KM2 = 1.0    # below this, a detection is treated as "no flood"
WARNING_PCT = 1.0      # % of covered study area flooded -> Warning
SEVERE_PCT = 5.0       # % of covered study area flooded -> Severe
SEVERE_ASSETS = 10     # assets flooded -> Severe
STALE_DAYS = 14        # a map older than this is not a "current" picture

CODING = {"1 = flooded (standard)": "eq1", "Any value above 0": "gt0"}
MODEL_THRESHOLDS = {"High and above": 3, "Medium and above": 2, "Very High only": 4}
BUFFERS_M = [0, 30, 60, 100, 250]

S_FLOODED, S_NEAR = "Flooded", "Near flood edge"
S_DRY, S_NODATA, S_OUT = "Not flooded", "No satellite data", "Outside study area"

LEVELS = {
    "severe":  ("#B71C1C", "#7F0000", "#FFFFFF", "SATELLITE FLOOD ALERT: SEVERE"),
    "warning": ("#E65100", "#BF360C", "#FFFFFF", "SATELLITE FLOOD ALERT: WARNING"),
    "watch":   ("#F9A825", "#F57F17", "#1F1500", "SATELLITE FLOOD ALERT: WATCH"),
    "clear":   ("#2E7D5B", "#1B5E3F", "#FFFFFF", "NO ACTIVE FLOODING DETECTED"),
    "stale":   ("#52708B", "#37506B", "#FFFFFF", "SATELLITE FLOOD MAP IS OUT OF DATE"),
    "none":    ("#64748B", "#475569", "#FFFFFF", "NO SATELLITE FLOOD MAP LOADED"),
    "error":   ("#7B1FA2", "#4A148C", "#FFFFFF", "SATELLITE FLOOD MAP COULD NOT BE READ"),
}

_DATE = re.compile(r"(20\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])(?!\d)")


# ------------------------------------------------------------------
# Files and dates
# ------------------------------------------------------------------
def _observed_dir(base_dir):
    return os.path.join(base_dir, "01_data", "flood_observed")


def _sidecar(path):
    side = os.path.splitext(path)[0] + ".json"
    try:
        if os.path.exists(side):
            with open(side, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def map_date(path):
    meta = _sidecar(path)
    for key in ("end_date", "date"):
        try:
            return date.fromisoformat(str(meta[key])[:10])
        except Exception:
            pass
    m = _DATE.search(os.path.basename(path))
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return datetime.fromtimestamp(os.path.getmtime(path)).date()


def list_flood_maps(observed_dir):
    """Newest first: [(date, path), ...]"""
    rows = []
    if os.path.isdir(observed_dir):
        for f in os.listdir(observed_dir):
            if f.lower().endswith(FLOOD_EXTS):
                full = os.path.join(observed_dir, f)
                rows.append((map_date(full), full))
    rows.sort(key=lambda r: (r[0], os.path.getmtime(r[1])), reverse=True)
    return rows


def _mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0


# ------------------------------------------------------------------
# Reading a flood map onto the risk-raster grid
# ------------------------------------------------------------------
def _flood_to_grid(path, dst_crs, dst_transform, shape, coding):
    """Return (flood_bool, covered_bool) on the risk grid."""
    H, W = shape
    ext = os.path.splitext(path)[1].lower()

    if ext in (".tif", ".tiff"):
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError("The flood GeoTIFF has no coordinate system (CRS).")
            arr = src.read(1)
            nodata = src.nodata
            src_crs, src_tf = src.crs, src.transform
        valid = np.ones(arr.shape, dtype=bool)
        if arr.dtype.kind == "f":
            valid &= np.isfinite(arr)
        if nodata is not None:
            valid &= arr != nodata
        valid &= arr != 255
        flood = (arr == 1) if coding == "eq1" else (arr > 0)
        flood = flood & valid
        fl = np.zeros((H, W), dtype="float32")
        cv = np.zeros((H, W), dtype="float32")
        for src_arr, dst_arr in ((flood, fl), (valid, cv)):
            reproject(
                source=src_arr.astype("float32"), destination=dst_arr,
                src_transform=src_tf, src_crs=src_crs,
                dst_transform=dst_transform, dst_crs=dst_crs,
                resampling=Resampling.average,
            )
        return fl >= 0.5, cv >= 0.5

    geoms = []
    if ext == ".geojson":
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        kind = gj.get("type")
        if kind == "FeatureCollection":
            feats = gj.get("features", [])
        elif kind == "Feature":
            feats = [gj]
        else:
            feats = [{"geometry": gj}]
        for ft in feats:
            g = ft.get("geometry")
            if g and g.get("type") in ("Polygon", "MultiPolygon"):
                geoms.append(transform_geom("EPSG:4326", dst_crs, g))
    elif ext in (".shp", ".gpkg"):
        try:
            import geopandas as gpd
        except ImportError:
            raise ValueError("Reading .shp/.gpkg needs geopandas "
                             "(pip install geopandas), or convert the file to GeoJSON.")
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            raise ValueError("The flood vector file has no coordinate system (CRS).")
        gdf = gdf.to_crs(dst_crs.to_wkt())
        geoms = [g.__geo_interface__ for g in gdf.geometry
                 if g is not None and g.geom_type in ("Polygon", "MultiPolygon")]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not geoms:
        return np.zeros((H, W), bool), np.ones((H, W), bool)
    flood = rasterize(((g, 1) for g in geoms), out_shape=(H, W), transform=dst_transform,
                      fill=0, all_touched=True, dtype="uint8").astype(bool)
    # Vector products carry no footprint, so full coverage is assumed.
    return flood, np.ones((H, W), bool)


@st.cache_resource(show_spinner="Reading satellite flood map...")
def analyse_flood_map(flood_path, flood_mtime, risk_path, risk_mtime, coding):
    """Cached. Arrays are shared between callers: treat them as read-only."""
    try:
        with rasterio.open(risk_path) as src:
            shape = (src.height, src.width)
            crs, tf = src.crs, src.transform
        flood, covered = _flood_to_grid(flood_path, crs, tf, shape, coding)
        return {"flood": flood, "covered": covered,
                "transform": tuple(tf)[:6], "crs_wkt": crs.to_wkt()}
    except Exception as exc:
        return {"error": str(exc)}


# ------------------------------------------------------------------
# Statistics and model comparison
# ------------------------------------------------------------------
def summarise(an, risk, px_km2, min_pred_class, names):
    valid = np.isfinite(risk) & (risk >= 1) & (risk <= 4)
    cov = valid & an["covered"]
    F = an["flood"] & cov
    n_valid, n_cov, n_flood = int(valid.sum()), int(cov.sum()), int(F.sum())
    flood_pct = 100.0 * n_flood / n_cov if n_cov else 0.0

    rows = []
    for i, name in enumerate(names, start=1):
        in_c = cov & (risk == i)
        n_c = int(in_c.sum())
        n_f = int((F & in_c).sum())
        share = 100.0 * n_f / n_c if n_c else 0.0
        rows.append({
            "Risk class": name,
            "Covered area (km²)": round(n_c * px_km2, 2),
            "Flooded (km²)": round(n_f * px_km2, 2),
            "Share of class flooded (%)": round(share, 2),
            "Share of all flooded area (%)": round(100.0 * n_f / n_flood, 2) if n_flood else 0.0,
            "Lift vs average": round(share / flood_pct, 2) if flood_pct > 0 else 0.0,
        })

    pred = cov & (risk >= min_pred_class)
    tp = int((pred & F).sum())
    fp = int((pred & ~F).sum())
    fn = int((~pred & F).sum())
    tn = int((cov & ~pred & ~F).sum())
    nan = float("nan")
    pred_frac = pred.sum() / n_cov if n_cov else nan
    pod = tp / (tp + fn) if (tp + fn) else nan
    metrics = {
        "POD": pod,                                              # share of flood the model flagged
        "Precision": tp / (tp + fp) if (tp + fp) else nan,       # share of flagged area that flooded
        "CSI": tp / (tp + fp + fn) if (tp + fp + fn) else nan,
        "F1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else nan,
        "Area flagged by model": pred_frac,
        "Enrichment": pod / pred_frac if (pred_frac and pred_frac > 0 and pod == pod) else nan,
    }
    return {
        "valid": valid, "cov": cov, "F": F, "pred": pred,
        "flood_km2": n_flood * px_km2, "covered_km2": n_cov * px_km2,
        "study_km2": n_valid * px_km2,
        "coverage_pct": 100.0 * n_cov / n_valid if n_valid else 0.0,
        "flood_pct": flood_pct, "by_class": pd.DataFrame(rows),
        "cm": {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
               "TP_km2": tp * px_km2, "FP_km2": fp * px_km2, "FN_km2": fn * px_km2},
        "metrics": metrics,
    }


def compute_event(map_path, risk_path, risk, px_km2, names, coding="eq1", min_pred_class=3):
    an = analyse_flood_map(map_path, _mtime(map_path), risk_path, _mtime(risk_path), coding)
    if "error" in an:
        return an
    return {"analysis": an, "summary": summarise(an, risk, px_km2, min_pred_class, names)}


# ------------------------------------------------------------------
# Assets hit by the real flood
# ------------------------------------------------------------------
def screen_assets(df, summary, an, risk, names, px_m, buffer_m, min_pred_class):
    """Classify every asset against the observed flood and the risk model."""
    crs = CRS.from_wkt(an["crs_wkt"])
    tf = Affine(*an["transform"])
    xs = df["lon"].to_numpy(dtype=float)
    ys = df["lat"].to_numpy(dtype=float)
    if crs.to_epsg() != 4326:
        xs, ys = warp_transform("EPSG:4326", crs, xs.tolist(), ys.tolist())
        xs, ys = np.asarray(xs), np.asarray(ys)
    rr, cc = rowcol(tf, xs, ys)
    rr = np.floor(np.asarray(rr, dtype=float)).astype(int)
    cc = np.floor(np.asarray(cc, dtype=float)).astype(int)

    F, cov = summary["F"], summary["cov"]
    H, W = F.shape
    n = int(np.ceil(buffer_m / px_m)) if buffer_m > 0 else 0

    status, cls, check = [], [], []
    for r, c in zip(rr, cc):
        if not (0 <= r < H and 0 <= c < W):
            status.append(S_OUT); cls.append("-"); check.append("-"); continue
        rv = risk[r, c]
        if not np.isfinite(rv) or rv < 1 or rv > 4:
            status.append(S_OUT); cls.append("-"); check.append("-"); continue
        cls.append(names[int(rv) - 1])
        if not cov[r, c]:
            status.append(S_NODATA); check.append("-"); continue
        if F[r, c]:
            st_ = S_FLOODED
        elif n > 0 and F[max(0, r - n):r + n + 1, max(0, c - n):c + n + 1].any():
            st_ = S_NEAR
        else:
            st_ = S_DRY
        status.append(st_)
        if st_ in (S_FLOODED, S_NEAR):
            check.append("Confirmed by model" if rv >= min_pred_class else "Missed by model")
        else:
            check.append("High risk, dry in this event" if rv >= min_pred_class
                         else "Consistent (lower risk, dry)")

    out = df.copy().reset_index(drop=True)
    out["Model risk class"] = cls
    out["Observed status"] = status
    out["Model check"] = check
    out["_r"], out["_c"] = rr, cc
    return out


def _find_infra(ctx):
    finder = ctx.get("find_infra")
    return finder() if finder else []


def _load_infra(ctx, path):
    loader = ctx.get("load_infra")
    if not loader:
        return None, "No infrastructure loader supplied."
    return loader(path, os.path.getmtime(path))


# ------------------------------------------------------------------
# Maps
# ------------------------------------------------------------------
def _display_max(arr, max_dim=1600):
    """Block-max reduction so small flooded patches stay visible."""
    step = max(1, int(np.ceil(max(arr.shape) / max_dim)))
    if step == 1:
        return arr, 1
    h, w = arr.shape
    hh, ww = -(-h // step), -(-w // step)
    pad = np.zeros((hh * step, ww * step), dtype=arr.dtype)
    pad[:h, :w] = arr
    return pad.reshape(hh, step, ww, step).max(axis=(1, 3)), step


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def _draw(cat, colors, labels, title, px_km2, decorate_map, assets=None):
    show, step = _display_max(cat)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(show, cmap=ListedColormap(colors), vmin=-0.5, vmax=len(colors) - 0.5,
              interpolation="nearest")
    decorate_map(ax, show.shape, step, px_km2, title)
    handles = [Patch(facecolor=c, edgecolor="#9FB3C8", label=l)
               for c, l, in zip(colors, labels) if l]
    if assets is not None and len(assets):
        styles = [(S_FLOODED, "#D64545", "^", 70, "Asset flooded"),
                  (S_NEAR, "#F2994A", "^", 55, "Asset near flood edge"),
                  (S_DRY, "#0B2545", "o", 10, "Asset not flooded")]
        for status, color, marker, size, label in styles:
            sub = assets[assets["Observed status"] == status]
            if sub.empty:
                continue
            ax.scatter((sub["_c"] + 0.5) / step - 0.5, (sub["_r"] + 0.5) / step - 0.5,
                       s=size, c=color, marker=marker, edgecolors="white", linewidths=0.7,
                       zorder=5)
            handles.append(Line2D([], [], marker=marker, linestyle="", markerfacecolor=color,
                                  markeredgecolor="white", markersize=8 if size > 20 else 5,
                                  label=label))
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.94,
              fontsize=8, title_fontsize=9)
    return fig


def flood_map_fig(summary, px_km2, decorate_map, title, assets=None):
    cat = np.zeros(summary["valid"].shape, dtype="uint8")
    cat[summary["valid"]] = 2
    cat[summary["valid"] & ~summary["cov"]] = 1
    cat[summary["F"]] = 3
    colors = ["#FFFFFF", "#CBD5E0", "#E8EEF4", "#1578C8"]
    labels = ["", "No satellite data", "Study area (dry)", "Observed flood"]
    return _draw(cat, colors, labels, title, px_km2, decorate_map, assets)


def agreement_map_fig(summary, px_km2, decorate_map, title):
    cat = np.zeros(summary["valid"].shape, dtype="uint8")
    cat[summary["valid"]] = 2
    cat[summary["valid"] & ~summary["cov"]] = 1
    cat[summary["pred"] & ~summary["F"]] = 3
    cat[summary["F"] & ~summary["pred"]] = 4
    cat[summary["F"] & summary["pred"]] = 5
    colors = ["#FFFFFF", "#CBD5E0", "#E8EEF4", "#F6D58E", "#C2185B", "#1578C8"]
    labels = ["", "No satellite data", "Neither", "Model only (not flooded)",
              "Flood missed by model", "Flood and model agree"]
    return _draw(cat, colors, labels, title, px_km2, decorate_map)


# ------------------------------------------------------------------
# Small UI helpers
# ------------------------------------------------------------------
def _fmt_pct(x):
    return "n/a" if x != x else f"{100 * x:.1f}%"


def _kpi(ctx, top, value, sub=""):
    if ctx.get("kpi"):
        st.markdown(ctx["kpi"](top, value, sub), unsafe_allow_html=True)
    else:
        st.metric(top, value, help=sub)


def _compact(s):
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


def _age_days(d):
    return max(0, (date.today() - d).days)


# ------------------------------------------------------------------
# Alert banner
# ------------------------------------------------------------------
def _alert_level(flood_km2, flood_pct, assets_hit):
    if flood_km2 < ALERT_MIN_KM2:
        return "clear"
    if flood_pct >= SEVERE_PCT or assets_hit >= SEVERE_ASSETS:
        return "severe"
    if flood_pct >= WARNING_PCT or assets_hit >= 1:
        return "warning"
    return "watch"


def assess_latest(ctx):
    """Assess the newest satellite flood map with the user's current settings."""
    maps = list_flood_maps(_observed_dir(ctx["base_dir"]))
    if not maps:
        return None
    d, path = maps[0]
    coding = CODING.get(st.session_state.get("fd_coding"), "eq1")
    thr = MODEL_THRESHOLDS.get(st.session_state.get("fd_model"), 3)
    buffer_m = st.session_state.get("fd_buffer", 100)
    names = ctx["risk_names"]
    ev = compute_event(path, ctx["risk_path"], ctx["risk_array"], ctx["pixel_km2"],
                       names, coding, thr)
    if "error" in ev:
        return {"error": ev["error"], "date": d, "path": path}
    s = ev["summary"]
    res = {"date": d, "path": path, "summary": s, "thr": thr, "buffer_m": buffer_m,
           "assets_flooded": 0, "assets_near": 0, "assets_total": 0, "assets_missed": 0}
    files = _find_infra(ctx)
    if files and s["flood_km2"] >= ALERT_MIN_KM2:
        df, err = _load_infra(ctx, files[0])
        if err is None and df is not None:
            px_m = float(np.sqrt(ctx["pixel_km2"]) * 1000)
            a = screen_assets(df, s, ev["analysis"], ctx["risk_array"], names,
                              px_m, buffer_m, thr)
            res["assets_total"] = len(a)
            res["assets_flooded"] = int((a["Observed status"] == S_FLOODED).sum())
            res["assets_near"] = int((a["Observed status"] == S_NEAR).sum())
            res["assets_missed"] = int(((a["Observed status"].isin([S_FLOODED, S_NEAR])) &
                                        (a["Model check"] == "Missed by model")).sum())
    return res


def render_flood_alert_banner(ctx):
    r = assess_latest(ctx)
    if r is None:
        level, text = "none", ("Add a satellite flood map to 01_data\\flood_observed, or run the "
                               "Sentinel-1 detection from the Real flood detection tab.")
    elif "error" in r:
        level, text = "error", r["error"]
    else:
        s, d = r["summary"], r["date"]
        age = _age_days(d)
        when = f"{d.isoformat()} ({age} day{'s' if age != 1 else ''} ago)"
        if age > STALE_DAYS:
            level = "stale"
            text = (f"The newest map is from {when} and showed {s['flood_km2']:,.1f} km² "
                    f"flooded. Run a new detection for a current picture.")
        else:
            level = _alert_level(s["flood_km2"], s["flood_pct"], r["assets_flooded"])
            if level == "clear":
                text = (f"Satellite map of {when} shows no significant flooding "
                        f"({s['flood_km2']:,.1f} km²). Satellite coverage of the study area: "
                        f"{s['coverage_pct']:.0f}%.")
            else:
                parts = [f"Satellite map of {when}: {s['flood_km2']:,.1f} km² inundated "
                         f"({s['flood_pct']:.1f}% of the covered study area)."]
                if r["assets_total"]:
                    parts.append(f"{r['assets_flooded']} of {r['assets_total']} assets flooded, "
                                 f"{r['assets_near']} more within {r['buffer_m']} m of the flood edge.")
                pod = s["metrics"]["POD"]
                if pod == pod:
                    parts.append(f"The risk model flagged {100 * pod:.0f}% of the flooded area.")
                if s["coverage_pct"] < 90:
                    parts.append(f"Satellite coverage is {s['coverage_pct']:.0f}%, so the true "
                                 f"extent may be larger.")
                text = " ".join(parts)
    c1, c2, fg, title = LEVELS[level]
    pulse = "fd-pulse" if level in ("severe", "warning", "watch") else ""
    html = f"""
    <style>
    @keyframes fdPulse {{ 0% {{box-shadow:0 0 0 0 rgba(255,255,255,.75)}}
        100% {{box-shadow:0 0 0 14px rgba(255,255,255,0)}} }}
    .fd-banner {{ display:flex; align-items:center; gap:16px; padding:16px 20px;
        border-radius:14px; margin:0 0 22px; box-shadow:0 6px 18px rgba(11,37,69,.18);
        background:linear-gradient(100deg,{c1},{c2}); color:{fg}; }}
    .fd-dot {{ flex:none; width:14px; height:14px; border-radius:50%; background:{fg}; }}
    .fd-pulse {{ animation:fdPulse 1.6s ease-out infinite; }}
    .fd-title {{ font-weight:850; letter-spacing:1.2px; font-size:13px; }}
    .fd-text {{ font-size:13px; line-height:1.55; margin-top:4px; opacity:.96; }}
    .fd-note {{ font-size:11px; opacity:.8; margin-top:6px; }}
    </style>
    <div class="fd-banner">
      <span class="fd-dot {pulse}"></span>
      <div>
        <div class="fd-title">{title}</div>
        <div class="fd-text">{text}</div>
        <div class="fd-note">Satellite-derived decision support. Not an official warning:
        follow guidance from the Department of Disaster Management Affairs.</div>
      </div>
    </div>
    """
    st.markdown(_compact(html), unsafe_allow_html=True)


# ------------------------------------------------------------------
# Getting data: upload or run the Sentinel-1 script
# ------------------------------------------------------------------
def _acquisition_panel(ctx, observed_dir, expanded):
    with st.expander("Get or add a satellite flood map", expanded=expanded):
        script = os.path.join(ctx["base_dir"], "02_scripts", "flood_detection",
                              "detect_sar_flood.py")
        st.markdown("**Option 1: detect automatically with Sentinel-1 radar** "
                    "(needs a one-time Google Earth Engine login, see the script header).")
        c1, c2 = st.columns([1, 2])
        with c1:
            days = st.number_input("Look back (days)", 6, 60, 12, step=1, key="fd_days")
        with c2:
            st.write("")
            st.write("")
            if st.button("Run Sentinel-1 detection now", disabled=not os.path.exists(script)):
                with st.spinner("Querying Sentinel-1 and downloading the flood map..."):
                    try:
                        res = subprocess.run([sys.executable, script, "--days", str(int(days))],
                                             capture_output=True, text=True, timeout=1500)
                        log = (res.stdout or "") + (res.stderr or "")
                    except subprocess.TimeoutExpired:
                        log = "The detection timed out after 25 minutes."
                st.session_state["fd_run_log"] = log[-3500:]
                st.rerun()
        if not os.path.exists(script):
            st.caption(f"Script not found: {script}")
        if st.session_state.get("fd_run_log"):
            st.code(st.session_state["fd_run_log"])

        st.markdown("**Option 2: add a flood map you already have** "
                    "(Copernicus GFM, UNOSAT / Copernicus EMS, NASA Disasters, or your own export).")
        up = st.file_uploader("Flood map (GeoTIFF or GeoJSON)",
                              type=["tif", "tiff", "geojson"], key="fd_upload")
        acq = st.date_input("Satellite acquisition date", value=date.today(), key="fd_acq")
        if up is not None and st.button("Save to flood_observed folder"):
            os.makedirs(observed_dir, exist_ok=True)
            ext = os.path.splitext(up.name)[1].lower()
            target = os.path.join(observed_dir, f"flood_extent_{acq.isoformat()}{ext}")
            with open(target, "wb") as f:
                f.write(up.getbuffer())
            st.success(f"Saved {os.path.basename(target)}")
            st.rerun()
        st.caption(f"Folder: {observed_dir}")


# ------------------------------------------------------------------
# The main tab
# ------------------------------------------------------------------
def render_flood_detection(ctx):
    sec = ctx.get("section") or (lambda t, s="": (st.subheader(t), st.caption(s)))
    cap = ctx.get("caption") or (lambda label, text: st.caption(f"**{label}.** {text}"))
    base_dir = ctx["base_dir"]
    observed_dir = _observed_dir(base_dir)
    risk, px_km2 = ctx["risk_array"], ctx["pixel_km2"]
    px_m = float(np.sqrt(px_km2) * 1000)
    names, deco = ctx["risk_names"], ctx["decorate_map"]

    sec("Real flood detection",
        "Satellite-observed flooding compared with the susceptibility model.")

    maps = list_flood_maps(observed_dir)
    _acquisition_panel(ctx, observed_dir, expanded=not maps)
    if not maps:
        st.info("No satellite flood map found yet. Use the panel above to run the Sentinel-1 "
                f"detection or add a map.\n\n`{observed_dir}`")
        return

    labels = [f"{d.isoformat()}  |  {os.path.basename(p)}" for d, p in maps]
    idx = st.selectbox("Satellite flood map", range(len(maps)),
                       format_func=lambda i: labels[i], key="fd_map")
    map_d, map_path = maps[idx]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        coding_label = st.radio("Flood pixel coding (GeoTIFF)", list(CODING), key="fd_coding")
    with k2:
        model_label = st.selectbox("Model counts as 'flood-prone' if class is",
                                   list(MODEL_THRESHOLDS), key="fd_model")
    with k3:
        buffer_m = st.select_slider("Asset search distance (m)", BUFFERS_M, value=100,
                                    key="fd_buffer")
    infra_files = _find_infra(ctx)
    with k4:
        chosen = (st.selectbox("Infrastructure dataset", infra_files,
                               format_func=os.path.basename, key="fd_infra")
                  if infra_files else None)
        if not infra_files:
            st.caption("No infrastructure dataset found.")

    coding, thr = CODING[coding_label], MODEL_THRESHOLDS[model_label]
    ev = compute_event(map_path, ctx["risk_path"], risk, px_km2, names, coding, thr)
    if "error" in ev:
        st.error(f"Could not read this flood map: {ev['error']}")
        return
    an, s = ev["analysis"], ev["summary"]
    meta = _sidecar(map_path)

    assets, infra_err = None, None
    if chosen:
        infra_df, infra_err = _load_infra(ctx, chosen)
        if infra_err is None and infra_df is not None:
            assets = screen_assets(infra_df, s, an, risk, names, px_m, buffer_m, thr)

    age = _age_days(map_d)
    src_txt = meta.get("source", "user-supplied flood map")
    win = (f" | Window {meta['start_date']} to {meta['end_date']}"
           if "start_date" in meta and "end_date" in meta else "")
    st.caption(f"Map date {map_d.isoformat()} ({age} d ago) | {src_txt}{win}")
    if age > STALE_DAYS:
        st.warning(f"This map is {age} days old. It shows a past event, not current conditions.")
    if s["coverage_pct"] < 90:
        st.warning(f"The satellite covered {s['coverage_pct']:.0f}% of the study area. Flooding "
                   "outside the covered part is not captured, so totals are minimums.")

    n_fl = int((assets["Observed status"] == S_FLOODED).sum()) if assets is not None else None
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi(ctx, "Observed flooded area", f"{s['flood_km2']:,.1f} km²",
             "Open water detected by satellite.")
    with c2:
        _kpi(ctx, "Share of covered area", f"{s['flood_pct']:.2f}%",
             f"Of {s['covered_km2']:,.0f} km² covered.")
    with c3:
        _kpi(ctx, "Satellite coverage", f"{s['coverage_pct']:.0f}%",
             "Share of the study area imaged.")
    with c4:
        _kpi(ctx, "Assets flooded", "N/A" if n_fl is None else f"{n_fl:,}",
             "Add an infrastructure dataset." if n_fl is None
             else f"Plus {int((assets['Observed status'] == S_NEAR).sum())} near the flood edge.")

    t_area, t_assets, t_model = st.tabs(["Flooded area", "Assets hit", "Model vs observed"])

    # ---- Flooded area -------------------------------------------------
    with t_area:
        m_col, s_col = st.columns([3, 2])
        with m_col:
            fig = flood_map_fig(s, px_km2, deco, f"Observed flood extent, {map_d.isoformat()}")
            st.pyplot(fig)
            png = _png(fig)
            plt.close(fig)
            st.download_button("Download map (PNG)", png,
                               file_name=f"duwe_observed_flood_{map_d.isoformat()}.png",
                               mime="image/png")
        with s_col:
            st.markdown("**Flooding by model risk class**")
            st.bar_chart(s["by_class"].set_index("Risk class")[["Flooded (km²)"]],
                         color="#1578C8")
            st.dataframe(s["by_class"], hide_index=True, width="stretch")
            st.download_button("Download table (CSV)",
                               s["by_class"].to_csv(index=False).encode("utf-8"),
                               file_name=f"duwe_flood_by_class_{map_d.isoformat()}.csv",
                               mime="text/csv")
        cap("Figure", "Blue cells are flooded according to the satellite map, resampled to the "
            "risk-raster grid. Grey cells were not imaged. Radar sees open water well but can miss "
            "flooding under dense vegetation or among buildings, so the extent is a minimum.")

    # ---- Assets hit ---------------------------------------------------
    with t_assets:
        if not infra_files:
            st.info("No infrastructure dataset found. Add a CSV or GeoJSON of point locations to "
                    "01_data\\infrastructure (longitude/latitude columns, plus name and type).")
        elif infra_err:
            st.error(infra_err)
        else:
            hit = assets[assets["Observed status"].isin([S_FLOODED, S_NEAR])]
            missed = int((hit["Model check"] == "Missed by model").sum())
            a1, a2, a3 = st.columns(3)
            with a1:
                _kpi(ctx, "Assets screened", f"{len(assets):,}",
                     f"{int((assets['Observed status'] == S_OUT).sum())} outside the study area.")
            with a2:
                _kpi(ctx, "Flooded or at the edge", f"{len(hit):,}",
                     f"Within {buffer_m} m of observed flood water.")
            with a3:
                _kpi(ctx, "Hit but missed by model", f"{missed:,}",
                     f"Below the '{model_label}' level in the risk model.")
            am, at = st.columns([3, 2])
            with am:
                fig = flood_map_fig(s, px_km2, deco,
                                    f"Assets and observed flood, {map_d.isoformat()}", assets)
                st.pyplot(fig)
                plt.close(fig)
            with at:
                summary = (pd.crosstab(assets["Type"], assets["Observed status"])
                           .reindex(columns=[S_FLOODED, S_NEAR, S_DRY, S_NODATA, S_OUT],
                                    fill_value=0))
                summary["Total"] = summary.sum(axis=1)
                st.markdown("**Assets by type and observed status**")
                st.dataframe(summary, width="stretch")
            if hit.empty:
                st.success("No assets in this dataset were flooded or at the flood edge.")
            else:
                order = {S_FLOODED: 0, S_NEAR: 1}
                table = (hit[["Name", "Type", "Observed status", "Model risk class",
                              "Model check", "lat", "lon"]]
                         .sort_values(by="Observed status", key=lambda c: c.map(order)))
                st.markdown("**Assets hit by the flood**")
                st.dataframe(table, hide_index=True, width="stretch")
                pts = hit[["lat", "lon"]].copy()
                pts["color"] = np.where(hit["Observed status"] == S_FLOODED, "#D64545", "#F2994A")
                st.map(pts, latitude="lat", longitude="lon", color="color", size=250)
                st.download_button("Download assets hit (CSV)",
                                   table.to_csv(index=False).encode("utf-8"),
                                   file_name=f"duwe_assets_hit_{map_d.isoformat()}.csv",
                                   mime="text/csv")
            cap("Table", f"An asset is 'Flooded' if its cell is flooded and 'Near flood edge' if "
                f"flood water lies within {buffer_m} m. The distance allows for the 30 m grid and "
                "for location error in the point data.")

    # ---- Model vs observed -------------------------------------------
    with t_model:
        m, cm = s["metrics"], s["cm"]
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            _kpi(ctx, "Flood captured by model", _fmt_pct(m["POD"]),
                 "POD: observed flood inside flood-prone zones.")
        with g2:
            _kpi(ctx, "Flagged area that flooded", _fmt_pct(m["Precision"]),
                 "Precision of the flood-prone zones.")
        with g3:
            _kpi(ctx, "Critical success index", "n/a" if m["CSI"] != m["CSI"]
                 else f"{m['CSI']:.2f}", "Hits / (hits + misses + false alarms).")
        with g4:
            _kpi(ctx, "Enrichment over random", "n/a" if m["Enrichment"] != m["Enrichment"]
                 else f"{m['Enrichment']:.1f}x",
                 f"Model flags {_fmt_pct(m['Area flagged by model'])} of the area.")
        mm, mt = st.columns([3, 2])
        with mm:
            fig = agreement_map_fig(s, px_km2, deco,
                                    f"Model vs observed flood ({model_label.lower()})")
            st.pyplot(fig)
            png = _png(fig)
            plt.close(fig)
            st.download_button("Download comparison map (PNG)", png,
                               file_name=f"duwe_model_vs_observed_{map_d.isoformat()}.png",
                               mime="image/png")
        with mt:
            st.markdown("**Flooding by risk class**")
            st.dataframe(s["by_class"], hide_index=True, width="stretch")
            cmx = pd.DataFrame({
                "Outcome": ["Flood and model agree", "Model only (not flooded)",
                            "Flood missed by model"],
                "Area (km²)": [round(cm["TP_km2"], 2), round(cm["FP_km2"], 2),
                               round(cm["FN_km2"], 2)]})
            st.dataframe(cmx, hide_index=True, width="stretch")
        if s["flood_km2"] < ALERT_MIN_KM2:
            st.info("There is too little observed flooding in this map for a meaningful comparison.")
        else:
            low = s["by_class"].set_index("Risk class")
            verdict = (f"The model's {model_label.lower()} zones cover "
                       f"{_fmt_pct(m['Area flagged by model'])} of the imaged area and contain "
                       f"{_fmt_pct(m['POD'])} of the observed flooding"
                       + (f", {m['Enrichment']:.1f} times more than a random map of the same size"
                          if m["Enrichment"] == m["Enrichment"] else "")
                       + f". {cm['FN_km2']:,.1f} km² flooded outside those zones.")
            st.markdown(f'<div class="note">{verdict}</div>', unsafe_allow_html=True)
        cap("Reading the comparison", "A susceptibility model shows where flooding is possible, "
            "while one satellite image shows one event. 'Model only' areas are expected: no single "
            "flood fills every susceptible cell. Pay most attention to 'Flood missed by model' and "
            "to whether the share flooded rises from Low to Very High. Compare several events "
            "before changing the model.")
