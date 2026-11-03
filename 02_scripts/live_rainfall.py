"""
DUWE AI FLOOD INTELLIGENCE ENGINE
Live rainfall detection for the dashboard.

Asks the free Open-Meteo weather service (no account or key needed) for rainfall at a grid of
points spread over the catchment, and shows:
  - whether it is raining now, and how much
  - rain in the last 24 / 72 hours
  - forecast rain for the next 24 / 48 hours

Save in 02_scripts, next to duwe_flood_dashboard.py. Needs internet; if the connection is down,
the last saved reading is shown with its time.
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform
import streamlit as st

BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
CATCHMENT_PATH = os.path.join(BASE_DIR, r"01_data\processed\catchment_duwe.tif")
LIVE_CACHE = os.path.join(BASE_DIR, r"01_data\processed\live_rainfall_latest.json")

API_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Africa/Blantyre"
USER_AGENT = "DUWE-Flood-Intelligence/1.0 (Chikwawa, Malawi)"

SPACING_KM = 12          # distance between sample points
MAX_POINTS = 40          # spacing is widened automatically if this would be exceeded
PAST_DAYS = 3
FORECAST_DAYS = 3
REFRESH_SECONDS = 600    # re-check every 10 minutes

# Adjustable thresholds. These are common general-purpose values, NOT official warning
# levels: replace them with the thresholds used by Malawi's DCCMS if you have them.
RAINING_MM_H = 0.1       # at or above this = raining
LIGHT_MAX_MM_H = 2.5     # light below this
MODERATE_MAX_MM_H = 7.5  # moderate below this, heavy above
HEAVY_24H_MM = 50
EXTREME_24H_MM = 100

STATE_COLORS = {"Dry": "#95a5a6", "Light": "#5dade2", "Moderate": "#2e86c1", "Heavy": "#8e44ad"}


# -------------------------------------------------
# SAMPLE POINTS (a grid of points that lie inside the catchment)
# -------------------------------------------------
@st.cache_data
def sample_points(spacing_km=SPACING_KM, max_points=MAX_POINTS, catchment_path=CATCHMENT_PATH):
    with rasterio.open(catchment_path) as src:
        cat = src.read(1)
        tr, crs = src.transform, src.crs
        left, bottom, right, top = src.bounds

    spacing = spacing_km * 1000.0
    while True:
        xs = np.arange(left + spacing / 2, right, spacing)
        ys = np.arange(bottom + spacing / 2, top, spacing)
        gx, gy = np.meshgrid(xs, ys)
        gx, gy = gx.ravel(), gy.ravel()
        rows, cols = rowcol(tr, list(gx), list(gy))
        rows, cols = np.asarray(rows), np.asarray(cols)
        ok = (rows >= 0) & (rows < cat.shape[0]) & (cols >= 0) & (cols < cat.shape[1])
        keep = np.zeros(len(gx), dtype=bool)
        keep[ok] = cat[rows[ok], cols[ok]] == 1
        if keep.sum() <= max_points or spacing > 60000:
            break
        spacing *= 1.25

    if keep.sum() == 0:                               # tiny or odd catchment: use the centre
        gx, gy = np.array([(left + right) / 2]), np.array([(bottom + top) / 2])
        keep = np.array([True])

    lons, lats = transform(crs, "EPSG:4326", list(gx[keep]), list(gy[keep]))
    return [round(float(v), 4) for v in lats], [round(float(v), 4) for v in lons]


# -------------------------------------------------
# FETCH (with a saved copy for when the internet is down)
# -------------------------------------------------
def _request(lats, lons):
    params = {
        "latitude": ",".join(str(v) for v in lats),
        "longitude": ",".join(str(v) for v in lons),
        "current": "precipitation",
        "hourly": "precipitation",
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
        "timezone": TIMEZONE,
    }
    r = requests.get(API_URL, params=params, timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        if data.get("error"):
            raise RuntimeError(data.get("reason", "weather service returned an error"))
        data = [data]
    return data


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def fetch_live(lats, lons):
    """Download fresh data. Raises on failure, so failures are never cached."""
    payload = {"fetched_at": datetime.now().isoformat(timespec="seconds"),
               "locations": _request(lats, lons)}
    try:
        os.makedirs(os.path.dirname(LIVE_CACHE), exist_ok=True)
        with open(LIVE_CACHE, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError:
        pass
    return payload


def get_data():
    """Return (payload, error_message). payload is None only if there is nothing at all."""
    try:
        lats, lons = sample_points()
        return fetch_live(tuple(lats), tuple(lons)), None
    except Exception as ex:
        try:
            with open(LIVE_CACHE, "r", encoding="utf-8") as f:
                return json.load(f), f"{type(ex).__name__}: {ex}"
        except Exception:
            return None, f"{type(ex).__name__}: {ex}"


# -------------------------------------------------
# ANALYSIS
# -------------------------------------------------
def rain_state(rate_mm_h):
    if rate_mm_h < RAINING_MM_H:
        return "Dry"
    if rate_mm_h < LIGHT_MAX_MM_H:
        return "Light"
    if rate_mm_h < MODERATE_MAX_MM_H:
        return "Moderate"
    return "Heavy"


def summarise(payload):
    """Return (per-point table, catchment-mean hourly series, 'now' timestamp)."""
    rows, series, now = [], [], None
    for i, loc in enumerate(payload.get("locations", [])):
        h = loc.get("hourly") or {}
        if not h.get("time"):
            continue
        cur = loc.get("current") or {}
        idx = pd.to_datetime(h["time"])
        p = pd.Series(pd.to_numeric(pd.Series(h["precipitation"]), errors="coerce").to_numpy(),
                      index=idx).fillna(0.0)
        t_now = pd.to_datetime(cur["time"]) if cur.get("time") else pd.Timestamp.now()
        if now is None:
            now = t_now

        interval = cur.get("interval") or 3600
        amount = float(cur.get("precipitation") or 0.0)
        rate = amount * 3600.0 / interval            # convert to mm per hour

        def past(hours):
            return float(p[(p.index > t_now - pd.Timedelta(hours=hours)) & (p.index <= t_now)].sum())

        def ahead(hours):
            return float(p[(p.index > t_now) & (p.index <= t_now + pd.Timedelta(hours=hours))].sum())

        rows.append({
            "point": i + 1,
            "lat": float(loc.get("latitude", np.nan)),
            "lon": float(loc.get("longitude", np.nan)),
            "now_mm_h": round(rate, 2),
            "state": rain_state(rate),
            "past_24h_mm": round(past(24), 1),
            "past_72h_mm": round(past(72), 1),
            "next_24h_mm": round(ahead(24), 1),
            "next_48h_mm": round(ahead(48), 1),
        })
        series.append(p)

    if not rows:
        raise ValueError("No usable rainfall data in the response")
    return pd.DataFrame(rows), pd.concat(series, axis=1).mean(axis=1), now


def catchment_status(df):
    n = len(df)
    raining = int((df["now_mm_h"] >= RAINING_MM_H).sum())
    return {
        "n": n,
        "raining": raining,
        "share": 100.0 * raining / n,
        "mean_rate": float(df["now_mm_h"].mean()),
        "max_rate": float(df["now_mm_h"].max()),
        "mean_past24": float(df["past_24h_mm"].mean()),
        "max_past24": float(df["past_24h_mm"].max()),
        "mean_next24": float(df["next_24h_mm"].mean()),
        "max_next24": float(df["next_24h_mm"].max()),
    }


def headline(s):
    if s["raining"] == 0:
        return "☀️ No rain detected right now", "info"
    if s["share"] >= 50:
        return f"🌧️ Raining across much of the catchment ({s['raining']} of {s['n']} points)", "warning"
    return f"🌦️ Rain at {s['raining']} of {s['n']} points in the catchment", "info"


def advisories(s):
    notes = []
    if s["max_past24"] >= EXTREME_24H_MM:
        notes.append(f"Very heavy rain in the last 24 h at some points (up to {s['max_past24']:.0f} mm).")
    elif s["max_past24"] >= HEAVY_24H_MM:
        notes.append(f"Heavy rain in the last 24 h at some points (up to {s['max_past24']:.0f} mm).")
    if s["max_next24"] >= EXTREME_24H_MM:
        notes.append(f"Very heavy rain is forecast in the next 24 h at some points (up to {s['max_next24']:.0f} mm).")
    elif s["max_next24"] >= HEAVY_24H_MM:
        notes.append(f"Heavy rain is forecast in the next 24 h at some points (up to {s['max_next24']:.0f} mm).")
    return notes


def asset_link_message(s):
    """Message linking rain (and river discharge, if available) to the high-risk assets, or None."""
    try:
        from rain_risk_link import get_high_risk_summary, build_rain_risk_message
    except Exception:
        return None
    try:
        from live_discharge import worst_status
        level = worst_status()
    except Exception:
        level = None
    try:
        return build_rain_risk_message(s, get_high_risk_summary(), level)
    except Exception:
        return None


# -------------------------------------------------
# STREAMLIT DISPLAY
# -------------------------------------------------
def _auto_refresh(fn):
    """Re-run this block every REFRESH_SECONDS if the installed Streamlit supports it."""
    frag = getattr(st, "fragment", None)
    if frag is None:
        return fn
    try:
        return frag(run_every=REFRESH_SECONDS)(fn)
    except TypeError:
        return fn


def _banner():
    payload, error = get_data()
    if payload is None:
        st.caption("Live rainfall is not available (no internet and no saved reading).")
        return
    try:
        df, _, _ = summarise(payload)
    except Exception:
        st.caption("Live rainfall data could not be read.")
        return
    s = catchment_status(df)
    text, level = headline(s)
    if error:
        text += f" (saved reading from {payload.get('fetched_at', '?')} - no internet)"
    (st.warning if level == "warning" else st.info)(text)
    for note in advisories(s):
        st.warning(note)

    # Link with high-risk infrastructure (shown here so it appears on every tab)
    msg = asset_link_message(s)
    if msg:
        st.error(msg)
        st.caption("See the **Infrastructure at risk** tab for the full list of these assets.")


def _panel(high_risk_km2=None):
    st.header("Live rainfall")
    st.caption(
        "Rainfall estimates from the Open-Meteo weather service (a weather-model and satellite "
        "blend, not rain-gauge readings), sampled at points spread across the catchment. "
        "Showers in Malawi are patchy, so a point can read dry while rain falls nearby. Use this "
        "as a guide and follow official warnings from Malawi's Department of Climate Change and "
        "Meteorological Services (DCCMS)."
    )

    if st.button("Refresh now"):
        fetch_live.clear()

    payload, error = get_data()
    if payload is None:
        st.error(f"Could not get rainfall data and no earlier reading is saved. ({error})")
        return
    if error:
        st.warning(f"No internet connection - showing the last saved reading from "
                   f"{payload.get('fetched_at', '?')}. ({error})")

    try:
        df, hourly, now = summarise(payload)
    except Exception as ex:
        st.error(f"The weather data could not be read: {ex}")
        return

    s = catchment_status(df)
    text, level = headline(s)
    (st.warning if level == "warning" else st.info)(text)
    st.write(f"Local time (Malawi): **{now.strftime('%Y-%m-%d %H:%M')}** | "
             f"data fetched: {payload.get('fetched_at', '?')} | {s['n']} sample points")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raining now (share of points)", f"{s['share']:.0f}%")
    c2.metric("Rain rate now (mean / max)", f"{s['mean_rate']:.1f} / {s['max_rate']:.1f} mm/h")
    c3.metric("Last 24 h (mean / max)", f"{s['mean_past24']:.0f} / {s['max_past24']:.0f} mm")
    c4.metric("Next 24 h forecast (mean / max)", f"{s['mean_next24']:.0f} / {s['max_next24']:.0f} mm")

    notes = advisories(s)
    for note in notes:
        st.warning(note)
    if notes and high_risk_km2:
        st.write(f"About **{high_risk_km2:,.0f} km²** of the catchment is in High or Very High "
                 "risk zones. See the Infrastructure tab for the schools, clinics and villages in those zones.")

    # ---- hourly chart: past vs forecast, catchment average ----
    chart = pd.DataFrame({
        "Past (model estimate)": hourly.where(hourly.index <= now),
        "Forecast": hourly.where(hourly.index > now),
    }).fillna(0.0)
    st.subheader("Hourly rainfall, catchment average (mm per hour)")
    st.bar_chart(chart)

    # ---- map ----
    st.subheader("Where it is raining")
    m = df[["lat", "lon", "state", "past_24h_mm"]].copy()
    m["size"] = 1500 + 300 * m["past_24h_mm"].clip(upper=100)
    m["color"] = m["state"].map(STATE_COLORS)
    try:
        st.map(m, latitude="lat", longitude="lon", size="size", color="color")
    except TypeError:
        st.map(m[["lat", "lon"]])
    st.caption("Colour = rain state now (grey dry, light blue light, blue moderate, purple heavy). "
               "Size = rain in the last 24 hours.")

    # ---- table ----
    st.subheader("Readings by point")
    show = df.rename(columns={
        "now_mm_h": "Now (mm/h)", "state": "State", "past_24h_mm": "Last 24h (mm)",
        "past_72h_mm": "Last 72h (mm)", "next_24h_mm": "Next 24h (mm)", "next_48h_mm": "Next 48h (mm)"})
    st.dataframe(show, hide_index=True)
    st.download_button("Download readings (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="live_rainfall_readings.csv", mime="text/csv")

    st.caption("Weather data by Open-Meteo.com (CC BY 4.0). Thresholds: raining from "
               f"{RAINING_MM_H} mm/h; light below {LIGHT_MAX_MM_H}, moderate below {MODERATE_MAX_MM_H}, "
               f"heavy above; 24 h alerts at {HEAVY_24H_MM} and {EXTREME_24H_MM} mm. These are general "
               "values, not official warning levels.")


render_live_banner = _auto_refresh(_banner)
render_live_rainfall = _auto_refresh(_panel)
