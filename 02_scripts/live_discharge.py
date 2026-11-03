"""
DUWE AI FLOOD INTELLIGENCE ENGINE
River discharge detection for the dashboard.

Asks the free Open-Meteo Flood API (GloFAS river model, no account or key needed) for
daily river discharge at the Shire River near Chikwawa and shows:
  - discharge now, the trend, and the peak expected in the next 7 days
  - a status (Normal / Elevated / Watch / Alert) judged against this river's own history
  - the last 30 days and the 7-day forecast as a chart

If you have real gauge readings (from DWR or your own sensor), save them as
01_data\\raw\\gauge_readings.csv with columns: datetime, level_m, discharge_m3s
(either value column can be left out) and they are shown next to the model.

Save in 02_scripts, next to duwe_flood_dashboard.py. Needs internet; if the connection is
down, the last saved reading is shown with its time.

IMPORTANT: this is MODELLED discharge (about 5 km grid), not a gauge measurement, and it
gives discharge only, not water level. Treat it as a guide, not an official warning.
"""

import os
import json
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
PROCESSED_DIR = os.path.join(BASE_DIR, r"01_data\processed")
CACHE_FILE = os.path.join(PROCESSED_DIR, "live_discharge_latest.json")
SITES_FILE = os.path.join(PROCESSED_DIR, "discharge_sites.json")
GAUGE_CSV = os.path.join(BASE_DIR, r"01_data\raw\gauge_readings.csv")
HIGH_RISK_CSV = os.path.join(PROCESSED_DIR, "assets_high_risk.csv")

API_URL = "https://flood-api.open-meteo.com/v1/flood"
USER_AGENT = "DUWE-Flood-Intelligence/1.0 (Chikwawa, Malawi)"

# Approximate search centre. The model grid is ~5 km, so the script looks at the cells around
# this point and picks the one carrying the most water (the river). The result is saved in
# discharge_sites.json: open it and edit lat/lon if the chosen cell is not on the Shire.
SITES = {"Shire River near Chikwawa": (-16.03, 34.85)}
SEARCH_STEP = 0.05       # degrees, about the model grid size
SEARCH_RADIUS = 2        # 2 means a 5 x 5 block of cells around the centre

PAST_DAYS = 30
FORECAST_DAYS = 7
CLIMATE_START_YEAR = 1991
REFRESH_SECONDS = 3600   # the model updates daily, so hourly is plenty

STATUS_COLORS = {"Normal": "#2E9E6B", "Elevated": "#F2C94C", "Watch": "#F2994A", "Alert": "#D64545"}


# -------------------------------------------------
# API HELPERS
# -------------------------------------------------
def _get(params):
    r = requests.get(API_URL, params=params, timeout=40, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(data.get("reason", "flood service returned an error"))
    return data


def _as_list(data):
    return data if isinstance(data, list) else [data]


def _clean(values):
    return [np.nan if v is None else float(v) for v in values]


def find_river_cell(lat, lon):
    """Pick the model cell with the largest recent flow near (lat, lon): that is the river."""
    offs = np.arange(-SEARCH_RADIUS, SEARCH_RADIUS + 1) * SEARCH_STEP
    cands = [(round(lat + dy, 4), round(lon + dx, 4)) for dy in offs for dx in offs]
    data = _as_list(_get({
        "latitude": ",".join(str(c[0]) for c in cands),
        "longitude": ",".join(str(c[1]) for c in cands),
        "daily": "river_discharge", "past_days": 60, "forecast_days": 1,
    }))
    best, best_mean = None, -1.0
    for cand, loc in zip(cands, data):
        vals = np.array(_clean((loc.get("daily") or {}).get("river_discharge") or []))
        if vals.size == 0 or np.all(np.isnan(vals)):
            continue
        m = float(np.nanmean(vals))
        if m > best_mean:
            best_mean = m
            best = (round(float(loc.get("latitude", cand[0])), 4),
                    round(float(loc.get("longitude", cand[1])), 4))
    if best is None:
        raise RuntimeError("No river discharge found near the search point")
    return best


def resolve_sites():
    """Return [(name, lat, lon)], finding the river cell once and remembering it."""
    if os.path.exists(SITES_FILE):
        try:
            with open(SITES_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            return [(s["name"], float(s["lat"]), float(s["lon"])) for s in saved]
        except Exception:
            pass
    sites = []
    for name, (lat, lon) in SITES.items():
        rlat, rlon = find_river_cell(lat, lon)
        sites.append((name, rlat, rlon))
    try:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        with open(SITES_FILE, "w", encoding="utf-8") as f:
            json.dump([{"name": n, "lat": a, "lon": b} for n, a, b in sites], f, indent=2)
    except OSError:
        pass
    return sites


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def get_thresholds(lat, lon):
    """Flood levels from this cell's own history (complete years only)."""
    end_year = date.today().year - 1
    data = _get({
        "latitude": lat, "longitude": lon, "daily": "river_discharge",
        "start_date": f"{CLIMATE_START_YEAR}-01-01", "end_date": f"{end_year}-12-31",
    })
    d = _as_list(data)[0].get("daily") or {}
    s = pd.Series(_clean(d.get("river_discharge") or []), index=pd.to_datetime(d.get("time") or [])).dropna()
    if len(s) < 365 * 5:
        raise ValueError("Not enough history to set flood levels")
    annual_max = s.groupby(s.index.year).max()
    return {
        "median": float(s.median()),
        "elevated": float(s.quantile(0.90)),            # a high-flow day
        "watch": float(annual_max.median()),            # roughly a 2-year flood
        "alert": float(annual_max.quantile(0.80)),      # roughly a 5-year flood
        "years": int(len(annual_max)),
    }


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def fetch_live(sites):
    """Download fresh data. Raises on failure, so failures are never cached."""
    out = []
    for name, lat, lon in sites:
        d = _as_list(_get({
            "latitude": lat, "longitude": lon, "daily": "river_discharge",
            "past_days": PAST_DAYS, "forecast_days": FORECAST_DAYS,
        }))[0].get("daily") or {}
        out.append({"name": name, "lat": lat, "lon": lon,
                    "time": d.get("time") or [], "discharge": d.get("river_discharge") or [],
                    "thresholds": get_thresholds(lat, lon)})
    payload = {"fetched_at": datetime.now().isoformat(timespec="seconds"), "sites": out}
    try:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError:
        pass
    return payload


def get_data():
    """Return (payload, error_message). payload is None only if there is nothing at all."""
    try:
        return fetch_live(tuple(resolve_sites())), None
    except Exception as ex:
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f), f"{type(ex).__name__}: {ex}"
        except Exception:
            return None, f"{type(ex).__name__}: {ex}"


# -------------------------------------------------
# ANALYSIS
# -------------------------------------------------
def status_for(q, th):
    if q >= th["alert"]:
        return "Alert"
    if q >= th["watch"]:
        return "Watch"
    if q >= th["elevated"]:
        return "Elevated"
    return "Normal"


def summarise(site, today=None):
    """Turn one site's raw lists into the numbers the panel shows."""
    today = pd.Timestamp(today or date.today()).normalize()
    s = pd.Series(_clean(site["discharge"]), index=pd.to_datetime(site["time"])).dropna()
    if s.empty:
        raise ValueError("No discharge values in the response")
    past, fut = s[s.index <= today], s[s.index > today]
    if past.empty:
        past = s.iloc[:1]
    now_q = float(past.iloc[-1])
    prev_q = float(past.iloc[-2]) if len(past) > 1 else now_q
    th = site["thresholds"]
    peak_q = float(fut.max()) if not fut.empty else now_q
    peak_day = fut.idxmax() if not fut.empty else past.index[-1]
    change = 100.0 * (now_q - prev_q) / prev_q if prev_q > 0 else 0.0
    return {
        "series": s, "past": past, "future": fut, "th": th,
        "now": now_q, "now_date": past.index[-1], "change_pct": change,
        "trend": "Rising" if change > 5 else ("Falling" if change < -5 else "Steady"),
        "status_now": status_for(now_q, th),
        "peak": peak_q, "peak_day": peak_day, "status_peak": status_for(peak_q, th),
        "vs_median": now_q / th["median"] if th["median"] > 0 else np.nan,
    }


def load_gauge():
    """Optional real readings. Returns a DataFrame or None."""
    try:
        if not os.path.exists(GAUGE_CSV):
            return None
        g = pd.read_csv(GAUGE_CSV)
        g.columns = [c.strip().lower() for c in g.columns]
        g["datetime"] = pd.to_datetime(g["datetime"], errors="coerce")
        return g.dropna(subset=["datetime"]).sort_values("datetime")
    except Exception:
        return None


def high_risk_count():
    try:
        return int(len(pd.read_csv(HIGH_RISK_CSV)))
    except Exception:
        return None


# -------------------------------------------------
# STREAMLIT DISPLAY
# -------------------------------------------------
def _auto_refresh(fn):
    frag = getattr(st, "fragment", None)
    if frag is None:
        return fn
    try:
        return frag(run_every=REFRESH_SECONDS)(fn)
    except TypeError:
        return fn


def _worst(info):
    order = ["Normal", "Elevated", "Watch", "Alert"]
    return max((info["status_now"], info["status_peak"]), key=order.index)


def worst_status():
    """Worst status (now or at the 7-day peak) across all sites, or None if there is no data.
    Used by rain_risk_link.py to add river discharge to the high-risk asset warning."""
    payload, _ = get_data()
    if payload is None:
        return None
    order = ["Normal", "Elevated", "Watch", "Alert"]
    worst = None
    for site in payload.get("sites", []):
        try:
            w = _worst(summarise(site))
        except Exception:
            continue
        if worst is None or order.index(w) > order.index(worst):
            worst = w
    return worst


def _banner():
    payload, error = get_data()
    if payload is None:
        return
    try:
        infos = [summarise(s) for s in payload["sites"]]
    except Exception:
        return
    for site, info in zip(payload["sites"], infos):
        worst = _worst(info)
        if worst in ("Watch", "Alert"):
            when = "now" if info["status_now"] == worst else f"around {info['peak_day']:%d %b}"
            msg = (f"River discharge {worst.upper()}: {site['name']} is modelled at "
                   f"{info['peak']:,.0f} m³/s {when}.")
            (st.error if worst == "Alert" else st.warning)(msg)


def _panel():
    st.header("River discharge")
    st.caption(
        "Modelled river discharge from the Open-Meteo Flood API (GloFAS), about 5 km resolution, "
        "daily values. This is a model estimate, not a gauge measurement, and it gives discharge "
        "only (not water level). Levels are judged against this river's own history, so they are "
        "relative guides, not official flood stages. Follow warnings from the Department of Water "
        "Resources (DWR), DCCMS and DoDMA."
    )

    if st.button("Refresh discharge now"):
        fetch_live.clear()

    payload, error = get_data()
    if payload is None:
        st.error(f"Could not get discharge data and no earlier reading is saved. ({error})")
        return
    if error:
        st.warning(f"No internet connection - showing the last saved reading from "
                   f"{payload.get('fetched_at', '?')}. ({error})")

    for site in payload["sites"]:
        try:
            info = summarise(site)
        except Exception as ex:
            st.error(f"{site.get('name', 'Site')}: the data could not be read ({ex})")
            continue

        st.subheader(site["name"])
        st.write(f"Model cell: {site['lat']:.3f}, {site['lon']:.3f} | "
                 f"latest value: {info['now_date']:%d %b %Y} | data fetched: {payload.get('fetched_at', '?')}")

        color = STATUS_COLORS[info["status_now"]]
        st.markdown(f'<div style="border-left:6px solid {color};padding:6px 12px;background:#FFFFFF;'
                    f'border-radius:8px;font-weight:700">Status now: {info["status_now"]} '
                    f'| Trend: {info["trend"]}</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Discharge now", f"{info['now']:,.0f} m³/s", f"{info['change_pct']:+.0f}% vs previous day")
        c2.metric("Typical (median)", f"{info['th']['median']:,.0f} m³/s", f"now is {info['vs_median']:.1f}x typical")
        c3.metric(f"Peak, next {FORECAST_DAYS} days", f"{info['peak']:,.0f} m³/s", f"{info['peak_day']:%d %b}")
        c4.metric("Status at peak", info["status_peak"])

        if info["status_peak"] in ("Watch", "Alert"):
            n = high_risk_count()
            extra = f" There are {n} high-risk assets in your exposure results." if n else ""
            st.error(f"Modelled discharge is expected to reach the {info['status_peak']} level "
                     f"around {info['peak_day']:%d %b}.{extra} See the **Infrastructure at risk** tab.")

        chart = pd.DataFrame({
            "Past (model)": info["past"],
            "Forecast": pd.concat([info["past"].iloc[-1:], info["future"]]),
        })
        chart["Watch level"] = info["th"]["watch"]
        chart["Alert level"] = info["th"]["alert"]
        st.markdown("**Daily discharge (m³/s): last 30 days and 7-day forecast**")
        st.line_chart(chart)

        th = info["th"]
        st.caption(f"Levels from {th['years']} years of model history ({CLIMATE_START_YEAR} onward): "
                   f"Elevated {th['elevated']:,.0f} m³/s (a high-flow day), Watch {th['watch']:,.0f} "
                   f"(about a 2-year flood), Alert {th['alert']:,.0f} (about a 5-year flood). "
                   "These are empirical estimates from the model, not official levels.")

    g = load_gauge()
    st.subheader("Gauge readings")
    if g is None or g.empty:
        st.info("No gauge file found. To compare the model with real readings, save a CSV at "
                "01_data\\raw\\gauge_readings.csv with columns: datetime, level_m, discharge_m3s.")
    else:
        latest = g.iloc[-1]
        cols = st.columns(2)
        if "level_m" in g.columns and pd.notna(latest.get("level_m")):
            cols[0].metric("Latest water level", f"{latest['level_m']:.2f} m", f"{latest['datetime']:%d %b %H:%M}")
        if "discharge_m3s" in g.columns and pd.notna(latest.get("discharge_m3s")):
            cols[1].metric("Latest gauge discharge", f"{latest['discharge_m3s']:,.0f} m³/s")
        plot_cols = [c for c in ("level_m", "discharge_m3s") if c in g.columns]
        if plot_cols:
            st.line_chart(g.set_index("datetime")[plot_cols])

    st.caption("Flood data by Open-Meteo.com (CC BY 4.0), based on the GloFAS river model.")


render_discharge_banner = _auto_refresh(_banner)
render_live_discharge = _auto_refresh(_panel)
