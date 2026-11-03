"""
DUWE AI FLOOD INTELLIGENCE ENGINE
Dashboard panel: infrastructure and vulnerable assets at risk.

Save in 02_scripts, next to duwe_flood_dashboard.py. The dashboard already calls
render_infrastructure_panel() inside its "Infrastructure at risk" tab, so do NOT also add
the import at the end of the dashboard (the panel would show twice).
"""

import os
import sys
import json
import subprocess

import pandas as pd
import streamlit as st

BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
AUTOMATION_DIR = os.path.join(BASE_DIR, r"02_scripts\automation")
INFRA_DIR = os.path.join(BASE_DIR, r"01_data\raw\infrastructure")
SUMMARY_CSV = os.path.join(BASE_DIR, r"01_data\processed\asset_exposure_summary.csv")
HIGH_RISK_CSV = os.path.join(BASE_DIR, r"01_data\processed\assets_high_risk.csv")
STATUS_JSON = os.path.join(INFRA_DIR, "infrastructure_status.json")

LABELS = {
    "schools": "Schools", "health": "Health facilities", "villages": "Villages / settlements",
    "water_points": "Water points", "markets": "Markets", "worship": "Places of worship",
    "power": "Power sites", "buildings": "Buildings", "roads": "Roads (km)", "bridges": "Bridges (km)",
}
MAP_POINT_LIMIT = 20000
CLASSES = ["Low", "Medium", "High", "Very High"]
SUMMARY_COLUMNS = ["category", "unit", "total_in_area", "pct_high_or_very_high"] + CLASSES
HIGH_RISK_COLUMNS = ["category", "name", "type", "risk_class", "lat", "lon"]


@st.cache_data(ttl=300)
def _read_csv(path, mtime):
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(ttl=300)
def _read_points(path, mtime):
    with open(path, "r", encoding="utf-8") as f:
        feats = json.load(f).get("features", [])
    rows = [{"lon": x["geometry"]["coordinates"][0], "lat": x["geometry"]["coordinates"][1]}
            for x in feats if x["geometry"]["type"] == "Point"]
    return pd.DataFrame(rows, columns=["lon", "lat"])


def _csv(path):
    return _read_csv(path, os.path.getmtime(path)) if os.path.exists(path) else None


def _missing(df, needed):
    return [c for c in needed if c not in df.columns]


def _refresh_data():
    """Run the downloader, then the exposure analysis."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    messages = []
    for script in ("03_download_infrastructure.py", "04_asset_exposure.py"):
        result = subprocess.run(
            [sys.executable, os.path.join(AUTOMATION_DIR, script)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env, cwd=AUTOMATION_DIR, timeout=5400)
        messages.append((script, result.returncode, (result.stdout or "")[-1500:] + (result.stderr or "")[-1500:]))
    st.cache_data.clear()
    return messages


def render_infrastructure_panel():
    st.header("Infrastructure and vulnerable assets at risk")
    st.caption("Source: OpenStreetMap contributors. Coverage depends on how well the area is mapped, "
               "so counts are minimums, not a full census.")

    # ---- refresh button ------------------------------------------------
    if st.button("Refresh infrastructure data now"):
        with st.spinner("Downloading from OpenStreetMap and re-checking exposure. This can take several minutes..."):
            try:
                messages = _refresh_data()
            except Exception as ex:
                messages = [("refresh", 1, str(ex))]
        for script, code, tail in messages:
            (st.success if code == 0 else st.error)(f"{script}: {'OK' if code == 0 else 'failed (see below)'}")
            if code != 0:
                st.code(tail)

    # ---- download status ----------------------------------------------
    if os.path.exists(STATUS_JSON):
        with open(STATUS_JSON, "r", encoding="utf-8") as f:
            status = json.load(f)
        meta = status.get("_meta", {})
        st.write(f"Last checked: **{meta.get('checked', 'unknown')}**")
        failed = [k for k, v in status.items() if k != "_meta" and str(v.get("status", "")).startswith("FAILED")]
        if failed:
            st.warning("Some categories could not be refreshed (older data or none): " + ", ".join(failed))
    else:
        st.info("No infrastructure data yet. Click the button above, or run "
                "03_download_infrastructure.py once (internet needed).")
        return

    summary = _csv(SUMMARY_CSV)
    if summary is None or summary.empty:
        st.info("Infrastructure is downloaded, but the exposure analysis has not run yet "
                "(it needs the risk map). Run 04_asset_exposure.py or the daily pipeline.")
        return

    gone = _missing(summary, SUMMARY_COLUMNS)
    if gone:
        st.error(f"asset_exposure_summary.csv is missing the columns {gone}. "
                 f"Columns found: {list(summary.columns)}. Rename them in this file or in 04_asset_exposure.py.")
        return

    # ---- headline numbers ----------------------------------------------
    st.subheader("In High or Very High risk zones")
    key = [c for c in ("schools", "health", "villages", "roads") if c in set(summary["category"])]
    cols = st.columns(len(key)) if key else []
    for col, cat in zip(cols, key):
        row = summary[summary["category"] == cat].iloc[0]
        at_risk = row["High"] + row["Very High"]
        unit = " km" if row["unit"] == "km" else ""
        col.metric(LABELS.get(cat, cat), f"{at_risk:,.0f}{unit}",
                   f"{row['pct_high_or_very_high']:.1f}% of {row['total_in_area']:,.0f}", delta_color="off")

    # ---- table and chart -------------------------------------------------
    table = summary.copy()
    table["category"] = table["category"].map(lambda c: LABELS.get(c, c))
    st.dataframe(table, hide_index=True)

    # Share of each risk class per category, so counts and kilometres can sit on one chart.
    chart = summary.copy()
    totals = chart[CLASSES].sum(axis=1).replace(0, float("nan"))
    shares = chart[CLASSES].div(totals, axis=0).mul(100).fillna(0.0)
    shares.index = chart["category"].map(lambda c: LABELS.get(c, c))
    st.markdown("**Share of each category by risk class (%)**")
    st.bar_chart(shares)
    st.caption("Each bar adds up to 100% of that category's assets inside the flood-risk area, so counts "
               "and kilometres can be compared fairly. Use the table above for the actual numbers.")

    # ---- map and list of high-risk assets -----------------------------------
    high = _csv(HIGH_RISK_CSV)
    if high is not None and not high.empty:
        gone = _missing(high, HIGH_RISK_COLUMNS)
        if gone:
            st.error(f"assets_high_risk.csv is missing the columns {gone}. Columns found: {list(high.columns)}.")
        else:
            st.subheader("Named assets in High / Very High risk zones")
            cats = sorted(high["category"].dropna().unique())
            choice = st.selectbox("Category", cats, format_func=lambda c: LABELS.get(c, c))
            sub = high[high["category"] == choice]
            st.write(f"{len(sub):,} assets")
            st.map(sub[["lat", "lon"]].dropna())
            st.dataframe(sub[["name", "type", "risk_class", "lat", "lon"]], hide_index=True)
            st.download_button("Download this list (CSV)",
                               sub.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"high_risk_{choice}.csv", mime="text/csv")

    # ---- optional: every point of a category -----------------------------------
    with st.expander("Show every mapped point of a category"):
        point_cats = [c for c in LABELS if c not in ("roads", "bridges")
                      and os.path.exists(os.path.join(INFRA_DIR, f"{c}.geojson"))]
        if point_cats:
            pick = st.selectbox("Category to map", point_cats, format_func=lambda c: LABELS[c], key="all_points")
            path = os.path.join(INFRA_DIR, f"{pick}.geojson")
            pts = _read_points(path, os.path.getmtime(path))
            if len(pts) > MAP_POINT_LIMIT:
                st.caption(f"Showing a random {MAP_POINT_LIMIT:,} of {len(pts):,} points.")
                pts = pts.sample(MAP_POINT_LIMIT, random_state=1)
            if not pts.empty:
                st.map(pts)
