"""
DUWE FLOOD INTELLIGENCE ENGINE
Dashboard panel: Flood Forecast (screening)

Save this file in the same folder as your dashboard and live_rainfall.py.
It only READS the JSON files written by the forecasting engine. Run the
engine separately with run_all.py.

This panel uses the CSS classes already defined in the main dashboard
(.kpi, .cls, .note, .sec-title), so it matches the existing design.
"""

import os
import json
from datetime import datetime, timezone

import streamlit as st

BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"

FORECAST_DIR = os.path.join(
    BASE_DIR, r"01_data\processed\forecasting"
)

REFRESH_SECONDS = 60
MAX_DATA_AGE_MINUTES = 30

FILES = {
    "monitoring": "monitoring_latest.json",
    "rainfall": "rainfall_analysis_latest.json",
    "runoff": "runoff_latest.json",
    "discharge": "discharge_indicator_latest.json",
    "timing": "flood_timing_latest.json",
    "assets": "infrastructure_screening_summary.json",
    "warning": "early_warning_latest.json",
}

ASSET_CSV = "infrastructure_screening_latest.csv"

LEVEL_COLORS = {
    "INFORMATION": "#2E9E6B",
    "MONITOR": "#F2C94C",
    "WATCH_SCREENING": "#F2994A",
    "ELEVATED_SCREENING": "#D64545",
    "URGENT_REVIEW": "#9B1C1C",
}


def _html(s):
    return "".join(line.strip() for line in s.splitlines())


def _path(name):
    return os.path.join(FORECAST_DIR, FILES[name])


def _load(name):
    try:
        with open(_path(name), "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError):
        return {}


def _num(value, decimals=1, unit=""):
    if value is None or isinstance(value, str):
        return "n/a"
    return f"{value:,.{decimals}f}{unit}"


def _kpi(label, value, sub="", color="#1596A6"):
    return _html(
        f'<div class="kpi" style="border-top:4px solid {color}">'
        f'<div class="kpi-top">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>'
    )


def _section(title, subtitle=""):
    st.markdown(
        _html(
            f'<div class="sec-title">{title}</div>'
            f'<div class="sec-sub">{subtitle}</div>'
        ),
        unsafe_allow_html=True,
    )


def _age_minutes(iso_text):
    try:
        stamp = datetime.fromisoformat(iso_text)
        if stamp.tzinfo is None:
            stamp = stamp.astimezone()
        return (datetime.now(timezone.utc) - stamp).total_seconds() / 60
    except (TypeError, ValueError):
        return None


def _render():

    warning = _load("warning")
    monitoring = _load("monitoring")

    if not warning or not monitoring:
        st.info(
            "No forecasting results yet. Start the engine by running "
            "`python run_all.py` in 02_scripts\\forecasting_engine, then "
            "wait for the first cycle to finish."
        )
        return

    rainfall = _load("rainfall")
    runoff = _load("runoff")
    discharge = _load("discharge")
    timing = _load("timing")
    assets = _load("assets")

    level = warning.get("screening_level", "UNKNOWN")
    color = LEVEL_COLORS.get(level, "#64748B")

    age = _age_minutes(monitoring.get("monitoring_timestamp"))
    stale = age is None or age > MAX_DATA_AGE_MINUTES

    # ---------- Screening level ----------
    reasons = warning.get("reasons") or ["No screening triggers."]
    reasons_html = "".join(f"<li>{r}</li>" for r in reasons)

    st.markdown(
        _html(
            f'<div class="cls" style="border-left:8px solid {color}">'
            f'<div class="cls-name">Current screening level</div>'
            f'<div class="cls-val" style="color:{color}">'
            f'{level.replace("_", " ")}</div>'
            f'<ul style="margin:10px 0 0 18px;color:#334155;font-size:13px">'
            f'{reasons_html}</ul></div>'
        ),
        unsafe_allow_html=True,
    )

    if stale:
        shown = "unknown" if age is None else f"{age:,.0f} minutes"
        st.warning(
            f"Forecast data may be out of date (age: {shown}). Check that "
            "run_all.py is running and that the rainfall download is working."
        )

    st.markdown(
        _html(
            '<div class="note">'
            + warning.get("message", "")
            + " Public warning authorised: "
            + str(warning.get("public_warning_authorised", False))
            + ".</div>"
        ),
        unsafe_allow_html=True,
    )

    # ---------- Rainfall ----------
    rain = monitoring.get("rainfall", {})

    _section(
        "Rainfall (Open-Meteo)",
        "Model-based hourly rainfall at the catchment centre. "
        "Not a rain-gauge measurement.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        _kpi("Now", _num(rain.get("current_rainfall_mm"), 1, " mm"), "", "#1596A6"),
        unsafe_allow_html=True,
    )
    c2.markdown(
        _kpi("Past 24 h", _num(rain.get("past_24h_mm"), 1, " mm"),
             f"72 h: {_num(rain.get('past_72h_mm'), 1, ' mm')}", "#134074"),
        unsafe_allow_html=True,
    )
    c3.markdown(
        _kpi("Next 24 h", _num(rain.get("forecast_next_24h_mm"), 1, " mm"),
             "Forecast", "#2E86C1"),
        unsafe_allow_html=True,
    )
    c4.markdown(
        _kpi("Next 48 h", _num(rain.get("forecast_next_48h_mm"), 1, " mm"),
             "Forecast", "#2E86C1"),
        unsafe_allow_html=True,
    )

    st.caption(
        "Rainfall status: "
        + str(rainfall.get("rainfall_status", "n/a"))
        + ". Provisional screening thresholds "
        + "(high 50 mm, very high 100 mm in 24 h); not official."
    )

    # ---------- Runoff ----------
    _section(
        "Runoff and flow indicators",
        "Preliminary and uncalibrated. Uses a placeholder Curve Number.",
    )

    r1, r2, r3 = st.columns(3)
    r1.markdown(
        _kpi("Forecast runoff", _num(runoff.get("forecast_runoff_mm"), 2, " mm"),
             f"Curve Number {runoff.get('curve_number', 'n/a')}", "#117C8B"),
        unsafe_allow_html=True,
    )
    r2.markdown(
        _kpi("Forecast runoff volume",
             _num(runoff.get("forecast_runoff_volume_m3"), 0, " m³"),
             f"Catchment {_num(runoff.get('catchment_area_km2'), 1, ' km²')}",
             "#117C8B"),
        unsafe_allow_html=True,
    )
    r3.markdown(
        _kpi("Flow indicator",
             _num(discharge.get("forecast_flow_indicator_m3s"), 2, " m³/s"),
             "Catchment average, not Shire River discharge", "#117C8B"),
        unsafe_allow_html=True,
    )

    if discharge.get("warning"):
        st.caption(discharge["warning"])

    # ---------- Gauge and timing ----------
    _section(
        "River gauge and flood timing",
        "Needs verified gauge observations and approved thresholds.",
    )

    gauge = monitoring.get("river_gauge", {})

    g1, g2 = st.columns(2)

    with g1:
        if gauge.get("available"):
            st.metric("Station", gauge.get("station", "n/a"))
            st.metric("Water level", _num(gauge.get("water_level_m"), 2, " m"))
            st.caption(f"Observed: {gauge.get('timestamp', 'n/a')}")
        else:
            st.info(gauge.get("message", "River gauge data not configured."))

    with g2:
        if timing.get("status") == "OBSERVATION_BASED_ESTIMATE":
            st.metric(
                "Time to next threshold",
                _num(timing.get("estimated_time_to_threshold_hours"), 1, " h"),
                timing.get("next_threshold", ""),
            )
            st.caption(timing.get("message", ""))
        else:
            st.info(timing.get("message", "Flood timing unavailable."))

    # ---------- Infrastructure ----------
    _section(
        "Infrastructure exposure",
        "Existing exposure classification, not a prediction of flooding.",
    )

    a1, a2 = st.columns([1, 2])

    a1.markdown(
        _kpi("Assets classed high / very high risk",
             f"{assets.get('high_or_very_high_assets', 'n/a')}",
             "Already classified in the risk map. Does not mean they "
             "will flood.", "#D64545"),
        unsafe_allow_html=True,
    )

    counts = assets.get("category_counts") or {}

    with a2:
        if counts:
            st.bar_chart(counts)
        else:
            st.caption("No category breakdown available.")

    csv_path = os.path.join(FORECAST_DIR, ASSET_CSV)

    if os.path.exists(csv_path):
        try:
            import pandas as pd

            table = pd.read_csv(csv_path)

            with st.expander(f"View high-risk asset list ({len(table)})"):
                st.dataframe(table.head(500), use_container_width=True)
                st.download_button(
                    "Download asset list (CSV)",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name="duwe_high_risk_assets.csv",
                    mime="text/csv",
                    key="download_high_risk_assets",
                )
        except Exception as ex:
            st.caption(f"Asset list could not be loaded: {ex}")

    # ---------- Engine status ----------
    with st.expander("Engine status (file update times)"):
        rows = []
        for name, filename in FILES.items():
            path = _path(name)
            if os.path.exists(path):
                modified = datetime.fromtimestamp(os.path.getmtime(path))
                minutes = (datetime.now() - modified).total_seconds() / 60
                rows.append({
                    "File": filename,
                    "Updated": modified.strftime("%d %b %Y %H:%M:%S"),
                    "Age (min)": round(minutes, 1),
                })
            else:
                rows.append({
                    "File": filename,
                    "Updated": "missing",
                    "Age (min)": None,
                })
        st.table(rows)


def render_forecast_panel():

    """Draw the panel and refresh it automatically every minute."""

    fragment = getattr(st, "fragment", None)

    if fragment is not None:
        fragment(run_every=REFRESH_SECONDS)(_render)()
    else:
        _render()
