"""
DUWE DISASTER INTELLIGENCE
DUWE FLOOD INTELLIGENCE ENGINE
Professional geospatial flood-susceptibility dashboard.

Expected logo location:
    04_assets/logo.png
Also accepts .jpg, .jpeg, .webp and .svg.

Note:
This dashboard presents geospatial decision-support information. It is not,
by itself, an operational flood forecast or an official warning system.
"""

import io
import os
import re
import base64
import heapq
import json
import warnings
import calendar
from datetime import datetime, date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import rasterio
import streamlit as st

# Clean, publication-style defaults for every matplotlib figure.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#9FB3C8",
    "axes.labelcolor": "#344B64",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# =================================================
# 1. PROJECT PATHS AND SETTINGS
# =================================================

BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
RAW_DIR = os.path.join(BASE_DIR, r"01_data\raw")
ASSETS_DIR = os.path.join(BASE_DIR, "04_assets")
CHIRPS_STATUS_FILE = os.path.join(
    BASE_DIR, r"02_scripts\automation\logs\chirps_status.txt"
)

PATHS = {
    "dem": os.path.join(
        BASE_DIR, r"01_data\processed\dem_conditioned_duwe.tif"
    ),
    "catchment": os.path.join(
        BASE_DIR, r"01_data\processed\catchment_duwe.tif"
    ),
    "accumulation": os.path.join(
        BASE_DIR, r"01_data\processed\flow_accumulation_duwe.tif"
    ),
    "rainfall": os.path.join(
        BASE_DIR, r"01_data\processed\chirps_rainfall_aligned_duwe.tif"
    ),
    "rain_hazard": os.path.join(
        BASE_DIR, r"01_data\processed\rainfall_enhanced_hazard_duwe.tif"
    ),
    "hazard_class": os.path.join(
        BASE_DIR, r"01_data\processed\rainfall_hazard_classified_duwe.tif"
    ),
    "risk": os.path.join(
        BASE_DIR, r"01_data\processed\risk_updated_duwe.tif"
    ),
    "risk_class": os.path.join(
        BASE_DIR, r"01_data\processed\risk_updated_classified_duwe.tif"
    ),
    "exposure": os.path.join(
        BASE_DIR, r"01_data\processed\exposure_proxy_duwe.tif"
    ),
}

PIXEL_AREA_KM2 = 0.0009
PIXEL_SIZE_M = 30
DEVELOPER = "Ashley Duwe"

RISK_NAMES = ["Low", "Medium", "High", "Very High"]
RISK_COLORS = ["#2E9E6B", "#F2C94C", "#F2994A", "#D64545"]
LOGO_FILES = ["logo.png", "logo.jpg", "logo.jpeg", "logo.webp", "logo.svg"]
MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}

EXPLORER_LAYERS = {
    "Flood risk class": ("risk_class", None, True, "Risk class"),
    "Rainfall hazard class": ("hazard_class", None, True, "Hazard class"),
    "Flood risk score": ("risk", "YlOrRd", False, "Risk score"),
    "Rainfall-enhanced hazard": ("rain_hazard", "YlGnBu", False, "Hazard score"),
    "Exposure proxy": ("exposure", "Purples", False, "Exposure index"),
    "CHIRPS rainfall": ("rainfall", "Blues", False, "Rainfall (mm)"),
    "Flow accumulation": ("accumulation", "cividis", False, "Cells (log scale)"),
    "Conditioned DEM": ("dem", "terrain", False, "Elevation (m)"),
}


# =================================================
# 2. LOGO, PAGE, AND DATA DETECTOR CONFIGURATION
# =================================================

_DATE_FULL = re.compile(r"(20\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])(?!\d)")
_DATE_YM = re.compile(r"(20\d{2})[-_.](0[1-9]|1[0-2])(?!\d)")


def _date_from_text(text):
    m = _DATE_FULL.search(text)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return d, d.isoformat()
        except ValueError:
            pass
    m = _DATE_YM.search(text)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        last = calendar.monthrange(y, mo)[1]
        return date(y, mo, last), f"{y}-{mo:02d}"
    return None, None


def scan_chirps():
    found = []
    if os.path.isdir(RAW_DIR):
        for dirpath, _, files in os.walk(RAW_DIR):
            for f in files:
                if "chirps" not in f.lower():
                    continue
                full = os.path.join(dirpath, f)
                try:
                    mod = datetime.fromtimestamp(os.path.getmtime(full)).date()
                except OSError:
                    continue
                d, label = _date_from_text(f)
                source = "file name"
                if d is None:
                    d, label, source = mod, mod.isoformat(), "file modified time"
                found.append({"File": f, "Date": label, "Source": source,
                              "_date": d, "_mod": mod})
    found.sort(key=lambda r: (r["_date"], r["_mod"]), reverse=True)

    latest = found[0] if found else None
    status_date = None
    text = read_status_file()
    if text:
        status_date, status_label = _date_from_text(text)
        if status_date and (latest is None or status_date > latest["_date"]):
            latest = {"File": "(status log)", "Date": status_label,
                      "Source": "status log", "_date": status_date,
                      "_mod": status_date}
    return {"latest": latest, "files": found, "status_date": status_date}


def check_latest_chirps():
    info = scan_chirps()
    latest = info["latest"]
    if latest is None:
        return "no files found", "bad"
    age = (date.today() - latest["_date"]).days
    return latest["Date"], ("ok" if age <= 45 else "warn")


def read_status_file():
    try:
        if os.path.exists(CHIRPS_STATUS_FILE):
            with open(CHIRPS_STATUS_FILE, "r", encoding="utf-8") as f:
                text = f.read().strip()
            return text[:300] if text else None
    except Exception:
        pass
    return None


def find_logo():
    for name in LOGO_FILES:
        path = os.path.join(ASSETS_DIR, name)
        if os.path.exists(path):
            return path
    return None


def logo_data_uri():
    path = find_logo()
    if not path:
        return ""
    try:
        ext = os.path.splitext(path)[1].lower()
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:{MIME.get(ext, 'image/png')};base64,{encoded}"
    except Exception:
        return ""


def page_icon():
    path = find_logo()
    if path and not path.lower().endswith(".svg"):
        try:
            from PIL import Image
            return Image.open(path)
        except Exception:
            pass
    return "🌊"


st.set_page_config(
    page_title="DUWE Flood Intelligence Engine",
    page_icon=page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
)


# =================================================
# 3. OPTIONAL DASHBOARD MODULES
# =================================================

try:
    from live_rainfall import render_live_banner, render_live_rainfall
    LIVE_RAIN_AVAILABLE = True
except ImportError:
    LIVE_RAIN_AVAILABLE = False

try:
    from flow_animation import render_flow_animation
    FLOW_AVAILABLE = True
except ImportError:
    FLOW_AVAILABLE = False

try:
    from forecast_panel import render_forecast_panel
    FORECAST_AVAILABLE = True
except ImportError:
    FORECAST_AVAILABLE = False


def run_module(func, label):
    try:
        func()
    except TypeError as exc:
        st.warning(
            f"{label} is installed but its function signature does not "
            f"match what the dashboard expects ({exc})."
        )
    except Exception as exc:
        st.warning(f"{label} could not be displayed: {exc}")


# =================================================
# 4. PROFESSIONAL VISUAL DESIGN
# =================================================

def compact(s):
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

.stApp {
    background-color: #F1F5F9;
    background-image:
        radial-gradient(1200px 520px at 94% -6%, rgba(94,200,242,.18), transparent 60%),
        radial-gradient(900px 420px at -6% 8%, rgba(21,150,166,.10), transparent 58%),
        linear-gradient(rgba(11,37,69,.028) 1px, transparent 1px),
        linear-gradient(90deg, rgba(11,37,69,.028) 1px, transparent 1px);
    background-size: auto, auto, 34px 34px, 34px 34px;
    color: #172B4D;
    font-family: 'Inter', "Segoe UI", system-ui, sans-serif;
}
.block-container {
    max-width: 1480px;
    padding-top: 1.35rem;
    padding-bottom: 2.8rem;
}
header[data-testid="stHeader"] { background: rgba(241,245,249,.92); }
footer { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFE 100%);
    border-right: 1px solid #DCE6F0;
    box-shadow: 4px 0 24px rgba(11,37,69,.03);
}
section[data-testid="stSidebar"] > div { padding-top: 1.4rem; }
.side-brand {
    display: flex; align-items: center; gap: 13px;
    padding: 8px 4px 16px;
}
.side-brand img {
    max-height: 52px; max-width: 115px; object-fit: contain;
    filter: drop-shadow(0 2px 5px rgba(11,37,69,.10));
}
.side-name {
    color: #0B2545; font-size: 15.5px; font-weight: 800; line-height: 1.25;
    letter-spacing: -0.2px;
}
.side-sub {
    color: #52708B; font-size: 11.5px; margin-top: 4px; line-height: 1.4; font-weight: 500;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #0B2545; font-weight: 800;
}
section[data-testid="stSidebar"] label {
    color: #344B64; font-weight: 600;
}
section[data-testid="stSidebar"] hr { border-color: #DCE6F0; }

/* ---------- HERO ---------- */
.hero {
    position: relative; overflow: hidden;
    background: linear-gradient(125deg, #06182C 0%, #0B2545 42%, #134074 72%, #117C8B 100%);
    border-radius: 20px; padding: 28px 30px 62px;
    margin-bottom: 22px;
    box-shadow: 0 14px 36px rgba(11,37,69,.20), 0 4px 12px rgba(11,37,69,.08);
    border: 1px solid rgba(255,255,255,.10);
}
.hero::before {
    content:""; position:absolute; inset:0; pointer-events:none; z-index:0;
    background:
        radial-gradient(ellipse 70% 55% at 12% 18%, rgba(31,122,156,.22) 0%, transparent 55%),
        radial-gradient(ellipse 55% 45% at 88% 75%, rgba(19,64,116,.30) 0%, transparent 50%),
        repeating-radial-gradient(circle at 88% 10%, rgba(255,255,255,.05) 0 1px, transparent 1px 28px);
}
.hero-left {
    position: relative; z-index: 2;
    display: flex; align-items: center; gap: 22px;
}
.logo-plate {
    display: flex; align-items: center; justify-content: center;
    min-width: 92px; min-height: 82px; padding: 12px 14px;
    border-radius: 16px; background: rgba(255,255,255,.97);
    box-shadow: 0 6px 20px rgba(0,0,0,.14), inset 0 1px 0 rgba(255,255,255,.85);
    border: 1px solid rgba(255,255,255,.55);
}
.logo-plate img {
    max-width: 195px; max-height: 78px; object-fit: contain;
}
.badge {
    color: #0B2545; font-size: 24px; font-weight: 900;
    letter-spacing: 2.5px; padding: 8px;
}
.eyebrow {
    color: #8FE3DE; font-size: 11px; font-weight: 800; letter-spacing: 2.2px;
    text-transform: uppercase; margin-bottom: 9px;
}
.hero-title {
    color: #FFFFFF !important;
    font-size: clamp(22px, 2.9vw, 33px);
    font-weight: 800; line-height: 1.12; letter-spacing: -.5px;
    text-shadow: 0 2px 10px rgba(0,0,0,.18);
}
.hero-sub {
    color: #C8DDF0 !important; font-size: 14.5px;
    margin-top: 9px; line-height: 1.55; font-weight: 500;
}
.hero-credit {
    color: #A8D4E8 !important; font-size: 13px;
    margin-top: 11px; font-weight: 600; letter-spacing: .2px;
    display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;
}
.pills {
    position: relative; z-index: 2; display: flex;
    flex-wrap: wrap; gap: 9px; margin-top: 20px;
}
.pill {
    display: inline-block; color: #FFFFFF !important;
    background: rgba(255,255,255,.11);
    border: 1px solid rgba(255,255,255,.22);
    border-radius: 999px; padding: 7px 14px;
    font-size: 11.5px; font-weight: 650; line-height: 1.45;
    backdrop-filter: blur(5px);
    transition: background .2s ease, transform .2s ease;
}
.pill:hover { background: rgba(255,255,255,.18); transform: translateY(-1px); }
.pill.ok {
    background: rgba(46,158,107,.28);
    border-color: rgba(117,228,177,.50);
}
.pill.warn {
    background: rgba(242,153,74,.24);
    border-color: rgba(255,199,120,.50);
}
.pill.bad {
    background: rgba(214,69,69,.26);
    border-color: rgba(255,140,140,.50);
}

/* ---------- RINGS & WAVES ---------- */
.ring {
    position: absolute; right: 72px; top: 48px; width: 18px; height: 18px;
    border-radius: 50%; border: 2px solid rgba(143,227,222,.55);
    animation: ring 4.6s ease-out infinite; z-index: 1; pointer-events: none;
}
.ring.r2 { animation-delay: 1.5s; }
.ring.r3 { animation-delay: 3.0s; }
@keyframes ring {
    0% { transform: scale(.35); opacity: .85; }
    100% { transform: scale(12); opacity: 0; }
}

.hero-waves, .intro-waves {
    position: absolute; left: 0; right: 0; bottom: 0; height: 58px;
    overflow: hidden; pointer-events: none; z-index: 1;
}
.hero-waves::before, .hero-waves::after,
.intro-waves::before, .intro-waves::after {
    content: ""; position: absolute; left: 0; bottom: 0; width: 200%; height: 100%;
    background-repeat: repeat-x; background-size: 50% 100%;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 120' preserveAspectRatio='none'%3E%3Cpath d='M0,60 C150,100 350,10 600,60 C850,110 1050,20 1200,60 L1200,120 L0,120 Z' fill='%23ffffff'/%3E%3C/svg%3E");
}
.hero-waves::before, .intro-waves::before {
    opacity: .20; animation: waveSlide 12s linear infinite;
}
.hero-waves::after, .intro-waves::after {
    opacity: .12; bottom: -9px; animation: waveSlide 18s linear infinite reverse;
}
.intro-waves { height: 150px; }
@keyframes waveSlide {
    from { transform: translateX(0); }
    to { transform: translateX(-50%); }
}

/* ---------- ASHLEY DUWE NAME ANIMATION ---------- */
.name-anim {
    display: inline-flex; font-weight: 900; letter-spacing: .5px; vertical-align: baseline;
}
.name-anim span {
    display: inline-block; opacity: 0; color: transparent;
    background: linear-gradient(105deg, #8FE3DE 0%, #5EC8F2 22%, #FFFFFF 48%, #5EC8F2 75%, #8FE3DE 100%);
    background-size: 300% 100%;
    -webkit-background-clip: text; background-clip: text;
    animation: dropIn .9s cubic-bezier(.2,.9,.25,1.15) forwards,
               shimmer 5.5s linear infinite;
}
.name-anim span:nth-child(1)  { animation-delay: 0.18s, 1.35s; }
.name-anim span:nth-child(2)  { animation-delay: 0.30s, 1.47s; }
.name-anim span:nth-child(3)  { animation-delay: 0.42s, 1.59s; }
.name-anim span:nth-child(4)  { animation-delay: 0.54s, 1.71s; }
.name-anim span:nth-child(5)  { animation-delay: 0.66s, 1.83s; }
.name-anim span:nth-child(6)  { animation-delay: 0.78s, 1.95s; }
.name-anim span:nth-child(7)  { animation-delay: 0.90s, 2.07s; }
.name-anim span:nth-child(8)  { animation-delay: 1.02s, 2.19s; }
.name-anim span:nth-child(9)  { animation-delay: 1.14s, 2.31s; }
.name-anim span:nth-child(10) { animation-delay: 1.26s, 2.43s; }
.name-anim span:nth-child(11) { animation-delay: 1.38s, 2.55s; }
@keyframes dropIn {
    0%   { opacity: 0; transform: translateY(-22px) scale(.72); filter: blur(5px); }
    65%  { opacity: 1; transform: translateY(2px) scale(1.04); filter: blur(0); }
    100% { opacity: 1; transform: none; filter: blur(0); }
}
@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    100% { background-position: 300% 50%; }
}

/* ---------- INTRO OVERLAY ---------- */
.intro {
    position: fixed; inset: 0; z-index: 999999; overflow: hidden;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    background: radial-gradient(circle at 50% 38%, #134074 0%, #0B2545 48%, #050F1E 100%);
    animation: introOut 1.1s ease 4.1s forwards;
}
.intro .name-anim { font-size: clamp(42px, 8.5vw, 96px); position: relative; z-index: 2; }
.intro-kicker {
    color: #8FE3DE; font-size: 12px; font-weight: 800; letter-spacing: 5.5px;
    text-transform: uppercase; margin-bottom: 20px; opacity: 0;
    animation: fadeUp .85s ease .18s forwards; position: relative; z-index: 2;
}
.intro-sub {
    color: #9FD8EA; letter-spacing: 4.5px; font-size: 12.5px; font-weight: 700;
    text-transform: uppercase; margin-top: 20px; opacity: 0;
    animation: fadeUp .85s ease 1.85s forwards; position: relative; z-index: 2;
}
.irg {
    position: absolute; left: 50%; top: 42%; width: 28px; height: 28px;
    margin: -14px 0 0 -14px; border-radius: 50%;
    border: 2px solid rgba(143,227,222,.50);
    animation: ringBig 3.5s ease-out infinite; z-index: 1;
}
.irg.b { animation-delay: 1.15s; }
.irg.c { animation-delay: 2.3s; }
@keyframes ringBig {
    0%   { transform: scale(.28); opacity: .9; }
    100% { transform: scale(36); opacity: 0; }
}
@keyframes introOut {
    to { opacity: 0; visibility: hidden; pointer-events: none; }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: none; }
}

/* ---------- KPI ---------- */
.kpi {
    position: relative; overflow: hidden; background: #FFFFFF;
    border: 1px solid #E1E8F0; border-radius: 16px;
    padding: 20px 20px 18px; height: 100%; min-height: 142px;
    box-shadow: 0 3px 14px rgba(16,42,67,.05);
    transition: transform .25s ease, box-shadow .25s ease;
    animation: fadeUp .65s ease both;
}
.kpi:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 26px rgba(16,42,67,.10);
}
.kpi::before {
    content: ""; position: absolute; left: 0; top: 0; right: 0; height: 3.5px;
    background: linear-gradient(90deg, #1596A6, #5EC8F2, #E9FBFF, #5EC8F2, #1596A6);
    background-size: 300% 100%;
    animation: flowLine 5.5s linear infinite;
}
.kpi-top {
    color: #64748B; font-size: 10.5px; font-weight: 800;
    text-transform: uppercase; letter-spacing: .9px; line-height: 1.5;
}
.kpi-value {
    color: #0B2545; font-size: clamp(22px, 2.4vw, 30px);
    font-weight: 800; margin-top: 11px; line-height: 1.15;
    font-family: 'JetBrains Mono', 'Inter', monospace;
    letter-spacing: -0.4px; font-variant-numeric: tabular-nums;
}
.kpi-sub {
    color: #718096; font-size: 12.5px; line-height: 1.45; margin-top: 7px; font-weight: 500;
}
@keyframes flowLine {
    from { background-position: 0% 0; }
    to   { background-position: 300% 0; }
}

/* ---------- SECTIONS ---------- */
.sec-title {
    color: #0B2545; font-size: 20.5px; font-weight: 800;
    line-height: 1.3; display: flex; align-items: center; gap: 12px;
    margin-top: 20px; margin-bottom: 4px; letter-spacing: -0.3px;
}
.sec-num {
    font-family: 'JetBrains Mono', Consolas, monospace; font-size: 11.5px; font-weight: 800;
    color: #FFFFFF; background: linear-gradient(135deg, #1596A6, #117C8B);
    padding: 4px 9px; border-radius: 8px; letter-spacing: .8px;
}
.sec-flow {
    height: 3px; width: 140px; border-radius: 3px; margin: 6px 0 3px;
    background: linear-gradient(90deg, #1596A6, #5EC8F2, #E9FBFF, #5EC8F2, #1596A6);
    background-size: 300% 100%;
    animation: flowLine 3.8s linear infinite;
}
.sec-sub {
    color: #64748B; font-size: 12.5px; line-height: 1.55;
    margin: 4px 0 15px 0; font-weight: 500;
}

/* ---------- CLASS CARDS ---------- */
.cls-grid {
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 13px; margin-top: 12px;
}
.cls {
    background: #FFFFFF; border: 1px solid #E1E8F0;
    border-left: 5px solid #1596A6; border-radius: 14px;
    padding: 15px 16px;
    box-shadow: 0 2px 10px rgba(16,42,67,.04);
    transition: transform .22s ease, box-shadow .22s ease;
}
.cls:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(16,42,67,.08);
}
.cls-name {
    color: #64748B; font-size: 11px; font-weight: 800;
    text-transform: uppercase; letter-spacing: .7px;
}
.cls-value {
    color: #0B2545; font-size: 23px; font-weight: 800; margin-top: 7px;
    font-family: 'JetBrains Mono', monospace; letter-spacing: -0.3px;
}
.cls-sub { color: #718096; font-size: 12.5px; margin-top: 3px; font-weight: 500; }

/* ---------- NOTES / DISCLAIMER / FOOTER ---------- */
.note {
    background: #FFFFFF; border: 1px solid #E1E8F0; border-radius: 14px;
    padding: 14px 17px; color: #475A70; font-size: 12.5px;
    line-height: 1.65; margin-top: 12px; font-weight: 500;
}
.disclaimer {
    background: #FFF8EB; border: 1px solid #F0D090; border-radius: 14px;
    padding: 14px 17px; color: #7A5410; font-size: 12.5px;
    line-height: 1.65; margin-top: 26px; font-weight: 500;
}
.footer {
    text-align: center; color: #7A8C9E; font-size: 12px;
    margin-top: 24px; padding-bottom: 8px; font-weight: 500;
}
.figcap {
    color: #52708B; font-size: 12.5px; line-height: 1.55; margin: 8px 0 14px;
    padding-left: 12px; border-left: 3px solid #BCD7E6; font-weight: 500;
}
.figcap b { color: #0B2545; }

/* ---------- SIDEBAR SIGNATURE ---------- */
.sig {
    position: relative; overflow: hidden; margin-top: 18px; padding: 14px 12px 24px;
    border-radius: 14px; text-align: center;
    background: linear-gradient(135deg, #0B2545, #134074);
}
.sig-by {
    display: block; color: #9FC3DA; font-size: 10px; letter-spacing: 1.5px;
    text-transform: uppercase; margin-bottom: 6px; font-weight: 700;
}
.sig .name-anim { font-size: 19px; }
.sig .hero-waves { height: 18px; }

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 5px; border-bottom: 1px solid #DCE6F0; padding-bottom: 1px;
}
.stTabs [data-baseweb="tab"] {
    height: 45px; padding: 0 18px; border-radius: 11px 11px 0 0;
    font-weight: 700; color: #52708B; background: transparent;
    transition: color .2s ease, background .2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #0B2545; background: rgba(21,150,166,.07);
}
.stTabs [aria-selected="true"] {
    color: #0B2545 !important; background: #FFFFFF;
}
.stTabs [data-baseweb="tab-highlight"] {
    height: 3px; background: linear-gradient(90deg, #1596A6, #5EC8F2);
}

/* ---------- RESPONSIVE ---------- */
@media (max-width: 900px) {
    .cls-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .hero-left { flex-direction: column; align-items: flex-start; }
    .ring { display: none; }
    .kpi-value { font-size: 24px; }
}

@media (prefers-reduced-motion: reduce) {
    .ring, .hero-waves::before, .hero-waves::after,
    .intro-waves::before, .intro-waves::after,
    .name-anim span, .kpi::before, .sec-flow, .irg {
        animation: none !important;
    }
    .name-anim span { opacity: 1; color: #8FE3DE; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def kpi(top, value, sub=""):
    return compact(f"""
    <div class="kpi">
        <div class="kpi-top">{top}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """)


_SECTION = {"n": 0}


def section(title, sub=""):
    _SECTION["n"] += 1
    num = f"{_SECTION['n']:02d}"
    st.markdown(
        compact(f"""
        <div class="sec-title"><span class="sec-num">{num}</span>{title}</div>
        <div class="sec-flow"></div>
        <div class="sec-sub">{sub}</div>
        """),
        unsafe_allow_html=True,
    )


def caption(label, text):
    st.markdown(f'<div class="figcap"><b>{label}.</b> {text}</div>',
                unsafe_allow_html=True)


# =================================================
# 5. DATA LOADING  (unchanged)
# =================================================

@st.cache_data(show_spinner=False)
def load_raster(path, mtime):
    if not os.path.exists(path):
        return None
    try:
        with rasterio.open(path) as src:
            arr = src.read(1).astype("float32")
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            arr[~np.isfinite(arr)] = np.nan

            pixel_km2 = PIXEL_AREA_KM2
            if src.crs is not None and src.crs.is_projected:
                area = abs(src.transform.a * src.transform.e) / 1e6
                if 1e-8 < area < 1:
                    pixel_km2 = area

            return {
                "array": arr,
                "pixel_km2": pixel_km2,
                "crs": str(src.crs) if src.crs else "Unknown",
                "shape": arr.shape,
            }
    except Exception:
        return None


def get_layer(key):
    path = PATHS[key]
    mtime = os.path.getmtime(path) if os.path.exists(path) else 0
    return load_raster(path, mtime)


def class_stats(layer):
    arr = layer["array"]
    px = layer["pixel_km2"]
    rows = []
    valid = arr[(arr >= 1) & (arr <= 4)]
    total = valid.size
    for i, name in enumerate(RISK_NAMES, start=1):
        count = int(np.sum(valid == i))
        rows.append({
            "Class": name,
            "Pixels": count,
            "Area (km²)": round(count * px, 2),
            "Share (%)": round(100 * count / total, 2) if total else 0.0,
        })
    return pd.DataFrame(rows), total


def downsample(arr, max_dim=1600):
    step = max(1, int(np.ceil(max(arr.shape) / max_dim)))
    return arr[::step, ::step]


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor="white")
    return buf.getvalue()


def decorate_map(ax, shape, step, px_km2, title):
    h, w = shape
    px_m = float(np.sqrt(px_km2) * 1000.0 * step)
    total_m = w * px_m
    length = 500
    for c in (500, 1000, 2000, 5000, 10000, 20000, 50000, 100000):
        if c <= total_m * 0.25:
            length = c
    x1, y = w * 0.97, h * 0.965
    x0 = x1 - length / px_m
    ax.plot([x0, x1], [y, y], color="#0B2545", lw=4.5, solid_capstyle="butt")
    ax.plot([x0, x1], [y, y], color="white", lw=1.8, solid_capstyle="butt")
    label = f"{length / 1000:g} km" if length >= 1000 else f"{length} m"
    ax.text((x0 + x1) / 2, y - h * 0.018, label, ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#0B2545")
    ax.text(0.955, 0.985, "N", transform=ax.transAxes, ha="center",
            va="top", fontsize=10, fontweight="bold", color="#0B2545")
    ax.annotate("", xy=(0.955, 0.955), xytext=(0.955, 0.885),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#0B2545", lw=1.6))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#9FB3C8")
        sp.set_linewidth(0.9)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold",
                 color="#0B2545", pad=10)


def _display_step(arr, max_dim=1600):
    return max(1, int(np.ceil(max(arr.shape) / max_dim)))


def draw_class_map(arr, title, visible=(1, 2, 3, 4), px_km2=PIXEL_AREA_KM2):
    show = downsample(arr).copy()
    for i in (1, 2, 3, 4):
        if i not in visible:
            show[show == i] = np.nan
    show[(show < 1) | (show > 4)] = np.nan

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = ListedColormap(RISK_COLORS)
    cmap.set_bad(color="#EEF2F7")
    ax.imshow(np.ma.masked_invalid(show), cmap=cmap, vmin=0.5, vmax=4.5,
              interpolation="nearest")
    decorate_map(ax, show.shape, _display_step(arr), px_km2, title)
    handles = [Patch(facecolor=c, edgecolor="#FFFFFF", label=n)
               for c, n in zip(RISK_COLORS, RISK_NAMES)]
    ax.legend(handles=handles, title="Risk class", loc="lower left",
              frameon=True, framealpha=0.94, fontsize=8, title_fontsize=9)
    return fig


def draw_continuous_map(arr, title, cmap_name, unit, log=False,
                        px_km2=PIXEL_AREA_KM2):
    show = downsample(arr).copy()
    if log:
        show = np.where(show > 0, np.log10(show), np.nan)
        unit = unit.replace("log scale", "log10")
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#EEF2F7")
    im = ax.imshow(np.ma.masked_invalid(show), cmap=cmap,
                   interpolation="nearest")
    decorate_map(ax, show.shape, _display_step(arr), px_km2, title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(unit, fontsize=9)
    cbar.outline.set_edgecolor("#9FB3C8")
    return fig


# =================================================
# 5b. HYDROLOGY ANIMATIONS (WATER FLOW AND FLOOD RISE)
# =================================================
# (Identical to your advanced version – kept fully intact)

STAGE_CAP_M = 15.0


def _blur3(a):
    p = np.pad(a, 1, mode="edge")
    out = np.zeros_like(a, dtype="float64")
    for dy in range(3):
        for dx in range(3):
            out += p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return out / 9.0


def _block_reduce(arr, step, how):
    h, w = arr.shape
    hh, ww = h // step, w // step
    blocks = arr[:hh * step, :ww * step].reshape(hh, step, ww, step)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if how == "max":
            return np.nanmax(blocks, axis=(1, 3))
        return np.nanmean(blocks, axis=(1, 3))


def _d8_accumulation(z, mask):
    H, W = z.shape
    key = np.where(mask, -z, np.inf).ravel()
    order = np.argsort(key, kind="stable").tolist()
    zl = z.ravel().tolist()
    mk = mask.ravel().tolist()
    acc = [1.0] * (H * W)
    nbs = ((-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414), (0, -1, 1.0),
           (0, 1, 1.0), (1, -1, 1.414), (1, 0, 1.0), (1, 1, 1.414))
    for i in order:
        if not mk[i]:
            break
        y, x = divmod(i, W)
        zi = zl[i]
        best, best_s = -1, 0.0
        for dy, dx, d in nbs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                j = ny * W + nx
                if mk[j]:
                    s = (zi - zl[j]) / d
                    if s > best_s:
                        best_s, best = s, j
        if best >= 0:
            acc[best] += acc[i]
    return np.array(acc, dtype="float64").reshape(H, W)


def _hand_threshold(z, mask, seeds, cap):
    H, W = z.shape
    N = H * W
    zl = z.ravel().tolist()
    mk = mask.ravel().tolist()
    t = [float("inf")] * N
    origin = [0.0] * N
    heap = []
    for i in seeds:
        t[i] = 0.0
        origin[i] = zl[i]
        heap.append((0.0, i))
    heapq.heapify(heap)
    nbs = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0),
           (1, 1))
    while heap:
        ti, i = heapq.heappop(heap)
        if ti > t[i]:
            continue
        if ti > cap:
            break
        y, x = divmod(i, W)
        oe = origin[i]
        for dy, dx in nbs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                j = ny * W + nx
                if not mk[j]:
                    continue
                tj = max(ti, zl[j] - oe)
                if tj < t[j]:
                    t[j] = tj
                    origin[j] = oe
                    heapq.heappush(heap, (tj, j))
    return np.array(t, dtype="float64").reshape(H, W)


def _b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


@st.cache_data(show_spinner="Preparing terrain for the animations...")
def prepare_terrain(dem_path, dem_mtime, acc_path, acc_mtime, max_dim=220):
    dem_layer = load_raster(dem_path, dem_mtime)
    if dem_layer is None:
        return None

    dem = dem_layer["array"]
    step = max(1, int(np.ceil(max(dem.shape) / max_dim)))
    z = _block_reduce(dem, step, "mean")
    mask = np.isfinite(z)
    if mask.sum() < 100:
        return None

    H, W = z.shape
    px_m = float(np.sqrt(dem_layer["pixel_km2"]) * 1000.0 * step)
    zf = np.where(mask, z, np.nanmin(z))
    zs = _blur3(zf)

    gy, gx = np.gradient(zs, px_m)
    vexag = 2.0
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy) * vexag)
    aspect = np.arctan2(-gy * vexag, gx * vexag)
    az, alt = np.radians(315.0), np.radians(45.0)
    shaded = (np.sin(alt) * np.sin(slope)
              + np.cos(alt) * np.cos(slope) * np.cos((az - np.pi / 2.0) - aspect))
    hs = np.clip(shaded, 0, 1)
    lo, hi = np.nanpercentile(z[mask], [1, 99])
    elv = np.clip((zf - lo) / max(hi - lo, 1e-6), 0, 1)

    vx, vy = -gx, -gy
    mag = np.hypot(vx, vy)
    p95 = max(float(np.percentile(mag[mask], 95)), 1e-9)
    vscale = 1.5
    fx = np.clip(vx / (p95 * vscale), -1, 1) * 127
    fy = np.clip(vy / (p95 * vscale), -1, 1) * 127

    acc = None
    acc_layer = load_raster(acc_path, acc_mtime) if os.path.exists(acc_path) else None
    if acc_layer is not None and acc_layer["array"].shape == dem.shape:
        acc = _block_reduce(np.where(np.isfinite(acc_layer["array"]),
                                     acc_layer["array"], np.nan), step, "max")
    if acc is None or not np.isfinite(acc[mask]).any():
        acc = _d8_accumulation(zs, mask)
    acc = np.where(mask & np.isfinite(acc), np.maximum(acc, 0), 0)
    la = np.log10(1.0 + acc)
    la_max = max(float(np.percentile(la[mask], 99.7)), 1e-6)
    acc_n = np.clip(la / la_max, 0, 1) ** 1.6

    cut = np.percentile(la[mask], 98.5)
    seeds = np.flatnonzero((mask & (la >= cut)).ravel()).tolist()
    if not seeds:
        seeds = np.argsort(np.where(mask, zf, np.inf).ravel())[:20].tolist()
    t = _hand_threshold(zf, mask, seeds, STAGE_CAP_M)
    thr = np.where(np.isfinite(t) & (t <= STAGE_CAP_M),
                   np.round(t / STAGE_CAP_M * 254), 255)

    return {
        "w": int(W), "h": int(H), "cap": STAGE_CAP_M,
        "px_m": px_m, "cell_km2": (px_m / 1000.0) ** 2,
        "emin": float(lo), "emax": float(hi),
        "hs": _b64((hs * 255).astype("uint8")),
        "el": _b64((elv * 255).astype("uint8")),
        "mask": _b64(mask.astype("uint8")),
        "acc": _b64((acc_n * 255).astype("uint8")),
        "fx": _b64(np.round(fx).astype("int8")),
        "fy": _b64(np.round(fy).astype("int8")),
        "thr": _b64(thr.astype("uint8")),
    }


def get_terrain():
    dem_p, acc_p = PATHS["dem"], PATHS["accumulation"]
    dm = os.path.getmtime(dem_p) if os.path.exists(dem_p) else 0
    am = os.path.getmtime(acc_p) if os.path.exists(acc_p) else 0
    return prepare_terrain(dem_p, dm, acc_p, am)


# -------------------------------------------------
# Shared front-end code for both animations
# (kept identical – these are already excellent)
# -------------------------------------------------

ANIM_CSS = """
<style>
*{box-sizing:border-box}
body{margin:0;font-family:"Segoe UI",Arial,sans-serif;color:#E7F1FA;background:transparent}
.card{background:linear-gradient(160deg,#08203A,#0B2F4F 60%,#0E4660);border:1px solid rgba(255,255,255,.10);border-radius:18px;padding:14px;box-shadow:0 10px 28px rgba(8,32,58,.28)}
.row{display:flex;gap:14px;align-items:flex-start}
.stagewrap{flex:1;min-width:0;display:flex;justify-content:center}
.stage{position:relative;width:100%;border-radius:12px;overflow:hidden;background:#07121F;border:1px solid rgba(255,255,255,.12)}
.stage canvas{position:absolute;left:0;top:0;width:100%;height:100%;display:block}
.chip{position:absolute;left:10px;top:10px;font-size:11px;font-weight:700;letter-spacing:.4px;padding:6px 10px;border-radius:999px;background:rgba(7,18,31,.66);border:1px solid rgba(255,255,255,.18);color:#BFE9F7;backdrop-filter:blur(4px)}
.hint{position:absolute;left:10px;bottom:34px;font-size:10.5px;color:#cfe6f3;opacity:.85;background:rgba(7,18,31,.55);padding:3px 8px;border-radius:8px}
.north{position:absolute;right:12px;top:10px;text-align:center;font-size:11px;font-weight:800;color:#fff;line-height:1;text-shadow:0 1px 3px rgba(0,0,0,.6)}
.sb{position:absolute;left:12px;bottom:12px;height:5px;background:#fff;border:1px solid #0B2545;box-shadow:0 1px 3px rgba(0,0,0,.5)}
.sbl{position:absolute;left:12px;bottom:19px;font-size:10px;font-weight:700;color:#fff;text-shadow:0 1px 3px rgba(0,0,0,.7)}
.ctl{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin-top:12px;font-size:12px;color:#BFD7E8}
.ctl label{display:flex;gap:7px;align-items:center}
.btn{cursor:pointer;border:0;border-radius:999px;padding:8px 16px;font-weight:800;font-size:12px;color:#06263F;background:linear-gradient(90deg,#8FE3DE,#5EC8F2)}
.btn:hover{filter:brightness(1.07)}
input[type=range]{accent-color:#5EC8F2;width:120px}
.lg{display:flex;align-items:center;gap:6px;margin-left:auto;font-size:11px;color:#9FC3DA}
.lg i{display:inline-block;width:46px;height:8px;border-radius:5px}
.g1{background:linear-gradient(90deg,#1b4a6b,#4aa3d8)}
.g2{background:linear-gradient(90deg,#4aa3d8,#e9fbff)}
.panel{width:290px;flex:none;display:flex;flex-direction:column;gap:10px}
.stat{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.10);border-radius:12px;padding:11px 13px}
.stat b{display:block;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#8FB6CF;font-weight:800}
.stat span{display:block;margin-top:4px;font-size:23px;font-weight:850;color:#fff;font-variant-numeric:tabular-nums}
.stat small{display:block;margin-top:2px;font-size:11px;color:#9FC3DA}
.sev{display:inline-block;margin-top:6px;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:800;color:#06263F;background:#8FE3DE}
.chart{width:100%;height:auto;border-radius:12px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.10)}
.dl{height:9px;border-radius:6px;background:linear-gradient(90deg,#8fdcf0,#40b4e6,#1578c8,#0e46a0,#08286e)}
.dlt{display:flex;justify-content:space-between;font-size:10px;color:#9FC3DA;margin-top:3px}
.note{font-size:10.5px;line-height:1.5;color:#8FB6CF}
@media (max-width:820px){.row{flex-direction:column}.panel{width:100%}}
</style>
"""

COMMON_JS = """
const D = __DATA__;
const w = D.w, h = D.h, S = 4;
function b64(s){const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a;}
const hs=b64(D.hs), mask=b64(D.mask), elv=b64(D.el), acc=b64(D.acc), thr=b64(D.thr);
const fx=new Int8Array(b64(D.fx).buffer), fy=new Int8Array(b64(D.fy).buffer);
function lerp(a,b,t){return a+(b-a)*t;}
function ramp(stops,t){
  t=Math.max(0,Math.min(1,t));
  for(let i=1;i<stops.length;i++){
    if(t<=stops[i][0]){
      const a=stops[i-1],b=stops[i],k=(t-a[0])/((b[0]-a[0])||1);
      return [lerp(a[1][0],b[1][0],k),lerp(a[1][1],b[1][1],k),lerp(a[1][2],b[1][2],k)];
    }
  }
  return stops[stops.length-1][1];
}
const LIGHT=[[0,[190,222,198]],[.25,[224,232,194]],[.5,[224,206,160]],[.75,[192,162,122]],[1,[150,126,102]]];
const DARK=[[0,[18,50,76]],[.5,[34,72,100]],[1,[78,108,132]]];
function makeCanvas(cw,ch){const c=document.createElement('canvas');c.width=cw;c.height=ch;return c;}
function makeBase(theme){
  const c=makeCanvas(w,h),x=c.getContext('2d'),img=x.createImageData(w,h),d=img.data;
  const st=theme==='dark'?DARK:LIGHT, out=theme==='dark'?[7,18,32]:[238,242,247];
  for(let k=0;k<w*h;k++){
    let r,g,b;
    if(!mask[k]){r=out[0];g=out[1];b=out[2];}
    else{
      const col=ramp(st,elv[k]/255), f=theme==='dark'?(0.35+0.95*hs[k]/255):(0.5+0.7*hs[k]/255);
      r=col[0]*f;g=col[1]*f;b=col[2]*f;
    }
    d[4*k]=r;d[4*k+1]=g;d[4*k+2]=b;d[4*k+3]=255;
  }
  x.putImageData(img,0,0);
  return c;
}
function sizeStage(stage,maxH){
  stage.style.aspectRatio=w+' / '+h;
  stage.parentElement.style.maxWidth='100%';
  stage.style.width='min(100%, '+(maxH*w/h)+'px)';
}
function addMapFurniture(stage){
  const totalM=w*D.px_m, cands=[500,1000,2000,5000,10000,20000,50000,100000];
  let L=cands[0]; for(const c of cands){ if(c<=totalM*0.3) L=c; }
  const bar=document.createElement('div'); bar.className='sb'; bar.style.width=(L/totalM*100)+'%';
  const lab=document.createElement('div'); lab.className='sbl'; lab.textContent=(L>=1000?(L/1000)+' km':L+' m');
  const n=document.createElement('div'); n.className='north'; n.innerHTML='N<br>&#9650;';
  stage.appendChild(bar); stage.appendChild(lab); stage.appendChild(n);
}
"""

FLOW_BODY = """
<div class="card">
  <div class="row"><div class="stagewrap">
    <div class="stage" id="stage">
      <canvas id="base"></canvas><canvas id="net"></canvas><canvas id="trail"></canvas>
      <div class="chip">SURFACE WATER FLOW &middot; STEEPEST DESCENT ON YOUR DEM</div>
      <div class="hint">Click the map to release a tracer burst</div>
    </div>
  </div></div>
  <div class="ctl">
    <button class="btn" id="pp">&#10074;&#10074; Pause</button>
    <label>Speed <input id="spd" type="range" min="0.3" max="3" step="0.1" value="1"></label>
    <label>Particles <input id="cnt" type="range" min="500" max="6000" step="250" value="3000"></label>
    <label><input id="netc" type="checkbox" checked> Drainage network</label>
    <span class="lg"><i class="g1"></i>hillslope runoff<i class="g2"></i>channel flow</span>
  </div>
</div>
<script>
__COMMON__
(function(){
  const stage=document.getElementById('stage');
  sizeStage(stage,560); addMapFurniture(stage);
  const base=document.getElementById('base'), net=document.getElementById('net'), trail=document.getElementById('trail');
  base.width=w; base.height=h; net.width=w; net.height=h;
  base.getContext('2d').drawImage(makeBase('dark'),0,0);
  (function(){
    const x=net.getContext('2d'),img=x.createImageData(w,h),d=img.data;
    for(let k=0;k<w*h;k++){
      if(mask[k]&&acc[k]>150){const a=(acc[k]-150)/105;d[4*k]=120;d[4*k+1]=210;d[4*k+2]=255;d[4*k+3]=40+150*a;}
    }
    x.putImageData(img,0,0);
  })();
  const W=w*S,H=h*S; trail.width=W; trail.height=H;
  const tctx=trail.getContext('2d'); tctx.lineCap='round';
  const valid=[]; for(let k=0;k<w*h;k++) if(mask[k]) valid.push(k);
  const NMAX=6000;
  const px=new Float32Array(NMAX),py=new Float32Array(NMAX),age=new Float32Array(NMAX),life=new Float32Array(NMAX),hot=new Uint8Array(NMAX);
  function spawn(i){
    const k=valid[(Math.random()*valid.length)|0];
    px[i]=(k%w)+Math.random(); py[i]=((k/w)|0)+Math.random();
    age[i]=0; life[i]=90+Math.random()*200; hot[i]=0;
  }
  for(let i=0;i<NMAX;i++){spawn(i); age[i]=Math.random()*life[i];}
  let N=3000, speed=1, playing=true;
  document.getElementById('pp').onclick=function(){playing=!playing;this.innerHTML=playing?'&#10074;&#10074; Pause':'&#9654; Play';};
  document.getElementById('spd').oninput=function(){speed=parseFloat(this.value);};
  document.getElementById('cnt').oninput=function(){N=parseInt(this.value);};
  document.getElementById('netc').onchange=function(){net.style.display=this.checked?'block':'none';};
  trail.style.cursor='crosshair';
  trail.addEventListener('click',function(e){
    const r=trail.getBoundingClientRect();
    const cx=(e.clientX-r.left)/r.width*w, cy=(e.clientY-r.top)/r.height*h;
    for(let j=0;j<90;j++){
      const i=NMAX-1-j;
      px[i]=cx+(Math.random()-.5)*2.2; py[i]=cy+(Math.random()-.5)*2.2;
      age[i]=0; life[i]=520; hot[i]=1;
    }
  });
  function frame(){
    if(playing){
      tctx.globalCompositeOperation='destination-out';
      tctx.fillStyle='rgba(0,0,0,0.075)'; tctx.fillRect(0,0,W,H);
      tctx.globalCompositeOperation='lighter';
      const total=Math.min(NMAX,N);
      for(let n=0;n<total+90;n++){
        const i=n<total?n:NMAX-1-(n-total);
        if(i>=NMAX) continue;
        const x=px[i],y=py[i],ix=x|0,iy=y|0;
        if(ix<0||iy<0||ix>=w||iy>=h){spawn(i);continue;}
        const k=iy*w+ix;
        if(!mask[k]||age[i]>life[i]){spawn(i);continue;}
        const vx=fx[k]*1.5/127, vy=fy[k]*1.5/127, m=Math.hypot(vx,vy), a=acc[k]/255;
        if(m<1e-4){age[i]+=6;continue;}
        const stp=speed*0.32*(0.25+Math.min(m,1.5))*(0.7+1.9*a);
        const nx=x+vx/m*stp, ny=y+vy/m*stp;
        if(hot[i]){tctx.strokeStyle='rgba(255,214,102,0.95)';tctx.lineWidth=2.4;}
        else{tctx.strokeStyle='rgba('+((90+150*a)|0)+','+((190+60*a)|0)+',255,'+(0.30+0.55*a)+')';tctx.lineWidth=0.7+1.9*a;}
        tctx.beginPath();tctx.moveTo(x*S,y*S);tctx.lineTo(nx*S,ny*S);tctx.stroke();
        px[i]=nx;py[i]=ny;age[i]+=1;
      }
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
"""

FLOOD_BODY = """
<div class="card">
  <div class="row">
    <div class="stagewrap">
      <div class="stage" id="stage">
        <canvas id="base"></canvas><canvas id="water"></canvas><canvas id="rain"></canvas>
        <div class="chip">FLOOD RISE &middot; HAND-STYLE INUNDATION ON YOUR DEM</div>
      </div>
    </div>
    <div class="panel">
      <div class="stat"><b>River stage above channel</b><span id="stg">0.0 m</span>
        <small id="sev"></small></div>
      <div class="stat"><b>Inundated area</b><span id="area">0 km&sup2;</span><small id="pct"></small></div>
      <canvas id="chart" class="chart" width="300" height="150"></canvas>
      <div><div class="dl"></div><div class="dlt"><span>0 m</span><span>Water depth</span><span id="dmax">6 m+</span></div></div>
      <div class="note">Illustrative terrain-based inundation (water connected to the drainage network). Not a hydrodynamic model and not a forecast.</div>
    </div>
  </div>
  <div class="ctl">
    <button class="btn" id="pp">&#10074;&#10074; Pause</button>
    <label>Stage <input id="lvl" type="range" min="0" max="__CAP__" step="0.05" value="0" style="width:220px"></label>
    <label>Speed <input id="spd" type="range" min="0.3" max="3" step="0.1" value="1"></label>
    <label><input id="rn" type="checkbox" checked> Storm rain</label>
  </div>
</div>
<script>
__COMMON__
(function(){
  const cap=D.cap;
  const stage=document.getElementById('stage');
  sizeStage(stage,560); addMapFurniture(stage);
  const base=document.getElementById('base'), water=document.getElementById('water'), rain=document.getElementById('rain');
  base.width=w; base.height=h; water.width=w; water.height=h;
  base.getContext('2d').drawImage(makeBase('light'),0,0);
  const wctx=water.getContext('2d'), wimg=wctx.createImageData(w,h), wd=wimg.data;
  const RW=800, RH=Math.round(800*h/w); rain.width=RW; rain.height=RH;
  const rctx=rain.getContext('2d');
  const drops=[]; for(let i=0;i<140;i++) drops.push([Math.random()*RW,Math.random()*RH,6+Math.random()*10,2+Math.random()*3]);
  const WS=[[0,[143,220,240]],[.15,[64,180,230]],[.4,[21,120,200]],[.7,[14,70,160]],[1,[8,40,110]]];
  const cnt=new Float64Array(256); let valid=0;
  for(let k=0;k<w*h;k++){ if(!mask[k]) continue; valid++; if(thr[k]<255) cnt[thr[k]]++; }
  const cum=new Float64Array(255); let run=0; for(let i=0;i<255;i++){run+=cnt[i];cum[i]=run;}
  const cellKm2=D.cell_km2, totalKm2=valid*cellKm2;
  const cv=document.getElementById('chart'), cx=cv.getContext('2d');
  const maxArea=Math.max(cum[254]*cellKm2,1e-6);
  function drawChart(level){
    const CW=cv.width,CH=cv.height,pl=40,pr=10,pt=12,pb=26;
    cx.clearRect(0,0,CW,CH);
    cx.strokeStyle='rgba(255,255,255,.18)';cx.lineWidth=1;
    cx.beginPath();cx.moveTo(pl,pt);cx.lineTo(pl,CH-pb);cx.lineTo(CW-pr,CH-pb);cx.stroke();
    cx.fillStyle='#9FC3DA';cx.font='10px Segoe UI, Arial';
    cx.fillText('Stage (m)',CW/2-18,CH-6);
    cx.save();cx.translate(10,CH/2+22);cx.rotate(-Math.PI/2);cx.fillText('Area (km\\u00b2)',0,0);cx.restore();
    cx.fillText('0',pl-3,CH-pb+12);cx.fillText(cap+'',CW-pr-12,CH-pb+12);
    cx.fillText(Math.round(maxArea)+'',4,pt+8);
    const g=cx.createLinearGradient(0,pt,0,CH-pb);g.addColorStop(0,'rgba(94,200,242,.55)');g.addColorStop(1,'rgba(94,200,242,.05)');
    cx.beginPath();cx.moveTo(pl,CH-pb);
    for(let i=0;i<255;i++){cx.lineTo(pl+(CW-pl-pr)*i/254,CH-pb-(CH-pb-pt)*(cum[i]*cellKm2/maxArea));}
    cx.lineTo(CW-pr,CH-pb);cx.closePath();cx.fillStyle=g;cx.fill();
    cx.beginPath();
    for(let i=0;i<255;i++){const X=pl+(CW-pl-pr)*i/254,Y=CH-pb-(CH-pb-pt)*(cum[i]*cellKm2/maxArea);if(i===0)cx.moveTo(X,Y);else cx.lineTo(X,Y);}
    cx.strokeStyle='#8FE3DE';cx.lineWidth=2;cx.stroke();
    const li=Math.max(0,Math.min(254,Math.round(level/cap*254)));
    const mx=pl+(CW-pl-pr)*li/254,my=CH-pb-(CH-pb-pt)*(cum[li]*cellKm2/maxArea);
    cx.strokeStyle='rgba(255,255,255,.5)';cx.setLineDash([3,3]);cx.beginPath();cx.moveTo(mx,CH-pb);cx.lineTo(mx,my);cx.stroke();cx.setLineDash([]);
    cx.fillStyle='#fff';cx.beginPath();cx.arc(mx,my,4,0,Math.PI*2);cx.fill();
  }
  let level=0, playing=true, speed=1, hold=0, last=performance.now(), rainOn=true;
  const lvl=document.getElementById('lvl');
  document.getElementById('pp').onclick=function(){playing=!playing;this.innerHTML=playing?'&#10074;&#10074; Pause':'&#9654; Play';};
  document.getElementById('spd').oninput=function(){speed=parseFloat(this.value);};
  document.getElementById('rn').onchange=function(){rainOn=this.checked;};
  lvl.oninput=function(){playing=false;document.getElementById('pp').innerHTML='&#9654; Play';level=parseFloat(this.value);hold=0;};
  function severity(p){
    if(p<=0) return ['Within channel','#8FE3DE'];
    if(p<2) return ['Localised flooding','#BDF2A8'];
    if(p<8) return ['Moderate extent','#F2E27A'];
    if(p<20) return ['Extensive flooding','#F5A962'];
    return ['Widespread flooding','#F26B6B'];
  }
  function drawWater(level,t){
    const lv=level/cap*254;
    for(let k=0;k<w*h;k++){
      const o=4*k;
      if(level>0&&mask[k]&&thr[k]<255&&thr[k]<=lv){
        const depth=(lv-thr[k])/254*cap, u=Math.min(1,depth/6), col=ramp(WS,u);
        const sh=0.92+0.08*Math.sin(t*2.2+(k%w)*0.35+((k/w)|0)*0.21);
        const edge=depth<0.25?0.6+0.4*depth/0.25:1;
        wd[o]=col[0];wd[o+1]=col[1];wd[o+2]=col[2];wd[o+3]=255*(0.55+0.35*u)*sh*edge;
      } else { wd[o+3]=0; }
    }
    wctx.putImageData(wimg,0,0);
  }
  function drawRain(){
    rctx.clearRect(0,0,RW,RH);
    if(!(rainOn&&playing&&level<cap)) return;
    rctx.strokeStyle='rgba(205,232,255,.45)';rctx.lineWidth=1;rctx.beginPath();
    for(const d of drops){
      rctx.moveTo(d[0],d[1]);rctx.lineTo(d[0]-d[2]*0.25,d[1]+d[2]);
      d[1]+=d[3]*3;d[0]-=d[3]*0.7;
      if(d[1]>RH||d[0]<0){d[1]=-10;d[0]=Math.random()*RW+40;}
    }
    rctx.stroke();
  }
  function tick(now){
    const dt=Math.min(0.05,(now-last)/1000); last=now;
    if(playing){
      if(level>=cap){hold-=dt;if(hold<=0){level=0;}}
      else{level=Math.min(cap,level+dt*speed*cap/16);if(level>=cap)hold=2;}
      lvl.value=level;
    }
    drawWater(level,now/1000); drawRain(); drawChart(level);
    const li=Math.max(0,Math.min(254,Math.round(level/cap*254)));
    const a=level<=0?0:cum[li]*cellKm2, p=totalKm2?100*a/totalKm2:0, sv=severity(p);
    document.getElementById('stg').textContent=level.toFixed(1)+' m';
    document.getElementById('area').innerHTML=(a>=100?a.toFixed(0):a.toFixed(1))+' km&sup2;';
    document.getElementById('pct').textContent=p.toFixed(1)+'% of the study area';
    const s=document.getElementById('sev');s.innerHTML='<span class="sev" style="background:'+sv[1]+'">'+sv[0]+'</span>';
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();
</script>
"""


def _anim_page(body):
    return "<!DOCTYPE html><html><head><meta charset='utf-8'>" + ANIM_CSS + \
           "</head><body>" + body + "</body></html>"


def build_flow_html(terrain):
    body = FLOW_BODY.replace("__COMMON__", COMMON_JS)
    return _anim_page(body).replace("__DATA__", json.dumps(terrain))


def build_flood_html(terrain):
    body = FLOOD_BODY.replace("__COMMON__", COMMON_JS)
    body = body.replace("__CAP__", str(STAGE_CAP_M))
    return _anim_page(body).replace("__DATA__", json.dumps(terrain))


def embed_html(html_str, height):
    if hasattr(st, "iframe"):
        st.iframe(html_str, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html_str, height=height, scrolling=False)


def render_flow_builtin():
    terrain = get_terrain()
    if terrain is None:
        st.warning("The conditioned DEM could not be loaded, so the water "
                   f"flow animation is unavailable:\n\n`{PATHS['dem']}`")
        return
    embed_html(build_flow_html(terrain), 690)


def render_flood_builtin():
    terrain = get_terrain()
    if terrain is None:
        st.warning("The conditioned DEM could not be loaded, so the flood "
                   f"simulation is unavailable:\n\n`{PATHS['dem']}`")
        return
    embed_html(build_flood_html(terrain), 690)


# =================================================
# 6. SIDEBAR
# =================================================

NAME_HTML = "".join(
    f"<span>{c if c != ' ' else '&nbsp;'}</span>" for c in DEVELOPER)

chirps = scan_chirps()
latest_chirps = chirps["latest"]
if latest_chirps:
    _age = (date.today() - latest_chirps["_date"]).days
    chirps_date = latest_chirps["Date"] + (f" ({_age} d ago)" if _age >= 0 else "")
    status_class = "ok" if _age <= 45 else "warn"
else:
    chirps_date, status_class = "no files found", "bad"

proc_path = PATHS["rainfall"]
if os.path.exists(proc_path):
    proc_dt = datetime.fromtimestamp(os.path.getmtime(proc_path)).date()
    proc_stale = bool(latest_chirps and proc_dt < latest_chirps["_mod"])
    proc_label = proc_dt.isoformat()
    proc_text = f"Rainfall layer processed: {proc_label}"
    if proc_stale:
        proc_text += " (older than the latest download: re-run alignment)"
    proc_class = "warn" if proc_stale else "ok"
else:
    proc_label, proc_stale = "missing", False
    proc_text, proc_class = "Rainfall layer not found", "bad"

status_text = read_status_file()
logo_uri = logo_data_uri()

layers_found = {k: os.path.exists(p) for k, p in PATHS.items()}
n_found = sum(layers_found.values())

with st.sidebar:
    logo_tag = f'<img src="{logo_uri}" alt="logo">' if logo_uri else ""
    st.markdown(
        compact(f"""
        <div class="side-brand">
            {logo_tag}
            <div>
                <div class="side-name">DUWE Disaster Intelligence</div>
                <div class="side-sub">Flood Intelligence Engine</div>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.markdown("### Controls")
    visible_names = st.multiselect(
        "Risk classes to display",
        RISK_NAMES,
        default=RISK_NAMES,
        help="Filters the risk map and the area summary chart.",
    )
    visible_classes = tuple(RISK_NAMES.index(n) + 1 for n in visible_names)

    st.markdown("### Modules")
    show_live = st.checkbox("Live rainfall", value=LIVE_RAIN_AVAILABLE,
                            disabled=not LIVE_RAIN_AVAILABLE)
    show_flow = st.checkbox("Legacy flow module", value=FLOW_AVAILABLE,
                            disabled=not FLOW_AVAILABLE)
    show_forecast = st.checkbox("Forecast panel", value=FORECAST_AVAILABLE,
                                disabled=not FORECAST_AVAILABLE)

    st.markdown("---")
    st.caption(f"Data layers found: {n_found} of {len(PATHS)}")
    if latest_chirps:
        if status_class == "ok":
            st.success(f"CHIRPS current: latest download {latest_chirps['Date']}")
        else:
            st.warning(f"CHIRPS older than 45 days: latest download "
                       f"{latest_chirps['Date']}")
    else:
        st.error("No CHIRPS download found in the raw folder.")
    if status_text:
        st.caption(f"Automation log says (not checked by the dashboard): "
                   f"{status_text}")
        if (latest_chirps and status_class == "ok"
                and "outdated" in status_text.lower()):
            st.info("The log still says outdated, but a recent download was "
                    "found. The log is only rewritten when the CHIRPS check "
                    "script runs, so run that script again.")
    with st.expander("CHIRPS diagnostics"):
        st.caption(f"Scanning: {RAW_DIR}")
        if chirps["files"]:
            diag = pd.DataFrame(chirps["files"])[["File", "Date", "Source"]]
            st.dataframe(diag.head(8), hide_index=True, width="stretch")
            st.caption(f"{len(chirps['files'])} CHIRPS file(s) found.")
        else:
            st.warning("No file with 'chirps' in its name was found in the "
                       "raw folder or its sub-folders. Check where the "
                       "download script saves its files.")
        st.caption(f"Processed rainfall layer: {proc_label}")
    if st.button("Reload data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        f'<div class="sig"><span class="sig-by">Designed &amp; developed by</span>'
        f'<span class="name-anim">{NAME_HTML}</span>'
        f'<div class="hero-waves"></div></div>',
        unsafe_allow_html=True,
    )


# =================================================
# 7. HERO BANNER + INTRO
# =================================================

logo_img_tag = f'<img src="{logo_uri}" alt="logo">' if logo_uri else ""
badge_tag = "" if logo_uri else '<div class="badge">DUWE</div>'
layers_class = "ok" if n_found == len(PATHS) else (
    "warn" if n_found else "bad")
now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

if not st.session_state.get("intro_done"):
    st.session_state["intro_done"] = True
    st.markdown(
        compact(f"""
        <div class="intro">
            <div class="irg"></div><div class="irg b"></div><div class="irg c"></div>
            <div class="intro-kicker">DUWE Disaster Intelligence</div>
            <div class="name-anim">{NAME_HTML}</div>
            <div class="intro-sub">Flood Intelligence Engine</div>
            <div class="intro-waves"></div>
        </div>
        """),
        unsafe_allow_html=True,
    )

st.markdown(
    compact(f"""
    <div class="hero">
        <div class="ring"></div><div class="ring r2"></div><div class="ring r3"></div>
        <div class="hero-left">
            <div class="logo-plate">
                {logo_img_tag}
                {badge_tag}
            </div>
            <div>
                <div class="eyebrow">Hydrological geospatial analysis · Chikwawa District, Malawi</div>
                <div class="hero-title">DUWE Flood Intelligence Engine</div>
                <div class="hero-sub">
                    Terrain, rainfall and exposure combined into a transparent,
                    map-based flood-susceptibility picture for decision support.
                </div>
                <div class="hero-credit">Developed by <span class="name-anim">{NAME_HTML}</span></div>
            </div>
        </div>
        <div class="pills">
            <span class="pill {status_class}">Latest CHIRPS download: {chirps_date}</span>
            <span class="pill {proc_class}">{proc_text}</span>
            <span class="pill {layers_class}">Layers: {n_found}/{len(PATHS)}</span>
            <span class="pill">Resolution: {PIXEL_SIZE_M} m</span>
            <span class="pill">Session: {now_str}</span>
        </div>
        <div class="hero-waves"></div>
    </div>
    """),
    unsafe_allow_html=True,
)

if LIVE_RAIN_AVAILABLE and show_live:
    run_module(render_live_banner, "Live rainfall banner")


# =================================================
# 8. KEY INDICATORS + TABS
# =================================================

risk_layer = get_layer("risk_class")
rain_layer = get_layer("rainfall")
exp_layer = get_layer("exposure")

if risk_layer is None:
    st.error(
        "The classified risk raster could not be loaded. Expected file:\n\n"
        f"`{PATHS['risk_class']}`"
    )
    st.stop()

stats_df, valid_pixels = class_stats(risk_layer)
px_km2 = risk_layer["pixel_km2"]
total_km2 = valid_pixels * px_km2
high_km2 = float(
    stats_df.loc[stats_df["Class"].isin(["High", "Very High"]),
                 "Area (km²)"].sum()
)
high_pct = 100 * high_km2 / total_km2 if total_km2 else 0.0

if rain_layer is not None and np.isfinite(rain_layer["array"]).any():
    rain_mean = float(np.nanmean(rain_layer["array"]))
    rain_max = float(np.nanmax(rain_layer["array"]))
    rain_val = f"{rain_mean:,.1f} mm"
    rain_sub = (f"Mean over study area. Peak cell {rain_max:,.1f} mm. "
                f"Layer processed {proc_label}.")
else:
    rain_val, rain_sub = "N/A", "Rainfall raster not available."

if exp_layer is not None and np.isfinite(exp_layer["array"]).any():
    exp_val = f"{float(np.nanmean(exp_layer['array'])):,.2f}"
    exp_sub = "Mean exposure proxy value."
else:
    exp_val, exp_sub = "N/A", "Exposure raster not available."


# Infrastructure helpers (kept intact)
INFRA_DIRS = [
    os.path.join(BASE_DIR, r"01_data\infrastructure"),
    RAW_DIR,
    os.path.join(BASE_DIR, r"01_data\processed"),
]
INFRA_PREFIXES = ("infrastructure", "facilities", "critical_infrastructure")
INFRA_EXTS = (".csv", ".geojson", ".json")


def find_infrastructure_files():
    found = []
    for d in INFRA_DIRS:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            low = f.lower()
            if not low.endswith(INFRA_EXTS):
                continue
            if d == INFRA_DIRS[0] or low.startswith(INFRA_PREFIXES):
                found.append(os.path.join(d, f))
    return found


def _coords(c):
    if c and isinstance(c[0], (int, float)):
        yield c
    else:
        for s in c:
            yield from _coords(s)


def _geom_center(geom):
    pts = list(_coords(geom.get("coordinates", [])))
    if not pts:
        return None
    return (float(np.mean([p[0] for p in pts])),
            float(np.mean([p[1] for p in pts])))


def _pick(props, keys, default):
    lowered = {str(k).lower(): v for k, v in props.items()}
    for k in keys:
        if k in lowered and lowered[k] not in (None, ""):
            return lowered[k]
    return default


@st.cache_data(show_spinner=False)
def load_infrastructure(path, mtime):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(path)
            cols = {c.lower().strip(): c for c in df.columns}
            lon = next((cols[k] for k in ("longitude", "lon", "long", "lng", "x")
                        if k in cols), None)
            lat = next((cols[k] for k in ("latitude", "lat", "y")
                        if k in cols), None)
            if lon is None or lat is None:
                return None, ("CSV needs longitude and latitude columns "
                              "(lon/lat or x/y, in decimal degrees).")
            name = next((cols[k] for k in ("name", "facility", "facility_name",
                                           "title") if k in cols), None)
            typ = next((cols[k] for k in ("type", "category", "class",
                                          "infrastructure_type",
                                          "facility_type") if k in cols), None)
            out = pd.DataFrame({
                "Name": df[name] if name else "Unnamed",
                "Type": df[typ] if typ else "Infrastructure",
                "lon": pd.to_numeric(df[lon], errors="coerce"),
                "lat": pd.to_numeric(df[lat], errors="coerce"),
            })
        else:
            with open(path, "r", encoding="utf-8") as f:
                gj = json.load(f)
            rows = []
            for ft in (gj.get("features", []) if isinstance(gj, dict) else []):
                center = _geom_center(ft.get("geometry") or {})
                if center is None:
                    continue
                props = ft.get("properties") or {}
                rows.append({
                    "Name": _pick(props, ("name", "facility", "title"),
                                  "Unnamed"),
                    "Type": _pick(props, ("type", "category", "amenity",
                                          "fclass", "building", "class"),
                                  "Infrastructure"),
                    "lon": center[0],
                    "lat": center[1],
                })
            out = pd.DataFrame(rows, columns=["Name", "Type", "lon", "lat"])
        out = out.dropna(subset=["lon", "lat"]).reset_index(drop=True)
        if out.empty:
            return None, "No valid point locations were found in this file."
        return out, None
    except Exception as exc:
        return None, f"Could not read this file: {exc}"


def sample_risk_class(df):
    from rasterio.warp import transform
    xs, ys = df["lon"].tolist(), df["lat"].tolist()
    classes = []
    with rasterio.open(PATHS["risk_class"]) as src:
        if src.crs is not None and src.crs.to_epsg() != 4326:
            xs, ys = transform("EPSG:4326", src.crs, xs, ys)
        for v in src.sample(zip(xs, ys), indexes=1, masked=True):
            val = v[0]
            if np.ma.is_masked(val):
                classes.append(np.nan)
                continue
            val = float(val)
            classes.append(val if 1 <= val <= 4 else np.nan)
    return np.array(classes)


def exposure_by_class(risk_arr, exp_arr, px_km2):
    in_risk = (risk_arr >= 1) & (risk_arr <= 4)
    total_exp = float(np.nansum(exp_arr[in_risk & np.isfinite(exp_arr)]))
    thr = float(np.nanpercentile(exp_arr[in_risk & np.isfinite(exp_arr)], 75))
    rows = []
    for i, name in enumerate(RISK_NAMES, start=1):
        m = (risk_arr == i) & np.isfinite(exp_arr)
        cls_exp = float(np.sum(exp_arr[m]))
        rows.append({
            "Class": name,
            "Mean exposure": round(float(np.mean(exp_arr[m])), 3)
            if m.any() else 0.0,
            "Share of total exposure (%)": round(
                100 * cls_exp / total_exp, 2) if total_exp else 0.0,
            "High-exposure area (km²)": round(
                float(np.sum(m & (exp_arr >= thr))) * px_km2, 2),
        })
    return pd.DataFrame(rows), thr


tab_over, tab_flow, tab_flood, tab_explore, tab_infra, tab_live = st.tabs([
    "Overview", "Water flow", "Flood simulation", "Layer explorer",
    "Infrastructure at risk", "Live & forecast",
])

with tab_over:
    section("Key indicators",
            "Summary of the classified flood-susceptibility surface.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("Study area", f"{total_km2:,.0f} km²",
                        f"{valid_pixels:,} valid cells classified."),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("High + Very High", f"{high_km2:,.1f} km²",
                        f"{high_pct:.1f}% of the classified area."),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Rainfall (CHIRPS)", rain_val, rain_sub),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Exposure proxy", exp_val, exp_sub),
                    unsafe_allow_html=True)

    section("Flood risk map",
            "Classified flood susceptibility. Use the sidebar to filter classes.")

    map_col, chart_col = st.columns([3, 2])

    with map_col:
        if visible_classes:
            fig = draw_class_map(risk_layer["array"],
                                 "Updated flood risk classification",
                                 visible_classes, px_km2=risk_layer["pixel_km2"])
            st.pyplot(fig)
            png_bytes = fig_to_png(fig)
            plt.close(fig)
            st.download_button("Download map (PNG)", png_bytes,
                               file_name="duwe_flood_risk_map.png",
                               mime="image/png")
        else:
            st.info("Select at least one risk class in the sidebar.")

    with chart_col:
        st.markdown("**Area by risk class (km²)**")
        chart_df = stats_df.set_index("Class")[["Area (km²)"]]
        if visible_classes:
            chart_df = chart_df.loc[[RISK_NAMES[i - 1] for i in visible_classes]]
        st.bar_chart(chart_df, color="#1596A6")
        st.dataframe(stats_df, hide_index=True, width="stretch")
        st.download_button("Download statistics (CSV)",
                           stats_df.to_csv(index=False).encode("utf-8"),
                           file_name="duwe_risk_class_statistics.csv",
                           mime="text/csv")

    cards = []
    for color, (_, row) in zip(RISK_COLORS, stats_df.iterrows()):
        cards.append(
            f'<div class="cls" style="border-left-color:{color}">'
            f'<div class="cls-name">{row["Class"]}</div>'
            f'<div class="cls-value">{row["Area (km²)"]:,.1f} km²</div>'
            f'<div class="cls-sub">{row["Share (%)"]:.1f}% of classified area</div>'
            f'</div>'
        )
    st.markdown(f'<div class="cls-grid">{"".join(cards)}</div>',
                unsafe_allow_html=True)


with tab_flow:
    section("Water flow animation",
            "Surface runoff paths computed from your conditioned DEM.")
    render_flow_builtin()
    caption("Animation A",
            "Each particle moves in the steepest-descent direction of the "
            "terrain. Brightness and speed increase with log flow "
            "accumulation, so hillslope runoff gathers into the drainage "
            "network. Click the map to release a burst of tracer particles "
            "and watch where water from that point travels.")

with tab_flood:
    section("Flood simulation",
            "Stage-driven inundation connected to the drainage network.")
    render_flood_builtin()
    caption("Animation B",
            "River stage rises from 0 to "
            f"{STAGE_CAP_M:g} m above the drainage network. A cell is shown "
            "flooded when its height above the nearest connected channel is "
            "below the current stage (a HAND-style approach). The curve "
            "shows how inundated area grows with stage. This is an "
            "illustrative terrain-based visualisation, not a hydrodynamic "
            "model or a forecast.")

with tab_explore:
    section("Layer explorer",
            "Inspect any input or output layer used by the engine.")

    choice = st.selectbox("Layer", list(EXPLORER_LAYERS.keys()))
    key, cmap_name, categorical, unit = EXPLORER_LAYERS[choice]
    layer = get_layer(key)

    if layer is None:
        st.warning(f"This layer is not available yet:\n\n`{PATHS[key]}`")
    else:
        arr = layer["array"]
        if categorical:
            fig = draw_class_map(arr, choice, px_km2=layer["pixel_km2"])
        else:
            fig = draw_continuous_map(arr, choice, cmap_name, unit,
                                      log=(key == "accumulation"),
                                      px_km2=layer["pixel_km2"])

        e1, e2 = st.columns([3, 2])
        with e1:
            st.pyplot(fig)
            plt.close(fig)
        with e2:
            finite = arr[np.isfinite(arr)]
            if finite.size:
                summary = pd.DataFrame({
                    "Statistic": ["Minimum", "Mean", "Median", "Maximum",
                                  "Valid cells"],
                    "Value": [f"{finite.min():,.3f}", f"{finite.mean():,.3f}",
                              f"{np.median(finite):,.3f}", f"{finite.max():,.3f}",
                              f"{finite.size:,}"],
                })
                st.dataframe(summary, hide_index=True, width="stretch")
            st.caption(f"CRS: {layer['crs']}  |  Grid: "
                       f"{layer['shape'][1]} × {layer['shape'][0]} cells")


with tab_infra:
    section("Infrastructure at risk",
            "Where exposure and flood susceptibility overlap, and which "
            "facilities sit in High or Very High risk zones.")

    st.markdown("#### Exposure in each risk class")
    if exp_layer is None:
        st.warning("The exposure proxy raster is not available:\n\n"
                   f"`{PATHS['exposure']}`")
    elif exp_layer["array"].shape != risk_layer["array"].shape:
        st.warning("The exposure raster and the risk raster have different "
                   "grid sizes. Re-align them before this analysis can run.")
    else:
        exp_arr = exp_layer["array"]
        risk_arr = risk_layer["array"]
        exp_df, exp_thr = exposure_by_class(risk_arr, exp_arr, px_km2)

        hi_share = float(exp_df.loc[exp_df["Class"].isin(["High", "Very High"]),
                                    "Share of total exposure (%)"].sum())
        hi_area = float(exp_df.loc[exp_df["Class"].isin(["High", "Very High"]),
                                   "High-exposure area (km²)"].sum())

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(kpi("Exposure in High + Very High",
                            f"{hi_share:.1f}%",
                            "Share of total exposure-proxy value."),
                        unsafe_allow_html=True)
        with k2:
            st.markdown(kpi("High-exposure area at risk",
                            f"{hi_area:,.1f} km²",
                            "Top-quartile exposure inside High or Very High "
                            "risk zones."),
                        unsafe_allow_html=True)
        with k3:
            st.markdown(kpi("Exposure threshold",
                            f"{exp_thr:,.2f}",
                            "75th percentile of the exposure proxy."),
                        unsafe_allow_html=True)

        a1, a2 = st.columns([3, 2])
        with a1:
            overlay = np.where(risk_arr >= 3, exp_arr, np.nan)
            fig = draw_continuous_map(
                overlay, "Exposure inside High and Very High risk zones",
                "OrRd", "Exposure index", px_km2=px_km2)
            st.pyplot(fig)
            plt.close(fig)
        with a2:
            st.dataframe(exp_df, hide_index=True, width="stretch")
            st.download_button(
                "Download exposure summary (CSV)",
                exp_df.to_csv(index=False).encode("utf-8"),
                file_name="duwe_exposure_by_risk_class.csv", mime="text/csv")

    st.markdown("#### Facilities in High and Very High risk zones")
    infra_files = find_infrastructure_files()

    if not infra_files:
        st.info(
            "No infrastructure dataset was found. To screen individual "
            "facilities (schools, clinics, bridges, markets and so on), add a "
            "CSV or GeoJSON file of point locations to:\n\n"
            f"`{INFRA_DIRS[0]}`\n\n"
            "A CSV needs `longitude` and `latitude` columns in decimal degrees. "
            "`name` and `type` columns are optional but recommended."
        )
    else:
        chosen = st.selectbox(
            "Infrastructure dataset", infra_files,
            format_func=lambda p: os.path.basename(p))
        infra_df, infra_err = load_infrastructure(chosen, os.path.getmtime(chosen))

        if infra_err:
            st.error(infra_err)
        else:
            classes = sample_risk_class(infra_df)
            infra_df = infra_df.copy()
            infra_df["Risk class"] = [
                RISK_NAMES[int(c) - 1] if np.isfinite(c) else "Outside study area"
                for c in classes
            ]
            total_assets = len(infra_df)
            at_risk = infra_df[infra_df["Risk class"].isin(["High", "Very High"])]
            outside = int((infra_df["Risk class"] == "Outside study area").sum())

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(kpi("Facilities screened", f"{total_assets:,}",
                                f"{outside:,} outside the study area."),
                            unsafe_allow_html=True)
            with m2:
                st.markdown(kpi("In High or Very High risk", f"{len(at_risk):,}",
                                "Candidates for priority attention."),
                            unsafe_allow_html=True)
            with m3:
                inside = total_assets - outside
                pct = 100 * len(at_risk) / inside if inside else 0.0
                st.markdown(kpi("Share at high risk", f"{pct:.1f}%",
                                "Of facilities inside the study area."),
                            unsafe_allow_html=True)

            order = RISK_NAMES + ["Outside study area"]
            summary = (pd.crosstab(infra_df["Type"], infra_df["Risk class"])
                       .reindex(columns=order, fill_value=0))
            summary["Total"] = summary.sum(axis=1)
            st.markdown("**Facilities by type and risk class**")
            st.dataframe(summary, width="stretch")

            if at_risk.empty:
                st.success("No facilities in this dataset fall in High or Very "
                           "High risk zones.")
            else:
                st.markdown("**Facilities at risk**")
                table = at_risk[["Name", "Type", "Risk class", "lat", "lon"]]
                st.dataframe(table.sort_values(["Risk class", "Type"]),
                             hide_index=True, width="stretch")
                st.map(at_risk[["lat", "lon"]])
                st.download_button(
                    "Download facilities at risk (CSV)",
                    table.to_csv(index=False).encode("utf-8"),
                    file_name="duwe_infrastructure_at_risk.csv",
                    mime="text/csv")


with tab_live:
    if LIVE_RAIN_AVAILABLE and show_live:
        section("Live rainfall", "Near-real-time rainfall observations.")
        run_module(render_live_rainfall, "Live rainfall")

    if FORECAST_AVAILABLE and show_forecast:
        section("Rainfall forecast", "Short-range forecast information.")
        run_module(render_forecast_panel, "Forecast panel")

    if FLOW_AVAILABLE and show_flow:
        section("Flow animation", "Simulated drainage flow paths.")
        run_module(render_flow_animation, "Flow animation")

    if not ((LIVE_RAIN_AVAILABLE and show_live)
            or (FORECAST_AVAILABLE and show_forecast)
            or (FLOW_AVAILABLE and show_flow)):
        st.info("No live or forecast modules are enabled or installed. "
                "Enable them in the sidebar, or add live_rainfall.py, "
                "forecast_panel.py and flow_animation.py next to this script.")


# =================================================
# 12. METHOD NOTES, DISCLAIMER AND FOOTER
# =================================================

section("About this analysis")
st.markdown(
    compact("""
    <div class="note">
        The risk classes combine terrain (conditioned DEM), flow
        accumulation, catchment extent, CHIRPS rainfall and an exposure
        proxy. Class areas assume the pixel size read from each raster.
        If a raster uses geographic coordinates, the dashboard falls back to
        the configured 30 m pixel area.
    </div>
    """),
    unsafe_allow_html=True,
)

st.markdown(
    compact("""
    <div class="note">
        <b>Animation methods.</b> Water flow: the DEM is block-averaged and
        smoothed, and each particle follows the local downslope gradient;
        brightness uses flow accumulation. Flood rise: drainage cells are
        taken from the highest-accumulation cells, and every other cell is
        assigned the lowest stage at which water connected to the drainage
        network can reach it (height above nearest drainage, computed with
        a priority-queue search). Animations use a reduced-resolution copy
        of the DEM for speed.
    </div>
    """),
    unsafe_allow_html=True,
)

st.markdown(
    compact("""
    <div class="disclaimer">
        <b>Decision-support notice.</b> This dashboard presents geospatial
        flood-susceptibility information. It is not an operational flood
        forecast or an official warning system. Follow guidance from the
        Department of Disaster Management Affairs and local authorities.
    </div>
    """),
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="footer">DUWE Disaster Intelligence &nbsp;|&nbsp; '
    f'Developed by {DEVELOPER}</div>',
    unsafe_allow_html=True,
)