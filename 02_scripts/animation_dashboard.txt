"""
DUWE DISASTER INTELLIGENCE
DUWE FLOOD INTELLIGENCE ENGINE
Professional Flood Intelligence Dashboard
Logo: save your logo as  04_assets/logo.png  (also accepted: .jpg .jpeg .webp .svg).
"""
import io
import os
import re
import base64
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
import streamlit as st
# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
RAW_DIR = os.path.join(BASE_DIR, r"01_data\raw")
ASSETS_DIR = os.path.join(BASE_DIR, "04_assets")
CHIRPS_STATUS_FILE = os.path.join(BASE_DIR, r"02_scripts\automation\logs\chirps_status.txt")
PATHS = {
    "dem": os.path.join(BASE_DIR, r"01_data\processed\dem_conditioned_duwe.tif"),
    "catchment": os.path.join(BASE_DIR, r"01_data\processed\catchment_duwe.tif"),
    "accumulation": os.path.join(BASE_DIR, r"01_data\processed\flow_accumulation_duwe.tif"),
    "rainfall": os.path.join(BASE_DIR, r"01_data\processed\chirps_rainfall_aligned_duwe.tif"),
    "rain_hazard": os.path.join(BASE_DIR, r"01_data\processed\rainfall_enhanced_hazard_duwe.tif"),
    "hazard_class": os.path.join(BASE_DIR, r"01_data\processed\rainfall_hazard_classified_duwe.tif"),
    "risk": os.path.join(BASE_DIR, r"01_data\processed\risk_updated_duwe.tif"),
    "risk_class": os.path.join(BASE_DIR, r"01_data\processed\risk_updated_classified_duwe.tif"),
    "exposure": os.path.join(BASE_DIR, r"01_data\processed\exposure_proxy_duwe.tif"),
}
PIXEL_AREA_KM2 = 0.0009   # 30 m x 30 m
PIXEL_SIZE_M = 30
DEVELOPER = "Ashley Duwe"
# -------------------------------------------------
# LOOK AND FEEL
# -------------------------------------------------
NAVY, BLUE, TEAL = "#0B2545", "#134074", "#1F7A9C"
RISK_NAMES = ["Low", "Medium", "High", "Very High"]
RISK_COLORS = ["#2E9E6B", "#F2C94C", "#F2994A", "#D64545"]
LOGO_FILES = ["logo.png", "logo.jpg", "logo.jpeg", "logo.webp", "logo.svg"]
ICON_FILES = ["icon.png", "icon.jpg", "icon.jpeg", "icon.webp", "icon.svg"]   # square emblem for small sizes
def find_logo():
    for name in LOGO_FILES:
        path = os.path.join(ASSETS_DIR, name)
        if os.path.exists(path):
            return path
    return None
def find_icon():
    for name in ICON_FILES:
        path = os.path.join(ASSETS_DIR, name)
        if os.path.exists(path):
            return path
    return None
def page_icon():
    path = find_icon() or find_logo()
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
    initial_sidebar_state="expanded"
)
# Live rainfall / infrastructure panels (the dashboard still works without them)
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
    from flood_simulation_panel import render_flood_simulation
    SIM_AVAILABLE = True
except ImportError:
    SIM_AVAILABLE = False
def html(s):
    """Collapse indentation so Streamlit's markdown never treats HTML as a code block."""
    return "".join(line.strip() for line in s.splitlines())
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

.stApp {
    background: linear-gradient(160deg, #F0F5FA 0%, #E8F0F8 40%, #F4F7FB 100%);
    color: #1F2937;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2.5rem;
    max-width: 1420px;
}
footer {visibility:hidden;}
header[data-testid="stHeader"] {background:transparent;}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFE 100%);
    border-right: 1px solid #D6E2F0;
    box-shadow: 4px 0 24px rgba(11,37,69,.04);
}
.side-brand {
    display:flex; align-items:center; gap:12px;
    margin-bottom:8px; padding:4px 2px;
}
.side-brand img {
    height:48px; max-width:130px; object-fit:contain;
    filter: drop-shadow(0 2px 4px rgba(11,37,69,.12));
}
.side-name {
    font-weight:800; color:#0B2545; font-size:15.5px;
    line-height:1.2; letter-spacing:-0.2px;
}
.side-sub {color:#5B6B7F; font-size:12.5px; font-weight:500;}

/* ---------- HERO ---------- */
.hero {
    background: linear-gradient(125deg, #0A1F3A 0%, #0F2F5A 35%, #134074 65%, #1A6B8A 100%);
    border-radius: 20px;
    padding: 26px 32px 34px;
    display:flex; align-items:center; justify-content:space-between;
    gap:22px; flex-wrap:wrap;
    box-shadow: 0 12px 40px rgba(11,37,69,.28), 0 4px 12px rgba(11,37,69,.12);
    margin-bottom: 18px;
    position:relative; overflow:hidden;
}
.hero::before {
    content:"";
    position:absolute; inset:0;
    background:
        radial-gradient(ellipse 80% 60% at 15% 20%, rgba(31,122,156,.25) 0%, transparent 55%),
        radial-gradient(ellipse 60% 50% at 85% 70%, rgba(19,64,116,.35) 0%, transparent 50%);
    pointer-events:none; z-index:0;
}
.hero-left, .pills {position:relative; z-index:2;}
.logo-plate {
    background: rgba(255,255,255,.97);
    border-radius: 16px;
    padding: 10px 16px;
    display:flex; align-items:center; justify-content:center;
    min-height:72px;
    box-shadow: 0 4px 16px rgba(0,0,0,.12), inset 0 1px 0 rgba(255,255,255,.8);
    border: 1px solid rgba(255,255,255,.6);
}
.logo-plate img {height:68px; max-width:340px; object-fit:contain;}
.badge {
    font-weight:800; letter-spacing:3.5px; color:#0B2545;
    font-size:24px; padding:8px 12px;
}
.hero-title {
    color:#FFFFFF !important;
    font-size:31px; font-weight:800;
    line-height:1.12; letter-spacing:-0.4px;
    text-shadow: 0 2px 8px rgba(0,0,0,.18);
}
.hero-sub {
    color:#B8D4EA !important;
    font-size:15.5px; margin-top:7px; font-weight:500;
}
.hero-credit {
    color:#D4E8F7 !important;
    font-size:13.5px; margin-top:10px;
    font-weight:600; letter-spacing:.2px;
    display:flex; align-items:center; gap:8px;
}
.ashley-anim {
    display:inline-block;
    background: linear-gradient(90deg, #7EC8E8, #A8E6CF, #7EC8E8, #5DADE2);
    background-size: 300% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: ashleyShine 4.5s ease-in-out infinite;
    font-weight:800; letter-spacing:.4px;
}
@keyframes ashleyShine {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.ashley-pulse {
    display:inline-block;
    width:8px; height:8px; border-radius:50%;
    background:#5DADE2;
    box-shadow: 0 0 0 0 rgba(93,173,226,.7);
    animation: pulseDot 2.2s ease-out infinite;
}
@keyframes pulseDot {
    0% {box-shadow: 0 0 0 0 rgba(93,173,226,.65);}
    70% {box-shadow: 0 0 0 10px rgba(93,173,226,0);}
    100% {box-shadow: 0 0 0 0 rgba(93,173,226,0);}
}

/* ---------- PILLS ---------- */
.pills {display:flex; gap:10px; flex-wrap:wrap;}
.pill {
    background: rgba(255,255,255,.13);
    color:#FFFFFF !important;
    border: 1px solid rgba(255,255,255,.25);
    border-radius: 999px;
    padding: 7px 16px;
    font-size:13px; font-weight:600;
    backdrop-filter: blur(6px);
    transition: all .25s ease;
}
.pill:hover {background: rgba(255,255,255,.22); transform: translateY(-1px);}
.pill.ok {background: rgba(46,158,107,.38); border-color: rgba(46,158,107,.45);}
.pill.warn {background: rgba(242,153,74,.40); border-color: rgba(242,153,74,.5);}
.pill.bad {background: rgba(214,69,69,.42); border-color: rgba(214,69,69,.55);}

/* ---------- WAVES (hero bottom) ---------- */
.waves {
    position:absolute; left:0; right:0; bottom:0;
    height:48px; z-index:1; pointer-events:none; overflow:hidden;
}
.wave {
    position:absolute; left:0; bottom:0;
    width:200%; height:100%;
}
.wave.w1 {
    fill: rgba(255,255,255,.11);
    animation: waveflow 14s linear infinite;
}
.wave.w2 {
    fill: rgba(255,255,255,.07);
    bottom: -8px;
    animation: waveflow 22s linear infinite reverse;
}
.wave.w3 {
    fill: rgba(93,173,226,.12);
    bottom: -14px;
    animation: waveflow 18s linear infinite;
}
@keyframes waveflow {
    from {transform: translateX(0);}
    to {transform: translateX(-50%);}
}

/* ---------- WATER STRIP (footer) ---------- */
.water-strip {
    height: 7px; border-radius: 5px; margin-top: 28px;
    background: linear-gradient(90deg,
        #0B2545 0%, #134074 18%, #1F7A9C 36%,
        #5DADE2 50%, #1F7A9C 64%, #134074 82%, #0B2545 100%);
    background-size: 280% 100%;
    animation: stripflow 9s linear infinite;
    box-shadow: 0 2px 10px rgba(31,122,156,.25);
}
@keyframes stripflow {
    from {background-position: 0% 0;}
    to {background-position: 280% 0;}
}

/* ---------- FLOWING WATER ACCENT (used near simulation areas) ---------- */
.flow-accent {
    height: 4px; border-radius: 3px; margin: 14px 0 8px;
    background: linear-gradient(90deg,
        transparent 0%, #1F7A9C 20%, #5DADE2 45%,
        #A8E6CF 55%, #5DADE2 70%, #1F7A9C 90%, transparent 100%);
    background-size: 220% 100%;
    animation: flowAccent 6.5s ease-in-out infinite;
    opacity: .85;
}
@keyframes flowAccent {
    0% {background-position: 100% 0;}
    100% {background-position: -100% 0;}
}

/* ---------- KPI CARDS ---------- */
.kpi {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 3px 14px rgba(16,42,67,.07), 0 1px 3px rgba(16,42,67,.04);
    border: 1px solid #E0E9F4;
    height: 100%;
    transition: transform .28s ease, box-shadow .28s ease;
    position: relative; overflow: hidden;
}
.kpi::after {
    content:"";
    position:absolute; top:0; right:0;
    width:80px; height:80px;
    background: radial-gradient(circle at top right, rgba(31,122,156,.06), transparent 70%);
    pointer-events:none;
}
.kpi:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 28px rgba(16,42,67,.11), 0 3px 8px rgba(16,42,67,.06);
}
.kpi-top {
    color: #5B6B7F;
    font-size: 12px; font-weight: 700;
    text-transform: uppercase; letter-spacing: .6px;
}
.kpi-value {
    font-size: 31px; font-weight: 800;
    color: #0B2545; margin-top: 7px;
    line-height: 1.1;
    font-family: 'JetBrains Mono', 'Inter', monospace;
    letter-spacing: -0.5px;
}
.kpi-sub {
    font-size: 13px; color: #6B7A90; margin-top: 5px; font-weight: 500;
}

/* ---------- SECTION TITLES ---------- */
.sec-title {
    font-size: 21px; font-weight: 800; color: #0B2545;
    border-left: 5px solid #1F7A9C;
    padding-left: 12px; margin-top: 8px;
    letter-spacing: -0.3px;
}
.sec-sub {
    color: #6B7A90; font-size: 13.5px;
    margin: 3px 0 14px 17px; font-weight: 500;
}

/* ---------- CLASS CARDS ---------- */
.cls-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px; margin-top: 8px;
}
.cls {
    background: #FFFFFF;
    border: 1px solid #E0E9F4;
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 2px 10px rgba(16,42,67,.05);
    transition: transform .25s ease, box-shadow .25s ease;
}
.cls:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(16,42,67,.09);
}
.cls-name {
    font-size: 12px; font-weight: 700;
    text-transform: uppercase; letter-spacing: .55px; color: #5B6B7F;
}
.cls-val {
    font-size: 25px; font-weight: 800; color: #0B2545;
    font-family: 'JetBrains Mono', monospace; letter-spacing: -0.4px;
}
.cls-pct {font-size: 13.5px; color: #6B7A90; font-weight: 500;}
.stackbar {
    display: flex; height: 15px; border-radius: 9px;
    overflow: hidden; margin-top: 14px;
    background: #E3E9F2;
    box-shadow: inset 0 1px 3px rgba(0,0,0,.06);
}

/* ---------- NOTES & FOOTER ---------- */
.note {
    color: #6B7A90; font-size: 13px; margin-top: 12px;
    line-height: 1.55; font-weight: 500;
}
.foot {
    text-align: center; color: #5B6B7F;
    font-size: 13.5px; margin-top: 32px;
    padding-top: 20px; border-top: 1px solid #D4E0EE;
    line-height: 1.65;
}
.foot img {
    height: 30px; vertical-align: middle; margin-right: 10px;
    filter: drop-shadow(0 1px 3px rgba(11,37,69,.12));
}
.foot b {color: #0B2545;}

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; border-bottom: 1px solid #D4E0EE;
    padding-bottom: 2px;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 700; padding: 11px 20px;
    border-radius: 11px 11px 0 0;
    transition: background .2s ease, color .2s ease;
}
.stTabs [aria-selected="true"] {
    color: #134074 !important;
    background: linear-gradient(180deg, #EAF1FA 0%, #F4F8FC 100%);
    box-shadow: 0 -2px 0 #1F7A9C inset;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #F0F5FA;
}

/* ---------- METRIC & CONTAINER POLISH ---------- */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E0E9F4;
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 2px 10px rgba(16,42,67,.05);
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border-radius: 16px;
    border: 1px solid #E0E9F4 !important;
    box-shadow: 0 3px 14px rgba(16,42,67,.05);
}

/* ---------- SCIENCE BADGE (small floating label) ---------- */
.sci-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(31,122,156,.09);
    color: #1F7A9C;
    font-size: 11.5px; font-weight: 700;
    letter-spacing: .4px; text-transform: uppercase;
    padding: 4px 11px; border-radius: 999px;
    border: 1px solid rgba(31,122,156,.18);
    margin-bottom: 6px;
}
.sci-badge::before {
    content: "";
    width: 6px; height: 6px; border-radius: 50%;
    background: #1F7A9C;
    animation: pulseDot 2.4s ease-out infinite;
}

/* ---------- REDUCED MOTION ---------- */
@media (prefers-reduced-motion: reduce) {
    .wave, .water-strip, .flow-accent, .ashley-anim, .ashley-pulse {
        animation: none !important;
    }
}
@media (max-width: 900px) {
    .cls-grid {grid-template-columns: repeat(2, 1fr);}
    .hero-title {font-size: 23px;}
    .kpi-value {font-size: 26px;}
}
</style>
"""
st.markdown(html(CSS), unsafe_allow_html=True)
@st.cache_data
def logo_data_uri(path, mtime):
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp", "svg": "image/svg+xml"}.get(ext, "image/png")
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")
def logo_uri():
    path = find_logo()
    return logo_data_uri(path, os.path.getmtime(path)) if path else None
def icon_uri():
    path = find_icon()
    return logo_data_uri(path, os.path.getmtime(path)) if path else None
_WAVE_PATH = ("M0 30 Q 150 0 300 30 T 600 30 T 900 30 T 1200 30 T 1500 30 T 1800 30 T 2100 30 T 2400 30 V60 H0 Z")
WAVES_HTML = (f'<div class="waves">'
              f'<svg class="wave w1" viewBox="0 0 2400 60" preserveAspectRatio="none"><path d="{_WAVE_PATH}"/></svg>'
              f'<svg class="wave w2" viewBox="0 0 2400 60" preserveAspectRatio="none"><path d="{_WAVE_PATH}"/></svg>'
              f'<svg class="wave w3" viewBox="0 0 2400 60" preserveAspectRatio="none"><path d="{_WAVE_PATH}"/></svg>'
              f'</div>')
def section(title, subtitle=""):
    st.markdown(html(f'<div class="sec-title">{title}</div><div class="sec-sub">{subtitle}</div>'),
                unsafe_allow_html=True)
def card():
    try:
        return st.container(border=True)
    except TypeError:                       # older Streamlit without bordered containers
        return st.container()
def kpi_card(label, value, sub="", color=TEAL):
    return html(f'<div class="kpi" style="border-top:4px solid {color}">'
                f'<div class="kpi-top">{label}</div><div class="kpi-value">{value}</div>'
                f'<div class="kpi-sub">{sub}</div></div>')
# -------------------------------------------------
# DATA HELPERS
# -------------------------------------------------
@st.cache_data
def _load_raster_cached(path, mtime):
    # mtime is part of the cache key, so the dashboard reloads a layer after the pipeline rewrites it
    with rasterio.open(path) as src:
        data = src.read(1).astype("float64")
        data[data < -1000] = np.nan
    return data
def load_raster(path):
    return _load_raster_cached(path, os.path.getmtime(path))
def calculate_area_stats(classified, valid_mask, pixel_area_km2=PIXEL_AREA_KM2):
    stats = {}
    total_pixels = np.sum(~np.isnan(classified) & valid_mask)
    for cls, name in enumerate(RISK_NAMES, start=1):
        pixels = np.sum((classified == cls) & valid_mask)
        pct = (pixels / total_pixels) * 100 if total_pixels > 0 else 0
        stats[name] = {"area": pixels * pixel_area_km2, "pct": pct}
    return stats
def read_chirps_status():
    info = {}
    try:
        with open(CHIRPS_STATUS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    info[key] = value
    except Exception:
        pass
    return info
def current_season_label():
    try:
        from chirps_utils import latest_chirps_file
        path = latest_chirps_file(RAW_DIR, clipped=True)
        years = re.findall(r"(?<!\d)(20\d{2})(?!\d)", os.path.basename(path))
        if len(years) >= 2:
            return f"{years[0]}-{years[-1]}"
    except Exception:
        pass
    return "latest season"
def class_cards(stats):
    cards = "".join(
        f'<div class="cls" style="border-left:6px solid {c}"><div class="cls-name">{n}</div>'
        f'<div class="cls-val">{stats[n]["area"]:,.0f} km²</div>'
        f'<div class="cls-pct">{stats[n]["pct"]:.1f}% of catchment</div></div>'
        for n, c in zip(RISK_NAMES, RISK_COLORS))
    bar = "".join(f'<div style="width:{stats[n]["pct"]:.2f}%;background:{c}"></div>'
                  for n, c in zip(RISK_NAMES, RISK_COLORS))
    return html(f'<div class="cls-grid">{cards}</div><div class="stackbar">{bar}</div>')
def make_map(data, valid_mask, cmap, vmin=None, vmax=None, label="", ticks=None, ticklabels=None):
    """Cropped, clean map with colour bar, scale bar and north arrow."""
    rows, cols = np.where(valid_mask)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    shown = np.where(valid_mask, data, np.nan)[r0:r1, c0:c1]
    h, w = shown.shape
    fig, ax = plt.subplots(figsize=(9, 7.4), dpi=110)
    fig.patch.set_facecolor("white")
    im = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, shrink=0.62, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9, colors="#374151")
    if ticks is not None:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    if label:
        cbar.set_label(label, fontsize=10, color="#374151")
    km = 10 if w > 900 else 5
    length = km * 1000 / PIXEL_SIZE_M
    x0, y0 = 0.04 * w, 0.97 * h
    ax.plot([x0, x0 + length], [y0, y0], color="#1F2937", lw=3, solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 - 0.015 * h, f"{km} km", ha="center", va="bottom", fontsize=9, color="#1F2937")
    ax.annotate("N", xy=(0.95, 0.97), xytext=(0.95, 0.88), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#1F2937", lw=1.6),
                ha="center", va="center", fontsize=12, fontweight="bold", color="#1F2937")
    fig.tight_layout()
    return fig
# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
SEASON = current_season_label()
RAIN_LABEL = f"CHIRPS Rainfall ({SEASON})"
try:
    with st.spinner("Loading spatial layers..."):
        dem = load_raster(PATHS["dem"])
        catchment = load_raster(PATHS["catchment"])
        rainfall = load_raster(PATHS["rainfall"])
        rain_hazard = load_raster(PATHS["rain_hazard"])
        hazard_class = load_raster(PATHS["hazard_class"])
        risk_class = load_raster(PATHS["risk_class"])
        exposure = load_raster(PATHS["exposure"])
except FileNotFoundError as ex:
    st.error(f"A required map layer is missing: {ex}. Run the daily pipeline "
             "(02_daily_pipeline.py) once to create it.")
    st.stop()
mask = catchment == 1
# -------------------------------------------------
# STATUS INFORMATION
# -------------------------------------------------
updated = datetime.fromtimestamp(os.path.getmtime(PATHS["risk_class"]))
chirps_info = read_chirps_status()
chirps_status = chirps_info.get("status", "UNKNOWN")
if chirps_status == "UP_TO_DATE":
    rain_pill = ("Rainfall data: up to date", "ok")
elif chirps_status in ("ONE_SEASON_BEHIND", "OUTDATED"):
    rain_pill = (f"Rainfall data: {chirps_status.replace('_', ' ').lower()} "
                 f"(ends {chirps_info.get('data_end', '?')})", "warn" if chirps_status == "ONE_SEASON_BEHIND" else "bad")
else:
    rain_pill = (f"Rainfall data: {chirps_status.replace('_', ' ').lower()}", "")
uri = logo_uri()
logo_block = (f'<div class="logo-plate"><img src="{uri}" alt="logo"></div>' if uri
              else '<div class="logo-plate"><div class="badge">DUWE</div></div>')
# -------------------------------------------------
# HERO HEADER
# -------------------------------------------------
st.markdown(html(f"""
<div class="hero">
<div class="hero-left">{logo_block}
<div>
<div class="hero-title">DUWE Flood Intelligence Engine</div>
<div class="hero-sub">Chikwawa Catchment · Southern Malawi · Automated geospatial intelligence</div>
<div class="hero-credit">
<span class="ashley-pulse"></span>
Dashboard developed by <span class="ashley-anim">{DEVELOPER}</span>
</div>
</div></div>
<div class="pills">
<span class="pill">Risk map updated {updated.strftime('%d %b %Y, %H:%M')}</span>
<span class="pill {rain_pill[1]}">{rain_pill[0]}</span>
</div>{WAVES_HTML}</div>"""), unsafe_allow_html=True)
if LIVE_RAIN_AVAILABLE:
    render_live_banner()          # raining now? heavy-rain warnings
# -------------------------------------------------
# KEY INDICATORS
# -------------------------------------------------
catchment_area = np.sum(mask) * PIXEL_AREA_KM2
high_risk_area = np.sum((risk_class >= 3) & mask) * PIXEL_AREA_KM2
mean_rain = np.nanmean(rainfall[mask])
max_rain = np.nanmax(rainfall[mask])
high_pct = f"{(high_risk_area / catchment_area) * 100:.1f}% of the catchment" if catchment_area else ""
k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi_card("Catchment area", f"{catchment_area:,.0f} km²", "Study area analysed", BLUE), unsafe_allow_html=True)
k2.markdown(kpi_card("High + Very High risk", f"{high_risk_area:,.0f} km²", high_pct, "#D64545"), unsafe_allow_html=True)
k3.markdown(kpi_card("Mean seasonal rainfall", f"{mean_rain:,.0f} mm", f"Rainy season {SEASON}", TEAL), unsafe_allow_html=True)
k4.markdown(kpi_card("Max seasonal rainfall", f"{max_rain:,.0f} mm", f"Rainy season {SEASON}", "#2E86C1"), unsafe_allow_html=True)
st.write("")
# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
side_uri = icon_uri() or uri
side_logo = f'<img src="{side_uri}" alt="logo">' if side_uri else ""
st.sidebar.markdown(html(f"""
<div class="side-brand">{side_logo}<div><div class="side-name">DUWE Disaster<br>Intelligence</div>
<div class="side-sub">Flood Intelligence Engine</div></div></div>"""), unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.subheader("Map selection")
map_option = st.sidebar.selectbox(
    "Choose layer to display",
    ["Classified Risk", "Rainfall-Enhanced Hazard (Classified)", RAIN_LABEL, "Exposure Proxy", "Digital Elevation Model"]
)
with st.sidebar.expander("About this dashboard"):
    st.markdown(
        "**Method.** Rainfall-enhanced hazard combines seasonal rainfall (CHIRPS, 40%), a terrain flood "
        "proxy (35%) and flow accumulation (25%). Risk = hazard × exposure. Classes are cut at the 30th, "
        "60th and 85th percentile, so they show *relative* risk inside the catchment.\n\n"
        "**Sources.** CHIRPS (Climate Hazards Center), OpenStreetMap contributors, Open-Meteo.\n\n"
        "**Use.** A decision-support tool that complements Malawi's national institutions: DCCMS for "
        "weather and climate information and DoDMA for disaster management. It shows where and who is "
        "most exposed so that preparedness and response can be targeted. Official forecasts and "
        "warnings are issued by DCCMS and DoDMA."
    )
    st.markdown(f"**Developed by** {DEVELOPER}")
    if not uri:
        st.caption("Logo not found. Save it as 04_assets\\logo.png to show it here.")
st.sidebar.caption(f"Dashboard developed by {DEVELOPER}")
# -------------------------------------------------
# TABS
# -------------------------------------------------
tab_maps, tab_flow, tab_sim, tab_live, tab_infra = st.tabs(
    ["Risk maps", "Water flow", "Flood simulation", "Live rainfall", "Infrastructure at risk"])
# ---------------- TAB 1: MAPS ----------------
with tab_maps:
    st.markdown(html('<div class="sci-badge">Geospatial Layer</div>'), unsafe_allow_html=True)
    section(map_option, "Chikwawa catchment, 30 m resolution")
    stats = None
    summary_unit = None
    if map_option == "Classified Risk":
        fig = make_map(risk_class, mask, ListedColormap(RISK_COLORS), 1, 4, "Risk class",
                       [1.375, 2.125, 2.875, 3.625], RISK_NAMES)
        stats = calculate_area_stats(risk_class, mask)
        slug = "classified_risk"
    elif map_option == "Rainfall-Enhanced Hazard (Classified)":
        fig = make_map(hazard_class, mask, ListedColormap(RISK_COLORS), 1, 4, "Hazard class",
                       [1.375, 2.125, 2.875, 3.625], RISK_NAMES)
        stats = calculate_area_stats(hazard_class, mask)
        slug = "rainfall_hazard"
    elif map_option == RAIN_LABEL:
        fig = make_map(rainfall, mask, "YlGnBu", label="Seasonal rainfall (mm)")
        summary_unit, summary_data, slug = "mm", rainfall, "rainfall"
    elif map_option == "Exposure Proxy":
        fig = make_map(exposure, mask, "YlOrRd", label="Exposure index")
        summary_unit, summary_data, slug = "", exposure, "exposure"
    else:
        fig = make_map(dem, mask, "terrain", label="Elevation (m)")
        summary_unit, summary_data, slug = "m", dem, "elevation"
    with card():
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
        st.download_button("Download this map (PNG)", buf.getvalue(),
                           file_name=f"duwe_{slug}.png", mime="image/png")
    plt.close(fig)
    if stats:
        section("Area statistics", "Share of the catchment in each class")
        st.markdown(class_cards(stats), unsafe_allow_html=True)
        st.markdown(html('<div class="note">Classes are relative: they are cut at the 30th, 60th and 85th '
                         'percentile of the current map, so the shares are always about 30 / 30 / 25 / 15%. '
                         'They show where risk is higher inside the catchment, not how severe a flood will be.</div>'),
                    unsafe_allow_html=True)
    else:
        vals = summary_data[mask & np.isfinite(summary_data)]
        u = f" {summary_unit}" if summary_unit else ""
        section("Layer summary", "Within the catchment")
        s1, s2, s3 = st.columns(3)
        s1.markdown(kpi_card("Minimum", f"{vals.min():,.1f}{u}", "", "#2E9E6B"), unsafe_allow_html=True)
        s2.markdown(kpi_card("Average", f"{vals.mean():,.1f}{u}", "", TEAL), unsafe_allow_html=True)
        s3.markdown(kpi_card("Maximum", f"{vals.max():,.1f}{u}", "", "#D64545"), unsafe_allow_html=True)
# ---------------- TAB: WATER FLOW ----------------
with tab_flow:
    st.markdown(html('<div class="sci-badge">Hydrological Flow</div>'), unsafe_allow_html=True)
    st.markdown(html('<div class="flow-accent"></div>'), unsafe_allow_html=True)
    if FLOW_AVAILABLE:
        render_flow_animation()
    else:
        st.info("The water-flow animation is not installed yet. Save flow_animation.py "
                "in the same folder as this dashboard (02_scripts).")
# ---------------- TAB: FLOOD SIMULATION ----------------
with tab_sim:
    st.markdown(html('<div class="sci-badge">Dynamic Flood Simulation</div>'), unsafe_allow_html=True)
    st.markdown(html('<div class="flow-accent"></div>'), unsafe_allow_html=True)
    if SIM_AVAILABLE:
        render_flood_simulation()
    else:
        st.info("The flood simulation panel is not installed yet. Save flood_simulation_panel.py "
                "in the same folder as this dashboard (02_scripts).")
# ---------------- TAB: LIVE RAINFALL ----------------
with tab_live:
    st.markdown(html('<div class="sci-badge">Real-time Monitoring</div>'), unsafe_allow_html=True)
    if LIVE_RAIN_AVAILABLE:
        render_live_rainfall(high_risk_area)
    else:
        st.info("The live rainfall panel is not installed yet. Save live_rainfall.py "
                "in the same folder as this dashboard (02_scripts).")
# ---------------- TAB 3: INFRASTRUCTURE ----------------
with tab_infra:
    st.markdown(html('<div class="sci-badge">Exposure Analysis</div>'), unsafe_allow_html=True)
    try:
        from infrastructure_panel import render_infrastructure_panel
    except ImportError:
        st.info("The infrastructure panel is not installed yet. Save infrastructure_panel.py "
                "in the same folder as this dashboard (02_scripts).")
    else:
        render_infrastructure_panel()
# -------------------------------------------------
# FOOTER
# -------------------------------------------------
foot_logo = f'<img src="{uri}" alt="logo">' if uri else ""
st.markdown(html(f"""
<div class="water-strip"></div>
<div class="foot">{foot_logo}<b>DUWE DISASTER INTELLIGENCE</b> | DUWE Flood Intelligence Engine<br>
Chikwawa Catchment • Southern Malawi • Automated Geospatial Intelligence<br>
<span class="ashley-pulse" style="display:inline-block;vertical-align:middle;margin-right:6px;"></span>
<b>Dashboard developed by <span class="ashley-anim">{DEVELOPER}</span></b></div>"""), unsafe_allow_html=True)