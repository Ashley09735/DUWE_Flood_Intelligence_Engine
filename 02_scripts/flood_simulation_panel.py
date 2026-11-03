"""
DUWE AI FLOOD INTELLIGENCE ENGINE
Dashboard panel: results of the 2D hydraulic flood simulation.

Save in 02_scripts, next to duwe_flood_dashboard.py. It shows runs made with
automation\\05_hydraulic_simulation.py. The simulation itself is NOT run from the dashboard
because a full run takes minutes to hours.
"""

import io
import os
import json

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colormaps
import streamlit as st

BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
OUT_ROOT = os.path.join(BASE_DIR, r"01_data\processed\hydraulic")

LAND = "#EEF2F7"
DEPTH_EDGES = [0.1, 0.5, 1.0, 2.0]
DEPTH_COLORS = ["#BFE3F5", "#5DADE2", "#1F6FB2", "#0B2545"]
DEPTH_LABELS = ["0.1 - 0.5 m", "0.5 - 1 m", "1 - 2 m", "more than 2 m"]
HAZARD_EDGES = [0.0, 0.75, 1.25, 2.5]
HAZARD_COLORS = ["#F7E08A", "#F2994A", "#D64545", "#7B1F2B"]
HAZARD_LABELS = ["Low (under 0.75)", "Danger for some (0.75 - 1.25)", "Danger for most (1.25 - 2.5)", "Danger for all (over 2.5)"]

LAYERS = {
    "Maximum flood depth": ("max_depth.tif", "classes", "depth"),
    "Flood hazard to people": ("max_hazard.tif", "classes", "hazard"),
    "Arrival time of flooding (hours)": ("arrival_h.tif", "continuous", "viridis_r"),
    "Flood duration (hours above 0.1 m)": ("wet_hours.tif", "continuous", "YlGnBu"),
    "Maximum flow speed (m/s)": ("max_speed.tif", "continuous", "magma_r"),
}


def _hex(c):
    return [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]


def list_runs():
    runs = []
    if os.path.isdir(OUT_ROOT):
        for name in os.listdir(OUT_ROOT):
            path = os.path.join(OUT_ROOT, name, "summary.json")
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        runs.append((name, json.load(f)))
                except Exception:
                    pass
    runs.sort(key=lambda r: r[1].get("created", ""), reverse=True)
    return runs


@st.cache_data(show_spinner=False)
def _read_tif(path, mtime):
    with rasterio.open(path) as s:
        a = s.read(1).astype("float32")
    return np.where(a <= -9990, np.nan, a)


@st.cache_data(show_spinner=False)
def _read_snapshots(path, mtime):
    z = np.load(path)
    return z["depth"].astype("float32"), z["hours"], int(z["pool"])


def _read(run, fname):
    p = os.path.join(OUT_ROOT, run, fname)
    return _read_tif(p, os.path.getmtime(p)) if os.path.exists(p) else None


def _crop(*arrays):
    inside = np.isfinite(arrays[0])
    rows, cols = np.where(inside)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    return [a[r0:r1, c0:c1] for a in arrays]


def _classes_rgba(values, edges, colors):
    rgba = np.zeros(values.shape + (4,), dtype="float32")
    inside = np.isfinite(values)
    rgba[inside] = _hex(LAND) + [1.0]
    idx = np.digitize(values, edges) - 1
    for k, c in enumerate(colors):
        sel = inside & (idx == k)
        rgba[sel] = _hex(c) + [1.0]
    rgba[inside & (values < edges[0])] = _hex(LAND) + [1.0]
    return rgba


def _figure(run, layer):
    fname, kind, spec = LAYERS[layer]
    data = _read(run, fname)
    depth = _read(run, "max_depth.tif")
    if data is None:
        return None
    model = np.where(np.isfinite(depth if depth is not None else data), 1.0, np.nan)   # cells inside the model
    if layer.startswith("Maximum flow speed") and depth is not None:
        data = np.where(depth >= 0.1, data, np.nan)                                       # speed only where there is flood water
    model, data = _crop(model, data)
    inside = np.isfinite(model)

    fig, ax = plt.subplots(figsize=(9, 7.4), dpi=110)
    fig.patch.set_facecolor("white")
    if kind == "classes":
        edges, colors, labels = (DEPTH_EDGES, DEPTH_COLORS, DEPTH_LABELS) if spec == "depth" else (HAZARD_EDGES, HAZARD_COLORS, HAZARD_LABELS)
        ax.imshow(_classes_rgba(np.where(inside, np.nan_to_num(data, nan=0.0), np.nan), edges, colors), interpolation="nearest")
        ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(colors, labels)], loc="lower right",
                  frameon=True, framealpha=0.95, fontsize=9, title=layer)
    else:
        base = np.zeros(data.shape + (4,), dtype="float32")
        base[inside] = _hex(LAND) + [1.0]
        ax.imshow(base, interpolation="nearest")
        shown = np.where(data > 0, data, np.nan) if ("duration" in layer or "speed" in layer) else data
        vmax = np.nanpercentile(shown, 98) if np.isfinite(shown).any() else 1.0
        im = ax.imshow(shown, cmap=colormaps[spec], vmin=0, vmax=max(float(vmax), 1e-6), interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, shrink=0.62, pad=0.02)
        cb.outline.set_visible(False)
        cb.set_label(layer, fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def _snapshot_figure(depth_grid, hour):
    d, = _crop(depth_grid)
    fig, ax = plt.subplots(figsize=(9, 7.4), dpi=110)
    fig.patch.set_facecolor("white")
    ax.imshow(_classes_rgba(d, DEPTH_EDGES, DEPTH_COLORS), interpolation="nearest")
    ax.set_title(f"Water depth at hour {hour:.0f}", fontsize=12, fontweight="bold")
    ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(DEPTH_COLORS, DEPTH_LABELS)],
              loc="lower right", frameon=True, framealpha=0.95, fontsize=9)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def render_flood_simulation():
    st.header("Flood simulation")
    st.warning(
        "A screening-level scenario model: not a forecast, and not yet calibrated or validated against "
        "recorded floods. It simulates rain falling on the whole catchment; river inflow is included only "
        "if one was supplied, and the elevation model has no river-channel depths. Use it to compare "
        "scenarios and find places worth a closer look, and follow official forecasts and warnings from "
        "DCCMS and DoDMA."
    )

    runs = list_runs()
    if not runs:
        st.info("No simulation results yet. Run one from PowerShell, then reload this page:")
        st.code('cd "C:\\Users\\User\\Desktop\\DUWE_DISASTER_INTELLIGENCE\\02_scripts\\automation"\n'
                "python 05_hydraulic_simulation.py --scenario storm_100mm_24h --hours 2 --resolution 120   # quick test\n"
                "python 05_hydraulic_simulation.py --scenario storm_100mm_24h                              # full run",
                language="powershell")
        return

    labels = {n: f"{n}  |  {s.get('scenario', '')}  |  {s.get('created', '')[:16].replace('T', ' ')}" for n, s in runs}
    run = st.selectbox("Simulation run", [n for n, _ in runs], format_func=lambda n: labels[n], key="hyd_run")
    s = dict(runs)[run]

    assets = s.get("assets", {})
    school = assets.get("schools", {})
    roads = assets.get("roads", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flooded area (0.1 m or more)", f"{s.get('flooded_ge_0.1m_km2', 0):,.1f} km²",
              f"{s.get('flooded_ge_0.1m_percent', 0):.1f}% of the model", delta_color="off")
    c2.metric("Deepest water", f"{s.get('max_depth_m', 0):.1f} m", f"fastest flow {s.get('max_speed_ms', 0):.1f} m/s", delta_color="off")
    c3.metric("Schools flooded", f"{school.get('depth_ge_0.1m', 0):,}",
              f"of {school.get('in_model', 0):,} in the model", delta_color="off")
    c4.metric("Road under 0.3 m or more of water", f"{roads.get('depth_ge_0.3m', 0):,.1f} km",
              f"of {roads.get('in_model', 0):,.1f} km", delta_color="off")
    st.caption(f"Rain total {s.get('rain_total_mm', 0):,.0f} mm | simulated {s.get('simulated_hours', 0):,.0f} h | "
               f"{s.get('resolution_m', '?')} m cells | run took {s.get('runtime_min', '?')} min")

    tab_max, tab_time, tab_assets = st.tabs(["Maps", "Flood over time", "Assets affected"])

    with tab_max:
        layer = st.selectbox("Layer", list(LAYERS), key="hyd_layer")
        fig = _figure(run, layer)
        if fig is None:
            st.info("This layer is not available for the selected run.")
        else:
            with st.container(border=True):
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
                st.download_button("Download this map (PNG)", buf.getvalue(), file_name=f"{run}_{LAYERS[layer][0][:-4]}.png", mime="image/png")
            plt.close(fig)
        fl = s.get("flooded_area_km2", {})
        if fl:
            st.subheader("Flooded area by depth")
            st.bar_chart(pd.DataFrame({"km²": list(fl.values())}, index=list(fl.keys())))

    with tab_time:
        snap_path = os.path.join(OUT_ROOT, run, "snapshots.npz")
        if not os.path.exists(snap_path):
            st.info("No time snapshots were saved for this run.")
        else:
            depth_t, hours, pool = _read_snapshots(snap_path, os.path.getmtime(snap_path))
            options = [f"{h:.1f} h" for h in hours]
            choice = st.select_slider("Time since the start of the simulation", options=options,
                                      value=options[len(options) // 2], key="hyd_step")
            k = options.index(choice)
            fig = _snapshot_figure(depth_t[k], hours[k])
            with st.container(border=True):
                st.pyplot(fig)
            plt.close(fig)
            wet = (depth_t >= 0.1).sum(axis=(1, 2)) * (s.get("resolution_m", 60) * pool) ** 2 / 1e6
            st.subheader("Flooded area over time")
            st.line_chart(pd.DataFrame({"km² with 0.1 m or more of water": wet}, index=np.round(hours, 1)))
            if pool > 1:
                st.caption(f"Time snapshots are shown at a coarser resolution ({pool}x) to keep them small.")

    with tab_assets:
        rows = []
        for cat, v in assets.items():
            if v.get("unit") == "count":
                rows.append({"Asset": cat, "In model": v["in_model"], "Flooded (0.1 m+)": v["depth_ge_0.1m"],
                             "Deep (0.5 m+)": v["depth_ge_0.5m"], "Very deep (1 m+)": v["depth_ge_1m"]})
            else:
                rows.append({"Asset": f"{cat} (km)", "In model": v["in_model"], "Flooded (0.1 m+)": v["depth_ge_0.1m"],
                             "Deep (0.5 m+)": v["depth_ge_0.5m"], "Very deep (1 m+)": None})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        else:
            st.info("No infrastructure data was available for this run. Run 03_download_infrastructure.py first.")
        path = os.path.join(OUT_ROOT, run, "asset_flooding.csv")
        if os.path.exists(path):
            listed = pd.read_csv(path, encoding="utf-8-sig")
            st.subheader("Named facilities reached by water")
            st.dataframe(listed, hide_index=True)
            st.download_button("Download this list (CSV)", listed.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{run}_flooded_assets.csv", mime="text/csv")

    with st.expander("Model settings and water balance"):
        p = s.get("parameters", {})
        mb = s.get("mass_balance_m3", {})
        st.markdown(
            f"- Manning roughness: **{p.get('manning_n')}**\n"
            f"- Initial loss: **{p.get('initial_loss_mm')} mm**, infiltration **{p.get('infiltration_mm_h')} mm/h**, "
            f"soil storage **{p.get('soil_storage_mm')} mm**\n"
            f"- Spatial rain pattern from CHIRPS: **{p.get('chirps_pattern')}**\n"
            f"- Water balance error: **{mb.get('balance_error_percent', 'n/a')}%** (should be close to zero)\n"
            "- These loss and roughness values are placeholders. They need calibrating against recorded floods "
            "before the numbers are used for decisions."
        )
