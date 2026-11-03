"""
DUWE AI FLOOD INTELLIGENCE ENGINE
Water-flow animation tab.

Traces the catchment's drainage network from the elevation model and the flow-accumulation
map, then animates water particles moving downstream along it. It shows MODELLED drainage
paths (where water tends to flow), not live river levels, rainfall or flooding.

Save in 02_scripts, next to duwe_flood_dashboard.py. Works offline.
The first time, the network is built (usually well under a minute); it is then saved and reused.
"""

import os
import io
import json
import base64
from datetime import datetime

import numpy as np
import rasterio
import streamlit as st

BASE_DIR = r"C:\Users\User\Desktop\DUWE_DISASTER_INTELLIGENCE"
DEM_PATH = os.path.join(BASE_DIR, r"01_data\processed\dem_conditioned_duwe.tif")
ACC_PATH = os.path.join(BASE_DIR, r"01_data\processed\flow_accumulation_duwe.tif")
CATCHMENT_PATH = os.path.join(BASE_DIR, r"01_data\processed\catchment_duwe.tif")
RISK_PATH = os.path.join(BASE_DIR, r"01_data\processed\risk_updated_classified_duwe.tif")
NETWORK_PATH = os.path.join(BASE_DIR, r"01_data\processed\flow_network_duwe.json")

STRIDE = 6               # animation cells are 6 x 30 m = 180 m
CHANNEL_SHARE = 0.05     # the busiest 5% of cells are drawn as rivers (raise for a denser network)
RISK_COLORS = ["#2E9E6B", "#F2C94C", "#F2994A", "#D64545"]

# 8 neighbours (row step, column step)
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _paths():
    """Read the paths when called, so they can be changed after import (and in tests)."""
    return {"dem": DEM_PATH, "acc": ACC_PATH, "catchment": CATCHMENT_PATH,
            "risk": RISK_PATH, "network": NETWORK_PATH}


def _mtimes(p):
    return {k: round(os.path.getmtime(p[k]), 1) for k in ("dem", "acc", "catchment") if os.path.exists(p[k])}


# -------------------------------------------------
# DRAINAGE NETWORK
# -------------------------------------------------
def d8_directions(dem, chunk=600):
    """For each cell, which of the 8 neighbours is steepest downhill (index 0-7), or -1 if none."""
    h, w = dem.shape
    dirs = np.full((h, w), -1, dtype=np.int8)
    pad = np.pad(dem, 1, constant_values=np.nan)
    with np.errstate(invalid="ignore"):
        for r0 in range(0, h, chunk):
            r1 = min(h, r0 + chunk)
            centre = dem[r0:r1]
            best = np.zeros(centre.shape, dtype=np.float32)
            d = dirs[r0:r1]
            for k, (dr, dc) in enumerate(OFFSETS):
                nb = pad[1 + r0 + dr:1 + r1 + dr, 1 + dc:1 + w + dc]
                slope = (centre - nb) / np.float32(np.hypot(dr, dc))
                better = slope > best                 # NaN compares False, so the outside is ignored
                best[better] = slope[better]
                d[better] = k
    return dirs


def _trace_downstream(r, c, dirs, stride, index_map, own, max_steps):
    h, w = dirs.shape
    bh, bw = index_map.shape
    for _ in range(max_steps):
        k = dirs[r, c]
        if k < 0:
            return -1
        dr, dc = OFFSETS[k]
        r, c = r + dr, c + dc
        if r < 0 or c < 0 or r >= h or c >= w:
            return -1
        br, bc = r // stride, c // stride
        if (br, bc) == own or br >= bh or bc >= bw:
            continue
        j = index_map[br, bc]
        if j >= 0:
            return int(j)
    return -1


def build_network(paths=None):
    """Trace the river network and save it. Returns the network dict."""
    p = paths or _paths()
    with rasterio.open(p["dem"]) as s:
        dem = s.read(1).astype("float32")
    with rasterio.open(p["acc"]) as s:
        acc = s.read(1).astype("float32")
    with rasterio.open(p["catchment"]) as s:
        catch = s.read(1)

    valid = (catch == 1) & np.isfinite(dem) & (dem > -1000)
    dem = np.where(valid, dem, np.nan).astype("float32")
    acc = np.where(valid & np.isfinite(acc) & (acc > 0), acc, 0).astype("float32")

    dirs = d8_directions(dem)

    S = STRIDE
    h, w = dem.shape
    bh, bw = h // S, w // S
    blocks = acc[:bh * S, :bw * S].reshape(bh, S, bw, S).swapaxes(1, 2).reshape(bh, bw, S * S)
    idx = blocks.argmax(axis=2)
    bmax = np.take_along_axis(blocks, idx[..., None], axis=2)[..., 0]
    del blocks

    positive = bmax[bmax > 0]
    if positive.size == 0:
        raise ValueError("No flow accumulation found inside the catchment.")
    thr = np.quantile(positive, 1 - CHANNEL_SHARE)
    channel = (bmax >= thr) & (bmax > 0)

    ch_r, ch_c = np.nonzero(channel)
    n = len(ch_r)
    index_map = -np.ones((bh, bw), dtype=np.int32)
    index_map[ch_r, ch_c] = np.arange(n, dtype=np.int32)

    start_r = ch_r * S + idx[ch_r, ch_c] // S
    start_c = ch_c * S + idx[ch_r, ch_c] % S

    down = [_trace_downstream(int(start_r[i]), int(start_c[i]), dirs, S, index_map,
                              (int(ch_r[i]), int(ch_c[i])), S * 8) for i in range(n)]

    logs = np.log10(bmax[ch_r, ch_c])
    span = max(float(logs.max() - logs.min()), 1e-6)
    weight = np.round(100 * (logs - logs.min()) / span).astype(int)

    net = {
        "w": int(bw), "h": int(bh), "stride": S,
        "x": ch_c.astype(int).tolist(), "y": ch_r.astype(int).tolist(),
        "down": down, "wt": weight.tolist(),
        "built": datetime.now().isoformat(timespec="seconds"),
        "mtimes": _mtimes(p),
    }
    os.makedirs(os.path.dirname(p["network"]), exist_ok=True)
    with open(p["network"], "w", encoding="utf-8") as f:
        json.dump(net, f)
    return net


def load_network(paths=None):
    """Saved network, or None if it is missing or the elevation/flow maps changed since."""
    p = paths or _paths()
    try:
        with open(p["network"], "r", encoding="utf-8") as f:
            net = json.load(f)
    except Exception:
        return None
    return net if net.get("mtimes") == _mtimes(p) else None


# -------------------------------------------------
# BACKGROUND IMAGE
# -------------------------------------------------
def _cmap(name):
    try:
        from matplotlib import colormaps
        return colormaps[name]
    except Exception:
        from matplotlib import cm
        return cm.get_cmap(name)


def _read_sampled(path, bh, bw, stride):
    with rasterio.open(path) as s:
        a = s.read(1).astype("float32")
    a = a[stride // 2::stride, stride // 2::stride][:bh, :bw]
    a = np.where(a < -1000, np.nan, a)
    return a


@st.cache_data(show_spinner=False)
def _background_cached(kind, bh, bw, stride, key, dem_path, risk_path, catchment_path):
    from PIL import Image
    catch = _read_sampled(catchment_path, bh, bw, stride)
    inside = np.nan_to_num(catch, nan=0) == 1
    rgba = np.zeros((bh, bw, 4), dtype="float32")

    if kind == "Risk classes" and os.path.exists(risk_path):
        cls = np.nan_to_num(_read_sampled(risk_path, bh, bw, stride), nan=0)
        for k, hexcol in enumerate(RISK_COLORS, start=1):
            rgb = np.array([int(hexcol[i:i + 2], 16) for i in (1, 3, 5)], dtype="float32")
            rgba[cls == k, :3] = rgb
        shown = inside & (cls >= 1)
        rgba[..., :3] = rgba[..., :3] * 0.72 + 255 * 0.28
        rgba[shown, 3] = 235
    elif kind == "Elevation" and os.path.exists(dem_path):
        z = _read_sampled(dem_path, bh, bw, stride)
        ok = inside & np.isfinite(z)
        lo, hi = np.nanpercentile(z[ok], [2, 98]) if ok.any() else (0, 1)
        norm = np.clip((np.nan_to_num(z, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
        col = _cmap("terrain")(norm)[..., :3] * 255
        rgba[..., :3] = col * 0.65 + 255 * 0.35
        rgba[ok, 3] = 235
    else:                                         # plain
        rgba[..., :3] = (232, 238, 246)
        rgba[inside, 3] = 255

    buf = io.BytesIO()
    Image.fromarray(rgba.astype("uint8"), "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def background_image(kind, net, paths=None):
    p = paths or _paths()
    key = (os.path.getmtime(p["risk"]) if os.path.exists(p["risk"]) else 0,)
    return _background_cached(kind, net["h"], net["w"], net["stride"], key, p["dem"], p["risk"], p["catchment"])


# -------------------------------------------------
# HTML / JAVASCRIPT ANIMATION
# -------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
html,body{margin:0;background:transparent;font-family:'Segoe UI',system-ui,sans-serif;}
.bar{display:flex;gap:18px;align-items:center;flex-wrap:wrap;padding:6px 4px 10px 4px;color:#374151;font-size:13px;font-weight:600;}
button{background:#134074;color:#fff;border:0;border-radius:8px;padding:7px 16px;font-weight:700;cursor:pointer;}
button:hover{background:#1F7A9C;}
input[type=range]{vertical-align:middle;width:130px;}
.stage{position:relative;display:block;margin:0 auto;width:fit-content;max-width:100%;background:#fff;border:1px solid #E3E9F2;border-radius:14px;overflow:hidden;box-shadow:0 2px 10px rgba(16,42,67,.08);}
canvas{display:block;max-width:100%;max-height:640px;width:auto;height:auto;}
canvas.layer{position:absolute;left:0;top:0;width:100%;height:100%;}
.msg{padding:30px;text-align:center;color:#6B7A90;}
</style></head><body>
<div class="bar">
<button id="pp">Pause</button>
<label>Speed <input id="sp" type="range" min="0.2" max="3" step="0.1" value="1"></label>
<label>Water <input id="dn" type="range" min="200" max="4000" step="100" value="1500"></label>
</div>
<div class="stage"><canvas id="base"></canvas><canvas id="fx" class="layer"></canvas></div>
<script>
const D = __DATA__;
const W = D.w, H = D.h, S = 2.2;
const X = D.x, Y = D.y, DOWN = D.down, WT = D.wt, M = X.length;
const base = document.getElementById('base'), fx = document.getElementById('fx');
if (M === 0) {
  document.querySelector('.stage').innerHTML = '<div class="msg">No drainage network found.</div>';
} else {
  base.width = fx.width = Math.round(W * S);
  base.height = fx.height = Math.round(H * S);
  const bctx = base.getContext('2d'), fctx = fx.getContext('2d');

  function drawBase(img) {
    bctx.clearRect(0, 0, base.width, base.height);
    if (img) { bctx.imageSmoothingEnabled = true; bctx.drawImage(img, 0, 0, base.width, base.height); }
    bctx.lineCap = 'round';
    for (let i = 0; i < M; i++) {
      const j = DOWN[i];
      if (j < 0) continue;
      bctx.strokeStyle = 'rgba(19,64,116,' + (0.22 + 0.38 * WT[i] / 100) + ')';
      bctx.lineWidth = (0.5 + 1.9 * WT[i] / 100) * S * 0.6;
      bctx.beginPath();
      bctx.moveTo((X[i] + 0.5) * S, (Y[i] + 0.5) * S);
      bctx.lineTo((X[j] + 0.5) * S, (Y[j] + 0.5) * S);
      bctx.stroke();
    }
  }
  drawBase(null);
  const img = new Image();
  img.onload = function () { drawBase(img); };
  img.src = D.bg;

  const parts = [];
  let target = D.n, running = true, speed = 1, last = performance.now();
  const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function spawn(p) {
    p.i = Math.floor(Math.random() * M);
    p.t = Math.random();
    p.v = 0.07 + Math.random() * 0.07;
    p.life = 150 + Math.random() * 350;
    return p;
  }
  function pos(i, t) {
    const j = DOWN[i];
    return [(X[i] + 0.5 + (X[j] - X[i]) * t) * S, (Y[i] + 0.5 + (Y[j] - Y[i]) * t) * S];
  }
  function fit() {
    while (parts.length < target) parts.push(spawn({}));
    if (parts.length > target) parts.length = target;
  }

  function frame(now) {
    const dt = Math.min(3, (now - last) / 16.67);
    last = now;
    if (running) {
      fit();
      fctx.globalCompositeOperation = 'destination-out';
      fctx.fillStyle = 'rgba(0,0,0,0.16)';
      fctx.fillRect(0, 0, fx.width, fx.height);
      fctx.globalCompositeOperation = 'source-over';
      fctx.lineCap = 'round';
      for (let n = 0; n < parts.length; n++) {
        const p = parts[n];
        let j = DOWN[p.i];
        if (j < 0 || --p.life <= 0) { spawn(p); continue; }
        const a = pos(p.i, p.t);
        const len = Math.hypot(X[j] - X[p.i], Y[j] - Y[p.i]) || 1;
        p.t += p.v * speed * dt / len;
        while (p.t >= 1) {
          p.t -= 1;
          p.i = DOWN[p.i];
          if (p.i < 0 || DOWN[p.i] < 0) { break; }
        }
        if (p.i < 0 || DOWN[p.i] < 0) { spawn(p); continue; }
        const b = pos(p.i, p.t);
        const w = WT[p.i] / 100;
        fctx.strokeStyle = 'rgba(' + Math.round(130 + 60 * (1 - w)) + ',' + Math.round(215 + 30 * (1 - w)) + ',255,0.95)';
        fctx.lineWidth = (0.9 + 1.7 * w) * S * 0.55;
        fctx.beginPath();
        fctx.moveTo(a[0], a[1]);
        fctx.lineTo(b[0], b[1]);
        fctx.stroke();
      }
    }
    requestAnimationFrame(frame);
  }

  const pp = document.getElementById('pp'), sp = document.getElementById('sp'), dn = document.getElementById('dn');
  dn.value = D.n;
  if (reduce) { running = false; pp.textContent = 'Play'; }
  pp.onclick = function () { running = !running; pp.textContent = running ? 'Pause' : 'Play'; };
  sp.oninput = function () { speed = parseFloat(sp.value); };
  dn.oninput = function () { target = parseInt(dn.value, 10); };
  fit();
  requestAnimationFrame(frame);
}
</script></body></html>
"""


def build_html(net, bg_uri, density):
    data = {"w": net["w"], "h": net["h"], "x": net["x"], "y": net["y"], "down": net["down"],
            "wt": net["wt"], "bg": bg_uri, "n": int(density)}
    return HTML_TEMPLATE.replace("__DATA__", json.dumps(data))


# -------------------------------------------------
# STREAMLIT PANEL
# -------------------------------------------------
def _embed(html, height):
    """Show an HTML page in the app. Newer Streamlit uses st.iframe; older versions use components.html."""
    if hasattr(st, "iframe"):
        try:
            return st.iframe(html, height=height)
        except TypeError:
            pass
    import streamlit.components.v1 as components      # older Streamlit only
    return components.html(html, height=height)


def render_flow_animation():
    st.header("Water flow")
    st.caption(
        "An illustrative animation of how water tends to drain across the catchment, traced from the "
        "elevation model. It shows modelled drainage paths, not live river levels, rainfall or flooding."
    )

    p = _paths()
    if not all(os.path.exists(p[k]) for k in ("dem", "acc", "catchment")):
        st.info("The elevation, flow-accumulation and catchment maps are needed for this tab. "
                "Run the earlier lessons to create them.")
        return

    net = load_network()
    if net is None:
        st.info("The drainage network has not been built yet (or the elevation maps changed). "
                "This is quick (usually under a minute) and is saved, so it only happens once.")
        if st.button("Build the water-flow network"):
            with st.spinner("Tracing drainage paths from the elevation model..."):
                try:
                    build_network()
                except Exception as ex:
                    st.error(f"Could not build the network: {ex}")
                    return
            st.rerun()
        return

    c1, c2 = st.columns(2)
    kind = c1.selectbox("Background", ["Risk classes", "Elevation", "Plain"], key="flow_bg")
    density = c2.slider("Water particles", 200, 4000, 1500, 100, key="flow_n")

    _embed(build_html(net, background_image(kind, net), density), 740)
    st.caption(
        f"Rivers are the busiest {int(CHANNEL_SHARE * 100)}% of drainage cells (each cell is "
        f"{STRIDE * 30} m wide); thicker, brighter lines carry more water. Use Pause, Speed and Water "
        "above the map to adjust the animation."
    )
