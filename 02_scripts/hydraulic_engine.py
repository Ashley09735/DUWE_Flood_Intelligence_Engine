"""
DUWE AI FLOOD INTELLIGENCE ENGINE
2D hydraulic engine: local-inertial shallow-water solver with rain on the whole grid.

Method: the q-centred local-inertial scheme of Bates et al. (2010) / de Almeida et al. (2012),
the same family of equations used in LISFLOOD-FP. Water moves between neighbouring cells
according to the water-surface slope, with Manning friction, a Froude limiter and a
donor-volume limiter that stops cells from going below zero depth.

Included:   rainfall (any hyetograph, optional spatial pattern), initial abstraction and
            infiltration with a soil-storage limit, point inflows (e.g. a river hydrograph),
            free outflow at designated outlet cells, mass-balance accounting.
Not included: river-channel bathymetry (a DEM does not contain it), structures (bridges,
            culverts, embankments below the DEM resolution), sediment, tides.

Needs numba:   pip install numba
"""

import math
import time

import numpy as np

try:
    from numba import njit, prange
except ImportError as ex:                                     # pragma: no cover
    raise ImportError("The hydraulic engine needs numba. Install it with:  pip install numba") from ex

G = 9.81


# -------------------------------------------------
# NUMBA KERNELS (arrays are padded with a one-cell inactive border)
# -------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def flux_x(z, h, active, outlet, q_old, q_new, n, dx, dt, theta, hmin, frmax, s0out):
    NY, NX = z.shape
    c73 = 7.0 / 3.0
    c53 = 5.0 / 3.0
    for i in prange(1, NY - 1):
        for j in range(1, NX):
            aL = active[i, j - 1]
            aR = active[i, j]
            if aL and aR:
                etaL = z[i, j - 1] + h[i, j - 1]
                etaR = z[i, j] + h[i, j]
                zf = z[i, j - 1] if z[i, j - 1] > z[i, j] else z[i, j]
                hf = (etaL if etaL > etaR else etaR) - zf
                if hf <= hmin:
                    q_new[i, j] = 0.0
                    continue
                q = q_old[i, j]
                nf = 0.5 * (n[i, j - 1] + n[i, j])
                num = theta * q + 0.5 * (1.0 - theta) * (q_old[i, j - 1] + q_old[i, j + 1]) \
                    - 9.81 * hf * dt * (etaR - etaL) / dx
                den = 1.0 + 9.81 * dt * nf * nf * abs(q) / hf ** c73
                qq = num / den
                qfr = frmax * hf * math.sqrt(9.81 * hf)
                if qq > qfr:
                    qq = qfr
                elif qq < -qfr:
                    qq = -qfr
                if qq > 0.0:
                    lim = h[i, j - 1] * dx / (4.0 * dt)
                    if qq > lim:
                        qq = lim
                else:
                    lim = h[i, j] * dx / (4.0 * dt)
                    if -qq > lim:
                        qq = -lim
                q_new[i, j] = qq
            elif aL and outlet[i, j - 1]:               # outflow to the right
                hd = h[i, j - 1]
                qq = 0.0
                if hd > hmin:
                    qq = hd ** c53 * math.sqrt(s0out) / n[i, j - 1]
                    qq = min(qq, hd * math.sqrt(9.81 * hd), hd * dx / (4.0 * dt))
                q_new[i, j] = qq
            elif aR and outlet[i, j]:                   # outflow to the left
                hd = h[i, j]
                qq = 0.0
                if hd > hmin:
                    qq = hd ** c53 * math.sqrt(s0out) / n[i, j]
                    qq = min(qq, hd * math.sqrt(9.81 * hd), hd * dx / (4.0 * dt))
                q_new[i, j] = -qq
            else:
                q_new[i, j] = 0.0


@njit(parallel=True, fastmath=True, cache=True)
def flux_y(z, h, active, outlet, q_old, q_new, n, dx, dt, theta, hmin, frmax, s0out):
    NY, NX = z.shape
    c73 = 7.0 / 3.0
    c53 = 5.0 / 3.0
    for i in prange(1, NY):
        for j in range(1, NX - 1):
            aU = active[i - 1, j]
            aD = active[i, j]
            if aU and aD:
                etaU = z[i - 1, j] + h[i - 1, j]
                etaD = z[i, j] + h[i, j]
                zf = z[i - 1, j] if z[i - 1, j] > z[i, j] else z[i, j]
                hf = (etaU if etaU > etaD else etaD) - zf
                if hf <= hmin:
                    q_new[i, j] = 0.0
                    continue
                q = q_old[i, j]
                nf = 0.5 * (n[i - 1, j] + n[i, j])
                num = theta * q + 0.5 * (1.0 - theta) * (q_old[i - 1, j] + q_old[i + 1, j]) \
                    - 9.81 * hf * dt * (etaD - etaU) / dx
                den = 1.0 + 9.81 * dt * nf * nf * abs(q) / hf ** c73
                qq = num / den
                qfr = frmax * hf * math.sqrt(9.81 * hf)
                if qq > qfr:
                    qq = qfr
                elif qq < -qfr:
                    qq = -qfr
                if qq > 0.0:
                    lim = h[i - 1, j] * dx / (4.0 * dt)
                    if qq > lim:
                        qq = lim
                else:
                    lim = h[i, j] * dx / (4.0 * dt)
                    if -qq > lim:
                        qq = -lim
                q_new[i, j] = qq
            elif aU and outlet[i - 1, j]:               # outflow downwards (+y)
                hd = h[i - 1, j]
                qq = 0.0
                if hd > hmin:
                    qq = hd ** c53 * math.sqrt(s0out) / n[i - 1, j]
                    qq = min(qq, hd * math.sqrt(9.81 * hd), hd * dx / (4.0 * dt))
                q_new[i, j] = qq
            elif aD and outlet[i, j]:                   # outflow upwards (-y)
                hd = h[i, j]
                qq = 0.0
                if hd > hmin:
                    qq = hd ** c53 * math.sqrt(s0out) / n[i, j]
                    qq = min(qq, hd * math.sqrt(9.81 * hd), hd * dx / (4.0 * dt))
                q_new[i, j] = -qq
            else:
                q_new[i, j] = 0.0


@njit(parallel=True, fastmath=True, cache=True)
def update_depth(h, active, qx, qy, rate, rf, ia, fcum, f_rate, fmax, dt, dx, t, hwet,
                 maxd, maxs, maxhz, arr, wet, row_hmax, row_rain, row_inf, row_neg):
    NY, NX = h.shape
    for i in prange(1, NY - 1):
        hm = 0.0
        vr = 0.0
        vi = 0.0
        vn = 0.0
        for j in range(1, NX - 1):
            if not active[i, j]:
                continue
            hij = h[i, j] + dt * (qx[i, j] - qx[i, j + 1] + qy[i, j] - qy[i + 1, j]) / dx
            r = rate * rf[i, j] * dt
            hij += r
            vr += r
            if hij < 0.0:
                vn -= hij                                   # numerical negative depth, tracked as mass error
                hij = 0.0
            if hij > 0.0:
                a = ia[i, j]
                if a > 0.0:
                    d = hij if hij < a else a
                    hij -= d
                    ia[i, j] = a - d
                    vi += d
                if f_rate > 0.0 and hij > 0.0:
                    room = fmax - fcum[i, j]
                    if room > 0.0:
                        cap = f_rate * dt
                        if cap > room:
                            cap = room
                        d = hij if hij < cap else cap
                        hij -= d
                        fcum[i, j] += d
                        vi += d
            h[i, j] = hij
            if hij > hm:
                hm = hij
            if hij > maxd[i, j]:
                maxd[i, j] = hij
            if hij >= hwet:
                u = 0.5 * (qx[i, j] + qx[i, j + 1])
                v = 0.5 * (qy[i, j] + qy[i + 1, j])
                sp = math.sqrt(u * u + v * v) / hij
                if sp > 15.0:
                    sp = 15.0
                if sp > maxs[i, j]:
                    maxs[i, j] = sp
                hz = hij * (sp + 0.5)
                if hz > maxhz[i, j]:
                    maxhz[i, j] = hz
                if arr[i, j] < 0.0:
                    arr[i, j] = t + dt
                wet[i, j] += dt
        row_hmax[i] = hm
        row_rain[i] = vr
        row_inf[i] = vi
        row_neg[i] = vn


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def _pad(a, fill, dtype):
    out = np.full((a.shape[0] + 2, a.shape[1] + 2), fill, dtype=dtype)
    out[1:-1, 1:-1] = a
    return out


def _boundary_faces(active, outlet):
    """Faces between an outlet cell and the outside: index arrays and +1/-1 outflow direction."""
    ax_i, ax_j, ax_s = [], [], []
    aL, aR = active[:, :-1], active[:, 1:]
    oL, oR = outlet[:, :-1], outlet[:, 1:]
    i, j = np.nonzero(aL & ~aR & oL)
    ax_i.append(i); ax_j.append(j + 1); ax_s.append(np.ones(len(i)))
    i, j = np.nonzero(~aL & aR & oR)
    ax_i.append(i); ax_j.append(j + 1); ax_s.append(-np.ones(len(i)))
    bx = (np.concatenate(ax_i), np.concatenate(ax_j), np.concatenate(ax_s))

    ay_i, ay_j, ay_s = [], [], []
    aU, aD = active[:-1, :], active[1:, :]
    oU, oD = outlet[:-1, :], outlet[1:, :]
    i, j = np.nonzero(aU & ~aD & oU)
    ay_i.append(i + 1); ay_j.append(j); ay_s.append(np.ones(len(i)))
    i, j = np.nonzero(~aU & aD & oD)
    ay_i.append(i + 1); ay_j.append(j); ay_s.append(-np.ones(len(i)))
    by = (np.concatenate(ay_i), np.concatenate(ay_j), np.concatenate(ay_s))
    return bx, by


def estimate_memory_mb(ny, nx):
    cells = (ny + 2) * (nx + 2)
    return cells * (8 * 2 + 4 * 4 + 4 * 4 + 4 * 5 + 4 * 2 + 2) / 1e6


# -------------------------------------------------
# MAIN SOLVER
# -------------------------------------------------
def simulate(z, dx, hours, rain_edges_s, rain_rates_ms, outlet, n_manning=0.05, rain_factor=None,
             init_abstraction_m=0.0, infil_rate_ms=0.0, soil_storage_m=0.0,
             inflows=None, snapshot_interval_s=None, on_snapshot=None, on_progress=None,
             theta=0.9, alpha=0.5, hmin=1e-4, hwet=0.1, froude_max=1.0,
             dt_max=30.0, dt_min=1e-3, outlet_slope=0.001, depth_cap_for_dt=10.0, h0=None):
    """
    z       (ny, nx) bed elevation in metres, NaN where outside the model domain
    dx      cell size in metres
    rain_edges_s / rain_rates_ms   rainfall as steps: rate k (m/s) applies from edge k to edge k+1
    outlet  (ny, nx) bool, domain-edge cells where water may leave freely
    h0      optional (ny, nx) initial water depth in metres (default: dry)
    inflows list of dicts: {"i": row, "j": col, "t_s": times, "q_m3s": discharge}
    Returns a dict of result arrays (NaN outside the domain) and a mass balance.
    """
    t_wall0 = time.time()
    ny, nx = z.shape
    active_in = np.isfinite(z)
    active = _pad(active_in, False, bool)
    zp = _pad(np.where(active_in, z, 0.0), 0.0, np.float64)
    h = np.zeros_like(zp)
    if h0 is not None:
        h[1:-1, 1:-1] = np.where(active_in, np.maximum(h0, 0.0), 0.0)
    outlet_p = _pad(outlet & active_in, False, bool)

    n_arr = np.full(active_in.shape, n_manning, dtype=np.float32) if np.isscalar(n_manning) else n_manning.astype(np.float32)
    n_p = _pad(n_arr, 0.05, np.float32)
    rf = _pad(np.ones(active_in.shape, np.float32) if rain_factor is None else rain_factor.astype(np.float32), 1.0, np.float32)
    ia = _pad(np.full(active_in.shape, init_abstraction_m, np.float32), 0.0, np.float32)
    fcum = np.zeros_like(ia)

    qx = np.zeros((zp.shape[0], zp.shape[1] + 1), np.float32)
    qx2 = np.zeros_like(qx)
    qy = np.zeros((zp.shape[0] + 1, zp.shape[1]), np.float32)
    qy2 = np.zeros_like(qy)

    maxd = np.zeros_like(ia)
    maxs = np.zeros_like(ia)
    maxhz = np.zeros_like(ia)
    arr = np.full_like(ia, -1.0)
    wet = np.zeros_like(ia)
    row_hmax = np.zeros(zp.shape[0])
    row_rain = np.zeros(zp.shape[0])
    row_inf = np.zeros(zp.shape[0])
    row_neg = np.zeros(zp.shape[0])

    bx, by = _boundary_faces(active, outlet_p)
    inflows = inflows or []
    for f in inflows:
        f["_i"], f["_j"] = int(f["i"]) + 1, int(f["j"]) + 1

    edges = np.asarray(rain_edges_s, dtype=float)
    rates = np.asarray(rain_rates_ms, dtype=float)
    T = hours * 3600.0
    t, steps = 0.0, 0
    hmax = float(h.max())
    dts = []
    v_rain = v_inf = v_out = v_in = v_neg = 0.0
    next_snap = 0.0 if snapshot_interval_s else math.inf
    next_prog = time.time() + 15.0
    cell_area = dx * dx

    while t < T - 1e-9:
        k = int(np.searchsorted(edges, t + 1e-9, side="right") - 1)
        k = max(k, 0)
        rate = float(rates[k]) if k < len(rates) else 0.0
        nxt = T
        if k + 1 < len(edges) and edges[k + 1] > t + 1e-9:
            nxt = min(nxt, edges[k + 1])
        nxt = min(nxt, next_snap if next_snap > t + 1e-9 else T)

        if hmax > 0.0:
            dt = alpha * dx / math.sqrt(G * min(max(hmax, 1e-3), depth_cap_for_dt))
        else:
            dt = dt_max
        dt = min(max(dt, dt_min), dt_max, nxt - t)
        if dt <= 0.0:
            dt = max(nxt - t, 1e-6)

        flux_x(zp, h, active, outlet_p, qx, qx2, n_p, dx, dt, theta, hmin, froude_max, outlet_slope)
        flux_y(zp, h, active, outlet_p, qy, qy2, n_p, dx, dt, theta, hmin, froude_max, outlet_slope)
        qx, qx2 = qx2, qx
        qy, qy2 = qy2, qy

        update_depth(h, active, qx, qy, rate, rf, ia, fcum, infil_rate_ms, soil_storage_m, dt, dx, t, hwet,
                     maxd, maxs, maxhz, arr, wet, row_hmax, row_rain, row_inf, row_neg)

        for f in inflows:
            q = float(np.interp(t + 0.5 * dt, f["t_s"], f["q_m3s"], left=0.0, right=0.0))
            if q > 0.0 and active[f["_i"], f["_j"]]:
                h[f["_i"], f["_j"]] += dt * q / cell_area
                v_in += dt * q

        hmax = float(row_hmax.max())
        v_rain += float(row_rain.sum()) * cell_area
        v_inf += float(row_inf.sum()) * cell_area
        v_neg += float(row_neg.sum()) * cell_area
        out_flux = float(np.sum(bx[2] * qx[bx[0], bx[1]]) + np.sum(by[2] * qy[by[0], by[1]]))
        v_out += max(out_flux, 0.0) * dx * dt

        t += dt
        steps += 1
        if steps % 50 == 0:
            dts.append(dt)

        if on_snapshot and t >= next_snap - 1e-9:
            on_snapshot(t, np.where(active_in, h[1:-1, 1:-1], np.nan))
            next_snap += snapshot_interval_s
        if on_progress and time.time() >= next_prog:
            on_progress(t, T, steps, hmax, time.time() - t_wall0)
            next_prog = time.time() + 15.0

    storage = float(h[active].sum()) * cell_area
    added = v_rain + v_in
    removed = v_inf + v_out + storage
    mass = {
        "rain_m3": v_rain, "inflow_m3": v_in, "infiltration_m3": v_inf, "outflow_m3": v_out,
        "storage_end_m3": storage, "negative_depth_fixed_m3": v_neg,
        "balance_error_percent": 100.0 * (added - removed) / added if added > 0 else 0.0,
    }

    def out(a):
        return np.where(active_in, a[1:-1, 1:-1], np.nan).astype(np.float32)

    arrival = out(arr)
    arrival = np.where(arrival < 0, np.nan, arrival / 3600.0).astype(np.float32)
    return {
        "max_depth": out(maxd), "max_speed": out(maxs), "max_hazard": out(maxhz),
        "arrival_h": arrival, "wet_hours": (out(wet) / 3600.0).astype(np.float32),
        "final_depth": out(h), "mass_balance": mass, "steps": steps,
        "dt_median_s": float(np.median(dts)) if dts else None,
        "runtime_s": time.time() - t_wall0,
    }
