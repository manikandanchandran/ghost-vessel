"""Shared utilities: representation, features, realism metrics."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for _folder in ("samples", "results", "paper", "paper/figs"):
    (ROOT / _folder).mkdir(parents=True, exist_ok=True)
import numpy as np, pickle, shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

LAT0, LAT1, LON0, LON1 = 55.5, 58.0, 10.3, 13.0
LATC = (LAT0 + LAT1) / 2
LONC = (LON0 + LON1) / 2
KLAT = 111.32
KLON = 111.32 * np.cos(np.deg2rad(LATC))
DT_H = 600 / 3600.0
KN2KMH = 1.852

_LAND = None


def land():
    global _LAND
    if _LAND is None:
        polys = pickle.load(open(f"{ROOT}/coast.pkl", "rb"))
        g = unary_union([Polygon(np.asarray(p)[:, [1, 0]]).buffer(0) for p in polys])
        shapely.prepare(g)
        _LAND = g
    return _LAND


def coastline_intersections(w, polygon=None, batch_windows=256):
    """Return (N,) bool: a reported position is on land OR any straight segment
    joining consecutive 10-minute AIS reports intersects a land polygon.

    Coordinates supplied to Shapely are always (longitude, latitude).  This
    is a geographic chord check, NOT a claim that the vessel physically used
    the straight line between samples or that navigation on a sphere was modelled.
    A preconstructed `polygon` may be supplied by unit tests; otherwise the
    public coastline polygons loaded by `land()` are used.
    """
    w = np.asarray(w)
    if w.ndim != 3 or w.shape[-1] < 2:
        raise ValueError("expected AIS windows of shape (N,L,>=2) with lat/lon first")
    if w.shape[1] < 2:
        raise ValueError("at least two points are required per window")
    geometry = land() if polygon is None else polygon
    # Shapely 2.x preparation accelerates predicates against one coastline.
    shapely.prepare(geometry)
    found = np.zeros(len(w), dtype=bool)
    for start in range(0, len(w), batch_windows):
        chunk = w[start:start + batch_windows]
        coords = np.stack((chunk[..., 1], chunk[..., 0]), axis=-1)
        n, t, _ = coords.shape
        points = shapely.points(coords.reshape(-1, 2))
        hits_points = shapely.intersects(geometry, points).reshape(n, t).any(axis=1)
        # Shapely linestrings input: (num_segments, two_endpoints, xy).
        seg_coords = np.stack((coords[:, :-1], coords[:, 1:]), axis=2)
        segments = shapely.linestrings(seg_coords.reshape(-1, 2, 2))
        hits_lines = shapely.intersects(geometry, segments).reshape(n, t - 1).any(axis=1)
        found[start:start + n] = hits_points | hits_lines
    return found


def load():
    d = np.load(f"{ROOT}/data.npz")
    return {k: d[k] for k in d.files}


# ---- representation ---------------------------------------------------------
def to_xy(w):
    """w: (N,L,4) lat,lon,sog,cog -> (N,L,4) x_km,y_km,sog,cog"""
    out = w.copy()
    out[..., 0] = (w[..., 1] - LONC) * KLON
    out[..., 1] = (w[..., 0] - LATC) * KLAT
    return out


def to_latlon(xy):
    lat = xy[..., 1] / KLAT + LATC
    lon = xy[..., 0] / KLON + LONC
    return lat, lon


def encode(w):
    """(N,L,4) lat/lon/sog/cog -> start state (N,5) and steps (N,L-1,5): dx,dy,sog,sin,cos"""
    xy = to_xy(w)
    c = np.deg2rad(xy[..., 3])
    s0 = np.stack([xy[:, 0, 0], xy[:, 0, 1], xy[:, 0, 2], np.sin(c[:, 0]), np.cos(c[:, 0])], 1)
    d = np.diff(xy[..., :2], axis=1)
    st = np.concatenate([d, xy[:, 1:, 2:3], np.sin(c[:, 1:, None]), np.cos(c[:, 1:, None])], 2)
    return s0.astype(np.float32), st.astype(np.float32)


def decode(s0, st):
    """inverse of encode -> (N,L,4) lat,lon,sog,cog (degrees)"""
    xy0 = s0[:, None, :2]
    pos = np.concatenate([xy0, xy0 + np.cumsum(st[..., :2], 1)], 1)
    sog = np.concatenate([s0[:, None, 2], st[..., 2]], 1)
    sn = np.concatenate([s0[:, None, 3], st[..., 3]], 1)
    cs = np.concatenate([s0[:, None, 4], st[..., 4]], 1)
    cog = (np.rad2deg(np.arctan2(sn, cs)) + 360) % 360
    lat, lon = to_latlon(pos)
    return np.stack([lat, lon, np.clip(sog, 0, None), cog], -1).astype(np.float32)


# ---- handcrafted features -----------------------------------------------------
def wrap(a):
    return (a + 180) % 360 - 180


FEATURE_GROUPS = {}


def features(w, with_land=True):
    """w: (N,L,4) lat,lon,sog,cog. Returns (N,F) array and group index dict."""
    xy = to_xy(w)
    x, y, sog, cog = xy[..., 0], xy[..., 1], xy[..., 2], xy[..., 3]
    dx, dy = np.diff(x, axis=1), np.diff(y, axis=1)
    dist = np.hypot(dx, dy)
    v_imp = dist / DT_H / KN2KMH  # knots
    brg = (np.rad2deg(np.arctan2(dx, dy)) + 360) % 360
    dsog = np.diff(sog, axis=1)
    dcog = wrap(np.diff(cog, axis=1))
    dbrg = wrap(np.diff(brg, axis=1))
    sog_mid = 0.5 * (sog[:, 1:] + sog[:, :-1])
    res_v = v_imp - sog_mid
    cog_mid = cog[:, 1:]
    moving = sog_mid > 1.0
    ang = np.abs(wrap(brg - cog_mid))
    ang_m = np.where(moving, ang, np.nan)
    acc = np.diff(v_imp, axis=1)
    disp = np.hypot(x[:, -1] - x[:, 0], y[:, -1] - y[:, 0])
    plen = dist.sum(1) + 1e-6

    def q(a, p):
        return np.nanpercentile(a, p, axis=1)

    A = [sog.mean(1), sog.std(1), sog.min(1), sog.max(1), q(sog, 10), q(sog, 90),
         np.abs(dsog).mean(1), dsog.std(1), np.abs(dsog).max(1),
         np.abs(dcog).mean(1), dcog.std(1), np.abs(dcog).max(1), (np.abs(dcog) > 30).mean(1)]
    B = [v_imp.mean(1), v_imp.std(1), v_imp.max(1), np.abs(dbrg).mean(1), dbrg.std(1), np.abs(dbrg).max(1),
         plen, disp, disp / plen, np.abs(acc).mean(1), acc.std(1), np.abs(np.diff(acc, axis=1)).mean(1)]
    C = [np.abs(res_v).mean(1), np.median(np.abs(res_v), 1), np.abs(res_v).max(1), res_v.std(1), res_v.mean(1),
         np.nan_to_num(np.nanmean(ang_m, 1), nan=0), np.nan_to_num(np.nanmedian(ang_m, 1), nan=0),
         np.nan_to_num(np.nanmax(ang_m, 1), nan=0),
         np.array([np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 and b.std() > 1e-6 else 0 for a, b in zip(v_imp, sog_mid)])]
    D = [x[:, 0], y[:, 0], x.mean(1), y.mean(1)]
    if with_land:
        lat, lon = w[..., 0], w[..., 1]
        onl = shapely.contains_xy(land(), lon.ravel(), lat.ravel()).reshape(lat.shape)
        # Historical geographic feature `land_cross` counts transitions between
        # reported on-land/off-land states. It is intentionally not changed to
        # avoid silently altering the 40-feature trained models and paper tables.
        # The separate coastline-only detector uses coastline_intersections().
        D += [onl.mean(1), (np.diff(onl.astype(int), axis=1) != 0).sum(1)]
    cols = A + B + C + D
    F = np.stack(cols, 1).astype(np.float64)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    na, nb, nc = len(A), len(B), len(C)
    groups = {"A_reported": list(range(0, na)), "B_position": list(range(na, na + nb)),
              "C_consistency": list(range(na + nb, na + nb + nc)), "D_geographic": list(range(na + nb + nc, F.shape[1]))}
    return F, groups


FEATURE_NAMES = (
    ["sog_mean", "sog_std", "sog_min", "sog_max", "sog_q10", "sog_q90", "dsog_absmean", "dsog_std", "dsog_absmax",
     "turn_absmean", "turn_std", "turn_absmax", "turn_gt30_frac"]
    + ["vimp_mean", "vimp_std", "vimp_max", "brgturn_absmean", "brgturn_std", "brgturn_absmax", "path_len",
       "displacement", "straightness", "acc_absmean", "acc_std", "jerk_absmean"]
    + ["resv_absmean", "resv_absmed", "resv_absmax", "resv_std", "resv_mean", "ang_mean", "ang_median", "ang_max",
       "corr_vimp_sog"]
    + ["x0", "y0", "xmean", "ymean", "land_frac", "land_cross"])


def violations(w):
    xy = to_xy(w)
    d = np.hypot(np.diff(xy[..., 0], axis=1), np.diff(xy[..., 1], axis=1))
    v = d / DT_H / KN2KMH
    sog_mid = 0.5 * (xy[:, 1:, 2] + xy[:, :-1, 2])
    onl = shapely.contains_xy(land(), w[..., 1].ravel(), w[..., 0].ravel()).reshape(w.shape[:2])
    return dict(speed_gt40=float((v > 40).mean()), resv_gt5=float((np.abs(v - sog_mid) > 5).mean()),
                land_pts=float(onl.mean()), land_windows=float(onl.any(1).mean()))


# ---- trajectory distances (vectorised over pairs) -----------------------------
def _pair_dp(A, B, mode):
    """A,B: (P,L,2) -> (P,) DTW (mean step cost) or discrete Frechet."""
    P, n, _ = A.shape
    m = B.shape[1]
    C = np.sqrt(((A[:, :, None, :] - B[:, None, :, :]) ** 2).sum(-1))  # (P,n,m)
    D = np.full((P, n, m), np.inf)
    for i in range(n):
        for j in range(m):
            c = C[:, i, j]
            if i == 0 and j == 0:
                D[:, 0, 0] = c
                continue
            prev = np.full(P, np.inf)
            if i > 0:
                prev = np.minimum(prev, D[:, i - 1, j])
            if j > 0:
                prev = np.minimum(prev, D[:, i, j - 1])
            if i > 0 and j > 0:
                prev = np.minimum(prev, D[:, i - 1, j - 1])
            D[:, i, j] = (prev + c) if mode == "dtw" else np.maximum(prev, c)
    out = D[:, -1, -1]
    return out / (n + m) if mode == "dtw" else out


def nn_distance(Q, R, mode, shape_only=False, chunk=40000, prefilter=25):
    """For each query trajectory in Q (Nq,L,2 km) the distance to its nearest neighbour in R.
    Candidates pre-filtered by Euclidean (lock-step) distance for tractability."""
    if shape_only:
        Q = Q - Q[:, :1]
        R = R - R[:, :1]
    Qf, Rf = Q.reshape(len(Q), -1), R.reshape(len(R), -1)
    best = np.empty(len(Q))
    for s in range(0, len(Q), 200):
        q = Qf[s:s + 200]
        d2 = (q ** 2).sum(1)[:, None] + (Rf ** 2).sum(1)[None] - 2 * q @ Rf.T
        idx = np.argsort(d2, 1)[:, :prefilter]
        A = np.repeat(Q[s:s + 200], prefilter, 0)
        B = R[idx.ravel()]
        dd = _pair_dp(A, B, mode).reshape(-1, prefilter)
        best[s:s + 200] = dd.min(1)
    return best


def mmd_rbf(X, Y, n=2000, seed=0):
    """unbiased MMD^2 with RBF kernel, bandwidth = median pairwise squared distance"""
    rng = np.random.default_rng(seed)
    X = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    Y = Y[rng.choice(len(Y), min(n, len(Y)), replace=False)]
    Z = np.vstack([X, Y]).astype(np.float64)
    sq = (Z ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None] - 2 * Z @ Z.T, 0)
    s = np.median(d2[d2 > 0])
    K = np.exp(-d2 / s)
    nx, ny = len(X), len(Y)
    kxx, kyy, kxy = K[:nx, :nx], K[nx:, nx:], K[:nx, nx:]
    return float((kxx.sum() - np.trace(kxx)) / nx / (nx - 1) + (kyy.sum() - np.trace(kyy)) / ny / (ny - 1) - 2 * kxy.mean())
