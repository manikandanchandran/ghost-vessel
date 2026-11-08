"""Build a vessel-disjoint windowed AIS dataset from the public DMA (ct_dma) release.

Source: Danish Maritime Authority AIS data, cargo and tanker vessels, ROI
(55.5N,10.3E)-(58.0N,13.0E), as pre-processed and released with TrAISformer
(Nguyen & Fablet, IEEE Access 2024): 10-min resampling, SOG<30 kn, moored and
near-coast (<1 nmi) points removed, voyages split at 2-h gaps, >=4 h, <=20 h.
Format per point: [lat, lon, sog, cog, unix_ts, mmsi], lat/lon/sog/cog scaled to [0,1].

We pool the three released files, re-split by MMSI (70/15/15) so that no vessel
appears in more than one partition, and cut non-overlapping windows of L steps.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for _folder in ("samples", "results", "paper", "paper/figs"):
    (ROOT / _folder).mkdir(parents=True, exist_ok=True)
import pickle, json, numpy as np, collections, datetime as dt
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from shapely.ops import unary_union

SRC = f"{ROOT}/raw/"
LAT0, LAT1, LON0, LON1, SOGMAX = 55.5, 58.0, 10.3, 13.0, 30.0
L = 36  # 6 h at 10-min sampling
SEED = 2026
LATC = np.deg2rad((LAT0 + LAT1) / 2)
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(LATC)


def denorm(t):
    lat = LAT0 + t[:, 0] * (LAT1 - LAT0)
    lon = LON0 + t[:, 1] * (LON1 - LON0)
    sog = t[:, 2] * SOGMAX
    cog = (t[:, 3] * 360.0) % 360.0
    return np.stack([lat, lon, sog, cog, t[:, 4]], 1)


def main():
    tracks = []
    for s in ["train", "valid", "test"]:
        for x in pickle.load(open(SRC + f"ct_dma_{s}.pkl", "rb")):
            tracks.append((int(x["mmsi"]), denorm(np.asarray(x["traj"], dtype=np.float64))))
    mmsis = np.array(sorted(set(m for m, _ in tracks)))
    rng = np.random.default_rng(SEED)
    rng.shuffle(mmsis)
    n = len(mmsis)
    part = {}
    for i, m in enumerate(mmsis):
        part[m] = "train" if i < 0.70 * n else ("val" if i < 0.85 * n else "test")

    W = {"train": [], "val": [], "test": []}
    meta = {"train": [], "val": [], "test": []}
    for m, t in tracks:
        for k in range(len(t) // L):
            w = t[k * L:(k + 1) * L]
            if np.any(np.diff(w[:, 4]) > 1200):  # safety: contiguous 10-min samples only
                continue
            W[part[m]].append(w[:, :4])
            meta[part[m]].append((m, w[0, 4]))

    polys = pickle.load(open(SRC + "dma_coastline_polygons.pkl", "rb"))
    land = prep(unary_union([Polygon(np.asarray(p)[:, [1, 0]]).buffer(0) for p in polys]))

    out = {}
    stats = {}
    for k in W:
        a = np.stack(W[k]).astype(np.float32)
        out[k] = a
        out[k + "_mmsi"] = np.array([m for m, _ in meta[k]])
        out[k + "_t0"] = np.array([t0 for _, t0 in meta[k]])
        med_sog = np.median(a[:, :, 2], 1)
        stats[k] = dict(
            windows=int(len(a)), vessels=int(len(set(out[k + "_mmsi"]))),
            sog_mean=float(a[:, :, 2].mean()), sog_std=float(a[:, :, 2].std()),
            frac_slow=float((med_sog < 8).mean()), frac_mid=float(((med_sog >= 8) & (med_sog < 14)).mean()),
            frac_fast=float((med_sog >= 14).mean()))
    lens = [len(t) for _, t in tracks]
    ts0 = min(t[0, 4] for _, t in tracks); ts1 = max(t[-1, 4] for _, t in tracks)
    months = collections.Counter(dt.datetime.fromtimestamp(t[0, 4], dt.UTC).strftime("%Y-%m") for _, t in tracks)
    stats["global"] = dict(tracks=len(tracks), vessels=int(n), points=int(sum(lens)),
                           track_len_steps_min=int(min(lens)), track_len_steps_median=float(np.median(lens)),
                           track_len_steps_max=int(max(lens)),
                           start=dt.datetime.fromtimestamp(ts0, dt.UTC).isoformat(),
                           end=dt.datetime.fromtimestamp(ts1, dt.UTC).isoformat(),
                           months=dict(sorted(months.items())), window_len=L, sample_s=600)
    # fraction of real points on land (sanity check of coastline polygon orientation)
    tr = out["train"].reshape(-1, 4)[::50]
    stats["global"]["real_on_land_frac_check"] = float(np.mean([land.contains(Point(lo, la)) for la, lo, _, _ in tr]))
    np.savez_compressed(f"{ROOT}/data.npz", **out)
    pickle.dump(polys, open(f"{ROOT}/coast.pkl", "wb"))
    json.dump(stats, open(f"{ROOT}/results/dataset_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
