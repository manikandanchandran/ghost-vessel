"""Kin-WGAN+noise, coastline-aware: the attacker uses the public coastline and redraws (fresh noise,
same start state) any ghost window that touches land, up to TRIES times. Same generator weights
(ckpt_kingan.pt) and the same calibrated noise as kin_noise.py."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for _folder in ("samples", "results", "paper", "paper/figs"):
    (ROOT / _folder).mkdir(parents=True, exist_ok=True)
import json, numpy as np, torch, shapely
import gvlib as g
from models import Norm, GenGRU

TRIES = 50
KMSTEP = 1.852 / 6.0
cfg = json.load(open(f"{ROOT}/results/train_kinnoise.json"))
SIG = (cfg["sigma_alongtrack_rel"], cfg["sigma_crosstrack_deg"])
torch.manual_seed(21)
rng = np.random.default_rng(21)
D = g.load()
enc = {k: g.encode(D[k]) for k in ["train", "val", "test"]}
norm = Norm(*enc["train"])
T = enc["train"][1].shape[1]
G = GenGRU(norm, T=T, kin=True)
G.load_state_dict(torch.load(f"{ROOT}/ckpt_kingan.pt", map_location="cpu", weights_only=True)["G"])
G.eval()


def perturb(w):
    s0, st = g.encode(w)
    st = st.copy()
    v_prev = np.concatenate([s0[:, None, 2], st[:, :-1, 2]], 1)
    dist = 0.5 * (v_prev + st[..., 2]) * KMSTEP * (1 + rng.normal(0, SIG[0], st[..., 2].shape))
    a = np.arctan2(st[..., 3], st[..., 4]) + np.deg2rad(rng.normal(0, SIG[1], st[..., 2].shape))
    st[..., 0] = dist * np.sin(a)
    st[..., 1] = dist * np.cos(a)
    return g.decode(s0, st)


def onland(w):
    """True when any report lies on land or any consecutive segment intersects land."""
    return g.coastline_intersections(w)


def sample(idx, k):
    s0n = torch.tensor(norm.ns(enc[k][0][idx]))
    with torch.no_grad():
        x = np.concatenate([G(s0n[i:i + 1024]).numpy() for i in range(0, len(idx), 1024)])
    return perturb(g.decode(enc[k][0][idx], norm.dt(x)))


out, stats = {}, {}
for k in ["train", "val", "test"]:
    n = len(enc[k][0])
    w = sample(np.arange(n), k)
    first = float(onland(w).mean())
    attempts = 0
    for t in range(TRIES):
        bad = np.where(onland(w))[0]
        if len(bad) == 0:
            break
        w[bad] = sample(bad, k)
        attempts += 1
    stats[k] = dict(intersection_before=first, intersection_after=float(onland(w).mean()),
                    redraw_passes=attempts, windows=int(n),
                    note="Includes land points and straight-line segment intersections")
    out[k] = w
    print(k, stats[k], flush=True)
np.savez_compressed(f"{ROOT}/samples/kinland.npz", **out)
json.dump(dict(tries=TRIES, **stats), open(f"{ROOT}/results/train_kinland.json", "w"), indent=1)
