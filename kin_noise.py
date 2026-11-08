"""Kin-WGAN + calibrated residual noise: an attacker who notices that kinematically integrated
ghosts are *too* consistent adds displacement noise calibrated (on training data only) so that the
speed-residual and course/bearing-angle statistics match real traffic."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for _folder in ("samples", "results", "paper", "paper/figs"):
    (ROOT / _folder).mkdir(parents=True, exist_ok=True)
import json, numpy as np
import gvlib as g

rng = np.random.default_rng(5)
D = g.load()
K = np.load(f"{ROOT}/samples/kingan.npz")
Fr, _ = g.features(D["train"])
i_res, i_ang = g.FEATURE_NAMES.index("resv_absmed"), g.FEATURE_NAMES.index("ang_median")
target = np.array([np.median(Fr[:, i_res]), np.median(Fr[:, i_ang])])


KMSTEP = 1.852 / 6.0


def perturb(w, sigma, seed):
    """re-integrate positions along the *reported* course (as the detector's consistency feature
    assumes), then perturb each step's displacement: along-track relative error s.d. sigma[0]
    and cross-track rotation s.d. sigma[1] degrees"""
    r = np.random.default_rng(seed)
    s0, st = g.encode(w)
    st = st.copy()
    v_prev = np.concatenate([s0[:, None, 2], st[:, :-1, 2]], 1)
    dist = 0.5 * (v_prev + st[..., 2]) * KMSTEP * (1 + r.normal(0, sigma[0], st[..., 2].shape))
    a = np.arctan2(st[..., 3], st[..., 4]) + np.deg2rad(r.normal(0, sigma[1], st[..., 2].shape))
    st[..., 0] = dist * np.sin(a)
    st[..., 1] = dist * np.cos(a)
    return g.decode(s0, st)


best = None
for sigma in [(a, b) for a in [0.005, 0.01, 0.015, 0.02] for b in [0.5, 1.0, 1.5, 2.0]]:
    F, _ = g.features(perturb(K["train"][:3000], sigma, 0))
    got = np.array([np.median(F[:, i_res]), np.median(F[:, i_ang])])
    err = abs(got[0] - target[0]) / target[0] + abs(got[1] - target[1]) / target[1]
    print(sigma, got.round(3), target.round(3), round(err, 3))
    if best is None or err < best[1]:
        best = (sigma, err)
sigma = best[0]
out = {k: perturb(K[k], sigma, i + 1) for i, k in enumerate(["train", "val", "test"])}
np.savez_compressed(f"{ROOT}/samples/kinnoise.npz", **out)
json.dump({"sigma_alongtrack_rel": sigma[0], "sigma_crosstrack_deg": sigma[1], "target_resv_absmed": float(target[0]), "target_ang_median": float(target[1])},
          open(f"{ROOT}/results/train_kinnoise.json", "w"))
print("chosen sigma", sigma)
