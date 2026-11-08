"""Adaptive attacker-defender experiments.
The pre-trained WGAN-GP generator is fine-tuned to evade a differentiable surrogate detector
(1-D CNN) while the WGAN critic keeps it close to the real distribution:
    L_G = -E[C(G(z))] + lambda * BCE(D_s(G(z)), real)
Usage: python adapt.py <base: gan|kingan> sweep|game"""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for _folder in ("samples", "results", "paper", "paper/figs"):
    (ROOT / _folder).mkdir(parents=True, exist_ok=True)
import sys, json, copy, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
import gvlib as g
import evaluate as E
from models import GenGRU, ConvNet, gradient_penalty

base, mode = sys.argv[1], sys.argv[2]
torch.manual_seed(11)
rng = np.random.default_rng(11)
NORM = E.NORM
enc = {k: g.encode(E.R[k]) for k in ["train", "val", "test"]}
S0 = {k: torch.tensor(NORM.ns(enc[k][0])) for k in enc}
ST = {k: torch.tensor(NORM.nt(enc[k][1])) for k in enc}
T = ST["train"].shape[1]
ck = torch.load(f"{ROOT}/ckpt_{base}.pt", map_location="cpu", weights_only=True)
G0 = GenGRU(NORM, T=T, kin=(base == "kingan")); G0.load_state_dict(ck["G"])
C0 = ConvNet(); C0.load_state_dict(ck["D"])
KIN = E.GROUPS["A_reported"] + E.GROUPS["B_position"] + E.GROUPS["C_consistency"]


def onland(w):
    import shapely
    lat, lon = w[..., 0], w[..., 1]
    return shapely.contains_xy(g.land(), lon.ravel(), lat.ravel()).reshape(lat.shape).any(1)


def gen(G, reject_land=False, tries=15):
    """reject_land: the attacker uses the public coastline and redraws ghosts that cross land"""
    out = {}
    G.eval()
    with torch.no_grad():
        for k in ["train", "val", "test"]:
            xs = [G(S0[k][i:i + 1024]).numpy() for i in range(0, len(S0[k]), 1024)]
            w = g.decode(enc[k][0], NORM.dt(np.concatenate(xs)))
            if reject_land:
                for _ in range(tries):
                    bad = np.where(onland(w))[0]
                    if len(bad) == 0: break
                    x = G(S0[k][bad]).numpy()
                    w[bad] = g.decode(enc[k][0][bad], NORM.dt(x))
            out[k] = w
    G.train()
    return out


def split_from(samples, tag):
    out = {}
    for k in ["train", "val", "test"]:
        Fr = E.feats(f"real_{k}", E.R[k]); Fg = g.features(samples[k])[0]
        out[k] = dict(F=np.vstack([Fr, Fg]), W=np.concatenate([E.R[k], samples[k]]),
                      y=np.r_[np.zeros(len(Fr)), np.ones(len(Fg))], c=np.r_[E.MM[k], E.MM[k]])
    return out


def surrogate(samples, seed=0):
    Wtr = np.concatenate([E.R["train"], samples["train"]]); ytr = np.r_[np.zeros(len(E.R["train"])), np.ones(len(samples["train"]))]
    Wva = np.concatenate([E.R["val"], samples["val"]]); yva = np.r_[np.zeros(len(E.R["val"])), np.ones(len(samples["val"]))]
    m = E.train_cnn(Wtr, ytr, Wva, yva, seed)
    for p in m.parameters(): p.requires_grad_(False)
    return m


def finetune(G, C, Ds, lam, iters):
    oG = torch.optim.Adam(G.parameters(), 5e-5, betas=(0.0, 0.9))
    oC = torch.optim.Adam(C.parameters(), 1e-4, betas=(0.0, 0.9))
    n = len(S0["train"])
    for it in range(iters):
        for _ in range(3):
            b = rng.integers(0, n, 128)
            s0, real = S0["train"][b], ST["train"][b]
            fake = G(s0).detach()
            lC = C(s0, fake).mean() - C(s0, real).mean() + 10 * gradient_penalty(C, s0, real, fake)
            oC.zero_grad(); lC.backward(); oC.step()
        b = rng.integers(0, n, 128)
        s0 = S0["train"][b]
        fake = G(s0)
        lG = -C(s0, fake).mean()
        if lam > 0:
            lG = lG + lam * nn.functional.binary_cross_entropy_with_logits(Ds(s0, fake), torch.zeros(len(b)))
        oG.zero_grad(); lG.backward(); oG.step()
    return G, C


def assess(samples, Ds, rf_static):
    S = split_from(samples, "x")
    y = S["test"]["y"]
    t0, tt = E.seq_tensors(S["test"]["W"])
    with torch.no_grad():
        a_sur = roc_auc_score(y, Ds(t0, tt).numpy())
    a_static = roc_auc_score(y, rf_static.predict_proba(S["test"]["F"][:, E.ALLC])[:, 1])
    s_rf_all, _ = E.det_rf(S, E.ALLC, 0)
    s_rf_kin, _ = E.det_rf(S, KIN, 0)
    s_cnn = E.det_cnn(S, None, 0)
    s_rule = E.det_rule(S)
    rq = rng.choice(len(samples["test"]), 400, replace=False)
    Q = g.to_xy(samples["test"][rq])[..., :2]
    ref = g.to_xy(E.R["test"])[..., :2]
    Fr = E.feats("real_train", E.R["train"])[:, KIN]
    mu, sd = Fr.mean(0), Fr.std(0) + 1e-9
    r = dict(auc_surrogate_frozen=a_sur, auc_rf_static=a_static,
             auc_rf_retrained=roc_auc_score(y, s_rf_all), auc_rf_retrained_kin=roc_auc_score(y, s_rf_kin),
             auc_cnn_retrained=roc_auc_score(y, s_cnn), auc_rule=roc_auc_score(y, s_rule),
             tpr1_rf_retrained=E.tpr_at(y, s_rf_all), tpr1_rule=E.tpr_at(y, s_rule),
             dtw_shape=float(np.mean(g.nn_distance(Q, ref, "dtw", shape_only=True))),
             mmd=g.mmd_rbf((g.features(samples["test"])[0][:, KIN] - mu) / sd,
                           (E.feats("real_test", E.R["test"])[:, KIN] - mu) / sd))
    r.update(g.violations(samples["test"]))
    return {k: float(v) for k, v in r.items()}


samp0 = gen(G0)
S_base = split_from(samp0, "base")
_, rf_static = E.det_rf(S_base, E.ALLC, 0)

if mode == "sweep":
    Ds = surrogate(samp0)
    res = {}
    for lam in [0.0, 0.3, 1.0, 3.0, 10.0]:
        G, C = copy.deepcopy(G0), copy.deepcopy(C0)
        G, C = finetune(G, C, Ds, lam, 800)
        res[str(lam)] = assess(gen(G), Ds, rf_static)
        print(lam, {k: round(v, 4) for k, v in res[str(lam)].items()}, flush=True)
        json.dump(res, open(f"{ROOT}/results/adapt_sweep_{base}.json", "w"), indent=1)

elif mode == "game":
    lam = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    G, C = copy.deepcopy(G0), copy.deepcopy(C0)
    samp = gen(G0, reject_land=True)
    res = {"0": assess(samp, surrogate(samp), rf_static)}
    print(0, res["0"], flush=True)
    for rnd in range(1, 5):
        Ds = surrogate(samp, seed=rnd)  # defender retrains on the attacker's latest ghosts
        G, C = finetune(G, C, Ds, lam, 600)  # attacker adapts to the retrained defender
        samp = gen(G, reject_land=True)
        res[str(rnd)] = assess(samp, Ds, rf_static)
        print(rnd, {k: round(v, 4) for k, v in res[str(rnd)].items()}, flush=True)
        json.dump(res, open(f"{ROOT}/results/adapt_game_{base}.json", "w"), indent=1)
