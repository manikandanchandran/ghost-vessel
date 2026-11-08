"""Realism metrics and detection experiments for all ghost-vessel generators.
Usage: python evaluate.py realism|detect|ablate|cross"""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for _folder in ("samples", "results", "paper", "paper/figs"):
    (ROOT / _folder).mkdir(parents=True, exist_ok=True)
import sys, json, warnings, numpy as np, torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import wasserstein_distance
import gvlib as g
from models import ConvNet, Norm

warnings.filterwarnings("ignore")
import os
ALL_GENS = ["replay", "constvel", "lstm", "transformer", "vae", "gan", "kingan", "kinnoise", "kinland"]
GENS = os.environ.get("GENS", ",".join(ALL_GENS)).split(",")
LABEL = {"replay": "Replay+offset", "constvel": "Const.-velocity", "lstm": "LSTM", "transformer": "Transformer",
         "vae": "VAE", "gan": "WGAN-GP", "kingan": "Kin-WGAN", "kinnoise": "Kin-WGAN+noise", "kinland": "Kin-WGAN+noise+coast"}
D = g.load()
R = {k: D[k] for k in ["train", "val", "test"]}
MM = {k: D[k + "_mmsi"] for k in ["train", "val", "test"]}
OUT = f"{ROOT}/results/"


def dump_merge(res, fn):
    p = OUT + fn
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(res)
    json.dump(old, open(p, "w"), indent=1)


def load_gen(name):
    s = np.load(f"{ROOT}/samples/{name}.npz")
    return {k: s[k] for k in ["train", "val", "test"]}


_FC = {}


def feats(key, w):
    if key not in _FC:
        _FC[key] = g.features(w)[0]
    return _FC[key]


def tpr_at(y, s, fpr=0.01):
    f, t, _ = roc_curve(y, s)
    return float(np.interp(fpr, f, t))


def cluster_boot(y, s, clusters, B=300, seed=0):
    rng = np.random.default_rng(seed)
    uc = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uc}
    aucs = []
    for _ in range(B):
        pick = rng.choice(uc, len(uc), replace=True)
        ii = np.concatenate([idx_by[c] for c in pick])
        if len(np.unique(y[ii])) < 2: continue
        aucs.append(roc_auc_score(y[ii], s[ii]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ------------------------------------------------------------------ realism
def realism():
    rng = np.random.default_rng(0)
    test_xy = g.to_xy(R["test"])[..., :2]
    Ftr, groups = g.features(R["train"])
    kin_cols = groups["A_reported"] + groups["B_position"] + groups["C_consistency"]
    sc = StandardScaler().fit(Ftr[:, kin_cols])
    Fte = sc.transform(feats("real_test", R["test"])[:, kin_cols])

    def per_step(w):
        xy = g.to_xy(w)
        d = np.hypot(np.diff(xy[..., 0], axis=1), np.diff(xy[..., 1], axis=1))
        return xy[..., 2].ravel(), np.abs(g.wrap(np.diff(xy[..., 3], axis=1))).ravel(), (d / g.DT_H / g.KN2KMH).ravel()

    ref_sog, ref_turn, ref_v = per_step(R["test"])
    res = {}
    for name in (["real"] if "real" not in os.environ.get("SKIP", "") else []) + GENS:
        w = R["val"] if name == "real" else load_gen(name)["test"]
        q = rng.choice(len(w), 500, replace=False)
        Q = g.to_xy(w[q])[..., :2]
        r = {}
        r["dtw_abs"] = float(np.mean(g.nn_distance(Q, test_xy, "dtw")))
        r["frechet_abs"] = float(np.mean(g.nn_distance(Q, test_xy, "frechet")))
        r["dtw_shape"] = float(np.mean(g.nn_distance(Q, test_xy, "dtw", shape_only=True)))
        r["frechet_shape"] = float(np.mean(g.nn_distance(Q, test_xy, "frechet", shape_only=True)))
        Fg = sc.transform(feats(f"{name}_test" if name != "real" else "real_val", w)[:, kin_cols])
        r["mmd"] = g.mmd_rbf(Fg, Fte)
        s, t, v = per_step(w)
        r["w1_sog"] = float(wasserstein_distance(s, ref_sog))
        r["w1_turn"] = float(wasserstein_distance(t, ref_turn))
        r["w1_vimp"] = float(wasserstein_distance(v, ref_v))
        r.update(g.violations(w))
        res[name] = r
        print(name, {k: round(x, 4) for k, x in r.items()}, flush=True)
    dump_merge(res, "realism.json")


# ------------------------------------------------------------------ detectors
def make_split(name):
    G = load_gen(name)
    out = {}
    for k in ["train", "val", "test"]:
        Fr = feats(f"real_{k}", R[k]); Fg = feats(f"{name}_{k}", G[k])
        out[k] = dict(F=np.vstack([Fr, Fg]), W=np.concatenate([R[k], G[k]]),
                      y=np.r_[np.zeros(len(Fr)), np.ones(len(Fg))],
                      c=np.r_[MM[k], MM[k]])  # ghost windows clustered with their seed vessel
    return out


_, GROUPS = g.features(R["train"][:5])
ALLC = sum(GROUPS.values(), [])


def det_rf(S, cols, seed):
    m = RandomForestClassifier(300, min_samples_leaf=2, n_jobs=2, random_state=seed,
                               class_weight="balanced").fit(S["train"]["F"][:, cols], S["train"]["y"])
    return m.predict_proba(S["test"]["F"][:, cols])[:, 1], m


def det_svm(S, cols, seed):
    sc = StandardScaler().fit(S["train"]["F"][:, cols])
    m = LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=20000).fit(
        sc.transform(S["train"]["F"][:, cols]), S["train"]["y"])
    return m.decision_function(sc.transform(S["test"]["F"][:, cols]))


def det_iforest(S, cols, seed):
    Fr = feats("real_train", R["train"])[:, cols]
    sc = StandardScaler().fit(Fr)
    m = IsolationForest(n_estimators=300, random_state=seed).fit(sc.transform(Fr))
    return -m.score_samples(sc.transform(S["test"]["F"][:, cols]))


RULE_STATS = ["resv_absmed", "ang_median", "vimp_max", "turn_absmax", "land_frac", "dsog_absmax"]


def det_rule(S, *_):
    """Physics/plausibility rule detector: max over statistics of the real-data empirical CDF."""
    idx = [g.FEATURE_NAMES.index(n) for n in RULE_STATS]
    Fr = feats("real_train", R["train"])[:, idx]
    Ft = S["test"]["F"][:, idx]
    cdf = np.stack([np.searchsorted(np.sort(Fr[:, j]), Ft[:, j], side="left") / len(Fr) for j in range(len(idx))], 1)
    return cdf.max(1)


NORM = Norm(*g.encode(R["train"]))


def seq_tensors(W):
    s0, st = g.encode(W)
    return torch.tensor(NORM.ns(s0)), torch.tensor(NORM.nt(st))


def train_cnn(Wtr, ytr, Wva, yva, seed, epochs=25):
    torch.manual_seed(seed)
    m = ConvNet()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    s0, st = seq_tensors(Wtr); y = torch.tensor(ytr, dtype=torch.float32)
    v0, vt = seq_tensors(Wva)
    best, state, bad = -1, None, 0
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        m.train()
        for i in np.array_split(rng.permutation(len(y)), max(1, len(y) // 256)):
            loss = nn.functional.binary_cross_entropy_with_logits(m(s0[i], st[i]), y[i])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            a = roc_auc_score(yva, m(v0, vt).numpy())
        if a > best + 1e-4:
            best, state, bad = a, {k: x.clone() for k, x in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4: break
    m.load_state_dict(state); m.eval()
    return m


def det_cnn(S, cols, seed):
    m = train_cnn(S["train"]["W"], S["train"]["y"], S["val"]["W"], S["val"]["y"], seed)
    t0, tt = seq_tensors(S["test"]["W"])
    with torch.no_grad():
        return m(t0, tt).numpy()


DETS = {"RF": det_rf, "LinSVM": det_svm, "1D-CNN": det_cnn, "IsoForest": det_iforest, "Rule": det_rule}


def detect():
    res = {}
    for name in GENS:
        S = make_split(name)
        y, c = S["test"]["y"], S["test"]["c"]
        res[name] = {}
        for dn, fn in DETS.items():
            seeds = [0, 1, 2] if dn in ["RF", "LinSVM", "1D-CNN"] else [0]
            aucs, tprs = [], []
            for sd in seeds:
                s = fn(S, ALLC, sd)
                s = s[0] if isinstance(s, tuple) else s
                aucs.append(roc_auc_score(y, s)); tprs.append(tpr_at(y, s))
                if sd == 0: ci = cluster_boot(y, s, c)
            res[name][dn] = dict(auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)), ci=ci,
                                 tpr1=float(np.mean(tprs)), tpr1_sd=float(np.std(tprs)))
            print(name, dn, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name][dn].items()}, flush=True)
        dump_merge(res, "detect.json")


def ablate():
    sets = {"A": GROUPS["A_reported"], "B": GROUPS["B_position"], "C": GROUPS["C_consistency"],
            "D": GROUPS["D_geographic"], "A+B": GROUPS["A_reported"] + GROUPS["B_position"],
            "A+B+C": GROUPS["A_reported"] + GROUPS["B_position"] + GROUPS["C_consistency"], "All": ALLC}
    res = {}
    for name in GENS:
        S = make_split(name)
        res[name] = {}
        for sn, cols in sets.items():
            s, m = det_rf(S, cols, 0)
            res[name][sn] = float(roc_auc_score(S["test"]["y"], s))
            if sn == "All":
                imp = m.feature_importances_
                top = np.argsort(-imp)[:5]
                res[name]["top_features"] = [(g.FEATURE_NAMES[cols[i]], float(imp[i])) for i in top]
        print(name, res[name], flush=True)
    dump_merge(res, "ablation.json")


def cross():
    """train RF on generator i, test on generator j (kinematic features A+B+C, no geography)."""
    cols = GROUPS["A_reported"] + GROUPS["B_position"] + GROUPS["C_consistency"]
    Ss = {n: make_split(n) for n in GENS}
    res = {}
    for i in GENS:
        m = RandomForestClassifier(300, min_samples_leaf=2, n_jobs=2, random_state=0,
                                   class_weight="balanced").fit(Ss[i]["train"]["F"][:, cols], Ss[i]["train"]["y"])
        res[i] = {j: float(roc_auc_score(Ss[j]["test"]["y"], m.predict_proba(Ss[j]["test"]["F"][:, cols])[:, 1]))
                  for j in GENS}
        print(i, {k: round(v, 3) for k, v in res[i].items()}, flush=True)
    json.dump(res, open(OUT + "cross.json", "w"), indent=1)


if __name__ == "__main__":
    {"realism": realism, "detect": detect, "ablate": ablate, "cross": cross}[sys.argv[1]]()
