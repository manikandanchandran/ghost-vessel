"""Operational detection. Statistical thresholds use REAL validation data (target 1% FPR).
Coastline is a fixed binary point-or-segment intersection rule, with *observed* FPR.
The combined detector ORs that rule with a validation-calibrated physics rule.
A fixed geographical predicate cannot be threshold-tuned to an arbitrary 1% FPR."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for _folder in ("samples", "results", "paper", "paper/figs"):
    (ROOT / _folder).mkdir(parents=True, exist_ok=True)
import json, os, numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import gvlib as g
import evaluate as E

TARGET = 0.01
OUT = f"{ROOT}/results/oppoint.json"


def op(val_real, test_real, test_ghost):
    thr = np.quantile(val_real, 1 - TARGET)
    return dict(thr=float(thr), tpr=float(np.mean(test_ghost > thr)), fpr=float(np.mean(test_real > thr)))


def rule_scores(Fr_train, F):
    idx = [g.FEATURE_NAMES.index(n) for n in E.RULE_STATS]
    ref = np.sort(Fr_train[:, idx], 0)
    X = F[:, idx]
    return np.stack([np.searchsorted(ref[:, j], X[:, j], side="left") / len(ref) for j in range(len(idx))], 1).max(1)


def coastline_flag(w):
    """Binary geographical rule: detect any point or segment intersecting land."""
    return g.coastline_intersections(w)


def binary_coast_operating_point(val_real, test_real, test_ghost):
    """Coastline has a fixed geometrical decision rule, not a tunable score.
    Report its observed validation and test false-positive rates transparently.
    """
    return dict(thr=0.5, tpr=float(np.mean(test_ghost)),
                fpr=float(np.mean(test_real)), validation_fpr=float(np.mean(val_real)),
                threshold_method="fixed any-intersection rule")


def combined_operating_point(val_rule, test_rule, ghost_rule, val_coast, test_coast, ghost_coast):
    """OR between 99th-percentile validation-calibrated physics rule and coastline rule."""
    phys_threshold = float(np.quantile(val_rule, 1 - TARGET))
    return dict(thr=phys_threshold,
                tpr=float(np.mean((ghost_rule > phys_threshold) | ghost_coast)),
                fpr=float(np.mean((test_rule > phys_threshold) | test_coast)),
                validation_fpr=float(np.mean((val_rule > phys_threshold) | val_coast)),
                threshold_method="physics validation percentile OR fixed coastline")


Frt = E.feats("real_train", E.R["train"])
Frv = E.feats("real_val", E.R["val"])
Fre = E.feats("real_test", E.R["test"])
cols = E.ALLC
# unsupervised detectors (independent of generator)
sc = StandardScaler().fit(Frt[:, cols])
iso = IsolationForest(n_estimators=300, random_state=0).fit(sc.transform(Frt[:, cols]))
iso_s = lambda F: -iso.score_samples(sc.transform(F[:, cols]))
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
# The coastline rule is independent of the generator, so compute it just once.
coast_val = coastline_flag(E.R["val"])
coast_test = coastline_flag(E.R["test"])
rule_val = rule_scores(Frt, Frv)
rule_test = rule_scores(Frt, Fre)
SKIP_RF = os.environ.get("SKIP_RF") == "1"
for name in os.environ.get("GENS", ",".join(E.ALL_GENS)).split(","):
    G = E.load_gen(name)
    Fg = {k: g.features(G[k])[0] for k in ["train", "val", "test"]}
    r = dict(res.get(name, {}))
    rf = []
    for seed in ([] if SKIP_RF else [0, 1, 2]):
        m = RandomForestClassifier(300, min_samples_leaf=2, n_jobs=2, random_state=seed, class_weight="balanced").fit(
            np.vstack([Frt, Fg["train"]])[:, cols], np.r_[np.zeros(len(Frt)), np.ones(len(Fg["train"]))])
        p = lambda F: m.predict_proba(F[:, cols])[:, 1]
        rf.append(op(p(Frv), p(Fre), p(Fg["test"])))
    if rf:
        r["RF"] = {k: float(np.mean([x[k] for x in rf])) for k in ["tpr", "fpr"]}
    r["IsoForest"] = op(iso_s(Frv), iso_s(Fre), iso_s(Fg["test"]))
    rule_ghost = rule_scores(Frt, Fg["test"])
    coast_ghost = coastline_flag(G["test"])
    r["Rule"] = op(rule_val, rule_test, rule_ghost)
    r["Land"] = binary_coast_operating_point(coast_val, coast_test, coast_ghost)
    r["RuleLand"] = combined_operating_point(rule_val, rule_test, rule_ghost,
                                               coast_val, coast_test, coast_ghost)
    res[name] = r
    print(name, {d: (round(100 * v["tpr"], 1), round(100 * v["fpr"], 2)) for d, v in r.items()}, flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
