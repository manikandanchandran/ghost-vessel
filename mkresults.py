"""Turn result JSON files into LaTeX macros and tables so every number in the paper is traceable."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for _folder in ("samples", "results", "paper", "paper/figs"):
    (ROOT / _folder).mkdir(parents=True, exist_ok=True)
import json, os, numpy as np

R = f"{ROOT}/results/"
P = f"{ROOT}/paper/"
GENS = ["replay", "constvel", "lstm", "transformer", "vae", "gan", "kingan", "kinnoise", "kinland"]
LAB = {"real": "Real (val.\\ vs.\\ test)", "replay": "Replay+offset", "constvel": "Const.\\ velocity", "lstm": "LSTM",
       "transformer": "Transformer", "vae": "VAE", "gan": "WGAN-GP", "kingan": "Kin-WGAN", "kinnoise": "Kin-WGAN+noise", "kinland": "Kin-WGAN+noise+coast"}
M = []


DIG = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def fmt(key, v):
    if key.startswith(("tpr", "land", "resv", "speed")):
        return f"{100*v:.1f}"
    if key.startswith("dtw"):
        return f"{v:.2f}"
    return f"{v:.3f}"


def mac(name, val):
    name = "".join(DIG.get(c, c) for c in name)
    M.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def f(x, d=2):
    return f"{x:.{d}f}"


def load(n):
    p = R + n
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------- dataset
ds = load("dataset_stats.json")
gl = ds["global"]
mac("NTracks", f"{gl['tracks']:,}".replace(",", "{,}"))
mac("NVessels", f"{gl['vessels']:,}".replace(",", "{,}"))
mac("NPoints", f"{gl['points']:,}".replace(",", "{,}"))
mac("DataStart", "1~January~2019"); mac("DataEnd", "3~December~2019")
for k, pre in [("train", "Tr"), ("val", "Va"), ("test", "Te")]:
    s = ds[k]
    mac(pre + "Ves", f"{s['vessels']:,}".replace(",", "{,}"))
    mac(pre + "Win", f"{s['windows']:,}".replace(",", "{,}"))
    mac(pre + "Sog", f"{s['sog_mean']:.1f} $\\pm$ {s['sog_std']:.1f}")
    mac(pre + "Mix", f"{100*s['frac_slow']:.0f} / {100*s['frac_mid']:.0f} / {100*s['frac_fast']:.0f}")
tt = [json.load(open(R + f"train_{n}.json"))["train_seconds"] for n in ["lstm", "transformer", "vae", "gan", "kingan"]]
mac("TrainMinMin", f"{min(tt)/60:.0f}"); mac("TrainMaxMin", f"{max(tt)/60:.0f}")

# ---------------- budget
b = load("budget.json")
TAGS = {"180": "Moor", "10": "Ten", "6": "Six", "3.33": "Third", "2": "Two"}
for ri, t in TAGS.items():
    pi = b["per_interval"][ri]
    for var, suf in [("etb32", ""), ("etb64", "Sixfour")]:
        e = pi[var]
        mac("Slots" + t + suf, f(e["extra_slots_min"]))
        mac("Ovh" + t + suf, f"{e['overhead_pct']:.1f}")
        mac("ETBBits" + t + suf, str(e["bits"]))
        mac("ETBSlots" + t + suf, str(e["slots"]))
        mac("ETBEntries" + t + suf, str(e["entries"]))
        mac("ETBDelta" + t + suf, f"{e['delta_s']:g}" if e["delta_s"] == int(e["delta_s"]) else f"{e['delta_s']:.1f}")
        mac("Latency" + t + suf, f"{e['latency_s']:.0f}")
mac("CommitBits", str(b["commit_bits"])); mac("CommitSlots", str(b["commit_slots"]))
mac("CommitFragOne", str(b["commit_fragments"][0])); mac("CommitFragTwo", str(b["commit_fragments"][1]))
hs = b["handshake_bytes"]
mac("HSTotal", str(hs["total"])); mac("HSCH", str(hs["ClientHello"])); mac("HSSH", str(hs["ServerHello"]))
mac("HSVF", str(hs["VesselFinished"])); mac("HSCert", str(hs["compact_cert"]))
sv = b["simultaneous_cargo_tanker_vessels"]
mac("SimMed", f"{sv['median']:.0f}"); mac("SimPninetyfive", f"{sv['p95']:.0f}"); mac("SimMax", str(sv["max"]))
mac("VerifyLatency", f"{b['per_interval']['10']['etb32']['latency_s']:.0f}")
al = b["alt10"]
mac("SlotsTenLong", f(al["extra_slots_min"])); mac("OvhTenLong", f"{al['overhead_pct']:.1f}")
mac("VerifyLatencyLong", f"{al['latency_s']:.0f}"); mac("DeltaTenLong", f"{al['delta_s']:.0f}")
# Table X rows (all Class A reporting intervals of ITU-R M.1371-6, Table 1)
order = ["180", "10", "6", "3.33", "2"]
labels = {"sig": "Per-report signature (512\\,b, 3-slot Msg~8)", "mac": "Per-report MAC, shared key (Msg~25)\\textsuperscript{a}",
          "etb32": "\\textbf{ETB, 32-bit tags (proposed)}", "etb64": "ETB, 64-bit tags"}
rows = ["Position reports only (slots/min) & " + " & ".join(
    (f"{60/float(k if k != '3.33' else 10/3):.2f}" if k == "180" else f"{60/(10/3 if k == '3.33' else float(k)):.0f}") for k in order) + " \\\\",
        "\\midrule"]
for key, lab in labels.items():
    cells = []
    for k in order:
        x = [s for s in b["per_interval"][k]["schemes"] if s["key"] == key][0]
        c = f"+{x['extra']:.2f} ({x['overhead_pct']:.0f}\\,\\%)" if key in ["sig", "mac"] else f"+{x['extra']:.2f} ({x['overhead_pct']:.1f}\\,\\%)"
        if not x["ratdma_ok"]:
            c += "\\textsuperscript{b}"
        if key == "etb32":
            c = "\\textbf{" + c + "}"
        cells.append(c)
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
mac("BudgetRows", "\n".join(rows))
# Table VIII rows
erows = []
for k, lab in [("10", "10\\,s"), ("6", "6\\,s"), ("3.33", "3$\\frac{1}{3}$\\,s"), ("2", "2\\,s"), ("180", "3\\,min")]:
    e = b["per_interval"][k]["etb32"]
    d = f"{e['delta_s']:g}"
    erows.append(f"{lab} reporting, $\\Delta={d}$\\,s: {e['entries']} report{'s' if e['entries'] > 1 else ''} & {e['bits']} ({e['slots']} slots, cap.\\ {b['msg8_app'][str(e['slots'])]}) \\\\")
mac("EtbRows", "\n".join(erows))
lbl = {"legacy": "No authentication", "sig": "Per-report signature", "mac": "Per-report MAC (shared key)",
       "etb32": "\\textbf{ETB, 32-bit tags (proposed)}", "etb64": "ETB, 64-bit tags"}
rows = []
for key, lab in lbl.items():
    vals = [r[key] for r in b["vdl_load_pct_10s"]]
    cells = [(f"{v:.1f}" if v <= 100 else f"\\textit{{{v:.0f}}}") for v in vals]
    rows.append(f"{lab} & " + " & ".join(cells) + f" & {b['max_vessels_10s'][key]['at50']} \\\\")
mac("LoadRows", "\n".join(rows))
L250 = [r for r in b["vdl_load_pct_10s"] if r["N"] == 250][0]
mac("LoadLegacyTwoFifty", f"{L250['legacy']:.1f}")
mac("LoadETBTwoFifty", f"{L250['etb32']:.1f}")
mac("MaxFiftyETB", str(b["max_vessels_10s"]["etb32"]["at50"]))
mac("MaxFiftyETBSixfour", str(b["max_vessels_10s"]["etb64"]["at50"]))
mac("MaxFiftyLegacy", str(b["max_vessels_10s"]["legacy"]["at50"]))
fo = b["forgery"]
import math
# exponents rounded UP so that "value < 2^x" is always true
mac("ForgeKeyLog", f"{math.ceil(math.log2(fo['any_report_per_epoch_key_nocap'])):d}")
mac("ForgeChainLog", f"{math.ceil(math.log2(fo['any_report_per_chain'])):d}")
mac("ForgeTrackLog", f"{math.ceil(math.log2(fo['fabricated_track_per_chain'])):d}")
mac("FailCapPlusOne", str(fo["fail_cap_per_chain"] + 1))
mac("FailCap", str(fo["fail_cap_per_chain"]))

# ---------------- realism table
re_ = load("realism.json")
if re_:
    lines = []
    for k in ["real"] + [x for x in GENS if x in re_]:
        r = re_[k]
        lines.append(f"{LAB[k]} & {f(r['dtw_abs'])} & {f(r['frechet_abs'],1)} & {f(r['dtw_shape'])} & {f(r['frechet_shape'],1)} & "
                     f"{f(r['mmd'],3)} & {f(r['w1_sog'])} & {f(r['w1_turn'])} & {100*r['resv_gt5']:.2f} & {100*r['land_windows']:.1f} \\\\"
                     + ("\\midrule" if k == "real" else ""))
    mac("RealismRows", "\n".join(lines))
    for k in ["real"] + [x for x in GENS if x in re_]:
        for key in ["dtw_abs", "dtw_shape", "mmd", "land_windows", "resv_gt5", "frechet_shape"]:
            v = re_[k][key]
            nm = "Re" + k.capitalize().replace("Kingan", "Kin").replace("Constvel", "Cv") + key.replace("_", "").capitalize()
            mac(nm, f(100 * v, 1) if key in ["land_windows", "resv_gt5"] else f(v, 3 if key == "mmd" else 2))

# ---------------- detection table
de = load("detect.json")
DETS = ["RF", "LinSVM", "1D-CNN", "IsoForest", "Rule"]
if de:
    lines = []
    for k in GENS:
        if k not in de: continue
        cells = []
        for d in DETS:
            r = de[k][d]
            cells.append(f"{r['auc']:.3f}")
        for d in ["RF", "Rule"]:
            cells.append(f"{100*de[k][d]['tpr1']:.1f}")
        ci = de[k]["RF"]["ci"]
        lines.append(f"{LAB[k]} & " + " & ".join(cells) + f" & [{ci[0]:.3f}, {ci[1]:.3f}] \\\\")
    mac("DetectRows", "\n".join(lines))
    for k in de:
        for d in DETS:
            nm = "De" + k.capitalize().replace("Kingan", "Kin").replace("Constvel", "Cv") + d.replace("-", "").replace("1D", "Cnn").replace("LinSVM", "Svm").replace("IsoForest", "Iso")
            mac(nm + "Auc", f"{de[k][d]['auc']:.3f}")
            mac(nm + "Tpr", f"{100*de[k][d]['tpr1']:.1f}")
            mac(nm + "Sd", f"{de[k][d]['auc_sd']:.3f}")

# ---------------- ablation
ab = load("ablation.json")
SETS = ["A", "B", "C", "D", "A+B+C", "All"]
if ab:
    lines = []
    for k in GENS:
        if k not in ab: continue
        lines.append(f"{LAB[k]} & " + " & ".join(f"{ab[k][s]:.3f}" for s in SETS) + " \\\\")
    mac("AblateRows", "\n".join(lines))
    for k in ab:
        for s in SETS:
            nm = "Ab" + k.capitalize().replace("Kingan", "Kin").replace("Constvel", "Cv") + s.replace("+", "")
            mac(nm, f"{ab[k][s]:.3f}")
        mac("Ab" + k.capitalize().replace("Kingan", "Kin").replace("Constvel", "Cv") + "TopFeat",
            ab[k]["top_features"][0][0].replace("_", "\\_"))

# ---------------- cross-generator
cr = load("cross.json")
if cr:
    short = {"replay": "Rep", "constvel": "CV", "lstm": "LSTM", "transformer": "Trf", "vae": "VAE", "gan": "GAN", "kingan": "Kin", "kinnoise": "Kin+N"}
    lines = []
    for i in GENS:
        if i not in cr: continue
        cells = []
        for j in [x for x in GENS if x in cr[i]]:
            v = cr[i][j]
            s = f"{v:.2f}"
            cells.append(f"\\textbf{{{s}}}" if i == j else s)
        lines.append(f"{short[i]} & " + " & ".join(cells) + " \\\\")
    mac("CrossRows", "\n".join(lines))
    off = [cr[i][j] for i in cr for j in cr[i] if i != j]
    mac("CrossOffMean", f"{np.mean(off):.2f}")

# ---------------- adaptive
sw = load("adapt_sweep_gan.json")
if sw:
    lines = []
    for lam, r in sw.items():
        lines.append(f"{float(lam):g} & {r['auc_surrogate_frozen']:.3f} & {r['auc_rf_static']:.3f} & {r['auc_rf_retrained']:.3f} & "
                     f"{r['auc_rf_retrained_kin']:.3f} & {r['auc_cnn_retrained']:.3f} & {r['auc_rule']:.3f} & {r['dtw_shape']:.2f} & {r['mmd']:.3f} & {100*r['land_windows']:.0f} \\\\")
    mac("AdaptRows", "\n".join(lines))
    for lam, r in sw.items():
        t = {"0.0": "Zero", "0.3": "Pthree", "1.0": "One", "3.0": "Three", "10.0": "Ten"}[lam]
        for key in r:
            mac("Sw" + t + key.replace("_", ""), fmt(key, r[key]))
ga = load("adapt_game_gan.json")
if ga:
    for rd, r in ga.items():
        t = ["Zero", "One", "Two", "Three", "Four"][int(rd)]
        for key in r:
            mac("Ga" + t + key.replace("_", ""), fmt(key, r[key]))

if ga:
    rounds = [r for k, r in ga.items() if k != "0"]
    mac("GaSurMin", f"{min(r['auc_surrogate_frozen'] for r in rounds):.2f}")
    mac("GaSurMax", f"{max(r['auc_surrogate_frozen'] for r in rounds):.2f}")
    mac("GaRfMin", f"{min(min(r['auc_rf_retrained'], r['auc_rf_retrained_kin']) for r in ga.values()):.3f}")
    mac("GaLandMin", f"{100*min(r['land_windows'] for r in ga.values()):.1f}")
    mac("GaLandMax", f"{100*max(r['land_windows'] for r in ga.values()):.1f}")
    mac("GaCnnMin", f"{min(r['auc_cnn_retrained'] for r in rounds):.3f}")
# ---------------- validation-threshold operating points
opp = load("oppoint.json")
if opp:
    rows = []
    for k in GENS:
        if k not in opp: continue
        o = opp[k]
        rows.append(f"{LAB[k]} & {100*o['RF']['tpr']:.1f} ({100*o['RF']['fpr']:.2f}) & {100*o['IsoForest']['tpr']:.1f} & {100*o['Rule']['tpr']:.1f} & {100*o['Land']['tpr']:.1f} & {100*o['RuleLand']['tpr']:.1f} \\\\")
        nm = "Op" + k.capitalize().replace("Kingan", "Kin").replace("Constvel", "Cv")
        for d, s in [("RF", "RF"), ("IsoForest", "Iso"), ("Rule", "Rule"), ("Land", "Land"), ("RuleLand", "RuleLand")]:
            mac(nm + s + "Tpr", f"{100*o[d]['tpr']:.1f}")
            mac(nm + s + "Fpr", f"{100*o[d]['fpr']:.2f}")
    mac("OppRows", "\n".join(rows))
    learned = ["lstm", "transformer", "vae", "gan", "kingan", "kinnoise"]
    mac("OpLearnedRFTprMin", f"{100*min(opp[k]['RF']['tpr'] for k in learned):.1f}")
    mac("OpLearnedRFFprMin", f"{100*min(opp[k]['RF']['fpr'] for k in learned):.2f}")
    mac("OpLearnedRFFprMax", f"{100*max(opp[k]['RF']['fpr'] for k in learned):.2f}")
    mac("OpIsoFpr", f"{100*opp['lstm']['IsoForest']['fpr']:.2f}")
    mac("OpRuleFpr", f"{100*opp['lstm']['Rule']['fpr']:.2f}")
    mac("OpLandFpr", f"{100*opp['lstm']['Land']['fpr']:.2f}")
    mac("OpRuleLandFpr", f"{100*opp['lstm']['RuleLand']['fpr']:.2f}")
    nos = ["IsoForest", "Rule", "Land", "RuleLand"]
    if "kinland" in opp:
        mac("OpKinlandBest", f"{100*max(opp['kinland'][d]['tpr'] for d in nos):.1f}")
    mac("OpKinnoiseBest", f"{100*max(opp['kinnoise'][d]['tpr'] for d in nos):.1f}")
dr = load("dataset_reconcile.json")
if dr:
    mac("NVesselsWin", f"{dr['vessels_with_windows']:,}".replace(",", "{,}"))
    mac("NVesselsExcl", str(dr["vessels_excluded"]))
    mac("ExclLenMin", str(dr["excluded_max_voyage_len_min"])); mac("ExclLenMax", str(dr["excluded_max_voyage_len_max"]))
    mac("NShortVoy", f"{dr['voyages_shorter_than_window']:,}".replace(",", "{,}"))
rc = load("rule_calibration.json")
if rc:
    mac("RuleCalOne", f"{100*rc['land_cdf_one_point']:.1f}")
    mac("RuleThr", f"{100*rc['rule_threshold']:.1f}")
    mac("TrainLandPct", f"{100*rc['real_train_land_frac']:.1f}")
open(P + "results.tex", "w").write("% AUTO-GENERATED by mkresults.py from results/*.json -- do not edit\n" + "\n".join(M) + "\n")
print(len(M), "macros")
