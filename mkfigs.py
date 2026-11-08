"""Figures for the manuscript."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for _folder in ("samples", "results", "paper", "paper/figs"):
    (ROOT / _folder).mkdir(parents=True, exist_ok=True)
import json, pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gvlib as g

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
OUT = f"{ROOT}/paper/figs/"
NAMES = [("real", "Real (test)"), ("replay", "Replay+offset"), ("constvel", "Const.-velocity"), ("lstm", "LSTM"),
         ("transformer", "Transformer"), ("vae", "VAE"), ("gan", "WGAN-GP"), ("kingan", "Kin-WGAN")]


def trajectories():
    D = g.load()
    polys = pickle.load(open(f"{ROOT}/coast.pkl", "rb"))
    rng = np.random.default_rng(3)
    idx = rng.choice(len(D["test"]), 40, replace=False)
    fig, axs = plt.subplots(2, 4, figsize=(7.1, 3.9), sharex=True, sharey=True)
    for ax, (k, lab) in zip(axs.ravel(), NAMES):
        for p in polys:
            p = np.asarray(p)
            ax.fill(p[:, 1], p[:, 0], color="#d9d4c7", lw=0)
        w = D["test"][idx] if k == "real" else np.load(f"{ROOT}/samples/{k}.npz")["test"][idx]
        for t in w:
            ax.plot(t[:, 1], t[:, 0], lw=0.6, color="#1f4e79" if k == "real" else "#b2182b", alpha=0.8)
            ax.plot(t[0, 1], t[0, 0], "o", ms=1.4, color="k")
        ax.set_xlim(g.LON0, g.LON1); ax.set_ylim(g.LAT0, g.LAT1)
        ax.set_title(lab, fontsize=8, pad=2)
        ax.tick_params(length=2, labelsize=6.5)
        ax.set_aspect(1 / np.cos(np.deg2rad(g.LATC)))
    for ax in axs[:, 0]: ax.set_ylabel("Lat. (°N)", fontsize=7)
    for ax in axs[1]: ax.set_xlabel("Lon. (°E)", fontsize=7)
    fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.4)
    fig.savefig(OUT + "trajectories.pdf"); plt.close(fig)


def game():
    res = json.load(open(f"{ROOT}/results/adapt_game_gan.json"))
    r = sorted(res, key=int)
    x = [int(k) for k in r]
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 2.1), gridspec_kw={"width_ratios": [1.5, 1]})
    series = [("auc_surrogate_frozen", "Surrogate (targeted)", "-o"), ("auc_rf_static", "RF static (orig. ghosts)", "--x"),
              ("auc_rf_retrained", "RF retrained, all feat.", "-s"),
              ("auc_rf_retrained_kin", "RF retrained, no geography", "-^"), ("auc_cnn_retrained", "1-D CNN retrained", "-d"),
              ("auc_rule", "Physics rules", "-v")]
    for key, lab, st in series:
        ax[0].plot(x, [res[k][key] for k in r], st, ms=2.5, lw=0.9, label=lab)
    ax[0].set_xlabel("Round"); ax[0].set_ylabel("ROC-AUC"); ax[0].set_xticks(x)
    ax[0].grid(alpha=0.3, lw=0.4)
    ax[1].plot(x, [res[k]["mmd"] for k in r], "-o", ms=2.5, lw=0.9, color="k", label="MMD$^2$")
    ax[1].set_xlabel("Round"); ax[1].set_title("Ghost MMD$^2$", fontsize=7, pad=2); ax[1].set_xticks(x)
    ax[1].grid(alpha=0.3, lw=0.4)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=5.6, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.80))
    fig.savefig(OUT + "game.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        {"traj": trajectories, "game": game}[f]()
