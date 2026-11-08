"""Train ghost-vessel generators on the training partition and generate ghost windows.
Usage: python gen.py <lstm|transformer|vae|gan|kingan|baselines>"""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for _folder in ("samples", "results", "paper", "paper/figs"):
    (ROOT / _folder).mkdir(parents=True, exist_ok=True)
import sys, time, json, numpy as np, torch
import gvlib as g
from models import *

name = sys.argv[1]
SEED = 7
torch.manual_seed(SEED); np.random.seed(SEED)
D = g.load()
enc = {k: g.encode(D[k]) for k in ["train", "val", "test"]}
norm = Norm(*enc["train"])
T = enc["train"][1].shape[1]
tt = lambda a: torch.tensor(a, dtype=torch.float32)
S0 = {k: tt(norm.ns(enc[k][0])) for k in enc}
ST = {k: tt(norm.nt(enc[k][1])) for k in enc}
log = {"model": name}
t0 = time.time()


def batches(n, bs, rng):
    idx = rng.permutation(n)
    for i in range(0, n - bs + 1, bs):
        yield idx[i:i + bs]


def gen_all(model, temp=1.0):
    out = {}
    model.eval()
    for k in ["train", "val", "test"]:
        xs = []
        with torch.no_grad():
            for i in range(0, len(S0[k]), 1024):
                xs.append(model.sample(S0[k][i:i + 1024], T, temp).numpy())
        st = norm.dt(np.concatenate(xs))
        s0 = enc[k][0]
        out[k] = g.decode(s0, st)
    return out


rng = np.random.default_rng(SEED)
if name in ["lstm", "transformer"]:
    m = LSTMAR() if name == "lstm" else TransAR(T=T)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    best, bad, state = 1e9, 0, None
    for ep in range(60):
        m.train()
        for b in batches(len(S0["train"]), 128, rng):
            mu, ls = m(S0["train"][b], ST["train"][b])
            loss = gauss_nll(mu, ls, ST["train"][b])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            mu, ls = m(S0["val"], ST["val"]); v = gauss_nll(mu, ls, ST["val"]).item()
        print(ep, round(loss.item(), 4), round(v, 4), flush=True)
        if v < best - 1e-4:
            best, bad, state = v, 0, {k: x.clone() for k, x in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 6: break
    m.load_state_dict(state); log.update(val_nll=best, epochs=ep + 1)
    torch.save(m.state_dict(), f"{ROOT}/ckpt_{name}.pt")
    samples = gen_all(m)

elif name == "vae":
    m = SeqVAE(T=T)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    best, bad, state, step = 1e9, 0, None, 0
    for ep in range(80):
        m.train()
        for b in batches(len(S0["train"]), 128, rng):
            x = ST["train"][b]
            rec, mu, lv = m(S0["train"][b], x)
            beta = min(1.0, step / 3000); step += 1
            nll = (m.logsig + 0.5 * ((x - rec) / torch.exp(m.logsig)) ** 2).mean()
            kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp())).sum(1).mean() / (T * 5)
            loss = nll + beta * kl
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            x = ST["val"]; rec, mu, lv = m(S0["val"], x)
            v = ((m.logsig + 0.5 * ((x - rec) / torch.exp(m.logsig)) ** 2).mean()
                 + (-0.5 * (1 + lv - mu ** 2 - lv.exp())).sum(1).mean() / (T * 5)).item()
        print(ep, round(loss.item(), 4), round(v, 4), flush=True)
        if ep > 10 and v < best - 1e-4:
            best, bad, state = v, 0, {k: x.clone() for k, x in m.state_dict().items()}
        elif ep > 10:
            bad += 1
            if bad >= 6: break
    m.load_state_dict(state); log.update(val_elbo=best, epochs=ep + 1)
    torch.save(m.state_dict(), f"{ROOT}/ckpt_{name}.pt")
    samples = gen_all(m)

elif name in ["gan", "kingan"]:
    G = GenGRU(norm, T=T, kin=(name == "kingan")); Dn = ConvNet()
    oG = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.0, 0.9))
    oD = torch.optim.Adam(Dn.parameters(), 1e-4, betas=(0.0, 0.9))
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    n = len(S0["train"])
    for it in range(iters):
        for _ in range(3):
            b = rng.integers(0, n, 128)
            s0, real = S0["train"][b], ST["train"][b]
            fake = G(s0).detach()
            lD = Dn(s0, fake).mean() - Dn(s0, real).mean() + 10 * gradient_penalty(Dn, s0, real, fake)
            oD.zero_grad(); lD.backward(); oD.step()
        b = rng.integers(0, n, 128)
        lG = -Dn(S0["train"][b], G(S0["train"][b])).mean()
        oG.zero_grad(); lG.backward(); oG.step()
        if it % 200 == 0:
            print(it, round(lD.item(), 4), round(lG.item(), 4), round(time.time() - t0), flush=True)
    torch.save({"G": G.state_dict(), "D": Dn.state_dict()}, f"{ROOT}/ckpt_{name}.pt")
    log.update(iters=iters)
    samples = gen_all(G)

elif name == "baselines":
    out_r, out_c = {}, {}
    tr_w = D["train"]; tr_m = D["train_mmsi"]
    for k in ["train", "val", "test"]:
        s0 = enc[k][0]
        # (a) replay-with-offset: a real training trajectory of ANOTHER vessel, translated to the seed start
        idx = rng.integers(0, len(tr_w), len(s0))
        s0r, str_ = g.encode(tr_w[idx])
        s0r[:, :2] = s0[:, :2]
        out_r[k] = g.decode(s0r, str_)
        # (b) constant-velocity spoof with small noise, kinematically integrated
        n = len(s0)
        sog = np.clip(s0[:, None, 2] + rng.normal(0, 0.3, (n, T)), 0, None)
        cog = np.rad2deg(np.arctan2(s0[:, 3], s0[:, 4]))[:, None] + rng.normal(0, 2.0, (n, T))
        c = np.deg2rad(cog)
        dist = sog * KMSTEP
        st = np.stack([dist * np.sin(c), dist * np.cos(c), sog, np.sin(c), np.cos(c)], -1).astype(np.float32)
        out_c[k] = g.decode(s0, st)
    np.savez_compressed(f"{ROOT}/samples/replay.npz", **out_r)
    np.savez_compressed(f"{ROOT}/samples/constvel.npz", **out_c)
    print("baselines done"); sys.exit()

log["train_seconds"] = round(time.time() - t0)
np.savez_compressed(f"{ROOT}/samples/{name}.npz", **samples)
json.dump(log, open(f"{ROOT}/results/train_{name}.json", "w"))
print(log)
