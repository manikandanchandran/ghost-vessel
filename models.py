from pathlib import Path
ROOT = Path(__file__).resolve().parent
for _folder in ("samples", "results", "paper", "paper/figs"):
    (ROOT / _folder).mkdir(parents=True, exist_ok=True)
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

import os; torch.set_num_threads(int(os.environ.get("THREADS", "2")))
KMSTEP = 1.852 / 6.0  # km per (knot * 10 min)


class Norm:
    """Per-channel normalisation of start states and steps (dx,dy,sog; sin/cos left raw)."""

    def __init__(self, s0, st):
        self.sm = s0.mean(0); self.ss = s0.std(0) + 1e-6
        self.sm[3:], self.ss[3:] = 0, 1
        self.tm = st.reshape(-1, 5).mean(0); self.ts = st.reshape(-1, 5).std(0) + 1e-6
        self.tm[3:], self.ts[3:] = 0, 1

    def ns(self, s0): return (s0 - self.sm) / self.ss
    def nt(self, st): return (st - self.tm) / self.ts
    def ds(self, s0): return s0 * self.ss + self.sm
    def dt(self, st): return st * self.ts + self.tm

    def torch(self):
        return {k: torch.tensor(getattr(self, k), dtype=torch.float32) for k in ["sm", "ss", "tm", "ts"]}


def unit_cs(x):
    """normalise the (sin,cos) pair in channels 3:5"""
    sc = x[..., 3:5]
    sc = sc / (sc.norm(dim=-1, keepdim=True) + 1e-6)
    return torch.cat([x[..., :3], sc], -1)


def first_prev(s0n):
    z = torch.zeros_like(s0n[:, :2])
    return torch.cat([z, s0n[:, 2:]], 1)


# ---------------- autoregressive Gaussian models ------------------------------
class LSTMAR(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.rnn = nn.LSTM(10, h, 2, batch_first=True)
        self.out = nn.Linear(h, 10)

    def forward(self, s0n, stn):
        prev = torch.cat([first_prev(s0n)[:, None], stn[:, :-1]], 1)
        inp = torch.cat([prev, s0n[:, None].expand(-1, prev.shape[1], -1)], -1)
        o, _ = self.rnn(inp)
        p = self.out(o)
        return p[..., :5], p[..., 5:].clamp(-5, 2)

    @torch.no_grad()
    def sample(self, s0n, T, temp=1.0):
        prev, hc, outs = first_prev(s0n), None, []
        for _ in range(T):
            o, hc = self.rnn(torch.cat([prev, s0n], -1)[:, None], hc)
            p = self.out(o[:, 0])
            x = p[:, :5] + temp * torch.exp(p[:, 5:].clamp(-5, 2)) * torch.randn_like(p[:, :5])
            x = unit_cs(x)
            outs.append(x); prev = x
        return torch.stack(outs, 1)


class TransAR(nn.Module):
    def __init__(self, d=64, T=35):
        super().__init__()
        self.inp = nn.Linear(10, d)
        self.pos = nn.Parameter(torch.zeros(1, T, d))
        layer = nn.TransformerEncoderLayer(d, 4, 128, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.out = nn.Linear(d, 10)
        self.T = T

    def _run(self, prev, s0n):
        n = prev.shape[1]
        h = self.inp(torch.cat([prev, s0n[:, None].expand(-1, n, -1)], -1)) + self.pos[:, :n]
        mask = torch.triu(torch.full((n, n), float("-inf")), 1)
        return self.out(self.enc(h, mask=mask))

    def forward(self, s0n, stn):
        prev = torch.cat([first_prev(s0n)[:, None], stn[:, :-1]], 1)
        p = self._run(prev, s0n)
        return p[..., :5], p[..., 5:].clamp(-5, 2)

    @torch.no_grad()
    def sample(self, s0n, T, temp=1.0):
        seq = first_prev(s0n)[:, None]
        for t in range(T):
            p = self._run(seq, s0n)[:, -1]
            x = unit_cs(p[:, :5] + temp * torch.exp(p[:, 5:].clamp(-5, 2)) * torch.randn_like(p[:, :5]))
            seq = torch.cat([seq, x[:, None]], 1)
        return seq[:, 1:]


def gauss_nll(mu, ls, x):
    return (ls + 0.5 * ((x - mu) / torch.exp(ls)) ** 2).mean()


# ---------------- VAE -------------------------------------------------------------
class SeqVAE(nn.Module):
    def __init__(self, h=128, z=16, T=35):
        super().__init__()
        self.enc = nn.GRU(10, h, batch_first=True, bidirectional=True)
        self.mu = nn.Linear(2 * h, z); self.lv = nn.Linear(2 * h, z)
        self.h0 = nn.Linear(z + 5, h)
        self.dec = nn.GRU(z + 5 + 8, h, batch_first=True)
        self.out = nn.Linear(h, 5)
        self.pe = nn.Parameter(torch.randn(1, T, 8) * 0.1)
        self.z, self.T = z, T
        self.logsig = nn.Parameter(torch.zeros(5))

    def decode(self, z, s0n):
        B = z.shape[0]
        c = torch.cat([z, s0n], -1)
        inp = torch.cat([c[:, None].expand(-1, self.T, -1), self.pe.expand(B, -1, -1)], -1)
        o, _ = self.dec(inp, torch.tanh(self.h0(c))[None])
        return self.out(o)

    def forward(self, s0n, stn):
        e, _ = self.enc(torch.cat([stn, s0n[:, None].expand(-1, stn.shape[1], -1)], -1))
        e = e.mean(1)
        mu, lv = self.mu(e), self.lv(e).clamp(-8, 4)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return self.decode(z, s0n), mu, lv

    @torch.no_grad()
    def sample(self, s0n, T, temp=1.0):
        z = torch.randn(s0n.shape[0], self.z)
        x = self.decode(z, s0n)
        x = x + temp * torch.exp(self.logsig) * torch.randn_like(x)  # observation noise of the decoder
        return unit_cs(x)


# ---------------- GANs --------------------------------------------------------------
class GenGRU(nn.Module):
    """Recurrent generator. kin=True: generates (sog, sin, cos) only and integrates
    position kinematically (dx,dy derived from SOG/COG) -> consistency by construction."""

    def __init__(self, norm, h=128, zg=16, zs=8, T=35, kin=False):
        super().__init__()
        self.h0 = nn.Linear(zg + 5, h)
        self.rnn = nn.GRU(zs + 5, h, batch_first=True)
        self.out = nn.Linear(h, 5)
        self.zg, self.zs, self.T, self.kin = zg, zs, T, kin
        for k, v in norm.torch().items():
            self.register_buffer(k, v)

    def forward(self, s0n, zg=None, zs=None):
        B = s0n.shape[0]
        zg = torch.randn(B, self.zg) if zg is None else zg
        zs = torch.randn(B, self.T, self.zs) if zs is None else zs
        h0 = torch.tanh(self.h0(torch.cat([zg, s0n], -1)))[None]
        o, _ = self.rnn(torch.cat([zs, s0n[:, None].expand(-1, self.T, -1)], -1), h0)
        x = unit_cs(self.out(o))
        if self.kin:
            sog = x[..., 2] * self.ts[2] + self.tm[2]
            sog = F.softplus(sog * 2) / 2  # non-negative knots
            sog0 = (s0n[:, 2] * self.ss[2] + self.sm[2])[:, None]
            sog_prev = torch.cat([sog0, sog[:, :-1]], 1)
            cs_prev = torch.cat([s0n[:, None, 3:5], x[:, :-1, 3:5]], 1)
            hd = cs_prev + x[..., 3:5]
            hd = hd / (hd.norm(dim=-1, keepdim=True) + 1e-6)
            dist = 0.5 * (sog + sog_prev) * KMSTEP
            dx, dy = dist * hd[..., 0], dist * hd[..., 1]
            x = torch.stack([(dx - self.tm[0]) / self.ts[0], (dy - self.tm[1]) / self.ts[1],
                             (sog - self.tm[2]) / self.ts[2], x[..., 3], x[..., 4]], -1)
        return x

    @torch.no_grad()
    def sample(self, s0n, T, temp=1.0):
        return self.forward(s0n)


class ConvNet(nn.Module):
    """1-D CNN over the step sequence (+ broadcast start state): used as WGAN critic
    (sn=False, scalar output) and as the neural detector / surrogate detector."""

    def __init__(self, cin=10, h=64):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv1d(cin, h, 5, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(h, h, 5, padding=2, stride=2), nn.LeakyReLU(0.2),
            nn.Conv1d(h, h, 5, padding=2, stride=2), nn.LeakyReLU(0.2))
        self.f = nn.Sequential(nn.Linear(h * 2, h), nn.LeakyReLU(0.2), nn.Linear(h, 1))

    def forward(self, s0n, stn):
        x = torch.cat([stn, s0n[:, None].expand(-1, stn.shape[1], -1)], -1).transpose(1, 2)
        h = self.c(x)
        h = torch.cat([h.mean(-1), h.amax(-1)], -1)
        return self.f(h).squeeze(-1)


def gradient_penalty(D, s0n, real, fake):
    a = torch.rand(real.shape[0], 1, 1)
    x = (a * real + (1 - a) * fake).requires_grad_(True)
    g, = torch.autograd.grad(D(s0n, x).sum(), x, create_graph=True)
    return ((g.reshape(g.shape[0], -1).norm(dim=1) - 1) ** 2).mean()
