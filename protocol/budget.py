"""Bit-level budget of Secure-AIS against ITU-R M.1371-6 (02/2026) and VHF data-link (VDL) load.

Limits used (ITU-R M.1371-6):
  Msg 1/2/3 position report: 168 data bits, 1 slot.
  Msg 8 binary broadcast, maximum binary data incl. bit stuffing (Table 56): 12/40/68 bytes for 1/2/3 slots,
      i.e. 80/304/528 application bits after the 16-bit application identifier.
      A mobile station may use at most 3 consecutive slots per Msg 8, and its RATDMA transmissions of
      Msgs 6, 8, 12, 14 and 25 should not exceed 20 slots per frame (Table 44, note 10).
  Msg 25 single-slot binary, broadcast, structured: 112 application bits.
  VDL: 2250 slots per frame (1 min) per channel, two channels.
  Class A reporting intervals (Table 1): 3 min (anchored/moored, <=3 kn), 10 s (0-14 kn), 3 1/3 s (0-14 kn,
      changing course), 6 s (14-23 kn), 2 s (14-23 kn changing course, and >23 kn).

Epoch tag block (ETB), one Msg 8 (<=3 slots) per epoch of length Delta:
  header: version 2 | chain-id 6 | epoch index 16                 = 24 bits
  disclosed key K_{i-d}                                           = 128 bits
  per report: UTC second (6) + truncated tag (TAG bits)
Chain commitment, once per chain (1 h): 740 bits, sent as two Msg 8 fragments (3 + 2 slots), see FRAG.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for _folder in ("samples", "results", "paper", "paper/figs"):
    (ROOT / _folder).mkdir(parents=True, exist_ok=True)
import json, pickle, numpy as np, collections

SLOTS_PER_MIN = 2 * 2250
MSG8_APP = {1: 80, 2: 304, 3: 528}  # mobile stations: max 3 slots
RATDMA_CAP = 20
HDR, KEY, UTC = 24, 128, 6
COMMIT_FIELDS = dict(version=2, chain_id=6, epoch0=16, delta=8, d=4, commitment=128, cert_hash=64, signature=512)
COMMIT_BITS = sum(COMMIT_FIELDS.values())
FRAG_HDR = 2 + 6 + 2  # version, chain-id, fragment index
# fragment 1 fills a 3-slot message; fragment 2 carries the rest
FRAG1 = 528
FRAG2 = COMMIT_BITS - (FRAG1 - FRAG_HDR) + FRAG_HDR
COMMIT_SLOTS = 3 + min(s for s, c in MSG8_APP.items() if FRAG2 <= c)
CHAIN_MIN = 60
INTERVALS = [("180", 180.0, "3 min"), ("10", 10.0, "10 s"), ("6", 6.0, "6 s"), ("3.33", 10 / 3, "3 1/3 s"),
             ("2", 2.0, "2 s")]


def slots_for(bits):
    for s, cap in MSG8_APP.items():
        if bits <= cap:
            return s
    return None


def etb_plan(interval, tag):
    """smallest extra slot rate over message sizes 2-3 slots, epoch Delta <= max(60 s, interval);
    ties broken by the longer epoch (fewer transmissions)."""
    entry = UTC + tag
    best = None
    dmax = max(60.0, interval)
    for slots in [2, 3]:
        n = (MSG8_APP[slots] - HDR - KEY) // entry
        if n < 1:
            continue
        k = int(min(n, round(dmax / interval)))  # reports per epoch
        delta = k * interval
        extra = (60.0 / delta) * slots + COMMIT_SLOTS / CHAIN_MIN
        c = dict(slots=slots, entries=k, delta_s=delta, extra_slots_min=extra, bits=HDR + KEY + entry * k)
        if best is None or extra < best["extra_slots_min"] - 1e-9 or (
                abs(extra - best["extra_slots_min"]) < 1e-9 and delta > best["delta_s"]):
            best = c
    r = 60.0 / interval
    best["reports_per_min"] = r
    best["overhead_pct"] = 100 * best["extra_slots_min"] / r
    best["latency_s"] = 3 * best["delta_s"]
    return best


def schemes(interval):
    r = 60.0 / interval
    rows = [
        dict(key="legacy", scheme="No authentication", extra=0.0),
        dict(key="sig", scheme="Per-report signature, 512-bit Ed25519 (3-slot Msg 8)", extra=r * slots_for(6 + 512)),
        dict(key="mac", scheme="Per-report MAC, 32-bit, shared key (Msg 25)", extra=r * 1.0),
        dict(key="etb32", scheme="ETB, 32-bit tags (proposed)", extra=etb_plan(interval, 32)["extra_slots_min"]),
        dict(key="etb64", scheme="ETB, 64-bit tags", extra=etb_plan(interval, 64)["extra_slots_min"]),
    ]
    for x in rows:
        x["base"] = r
        x["total"] = r + x["extra"]
        x["overhead_pct"] = 100 * x["extra"] / r
        # authentication messages sent with RATDMA by the vessel itself
        x["ratdma_ok"] = x["extra"] <= RATDMA_CAP + 1e-9
    return rows


def main():
    out = dict(etb_fields=dict(header=HDR, key=KEY, utc=UTC), commit_bits=COMMIT_BITS, commit_slots=COMMIT_SLOTS,
               commit_fragments=[FRAG1, FRAG2], ratdma_cap=RATDMA_CAP, msg8_app=MSG8_APP)
    out["per_interval"] = {}
    for key, iv, label in INTERVALS:
        out["per_interval"][key] = dict(label=label, etb32=etb_plan(iv, 32), etb64=etb_plan(iv, 64),
                                        schemes=schemes(iv))
    # 10-s alternative with the longest epoch that still fits 3 slots
    out["alt10"] = dict(delta_s=90.0, entries=9, slots=3, extra_slots_min=3 * 60 / 90 + COMMIT_SLOTS / CHAIN_MIN)
    out["alt10"]["overhead_pct"] = 100 * out["alt10"]["extra_slots_min"] / 6
    out["alt10"]["latency_s"] = 3 * 90
    # simultaneous vessels in the DMA ROI (cargo/tanker only -> lower bound of real traffic)
    cnt = collections.Counter()
    for s in ["train", "valid", "test"]:
        for x in pickle.load(open(f"{ROOT}/raw/ct_dma_{s}.pkl", "rb")):
            for ts in np.unique((x["traj"][:, 4] // 600).astype(np.int64)):
                cnt[ts] += 1
    c = np.array(list(cnt.values()))
    out["simultaneous_cargo_tanker_vessels"] = dict(median=float(np.median(c)), p95=float(np.percentile(c, 95)),
                                                    max=int(c.max()))
    rows = schemes(10.0)
    out["vdl_load_pct_10s"] = [{"N": N, **{x["key"]: 100 * N * x["total"] / SLOTS_PER_MIN for x in rows}}
                               for N in [100, 250, 500, 750]]
    out["max_vessels_10s"] = {x["key"]: dict(at50=int(0.5 * SLOTS_PER_MIN / x["total"]),
                                             at100=int(SLOTS_PER_MIN / x["total"])) for x in rows}
    # forgery analysis for 32-bit tags (see paper, Sec. VII-F)
    tag, F = 32, 16
    out["forgery"] = dict(per_attempt=2.0 ** -tag, max_entries_per_epoch_key=60,
                          any_report_per_epoch_key_nocap=60 * 2.0 ** -tag,
                          fail_cap_per_chain=F,
                          # a single forged report (e.g. inserted next to genuine reports) under one chain
                          any_report_per_chain=(F + 1) * 2.0 ** -tag,
                          # a wholly fabricated track: needs two independent successful forgeries
                          fabricated_track_per_chain=((F + 1) * 2.0 ** -tag) ** 2)
    cert = 32 + 64 + 8 + 9 + 32
    hs = {"ClientHello": 1 + 9 + 32 + 32 + 2 + cert, "ServerHello": 1 + 9 + 32 + 32 + cert + 64 + 32,
          "VesselFinished": 1 + 64 + 32, "compact_cert": cert}
    hs["total"] = hs["ClientHello"] + hs["ServerHello"] + hs["VesselFinished"]
    out["handshake_bytes"] = hs
    json.dump(out, open(f"{ROOT}/results/budget.json", "w"), indent=1)
    for k, v in out["per_interval"].items():
        print(v["label"], "ETB32", {a: round(b, 3) for a, b in v["etb32"].items()})
        print("      ETB64", {a: round(b, 3) for a, b in v["etb64"].items()})
        for x in v["schemes"]:
            print("     ", x["key"], round(x["extra"], 3), f"{x['overhead_pct']:.1f}%", "ratdma_ok" if x["ratdma_ok"] else "EXCEEDS 20")
    print("commit", COMMIT_BITS, out["commit_fragments"], COMMIT_SLOTS)
    print("alt10", out["alt10"])
    print(out["max_vessels_10s"])
    print(out["vdl_load_pct_10s"])


if __name__ == "__main__":
    main()
