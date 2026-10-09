"""Head-to-head invasion of the matched L = 50 pair (pre-registered: REVISION_PREREG.md, I).

    .venv/bin/python invasion_pair.py --yes [--seeds 5] [--steps 20000] [--out results/invasion_pair]

Resident = every cell one tape; invader = 1% of cells the other tape at step 0; Stage G dynamics (L = 50, 8,192 pairs,
128 instructions, k = 4, square lattice). Both directions. Records every 250 steps the share of cells within Hamming 4 of
either tape under any cyclic shift (quasispecies radius 13, nearest class), the Hamming-4 cores, the share of cells carrying the jump word 20 f0, zero fraction; every 2,500 steps the heritable fraction of 16 cells.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from algocell_exp.assay import assay_many  # noqa: E402
from algocell_exp.soup import Soup  # noqa: E402

L = 50


def parse(s):
    return np.array([int(b, 16) for b in s.split()], dtype=np.uint8)


def shifts_of(t):
    return np.stack([np.roll(t, s) for s in range(L)])


def min_hamming(soup, ph):
    best = np.full(len(soup), L, dtype=int)
    for p in ph:
        best = np.minimum(best, (soup != p).sum(1))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--assay-every", type=int, default=2500)
    ap.add_argument("--early-every", type=int, default=10, help="sampling interval before --early-until")
    ap.add_argument("--early-until", type=int, default=500)
    ap.add_argument("--controls", type=int, default=3, help="seeds per resident with no invader")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "invasion_pair"))
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if not a.yes:
        sys.exit("refusing to run: pass --yes after pausing the browser simulation (local GPU)")
    g = pd.read_csv(os.path.join(HERE, "results", "stageG", "stageG", "stage_g_runs.csv"))
    m = g[(g.L == 50) & g.final_tape.str.startswith("01 c5") & g.final_tape.str.contains("20 f0")]
    closed = parse(m.final_tape.mode().iloc[0])
    pusher = parse(m.first_tape.mode().iloc[0])
    assert (pusher != closed).sum() > 4
    print("pusher :", " ".join(f"{b:02x}" for b in pusher))
    print("closed :", " ".join(f"{b:02x}" for b in closed), "| differ at", int((pusher != closed).sum()), "positions")
    ph = {"pusher": shifts_of(pusher), "closed": shifts_of(closed)}
    os.makedirs(a.out, exist_ok=True)
    rows = []
    t_all = time.time()
    plan = [("pusher", "closed", sd) for sd in range(1, a.seeds + 1)] + [("closed", "pusher", sd) for sd in range(1, a.seeds + 1)]
    plan += [("pusher", "none", sd) for sd in range(1, a.controls + 1)] + [("closed", "none", sd) for sd in range(1, a.controls + 1)]
    for resident, invader, seed in plan:
        if True:
            soup = Soup(160, 125, "square", L, 100 + seed, 8192, 128, 4, [])
            rng = np.random.default_rng([seed, L, 1 if resident == "pusher" else 2])
            cells = soup.read_soup()
            cells[:] = (pusher if resident == "pusher" else closed)[None, :]
            if invader != "none":
                idx = rng.choice(len(cells), size=int(0.01 * len(cells)), replace=False)
                cells[idx] = (closed if invader == "closed" else pusher)[None, :]
            soup.write_soup(cells)
            step, t0 = 0, time.time()
            while step <= a.steps:
                s = soup.read_soup()
                dp, dc = min_hamming(s, ph["pusher"]), min_hamming(s, ph["closed"])
                is_p = (dp <= 13) & (dp < dc)                 # quasispecies radius ceil(L/4) = 13, nearest class wins
                is_c = (dc <= 13) & (dc <= dp)
                core_p, core_c = (dp <= 4) & (dp < dc), (dc <= 4) & (dc <= dp)
                motif = ((s == 0x20) & (np.roll(s, -1, axis=1) == 0xF0)).any(axis=1)   # carries the jump word 20 f0
                rec = {"resident": resident, "invader": invader, "seed": seed, "step": step, "pusher_share": float(is_p.mean()), "closed_share": float(is_c.mean()),
                       "other_share": float((~is_p & ~is_c).mean()), "pusher_core": float(core_p.mean()), "closed_core": float(core_c.mean()),
                       "jump_share": float(motif.mean()), "zero_frac": float((s == 0).mean())}
                if step % a.assay_every == 0 and invader != "none":
                    ridx = rng.integers(0, len(s), size=16)
                    res = assay_many(s[ridx], z80_steps=128, suppress=[], n=32, seed=seed * 7919 + step)
                    rec["frac_heritable"] = float(np.mean([x["is_replicator"] for x in res]))
                rows.append(rec)
                dt = a.early_every if step < a.early_until else a.every
                soup.step(dt)
                step += dt
            last = rows[-1]
            print(f"{invader:>6} into {resident:<6} seed {seed}: closed {last['closed_share']:.3f} pusher {last['pusher_share']:.3f} other {last['other_share']:.3f} zero {last['zero_frac']:.3f} ({time.time() - t0:.0f} s)", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(a.out, "invasion_pair.csv"), index=False)
    df = pd.DataFrame(rows)
    fin = df[df.step == df.step.max()]
    lines = ["# Head-to-head invasion of the matched L = 50 pair (generated)", ""]
    for (res, inv), d in fin.groupby(["resident", "invader"]):
        share = d["closed_share"] if inv == "closed" else d["pusher_share"]
        lines.append(f"- {inv} into {res}-filled soup, step {int(d.step.iloc[0]):,}: pre-registered class share of the invader " + ", ".join(f"{v:.3f}" for v in share)
                     + f" (median {share.median():.3f}; > 50% in {int((share > 0.5).sum())}/{len(d)}; < 5% in {int((share < 0.05).sum())}/{len(d)}); jump-word share (post hoc) median {d.jump_share.median():.3f}; pusher core {d.pusher_core.median():.3f}, closed core {d.closed_core.median():.3f}")
    for seed, d in df[(df.invader == "closed")].groupby("seed"):
        t50 = d[d.jump_share > 0.5].step.min()
        lines.append(f"- closed into pusher, seed {seed}: jump-word share first above 50% at step {t50 if t50 == t50 else 'never'}; at steps 50/100/200/500: " + "/".join(f"{d[d.step == st].jump_share.iloc[0]:.2f}" for st in (50, 100, 200, 500) if (d.step == st).any()))
    out = "\n".join(lines)
    open(os.path.join(a.out, "REPORT.md"), "w").write(out + "\n")
    print(out)
    print(f"done in {time.time() - t_all:.0f} s")


if __name__ == "__main__":
    main()
