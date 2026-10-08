"""Invasion assay — can a known replicator take over a soup it did not evolve in?

    python invasion.py [--out runs/invasion] [--seeds 5] [--steps 20000]

Motivation (Stage C, E7): at 32 Z80 steps · mutation 1/4 the pusher `LD rr,nn ; PUSH rr` never appears although it is
heritable in isolation at that budget (gen2 0.54 at L = 16); every 32-step replicator is an LDIR unit. Is the pusher
unable to persist there (maintenance), or merely never discovered (discovery)? Seed it and watch.

Design: L = 16, 20,000 cells, 8,192 pairs, `none`. Conditions (z80_steps, k) ∈ {(32, 2), (128, 4), (32, 4), (128, 2)};
units ∈ {pusher `01 c5 ×8`, LDIR-4 `04 5e ed b0 ×4`}; seeding at step 0 (random soup) or at step 1,000 (after the zero
flood has formed), 1% of cells (200, chosen by a seeded RNG); SEEDS soups each. Recorded every 250 steps: exact share of
the unit (either phase), share within Hamming distance 4 of either phase, share matching the unit at ≥ 75% of bytes under
some cyclic shift (2,000-cell sample), zero fraction; every 2,500 steps the heritable fraction of 16 random cells
(assay_many, 32 partners). Outputs invasion.csv and a figure per unit.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

import figstyle as fs
from algocell_exp.assay import assay_many
from algocell_exp.soup import Soup

L = 16
UNITS = {"pusher": bytes.fromhex("01c5"), "ldir4": bytes.fromhex("045eedb0")}
CONDS = [(32, 2), (128, 4), (32, 4), (128, 2)]
SEED_AT = [0, 1000]
FRACTION = 0.01


def phases(unit: bytes) -> np.ndarray:
    u = np.frombuffer(unit, dtype=np.uint8)
    tiled = np.resize(u, L)
    return np.stack([np.roll(tiled, s) for s in range(len(u))])


def occupancy(soup: np.ndarray, ph: np.ndarray, rng: np.random.Generator) -> dict:
    exact = np.zeros(len(soup), bool)
    near = np.zeros(len(soup), bool)
    for p in ph:
        d = (soup != p).sum(1)
        exact |= d == 0
        near |= d <= 4
    idx = rng.integers(0, len(soup), size=min(2000, len(soup)))
    sub = soup[idx]
    best = np.zeros(len(sub))
    for s in range(L):
        rolled = np.roll(ph[0], s)
        best = np.maximum(best, (sub == rolled).mean(1))
    return {"exact": float(exact.mean()), "near4": float(near.mean()), "shift75": float((best >= 0.75).mean()), "zero_frac": float((soup == 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/invasion")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--assay-every", type=int, default=2500)
    ap.add_argument("--plot-only", action="store_true", help="re-draw the figures and tables from an existing invasion.csv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    t0 = time.time()
    n_runs = len(CONDS) * len(UNITS) * len(SEED_AT) * a.seeds
    done = 0
    if a.plot_only:
        rows = pd.read_csv(os.path.join(a.out, "invasion.csv")).to_dict("records")
        CONDS_RUN = []
    else:
        CONDS_RUN = CONDS
    for z80_steps, k in CONDS_RUN:
        for uname, unit in UNITS.items():
            ph = phases(unit)
            for seed_at in SEED_AT:
                for seed in range(1, a.seeds + 1):
                    soup = Soup(160, 125, "square", L, seed, 8192, z80_steps, k, [])
                    rng = np.random.default_rng([seed, z80_steps, k, seed_at, len(unit)])
                    seeded = False
                    step = 0
                    while step <= a.steps:
                        if not seeded and step >= seed_at:
                            cells = soup.read_soup()
                            idx = rng.choice(len(cells), size=int(FRACTION * len(cells)), replace=False)
                            cells[idx] = ph[0]
                            soup.write_soup(cells)
                            seeded = True
                        s = soup.read_soup()
                        rec = {"unit": uname, "z80_steps": z80_steps, "k": k, "seed_at": seed_at, "seed": seed, "step": step, **occupancy(s, ph, rng)}
                        if step % a.assay_every == 0:
                            ridx = rng.integers(0, len(s), size=16)
                            res = assay_many(s[ridx], z80_steps=z80_steps, suppress=[], n=32, seed=seed * 7919 + step)
                            rec["frac_heritable"] = float(np.mean([x["is_replicator"] for x in res]))
                            rec["frac_faithful"] = float(np.mean([x["faithful"] for x in res]))
                        rows.append(rec)
                        soup.step(a.every)
                        step += a.every
                    done += 1
                    last = rows[-1]
                    print(f"[{done}/{n_runs}] {uname} st{z80_steps} k{k} seed_at{seed_at} s{seed}: final exact {last['exact']:.3f} near4 {last['near4']:.3f} shift75 {last['shift75']:.3f} zero {last['zero_frac']:.3f} ({time.time() - t0:.0f}s)", flush=True)
                    pd.DataFrame(rows).to_csv(os.path.join(a.out, "invasion.csv"), index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "invasion.csv"), index=False)

    fs.setup()
    import matplotlib.pyplot as plt
    for uname in UNITS:
        fig, axes = plt.subplots(2, len(CONDS), figsize=(fs.DOUBLE, 3.8), sharex=True, sharey="row")
        for j, (z80_steps, k) in enumerate(CONDS):
            for seed_at, ls in zip(SEED_AT, ("-", "--")):
                g = df[(df["unit"] == uname) & (df["z80_steps"] == z80_steps) & (df["k"] == k) & (df["seed_at"] == seed_at)]
                for seed, gg in g.groupby("seed"):
                    axes[0, j].plot(gg["step"], gg["near4"], color=fs.color("none"), lw=0.7, ls=ls, alpha=0.7)
                    h = gg.dropna(subset=["frac_heritable"]) if "frac_heritable" in gg else gg.iloc[0:0]
                    axes[1, j].plot(h["step"], h["frac_heritable"], color=fs.color("none"), lw=0.7, ls=ls, alpha=0.7, marker=".", ms=3)
            axes[0, j].set_title(f"{z80_steps} steps, mutation 1/{2**k}", fontsize=8)
            axes[1, j].set_xlabel("steps after seeding")
            for ax in axes[:, j]:
                ax.set_ylim(-0.02, 1.02)
        axes[0, 0].set_ylabel("occupancy of the unit\n(cells within Hamming 4)")
        axes[1, 0].set_ylabel("heritable fraction\n(16 random cells)")
        fig.subplots_adjust(hspace=0.35, wspace=0.25)
        from matplotlib.lines import Line2D
        fig.legend([Line2D([], [], color="k", ls="-"), Line2D([], [], color="k", ls="--")], ["seeded at step 0", "seeded at step 1,000"], loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False, title=f"{uname} seeded into 1% of cells")
        fs.save(fig, os.path.join(a.out, f"invasion_{uname}"))
    tab = df[df["step"].isin([0, 1000, 2500, 5000, 10000, 20000])].groupby(["unit", "z80_steps", "k", "seed_at", "step"]).agg(near4=("near4", "mean"), exact=("exact", "mean"), zero=("zero_frac", "mean"), heritable=("frac_heritable", "mean")).round(3)
    with open(os.path.join(a.out, "NUMBERS_INVASION.md"), "w") as fh:
        fh.write("# Invasion assay (generated)\n\n" + tab.to_markdown() + "\n")
    print(tab.to_string())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
