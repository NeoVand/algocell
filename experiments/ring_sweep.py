"""Local ring sweep (post Stage F, exploratory, no GPU cost): does the pusher regime exist at (L, P)?

    python ring_sweep.py [--out runs/ring_sweep] [--seeds 101,102,103] [--horizon 20000]

`none`, 128 steps, mutation 1/16, every (L, P) for which a sim shader exists at L ∈ {8, 10, 12, 16} (P even, from 2L
up to 2L + 24, plus the Stage F values), run through the exact Modal code path (`batch.run_to_dir`) for HORIZON steps
with the standard sampling; then assay_batch gives t_rep and the first-replicator period. Emergence of the pusher is
fast (≤ 2,000 steps) wherever it occurs in Stages E and F, so a 20,000-step horizon separates "pusher cell" from "not".
Writes conds/ring_sweep.json (so select_summaries filters correctly), the run directory, and analysis/ring_sweep.csv +
a figure after `python assay_batch.py runs/ring_sweep`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from algocell_exp.batch import run_stem, run_to_dir
from make_conds import _c

SHADER_DIR = os.path.join(os.path.dirname(__file__), "algocell_exp", "shader")


def rings_for(L: int) -> list[int]:
    Ps = {2 * L}
    for f in glob.glob(os.path.join(SHADER_DIR, f"sim_square_L{L}_P*.wgsl")):
        Ps.add(int(re.search(r"_P(\d+)\.wgsl$", f).group(1)))
    return sorted(Ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ring_sweep")
    ap.add_argument("--seeds", default="101,102,103")
    ap.add_argument("--horizon", type=int, default=20000)
    ap.add_argument("--lengths", default="8,10,12,16")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    conds = []
    for L in (int(x) for x in a.lengths.split(",")):
        for P in rings_for(L):
            for s in seeds:
                c = _c(f"none@ring{P}", L, 128, 4, s, horizon=a.horizon)
                if P != 2 * L:
                    c["mem_length"] = P
                c["snapshot_steps"] = [t for t in c["snapshot_steps"] if t <= a.horizon]
                conds.append(c)
    os.makedirs("conds", exist_ok=True)
    with open("conds/ring_sweep.json", "w") as f:
        json.dump(conds, f, indent=0)
    print(len(conds), "conditions")
    if not a.analyze_only:
        for i, c in enumerate(conds):
            stem = run_stem(c)
            if os.path.exists(os.path.join(a.out, stem + ".summary.json")):
                continue
            run_to_dir(c, a.out, {"local": True, "purpose": "ring_sweep"})
            print(f"[{i + 1}/{len(conds)}] {stem}", flush=True)
        subprocess.run([sys.executable, "assay_batch.py", a.out], check=True, stdout=subprocess.DEVNULL)
    asy = pd.read_csv(os.path.join(a.out, "analysis", "assays.csv"))
    asy["P"] = asy["mem_length"]
    rows = []
    for (L, P), g in asy.groupby(["tape_len", "P"]):
        em = g["t_rep"] > 0
        rep = g[em]
        rows.append({"L": int(L), "P": int(P), "P_minus_2L": int(P - 2 * L), "n": len(g), "heritable": int(em.sum()), "t_rep_median": float(rep["t_rep"].median()) if len(rep) else np.nan,
                     "pusher_first": int((rep["trep_period"] == 2).sum()), "periods": ", ".join(f"{int(p)}×{c}" for p, c in rep["trep_period"].value_counts().sort_index().items()) or "–",
                     "final_gen2_median": float(g["final_gen2"].median()), "final_func_rnd_median": float(g["final_func_rnd"].median()), "final_zero_median": float(g["final_zero_frac"].median())})
    tab = pd.DataFrame(rows).sort_values(["L", "P"])
    os.makedirs(os.path.join(a.out, "analysis"), exist_ok=True)
    tab.to_csv(os.path.join(a.out, "analysis", "ring_sweep.csv"), index=False)
    with open(os.path.join(a.out, "analysis", "NUMBERS_RING_SWEEP.md"), "w") as f:
        f.write(f"# Local ring sweep (generated; `none`, 128 steps, 1/16, seeds {a.seeds}, horizon {a.horizon:,})\n\n" + tab.to_markdown(index=False, floatfmt=".2f") + "\n")
    print(tab.to_string(index=False))
    import figstyle as fs
    fs.setup()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE, 2.6))
    for L, m, c in zip((8, 10, 12, 16), ("o", "s", "^", "D"), ("#0072B2", "#E69F00", "#009E73", "#000000")):
        s = tab[tab["L"] == L]
        if s.empty:
            continue
        axes[0].plot(s["P"], s["pusher_first"] / s["n"], marker=m, color=c, lw=0.8, label=f"L = {L}")
        axes[1].plot(s["P"], s["final_func_rnd_median"], marker=m, color=c, lw=0.8)
    axes[0].set_ylabel(f"seeds where the pusher is the first replicator (of {len(seeds)})")
    axes[1].set_ylabel("final heritable fraction of random cells (median)")
    for ax in axes:
        ax.set_xlabel("ring length P (bytes)")
        ax.set_ylim(-0.03, 1.03)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False, title=f"`none`, 128 steps, 1/16, {a.horizon:,} steps")
    fs.save(fig, os.path.join(a.out, "analysis", "ring_sweep"))
    print("wrote", os.path.join(a.out, "analysis"))


if __name__ == "__main__":
    main()
