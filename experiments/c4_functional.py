"""C4 — functional fraction over time, from the 16 uniformly random cell tapes recorded in every sample.

    python c4_functional.py runs/stageC [--out runs/stageC/analysis/c4] [--labels none,ld-imm,...]

For every run and every sample step in STEPS that the run recorded, the 16 random tapes are assayed
together (assay_many: A role, 32 random partners, gen2 and faithfulness, under the run's own
suppression set, step budget and rule — `zero_halts` runs select the lethal executor, recorded per row).
Output: functional.csv (one row per run × step) and figures of the
mean fraction of heritable / faithful cells versus step per arm, at (128, 1/16) and (32, 1/4).
Pre-registered as C4 ("functional fraction over time") in PLAN.md; no prediction was attached.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import figstyle as fs
from algocell_exp.assay import assay_many, executor_file
from algocell_exp.batch import select_summaries

STEPS = [50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 50000, 75000, 100000, 150000, 200000, 300000, 500000, 750000, 1000000]


def hexbytes(h: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(h.replace(" ", "")), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--labels", default=None, help="comma-separated labels to restrict to (default: all)")
    ap.add_argument("--partners", type=int, default=32)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "analysis", "c4")
    os.makedirs(out, exist_ok=True)
    want = set(a.labels.split(",")) if a.labels else None
    rows = []
    files = select_summaries(a.dir)
    for i, f in enumerate(files):
        s = json.load(open(f))
        if want and s["label"] not in want:
            continue
        stem = f[: -len(".summary.json")]
        L = s.get("tape_length", 16)
        patterns = s.get("suppress") or []
        zero_halts = bool(s.get("zero_halts", False))   # the run's rule: Stage I runs are assayed on the lethal executor
        steps_wanted = set(STEPS)
        n_rows = 0
        with open(stem + ".jsonl") as fh:
            for line in fh:
                if '"kind": "sample"' not in line:
                    continue
                r = json.loads(line)
                if r["step"] not in steps_wanted or not r.get("random_tapes"):
                    continue
                tapes = np.stack([hexbytes(h) for h in r["random_tapes"]])
                res = assay_many(tapes, z80_steps=s["z80_steps"], suppress=patterns, n=a.partners, seed=s["seed"] * 7919 + r["step"], zero_halts=zero_halts)
                rows.append({"file": os.path.basename(f), "label": s["label"], "tape_len": L, "steps": s["z80_steps"], "k": s["noise_exp"], "seed": s["seed"], "zero_halts": zero_halts, "step": r["step"], "n": len(res),
                             "frac_heritable": float(np.mean([x["is_replicator"] for x in res])), "frac_faithful": float(np.mean([x["faithful"] for x in res])),
                             "gen2_mean": float(np.mean([x["gen2_score"] for x in res])), "zero_frac": r.get("zero_frac", np.nan), "q_share": r.get("q_share", np.nan)})
                n_rows += 1
        print(f"[{i + 1}/{len(files)}] {s['label']} st{s['z80_steps']} k{s['noise_exp']} s{s['seed']}: {n_rows} steps; executor {executor_file(L, None, zero_halts)}{' (zero_halts)' if zero_halts else ''}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "functional.csv"), index=False)

    fs.setup()
    import matplotlib.pyplot as plt
    for steps, k in ((128, 4), (32, 2)):
        sub = df[(df["tape_len"] == 16) & (df["steps"] == steps) & (df["k"] == k)]
        if sub.empty:
            continue
        labels = fs.order(sub["label"].unique())
        fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE, 2.6), sharey=True)
        for ax, col, title in zip(axes, ("frac_heritable", "frac_faithful"), ("heritable (gen2 ≥ 0.3)", "faithful")):
            for lab in labels:
                g = sub[sub["label"] == lab].groupby("step")[col].agg(["mean", "count"]).reset_index()
                g = g[g["count"] >= 5]
                ax.plot(g["step"], g["mean"], color=fs.color(lab), lw=1.1, label=lab)
            ax.set_xscale("log")
            ax.set_xlim(40, 1.2e6)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("steps")
            ax.set_title(f"fraction of random cells {title}", fontsize=8)
        axes[0].set_ylabel("mean over seeds (16 cells × 32 partners each)")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.98), frameon=False, title=f"{steps} steps, mutation 1/{2**k}")
        fs.save(fig, os.path.join(out, f"C4_functional_st{steps}_k{k}"))
    # summary table at fixed steps
    tab = df[df["step"].isin([200, 1000, 5000, 50000, 300000, 1000000])].groupby(["label", "tape_len", "steps", "k", "step"]).agg(n=("seed", "size"), heritable=("frac_heritable", "mean"), faithful=("frac_faithful", "mean")).reset_index()
    tab.to_csv(os.path.join(out, "functional_fixed_steps.csv"), index=False)
    with open(os.path.join(out, "NUMBERS_C4.md"), "w") as fh:
        fh.write("# C4 — functional fraction of random cells at fixed steps (generated)\n\n" + tab.pivot_table(index=["label", "tape_len", "steps", "k"], columns="step", values="heritable").round(2).to_markdown() + "\n\n### faithful\n\n" + tab.pivot_table(index=["label", "tape_len", "steps", "k"], columns="step", values="faithful").round(2).to_markdown() + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
