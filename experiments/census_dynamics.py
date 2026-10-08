"""Interaction-census dynamics — what the active pairs do, over time, per arm (CPU only).

    python census_dynamics.py runs/stageD --labels none,stack-write-only,stack-read-only,push,call-rst-write [--steps 128] [--out ...]
    python census_dynamics.py runs/stageC --labels none,push-only,stack-write-only,ld-imm --steps 128 --k 4

Every snapshot (500, 1k, 2k, 3k, 5k, 7.5k, 10k, 15k, 20k, 30k, 50k, 75k, 100k, 150k, 200k, 300k) carries a census of
all active pairs of that step: copy events A→B and B→A (best cyclic shift ≥ 0.75), partial copies, zero bytes written
into B, programs destroyed, bytes changed. Per arm and step this script reports medians over seeds of
  copy events per 1,000 interactions, zero writes per interaction, destroyed programs per 1,000 interactions,
  mean bytes changed in B,
and separately for runs that have NOT yet produced a heritable replicator at that step (pre-emergence), to ask whether
nascent copiers exist and are destroyed (copy events present before emergence) or never assemble (absent).
Pre-registered question (Stage D FINDINGS): destruction of nascent copiers vs niche occupation by sterile smears.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import figstyle as fs
from algocell_exp.batch import select_summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    labels = a.labels.split(",")
    out = a.out or os.path.join(a.dir, "analysis", "census")
    os.makedirs(out, exist_ok=True)
    asy = pd.read_csv(os.path.join(a.dir, "analysis", "assays.csv")).set_index("file")
    rows = []
    for f in select_summaries(a.dir):
        s = json.load(open(f))
        if s["label"] not in labels or s["z80_steps"] != a.steps or s["noise_exp"] != a.k:
            continue
        stem = f[: -len(".summary.json")]
        t_rep = float(asy.loc[os.path.basename(f), "t_rep"]) if os.path.basename(f) in asy.index else np.nan
        for line in open(stem + ".jsonl"):
            if '"census"' not in line or '"kind": "sample"' not in line:
                continue
            r = json.loads(line)
            c = r.get("census")
            if not c or not c.get("n"):
                continue
            n = c["n"]
            rows.append({"label": s["label"], "seed": s["seed"], "tape_len": s.get("tape_length", 16), "step": r["step"], "n": n,
                         "copy_per_1k": 1000 * (c["copy_ab"] + c["copy_ba"]) / n, "partial_per_1k": 1000 * c.get("partial_ab", 0) / n,
                         "zero_writes_per_ix": c["zero_writes_b"] / n, "writes_per_ix": c["writes_b"] / n, "destroyed_per_1k": 1000 * c["destroyed_a"] / n,
                         "bytes_changed_b": c["bytes_changed_b_mean"], "zero_frac": r.get("zero_frac"), "pre_emergence": not (t_rep > 0 and r["step"] >= t_rep), "emerged_run": t_rep > 0})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "census_rows.csv"), index=False)
    metrics = ["copy_per_1k", "partial_per_1k", "zero_writes_per_ix", "destroyed_per_1k", "bytes_changed_b", "zero_frac"]
    med = df.groupby(["label", "step"])[metrics].median().reset_index()
    pre = df[df["pre_emergence"]].groupby(["label", "step"]).agg(n_runs=("seed", "size"), **{m: (m, "median") for m in metrics}).reset_index()
    med.to_csv(os.path.join(out, "census_medians.csv"), index=False)
    pre.to_csv(os.path.join(out, "census_pre_emergence.csv"), index=False)
    md = [f"# Interaction census dynamics (generated) — {os.path.basename(a.dir)}, {a.steps} steps, mutation 1/{2**a.k}\n",
          "## Medians over all seeds, per arm and step\n", med.round(3).to_markdown(index=False) + "\n",
          "## Runs that have not yet produced a heritable replicator at that step (pre-emergence)\n", pre.round(3).to_markdown(index=False) + "\n"]
    # pooled pre-emergence copy rate per arm at steps ≤ 20k
    lines = ["## Pre-emergence copy events per 1,000 interactions, pooled over snapshots ≤ 20,000 steps (median, IQR, n run×steps)\n"]
    for lab in labels:
        g = df[(df["label"] == lab) & df["pre_emergence"] & (df["step"] <= 20000)]
        if len(g):
            q = g["copy_per_1k"].quantile([0.25, 0.5, 0.75])
            lines.append(f"- {lab}: {q[0.5]:.2f} ({q[0.25]:.2f}–{q[0.75]:.2f}), n = {len(g)}; destroyed per 1k {g['destroyed_per_1k'].median():.1f}; zero writes per interaction {g['zero_writes_per_ix'].median():.3f}")
    md.append("\n".join(lines) + "\n")
    with open(os.path.join(out, "NUMBERS_CENSUS.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(lines))

    fs.setup()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(fs.DOUBLE, 4.2))
    axes = axes.ravel()
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    titles = {"copy_per_1k": "copy events per 1,000 interactions\n(0 drawn at 0.05)", "zero_writes_per_ix": "zero bytes written into B\nper interaction", "destroyed_per_1k": "programs destroyed\nper 1,000 interactions", "zero_frac": "zero fraction of the soup"}
    for ax, m in zip(axes, ["copy_per_1k", "zero_writes_per_ix", "destroyed_per_1k", "zero_frac"]):
        for lab in labels:
            g = med[med["label"] == lab].sort_values("step")
            y = g[m].to_numpy(float)
            if m in ("copy_per_1k", "destroyed_per_1k"):
                y = np.where(y <= 0, 0.05, y)
            ax.plot(g["step"], y, color=fs.color(lab), lw=1.1, marker="o", ms=2.5, label=lab)
            gp = pre[(pre["label"] == lab) & (pre["n_runs"] >= 3)].sort_values("step")
            if m == "copy_per_1k" and len(gp):
                yp = np.where(gp[m].to_numpy(float) <= 0, 0.05, gp[m].to_numpy(float))
                ax.plot(gp["step"], yp, color=fs.color(lab), lw=0.8, ls=":", marker="x", ms=3)
        ax.set_xscale("log")
        if m in ("copy_per_1k", "destroyed_per_1k"):
            ax.set_yscale("log")
        ax.set_title(titles[m], fontsize=7.5)
        ax.set_xlabel("step")
    h, l = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h.append(Line2D([], [], color="k", ls=":", marker="x", ms=3, lw=0.8))
    l.append("pre-emergence runs only")
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.9), frameon=False, title=f"{os.path.basename(a.dir)}, {a.steps} steps, 1/{2**a.k}\n(medians over seeds)")
    fs.save(fig, os.path.join(out, f"census_dynamics_st{a.steps}_k{a.k}"))
    print("wrote", out)


if __name__ == "__main__":
    main()
