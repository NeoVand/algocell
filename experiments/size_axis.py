"""Organism-size axis (Stage B + the L=16 cells of Stage A).

Figures: fraction of seeds with a heritable replicator vs L, median t_rep vs L,
final family composition vs L, final functional fraction vs L — per ablation
and step budget, mutation 1/16.

    python size_axis.py runs/stageB runs/stageA --out results/size_axis
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="results/size_axis")
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    asy = pd.concat([pd.read_csv(os.path.join(d, "analysis", "assays.csv")) for d in a.dirs], ignore_index=True)
    suc = pd.concat([pd.read_csv(os.path.join(d, "analysis", "succession.csv")) for d in a.dirs if os.path.exists(os.path.join(d, "analysis", "succession.csv"))], ignore_index=True)
    asy = asy[asy["k"] == a.k]
    labels = [l for l in ("none", "block-copy", "stack-writes", "no-copy") if l in set(asy["label"])]
    steps_levels = sorted(asy["steps"].unique())
    steps_levels = [s for s in steps_levels if s in (128, 512)]
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for (label, L, steps), g in asy.groupby(["label", "tape_len", "steps"]):
        if steps not in steps_levels:
            continue
        n = len(g)
        ne = int((g["t_rep"] > 0).sum())
        nf = int(g["final_replicator_insitu"].fillna(False).astype(bool).sum())
        lo, hi = wilson(ne, n)
        em = g[g["t_rep"] > 0]["t_rep"]
        rows.append({"label": label, "L": L, "steps": steps, "n": n, "emerged": ne, "frac": ne / n, "lo": lo, "hi": hi,
                     "final_rep": nf, "final_rep_frac": nf / n, "median_trep": float(em.median()) if ne else np.nan,
                     "func_frac": float(g["final_rep_fraction"].median()) if "final_rep_fraction" in g else np.nan})
    tab = pd.DataFrame(rows).sort_values(["label", "steps", "L"])
    tab.to_csv(os.path.join(a.out, "size_cells.csv"), index=False)
    pd.set_option("display.width", 200)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Figure 1: fraction emerged (t_rep) and final heritable vs L
    fig, axes = plt.subplots(1, len(steps_levels), figsize=(5.5 * len(steps_levels), 4), squeeze=False, sharey=True)
    for ax, steps in zip(axes[0], steps_levels):
        for label in labels:
            s = tab[(tab["label"] == label) & (tab["steps"] == steps)]
            if s.empty:
                continue
            ax.errorbar(s["L"], s["frac"], yerr=[s["frac"] - s["lo"], s["hi"] - s["frac"]], marker="o", capsize=3, label=f"{label} (first replicator)")
            ax.plot(s["L"], s["final_rep_frac"], marker="x", linestyle=":", alpha=0.7, label=f"{label} (heritable at end)")
        ax.set_xscale("log")
        ax.set_xticks([4, 9, 16, 25, 36, 49, 64, 81, 100], [4, 9, 16, 25, 36, 49, 64, 81, 100])
        ax.set_xlabel("tape length L (bytes)")
        ax.set_title(f"{steps} steps · mutation 1/2^{a.k}")
        ax.set_ylim(-0.05, 1.05)
    axes[0][0].set_ylabel("fraction of seeds")
    axes[0][-1].legend(fontsize=7)
    fig.savefig(os.path.join(a.out, "size_emergence.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: median t_rep vs L
    fig, axes = plt.subplots(1, len(steps_levels), figsize=(5.5 * len(steps_levels), 4), squeeze=False, sharey=True)
    for ax, steps in zip(axes[0], steps_levels):
        for label in labels:
            s = tab[(tab["label"] == label) & (tab["steps"] == steps)]
            ax.plot(s["L"], s["median_trep"], marker="o", label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([4, 9, 16, 25, 36, 49, 64, 81, 100], [4, 9, 16, 25, 36, 49, 64, 81, 100])
        ax.set_xlabel("tape length L (bytes)")
        ax.set_title(f"median emergence step (emerged seeds only) · {steps} steps")
    axes[0][0].set_ylabel("steps to first heritable replicator")
    axes[0][-1].legend(fontsize=8)
    fig.savefig(os.path.join(a.out, "size_trep.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: final family composition vs L (stacked bars), unablated
    if len(suc):
        suc = suc[suc["k"] == a.k]
        fams = ["push", "ex_sp", "ldir", "ld_hl", "cb_hl", "rst", "flooded", "none"]
        fig, axes = plt.subplots(1, len(steps_levels), figsize=(5.5 * len(steps_levels), 4), squeeze=False, sharey=True)
        for ax, steps in zip(axes[0], steps_levels):
            g = suc[(suc["label"] == "none") & (suc["steps"] == steps)]
            Ls = sorted(g["tape"].unique())
            bottom = np.zeros(len(Ls))
            for fam in fams:
                vals = np.array([((g[g["tape"] == L]["final_family"] == fam).mean()) if len(g[g["tape"] == L]) else 0 for L in Ls])
                if vals.sum() == 0:
                    continue
                ax.bar(range(len(Ls)), vals, bottom=bottom, label=fam)
                bottom += vals
            ax.set_xticks(range(len(Ls)), Ls)
            ax.set_xlabel("tape length L (bytes)")
            ax.set_title(f"final family (census), unablated · {steps} steps")
        axes[0][0].set_ylabel("fraction of seeds")
        axes[0][-1].legend(fontsize=8)
        fig.savefig(os.path.join(a.out, "size_family.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
