"""Stage D figure and numbers — the L = 9 reversal (280 runs, seeds 1001–1020).

    python stage_d.py runs/stageD [--out runs/stageD/analysis/stage_d]

Forest plot: heritable (●) and faithful (○) emergence fraction with Wilson intervals, and KM median steps, for every
arm at 128 and 512 Z80 steps, plus `none` at 32 steps and at mutation 1/64; one-sided Fisher p against `none` at the
same budget and the CMH statistic pooled over budgets, as in results/stageD/FINDINGS.md. Writes NUMBERS_D.md.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import figstyle as fs
from analyze import km_median, wilson
from report import fisher, km_str

ORDER = ["none", "stack-writes", "stack-write-only", "stack-read-only", "push", "call-rst-write"]


def cmh_one_sided(tables: list[tuple[int, int, int, int]]) -> tuple[float, float]:
    """Cochran–Mantel–Haenszel z (and one-sided p) for 2×2 tables (a, b, c, d) = (arm+, arm−, none+, none−)."""
    from math import erfc, sqrt
    num = var = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        r1, r2, c1, c2 = a + b, c + d, a + c, b + d
        num += a - r1 * c1 / n
        var += r1 * r2 * c1 * c2 / (n * n * (n - 1))
    z = num / sqrt(var) if var > 0 else float("nan")
    return z, 0.5 * erfc(z / sqrt(2))


def cell(g: pd.DataFrame) -> dict:
    n = len(g)
    out = {"n": n}
    for ev in ("t_rep", "t_faith", "tq_10"):
        t = g[ev].astype(float)
        em = t > 0
        ne = int(em.sum())
        lo, hi = wilson(ne, n)
        out.update({f"{ev}_n": ne, f"{ev}_frac": ne / n, f"{ev}_lo": lo, f"{ev}_hi": hi, f"{ev}_km": km_median(np.where(em, t, g["steps_run"]).astype(float), em.to_numpy())})
    rep = g[g["t_rep"] > 0]
    out["periods"] = ", ".join(f"{int(p)}×{c}" for p, c in rep["trep_period"].value_counts().sort_index().items()) or "–"
    out["ldir_first"] = int(rep["trep_mechs"].astype(str).str.contains("block-copy").sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "analysis", "stage_d")
    os.makedirs(out, exist_ok=True)
    asy = pd.read_csv(os.path.join(a.dir, "analysis", "assays.csv"))
    asy = asy[asy["tape_len"] == 9]
    rows = []
    for (lab, steps, k), g in asy.groupby(["label", "steps", "k"]):
        rows.append({"label": lab, "steps": steps, "k": k, **cell(g)})
    tab = pd.DataFrame(rows)
    none = {(r["steps"], r["k"]): r for _, r in tab[tab["label"] == "none"].iterrows()}
    for i, r in tab.iterrows():
        ref = none.get((r["steps"], r["k"]))
        if ref is not None and r["label"] != "none":
            tab.loc[i, "fisher_p_greater"] = fisher(int(r["t_rep_n"]), int(r["n"] - r["t_rep_n"]), int(ref["t_rep_n"]), int(ref["n"] - ref["t_rep_n"]), "greater")
    cmh = {}
    for lab in ORDER[1:]:
        tables = []
        for steps in (128, 512):
            r = tab[(tab["label"] == lab) & (tab["steps"] == steps) & (tab["k"] == 4)]
            ref = none.get((steps, 4))
            if len(r) and ref is not None:
                r = r.iloc[0]
                tables.append((int(r["t_rep_n"]), int(r["n"] - r["t_rep_n"]), int(ref["t_rep_n"]), int(ref["n"] - ref["t_rep_n"])))
        if tables:
            cmh[lab] = cmh_one_sided(tables)
    tab.to_csv(os.path.join(out, "cells_D.csv"), index=False)
    md = ["# Stage D numbers (generated)\n", tab[["label", "steps", "k", "n", "t_rep_n", "t_rep_km", "t_faith_n", "tq_10_n", "fisher_p_greater", "periods", "ldir_first"]].sort_values(["steps", "k", "label"]).to_markdown(index=False, floatfmt=".3g") + "\n",
          "## CMH one-sided (arm > none), pooled over 128 and 512 steps at 1/16\n"]
    for lab, (z, p) in cmh.items():
        md.append(f"- {lab}: z = {z:.2f}, p = {p:.2g}")
    with open(os.path.join(out, "NUMBERS_D.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))

    fs.setup()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    labels = [l for l in ORDER if l in set(tab["label"])]
    fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE, 0.3 * len(labels) + 1.4), sharey=True, gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(labels))[::-1].astype(float)
    off = {128: 0.16, 512: -0.16}
    mk = {128: "o", 512: "s"}
    for steps in (128, 512):
        s = tab[(tab["steps"] == steps) & (tab["k"] == 4)].set_index("label").reindex(labels)
        yy = y + off[steps]
        cols = [fs.color(l) for l in labels]
        axes[0].errorbar(s["t_rep_frac"], yy, xerr=[s["t_rep_frac"] - s["t_rep_lo"], s["t_rep_hi"] - s["t_rep_frac"]], fmt="none", ecolor="#bbbbbb", elinewidth=0.8)
        axes[0].scatter(s["t_rep_frac"], yy, c=cols, s=22, marker=mk[steps], zorder=3)
        axes[0].scatter(s["t_faith_frac"], yy, facecolors="none", edgecolors=cols, s=40, marker=mk[steps], lw=0.8, zorder=3)
        km = s["t_rep_km"].to_numpy(float)
        ok = np.isfinite(km)
        axes[1].scatter(km[ok], yy[ok], c=np.array(cols)[ok], s=22, marker=mk[steps], zorder=3)
        axes[1].scatter([400_000] * int((~ok).sum()), yy[~ok], c=np.array(cols)[~ok], s=22, marker=">", zorder=3)
        for lab, yv in zip(labels, yy):
            p = s.loc[lab, "fisher_p_greater"]
            if lab != "none" and np.isfinite(p):
                axes[0].text(1.06, yv, f"p = {p:.2g}", va="center", fontsize=5.5, color="#555555")
    # extra `none` cells: 32 steps and 1/64
    extra = tab[(tab["label"] == "none") & ((tab["steps"] == 32) | (tab["k"] == 6))]
    for j, (_, r) in enumerate(extra.iterrows()):
        yv = -1.0 - 0.5 * j
        axes[0].scatter(r["t_rep_frac"], yv, c="k", s=22, marker="^", zorder=3)
        axes[0].errorbar(r["t_rep_frac"], yv, xerr=[[r["t_rep_frac"] - r["t_rep_lo"]], [r["t_rep_hi"] - r["t_rep_frac"]]], fmt="none", ecolor="#bbbbbb", elinewidth=0.8)
        axes[0].text(-0.08, yv, f"none, {r['steps']} steps, 1/{2**int(r['k'])}", va="center", ha="right", fontsize=6.5)
        if np.isfinite(r["t_rep_km"]):
            axes[1].scatter(r["t_rep_km"], yv, c="k", s=22, marker="^", zorder=3)
        else:
            axes[1].scatter(400_000, yv, c="k", s=22, marker=">", zorder=3)
    axes[0].set_yticks(y, labels)
    for t, lab in zip(axes[0].get_yticklabels(), labels):
        t.set_color(fs.color(lab))
    axes[0].set_xlim(-0.03, 1.25)
    axes[0].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_xlabel("fraction of 20 seeds — heritable (filled), faithful (open); p: one-sided Fisher vs none")
    axes[1].set_xscale("log")
    axes[1].set_xlim(5_000, 600_000)
    axes[1].set_xlabel("KM median steps to first heritable replicator (▶ not reached in 300k)")
    axes[0].set_ylim(-1.6 - 0.5 * max(0, len(extra) - 1), len(labels) - 0.4)
    fig.suptitle("L = 9, mutation 1/16: removing stack writers raises emergence (Stage D, seeds 1001–1020)", fontsize=8, y=1.0)
    h = [Line2D([], [], marker="o", color="k", ls="none", ms=4), Line2D([], [], marker="s", color="k", ls="none", ms=4)]
    fig.legend(h, ["128 Z80 steps", "512 Z80 steps"], loc="upper left", bbox_to_anchor=(1.0, 0.9), frameon=False)
    fs.save(fig, os.path.join(out, "D_forest"))
    print("wrote", out)


if __name__ == "__main__":
    main()
