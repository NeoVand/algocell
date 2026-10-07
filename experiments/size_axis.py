"""Organism-size axis (Stage B + the L = 16 cells of Stage A) → size_cells.csv and figures F3/F5.

    python size_axis.py runs/stageB runs/stageA --out runs/stageB/analysis/size_axis

Per (label, L, steps) at mutation 1/2^k: seeds with a heritable (t_rep) and a faithful
(t_faith) replicator with Wilson 95% intervals; Kaplan–Meier median time to each (NR when
not reached, censored at the run's last step); the conditional median among emerged seeds
(labelled); functional fraction of random final cells against random partners; final
family composition at 300k steps when the run got there (else at its last step, with the
number of early-stopped runs shown); tolerant period of the first-replicator tape.

Review 2026-10-07: the previous figures mixed measures across panels, used a different
colour per figure, overlapped titles and plotted n = 1 medians like n = 10 ones.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import figstyle as fs
from analyze import km_median, wilson

TAPES = [4, 9, 16, 25, 36, 49, 64, 81, 100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="runs/stageB/analysis/size_axis")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--labels", default="none,block-copy,stack-writes,no-copy")
    a = ap.parse_args()
    asy = pd.concat([pd.read_csv(os.path.join(d, "analysis", "assays.csv")) for d in a.dirs], ignore_index=True)
    suc = pd.concat([pd.read_csv(os.path.join(d, "analysis", "succession.csv")) for d in a.dirs if os.path.exists(os.path.join(d, "analysis", "succession.csv"))], ignore_index=True)
    asy = asy[(asy["k"] == a.k) & asy["steps"].isin([128, 512])]
    labels = [l for l in a.labels.split(",") if l in set(asy["label"])]
    steps_levels = sorted(asy["steps"].unique())
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for (label, L, steps), g in asy.groupby(["label", "tape_len", "steps"]):
        n = len(g)
        row = {"label": label, "L": L, "steps": steps, "n": n, "stopped_early": int(g["stopped_early"].sum())}
        for ev in ("t_rep", "t_faith"):
            t = g[ev].astype(float)
            em = t > 0
            ne = int(em.sum())
            lo, hi = wilson(ne, n)
            times = np.where(em, t, g["steps_run"]).astype(float)
            row.update({f"{ev}_n": ne, f"{ev}_frac": ne / n, f"{ev}_lo": lo, f"{ev}_hi": hi,
                        f"{ev}_km_median": km_median(times, em.to_numpy()), f"{ev}_cond_median": float(t[em].median()) if ne else np.nan})
        row.update({
            "final_faithful_n": int(g["final_faithful"].fillna(False).astype(bool).sum()),
            "func_rnd_median": float(g["final_func_rnd"].median()) if "final_func_rnd" in g else np.nan,
            "func_rnd_faithful_median": float(g["final_func_rnd_faithful"].median()) if "final_func_rnd_faithful" in g else np.nan,
            "trep_period_median": float(g.loc[g["t_rep"] > 0, "trep_period"].median()) if (g["t_rep"] > 0).any() else np.nan,
            "trep_period_divides_2L": float(((2 * L) % g.loc[g["t_rep"] > 0, "trep_period"].fillna(1).astype(int) == 0).mean()) if (g["t_rep"] > 0).any() else np.nan,
            "final_hoe_median": float(g["final_hoe"].median()),
        })
        rows.append(row)
    tab = pd.DataFrame(rows).sort_values(["label", "steps", "L"])
    tab.to_csv(os.path.join(a.out, "size_cells.csv"), index=False)
    pd.set_option("display.width", 250)
    print(tab[["label", "L", "steps", "n", "stopped_early", "t_rep_n", "t_rep_km_median", "t_faith_n", "t_faith_km_median", "func_rnd_median", "trep_period_median", "trep_period_divides_2L"]]
          .to_string(index=False, float_format=lambda x: "NR" if x == np.inf else f"{x:.2f}"))

    fs.setup()
    import matplotlib.pyplot as plt

    # ── F3a: fraction of seeds with a heritable (solid) / faithful (dashed) replicator vs L ──
    fig, axes = plt.subplots(1, len(steps_levels), figsize=(fs.DOUBLE, 2.4), squeeze=False, sharey=True)
    for ax, steps in zip(axes[0], steps_levels):
        for label in labels:
            s = tab[(tab["label"] == label) & (tab["steps"] == steps)].sort_values("L")
            if s.empty:
                continue
            c = fs.color(label)
            ax.errorbar(s["L"], s["t_rep_frac"], yerr=[s["t_rep_frac"] - s["t_rep_lo"], s["t_rep_hi"] - s["t_rep_frac"]], color=c, marker="o", capsize=2, lw=1, elinewidth=0.6, label=f"{label}")
            ax.plot(s["L"], s["t_faith_frac"], color=c, marker="s", ls="--", lw=0.9, mfc="white", alpha=0.9)
        ax.set_xscale("log")
        ax.set_xticks(TAPES, TAPES)
        ax.minorticks_off()
        ax.set_xlabel("tape length L (bytes)")
        ax.set_title(f"{steps} Z80 steps · mutation 1/2^{a.k}")
        ax.set_ylim(-0.03, 1.03)
    axes[0][0].set_ylabel("fraction of seeds (n = 10)")
    h, l = axes[0][0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], color="k", marker="o", ls="-"), Line2D([], [], color="k", marker="s", ls="--", mfc="white")]
    l += ["heritable (gen2 ≥ 0.3), Wilson 95%", "faithful (≥ 50% of partners became copies)"]
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False)
    fs.save(fig, os.path.join(a.out, "size_emergence"))

    # ── F3b: KM median time to first heritable replicator vs L (NR as open marker at the horizon) ──
    fig, axes = plt.subplots(1, len(steps_levels), figsize=(fs.DOUBLE, 2.4), squeeze=False, sharey=True)
    for ax, steps in zip(axes[0], steps_levels):
        for label in labels:
            s = tab[(tab["label"] == label) & (tab["steps"] == steps)].sort_values("L")
            if s.empty or s["t_rep_n"].sum() == 0:
                continue
            c = fs.color(label)
            reached = s[np.isfinite(s["t_rep_km_median"])]
            nr = s[~np.isfinite(s["t_rep_km_median"])]
            ax.plot(reached["L"], reached["t_rep_km_median"], color=c, marker="o", label=label)
            if len(nr):
                ax.plot(nr["L"], [300_000] * len(nr), color=c, marker="^", ls="none", mfc="white")
        ax.axhline(300_000, color="#999999", lw=0.5, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(TAPES, TAPES)
        ax.minorticks_off()
        ax.set_xlabel("tape length L (bytes)")
        ax.set_title(f"{steps} Z80 steps")
    axes[0][0].set_ylabel("KM median steps to first heritable replicator")
    h, l = axes[0][0].get_legend_handles_labels()
    h.append(Line2D([], [], color="k", marker="^", ls="none", mfc="white"))
    l.append("median not reached within 300k")
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False)
    fs.save(fig, os.path.join(a.out, "size_trep"))

    # ── F3c: family composition vs L at 300k steps (or the last step) for none and stack-writes ──
    if len(suc):
        suc = suc[(suc["k"] == a.k) & suc["steps"].isin(steps_levels)]
        fams = ["push", "ex_sp", "ldir", "ld_hl", "cb_hl", "rst", "flooded", "none"]
        show = [l for l in ("none", "stack-writes") if l in set(suc["label"])]
        fig, axes = plt.subplots(len(show), len(steps_levels), figsize=(fs.DOUBLE, 1.9 * len(show) + 0.3), squeeze=False, sharey=True)
        for r, label in enumerate(show):
            for c, steps in enumerate(steps_levels):
                ax = axes[r][c]
                g = suc[(suc["label"] == label) & (suc["steps"] == steps)]
                Ls = sorted(g["tape"].unique())
                fam_col = g["family_300k"].where(g["family_300k"].notna(), g["final_family"]) if "family_300k" in g else g["final_family"]
                bottom = np.zeros(len(Ls))
                for fam in fams:
                    vals = np.array([(fam_col[g["tape"] == L] == fam).mean() if (g["tape"] == L).any() else 0 for L in Ls])
                    if vals.sum() == 0:
                        continue
                    ax.bar(range(len(Ls)), vals, bottom=bottom, color=fs.FAMILY_COLOR[fam], edgecolor="white", lw=0.3, label=fam)
                    bottom += vals
                for i, L in enumerate(Ls):
                    stopped = int(g[(g["tape"] == L)]["stopped_early"].fillna(False).astype(bool).sum()) if "stopped_early" in g else 0
                    if stopped:
                        ax.text(i, 1.02, f"{stopped}▾", ha="center", va="bottom", fontsize=6, color="#555555")
                ax.set_xticks(range(len(Ls)), Ls)
                if r == len(show) - 1:
                    ax.set_xlabel("tape length L (bytes)")
                ax.set_title(f"{label} · {steps} steps", color=fs.color(label))
                ax.set_ylim(0, 1.12)
            axes[r][0].set_ylabel("fraction of seeds")
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False, title="census family at 300k steps\n(▾n = runs stopped early; shown at their last step)")
        fs.save(fig, os.path.join(a.out, "size_family"))

    # ── F5: tiling — tolerant period of the first-replicator tape vs L; filled = divides 2L ──
    fig, axes = plt.subplots(1, len(steps_levels), figsize=(fs.DOUBLE, 2.4), squeeze=False, sharey=True)
    rng = np.random.default_rng(0)
    for ax, steps in zip(axes[0], steps_levels):
        for label in [l for l in labels if l in ("none", "block-copy", "stack-writes")]:
            g = asy[(asy["label"] == label) & (asy["steps"] == steps) & (asy["t_rep"] > 0)]
            if g.empty:
                continue
            x = g["tape_len"].to_numpy(dtype=float) * np.exp(rng.normal(0, 0.03, len(g)))
            p = g["trep_period"].to_numpy(dtype=float)
            div = (2 * g["tape_len"].to_numpy()) % np.maximum(p, 1).astype(int) == 0
            ax.scatter(x[div], p[div], s=10, color=fs.color(label), label=label, alpha=0.8, lw=0)
            ax.scatter(x[~div], p[~div], s=12, facecolors="none", edgecolors=fs.color(label), alpha=0.9, lw=0.7)
        ax.plot(TAPES, TAPES, color="#aaaaaa", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(TAPES, TAPES)
        ax.minorticks_off()
        ax.set_xlabel("tape length L (bytes)")
        ax.set_title(f"{steps} Z80 steps")
    axes[0][0].set_ylabel("period of first replicator tape (bytes)")
    h, l = axes[0][0].get_legend_handles_labels()
    h += [Line2D([], [], marker="o", color="k", ls="none", ms=4), Line2D([], [], marker="o", color="k", ls="none", mfc="none", ms=4)]
    l += ["period divides 2L", "does not divide 2L"]
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False)
    fs.save(fig, os.path.join(a.out, "size_period"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
