"""Stage E — size-axis control arms, budget and mutation sweeps, tiny lengths, within-seed variance.

    python stage_e.py runs/stageE [--out runs/stageE/analysis/stage_e]

Reads assays.csv (+ the jsonl for the interaction clock) and writes figures + a NUMBERS_E.md:
  E1  heritable / faithful fraction vs L per arm (@nominal, @mubyte, @bytes, @steps8L), both ablations
  E2  KM median time to first heritable replicator vs L per arm, in steps and in active interactions per cell
  E3  tiled fraction and period of the first replicator vs L per arm (does the period divide 2L in every arm?)
  E4  budget sweep at L = 100: heritable, faithful, whole-tape copiers vs Z80 steps
  E5  mutation sweep at L = 100: heritable fraction, first-replicator period, final HOE vs mutation rate
  E6  within-seed variance (10 repeats of seed 1) vs between-seed variance in the same cells
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import figstyle as fs
from analyze import km_median, wilson
from report import km_str

ARMS = ["nominal", "mubyte", "bytes", "steps8L"]
ARM_STYLE = {"nominal": ("o", "-"), "mubyte": ("s", "--"), "bytes": ("^", "-."), "steps8L": ("D", ":")}
ARM_COLOR = {"nominal": "#000000", "mubyte": "#D55E00", "bytes": "#0072B2", "steps8L": "#009E73"}
L_ALL = [3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 18, 20, 24, 25, 32, 36, 49, 50, 64, 81, 100]


def interaction_clock(run_dir: str, files: list[str]) -> pd.Series:
    """Mean active interactions per cell per step, per run (from the recorded active_pairs and the cell count)."""
    out = {}
    for f in files:
        stem = f[: -len(".summary.json")]
        s = json.load(open(f))
        cells = s["cells"]
        acts = []
        for line in open(stem + ".jsonl"):
            if '"kind": "sample"' in line:
                r = json.loads(line)
                acts.append(r.get("active_pairs", np.nan))
        out[os.path.basename(f)] = 2.0 * float(np.nanmean(acts)) / cells if acts else np.nan
    return pd.Series(out, name="ix_per_cell_step")


def cell_rows(a: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in a.groupby(keys):
        n = len(g)
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row["n"] = n
        for ev in ("t_rep", "t_faith"):
            t = g[ev].astype(float)
            em = t > 0
            ne = int(em.sum())
            lo, hi = wilson(ne, n)
            times = np.where(em, t, g["steps_run"]).astype(float)
            row.update({f"{ev}_n": ne, f"{ev}_frac": ne / n, f"{ev}_lo": lo, f"{ev}_hi": hi, f"{ev}_km": km_median(times, em.to_numpy())})
            if "ix_per_cell_step" in g:
                row[f"{ev}_km_ix"] = km_median(times * g["ix_per_cell_step"].to_numpy(), em.to_numpy())
        rep = g[g["t_rep"] > 0]
        row["tiled_frac"] = float((rep["trep_period"] <= rep["tape_len"] / 2).mean()) if len(rep) else np.nan
        tiled = rep[rep["trep_period"] <= rep["tape_len"] / 2]
        row["div2L_tiled"] = float(((2 * tiled["tape_len"]) % tiled["trep_period"].astype(int) == 0).mean()) if len(tiled) else np.nan
        row["period_median"] = float(rep["trep_period"].median()) if len(rep) else np.nan
        row["whole_tape_n"] = int(((rep["trep_period"] == rep["tape_len"]) & (rep["trep_offset"].fillna(-1) == 0)).sum()) if len(rep) else 0
        row["final_faithful_n"] = int(g["final_faithful"].fillna(False).astype(bool).sum())
        row["final_rep_n"] = int(g["final_replicator"].fillna(False).astype(bool).sum())
        row["func_rnd_min"] = float(g["final_func_rnd"].min())
        row["func_rnd_median"] = float(g["final_func_rnd"].median())
        row["final_hoe_median"] = float(g["final_hoe"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "analysis", "stage_e")
    os.makedirs(out, exist_ok=True)
    asy = pd.read_csv(os.path.join(a.dir, "analysis", "assays.csv"))
    files = [os.path.join(a.dir, f) for f in asy["file"]]
    clock = interaction_clock(a.dir, files)
    asy = asy.merge(clock.rename_axis("file").reset_index(), on="file", how="left")
    asy["ablation"] = asy["label"].str.split("@").str[0]
    asy["arm"] = asy["label"].str.split("@").str[1].fillna("nominal")
    fs.setup()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    md = ["# Stage E numbers (generated)\n"]

    # ── E1/E2/E3: size axis per arm ──
    size = asy[asy["arm"].isin(ARMS) & (asy["replicate"].isna())]
    tab = cell_rows(size, ["ablation", "arm", "tape_len"]).sort_values(["ablation", "arm", "tape_len"])
    tab.to_csv(os.path.join(out, "size_arms.csv"), index=False)
    md.append("## E1–E3 size axis per arm\n\n" + tab[["ablation", "arm", "tape_len", "n", "t_rep_n", "t_rep_km", "t_rep_km_ix", "t_faith_n", "tiled_frac", "div2L_tiled", "period_median", "whole_tape_n", "func_rnd_median"]]
              .to_markdown(index=False, floatfmt=".2f") + "\n")
    pooled = []
    for (abl, arm), g in size[size["t_rep"] > 0].groupby(["ablation", "arm"]):
        tiled = g[g["trep_period"] <= g["tape_len"] / 2]
        pooled.append({"ablation": abl, "arm": arm, "first_replicators": len(g), "tiled": len(tiled),
                       "divides_L_of_tiled": float((tiled["tape_len"] % tiled["trep_period"].astype(int) == 0).mean()) if len(tiled) else np.nan,
                       "divides_2L_of_tiled": float(((2 * tiled["tape_len"]) % tiled["trep_period"].astype(int) == 0).mean()) if len(tiled) else np.nan,
                       "non_divisors_2L": ", ".join(f"L{int(L)}:p{int(p)}" for L, p in zip(tiled["tape_len"], tiled["trep_period"]) if (2 * L) % int(p)) or "–"})
    md.append("### Tiling pooled per arm (first heritable replicators, L ≥ 3)\n\n" + pd.DataFrame(pooled).to_markdown(index=False, floatfmt=".3f") + "\n")
    abls = [x for x in ("none", "stack-write-only") if x in set(tab["ablation"])]
    for ev, fname, ylabel in (("t_rep", "E1_fraction", "seeds with a heritable replicator (of 10)"), ("t_faith", "E1b_fraction_faithful", "seeds with a faithful replicator (of 10)")):
        fig, axes = plt.subplots(1, len(abls), figsize=(fs.DOUBLE, 2.5), squeeze=False, sharey=True)
        for ax, abl in zip(axes[0], abls):
            for arm in ARMS:
                s = tab[(tab["ablation"] == abl) & (tab["arm"] == arm)].sort_values("tape_len")
                if s.empty:
                    continue
                m, ls = ARM_STYLE[arm]
                ax.errorbar(s["tape_len"], s[f"{ev}_frac"], yerr=[s[f"{ev}_frac"] - s[f"{ev}_lo"], s[f"{ev}_hi"] - s[f"{ev}_frac"]], color=ARM_COLOR[arm], marker=m, ls=ls, lw=1, capsize=2, elinewidth=0.5, label=f"@{arm}")
            ax.set_xscale("log")
            ax.set_xticks(L_ALL, [str(x) if x in (3, 4, 9, 16, 25, 36, 49, 64, 81, 100) else "" for x in L_ALL])
            ax.minorticks_off()
            ax.set_title(abl, color=fs.color(abl))
            ax.set_xlabel("tape length L (bytes)")
            ax.set_ylim(-0.03, 1.03)
        axes[0][0].set_ylabel(ylabel)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False, title="arm (128 steps, 1/16 unless the arm changes it)")
        fs.save(fig, os.path.join(out, fname))
    for unit, col, fname, ylabel in (("steps", "t_rep_km", "E2_km_steps", "KM median steps to first heritable replicator"), ("interactions", "t_rep_km_ix", "E2b_km_interactions", "KM median active interactions per cell")):
        fig, axes = plt.subplots(1, len(abls), figsize=(fs.DOUBLE, 2.5), squeeze=False, sharey=True)
        for ax, abl in zip(axes[0], abls):
            for arm in ARMS:
                s = tab[(tab["ablation"] == abl) & (tab["arm"] == arm)].sort_values("tape_len")
                if s.empty:
                    continue
                m, ls = ARM_STYLE[arm]
                ok = np.isfinite(s[col])
                ax.plot(s["tape_len"][ok], s[col][ok], color=ARM_COLOR[arm], marker=m, ls=ls, lw=1, label=f"@{arm}")
                if (~ok).any():
                    top = 300_000 if unit == "steps" else float(np.nanmax(tab[col][np.isfinite(tab[col])])) * 1.5
                    ax.plot(s["tape_len"][~ok], [top] * int((~ok).sum()), color=ARM_COLOR[arm], marker="^", ls="none", mfc="white")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(L_ALL, [str(x) if x in (3, 4, 9, 16, 25, 36, 49, 64, 81, 100) else "" for x in L_ALL])
            ax.minorticks_off()
            ax.set_title(abl, color=fs.color(abl))
            ax.set_xlabel("tape length L (bytes)")
        axes[0][0].set_ylabel(ylabel)
        h, l = axes[0][0].get_legend_handles_labels()
        h.append(Line2D([], [], marker="^", color="k", ls="none", mfc="white"))
        l.append("median not reached")
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False)
        fs.save(fig, os.path.join(out, fname))
    fig, axes = plt.subplots(1, len(abls), figsize=(fs.DOUBLE, 2.5), squeeze=False, sharey=True)
    rng = np.random.default_rng(0)
    for ax, abl in zip(axes[0], abls):
        g = size[(size["ablation"] == abl) & (size["t_rep"] > 0)]
        for arm in ARMS:
            gg = g[g["arm"] == arm]
            if gg.empty:
                continue
            x = gg["tape_len"].to_numpy(float) * np.exp(rng.normal(0, 0.03, len(gg)))
            p = gg["trep_period"].to_numpy(float)
            div = (2 * gg["tape_len"].to_numpy()) % np.maximum(p, 1).astype(int) == 0
            m, _ = ARM_STYLE[arm]
            ax.scatter(x[div], p[div], s=10, marker=m, color=ARM_COLOR[arm], alpha=0.8, lw=0, label=f"@{arm}")
            ax.scatter(x[~div], p[~div], s=12, marker=m, facecolors="none", edgecolors=ARM_COLOR[arm], lw=0.7)
        ax.plot(L_ALL, L_ALL, color="#aaaaaa", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(L_ALL, [str(x) if x in (3, 4, 9, 16, 25, 36, 49, 64, 81, 100) else "" for x in L_ALL])
        ax.minorticks_off()
        ax.set_title(abl, color=fs.color(abl))
        ax.set_xlabel("tape length L (bytes)")
    axes[0][0].set_ylabel("period of first replicator (bytes)")
    h, l = axes[0][0].get_legend_handles_labels()
    h += [Line2D([], [], marker="o", color="k", ls="none", ms=4), Line2D([], [], marker="o", color="k", ls="none", mfc="none", ms=4)]
    l += ["divides 2L", "does not divide 2L"]
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False)
    fs.save(fig, os.path.join(out, "E3_period"))

    # ── E4: budget sweep at L = 100 ──
    bud = asy[(asy["tape_len"] == 100) & asy["arm"].isin(["budget", "nominal"]) & (asy["k"] == 4) & asy["replicate"].isna()]
    tb = cell_rows(bud, ["ablation", "steps"]).sort_values(["ablation", "steps"])
    tb.to_csv(os.path.join(out, "budget_L100.csv"), index=False)
    md.append("## E4 budget sweep, L = 100\n\n" + tb[["ablation", "steps", "n", "t_rep_n", "t_rep_km", "t_faith_n", "whole_tape_n", "tiled_frac", "period_median", "final_rep_n", "func_rnd_median", "func_rnd_min"]].to_markdown(index=False, floatfmt=".2f") + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE, 2.5), sharey=True)
    for ax, abl in zip(axes, abls):
        s = tb[tb["ablation"] == abl]
        ax.errorbar(s["steps"], s["t_rep_frac"], yerr=[s["t_rep_frac"] - s["t_rep_lo"], s["t_rep_hi"] - s["t_rep_frac"]], color=fs.color(abl), marker="o", capsize=2, lw=1, label="heritable")
        ax.plot(s["steps"], s["t_faith_frac"], color=fs.color(abl), marker="s", ls="--", mfc="white", label="faithful")
        ax.plot(s["steps"], s["whole_tape_n"] / s["n"], color="#888888", marker="D", ls=":", label="whole-tape copier first")
        ax.axvline(100, color="#bbbbbb", lw=0.6, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 64, 128, 256, 512, 1024, 2048], [32, 64, 128, 256, 512, 1024, 2048])
        ax.set_title(f"{abl}, L = 100", color=fs.color(abl))
        ax.set_xlabel("Z80 steps per encounter")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("fraction of seeds (of 10)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False, title="dotted line: steps = L")
    fs.save(fig, os.path.join(out, "E4_budget"))

    # ── E5: mutation sweep at L = 100 ──
    mu = asy[(asy["tape_len"] == 100) & asy["arm"].isin(["musweep", "nominal"]) & (asy["steps"] == 128) & asy["replicate"].isna()]
    tm = cell_rows(mu, ["ablation", "k"]).sort_values(["ablation", "k"])
    tm["per_byte_mutation_rate"] = 8192 / np.power(2.0, tm["k"].astype(float)) / (20000 * 100)
    tm.to_csv(os.path.join(out, "mutation_L100.csv"), index=False)
    md.append("## E5 mutation sweep, L = 100\n\n" + tm[["ablation", "k", "per_byte_mutation_rate", "n", "t_rep_n", "t_rep_km", "t_faith_n", "period_median", "tiled_frac", "final_hoe_median"]].to_markdown(index=False, floatfmt=".3g") + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(fs.DOUBLE, 2.5))
    fig.subplots_adjust(wspace=0.55)
    for abl in abls:
        s = tm[tm["ablation"] == abl]
        axes[0].plot(s["per_byte_mutation_rate"], s["t_rep_frac"], color=fs.color(abl), marker="o", label=f"{abl}: heritable")
        axes[0].plot(s["per_byte_mutation_rate"], s["t_faith_frac"], color=fs.color(abl), marker="s", mfc="white", ls="--", label=f"{abl}: faithful")
        axes[1].plot(s["per_byte_mutation_rate"], s["period_median"], color=fs.color(abl), marker="o")
        axes[2].plot(s["per_byte_mutation_rate"], s["final_hoe_median"], color=fs.color(abl), marker="o")
    for ax, yl in zip(axes, ("fraction of seeds (of 10)", "median period of first replicator (bytes)", "final high-order entropy (bits/byte)")):
        ax.set_xscale("log")
        ax.set_xlabel("mutations per byte per step")
        ax.set_ylabel(yl)
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_yscale("log")
    axes[1].set_yticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
    axes[1].minorticks_off()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False)
    fig.suptitle("L = 100, 128 steps: mutation sweep (k = 1 … 8)", y=1.03)
    fs.save(fig, os.path.join(out, "E5_mutation"))

    # ── E6: within-seed vs between-seed variance ──
    lines = ["## E6 within-seed (10 repeats of seed 1) vs between-seed (seeds 101–110; `none` from Stage E @nominal, `stack-writes` from Stage C) variance\n"]
    for lab, L in (("none", 16), ("stack-writes", 16), ("none", 9)):
        within = asy[(asy["label"] == f"{lab}@var") & (asy["tape_len"] == L)]
        pool = asy
        c_path = os.path.join(os.path.dirname(a.dir.rstrip("/")), "stageC", "analysis", "assays.csv")
        if os.path.exists(c_path):
            c_asy = pd.read_csv(c_path)
            c_asy["source"] = "stageC"
            pool = pd.concat([asy, c_asy], ignore_index=True)
        between = pool[(pool["label"].isin([f"{lab}@nominal", lab])) & (pool["tape_len"] == L) & (pool["steps"] == 128) & (pool["k"] == 4) & pool["replicate"].isna() & (pool["horizon"] == 300_000)]
        between = between.drop_duplicates(subset=["label", "seed"])
        def desc(g):
            t = g["t_rep"].astype(float)
            em = t > 0
            return f"{int(em.sum())}/{len(g)} emerged, KM median {km_str(km_median(np.where(em, t, g['steps_run']).astype(float), em.to_numpy()))}, log10 SD among emerged {np.log10(t[em]).std():.2f}" if len(g) else "–"
        lines.append(f"- {lab} L = {L}: within-seed {desc(within)}; between-seed {desc(between)}")
    md.append("\n".join(lines) + "\n")

    # ── intrinsic fitness of the canonical units vs L: tiled tapes assayed in isolation (64 random partners, seed 0) ──
    from algocell_exp.assay import assay, GEN2_MIN
    units = {"pusher 01 c5": bytes.fromhex("01c5"), "LDIR-3 1d ed b0": bytes.fromhex("1dedb0"), "LDIR-4 04 5e ed b0": bytes.fromhex("045eedb0")}
    rows = []
    for L in L_ALL:
        for name, u in units.items():
            t = np.resize(np.frombuffer(u, dtype=np.uint8), L).copy()
            for steps in (32, 128):
                r = assay(t, z80_steps=steps, suppress=[], n=64, seed=0)
                rows.append({"unit": name, "L": L, "steps": steps, "tiles_exactly": L % len(u) == 0, "score": r["score"], "gen2": r["gen2_score"], "heritable": r["gen2_score"] >= GEN2_MIN,
                             "faithful": r["faithful"], "self_preserved_as_B": r["self_preserved_as_B"], "copy_offset": r.get("copy_offset")})
    tu = pd.DataFrame(rows)
    tu.to_csv(os.path.join(out, "unit_fitness_vs_L.csv"), index=False)
    piv = tu[tu["steps"] == 128].pivot_table(index="L", columns="unit", values="gen2")
    md.append("## Intrinsic heritability (gen2, isolated assay, 128 steps) of the canonical units tiled to each L\n\n" + piv.to_markdown(floatfmt=".2f") + "\n")
    piv32 = tu[tu["steps"] == 32].pivot_table(index="L", columns="unit", values="gen2")
    md.append("### same at 32 steps\n\n" + piv32.to_markdown(floatfmt=".2f") + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE, 2.5), sharey=True)
    for ax, steps in zip(axes, (128, 32)):
        for (name, u), m, c in zip(units.items(), ("o", "s", "^"), ("#000000", "#009E73", "#D55E00")):
            s_ = tu[(tu["unit"] == name) & (tu["steps"] == steps)].sort_values("L")
            ax.plot(s_["L"], s_["gen2"], marker=m, color=c, lw=1, ms=4, label=name)
            ex = s_[~s_["tiles_exactly"]]
            ax.scatter(ex["L"], ex["gen2"], marker=m, s=40, facecolors="none", edgecolors=c, lw=0.8, zorder=4)
        ax.axhline(GEN2_MIN, color="#999999", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(L_ALL, [str(x) if x in (3, 4, 9, 16, 25, 36, 49, 64, 81, 100) else "" for x in L_ALL])
        ax.minorticks_off()
        ax.set_xlabel("tape length L (bytes)")
        ax.set_title(f"{steps} Z80 steps per encounter")
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("gen2 of the tiled unit (isolated assay)")
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], marker="o", color="k", ls="none", mfc="none"))
    l.append("L not a multiple of the unit (truncated tiling)")
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False, title="dotted: heritability threshold")
    fs.save(fig, os.path.join(out, "E8_unit_fitness"))

    # ── E7: the 32-step · 1/4 block-copy exception, 20 new seeds ──
    from report import fisher
    x = asy[asy["arm"] == "x32k2"]
    if len(x):
        lines = ["## E7 the 32-step · mutation 1/4 block-copy exception (seeds 1001–1020)\n"]
        cnt = {}
        for lab, g in x.groupby("ablation"):
            n = len(g)
            cnt[lab] = (int((g["t_rep"] > 0).sum()), int((g["t_faith"] > 0).sum()), int((g["tq_10"] > 0).sum()), n)
            t = g["t_rep"].astype(float)
            em = t > 0
            lines.append(f"- {lab}: heritable {cnt[lab][0]}/{n}, faithful {cnt[lab][1]}/{n}, tq_10 {cnt[lab][2]}/{n}; KM median t_rep {km_str(km_median(np.where(em, t, g['steps_run']).astype(float), em.to_numpy()))}; periods {', '.join(f'{int(p)}×{c}' for p, c in g.loc[em, 'trep_period'].value_counts().sort_index().items()) or '–'}")
        if "none" in cnt and "block-copy" in cnt:
            a1, _, _, n1 = cnt["none"]
            a2, _, _, n2 = cnt["block-copy"]
            lines.append(f"- Fisher exact, heritable: none {a1}/{n1} vs block-copy {a2}/{n2}: two-sided p = {fisher(a1, n1 - a1, a2, n2 - a2):.3g}, one-sided (none greater) p = {fisher(a1, n1 - a1, a2, n2 - a2, 'greater'):.3g}")
        md.append("\n".join(lines) + "\n")

    with open(os.path.join(out, "NUMBERS_E.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print("wrote", out)


if __name__ == "__main__":
    main()
