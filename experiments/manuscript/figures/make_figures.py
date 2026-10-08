"""Composite main figures for the Nature manuscript, data panels only (conceptual panels are placeholders until designed
with the user). Every number plotted comes from a generated table under results/ or a recorded run file.

    python manuscript/figures/make_figures.py [--out manuscript/figures/out]

Figures (180 mm double column, depth ≤ 170 mm, Nature profile from figstyle):
  fig1  The soup and the order of events      a,b conceptual | c one world's time course | d emergence by L (KM)
  fig2  First replicator open, successor closed  a copied b self-damage (first vs final, 4 L) | c heritable fraction vs step | d conceptual | e convergence
  fig3  What the instruction set must provide    a atlas forest | b size axis + unit fitness | c dead-zone switch | d L = 9 reversal
  fig4  One instruction decides how life begins in BFF   a heritable fraction vs epoch by variant | b first vs final openness | c the all-P wave | d conceptual
  fig5  Closure as a cycle                     a conceptual | b closure window (data placeholder) | c fidelity vs context-dependence
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, EXP)
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

R = os.path.join(EXP, "results")
L_COL = dict(fs.L_COLOR)
L_COL.update({8: "#56B4E9", 9: "#CC79A7", 12: "#F0E442", 10: "#999999"})


def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def placeholder(ax, text):
    ax.set_axis_off()
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=6, color="#777777", transform=ax.transAxes, wrap=True)
    for sp in ax.spines.values():
        sp.set_visible(False)


def km_curve(times, horizon):
    """Kaplan–Meier fraction of worlds WITHOUT the event vs step (events at `times`; NaN or < 0 = censored at horizon)."""
    t = np.array([x if (x is not None and x == x and x >= 0) else np.inf for x in times], float)
    xs = [0.0]
    ys = [1.0]
    n = len(t)
    for v in sorted(set(t[np.isfinite(t)])):
        d = int((t == v).sum())
        at_risk = int((t >= v).sum())
        s = ys[-1] * (1 - d / at_risk)
        xs += [v, v]
        ys += [ys[-1], s]
    xs.append(horizon)
    ys.append(ys[-1])
    return np.array(xs), np.array(ys)


# ----------------------------------------------------------------------------------------------------------------- fig 1
def fig1(out):
    fig = plt.figure(figsize=(fs.DOUBLE, 62 * fs.MM))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.1, 1.1, 1.25, 1.25], wspace=0.55, left=0.04, right=0.99, top=0.9, bottom=0.2)
    axa, axb, axc, axd = [fig.add_subplot(gs[0, i]) for i in range(4)]
    placeholder(axa, "a  conceptual: the pair, the ring,\nthe pointer and the stack\n(to design)")
    placeholder(axb, "b  conceptual: the pusher\n01 c5 — code = data = literal\n(to design)")
    # c: one world's time course (Stage G, L = 16, seed 2001)
    try:
        run = glob.glob(os.path.join(EXP, "runs", "stageG", "none@closure_L16_st128_k4_s2001.jsonl"))[0]
        rows = [json.loads(l) for l in open(run) if '"kind": "sample"' in l]
        S = pd.DataFrame([{"step": r["step"], "zero": r.get("zero_frac", np.nan), "q": r.get("q_share", np.nan)} for r in rows]).sort_values("step")
        S = S[S["step"] > 0]
        c4 = pd.read_csv(os.path.join(R, "stageG", "c4", "functional.csv"))
        h = c4[(c4["tape_len"] == 16) & (c4["seed"] == 2001)].sort_values("step")
        axc.plot(S["step"], S["zero"], color=fs.CONCEPT["tar"], lw=0.9, label="zero bytes (tar)")
        axc.plot(S["step"], S["q"], color="#0072B2", lw=0.9, label="dominant tape, occupancy")
        axc.plot(h["step"], h["frac_heritable"], color=fs.CONCEPT["closed"], lw=0.9, label="heritable random cells")
        g = pd.read_csv(os.path.join(R, "stageG", "stageG", "stage_g_runs.csv"))
        trep = float(g[(g["L"] == 16) & (g["seed"] == 2001)]["t_rep"].iloc[0])
        axc.axvline(trep, color="#000000", lw=0.5, ls=":")
        axc.text(trep * 1.15, 0.97, "first\nreplicator", fontsize=5, va="top")
        axc.set_xscale("log")
        axc.set_xlim(40, 3.5e5)
        axc.set_ylim(0, 1.0)
        fs.tidy(axc, "step", "fraction")
        axc.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=5)
        fs.panel_label(axc, "c")
    except Exception as e:  # noqa: BLE001
        placeholder(axc, f"c  (data missing: {e})")
    # d: Kaplan–Meier emergence by L (Stage E @nominal, none)
    try:
        A = pd.read_csv(os.path.join(R, "stageE", "assays.csv"))
        A = A[(A["label"] == "none@nominal") & (A["steps"] == 128) & (A["k"] == 4)]
        if "replicate" in A:
            A = A[A["replicate"].isna()]
        for L in (8, 9, 16, 36, 64, 100):
            g = A[A["tape_len"] == L]
            if g.empty:
                continue
            x, y = km_curve(g["t_rep"].tolist(), 300000)
            axd.step(np.maximum(x, 40), 1 - y, where="post", color=L_COL.get(L, "#444444"), lw=0.9, label=f"L = {L}")
        axd.set_xscale("log")
        axd.set_xlim(40, 3.5e5)
        axd.set_ylim(0, 1.02)
        fs.tidy(axd, "step", "worlds with a heritable replicator")
        axd.legend(loc="lower right", fontsize=5, ncol=2)
        fs.panel_label(axd, "d")
    except Exception as e:  # noqa: BLE001
        placeholder(axd, f"d  (data missing: {e})")
    fs.save(fig, os.path.join(out, "fig1"))


# ----------------------------------------------------------------------------------------------------------------- fig 2
def fig2(out):
    g = pd.read_csv(os.path.join(R, "stageG", "stageG", "stage_g_runs.csv"))
    c4 = pd.read_csv(os.path.join(R, "stageG", "c4", "functional.csv"))
    Ls = [16, 20, 50, 64]
    fig = plt.figure(figsize=(fs.DOUBLE, 105 * fs.MM))
    gs = GridSpec(2, 4, figure=fig, hspace=0.6, wspace=0.5, left=0.07, right=0.99, top=0.95, bottom=0.14)
    rng = np.random.default_rng(0)
    # a, b: per-world first vs final
    for row, (col, ylab) in enumerate((("copied", "random partners that\nbecome a copy"), ("damaged", "encounters that damage\nthe organism"))):
        for i, L in enumerate(Ls):
            ax = fig.add_subplot(gs[row, i])
            d = g[g["L"] == L]
            jit = rng.uniform(-0.12, 0.12, len(d))
            for x, which in ((0, "first"), (1, "final")):
                loop = (d[f"{which}_has_cf"] | d[f"{which}_has_block"]).values
                v = d[f"{which}_{col}"].values
                ax.scatter(x + jit[~loop], v[~loop], s=9, facecolors="none", edgecolors=fs.CONCEPT["open"], lw=0.6)
                ax.scatter(x + jit[loop], v[loop], s=9, color=fs.CONCEPT["closed"], lw=0)
            ax.set_xticks([0, 1], ["first", "final"])
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_ylabel(ylab)
            else:
                ax.set_yticklabels([])
            if row == 0:
                hz = int(d['horizon'].iloc[0])
                ax.set_title(f"L = {L}, {'1M' if hz >= 1_000_000 else str(hz // 1000) + 'k'} steps", fontsize=6)
            fs.tidy(ax)
            if i == 0:
                fs.panel_label(ax, "ab"[row], x=-0.45)
    # legend for a/b
    h1 = plt.Line2D([], [], marker="o", ls="none", mfc="none", mec=fs.CONCEPT["open"], ms=3.5, label="no loop instruction")
    h2 = plt.Line2D([], [], marker="o", ls="none", color=fs.CONCEPT["closed"], ms=3.5, label="loop instruction (jump, return or LDIR)")
    fig.legend(handles=[h1, h2], loc="lower center", bbox_to_anchor=(0.5, -0.01), fontsize=5, ncol=2, frameon=False)
    fs.save(fig, os.path.join(out, "fig2_ab"))

    fig = plt.figure(figsize=(fs.DOUBLE, 55 * fs.MM))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.4, 1.1, 1.0], wspace=0.5, left=0.07, right=0.99, top=0.93, bottom=0.2)
    axc, axd, axe = [fig.add_subplot(gs[0, i]) for i in range(3)]
    # c: heritable fraction vs step per L (median + IQR)
    for L in Ls:
        c = c4[c4["tape_len"] == L].groupby("step")["frac_heritable"]
        med, lo, hi = c.median(), c.quantile(0.25), c.quantile(0.75)
        axc.plot(med.index, med.values, color=L_COL[L], lw=0.9, label=f"L = {L}")
        axc.fill_between(med.index, lo.values, hi.values, color=L_COL[L], alpha=0.15, lw=0)
    axc.set_xscale("log")
    axc.set_xlim(40, 1.2e6)
    axc.set_ylim(-0.02, 1.02)
    fs.tidy(axc, "step", "heritable fraction of random cells")
    axc.legend(fontsize=5, loc="upper left")
    fs.panel_label(axc, "c")
    placeholder(axd, "d  conceptual: the closers\n(RET NZ, JR NZ, DJNZ, LDIR)\nwith the cycle marked\n(to design)")
    # e: convergence and closure counts per L
    rows = []
    for L in Ls:
        d = g[g["L"] == L]
        modal = d["final_tape"].mode().iloc[0]
        rows.append({"L": L, "identical": int((d["final_tape"] == modal).sum()), "closed": int((d["final_copied"] >= 0.95).sum()),
                     "loop": int((d["final_has_cf"] | d["final_has_block"]).sum()), "n": len(d)})
    T = pd.DataFrame(rows)
    x = np.arange(len(Ls))
    w = 0.26
    axe.bar(x - w, T["loop"], w, color=fs.CONCEPT["closed"], label="loop instruction")
    axe.bar(x, T["closed"], w, color="#0072B2", label="copies ≥ 95% of partners")
    axe.bar(x + w, T["identical"], w, color="#999999", label="byte-identical to the modal tape")
    axe.set_xticks(x, [f"L = {L}" for L in Ls])
    axe.set_ylim(0, 20.5)
    fs.tidy(axe, None, "worlds (of 20)")
    axe.legend(fontsize=4.8, loc="upper right")
    fs.panel_label(axe, "e")
    fs.save(fig, os.path.join(out, "fig2_cde"))


# ----------------------------------------------------------------------------------------------------------------- fig 3
def fig3(out):
    fig = plt.figure(figsize=(fs.DOUBLE, 100 * fs.MM))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1.0], hspace=0.55, wspace=0.45, left=0.2, right=0.98, top=0.96, bottom=0.1)
    axa, axb, axc, axd = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    # a: atlas forest (Stage C, 128 steps, 1/16)
    try:
        C = pd.read_csv(os.path.join(R, "stageC", "stage_c", "c3_ablations_st128_k4.csv"))
        C = C[C["label"] != "none"].copy()
        C["ratio"] = C["km_ratio_vs_none"].replace([np.inf, -np.inf], np.nan)
        C = C.sort_values("ratio", na_position="last")
        y = np.arange(len(C))
        for yi, (_, r) in zip(y, C.iterrows()):
            col = fs.color(r["label"])
            if np.isfinite(r["ratio"]):
                axa.plot(r["ratio"], yi, "o", color=col, ms=3.2)
            else:
                axa.annotate("", xy=(3000, yi), xytext=(600, yi), arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
            axa.text(1.03, yi, f"{int(r['t_rep_n'])}/{int(r['n'])}", transform=axa.get_yaxis_transform(), fontsize=5, va="center")
        axa.axvline(1, color="#000000", lw=0.5, ls=":")
        axa.set_xscale("log")
        axa.set_xlim(0.3, 3000)
        axa.set_yticks(y, C["label"].tolist())
        fs.tidy(axa, "emergence delay vs unablated (ratio of KM medians)")
        axa.text(1.03, len(C) - 0.3, "alive", transform=axa.get_yaxis_transform(), fontsize=5, va="center", color="#555555")
        fs.panel_label(axa, "a", x=-0.62)
    except Exception as e:  # noqa: BLE001
        placeholder(axa, f"a (data missing: {e})")
    # b: size axis — fraction alive by L (none @nominal) with Wilson CI, plus the pusher's isolated heritability
    try:
        S = pd.read_csv(os.path.join(R, "stageE", "stage_e", "size_arms.csv"))
        S = S[(S["ablation"] == "none") & (S["arm"] == "nominal")].sort_values("tape_len")
        axb.errorbar(S["tape_len"], S["t_rep_frac"], yerr=[S["t_rep_frac"] - S["t_rep_lo"], S["t_rep_hi"] - S["t_rep_frac"]], fmt="o", color="#000000", ms=3, lw=0.6, capsize=1.5, label="worlds alive by 300k steps")
        U = pd.read_csv(os.path.join(R, "stageE", "stage_e", "unit_fitness_vs_L.csv"))
        U = U[(U["unit"].str.startswith("pusher")) & (U["steps"] == 128)].sort_values("L")
        axb.plot(U["L"], U["gen2"], color=fs.CONCEPT["closed"], lw=0.9, ls="--", label="pusher heritability in isolation")
        axb.axhline(0.3, color="#999999", lw=0.5, ls=":")
        axb.set_xscale("log")
        axb.set_xticks([3, 5, 8, 12, 16, 25, 36, 50, 64, 100], ["3", "5", "8", "12", "16", "25", "36", "50", "64", "100"])
        axb.minorticks_off()
        axb.set_ylim(-0.03, 1.03)
        fs.tidy(axb, "tape length L (bytes)", "fraction")
        axb.legend(fontsize=5, loc="lower right")
        fs.panel_label(axb, "b")
    except Exception as e:  # noqa: BLE001
        placeholder(axb, f"b (data missing: {e})")
    # c: dead-zone switch (Stage F4 + Stage E reference)
    try:
        F = pd.read_csv(os.path.join(R, "stageF", "stage_f", "rings.csv"))
        F = F[(F["ablation"] == "none") & (F["L"].isin([8, 10, 12]))].sort_values(["L", "P"])
        for L, mk in ((8, "s"), (10, "^"), (12, "o")):
            d = F[F["L"] == L]
            frac = d["t_rep_n"] / d["n"]
            axc.errorbar(d["P"], frac, yerr=[frac - d["t_rep_lo"], d["t_rep_hi"] - frac], fmt=mk, color=L_COL[L], ms=3.2, lw=0.6, capsize=1.5, label=f"L = {L}")
            for _, r in d.iterrows():
                if r["P"] == 2 * r["L"]:
                    axc.annotate("native ring", (r["P"], r["t_rep_n"] / r["n"]), textcoords="offset points", xytext=(0, -9), fontsize=4.5, ha="center", color="#555555")
        axc.set_ylim(-0.03, 1.03)
        fs.tidy(axc, "pair memory ring P (bytes)", "worlds alive by 300k steps")
        axc.legend(fontsize=5, loc="center right")
        fs.panel_label(axc, "c", x=-0.3)
    except Exception as e:  # noqa: BLE001
        placeholder(axc, f"c (data missing: {e})")
    # d: L = 9 reversal (Stage D)
    try:
        D = pd.read_csv(os.path.join(R, "stageD", "stage_d", "cells_D.csv"))
        if "k" in D:
            D = D[D["k"] == 4]
        D = D.drop_duplicates(subset=["label", "steps"])
        arms = [a for a in ["none", "stack-writes", "stack-write-only", "stack-read-only", "push", "call-rst-write"] if a in set(D["label"])]
        y = np.arange(len(arms))
        for st, mk, off in ((128, "o", -0.15), (512, "s", 0.15)):
            d = D[D["steps"] == st].set_index("label").reindex(arms)
            axd.errorbar(d["t_rep_frac"], y + off, xerr=[d["t_rep_frac"] - d["t_rep_lo"], d["t_rep_hi"] - d["t_rep_frac"]], fmt=mk, color="#000000" if st == 128 else "#0072B2", ms=3, lw=0.6, capsize=1.5, label=f"{st} steps per encounter")
        axd.set_yticks(y, arms)
        axd.set_xlim(-0.03, 1.03)
        fs.tidy(axd, "worlds alive at L = 9 (of 20)")
        axd.legend(fontsize=5, loc="lower right")
        fs.panel_label(axd, "d", x=-0.6)
    except Exception as e:  # noqa: BLE001
        placeholder(axd, f"d (data missing: {e})")
    fs.save(fig, os.path.join(out, "fig3"))


# ----------------------------------------------------------------------------------------------------------------- fig 4
def _allp_share(s):
    """Share of the all-`P` class (hex 50) among the ten largest classes of a sample; 0 when it is not among them."""
    for t in s["top"]:
        tp = t["tape"]
        if tp and set(tp[i:i + 2] for i in range(0, len(tp), 2)) == {"50"}:
            return t["share"]
    return 0.0


def fig4(out):
    from matplotlib import ticker as mticker
    bdir = os.path.join(EXP, "runs", "bff_modal", "bff")
    runs = pd.read_csv(os.path.join(R, "bff", "runs.csv"))
    runs["variant"] = runs["variant"].replace({"stdlit": "lit"})
    variants = [v for v in ["std", "wrap", "lit", "wraplit", "wraplitnh"] if v in set(runs["variant"])]
    xticks, xlabels = [0, 64, 256, 1024, 4096, 16384], ["0", "64", "256", "1,024", "4,096", "16,384"]

    def epoch_axis(ax):
        ax.set_xscale("symlog", linthresh=64)
        ax.set_xticks(xticks, xlabels)
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlim(0, 17500)
        ax.set_ylim(-0.02, 1.02)

    fig = plt.figure(figsize=(fs.DOUBLE, 112 * fs.MM))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.4, 1.0, 1.0], height_ratios=[1.0, 0.85], hspace=0.8, wspace=0.45, left=0.07, right=0.99, top=0.95, bottom=0.07)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    axd = fig.add_subplot(gs[1, :])
    # a: heritable fraction vs epoch, every run, by variant
    for v in variants:
        for run in runs[runs["variant"] == v]["run"]:
            p = os.path.join(bdir, run, "samples.jsonl")
            if not os.path.exists(p):
                continue
            S = pd.DataFrame([{"epoch": s["epoch"], "h": s["frac_heritable"]} for s in map(json.loads, open(p))])
            axa.plot(S["epoch"], S["h"].rolling(4, min_periods=1).mean(), color=fs.BFF_VARIANT[v], lw=0.5, alpha=0.55)
        axa.plot([], [], color=fs.BFF_VARIANT[v], lw=1.2, label=f"{fs.BFF_VARIANT_LABEL[v]} (n = {int((runs['variant'] == v).sum())})")
    epoch_axis(axa)
    fs.tidy(axa, "epoch", "heritable fraction of random tapes")
    fs.panel_label(axa, "a")
    handles, labels = axa.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.505), ncol=5, frameon=False, fontsize=5, handlelength=1.8, columnspacing=1.6)
    # b: first vs final openness per variant (black labels: Nature forbids coloured text)
    rng = np.random.default_rng(0)
    for i, v in enumerate(variants):
        d = runs[(runs["variant"] == v) & runs["t_top"].notna()]
        for j, which in enumerate(("first", "final")):
            x = i * 2.6 + j + rng.uniform(-0.15, 0.15, len(d))
            loop = d[f"{which}_loop"].astype(bool).values
            vals = d[f"{which}_entered"].values
            axb.scatter(x[~loop], vals[~loop], s=8, facecolors="none", edgecolors=fs.BFF_VARIANT[v], lw=0.6)
            axb.scatter(x[loop], vals[loop], s=8, color=fs.BFF_VARIANT[v], lw=0)
        axb.text(i * 2.6 + 0.5, 1.08, v, ha="center", fontsize=4.8, color="black")
    axb.set_xticks([i * 2.6 + j for i in range(len(variants)) for j in (0, 1)], [w for _ in variants for w in ("first", "final")], fontsize=4.5, rotation=90)
    axb.set_ylim(-0.03, 1.03)
    fs.tidy(axb, None, "encounters whose pointer\nenters the partner")
    fs.panel_label(axb, "b")
    # c: the all-P class under lethal (wraplit) and benign (wraplitnh) tar: one switch, collapse vs persistence
    for v in ("wraplit", "wraplitnh"):
        if v not in variants:
            continue
        for run in runs[runs["variant"] == v]["run"]:
            p = os.path.join(bdir, run, "samples.jsonl")
            if not os.path.exists(p):
                continue
            S = pd.DataFrame([{"epoch": s["epoch"], "share": _allp_share(s)} for s in map(json.loads, open(p))])
            axc.plot(S["epoch"], S["share"], color=fs.BFF_VARIANT[v], lw=0.6, alpha=0.6)
    epoch_axis(axc)
    fs.tidy(axc, "epoch", "share of the all-P class")
    fs.panel_label(axc, "c")
    placeholder(axd, "d  conceptual: the 2 × 2 classification — literal write channel × lethality of the tar — with the Z80, BFF, BFF+literal and the benign-tar cell placed (to design)")
    fs.save(fig, os.path.join(out, "fig4"))


# ----------------------------------------------------------------------------------------------------------------- fig 5
def fig5(out):
    fig = plt.figure(figsize=(fs.DOUBLE, 58 * fs.MM))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1.0, 1.0], wspace=0.5, left=0.05, right=0.99, top=0.93, bottom=0.2)
    axa, axb, axc = [fig.add_subplot(gs[0, i]) for i in range(3)]
    placeholder(axa, "a  conceptual: Theorem 2 — the pointer trajectory of an\nopen organism (runs into the partner) vs a closed one\n(a cycle inside its own bytes) (to design)")
    placeholder(axb, "b  closure window q∫n dt: open-population integral for\nthe Z80 (benign tar), BFF + literal (lethal tar) and\nthe benign-tar BFF cell — data + model (after follow-ups)")
    # c: fidelity vs context dependence, every measured replicator (Z80 first/final partner tests; BFF culture tests)
    try:
        rng = np.random.default_rng(3)
        runs = pd.read_csv(os.path.join(R, "bff", "runs.csv"))
        d = runs[runs["t_top"].notna()]
        for which, mk in (("first", "o"), ("final", "s")):
            loop = d[f"{which}_loop"].astype(bool).values
            xj = d[f"{which}_self_damage"].values + rng.uniform(-0.012, 0.012, len(d))
            yj = d[f"{which}_copies"].values + rng.uniform(-0.012, 0.012, len(d))
            axc.scatter(xj[~loop], yj[~loop], s=8, marker=mk, facecolors="none", edgecolors="#0072B2", lw=0.6)
            axc.scatter(xj[loop], yj[loop], s=8, marker=mk, color="#0072B2", lw=0)
        g = pd.read_csv(os.path.join(R, "stageG", "stageG", "stage_g_runs.csv"))
        for which, mk in (("first", "o"), ("final", "s")):
            loop = (g[f"{which}_has_cf"] | g[f"{which}_has_block"]).values
            xj = g[f"{which}_damaged"].values + rng.uniform(-0.012, 0.012, len(g))
            yj = g[f"{which}_copied"].values + rng.uniform(-0.012, 0.012, len(g))
            axc.scatter(xj[~loop], yj[~loop], s=8, marker=mk, facecolors="none", edgecolors=fs.CONCEPT["open"], lw=0.6)
            axc.scatter(xj[loop], yj[loop], s=8, marker=mk, color=fs.CONCEPT["closed"], lw=0)
        axc.set_xlim(-0.03, 1.03)
        axc.set_ylim(-0.03, 1.03)
        fs.tidy(axc, "encounters that damage the organism", "random partners that become a copy")
        h = [plt.Line2D([], [], marker="o", ls="none", mfc="none", mec="#000000", ms=3, label="Z80, no loop"), plt.Line2D([], [], marker="o", ls="none", color=fs.CONCEPT["closed"], ms=3, label="Z80, loop"),
             plt.Line2D([], [], marker="o", ls="none", mfc="none", mec="#0072B2", ms=3, label="BFF, no loop"), plt.Line2D([], [], marker="o", ls="none", color="#0072B2", ms=3, label="BFF, loop")]
        axc.legend(handles=h, fontsize=4.6, loc="lower left")
        fs.panel_label(axc, "c")
    except Exception as e:  # noqa: BLE001
        placeholder(axc, f"c (data missing: {e})")
    fs.save(fig, os.path.join(out, "fig5"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fs.setup()
    for name, fn in (("fig1", fig1), ("fig2", fig2), ("fig3", fig3), ("fig4", fig4), ("fig5", fig5)):
        if a.only and name not in a.only.split(","):
            continue
        try:
            fn(a.out)
            print("built", name)
        except Exception as e:  # noqa: BLE001
            print("FAILED", name, repr(e))


if __name__ == "__main__":
    main()
