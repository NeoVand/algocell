"""Revision-2 figures (round-2 reports, R3/R4/E): Fig. 6 v5 (What is proved: a Theorem 2; b census of self-writers by
period; c the copy-period law), Extended Data Fig. 9 v2 (invasions, with the aligned-length tests) and Extended Data
Fig. 10 (the copy-offset switch and closure under random registers). Every number plotted is read from a generated table.

    .venv/bin/python manuscript/figures/figs_r3.py [--only fig6v5,ed_invasions2,ed_switch]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_figures as mf  # noqa: E402
from make_figures import fs, cp, plt, GridSpec, R, save, placeholder, INK, GREY, RED, TEAL  # noqa: E402

EXP = mf.EXP
OPEN_C, REGEN_C, TRANS_C = mf.OPEN_C, mf.REGEN_C, mf.TRANS_C
LC = {16: "#000000", 20: "#E69F00", 32: "#009E73", 64: "#D55E00"}


def _letter(fig, ax, letter, y, dx=0.06, x=None):
    ax.apply_aspect()
    if x is None:
        x = max(0.004, ax.get_position().x0 - dx)
    fig.text(x, y, letter, fontsize=8, fontweight="bold", va="top", ha="left", gid="panel-label")


# ------------------------------------------------------------------------------------------------ Fig. 6 v5
def _census_panel(ax):
    """Self-writers among all words of period 2, 3 and 4 tiled to 16 bytes (Proposition 3), open against closed."""
    counts = {}
    for k in (2, 3, 4):
        d = pd.read_csv(os.path.join(R, "review_r2", f"census_p{k}.csv"))
        d = d[d.primitive_period == k] if "primitive_period" in d else d
        counts[k] = (int((~d.closed.astype(bool)).sum()), int(d.closed.astype(bool).sum()))
    x = np.arange(3)
    w = 0.36
    for i, k in enumerate((2, 3, 4)):
        o, c = counts[k]
        for dx, v, col in ((-w / 2, o, OPEN_C), (w / 2, c, REGEN_C)):
            if v > 0:
                ax.bar(i + dx, v, width=w * 0.92, color=col, edgecolor="none")
            ax.text(i + dx, max(v, 1) * 1.25, f"{v:,}", ha="center", va="bottom", fontsize=5.0, color=col)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 3000)
    ax.set_xticks(x, ["2", "3", "4"])
    ax.set_xlim(-0.6, 2.6)
    fs.tidy(ax, "period of the tiled word (bytes)", "self-writers")
    ax.text(-0.5, 2200, "open", color=OPEN_C, fontsize=5.5, va="top")
    ax.text(-0.5, 1050, "closed", color=REGEN_C, fontsize=5.5, va="top")


def _copy_period_panel(ax):
    """Transmissible sites against the copy offset d for confined block-copy cells of the population scans."""
    D = pd.read_csv(os.path.join(R, "offset", "offset_cells.csv"))
    D = D[D.offset <= D.L]
    xs = np.geomspace(4, 67, 200)
    ax.plot(xs, xs - 4, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(40, 30, "d − 4", color=GREY, fontsize=5.5, ha="left", va="top")
    for (L, d), e in D.groupby(["L", "offset"]):
        if len(e) < 3:
            continue
        med = e.n_sites.median()
        q1, q3 = e.n_sites.quantile(0.25), e.n_sites.quantile(0.75)
        col = LC.get(int(L), GREY)
        ms = 2.2 + 1.1 * np.log10(len(e))
        filled = d == L
        d = d * {16: 0.93, 20: 0.98, 32: 1.03, 64: 1.08}.get(int(L), 1.0)
        ax.errorbar(d, med, yerr=[[med - q1], [q3 - med]], fmt="none", ecolor=col, elinewidth=0.5, zorder=2)
        ax.plot(d, med, marker="o", ms=ms, mfc=col if filled else "white", mec=col, mew=0.7, zorder=3, ls="none")
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64], ["4", "8", "16", "32", "64"])
    ax.minorticks_off()
    ax.set_xlim(3.3, 80)
    ax.set_ylim(-3, 64)
    fs.tidy(ax, "copy offset d (bytes)", "transmissible sites")
    for i, L in enumerate((16, 20, 32, 64)):
        ax.text(3.6, 61 - 5.2 * i, f"L = {L}", color=LC[L], fontsize=5.2, va="center")
    ax.plot([11.5], [61], marker="o", ms=3.0, mfc="white", mec=INK, mew=0.7, ls="none")
    ax.text(13.2, 61, "d < L, regenerates", fontsize=5.2, va="center", color=INK)
    ax.plot([11.5], [55.8], marker="o", ms=3.0, mfc=INK, mec=INK, mew=0.7, ls="none")
    ax.text(13.2, 55.8, "d = L, transmits", fontsize=5.2, va="center", color=INK)


def fig6v5(out):
    """v5 Fig. 6 | What is proved and what is counted."""
    fig = plt.figure(figsize=(fs.DOUBLE, 62 * fs.MM))
    axa = fig.add_axes([0.01, 0.04, 0.44, 0.86])
    axb = fig.add_axes([0.535, 0.2, 0.15, 0.66])
    axc = fig.add_axes([0.775, 0.2, 0.215, 0.66])
    cp.fig6a(axa)
    axa.set_anchor("NW")
    for ax, fn, letter in ((axb, _census_panel, "b"), (axc, _copy_period_panel, "c")):
        try:
            fn(ax)
        except Exception as e:  # noqa: BLE001
            placeholder(ax, f"{letter} (data missing: {e})")
    fig.text(0.01, 0.985, "a", fontsize=8, fontweight="bold", va="top", ha="left", gid="panel-label")
    _letter(fig, axb, "b", 0.985, dx=0.07)
    _letter(fig, axc, "c", 0.985, dx=0.065)
    save(fig, os.path.join(out, "fig6v5"))


# ------------------------------------------------------------------------------------------------ ED Fig. 9 v2
def _aligned_panel(ax, L):
    D = pd.read_csv(os.path.join(R, "invasion_aligned", "invasion_aligned.csv"))
    d0 = D[D.L == L]
    first = True
    for sd, d in d0[(d0.resident == "pusher") & (d0.invader == "closer")].groupby("seed"):
        d = d[d.step > 0].sort_values("step")
        ax.plot(d.step, d.closer_share, color=REGEN_C, lw=0.8, label="closer, seeded into a pusher world" if first else None)
        ax.plot(d.step, d.pusher_share, color=OPEN_C, lw=0.8, label="pusher in that world" if first else None)
        first = False
    first = True
    for sd, d in d0[(d0.resident == "closer") & (d0.invader == "pusher")].groupby("seed"):
        d = d[d.step > 0].sort_values("step")
        ax.plot(d.step, d.pusher_share, color=OPEN_C, lw=0.8, ls=(0, (1, 1.2)), label="pusher, seeded into a closer world" if first else None)
        first = False
    ax.set_xscale("log")
    ax.set_xlim(8, 2.5e4)
    ax.set_ylim(-0.02, 1.02)
    fs.tidy(ax, "step", "share of cells (class)")
    name = "return closer" if L == 16 else "block-copy tiling"
    ax.text(0.02, 1.0, f"L = {L}, {name}", transform=ax.transAxes, fontsize=6, ha="left", va="bottom", gid="allow-outside")
    ax.legend(fontsize=5.3, loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=3, frameon=False, columnspacing=1.0)


def ed_invasions2(out):
    """Extended Data Fig. 9: invasion tests (V, A1, A2 of round 2; B of R3)."""
    fig = plt.figure(figsize=(fs.DOUBLE, 158 * fs.MM))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35, left=0.08, right=0.985, top=0.9, bottom=0.71)
    gs2 = GridSpec(1, 1, figure=fig, left=0.33, right=0.80, top=0.585, bottom=0.355)
    gs3 = GridSpec(1, 2, figure=fig, wspace=0.35, left=0.08, right=0.985, top=0.225, bottom=0.06)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs2[0, 0])
    axd, axe = fig.add_subplot(gs3[0, 0]), fig.add_subplot(gs3[0, 1])
    for ax, fn, letter in ((axa, mf._marker_vs_confined, "a"), (axb, mf._a1_curves, "b"), (axc, mf._inv_classes, "c"),
                           (axd, lambda a: _aligned_panel(a, 16), "d"), (axe, lambda a: _aligned_panel(a, 32), "e")):
        try:
            fn(ax)
        except Exception as e:  # noqa: BLE001
            placeholder(ax, f"{letter} (data missing: {e})")
    for ax, letter, y in ((axa, "a", 0.99), (axb, "b", 0.99), (axc, "c", 0.66), (axd, "d", 0.295), (axe, "e", 0.295)):
        _letter(fig, ax, letter, y, x=0.01 if letter == "c" else None, dx=0.07)
    save(fig, os.path.join(out, "ed_invasions2"))


# ------------------------------------------------------------------------------------------------ ED Fig. 10
START_C = {"R_into_T": REGEN_C, "T_into_R": TRANS_C, "mix50": "#6B7280"}
START_LAB = {"R_into_T": "1% regenerators", "T_into_R": "1% transmitters", "mix50": "50:50"}


def _traj(L, tar, mut):
    out = []
    for f in sorted(glob.glob(os.path.join(EXP, "runs", "offset", "offset", f"L{L}_{tar}_mut{mut}_*.jsonl"))):
        recs = [json.loads(x) for x in open(f)]
        c = recs[0]
        t = pd.DataFrame(recs[1:])
        t["Tsh"] = t["T"] / (t["R"] + t["T"]).where(t["R"] + t["T"] >= 1000)
        out.append((c, t))
    return out


def _switch_panel(ax, L, tar, title):
    seen = set()
    for c, t in _traj(L, tar, "on"):
        t = t[t.step > 0]
        lab = START_LAB[c["start"]] if c["start"] not in seen else None
        seen.add(c["start"])
        ax.plot(t.step, t.Tsh, color=START_C[c["start"]], lw=0.55, alpha=0.85, label=lab)
    ax.set_xscale("log")
    ax.set_xlim(40, 3.2e5)
    ax.set_ylim(-0.02, 1.02)
    fs.tidy(ax, "step", "transmitters among\ncore-carrying cells")
    ax.text(0.02, 1.02, title, transform=ax.transAxes, fontsize=6, ha="left", va="bottom", gid="allow-outside")


def _drift_panel(ax):
    rows = [(16, "benign", "L = 16, benign"), (16, "lethal", "L = 16, lethal"), (32, "benign", "L = 32, benign")]
    for i, (L, tar, lab) in enumerate(rows):
        ends = []
        for c, t in _traj(L, tar, "off"):
            ends.append(t.Tsh.iloc[-1])
        ends = np.array(ends)
        jit = (np.arange(len(ends)) - len(ends) / 2) * 0.035
        ax.plot(i + jit, ends, marker="o", ls="none", ms=2.6, mfc=INK, mec="none")
    ax.axhline(0.5, color=GREY, lw=0.5, ls=(0, (3, 2)))
    ax.set_xticks(range(3), [r[2] for r in rows])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.03, 1.03)
    fs.tidy(ax, None, "transmitter share at\n100,000 steps (no mutation)")
    ax.tick_params(axis="x", labelsize=5)


def _parse_md_table(path):
    rows = []
    for ln in open(path):
        if ln.startswith("|") and not ln.startswith("|---"):
            rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
    return rows[0], rows[1:]


def _encounter_panel(ax):
    head, rows = _parse_md_table(os.path.join(R, "offset", "OFFSET_ENCOUNTER.md"))
    comps = ["copies into a\nsoup partner", "survives as partner\nof a soup executor", "core intact\nas partner", "single mutant\nstill copies"]
    y = np.arange(4)[::-1]
    for r in rows:
        tar, typ = r[0], r[1]
        vals = [float(v) for v in r[2:6]]
        col = REGEN_C if typ == "R" else TRANS_C
        mk = "o" if tar == "benign" else "s"
        off = 0.12 if typ == "R" else -0.12
        ax.plot(vals, y + off, marker=mk, ls="none", ms=2.8, mfc=col if tar == "lethal" else "white", mec=col, mew=0.7)
    ax.set_yticks(y, comps)
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.6, 3.6)
    fs.tidy(ax, "share of 4,096 encounters", None)
    ax.tick_params(axis="y", labelsize=5)
    for i, (t, col) in enumerate((("regenerator", REGEN_C), ("transmitter", TRANS_C))):
        ax.text(0.3, 3.45 - 0.34 * i, t, color=col, fontsize=5.2, ha="left", va="center")
    ax.text(0.3, 2.77, "open, benign tar; filled, lethal", color=INK, fontsize=5.0, ha="left", va="center")


def _ref22_panel(ax):
    txt = open(os.path.join(R, "review_r2", "ROBUST_REF22.md")).read()
    vals = {}
    for name in ("regenerator R", "transmitter T"):
        m = re.search(r"\| " + name + r" \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|", txt)
        vals[name] = [float(v) for v in m.groups()]
    ks = [1, 4, 8]
    ax.plot(ks, vals["transmitter T"], color=TRANS_C, marker="o", ms=2.6, lw=0.8)
    ax.plot(ks, vals["regenerator R"], color=REGEN_C, marker="o", ms=2.6, lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, ["1", "4", "8"])
    ax.minorticks_off()
    ax.set_xlim(0.8, 10)
    ax.set_ylim(-0.03, 1.0)
    fs.tidy(ax, "successive random mutations", "exact self-copy into the\nzero partner (ref. 22)")
    ax.text(1.1, 0.95, "transmitter", color=TRANS_C, fontsize=5.2, va="center")
    ax.text(1.1, 0.07, "regenerator", color=REGEN_C, fontsize=5.2, va="center")


def _randreg_panel(ax):
    F = pd.read_csv(os.path.join(R, "review_r2", "first_closure.csv"))
    g = F[(F.stage.isin(["G", "L"])) & (F.L == 16)]
    t = np.sort(g.t_closed.dropna().values)
    n = len(g)
    xs = np.concatenate([[1e3], t, [3e6]])
    ys = np.concatenate([[0], np.arange(1, len(t) + 1) / n, [len(t) / n]])
    ax.step(xs, ys, where="post", color=INK, lw=0.9, label=None)
    D = pd.read_csv(os.path.join(R, "r4", "r4_long.csv"))
    steps = sorted(D.step.unique())
    frac = [(D[D.step == s].confined_given_heritable > 0.5).mean() for s in steps]
    ax.step([1e3] + steps + [3e6], [0] + frac + [frac[-1]], where="post", color=TRANS_C, lw=0.9, label=None)
    ax.plot(steps, frac, marker="o", ms=2.2, color=TRANS_C, ls="none")
    ax.set_xscale("log")
    ax.set_xlim(1e3, 3.3e6)
    ax.set_ylim(-0.02, 1.05)
    fs.tidy(ax, "step", "worlds closed")
    ax.text(1.1e3, 0.72, "zero registers\n(Stages G and L,\n30 worlds)", color=INK, fontsize=5.2, ha="left", va="center")
    ax.text(2.5e4, 0.2, "random registers\n(20 worlds)", color=TRANS_C, fontsize=5.2, ha="left", va="center")


def ed_switch(out):
    """Extended Data Fig. 10 | The copy-offset switch and closure without supplied registers."""
    fig = plt.figure(figsize=(fs.DOUBLE, 118 * fs.MM))
    gs = GridSpec(2, 3, figure=fig, wspace=0.62, hspace=0.75, left=0.09, right=0.985, top=0.9, bottom=0.1)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))
    axd, axe, axf = (fig.add_subplot(gs[1, i]) for i in range(3))
    for ax, fn, letter in ((axa, lambda a: _switch_panel(a, 32, "benign", "L = 32, benign tar"), "a"),
                           (axb, lambda a: _switch_panel(a, 16, "lethal", "L = 16, lethal tar"), "b"),
                           (axc, _drift_panel, "c"), (axd, _encounter_panel, "d"), (axe, _ref22_panel, "e"), (axf, _randreg_panel, "f")):
        try:
            fn(ax)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            placeholder(ax, f"{letter} (data missing: {e})")
    axa.legend(fontsize=5.2, loc="lower center", bbox_to_anchor=(1.25, 1.12), ncol=3, frameon=False)
    for ax, letter, y in ((axa, "a", 0.985), (axb, "b", 0.985), (axc, "c", 0.985), (axd, "d", 0.47), (axe, "e", 0.47), (axf, "f", 0.47)):
        _letter(fig, ax, letter, y, dx=0.075 if letter != "d" else 0.13)
    save(fig, os.path.join(out, "ed_switch"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    fs.setup()
    for name, fn in (("fig6v5", fig6v5), ("ed_invasions2", ed_invasions2), ("ed_switch", ed_switch)):
        if a.only and name not in a.only.split(","):
            continue
        fn(a.out)
        print("built", name)


if __name__ == "__main__":
    main()
