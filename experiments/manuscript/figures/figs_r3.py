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
sys.path.insert(0, mf.EXP)
from make_figures import fs, cp, plt, GridSpec, R, save, placeholder, INK, GREY, RED, TEAL  # noqa: E402

EXP = mf.EXP
OPEN_C, REGEN_C, TRANS_C = mf.OPEN_C, mf.REGEN_C, mf.TRANS_C


SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _logfmt(ax, axis="x"):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    f = FuncFormatter(lambda v, _: ("10" + str(int(round(np.log10(v)))).translate(SUP)) if v > 0 and abs(np.log10(v) - round(np.log10(v))) < 1e-9 else "")
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(LogLocator(base=10))
    a.set_major_formatter(f)
    a.set_minor_formatter(NullFormatter())


def _letter(fig, ax, letter, y, dx=0.06, x=None):
    ax.apply_aspect()
    if x is None:
        x = max(0.004, ax.get_position().x0 - dx)
    fig.text(x, y, letter, fontsize=8, fontweight="bold", va="top", ha="left", gid="panel-label")


# ------------------------------------------------------------------------------------------------ Fig. 6 v5
LC = {16: "#000000", 20: "#56B4E9", 32: "#0072B2", 64: "#999999"}


def _theorem_panel(ax):
    """Theorem 2 as used for the Z80 with benign tar (concept): open in black, closed in vermilion (the paper's concepts)."""
    cp.finish(ax, (-0.5, 30.5), (0.2, 13.0))
    W = 1.25
    org = ["", "", "", "…", "", "", "", ""]
    ax.add_patch(mf.Rectangle((0.5, 11.25), 29.0, 1.05, facecolor="white", edgecolor=INK, lw=0.5, zorder=2))
    ax.text(15.0, 11.77, "128 executions per encounter", ha="center", va="center", fontsize=6, color=INK, zorder=3)
    ax.add_patch(mf.Rectangle((0.5, 9.95), 29.0 * 100 / 128, 1.05, facecolor=cp.TEAL_FILL, edgecolor=INK, lw=0.5, zorder=2))
    ax.text(0.5 + 29.0 * 100 / 256, 10.47, "at most 100 cells of the organism", ha="center", va="center", fontsize=6, color=INK, zorder=3)
    y = 6.0
    cp.text_runs(ax, 0.5, y + 1.6, [("open", INK, True), (" · leaves its cells", INK, False)], fs=6.5)
    cp.strip(ax, 0.5, y, org, w=W, h=1.2)
    cp.strip(ax, 0.5 + W * len(org), y, ["", "", "", "…", "", ""], fill=cp.GREY_FILL, w=W, h=1.2)
    cp.arrow(ax, (0.9, y - 0.4), (0.5 + W * 13.7, y - 0.4), color=INK, lw=1.0)
    ax.text(0.5 + W * 14 + 0.6, y + 0.6, "partner", ha="left", va="center", fontsize=6, color=INK)
    y = 1.6
    cp.text_runs(ax, 0.5, y + 1.6, [("closed", RED, True), (" · revisits a cell", INK, False)], fs=6.5)
    cp.strip(ax, 0.5, y, org, w=W, h=1.2)
    x_end = 0.5 + W * len(org)
    cp.arrow(ax, (0.9, y - 0.4), (x_end - 0.4, y - 0.4), color=INK, lw=1.0)
    cp.arrow(ax, (x_end - 0.4, y - 0.6), (1.1, y - 0.6), color=RED, lw=1.0, rad=-0.18)
    ax.text(x_end + 0.9, y + 0.8, "128 executions in L cells:", ha="left", va="bottom", fontsize=6, color=INK)
    ax.text(x_end + 0.9, y + 0.6, "some cell runs twice, a cycle", ha="left", va="top", fontsize=6, color=INK)


def _census_panel(ax):
    """Self-writers among all words of period 2, 3 and 4 tiled to 16 bytes (Proposition 3), open against closed."""
    counts = {}
    for k in (2, 3, 4):
        d = pd.read_csv(os.path.join(R, "review_r2", f"census_p{k}.csv"))
        d = d[d.primitive_period == k] if "primitive_period" in d else d
        counts[k] = (int((~d.closed.astype(bool)).sum()), int(d.closed.astype(bool).sum()))
    for i, k in enumerate((2, 3, 4)):
        o, c = counts[k]
        for dx, v, col in ((-0.13, o, OPEN_C), (0.13, c, REGEN_C)):
            ax.plot([i + dx, i + dx], [0, v], color=col, lw=0.8, solid_capstyle="butt")
            ax.plot(i + dx, v, marker="o", ms=3.2, mfc=col, mec=col, ls="none")
            ax.text(i + dx, v + 22, f"{v:,}", ha="center", va="bottom", fontsize=5.0, color=col)
    ax.set_ylim(0, 650)
    ax.set_xticks(range(3), ["2", "3", "4"])
    ax.set_xlim(-0.55, 2.55)
    fs.tidy(ax, "period of the tiled word (bytes)", "self-writers")
    ax.text(-0.45, 630, "open", color=OPEN_C, fontsize=5.5, va="top")
    ax.text(-0.45, 565, "closed", color=REGEN_C, fontsize=5.5, va="top")


def _copy_period_panel(ax):
    """Transmissible sites against the copy offset d for every confined block-copy cell with d <= L (medians by group)."""
    D = pd.read_csv(os.path.join(R, "offset", "offset_cells.csv"))
    D = D[D.offset <= D.L]
    xs = np.geomspace(4, 67, 200)
    ax.plot(xs, xs - 4, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(40, 30, "d − 4", color=GREY, fontsize=5.5, ha="left", va="top")
    for (L, d), e in D.groupby(["L", "offset"]):
        med = e.n_sites.median()
        col = LC.get(int(L), GREY)
        x = d * {16: 0.93, 20: 0.98, 32: 1.03, 64: 1.08}.get(int(L), 1.0)
        if int(L) % int(d):
            ax.plot(x, med, marker="x", ms=3.0, mec=col, mew=0.8, ls="none", zorder=3)
            continue
        ms = 2.0 + 1.1 * np.log10(len(e))
        ax.plot(x, med, marker="o", ms=ms, mfc=col if d == L else "white", mec=col, mew=0.7, zorder=3, ls="none")
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64], ["4", "8", "16", "32", "64"])
    ax.minorticks_off()
    ax.set_xlim(3.3, 80)
    ax.set_ylim(-3, 64)
    fs.tidy(ax, "copy offset d (bytes)", "transmissible sites")
    for i, L in enumerate((16, 20, 32, 64)):
        ax.text(3.6, 61 - 5.2 * i, f"L = {L}", color=LC[L], fontsize=5.2, va="center")
    for yy, kw, txt in ((61, dict(marker="o", mfc="white"), "d < L, regenerates"), (55.8, dict(marker="o", mfc=GREY), "d = L, transmits"),
                        (50.6, dict(marker="x"), "d does not divide L")):
        ax.plot([11.5], [yy], ms=3.0, mec=GREY, mew=0.7, ls="none", **kw)
        ax.text(13.2, yy, txt, fontsize=5.2, va="center", color=INK)


def fig6v5(out):
    """v5 Fig. 6 | What is proved and what is counted."""
    fig = plt.figure(figsize=(fs.DOUBLE, 62 * fs.MM))
    axa = fig.add_axes([0.01, 0.04, 0.44, 0.86])
    axb = fig.add_axes([0.535, 0.2, 0.15, 0.66])
    axc = fig.add_axes([0.775, 0.2, 0.215, 0.66])
    _theorem_panel(axa)
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
def _aligned_panel(ax, L, legend=False):
    D = pd.read_csv(os.path.join(R, "invasion_aligned", "invasion_aligned.csv"))
    d0 = D[D.L == L]
    first = True
    for sd, d in d0[(d0.resident == "pusher") & (d0.invader == "none")].groupby("seed"):
        d = d[d.step > 0].sort_values("step")
        ax.plot(d.step, d.pusher_share, color="#B4BAC1", lw=0.8, label="pusher, unseeded pusher world" if first else None, zorder=1)
        first = False
    first = True
    for sd, d in d0[(d0.resident == "pusher") & (d0.invader == "closer")].groupby("seed"):
        d = d[d.step > 0].sort_values("step")
        ax.plot(d.step, d.closer_share, color=REGEN_C, lw=0.8, label="closer, seeded at 1% into a pusher world" if first else None, zorder=3)
        ax.plot(d.step, d.pusher_share, color=OPEN_C, lw=0.8, label="pusher, in those worlds" if first else None, zorder=2)
        first = False
    first = True
    for sd, d in d0[(d0.resident == "closer") & (d0.invader == "pusher")].groupby("seed"):
        d = d[d.step > 0].sort_values("step")
        ax.plot(d.step, d.pusher_share, color=OPEN_C, lw=0.8, ls=(0, (1, 1.2)), label="pusher, seeded at 1% into a closer world (at zero)" if first else None, zorder=2)
        first = False
    ax.set_xscale("log")
    ax.set_xlim(8, 2.5e4)
    ax.set_ylim(-0.02, 1.02)
    fs.tidy(ax, "step", "share of cells (class)")
    name = "return closer" if L == 16 else "block-copy tiling"
    ax.text(0.02, 1.0, f"L = {L}, {name}", transform=ax.transAxes, fontsize=6, ha="left", va="bottom", gid="allow-outside")
    if legend:
        ax.legend(fontsize=5.3, loc="lower left", bbox_to_anchor=(0.0, 1.1), ncol=2, frameon=False, columnspacing=1.2)


def ed_invasions2(out):
    """Extended Data Fig. 9: invasion tests (V, A1, A2 of round 2; B of R3)."""
    fig = plt.figure(figsize=(fs.DOUBLE, 158 * fs.MM))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35, left=0.08, right=0.985, top=0.9, bottom=0.71)
    gs2 = GridSpec(1, 1, figure=fig, left=0.33, right=0.80, top=0.585, bottom=0.355)
    gs3 = GridSpec(1, 2, figure=fig, wspace=0.35, left=0.08, right=0.985, top=0.2, bottom=0.06)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs2[0, 0])
    axd, axe = fig.add_subplot(gs3[0, 0]), fig.add_subplot(gs3[0, 1])
    for ax, fn, letter in ((axa, mf._marker_vs_confined, "a"), (axb, mf._a1_curves, "b"), (axc, mf._inv_classes, "c"),
                           (axd, lambda a: _aligned_panel(a, 16, legend=True), "d"), (axe, lambda a: _aligned_panel(a, 32), "e")):
        try:
            fn(ax)
        except Exception as e:  # noqa: BLE001
            placeholder(ax, f"{letter} (data missing: {e})")
    for ax, letter, y in ((axa, "a", 0.99), (axb, "b", 0.99), (axc, "c", 0.66), (axd, "d", 0.285), (axe, "e", 0.285)):
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


def _switch_panel(ax, L, tar, title, ylabel=True):
    seen = set()
    for c, t in _traj(L, tar, "on"):
        t = t[t.step > 0].copy()
        raw = t["T"] / (t["R"] + t["T"]).replace(0, np.nan)
        ok = (t["R"] + t["T"]) >= 1000
        sm = raw.copy()
        late = t.step > 5000
        sm[late] = raw[late].rolling(9, center=True, min_periods=3).median()
        sm[~ok] = np.nan
        lab = START_LAB[c["start"]] if c["start"] not in seen else None
        seen.add(c["start"])
        ax.plot(t.step, sm, color=START_C[c["start"]], lw=0.55, alpha=0.85, label=lab)
        if (~ok & (t.step > 20000)).any():
            last = sm.last_valid_index()
            ax.plot(t.step[last], sm[last], marker="x", ms=3.2, mew=0.8, color=START_C[c["start"]], zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(40, 3.2e5)
    ax.set_ylim(-0.02, 1.02)
    fs.tidy(ax, "step", "transmitters among\ncore-carrying cells" if ylabel else None)
    _logfmt(ax)
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
    ax.set_xticks(range(3), [r[2].replace(", ", ",\n") for r in rows])
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
    comps = ["copies", "survives", "core", "mutant"]
    off = {("benign", "R"): -0.27, ("benign", "T"): -0.11, ("lethal", "R"): 0.11, ("lethal", "T"): 0.27}
    for r in rows:
        tar, typ = r[0], r[1]
        col = REGEN_C if typ == "R" else TRANS_C
        for k in range(4):
            v = float(r[2 + k])
            ax.plot(k + off[(tar, typ)], v, marker="o", ls="none", ms=2.8, mfc=col if tar == "lethal" else "white", mec=col, mew=0.7)
    ax.set_xticks(range(4), comps, fontsize=5)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(-0.03, 1.3)
    ax.set_yticks([0, 0.5, 1.0])
    fs.tidy(ax, None, "share of 4,096 encounters")
    ax.spines["left"].set_bounds(0, 1.0)
    for k, (t, col) in enumerate((("regenerator", REGEN_C), ("transmitter", TRANS_C))):
        ax.text(1.45, 1.27 - 0.085 * k, t, color=col, fontsize=5.0, ha="left", va="center")
    for k, (t, fill) in enumerate((("benign tar", "white"), ("lethal tar", INK))):
        y = 1.27 - 0.085 * (k + 2)
        ax.plot([1.55], [y], marker="o", ms=2.6, mfc=fill, mec=INK, mew=0.6, ls="none")
        ax.text(1.75, y, t, color=INK, fontsize=5.0, ha="left", va="center")


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
    """Fraction of worlds closed (most heritable cells among 64 random cells confined) at recorded snapshots, L = 16."""
    C = pd.read_csv(os.path.join(R, "population", "classes_snapshots.csv"))
    g = C[C.stage.isin(["G", "L"]) & (C.L == 16) & C.step.notna() & (C.step <= 3e6)]
    g = g.drop_duplicates(["stage", "seed", "step"])
    zs = g.groupby("step").apply(lambda d: ((d.frac_confined_of_heritable > 0.5).sum() / d.seed.nunique(), d.seed.nunique()))
    steps0 = [st for st in zs.index if zs[st][1] >= 20]
    ax.plot(steps0, [zs[st][0] for st in steps0], marker="o", ms=2.2, color=INK, lw=0.6)
    D = pd.concat([pd.read_csv(os.path.join(R, "r4", f)) for f in ("r4_long.csv", "r4_rep.csv")])
    steps = sorted(D.step.unique())
    frac = [(D[D.step == st].confined_given_heritable > 0.5).mean() for st in steps]
    ax.plot(steps, frac, marker="s", ms=2.2, color="#CC79A7", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlim(300, 4e6)
    ax.set_ylim(-0.02, 1.05)
    fs.tidy(ax, "step", "fraction of worlds closed")
    _logfmt(ax)
    ax.text(3.6e6, 0.86, "zero registers", color=INK, fontsize=5.2, ha="right", va="center")
    ax.text(3.6e6, 0.45, "random registers\n(40 worlds)", color="#CC79A7", fontsize=5.2, ha="right", va="center")


def ed_switch(out):
    """Extended Data Fig. 10 | The copy-offset switch and closure without supplied registers."""
    fig = plt.figure(figsize=(fs.DOUBLE, 122 * fs.MM))
    gs1 = GridSpec(1, 3, figure=fig, wspace=0.55, left=0.09, right=0.985, top=0.87, bottom=0.58)
    gs2 = GridSpec(1, 4, figure=fig, wspace=0.62, left=0.09, right=0.985, top=0.4, bottom=0.1)
    axa, axb, axc = (fig.add_subplot(gs1[0, i]) for i in range(3))
    axd, axe, axf, axg = (fig.add_subplot(gs2[0, i]) for i in range(4))
    for ax, fn, letter in ((axa, lambda a: _switch_panel(a, 32, "benign", "L = 32, benign tar"), "a"),
                           (axb, lambda a: _switch_panel(a, 32, "lethal", "L = 32, lethal tar", ylabel=False), "b"),
                           (axc, lambda a: _switch_panel(a, 16, "lethal", "L = 16, lethal tar", ylabel=False), "c"),
                           (axd, _drift_panel, "d"), (axe, _encounter_panel, "e"), (axf, _ref22_panel, "f"), (axg, _randreg_panel, "g")):
        try:
            fn(ax)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            placeholder(ax, f"{letter} (data missing: {e})")
    axb.legend(fontsize=5.2, loc="lower center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False)
    for ax, letter, y in ((axa, "a", 0.95), (axb, "b", 0.95), (axc, "c", 0.95), (axd, "d", 0.45), (axe, "e", 0.45), (axf, "f", 0.45), (axg, "g", 0.45)):
        _letter(fig, ax, letter, y, dx=0.07)
    save(fig, os.path.join(out, "ed_switch"))


# ------------------------------------------------------------------------------------------------ ED Fig. 11
MODE_C = {"open": OPEN_C, "regenerator": REGEN_C, "transmitter": TRANS_C}


def _pe_panel(ax):
    D = pd.read_csv(os.path.join(R, "pe", "pe_matrix.csv"))
    envs = list(dict.fromkeys(D.environment))
    pars = list(dict.fromkeys(D.parent))
    env_lab = {e: e.replace("random partners, ", "random partners,\n").replace("closed benign world", "closed\nbenign world").replace("open phase (t5000)", "open phase\n(step 5,000)").replace("random registers", "random\nregisters").replace("lethal world", "lethal\nworld") for e in envs}
    for i, pa in enumerate(pars):
        for j, e in enumerate(envs):
            r = D[(D.parent == pa) & (D.environment == e)].iloc[0]
            col = MODE_C[r["mode"]]
            alive = r.alive_g8 >= 0.5
            ax.plot(j, -i, marker="o", ms=9, mfc=col if alive else "white", mec=col, mew=0.8, ls="none")
            ax.text(j, -i, f"{int(r.sites_g8)}" if alive else "", ha="center", va="center", fontsize=5.2, color="white", fontweight="bold")
    ax.set_xticks(range(len(envs)), [env_lab[e] for e in envs], fontsize=5)
    ax.set_yticks([-i for i in range(len(pars))], pars, fontsize=5.2)
    ax.set_xlim(-0.6, len(envs) - 0.4)
    ax.set_ylim(-len(pars) + 0.4, 0.6)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.set_ticks_position("top")


DIAL_ORDER = ["lp0", "lp001", "lp003", "lp01", "lp03", "lp1"]
DIAL_LAB = ["0", "0.01", "0.03", "0.1", "0.3", "1"]


def _dial_df():
    return pd.read_csv(os.path.join(R, "dial", "dial_worlds.csv"))


def _dial_open(ax):
    D = _dial_df()
    xs = range(len(DIAL_ORDER))
    o = [int(D[D.variant == v].first_open.fillna(False).astype(bool).sum()) for v in DIAL_ORDER]
    n = [int((D.variant == v).t_rep.notna().sum()) if False else int(D[D.variant == v].t_rep.notna().sum()) for v in DIAL_ORDER]
    ax.plot(xs, [a / b if b else np.nan for a, b in zip(o, n)], marker="o", color=OPEN_C, ms=3, ls="none")
    ax.set_xticks(list(xs), DIAL_LAB)
    ax.set_ylim(-0.03, 1.05)
    fs.tidy(ax, "probability p that a zero halts", "worlds whose first\nreplicator is open (fraction)")


def _dial_closure(ax):
    D = _dial_df()
    hz = D.horizon.max()
    from dial_soups import _km_median
    for k, v in enumerate(DIAL_ORDER):
        d = D[D.variant == v]
        t = d.t_closed.values.astype(float)
        cens = ~np.isfinite(t)
        y = np.where(cens, hz * 1.35, t)
        jit = (np.arange(len(t)) - len(t) / 2) * 0.04
        ax.plot((k + jit)[~cens], y[~cens], marker="o", ls="none", ms=2.4, mfc="#8A929B", mec="none")
        if cens.any():
            ax.plot((k + jit)[cens], y[cens], marker="o", ls="none", ms=2.4, mfc="white", mec="#8A929B", mew=0.6)
        ax.plot([k - 0.25, k + 0.25], [_km_median(d.t_closed.tolist(), hz)] * 2, color=INK, lw=1.0)
    ax.set_yscale("log")
    ax.axhline(hz, color=GREY, lw=0.5, ls=(0, (3, 2)))
    ax.set_xticks(range(len(DIAL_ORDER)), DIAL_LAB)
    fs.tidy(ax, "probability p that a zero halts", "first closed replicator (step)")
    _logfmt(ax, "y")


def _dial_trans(ax):
    D = _dial_df()
    xs = range(len(DIAL_ORDER))
    for k, v in enumerate(DIAL_ORDER):
        d = D[D.variant == v]
        y = d.frac_transmitter_of_heritable.values
        jit = (np.arange(len(y)) - len(y) / 2) * 0.04
        ax.plot(k + jit, y, marker="o", ls="none", ms=2.4, mfc=TRANS_C, mec="none", alpha=0.85)
        ax.plot([k - 0.25, k + 0.25], [np.nanmedian(y)] * 2, color=INK, lw=0.9)
    ax.set_xticks(list(xs), DIAL_LAB)
    ax.set_ylim(-0.03, 1.05)
    fs.tidy(ax, "probability p that a zero halts", "transmitters among heritable\ncells at 300,000 steps")


def ed_dial(out):
    """Extended Data Fig. 11 | Parent, environment and the lethality dial."""
    fig = plt.figure(figsize=(fs.DOUBLE, 118 * fs.MM))
    axa = fig.add_axes([0.17, 0.56, 0.8, 0.33])
    gs = GridSpec(1, 3, figure=fig, wspace=0.5, left=0.08, right=0.985, top=0.41, bottom=0.1)
    axb, axc, axd = (fig.add_subplot(gs[0, i]) for i in range(3))
    for ax, fn, letter in ((axa, _pe_panel, "a"), (axb, _dial_open, "b"), (axc, _dial_closure, "c"), (axd, _dial_trans, "d")):
        try:
            fn(ax)
        except Exception as e:  # noqa: BLE001
            placeholder(ax, f"{letter} (data missing: {e})")
    fig.text(0.01, 0.97, "a", fontsize=8, fontweight="bold", va="top", ha="left", gid="panel-label")
    for ax, letter in ((axb, "b"), (axc, "c"), (axd, "d")):
        _letter(fig, ax, letter, 0.465, dx=0.07)
    save(fig, os.path.join(out, "ed_dial"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    fs.setup()
    for name, fn in (("fig6v5", fig6v5), ("ed_invasions2", ed_invasions2), ("ed_switch", ed_switch), ("ed_dial", ed_dial)):
        if a.only and name not in a.only.split(","):
            continue
        fn(a.out)
        print("built", name)


if __name__ == "__main__":
    main()
