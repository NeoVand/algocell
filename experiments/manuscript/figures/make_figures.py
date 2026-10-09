"""Composite figures for the Nature manuscript. Every number plotted comes from a generated table under results/ or a
recorded run file; the conceptual panels come from concept.py (code-drawn stand-ins until the designer's set arrives).

    python manuscript/figures/make_figures.py [--out manuscript/figures/out] [--only fig2,fig3]

Main figures (180 mm double column, depth <= 170 mm, Nature profile from figstyle):
  fig1  The first replicator and its closure     a design panel (refs/designer_fig1_round3.png if present) | b emergence by L (KM)
  fig2  One world, watched                        a seven lattice frames (seed 2002) | b its time course | c a byte-resolution window
  fig3  The first replicator is open, the successor closed
                                                 a copies b self-damage c information inflow (first -> final per world, slope charts)
                                                 d heritable fraction vs step | e convergence counts | f control flow of the closers (concept)
  fig4  What the instruction set must provide    a atlas forest | b size axis + unit fitness | c dead-zone switch | d L = 9 reversal
  fig5  One instruction decides how life begins in BFF   a heritable fraction vs epoch | b first vs final openness | c the all-P wave | d classification (concept)
  fig6  Closure requires a cycle                 the theorem as a diagram (concept, single panel)
Extended Data drawn here: ed12 (assembly measure against the culture test), ed13 (lethal tar in the first machine).
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
EXP = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, EXP)
import figstyle as fs  # noqa: E402
sys.path.insert(0, HERE)
import concept as cp  # noqa: E402
import figcheck  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

R = os.path.join(EXP, "results")
L_COL = dict(fs.L_COLOR)
L_COL.update({8: "#56B4E9", 9: "#E69F00", 12: "#F0E442", 10: "#999999"})
INK, GREY, RULE, TEAL, RED = cp.INK, "#6B7280", "#B4BAC1", cp.TEAL, cp.RED
LOOP_MS = 7          # marker area for the slope charts


def save(fig, path_no_ext):
    """Run the layout checks, print the report, then save. A figure with problems is still written so it can be inspected."""
    figcheck.print_report(figcheck.check(fig), os.path.basename(path_no_ext))
    fs.save(fig, path_no_ext)


def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def label(ax, letter, dx=0.052, dy=0.006):
    """Panel letter at a fixed offset from the panel's box, so letters align across rows and panel types."""
    ax.apply_aspect()
    pos = ax.get_position()
    ax.figure.text(pos.x0 - dx, pos.y1 + dy, letter, fontsize=8, fontweight="bold", va="bottom", ha="left", gid="panel-label")


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
    for v in sorted(set(t[np.isfinite(t)])):
        d = int((t == v).sum())
        at_risk = int((t >= v).sum())
        s = ys[-1] * (1 - d / at_risk)
        xs += [v, v]
        ys += [ys[-1], s]
    xs.append(horizon)
    ys.append(ys[-1])
    return np.array(xs), np.array(ys)


def loop_flags(d, which):
    return (d[f"{which}_has_cf"].astype(bool) | d[f"{which}_has_block"].astype(bool)).values


# ----------------------------------------------------------------------------------------------------------------- fig 1
def km_panel(ax, ncol=3):
    """b: Kaplan–Meier emergence by tape length (Stage E, none@nominal, 128 steps, k = 4)."""
    A = pd.read_csv(os.path.join(R, "stageE", "assays.csv"))
    A = A[(A["label"] == "none@nominal") & (A["steps"] == 128) & (A["k"] == 4)]
    if "replicate" in A:
        A = A[A["replicate"].isna()]
    for L in (8, 9, 16, 36, 64, 100):
        g = A[A["tape_len"] == L]
        if g.empty:
            continue
        x, y = km_curve(g["t_rep"].tolist(), 300000)
        ax.step(np.maximum(x, 40), 1 - y, where="post", color=L_COL.get(L, "#444444"), lw=0.9, label=f"L = {L}")
    ax.set_xscale("log")
    ax.set_xlim(40, 3.5e5)
    ax.set_ylim(-0.03, 1.03)
    fs.tidy(ax, "step", "worlds with a heritable replicator")
    ax.set_gid("allow-clip")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, fontsize=5, frameon=False, columnspacing=1.0, handlelength=1.4)


COMPOSE_TEX = r"""\documentclass{article}
\usepackage[paperwidth=180mm,paperheight=%(h)smm,margin=0mm,top=2mm,headheight=0pt,headsep=0pt]{geometry}
\usepackage{fontspec}
\setsansfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\usepackage{graphicx}
\pagestyle{empty}\setlength{\parindent}{0pt}
\begin{document}
\noindent\begin{minipage}[t]{%(wa)smm}\vspace{0pt}{\sffamily\bfseries\fontsize{8}{9}\selectfont a}\par\vspace{0.4mm}\includegraphics[width=%(wa)smm]{%(a)s}\end{minipage}\hfill
\begin{minipage}[t]{%(wb)smm}\vspace{0pt}{\sffamily\bfseries\fontsize{8}{9}\selectfont b}\par\vspace{0.4mm}\includegraphics[width=%(wb)smm]{%(b)s}\end{minipage}
\end{document}
"""


def fig1(out):
    """a: the designer's conceptual panel (vector PDF, round 4, composed with tectonic so it stays vector; else the round-3 PNG;
    else the code-drawn stand-in); b: Kaplan–Meier emergence by L."""
    import shutil
    import subprocess
    design_pdf = os.path.join(HERE, "refs", "designer_fig1_round4.pdf")
    design_png = os.path.join(HERE, "refs", "designer_fig1_round3.png")
    if os.path.exists(design_pdf) and shutil.which("tectonic"):
        fig = plt.figure(figsize=(62 * fs.MM, 62 * fs.MM))
        ax = fig.add_axes([0.16, 0.12, 0.82, 0.74])
        km_panel(ax, ncol=3)
        figcheck.print_report(figcheck.check(fig), "fig1_km")
        fs.save(fig, os.path.join(out, "fig1_km"), formats=("pdf",))
        wa = 112.0                                   # designer panel width; its page is 510 x 340.08 pt (3:2)
        ha = wa * 340.08 / 510.0
        tex = COMPOSE_TEX % {"h": f"{ha + 6.5:.1f}", "wa": f"{wa:.0f}", "wb": "62", "a": design_pdf, "b": os.path.join(out, "fig1_km.pdf")}
        tex_path = os.path.join(out, "fig1_compose.tex")
        open(tex_path, "w").write(tex)
        subprocess.run(["tectonic", "--outdir", out, tex_path], check=True, capture_output=True)
        os.replace(os.path.join(out, "fig1_compose.pdf"), os.path.join(out, "fig1.pdf"))
        for stale in ("fig1.svg", "fig1_compose.tex"):
            if os.path.exists(os.path.join(out, stale)):
                os.remove(os.path.join(out, stale))
        if shutil.which("pdftoppm"):
            subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", os.path.join(out, "fig1.pdf"), os.path.join(out, "fig1")], check=False)
        print("  fig1 composed from the designer's vector PDF (a) and the KM panel (b)")
        return
    fig = plt.figure(figsize=(fs.DOUBLE, 72 * fs.MM))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[112, 58], wspace=0.22, left=0.012, right=0.99, top=0.9, bottom=0.14)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    if os.path.exists(design_png):
        axa.imshow(plt.imread(design_png))
        axa.set_axis_off()
    else:
        cp.fig1a(axa)
    axa.set_anchor("NW")
    label(axa, "a", dx=0.01)
    try:
        km_panel(axb, ncol=3)
        label(axb, "b")
    except Exception as e:  # noqa: BLE001
        placeholder(axb, f"b  (data missing: {e})")
    save(fig, os.path.join(out, "fig1"))


# ----------------------------------------------------------------------------------------------------------------- fig 2
VIDEO_STEM = os.path.join(EXP, "runs", "video", "video_L16_st128_k4_s2002")
FRAME_STEPS = [5, 320, 600, 1200, 10000, 41000, 100000]
FRAME_CAPS = ["random programs", "tar: zeros spread", "first replicators", "the wave", "the open phase", "closure spreads", "closed"]
WINDOW_STEP, WIN_W, WIN_H = 600, 24, 16


def _frame_rgb(soup, cc, k=4, min_count=20):
    """Lattice map, one pixel per tape: the k most common classes (>= min_count tapes, not zero-rich) in the mean colour of
    their bytes (the same rule as the supplementary video); every other tape grey, lighter the more of its bytes are zero,
    so the tar flood and the zero pockets are visible as bleaching."""
    uniq, inv, counts = np.unique(soup, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    zf = (uniq == 0).mean(axis=1)
    base = np.array([168.0, 173.0, 181.0])
    col = (base[None, :] + (255.0 - base)[None, :] * zf[:, None]).astype(np.uint8)
    order = np.argsort(-counts, kind="stable")
    top = []
    for i in order[:k]:
        if counts[i] >= min_count and zf[i] < 0.5:
            col[i] = cc.colour(uniq[i].tobytes())
            top.append(i)
    return col[inv].reshape(125, 160, 3), [(uniq[i], int(counts[i])) for i in top]


def _best_window(soup, top_class):
    """Window (row, col) of WIN_W x WIN_H tapes whose share of the top class is closest to 0.45 (a patch edge)."""
    member = (soup == top_class[None, :]).all(axis=1).reshape(125, 160).astype(float)
    best, score = (0, 0), 9.0
    for r in range(0, 125 - WIN_H + 1, 2):
        for c in range(0, 160 - WIN_W + 1, 2):
            s = abs(member[r:r + WIN_H, c:c + WIN_W].mean() - 0.45)
            if s < score:
                best, score = (r, c), s
    return best


def fig2(out):
    import soup_stills as ss
    rows = [json.loads(l) for l in open(VIDEO_STEM + ".jsonl") if '"kind": "sample"' in l]
    rows.sort(key=lambda r: r["step"])
    S = pd.DataFrame([{"step": r["step"], "zero": r["zero_frac"], "q": r.get("q_share", np.nan), "unique": r.get("unique", np.nan)} for r in rows])
    S = S[S["step"] > 0]
    snaps = ss.snapshot_files(VIDEO_STEM)
    files = {st: min(snaps, key=lambda t: abs((t[1] if t[1] is not None else -1) - st)) for st in FRAME_STEPS}
    cc = ss.ClassColours(k=8)
    frames = {}
    for st in FRAME_STEPS:
        name, sp, f = files[st]
        soup = ss.load(f, 16)
        frames[st] = (sp, soup, *_frame_rgb(soup, cc))

    fig = plt.figure(figsize=(fs.DOUBLE, 84 * fs.MM))
    gs_top = GridSpec(1, 7, figure=fig, wspace=0.06, left=0.02, right=0.99, top=0.9, bottom=0.6)
    gs_bot = GridSpec(1, 2, figure=fig, width_ratios=[2.35, 1.0], wspace=0.12, left=0.075, right=0.99, top=0.44, bottom=0.11)
    for i, st in enumerate(FRAME_STEPS):
        ax = fig.add_subplot(gs_top[0, i])
        sp, soup, rgb, top = frames[st]
        ax.imshow(np.repeat(np.repeat(rgb, 4, axis=0), 4, axis=1), interpolation="nearest")
        ax.set_axis_off()
        ax.set_title(FRAME_CAPS[i], fontsize=5.5, color=INK, pad=2)
        ax.text(0.5, -0.06, f"step {sp:,}", transform=ax.transAxes, ha="center", va="top", fontsize=5, color=GREY, gid="allow-outside")
        ax.text(0.03, 0.97, str(i + 1), transform=ax.transAxes, ha="left", va="top", fontsize=5, color=INK, gid="allow-outside",
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="none"))
        if st == WINDOW_STEP:
            r0, c0 = _best_window(soup, top[0][0])
            ax.add_patch(Rectangle((c0 * 4 - 0.5, r0 * 4 - 0.5), WIN_W * 4, WIN_H * 4, fill=False, ec=RED, lw=0.7))
        if i == 0:
            label(ax, "a", dx=0.012)
    # b: the world's time course with the frames marked
    axb = fig.add_subplot(gs_bot[0, 0])
    axb.plot(S["step"], S["zero"], color=GREY, lw=0.9, label="zero bytes (fraction of all bytes)")
    axb.plot(S["step"], S["q"], color=TEAL, lw=0.9, label="occupancy of the dominant tape")
    axb.plot(S["step"], S["unique"] / 20000.0, color=INK, lw=0.8, ls="--", label="distinct tapes (fraction of 20,000)")
    for i, st in enumerate(FRAME_STEPS):
        sp = frames[st][0]
        axb.axvline(sp, color=RULE, lw=0.5, ls=":", zorder=0)
        axb.text(sp, 1.03, str(i + 1), ha="center", va="bottom", fontsize=5, color=INK, gid="allow-outside")
    axb.set_xscale("log")
    axb.set_xlim(4, 1.3e5)
    axb.set_ylim(0, 1.0)
    axb.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    fs.tidy(axb, "step", "fraction")
    axb.set_gid("allow-clip")
    axb.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=5, frameon=False, columnspacing=1.2, handlelength=1.6)
    label(axb, "b")
    # c: a byte-resolution window of frame 3
    axc = fig.add_subplot(gs_bot[0, 1])
    sp, soup, rgb, top = frames[WINDOW_STEP]
    r0, c0 = _best_window(soup, top[0][0])
    idx = np.array([[(r0 + r) * 160 + (c0 + c) for c in range(WIN_W)] for r in range(WIN_H)]).ravel()
    # one 4 x 4 block per tape with a one-pixel white gutter, so the tapes read as tiles
    win = np.full((WIN_H * 5 + 1, WIN_W * 5 + 1, 3), 255, np.uint8)
    for r in range(WIN_H):
        for c in range(WIN_W):
            win[1 + r * 5:5 + r * 5, 1 + c * 5:5 + c * 5] = ss.LUT[soup[idx[r * WIN_W + c]]].reshape(4, 4, 3)
    axc.imshow(np.repeat(np.repeat(win, 5, axis=0), 5, axis=1), interpolation="nearest")
    for sp_ in axc.spines.values():
        sp_.set_edgecolor(RED)
        sp_.set_linewidth(0.7)
    axc.set_xticks([])
    axc.set_yticks([])
    word = " ".join(f"{b:02x}" for b in top[0][0][:2])
    axc.set_title(f"{WIN_W} × {WIN_H} tapes of frame 3 at byte resolution", fontsize=5.5, color=INK, pad=2)
    axc.text(0.5, -0.05, f"one block per tape, one pixel per byte; zero bytes white;\nthe replicator is the repeated word {word}", transform=axc.transAxes,
             ha="center", va="top", fontsize=5, color=GREY, gid="allow-outside")
    label(axc, "c", dx=0.03)
    save(fig, os.path.join(out, "fig2"))
    # the numbers the legend quotes
    z = S.set_index("step")["zero"]
    print("  fig2 numbers: zero peak %.3f at step %d; tq_10 %s; zero at 100k %.3f; q at 100k %.3f; distinct at 100k %d" % (
        z.max(), z.idxmax(), json.load(open(VIDEO_STEM + ".summary.json")).get("tq_10"), z.iloc[-1], S["q"].iloc[-1], S["unique"].iloc[-1]))
    print("  fig2 window: rows %d–%d, cols %d–%d of frame at step %d; top classes %s" % (r0, r0 + WIN_H, c0, c0 + WIN_W, sp,
          [(" ".join(f"{b:02x}" for b in t[:4]), n) for t, n in top]))


# ----------------------------------------------------------------------------------------------------------------- fig 3
def slope_chart(ax, groups, first, final, loop_first, loop_final, ylab, ylim, yticks, seed=0, header_y=None):
    """Paired first -> final per world, grouped (one group per tape length): grey lines join the same world; filled
    vermilion = loop instruction present, open grey = absent (the convention of every first/final chart in the paper)."""
    rng = np.random.default_rng(seed)
    for gi, (name, idx) in enumerate(groups):
        x0 = gi * 3.0
        j = rng.uniform(-0.14, 0.14, len(idx))
        for k, w in enumerate(idx):
            ax.plot([x0 + j[k], x0 + 1 + j[k]], [first[w], final[w]], color=RULE, lw=0.5, alpha=0.9, zorder=1)
        for dx, vals, loop in ((0, first, loop_first), (1, final, loop_final)):
            y = np.array([vals[w] for w in idx])
            lp = np.array([loop[w] for w in idx], bool)
            xs = x0 + dx + j
            ax.scatter(xs[~lp], y[~lp], s=LOOP_MS, facecolors="white", edgecolors=GREY, lw=0.6, zorder=3)
            ax.scatter(xs[lp], y[lp], s=LOOP_MS, color=RED, lw=0, zorder=3)
        ax.text(x0 + 0.5, header_y if header_y is not None else ylim[1], name, ha="center", va="bottom", fontsize=6, color=INK)
    ax.set_xticks([gi * 3.0 + dx for gi in range(len(groups)) for dx in (0, 1)], ["first", "final"] * len(groups), fontsize=5)
    ax.set_xlim(-0.7, (len(groups) - 1) * 3.0 + 1.7)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    fs.tidy(ax, None, ylab)


def fig3(out):
    g = pd.read_csv(os.path.join(R, "stageG", "stageG", "stage_g_runs.csv"))
    c4 = pd.read_csv(os.path.join(R, "stageG", "c4", "functional.csv"))
    Ls = [16, 20, 50, 64]
    fig = plt.figure(figsize=(fs.DOUBLE, 160 * fs.MM))
    gs1 = GridSpec(1, 3, figure=fig, wspace=0.42, left=0.075, right=0.99, top=0.95, bottom=0.74)
    gs2 = GridSpec(1, 2, figure=fig, width_ratios=[1.45, 1.0], wspace=0.3, left=0.075, right=0.99, top=0.63, bottom=0.42)
    gs3 = GridSpec(1, 1, figure=fig, left=0.03, right=0.99, top=0.335, bottom=0.005)   # 52.8 mm: the 3.35:1 drawing fills the width
    axa, axb, axc = fig.add_subplot(gs1[0, 0]), fig.add_subplot(gs1[0, 1]), fig.add_subplot(gs1[0, 2])
    axd, axe = fig.add_subplot(gs2[0, 0]), fig.add_subplot(gs2[0, 1])
    axf = fig.add_subplot(gs3[0, 0])
    # a, b: partner test, first replicator -> final dominant per world
    groups, first_c, final_c, first_d, final_d, lf, ll = [], {}, {}, {}, {}, {}, {}
    for L in Ls:
        d = g[g["L"] == L].reset_index(drop=True)
        idx = [f"{L}:{i}" for i in range(len(d))]
        hz = int(d["horizon"].iloc[0])
        groups.append((f"L = {L}", idx))
        for i, key in enumerate(idx):
            first_c[key], final_c[key] = d.loc[i, "first_copied"], d.loc[i, "final_copied"]
            first_d[key], final_d[key] = d.loc[i, "first_damaged"], d.loc[i, "final_damaged"]
            lf[key], ll[key] = loop_flags(d, "first")[i], loop_flags(d, "final")[i]
    slope_chart(axa, groups, first_c, final_c, lf, ll, "random partners that become a copy", (-0.04, 1.12), [0, 0.25, 0.5, 0.75, 1.0], seed=0, header_y=1.04)
    slope_chart(axb, groups, first_d, final_d, lf, ll, "encounters that damage the organism", (-0.04, 1.12), [0, 0.25, 0.5, 0.75, 1.0], seed=1, header_y=1.04)
    h = [plt.Line2D([], [], marker="o", ls="none", color=RED, ms=3, label="loop instruction (jump, return or LDIR)"),
         plt.Line2D([], [], marker="o", ls="none", mfc="white", mec=GREY, ms=3, label="no loop instruction")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.5, frameon=False, columnspacing=2.0)
    label(axa, "a")
    label(axb, "b")
    # c: information inflow H(o | x) in bits (analysis B)
    try:
        P = pd.read_csv(os.path.join(R, "biology", "individuality", "per_replicator.csv"))
        P = P[P["machine"] == "z80"].copy()
        P["loop"] = P["has_loop"].map(lambda v: str(v) == "True")
        groups_i, fH, lH, lfH, llH = [], {}, {}, {}, {}
        for L in Ls:
            d = P[P["group"].astype(int) == L]
            first = d[d["which"] == "first"].set_index("world")
            final = d[d["which"] == "final"].set_index("world")
            worlds = first.index.intersection(final.index)
            idx = [f"{L}:{w}" for w in worlds]
            groups_i.append((f"L = {L}", idx))
            for w, key in zip(worlds, idx):
                fH[key], lH[key] = first.loc[w, "H_bits"], final.loc[w, "H_bits"]
                lfH[key], llH[key] = first.loc[w, "loop"], final.loc[w, "loop"]
        slope_chart(axc, groups_i, fH, lH, lfH, llH, "information from the partner (bits)", (-0.3, 9.0), [0, 2, 4, 6, 8], seed=2, header_y=8.35)
        axc.axhline(8.0, color=RULE, lw=0.5, ls=":", zorder=0)
        axc.text(axc.get_xlim()[0] + 0.15, 7.6, "8-bit ceiling", ha="left", va="top", fontsize=5, color=GREY)
        label(axc, "c")
    except Exception as e:  # noqa: BLE001
        placeholder(axc, f"c (data missing: {e})")
    # d: heritable fraction of random cells vs step per L (median + IQR)
    for L in Ls:
        c = c4[c4["tape_len"] == L].groupby("step")["frac_heritable"]
        med, lo, hi = c.median(), c.quantile(0.25), c.quantile(0.75)
        axd.plot(med.index, med.values, color=L_COL[L], lw=0.9, label=f"L = {L}")
        axd.fill_between(med.index, lo.values, hi.values, color=L_COL[L], alpha=0.15, lw=0)
    axd.set_xscale("log")
    axd.set_xlim(40, 1.2e6)
    axd.set_ylim(-0.02, 1.02)
    fs.tidy(axd, "step", "heritable fraction of random cells")
    axd.set_gid("allow-clip")
    axd.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    label(axd, "d")
    # e: convergence and closure counts per L as a count table (filled proportion behind each count)
    rows = []
    for L in Ls:
        d = g[g["L"] == L]
        modal = d["final_tape"].mode().iloc[0]
        hz = int(d["horizon"].iloc[0])
        rows.append({"L": L, "hz": hz, "loop": int(loop_flags(d, "final").sum()), "closed": int((d["final_copied"] >= 0.95).sum()),
                     "identical": int((d["final_tape"] == modal).sum()), "n": len(d)})
    T = pd.DataFrame(rows)
    cols = [("loop", "loop\ninstruction"), ("closed", "copies ≥ 95%\nof partners"), ("identical", "byte-identical\nto the modal tape")]
    axe.set_xlim(0, 3.9)
    axe.set_ylim(-0.1, len(Ls) + 1.1)
    axe.set_axis_off()
    for j, (_, head) in enumerate(cols):
        axe.text(1.3 + j * 0.9, len(Ls) + 0.05, head, ha="center", va="bottom", fontsize=5, color=INK, linespacing=1.1)
    axe.text(0.42, len(Ls) + 0.05, "worlds of 20\nby horizon", ha="center", va="bottom", fontsize=5, color=GREY, linespacing=1.1)
    for i, r in T.iterrows():
        y = len(Ls) - 1 - i
        axe.text(0.42, y + 0.5, f"L = {r['L']}\n{'1M' if r['hz'] >= 10**6 else str(r['hz'] // 1000) + 'k'} steps", ha="center", va="center", fontsize=5, color=INK, linespacing=1.1)
        for j, (key, _) in enumerate(cols):
            x = 0.85 + j * 0.9
            frac = r[key] / r["n"]
            axe.add_patch(Rectangle((x, y + 0.08), 0.9, 0.84, facecolor=cp.TEAL_FILL, edgecolor="none", alpha=0.25 + 0.75 * frac))
            axe.text(x + 0.45, y + 0.5, f"{r[key]}", ha="center", va="center", fontsize=6, color=INK)
    label(axe, "e", dx=0.03)
    # f: control flow of the first replicator and four closed successors (conceptual)
    cp.fig2d(axf)
    axf.set_anchor("NW")
    label(axf, "f", dx=0.01)
    save(fig, os.path.join(out, "fig3"))
    print("  fig3 table:", T.to_dict("records"))


# ----------------------------------------------------------------------------------------------------------------- fig 4
def fig4(out):
    fig = plt.figure(figsize=(fs.DOUBLE, 100 * fs.MM))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1.0], hspace=0.55, wspace=0.45, left=0.2, right=0.98, top=0.96, bottom=0.1)
    axa, axb, axc, axd = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    # a: atlas forest (Stage C, 128 steps, 1/16). One colour; filled = all ten worlds alive; open = fewer; arrow = no emergence.
    try:
        C = pd.read_csv(os.path.join(R, "stageC", "stage_c", "c3_ablations_st128_k4.csv"))
        C = C[C["label"] != "none"].copy()
        C["ratio"] = C["km_ratio_vs_none"].replace([np.inf, -np.inf], np.nan)
        C = C.sort_values("ratio", na_position="last")
        y = np.arange(len(C))
        for yi, (_, r) in zip(y, C.iterrows()):
            alive, n = int(r["t_rep_n"]), int(r["n"])
            if np.isfinite(r["ratio"]):
                if alive == n:
                    axa.plot(r["ratio"], yi, "o", color=INK, ms=3.2)
                else:
                    axa.plot(r["ratio"], yi, "o", mfc="white", mec=INK, mew=0.7, ms=3.2)
            else:
                axa.annotate("", xy=(3000, yi), xytext=(600, yi), arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
            axa.text(1.03, yi, f"{alive}/{n}", transform=axa.get_yaxis_transform(), fontsize=5, va="center", color=INK if alive == n else GREY, gid="allow-outside")
        axa.axvline(1, color=INK, lw=0.5, ls=":")
        axa.set_xscale("log")
        axa.set_xlim(0.3, 3000)
        axa.set_yticks(y, C["label"].tolist())
        axa.set_ylim(-0.7, len(C) + 0.3)
        fs.tidy(axa, "emergence delay vs unablated (ratio of KM medians)")
        axa.text(1.03, len(C) + 0.05, "alive", transform=axa.get_yaxis_transform(), fontsize=5, va="center", color=GREY, gid="allow-outside")
        h = [plt.Line2D([], [], marker="o", ls="none", color=INK, ms=3, label="all 10 worlds alive"),
             plt.Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, ms=3, label="fewer alive"),
             plt.Line2D([], [], marker=r"$\rightarrow$", ls="none", color=GREY, ms=5, label="no median: fewer than half the worlds alive")]
        axa.legend(handles=h, fontsize=5, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.3)
        label(axa, "a")
    except Exception as e:  # noqa: BLE001
        placeholder(axa, f"a (data missing: {e})")
    # b: size axis: fraction alive by L (none @nominal) with Wilson CI, plus the pusher's isolated heritability
    try:
        S = pd.read_csv(os.path.join(R, "stageE", "stage_e", "size_arms.csv"))
        S = S[(S["ablation"] == "none") & (S["arm"] == "nominal")].sort_values("tape_len")
        axb.errorbar(S["tape_len"], S["t_rep_frac"], yerr=[S["t_rep_frac"] - S["t_rep_lo"], S["t_rep_hi"] - S["t_rep_frac"]], fmt="o", color=INK, ms=3, lw=0.6, capsize=1.5, label="worlds alive by 300k steps")
        U = pd.read_csv(os.path.join(R, "stageE", "stage_e", "unit_fitness_vs_L.csv"))
        U = U[(U["unit"].str.startswith("pusher")) & (U["steps"] == 128)].sort_values("L")
        axb.plot(U["L"], U["gen2"], color=TEAL, lw=0.9, ls="--", label="the first replicator's heritability in isolation")
        axb.axhline(0.3, color=RULE, lw=0.5, ls=":")
        axb.set_xscale("log")
        axb.set_xticks([3, 5, 8, 12, 16, 25, 36, 50, 64, 100], ["3", "5", "8", "12", "16", "25", "36", "50", "64", "100"])
        axb.minorticks_off()
        axb.set_ylim(-0.03, 1.03)
        fs.tidy(axb, "tape length L (bytes)", "fraction")
        axb.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
        label(axb, "b")
    except Exception as e:  # noqa: BLE001
        placeholder(axb, f"b (data missing: {e})")
    # c: dead-zone switch (Stage F4 + Stage E reference)
    try:
        F = pd.read_csv(os.path.join(R, "stageF", "stage_f", "rings.csv"))
        F = F[(F["ablation"] == "none") & (F["L"].isin([8, 10, 12]))].sort_values(["L", "P"])
        for L, mk, col in ((8, "s", INK), (10, "^", TEAL), (12, "o", RED)):
            d = F[F["L"] == L]
            frac = d["t_rep_n"] / d["n"]
            axc.errorbar(d["P"], frac, yerr=[frac - d["t_rep_lo"], d["t_rep_hi"] - frac], fmt=mk, color=col, ms=3.2, lw=0.6, capsize=1.5, label=f"L = {L}")
            for _, r in d.iterrows():
                if r["P"] == 2 * r["L"]:
                    axc.annotate("native ring", (r["P"], r["t_rep_n"] / r["n"]), textcoords="offset points", xytext=(6, 0), fontsize=5, ha="left", va="center", color=GREY)
        axc.set_ylim(-0.03, 1.03)
        fs.tidy(axc, "pair memory ring P (bytes)", "worlds alive by 300k steps")
        axc.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
        label(axc, "c")
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
        for st, mk, off, col in ((128, "o", -0.15, INK), (512, "s", 0.15, TEAL)):
            d = D[D["steps"] == st].set_index("label").reindex(arms)
            axd.errorbar(d["t_rep_frac"], y + off, xerr=[d["t_rep_frac"] - d["t_rep_lo"], d["t_rep_hi"] - d["t_rep_frac"]], fmt=mk, color=col, ms=3, lw=0.6, capsize=1.5, label=f"{st} steps per encounter")
        axd.set_yticks(y, arms)
        axd.set_xlim(-0.03, 1.03)
        fs.tidy(axd, "worlds alive at L = 9 (of 20)")
        axd.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
        label(axd, "d")
    except Exception as e:  # noqa: BLE001
        placeholder(axd, f"d (data missing: {e})")
    save(fig, os.path.join(out, "fig4"))


# ----------------------------------------------------------------------------------------------------------------- fig 5
def _allp_share(s):
    """Share of the all-`P` class (hex 50) among the ten largest classes of a sample; 0 when it is not among them."""
    for t in s["top"]:
        tp = t["tape"]
        if tp and set(tp[i:i + 2] for i in range(0, len(tp), 2)) == {"50"}:
            return t["share"]
    return 0.0


def fig5(out):
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

    fig = plt.figure(figsize=(fs.DOUBLE, 120 * fs.MM))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.4, 1.0, 1.0], height_ratios=[1.0, 1.0], hspace=0.5, wspace=0.45, left=0.07, right=0.99, top=0.96, bottom=0.02)
    gs2 = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.0], hspace=0.5, left=0.065, right=0.99, top=0.96, bottom=0.02)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    axd = fig.add_subplot(gs2[1, 0])
    for v in variants:
        for run in runs[runs["variant"] == v]["run"]:
            p = os.path.join(bdir, run, "samples.jsonl")
            if not os.path.exists(p):
                continue
            S = pd.DataFrame([{"epoch": s["epoch"], "h": s["frac_heritable"]} for s in map(json.loads, open(p))])
            axa.plot(S["epoch"], S["h"].rolling(4, min_periods=1).mean(), color=fs.BFF_VARIANT[v], lw=0.6, alpha=0.75)
        axa.plot([], [], color=fs.BFF_VARIANT[v], lw=0.9, alpha=0.9, label=f"{fs.BFF_VARIANT_LABEL[v]} (n = {int((runs['variant'] == v).sum())})")
    epoch_axis(axa)
    fs.tidy(axa, "epoch", "heritable fraction of random tapes")
    label(axa, "a")
    handles, labels = axa.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.49), ncol=5, frameon=False, fontsize=5, handlelength=1.8, columnspacing=1.6)
    rng = np.random.default_rng(0)
    ROW, SUB, JIT = 2.6, 0.45, 0.22
    yticks, ylabels = [], []
    for i, v in enumerate(variants):
        d = runs[(runs["variant"] == v) & runs["t_top"].notna()]
        y0 = -ROW * i
        for which, dy, mk in (("first", SUB, "o"), ("final", -SUB, "s")):
            y = y0 + dy + rng.uniform(-JIT, JIT, len(d))
            loop = d[f"{which}_loop"].astype(bool).values
            vals = d[f"{which}_entered"].values
            axb.scatter(vals[~loop], y[~loop], s=6, marker=mk, facecolors="none", edgecolors=fs.BFF_VARIANT[v], lw=0.6)
            axb.scatter(vals[loop], y[loop], s=6, marker=mk, color=fs.BFF_VARIANT[v], lw=0)
            yticks.append(y0 + dy)
            ylabels.append(which)
        axb.text(0.0, y0 + SUB + JIT + 0.3, fs.BFF_VARIANT_LABEL[v], ha="left", va="bottom", fontsize=5, fontweight="bold", color="black")
    axb.set_yticks(yticks, ylabels, fontsize=5)
    axb.set_ylim(-ROW * (len(variants) - 1) - SUB - JIT - 0.35, SUB + JIT + 0.3 + 0.75)
    axb.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    axb.set_xlim(-0.04, 1.04)
    axb.tick_params(axis="y", length=2)
    fs.tidy(axb, "encounters whose pointer\nenters the partner")
    label(axb, "b")
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
    fs.tidy(axc, "epoch", "share of the all-P tape in the soup")
    label(axc, "c")
    cp.fig4d(axd)
    axd.set_anchor("NW")
    label(axd, "d")
    save(fig, os.path.join(out, "fig5"))


# ----------------------------------------------------------------------------------------------------------------- fig 6
def fig6(out):
    """Theorem 2 as a diagram, a single panel (no letter): 120 mm wide."""
    fig = plt.figure(figsize=(120 * fs.MM, 52 * fs.MM))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    cp.fig5a(ax)
    ax.set_anchor("NW")
    save(fig, os.path.join(out, "fig6"))


# ------------------------------------------------------------------------------------------------------ Extended Data 12
def ed12(out):
    A = os.path.join(R, "biology", "assembly")
    fig = plt.figure(figsize=(fs.DOUBLE, 62 * fs.MM))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.25, 1.0, 1.0], height_ratios=[1.0, 0.8], hspace=0.12, wspace=0.42, left=0.075, right=0.99, top=0.9, bottom=0.14)
    axa1, axa2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    axb, axc = fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[:, 2])
    world = "none@closure_L16_st128_k4_s2001"
    S = pd.read_csv(os.path.join(A, "per_sample.csv"))
    S = S[S["world"] == world].sort_values("step")
    S = S[S["step"] > 0]
    W = pd.read_csv(os.path.join(A, "per_world.csv"))
    w = W[W["world"] == world].iloc[0]
    axa1.plot(S["step"], S["A_top10"], color=TEAL, lw=0.8)
    axa1.axhline(w["threshold"], color=RULE, lw=0.5, ls=":")
    axa1.text(2.6e4, w["threshold"] * 1.35, "10 × baseline", fontsize=5, color=GREY, va="bottom", ha="right")
    axa1.set_yscale("log")
    axa1.set_ylim(1, 1e4)
    axa1.set_yticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1,000", "10,000"])
    fs.tidy(axa1, None, "assembly measure\nover the ten most common classes")
    axa1.set_xticklabels([])
    axa2.plot(S["step"], S["hoe"], color=INK, lw=0.8)
    axa2.set_ylim(0, 4.2)
    axa2.set_yticks([0, 1, 2, 3, 4])
    fs.tidy(axa2, "step", "high-order entropy\n(bits per byte)")
    for ax in (axa1, axa2):
        ax.set_xscale("log")
        ax.set_xlim(1, 3.5e5)
        ax.axvline(w["t_rep"], color=RED, lw=0.6, ls="--")
        ax.set_gid("allow-clip")
    axa1.text(w["t_rep"] * 1.15, 5e3, "first heritable\nreplicator", fontsize=5, color=RED, va="top")
    axa1.plot([w["step_first_cross"]], [w["A_first_cross"]], "o", mfc="white", mec=TEAL, ms=4, mew=0.8)
    axa1.text(w["step_first_cross"] / 1.25, w["A_first_cross"] * 1.9, "first tenfold rise", fontsize=5, color=TEAL, va="bottom", ha="right")
    label(axa1, "a")
    # b: first tenfold rise vs the first heritable replicator, all Stage G worlds
    mk = {16: "o", 20: "s", 50: "^", 64: "D"}
    top_edge = 2.0e4
    for L in (16, 20, 50, 64):
        d = W[W["L"] == L]
        y = d["step_first_cross"].astype(float).values
        ok = np.isfinite(y)
        axb.scatter(d["t_rep"].values[ok], y[ok], s=9, marker=mk[L], facecolors="white", edgecolors=INK, lw=0.6, label=f"L = {L} (n = {len(d)})")
        if (~ok).any():
            axb.scatter(d["t_rep"].values[~ok], np.full((~ok).sum(), top_edge), s=12, marker="x", color=RED, lw=0.7)
    axb.plot([60, 1e4], [60, 1e4], color=RULE, lw=0.5, ls=":")
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlim(60, 1e4)
    axb.set_ylim(1, 3e4)
    axb.text(70, 1.4, "rises before\nthe replicator", fontsize=5, color=GREY, va="bottom")
    axb.text(70, 1.6e4, "no rise (x)", fontsize=5, color=RED, va="center")
    fs.tidy(axb, "first heritable replicator (step)", "first tenfold rise of the measure (step)")
    axb.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, columnspacing=0.8, handletextpad=0.2)
    label(axb, "b")
    # c: AUC of the two detectors against the heredity event by tape length (Stage E, snapshot steps)
    U = pd.read_csv(os.path.join(A, "auc.csv"))
    recs = []
    for _, r in U.iterrows():
        m = re.match(r"E \| L=(\d+) \| all arms \| snapshot steps", str(r["stratum"]))
        if m and np.isfinite(r["AUC"]):
            recs.append({"L": int(m.group(1)), "det": r["detector"], "auc": r["AUC"]})
    T = pd.DataFrame(recs)
    for det, col, mk_, lab in (("A_top10", TEAL, "o", "assembly measure"), ("hoe", INK, "s", "high-order entropy")):
        d = T[T["det"] == det].sort_values("L")
        axc.plot(d["L"], d["auc"], ls="-", lw=0.6, color=col, marker=mk_, ms=3, label=lab, mfc="white" if det == "hoe" else col)
    axc.axhline(0.5, color=RULE, lw=0.5, ls=":")
    axc.set_xscale("log")
    axc.set_xticks([4, 8, 16, 32, 64, 100], ["4", "8", "16", "32", "64", "100"])
    axc.minorticks_off()
    axc.set_ylim(-0.03, 1.03)
    fs.tidy(axc, "tape length L (bytes)", "AUC against the heredity event")
    axc.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    label(axc, "c")
    save(fig, os.path.join(out, "ed12"))
    print("  ed12 numbers:", {k: w[k] for k in ("t_rep", "step_first_cross", "A_first_cross", "threshold")}, "AUC rows", len(T))


# ------------------------------------------------------------------------------------------------------ Extended Data 13
def ed13(out):
    leth = pd.read_csv(os.path.join(R, "stageI", "c4", "functional.csv"))
    ben = pd.read_csv(os.path.join(R, "stageG", "c4", "functional.csv"))
    ben = ben[(ben["tape_len"] == 16) & (ben["label"].str.startswith("none@closure"))]
    gl = pd.read_csv(os.path.join(R, "stageI", "stageI", "stage_g_runs.csv"))
    gb = pd.read_csv(os.path.join(R, "stageG", "stageG", "stage_g_runs.csv"))
    gb = gb[gb["L"] == 16]
    fig = plt.figure(figsize=(fs.DOUBLE, 55 * fs.MM))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 0.9], wspace=0.42, left=0.07, right=0.99, top=0.86, bottom=0.17)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    for ax, col, ylab in ((axa, "frac_heritable", "heritable fraction of random cells"), (axb, "zero_frac", "zero bytes (fraction of all bytes)")):
        for seed, d in ben.groupby("seed"):
            d = d.sort_values("step")
            ax.plot(d["step"], d[col], color=RULE, lw=0.5, alpha=0.9)
        for seed, d in leth.groupby("seed"):
            d = d.sort_values("step")
            ax.plot(d["step"], d[col], color=RED, lw=0.6, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlim(40, 3.5e5)
        ax.set_ylim(-0.02, 1.02 if col == "frac_heritable" else 0.5)
        fs.tidy(ax, "step", ylab)
        ax.set_gid("allow-clip")
    h = [plt.Line2D([], [], color=RED, lw=0.9, label="zero byte halts the pair (lethal tar, 10 worlds)"),
         plt.Line2D([], [], color=RULE, lw=0.9, label="zero byte is a no-op (benign tar, 20 worlds)")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.42, 1.0), ncol=2, fontsize=5.5, frameon=False, columnspacing=2.0)
    label(axa, "a")
    label(axb, "b")
    # c: the first heritable replicator's step, lethal vs benign
    rng = np.random.default_rng(0)
    for y, d, col, name in ((1, gl, RED, "lethal"), (0, gb, RULE, "benign")):
        t = d["t_rep"].astype(float).values
        t = t[np.isfinite(t) & (t > 0)]
        axc.scatter(t, y + rng.uniform(-0.12, 0.12, len(t)), s=8, facecolors="white" if col == RULE else col, edgecolors=INK if col == RULE else col, lw=0.6, zorder=3)
        med = float(np.median(t))
        axc.plot([med, med], [y - 0.3, y + 0.3], color=INK, lw=0.8, zorder=4)
        axc.text(med, y + 0.36, f"median {med:,.0f}", ha="center", va="bottom", fontsize=5, color=INK)
        print(f"  ed13 {name}: n = {len(t)}, median t_rep = {med:,.0f}")
    axc.set_xscale("log")
    axc.set_xlim(60, 3.5e5)
    axc.set_ylim(-0.6, 1.9)
    axc.set_yticks([0, 1], ["benign", "lethal"])
    fs.tidy(axc, "first heritable replicator (step)")
    label(axc, "c")
    save(fig, os.path.join(out, "ed13"))


# ------------------------------------------------------------------------------------------------------ Extended Data 14
def ed14(out):
    """Mutational scan (results/mutscan/mutscan_tapes.csv): transmissible sites, capacity and robustness, first -> final per world."""
    T = pd.read_csv(os.path.join(R, "mutscan", "mutscan_tapes.csv"))
    Ls = [16, 20, 50, 64]
    fig = plt.figure(figsize=(fs.DOUBLE, 58 * fs.MM))
    gs = GridSpec(1, 3, figure=fig, wspace=0.42, left=0.075, right=0.99, top=0.84, bottom=0.14)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    specs = [("n_sites", "transmissible sites (positions)", (-1.5, 38.0), [0, 10, 20, 30], 35.2),
             ("capacity_bits", "capacity for inherited variation (bits)", (-15, 440), [0, 100, 200, 300, 400], 410),
             ("robustness_h", "single mutants that remain heritable", (-0.04, 1.12), [0, 0.25, 0.5, 0.75, 1.0], 1.04)]
    for ax, (col, ylab, ylim, yt, hy), sd in zip(axes, specs, (0, 1, 2)):
        groups, first, final, lf, ll = [], {}, {}, {}, {}
        for L in Ls:
            f = T[(T.L == L) & (T.which == "first")].set_index("seed")
            n = T[(T.L == L) & (T.which == "final")].set_index("seed")
            idx = [f"{L}:{sd_}" for sd_ in f.index.intersection(n.index)]
            groups.append((f"L = {L}", idx))
            for sd_, key in zip(f.index.intersection(n.index), idx):
                first[key], final[key] = f.loc[sd_, col], n.loc[sd_, col]
                lf[key], ll[key] = bool(f.loc[sd_, "has_loop"]), bool(n.loc[sd_, "has_loop"])
        slope_chart(ax, groups, first, final, lf, ll, ylab, ylim, yt, seed=sd, header_y=hy)
    h = [plt.Line2D([], [], marker="o", ls="none", color=RED, ms=3, label="loop instruction (jump, return or LDIR)"),
         plt.Line2D([], [], marker="o", ls="none", mfc="white", mec=GREY, ms=3, label="no loop instruction")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.5, frameon=False, columnspacing=2.0)
    for ax, letter in zip(axes, "abc"):
        label(ax, letter)
    save(fig, os.path.join(out, "ed14"))


# ------------------------------------------------------------------------------------------- the variation figure (v4 Fig. 4)
def figvar(out):
    """The cost of closure and the birth of the genotype: a executed/transmissible position maps of exemplar genomes;
    b the matched-pair invasion (jump-word share vs step); c capacity of the dominant tape over evolutionary time."""
    sys.path.insert(0, EXP)
    from algocell_exp import exectrace as X
    fig = plt.figure(figsize=(fs.DOUBLE, 118 * fs.MM))
    gs_a = GridSpec(1, 1, figure=fig, left=0.2, right=0.84, top=0.9, bottom=0.6)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.3, left=0.07, right=0.99, top=0.45, bottom=0.1)
    axa = fig.add_subplot(gs_a[0, 0])
    axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    # a: position maps
    gG = pd.read_csv(os.path.join(R, "stageG", "stageG", "stage_g_runs.csv"))
    gI = pd.read_csv(os.path.join(R, "stageI", "stageI", "stage_g_runs.csv"))
    sG = pd.read_csv(os.path.join(R, "mutscan", "mutscan_sites.csv"))
    sI = pd.read_csv(os.path.join(R, "mutscan_I", "mutscan_sites.csv"))
    ex_rows = [("the first replicator (L = 50)", gG, 50, 2001, "first", sG, False),
               ("its successor: the pusher with a jump (L = 50)", gG, 50, 2001, "final", sG, False),
               ("return closer (L = 16)", gG, 16, 2001, "final", sG, False),
               ("block copy with a skipped segment (L = 16)", gG, 16, 2002, "final", sG, False),
               ("born closed under lethal tar (L = 16)", gI, 16, 4009, "final", sI, True)]
    y = 0
    for title, g, L, seed, which, sites, zh in ex_rows:
        w = g[(g.L == L) & (g.seed == seed)].iloc[0]
        tape = np.array([int(b, 16) for b in str(w[f"{which}_tape"]).split()], np.uint8)
        rng = np.random.default_rng([20261010, L, seed])
        Rp = rng.integers(0, 256, size=(64, L), dtype=np.uint8)
        res, masks = X.execute_pairs_traced(np.concatenate([np.repeat(tape[None, :], 64, 0), Rp], 1), L, 128, zero_halts=zh)
        union = X.exec_positions(masks, 2 * L)[:, :L].any(axis=0)
        st = sites[(sites.L == L) & (sites.seed == seed) & (sites.which == which)].sort_values("pos")
        tr = st["transmissible"].astype(bool).values if len(st) == L else np.zeros(L, bool)
        scale = 64.0 / L
        for i in range(L):
            axa.add_patch(Rectangle((i * scale, y), scale, 0.8, facecolor=INK if union[i] else "#FFFFFF", edgecolor=RULE, lw=0.3))
            if tr[i]:
                axa.plot(i * scale + scale / 2, y + 0.4, "o", color=RED, ms=2.6, mec="white", mew=0.3, zorder=5)
        axa.text(-1.0, y + 0.4, title, ha="right", va="center", fontsize=5.5, color=INK)
        axa.text(65.0, y + 0.4, f"executed {int(union.sum())}/{L} · transmissible {int(tr.sum())}" + (f" ({int((tr & ~union).sum())} unexecuted)" if tr.any() else ""), ha="left", va="center", fontsize=5, color=GREY)
        y += 1.2
    axa.set_xlim(-0.5, 64.5)
    axa.set_ylim(-0.2, y)
    axa.invert_yaxis()
    axa.set_axis_off()
    h = [Rectangle((0, 0), 1, 1, facecolor=INK, edgecolor=RULE, lw=0.3, label="byte executed (fetched as instruction stream)"),
         Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=RULE, lw=0.3, label="byte never executed"),
         plt.Line2D([], [], marker="o", ls="none", color=RED, ms=3, label="transmissible site: a mutation here is inherited")]
    axa.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=5.5, frameon=False, columnspacing=1.5)
    label(axa, "a", dx=0.185)
    # b: matched-pair invasion
    try:
        d = pd.read_csv(os.path.join(R, "invasion_pair", "invasion_pair.csv"))
        for (res_, inv), col, ls, lab in ((("pusher", "closed"), RED, "-", "closed form seeded at 1% into a pusher world"),
                                          (("closed", "pusher"), GREY, "-", "pusher seeded at 1% into a closed world"),
                                          (("pusher", "none"), INK, ":", "pusher world, no seeding (closure arises by mutation)"),
                                          (("closed", "none"), RULE, ":", "closed world, no seeding")):
            g_ = d[(d.resident == res_) & (d.invader == inv)]
            first = True
            for sd, e in g_.groupby("seed"):
                e = e[e.step > 0].sort_values("step")
                axb.plot(e.step, e.jump_share, color=col, ls=ls, lw=0.8, alpha=0.85, label=lab if first else None)
                first = False
        axb.set_xscale("log")
        axb.set_xlim(8, 2.2e4)
        axb.set_ylim(-0.02, 1.0)
        fs.tidy(axb, "step", "cells carrying the jump word")
        axb.set_gid("allow-clip")
        axb.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
        label(axb, "b")
    except Exception as e:  # noqa: BLE001
        placeholder(axb, f"b (data missing: {e})")
    # c: capacity of the dominant tape over time
    try:
        C = pd.read_csv(os.path.join(R, "capacity_time", "capacity_over_time.csv"))
        C = C[C.tape != "all-zero"].dropna(subset=["capacity_bits"])
        for L, col in ((16, L_COL[16]), (20, L_COL[20]), (50, L_COL[50]), (64, L_COL[64])):
            d = C[(C.stage == "G") & (C.L == L)]
            if d.empty:
                continue
            med = d.groupby("step").capacity_bits.median()
            lo, hi = d.groupby("step").capacity_bits.quantile(0.25), d.groupby("step").capacity_bits.quantile(0.75)
            axc.plot(med.index, med.values, color=col, lw=0.9, label=f"L = {L}")
            axc.fill_between(med.index, lo.values, hi.values, color=col, alpha=0.15, lw=0)
        axc.set_xscale("log")
        axc.set_xlim(400, 1.2e6)
        fs.tidy(axc, "step", "capacity for inherited variation\nof the dominant tape (bits)")
        axc.set_gid("allow-clip")
        axc.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
        label(axc, "c")
    except Exception as e:  # noqa: BLE001
        placeholder(axc, f"c (data missing: {e})")
    save(fig, os.path.join(out, "figvar"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fs.setup()
    for name, fn in (("fig1", fig1), ("fig2", fig2), ("fig3", fig3), ("fig4", fig4), ("fig5", fig5), ("fig6", fig6), ("ed12", ed12), ("ed13", ed13), ("ed14", ed14), ("figvar", figvar)):
        if a.only and name not in a.only.split(","):
            continue
        try:
            fn(a.out)
            print("built", name)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("FAILED", name, repr(e))


if __name__ == "__main__":
    main()
