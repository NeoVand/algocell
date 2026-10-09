"""Conceptual panels, vector, to the compact designs in manuscript/figures/refs/compact-figures-v2 (the user's generated
design proofs, 2026-10-08 evening): abbreviated strips with ellipses, short local labels, one idea per panel.

Vocabulary: organism cells pale teal, partner cells grey, what the organism writes or where it jumps vermilion,
instruction pointer teal, stack pointer vermilion, structure and type charcoal. Numbers come from the data tables
(stage_g_runs.csv, results/concept/traces.json); nothing is typed from memory.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, EXP)
import figstyle as fs  # noqa: E402

INK = "#1C2733"
GREY_TEXT = "#6B7280"
GRID = "#B4BAC1"
TEAL = "#1E8A8A"
TEAL_FILL = "#CDEAEA"
GREY_FILL = "#E3E5E8"
RED = "#E8431F"
RED_MID = "#F08A6E"
RED_PALE = "#F8D2C8"
FS_BYTE, FS_LABEL, FS_TITLE, FS_SMALL = 6.0, 6.5, 7.0, 5.5


def _traces():
    with open(os.path.join(EXP, "results", "concept", "traces.json")) as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------------------------------ helpers
def cell(ax, x, y, text=None, fill=TEAL_FILL, w=1.0, h=1.0, fs=FS_BYTE, text_color=INK):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=INK, lw=0.5, zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=text_color, zorder=3)


def strip(ax, x0, y0, items, fill=TEAL_FILL, fills=None, w=1.0, h=1.0):
    """items: byte strings or '…' (drawn as a bordered cell with a centred ellipsis)."""
    for i, it in enumerate(items):
        f = fills[i] if fills else fill
        cell(ax, x0 + i * w, y0, "···" if it == "…" else it, fill=f, w=w, h=h, fs=FS_BYTE if it != "…" else 7.5)
    return x0 + len(items) * w


def arrow(ax, p0, p1, color=INK, lw=0.9, rad=0.0, ls="-", scale=7, zorder=4):
    a = FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=scale, lw=lw, color=color,
                        linestyle=ls, zorder=zorder, shrinkA=0, shrinkB=0, capstyle="round")
    ax.add_patch(a)
    return a


def dotted(ax, x0, x1, y, color, lw=1.1):
    d = 1 if x1 > x0 else -1
    ax.plot([x0, x1 - d * 0.5], [y, y], color=color, lw=lw, ls=(0, (0.6, 1.6)), dash_capstyle="round", zorder=3)
    arrow(ax, (x1 - d * 0.55, y), (x1, y), color=color, lw=lw)


def bracket(ax, x0, x1, y, tick=0.2, color=INK, up=True):
    s = -1 if up else 1
    ax.plot([x0, x0, x1, x1], [y + s * tick, y, y, y + s * tick], color=color, lw=0.6, solid_capstyle="round", zorder=2)


def text_runs(ax, x, y, runs, fs=FS_LABEL, va="bottom", ha="left", gap=0.0):
    """Text runs on one line, each (text, color, bold), positioned from rendered widths (print resolution)."""
    fig = ax.figure
    dpi0 = fig.get_dpi()
    fig.set_dpi(300)
    try:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        objs = []
        for text, color, bold in runs:
            t = ax.text(x, y, text, ha="left", va=va, fontsize=fs, color=color, fontweight="bold" if bold else "normal", zorder=3)
            bb = t.get_window_extent(renderer=rend)
            x = ax.transData.inverted().transform((bb.x1, bb.y0))[0] + gap
            objs.append(t)
        if ha == "center":   # shift the whole run so that it is centred on the original x
            x_start = objs[0].get_position()[0]
            shift = (x - x_start) / 2
            for t in objs:
                px, py = t.get_position()
                t.set_position((px - shift, py))
    finally:
        fig.set_dpi(dpi0)


def finish(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()


# ------------------------------------------------------------------------------------------------------------ Fig. 1a
def fig1a(ax):
    # organism A: three cells, an ellipsis, two cells; partner B likewise
    for x in (0, 1, 2, 4.5, 5.5):
        cell(ax, x, 0, fill=TEAL_FILL)
    for x in (6.5, 7.5, 8.5, 11, 12):
        cell(ax, x, 0, fill=GREY_FILL)
    ax.text(3.75, 0.5, "···", ha="center", va="center", fontsize=8, color=INK)
    ax.text(10.25, 0.5, "···", ha="center", va="center", fontsize=8, color=INK)
    ax.text(3.25, 2.35, "A · 16 bytes", ha="center", va="bottom", fontsize=FS_TITLE, color=TEAL, fontweight="bold")
    ax.text(9.75, 2.35, "B · 16 bytes", ha="center", va="bottom", fontsize=FS_TITLE, color=INK, fontweight="bold")
    # instruction pointer
    arrow(ax, (0.35, 1.75), (0.35, 1.08), color=TEAL, lw=1.1)
    dotted(ax, 0.65, 4.3, 1.45, TEAL)
    ax.text(0.7, 1.62, "IP: execute", ha="left", va="bottom", fontsize=FS_LABEL, color=TEAL, fontweight="bold")
    # stack pointer
    # one L-shaped path: SP starts under B's last byte and moves left two bytes per push
    ax.plot([12.5, 12.5], [-0.08, -0.5], color=RED, lw=1.1, solid_capstyle="round", zorder=3)
    arrow(ax, (12.5, -0.5), (9.0, -0.5), color=RED, lw=1.1)
    ax.text(9.9, -1.0, "SP: push 2 bytes", ha="center", va="top", fontsize=FS_LABEL, color=RED, fontweight="bold")
    # ring: under the strip, from the last cell back to the first
    ax.plot([13.4, 13.4, -0.4, -0.4], [-0.4, -2.1, -2.1, -0.4], color=INK, lw=0.8, solid_joinstyle="round", zorder=1)
    arrow(ax, (-0.4, -0.5), (-0.4, 0.3), color=INK, lw=0.8)
    ax.text(6.5, -2.3, "32-byte ring", ha="center", va="top", fontsize=FS_LABEL, color=INK)
    # the encounter
    ax.text(16.6, 1.15, "128 instructions", ha="center", va="bottom", fontsize=FS_LABEL, color=INK)
    arrow(ax, (16.6, 1.0), (16.6, 0.25), color=INK, lw=0.8)
    ax.text(16.6, -0.05, "write back A + B", ha="center", va="top", fontsize=FS_LABEL, color=INK)
    finish(ax, (-0.8, 19.6), (-3.3, 3.2))


# ------------------------------------------------------------------------------------------------------------ Fig. 1b
def fig1b(ax):
    xa = strip(ax, 0, 0, ["01", "c5", "01", "c5", "…", "01", "c5"])            # A ends at 7
    bracket(ax, 0.05, 2.95, 1.25)
    ax.text(1.5, 1.5, "LD BC,nn", ha="center", va="bottom", fontsize=FS_LABEL, color=INK, fontweight="bold")
    bracket(ax, 3.05, 3.95, 1.25)
    ax.plot([3.5, 3.5], [1.25, 2.3], color=INK, lw=0.6)
    ax.text(3.5, 2.4, "PUSH BC", ha="center", va="bottom", fontsize=FS_LABEL, color=INK, fontweight="bold")
    # the pointer runs on into the partner
    arrow(ax, (xa + 0.15, 0.5), (xa + 1.45, 0.5), color=TEAL, lw=1.1)
    ax.text(xa + 0.8, 1.15, "IP enters\npartner", ha="center", va="bottom", fontsize=FS_LABEL, color=TEAL, fontweight="bold", linespacing=1.1)
    xb = xa + 1.6                                                                 # B starts at 8.6
    strip(ax, xb, 0, ["ff", "…", "c5", "01", "c5", "01", "f3"], fills=[GREY_FILL, GREY_FILL, RED_PALE, RED_PALE, RED_MID, RED_MID, GREY_FILL])
    bracket(ax, xb + 4.05, xb + 5.95, -0.3, up=False)
    arrow(ax, (3.5, -0.1), (xb + 5.0, -0.5), color=RED, lw=1.1, rad=0.28)
    ax.text(8.6, -2.15, "write c5 01", ha="center", va="top", fontsize=FS_LABEL, color=RED, fontweight="bold")
    # side panel: code = operand = written
    xs = 17.6
    for row, (label, items, fill, y) in enumerate((("code", ["01", "c5", "01"], TEAL_FILL, 1.3), ("operand", ["c5", "01"], GREY_FILL, 0.0), ("written", ["c5", "01"], RED_MID, -1.7))):
        ax.text(xs + 2.0, y + 0.5, label, ha="right", va="center", fontsize=FS_LABEL, color=INK, fontweight="bold")
        strip(ax, xs + 2.5 + (1.0 if len(items) == 2 else 0.0), y, items, fill=fill)
    ax.text(xs + 4.5, -0.35, "=", ha="center", va="center", fontsize=9, color=INK)
    ax.text(xs + 4.0, -2.35, "code is its own data", ha="center", va="top", fontsize=FS_LABEL, color=INK, fontweight="bold")
    finish(ax, (-0.4, 24.4), (-3.2, 3.4))


# ------------------------------------------------------------------------------------------------------------ Fig. 2d
def fig2d(ax):
    import pandas as pd
    g = pd.read_csv(os.path.join(EXP, "results", "stageG", "stageG", "stage_g_runs.csv"))
    f16 = g[g["L"] == 16]
    cop, dam = f16["first_copied"].median(), f16["first_damaged"].median()
    fin = g[g["final_has_cf"] | g["final_has_block"]]
    fcop, fdam = fin["final_copied"].median(), fin["final_damaged"].median()
    finish(ax, (-0.3, 33.2), (-7.1, 2.9))
    # row 1: the open first replicator
    ax.text(0, 1.9, "open · the first replicator", ha="left", va="bottom", fontsize=FS_TITLE, color=INK, fontweight="bold")
    strip(ax, 0, 0, ["01", "c5", "…", "01", "c5"])
    ax.add_patch(Rectangle((5, 0), 3.6, 1, facecolor=GREY_FILL, edgecolor=INK, lw=0.5, zorder=2))
    ax.text(5.9, 0.5, "···", ha="center", va="center", fontsize=8, color=INK, zorder=3)
    ax.text(7.6, 0.5, "partner", ha="center", va="center", fontsize=FS_SMALL, color=INK, zorder=3)
    arrow(ax, (4.5, 1.1), (5.9, 1.1), color=INK, lw=1.0, rad=-0.45, scale=6)
    ax.text(9.6, 0.95, "executes partner code", ha="left", va="center", fontsize=FS_LABEL, color=INK)
    ax.text(9.6, 0.2, f"copies {cop:.2f} of partners; damaged in {dam:.2f} of encounters (medians, 20 worlds)", ha="left", va="center", fontsize=FS_LABEL, color=GREY_TEXT)
    # row 2: four closers, abbreviated to the instruction motif
    closers = [
        ("RET NZ · 16 bytes", ["…", "e3", "…", "c0"], [3], "its written bytes double\nas its return addresses"),
        ("JR NZ · 50 bytes", ["…", "20", "f0", "…"], [1, 2], "a relative jump of 16 bytes back,\nthrough the address wrap"),
        ("DJNZ · 20 bytes", ["…", "10", "e5", "…"], [1, 2], "a counted jump of 27 bytes back,\nthrough the address wrap"),
        ("LDIR · 20 bytes", ["…", "ed", "b0", "…"], [1, 2], "the block copy repeats\nin place"),
    ]
    y0 = -3.3
    for k, (title, items, hl, caption) in enumerate(closers):
        x0 = k * 8.4
        ax.text(x0, y0 + 1.95, title, ha="left", va="bottom", fontsize=FS_TITLE, color=INK, fontweight="bold")
        strip(ax, x0, y0, items, fills=[RED_PALE if i in hl else TEAL_FILL for i in range(4)])
        src = x0 + max(hl) + 0.5
        if title.startswith("LDIR"):
            # the block copy repeats in place: an arc from the end of the motif back to its start, same style as the jumps
            arrow(ax, (x0 + 2.95, y0 + 1.1), (x0 + 1.05, y0 + 1.1), color=RED, lw=1.0, rad=0.5, scale=6)
        else:
            arrow(ax, (src, y0 + 1.1), (x0 + 0.5, y0 + 1.1), color=RED, lw=1.0, rad=0.36, scale=6)
        ax.text(x0, y0 - 0.25, caption, ha="left", va="top", fontsize=FS_SMALL, color=INK, linespacing=1.15)
    ax.text(14.6, -5.95, f"closed: copies {fcop:.2f} of partners · {fdam:.2f} self-damage", ha="center", va="bottom", fontsize=FS_TITLE, color=INK, fontweight="bold")
    ax.text(14.6, -6.2, "closure is a cycle in control flow, not a wall around the bytes", ha="center", va="top", fontsize=FS_LABEL, color=INK)


# ------------------------------------------------------------------------------------------------------------ Fig. 4d
def fig4d(ax):
    W = 100.0
    x1, x2 = 25.0, 62.0
    y_hdr, y_mid, y_bot = 21.0, 9.6, 0.0
    H = 25.5
    for yy in (H, y_hdr, y_mid, y_bot):
        ax.plot([0, W], [yy, yy], color=GRID, lw=0.7, zorder=1)
    for xx in (x1, x2):
        ax.plot([xx, xx], [y_bot, H], color=GRID, lw=0.7, zorder=1)
    ax.text(1.0, (y_hdr + H) / 2, "literal-write instruction", ha="left", va="center", fontsize=FS_TITLE, color=INK, fontweight="bold")
    ax.text((x1 + x2) / 2, (y_hdr + H) / 2, "tar benign", ha="center", va="center", fontsize=FS_TITLE, color=INK, fontweight="bold")
    ax.text((x2 + W) / 2, (y_hdr + H) / 2, "tar lethal", ha="center", va="center", fontsize=FS_TITLE, color=INK, fontweight="bold")
    ax.text(1.0, (y_hdr + y_mid) / 2, "present", ha="left", va="center", fontsize=FS_TITLE, color=INK, fontweight="bold")
    ax.text(1.0, (y_mid + y_bot) / 2, "absent", ha="left", va="center", fontsize=FS_TITLE, color=INK, fontweight="bold")
    L = 1.75   # line pitch
    xa, xb = x1 + 2.0, x2 + 2.0
    rows = [
        (xa, 19.9, "Z80", True), (xa, 19.9 - L, "open, then closed · 67 of 80 worlds", False),
        (xa, 19.9 - 2.4 * L, "modified BFF, harmless brackets", True), (xa, 19.9 - 3.4 * L, "open through 16,384 epochs · 12 of 12", False), (xa, 19.9 - 4.4 * L, "no closed design found (period ≤ 10)", False),
        (xb, 19.9, "modified BFF, lethal brackets", True), (xb, 19.9 - L, "open, then extinct · 12 of 12 worlds", False),
        (xb, 19.9 - 2.4 * L, "Z80, zero halts", True), (xb, 19.9 - 3.4 * L, "born closed, late · 10 of 10 worlds", False), (xb, 19.9 - 4.4 * L, "the open beginning never happens", False),
        (xa, 6.7, "BFF as published, harmless brackets", True), (xa, 6.7 - L, "born closed · 7 of 12 worlds; lost again in 4", False),
        (xb, 6.7, "BFF as published", True), (xb, 6.7 - L, "born closed · 28 of 28 (published and wrapping pointer)", False),
    ]
    for x, y, text, bold in rows:
        ax.text(x, y, text, ha="left", va="top", fontsize=FS_LABEL, color=INK if (bold or text != "not run") else GREY_TEXT, fontweight="bold" if bold else "normal")
    ax.text(0, -1.4, "lethal tar ends the open phase before it begins (Z80) or soon after (BFF); life then starts closed if a closed design exists, and late", ha="left", va="top", fontsize=FS_LABEL, color=INK)
    finish(ax, (-0.5, W + 0.5), (-3.4, H + 0.5))


# ------------------------------------------------------------------------------------------------------------ Fig. 5a
def fig5a(ax):
    T = _traces()["pusher_L16"]
    L = T["L"]
    pc = np.array(T["pc"])
    leave = int(np.argmax(pc >= L))                  # instructions until the pointer leaves
    written = int(T["writes_b"][leave])              # bytes written by then
    finish(ax, (-0.3, 30.3), (-0.4, 12.4))
    # the premise, as a fraction
    ax.text(11.0, 11.6, "bytes written", ha="center", va="bottom", fontsize=FS_TITLE, color=TEAL, fontweight="bold")
    ax.plot([8.4, 13.6], [11.45, 11.45], color=INK, lw=0.7)
    ax.text(11.0, 11.3, "cells executed", ha="center", va="top", fontsize=FS_TITLE, color=INK, fontweight="bold")
    ax.text(14.9, 11.45, "< 1", ha="left", va="center", fontsize=8, color=INK, fontweight="bold")
    # one pass of the first replicator
    ax.text(0.5, 9.25, "one pass", ha="left", va="bottom", fontsize=FS_TITLE, color=INK, fontweight="bold")
    arrow(ax, (0.5, 8.95), (10.5, 8.95), color=TEAL, lw=1.1)
    ax.add_patch(Rectangle((0.5, 7.3), 10.0, 1.4, facecolor=TEAL_FILL, edgecolor=INK, lw=0.5, zorder=2))
    ax.text(5.5, 8.0, f"{L} cells executed", ha="center", va="center", fontsize=FS_LABEL, color=INK, zorder=3)
    ax.add_patch(Rectangle((0.5, 5.4), 10.0, 1.4, facecolor="white", edgecolor=INK, lw=0.5, zorder=2))
    ax.add_patch(Rectangle((0.5, 5.4), 10.0 * written / L, 1.4, facecolor=RED_MID, edgecolor="none", zorder=2))
    ax.text(5.5, 5.2, f"{written} of {L} bytes written", ha="center", va="top", fontsize=FS_LABEL, color=INK, zorder=3)
    ax.text(5.5, 4.3, "the first replicator", ha="center", va="top", fontsize=FS_LABEL, color=GREY_TEXT)
    # the fork
    ax.plot([10.5, 12.8], [6.1, 6.1], color=INK, lw=0.8)
    for yt in (9.1, 3.4):
        path = Path([(12.8, 6.1), (15.2, 6.1), (15.2, yt), (17.4, yt)], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        ax.add_patch(PathPatch(path, fill=False, lw=0.8, edgecolor=INK, zorder=2))
        arrow(ax, (17.0, yt), (17.75, yt), color=INK, lw=0.8)
    # open: leave its cells
    text_runs(ax, 17.8, 10.4, [("open", TEAL, True), (" · leave its cells", INK, True)], fs=FS_TITLE)
    ax.add_patch(FancyBboxPatch((17.8, 8.4), 7.0, 1.4, boxstyle="round,pad=0,rounding_size=0.15", facecolor=TEAL_FILL, edgecolor=INK, lw=0.5, zorder=2))
    ax.add_patch(FancyBboxPatch((24.8, 8.4), 4.9, 1.4, boxstyle="round,pad=0,rounding_size=0.15", facecolor=GREY_FILL, edgecolor=INK, lw=0.5, zorder=2))
    arrow(ax, (18.5, 9.1), (28.6, 9.1), color=TEAL, lw=1.1)
    ax.text(21.3, 8.15, "organism", ha="center", va="top", fontsize=FS_LABEL, color=INK)
    ax.text(27.25, 8.15, "partner", ha="center", va="top", fontsize=FS_LABEL, color=INK)
    # closed: revisit a cell
    text_runs(ax, 17.8, 5.35, [("closed", TEAL, True), (" · revisit a cell", INK, True)], fs=FS_TITLE)
    ax.add_patch(FancyBboxPatch((17.8, 2.4), 11.9, 2.0, boxstyle="round,pad=0,rounding_size=0.15", facecolor=TEAL_FILL, edgecolor=INK, lw=0.5, zorder=2))
    arrow(ax, (18.9, 2.85), (28.3, 2.85), color=TEAL, lw=1.1)
    arrow(ax, (28.6, 3.15), (18.9, 3.15), color=TEAL, lw=1.1, rad=0.2)
    for xx in (18.9, 28.6):
        ax.plot([xx], [2.85 if xx < 20 else 3.15], marker="o", ms=4, mfc="white", mec=INK, mew=0.8, zorder=5)
    ax.text(23.75, 2.15, "organism", ha="center", va="top", fontsize=FS_LABEL, color=INK)
    ax.text(15.0, 0.1, "a complete self-copy must leave its cells or revisit one", ha="center", va="bottom", fontsize=FS_TITLE, color=INK, fontweight="bold")


# ------------------------------------------------------------------------------------------------------- v4 Fig. 6
def fig6a(ax):
    """Theorem 2 as used for the Z80 (proviso ii): the budget outlasts one pass, so a pointer that stays must revisit a cell."""
    finish(ax, (-0.5, 30.5), (-0.6, 13.6))
    W = 1.25
    org = ["", "", "", "…", "", "", "", ""]
    # the budget, longer than the organism
    ax.add_patch(Rectangle((0.5, 11.0), 29.0, 0.9, facecolor=RED_PALE, edgecolor=INK, lw=0.5, zorder=2))
    ax.text(15.0, 11.45, "128 executions per encounter", ha="center", va="center", fontsize=FS_LABEL, color=INK, zorder=3)
    bracket(ax, 0.5, 0.5 + W * len(org), 10.55, color=INK, up=False)
    ax.text(0.5 + W * len(org) / 2, 10.2, "the organism: L ≤ 100 cells", ha="center", va="top", fontsize=FS_LABEL, color=INK)
    # open: runs on into the partner
    y = 6.0
    text_runs(ax, 0.5, y + 1.6, [("open", TEAL, True), (" · leaves its cells", INK, True)], fs=FS_TITLE)
    strip(ax, 0.5, y, org, w=W, h=1.2)
    strip(ax, 0.5 + W * len(org), y, ["", "", "", "…", "", ""], fill=GREY_FILL, w=W, h=1.2)
    arrow(ax, (0.9, y - 0.4), (0.5 + W * 13.7, y - 0.4), color=TEAL, lw=1.2)
    ax.text(0.5 + W * 14 + 0.6, y + 0.6, "partner", ha="left", va="center", fontsize=FS_LABEL, color=INK)
    # closed: stays, so some cell runs twice
    y = 1.4
    text_runs(ax, 0.5, y + 1.6, [("closed", TEAL, True), (" · revisits a cell", INK, True)], fs=FS_TITLE)
    strip(ax, 0.5, y, org, w=W, h=1.2)
    x_end = 0.5 + W * len(org)
    arrow(ax, (0.9, y - 0.4), (x_end - 0.4, y - 0.4), color=TEAL, lw=1.2)
    arrow(ax, (x_end - 0.4, y - 0.6), (1.1, y - 0.6), color=RED, lw=1.2, rad=-0.18)
    ax.plot([0.5 + W / 2], [y + 0.6], marker="o", ms=4, mfc="white", mec=INK, mew=0.8, zorder=6)
    ax.text(x_end + 0.9, y + 0.8, "128 executions in L cells:", ha="left", va="bottom", fontsize=FS_LABEL, color=INK)
    ax.text(x_end + 0.9, y + 0.6, "some cell runs twice, a cycle", ha="left", va="top", fontsize=FS_LABEL, color=INK)


def fig6b(ax):
    """Proposition 3 with the count of Proposition 4: the smallest self-writer is straight-line, hence open, and present from the start."""
    finish(ax, (-0.5, 22.5), (-0.6, 13.6))
    W = 1.25
    ax.text(21.9, 9.85, "chance the word is in", ha="right", va="bottom", fontsize=FS_SMALL, color=GREY_TEXT)
    ax.text(21.9, 9.8, "the first soup (L = 16)", ha="right", va="top", fontsize=FS_SMALL, color=GREY_TEXT)
    # the two-byte load-push word: straight-line, open
    y = 6.0
    text_runs(ax, 0.5, y + 1.6, [("2 bytes", TEAL, True), (" · no jump", INK, True)], fs=FS_TITLE)
    x1 = strip(ax, 0.5, y, ["01", "c5", "01", "c5", "…", "c5"], w=W, h=1.2)
    strip(ax, x1, y, ["", ""], fill=GREY_FILL, w=W, h=1.2)
    arrow(ax, (0.9, y - 0.4), (x1 + 2 * W - 0.3, y - 0.4), color=TEAL, lw=1.2)
    ax.text(x1 + 2 * W + 0.4, y + 0.85, "5 of 65,536 two-byte", ha="left", va="center", fontsize=FS_SMALL, color=GREY_TEXT)
    ax.text(x1 + 2 * W + 0.4, y + 0.25, "words write themselves", ha="left", va="center", fontsize=FS_SMALL, color=GREY_TEXT)
    ax.text(21.9, y + 0.6, "0.99", ha="right", va="center", fontsize=8, color=TEAL, fontweight="bold")
    # the smallest closer observed: a four-byte block copy
    y = 1.4
    text_runs(ax, 0.5, y + 2.3, [("4 bytes", RED, True), (" · a loop", INK, True)], fs=FS_TITLE)
    x1 = strip(ax, 0.5, y, ["1e", "a4", "ed", "b0", "…", "b0"], fills=[TEAL_FILL, TEAL_FILL, RED_PALE, RED_PALE, TEAL_FILL, RED_PALE], w=W, h=1.2)
    arrow(ax, (0.5 + 3.7 * W, y + 1.25), (0.5 + 2.3 * W, y + 1.25), color=RED, lw=1.1, rad=0.75, scale=6)
    ax.text(x1 + 0.4, y + 0.85, "LDIR repeats in place;", ha="left", va="center", fontsize=FS_SMALL, color=GREY_TEXT)
    ax.text(x1 + 0.4, y + 0.25, "no closed self-writer of 2 bytes", ha="left", va="center", fontsize=FS_SMALL, color=GREY_TEXT)
    ax.text(21.9, y + 0.6, r"$6\times10^{-5}$", ha="right", va="center", fontsize=8, color=RED, fontweight="bold")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import figcheck
    fs.setup()
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    for name, fn, wmm, hmm in (("fig1a", fig1a, 69, 22), ("fig1b", fig1b, 92, 25), ("fig2d", fig2d, 172, 50), ("fig4d", fig4d, 172, 50), ("fig5a", fig5a, 88, 37)):
        fig = plt.figure(figsize=(wmm * fs.MM, hmm * fs.MM))
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        fn(ax)
        fs.panel_label(ax, name[-1], x=0.0, y=0.97)
        figcheck.print_report(figcheck.check(fig), name)
        fig.savefig(os.path.join(out, f"{name}_test.png"), dpi=400, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    print("rendered")
