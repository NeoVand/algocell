"""Conceptual panels for the Nature manuscript, drawn in vector to the visual language fixed on 2026-10-08 from the
user's generated reference set (manuscript/figures/refs/journal-figures-v1, gitignored):

  organism cells: pale teal fill · partner cells: light grey · what the organism writes or where it jumps: vermilion
  instruction pointer: teal · stack pointer: vermilion · structure and type: charcoal · cells numbered from 0.

Panels: fig1a (the pair), fig1b (the pusher), fig2d (the closers; exact traces from results/concept/traces.json),
fig4d (the classification), fig5a (Theorem 2 trajectories, exact traces). Every byte and every arrow endpoint comes from
the data (stage_g_runs.csv, traces.json); nothing is drawn from memory.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
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
BAND = "#E8F4F4"
GREY_FILL = "#E3E5E8"
RED = "#E8431F"
RED_MID = "#F08A6E"
RED_PALE = "#F8D2C8"
FS_BYTE, FS_IDX, FS_LABEL, FS_TITLE = 6.0, 5.0, 6.5, 7.0


def _traces():
    with open(os.path.join(EXP, "results", "concept", "traces.json")) as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------------------------------ helpers
def cells(ax, x0, y0, n, texts=None, fill=TEAL_FILL, fills=None, w=1.0, h=1.0, fs=FS_BYTE, index=True, idx_from=0):
    for i in range(n):
        f = fills[i] if fills is not None else fill
        ax.add_patch(Rectangle((x0 + i * w, y0), w, h, facecolor=f, edgecolor=INK, lw=0.5, zorder=2))
        if texts is not None and i < len(texts) and texts[i]:
            ax.text(x0 + (i + 0.5) * w, y0 + h / 2, texts[i], ha="center", va="center", fontsize=fs, color=INK, zorder=3)
        if index:
            ax.text(x0 + (i + 0.5) * w, y0 - 0.12, str(idx_from + i), ha="center", va="top", fontsize=FS_IDX, color=GREY_TEXT, zorder=3)


def bracket(ax, x0, x1, y, label=None, color=INK, tick=0.22, fs=FS_TITLE, bold=True, pad=0.14, down=True):
    """A square bracket spanning x0..x1 at height y with ticks pointing down (toward the strip) and a label above."""
    s = 1 if down else -1
    ax.plot([x0, x0, x1, x1], [y - s * tick, y, y, y - s * tick], color=color, lw=0.7, solid_capstyle="round", zorder=2)
    if label:
        ax.text((x0 + x1) / 2, y + s * pad, label, ha="center", va="bottom" if down else "top", fontsize=fs, color=color, fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, p0, p1, color=INK, lw=0.9, rad=0.0, ls="-", scale=7, dot=False, zorder=4):
    a = FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=scale, lw=lw, color=color,
                        linestyle=ls, zorder=zorder, shrinkA=0, shrinkB=0, capstyle="round")
    ax.add_patch(a)
    if dot:
        ax.plot([p0[0]], [p0[1]], marker="o", ms=2.6, color=color, zorder=zorder + 1)
    return a


def dotted(ax, x0, x1, y, color, lw=1.1, head=True):
    """Dotted pointer path from x0 to x1 (either direction) with an arrowhead at x1."""
    d = 1 if x1 > x0 else -1
    ax.plot([x0, x1 - d * 0.55], [y, y], color=color, lw=lw, ls=(0, (0.6, 1.6)), dash_capstyle="round", zorder=3)
    if head:
        arrow(ax, (x1 - d * 0.6, y), (x1, y), color=color, lw=lw, scale=7)


def leader(ax, p0, p1, color=INK):
    """Thin leader line (vertical then horizontal) from a label to a feature."""
    ax.plot([p0[0], p0[0], p1[0]], [p0[1], p1[1], p1[1]], color=color, lw=0.6, zorder=2)


def text_runs(ax, x, y, runs, fs=FS_LABEL, va="bottom", gap=0.35):
    """Several text runs on one line, each (text, color, bold), placed from measured text widths (TextPath, no canvas
    draw). The axes limits and aspect must already be set."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    ax.apply_aspect()
    pos = ax.get_position()
    w_in = pos.width * ax.figure.get_figwidth()
    x0, x1 = ax.get_xlim()
    data_per_pt = (x1 - x0) / (w_in * 72.0)
    for text, color, bold in runs:
        ax.text(x, y, text, ha="left", va=va, fontsize=fs, color=color, fontweight="bold" if bold else "normal", zorder=3)
        fp = FontProperties(family=["Helvetica", "Arial", "DejaVu Sans"], weight="bold" if bold else "normal", size=fs)
        width_pt = TextPath((0, 0), text, prop=fp).get_extents().width
        x += width_pt * data_per_pt + gap


def finish(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()


# ------------------------------------------------------------------------------------------------------------ Fig. 1a
def fig1a(ax, L=16):
    n = 2 * L
    cells(ax, 0, 0, L, fill=TEAL_FILL)
    cells(ax, L, 0, L, fill=GREY_FILL, idx_from=L)
    # group brackets
    bracket(ax, 0.05, L - 0.05, 3.75, "organism A", color=TEAL)
    bracket(ax, L + 0.05, n - 0.05, 3.75, "partner B", color=INK)
    # instruction pointer
    arrow(ax, (0.5, 2.25), (0.5, 1.1), color=TEAL, lw=1.1)
    dotted(ax, 1.0, L + 0.4, 1.7, TEAL)
    ax.text(1.1, 2.1, "instruction pointer: starts here, executes one instruction at a time, moves right", ha="left", va="bottom", fontsize=FS_LABEL, color=INK)
    # stack pointer
    arrow(ax, (n - 0.5, -2.0), (n - 0.5, -1.1), color=RED, lw=1.1)
    dotted(ax, n - 1.0, L - 0.4, -1.55, RED)
    ax.text(n - 1.1, -2.0, "stack pointer: starts here; each push writes two cells and moves two cells left", ha="right", va="top", fontsize=FS_LABEL, color=INK)
    # ring: a wide U under the strip from the last cell back to the first
    verts = [(n + 0.15, 0.5), (n + 1.6, -5.6), (-1.6, -5.6), (-0.15, 0.5)]
    ax.add_patch(PathPatch(Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]), fill=False, lw=0.8, edgecolor=INK, zorder=1))
    arrow(ax, (-0.3, -0.1), (-0.15, 0.5), color=INK, lw=0.8, scale=7)
    ax.text(n / 2, -4.55, "a ring: after the last cell comes the first", ha="center", va="top", fontsize=FS_LABEL, color=INK)
    finish(ax, (-1.0, n + 1.0), (-5.0, 4.6))


# ------------------------------------------------------------------------------------------------------------ Fig. 1b
PARTNER_EXAMPLE = "ff e4 22 79 f3 bd 06 83 66 a8 52 c1 bb 96 51 f3".split()


def fig1b(ax, L=16):
    n = 2 * L
    word = ["01", "c5"] * (L // 2)
    partner = list(PARTNER_EXAMPLE[:L])
    partner[13], partner[14] = "c5", "01"   # first push → cells 29, 30
    partner[11], partner[12] = "c5", "01"   # second push → cells 27, 28
    fills = [GREY_FILL] * L
    fills[13] = fills[14] = RED_MID
    fills[11] = fills[12] = RED_PALE
    cells(ax, 0, 0, L, texts=word, fill=TEAL_FILL)
    cells(ax, L, 0, L, texts=partner, fills=fills, idx_from=L)
    # the instruction and the push, labelled above the first four cells
    bracket(ax, 0.05, 2.95, 1.35, color=INK, tick=0.2)
    bracket(ax, 3.05, 3.95, 1.35, color=INK, tick=0.2)
    ax.plot([1.5, 1.5], [1.35, 2.05], color=INK, lw=0.6)
    ax.text(0.0, 3.4, "LD BC,nn", ha="left", va="bottom", fontsize=FS_LABEL, color=INK, fontweight="bold")
    ax.text(0.0, 3.3, "load the next two bytes\ninto register BC", ha="left", va="top", fontsize=FS_LABEL, color=INK, linespacing=1.15)
    ax.plot([3.5, 3.5, 5.6], [1.35, 1.8, 1.8], color=INK, lw=0.6)
    ax.text(5.8, 3.4, "PUSH BC", ha="left", va="bottom", fontsize=FS_LABEL, color=INK, fontweight="bold")
    ax.text(5.8, 3.3, "write BC's two bytes\nwhere the stack pointer is", ha="left", va="top", fontsize=FS_LABEL, color=INK, linespacing=1.15)
    # the pointer runs on into the partner
    arrow(ax, (L - 1.2, 1.3), (L + 1.6, 1.3), color=INK, lw=1.0)
    ax.plot([L + 0.2, L + 0.2], [1.5, 2.25], color=INK, lw=0.6)
    ax.text(L + 0.2, 2.35, "the pointer runs on into the partner\nand executes the partner's bytes", ha="center", va="bottom", fontsize=FS_LABEL, color=INK, linespacing=1.15)
    # inset: the instruction, its operand, what the push writes
    bx, by, bw, bh, hh = L + 3.2, 3.45, 4.1, 0.95, 0.75
    for k, (title, content, hdr) in enumerate((("the instruction", "01 c5 01", TEAL_FILL), ("its operand", "c5 01", GREY_FILL), ("what the push writes", "c5 01", RED_PALE))):
        x = bx + k * (bw + 0.75)
        ax.add_patch(Rectangle((x, by), bw, bh, facecolor="white", edgecolor=INK, lw=0.5))
        ax.add_patch(Rectangle((x, by + bh), bw, hh, facecolor=hdr, edgecolor=INK, lw=0.5))
        ax.text(x + bw / 2, by + bh / 2, content, ha="center", va="center", fontsize=FS_BYTE, color=INK)
        ax.text(x + bw / 2, by + bh + hh / 2, title, ha="center", va="center", fontsize=5.5, color=INK)
        if k == 2:
            ax.text(x - 0.38, by + bh / 2, "=", ha="center", va="center", fontsize=8, color=INK)
    ax.text(bx + 3 * bw + 1.5, 3.2, "the tape is one two-byte word repeated:\nthe code is its own data, and what it writes is itself", ha="right", va="top", fontsize=FS_LABEL, color=INK, linespacing=1.15)
    # the writes: from the push to the far end of the partner
    arrow(ax, (3.5, -0.6), (L + 14.0, -0.95), color=RED, lw=1.1, rad=0.2)
    arrow(ax, (7.5, -0.6), (L + 12.0, -0.95), color=RED_MID, lw=1.0, rad=0.17)
    for a, b in ((L + 13.05, L + 14.95), (L + 11.05, L + 12.95)):
        ax.plot([a, a, b, b], [-0.78, -0.62, -0.62, -0.78], color=INK, lw=0.5)
    finish(ax, (-1.0, n + 1.0), (-3.8, 5.3))


# ------------------------------------------------------------------------------------------------------------ Fig. 2d
def _jumps(pc):
    out = []
    for a, b in zip(pc[:-1], pc[1:]):
        if (b < a or b > a + 4) and (a, b) not in out:
            out.append((a, b))
    return out


def fig2d(ax):
    import pandas as pd
    T = _traces()
    g = pd.read_csv(os.path.join(EXP, "results", "stageG", "stageG", "stage_g_runs.csv"))
    f16 = g[g["L"] == 16]
    pusher_stats = f"copies {f16['first_copied'].median():.2f} of partners, damaged in {f16['first_damaged'].median():.2f} of encounters (medians of 20 worlds)"

    def closer_stats(L, prefix):
        d = g[(g["L"] == L) & g["final_tape"].str.replace(" ", "").str.startswith(prefix)]
        return f"copies {d['final_copied'].median():.2f}, damaged {d['final_damaged'].median():.2f} ({len(d)} worlds)"

    rows = [
        ("pusher_L16", "open: the first replicator", "no backward jump, so the pointer leaves the organism", pusher_stats, []),
        ("retnz_L16", "closed with RET NZ", "its own bytes, read as return addresses, point back into itself", closer_stats(16, "ade321"), [5, 7]),
        ("jrnz_L50", "closed with JR NZ", "relative jumps of 16 bytes back; dashed, the 16-bit program counter wraps to 0", closer_stats(50, "21e521e5"), [9, 10, 23, 24, 37, 38]),
        ("djnz_L20", "closed with DJNZ", "a counted loop; dashed, the 16-bit program counter wraps to 0", closer_stats(20, "21e521e5214e10"), [6, 7, 16, 17]),
        ("ldir_L20", "closed with LDIR", "one instruction repeats in place until the block is copied", closer_stats(20, "1ea4edb0"), [2, 3]),
    ]
    pitch = 4.6
    finish(ax, (-0.6, 51.0), (-(len(rows) - 1) * pitch - 0.7, 3.6))
    for r, (key, title, mech, stats, hl) in enumerate(rows):
        t = T[key]
        L, tape, pc = t["L"], t["tape"].split(), t["pc"]
        y0 = -r * pitch
        fills = [RED_PALE if i in hl else TEAL_FILL for i in range(L)]
        cells(ax, 0, y0, L, texts=tape, fills=fills)
        text_runs(ax, 0, y0 + 3.0, [(f"{title} ({L} cells)", INK, True), (mech, INK, False), (stats, GREY_TEXT, False)])
        executed = sorted(set(pc))
        lo, hi = executed[0], executed[-1]
        if key.startswith("pusher"):
            cells(ax, L, y0, 4, texts=PARTNER_EXAMPLE[:4], fill=GREY_FILL, idx_from=L)
            ax.text(L + 4.4, y0 + 0.5, "partner", ha="left", va="center", fontsize=FS_IDX, color=GREY_TEXT)
            ax.plot([0.5, L + 3.9], [y0 + 1.5, y0 + 1.5], color=INK, lw=0.7, zorder=3)
            arrow(ax, (L + 3.9, y0 + 1.5), (L + 4.6, y0 + 1.5), color=INK, lw=0.7)
            continue
        ax.plot([lo + 0.5, hi + 0.9], [y0 + 1.25, y0 + 1.25], color=INK, lw=0.7, zorder=3)
        arrow(ax, (hi + 0.9, y0 + 1.25), (hi + 1.5, y0 + 1.25), color=INK, lw=0.7)
        if key.startswith("ldir"):
            a = FancyArrowPatch((3.0, y0 + 1.15), (2.1, y0 + 1.15), connectionstyle="arc3,rad=1.6", arrowstyle="-|>", mutation_scale=6, lw=0.9, color=RED, zorder=4)
            ax.add_patch(a)
            ax.plot([3.0], [y0 + 1.15], marker="o", ms=2.6, color=RED, zorder=5)
            continue
        for j, (a_, b_) in enumerate(_jumps(pc)):
            wrap = b_ == 0 and a_ in (35, 15) and not key.startswith("retnz")
            dist = abs(b_ - a_)
            height = 0.75 + 0.4 * j
            rad = (2 * height / dist) * (1 if b_ < a_ else -1)
            arrow(ax, (a_ + 0.5, y0 + 1.12), (b_ + 0.5, y0 + 1.12), color=INK if wrap else RED, lw=0.9, rad=rad, ls=(0, (2.2, 1.4)) if wrap else "-", scale=6, dot=not wrap)


# ------------------------------------------------------------------------------------------------------------ Fig. 4d
def icon(ax, x, y, kind, w=1.25):
    cells(ax, x, y, 6, fill=TEAL_FILL, w=w, h=w, index=False)
    top = y + w * 1.45
    if kind in ("open", "extinct", "forever"):
        arrow(ax, (x + 0.3 * w, top), (x + 7.4 * w, top), color=INK, lw=0.9, scale=7)
        if kind == "extinct":
            ax.plot([x + 7.75 * w] * 2, [y - 0.15 * w, top + 0.45 * w], color=RED, lw=2.0, solid_capstyle="butt")
            return x + 8.4 * w
        if kind == "forever":
            cx, cy, rr = x + 9.3 * w, y + 0.8 * w, 0.7 * w
            for ang in (0, 180):
                th = np.radians(np.linspace(ang + 20, ang + 160, 20))
                ax.plot(cx + rr * np.cos(th), cy + rr * np.sin(th), color=INK, lw=0.9)
                arrow(ax, (cx + rr * np.cos(th[-2]), cy + rr * np.sin(th[-2])), (cx + rr * np.cos(th[-1]), cy + rr * np.sin(th[-1])), color=INK, lw=0.9, scale=6)
            return x + 10.4 * w
        return x + 7.6 * w
    if kind == "closed":
        arrow(ax, (x + 5.5 * w, y + 1.15 * w), (x + 0.5 * w, y + 1.15 * w), color=INK, lw=0.9, rad=0.55, scale=7)
        return x + 6.4 * w


def fig4d(ax):
    W, H = 100.0, 36.0
    xl, ymid, yhdr = 14.0, 10.5, 31.0
    xm = (xl + W) / 2
    for xx in (xl, xm):
        ax.plot([xx, xx], [0, H], color=GRID, lw=0.7)
    for yy in (ymid, yhdr):
        ax.plot([0, W], [yy, yy], color=GRID, lw=0.7)
    ax.text((xl + xm) / 2, (yhdr + H) / 2, "tar benign", ha="center", va="center", fontsize=FS_TITLE, color=INK, fontweight="bold")
    ax.text((xm + W) / 2, (yhdr + H) / 2, "tar lethal", ha="center", va="center", fontsize=FS_TITLE, color=INK, fontweight="bold")
    ax.text(0.5, (ymid + yhdr) / 2, "has a literal-write\ninstruction", ha="left", va="center", fontsize=FS_TITLE, color=INK, fontweight="bold", linespacing=1.2)
    ax.text(0.5, ymid / 2, "has none", ha="left", va="center", fontsize=FS_TITLE, color=INK, fontweight="bold")

    def entry(x, y, kind, head, sub):
        xe = icon(ax, x, y, kind)
        if kind == "open->closed":
            pass
        ax.text(x, y - 0.9, head, ha="left", va="top", fontsize=FS_LABEL, color=INK, fontweight="bold")
        ax.text(x, y - 2.3, sub, ha="left", va="top", fontsize=FS_LABEL, color=INK, linespacing=1.2)
        return xe

    # top left: Z80 open → closed; BFF literal + harmless brackets: open for ever
    x0 = xl + 2.0
    xe = icon(ax, x0, 26.0, "open")
    arrow(ax, (xe + 0.3, 26.9), (xe + 2.2, 26.9), color=INK, lw=0.9)
    icon(ax, xe + 2.8, 26.0, "closed")
    ax.text(x0, 25.1, "Z80 soup: open first, then closed", ha="left", va="top", fontsize=FS_LABEL, color=INK, fontweight="bold")
    ax.text(x0, 23.7, "every one of 40 worlds", ha="left", va="top", fontsize=FS_LABEL, color=INK)
    icon(ax, x0, 18.6, "forever")
    ax.text(x0, 17.7, "BFF with a literal-write instruction and harmless brackets: open for ever", ha="left", va="top", fontsize=FS_LABEL, color=INK, fontweight="bold")
    ax.text(x0, 16.3, "12 of 12 worlds; no closed design exists", ha="left", va="top", fontsize=FS_LABEL, color=INK)
    ax.text(x0, 13.6, "which of the two happens depends on whether the machine\noffers a jump that the organism can copy along with itself", ha="left", va="top", fontsize=FS_LABEL, color=GREY_TEXT, linespacing=1.2)
    # top right: BFF literal: open then extinct
    xr = xm + 2.0
    icon(ax, xr, 22.5, "extinct")
    ax.text(xr, 21.6, "BFF with a literal-write instruction: open first, then extinct", ha="left", va="top", fontsize=FS_LABEL, color=INK, fontweight="bold")
    ax.text(xr, 20.2, "12 of 12 worlds (12 of 12 also without the wrapping pointer)", ha="left", va="top", fontsize=FS_LABEL, color=INK)
    # bottom right: BFF as published: born closed
    icon(ax, xr, 5.6, "closed")
    ax.text(xr, 4.7, "BFF as published: born closed", ha="left", va="top", fontsize=FS_LABEL, color=INK, fontweight="bold")
    ax.text(xr, 3.3, "28 of 28 worlds that produced life", ha="left", va="top", fontsize=FS_LABEL, color=INK)
    # bottom left: not run
    ax.text((xl + xm) / 2, ymid / 2, "not run", ha="center", va="center", fontsize=FS_LABEL, color=GREY_TEXT)
    finish(ax, (-0.5, W + 0.5), (-0.5, H + 0.5))


# ------------------------------------------------------------------------------------------------------------ Fig. 5a
def fig5a(ax_open, ax_closed, K=22):
    T = _traces()
    for ax, key, title in ((ax_open, "pusher_L20", "open: the first replicator"), (ax_closed, "djnz_L20", "closed: the DJNZ closer")):
        t = T[key]
        L = t["L"]
        k = np.arange(K + 1)
        pc, sp, wb = (np.array(t[c][:K + 1]) for c in ("pc", "sp", "writes_b"))
        ax.axhspan(0, L, color=BAND, lw=0, zorder=0)
        ax.step(k, pc, where="post", color=TEAL, lw=1.3, label="instruction pointer", zorder=3)
        ax.step(k, sp, where="post", color=RED, lw=1.3, label="stack pointer", zorder=3)
        wrote = np.where(np.diff(sp) < 0)[0] + 1
        ax.plot(k[wrote], sp[wrote], ls="none", marker="o", color=RED, ms=3.2, label="two bytes written", zorder=4)
        ax.set_xlim(0, K)
        ax.set_ylim(-0.5, 2 * L + 0.5)
        ax.set_yticks([0, L, 2 * L])
        ax.set_xticks([0, 8, 16])
        ax.set_title(title, fontsize=FS_TITLE, color=INK, fontweight="bold", pad=4)
        fs.tidy(ax, "instructions executed", "memory cell" if ax is ax_open else None)
        for sp_ in ("left", "bottom"):
            ax.spines[sp_].set_color(INK)
        ax.tick_params(colors=INK)
        ax.text(0.4, L - 0.7, "the organism's own cells", ha="left", va="top", fontsize=5.5, color=GREY_TEXT)
        ax.text(K - 0.4, 2 * L - 0.6, "the partner's cells", ha="right", va="top", fontsize=5.5, color=GREY_TEXT)
        if key.startswith("pusher"):
            leave = int(np.argmax(pc >= L))
            ax.plot([leave], [L], marker="o", ms=4.5, mfc="white", mec=TEAL, mew=1.1, zorder=5)
            ax.plot([leave, leave], [-0.5, L], color=INK, lw=0.6, ls=(0, (2, 2)), zorder=2)
            ax.text(leave + 1.6, 2 * L - 9.6, f"leaves after {leave} instructions\nwith {wb[leave]} of {L} bytes written", ha="left", va="bottom", fontsize=5.5, color=INK, linespacing=1.15)
        else:
            back = [i for i in range(1, K + 1) if pc[i] < pc[i - 1]]
            if back:
                ax.annotate("returns to an\nearlier cell", xy=(back[0], pc[back[0]] + 0.5), xytext=(back[0] + 1.6, L + 1.8), fontsize=5.5, ha="left", va="bottom", color=INK,
                            arrowprops=dict(arrowstyle="-", color=INK, lw=0.5))
    ax_open.legend(fontsize=5.5, loc="center left", bbox_to_anchor=(0.0, 0.64), frameon=False, handlelength=1.6)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fs.setup()
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    for name, fn, wmm, hmm in (("fig1a", fig1a, 180, 54), ("fig1b", fig1b, 180, 48), ("fig2d", fig2d, 180, 82), ("fig4d", fig4d, 180, 66)):
        fig = plt.figure(figsize=(wmm * fs.MM, hmm * fs.MM))
        ax = fig.add_axes([0.02, 0.0, 0.98, 1.0])
        fn(ax)
        fs.panel_label(ax, name[-1], x=0.0, y=0.97)
        fig.savefig(os.path.join(out, f"{name}_test.png"), dpi=400, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(95 * fs.MM, 46 * fs.MM), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.86, bottom=0.2, wspace=0.12)
    fig5a(a1, a2)
    fs.panel_label(a1, "a", x=-0.3)
    fig.savefig(os.path.join(out, "fig5a_test.png"), dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("rendered")
