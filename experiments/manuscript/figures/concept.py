"""Conceptual panels for the Nature manuscript, drawn with one shared vocabulary (CONCEPTUAL_FIGURES.md):

memory = a horizontal strip of cells; the organism's bytes = black outline on white; the partner's = grey outline on
light grey; the instruction pointer = a small black arrow above the strip; the stack/write pointer = a hollow arrow
below; a written byte = vermilion fill; a control transfer (jump, return, hardware repeat) = a vermilion arrow, dashed
when it is the 16-bit program counter wrapping from 65535 to 0.

Panels: fig1a (the pair), fig1b (the pusher), fig2d (the closers; traces from results/concept/traces.json produced by
trace_z80.py), fig4d (the classification), fig5a (Theorem 2 trajectories, same traces).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path
from matplotlib.patches import PathPatch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, EXP)
import figstyle as fs  # noqa: E402

VERM = fs.CONCEPT["closed"]
GREY = fs.CONCEPT["tar"]
LIGHT = "#EBEBEB"
BLACK = "#000000"
FS_BYTE = 5.0
FS_LABEL = 5.0
FS_SMALL = 4.8


def _traces():
    with open(os.path.join(EXP, "results", "concept", "traces.json")) as fh:
        return json.load(fh)


def strip(ax, x0, y0, n, texts=None, organism=True, written=(), faint=(), w=1.0, h=1.0, text_color=None):
    """n cells from (x0, y0). organism: black on white; else grey on light grey. written: vermilion cells; faint: lighter."""
    for i in range(n):
        fc, ec, tc = (("white", BLACK, BLACK) if organism else (LIGHT, GREY, "#444444"))
        if i in written:
            fc, tc = VERM, "white"
        elif i in faint:
            fc, tc = "#EDAE8A", BLACK
        ax.add_patch(Rectangle((x0 + i * w, y0), w, h, facecolor=fc, edgecolor=ec, lw=0.5, zorder=2))
        if texts is not None and i < len(texts) and texts[i]:
            ax.text(x0 + (i + 0.5) * w, y0 + h / 2, texts[i], ha="center", va="center", fontsize=FS_BYTE, color=text_color or tc, zorder=3)


def bracket(ax, x0, x1, y, label, above=True, tick=0.18, fontsize=FS_LABEL, color=BLACK, pad=0.12):
    """Thin bracket with end ticks and a centred label."""
    s = 1 if above else -1
    ax.plot([x0, x0, x1, x1], [y - s * tick, y, y, y - s * tick], color=color, lw=0.5, solid_capstyle="butt", zorder=2)
    ax.text((x0 + x1) / 2, y + s * pad, label, ha="center", va="bottom" if above else "top", fontsize=fontsize, color=color, zorder=3)


def arrow(ax, p0, p1, color=BLACK, lw=0.7, rad=0.0, ls="-", hollow=False, scale=6, zorder=4):
    a = FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=scale, lw=lw,
                        color=color, linestyle=ls, zorder=zorder, shrinkA=0, shrinkB=0)
    if hollow:
        a.set_facecolor("white")
    ax.add_patch(a)
    return a


def dotted_path(ax, x0, x1, y, color=BLACK, head=True):
    ax.plot([x0, x1 - (0.6 if head else 0)], [y, y], color=color, lw=0.6, ls=(0, (1, 1.5)), zorder=3)
    if head:
        arrow(ax, (x1 - 0.7, y), (x1, y), color=color, lw=0.6)


def finish(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()


# ------------------------------------------------------------------------------------------------------------ Fig. 1a
def fig1a(ax, L=16):
    n = 2 * L
    strip(ax, 0, 0, L, organism=True)
    strip(ax, L, 0, L, organism=False)
    for x, lab, ha in ((0.5, "0", "center"), (L - 0.15, str(L - 1), "right"), (L + 0.15, str(L), "left"), (n - 0.5, str(n - 1), "center")):
        ax.text(x, -0.12, lab, ha=ha, va="top", fontsize=FS_SMALL, color="#666666")
    # pointer above, dotted path to the right
    arrow(ax, (0.5, 1.95), (0.5, 1.08), lw=0.7)
    dotted_path(ax, 1.0, L + 0.5, 1.5)
    ax.text(L / 2 + 1.0, 1.72, "instruction pointer: starts at A's first byte, moves right", ha="center", va="bottom", fontsize=FS_LABEL)
    bracket(ax, 0.05, L - 0.05, 3.1, f"organism A, L = {L} bytes", above=True)
    bracket(ax, L + 0.05, n - 0.05, 3.1, "partner B", above=True)
    # stack pointer below, dotted path to the left
    arrow(ax, (n - 0.5, -1.65), (n - 0.5, -0.78), lw=0.7, hollow=True)
    dotted_path(ax, n - 1.0, L + 0.5, -1.2)
    ax.text(L / 2, -0.95, "stack pointer: starts at B's last byte;\neach push writes two bytes to its left and moves there", ha="center", va="top", fontsize=FS_LABEL)
    # ring: a U beneath joining the two ends
    verts = [(n + 0.1, 0.5), (n + 1.9, 0.5), (n + 1.9, -3.9), (n / 2, -3.9), (-1.9, -3.9), (-1.9, 0.5), (-0.1, 0.5)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    ax.add_patch(PathPatch(Path(verts, codes), fill=False, lw=0.6, edgecolor="#555555", zorder=1))
    arrow(ax, (-0.35, 0.2), (0.0, 0.5), color="#555555", lw=0.6)
    ax.text(n / 2, -4.25, f"a ring of {n} bytes: after byte {n - 1} comes byte 0", ha="center", va="top", fontsize=FS_LABEL, color="#555555")
    ax.text(n / 2, 4.3, "one encounter: 128 instructions, then both halves are written back", ha="center", va="bottom", fontsize=FS_LABEL)
    finish(ax, (-2.6, n + 2.6), (-5.3, 5.1))


# ------------------------------------------------------------------------------------------------------------ Fig. 1b
PARTNER_EXAMPLE = "ff e4 22 79 f3 bd 06 83 66 a8 52 c1 bb 96 51 f3".split()


def fig1b(ax, L=16):
    n = 2 * L
    word = ["01", "c5"] * (L // 2)
    partner = list(PARTNER_EXAMPLE[:L])
    partner[13], partner[14] = "c5", "01"   # first push: cells 29, 30
    partner[11], partner[12] = "c5", "01"   # second push: cells 27, 28
    strip(ax, 0, 0, L, texts=word, organism=True)
    strip(ax, L, 0, L, texts=partner, organism=False, written=(13, 14), faint=(11, 12))
    for i, lab in ((0, "0"), (3, "3"), (L, str(L)), (n - 1, str(n - 1))):
        ax.text(i + 0.5, -0.12, lab, ha="center", va="top", fontsize=FS_SMALL, color="#666666")
    # pointer above: runs through A and on into B
    arrow(ax, (0.5, 1.9), (0.5, 1.08), lw=0.7)
    dotted_path(ax, 1.0, L + 4.5, 1.45)
    ax.text(L + 8.5, 1.75, "the pointer runs on into the partner\nand executes its bytes", ha="center", va="bottom", fontsize=FS_LABEL)
    # the instruction, below cells 0-2, and the push, below cell 3
    ax.plot([0.05, 0.05, 2.95, 2.95], [-0.4, -0.55, -0.55, -0.4], color=BLACK, lw=0.5)
    ax.text(0.0, -0.7, "LD BC,nn: loads the two bytes\nafter the opcode, c5 01, into BC", ha="left", va="top", fontsize=FS_LABEL)
    ax.plot([3.05, 3.05, 3.95, 3.95], [-0.4, -0.55, -0.55, -0.4], color=BLACK, lw=0.5)
    ax.text(3.5, -2.05, "PUSH BC: writes them", ha="left", va="top", fontsize=FS_LABEL, color=VERM)
    # write arrows under the strip to the far end of the partner
    arrow(ax, (3.5, -0.6), (L + 14.0, -0.05), color=VERM, lw=0.8, rad=0.17)
    arrow(ax, (7.5, -0.35), (L + 12.0, -0.05), color="#EDAE8A", lw=0.7, rad=0.13)
    ax.text(L + 7.0, -2.9, "every push lands two bytes further left: the partner\nfills from its far end, in phase with the organism", ha="center", va="top", fontsize=FS_LABEL, color=VERM)
    arrow(ax, (n - 0.5, -1.35), (n - 0.5, -0.78), lw=0.7, hollow=True)
    # inset: instruction / operand / written
    bx, by, bw, bh = L + 0.0, 3.55, 3.8, 1.1
    for k, (title, content) in enumerate((("the instruction", "01 c5 01"), ("its operand", "c5 01"), ("what PUSH writes", "c5 01"))):
        x = bx + k * (bw + 2.6)
        ax.add_patch(Rectangle((x, by), bw, bh, facecolor="white", edgecolor=BLACK, lw=0.5))
        ax.text(x + bw / 2, by + bh / 2, content, ha="center", va="center", fontsize=FS_BYTE)
        ax.text(x + bw / 2, by + bh + 0.15, title, ha="center", va="bottom", fontsize=FS_SMALL, color="#444444")
        if k:
            ax.text(x - 1.3, by + bh / 2, "=", ha="center", va="center", fontsize=6.5) if k == 2 else ax.text(x - 1.3, by + bh / 2, "last two\nbytes", ha="center", va="center", fontsize=4.6, color="#444444")
    ax.text(0, 3.0, "the tape is one two-byte word repeated:\ncode = data = literal", ha="left", va="bottom", fontsize=FS_LABEL)
    ax.text(0, 2.35, "no loop, no counter, no reading of the surroundings", ha="left", va="bottom", fontsize=FS_SMALL, color="#444444")
    finish(ax, (-0.6, n + 1.0), (-4.2, 5.45))


# ------------------------------------------------------------------------------------------------------------ Fig. 2d
def _jumps(pc):
    """Non-sequential moves (from executed cell, to next cell) in a pc sequence, deduplicated, in order of first use."""
    out = []
    for a, b in zip(pc[:-1], pc[1:]):
        if b < a or b > a + 4:
            if (a, b) not in out:
                out.append((a, b))
    return out


def fig2d(ax):
    import pandas as pd
    T = _traces()
    g = pd.read_csv(os.path.join(EXP, "results", "stageG", "stageG", "stage_g_runs.csv"))
    first16 = g[g["L"] == 16]
    stat_pusher = f"copies {first16['first_copied'].median():.2f} of partners · damaged in {first16['first_damaged'].median():.2f} of encounters (medians, 20 worlds)"

    def closer_stats(L, prefix):
        d = g[(g["L"] == L) & g["final_tape"].str.replace(" ", "").str.startswith(prefix)]
        return f"copies {d['final_copied'].median():.2f} · damaged {d['final_damaged'].median():.2f} ({len(d)} worlds)"

    rows = [
        ("pusher_L16", "open: the pusher\nL = 16", "LD BC,nn ; PUSH BC  ×8", stat_pusher, {}),
        ("retnz_L16", "closed with RET NZ\nL = 16", "XOR L ; EX (SP),HL ; LD HL,nn ; RET NZ ; XOR L ; RET NZ  ×2", closer_stats(16, "ade321"), {}),
        ("jrnz_L50", "closed with JR NZ\nL = 50", "LD HL,nn ; PUSH HL … with JR NZ,−16 every 14 bytes", closer_stats(50, "21e521e5"), {}),
        ("djnz_L20", "closed with DJNZ\nL = 20", "LD HL,nn ; PUSH HL ; … ; LD C,(HL) ; DJNZ −27", closer_stats(20, "21e521e5214e10"), {}),
        ("ldir_L20", "closed with LDIR\nL = 20", "LD E,n ; LDIR  ×5  (the hardware loop repeats on the spot)", closer_stats(20, "1ea4edb0"), {}),
    ]
    mech = {"retnz_L16": "its own bytes, read as return addresses, point back into itself",
            "jrnz_L50": "relative jumps; dashed: the 16-bit program counter wraps to 0",
            "djnz_L20": "a counted loop; dashed: the 16-bit program counter wraps to 0",
            "ldir_L20": "one instruction repeats in place until the block is copied"}
    pitch = 4.7
    for r, (key, label, disasm, stats, _) in enumerate(rows):
        t = T[key]
        L = t["L"]
        y0 = -r * pitch
        tape = t["tape"].split()
        pc = t["pc"]
        executed = sorted(set(pc))
        strip(ax, 0, y0, L, texts=tape, organism=True)
        ax.text(-0.9, y0 + 0.5, label, ha="right", va="center", fontsize=FS_LABEL)
        ax.text(0, y0 - 0.15, disasm, ha="left", va="top", fontsize=FS_SMALL, color="#444444")
        if key.startswith("pusher"):
            strip(ax, L, y0, 4, texts=["", "", "", ""], organism=False)
            ax.text(L + 2.0, y0 - 0.15, "partner", ha="center", va="top", fontsize=FS_SMALL, color="#444444")
            dotted_path(ax, 0.5, L + 4.6, y0 + 1.35)
            xr = L + 5.4
            ax.text(xr, y0 + 1.05, "no backward jump: the pointer leaves the organism", ha="left", va="center", fontsize=FS_LABEL)
            ax.text(xr, y0 + 0.2, stats, ha="left", va="center", fontsize=FS_SMALL, color="#444444")
            continue
        lo, hi = executed[0], executed[-1]
        dotted_path(ax, lo + 0.5, hi + 1.0, y0 + 1.35, head=False)
        if key.startswith("ldir"):
            a = FancyArrowPatch((2.95, y0 + 1.15), (2.05, y0 + 1.15), connectionstyle="arc3,rad=1.7", arrowstyle="-|>", mutation_scale=5, lw=0.7, color=VERM, zorder=4)
            ax.add_patch(a)
        else:
            for j, (a_, b_) in enumerate(_jumps(pc)):
                wrap = b_ == 0 and a_ in (35, 15) and not key.startswith("retnz")   # sequential increment past 65535
                dist = abs(b_ - a_)
                height = 1.0 + 0.55 * j
                rad = (2 * height / dist) * (1 if b_ < a_ else -1)   # arc3: positive bulges up for leftward arrows, negative for rightward
                arrow(ax, (a_ + 0.5, y0 + 1.12), (b_ + 0.5, y0 + 1.12), color=VERM, lw=0.75, rad=rad, ls=(0, (2, 1.2)) if wrap else "-", scale=5)
        if L <= 20:
            xr = L + 1.2
            ax.text(xr, y0 + 1.05, mech[key], ha="left", va="center", fontsize=FS_LABEL)
            ax.text(xr, y0 + 0.2, stats, ha="left", va="center", fontsize=FS_SMALL, color="#444444")
        else:
            ax.text(L, y0 - 0.15, mech[key], ha="right", va="top", fontsize=FS_LABEL)
            ax.text(L, y0 - 0.85, stats, ha="right", va="top", fontsize=FS_SMALL, color="#444444")
    ax.text(25, -(len(rows) - 1) * pitch - 1.9, "closure is a cycle in control flow, not a wall around the bytes: four unrelated instructions, one property", ha="center", va="top", fontsize=FS_LABEL)
    finish(ax, (-9.5, 56.5), (-(len(rows) - 1) * pitch - 3.2, 3.0))


# ------------------------------------------------------------------------------------------------------------ Fig. 4d
def icon(ax, x, y, kind, w=1.0):
    """Tiny strip icon: 6 organism cells (+3 partner cells for open kinds) with the pointer's path."""
    strip(ax, x, y, 6, organism=True, w=w, h=w)
    if kind in ("open", "extinct", "forever"):
        strip(ax, x + 6 * w, y, 3, organism=False, w=w, h=w)
        ax.plot([x + 0.5 * w, x + 8.3 * w], [y + 1.35 * w] * 2, color=BLACK, lw=0.6, ls=(0, (1, 1.5)))
        if kind == "extinct":
            ax.plot([x + 8.5 * w] * 2, [y + 1.05 * w, y + 1.65 * w], color=BLACK, lw=1.0)
        else:
            arrow(ax, (x + 8.3 * w, y + 1.35 * w), (x + 9.0 * w, y + 1.35 * w), lw=0.6, scale=5)
        if kind == "forever":
            ax.text(x + 9.6 * w, y + 1.35 * w, "∞", ha="left", va="center", fontsize=6.5)
        return x + (11.5 if kind == "forever" else 9.6) * w
    if kind == "closed":
        arrow(ax, (x + 5.5 * w, y + 1.1 * w), (x + 1.5 * w, y + 1.1 * w), color=VERM, lw=0.75, rad=-0.45, scale=5)
        return x + 6.6 * w


def fig4d(ax):
    W, H = 100.0, 34.0
    x_lab, y_hdr = 15.0, 28.5
    cols = [(x_lab, (x_lab + W) / 2, "tar benign: running into it does not stop the pointer"), ((x_lab + W) / 2, W, "tar lethal: running into it halts the pointer")]
    rws = [(10.0, y_hdr, "a literal write channel\n(an instruction writes\nits own operand)"), (0.0, 10.0, "no literal\nwrite channel")]
    for x0, x1, lab in cols:
        ax.text((x0 + x1) / 2, (y_hdr + H) / 2, lab, ha="center", va="center", fontsize=FS_LABEL)
    for y0, y1, lab in rws:
        ax.text(x_lab - 0.8, (y0 + y1) / 2, lab, ha="right", va="center", fontsize=FS_LABEL)
    for xx in (x_lab, (x_lab + W) / 2, W):
        ax.plot([xx, xx], [0, H], color=BLACK, lw=0.5)
    for yy in (0, 10.0, y_hdr, H):
        ax.plot([x_lab if yy in (0, H) else 0, W], [yy, yy], color=BLACK, lw=0.5)
    ax.plot([0, x_lab], [0, 0], color=BLACK, lw=0.5)
    ax.plot([0, 0], [0, y_hdr], color=BLACK, lw=0.5)
    # top left: Z80 (open → closed) and BFF+literal, no-halt (open for ever)
    x0 = x_lab + 1.5
    xe = icon(ax, x0, 23.8, "open", w=0.9)
    arrow(ax, (xe + 0.2, 24.4), (xe + 1.8, 24.4), lw=0.6, scale=5)
    xe2 = icon(ax, xe + 2.2, 23.8, "closed", w=0.9)
    ax.text(xe2 + 0.8, 24.4, "Z80 soup: open first, then closed, 40 of 40 worlds\n(300,000 steps; L = 16 and 50)", ha="left", va="center", fontsize=FS_LABEL)
    xe = icon(ax, x0, 17.8, "forever", w=0.9)
    ax.text(xe2 + 0.8, 18.4, "BFF with a literal and harmless unmatched brackets:\nopen for ever, 12 of 12 worlds; no closed design\nexists (none among periodic programs to period 10)", ha="left", va="center", fontsize=FS_LABEL)
    ax.text((x_lab + (x_lab + W) / 2) / 2, 12.3, "which of the two happens depends on whether the instruction set\noffers a closer that the organism can copy along with itself", ha="center", va="center", fontsize=FS_SMALL, color="#444444")
    # top right: BFF + literal (lethal) → extinct
    xr = (x_lab + W) / 2 + 1.5
    xe = icon(ax, xr, 21.3, "extinct", w=0.9)
    ax.text(xe + 0.8, 21.9, "BFF with a literal: open first, then extinct,\n12 of 12 worlds (12 of 12 also without\nthe wrapping pointer)", ha="left", va="center", fontsize=FS_LABEL)
    # bottom right: BFF as published → born closed
    xe = icon(ax, xr, 4.3, "closed", w=0.9)
    ax.text(xe + 0.8, 4.9, "BFF as published: born closed, 28 of 28 worlds\nthat produced life (9 as published,\n19 with a wrapping pointer)", ha="left", va="center", fontsize=FS_LABEL)
    # bottom left: not run
    ax.text((x_lab + (x_lab + W) / 2) / 2, 5.0, "not run", ha="center", va="center", fontsize=FS_LABEL, color="#777777")
    finish(ax, (-0.5, W + 0.5), (-0.5, H + 0.5))


# ------------------------------------------------------------------------------------------------------------ Fig. 5a
def fig5a(ax_open, ax_closed, K=22):
    T = _traces()
    for ax, key, title in ((ax_open, "pusher_L20", "open: the pusher"), (ax_closed, "djnz_L20", "closed: the DJNZ closer")):
        t = T[key]
        L = t["L"]
        k = np.arange(K + 1)
        pc, sp, wb = np.array(t["pc"][:K + 1]), np.array(t["sp"][:K + 1]), np.array(t["writes_b"][:K + 1])
        ax.axhspan(0, L, color="#EFEFEF", lw=0, zorder=0)
        ax.step(k, pc, where="post", color=BLACK, lw=0.9, label="instruction pointer")
        ax.step(k, sp, where="post", color=GREY, lw=0.9, label="stack pointer")
        wrote = np.where(np.diff(sp) < 0)[0] + 1
        ax.plot(k[wrote], sp[wrote], ls="none", marker="_", color=VERM, ms=4, mew=0.9, label="two bytes written")
        ax.set_xlim(0, K)
        ax.set_ylim(-0.5, 2 * L + 0.5)
        ax.set_yticks([0, L, 2 * L])
        ax.set_xticks([0, 8, 16])
        ax.set_title(title, fontsize=6)
        fs.tidy(ax, "instructions executed", "address in the ring" if ax is ax_open else None)
        ax.text(K - 0.3, 1.2, f"the organism's own {L} bytes", ha="right", va="bottom", fontsize=FS_SMALL, color="#555555")
        ax.text(K - 0.3, 2 * L - 0.8, "the partner's bytes", ha="right", va="top", fontsize=FS_SMALL, color="#555555")
        if key.startswith("pusher"):
            leave = int(np.argmax(pc >= L))
            ax.annotate(f"leaves after {leave} instructions\nwith {wb[leave]} of {L} bytes written", xy=(leave, L), xytext=(leave + 0.8, 2 * L - 6.5), fontsize=FS_SMALL, ha="left", va="top",
                        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.5))
        else:
            back = [i for i in range(1, K + 1) if pc[i] < pc[i - 1]]
            if back:
                ax.annotate("returns to an\nearlier address", xy=(back[0], pc[back[0]]), xytext=(back[0] + 1.5, L + 1.5), fontsize=FS_SMALL, ha="left", va="bottom",
                            arrowprops=dict(arrowstyle="-", color="#555555", lw=0.5))
    ax_open.legend(fontsize=4.6, loc="center left", bbox_to_anchor=(0.0, 0.6), frameon=False, handlelength=1.5)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fs.setup()
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    for name, fn, wmm, hmm in (("fig1a", fig1a, 72, 30), ("fig1b", fig1b, 104, 30), ("fig2d", fig2d, 180, 70), ("fig4d", fig4d, 180, 62)):
        fig = plt.figure(figsize=(wmm * fs.MM, hmm * fs.MM))
        ax = fig.add_axes([0, 0, 1, 1])
        fn(ax)
        fs.panel_label(ax, name[-1], x=0.0, y=0.98)
        fig.savefig(os.path.join(out, f"{name}_test.png"), dpi=400, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(84 * fs.MM, 42 * fs.MM), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.88, bottom=0.2, wspace=0.12)
    fig5a(a1, a2)
    fs.panel_label(a1, "a", x=-0.3)
    fig.savefig(os.path.join(out, "fig5a_test.png"), dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("rendered")
