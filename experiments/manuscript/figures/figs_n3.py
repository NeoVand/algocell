"""Night of 2026-10-09 (N3): how the first self-confined copier is assembled. DRAFT figure.

    .venv/bin/python manuscript/figures/figs_n3.py      # manuscript/figures/out/fig_assembly.{pdf,svg,png}
"""
from __future__ import annotations

import ast
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_figures as mf  # noqa: E402
from make_figures import fs, plt, GridSpec, save, INK, GREY, RED  # noqa: E402

EXP = mf.EXP
sys.path.insert(0, EXP)
from lod_traj import classify_tapes  # noqa: E402

BLUE = "#0072B2"
SLATE = fs.MARK
LIGHT = "#AEB6BF"
PALE = "#E7E9EC"
K = {0: "start", 1: "copy", 2: "damage", 3: "recombination", 4: "mutation", 5: "copy"}


def byte_colour(b: int) -> str:
    if b in (0xE5, 0xC5):
        return SLATE                 # PUSH
    if b in (0x21, 0x01):
        return LIGHT                 # LD rr,nn
    if b == 0xE3:
        return BLUE                  # EX (SP),HL
    if b in (0xE0, 0xC0, 0xC9):
        return RED                   # RET (conditional or not)
    if b == 0x00:
        return "white"
    return PALE


def h(a):
    return " ".join(f"{int(x):02x}" for x in a)


def chain_of(seed: int, line: int = 0):
    z = np.load(os.path.join(EXP, "runs", "lod_v6_modal", "lod_v6", f"L16_benign_s{seed}", "line_records.npz"))
    tapes = [h(t) for t in z["tape"]]
    C = classify_tapes(sorted(set(tapes)))
    cls = np.array([C[t]["class"] for t in tapes])
    off = int(z["line_lengths"][:line].sum())
    n = int(z["line_lengths"][line])
    ids = z["line_ids"][off:off + n]
    conf = cls[ids] == "confined copier"
    frac = np.cumsum(conf) / np.arange(1, n + 1)
    tr = int(np.where(frac >= 0.9)[0].max())
    lo = next(q for q in range(tr, n) if cls[ids[q]] == "open copier")
    fq = next(q for q in range(lo - 1, -1, -1) if cls[ids[q]] == "confined copier")
    keep = [int(ids[q]) for q in range(lo, max(fq - 2, -1), -1)]          # oldest first, two records past F
    return z, cls, keep, int(ids[fq])


def _line_panel(ax, seed=31002):
    z, cls, keep, F = chain_of(seed)
    n = len(keep)
    for r, rid in enumerate(keep):
        y = n - 1 - r
        for p, b in enumerate(z["tape"][rid]):
            ax.add_patch(plt.Rectangle((p, y + 0.08), 0.92, 0.84, facecolor=byte_colour(int(b)), edgecolor=GREY if int(b) == 0 else "none", lw=0.25))
        c = cls[rid]
        mk = {"open copier": ("o", SLATE, SLATE), "confined copier": ("s", RED, RED), "non-copier": ("o", "white", GREY)}[c]
        ax.plot(17.0, y + 0.5, marker=mk[0], ms=2.6, mfc=mk[1], mec=mk[2], mew=0.5, ls="none", clip_on=False)
        ax.text(17.8, y + 0.5, f"{int(z['step'][rid]):,}  {K[int(z['kind'][rid])]}", fontsize=5.0, va="center", color=INK if rid == F else GREY,
                fontweight="bold" if rid == F else "normal", clip_on=False)
        if rid == F:
            ax.add_patch(plt.Rectangle((-0.15, y), 16.2, 1.0, facecolor="none", edgecolor=INK, lw=0.7))
    ax.set_xlim(-0.3, 16.1)
    ax.set_ylim(-0.2, n + 0.2)
    ax.set_axis_off()
    ax.text(0, n + 0.6, "byte position 0 … 15", fontsize=5, color=GREY, va="bottom")
    # key
    ky = -1.7
    keys = (("PUSH", SLATE), ("LD rr,nn", LIGHT), ("EX (SP),HL", BLUE), ("RET", RED), ("zero", "white"), ("other", PALE))
    xs = (0, 3.0, 6.6, 11.4, 14.4, 17.6)
    for (lab, col), x in zip(keys, xs):
        ax.add_patch(plt.Rectangle((x, ky), 0.8, 0.8, facecolor=col, edgecolor=GREY if col in ("white", PALE) else "none", lw=0.25, clip_on=False))
        ax.text(x + 1.0, ky + 0.4, lab, fontsize=5.0, va="center", color=INK, clip_on=False)
    ky2 = -3.1
    for (lab, mk, fc, ec), x in zip((("open copier", "o", SLATE, SLATE), ("non-copier", "o", "white", GREY), ("confined copier", "s", RED, RED)), (0.4, 6.0, 11.4)):
        ax.plot(x, ky2 + 0.4, marker=mk, ms=2.6, mfc=fc, mec=ec, mew=0.5, ls="none", clip_on=False)
        ax.text(x + 0.7, ky2 + 0.4, lab, fontsize=5.0, va="center", color=INK, clip_on=False)


def _pool_panel(ax, seed=31001):
    z = np.load(os.path.join(EXP, "runs", "lod_v6_modal", "lod_v6", f"L16_benign_s{seed}", "soups.npz"))
    D = pd.read_csv(os.path.join(EXP, "results", "lod", "chain_v6.csv"))
    fstep = int(D[D.seed == seed].F_step.min())

    def has(S, *w):
        m = np.ones(S.shape, bool)
        for i, b in enumerate(w):
            m &= np.roll(S, -i, 1) == b
        return m.any(1)
    rows = []
    for t, S in zip(z["steps"], z["soups"]):
        rows.append((t, has(S, 0x21, 0xE5).mean(), has(S, 0x21, 0xE0).mean(), has(S, 0x21, 0xE3).mean(), has(S, 0xE3, 0x21, 0xE3, 0x21, 0xE0).mean()))
    a = np.array(rows, float)
    floor = 1 / 20000
    for j, (lab, col, ls) in enumerate((("pusher word 21 e5", SLATE, "-"), ("21 e0 (RET PO)", RED, "-"), ("21 e3 (EX (SP),HL)", BLUE, "-"), ("closer motif e3 21 e3 21 e0", INK, (0, (2, 1.5))))):
        ax.plot(a[:, 0], np.maximum(a[:, j + 1], floor), color=col, lw=0.9, ls=ls, marker="o", ms=2.2, label=lab)
    ax.axvline(fstep, color=GREY, lw=0.6, ls=(0, (1, 1.5)))
    ax.text(fstep, 0.6, "founder ", fontsize=5.0, color=GREY, ha="right", va="center")
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.8, 1.2)
    fs.tidy(ax, "step", "share of cells carrying the word")
    ax.legend(fontsize=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, handlelength=1.6, columnspacing=1.0)


def _stats_panel(ax):
    D = pd.read_csv(os.path.join(EXP, "results", "lod", "chain_v6.csv")).drop_duplicates(["seed", "F_step", "F_tape"])
    tot = {"non-copier": 0, "open copier": 0, "mutation": 0, "kept from the line": 0}
    for r in D.itertuples():
        sc = ast.literal_eval(r.source_classes) if isinstance(r.source_classes, str) else {}
        src = ast.literal_eval(r.sources) if isinstance(r.sources, str) else {}
        tot["non-copier"] += sc.get("non-copier", 0)
        tot["open copier"] += sc.get("open copier", 0) + sc.get("confined copier", 0)
        tot["mutation"] += src.get("mutation", 0)
        tot["kept from the line"] += src.get("inherited from the last open copier", 0)
    cols = {"non-copier": PALE, "open copier": SLATE, "mutation": "white", "kept from the line": LIGHT}
    n = sum(tot.values())
    y = np.arange(len(tot))[::-1]
    for yy, (c, v) in zip(y, tot.items()):
        ax.barh(yy, v / n, color=cols[c], edgecolor=GREY, lw=0.3, height=0.62)
        ax.text(v / n + 0.015, yy, f"{v / n:.2f}", fontsize=5.0, va="center", color=INK)
    ax.set_yticks(y, list(tot))
    ax.set_xlim(0, 1)
    fs.tidy(ax, "share of the founders' bytes, by what wrote them", None)
    kinds = D.F_kind.value_counts()
    ev = D.contributing_events
    print(f"[fig_assembly] panel c: {len(D)} founders in {D.seed.nunique()} worlds; completed by recombination {kinds.get('novel', 0)}, partial overwrite {kinds.get('damage', 0)}, "
          f"point mutation {kinds.get('mut', 0)}; contributing events median {ev.median():.0f} ({ev.min()}-{ev.max()}); steps after the last open copier median {D.steps_elapsed.median():.0f}")


def fig_assembly(out):
    fig = plt.figure(figsize=(fs.DOUBLE, 120 * fs.MM))
    gs = GridSpec(1, 1, figure=fig, left=0.03, right=0.36, top=0.93, bottom=0.16)
    gs2 = GridSpec(2, 1, figure=fig, left=0.645, right=0.985, top=0.93, bottom=0.12, height_ratios=[2.0, 1], hspace=0.75)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs2[0, 0])
    axc = fig.add_subplot(gs2[1, 0])
    _line_panel(axa)
    _pool_panel(axb)
    _stats_panel(axc)
    for ax, letter, y in ((axa, "a", 0.985), (axb, "b", 0.985), (axc, "c", 0.33)):
        mf.label(ax, letter) if ax is not axa else fig.text(0.004, y, letter, fontsize=8, fontweight="bold", va="top", gid="panel-label")
    save(fig, os.path.join(out, "fig_assembly"))


if __name__ == "__main__":
    fs.setup()
    fig_assembly(os.path.join(HERE, "out"))
