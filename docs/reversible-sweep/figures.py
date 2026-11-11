"""Pilot figures for the reversible rule: a portrait of the lattice across the two dials, and maps of what was measured.

  python docs/reversible-sweep/figures.py docs/reversible-sweep/pilot.jsonl <dir with final lattices> <out dir>

Everything drawn is read from the sweep file; nothing is smoothed or interpolated. Pilot only: two runs per cell from a
random start, one from a replicator start, 4,000 and 2,000 updates, which is not long enough to call any cell settled.
"""
from __future__ import annotations

import colorsys
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

INK, MUTED, RULE = "#1B2A33", "#6B7A83", "#D5DCE0"
SEQ = LinearSegmentedColormap.from_list("seq", ["#F4F7F8", "#BFE0DE", "#5FB3B0", "#1F8A8A", "#12545A", "#0B2F36"])
W, H, L = 148, 135, 16
GAMMAS = [0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 1]
EXPS = [2, 4, 6, 8, 10]  # top row = most mutation
PUSHER = " ".join(["01", "c5"] * 8)
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 8, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": RULE, "pdf.fonttype": 42})


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def pick(rows, e, col, boundary="reflect", initial=None, updates=None):
    out = []
    for r in rows:
        q = r["run"]
        if q["mutationExponent"] != e or q.get("boundary", "reflect") != boundary or q.get("initial") != initial:
            continue
        if updates and q["updates"] != updates:
            continue
        if col == "overwrite" and q["interaction"] == "overwrite":
            out.append(r)
        elif col != "overwrite" and q["interaction"] == "reversible" and q["dissipation"] == col:
            out.append(r)
    return out


def mut_label(e):
    per_byte = (8192 // 2**e) / (W * H * L)
    mant, expo = f"{per_byte:.1e}".split("e")
    return f"1/{2**e}", f"{mant}×10$^{{{int(expo)}}}$"


def render(soup):
    """One pixel per cell. Hue names the tape's first two bytes, so a clone and its near copies share a colour;
    brightness is the share of non-zero bytes, so blank tapes are dark."""
    t = soup.reshape(H, W, L)
    key = t[..., 0].astype(np.int64) * 256 + t[..., 1]
    nz = (t != 0).mean(-1)
    return HUES[key] * (0.10 + 0.90 * nz)[..., None]


_h = ((np.arange(65536, dtype=np.uint64) * np.uint64(2654435761)) % np.uint64(2**32)) / 2**32
HUES = np.array([colorsys.hsv_to_rgb(float(h), 0.62, 1.0) for h in _h])


def soup_path(soups, r):
    return Path(soups) / (r["key"].replace("|", "_") + ".u8")


def portrait(rows, soups, out, initial=None, updates=4000, seed=101, title=""):
    cols = GAMMAS + ["overwrite"]
    fig = plt.figure(figsize=(7.2, 4.55))
    gs = fig.add_gridspec(len(EXPS), len(cols) + 1, width_ratios=[1] * len(GAMMAS) + [0.16, 1],
                          left=0.115, right=0.995, top=0.865, bottom=0.085, wspace=0.05, hspace=0.07)
    for i, e in enumerate(EXPS):
        for j, col in enumerate(cols):
            ax = fig.add_subplot(gs[i, j if j < len(GAMMAS) else j + 1])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            got = [r for r in pick(rows, e, col, initial=initial, updates=updates) if r["run"]["seed"] == seed]
            if got and soup_path(soups, got[0]).exists():
                soup = np.fromfile(soup_path(soups, got[0]), np.uint8)[: W * H * L]
                ax.imshow(render(soup), interpolation="nearest", aspect="equal")
            else:
                ax.text(0.5, 0.5, "not run", ha="center", va="center", color=MUTED, transform=ax.transAxes)
            if i == len(EXPS) - 1:
                ax.set_xlabel("ordinary\nrule" if col == "overwrite" else f"{col:g}", labelpad=3)
            if j == 0:
                a, b = mut_label(e)
                ax.set_ylabel(f"{a}\n{b}", rotation=0, ha="right", va="center", labelpad=6, linespacing=1.25)
    fig.text(0.115, 0.972, title, fontsize=10.5, weight="bold", va="top")
    fig.text(0.115, 0.928, f"Each square is the whole lattice (148 × 135 cells), one run (seed {seed}). Colour names a tape's first two bytes; darker = more blank bytes.",
             fontsize=7.5, color=MUTED, va="top")
    fig.text(0.50, 0.012, "dissipation γ: chance per cell and event of forgetting the past  (more cooling to the right)", ha="center", fontsize=8.5)
    fig.text(0.012, 0.475, "mutation  (more heating upwards)", rotation=90, va="center", fontsize=8.5)
    fig.text(0.115, 0.893, "0 = closed, fully reversible", fontsize=7, color=MUTED, va="top")
    fig.text(0.868, 0.893, "1 = same as the ordinary rule (checked exactly)", fontsize=7, color=MUTED, va="top", ha="right")
    for ext in ("png", "pdf"):
        fig.savefig(Path(out) / f"{'portrait_replicator_start' if initial else 'portrait_random_start'}.{ext}", dpi=300)
    plt.close(fig)


def cell_runs(rows, metric, initial=None, updates=4000):
    cols = GAMMAS + ["overwrite"]
    return [[[v for v in (metric(r) for r in sorted(pick(rows, e, col, initial=initial, updates=updates),
                                                    key=lambda r: r["run"]["seed"])) if v is not None]
             for col in cols] for e in EXPS]


def maps(rows, out):
    last = lambda k: (lambda r: r["rows"][-1][k])
    structure = lambda r: 1 - r["rows"][-1]["gzip"]
    panels = [
        ("a", "Disorder", "entropy of byte values, bits (8 = random)", last("entropy"), None, 4000, 3.5, 8, "{:.1f}"),
        ("b", "Heat", "share of bytes where now ≠ before", last("hotBytes"), None, 4000, 0, 0.25, "{:.2f}"),
        ("c", "Blank matter", "share of bytes that are zero", last("zero"), None, 4000, 0, 0.42, "{:.2f}"),
        ("d", "Structure, random start", "1 − gzip ratio of the lattice (0 = random)", structure, None, 4000, 0, 0.75, "{:.2f}"),
        ("e", "Structure, replicator start", "same measure; blank bytes compress too, read with f", structure, "pusher", 2000, 0, 0.75, "{:.2f}"),
        ("f", "Blank matter, replicator start", "left of the line: back at the waiting level", last("zero"), "pusher", 2000, 0, 0.42, "{:.2f}"),
    ]
    cols = GAMMAS + ["overwrite"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.0))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.80, bottom=0.075, wspace=0.17, hspace=0.42)
    xs = list(range(len(GAMMAS))) + [len(GAMMAS) + 0.35]
    for ax, (tag, name, sub, metric, initial, updates, lo, hi, fmt) in zip(axes.flat, panels):
        runs = cell_runs(rows, metric, initial, updates)
        most = max(len(v) for row in runs for v in row)
        for i in range(len(EXPS)):
            for j in range(len(cols)):
                vals = runs[i][j]
                for k, v in enumerate(vals):
                    f = float(np.clip((v - lo) / (hi - lo), 0, 1))
                    h = 0.96 / len(vals)
                    y0 = i - 0.48 + k * h
                    ax.add_patch(plt.Rectangle((xs[j] - 0.48, y0), 0.96, h - (0.03 if k < len(vals) - 1 else 0),
                                               color=SEQ(f), lw=0))
                    ax.text(xs[j], y0 + h / 2 - 0.01, fmt.format(v + 0.0 if abs(v) >= 0.005 else 0.0), ha="center", va="center",
                            fontsize=5.0 if len(vals) > 1 else 5.6, color="white" if f > 0.55 else INK)
        ax.set_xlim(-0.55, xs[-1] + 0.55); ax.set_ylim(len(EXPS) - 0.45, -0.55)
        ax.set_xticks(xs); ax.set_xticklabels([f"{g:g}" for g in GAMMAS] + ["ord."], fontsize=6.3)
        ax.set_yticks(range(len(EXPS))); ax.set_yticklabels([mut_label(e)[0] for e in EXPS], fontsize=6.5)
        ax.tick_params(length=0, pad=2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{tag}   {name}", loc="left", fontsize=8.6, weight="bold", pad=13)
        ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=6.4, color=MUTED, va="bottom")
        if tag == "b":
            ax.text(xs[-1], (len(EXPS) - 1) / 2, "no hidden layer", rotation=90, ha="center", va="center", fontsize=5.8, color=MUTED)
        if tag == "f":
            # Stepped line at the first dissipation above zero where under 30% of bytes are blank. The waiting
            # state a random start sits in has 31-40% blank bytes; the threshold was chosen after seeing the data.
            edge = []
            for i in range(len(EXPS)):
                ok = [j for j in range(1, len(GAMMAS)) if runs[i][j] and runs[i][j][0] < 0.30]
                edge.append(min(ok) if ok else len(GAMMAS))
            px, py = [], []
            for i, j in enumerate(edge):
                px += [j - 0.5, j - 0.5]; py += [i - 0.5, i + 0.5]
            ax.plot(px, py, color="#C2552D", lw=1.4, solid_joinstyle="miter", solid_capstyle="butt", zorder=5)
    for ax in axes[1]:
        ax.set_xlabel("dissipation γ", labelpad=2)
    for ax in axes[:, 0]:
        ax.set_ylabel("mutation", labelpad=3)
    fig.text(0.085, 0.982, "What was measured across the two dials (pilot)", fontsize=10.5, weight="bold", va="top")
    fig.text(0.085, 0.950, "a–d: random start, 4,000 updates; each cell shows its two runs, one above the other.  "
             "e, f: replicator start, 2,000 updates, one run.", fontsize=7.2, color=MUTED, va="top")
    fig.text(0.085, 0.926, "‘ord.’ = the ordinary overwrite rule.  Darker = more; colours saturate at the scale ends, so read the numbers.\n"
             "In a–d a run has either been taken over by a replicator or not, so two runs of a cell often disagree.", fontsize=7.2, color=MUTED, va="top")
    for ext in ("png", "pdf"):
        fig.savefig(Path(out) / f"maps.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    data, soups, out = sys.argv[1:4]
    Path(out).mkdir(parents=True, exist_ok=True)
    rows = load(data)
    portrait(rows, soups, out, None, 4000, 101, "The lattice after 4,000 updates from a random start")
    portrait(rows, soups, out, "pusher", 2000, 101, "The lattice after 2,000 updates from a world of one replicator")
    maps(rows, out)
