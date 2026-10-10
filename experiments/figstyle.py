"""One figure style for every script — Nature profile (2026-10-08; figure guide: sans-serif Helvetica/Arial, text 5–7 pt,
panel letters 8 pt bold lowercase, axis lines and ticks, no gridlines, accessible palette, RGB, vector with editable text;
widths 90 mm single / 120 mm 1.5-column / 180 mm double, depth ≤ 170 mm).

Colour = ablation, fixed across all figures (Okabe–Ito = Wong 2011, colour-blind safe; the control is black).
Census families use a separate muted set so a family is never confused with an ablation.
Concept colours (CONCEPT) and BFF variant colours (BFF_VARIANT) are fixed across every figure of the paper.
Solid lines = pre-registered measure, dashed = post hoc assay measure (say so in the caption).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOR = {
    "none": "#56616F",
    "block-copy": "#E69F00",
    "stack-writes": "#56B4E9",
    "ld-mem": "#009E73",
    "all-ld": "#CC79A7",
    "no-copy": "#D55E00",
    "rmw-only": "#999999",
    # Stage C finer ablations
    "push-only": "#0072B2",
    "ex-sp-only": "#7FBFE9",
    "call-rst": "#003F5C",
    "ld-imm": "#2CA02C",
    "ld-reg": "#8FD18F",
    "cb-page": "#F0E442",
    "ed-loads": "#B8860B",
    # Stage D / later write-side arms
    "stack-write-only": "#1F5F8B",
    "stack-read-only": "#9ECAE1",
    "push": "#0072B2",
    "call-rst-write": "#003F5C",
}
ORDER = ["none", "block-copy", "stack-writes", "stack-write-only", "stack-read-only", "push-only", "ex-sp-only", "call-rst",
         "ld-mem", "ld-imm", "ld-reg", "all-ld", "ed-loads", "cb-page", "no-copy", "rmw-only"]

FAMILY_COLOR = {"push": "#6A51A3", "ex_sp": "#9E9AC8", "ldir": "#1B9E77", "ld_hl": "#66C2A5", "cb_hl": "#A6761D", "rst": "#E7298A", "flooded": "#BDBDBD", "none": "#F0F0F0"}

MM = 1 / 25.4
SINGLE = 90 * MM
COL15 = 120 * MM
DOUBLE = 180 * MM
MAX_DEPTH = 170 * MM

# Concepts, fixed across the paper: open organisms slate (no black fills: black reads too heavy in print), closed vermilion,
# tar grey, intermediate orange. MARK is the neutral for filled markers, FILL_MID for filled bars and blocks.
MARK = "#56616F"
FILL_MID = "#8A95A1"
FILL_LIGHT = "#D6DADF"
CONCEPT = {"open": "#56616F", "closed": "#D55E00", "tar": "#999999", "intermediate": "#E69F00", "first": "#000000", "final": "#D55E00"}
# BFF variants, fixed across the paper (Fig. 5a-c; the headers of Supplementary Fig. 2). Rules (2026-10-10, critic round 4,
# N2): no variant takes a class hue (vermilion = loop or closed, teal = transmitter, slate = open/pusher, purple = lethal
# tar, orange = intermediate) or a length of LEN_COLOR. Checked with OKLab distances and the Machado (2009) protan/deutan
# simulation (dE x 100): against those 19 colours every variant is >= 10.1 normal and >= 5.8 colour-blind (the worst:
# green against the olive of L = 9, and burgundy against the dark brown of L = 12, neither of which shares a figure with
# it); within the five, >= 16.9 normal and >= 8.7 colour-blind. Rose "as published" and burgundy "wrapping pointer" are the
# loop-bearing machines, violet, indigo and green the three with the literal push; the curves stay solid (dashed means a
# post hoc measure in this paper).
BFF_VARIANT = {"std": "#A96464", "wrap": "#91123F", "lit": "#8843EA", "wraplit": "#2A25BB", "wraplitnh": "#3FAE6C"}
BFF_VARIANT_LABEL = {"std": "BFF as published", "wrap": "wrapping pointer", "lit": "literal push", "wraplit": "wrap + literal push",
                     "wraplitnh": "wrap + literal push, harmless brackets"}
# Tape lengths: ONE palette for every figure in which colour encodes L (Figs 1b, 3d, 4e, 6c; Extended Data Fig. 2c).
# Rules (2026-10-10, critic round 3, G2): a colour means one length everywhere, and no length reuses a class hue
# (vermilion = loop or closed, teal = transmitter, slate = open/pusher, purple = lethal tar, orange = intermediate).
# The lengths of the closure stages run light to dark in one blue (20 sky, 32 azure, 50 blue, 64 navy; L = 50 keeps the
# blue of Fig. 4a/b's "L = 50 jump genome" group) with L = 16, the reference length, in black; the size-axis lengths take
# hues of their own. Checked with OKLab distances and the Machado (2009) protan/deutan simulation (dE x 100): within each
# figure's set of lengths the worst normal-vision pair is >= 12 and the worst colour-blind pair >= 12; against the class
# hues drawn beside them >= 14 normal and >= 8 colour-blind (navy against Fig. 4e's dashed lethal-tar purple); any two
# lengths differ by >= 9 (azure 32 against blue 50, which never share a figure). L = 10 was darkened (round 4) from pear
# to green for contrast on white: >= 11 normal and >= 5.6 colour-blind from every other length (olive L = 9, never in the
# same figure), >= 25 normal and >= 20 colour-blind from L = 8 and 12, the lengths it shares ED Fig. 2c with, >= 13.9 normal
# from the class hues and >= 10 normal and colour-blind from the BFF variants.
LEN_COLOR = {
    8: "#8D78F5",    # periwinkle
    9: "#999933",    # olive
    10: "#4D8E16",   # green (was pear #BBCC33, 1.8:1 on white; now 4.0:1)
    12: "#5C3A1A",   # dark brown
    16: "#000000",   # black: the reference length
    20: "#56B4E9",   # sky blue
    32: "#2E8BD8",   # azure
    36: "#CC79A7",   # reddish purple
    50: "#0072B2",   # blue (= Fig. 4's L = 50 jump genome)
    64: "#1B3A6B",   # navy
    100: "#8C510A",  # brown
}
L_COLOR = LEN_COLOR   # old name, kept for the scripts that import it

EXP_PT = 5.0       # exponent size of every power of ten in the figures: the Nature minimum
EXP_RAISE = 0.4    # exponent baseline above the base's baseline, in em of the base
EXP_GAP = 0.06     # gap between "10" and its exponent, in em of the base


def _decade(v) -> bool:
    import numpy as np
    return bool(v > 0 and abs(np.log10(v) - round(np.log10(v))) < 1e-9)


def _width_pt(fig, text, prop) -> float:
    r = fig.canvas.get_renderer()
    w, _, _ = r.get_text_width_height_descent(text, prop, ismath=False)
    return w * 72.0 / r.dpi


def _ascent_pt(fig, prop) -> float:
    """Ascent of matplotlib's text layout box ("lp" metrics), so that two va="top" texts of different sizes can be
    aligned on their baselines."""
    r = fig.canvas.get_renderer()
    _, h, d = r.get_text_width_height_descent("lp", prop, ismath=False)
    return (h - d) * 72.0 / r.dpi


def log10_ticks(ax, axis: str = "x") -> None:
    """Decade tick labels 10^n with the exponent drawn as its own text at EXP_PT (5 pt), raised EXP_RAISE em.

    mathtext sets a superscript at 70% of the label (3.5 pt at 5 pt labels), and the Unicode superscript glyphs print at
    1.8-2.2 pt ink height in two weights in Helvetica; both fall below the 5 pt minimum. Here the base "10" stays the tick
    label (so the axis label is laid out as usual) and is moved left so that "10" + exponent is centred on its tick (x axis)
    or ends at the label pad (y axis); the exponent is an annotation anchored to that tick label. Call this after the
    axis limits are final: figcheck reports an exponent whose tick has moved (exponent-tick)."""
    import numpy as np
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    from matplotlib.transforms import ScaledTranslation
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(LogLocator(base=10))
    a.set_major_formatter(FuncFormatter(lambda v, _: "10" if _decade(v) else ""))
    a.set_minor_formatter(NullFormatter())
    fig = ax.figure
    store = ax.__dict__.setdefault("_fs_exponents", {})
    for t in store.pop(axis, []):
        t.remove()
    locs = list(a.get_majorticklocs())
    ticks = a.get_major_ticks(len(locs))
    lo, hi = sorted(a.get_view_interval())
    made = []
    for tick, loc in zip(ticks, locs):
        lab = tick.label1
        if not hasattr(lab, "_fs_trans0"):
            lab._fs_trans0 = lab.get_transform()
        lab.set_transform(lab._fs_trans0)
        if not (_decade(loc) and lo * (1 - 1e-9) <= loc <= hi * (1 + 1e-9)) or not lab.get_visible():
            continue
        e = str(int(round(np.log10(loc)))).replace("-", "\u2212")
        size = lab.get_fontsize()
        bprop = lab.get_fontproperties()
        eprop = bprop.copy()
        eprop.set_size(EXP_PT)
        gap = EXP_GAP * size
        w_e = _width_pt(fig, e, eprop)
        shift = (gap + w_e) / 2.0 if axis == "x" else gap + w_e
        lab.set_transform(lab._fs_trans0 + ScaledTranslation(-shift / 72.0, 0, fig.dpi_scale_trans))
        dy = EXP_RAISE * size + _ascent_pt(fig, eprop) - _ascent_pt(fig, bprop)    # top-to-top offset for a baseline raise
        t = ax.annotate(e, xy=(1, 1), xycoords=lab, xytext=(gap, dy), textcoords="offset points", ha="left", va="top",
                        fontsize=EXP_PT, color=lab.get_color(), annotation_clip=False, gid="allow-outside")
        t._fs_tick, t._fs_loc = tick, loc
        made.append(t)
    store[axis] = made


def steps(n) -> str:
    """A step count written out, as in the text ("300,000 steps", "a million steps", "ten million steps"): 300000 ->
    "300,000", 1000000 -> "1 million", 10000000 -> "10 million"."""
    n = int(round(n))
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000} million"
    return f"{n:,}"


def setup(profile: str = "nature") -> None:
    """Nature profile: 6 pt body, 5 pt ticks, 8 pt bold lowercase panel letters (use panel_label), Helvetica/Arial."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 5.5,
        "legend.title_fontsize": 5.5, "mathtext.default": "regular",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.minor.size": 1.3, "ytick.minor.size": 1.3,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
        "axes.labelpad": 2.0, "axes.titlepad": 3.0, "axes.grid": False,
        "lines.linewidth": 0.9, "lines.markersize": 3.0, "lines.markeredgewidth": 0.6, "patch.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.labelspacing": 0.3, "legend.borderaxespad": 0.3,
        "savefig.dpi": 600, "figure.dpi": 110, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
    })


def panel_label(ax, letter: str, x: float = -0.16, y: float = 1.02) -> None:
    """Nature panel letter: 8 pt, bold, lowercase, upright, outside the top-left corner of the axes."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right", clip_on=False, gid="panel-label")


def tidy(ax, xlabel: str | None = None, ylabel: str | None = None) -> None:
    """Axis lines and ticks only; labels as 'quantity (unit)'."""
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def color(label: str) -> str:
    base = label.split("@", 1)[0]
    return COLOR.get(label, COLOR.get(base, "#444444"))


def order(labels) -> list[str]:
    known = [l for l in ORDER if l in set(labels)]
    return known + sorted(set(labels) - set(known))


def save(fig, path_no_ext: str, formats=("pdf", "svg", "png")) -> None:
    """Vector first (pdf for submission, svg for us and arXiv, text kept editable), png for quick viewing."""
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    for ext in formats:
        fig.savefig(f"{path_no_ext}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def outside_legend(ax_or_fig, handles=None, labels=None, **kw):
    """Legend to the right of the axes, never on the data."""
    if handles is None:
        handles, labels = ax_or_fig.get_legend_handles_labels()
    return ax_or_fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, **kw)
