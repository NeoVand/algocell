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
# BFF variants, fixed across the paper.
BFF_VARIANT = {"std": "#56616F", "wrap": "#D55E00", "wraplit": "#0072B2", "wraplitnh": "#009E73", "lit": "#CC79A7"}
BFF_VARIANT_LABEL = {"std": "BFF as published", "wrap": "wrapping pointer", "wraplit": "wrap + literal push", "wraplitnh": "wrap + literal, no halt", "lit": "literal push"}
# Tape lengths, fixed across the paper (Okabe–Ito order).
L_COLOR = {16: "#000000", 20: "#E69F00", 50: "#0072B2", 64: "#D55E00", 36: "#009E73", 100: "#CC79A7", 9: "#56B4E9", 25: "#F0E442"}


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
