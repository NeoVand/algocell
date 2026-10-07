"""One figure style for every script (review 2026-10-07: colours differed in every figure family).

Colour = ablation, fixed across all figures (Okabe–Ito, colour-blind safe; the control is black).
Census families use a separate muted set so a family is never confused with an ablation.
Solid lines = pre-registered measure, dashed = post hoc assay measure (say so in the caption).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOR = {
    "none": "#000000",
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
SINGLE = 85 * MM
DOUBLE = 180 * MM


def setup() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 100, "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def color(label: str) -> str:
    base = label.split("@", 1)[0]
    return COLOR.get(label, COLOR.get(base, "#444444"))


def order(labels) -> list[str]:
    known = [l for l in ORDER if l in set(labels)]
    return known + sorted(set(labels) - set(known))


def save(fig, path_no_ext: str, formats=("pdf", "svg", "png")) -> None:
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    for ext in formats:
        fig.savefig(f"{path_no_ext}.{ext}", bbox_inches="tight")
    plt.close(fig)


def outside_legend(ax_or_fig, handles=None, labels=None, **kw):
    """Legend to the right of the axes, never on the data."""
    if handles is None:
        handles, labels = ax_or_fig.get_legend_handles_labels()
    return ax_or_fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, **kw)
