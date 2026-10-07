"""Assemble the analysis outputs of one or more batches into a results report
(Markdown) that answers each pre-registered hypothesis explicitly and shows
every cell of the grid, including nulls.

    python report.py runs/stageA [runs/stageB ...] --out runs/REPORT.md
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def load(dirs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cells, assays, succ = [], [], []
    for d in dirs:
        a = os.path.join(d, "analysis")
        cells.append(pd.read_csv(os.path.join(a, "cells.csv")))
        assays.append(pd.read_csv(os.path.join(a, "assays.csv")))
        if os.path.exists(os.path.join(a, "succession_cells.csv")):
            succ.append(pd.read_csv(os.path.join(a, "succession_cells.csv")))
    return pd.concat(cells, ignore_index=True), pd.concat(assays, ignore_index=True), (pd.concat(succ, ignore_index=True) if succ else pd.DataFrame())


def fmt_cell(r) -> str:
    rep = f"{int(r['rep_emerged'])}/{int(r['n'])}" if "rep_emerged" in r and pd.notna(r.get("rep_emerged")) else "–"
    med = f"{r['median_trep']:.0f}" if pd.notna(r.get("median_trep")) else "–"
    return f"{rep} (median t_rep {med})"


def cell_table(cells: pd.DataFrame, tape: int) -> str:
    sub = cells[cells["tape"] == tape]
    labels = list(dict.fromkeys(sub["label"]))
    steps = sorted(sub["steps"].unique())
    ks = sorted(sub["k"].unique())
    lines = ["| ablation | " + " | ".join(f"{s} steps · 1/2^{k}" for s in steps for k in ks) + " |", "|---|" + "---|" * (len(steps) * len(ks))]
    for lab in labels:
        row = [lab]
        for s in steps:
            for k in ks:
                r = sub[(sub["label"] == lab) & (sub["steps"] == s) & (sub["k"] == k)]
                row.append(fmt_cell(r.iloc[0]) if len(r) else "–")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def verdicts(cells: pd.DataFrame, assays: pd.DataFrame) -> list[str]:
    out = []
    L16 = cells[cells["tape"] == 16]

    def frac(label, steps, k):
        r = L16[(L16["label"] == label) & (L16["steps"] == steps) & (L16["k"] == k)]
        return (r.iloc[0]["rep_emerged"], r.iloc[0]["n"], r.iloc[0]["median_trep"]) if len(r) else (None, None, None)

    # H1
    a, n, ma = frac("none", 128, 4)
    b, n2, mb = frac("block-copy", 128, 4)
    if a is not None and b is not None:
        ratio = (mb / ma) if ma and mb else float("nan")
        corner = frac("block-copy", 32, 2)
        out.append(f"**H1 (block copy dispensable early):** at 128 steps / 1/16: none {a}/{n} (median {ma:.0f}), block-copy {b}/{n2} (median {mb:.0f}), ratio {ratio:.2f} → {'supported' if 0.5 <= ratio <= 2 else 'not supported'} at the default settings. Exception: at 32 steps / 1/4 block-copy emerges in {corner[0]}/{corner[1]} seeds vs none {frac('none',32,2)[0]}/{frac('none',32,2)[1]}.")
    # H2
    a, n, ma = frac("none", 128, 4)
    b, n2, mb = frac("stack-writes", 128, 4)
    if a is not None and b is not None and mb:
        out.append(f"**H2 (stack is the early bottleneck, >10× delay):** at 128 steps / 1/16: stack-writes {b}/{n2}, median t_rep {mb:.0f} vs {ma:.0f} (×{mb/ma:.1f}) → {'supported' if mb/ma > 10 or b <= n2/2 else 'magnitude not supported'}; mechanism part: see zoo (LDIR expected).")
    # H3
    nc = L16[L16["label"] == "no-copy"]
    if len(nc):
        tot = int(nc["rep_emerged"].sum())
        out.append(f"**H3 (no-copy leaves no replicator):** {tot} heritable emergences in {int(nc['n'].sum())} no-copy runs across all budgets/mutations → {'supported' if tot == 0 else 'REFUTED — see zoo for the tapes'}.")
    # H4/H5 from none
    none = L16[L16["label"] == "none"]
    if len(none):
        piv = none.pivot_table(index="k", columns="steps", values="median_trep")
        out.append("**H4/H5 (budget strongest; mutation non-monotone), unablated median t_rep (steps × mutation):**\n\n" + piv.to_string(float_format=lambda x: f"{x:.0f}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="runs/REPORT.md")
    a = ap.parse_args()
    cells, assays, succ = load(a.dirs)
    lines = ["# Instruction-set ablation atlas — results", "", f"Batches: {', '.join(a.dirs)}. Pre-registration: experiments/PLAN.md. All emergence numbers below use the heritability assay (`t_rep`, gen2 ≥ 0.3) unless marked; the pre-registered occupancy measure (`tq_10`) is in cells.csv.", ""]
    lines.append("## Hypothesis verdicts\n")
    for v in verdicts(cells, assays):
        lines.append(v + "\n")
    for tape in sorted(cells["tape"].unique()):
        lines.append(f"## Emergence grid — L = {tape}\n")
        lines.append(cell_table(cells, tape) + "\n")
        for d in a.dirs:
            for fn in (f"emergence_rep_L{tape}.png", f"emergence_L{tape}.png"):
                p = os.path.join(d, "analysis", fn)
                if os.path.exists(p):
                    lines.append(f"![{fn}]({os.path.relpath(p, os.path.dirname(a.out) or '.')})\n")
    if len(succ):
        lines.append("## Succession (census takeover times, final family)\n")
        lines.append(succ.to_markdown(index=False, floatfmt=".0f") if hasattr(succ, "to_markdown") else succ.to_string())
        lines.append("")
    for d in a.dirs:
        z = os.path.join(d, "analysis", "ZOO.md")
        if os.path.exists(z):
            lines.append(f"## Replicator zoo — {d}\n")
            lines.append(open(z).read().split("\n", 3)[-1])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
