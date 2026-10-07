"""Results report (Markdown) from one or more analysed batches.

    python report.py runs/stageA [runs/stageB ...] --out runs/REPORT.md

Two kinds of statement are kept apart (review 2026-10-07): the PRE-REGISTERED tests of
H1–H6 as written in PLAN.md, each with its outcome and the test actually applied, and the
POST HOC observations, which carry the forking-paths count and point at the confirmatory
run rather than a verdict. Every cell shows the occupancy event (tq_10), the heritable
(t_rep) and the faithful (t_faith) replicator counts with KM medians ("NR" = not reached).
"""

from __future__ import annotations

import argparse
import os
from math import comb

import numpy as np
import pandas as pd


def fisher(a: int, b: int, c: int, d: int, alternative: str = "two-sided") -> float:
    """Exact Fisher test on [[a, b], [c, d]] (a = events in arm 1, c = events in arm 2)."""
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def p(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    p0 = p(a)
    if alternative == "two-sided":
        return sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 + 1e-12)
    if alternative == "greater":      # arm 1 has MORE events
        return sum(p(x) for x in range(a, hi + 1))
    return sum(p(x) for x in range(lo, a + 1))


def sign_test(pos: int, neg: int) -> float:
    """Two-sided exact sign test ignoring ties."""
    n = pos + neg
    if n == 0:
        return float("nan")
    k = min(pos, neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2**n)


def load(dirs: list[str]):
    cells, assays, succ = [], [], []
    for d in dirs:
        a = os.path.join(d, "analysis")
        cells.append(pd.read_csv(os.path.join(a, "cells.csv")))
        assays.append(pd.read_csv(os.path.join(a, "assays.csv")))
        if os.path.exists(os.path.join(a, "succession_cells.csv")):
            succ.append(pd.read_csv(os.path.join(a, "succession_cells.csv")))
    return pd.concat(cells, ignore_index=True), pd.concat(assays, ignore_index=True), (pd.concat(succ, ignore_index=True) if succ else pd.DataFrame())


def km_str(x) -> str:
    return "NR" if (pd.isna(x) or x == np.inf) else f"{x:,.0f}"


def fmt_cell(r) -> str:
    return f"{int(r['tq_10_n'])}/{int(r['n'])} · **{int(r['t_rep_n'])}/{int(r['n'])}** ({km_str(r['t_rep_km_median'])}) · {int(r['t_faith_n'])}/{int(r['n'])}"


def cell_table(cells: pd.DataFrame, tape: int) -> str:
    sub = cells[cells["tape"] == tape]
    labels = list(dict.fromkeys(sub["label"]))
    steps = sorted(sub["steps"].unique())
    ks = sorted(sub["k"].unique())
    head = "| ablation | " + " | ".join(f"{s} steps · 1/2^{k}" for s in steps for k in ks) + " |"
    lines = [head, "|---|" + "---|" * (len(steps) * len(ks))]
    for lab in labels:
        row = [lab]
        for s in steps:
            for k in ks:
                r = sub[(sub["label"] == lab) & (sub["steps"] == s) & (sub["k"] == k)]
                row.append(fmt_cell(r.iloc[0]) if len(r) else "–")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Each cell: seeds reaching q_share ≥ 10% (pre-registered) · **seeds with a heritable replicator** (KM median step, NR = not reached within the horizon) · seeds with a faithful replicator.")
    return "\n".join(lines)


def cell(cells, label, tape, steps, k):
    r = cells[(cells["label"] == label) & (cells["tape"] == tape) & (cells["steps"] == steps) & (cells["k"] == k)]
    return r.iloc[0] if len(r) else None


def paired_sign(assays: pd.DataFrame, label: str, tape: int, k: int, a_steps: int, b_steps: int) -> tuple[int, int, int]:
    """Seed-paired comparison of t_rep between two budgets: (#a faster, #b faster, ties); a censored run is slower than any emerged one."""
    A = assays[(assays["label"] == label) & (assays["tape_len"] == tape) & (assays["k"] == k) & (assays["steps"] == a_steps)].set_index("seed")["t_rep"]
    B = assays[(assays["label"] == label) & (assays["tape_len"] == tape) & (assays["k"] == k) & (assays["steps"] == b_steps)].set_index("seed")["t_rep"]
    pos = neg = tie = 0
    for s in A.index.intersection(B.index):
        ta, tb = A[s], B[s]
        ta = np.inf if (pd.isna(ta) or ta < 0) else ta
        tb = np.inf if (pd.isna(tb) or tb < 0) else tb
        if ta < tb:
            pos += 1
        elif tb < ta:
            neg += 1
        else:
            tie += 1
    return pos, neg, tie


def preregistered(cells: pd.DataFrame, assays: pd.DataFrame) -> list[str]:
    out = []
    L = 16
    n_, bc, sw, nc = (cell(cells, l, L, 128, 4) for l in ("none", "block-copy", "stack-writes", "no-copy"))
    if n_ is not None and bc is not None:
        p = fisher(int(bc["t_rep_n"]), int(bc["n"] - bc["t_rep_n"]), int(n_["t_rep_n"]), int(n_["n"] - n_["t_rep_n"]))
        corner_bc, corner_n = cell(cells, "block-copy", L, 32, 2), cell(cells, "none", L, 32, 2)
        corner = ""
        if corner_bc is not None and corner_n is not None:
            pc = fisher(int(corner_bc["t_rep_n"]), int(corner_bc["n"] - corner_bc["t_rep_n"]), int(corner_n["t_rep_n"]), int(corner_n["n"] - corner_n["t_rep_n"]))
            corner = f" At 32 steps · 1/4 (one of 9 cells, exploratory): block-copy {int(corner_bc['t_rep_n'])}/{int(corner_bc['n'])} vs none {int(corner_n['t_rep_n'])}/{int(corner_n['n'])}, Fisher two-sided p = {pc:.2g}."
        out.append(f"**H1 — block copy is dispensable early (L = 16, 128 steps, 1/16).** none {int(n_['t_rep_n'])}/{int(n_['n'])} (KM median {km_str(n_['t_rep_km_median'])}), block-copy {int(bc['t_rep_n'])}/{int(bc['n'])} (KM median {km_str(bc['t_rep_km_median'])}); Fisher two-sided p = {p:.2g}. "
                   f"The pre-registered acceptance region (median ratio within [0.5, 2]) cannot be resolved: both medians sit at the first or second 500-step sample, so the test is only 'no difference detectable at 500-step resolution'.{corner}")
    if n_ is not None and sw is not None:
        ratio = sw["t_rep_km_median"] / n_["t_rep_km_median"] if np.isfinite(sw["t_rep_km_median"]) and n_["t_rep_km_median"] > 0 else np.nan
        mech = str(sw.get("trep_mechanisms", ""))
        out.append(f"**H2 — removing the stack-writing arm delays emergence > 10× and switches the mechanism (L = 16, 128 steps, 1/16).** stack-writes {int(sw['t_rep_n'])}/{int(sw['n'])}, KM median {km_str(sw['t_rep_km_median'])} vs none {km_str(n_['t_rep_km_median'])} (ratio {ratio:.0f}×); first-replicator mechanisms under the ablation: {mech or '–'}. "
                   f"Caveat (review): the arm removes 46 opcodes of which 24 write nothing (POP, RET, EX DE,HL, EXX), so the delay is 'stack+exchange+return removed', not 'stack writes removed'; Stage D separates the two.")
    nc_all = cells[cells["label"] == "no-copy"]
    rmw_all = cells[cells["label"] == "rmw-only"]
    if len(nc_all):
        tq = int(nc_all["tq_10_n"].sum())
        letter = (f"By the pre-registered occupancy event tq_10 the letter of H3 is REFUTED: {tq} no-copy runs crossed q_share ≥ 10%, all on zero-byte floods, not replicators." if tq
                  else "By the pre-registered occupancy event tq_10 as well: 0 crossings.")
        out.append(f"**H3 — no-copy leaves no replicator.** Heritable replicators (t_rep): {int(nc_all['t_rep_n'].sum())} in {int(nc_all['n'].sum())} no-copy runs"
                   + (f"; rmw-only: {int(rmw_all['t_rep_n'].sum())} in {int(rmw_all['n'].sum())}" if len(rmw_all) else "")
                   + f". {letter} The assay outcome (t_rep) was adopted after 19 runs were read (PLAN change log) and is the measure used here.")
    none16 = cells[(cells["label"] == "none") & (cells["tape"] == 16)]
    if len(none16) > 3:
        rows = []
        for k in sorted(none16["k"].unique()):
            pos, neg, tie = paired_sign(assays, "none", 16, k, 128, 32)
            rows.append(f"1/2^{k}: 128 steps faster than 32 steps in {pos}, slower in {neg}, tied in {tie} of {pos+neg+tie} seeds (sign test p = {sign_test(pos, neg):.2g})")
        piv = none16.pivot_table(index="k", columns="steps", values="t_rep_km_median").map(km_str)
        out.append("**H4 — the step budget is the strongest knob (unablated, L = 16).** KM median t_rep (steps × mutation):\n\n" + piv.to_markdown() + "\n\nSeed-paired budget contrast: " + "; ".join(rows) + ".")
        rows = []
        for st in sorted(none16["steps"].unique()):
            A = assays[(assays["label"] == "none") & (assays["tape_len"] == 16) & (assays["steps"] == st)]
            def t(k):
                return A[A["k"] == k].set_index("seed")["t_rep"].replace(-1, np.inf)
            pairs = []
            for k1, k2 in ((2, 4), (4, 6)):
                if k1 in set(A["k"]) and k2 in set(A["k"]):
                    x, y = t(k1), t(k2)
                    idx = x.index.intersection(y.index)
                    pos = int((x[idx] < y[idx]).sum()); neg = int((y[idx] < x[idx]).sum()); tie = len(idx) - pos - neg
                    pairs.append(f"1/2^{k1} faster than 1/2^{k2} in {pos}, slower in {neg}, tied in {tie} of {len(idx)} seeds (sign test p = {sign_test(pos, neg):.2g})")
            rows.append(f"{st} steps: " + "; ".join(pairs))
        out.append("**H5 — mutation is non-monotone (unablated, L = 16).** Seed-paired contrasts between adjacent mutation rates: " + ". ".join(rows) + ". The 'too low' side (k ≥ 8) was not run; a minimum cannot be claimed from k ∈ {2, 4, 6}.")
    sizes = cells[(cells["label"] == "none") & (cells["k"] == 4) & (cells["steps"] == 128)].sort_values("tape")
    if len(sizes) > 3:
        s = ", ".join(f"L={int(r['tape'])}: {int(r['t_rep_n'])}/{int(r['n'])} ({km_str(r['t_rep_km_median'])}; faithful {int(r['t_faith_n'])})" for _, r in sizes.iterrows())
        out.append(f"**H6 — emergence time grows with L; large organisms have more free tape; more mechanism classes at large L (unablated, 128 steps, 1/16).** {s}. "
                   "Time to emergence does not grow with L above 16 at 500-step resolution (first clause not supported); the free-tape and complexity clauses are addressed in the size-axis analysis (tiling), with the confounds listed in REVIEW.md §4 (per-byte mutation ∝ 1/L, steps per byte, parity-driven early stop) still open until the control arms run.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="runs/REPORT.md")
    a = ap.parse_args()
    cells, assays, succ = load(a.dirs)
    lines = ["# Instruction-set ablation atlas — results", "",
             f"Batches: {', '.join(a.dirs)}. Pre-registration and change log: `experiments/PLAN.md`; review: `experiments/REVIEW.md`. "
             "Emergence events: `tq_10` = quasispecies occupancy ≥ 10% (pre-registered primary; fires on zero-byte floods as well as replicators); "
             "`t_rep` = first top-3 exemplar with share ≥ 0.5% that is heritable (assay gen2 ≥ 0.3); `t_faith` = additionally ≥ 50% of partners became ≥ 75% copies. "
             "Times are Kaplan–Meier medians censored at each run's last step; sampling is every 500 steps, so 500 is the resolution floor.", ""]
    lines.append("## Pre-registered hypotheses\n")
    for v in preregistered(cells, assays):
        lines.append(v + "\n")
    lines.append("## Emergence grids\n")
    for tape in sorted(cells["tape"].unique()):
        lines.append(f"### L = {tape}\n")
        lines.append(cell_table(cells, tape) + "\n")
        for d in a.dirs:
            for fn in (f"atlas_L{tape}.png", f"km_L{tape}.png"):
                p = os.path.join(d, "analysis", fn)
                if os.path.exists(p):
                    lines.append(f"![{fn}]({os.path.relpath(p, os.path.dirname(a.out) or '.')})\n")
    if len(succ):
        lines.append("## Succession (census takeover times, family at 300k steps and at the last step)\n")
        keep = [c for c in succ.columns if c not in ("stack_plateau_med",)]
        lines.append(succ[keep].to_markdown(index=False, floatfmt=".0f"))
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
