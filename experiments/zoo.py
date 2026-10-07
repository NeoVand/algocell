"""Replicator zoo: which replicator designs emerge where.

For every run, take the heritable replicator the assay found first (t_rep tape)
and the final heritable exemplar, disassemble them, and reduce each to a
*design signature*: the ordered list of instructions that write memory or set
up the copy (loads of pointers, pushes, block copies, exchanges, jumps),
ignoring filler. Cluster signatures per (label, tape, steps, k) and report
counts with one example tape each.

    python zoo.py runs/stageA
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

import pandas as pd

from algocell_exp.isa import disassemble

CORE_FAMILIES = {"stack", "ex", "call-ret", "rst", "block-copy", "ld8-mem", "ld16-mem", "incdec-mem", "rotate-mem", "bit-set-mem", "block-io", "ld16-imm", "ld8-imm", "ld-special", "jump", "jump-rel"}


def signature(tape_hex: str, suppress: str = "") -> str:
    tape = bytes.fromhex(tape_hex.replace(" ", ""))
    parts = []
    for ins in disassemble(tape):
        if ins["family"] in CORE_FAMILIES or ins["writesMem"]:
            m = ins["mnemonic"]
            # generalise register names so LD B,n and LD E,n cluster together
            for reg in ("BC", "DE", "HL", "SP", "AF", "IX", "IY"):
                m = m.replace(f",{reg}", ",rr").replace(f"{reg},", "rr,").replace(f" {reg}", " rr") if ins["family"] in ("stack", "ld16-imm", "ld16-mem", "ex") else m
            parts.append(m)
    # collapse immediate repeats (e.g. 8× "LD rr,nn PUSH rr")
    out, prev, rep = [], None, 0
    for p in parts:
        if p == prev:
            rep += 1
        else:
            if prev is not None:
                out.append(prev + (f"×{rep}" if rep > 1 else ""))
            prev, rep = p, 1
    if prev is not None:
        out.append(prev + (f"×{rep}" if rep > 1 else ""))
    return " ; ".join(out) if out else "(no write instructions)"


def main(d: str) -> None:
    asy = pd.read_csv(os.path.join(d, "analysis", "assays.csv"))
    rows = []
    for _, r in asy.iterrows():
        for which, tape, ok in (("first", r.get("t_rep_tape"), r.get("t_rep", -1) > 0), ("final", r.get("tape"), bool(r.get("final_replicator_insitu", False)))):
            if not ok or not isinstance(tape, str):
                continue
            rows.append({"label": r["label"], "tape_len": r["tape_len"], "steps": r["steps"], "k": r["k"], "seed": r["seed"], "which": which, "tape": tape, "signature": signature(tape)})
    df = pd.DataFrame(rows)
    out = os.path.join(d, "analysis")
    df.to_csv(os.path.join(out, "zoo.csv"), index=False)
    lines = ["# Replicator zoo", "", "Design signatures of the first heritable replicator (`first`) and the final heritable exemplar (`final`) per condition. Counts are seeds.", ""]
    for (label, L, steps, k), g in df.groupby(["label", "tape_len", "steps", "k"]):
        lines.append(f"## {label} · L={L} · {steps} steps · mutation 1/2^{k}")
        for which in ("first", "final"):
            gg = g[g["which"] == which]
            if gg.empty:
                continue
            lines.append(f"**{which}** ({len(gg)} seeds)")
            c = Counter(gg["signature"])
            for sig, n in c.most_common(6):
                ex = gg[gg["signature"] == sig].iloc[0]["tape"]
                lines.append(f"- {n}× `{ex}` — {sig}")
            lines.append("")
    with open(os.path.join(out, "ZOO.md"), "w") as f:
        f.write("\n".join(lines))
    # global design table
    glob_c = Counter(df[df["which"] == "first"]["signature"])
    print("most common first-replicator designs (all cells):")
    for sig, n in glob_c.most_common(12):
        print(f"  {n:>3}  {sig}")
    print("wrote", os.path.join(out, "ZOO.md"))


if __name__ == "__main__":
    main(sys.argv[1])
