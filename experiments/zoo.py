"""Replicator zoo: which replicator designs emerge where → analysis/zoo.csv, ZOO.md.

For every run, take the first heritable replicator (`trep_tape`) and the final tape when it
is a faithful replicator against random partners (`final_tape`, `final_faithful`), and reduce
each to a design signature under the condition's own suppression set (suppressed opcodes
are NOPs there and are dropped). Tiled tapes (tolerant period p ≤ L/2) are reduced to ONE
period, rotated to a canonical phase, followed by "×n", so `LD rr,nn ; PUSH rr` repeated 25
times and the same design starting mid-unit are one signature (review 2026-10-07).

    python zoo.py runs/stageA
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

from algocell_exp.isa import disassemble, resolve
from algocell_exp.metrics import tolerant_period
from make_conds import ABLATIONS, ablation_of

CORE_FAMILIES = {"stack", "ex", "call-ret", "rst", "block-copy", "ld8-mem", "ld16-mem", "incdec-mem", "rotate-mem", "bit-set-mem", "block-io",
                 "ld16-imm", "ld8-imm", "ld-special", "jump", "jump-rel"}
REGS16 = ("BC", "DE", "HL", "SP", "AF", "IX", "IY")
REGS8 = ("A", "B", "C", "D", "E", "H", "L")


def _generalise(m: str, family: str) -> str:
    if family in ("stack", "ld16-imm", "ld16-mem", "ex"):
        for reg in REGS16:
            m = m.replace(f",{reg}", ",rr").replace(f"{reg},", "rr,").replace(f" {reg}", " rr")
    elif family in ("ld8-imm", "ld8", "incdec", "ld8-mem"):
        for reg in REGS8:
            m = m.replace(f" {reg},", " r,").replace(f",{reg}", ",r")
    return m


def _parts(tape: bytes, sets) -> list[str]:
    parts = []
    for ins in disassemble(tape, suppress=sets):
        if ins["family"] == "suppressed":
            continue
        if ins["family"] in CORE_FAMILIES or ins["writesMem"]:
            parts.append(_generalise(ins["mnemonic"], ins["family"]))
    return parts


def signature(tape_hex: str, label: str = "none") -> str:
    tape = bytes.fromhex(tape_hex.replace(" ", ""))
    sets = resolve(ABLATIONS[ablation_of(label)])
    L = len(tape)
    p, match = tolerant_period(tape)
    if p <= L // 2 and match >= 0.9:
        # one period, canonical phase = the rotation whose signature sorts first
        # One byte-period may hold a fraction of an instruction period (`01 c5` is 2 bytes; its code
        # `LD BC,nn ; PUSH BC` is 4). Disassemble enough repeats to see whole instructions, keep every
        # instruction of one INSTRUCTION period (pointer set-up such as DEC E matters for a tiled design),
        # and take the rotation whose signature sorts first as the canonical phase.
        arr = np.frombuffer(tape, dtype=np.uint8)
        unit = bytes(Counter(arr[i::p].tolist()).most_common(1)[0][0] for i in range(p))   # per-position majority vote: one mutated byte does not relabel the design
        reps = max(3, (12 // p) + 2)
        cands = []
        for r in range(p):
            rot = unit[r:] + unit[:r]
            parts = [_generalise(i["mnemonic"], i["family"]) for i in disassemble(rot * reps, suppress=sets) if i["family"] != "suppressed"]
            parts = parts[:-1] if len(parts) > 1 else parts     # the last one may be truncated
            q = next((q for q in range(1, len(parts)) if all(parts[i] == parts[i + q] for i in range(len(parts) - q))), len(parts))
            cands.append(" ; ".join(parts[:q]) if parts else "(no instructions)")
        core = min(cands, key=lambda c: (len(c), c))
        return f"[{core}] ×{L // p}" + (f"+{L % p}B" if L % p else "")
    parts = _parts(tape, sets)
    return " ; ".join(_collapse(parts)) if parts else "(no write instructions)"


def _collapse(parts: list[str]) -> list[str]:
    """Collapse immediate repeats of units up to 4 instructions long: a b a b a b → (a ; b)×3."""
    out: list[str] = []
    i = 0
    n = len(parts)
    while i < n:
        best = None
        for u in (1, 2, 3, 4):
            if i + u > n:
                break
            unit = parts[i : i + u]
            reps = 1
            while parts[i + reps * u : i + (reps + 1) * u] == unit:
                reps += 1
            if reps > 1 and (best is None or reps * u > best[0] * best[1]):
                best = (reps, u)
        if best:
            reps, u = best
            unit = " ; ".join(parts[i : i + u])
            out.append((f"({unit})" if u > 1 else unit) + f"×{reps}")
            i += reps * u
        else:
            out.append(parts[i])
            i += 1
    return out


def main(d: str) -> None:
    asy = pd.read_csv(os.path.join(d, "analysis", "assays.csv"))
    rows = []
    for _, r in asy.iterrows():
        cands = [("first", r.get("trep_tape"), pd.notna(r.get("t_rep")) and r.get("t_rep", -1) > 0),
                 ("final", r.get("final_tape"), bool(r.get("final_faithful", False)))]
        for which, tape, ok in cands:
            if not ok or not isinstance(tape, str):
                continue
            rows.append({"label": r["label"], "tape_len": r["tape_len"], "steps": r["steps"], "k": r["k"], "seed": r["seed"], "which": which,
                         "tape": tape, "signature": signature(tape, r["label"])})
    df = pd.DataFrame(rows)
    out = os.path.join(d, "analysis")
    df.to_csv(os.path.join(out, "zoo.csv"), index=False)
    lines = ["# Replicator zoo", "",
             "Design signatures of the first heritable replicator (`first`, the t_rep tape) and of the final tape when it is a "
             "faithful replicator against random partners (`final`). Tiled tapes are shown as one period ×n. Counts are seeds.", ""]
    for (label, L, steps, k), g in df.groupby(["label", "tape_len", "steps", "k"]):
        lines.append(f"## {label} · L={L} · {steps} steps · mutation 1/2^{k}")
        for which in ("first", "final"):
            gg = g[g["which"] == which]
            if gg.empty:
                continue
            lines.append(f"**{which}** ({len(gg)} seeds)")
            for sig, n in Counter(gg["signature"]).most_common(6):
                ex = gg[gg["signature"] == sig].iloc[0]["tape"]
                lines.append(f"- {n}× `{ex[:60]}{'…' if len(ex) > 60 else ''}` — {sig}")
            lines.append("")
    with open(os.path.join(out, "ZOO.md"), "w") as f:
        f.write("\n".join(lines))
    print("most common first-replicator designs (all cells):")
    for sig, n in Counter(df[df["which"] == "first"]["signature"]).most_common(12):
        print(f"  {n:>3}  {sig}")
    print("wrote", os.path.join(out, "ZOO.md"))


if __name__ == "__main__":
    main(sys.argv[1])
