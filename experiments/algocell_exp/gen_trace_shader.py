"""Derive the TRACED single-pair executor from an exported test executor: identical execution, plus a 256-bit bitmap of
every address fetched as instruction stream (opcodes, prefixes, operands, displacements: everything `z80_fetch()` reads),
returned after the twelve register words (20 words per case instead of 12).

    python -m algocell_exp.gen_trace_shader --source shader/z80_test_L16.wgsl [--out shader/z80_test_trace_L16.wgsl]

Used by algocell_exp.exectrace for pointer confinement (any fetch at an address >= L) and the copied-but-unexecuted
fraction U (NATURE_PLAN Move 2a). Four hunks, each anchored on a line that occurs exactly once; everything else is
byte-identical to the source.
"""
from __future__ import annotations

import argparse
from pathlib import Path

SHADER_DIR = Path(__file__).parent / "shader"

HUNKS = [
    ("var<private> cpu_halted: u32;", "var<private> cpu_halted: u32;\nvar<private> exec_mask: array<u32, 8>;   // traced: addresses fetched as instruction stream (mod MEM_LENGTH)"),
    ("    let val = mem_read(cpu_pc);\n    cpu_pc = (cpu_pc + 1u) & 0xffffu;\n    return val;",
     "    let val = mem_read(cpu_pc);\n    let ea = cpu_pc % MEM_LENGTH;\n    exec_mask[ea >> 5u] |= (1u << (ea & 31u));\n    cpu_pc = (cpu_pc + 1u) & 0xffffu;\n    return val;"),
    ("cpu_halted=0u; cpu_iff1=0u; cpu_iff2=0u; cpu_writes_a=0u; cpu_writes_b=0u;",
     "cpu_halted=0u; cpu_iff1=0u; cpu_iff2=0u; cpu_writes_a=0u; cpu_writes_b=0u;\n\tfor (var i = 0u; i < 8u; i++) { exec_mask[i] = 0u; }"),
    ("\tlet rbase = case_id * 12u;", "\tlet rbase = case_id * 20u;"),
    ("regs[rbase+8u]=cpu_sp; regs[rbase+9u]=cpu_pc; regs[rbase+10u]=cpu_writes_a; regs[rbase+11u]=cpu_writes_b;",
     "regs[rbase+8u]=cpu_sp; regs[rbase+9u]=cpu_pc; regs[rbase+10u]=cpu_writes_a; regs[rbase+11u]=cpu_writes_b;\n\tfor (var i = 0u; i < 8u; i++) { regs[rbase+12u+i] = exec_mask[i]; }"),
]


def derive(source: Path) -> str:
    text = source.read_text()
    for old, new in HUNKS:
        assert text.count(old) == 1, f"anchor not unique in {source.name}: {old[:50]!r} ({text.count(old)})"
        text = text.replace(old, new)
    return text


def traced_path(source: Path) -> Path:
    return source.with_name(source.name.replace("z80_test", "z80_test_trace", 1))


def ensure(source: Path) -> Path:
    out = traced_path(source)
    text = derive(source)
    if not out.exists() or out.read_text() != text:
        out.write_text(text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    a = ap.parse_args()
    print(ensure(Path(a.source)))


if __name__ == "__main__":
    main()
