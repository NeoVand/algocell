"""Derive the LETHALITY-DIAL shaders (REVISION_PREREG DZ, 2026-10-09): a zero byte fetched as the first byte of an
instruction halts the pair with probability p (a fresh draw for every such fetch), interpolating between the benign rule
(p = 0, the exported shaders) and the lethal-tar rule of Stage I (p = 1, gen_lethal_shader).

    python -m algocell_exp.gen_dial_shader            # writes sim_lp{tag}_L16.wgsl and z80_test_lp{tag}_L16.wgsl
    python -m algocell_exp.gen_dial_shader --check

Hunks, all inserted into the exported shaders (sim_square_L16, z80_test_L16), nothing else changed:
  1. declarations after `var<private> cpu_halted: u32;`: a private PCG state, the threshold T = floor(p * 2^32) and a flag
     for p = 1;
  2. in z80_step(), right after the opcode fetch (the same place as the Stage I hunk): if the raw fetched byte is zero and
     (p = 1 or a draw < T), halt as Stage I does (cpu_halted = 1, PC backed onto the byte);
  3. at the register reset of every encounter: the state is seeded from the call's batch seed and the pair (soup) or case
     (executor) index, so draws differ between encounters and steps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SHADER_DIR = Path(__file__).parent / "shader"
DIAL = {"lp0": 0.0, "lp001": 0.01, "lp003": 0.03, "lp01": 0.1, "lp03": 0.3, "lp1": 1.0}

DECL_ANCHOR = "var<private> cpu_halted: u32;\n"
FETCH_ANCHOR = "    var op = z80_fetch();\n    r_inc(); // M1: opcode (or prefix) fetch\n"
SOUP_ANCHOR = "    cpu_halted = 0u;\n    cpu_iff1 = 0u; cpu_iff2 = 0u;\n    cpu_writes_a = 0u;\n    cpu_writes_b = 0u;\n"
TEST_ANCHOR = "\tcpu_halted=0u; cpu_iff1=0u; cpu_iff2=0u; cpu_writes_a=0u; cpu_writes_b=0u;\n"


def _decl(p: float) -> str:
    T = min(int(p * 2 ** 32), 2 ** 32 - 1)
    return (DECL_ANCHOR + "// lethality dial (REVISION_PREREG DZ): p = " + repr(p) + "\n"
            "var<private> lethal_state: u32;\n"
            f"const LETHAL_T: u32 = {T}u;\n"
            f"const LETHAL_ALL: bool = {'true' if p >= 1.0 else 'false'};\n"
            "fn lethal_draw() -> u32 {\n    let s = lethal_state;\n    lethal_state = s * 747796405u + 2891336453u;\n"
            "    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;\n    return (w >> 22u) ^ w;\n}\n")


FETCH_HUNK = ("    // lethality dial (REVISION_PREREG DZ): a zero opcode byte halts the pair with probability p, as in Stage I\n"
              "    if (op == 0u) { if (LETHAL_ALL || lethal_draw() < LETHAL_T) { cpu_halted = 1u; cpu_pc = (cpu_pc - 1u) & 0xffffu; return; } }\n")
SOUP_HUNK = "    lethal_state = params.batch_seed * 2654435761u + pair_id * 2246822519u + 0x165667b1u; lethal_draw();\n"
TEST_HUNK = "\tlethal_state = params.batch_seed * 2654435761u + case_id * 2246822519u + 0x165667b1u; lethal_draw();\n"


def derive(wgsl: str, p: float, kind: str) -> str:
    for a in (DECL_ANCHOR, FETCH_ANCHOR):
        assert wgsl.count(a) == 1, f"anchor {a[:30]!r} found {wgsl.count(a)} times"
    out = wgsl.replace(DECL_ANCHOR, _decl(p), 1)
    at = out.index(FETCH_ANCHOR) + len(FETCH_ANCHOR)
    assert out.rfind("\nfn ", 0, at) == out.rfind("\nfn z80_step() {", 0, at), "fetch anchor not inside z80_step"
    out = out[:at] + FETCH_HUNK + out[at:]
    anchor, hunk, fn = (SOUP_ANCHOR, SOUP_HUNK, "\nfn z80_execute_batch(") if kind == "soup" else (TEST_ANCHOR, TEST_HUNK, "\nfn z80_test(")
    assert out.count(anchor) == 1, "register-reset anchor not unique"
    at = out.index(anchor) + len(anchor)
    assert out.rfind("\nfn ", 0, at) == out.rfind(fn, 0, at), "reset anchor not in the entry point"
    out = out[:at] + hunk + out[at:]
    assert len(out) - len(wgsl) == len(_decl(p)) - len(DECL_ANCHOR) + len(FETCH_HUNK) + len(hunk)
    return out


def targets():
    for tag, p in DIAL.items():
        yield SHADER_DIR / "sim_square_L16.wgsl", SHADER_DIR / f"sim_{tag}_L16.wgsl", p, "soup"
        yield SHADER_DIR / "z80_test_L16.wgsl", SHADER_DIR / f"z80_test_{tag}_L16.wgsl", p, "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    bad = 0
    for src, dst, p, kind in targets():
        text = derive(src.read_text(), p, kind)
        if a.check:
            ok = dst.exists() and dst.read_text() == text
            bad += not ok
            print(("ok  " if ok else "DIFF"), dst.name)
        else:
            dst.write_text(text)
            print("wrote", dst.name)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
