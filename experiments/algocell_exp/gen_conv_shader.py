"""Derive the CONVENTION-TEST shaders (round-2 review, REVISION_PREREG.md R3/C) from the exported soup shader and the
single-pair executor: byte-identical except for the register state at the start of every encounter.

    python -m algocell_exp.gen_conv_shader [--tapes 16,20,32,50,64]   # writes sim_{v}_L*.wgsl and z80_test_{v}_L*.wgsl
    python -m algocell_exp.gen_conv_shader --check

Variants: `randreg` draws A, F, B, C, D, E, H, L, their alternates (8 bits each), IX and IY (16 bits) uniformly at
random, keeping PC = 0 and the stack pointer at its usual start; `randsp` draws SP uniformly from 0–65,535 and keeps every
other register zero. Soup shader: the draw uses the shader's own PCG generator, seeded by the step's batch seed and the
pair index (a fresh draw for every encounter). Executor: a stateless hash of the call seed (params.batch_seed) and the
case index (a fresh draw for every case of every call).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SHADER_DIR = Path(__file__).parent / "shader"
VARIANTS = ("randreg", "randsp")

SOUP_ANCHOR = "    cpu_halted = 0u;\n    cpu_iff1 = 0u; cpu_iff2 = 0u;\n    cpu_writes_a = 0u;\n    cpu_writes_b = 0u;\n"
TEST_ANCHOR = "\tcpu_halted=0u; cpu_iff1=0u; cpu_iff2=0u; cpu_writes_a=0u; cpu_writes_b=0u;\n"
TEST_DECL_ANCHOR = "var<private> cpu_halted: u32;\n"

REG8 = ["cpu_a", "cpu_f", "cpu_b", "cpu_c", "cpu_d", "cpu_e", "cpu_h", "cpu_l", "cpu_a2", "cpu_f2", "cpu_b2", "cpu_c2", "cpu_d2", "cpu_e2", "cpu_h2", "cpu_l2"]


def _draws(variant: str, draw: str, indent: str) -> str:
    if variant == "randreg":
        lines = [f"{r} = {draw} & 0xffu;" for r in REG8] + [f"cpu_ix = {draw} & 0xffffu;", f"cpu_iy = {draw} & 0xffffu;"]
    elif variant == "randsp":
        lines = [f"cpu_sp = {draw} & 0xffffu;"]
    else:
        raise ValueError(variant)
    return "".join(indent + l + "\n" for l in lines)


def soup_source(wgsl: str, variant: str) -> str:
    assert wgsl.count(SOUP_ANCHOR) == 1, "soup register-reset anchor not unique"
    at = wgsl.index(SOUP_ANCHOR) + len(SOUP_ANCHOR)
    assert wgsl.rfind("\nfn ", 0, at) == wgsl.rfind("\nfn z80_execute_batch(", 0, at), "anchor not inside z80_execute_batch"
    hunk = (f"    // convention test ({variant}; REVISION_PREREG R3/C): a fresh random draw for every encounter\n"
            "    rng = params.batch_seed * 2246822519u + pair_id * 3266489917u + 0x27d4eb2fu;\n    rand();\n" + _draws(variant, "rand()", "    "))
    out = wgsl[:at] + hunk + wgsl[at:]
    assert out[:at] == wgsl[:at] and out[at + len(hunk):] == wgsl[at:]
    return out


def test_source(wgsl: str, variant: str) -> str:
    assert wgsl.count(TEST_ANCHOR) == 1 and wgsl.count(TEST_DECL_ANCHOR) == 1, "executor anchors not unique"
    decl = (TEST_DECL_ANCHOR + "var<private> conv_state: u32;\n"
            "fn conv_draw() -> u32 {   // convention test: PCG step on a private state seeded per case\n"
            "    let s = conv_state;\n    conv_state = s * 747796405u + 2891336453u;\n"
            "    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;\n    return (w >> 22u) ^ w;\n}\n")
    out = wgsl.replace(TEST_DECL_ANCHOR, decl, 1)
    hunk = (f"\t// convention test ({variant}; REVISION_PREREG R3/C): a fresh random draw for every case of every call\n"
            "\tconv_state = params.batch_seed * 2246822519u + case_id * 3266489917u + 0x27d4eb2fu;\n\tconv_draw();\n" + _draws(variant, "conv_draw()", "\t"))
    at = out.index(TEST_ANCHOR) + len(TEST_ANCHOR)
    assert out.rfind("\nfn ", 0, at) == out.rfind("\nfn z80_test(", 0, at), "anchor not inside z80_test"
    out = out[:at] + hunk + out[at:]
    return out


def targets(tapes):
    for L in tapes:
        for v in VARIANTS:
            s = SHADER_DIR / f"sim_square_L{L}.wgsl"
            if s.exists():
                yield s, SHADER_DIR / f"sim_{v}_L{L}.wgsl", v, soup_source
            t = SHADER_DIR / f"z80_test_L{L}.wgsl"
            if t.exists():
                yield t, SHADER_DIR / f"z80_test_{v}_L{L}.wgsl", v, test_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", default="16,20,32,50,64")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    bad = 0
    for src, dst, v, fn in targets([int(x) for x in a.tapes.split(",")]):
        text = fn(src.read_text(), v)
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
