"""Derive the LETHAL-TAR shaders (the `zero_halts` rule) from the exported square-lattice shader and its executor.

    python -m algocell_exp.gen_lethal_shader             # shader/sim_square_L16.wgsl -> shader/sim_lethal_L16.wgsl
                                                          # shader/z80_test_L16.wgsl  -> shader/z80_test_lethal_L16.wgsl
    python -m algocell_exp.gen_lethal_shader --check      # exit 1 unless both derived files equal the derived text

The rule (PLAN.md, Stage I pre-registration, 2026-10-08): when the Z80 core fetches an opcode byte equal to 0x00 (NOP)
at the start of an instruction, the pair's execution halts for the rest of the encounter — the remaining budget is
forfeited and the two memories are written back as they are — as an unmatched bracket halts a BFF program.

Where the hunk goes. `z80_step()` fetches the first byte of every instruction with `var op = z80_fetch();` and counts
the M1 cycle (`r_inc()`). ONE statement is inserted right after that fetch, i.e. BEFORE the DD/FD prefix resolution and
BEFORE the suppression hook `on_fetch_opcode` (which turns suppressed opcodes into no-ops): if the raw fetched byte is
zero, the core's own halt flag is set and the PC is backed up onto the lethal byte (exactly what the HALT instruction
does). Both entry points already stop their instruction loop on that flag (`if (cpu_halted != 0u) { break; }` in
`z80_execute_batch` and in `z80_test`), so the pair forfeits the rest of its budget and `store_pair_mem` writes the
memories back as they are. Because the check sits on the first fetch only, it never fires on prefixed opcodes
(`DD 00`, `FD 00`: the zero is fetched as `next`; `ED 00`, `CB 00`: fetched by the page dispatch) nor on operand
bytes (fetched by `z80_fetch()` inside the instruction bodies). Everything else is byte-identical (asserted below).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SHADER_DIR = Path(__file__).parent / "shader"

# The two lines of z80_step() after which the hunk is inserted (the exported core; present once in every shader).
ANCHOR = "    var op = z80_fetch();\n    r_inc(); // M1: opcode (or prefix) fetch\n"
HUNK = (
    "    // zero_halts (lethal tar, Stage I): a zero byte fetched as the FIRST byte of an instruction halts the pair for the\n"
    "    // rest of the encounter (budget forfeited, memories written back as they are). Judged on the raw fetched byte, before\n"
    "    // the DD/FD prefix resolution and before the suppression hook, so it never fires on prefixed opcodes or operand bytes.\n"
    "    // PC backs up onto the lethal byte, as HALT does; the instruction loops of both entry points stop on cpu_halted.\n"
    "    if (op == 0u) { cpu_halted = 1u; cpu_pc = (cpu_pc - 1u) & 0xffffu; return; }\n"
)
LOOP_BREAK = "if (cpu_halted != 0u) { break; }"


def lethal_source(wgsl: str) -> str:
    """The lethal shader text derived from an exported shader text; asserts that only the hunk is added."""
    assert wgsl.count(ANCHOR) == 1, f"opcode-fetch anchor found {wgsl.count(ANCHOR)} times"
    assert HUNK not in wgsl, "the source already carries the zero_halts hunk"
    at = wgsl.index(ANCHOR) + len(ANCHOR)
    # the anchor must be the opcode fetch of z80_step(), not another fetch
    assert wgsl.rfind("\nfn ", 0, at) == wgsl.rfind("\nfn z80_step() {", 0, at), "the anchor is not inside fn z80_step()"
    # the entry point's instruction loop must stop on the core's halt flag, otherwise the hunk would not end the pair
    assert LOOP_BREAK in wgsl, "the instruction loop does not break on cpu_halted"
    # the prefix resolution and the suppression hook follow the anchor (the hunk must precede both)
    assert wgsl.index("if (op == 0xddu || op == 0xfdu) {", at) < wgsl.index("if (on_fetch_opcode(pfx, op)) { return; }", at)
    out = wgsl[:at] + HUNK + wgsl[at:]
    # everything outside the hunk is byte-identical
    assert out[:at] == wgsl[:at] and out[at + len(HUNK):] == wgsl[at:]
    assert out.count(HUNK) == 1 and len(out) - len(wgsl) == len(HUNK)
    return out


def pairs(tape: int, mem_length: int | None) -> list[tuple[Path, Path]]:
    """(source, derived) for the simulation shader and the single-pair executor of this tape and ring length."""
    suffix = "" if not mem_length or mem_length == 2 * tape else f"_P{mem_length}"
    return [
        (SHADER_DIR / f"sim_square_L{tape}{suffix}.wgsl", SHADER_DIR / f"sim_lethal_L{tape}{suffix}.wgsl"),
        (SHADER_DIR / f"z80_test_L{tape}{suffix}.wgsl", SHADER_DIR / f"z80_test_lethal_L{tape}{suffix}.wgsl"),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tape", type=int, default=16, help="tape length L of the exported shaders (default 16)")
    ap.add_argument("--mem-length", type=int, default=None, help="ring length P of a padded shader (default 2L)")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 unless both derived shaders equal the derived text")
    a = ap.parse_args(argv)
    status = 0
    for src_path, dst_path in pairs(a.tape, a.mem_length):
        if not src_path.exists():
            print(f"no exported shader {src_path.name}", file=sys.stderr)
            return 2
        out = lethal_source(src_path.read_text())
        if a.check:
            if not dst_path.exists():
                print(f"{dst_path.name} does not exist", file=sys.stderr)
                status = 1
            elif dst_path.read_text() != out:
                print(f"{dst_path.name} differs from the text derived from {src_path.name}", file=sys.stderr)
                status = 1
            else:
                print(f"{dst_path.name} is up to date ({len(out)} bytes)")
            continue
        dst_path.write_text(out)
        print(f"wrote {dst_path} ({len(out) - len(HUNK)} -> {len(out)} bytes; {HUNK.count(chr(10))}-line hunk after `{ANCHOR.strip().splitlines()[0]}`)")
    return status


if __name__ == "__main__":
    sys.exit(main())
