"""Derive the WELL-MIXED simulation shader from the exported square-lattice shader.

    python -m algocell_exp.gen_mixed_shader            # shader/sim_square_L16.wgsl -> shader/sim_mixed_L16.wgsl
    python -m algocell_exp.gen_mixed_shader --tape 16 --check   # verify the existing mixed shader is up to date

The square shader's `prepare_batch` draws a random cell i and pairs it with one of its four
lattice neighbours (edges reflected). The mixed variant replaces exactly that neighbour-selection
block (from the comment `// --- Topology-specific neighbor selection ---` through
`let j = ny * w + nx;`) with a partner drawn uniformly from the whole soup:

    let j = rand_bounded(w * h);

Nothing else changes: j == i stays inactive and the collision claim, execution, absorption and
mutation are byte-identical to the square shader (asserted below). Used by Stage H, the
well-mixed control of Stage G (PLAN.md, 2026-10-08).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SHADER_DIR = Path(__file__).parent / "shader"

START = "    // --- Topology-specific neighbor selection ---\n"
END = "    let j = ny * w + nx;\n"
REPLACEMENT = "    let j = rand_bounded(w * h);\n"

# The exact block of the exported square shader that is replaced (the script refuses to run on anything else).
SQUARE_BLOCK = (
    START
    + "\n"
    + "    let dir = rand_bounded(4u);\n"
    + "    var nx = x;\n"
    + "    var ny = y;\n"
    + "    switch(dir) {\n"
    + "        case 0u: { if (x + 1u < w) { nx = x + 1u; } else { nx = x - 1u; } }  // right, reflect at edge\n"
    + "        case 1u: { if (y + 1u < h) { ny = y + 1u; } else { ny = y - 1u; } }  // down, reflect at edge\n"
    + "        case 2u: { if (x > 0u) { nx = x - 1u; } else { nx = x + 1u; } }       // left, reflect at edge\n"
    + "        case 3u: { if (y > 0u) { ny = y - 1u; } else { ny = y + 1u; } }       // up, reflect at edge\n"
    + "        default: {}\n"
    + "    }\n"
    + END
)


def mixed_source(square_wgsl: str) -> str:
    """The mixed shader text derived from the square shader text; asserts that only the block changed."""
    assert square_wgsl.count(START) == 1, f"neighbour-selection marker found {square_wgsl.count(START)} times"
    assert square_wgsl.count(END) == 1, f"`let j = ny * w + nx;` found {square_wgsl.count(END)} times"
    start = square_wgsl.index(START)
    end = square_wgsl.index(END, start) + len(END)
    block = square_wgsl[start:end]
    assert block == SQUARE_BLOCK, "the neighbour-selection block of the exported shader is not the one this script knows"
    # i must have been drawn just before the block, exactly as the square shader does it
    assert square_wgsl[:start].endswith("    let x = rand_bounded(w);\n    let y = rand_bounded(h);\n    let i = y * w + x;\n\n"), "unexpected code before the block"
    out = square_wgsl[:start] + REPLACEMENT + square_wgsl[end:]
    # everything outside the block is byte-identical
    assert out[:start] == square_wgsl[:start]
    assert out[start + len(REPLACEMENT):] == square_wgsl[end:]
    assert out.count(REPLACEMENT) == 1 and START not in out and END not in out
    assert len(square_wgsl) - len(out) == len(SQUARE_BLOCK) - len(REPLACEMENT)
    return out


def paths(tape: int, mem_length: int | None) -> tuple[Path, Path]:
    suffix = "" if not mem_length or mem_length == 2 * tape else f"_P{mem_length}"
    return SHADER_DIR / f"sim_square_L{tape}{suffix}.wgsl", SHADER_DIR / f"sim_mixed_L{tape}{suffix}.wgsl"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tape", type=int, default=16, help="tape length L of the exported square shader (default 16)")
    ap.add_argument("--mem-length", type=int, default=None, help="ring length P of a padded shader (default 2L)")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 unless the existing mixed shader equals the derived text")
    a = ap.parse_args(argv)
    src_path, dst_path = paths(a.tape, a.mem_length)
    if not src_path.exists():
        print(f"no exported square shader {src_path.name}", file=sys.stderr)
        return 2
    src = src_path.read_text()
    out = mixed_source(src)
    if a.check:
        if not dst_path.exists():
            print(f"{dst_path.name} does not exist", file=sys.stderr)
            return 1
        if dst_path.read_text() != out:
            print(f"{dst_path.name} differs from the text derived from {src_path.name}", file=sys.stderr)
            return 1
        print(f"{dst_path.name} is up to date ({len(out)} bytes)")
        return 0
    dst_path.write_text(out)
    print(f"wrote {dst_path} ({len(src)} -> {len(out)} bytes; block of {SQUARE_BLOCK.count(chr(10))} lines replaced by `{REPLACEMENT.strip()}`)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
