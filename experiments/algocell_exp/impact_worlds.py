"""The worlds of the emulator impact study (planning/IMPACT_STUDY_DRAFT.md) and their three cores.

Every world below is vendored three times, in `machines/worlds_impact.json`, with one world spec shared verbatim by
the three machines (same world SHA-256, same hooks); only `core` differs:

    C0  imp_<world>_c0   Zilion Z80, compat '0.2' (the 0.2.0 core every registered result ran on)
    C1  imp_<world>_c1   Zilion Z80, compat '0.3', variant classic (the fixed core)
    C2  imp_<world>_c2   Zilion Z80, compat '0.3', variant zilog-nmos (C1 plus the Zilog NMOS Q latch)

World tags:
    L<L>            Algocell's pair: tapes A, B of L bytes on a ring of P = 2L (Stages A–M, line-of-descent runs, ...)
    L<L>_P<P>       the same pair on a padded or odd ring of P > 2L bytes (Stage F, ring sweep, corpus-1 rings)
    lethal_L<L>     plus the lethal-tar rule (Stage I, E, E-L32L): a zero byte fetched as an instruction's first byte halts
    lp<tag>_L16     the lethality dial (DZ): that halt happens with probability p, Algocell's PCG stream (raw hooks)
    randreg_L<L>    random registers at every encounter (C, C4a, C4a-rep, C4b): Algocell's PCG stream (raw hooks)
    randsp_L<L>     a random stack pointer at every encounter (C)

Common to all: `init.stack: { topAliases: 'lastByte' }` (SP aliases the last byte of B, Algocell's convention), a
per-lane step budget (0 = the lane takes no step), the uniform ablation table in the Z80 `coarse` layout (Algocell's
params.suppress word for word), write counts per segment. The raw hooks are the text of zilion's
`test/world/algocell.mjs` (dialHooks, convHooks), which reproduce `gen_dial_shader.py` and `gen_conv_shader.py`.

    python -m algocell_exp.impact_worlds            # write machines/worlds_impact.json
    python -m algocell_exp.impact_worlds --check    # exit 1 unless the file on disk is what this module writes
then `node experiments/algocell_exp/machines/build_machines.mjs` generates the machines.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .gen_dial_shader import DIAL

MACHINES_DIR = Path(__file__).parent / "machines"
SPEC_FILE = MACHINES_DIR / "worlds_impact.json"
SHADER_DIR = Path(__file__).parent / "shader"

CORES = {
    "c0": {"isa": "z80", "compat": "0.2"},
    "c1": {"isa": "z80", "compat": "0.3", "variant": "classic"},
    "c2": {"isa": "z80", "compat": "0.3", "variant": "zilog-nmos"},
}
CORE_NOTES = {
    "c0": "C0: the Zilion 0.2.0 Z80 (compat '0.2'), the core every registered result ran on",
    "c1": "C1: the fixed Zilion Z80 (compat '0.3', variant classic)",
    "c2": "C2: the fixed Zilion Z80 with the Zilog NMOS Q latch (compat '0.3', variant zilog-nmos)",
}

# P = 2L pair worlds: every tape length a registered condition ran at (Stages A–E, G, K, L, M; lod; offset), plus L = 19,
# the P = 38 ring of the draft's corpus 1 (the ring difftest's size; no soup ran at L = 19).
PLAIN_L = (3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 18, 19, 20, 24, 25, 32, 36, 49, 50, 64, 81, 100)
# padded and odd rings (Stage F, every (L, P) it ran; the draft's corpus 1 uses P = 33, 35, 37 at L = 16, 73, 75, 79 at 36)
STAGE_F_PADDED = ((8, 32), (10, 28), (10, 40), (12, 28), (12, 36), (16, 33), (16, 34), (16, 35), (16, 37), (36, 73), (36, 74), (36, 75), (36, 79))
# The exploratory ring sweep supports Extended Data Fig. 3. Preserve its actual exported/archived rings: there are
# gaps in the L = 8 and 10 grids, so a blanket 2L..2L+24 range would invent unrun worlds.
RING_SWEEP_RINGS = {
    8: (16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40),
    10: (20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 44),
    12: tuple(range(24, 49, 2)),
    16: tuple(sorted((*range(32, 57, 2), 33, 35, 37))),
}
PADDED = STAGE_F_PADDED + tuple(sorted({(L, P) for L, Ps in RING_SWEEP_RINGS.items() for P in Ps if P > 2 * L}
                                      - set(STAGE_F_PADDED)))
LETHAL_L = (16, 32)                 # Stage I, offset (E), E-L32L
DIAL_L = 16                         # DZ: every p at L = 16
CONV_L = (16, 20, 32, 50, 64)       # C (soups at L = 16; its partner tests of Stage G and K tapes at 16, 20, 32, 50, 64)

# Algocell's PCG step, as zilion test/world/algocell.mjs writes it (gen_dial_shader.py, gen_conv_shader.py)
def _pcg(state: str, fn: str) -> str:
    return (f"var<private> {state}: u32;\nfn {fn}() -> u32 {{\n    let s = {state};\n    {state} = s * 747796405u + 2891336453u;\n"
            "    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;\n    return (w >> 22u) ^ w;\n}")


def dial_threshold(p: float) -> int:
    """gen_dial_shader's LETHAL_T = min(int(p · 2³²), 2³² − 1)."""
    return min(int(p * 2 ** 32), 2 ** 32 - 1)


def dial_hooks(p: float) -> dict:
    cond = "true" if p >= 1 else f"lethal_draw() < {dial_threshold(p)}u"
    return {
        "declarations": _pcg("lethal_state", "lethal_draw"),
        "onReset": "lethal_state = zil_seed * 2654435761u + zil_lane * 2246822519u + 0x165667b1u; lethal_draw();",
        "onInstructionStart": f"if (unit == 0u) {{ if ({cond}) {{ return ZIL_HALT; }} }}",
    }


REG8 = ["cpu_a", "cpu_f", "cpu_b", "cpu_c", "cpu_d", "cpu_e", "cpu_h", "cpu_l", "cpu_a2", "cpu_f2", "cpu_b2", "cpu_c2", "cpu_d2", "cpu_e2", "cpu_h2", "cpu_l2"]


def conv_hooks(variant: str) -> dict:
    lines = ["conv_state = zil_seed * 2246822519u + zil_lane * 3266489917u + 0x27d4eb2fu;", "conv_draw();"]
    if variant == "randreg":
        lines += [f"{r} = conv_draw() & 0xffu;" for r in REG8] + ["cpu_ix = conv_draw() & 0xffffu;", "cpu_iy = conv_draw() & 0xffffu;"]
    elif variant == "randsp":
        lines += ["cpu_sp = conv_draw() & 0xffffu;"]
    else:
        raise ValueError(variant)
    return {"declarations": _pcg("conv_state", "conv_draw"), "onReset": "\n".join(lines)}


@dataclass(frozen=True)
class World:
    tag: str
    L: int
    P: int
    rule: str = "plain"          # plain | lethal | dial | randreg | randsp
    p: float | None = None       # dial probability
    note: str = ""
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def lethal(self) -> bool:
        return self.rule == "lethal"

    def machine(self, core: str) -> str:
        return f"imp_{self.tag}_{core}"

    def machines(self) -> tuple[str, str, str]:
        return tuple(self.machine(c) for c in CORES)

    @property
    def resumable_twin(self) -> str | None:
        """Worlds whose machines declare private hook state (the PCG streams) cannot be resumed; attribution steps them
        through the plain world of the same ring (see impact.py). None when the world itself is resumable."""
        if self.rule in ("dial", "randreg", "randsp"):
            return f"L{self.L}" if self.P == 2 * self.L else f"L{self.L}_P{self.P}"
        return None

    def spec(self) -> dict:
        w = {
            "memory": {"kind": "segments", "segments": [{"bytes": self.L, "name": "A"}, {"bytes": self.L, "name": "B"}], "ring": self.P},
            "init": {"stack": {"topAliases": "lastByte"}},
            "budget": {"unit": "steps", "value": "perLane"},
        }
        if self.rule == "lethal":
            w["halting"] = {"lethal": {"firstUnit": 0}}
        w["ablation"] = {"table": "uniform", "layout": "coarse"}
        w["outputs"] = {"writeCounts": True}
        if self.rule == "dial":
            w["hooks"] = dial_hooks(self.p)
        elif self.rule in ("randreg", "randsp"):
            w["hooks"] = conv_hooks(self.rule)
        return w

    def algocell_executor(self) -> tuple[str, str | None]:
        """Algocell's own single-pair executor for this world (file in shader/, or a derivation): (file name, derivation).
        Ring changes affect MEM_LENGTH and the private memory array size (see impact.derive_executor; the tests
        regenerate every exported plain/ring executor). The unexported L19/P38 executor derives from L18/P36;
        those two use the same array size, so only MEM_LENGTH changes for that particular derivation."""
        if self.rule == "plain":
            if self.P == 2 * self.L:
                f = f"z80_test_L{self.L}.wgsl"
                if (SHADER_DIR / f).exists():
                    return f, None
                return "z80_test_L18.wgsl", f"MEM_LENGTH={self.P}"
            return f"z80_test_L{self.L}_P{self.P}.wgsl", None
        if self.rule == "lethal":
            return f"z80_test_lethal_L{self.L}.wgsl", None
        if self.rule == "dial":
            return f"z80_test_{self.tag.split('_')[0]}_L{self.L}.wgsl", None
        return f"z80_test_{self.rule}_L{self.L}.wgsl", None


def all_worlds() -> list[World]:
    ws = [World(f"L{L}", L, 2 * L, note=f"Algocell's pair at L = {L} (ring {2 * L})") for L in PLAIN_L]
    ws += [World(f"L{L}_P{P}", L, P, note=f"Algocell's pair at L = {L} on a {'odd' if P % 2 else 'padded'} ring of {P} "
                                       f"({'Stage F' if (L, P) in STAGE_F_PADDED else 'ring sweep, Extended Data Fig. 3'})") for L, P in PADDED]
    ws += [World(f"lethal_L{L}", L, 2 * L, "lethal", note=f"the lethal-tar rule at L = {L}") for L in LETHAL_L]
    ws += [World(f"{tag}_L{DIAL_L}", DIAL_L, 2 * DIAL_L, "dial", p, note=f"the lethality dial (DZ) at p = {p!r}, Algocell's PCG stream")
           for tag, p in DIAL.items()]
    ws += [World(f"{v}_L{L}", L, 2 * L, v, note=f"the {v} convention (C) at L = {L}, Algocell's PCG stream") for v in ("randreg", "randsp") for L in CONV_L]
    tags = [w.tag for w in ws]
    assert len(tags) == len(set(tags))
    return ws


WORLDS = {w.tag: w for w in all_worlds()}


def world(tag: str) -> World:
    return WORLDS[tag]


def plain_world(L: int, P: int | None = None, lethal: bool = False) -> World:
    P = P or 2 * L
    if lethal:
        return WORLDS[f"lethal_L{L}"] if P == 2 * L else None
    return WORLDS[f"L{L}" if P == 2 * L else f"L{L}_P{P}"]


def spec_json() -> str:
    out = {}
    for w in all_worlds():
        spec = w.spec()
        for core, sel in CORES.items():
            out[w.machine(core)] = {"note": f"Impact study ({CORE_NOTES[core]}): {w.note}. The world spec is identical for "
                                            f"{', '.join(w.machines())}.", "core": dict(sel), "world": spec}
    return json.dumps(out, indent=1) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = spec_json()
    if a.check:
        ok = SPEC_FILE.exists() and SPEC_FILE.read_text() == text
        print(("ok  " if ok else "DIFF"), SPEC_FILE.name)
        sys.exit(0 if ok else 1)
    SPEC_FILE.write_text(text)
    print(f"wrote {SPEC_FILE.name}: {len(WORLDS)} worlds, {3 * len(WORLDS)} machines")


if __name__ == "__main__":
    main()
