"""Generate the condition files for every stage (see PLAN.md).

    python make_conds.py            # writes conds/stage{A,...,I}.json
    python make_conds.py stageI     # (re)writes only the named stage files

Labels: `<ablation>` or `<ablation>@<arm>` for control arms that change a setting other than the
instruction set (`ablation_of(label)` strips the arm). Stems (see algocell_exp.batch.run_stem)
include label, L, steps, k, seed and the replicate index, so every condition in a stage maps
to a unique file.

Stages A and B are frozen (their files are checked byte-for-byte by the tests). Stages C, D
and E were re-specified on 2026-10-07 after the design review (REVIEW.md §4):
  * fine early sampling (every 50 steps to 5,000, then 500), the 256-bin byte histogram and
    the active-pair count in every sample, 8 random tapes per sample, no early stop;
  * new seeds (101–110 for C and E, 1001–1020 for D): Stage B's seeds 1–10 generated the
    hypotheses, and the same seed is the same initial soup and RNG stream;
  * the stack ablation split into write-side and read-side arms.
"""

from __future__ import annotations

import itertools
import json
import os

ABLATIONS = {
    "none": [],
    # ── Stage C finer ablations ──
    "push-only": ["family:stack"],                       # PUSH + POP (8): the stack family as a whole
    "ex-sp-only": ["EX (SP),HL"],
    "call-rst": ["family:call-ret", "family:rst"],       # CALL*, RET*, RST (34): calls and returns as a whole
    "ld-imm": ["family:ld8-imm", "family:ld16-imm"],     # only immediates removed
    "ld-reg": ["family:ld8", "family:ld-special"],       # register-to-register loads removed
    "cb-page": ["family:rotate", "family:rotate-mem", "family:bit", "family:bit-set", "family:bit-set-mem", "-RLD", "-RRD"],
    "ed-loads": ["ed:43", "ed:53", "ed:63", "ed:73", "ed:4B", "ed:5B", "ed:6B", "ed:7B"],
    # ── write-side / read-side split of the stack arm (review 2026-10-07) ──
    "stack-write-only": ["PUSH", "EX (SP),HL", "CALL", "RST N"],          # 22 opcodes that write memory through SP
    "stack-read-only": ["POP", "RET", "EX DE,HL", "EXX", "EX AF,AF'"],    # 24 opcodes of the old arm that write nothing
    "push": ["PUSH"],                                                     # 4
    "call-rst-write": ["CALL", "RST N"],                                  # 17
    # ── Stage A arms ──
    "block-copy": ["family:block-copy"],
    "stack-writes": ["family:stack", "family:ex", "family:call-ret", "family:rst"],   # 46: stack + exchange + call/return + RST (name kept for continuity)
    "ld-mem": ["family:ld8-mem", "family:ld16-mem"],
    "all-ld": ["family:ld8", "family:ld8-imm", "family:ld8-mem", "family:ld16-imm", "family:ld16-mem", "family:ld-special"],
    "no-copy": [
        "family:ld8", "family:ld8-imm", "family:ld8-mem", "family:ld16-imm", "family:ld16-mem", "family:ld-special",
        "family:stack", "family:ex", "family:block-copy",
    ],
    "rmw-only": ["family:writes-mem", "-family:incdec-mem", "-family:rotate-mem", "-family:bit-set-mem"],
    # ── Stage H topology control: NOT an instruction ablation. Nothing is suppressed; the label prefix names the
    #    pairing (`grid: mixed`, partner drawn uniformly from the whole soup) so that stems, zoo and preflight resolve it. ──
    "mixed": [],
    # ── Stage I lethal-tar control: NOT an instruction ablation either. Nothing is suppressed; the label prefix names the
    #    rule (`zero_halts: true`, a zero byte fetched as an opcode halts the pair) so that stems, zoo and preflight resolve it. ──
    "lethal": [],
    # ── 8080 subset (NATURE_PLAN Move 1): the Z80 restricted to the Intel 8080 instruction set by suppressing every Z80-only
    #    instruction: the whole CB page (rotates, shifts, bit ops), every defined ED-page opcode (block copies, 16-bit
    #    ADC/SBC, IN/OUT (C), NEG, RETN/RETI, IM, LD A,I/R, RLD/RRD, LDI/LDD/CPI/CPD and their repeats) and the eight
    #    Z80-only base opcodes (EX AF,AF', DJNZ, JR ×5, EXX). Deviations from a real 8080, stated in the paper: IX/IY
    #    prefixes still select the IX/IY form of an 8080 instruction (a real 8080 treats DD/FD as CALL aliases), and the
    #    suppressed bytes are NOPs rather than the 8080's undocumented aliases (CB = JMP, ED = CALL, D9 = RET). ──
    "i8080": ["family:bit", "family:bit-set", "family:bit-set-mem", "family:rotate", "family:rotate-mem", "EX AF,AF'", "DJNZ", "JR ", "EXX"] + ["ed:40", "ed:41", "ed:42", "ed:43", "ed:44", "ed:45", "ed:46", "ed:47", "ed:48", "ed:49", "ed:4a", "ed:4b", "ed:4c", "ed:4d", "ed:4e", "ed:4f", "ed:50", "ed:51", "ed:52", "ed:53", "ed:54", "ed:55", "ed:56", "ed:57", "ed:58", "ed:59", "ed:5a", "ed:5b", "ed:5c", "ed:5d", "ed:5e", "ed:5f", "ed:60", "ed:61", "ed:62", "ed:63", "ed:64", "ed:65", "ed:66", "ed:67", "ed:68", "ed:69", "ed:6a", "ed:6b", "ed:6c", "ed:6d", "ed:6e", "ed:6f", "ed:70", "ed:71", "ed:72", "ed:73", "ed:74", "ed:75", "ed:76", "ed:78", "ed:79", "ed:7a", "ed:7b", "ed:7c", "ed:7d", "ed:7e", "ed:a0", "ed:a1", "ed:a2", "ed:a3", "ed:a8", "ed:a9", "ed:aa", "ed:ab", "ed:b0", "ed:b1", "ed:b2", "ed:b3", "ed:b8", "ed:b9", "ed:ba", "ed:bb"],
}
STAGE_A_ABLATIONS = ["none", "block-copy", "stack-writes", "ld-mem", "all-ld", "no-copy", "rmw-only"]
SEEDS = list(range(1, 11))
HORIZON = 300_000
SAMPLE_EVERY = 500
SQUARE_TAPES = (4, 9, 16, 25, 36, 49, 64, 81, 100)
NONSQUARE_TAPES = (8, 10, 12, 18, 20, 24, 32, 50)   # headless only (no √L×√L display); shaders exported for them too
TINY_TAPES = (3, 5, 6, 7)                             # where does life start? (the 3-byte unit DEC E ; LDIR exists)
EARLY_STEPS = (1, 2, 3, 5, 8, 13, 21, 34)             # the zero flood forms within ~30 steps (31% zeros by step 50)
SNAPSHOTS = (500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 50000, 75000, 100000, 150000, 200000, 300000, 500000, 750000, 1000000)
CELLS = 20_000
PAIRS = 8_192


def ablation_of(label: str) -> str:
    return label.split("@", 1)[0]


def cond(label: str, tape: int, steps: int, k: int, seed: int) -> dict:
    """Stage A/B condition (frozen: early stop at 50% occupancy, 500-step sampling)."""
    return {
        "label": label,
        "tape": tape,
        "z80_steps": steps,
        "noise_exp": k,
        "seed": seed,
        "suppress": ";".join(ABLATIONS[ablation_of(label)]),
        "horizon": HORIZON,
        "sample_every": SAMPLE_EVERY,
        "stop_share": 0.5,
    }


def stage_a() -> list[dict]:
    return [cond(a, 16, st, k, s) for a, st, k, s in itertools.product(STAGE_A_ABLATIONS, (32, 128, 512), (2, 4, 6), SEEDS)]


def stage_b() -> list[dict]:
    tapes = (4, 9, 25, 36, 49, 64, 81, 100)
    return [cond(a, L, st, 4, s) for L, a, st, s in itertools.product(tapes, ("none", "block-copy", "stack-writes", "no-copy"), (128, 512), SEEDS)]


def _c(label, tape, steps, k, seed, horizon=HORIZON, every=SAMPLE_EVERY, **extra) -> dict:
    """Stage C/D/E condition: no early stop, fine early sampling, random tapes, instrumentation."""
    c = cond(label, tape, steps, k, seed)
    c.update({"horizon": horizon, "sample_every": every, "sample_every_early": 50, "early_until": 5_000, "stop_share": -1, "random_tapes": 16,
              "sample_steps": list(EARLY_STEPS), "snapshot_steps": [t for t in SNAPSHOTS if t <= horizon], "census_pairs": 128, "exemplar_count": 10})
    c.update(extra)
    return c


SEEDS_C = list(range(101, 111))


def stage_c() -> list[dict]:
    """PLAN.md Stage C (re-specified 2026-10-07): new seeds 101–110; C3 gains the write-only / read-only stack arms."""
    out = []
    for a in ("no-copy", "rmw-only", "all-ld"):                                   # C1 — 1M steps
        for st, k in ((128, 4), (32, 2)):
            out += [_c(a, 16, st, k, s, 1_000_000, 1000) for s in SEEDS_C]
    for a in ("none", "block-copy", "ld-mem"):                                      # C2 — succession without censoring
        for st in (128, 512):
            for k in (2, 4, 6):
                out += [_c(a, 16, st, k, s) for s in SEEDS_C]
    out += [_c("none", 16, 32, 2, s) for s in SEEDS_C]                              # the unablated comparator for the C3 (32, k=2) setting
    for a in ("push-only", "ex-sp-only", "call-rst", "ld-imm", "ld-reg", "cb-page", "ed-loads", "stack-write-only", "stack-read-only", "stack-writes"):   # C3 (+ the Stage A arm with new seeds)
        for st, k in ((128, 4), (32, 2)):
            out += [_c(a, 16, st, k, s) for s in SEEDS_C]
    for a in ("none", "stack-writes"):                                              # C5 — size without censoring
        for L in (36, 100):
            out += [_c(a, L, 128, 4, s) for s in SEEDS_C]
    return out


SEEDS_D = list(range(1001, 1021))


def stage_d() -> list[dict]:
    """PLAN.md Stage D (re-specified 2026-10-07): the L = 9 reverse ablation with new seeds and split arms."""
    out = []
    for a in ("none", "stack-writes", "stack-write-only", "stack-read-only", "push", "call-rst-write"):
        for st in (128, 512):
            out += [_c(a, 9, st, 4, s) for s in SEEDS_D]
    out += [_c("none", 9, 32, 4, s) for s in SEEDS_D]        # D5: fewer pushes per interaction
    out += [_c("none", 9, 128, 6, s) for s in SEEDS_D]       # exploratory: lower mutation
    return out


def grid_for_bytes(L: int, total_bytes: int = CELLS * 16) -> tuple[int, int, int]:
    """(width, height, pairs) keeping total soup bytes ≈ total_bytes and drawn pairs per cell ≈ PAIRS/CELLS."""
    cells = total_bytes // L
    w = int(round((cells * 1.28) ** 0.5))      # keep the 160:125 aspect
    h = max(1, int(round(cells / w)))
    pairs = int(round(PAIRS * (w * h) / CELLS))
    return w, h, pairs


def stage_e() -> list[dict]:
    """PLAN.md Stage E — size-axis control arms (review §4, points 6–7), 128 steps, k = 4, seeds 101–110, no stop:
       @nominal   the Stage B setting without early stop (all 9 square L + 8 non-square L)
       @mubyte    per-BYTE mutation rate held at the L = 16 value: mutations/step = 32·L
       @bytes     total soup bytes held at 320,000 (cells = 320,000/L; pairs scaled with cells)
       @steps8L   Z80 budget proportional to L: steps = 8·L (= 128 at L = 16)
       @musweep   L = 100, k ∈ {1,2,3,5,8} (k = 4 is @nominal): the error-threshold curve
       @budget    L = 100, steps ∈ {32,64,256,1024,2048} (128 is @nominal): bytes per encounter
       plus E6 replicate-variance (10 repeats of one seed in 3 cells) and E7 the 32-step/k=2 block-copy exception with new seeds.
       At L = 16 the three control arms coincide with @nominal and with Stage C cells: these are declared cross-batch replicates (PLAN)."""
    out = []
    for a in ("none", "stack-write-only"):
        for L in TINY_TAPES + SQUARE_TAPES + NONSQUARE_TAPES:
            out += [_c(f"{a}@nominal", L, 128, 4, s) for s in SEEDS_C]
        for k in (1, 2, 3, 5, 8):                                                  # E8 — mutation sweep at L = 100 (error-threshold curve)
            out += [_c(f"{a}@musweep", 100, 128, k, s) for s in SEEDS_C]
        for st in (32, 64, 256, 1024, 2048):                                       # E9 — budget sweep at L = 100 (bytes per encounter)
            out += [_c(f"{a}@budget", 100, st, 4, s) for s in SEEDS_C]
        for L in SQUARE_TAPES:
            out += [_c(f"{a}@mubyte", L, 128, 4, s, mutations_per_step=32 * L) for s in SEEDS_C]
            w, h, pairs = grid_for_bytes(L)
            out += [_c(f"{a}@bytes", L, 128, 4, s, width=w, height=h, pairs=pairs) for s in SEEDS_C]
            out += [_c(f"{a}@steps8L", L, 8 * L, 4, s) for s in SEEDS_C]
    for lab, L in (("none", 16), ("stack-writes", 16), ("none", 9)):                # E6 — within-seed variance
        out += [_c(f"{lab}@var", L, 128, 4, 1, replicate=r) for r in range(1, 11)]
    for a in ("none", "block-copy"):                                              # E7 — the 32-step · 1/4 exception, new seeds
        out += [_c(a + "@x32k2", 16, 32, 2, s) for s in SEEDS_D]
    return out


RINGS = {16: (33, 34, 35, 37), 36: (73, 74, 75, 79)}   # 33 = 3·11, 34 = 2·17, 35 = 5·7, 37 prime; 73 prime, 74 = 2·37, 75 = 3·5², 79 prime
RINGS_F4 = {12: (28, 36), 10: (28, 40), 8: (32,)}       # F4 dead-zone probes (PLAN.md): pusher gen2 in isolation 0.20 → 0.60/0.58 at L = 12; no rescue at L = 10; impaired at L = 8 · P = 32


def stage_f() -> list[dict]:
    """PLAN.md Stage F — ring arithmetic: pad the pair memory to P bytes (addresses wrap mod P; SP
    still aliases the end of B). A shift-copy converges to period gcd(offset, P), so the periods of the
    emergent LDIR replicators must track the divisors of P, and a prime P admits none (only
    BC-limited exact copiers and the stack family). The P = 2L rows are Stage E's @nominal runs."""
    out = []
    for a in ("none", "stack-write-only"):
        for L, rings in RINGS.items():
            for P in rings:
                out += [_c(f"{a}@ring{P}", L, 128, 4, s, mem_length=P) for s in SEEDS_C]
    # F4 (added 2026-10-07 after Stage E, before any F run): is the L = 9–12 dead zone ring arithmetic?
    # `none` only; the P = 2L rows are Stage E's @nominal runs.
    for L, rings in RINGS_F4.items():
        for P in rings:
            out += [_c(f"none@ring{P}", L, 128, 4, s, mem_length=P) for s in SEEDS_C]
    return out


SEEDS_G = list(range(2001, 2021))


def stage_g() -> list[dict]:
    """PLAN.md Stage G — closure, confirmatory (pre-registered 2026-10-08 after the post hoc closure.py finding).
    G1: `none` at L = 16 and L = 50, 300k steps, seeds 2001–2020 — the first replicator is straight-line and
    partner-dependent, the 300k dominant has control flow and is partner-independent, the heritable fraction rises.
    G2: `none` at L = 20 and L = 64 for 1,000,000 steps (no closure by 300k in Stage E) — does closure arrive later?"""
    out = []
    for L in (16, 50):
        out += [_c("none@closure", L, 128, 4, s) for s in SEEDS_G]
    for L in (20, 64):
        out += [_c("none@closure1M", L, 128, 4, s, 1_000_000, 1000) for s in SEEDS_G]
    return out


SEEDS_H = list(range(3001, 3011))


def stage_h() -> list[dict]:
    """PLAN.md Stage H — well-mixed control of Stage G L = 16 (pre-registered 2026-10-08, before any H run).
    `grid: mixed`: the partner j is drawn uniformly from the whole soup instead of one of the four lattice neighbours
    (shader derived from sim_square_L16.wgsl by algocell_exp.gen_mixed_shader; everything else byte-identical). 10 worlds,
    new seeds 3001–3010, every other field identical to the Stage G L = 16 conditions (`none@closure`: 20,000 cells, 8,192
    draws, 128 steps, k = 4, 300,000 steps, same sample/snapshot schedule and census). Run locally by stage_h_local.py."""
    return [_c("mixed@closure", 16, 128, 4, s, grid="mixed") for s in SEEDS_H]


SEEDS_I = list(range(4001, 4011))


def stage_i() -> list[dict]:
    """PLAN.md Stage I — lethal tar in the first machine (pre-registered 2026-10-08 night, before any I run).
    `zero_halts: true`: a zero byte (NOP) fetched as an opcode halts the pair's execution for the rest of the encounter, as an
    unmatched bracket halts BFF (shaders derived from sim_square_L16.wgsl / z80_test_L16.wgsl by algocell_exp.gen_lethal_shader;
    everything else byte-identical). 10 worlds on the square lattice, new seeds 4001–4010, every other field identical to the
    Stage G L = 16 conditions (`none@closure`: 20,000 cells, 8,192 draws, 128 steps, k = 4, 300,000 steps, same sample/snapshot
    schedule and census). Culture tests, partner tests and the c4 census run under the same rule. Run locally by
    `stage_h_local.py --stage I`."""
    return [_c("lethal@closure", 16, 128, 4, s, zero_halts=True) for s in SEEDS_I]


SEEDS_K = list(range(5001, 5021))


def stage_k() -> list[dict]:
    """REVISION_PREREG.md Stage K — closure at the aligned intermediate length L = 32 (2L = 64 divides 65,536, so no closer
    can use the 16-bit address wrap that the L = 20 and 50 relative-jump closers use). Stage G `none@closure1M` conditions
    at L = 32, seeds 5001–5020, one million steps, on Modal."""
    return [_c("none@closure1M", 32, 128, 4, s, 1_000_000, 1000) for s in SEEDS_K]


SEEDS_L = list(range(6001, 6011))
SNAPSHOTS_10M = tuple(SNAPSHOTS) + (2_000_000, 3_000_000, 5_000_000, 7_500_000, 10_000_000)


def stage_l() -> list[dict]:
    """NATURE_PLAN Move 2c (pre-registered in REVISION_PREREG.md, L) — ten-million-step extensions at L = 16 and 20, 10 worlds
    each, to see whether the capacity for inherited variation recovers after closure and in which lineages."""
    return [_c("none@closure10M", L, 128, 4, s, 10_000_000, 5000, snapshot_steps=list(SNAPSHOTS_10M)) for L in (16, 20) for s in SEEDS_L]


SEEDS_M = list(range(7001, 7021))


def stage_m() -> list[dict]:
    """NATURE_PLAN Move 1 (pre-registered in REVISION_PREREG.md, M) — the 8080 subset: Stage G conditions under the `i8080`
    suppression at L = 16 (300k steps) and L = 32 (one million steps), 20 worlds each."""
    return [_c("i8080@closure", 16, 128, 4, s) for s in SEEDS_M] + [_c("i8080@closure1M", 32, 128, 4, s, 1_000_000, 1000) for s in SEEDS_M]


STAGES = {"stageA": stage_a, "stageB": stage_b, "stageC": stage_c, "stageD": stage_d, "stageE": stage_e, "stageF": stage_f, "stageG": stage_g, "stageH": stage_h, "stageI": stage_i, "stageK": stage_k, "stageL": stage_l, "stageM": stage_m}


if __name__ == "__main__":
    import sys

    root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(root, "conds"), exist_ok=True)
    names = sys.argv[1:] or list(STAGES)   # `python make_conds.py stageH` writes only that file; no argument writes all
    for name in names:
        conds = STAGES[name]()
        with open(os.path.join(root, "conds", f"{name}.json"), "w") as f:
            json.dump(conds, f, indent=0)
        print(name, len(conds), "conditions")
