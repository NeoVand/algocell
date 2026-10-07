"""Generate the condition files for every stage (see PLAN.md).

    python make_conds.py            # writes conds/stage{A,B,C,D,E}.json

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
    return out


STAGES = {"stageA": stage_a, "stageB": stage_b, "stageC": stage_c, "stageD": stage_d, "stageE": stage_e, "stageF": stage_f}


if __name__ == "__main__":
    os.makedirs("conds", exist_ok=True)
    for name, fn in STAGES.items():
        conds = fn()
        with open(f"conds/{name}.json", "w") as f:
            json.dump(conds, f, indent=0)
        print(name, len(conds), "conditions")
