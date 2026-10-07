"""Generate the pre-registered condition lists (see PLAN.md). Writes conds/stageA.json and conds/stageB.json.

Each entry is a dict of algocell_exp.run.run() keyword arguments. Labels name the
ablation; the other factor levels are explicit fields so filenames and analysis
never have to parse labels.
"""

import itertools
import json
import os

ABLATIONS = {
    "none": [],
    # ── Stage C finer ablations ──
    "push-only": ["family:stack"],                       # PUSH/POP removed, EX (SP),HL and CALL/RST kept
    "ex-sp-only": ["EX (SP),HL"],
    "call-rst": ["family:call-ret", "family:rst"],
    "ld-imm": ["family:ld8-imm", "family:ld16-imm"],     # only immediates removed
    "ld-reg": ["family:ld8", "family:ld-special"],       # register-to-register loads removed
    "cb-page": ["family:rotate", "family:rotate-mem", "family:bit", "family:bit-set", "family:bit-set-mem", "-RLD", "-RRD"],
    "ed-loads": ["ed:43", "ed:53", "ed:63", "ed:73", "ed:4B", "ed:5B", "ed:6B", "ed:7B"],
    "block-copy": ["family:block-copy"],
    "stack-writes": ["family:stack", "family:ex", "family:call-ret", "family:rst"],
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


def cond(label: str, tape: int, steps: int, k: int, seed: int) -> dict:
    return {
        "label": label,
        "tape": tape,
        "z80_steps": steps,
        "noise_exp": k,
        "seed": seed,
        "suppress": ";".join(ABLATIONS[label]),
        "horizon": HORIZON,
        "sample_every": SAMPLE_EVERY,
        "stop_share": 0.5,
    }


def stage_a() -> list[dict]:
    return [cond(a, 16, st, k, s) for a, st, k, s in itertools.product(STAGE_A_ABLATIONS, (32, 128, 512), (2, 4, 6), SEEDS)]


def stage_b() -> list[dict]:
    tapes = [4, 9, 25, 36, 49, 64, 81, 100]  # L=16 cells come from Stage A
    return [cond(a, L, st, 4, s) for L, a, st, s in itertools.product(tapes, ("none", "block-copy", "stack-writes", "no-copy"), (128, 512), SEEDS)]


def _c(label, tape, steps, k, seed, horizon, every):
    c = cond(label, tape, steps, k, seed)
    c.update({"horizon": horizon, "sample_every": every, "stop_share": -1, "random_tapes": 8, "label": label})
    return c


def stage_c() -> list[dict]:
    """See PLAN.md Stage C. No early stop; 8 random tapes per sample."""
    out = []
    # C1 — slow or impossible? 1M steps for the null/rare ablations.
    for a in ("no-copy", "rmw-only", "all-ld"):
        for st, k in ((128, 4), (32, 2)):
            out += [_c(a, 16, st, k, s, 1_000_000, 1000) for s in SEEDS]
    # C2 — succession without censoring.
    for a in ("none", "block-copy", "ld-mem"):
        for st in (128, 512):
            for k in (2, 4, 6):
                out += [_c(a, 16, st, k, s, 300_000, 500) for s in SEEDS]
    # C3 — finer ablations.
    for a in ("push-only", "ex-sp-only", "call-rst", "ld-imm", "ld-reg", "cb-page", "ed-loads"):
        for st, k in ((128, 4), (32, 2)):
            out += [_c(a, 16, st, k, s, 300_000, 500) for s in SEEDS]
    # C5 — size without censoring: do large organisms stay tiled by short motifs?
    for a in ("none", "stack-writes"):
        for L in (36, 100):
            out += [_c(a, L, 128, 4, s, 300_000, 500) for s in SEEDS]
    return out


def stage_d() -> list[dict]:
    """See PLAN.md Stage D: confirmatory run for the L = 9 reverse ablation found post hoc in Stage B.
    20 seeds per arm, no early stop, 8 random tapes per sample."""
    out = []
    for a in ("none", "stack-writes", "push-only", "call-rst"):
        for st in (128, 512):
            out += [_c(a, 9, st, 4, s, 300_000, 500) for s in range(1, 21)]
    return out


if __name__ == "__main__":
    os.makedirs("conds", exist_ok=True)
    for name, conds in (("stageA", stage_a()), ("stageB", stage_b()), ("stageC", stage_c()), ("stageD", stage_d())):
        with open(f"conds/{name}.json", "w") as f:
            json.dump(conds, f, indent=0)
        print(name, len(conds), "conditions")
