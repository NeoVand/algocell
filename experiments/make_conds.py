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
    return [cond(a, 16, st, k, s) for a, st, k, s in itertools.product(ABLATIONS, (32, 128, 512), (2, 4, 6), SEEDS)]


def stage_b() -> list[dict]:
    tapes = [4, 9, 25, 36, 49, 64, 81, 100]  # L=16 cells come from Stage A
    return [cond(a, L, st, 4, s) for L, a, st, s in itertools.product(tapes, ("none", "block-copy", "stack-writes", "no-copy"), (128, 512), SEEDS)]


if __name__ == "__main__":
    os.makedirs("conds", exist_ok=True)
    for name, conds in (("stageA", stage_a()), ("stageB", stage_b())):
        with open(f"conds/{name}.json", "w") as f:
            json.dump(conds, f, indent=0)
        print(name, len(conds), "conditions")
