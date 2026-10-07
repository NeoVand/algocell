"""Write conds/<stage>_remaining.json with the conditions of a stage that have no
completed summary yet (locally in runs/<stage>/), so a sweep can be resumed after
an interruption without re-running finished cells.

    python remaining.py stageA
"""

import glob
import json
import os
import sys


def stem(c: dict) -> str:
    return f"{c['label']}_L{c.get('tape') or 16}_st{c['z80_steps']}_k{c['noise_exp']}_s{c['seed']}"


def main(stage: str) -> None:
    conds = json.load(open(f"conds/{stage}.json"))
    if not os.path.isdir(f"runs/{stage}"):
        raise SystemExit(f"runs/{stage} does not exist: fetch first (./fetch.sh {stage}) or every condition would be re-run")
    done = {os.path.basename(p)[: -len(".summary.json")] for p in glob.glob(f"runs/{stage}/*.summary.json")}
    remaining = [c for c in conds if stem(c) not in done]
    with open(f"conds/{stage}_remaining.json", "w") as f:
        json.dump(remaining, f, indent=0)
    print(f"{stage}: {len(done)} done, {len(remaining)} remaining -> conds/{stage}_remaining.json")
    for c in remaining[:20]:
        print("  ", stem(c))
    if len(remaining) > 20:
        print(f"   … {len(remaining) - 20} more")


if __name__ == "__main__":
    main(sys.argv[1])
