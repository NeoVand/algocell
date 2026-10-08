"""One local soup run with dense snapshots for the supplementary videos (NOT launched automatically: it uses the
local GPU for several minutes, and the browser simulation must be paused first).

    python manuscript/figures/video_run.py [--seed 2001] [--L 16] [--horizon 100000] [--out runs/video]

Writes runs/video/<stem>.jsonl, .summary.json and .soup_t<step>.u8.br at every snapshot step through the same
`run_to_dir` path as the experimental stages (identical kernel, sampling and mutation). The snapshot schedule is dense
early (every 25 steps to 3,000: the tar and the first wave), then every 250 to 30,000 (competition), then every 1,000
to the horizon (the closed takeover), about 300 frames. Render with:

    python manuscript/figures/soup_stills.py --run runs/video/<stem> --video --fps 24
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, EXP)


def schedule(horizon: int) -> list[int]:
    steps = set(range(25, 3001, 25)) | set(range(3250, 30001, 250)) | set(range(31000, horizon + 1, 1000))
    return sorted(s for s in steps if s <= horizon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=100000)
    ap.add_argument("--out", default=os.path.join(EXP, "runs", "video"))
    ap.add_argument("--yes", action="store_true", help="confirm that the browser simulation is paused")
    a = ap.parse_args()
    if not a.yes:
        sys.exit("refusing to run: pass --yes after pausing the browser simulation (local GPU run of several minutes)")
    from algocell_exp.batch import run_to_dir

    cond = {"label": "video", "grid": "square", "width": 160, "height": 125, "tape": a.L, "seed": a.seed, "pairs": 8192,
            "z80_steps": 128, "noise_exp": 4, "horizon": a.horizon, "sample_every": 250, "stop_share": None,
            "sample_every_early": 25, "early_until": 3000,   # snapshots are taken on sampling steps only
            "snapshot_steps": schedule(a.horizon)}
    print(f"{len(cond['snapshot_steps'])} snapshots, horizon {a.horizon}, seed {a.seed}, L = {a.L}")
    s = run_to_dir(cond, a.out, provenance={"purpose": "supplementary video frames"})
    print({k: s.get(k) for k in ("label", "seed", "steps_run", "tq_10", "tq_50", "wall_s")})


if __name__ == "__main__":
    main()
