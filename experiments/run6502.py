"""Registered S1 runner: records only, with no tape classification or console tapes.

Real runs use modal_s6502.py. Samples follow REVISION_PREREG.md exactly; soups are
brotli-compressed, row-major (20,000 cells × L). The pilot remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import brotli
import numpy as np

from algocell_exp.metrics import quasispecies_share
from algocell_exp.soup import adapter_summary
from algocell_exp.zsoup import ZSoup

HERE = Path(__file__).resolve().parent
HORIZON = 10_000_000
SEEDS = tuple(range(65001, 65021))
SMOKE_SEED = 999
SMOKE_STEPS = 30_000
ARMS = {
    "A": {"L": 16, "budget": 128, "brk": "faithful", "machine": "m6502_pair_maps_L16"},
    "B": {"L": 32, "budget": 128, "brk": "faithful", "machine": "m6502_pair_maps_L32"},
    "C": {"L": 16, "budget": 512, "brk": "faithful", "machine": "m6502_pair_maps_L16"},
    "D": {"L": 16, "budget": 128, "brk": "halt", "machine": "m6502_brkhalt_maps_L16"},
}
SAMPLE_SCHEDULE = ((0, 2_000, 50), (2_000, 20_000, 250),
                   (20_000, 1_000_000, 2_500), (1_000_000, HORIZON, 25_000))
SNAPSHOT_STEPS = (20_000, 100_000, 1_000_000, 3_000_000, HORIZON)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()


def sample_steps(steps: int = HORIZON) -> list[int]:
    return [s for start, stop, every in SAMPLE_SCHEDULE
            for s in range(start + every, min(stop, steps) + 1, every)]


def world_name(arm: str, seed: int) -> str:
    return f"s6502_{arm}_s{seed}"


def existing_summary(path: Path) -> dict | None:
    """Read completion metadata only; incomplete/corrupt files are never replaced."""
    if not path.exists():
        return None
    summary = None
    with path.open() as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return None
            if record.get("kind") == "summary":
                summary = record
    return summary


def exact_tapes(cells: np.ndarray, hashes: np.ndarray) -> tuple[dict, np.ndarray]:
    """Count full byte strings; GPU hash collisions cannot pool distinct tapes.

    The hash is the pilot's GPU hash, accompanied by SHA-256 of the exact tape.
    Equal-count ties use lexicographic tape order, independent of cell order.
    """
    tapes, indices, counts = np.unique(cells, axis=0, return_index=True, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    top3 = [{"hash": int(hashes[indices[i]]), "share": float(counts[i] / len(cells)),
             "tape": tapes[i].tobytes().hex(),
             "sha256": hashlib.sha256(tapes[i].tobytes()).hexdigest()} for i in order[:3]]
    return {"top3": top3, "top_share": top3[0]["share"], "top_hash": top3[0]["hash"],
            "top_tape": top3[0]["tape"], "unique": len(tapes)}, tapes[order[0]]


def run_world(arm: str, seed: int, out: Path, *, steps: int = HORIZON,
              git_commit: str | None = None, submit: int = 50,
              commit_output: Callable[[], None] | None = None) -> dict:
    """Run one world, reserving its file exclusively before any GPU work.

    commit_output is called after the condition, each snapshot, and the summary.
    A failed world is retained for reporting, never silently resumed/overwritten.
    """
    if arm not in ARMS or not 1 <= steps <= HORIZON or submit < 1:
        raise ValueError("invalid arm, horizon, or submit size")
    git_commit = git_commit or current_commit()
    if not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise ValueError("a full worktree git commit is required")
    cfg = ARMS[arm]
    out.mkdir(parents=True, exist_ok=True)
    world = world_name(arm, seed)
    path = out / f"{world}.jsonl"
    if any(out.glob(f"{world}.*")):
        raise FileExistsError(f"{world} exists; a world is never overwritten")
    # 'x' also prevents competing containers from opening the same world's file.
    with path.open("x") as f:
        def emit(record: dict) -> None:
            f.write(json.dumps(record) + "\n")
            f.flush()

        def commit() -> None:
            if commit_output is not None:
                commit_output()

        t_start = time.perf_counter()
        soup = ZSoup(cfg["machine"], width=160, height=125, pair_count=8192,
                     noise_exp=4, seed=seed, z80_steps=cfg["budget"])
        emit({"kind": "condition", "world": world, "arm": arm, **cfg,
              "machine": soup.machine_provenance, "machine_name": cfg["machine"],
              "seed": seed, "steps": steps, "width": soup.width, "height": soup.height,
              "pair_count": soup.pair_count, "noise_exp": soup.noise_exp,
              "sample_schedule": [list(s) for s in SAMPLE_SCHEDULE],
              "snapshot_steps": sorted({s for s in SNAPSHOT_STEPS if s <= steps} | {steps}),
              "snapshot_format": "brotli; row-major cells x L; uint8",
              "git_commit": git_commit, "adapter": adapter_summary(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "started": utc_now()})
        commit()
        snapshots = {s for s in SNAPSHOT_STEPS if s <= steps} | {steps}
        samples = set(sample_steps(steps))
        step = 0
        sample_count = 0
        checkpoint_count = 0
        for target in sorted(samples | snapshots):
            first = step
            t0 = time.perf_counter()
            while step < target:
                n = min(submit, target - step)
                soup.step(n)
                step += n
            soup.sync()
            dt = time.perf_counter() - t0
            cells = soup.read_soup()
            if step in samples:
                sp, top_tape = exact_tapes(cells, soup.read_hashes())
                qs = quasispecies_share(cells, top_tape)
                hist = soup.read_byte_counts()
                emit({"kind": "sample", "step": step, **sp, **qs,
                      "steps_per_s": round((step - first) / max(dt, 1e-9), 2),
                      "byte_hist": hist.astype(int).tolist(),
                      "zero_frac": float(hist[0] / max(int(hist.sum()), 1)),
                      "active_pairs": soup.read_active_pairs(),
                      "elapsed_s": round(time.perf_counter() - t_start, 3)})
                sample_count += 1
            if step in snapshots:
                raw = np.ascontiguousarray(cells, dtype=np.uint8).tobytes()
                compressed = brotli.compress(raw, quality=9)
                filename = f"{world}.t{step}.u8.br"
                with (out / filename).open("xb") as checkpoint:
                    checkpoint.write(compressed)
                emit({"kind": "snapshot", "step": step, "file": filename,
                      "shape": [soup.cell_count, cfg["L"]], "raw_bytes": len(raw),
                      "compressed_bytes": len(compressed),
                      "sha256_raw": hashlib.sha256(raw).hexdigest()})
                checkpoint_count += 1
                commit()
        wall = time.perf_counter() - t_start
        summary = {"kind": "summary", "world": world, "arm": arm, "seed": seed,
                   "steps_run": step, "samples": sample_count, "snapshots": checkpoint_count,
                   "wall_s": round(wall, 3), "steps_per_s": round(step / max(wall, 1e-9), 2),
                   "git_commit": git_commit, "finished": utc_now()}
        emit(summary)
        commit()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke and a.seed in SEEDS:
        ap.error("smoke seeds must be outside the registered range")
    print(json.dumps(run_world(a.arm, a.seed, a.out, steps=SMOKE_STEPS if a.smoke else HORIZON)))


if __name__ == "__main__":
    main()
