"""Preflight for a sweep: everything that must be true BEFORE a condition file is sent to Modal.

    .venv/bin/python preflight.py conds/stageC.json [--horizon 1500] [--full]

1. The test suite passes (pytest tests, incl. the condition-file integrity tests).
2. The condition file is well-formed: unique stems, every suppression string resolves, every
   tape length has an exported shader and executor, the label's ablation exists.
3. One representative of every distinct parameter signature (everything except seed and
   replicate) is run LOCALLY through the exact Modal code path (algocell_exp.batch.run_to_dir)
   with the horizon cut to --horizon steps (early_until cut accordingly), and the outputs are
   checked (sample schedule, byte histogram, active pairs, random tapes, snapshots, atomic
   summary, parameter echo, overrides applied).
4. A cost estimate is printed from measured per-step times (L40S: 0.30 ms/step at 128 steps for
   L ≤ 36, scaled for other budgets/sizes) plus a fixed container overhead.

Exit status is non-zero on any failure. Written after 2026-10-07, when two sweeps were launched
with bugs a two-minute local run would have caught.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

from algocell_exp.batch import check_outputs, run_stem, run_to_dir
from algocell_exp.isa import parse_patterns, resolve
from algocell_exp.soup import SHADER_DIR
from make_conds import ABLATIONS, ablation_of

L40S_USD_PER_H = 1.95
OVERHEAD_S = 25.0  # container start + image + volume commit, per run


def signature(c: dict) -> tuple:
    return tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in c.items() if k not in ("seed", "replicate")))


def step_seconds(c: dict) -> float:
    """Measured on L40S: 0.30 ms/step at 128 Z80 steps (L ≤ 36), ≈ 0.85 ms at 512; cells and pairs scale the work."""
    steps = c.get("z80_steps", 128)
    L = c.get("tape") or 16
    base = 0.30e-3 * (steps / 128) ** 0.75 if steps >= 128 else 0.16e-3 * (steps / 32) ** 0.5
    size = 1.0 + 0.35 * max(0, (L - 36)) / 64           # L = 100 is ~1.35× L = 16 at equal steps
    cells = (c.get("width", 160) * c.get("height", 125)) / 20000
    pairs = c.get("pairs", 8192) / 8192
    return base * size * max(cells, pairs)


def estimate(conds: list[dict]) -> tuple[float, float]:
    secs = sum(c.get("horizon", 300_000) * step_seconds(c) + OVERHEAD_S for c in conds)
    return secs / 3600, secs / 3600 * L40S_USD_PER_H


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("conds")
    ap.add_argument("--horizon", type=int, default=1500, help="local dry-run horizon per representative condition")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--full", action="store_true", help="dry-run EVERY condition (not one per signature)")
    a = ap.parse_args()
    conds = json.load(open(a.conds))
    problems: list[str] = []

    if not a.skip_tests:
        print("1. test suite …", flush=True)
        r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-x"], capture_output=True, text=True)
        print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:])
        if r.returncode != 0:
            problems.append("tests failed")

    print("2. condition file …")
    stems = [run_stem(c) for c in conds]
    if len(set(stems)) != len(stems):
        problems.append("duplicate stems")
    for c in conds:
        if ablation_of(c["label"]) not in ABLATIONS:
            problems.append(f"unknown ablation in label {c['label']}")
        try:
            resolve(parse_patterns(c.get("suppress", "")))
        except Exception as e:  # noqa: BLE001
            problems.append(f"{run_stem(c)}: suppression does not resolve: {e}")
        L = c.get("tape") or 16
        for f in (f"sim_square_L{L}.wgsl", f"z80_test_L{L}.wgsl"):
            if not (SHADER_DIR / f).exists():
                problems.append(f"missing exported shader {f}")
        if (c.get("stop_share", 0.5) or 0) > 0 and c.get("random_tapes"):
            problems.append(f"{run_stem(c)}: early stop with random tapes (Stage C+ runs must not stop early)")
    sigs: dict[tuple, dict] = {}
    for c in conds:
        sigs.setdefault(signature(c), c)
    print(f"   {len(conds)} conditions, {len(sigs)} distinct parameter signatures, {len(set(stems))} unique stems; problems so far: {len(problems)}")

    print(f"3. local dry run of {len(conds) if a.full else len(sigs)} condition(s) at horizon {a.horizon} through the Modal code path …", flush=True)
    todo = conds if a.full else list(sigs.values())
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for i, c in enumerate(todo):
            cc = dict(c)
            cc["horizon"] = min(cc.get("horizon", 300_000), a.horizon)
            if cc.get("early_until"):
                cc["early_until"] = min(cc["early_until"], cc["horizon"])
            cc["sample_every"] = min(cc.get("sample_every", 500), cc["horizon"])
            try:
                run_to_dir(cc, tmp, {"preflight": True})
                problems += check_outputs(cc, tmp)
            except Exception as e:  # noqa: BLE001
                problems.append(f"{run_stem(c)}: dry run raised {type(e).__name__}: {e}")
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                print(f"   {i+1}/{len(todo)} done ({time.time()-t0:.0f}s)", flush=True)

    hours, usd = estimate(conds)
    print(f"4. estimate: {hours:.1f} GPU-hours on L40S ≈ ${usd:.0f} at list price (incl. ≈{OVERHEAD_S:.0f}s container overhead per run)")
    if problems:
        print(f"\nPREFLIGHT FAILED — {len(problems)} problem(s):")
        for p in problems[:40]:
            print("  -", p)
        return 1
    print("\nPREFLIGHT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
