"""Validate smoke structure/provenance/timings only. Never classify or disassemble.

    python check_s6502_smoke.py results/s6502/smoke --report results/s6502/smoke_check.json

The cost gate uses total smoke wall time, including initialization, readbacks,
compression and volume commits before the summary, at the requested $1.95/h.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import brotli

import run6502 as R


def check_world(root: Path, arm: str, seed: int, git_commit: str | None = None) -> dict:
    path = root / arm / f"{R.world_name(arm, seed)}.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["kind"] == "condition", f"{arm}: first line is not condition"
    assert records[-1]["kind"] == "summary", f"{arm}: last line is not summary"
    cond, summary = records[0], records[-1]
    assert sum(r["kind"] == "condition" for r in records) == 1
    assert sum(r["kind"] == "summary" for r in records) == 1
    assert all(r["kind"] in {"condition", "sample", "snapshot", "summary"} for r in records)
    cfg = R.ARMS[arm]
    for key in ("L", "budget", "brk"):
        assert cond[key] == cfg[key], f"{arm}: wrong {key}"
    assert cond["arm"] == arm and cond["seed"] == seed and seed not in R.SEEDS
    assert (cond["width"], cond["height"], cond["pair_count"], cond["noise_exp"]) == (160, 125, 8192, 4)
    assert cond["steps"] == 30_000
    assert cond["machine_name"] == cfg["machine"] == cond["machine"]["machine"]
    assert cond["adapter"] and "L40S" in cond["adapter"] and "Vulkan" in cond["adapter"]
    assert len(cond["git_commit"]) == 40 and summary["git_commit"] == cond["git_commit"]
    if git_commit:
        assert cond["git_commit"] == git_commit, f"{arm}: wrong worktree commit"
    machines = R.HERE / "algocell_exp" / "machines"
    manifest = json.loads((machines / f"{cfg['machine']}.manifest.json").read_text())
    for key in ("core", "world", "wgsl"):
        assert cond["machine"][f"sha256_{key}"] == manifest["sha256"][key], f"{arm}: wrong {key} SHA"
    assert hashlib.sha256((machines / f"{cfg['machine']}.wgsl").read_bytes()).hexdigest() == manifest["sha256"]["wgsl"]
    assert cond["script_sha256"] == hashlib.sha256((R.HERE / "run6502.py").read_bytes()).hexdigest()
    samples = [r for r in records if r["kind"] == "sample"]
    # Independently spell out the registered smoke schedule, including boundaries.
    expected = list(range(50, 2001, 50)) + list(range(2250, 20001, 250)) + list(range(22500, 30001, 2500))
    assert [r["step"] for r in samples] == expected, f"{arm}: missing, duplicate, or off-schedule sample"
    for sample in samples:
        top3 = sample["top3"]
        assert len(top3) == min(3, sample["unique"]), f"{arm}: wrong top-3 count"
        assert len(top3) == 3, f"{arm}: smoke does not contain three tapes"
        shares = [t["share"] for t in top3]
        assert all(math.isfinite(s) and 0 < s <= 1 for s in shares)
        assert sum(shares) <= 1 + 1e-12 and shares == sorted(shares, reverse=True)
        assert len({t["tape"] for t in top3}) == len(top3)
        for tape in top3:
            raw = bytes.fromhex(tape["tape"])
            assert len(raw) == cfg["L"] and 0 <= tape["hash"] < 2**32
            assert hashlib.sha256(raw).hexdigest() == tape["sha256"]
        assert sample["top_share"] == shares[0]
        assert 0 <= sample["q_half_share"] <= sample["q_share"] <= 1
        assert len(sample["byte_hist"]) == 256 and min(sample["byte_hist"]) >= 0
        assert sum(sample["byte_hist"]) == 20000 * cfg["L"]
        assert sample["zero_frac"] == sample["byte_hist"][0] / (20000 * cfg["L"])
        assert 0 < sample["active_pairs"] <= 8192
        assert math.isfinite(sample["steps_per_s"]) and sample["steps_per_s"] > 0
    snapshots = [r for r in records if r["kind"] == "snapshot"]
    assert [r["step"] for r in snapshots] == [20_000, 30_000], f"{arm}: missing snapshot"
    for snapshot in snapshots:
        assert snapshot["file"] == f"{R.world_name(arm, seed)}.t{snapshot['step']}.u8.br"
        compressed = (root / arm / snapshot["file"]).read_bytes()
        raw = brotli.decompress(compressed)
        assert len(compressed) == snapshot["compressed_bytes"]
        assert len(raw) == snapshot["raw_bytes"] == 20000 * cfg["L"]
        assert snapshot["shape"] == [20000, cfg["L"]]
        assert hashlib.sha256(raw).hexdigest() == snapshot["sha256_raw"]
    assert summary["steps_run"] == 30_000 and summary["samples"] == len(expected) and summary["snapshots"] == 2
    assert summary["arm"] == arm and summary["seed"] == seed
    assert summary["steps_per_s"] > 0 and summary["wall_s"] > 0
    assert abs(summary["steps_per_s"] - 30000 / summary["wall_s"]) < 1
    return {"arm": arm, "checks": "passed", "samples": len(samples), "snapshot_steps": [20000, 30000],
            "steps_per_s": summary["steps_per_s"], "smoke_wall_s": summary["wall_s"],
            "sample_steps_per_s_min": min(r["steps_per_s"] for r in samples),
            "sample_steps_per_s_max": max(r["steps_per_s"] for r in samples),
            "git_commit": cond["git_commit"], "adapter": cond["adapter"],
            "projected_world_hours": R.HORIZON / summary["steps_per_s"] / 3600}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--git-commit")
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    worlds = [check_world(a.root, arm, a.seed, a.git_commit) for arm in R.ARMS]
    hours = 20 * sum(w["projected_world_hours"] for w in worlds)
    cost = hours * 1.95
    report = {"checks": "passed", "seed": a.seed, "steps": 30000, "arms": worlds,
              "registered_worlds": 80, "l40s_usd_per_hour": 1.95,
              "estimated_gpu_hours": hours, "estimated_usd": cost,
              "cost_limit_usd": 160, "launch_permitted": cost <= 160,
              "basis": "10M / total smoke wall steps/s, 20 worlds per arm; includes smoke startup/readback/checkpoint overhead"}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
