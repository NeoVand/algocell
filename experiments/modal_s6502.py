"""S1 on Modal L40S: one world per container, with no retries.

    modal run --detach modal_s6502.py --smoke
    modal run --detach modal_s6502.py
    modal run --detach modal_s6502.py --arms A,C --seeds 65001-65020

Smoke outputs live only in s6502_smoke/. Completed worlds are skipped; incomplete
worlds are reported and retained. A relaunch never overwrites a world.
"""

from __future__ import annotations

import json

import modal

app = modal.App("algocell-s6502")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libvulkan1", "libx11-6", "libxext6", "libxcb1", "libegl1", "libgles2", "libglvnd0")
    .pip_install("wgpu==0.32.0", "numpy==2.5.3", "brotli==1.2.0")
    .env({"WGPU_BACKEND_TYPE": "Vulkan"})
    .add_local_dir("algocell_exp", remote_path="/root/experiments/algocell_exp")
    .add_local_file("run6502.py", remote_path="/root/experiments/run6502.py")
)
vol = modal.Volume.from_name("algocell-atlas-runs", create_if_missing=True)

# Recalibrate against the four-arm smoke before launching. Six hours is the
# provisional ceiling; the registered launch uses the measured ceiling below.
WORLD_TIMEOUT_S = 6 * 60 * 60


@app.function(image=image, gpu="L40S", timeout=WORLD_TIMEOUT_S,
              volumes={"/vol": vol}, retries=0, single_use_containers=True)
def world(arm: str, seed: int, git_commit: str, smoke: bool = False) -> dict:
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/experiments")
    import run6502 as R

    if arm not in R.ARMS or (smoke and seed in R.SEEDS) or (not smoke and seed not in R.SEEDS):
        raise ValueError("invalid registered arm/seed or smoke seed overlaps registration")
    vol.reload()
    out = Path("/vol") / ("s6502_smoke" if smoke else "s6502") / arm
    path = out / f"{R.world_name(arm, seed)}.jsonl"
    summary = R.existing_summary(path)
    if summary is not None:
        print(f"skip completed arm={arm} seed={seed}", flush=True)
        return {"status": "skipped", "summary": summary}
    if any(out.glob(f"{R.world_name(arm, seed)}.*")):
        print(f"INCOMPLETE arm={arm} seed={seed}: {path}; retained, no overwrite", flush=True)
        return {"status": "incomplete", "arm": arm, "seed": seed, "path": str(path)}
    print(f"starting arm={arm} seed={seed} smoke={smoke} commit={git_commit}", flush=True)
    try:
        result = R.run_world(arm, seed, out, steps=R.SMOKE_STEPS if smoke else R.HORIZON,
                             git_commit=git_commit, commit_output=vol.commit)
    finally:
        vol.commit()
    print(json.dumps(result), flush=True)
    return {"status": "completed", "summary": result}


def parse_selection(arms: str, seeds: str, smoke: bool) -> tuple[list[str], list[int]]:
    selected_arms = arms.replace(",", " ").split()
    if not selected_arms or len(set(selected_arms)) != len(selected_arms) or any(a not in "ABCD" or len(a) != 1 for a in selected_arms):
        raise ValueError("arms must be unique members of A,B,C,D")
    selected_seeds = []
    for part in seeds.replace(",", " ").split():
        if "-" in part:
            lo, hi = map(int, part.split("-"))
            if hi < lo:
                raise ValueError("seed ranges must be ascending")
            selected_seeds.extend(range(lo, hi + 1))
        else:
            selected_seeds.append(int(part))
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("seeds must be nonempty and unique")
    registered = set(range(65001, 65021))
    if smoke and registered.intersection(selected_seeds):
        raise ValueError("smoke seeds must be outside 65001-65020")
    if not smoke and not set(selected_seeds) <= registered:
        raise ValueError("registered seeds must be within 65001-65020")
    return selected_arms, selected_seeds


@app.local_entrypoint()
def main(arms: str = "A,B,C,D", seeds: str = "", smoke: bool = False):
    from datetime import datetime, timezone

    import run6502 as R

    selected_arms, selected_seeds = parse_selection(arms, seeds or ("999" if smoke else "65001-65020"), smoke)
    commit = R.current_commit()
    print(json.dumps({"kind": "launch", "app_id": app.app_id, "git_commit": commit,
                      "time": datetime.now(timezone.utc).isoformat(), "smoke": smoke,
                      "worlds": len(selected_arms) * len(selected_seeds),
                      "timeout_s": WORLD_TIMEOUT_S}), flush=True)
    for arm in selected_arms:
        for seed in selected_seeds:
            call = world.spawn(arm, seed, commit, smoke)
            print(json.dumps({"kind": "spawn", "app_id": app.app_id, "arm": arm,
                              "seed": seed, "call_id": call.object_id}), flush=True)
