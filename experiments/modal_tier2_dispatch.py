"""Registered I1-T2 launch only: modal run --detach modal_tier2_dispatch.py.

Frozen runners are called unchanged. Persistence hooks commit each snapshot.
Serial/branching report functions are suppressed; no claim scoring is dispatched.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal
from tier2_dispatch import CAP, DATA, HERE, jobs, projection

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libvulkan1", "libx11-6", "libxext6", "libxcb1", "libegl1", "libgles2", "libglvnd0")
         .pip_install("wgpu==0.32.0", "numpy==2.5.3", "brotli==1.2.0", "pandas==3.0.6")
         .env({"WGPU_BACKEND_TYPE": "Vulkan", "XDG_RUNTIME_DIR": "/tmp"})
         .add_local_dir(HERE / "algocell_exp", remote_path="/root/algocell_exp")
         .add_local_dir(HERE / "conds", remote_path="/root/conds")
         .add_local_dir(DATA, remote_path="/root/results/tier2"))
for name in ("tier2_dispatch.py", "dial_soups.py", "offset_switch.py", "invasion_generic.py",
             "serial_retention.py", "branching_transfer.py", "make_conds.py", "threshold_grid.py",
             "population_classes.py", "mutscan.py", "stats_tests.py"):
    image = image.add_local_file(HERE / name, remote_path=f"/root/{name}")
# Read-only, hash-bound partner archive; never mutate the source files or other apps.
for record in json.loads((DATA / "control_inputs.json").read_text())["K_pool_files"]:
    image = image.add_local_file(record["path"], remote_path=f"/root/frozen_k/{Path(record['path']).name}")
vol = modal.Volume.from_name("algocell-atlas-runs")
app = modal.App("algocell-i1-tier2")


def volume_markers():
    from tier2_dispatch import marker_name
    result = {}
    for j in jobs():
        p = Path("/runs/tier2") / j["experiment"] / marker_name(j, "spend")
        if p.exists():
            result[j["id"]] = json.loads(p.read_text())
    return result


@app.function(image=image, cpu=1, memory=8192, timeout=120, retries=0, volumes={"/runs": vol})
def inspect_dispatch():
    import sys
    sys.path.insert(0, "/root")
    from tier2_dispatch import local_complete
    vol.reload()
    all_jobs = jobs()
    return {"complete": [j["id"] for j in all_jobs if local_complete(j, "/runs/tier2")],
            "markers": volume_markers()}


def execute_world(j, commit):
    """In-process adapters change persistence/report routing, never RNG or design."""
    import sys
    sys.path.insert(0, "/root")
    out = Path("/runs/tier2") / j["experiment"]
    out.mkdir(parents=True, exist_ok=True)
    if j["kind"] == "batch":
        from algocell_exp import batch as B
        original = B.run

        def run_with_commit(**kw):
            snapshot = kw["on_snapshot"]

            def persisted(name, soup):
                snapshot(name, soup)
                commit()
            kw["on_snapshot"] = persisted
            return original(**kw)
        B.run = run_with_commit
        try:
            result = B.run_to_dir(j["condition"], str(out), {"tier2": True, "git_commit": j["source_commit"]})
            problems = B.check_outputs(j["condition"], str(out))
            if problems:
                raise RuntimeError(problems)
            return result
        finally:
            B.run = original
    if j["kind"] in ("dial", "offset", "invasion"):
        import importlib
        module = importlib.import_module({"dial": "dial_soups", "offset": "offset_switch", "invasion": "invasion_generic"}[j["kind"]])
        original = module.np

        class PersistenceNumpy:
            def __getattr__(self, name):
                return getattr(original, name)

            def save(self, *args, **kwargs):
                original.save(*args, **kwargs)
                commit()
        module.np = PersistenceNumpy()
        try:
            if j["kind"] == "dial":
                return module.run_world(j["variant"], j["seed"], j["steps"], str(out), machine="c1")
            return module.run_world(j["condition"], str(out), machine="c1")
        finally:
            module.np = original
    import numpy as np
    from types import SimpleNamespace
    from algocell_exp import executor as E
    import serial_retention as S
    gt = dict(j["genotype"], tape=np.array(j["genotype"]["tape"], dtype=np.uint8), sup=tuple(j["genotype"]["sup"]))
    E.select("tier2c1_assay")
    target = out / j["stem"]
    S.panel = lambda: [gt]
    S.report = lambda a: None
    if j["kind"] == "serial":
        S.run(SimpleNamespace(out=str(target), only=gt["key"]))
    else:
        import branching_transfer as B
        import brotli
        import hashlib
        import re
        B.panel = lambda: [gt]
        B.report = lambda a: None

        def frozen_pool(L):
            records = json.loads((DATA / "control_inputs.json").read_text())["K_pool_files"]
            records = [r for r in records if f"_L{L}_" in r["path"]]
            records.sort(key=lambda r: tuple(map(int, re.search(r"_s(\d+)\.soup_t(\d+)", r["path"]).groups())))
            assert len(records) == 40
            arrays = []
            for r in records:
                blob = (Path("/root/frozen_k") / Path(r["path"]).name).read_bytes()
                if hashlib.sha256(blob).hexdigest() != r["sha256"]:
                    raise ValueError(f"K archive hash mismatch: {r['path']}")
                arrays.append(np.frombuffer(brotli.decompress(blob), dtype=np.uint8).reshape(-1, L))
            return np.concatenate(arrays), "20 original Stage G worlds; t5000/t10000, hash-bound archived pool"
        B.kin_pool = frozen_pool
        B.run(SimpleNamespace(out=str(target), only=gt["key"], arms=j["arm"], **B.REG))
    (target / "executor_usage.json").write_text(json.dumps(E.usage(), indent=2) + "\n")
    return {"measurement_only": True, "claim_report_deferred": True, "executor_usage": E.usage()}


def worker(j):
    import sys
    sys.path.insert(0, "/root")
    import subprocess
    from tier2_dispatch import local_complete, marker_name
    vol.reload()
    if local_complete(j, "/runs/tier2"):
        return {"id": j["id"], "skipped": True}
    if projection(jobs(), volume_markers())["total_forecast_usd"] > CAP:
        raise RuntimeError("I1-T2 remaining work exceeds $40; dispatch must stop for registration amendment")
    base = Path("/runs/tier2") / j["experiment"]
    base.mkdir(parents=True, exist_ok=True)
    state = dict(id=j["id"], status="running", spec_sha256=j["spec_sha256"], source_commit=j["source_commit"],
                 started_epoch=time.time(), machine=j["machine"])
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    if "L4" not in gpu:
        raise RuntimeError(f"Registered L4 required, got {gpu}")
    state["gpu"] = gpu

    def commit():
        state["updated_epoch"] = time.time()
        (base / marker_name(j, "spend")).write_text(json.dumps(state) + "\n")
        vol.commit()
    commit()
    try:
        result = execute_world(j, commit)
        state.update(status="complete", finished_epoch=time.time())
        (base / marker_name(j, "complete")).write_text(json.dumps(state) + "\n")
        commit()
        return {"id": j["id"], "wall_s": state["finished_epoch"] - state["started_epoch"]}
    except BaseException as exc:
        state.update(status="failed", finished_epoch=time.time(), error=repr(exc))
        commit()
        raise


# Exactly one input in each container, one GPU, one physical CPU core, 8 GiB.
COMMON = dict(image=image, gpu="L4", cpu=1, memory=8192, retries=0,
              single_use_containers=True, volumes={"/runs": vol})
dial_world = app.function(timeout=3600, **COMMON)(worker)
competition_world = app.function(name="competition_world", timeout=7200, **COMMON)(worker)
batch_world = app.function(name="batch_world", timeout=14400, **COMMON)(worker)
panel_world = app.function(name="panel_world", timeout=7200, **COMMON)(worker)


@app.local_entrypoint()
def main():
    import subprocess
    from collections import Counter
    all_jobs = jobs()
    forecast = projection(all_jobs, {})
    if forecast["total_forecast_usd"] > CAP:
        raise RuntimeError(f"Forecast ${forecast['total_forecast_usd']:.4f} exceeds $40; no worlds launched")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=HERE, text=True).strip():
        raise RuntimeError("Commit the dispatcher/forecast before launch")
    initial_call = inspect_dispatch.spawn()
    initial = initial_call.get(timeout=120)
    if projection(all_jobs, initial["markers"])["total_forecast_usd"] > CAP:
        raise RuntimeError("Existing work plus required remaining work exceeds $40")
    record = dict(app_id=app.app_id, source_commit=commit, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  forecast=forecast, calls=[], skipped=initial["complete"], inspection_calls=[initial_call.object_id])
    # Persist locally after every spawn, so a client interruption cannot lose IDs.
    path = DATA / "launch_calls.json"
    for n, j in enumerate(all_jobs):
        if j["id"] in initial["complete"]:
            continue
        if n and n % 32 == 0:
            inspection = inspect_dispatch.spawn()
            record["inspection_calls"].append(inspection.object_id)
            path.write_text(json.dumps(record, indent=2) + "\n")
            check = inspection.get(timeout=120)
            if projection(all_jobs, check["markers"])["total_forecast_usd"] > CAP:
                record["budget_stop"] = True
                path.write_text(json.dumps(record, indent=2) + "\n")
                raise RuntimeError("Budget guard stopped further dispatch; preserve full registered design")
        function = batch_world if j["kind"] == "batch" else panel_world if j["kind"] in ("serial", "branching") else dial_world if j["kind"] == "dial" else competition_world
        call = function.spawn(dict(j, source_commit=commit))
        r = dict(id=j["id"], experiment=j["experiment"], call_id=call.object_id, function_id=function.object_id,
                 launched_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), timeout_s=j["timeout"])
        record["calls"].append(r)
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"SPAWN {r['id']} {r['call_id']}", flush=True)
    record["dispatch_finished_utc"] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    record["function_stats"] = {f.object_id: vars(f.get_current_stats()) for f in (dial_world, competition_world, batch_world, panel_world)}
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"app_id": app.app_id, "launched": dict(Counter(r['experiment'] for r in record['calls'])), "stats": record['function_stats']}), flush=True)
