"""Run Algocell ablation conditions on Modal GPUs with the exact exported shader.

    modal run modal_app.py                 # smoke test: adapter + throughput
    modal run modal_app.py --conds conds.json --batch pilot1   # fan out a sweep

Each condition is a dict of algocell_exp.run.run() keyword arguments plus a
'label'. JSONL trajectories are persisted to the 'algocell-atlas-runs' volume
under /runs/<batch>/, and the summaries are returned to the caller.
"""

from __future__ import annotations

import modal

app = modal.App("algocell-atlas")

# libGLX_nvidia.so.0 is the NVIDIA Vulkan ICD. It needs the X11 client libs AND
# libglvnd/EGL to initialise; with any of them missing the loader silently drops
# the ICD and wgpu falls back to llvmpipe on the CPU (probed 2026-10-07).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libvulkan1", "libx11-6", "libxext6", "libxcb1", "libegl1", "libgles2", "libglvnd0")
    .pip_install("wgpu>=0.32,<0.40", "numpy>=2", "brotli>=1.1")
    .env({"WGPU_BACKEND_TYPE": "Vulkan"})
    .add_local_dir("algocell_exp", remote_path="/root/algocell_exp")
)
runs_volume = modal.Volume.from_name("algocell-atlas-runs", create_if_missing=True)

GPU = "H200"


@app.function(image=image, gpu=GPU, timeout=60 * 60 * 6, volumes={"/runs": runs_volume})
def run_condition(cond: dict, batch: str = "adhoc") -> dict:
    import io
    import os

    from algocell_exp.run import run

    import json

    buf = io.StringIO()
    summary = run(**cond, out=buf, quiet=True)
    d = f"/runs/{batch}"
    os.makedirs(d, exist_ok=True)
    stem = run_stem(cond)
    with open(f"{d}/{stem}.jsonl", "w") as f:
        f.write(buf.getvalue())
    with open(f"{d}/{stem}.summary.json", "w") as f:
        json.dump(summary, f)
    runs_volume.commit()
    return summary


def run_stem(cond: dict) -> str:
    return f"{cond.get('label', 'run')}_L{cond.get('tape') or 16}_st{cond.get('z80_steps', 128)}_k{cond.get('noise_exp', 4)}_s{cond.get('seed', 0)}"


def _run_one_to_file(args: tuple) -> dict:
    """Subprocess body: run one condition, write its JSONL, return the summary."""
    cond, path = args
    import io

    from algocell_exp.run import run

    buf = io.StringIO()
    summary = run(**cond, out=buf, quiet=True)
    with open(path, "w") as f:
        f.write(buf.getvalue())
    return summary


@app.function(image=image, gpu=GPU, timeout=60 * 60 * 12, volumes={"/runs": runs_volume})
def run_batch(conds: list[dict], batch: str = "adhoc", procs: int = 8) -> list[dict]:
    """Run several conditions on one GPU in parallel worker processes. A single
    soup is Python-bound on an H200 (~0.35 ms/step with the GPU mostly idle), so
    independent processes multiply throughput until the GPU saturates."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    d = f"/runs/{batch}"
    os.makedirs(d, exist_ok=True)
    jobs = [(c, f"{d}/{run_stem(c)}.jsonl") for c in conds]
    with ProcessPoolExecutor(max_workers=min(procs, len(jobs))) as ex:
        results = list(ex.map(_run_one_to_file, jobs))
    runs_volume.commit()
    return results


@app.function(image=image, gpu=GPU, timeout=1800)
def bench_procs(steps: int = 3000, procs_list: tuple = (1, 4, 8, 16)) -> dict:
    """Aggregate throughput with N worker processes each stepping its own soup."""
    import time
    from concurrent.futures import ProcessPoolExecutor

    out = {}
    for n in procs_list:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n) as ex:
            times = list(ex.map(_bench_one, [(steps, i + 1) for i in range(n)]))
        wall = time.perf_counter() - t0
        out[n] = {"wall_s": round(wall, 2), "ms_per_soup_step_aggregate": round(1000 * wall / (n * steps), 3), "g_slots_per_s": round(n * steps * 8192 * 128 / wall / 1e9, 2), "max_single_s": round(max(times), 2)}
    return out


def _bench_one(args: tuple) -> float:
    import time

    n_steps, seed = args
    from algocell_exp.soup import Soup

    s = Soup(seed=seed)
    s.step(32)
    s.sync()
    t0 = time.perf_counter()
    s.step(n_steps)
    s.sync()
    return time.perf_counter() - t0


@app.function(image=image, gpu=GPU, timeout=900)
def bench_single(steps: int = 3000) -> dict:
    """One soup per GPU: ms/step for L=16 and L=100 (the sweep's two extremes)."""
    import time

    from algocell_exp.soup import Soup, adapter_summary

    out = {"adapter": adapter_summary()}
    for L in (16, 100):
        s = Soup(seed=1, tape_length=L)
        s.step(32)
        s.sync()
        t0 = time.perf_counter()
        s.step(steps)
        s.sync()
        dt = time.perf_counter() - t0
        out[f"L{L}_ms_per_step"] = round(1000 * dt / steps, 3)
    return out


@app.function(image=image, gpu=GPU, timeout=900)
def bench(steps: int = 2048) -> dict:
    """Single-soup speed, plus aggregate speed with N soups interleaved in one process."""
    import time

    from algocell_exp.soup import Soup, adapter_summary

    info = adapter_summary()
    out = {"adapter": info, "steps_per_soup": steps, "by_n_soups": {}}
    for n in (1, 2, 4, 8, 16):
        soups = [Soup(seed=i + 1) for i in range(n)]
        for s in soups:
            s.step(32)
        soups[0].sync()
        t0 = time.perf_counter()
        for _ in range(steps // 32):
            for s in soups:
                s.step(32)
        for s in soups:
            s.sync()
        dt = time.perf_counter() - t0
        tot = n * steps
        out["by_n_soups"][n] = {
            "ms_per_soup_step": round(1000 * dt / tot, 3),
            "g_slots_per_s": round(tot * 8192 * 128 / dt / 1e9, 2),
        }
    hashes = soups[0].read_hashes()
    out["sanity_unique_species"] = int(len(set(hashes.tolist())))
    return out


@app.local_entrypoint()
def gpus(bench_steps: int = 3000, gpu_list: str = "L4,A10G,L40S,H100,H200") -> None:
    """Latency per step across GPU types (the workload is latency-bound, so cheap GPUs may do)."""
    for g in gpu_list.split(","):
        try:
            print(g, bench_single.with_options(gpu=g).remote(bench_steps))
        except Exception as e:  # noqa: BLE001
            print(g, "error:", str(e)[:200])


@app.local_entrypoint()
def main(conds: str = "", batch: str = "adhoc", bench_steps: int = 3000, procs_bench: bool = False) -> None:
    import json

    if procs_bench:
        print(json.dumps(bench_procs.remote(bench_steps), indent=1))
        return
    if not conds:
        print(json.dumps(bench.remote(bench_steps), indent=1))
        return
    with open(conds) as f:
        condition_list = json.load(f)
    print(f"fanning out {len(condition_list)} conditions to {GPU} (batch={batch})")
    results = list(run_condition.map(condition_list, kwargs={"batch": batch}))
    out = f"runs/{batch}_summaries.json"
    import os

    os.makedirs("runs", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    for r in results:
        print(r["label"], "seed", r["seed"], "t_02", r["t_02"], "t_10", r["t_10"], "steps", r["steps_run"], f"{r['wall_s']}s", r["adapter"])
    print("wrote", out)
