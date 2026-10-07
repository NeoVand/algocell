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
    .pip_install("wgpu==0.32.0", "numpy==2.5.3", "brotli==1.1.0")  # pinned to the versions the local tests run against
    .env({"WGPU_BACKEND_TYPE": "Vulkan"})
    .add_local_dir("algocell_exp", remote_path="/root/algocell_exp")
)
runs_volume = modal.Volume.from_name("algocell-atlas-runs", create_if_missing=True)

GPU = "L40S"  # fastest per step in the latency-bound regime (modal run modal_app.py::gpus), and cheaper than H200


@app.function(image=image, gpu=GPU, timeout=60 * 60 * 6, volumes={"/runs": runs_volume}, retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run_condition(cond: dict, batch: str = "adhoc", provenance: dict | None = None) -> dict:
    """One condition → /runs/<batch>/{stem}.* on the volume. The file-writing path is
    algocell_exp.batch.run_to_dir, the same function preflight.py runs locally."""
    from algocell_exp.batch import run_to_dir

    summary = run_to_dir(cond, f"/runs/{batch}", provenance)
    runs_volume.commit()
    return summary


def run_stem(cond: dict) -> str:
    from algocell_exp.batch import run_stem as _stem

    return _stem(cond)


def _run_one_to_file(args: tuple) -> dict:
    """Subprocess body: run one condition, write its JSONL, return the summary."""
    cond, path = args
    import io

    from algocell_exp.run import run

    buf = io.StringIO()
    summary = run(**cond, out=buf, quiet=True)
    with open(path, "w") as f:
        f.write(buf.getvalue())
    return summary  # (run_batch path: no snapshots; the sweeps use run_condition)


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
def main(conds: str = "", batch: str = "adhoc", bench_steps: int = 3000, procs_bench: bool = False, force_batch: bool = False) -> None:
    import json

    if procs_bench:
        print(json.dumps(bench_procs.remote(bench_steps), indent=1))
        return
    if not conds:
        print(json.dumps(bench.remote(bench_steps), indent=1))
        return
    with open(conds) as f:
        condition_list = json.load(f)
    import hashlib
    import os
    import subprocess
    import time

    # The batch name must match the condition file (a typo would overwrite another stage's files).
    expected = os.path.splitext(os.path.basename(conds))[0].replace("_remaining", "")
    if batch != expected and not force_batch:
        raise SystemExit(f"--batch {batch!r} does not match the condition file {conds!r} (expected {expected!r}); pass --force-batch to override")
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True).stdout.strip())
    except Exception:  # noqa: BLE001
        git, dirty = None, None
    provenance = {
        "git_commit": git, "git_dirty_experiments": dirty, "conds_file": os.path.basename(conds),
        "conds_sha256_16": hashlib.sha256(open(conds, "rb").read()).hexdigest()[:16],
        "launched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "batch": batch, "gpu": GPU,
    }
    print(f"fanning out {len(condition_list)} conditions to {GPU} (batch={batch}, commit={git}{' DIRTY' if dirty else ''})")
    os.makedirs("runs", exist_ok=True)
    out = f"runs/{batch}_summaries.json"
    results, failures = [], []
    # return_exceptions=True: one failed container must not abort the sweep (the default raises in the
    # client, which disconnects the app and kills every in-flight run). Summaries are flushed as they
    # arrive so a client crash loses nothing that finished.
    for r in run_condition.map(condition_list, kwargs={"batch": batch, "provenance": provenance}, return_exceptions=True):
        if isinstance(r, Exception):
            failures.append(repr(r)[:500])
            print("FAILED:", failures[-1])
            continue
        results.append(r)
        print(r["label"], "L", r["tape_length"], "seed", r["seed"], "steps", r["steps_run"], "/", r["horizon"], "tq_10", r["tq_10"], f"{r['wall_s']}s", r["adapter"])
        with open(out, "w") as f:
            json.dump({"results": results, "failures": failures, "provenance": provenance}, f, indent=1)
    print(f"done: {len(results)} ok, {len(failures)} failed; wrote {out}")
