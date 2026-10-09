"""DZ (lethality dial) and C4a-rep (random-register replication) on Modal. Skips worlds whose final snapshot is on the
volume. Records the GPU model of every world.

    .venv/bin/modal run modal_dial.py [--part dial|rep|all] [--dry]
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libvulkan1", "libx11-6", "libxext6", "libxcb1", "libegl1", "libgles2", "libglvnd0")
    .pip_install("wgpu==0.32.0", "numpy==2.5.3", "brotli==1.2.0", "pandas==3.0.6")
    .env({"WGPU_BACKEND_TYPE": "Vulkan"})
    .add_local_dir("algocell_exp", remote_path="/root/algocell_exp")
    .add_local_file("conv_soups.py", remote_path="/root/conv_soups.py")
    .add_local_file("threshold_grid.py", remote_path="/root/threshold_grid.py")
    .add_local_file("dial_soups.py", remote_path="/root/dial_soups.py")
)
runs_volume = modal.Volume.from_name("algocell-atlas-runs", create_if_missing=True)
app = modal.App("algocell-dial")


def _gpu():
    import subprocess
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e!r})"


@app.function(image=image, gpu=["L40S", "A10G", "L4"], timeout=60 * 60, volumes={"/runs": runs_volume}, retries=modal.Retries(max_retries=1, initial_delay=10.0))
def dial_world(v: str, seed: int, steps: int, outdir: str = "/runs/dial") -> dict:
    import os
    import sys
    sys.path.insert(0, "/root")
    import dial_soups
    if os.path.exists(os.path.join(outdir, f"{v}_s{seed}_t{steps}.npy")):
        return {"variant": v, "seed": seed, "skipped": True}
    r = dial_soups.run_world(v, seed, steps, outdir)
    r["gpu"] = _gpu()
    with open(os.path.join(outdir, f"{v}_s{seed}.gpu"), "w") as f:
        f.write(r["gpu"] + "\n")
    runs_volume.commit()
    return r


@app.function(image=image, gpu=["L40S", "A10G", "L4"], timeout=3 * 60 * 60, volumes={"/runs": runs_volume}, retries=modal.Retries(max_retries=1, initial_delay=10.0))
def rep_world(seed: int, steps: int) -> dict:
    import os
    import sys
    import time
    sys.path.insert(0, "/root")
    import conv_soups
    conv_soups.RUNS = "/runs/conv_rep"
    conv_soups.SNAPS = (20000, 100000, 300000, 1000000, 2000000, 3000000)
    if os.path.exists(os.path.join(conv_soups.RUNS, f"randreg_s{seed}_t{steps}.npy")):
        return {"seed": seed, "skipped": True}
    t0 = time.time()
    conv_soups.run_world("randreg", seed, steps)
    gpu = _gpu()
    with open(os.path.join(conv_soups.RUNS, f"randreg_s{seed}.gpu"), "w") as f:
        f.write(gpu + "\n")
    runs_volume.commit()
    return {"variant": "randreg", "seed": seed, "steps": steps, "wall_s": round(time.time() - t0, 1), "gpu": gpu}


@app.local_entrypoint()
def main(part: str = "all", dry: bool = False):
    import dial_soups
    if dry:
        print(dial_world.remote("lp01", 9999, 2000, "/runs/dial_dry"))
        return
    handles = []
    if part in ("all", "rep"):
        handles.append(("rep", rep_world.starmap([(s, 3000000) for s in range(8201, 8221)], return_exceptions=True)))
    if part in ("all", "dial"):
        jobs = [(v, s, 300000) for v in dial_soups.DIAL for s in dial_soups.SEEDS]
        handles.append(("dial", dial_world.starmap(jobs, return_exceptions=True)))
    for name, it in handles:
        for r in it:
            print(name, r if isinstance(r, dict) else f"FAILED: {r!r}"[:300], flush=True)
