"""R4 on Modal: C4a (long random-register worlds, conv_soups.run_world) and C4b/C4c (invasion_generic.run_world).

    .venv/bin/modal run modal_r4.py
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
    .add_local_file("invasion_generic.py", remote_path="/root/invasion_generic.py")
)
runs_volume = modal.Volume.from_name("algocell-atlas-runs", create_if_missing=True)
app = modal.App("algocell-r4")


@app.function(image=image, gpu=["L40S", "A10G", "L4"], timeout=3 * 60 * 60, volumes={"/runs": runs_volume}, retries=modal.Retries(max_retries=1, initial_delay=10.0))
def long_world(v: str, seed: int, steps: int) -> dict:
    import sys
    import time
    sys.path.insert(0, "/root")
    import conv_soups
    conv_soups.RUNS = "/runs/conv_long"
    conv_soups.SNAPS = (20000, 100000, 300000, 1000000, 2000000, 3000000)
    t0 = time.time()
    conv_soups.run_world(v, seed, steps)
    runs_volume.commit()
    return {"variant": v, "seed": seed, "steps": steps, "wall_s": round(time.time() - t0, 1)}


@app.function(image=image, gpu=["L40S", "A10G", "L4"], timeout=2 * 60 * 60, volumes={"/runs": runs_volume}, retries=modal.Retries(max_retries=1, initial_delay=10.0))
def inv_world(c: dict) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import invasion_generic
    r = invasion_generic.run_world(c, "/runs/r4_inv")
    runs_volume.commit()
    return r


@app.local_entrypoint()
def main(part: str = "all"):
    import invasion_generic
    calls = []
    if part in ("all", "inv"):
        for r in inv_world.map(invasion_generic.conditions(), return_exceptions=True):
            print(r if isinstance(r, dict) else f"FAILED: {r!r}"[:300], flush=True)
    if part in ("all", "long"):
        jobs = [("randreg", s, 3000000) for s in range(8101, 8121)]
        for r in long_world.starmap(jobs, return_exceptions=True):
            print(r if isinstance(r, dict) else f"FAILED: {r!r}"[:300], flush=True)
