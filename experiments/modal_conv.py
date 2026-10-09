"""C on Modal (REVISION_PREREG R3/C): the convention-test soups of conv_soups.py, one world per container.

    .venv/bin/modal run modal_conv.py --smoke                 # one randreg world, 2,500 steps (dry run of this path)
    .venv/bin/modal run modal_conv.py                         # 20 randreg + 20 randsp worlds, 300,000 steps
    .venv/bin/modal volume get algocell-atlas-runs conv runs/  # then: conv_soups.py --analyse

The world runner is conv_soups.run_world, the same function the local path calls; outputs go to /runs/conv on the volume.
"""
from __future__ import annotations

import modal

# The image of modal_app.py (same pins, same Vulkan ICD recipe) plus the two scripts the world runner imports.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libvulkan1", "libx11-6", "libxext6", "libxcb1", "libegl1", "libgles2", "libglvnd0")
    .pip_install("wgpu==0.32.0", "numpy==2.5.3", "brotli==1.2.0", "pandas==3.0.6")
    .env({"WGPU_BACKEND_TYPE": "Vulkan"})
    .add_local_dir("algocell_exp", remote_path="/root/algocell_exp")
    .add_local_file("conv_soups.py", remote_path="/root/conv_soups.py")
    .add_local_file("threshold_grid.py", remote_path="/root/threshold_grid.py")
)
runs_volume = modal.Volume.from_name("algocell-atlas-runs", create_if_missing=True)
GPU = "L40S"
app = modal.App("algocell-conv")


@app.function(image=image, gpu=GPU, timeout=2 * 60 * 60, volumes={"/runs": runs_volume}, retries=modal.Retries(max_retries=1, initial_delay=10.0))
def world(v: str, seed: int, steps: int) -> dict:
    import os
    import sys
    import time
    sys.path.insert(0, "/root")
    import conv_soups
    conv_soups.RUNS = "/runs/conv"
    conv_soups.SNAPS = tuple(s for s in (20000, 100000, 300000) if s <= steps)
    t0 = time.time()
    conv_soups.run_world(v, seed, steps)
    runs_volume.commit()
    from algocell_exp.assay import get_device
    return {"variant": v, "seed": seed, "steps": steps, "wall_s": round(time.time() - t0, 1), "adapter": str(get_device().adapter.info.get("device", "?")) if hasattr(get_device(), "adapter") else "?",
            "files": sorted(f for f in os.listdir("/runs/conv") if f.startswith(f"{v}_s{seed}"))}


@app.local_entrypoint()
def main(smoke: bool = False, steps: int = 300000):
    if smoke:
        print(world.remote("randreg", 9999, 2500))
        return
    jobs = [(v, s, steps) for s in range(8001, 8021) for v in ("randreg", "randsp")]
    for r in world.starmap(jobs, return_exceptions=True):
        print(r if isinstance(r, dict) else f"FAILED: {r!r}"[:300], flush=True)
