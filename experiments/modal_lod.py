"""N3 on Modal (REVISION_PREREG N3): the exact line of descent, one world per container.

    .venv/bin/modal run modal_lod.py --smoke
    .venv/bin/modal run modal_lod.py
    .venv/bin/modal volume get --force algocell-atlas-runs lod runs/lod_modal/
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libvulkan1", "libx11-6", "libxext6", "libxcb1", "libegl1", "libgles2", "libglvnd0")
    .pip_install("wgpu==0.32.0", "numpy==2.5.3", "brotli==1.2.0", "pandas==3.0.6")
    .env({"WGPU_BACKEND_TYPE": "Vulkan"})
    .add_local_dir("algocell_exp", remote_path="/root/algocell_exp")
    .add_local_file("lod.py", remote_path="/root/lod.py")
)
runs_volume = modal.Volume.from_name("algocell-atlas-runs", create_if_missing=True)
app = modal.App("algocell-lod")


@app.function(image=image, gpu=["L4", "A10G", "L40S"], cpu=2.0, memory=8192, timeout=3 * 60 * 60, volumes={"/runs": runs_volume},
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def world(seed: int, cap: int = 80000, outdir: str = "lod") -> dict:
    import sys
    sys.path.insert(0, "/root")
    import lod
    r = lod.run(seed, f"/runs/{outdir}", cap=cap)
    runs_volume.commit()
    return r


@app.local_entrypoint()
def main(smoke: bool = False, first: int = 31001, n: int = 20, outdir: str = "lod"):
    if smoke:
        print(world.remote(39999, cap=600))
        return
    seeds = list(range(first, first + n))
    for r in world.map(seeds, kwargs={"outdir": outdir}, order_outputs=False, return_exceptions=True):
        print(r if isinstance(r, dict) else f"FAILED: {r!r}"[:300], flush=True)
