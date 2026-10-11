"""The 6502 pilot (PREREG_6502_DRAFT.md: predictions locked 2026-10-11) on Modal GPUs, one world per container.

    cd experiments
    modal run --detach modal_pilot6502.py            # spawns the 8 worlds (L = 16 and 32, seeds 1-4, 100,000 steps)
    modal volume get algocell-atlas-runs pilot6502 results/   # when every <world>.jsonl has its summary line

Each world is pilot6502.run_world, unchanged, writing straight into the volume (no GPU sharing pauses: the GPU is the
container's). A world is never overwritten, so the function has no retries.
"""

from __future__ import annotations

import modal

app = modal.App("algocell-pilot6502")

# The Vulkan recipe of modal_app.py, repeated (a Modal app file must not import another local module).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libvulkan1", "libx11-6", "libxext6", "libxcb1", "libegl1", "libgles2", "libglvnd0")
    .pip_install("wgpu==0.32.0", "numpy==2.5.3", "brotli==1.2.0")
    .env({"WGPU_BACKEND_TYPE": "Vulkan"})
    .add_local_dir("algocell_exp", remote_path="/root/experiments/algocell_exp")
    .add_local_file("pilot6502.py", remote_path="/root/experiments/pilot6502.py")
)
vol = modal.Volume.from_name("algocell-atlas-runs", create_if_missing=True)


@app.function(image=image, gpu="L40S", timeout=2 * 60 * 60, volumes={"/vol": vol})
def world(L: int, seed: int, steps: int = 100_000) -> dict:
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/experiments")
    import pilot6502 as P
    from algocell_exp.soup import adapter_summary

    print("adapter:", adapter_summary(), flush=True)
    summary = P.run_world(L, seed, steps, Path("/vol/pilot6502"), burst=0.0)
    vol.commit()
    return summary


@app.local_entrypoint()
def main(steps: int = 100_000):
    for L in (16, 32):
        for seed in (1, 2, 3, 4):
            call = world.spawn(L, seed, steps)
            print(f"spawned L={L} seed={seed}: {call.object_id}")
