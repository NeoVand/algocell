"""E on Modal (REVISION_PREREG E): the copy-offset switch competitions, one world per container.

    .venv/bin/modal run modal_offset.py --smoke
    .venv/bin/modal run modal_offset.py
    .venv/bin/modal volume get algocell-atlas-runs offset runs/ && .venv/bin/python offset_switch.py --analyse
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libvulkan1", "libx11-6", "libxext6", "libxcb1", "libegl1", "libgles2", "libglvnd0")
    .pip_install("wgpu==0.32.0", "numpy==2.5.3", "brotli==1.2.0", "pandas==3.0.6")
    .env({"WGPU_BACKEND_TYPE": "Vulkan"})
    .add_local_dir("algocell_exp", remote_path="/root/algocell_exp")
    .add_local_file("offset_switch.py", remote_path="/root/offset_switch.py")
)
runs_volume = modal.Volume.from_name("algocell-atlas-runs", create_if_missing=True)
app = modal.App("algocell-offset")


@app.function(image=image, gpu="L40S", timeout=2 * 60 * 60, volumes={"/runs": runs_volume}, retries=modal.Retries(max_retries=1, initial_delay=10.0))
def world(c: dict) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import offset_switch
    r = offset_switch.run_world(c, "/runs/offset")
    runs_volume.commit()
    return r


@app.local_entrypoint()
def main(smoke: bool = False):
    import offset_switch
    if smoke:
        print(world.remote({"L": 16, "tar": "lethal", "mut": "off", "start": "mix50", "seed": 99, "steps": 2000}))
        return
    for r in world.map(offset_switch.conditions(), return_exceptions=True):
        print(r if isinstance(r, dict) else f"FAILED: {r!r}"[:300], flush=True)
