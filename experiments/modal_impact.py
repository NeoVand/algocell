"""Sampled-input impact study on Modal; do not import another local app module.

From experiments, with profile neovand:
  python impact_modal.py prepare
  python impact_modal.py cross-local       # a single brief foreground GPU check
  modal run modal_impact.py::cross          # uploads only sampled inputs, compares exact bytes
  modal run --detach modal_impact.py::main  # Function.spawn, committed per-condition results
  modal volume get algocell-atlas-runs impact_dryrun/<bundle>/results results/impact_dryrun
"""
from __future__ import annotations

import modal

app = modal.App("algocell-impact")
# Repeated from modal_app.py: these libraries are required for the NVIDIA Vulkan ICD.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libvulkan1", "libx11-6", "libxext6", "libxcb1", "libegl1", "libgles2", "libglvnd0")
    .pip_install("wgpu==0.32.0", "numpy==2.5.3", "brotli==1.2.0")
    .env({"WGPU_BACKEND_TYPE": "Vulkan", "PYTHONPATH": "/root/experiments"})
    .add_local_dir("algocell_exp", remote_path="/root/experiments/algocell_exp")
    .add_local_file("impact_study.py", remote_path="/root/experiments/impact_study.py")
    .add_local_file("impact_modal.py", remote_path="/root/experiments/impact_modal.py")
)
vol = modal.Volume.from_name("algocell-atlas-runs", create_if_missing=True)


@app.function(image=image, gpu=["L40S", "A10G"], cpu=(2, 2), memory=(8192, 8192), timeout=300,
              scaledown_window=2, volumes={"/vol": vol}, retries=0)
def check(bundle: str):
    import hashlib
    import json
    import time
    from pathlib import Path
    import impact_modal as M
    from algocell_exp.soup import adapter_summary

    start = time.monotonic()
    gpu = M.gpu_type()
    folder = Path("/vol/impact_dryrun") / bundle
    m, a = M.read_inputs(folder)
    local = json.loads((folder / "cross_local.json").read_text())
    if local["code_sha256"] != M.code_identity():
        raise RuntimeError("cross-platform code identity differs")
    data, components = M.cross_bytes(m, a)
    identical = data == (folder / "cross_local.bin").read_bytes()
    result = {"identical": identical, "bytes": len(data), "pairs": m["n"], "cid": local["cid"],
              "sha256": hashlib.sha256(data).hexdigest(), "components": components,
              "adapter": adapter_summary(), "wall_s": time.monotonic() - start,
              "gpu": gpu, "rate_usd_s": M.RATES[gpu],
              "local_wall_s": local["wall_s"], "local_adapter": local["adapter"],
              "code_sha256": M.code_identity(), "inputs_sha256": m["inputs_sha256"]}
    M.dump(folder / "cross_remote.json", result)
    vol.commit()
    if not identical:
        raise RuntimeError("STOP: Metal/Vulkan results differ byte for byte")
    return result


@app.function(image=image, gpu=["L40S", "A10G"], cpu=(2, 2), memory=(8192, 8192), timeout=6600,
              scaledown_window=2, volumes={"/vol": vol}, retries=0)
def finish(bundle: str):
    from pathlib import Path
    import impact_modal as M
    return M.finish(Path("/vol/impact_dryrun") / bundle, vol.commit)


def bundle_name():
    import json
    from pathlib import Path
    m = json.loads(Path("results/impact_dryrun/input_manifest.json").read_text())
    local = json.loads(Path("results/impact_dryrun/cross_local.json").read_text())
    if local["inputs_sha256"] != m["inputs_sha256"]:
        raise RuntimeError("local cross-platform inputs are stale")
    # Identical sampled bytes can be rerun after a worker fix. Keep that run's
    # checkpoints separate rather than colliding with a completed older worker.
    return f"{m['inputs_sha256'][:16]}-{local['code_sha256'][:12]}"


@app.local_entrypoint()
def cross():
    import json
    from pathlib import Path
    bundle = bundle_name()
    with vol.batch_upload(force=True) as upload:
        for name in ("inputs.npz", "input_manifest.json", "cross_local.bin", "cross_local.json"):
            upload.put_file(f"results/impact_dryrun/{name}", f"impact_dryrun/{bundle}/{name}")
    result = check.remote(bundle)
    Path("results/impact_dryrun/cross_remote.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path
    gate = json.loads(Path("results/impact_dryrun/cross_remote.json").read_text())
    if not gate["identical"]:
        raise RuntimeError("cross-platform check has not passed")
    bundle = bundle_name()
    call = finish.spawn(bundle)
    record = {"function_call_id": call.object_id, "volume_path": f"impact_dryrun/{bundle}", "bundle": bundle}
    Path("results/impact_dryrun/modal_call.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record))
