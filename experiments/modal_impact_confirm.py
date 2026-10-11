"""Only the detached registered I1 worker; no imports of any other Modal app."""
from __future__ import annotations

import modal

app = modal.App("algocell-impact-i1")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libvulkan1", "libx11-6", "libxext6", "libxcb1", "libegl1", "libgles2", "libglvnd0")
         .pip_install("wgpu==0.32.0", "numpy==2.5.3", "brotli==1.2.0")
         .env({"WGPU_BACKEND_TYPE": "Vulkan", "PYTHONPATH": "/root/experiments"})
         .add_local_dir("algocell_exp", remote_path="/root/experiments/algocell_exp")
         .add_local_file("impact_study.py", remote_path="/root/experiments/impact_study.py")
         .add_local_file("impact_modal.py", remote_path="/root/experiments/impact_modal.py")
         .add_local_file("impact_confirm.py", remote_path="/root/experiments/impact_confirm.py"))
vol = modal.Volume.from_name("algocell-atlas-runs")


@app.function(image=image, gpu=["L40S", "A10G"], cpu=(2, 2), memory=(8192, 8192),
              timeout=12200, scaledown_window=2, volumes={"/vol": vol}, retries=0)
def finish(bundle: str):
    from pathlib import Path
    import impact_confirm as C
    return C.finish(Path("/vol/impact_i1") / bundle, vol.commit)


@app.local_entrypoint()
def main():
    import hashlib
    import json
    from pathlib import Path
    import impact_confirm as C
    C.gate()
    p = Path("results/impact/input_manifest.json")
    m = json.loads(p.read_text())
    if C.source_hash() != m["code_sha256"]:
        raise RuntimeError("prepare again: source changed")
    if m["cost_estimate"]["estimate_usd"] > 8 or m["cost_estimate"]["allocation_ceiling_usd"] > 8:
        raise RuntimeError("STOP: $8 cost gate")
    record = Path("results/impact/modal_call.json")
    if record.exists():
        raise RuntimeError("already launched: no implicit duplicate spend")
    bundle = hashlib.sha256(p.read_bytes()).hexdigest()[:20]
    with vol.batch_upload() as upload:
        upload.put_file(p, f"impact_i1/{bundle}/input_manifest.json")
        upload.put_directory("results/impact/inputs", f"impact_i1/{bundle}/inputs")
    call = finish.spawn(bundle)
    info = dict(function_call_id=call.object_id, volume_path=f"impact_i1/{bundle}", bundle=bundle,
                code_sha256=m["code_sha256"])
    record.write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))
