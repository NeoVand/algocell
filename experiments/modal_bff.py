"""Run BFF soups (micro/bff_soup.py) on Modal GPUs, one soup per container.

    .venv/bin/modal run modal_bff.py --smoke                       # one tiny soup through the exact path
    .venv/bin/modal run --detach modal_bff.py --seeds 13-24 --variants std,wrap --batch bff

Writes /runs/<batch>/<variant>_s<seed>/ (cond.json, epochs.csv, samples.jsonl, soup_*.u8.br, summary.json) on the
'algocell-atlas-runs' volume. Same image recipe as modal_app.py (pinned wgpu/numpy/brotli, NVIDIA Vulkan ICD libs) plus
the micro package; the recipe is repeated here rather than imported because only this file is mounted in the container.
"""

from __future__ import annotations

import modal

app = modal.App("algocell-bff")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libvulkan1", "libx11-6", "libxext6", "libxcb1", "libegl1", "libgles2", "libglvnd0")
    .pip_install("wgpu==0.32.0", "numpy==2.5.3", "brotli==1.2.0")
    .env({"WGPU_BACKEND_TYPE": "Vulkan"})
    .add_local_dir("algocell_exp", remote_path="/root/algocell_exp")
    .add_local_dir("micro", remote_path="/root/micro")
)
runs_volume = modal.Volume.from_name("algocell-atlas-runs", create_if_missing=True)
GPU = "L40S"


# A 16,384-epoch soup of 2^17 programs takes ≈ 45–80 min (CPU-bound host loop); 3 h bounds a slow container.
@app.function(image=image, gpu=GPU, cpu=2.0, timeout=3 * 60 * 60, volumes={"/runs": runs_volume}, retries=modal.Retries(max_retries=1, initial_delay=10.0))
def run_soup(seed: int, ip_wrap: bool, density: int, n: int, epochs: int, batch: str, sample_every: int = 64) -> dict:
    import json
    import os

    from micro.bff_soup import run

    variant = ("wrap" if ip_wrap else "std") + (f"_d{density}" if density > 1 else "")
    out = f"/runs/{batch}/{variant}_s{seed}"
    if os.path.exists(os.path.join(out, "summary.json")):
        return json.load(open(os.path.join(out, "summary.json")))
    snaps = [e for e in (1024, 2048, 4096, 8192) if e < epochs]
    summary = run(out, n, epochs, seed, ip_wrap, density, sample_every, snaps)
    runs_volume.commit()
    return summary


def parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


@app.local_entrypoint()
def main(seeds: str = "13-24", variants: str = "std,wrap", density: int = 1, n: int = 1 << 17, epochs: int = 16384, batch: str = "bff", smoke: bool = False) -> None:
    import json
    import time

    if smoke:
        t0 = time.time()
        s = run_soup.remote(999, False, 1, 4096, 64, "bff_smoke", sample_every=32)
        print("smoke ok:", json.dumps(s), f"{time.time() - t0:.0f}s", flush=True)
        return
    jobs = [(sd, v == "wrap", density, n, epochs, batch) for v in variants.split(",") for sd in parse_seeds(seeds)]
    print(f"fanning out {len(jobs)} soups to {GPU} (batch={batch}, n={n}, epochs={epochs}, density={density})", flush=True)
    done = 0
    for s in run_soup.starmap(jobs, order_outputs=False, return_exceptions=True):
        done += 1
        if isinstance(s, Exception):
            print(f"[{done}/{len(jobs)}] FAILED: {s!r}", flush=True)
        else:
            print(f"[{done}/{len(jobs)}] {'wrap' if s['ip_wrap'] else 'std'} seed {s['seed']} t_rep {s['t_rep']} {s['elapsed_s']:.0f}s", flush=True)
    print("all soups returned", flush=True)
