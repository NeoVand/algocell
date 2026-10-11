"""Finish a checkpointed 6502 census on a Modal GPU (the local M4 is the user's working machine).

    cd experiments
    modal run --detach modal_census.py --machine m6502_brkhalt_maps_L16 --k 4

The local checkpoint (results/census6502/ckpt/<machine>_k<k>_r16_b128-512.jsonl) is sent with the call. The container:

1. Re-runs two chunks the checkpoint already holds and requires identical records. This is the cross-platform check
   (Metal locally, Vulkan on NVIDIA here); the run stops if they differ.
2. Continues the census with census6502.run_census in 5-minute slices.
3. Copies the checkpoint to the volume after every slice and commits.

If the container dies, or the client goes away, the volume holds every finished chunk, and a rerun resumes from the
local copy once it is fetched (`modal volume get algocell-atlas-runs census6502/<file>`). The report is regenerated
locally with `python census6502.py report`.
"""

from __future__ import annotations

import modal

app = modal.App("algocell-census6502")

# The Vulkan recipe of modal_app.py (repeated here: a Modal app file must not import another local module).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libvulkan1", "libx11-6", "libxext6", "libxcb1", "libegl1", "libgles2", "libglvnd0")
    .pip_install("wgpu==0.32.0", "numpy==2.5.3", "brotli==1.2.0")
    .env({"WGPU_BACKEND_TYPE": "Vulkan"})
    .add_local_dir("algocell_exp", remote_path="/root/experiments/algocell_exp")
    .add_local_file("census6502.py", remote_path="/root/experiments/census6502.py")
)
vol = modal.Volume.from_name("algocell-atlas-runs", create_if_missing=True)


@app.function(image=image, gpu="L40S", timeout=3 * 60 * 60, volumes={"/vol": vol})
def finish(machine: str, k: int, ckpt_name: str, ckpt_text: str, check_chunks: list[int]) -> dict:
    import json
    import shutil

    import numpy as np
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/experiments")
    import census6502 as C
    from algocell_exp.soup import adapter_summary

    adapter = adapter_summary()
    print("adapter:", adapter, flush=True)
    C.CKPT.mkdir(parents=True, exist_ok=True)
    local = C.CKPT / ckpt_name
    local.write_text(ckpt_text)
    dest = Path("/vol/census6502")
    dest.mkdir(parents=True, exist_ok=True)

    # 1. Cross-platform check: recompute chunks the local (Metal) run already holds and compare record for record.
    lines = [json.loads(x) for x in ckpt_text.splitlines() if x.strip()]
    have = {x["i"]: x for x in lines[1:] if x["type"] == "chunk"}
    c = C.TileCensus(machine, k=k, nrand=C.NRAND, budgets=C.BUDGETS, seg=C.SEGMENT, sample_bits=C.sample_bits_for(k), panel=False, variant_major=True)
    if c.describe().get("harness_wgsl_sha256") != lines[0].get("harness_wgsl_sha256"):
        raise RuntimeError("harness WGSL differs from the local run's; not resuming")
    checks = []
    for ci in check_chunks:
        ref = have[ci]
        out = c.run_range(ref["t0"], ref["t1"])
        for b in c.budgets:
            # Counters must be identical. Hits are appended through an atomic counter, so their order is the GPU's
            # scheduling order; they must be identical as a set of rows.
            got_counters = [out[b]["counters"][n] for n in c.counter_names]
            ref_counters = ref["budgets"][str(b)]["counters"]
            diff = {n: (r, g) for n, r, g in zip(c.counter_names, ref_counters, got_counters) if r != g}
            sort = lambda H: H[np.lexsort(H.T[::-1])] if len(H) else H  # noqa: E731
            H_ref, H_got = sort(C._unpack(ref["budgets"][str(b)]["hits"])), sort(out[b]["hits"])
            same_hits = H_ref.shape == H_got.shape and bool((H_ref == H_got).all())
            checks.append({"chunk": ci, "budget": b, "counters_identical": not diff, "hits_identical_as_set": same_hits,
                           "n_hits": [int(len(H_ref)), int(len(H_got))], "counter_diffs": dict(list(diff.items())[:20])})
            if diff or not same_hits:
                raise RuntimeError(f"chunk {ci} budget {b} differs from the local Metal run: {checks[-1]}")
    print("cross-platform check:", checks, flush=True)

    # 2. Continue in slices, persisting after each.
    total = 256 ** k
    while True:
        C.run_census(machine, k, max_minutes=5, pause=0)
        shutil.copy(C.ckpt_path(machine, k, C.NRAND, c.budgets), dest / ckpt_name)
        vol.commit()
        done = {json.loads(x)["i"] for x in local.read_text().splitlines()[1:] if x.strip()}
        print(f"{len(done)} chunks on the volume", flush=True)
        if len(done) * min(total, 1 << 24) >= total:
            break
    (dest / f"{ckpt_name}.done.json").write_text(json.dumps({"adapter": adapter, "checks": checks, "chunks": len(done)}))
    vol.commit()
    return {"adapter": adapter, "checks": checks, "chunks": len(done)}


@app.local_entrypoint()
def main(machine: str = "m6502_brkhalt_maps_L16", k: int = 4, checks: str = "0,200"):
    from pathlib import Path

    name = f"{machine}_k{k}_r16_b128-512.jsonl"
    text = (Path("results/census6502/ckpt") / name).read_text()
    call = finish.spawn(machine, k, name, text, [int(x) for x in checks.split(",")])
    print(f"spawned {call.object_id}; poll the volume for census6502/{name}.done.json")
