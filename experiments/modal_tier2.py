"""I1-T2 preparation only: C0 equivalence, C1 smoke, and throughput. NO registered launcher.

Run from experiments/: modal run modal_tier2.py --action validate|trace|variants|smoke|bench|mature|l4
All world horizons are hard-bounded here. Outputs go only to tier2_smoke/.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

FILES = ("dial_soups.py", "offset_switch.py", "invasion_generic.py", "threshold_grid.py",
         "population_classes.py", "mutscan.py", "make_conds.py", "stats_tests.py")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libvulkan1", "libx11-6", "libxext6", "libxcb1", "libegl1", "libgles2", "libglvnd0")
         .pip_install("wgpu==0.32.0", "numpy==2.5.3", "brotli==1.2.0", "pandas==3.0.6")
         .env({"WGPU_BACKEND_TYPE": "Vulkan", "XDG_RUNTIME_DIR": "/tmp"})
         .add_local_dir("algocell_exp", remote_path="/root/algocell_exp")
         .add_local_dir("conds", remote_path="/root/conds"))
for f in FILES:
    image = image.add_local_file(f, remote_path=f"/root/{f}")
volume = modal.Volume.from_name("algocell-atlas-runs")
app = modal.App("algocell-tier2-preparation")
OUT = Path("results/tier2")


@app.function(image=image, gpu="L40S", timeout=900, volumes={"/runs": volume})
def validate(tag: str) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    from algocell_exp.impact import compare_c0
    from algocell_exp.impact_worlds import WORLDS
    from algocell_exp.tier2 import assay_name
    from algocell_exp.zmachine import load_machine
    from algocell_exp.zsoup_check import Lockstep

    w = WORLDS[tag]
    v = tag.split("_L")[0] if w.rule in ("dial", "randreg") else None
    checker = Lockstep(L=w.L, seed=424242, zero_halts=w.lethal, shader_variant=v,
                       mem_length=w.P, machine=w.machine("c0"), steps_capacity=128)
    lock = checker.run(128, dense_until=4, every=32, checkpoint_every=128)
    assert lock["pair_mem_mismatches"] == lock["pair_wc_mismatches"] == lock["cpu_gpu_disagree"] == 0, lock
    assert lock["active_changed"] > 0 and lock["writes"] > 0, lock
    pairs = np.random.default_rng(424242).integers(0, 256, (65536, 2 * w.L), dtype=np.uint8)
    pairs[np.random.default_rng(42).random(pairs.shape) < 0.25] = 0
    mapped = compare_c0(w, load_machine(assay_name(tag, "c0")), pairs, 128, seed=232323)
    assert mapped["mem_mismatch"] == mapped["wc_mismatch"] == mapped["reg_mismatch"] == 0, mapped
    neg = compare_c0(w, load_machine(w.machine("c0")), pairs[:8192], 128, seed=232323, budget_delta=-1)
    assert neg["mem_mismatch"] + neg["wc_mismatch"] + neg["reg_mismatch"] > 0, neg
    return {"tag": tag, "lockstep": lock, "mapped_pairs": mapped, "negative_control": neg}


@app.function(image=image, gpu="L40S", timeout=900)
def trace_validate(tag: str) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    from algocell_exp import conv, exectrace, executor
    from algocell_exp.impact_worlds import WORLDS
    from algocell_exp.tier2 import assay_name
    from algocell_exp.zmachine import load_machine
    w = WORLDS[tag]
    m = load_machine(assay_name(tag, "c0"))
    pairs = np.random.default_rng(31337).integers(0, 256, (4096, 2 * w.L), dtype=np.uint8)
    pairs[np.random.default_rng(17).random(pairs.shape) < 0.25] = 0
    for budget in (0, 1, 128):
        if w.rule in ("dial", "randreg"):
            v = tag.split("_L")[0]
            memory, masks = conv.execute(pairs, w.L, budget, v, 987654, traced=True)
            routed, routed_masks = conv.execute(pairs, w.L, budget, v, 987654, traced=True, machine="c0")
            assert np.array_equal(memory, routed) and np.array_equal(masks, routed_masks)
        else:
            memory, masks = exectrace.execute_pairs_traced(pairs, w.L, budget, mem_length=w.P, zero_halts=w.lethal)
        r = m.run(pairs, budget=budget, seed=987654)
        assert np.array_equal(memory, r.memory)
        assert np.array_equal(masks, executor.pack_fetch_masks(r.fetch_map))
    return {"tag": tag, "pairs": 3 * len(pairs), "memory_mismatches": 0, "fetch_mismatches": 0,
            "budgets": [0, 1, 128], "seed": 987654}


@app.function(image=image, gpu="L40S", timeout=900)
def variant_validate(tag: str) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    from algocell_exp.impact_worlds import WORLDS
    from algocell_exp.impact import compare_c0
    from algocell_exp.zmachine import load_machine
    from algocell_exp.zsoup_check import Lockstep
    from algocell_exp.tier2 import assay_name
    from make_conds import ABLATIONS
    from algocell_exp.isa import masks, resolve
    w = WORLDS[tag]
    t = Lockstep(L=w.L, zero_halts=w.lethal, machine=w.machine("c0"), seed=424242,
                 mutations_per_step=0, steps_capacity=128)
    result = t.run(128, dense_until=4, every=32, checkpoint_every=128)
    assert result["pair_mem_mismatches"] == result["pair_wc_mismatches"] == result["cpu_gpu_disagree"] == 0
    ablated = None
    if tag == "L32":
        p = np.random.default_rng(981).integers(0, 256, (65536, 64), dtype=np.uint8)
        ablated = compare_c0(w, load_machine(assay_name(tag, "c0")), p, 128,
                             masks=masks(resolve(list(ABLATIONS["i8080"]))), seed=2)
        assert ablated["mem_mismatch"] == ablated["wc_mismatch"] == ablated["reg_mismatch"] == 0
    return {"tag": tag, "mutation_off_lockstep": result, "i8080_masked_pairs": ablated}


@app.function(image=image, gpu="L40S", timeout=900, volumes={"/runs": volume})
def mature_bench(tag: str) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import brotli
    import numpy as np
    import time
    from algocell_exp.zsoup import ZSoup
    s = ZSoup(f"imp_{tag}_c1", seed=424242)
    records = []
    for seed in (6001, 6003, 6006):
        paths = sorted(Path("/runs/stageL").glob(f"**/none@closure10M_{tag}_st128_k4_s{seed}.soup_final.u8.br"))
        if not paths:
            raise FileNotFoundError(f"Archived Modal Stage L {tag} seed {seed}")
        p = paths[0]
        cells = np.frombuffer(brotli.decompress(p.read_bytes()), dtype=np.uint8).reshape(20000, s.tape_length)
        s.write_soup(cells)
        s.step(256)
        s.sync()
        rates = []
        for _ in range(3):
            t0 = time.perf_counter()
            s.step(8192)
            s.sync()
            rates.append(8192 / (time.perf_counter() - t0))
        records.append({"archive": str(p), "steps_per_s": rates})
    return {"tag": tag, "records": records, "note": "C0 archived endpoints used only as performance inputs to C1; no endpoint scored"}


@app.function(image=image, gpu="L40S", timeout=900, volumes={"/runs": volume})
def smoke(experiment: str) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import time
    import numpy as np
    import brotli
    from algocell_exp.tier2 import machine_for, tag_for
    from algocell_exp.zmachine import load_machine

    out = Path("/runs/tier2_smoke") / experiment
    t0 = time.perf_counter()
    if experiment == "DZ":
        import dial_soups as D
        result = D.run_world("lp01", 424242, 2000, str(out), machine="c1")
        records = [json.loads(x) for x in (out / "lp01_s424242.jsonl").read_text().splitlines()]
        soup = np.load(out / "lp01_s424242_t2000.npy")
        assert all(len(r["top3"]) == len(r["shares"]) == 3 for r in records[1:])
        tag = "lp01_L16"
    elif experiment in ("E", "R4"):
        if experiment == "E":
            import offset_switch as R
            c = {"L": 16, "tar": "lethal", "mut": "on", "start": "mix50", "seed": 424242, "steps": 2000}
            tag, fields = "lethal_L16", ("core", "R", "T", "T_distinct_tails")
        else:
            import invasion_generic as R
            c = {"L": 16, "variant": "randreg", "resident": "pusher", "invader": "SI13", "seed": 424242, "steps": 2000}
            tag, fields = "randreg_L16", ("pusher", "SI13", "R4")
        result = R.run_world(c, str(out), machine="c1")
        records = [json.loads(x) for x in (out / (R.stem(c) + ".jsonl")).read_text().splitlines()]
        soup = np.load(out / (R.stem(c) + "_final.npy"))
        assert all(all(k in r for k in fields) for r in records[1:])
    else:
        from algocell_exp.batch import run_to_dir, run_stem, check_outputs
        source = "stageL" if experiment == "L" else "stageF"
        candidates = json.loads(Path(f"/root/conds/{source}.json").read_text())
        c = next(x for x in candidates if x["tape"] == (20 if experiment == "L" else 8))
        c = dict(c, seed=424242, horizon=2000, snapshot_steps=[500, 1000, 2000])
        tag = tag_for(c["tape"], P=c.get("mem_length"))
        c["machine"] = machine_for("c1", tag)
        result = run_to_dir(c, str(out))
        assert check_outputs(c, str(out)) == []
        records = [json.loads(x) for x in (out / (run_stem(c) + ".jsonl")).read_text().splitlines()]
        soup = np.frombuffer(brotli.decompress((out / (run_stem(c) + ".soup_final.u8.br")).read_bytes()), dtype=np.uint8).reshape(20000, c["tape"])
    L = soup.shape[1]
    assert soup.shape == (20000, L) and soup.dtype == np.uint8
    m = load_machine(machine_for("c1", tag, maps=True))
    probe = m.run(np.concatenate([soup[:16], soup[16:32]], axis=1), budget=128, seed=232323)
    assert probe.memory.shape == (16, 2 * L) and probe.fetch_map.shape == (16, m.ring_bytes)
    checks = {"experiment": experiment, "tag": tag, "steps": 2000, "soup_shape": list(soup.shape),
              "records": len(records), "mapped_assay_pairs": 16, "shape_ok": True,
              "wall_s": time.perf_counter() - t0, "runner_wall_s": result["wall_s"],
              "prefix": str(out), "world_machine": machine_for("c1", tag), "assay_machine": m.describe()}
    (out / "shape_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    volume.commit()
    return checks


@app.function(image=image, gpu="L40S", timeout=900, volumes={"/runs": volume})
def bench(tag: str) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import time
    from algocell_exp.zsoup import ZSoup
    s = ZSoup(f"imp_{tag}_c1", seed=424242)
    s.step(256)
    s.sync()
    rates = []
    for _ in range(3):
        t0 = time.perf_counter()
        s.step(4096)
        s.sync()
        rates.append(4096 / (time.perf_counter() - t0))
    return {"tag": tag, "steps_per_s": rates, "state": "fresh random soup, evolving during timing",
            "warmup": 256, "timed_steps_per_rep": 4096, "machine": s.machine_provenance}


@app.local_entrypoint()
def main(action: str = "smoke"):
    OUT.mkdir(parents=True, exist_ok=True)
    if action in ("validate", "trace"):
        from algocell_exp.impact_worlds import WORLDS
        specs = json.loads(Path("algocell_exp/machines/worlds_tier2.json").read_text())
        tags = sorted({next(t for t, w in WORLDS.items() if w.spec()["memory"] == s["world"]["memory"] and
                            w.spec().get("hooks") == s["world"].get("hooks") and
                            w.spec().get("halting") == s["world"].get("halting")) for s in specs.values()})
        results = list((validate if action == "validate" else trace_validate).map(tags))
    elif action == "smoke":
        results = list(smoke.map(["DZ", "E", "L", "R4", "F"]))
    elif action == "bench":
        tags = ["lp0_L16", "lp001_L16", "lp003_L16", "lp01_L16", "lp03_L16", "lp1_L16",
                "L16", "lethal_L16", "L32", "lethal_L32", "randreg_L16", "L20", "L8_P32"]
        results = list(bench.map(tags))
    elif action == "mature":
        results = list(mature_bench.map(["L16", "L20"]))
    elif action == "variants":
        results = list(variant_validate.map(["L16", "lethal_L16", "L32"]))
    elif action == "l4":
        tags = ["lp0_L16", "lp001_L16", "lp003_L16", "lp01_L16", "lp03_L16", "lp1_L16",
                "L16", "lethal_L16", "L32", "lethal_L32", "randreg_L16", "L20", "L8_P32"]
        results = {"fresh": list(bench.with_options(gpu="L4").map(tags)),
                   "mature": list(mature_bench.with_options(gpu="L4").map(["L16", "L20"]))}
    else:
        raise ValueError("Only preparation actions validate, trace, variants, smoke, bench, mature, l4 are available")
    (OUT / f"{action}.json").write_text(json.dumps(results, indent=2) + "\n")
    count = len(results["fresh"]) + len(results["mature"]) if isinstance(results, dict) else len(results)
    print(f"{action}: {count} completed; results in {OUT / (action + '.json')}")
