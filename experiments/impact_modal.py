"""Sample-only transport and resumable Modal worker for the impact dry run.

Local CPU: python impact_modal.py prepare
Local foreground GPU: python impact_modal.py cross-local
After fetching the remote results: python impact_modal.py exclusion
The app definition is deliberately separate, in modal_impact.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import impact_study as S
from algocell_exp import impact as I, impact_corpus as K
from algocell_exp.impact_worlds import CORES, WORLDS
from algocell_exp.soup import adapter_summary
from algocell_exp.zmachine import load_machine

OUT = S.RESULTS / "impact_dryrun"
N, NREG = 1000, 100000
# CLI billing rates, neovand, 2026-10-11. Two CPUs and eight GiB are hard limits.
RATES = {"L40S": (1.95 + 2 * .0473 + 8 * .008) / 3600,
         "A10G": (1.10 + 2 * .0473 + 8 * .008) / 3600}


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(p)


def gpu_type():
    adapter = adapter_summary()
    for name, token in (("L40S", "L40S"), ("A10G", "A10")):
        if token in adapter and "Vulkan" in adapter and "DiscreteGPU" in adapter:
            return name
    raise RuntimeError(f"refusing an unexpected GPU/backend: {adapter}")


def condition(d):
    d = dict(d)
    d["sources"] = [K.Source(Path(s["path"]), s["L"], s["width"], s["height"]) for s in d["sources"]]
    d["suppress"] = tuple(d["suppress"])
    return K.Condition(**d)


def descriptor(c):
    d = dataclasses.asdict(c)
    d["suppress"] = list(c.suppress)
    for s, src in zip(d["sources"], c.sources):
        s["path"] = src.rel
    return d


def code_identity():
    paths = [Path(__file__), S.HERE / "impact_study.py", *sorted((S.HERE / "algocell_exp").glob("*.py")),
             *sorted((S.HERE / "algocell_exp/machines").glob("*.wgsl")),
             *sorted((S.HERE / "algocell_exp/machines").glob("*.manifest.json"))]
    return hashlib.sha256(b"".join(p.name.encode() + b"\0" + p.read_bytes() for p in paths)).hexdigest()


def prepare():
    """Never execute registered pairs; send only sampled inputs and their frozen descriptors."""
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, specs = {}, []
    t0 = time.monotonic()
    conds = K.random_conditions() + K.conditions()
    for i, c in enumerate(conds):
        t = time.monotonic()
        if c.tier == "random":
            pairs, redrawn = K.dry_random(c, N, NREG)
            draws = pairs
        else:
            draws, redrawn = K.dry_slots(c, N, NREG)
            pairs = K.pairs_of(c, draws)
        key = f"dry_{i:04d}"
        arrays[key], arrays[key + "_draws"] = pairs, draws
        specs.append({"key": key, "condition": descriptor(c), "seed": K.run_seed(K.DRY_STREAM, c.cid),
                      "redrawn": redrawn, "pairs_sha256": digest(pairs), "draw_sha256": digest(draws),
                      "t_draw_s": round(time.monotonic() - t, 6)})
        if i % 100 == 0:
            print(f"sampled {i + 1}/{len(conds)}", flush=True)
    # Supplement the previous proof with archived pairs of both newly included families.
    grouped = defaultdict(list)
    for c in conds:
        if c.family in ("ring_sweep", "R2"):
            slots = K._slots(c, 2000, K.rng_for(S.EQ_STREAM, 4, c.cid))
            grouped[(c.world, c.budget, c.suppress)].append((c.cid, K.pairs_of(c, slots)))
    eq = []
    for i, ((world, budget, suppress), entries) in enumerate(sorted(grouped.items())):
        key = f"equiv_{i:04d}"
        arrays[key] = np.concatenate([p for _, p in entries])
        eq.append({"key": key, "world": world, "budget": budget, "suppress": list(suppress),
                   "conditions": [cid for cid, _ in entries], "pairs_sha256": digest(arrays[key])})
    np.savez_compressed(OUT / "inputs.npz", **arrays)
    manifest = {"version": 1, "n": N, "n_registered": NREG, "dry_stream": K.DRY_STREAM,
                "reg_stream": K.REG_STREAM, "git": S.git_head(), **S.machines_provenance(),
                "conditions": specs, "equiv": eq, "prepare_s": time.monotonic() - t0,
                "inputs_sha256": hashlib.sha256((OUT / "inputs.npz").read_bytes()).hexdigest(),
                "inputs_bytes": (OUT / "inputs.npz").stat().st_size,
                "price_source": "modal billing rates --json --profile neovand (2026-10-11)",
                "rates_usd_s": RATES, "estimated_total_usd": 2.50,
                "estimate_basis": "Prior 38-minute local dry run, scaled by conditions, plus proof, compilation and benchmark allowance: about 71 minutes at the L40S allocation rate.",
                "max_allocated_s": 7800, "max_modeled_usd": 7800 * RATES["L40S"]}
    dump(OUT / "input_manifest.json", manifest)
    print(f"prepared {len(specs)} conditions, {len(eq)} proof groups; {manifest['inputs_bytes']} bytes", flush=True)


def read_inputs(folder):
    manifest = json.loads((folder / "input_manifest.json").read_text())
    raw = (folder / "inputs.npz").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["inputs_sha256"]:
        raise ValueError("input bundle digest mismatch")
    return manifest, np.load(folder / "inputs.npz", allow_pickle=False)


def scientific(row):
    return {k: v for k, v in row.items() if not k.startswith("t_")}


def cross_bytes(manifest, arrays):
    """One actual dry-run condition: exact bytes of all records/memory and canonical attribution."""
    sp = next(s for s in manifest["conditions"] if s["condition"]["cid"] == "random/P32_L16")
    c, p = condition(sp["condition"]), arrays[sp["key"]]
    tr = I.Triple(c.world)
    results = tr.run(p, c.budget, S.masks_of(c.suppress), sp["seed"])
    chunks = []
    components = {}
    for core in CORES:
        for field in ("memory", "record", "steps", "halt_reason", "trap_cause", "state", "write_counts"):
            a = getattr(results[core], field)
            value = np.ascontiguousarray(a).tobytes()
            chunks.append(value)
            components[f"{core}/{field}"] = hashlib.sha256(value).hexdigest()
    row, affected = S.run_condition(c, p, sp["seed"], {c.world: tr}, census=True)
    data = json.dumps({"row": scientific(row), "affected": affected}, sort_keys=True, separators=(",", ":")).encode()
    chunks.append(data)
    return b"".join(chunks), components


def cross_local():
    m, a = read_inputs(OUT)
    t = time.monotonic()
    data, components = cross_bytes(m, a)
    (OUT / "cross_local.bin").write_bytes(data)
    dump(OUT / "cross_local.json", {"cid": "random/P32_L16", "pairs": N, "bytes": len(data),
                                   "sha256": hashlib.sha256(data).hexdigest(), "components": components,
                                   "adapter": adapter_summary(), "wall_s": time.monotonic() - t,
                                   "code_sha256": code_identity(), "inputs_sha256": m["inputs_sha256"]})
    print((OUT / "cross_local.json").read_text(), flush=True)


def supplemental_equiv(m, arrays, dest, commit, deadline):
    groups = defaultdict(list)
    for d in m["equiv"]:
        groups[d["world"]].append(d)
    for tag, specs in sorted(groups.items()):
        path = dest / "equiv" / f"{tag}.json"
        if path.exists():
            continue
        if time.monotonic() > deadline:
            raise TimeoutError("spend time limit before equivalence world")
        t = time.monotonic()
        w, machine = WORLDS[tag], load_machine(WORLDS[tag].machine("c0"))
        seed, rows, controls = K.h32("equiv/" + tag), [], []
        biggest = max(specs, key=lambda d: len(arrays[d["key"]]))
        for d in specs:
            p = arrays[d["key"]]
            if digest(p) != d["pairs_sha256"]:
                raise ValueError("equivalence input digest mismatch")
            r = I.compare_c0(w, machine, p, d["budget"], S.masks_of(d["suppress"]), seed)
            rows.append({"world": tag, "budget": d["budget"], "suppress": ";".join(d["suppress"]) or "-",
                         "conditions": len(d["conditions"]), **r})
        rng = np.random.default_rng([S.EQ_STREAM, 5, K.h32(tag)])
        rand = rng.integers(0, 256, size=(20000, 2 * w.L), dtype=np.uint8)
        masks = np.packbits(rng.random((24, 32)) < 1 / 6, axis=1, bitorder="little").view(np.uint32).reshape(24)
        for name, mask in (("random unmasked", None), ("random masks 1 in 6", masks)):
            rows.append({"world": tag, "budget": 128, "suppress": name, "conditions": 0,
                         **I.compare_c0(w, machine, rand, 128, mask, seed + 7)})
        p = arrays[biggest["key"]][:5000]
        controls.append({"world": tag, "control": "budget minus one",
                         **I.compare_c0(w, machine, p, biggest["budget"], S.masks_of(biggest["suppress"]), seed, budget_delta=-1)})
        fields = ("mem_mismatch", "wc_mismatch", "reg_mismatch")
        if any(r[f] for r in rows for f in fields) or any(not any(r[f] for f in fields) for r in controls):
            raise RuntimeError(f"C0 equivalence or negative control failed: {tag}")
        dump(path, {"world": tag, "rows": rows, "controls": controls, "wall_s": time.monotonic() - t,
                    "executor": w.algocell_executor(), "adapter": adapter_summary()})
        I._ALG_PIPES.clear()
        commit()
        print(f"equiv {tag}: zero mismatches", flush=True)
        del machine
        gc.collect()


def bench_item(tag, budget):
    start = time.monotonic()
    tr = I.Triple(tag)
    c = K.Condition(f"bench/{tag}_st{budget}", "bench", "random", tag, budget, L=tr.L, P=tr.P)
    p = K.draw_random(c, NREG, K.DRY_STREAM)
    tr.run(p[:1000], budget, None, 1)
    t = time.monotonic()
    for start in range(0, len(p), S.BLOCK):
        tr.run(p[start:start + S.BLOCK], budget, None, 1, lane_base=start)
    te = time.monotonic() - t
    sub = p[:S.ATTR_BATCH]
    results = tr.run(sub, budget, None, 1)
    t = time.monotonic()
    Att = I.Attributor if tr.w.rule == "dial" else I.FastAttributor
    Att(tr, budget, None, 1).attribute(sub, np.arange(len(sub)), results, census=True)
    return {"world": tag, "rule": tr.w.rule, "L": tr.L, "budget": budget, "pairs": len(p),
            "t_exec_s": te, "pairs_per_s": len(p) / te, "attr_batch_lanes": len(sub),
            "t_attr_batch_s": time.monotonic() - t, "wall_s": time.monotonic() - start}


def finish(folder, commit, max_seconds=6400):
    """One detached job; each condition/proof/benchmark is its own committed chunk."""
    start = time.monotonic()
    m, a = read_inputs(folder)
    out = folder / "results"
    out.mkdir(parents=True, exist_ok=True)
    cross = json.loads((folder / "cross_remote.json").read_text())
    if not cross["identical"] or cross["inputs_sha256"] != m["inputs_sha256"] or cross["code_sha256"] != code_identity():
        raise RuntimeError("cross-platform gate missing or stale")
    gpu = gpu_type()
    meta = {**{k: v for k, v in m.items() if k not in ("conditions", "equiv")},
            "adapter": adapter_summary(), "gpu": gpu, "rate_usd_s": RATES[gpu], "code_sha256": code_identity()}
    deadline = start + max_seconds
    try:
        identity = {"code_sha256": code_identity(), "inputs_sha256": m["inputs_sha256"]}
        identity_path = out / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise RuntimeError("refusing to resume chunks from different code or inputs")
        dump(identity_path, identity)
        commit()
        supplemental_equiv(m, a, out, commit, deadline)
        triples = {}
        # Group by world to avoid recompiling pipelines when families share a world.
        specs = sorted(m["conditions"], key=lambda d: (d["condition"]["world"], d["key"]))
        for sp in specs:
            path = out / "chunks" / f"{sp['key']}.json"
            if path.exists():
                continue
            if time.monotonic() > deadline:
                raise TimeoutError("spend time limit before dry-run condition")
            t = time.monotonic()
            c, p = condition(sp["condition"]), a[sp["key"]]
            if digest(p) != sp["pairs_sha256"] or digest(a[sp["key"] + "_draws"]) != sp["draw_sha256"]:
                raise ValueError("dry-run input digest mismatch")
            if c.world not in triples:
                triples.clear()
                gc.collect()
                triples[c.world] = I.Triple(c.world)
            tc = time.monotonic() - t
            row, pairs = S.run_condition(c, p, sp["seed"], triples, census=True)
            row.update({k: sp[k] for k in ("redrawn", "pairs_sha256", "draw_sha256", "t_draw_s")})
            row.update(t_compile_s=tc, t_wall_s=time.monotonic() - t)
            dump(path, {"key": sp["key"], "row": row, "pairs": pairs})
            commit()
            print(f"{sp['key']} {c.cid}: C1={row['aff_c1']} C2={row['aff_c2']} {row['t_wall_s']:.1f}s", flush=True)
        triples.clear()
        bench = list(S.BENCH) + [("lp003_L16", 128), ("randreg_L16", 128), ("randsp_L16", 128), ("lethal_L16", 128)]
        for tag, budget in bench:
            path = out / "bench" / f"{tag}_{budget}.json"
            if path.exists():
                continue
            if time.monotonic() > deadline:
                raise TimeoutError("spend time limit before benchmark")
            dump(path, bench_item(tag, budget))
            commit()
        records = [json.loads((out / "chunks" / f"{sp['key']}.json").read_text()) for sp in m["conditions"]]
        S.write_csv(out / "dryrun_conditions.csv", [r["row"] for r in records])
        S.write_csv(out / "dryrun_affected_pairs.csv", [p for r in records for p in r["pairs"]])
        S.write_csv(out / "bench.csv", [json.loads(p.read_text()) for p in sorted((out / "bench").glob("*.json"))])
        proofs = [json.loads(p.read_text()) for p in sorted((out / "equiv").glob("*.json"))]
        S.write_csv(out / "equiv_groups.csv", [r for p in proofs for r in p["rows"]])
        S.write_csv(out / "equiv_controls.csv", [r for p in proofs for r in p["controls"]])
        meta.update(status="complete", wall_s=time.monotonic() - start, conditions=len(records), chunks=len(records),
                    proof_worlds=len(proofs), work_s=sum(r["row"]["t_wall_s"] for r in records))
        meta["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta["modeled_cost_usd"] = meta["wall_s"] * RATES[gpu]
        dump(out / "dryrun_meta.json", meta)
        dump(folder / "done.json", meta)
        commit()
        return meta
    except Exception as exc:
        dump(folder / "failed.json", {"error": repr(exc), "wall_s": time.monotonic() - start,
                                       "modeled_cost_usd": (time.monotonic() - start) * RATES[gpu]})
        commit()
        raise


def exclusion():
    """Re-derive draws AND pair bytes; bind the exclusion proof to transported/executed inputs."""
    m, arrays = read_inputs(OUT)
    ran = {r["cid"]: r for r in S._read_csv(OUT / "dryrun_conditions.csv")}
    catalog = {c.cid: c for c in K.random_conditions() + K.conditions()}
    rows = []
    for sp in m["conditions"]:
        c = catalog[sp["condition"]["cid"]]
        if descriptor(c) != sp["condition"]:
            raise ValueError(f"archive registry changed: {c.cid}")
        if c.tier == "random":
            reg = {bytes(p) for p in K.draw_random(c, m["n_registered"], K.REG_STREAM)}
            dry, redraws = K.dry_random(c, m["n"], m["n_registered"])
            p, overlap = dry, len(reg & {bytes(x) for x in dry})
        else:
            reg = {tuple(x) for x in K.draw_slots(c, m["n_registered"], K.REG_STREAM).tolist()}
            dry, redraws = K.dry_slots(c, m["n"], m["n_registered"])
            p, overlap = K.pairs_of(c, dry), len(reg & {tuple(x) for x in dry.tolist()})
        r = ran[c.cid]
        same_draw = digest(dry) == sp["draw_sha256"] == r["draw_sha256"] == digest(arrays[sp["key"] + "_draws"])
        same_pair = digest(p) == sp["pairs_sha256"] == r["pairs_sha256"] == digest(arrays[sp["key"]])
        rows.append({"cid": c.cid, "registered_draws": m["n_registered"], "registered_distinct": len(reg),
                     "dry_draws": len(dry), "overlap": overlap, "redrawn": redraws, "dry_run_redrawn": int(r["redrawn"]),
                     "draw_sha256": digest(dry), "pairs_sha256": digest(p), "draw_identical": int(same_draw), "pairs_identical": int(same_pair)})
        if overlap or not same_draw or not same_pair or redraws != int(r["redrawn"]):
            raise RuntimeError(f"exclusion failed: {c.cid}")
    S.write_csv(OUT / "exclusion.csv", rows)
    print(f"{len(rows)} conditions: zero overlap, all transported/executed bytes and draws verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "cross-local", "exclusion"))
    args = parser.parse_args()
    {"prepare": prepare, "cross-local": cross_local, "exclusion": exclusion}[args.command]()
