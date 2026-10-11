"""Frozen I1 inputs, per-condition Modal checkpoints, and an $8 allocation gate.

CPU-only preparation: python impact_confirm.py prepare
Detached launch: modal run --detach modal_impact_confirm.py::main
Fetch only the run's results, never other volume folders.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import impact_modal as M
import impact_study as S
from algocell_exp import executor as E, impact as I, impact_corpus as K
from algocell_exp.impact_assays import assay_item, panel
from algocell_exp.soup import adapter_summary

OUT = S.RESULTS / "impact"
N = 100000
MAX_SECONDS = 11900
ALLOCATION_SECONDS = 12200
RESERVE_USD = .75  # startup, two-second idle, storage/egress; no retry/relaunch


def source_hash():
    paths = [S.HERE / name for name in ("impact_confirm.py", "impact_study.py", "impact_modal.py")]
    paths += sorted((S.HERE / "algocell_exp").glob("*.py"))
    paths += sorted((S.HERE / "algocell_exp/machines").glob("imp_*.wgsl"))
    paths += sorted((S.HERE / "algocell_exp/machines").glob("imp_*.manifest.json"))
    paths += [S.HERE / "algocell_exp/machines/worlds_impact.json"]
    return hashlib.sha256(b"".join(p.name.encode() + b"\0" + p.read_bytes() for p in paths)).hexdigest()


def gate():
    exclusion = S._read_csv(S.RESULTS / "impact_dryrun/exclusion.csv")
    if len(exclusion) != 787 or any(int(r["overlap"]) or int(r["registered_draws"]) != N or
                                   not int(r["draw_identical"]) or not int(r["pairs_identical"]) for r in exclusion):
        raise RuntimeError("dry-run exclusion proof has not passed")
    return exclusion


def prepare():
    exclusion = gate()
    old = json.loads((S.RESULTS / "impact_dryrun/input_manifest.json").read_text())
    conds = K.random_conditions() + K.conditions()
    if [M.descriptor(c) for c in conds] != [sp["condition"] for sp in old["conditions"]]:
        raise RuntimeError("registered registry changed since the dry run")
    assays = panel()
    projection = S._read_csv(S.RESULTS / "impact_dryrun/projection.csv")[-1]
    rate = M.RATES["L40S"]
    estimate = float(projection["upper_cost_usd"]) + 300 * rate + RESERVE_USD
    maximum = ALLOCATION_SECONDS * rate + RESERVE_USD
    if estimate > 8 or maximum > 8:
        raise RuntimeError(f"STOP: estimate {estimate}, allocation ceiling {maximum} > $8")
    OUT.mkdir(parents=True, exist_ok=True)
    M.dump(OUT / "cost_estimate.json", dict(upper_projection_usd=float(projection["upper_cost_usd"]),
           assay_allowance_s=300, reserve_usd=RESERVE_USD, estimate_usd=estimate,
           allocation_ceiling_usd=maximum, max_worker_s=MAX_SECONDS,
           allocation_timeout_s=ALLOCATION_SECONDS, cap_usd=8, rates_usd_s=M.RATES,
           basis="impact_dryrun/projection.csv draft+added upper, plus assays and startup/storage/egress reserve"))
    S.write_csv(OUT / "exclusion.csv", exclusion)
    specs, hashes = [], {}
    for i, c in enumerate(conds):
        key = f"reg_{i:04d}"
        path = OUT / "inputs" / f"{key}.npz"
        if c.tier == "random":
            draws = pairs = K.draw_random(c, N, K.REG_STREAM)
        else:
            # Draw the whole 100,000 once, never restart the RNG at a chunk boundary.
            draws = K.draw_slots(c, N, K.REG_STREAM)
            pairs = K.pairs_of(c, draws)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, pairs=pairs)
        for src in c.sources:
            if src.rel not in hashes:
                hashes[src.rel] = hashlib.sha256(src.path.read_bytes()).hexdigest()
        specs.append(dict(key=key, condition=M.descriptor(c), seed=K.run_seed(K.REG_STREAM, c.cid),
                          draw_sha256=M.digest(draws), pairs_sha256=M.digest(pairs),
                          file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size))
        if i % 100 == 0:
            print(f"prepared {i + 1}/{len(conds)}", flush=True)
    S.write_csv(OUT / "archive_hashes.csv", [dict(path=p, sha256=s) for p, s in sorted(hashes.items())])
    S.write_csv(OUT / "assay_tapes.csv", [{**e, **{k: json.dumps(e[k]) for k in ("sources", "names", "references", "suppress")}} for e in assays])
    manifest = dict(version=1, n=N, frozen_draw_commit="bdcdcb6", git=S.git_head(), reg_stream=K.REG_STREAM,
                    dry_stream=K.DRY_STREAM, conditions=specs, assays=assays, code_sha256=source_hash(),
                    archive_sha256=hashlib.sha256((OUT / "archive_hashes.csv").read_bytes()).hexdigest(),
                    exclusion_sha256=hashlib.sha256((OUT / "exclusion.csv").read_bytes()).hexdigest(),
                    cost_estimate=json.loads((OUT / "cost_estimate.json").read_text()))
    M.dump(OUT / "input_manifest.json", manifest)
    print(f"Prepared {len(specs)} conditions; {len(assays)} assay tape/world/mask combinations; "
          f"{sum(s['bytes'] for s in specs)} compressed bytes; estimate ${estimate:.3f}, ceiling ${maximum:.3f}")


def finish(folder, commit):
    start = time.monotonic()
    m = json.loads((folder / "input_manifest.json").read_text())
    if source_hash() != m["code_sha256"] or m["n"] != N or len(m["conditions"]) != 787:
        raise RuntimeError("input/code identity gate failed")
    if m["cost_estimate"]["estimate_usd"] > 8 or m["cost_estimate"]["allocation_ceiling_usd"] > 8:
        raise RuntimeError("cost gate failed")
    gpu = M.gpu_type()
    out = folder / "results"
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(status="running", gpu=gpu, adapter=adapter_summary(), rate_usd_s=M.RATES[gpu],
                code_sha256=source_hash(), conditions_expected=787, pairs_per_condition=N,
                assay_items_expected=len(m["assays"]), started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    triples = {}

    def deadline():
        if time.monotonic() - start > MAX_SECONDS:
            raise TimeoutError("STOP: allocation cost/time guard")

    try:
        M.dump(out / "meta.json", meta)
        commit()
        for item in m["assays"]:
            deadline()
            path = out / "assays" / f"{item['aid']}.json"
            if path.exists():
                raise RuntimeError("refusing an implicit relaunch/resume against the $8 cap")
            M.dump(path, assay_item(item))
            commit()
        M.dump(out / "executor_usage.json", E.usage())
        commit()
        for sp in sorted(m["conditions"], key=lambda sp: (sp["condition"]["world"], sp["key"])):
            deadline()
            path = out / "chunks" / f"{sp['key']}.json"
            if path.exists():
                raise RuntimeError("refusing an implicit relaunch/resume against the $8 cap")
            t = time.monotonic()
            infile = folder / "inputs" / f"{sp['key']}.npz"
            if hashlib.sha256(infile.read_bytes()).hexdigest() != sp["file_sha256"]:
                raise RuntimeError("transported input digest mismatch")
            with np.load(infile, allow_pickle=False) as bundle:
                pairs = bundle["pairs"]
            if M.digest(pairs) != sp["pairs_sha256"] or pairs.shape[0] != N:
                raise RuntimeError("registered pair digest/length mismatch")
            c = M.condition(sp["condition"])
            if c.world not in triples:
                triples.clear()
                gc.collect()
                triples[c.world] = I.Triple(c.world)
            row, affected = S.run_condition(c, pairs, sp["seed"], triples)
            row.update(draw_sha256=sp["draw_sha256"], pairs_sha256=sp["pairs_sha256"],
                       run_seed=sp["seed"], t_wall_s=time.monotonic() - t)
            if any(row[f"{core}_other"] for core in ("c1", "c2")):
                raise RuntimeError("STOP: other attribution")
            M.dump(path, dict(key=sp["key"], row=row, pairs=affected))
            commit()
            print(f"{sp['key']} {c.cid}: C1={row['aff_c1']} C2={row['aff_c2']} "
                  f"differ={row['c1_c2_differ']} {row['t_wall_s']:.2f}s", flush=True)
        meta.update(status="complete", conditions=787, assay_items=len(m["assays"]), wall_s=time.monotonic() - start)
    except Exception as exc:
        meta.update(status="stopped", error=repr(exc), wall_s=time.monotonic() - start)
        M.dump(out / "meta.json", {**meta, "modeled_cost_usd": meta["wall_s"] * M.RATES[gpu]})
        commit()
        raise
    meta.update(completed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                modeled_cost_usd=meta["wall_s"] * M.RATES[gpu])
    M.dump(out / "meta.json", meta)
    commit()
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("prepare",))
    ap.parse_args()
    prepare()
