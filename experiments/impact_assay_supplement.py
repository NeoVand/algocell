"""Complete corpus 3 with Stage E's explicitly named unit-tiling table.

The original detached worker and its manifest stay immutable. This supplement
uses that same sealed assay engine and imp_* machines, locally in <=55-second
GPU bursts followed by pauses at least as long. No confirmatory draws change.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

import impact_confirm as C
from algocell_exp import executor as E
from algocell_exp.impact_assays import assay_item
from algocell_exp.soup import adapter_summary

TABLE = C.S.RESULTS / "stageE/stage_e/unit_fitness_vs_L.csv"
UNITS = {"pusher 01 c5": "01 c5", "LDIR-3 1d ed b0": "1d ed b0", "LDIR-4 04 5e ed b0": "04 5e ed b0"}


def key(e):
    return (e["tape"], e["world"], tuple(e["suppress"]), e["budget"], e["n"], e["seed"])


def catalogue(primary):
    known = {key(e): e["aid"] for e in primary}
    extra, aliases = [], {}
    index = (C.S.HERE / "manuscript/NUMBERS_INDEX.md").read_text().splitlines()
    refs = [f"NUMBERS_INDEX.md:{i}" for i, line in enumerate(index, 1) if "unit_fitness_vs_L.csv" in line]
    assert refs
    with TABLE.open() as fh:
        for rowno, r in enumerate(csv.DictReader(fh), 2):
            L = int(r["L"])
            tape = np.resize(np.frombuffer(bytes.fromhex(UNITS[r["unit"]]), np.uint8), L).tobytes().hex(" ")
            e = dict(tape=tape, world=f"L{L}", L=L, suppress=[], budget=int(r["steps"]), n=64, seed=0,
                     sources=[f"results/stageE/stage_e/unit_fitness_vs_L.csv:{rowno}; stage_e.py:259 np.resize"],
                     names=[r["unit"] + f" ×/truncated to L{L}, {r['steps']} steps"],
                     references=refs + ["MAIN_nature.md:135 (Extended Data Fig. 2b)"])
            if L == 9 and r["unit"] == "LDIR-3 1d ed b0":
                e["references"].append("MAIN_nature.md:412 (Theorem 6 L9 example)")
            k = key(e)
            if k in known:
                aliases.setdefault(known[k], []).append(e)
                continue
            e["aid"] = hashlib.sha256(json.dumps(["I1-unit-supplement", *k]).encode()).hexdigest()[:16]
            assert e["world"] in C.K.WORLDS
            known[k] = e["aid"]
            extra.append(e)
    return sorted(extra, key=lambda e: (e["world"], e["aid"])), aliases


def run():
    original = json.loads((C.OUT / "input_manifest.json").read_text())
    if C.source_hash() != original["code_sha256"]:
        raise RuntimeError("the supplement must use the same sealed scientific engine")
    items, aliases = catalogue(original["assays"])
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = dict(assays=items, aliases=aliases, source_table="results/stageE/stage_e/unit_fitness_vs_L.csv",
                    source_table_sha256=hashlib.sha256(TABLE.read_bytes()).hexdigest(),
                    original_worker_code_sha256=original["code_sha256"], code_sha256=C.source_hash(),
                    supplement_script_sha256=script_sha, git=C.S.git_head(),
                    reason="Complete named canonical units in Extended Data Fig. 2b's table and the L9 example in Theorem 6",
                    burst_limit_s=55, pause_rule="at least each preceding GPU burst's wall time", modal_cost_usd=0)
    C.M.dump(C.OUT / "supplement_manifest.json", manifest)
    folder = C.OUT / "local_assays"
    folder.mkdir(exist_ok=True)
    bursts = []
    remaining = [e for e in items if not (folder / f"{e['aid']}.json").exists()]
    start = time.monotonic()
    while remaining:
        burst_start = time.monotonic()
        adapter = adapter_summary()
        ran, max_item_s = 0, 2.0
        while remaining and (not ran or time.monotonic() - burst_start + max_item_s + .5 < 55):
            item = remaining.pop(0)
            t = time.monotonic()
            rec = assay_item(item)
            rec.update(code_sha256=original["code_sha256"], adapter=adapter)
            C.M.dump(folder / f"{item['aid']}.json", rec)
            max_item_s = max(max_item_s, time.monotonic() - t)
            ran += 1
        wall = time.monotonic() - burst_start
        print(f"GPU burst: {ran} assay combinations, {wall:.3f}s; {len(remaining)} left", flush=True)
        if wall > 60:
            raise RuntimeError("local burst exceeded 60 seconds; stop")
        pause_start = time.monotonic()
        pause_s = math.ceil(wall)
        # Sleep in short pieces so the supervising agent remains responsive.
        until = pause_start + pause_s
        while time.monotonic() < until:
            time.sleep(min(5, until - time.monotonic()))
        actual_pause = time.monotonic() - pause_start
        bursts.append(dict(items=ran, gpu_burst_s=wall, pause_s=actual_pause, adapter=adapter))
        C.S.write_csv(C.OUT / "local_gpu_bursts.csv", bursts)
    meta = dict(status="complete", items=len(items), source_code_sha256=original["code_sha256"],
                adapter=adapter, usage=E.usage(), wall_s=time.monotonic() - start,
                bursts=len(bursts), max_gpu_burst_s=max(b["gpu_burst_s"] for b in bursts),
                min_pause_minus_burst_s=min(b["pause_s"] - b["gpu_burst_s"] for b in bursts), modal_cost_usd=0)
    C.M.dump(C.OUT / "supplement_meta.json", meta)
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    run()
