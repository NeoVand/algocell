"""Local exhaustive 8086 census and candidate trace audit. No population runs.

Run with the experiment interpreter: python census8086.py census | audit | smoke.
Every GPU call is a foreground burst followed by an equal wall-clock pause.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from algocell_exp.census8086_trace import DiagnosticMachine, mechanism, same_execution
from algocell_exp.zcensus import partner_bytes, tile_tapes
from algocell_exp.zmachine import load_machine

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/census8086"
SEED = 0x8086C0DE
NRAND = 16
BUDGETS = (128, 512)
GPU_SECONDS = (sum(float(r["burst_seconds"]) for r in csv.DictReader((OUT / "bursts.csv").open()))
               if (OUT / "bursts.csv").exists() else 0.0)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def burst(label, fn):
    global GPU_SECONDS
    t = time.monotonic()
    result = fn()
    elapsed = time.monotonic() - t
    GPU_SECONDS += elapsed
    if elapsed > 60 or GPU_SECONDS > 1800:
        raise RuntimeError(f"local GPU limit exceeded: {label}: {elapsed:.3f}s, cumulative {GPU_SECONDS:.3f}s")
    t = time.monotonic()
    time.sleep(elapsed)
    pause = time.monotonic() - t
    with (OUT / "bursts.csv").open("a", newline="") as f:
        csv.writer(f).writerow([label, f"{elapsed:.6f}", f"{pause:.6f}"])
    return result


def panel(L):
    return np.vstack([np.zeros(L, np.uint8)] + [partner_bytes(np.array([0], np.uint64), v, L, SEED, panel=True)[0] for v in range(1, 17)])


def pairs(A, L, partners=None):
    partners = panel(L) if partners is None else partners
    n, V = len(A), len(partners)
    mem = np.empty((n, V, 2*L), np.uint8)
    mem[:, :, :L] = A[:, None, :]
    mem[:, :, L:] = partners[None, :, :]
    return mem.reshape(n*V, 2*L)


def matches(B, A, k):
    return np.max(np.stack([(B == np.roll(A, s, axis=1)[:, None, :]).sum(2) for s in range(k)], axis=2), axis=2)


def census():
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "partners.csv", [{"L": L, "partner": v, "seed": SEED, "bytes": bytes(p).hex(), "sha256": sha(p.tobytes())}
                                  for L in (16, 32) for v, p in enumerate(panel(L))])
    provenance = {"seed": SEED, "random_partners": NRAND, "panel_shared_across_words": True, "budgets": BUDGETS,
                  "enumeration": "big-endian word index; both phases and constants included", "machines": {}, "runs": []}
    for L in (16, 32):
        m = burst(f"compile-L{L}", lambda: load_machine(f"i8086_pair_maps_L{L}"))
        d = burst(f"compile-diagnostics-L{L}", lambda: DiagnosticMachine(m))
        provenance["machines"][str(L)] = {**m.describe(), "diagnostic_sha256": d.wgsl_sha256}
        for k in (1, 2):
            for budget in BUDGETS:
                rows, digest, ndone = [], hashlib.sha256(), 0
                for lo in range(0, 256**k, 2048):
                    ids = np.arange(lo, min(lo+2048, 256**k), dtype=np.uint64)
                    A = tile_tapes(ids, k, L)
                    mem = pairs(A, L)
                    raw = burst(f"census-L{L}-k{k}-b{budget}-{lo}-raw", lambda: m.run(mem, budget=budget))
                    obs = burst(f"census-L{L}-k{k}-b{budget}-{lo}-diagnostics", lambda: d.run(mem, budget=budget))
                    if not same_execution(raw, obs, m.record_words):
                        raise RuntimeError("observation changed execution")
                    digest.update(raw.memory.tobytes()); digest.update(raw.record.tobytes())
                    B = raw.memory.reshape(len(ids), 17, 2*L)[:, :, L:]
                    score = matches(B, A, k)
                    D = d.diagnostics(obs).reshape(len(ids), 17, 6)
                    wb = raw.writes("B").reshape(len(ids), 17)
                    enters = raw.fetch_map.reshape(len(ids), 17, 2*L)[:, :, L:].any(2)
                    steps = raw.steps.reshape(len(ids), 17)
                    halt = raw.halt_reason.reshape(len(ids), 17)
                    for j, idx in enumerate(ids):
                        rows.append({"L": L, "k": k, "budget": budget, "index": int(idx),
                            "word": int(idx).to_bytes(k, "big").hex(), "primitive_period": 1 if k == 1 or (idx >> np.uint64(8)) == (idx & np.uint64(255)) else 2,
                            "zero_match": int(score[j, 0]), "rand_exact": int((score[j, 1:] == L).sum()),
                            "rand75": int((score[j, 1:] >= (3*L+3)//4).sum()), "entered": int(enters[j].sum()),
                            "zero_write_coverage": int(D[j, 0, 4]), "min_rand_write_coverage": int(D[j, 1:, 4].min()),
                            "max_rand_write_coverage": int(D[j, 1:, 4].max()),
                            "partner_output_difference_max": int((B[j, 1:] != B[j, 0]).sum(1).max()),
                            "matches": ";".join(map(str, score[j])), "write_counts_B": ";".join(map(str, wb[j])),
                            "write_coverage_B": ";".join(map(str, D[j, :, 4])),
                            "original_partner_fetch_reads": ";".join(map(str, D[j, :, 0])),
                            "original_partner_data_reads": ";".join(map(str, D[j, :, 1])),
                            "data_reads": ";".join(map(str, D[j, :, 2])), "steps": ";".join(map(str, steps[j])),
                            "halts": ";".join(map(str, halt[j])), "output_sha256": sha(B[j].tobytes())})
                    ndone += len(ids)
                    if lo % 16384 == 0:
                        print(f"L{L} k{k} b{budget}: {ndone}/{256**k}", flush=True)
                assert ndone == 256**k and [r["index"] for r in rows] == list(range(256**k))
                name = f"words_L{L}_k{k}_b{budget}.csv"
                write_csv(OUT / name, rows)
                provenance["runs"].append({"L": L, "k": k, "budget": budget, "words": ndone, "execution_sha256": digest.hexdigest(), "csv": name})
                (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")


def audit():
    for L in (16, 32):
        m = burst(f"audit-compile-L{L}", lambda: load_machine(f"i8086_pair_maps_L{L}"))
        d = burst(f"audit-trace-compile-L{L}", lambda: DiagnosticMachine(m, events=8192))
        for budget in BUDGETS:
            wanted = {(2, int(w, 16)) for w in ("b951", "51b9", "ff30", "30ff", "ad50", "50ad")}
            for k in (1, 2):
                with (OUT / f"words_L{L}_k{k}_b{budget}.csv").open() as f:
                    for r in csv.DictReader(f):
                        if int(r["zero_match"]) == L or int(r["rand_exact"]) or int(r["rand75"]) >= 8:
                            wanted.add((k, int(r["index"])))
            rows, logs = [], []
            for k, idx in sorted(wanted):
                A = tile_tapes(np.array([idx], np.uint64), k, L)
                mem = pairs(A, L)
                raw = burst(f"audit-L{L}-b{budget}-k{k}-{idx}-raw", lambda: m.run(mem, budget=budget))
                obs = burst(f"audit-L{L}-b{budget}-k{k}-{idx}-trace", lambda: d.run(mem, budget=budget))
                if not same_execution(raw, obs, m.record_words):
                    raise RuntimeError("trace changed execution")
                score = matches(raw.memory[:, L:].reshape(1, 17, L), A, k)[0]
                for v, log in enumerate(d.logs(obs)):
                    cls, evidence = mechanism(log, L)
                    diag = d.diagnostics(obs)[v]
                    rows.append({"L": L, "budget": budget, "k": k, "index": idx, "word": idx.to_bytes(k, "big").hex(), "partner": v,
                                 "match": int(score[v]), "mechanism": cls, "evidence": evidence,
                                 "write_coverage": int(diag[4]), "original_partner_fetch_reads": int(diag[0]),
                                 "original_partner_data_reads": int(diag[1]), "events": int(diag[5]), "trace_sha256": sha(log.tobytes()) if log is not None else "overflow"})
                    if log is not None:
                        # Full ordered traces for the named candidates and zero-partner hits.
                        if v == 0 or idx in (0xb951, 0xff30, 0xad50):
                            for step, kind, pos, value, origin in log:
                                logs.append({"k": k, "index": idx, "partner": v, "step": int(step), "kind": int(kind), "pos": int(pos), "value": int(value), "original_partner": int(origin)})
            write_csv(OUT / f"mechanisms_L{L}_b{budget}.csv", rows)
            write_csv(OUT / f"traces_L{L}_b{budget}.csv", logs)
            print(f"audit L{L} b{budget}: {len(wanted)} words, {len(rows)} traced encounters", flush=True)


def smoke():
    from algocell_exp.zsoup import ZSoup
    rows = []
    for L in (16, 32):
        def run():
            s = ZSoup(f"i8086_pair_maps_L{L}", width=2, height=2, pair_count=1, z80_steps=8, seed=8086, mutations_per_step=0)
            s.write_soup(np.full((4, L), 0x90, np.uint8))
            s.step(1); s.sync()
            out = s.read_soup()
            assert np.all(out == 0x90)
            # ZBridge keeps these soup shader items verbatim.
            return {"L": L, "fixed_input": "90"*L, "steps": 1, "pairs": 1, "mutations": 0,
                    "output_sha256": sha(out.tobytes()), **s.machine_provenance}
        rows.append(burst(f"fixed-input-smoke-L{L}", run))
    (OUT / "smoke.json").write_text(json.dumps(rows, indent=2)+"\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("census", "audit", "smoke"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "bursts.csv").exists():
        write_csv(OUT / "bursts.csv", [], ["label", "burst_seconds", "pause_seconds"])
    globals()[args.command]()
