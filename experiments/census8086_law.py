"""Freeze REP geometry before payload assays; constructed controls only.

Commands: qualify (trace and check eligibility), assay (all alternatives,
joint payloads, erased-coordinate controls, and eight fresh-partner passages).
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from census8086 import OUT, SEED, burst, panel, pairs, sha, write_csv
from algocell_exp.census8086_trace import DiagnosticMachine, same_execution
from algocell_exp.zcensus import partner_bytes
from algocell_exp.zmachine import load_machine

GEOMETRIES = ((16, 8), (16, 16), (32, 8), (32, 16), (32, 32))


def block(d, op):
    out = np.array([(33+13*i) % 256 for i in range(d)], np.uint8)
    out[:6] = [0xbf, d & 255, d >> 8, 0x49, 0xf3, op]
    return out


def parents(d, L, op):
    return np.tile(block(d, op), L//d)


def qualify():
    rows, events = [], []
    for L in (16, 32):
        m = burst(f"law-qualify-compile-L{L}", lambda: load_machine(f"i8086_pair_maps_L{L}"))
        t = burst(f"law-qualify-trace-L{L}", lambda: DiagnosticMachine(m, events=8192))
        for LL, d in GEOMETRIES:
            if LL != L:
                continue
            for op in (0xa4, 0xa5):
                A = parents(d, L, op)[None, :]
                mem = pairs(A, L)
                for budget in (128, 512):
                    r = burst(f"qualify-L{L}-d{d}-op{op}-b{budget}-raw", lambda: m.run(mem, budget=budget))
                    q = burst(f"qualify-L{L}-d{d}-op{op}-b{budget}-trace", lambda: t.run(mem, budget=budget))
                    assert same_execution(r, q, m.record_words)
                    assert np.all(r.memory[:, L:] == A)
                    for v, log in enumerate(t.logs(q)):
                        assert log is not None
                        decodes = set(log[log[:, 1] == 3, 3])
                        assert decodes == {0xbf, 0x49, op}
                        assert not np.any((log[:, 1] == 0) & (log[:, 2] >= L))
                        reads = log[log[:, 1] == 1]
                        writes = log[log[:, 1] == 2]
                        assert len(reads) == len(writes) == (budget-2)*(1 if op == 0xa4 else 2)
                        assert np.all(reads[:, 0] == writes[:, 0])
                        assert np.all(reads[:, 3] == writes[:, 3])
                        assert np.all((writes[:, 2].astype(np.int64)-reads[:, 2]) % (2*L) == d)
                        assert np.all(r.memory[v, L:L+6] == block(d, op)[:6])
                        rows.append({"L": L, "d": d, "opcode": f"{op:02x}", "budget": budget, "partner": v,
                                     "eligible": 1, "core_bytes": 6, "predicted_payload": d-6, "payload_positions": ";".join(map(str, range(6, d))),
                                     "byte_writes": len(writes), "partner_write_coverage": int(t.diagnostics(q)[v, 4]),
                                     "constant_displacement": d, "original_partner_reads": int(t.diagnostics(q)[v, :2].sum()),
                                     "cx_final": int(r.reg("cx")[v]), "core_preserved": 1, "trace_sha256": sha(log.tobytes())})
                        if v == 0 and budget == 128:
                            for step, kind, pos, value, origin in log:
                                events.append({"L": L, "d": d, "opcode": f"{op:02x}", "step": int(step), "kind": int(kind), "pos": int(pos), "value": int(value), "original_partner": int(origin)})
    write_csv(OUT / "law_eligibility.csv", rows)
    write_csv(OUT / "law_geometry_traces.csv", events)
    print(f"qualified {len(rows)} geometry/partner/budget rows", flush=True)


def variants(L, d, op):
    baseline = parents(d, L, op)
    cases = []
    for pos in range(6, d):
        for value in range(256):
            A = baseline.copy(); A[pos] = value
            cases.append((f"single:{pos}:{value}", "single", pos, value, A))
    # 256 counterbalanced full payload assignments + 256 independent joint
    # assignments. Every coordinate takes all values in the first panel.
    rng = np.random.default_rng([SEED, L, d, op])
    for j in range(512):
        A = baseline.copy()
        A[6:d] = (np.arange(d-6)*37+j) % 256 if j < 256 else rng.integers(0, 256, d-6, dtype=np.uint8)
        cases.append((f"joint:{j}", "joint", -1, j, A))
    return cases


def assay():
    # Eligibility must already have been frozen by a Git commit.
    import subprocess
    root = OUT.parents[2]
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "experiments/results/census8086/law_eligibility.csv", "experiments/results/census8086/LAW_CONTRACT.md"], cwd=root, capture_output=True)
    if tracked.returncode:
        raise RuntimeError("commit law eligibility and LAW_CONTRACT.md before assays")
    if subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "experiments/results/census8086/law_eligibility.csv", "experiments/results/census8086/LAW_CONTRACT.md"], cwd=root, capture_output=True).returncode:
        raise RuntimeError("law eligibility contract differs from frozen commit")
    rows, serial, erased = [], [], []
    for L in (16, 32):
        m = burst(f"law-assay-compile-L{L}", lambda: load_machine(f"i8086_pair_maps_L{L}"))
        for LL, d in GEOMETRIES:
            if LL != L:
                continue
            for op in (0xa4, 0xa5):
                cases = variants(L, d, op)
                for budget in (128, 512):
                    for lo in range(0, len(cases), 256):
                        batch = cases[lo:lo+256]
                        A = np.stack([c[4] for c in batch])
                        expected = np.tile(A[:, :d], (1, L//d))
                        mem = pairs(A, L)
                        r = burst(f"law-L{L}-d{d}-op{op}-b{budget}-v{lo}", lambda: m.run(mem, budget=budget))
                        child = r.memory.reshape(len(batch), 17, 2*L)[:, :, L:]
                        for j, (key, kind, pos, value, parent) in enumerate(batch):
                            # All 17 partners and all coordinates checked, not similarity thresholds.
                            ok = np.all(child[j] == expected[j], axis=1)
                            rows.append({"L": L, "d": d, "opcode": f"{op:02x}", "budget": budget, "case": key, "kind": kind,
                                         "position": pos, "value": value, "alternative": int(kind == "joint" or value != int(block(d, op)[pos])),
                                         "partners": 17, "passed": int(ok.sum()), "core_preserved": int(np.all(child[j, :, :6] == parent[:6])),
                                         "output_sha256": sha(child[j].tobytes())})
                            if not np.all(ok):
                                raise RuntimeError(f"law mismatch {L,d,op,budget,key}")
                        # All the same single and joint cases pass through g=8.
                        current = A.copy()
                        ids = np.arange(lo, lo+len(batch), dtype=np.uint64)
                        for g in range(1, 9):
                            fresh = partner_bytes(ids, g, L, seed=SEED ^ (L << 16) ^ (d << 8) ^ op ^ budget)
                            memory = np.concatenate((current, fresh), axis=1)
                            rr = burst(f"serial-L{L}-d{d}-op{op}-b{budget}-v{lo}-g{g}", lambda: m.run(memory, budget=budget))
                            current = rr.memory[:, L:]
                            good = np.all(current == expected, axis=1)
                            for j, (key, kind, pos, value, parent) in enumerate(batch):
                                serial.append({"L": L, "d": d, "opcode": f"{op:02x}", "budget": budget, "case": key, "generation": g,
                                               "passed": int(good[j]), "partner_sha256": sha(fresh[j].tobytes()), "offspring_sha256": sha(current[j].tobytes())})
                            if not np.all(good):
                                raise RuntimeError("serial mismatch")
                    # Isolated mutations outside the primary d-byte seed must be erased.
                    controls = [(pos, value) for pos in range(d, L) for value in range(256)]
                    for lo in range(0, len(controls), 256):
                        batch = controls[lo:lo+256]
                        A = np.tile(parents(d, L, op), (len(batch), 1))
                        for j, (pos, value) in enumerate(batch): A[j, pos] = value
                        r = burst(f"erased-L{L}-d{d}-op{op}-b{budget}-v{lo}", lambda: m.run(pairs(A, L), budget=budget))
                        child = r.memory.reshape(len(batch), 17, 2*L)[:, :, L:]
                        expected = parents(d, L, op)
                        for j, (pos, value) in enumerate(batch):
                            ok = np.all(child[j] == expected, axis=1)
                            erased.append({"L": L, "d": d, "opcode": f"{op:02x}", "budget": budget, "position": pos, "value": value,
                                           "alternative": int(value != expected[pos]), "partners": 17, "erased": int(ok.sum())})
                            if not np.all(ok): raise RuntimeError("erased-coordinate mismatch")
                print(f"law L{L} d{d} {op:02x}: {len(cases)} variants complete", flush=True)
    write_csv(OUT / "law_alternatives.csv", rows)
    write_csv(OUT / "law_serial.csv.gz", serial)
    write_csv(OUT / "law_erased.csv", erased)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("qualify", "assay"))
    args = ap.parse_args()
    globals()[args.command]()
