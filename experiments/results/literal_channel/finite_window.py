"""Finite-window check, explicitly separate from infinite-stream equality.

The input is the unmodified tiling T^infinity. Search copies of N=16 bytes,
with both input and output phases free, through execution period 16. This does
not model the zero-partner bootstrap or a shared-memory CPU ring.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

from direct_search import shapes


def check(word, pushed, length):
    k = len(word)
    output = [x[0] for x in reversed(pushed)]
    for phase in range(k):
        for output_phase in range(len(output)):
            parent = list(range(k))
            fixed = [x if x in ("A", "B") else None for x in word]

            def root(i):
                while parent[i] != i:
                    i = parent[i]
                return i

            good = True
            for i in range(length):
                a, b = root((phase + i) % k), root(output[(output_phase + i) % len(output)])
                if a == b:
                    continue
                if fixed[a] and fixed[b] and fixed[a] != fixed[b]:
                    good = False
                    break
                parent[a] = b
                fixed[b] = fixed[b] or fixed[a]
            if good:
                concrete = [{"A": 0xA9, "B": 0x48, None: 0}[fixed[root(i)]] for i in range(k)]
                return {"word": concrete, "input_phase": phase, "output_phase": output_phase}
    return None


def main():
    out = Path(__file__).parent
    rows, first = [], None
    start = time.monotonic()
    for k in range(1, 17):
        count = hits = 0
        for word, pushed in shapes(k, 1):
            count += 1
            witness = check(word, pushed, 16)
            if witness:
                hits += 1
                if first is None:
                    first = {"execution_period": k, "copy_length": 16, **witness}
        rows.append({"width": 1, "execution_period": k, "copy_length": 16,
                     "token_orders": count, "satisfiable_orders": hits})
    assert first and first["execution_period"] == 16
    # Concrete descending-stack execution of the first witness, no SMT.
    word = first["word"]
    ip, reg, chronological, instructions = 0, None, [], 0
    while len(chronological) < 21:
        op = word[ip % len(word)]
        if op == 0xA9:
            reg = word[(ip + 1) % len(word)]
            ip += 2
        else:
            assert op == 0x48 and reg is not None
            chronological.append(reg)
            ip += 1
        instructions += 1
    written = list(reversed(chronological))
    target = [word[(i + first["input_phase"]) % len(word)] for i in range(16)]
    assert written[:16] == target
    assert written[16] != word[first["input_phase"]]
    first.update({"instructions": instructions, "fetched_bytes": ip,
                  "pushes": len(chronological), "written_ascending": written,
                  "copied_window": target, "next_written_byte": written[16],
                  "next_target_byte": word[first["input_phase"]],
                  "scope": "Unmodified periodic input; not a zero-partner harness self-writer"})
    with (out / "finite_search.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (out / "finite_witness.json").write_text(json.dumps(first, indent=2) + "\n")
    # Optional replay of the supplied reference CPU; no GPU/project harness.
    from audit_census import DEFAULT_8086
    cpu_path = DEFAULT_8086 / "m6502.py"
    if cpu_path.exists():
        sys.path.insert(0, str(DEFAULT_8086))
        spec = importlib.util.spec_from_file_location("finite_m6502", cpu_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        trials = []
        tape = bytes(word)
        for budget in (39, 128, 512):
            for mode in ("halt", "faithful"):
                memory = bytearray(tape + bytes(16))
                cpu = module.CPU(memory, 16, brk_mode=mode, budget=budget)
                cpu.run()
                phases = [s for s in range(16) if memory[16:] == tape[s:] + tape[:s]]
                trials.append({"budget": budget, "brk_mode": mode, "exact_phases": phases,
                               "partner": memory[16:].hex(" ")})
                assert not phases
        (out / "6502_finite_harness.json").write_text(json.dumps({
            "source": str(cpu_path), "sha256": hashlib.sha256(cpu_path.read_bytes()).hexdigest(),
            "qualification": "Supplied reference interpreter, not independent full ISA validation",
            "trials": trials}, indent=2) + "\n")
    print(json.dumps({"elapsed_seconds": time.monotonic() - start,
                      "first_period": 16, "instructions": instructions,
                      "matching_orders_period16": rows[-1]["satisfiable_orders"]}))


if __name__ == "__main__":
    main()
