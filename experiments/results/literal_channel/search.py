"""CPU-only, exhaustive symbolic literal-channel search (Python + Z3).

An execution period k contains nL loads (w+1 bytes) and nP pushes (1 byte).
Literal bytes are unconstrained 8-bit values, including opcode values. Infinite
input/output equality implies input period gcd(k, w*nP). We impose that necessary
condition first, then exact descending-stack output equality when requested.
There is no sampling of operands or token orders. A SAT result is replayed by an
independent concrete decoder. See CHECK.md for scope and the opcode quotient.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from pathlib import Path

import z3


def solve(k, nl, np, w, layout, loads=(0xA9,), pushes=(0x48,), exact=True,
          timeout_ms=10000, minimal_cycle=False):
    """All token orders, all byte assignments, all phases for these counts."""
    assert set(loads).isdisjoint(pushes)
    g = math.gcd(k, w * np)
    nt = nl + np
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    code = z3.Array("code", z3.IntSort(), z3.BitVecSort(8))
    isload = [z3.Bool(f"load_{i}") for i in range(nt)]
    offset = [z3.Int(f"offset_{i}") for i in range(nt + 1)]
    solver.add(offset[0] == 0, offset[-1] == k,
               z3.PbEq([(x, 1) for x in isload], nl), isload[0])
    if minimal_cycle:
        # A minimal decoder cycle visits each code residue at most once.
        # See CHECK.md: every counterexample reduces to such a cycle.
        solver.add(z3.Distinct([offset[i] % g for i in range(nt)]))
    # Starting at a load loses no cyclic token order; output phase is free.
    for i in range(nt):
        solver.add(offset[i + 1] == offset[i] + z3.If(isload[i], w + 1, 1))
        byte = z3.Select(code, offset[i] % g)
        solver.add(z3.If(isload[i], z3.Or([byte == x for x in loads]),
                         z3.Or([byte == x for x in pushes])))
    if exact:
        phase = z3.Int("phase")
        rank = [z3.Int(f"rank_{i}") for i in range(nt + 1)]
        reg = [[z3.BitVec(f"reg_{i}_{j}", 8) for j in range(w)]
               for i in range(nt + 1)]
        solver.add(phase >= 0, phase < g, rank[0] == 0, rank[-1] == np)
        for j in range(w):
            solver.add(reg[0][j] == reg[-1][j])
        for i in range(nt):
            solver.add(rank[i + 1] == rank[i] + z3.If(isload[i], 0, 1))
            for j in range(w):
                solver.add(reg[i + 1][j] == z3.If(
                    isload[i], z3.Select(code, (offset[i] + 1 + j) % g), reg[i][j]))
                # Reverse push BLOCK order, not the bytes within a block.
                target = (phase + (np - 1 - rank[i]) * w + j) % g
                solver.add(z3.Implies(z3.Not(isload[i]),
                                     z3.Select(code, target) == reg[i][layout[j]]))
    result = solver.check()
    witness = None
    if result == z3.sat:
        model = solver.model()
        word = [model.eval(z3.Select(code, i % g), model_completion=True).as_long()
                for i in range(k)]
        tokens = ["L" if z3.is_true(model.eval(x)) else "P" for x in isload]
        witness = {"word": word, "tokens": "".join(tokens)}
        if exact:
            witness.update(replay(word, w, layout, loads, pushes))
    return str(result), witness, solver.reason_unknown() if result == z3.unknown else ""


def replay(word, w, layout, loads=(0xA9,), pushes=(0x48,)):
    """Concrete, non-SMT verification over one complete execution period."""
    k = len(word)
    ip, reg, blocks, tokens = 0, None, [], []
    while ip < k:
        op = word[ip]
        if op in loads:
            assert ip + w < k
            reg = word[ip + 1:ip + 1 + w]
            ip += w + 1
            tokens.append("L")
        else:
            assert op in pushes and reg is not None
            blocks.append([reg[j] for j in layout])
            ip += 1
            tokens.append("P")
    assert ip == k and blocks
    written = list(itertools.chain.from_iterable(reversed(blocks)))
    length = math.lcm(k, len(written))
    phases = [s for s in range(k) if all(
        written[i % len(written)] == word[(i + s) % k] for i in range(length))]
    assert phases, (word, written)
    primitive = next(d for d in range(1, k + 1) if k % d == 0 and all(
        word[i] == word[i % d] for i in range(k)))
    return {"written": written, "phases": phases, "primitive_period": primitive,
            "replayed_tokens": "".join(tokens)}


def write_rows(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pair_checks(out):
    """Independent finite-window enumeration of every layout, widths 1..8."""
    rows, examples = [], []
    a, b = 0xA9, 0x48
    for w in range(1, 9):
        ell, bound = w + 2, 2 * w + 2 - math.gcd(2, w)
        # Test exact byte equalities for EVERY layout, without calling
        # Fine-Wilf or sampling any literal values. No SMT solver in this check.
        for perm in itertools.permutations(range(w)):
            hits = []
            # Each union-find node is a freely assignable byte position.
            for phase in range(ell):
                parent = list(range(ell))

                def root(x):
                    while parent[x] != x:
                        x = parent[x]
                    return x

                for i in range(bound):
                    x, y = root((phase + i) % ell), root(1 + perm[i % w])
                    parent[x] = y
                if root(0) != root(ell - 1):
                    hits.append(phase)
            alternating = w % 2 == 0 and all(
                (perm[i] - perm[i - 1]) % 2 for i in range(1, w))
            assert bool(hits) == alternating
            rows.append({"width": w, "layout": " ".join(map(str, perm)),
                         "copy_bound": bound, "solutions": bool(hits),
                         "alternating_layout": alternating,
                         "phases": " ".join(map(str, hits))})
        print(f"pair width={w}: all {math.factorial(w)} layouts checked", flush=True)
    # Sharp finite-window failures of the draft's odd-width 2w bound.
    for word, perm in [([a, a, b], (0,)), ([a, b, a, a, b], (1, 0, 2))]:
        w = len(perm)
        lit = word[1:1 + w]
        block = [lit[i] for i in perm]
        n = 2 * w
        assert all(word[i % len(word)] == block[i % w] for i in range(n))
        assert word[n % len(word)] != block[n % w]
        examples.append({"width": w, "word": word, "layout": list(perm),
                         "written_block": block, "matching_length": n})
    write_rows(out / "pair_layouts.csv", rows)
    (out / "finite_bound_counterexamples.json").write_text(json.dumps(examples, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-period", type=int, default=64)
    parser.add_argument("--control-period", type=int, default=16)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--budget-seconds", type=float, default=850)
    parser.add_argument("--minimal-cycle", action="store_true")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    rows, witnesses = [], []
    metadata = {"z3_version": z3.get_version_string(), "max_period": args.max_period,
                "control_period": args.control_period, "cpu_only": True,
                "minimal_cycle": args.minimal_cycle,
                "width1_multi_opcode_method": "exact load/push alphabet quotient"}
    for w, limit, exact in [(1, args.max_period, False),
                            (2, args.control_period, True),
                            (3, args.control_period, True)]:
        for k in range(1, limit + 1):
            for nl in range(1, (k - 1) // (w + 1) + 1):
                np = k - (w + 1) * nl
                elapsed = time.monotonic() - start
                if elapsed >= args.budget_seconds:
                    metadata["budget_exhausted"] = True
                    break
                begin = time.monotonic()
                g = math.gcd(k, w * np)
                if args.minimal_cycle and nl + np > g:
                    result, witness, reason, method = "unsat", None, "", "minimal-cycle pigeonhole"
                elif g == 1:
                    # A constant word cannot have disjoint load/push opcodes.
                    result, witness, reason, method = "unsat", None, "", "gcd=1"
                else:
                    result, witness, reason = solve(k, nl, np, w, tuple(range(w)),
                                                    exact=exact, timeout_ms=args.timeout_ms,
                                                    minimal_cycle=args.minimal_cycle)
                    method = "SMT exact" if exact else "SMT necessary grammar"
                    if not exact and result == "sat":
                        result, witness, reason = solve(k, nl, np, w, tuple(range(w)),
                                                        exact=True, timeout_ms=args.timeout_ms,
                                                        minimal_cycle=args.minimal_cycle)
                        method = "SMT grammar + exact"
                rows.append({"width": w, "execution_period": k, "loads": nl,
                             "pushes": np, "output_period": w * np, "gcd": g,
                             "token_orders_starting_load": math.comb(nl + np - 1, nl - 1),
                             "result": result, "method": method,
                             "seconds": f"{time.monotonic() - begin:.6f}", "reason": reason})
                if witness is not None:
                    witnesses.append({"width": w, "execution_period": k,
                                      "loads": nl, "pushes": np, **witness})
            print(f"width={w} k={k} elapsed={time.monotonic() - start:.2f}s", flush=True)
            if metadata.get("budget_exhausted"):
                break
        if metadata.get("budget_exhausted"):
            break
    write_rows(args.out / f"{args.prefix}symbolic_search.csv", rows)
    (args.out / f"{args.prefix}witnesses.json").write_text(json.dumps(witnesses, indent=2) + "\n")
    if not args.minimal_cycle:
        pair_checks(args.out)
    metadata["elapsed_seconds"] = time.monotonic() - start
    metadata["cases"] = len(rows)
    metadata["results"] = {r: sum(x["result"] == r for x in rows)
                           for r in ("sat", "unsat", "unknown")}
    (args.out / f"{args.prefix}run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
