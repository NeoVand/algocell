"""Independent exhaustive token-order search with symbolic union-find operands.

Uses no SMT solver. Each literal position is a free byte; equality components
may acquire opcode constants. Unequal constants in a component are a conflict.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path


def shapes(remaining, w, word=None, pushed=None, last=None):
    if word is None:
        word, pushed = ["A"] + list(range(1, w + 1)), []
        remaining -= w + 1
        last = tuple(range(1, w + 1))
    if remaining == 0:
        if pushed:
            yield word, pushed
        return
    if remaining < 0:
        return
    if remaining >= w + 1:
        position = len(word)
        literal = tuple(range(position + 1, position + 1 + w))
        yield from shapes(remaining - w - 1, w, word + ["A"] + list(literal), pushed, literal)
    if remaining:
        yield from shapes(remaining - 1, w, word + ["B"], pushed + [last], last)


def check(word, pushed, w):
    k, p = len(word), len(pushed) * w
    g = math.gcd(k, p)
    constants = {}
    for i, symbol in enumerate(word):
        if symbol in ("A", "B"):
            if i % g in constants and constants[i % g] != symbol:
                return None, 0
            constants[i % g] = symbol
    written = [j % g for block in reversed(pushed) for j in block]
    tested = 0
    for phase in range(g):
        tested += 1
        parent = list(range(g))
        fixed = [constants.get(i) for i in range(g)]

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        good = True
        for i, source in enumerate(written):
            a, b = root((phase + i) % g), root(source)
            if a == b:
                continue
            if fixed[a] is not None and fixed[b] is not None and fixed[a] != fixed[b]:
                good = False
                break
            parent[a] = b
            fixed[b] = fixed[b] or fixed[a]
        if good:
            code = [{"A": 0xA9, "B": 0x48, None: 0}[fixed[root(i % g)]] for i in range(k)]
            return {"word": code, "phase": phase}, tested
    return None, tested


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-period", type=int, default=28)
    ap.add_argument("--control-period", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    rows, witnesses = [], []
    start = time.monotonic()
    for w, limit in [(1, args.max_period), (2, args.control_period), (3, args.control_period)]:
        for k in range(1, limit + 1):
            t0, orders, phases, hits = time.monotonic(), 0, 0, 0
            for word, pushed in shapes(k, w):
                orders += 1
                witness, tested = check(word, pushed, w)
                phases += tested
                if witness is not None:
                    hits += 1
                    if not any(x["width"] == w and x["execution_period"] == k for x in witnesses):
                        witnesses.append({"width": w, "execution_period": k, **witness})
            rows.append({"width": w, "execution_period": k, "token_orders": orders,
                         "phases_tested": phases, "satisfiable_orders": hits,
                         "seconds": f"{time.monotonic() - t0:.6f}"})
            print(f"direct width={w} k={k} orders={orders} hits={hits}", flush=True)
    with (args.out / "direct_search.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "direct_witnesses.json").write_text(json.dumps(witnesses, indent=2) + "\n")
    (args.out / "direct_run.json").write_text(json.dumps({
        "elapsed_seconds": time.monotonic() - start, "max_period": args.max_period,
        "control_period": args.control_period, "cpu_only": True,
        "width1_orders": sum(r["token_orders"] for r in rows if r["width"] == 1)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
