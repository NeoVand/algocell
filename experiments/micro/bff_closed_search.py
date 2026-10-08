"""Does a CLOSED self-replicator exist in BFF + literal push under the benign-tar (no-halt) rule and a wrapping pointer?

    python -m micro.bff_closed_search [--max-period 8] [--out runs/bff_closed_search]

Enumerates every periodic tiling of period 1..max_period over the alphabet {P, [, ], x} (x = a no-op byte; `.`/`,`/
heads are irrelevant to a push-based closed design and would only enlarge the space), tiles it to 64 bytes, and runs it
as A against 3 partners (the all-x partner and two random ones) for 2^13 steps with ip_wrap + literal + nohalt.
A candidate is CLOSED if its pointer never enters the partner in any of the three encounters, and a REPLICATOR if the
partner becomes a >= 75% copy (best cyclic shift, forwards or reversed) in all three. Reports closed replicators,
closed non-replicators (immune tar), and open replicators.
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd

from micro.bff import BFF, PAIR, TAPE
from micro.bff_soup import batch_best_similarity

ALPHA = "P[]x"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-period", type=int, default=8)
    ap.add_argument("--out", default="/Users/neo/repos/algocell/experiments/runs/bff_closed_search")
    ap.add_argument("--nohalt", type=int, default=1)
    ap.add_argument("--min-copy", type=float, default=0.9)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    units = [u for p in range(1, a.max_period + 1) for u in map("".join, itertools.product(ALPHA, repeat=p)) if "P" in u]
    print(len(units), "units")
    rng = np.random.default_rng(0)
    partners = [np.full(TAPE, ord("x"), dtype=np.uint8), rng.integers(0, 256, size=TAPE, dtype=np.uint8), rng.integers(0, 256, size=TAPE, dtype=np.uint8)]
    bff = BFF(max_pairs=3 * 16384, steps=1 << 13, ip_wrap=True, literal=True, nohalt=bool(a.nohalt))
    rows = []
    for i in range(0, len(units), 16384):
        chunk = units[i:i + 16384]
        A = np.stack([np.frombuffer((u * (TAPE // len(u) + 1))[:TAPE].encode(), dtype=np.uint8) for u in chunk])
        pairs = np.concatenate([np.repeat(A, 3, axis=0), np.tile(np.stack(partners), (len(chunk), 1))], axis=1)
        mem, out = bff.execute(pairs)
        best, _, _ = batch_best_similarity(np.repeat(A, 3, axis=0), mem[:, TAPE:])
        best = best.reshape(len(chunk), 3)
        entered = out[:, 1].reshape(len(chunk), 3)
        intact = (mem[:, :TAPE] == np.repeat(A, 3, axis=0)).all(1).reshape(len(chunk), 3)
        for j, u in enumerate(chunk):
            rows.append({"unit": u, "period": len(u), "closed": bool((entered[j] == 0).all()), "entered_any": int(entered[j].sum()),
                         "min_copy": float(best[j].min()), "mean_copy": float(best[j].mean()), "self_intact": bool(intact[j].all())})
        print(f"  {i + len(chunk)}/{len(units)}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "closed_search.csv"), index=False)
    closed_rep = D[D["closed"] & (D["min_copy"] >= a.min_copy)]
    open_rep = D[~D["closed"] & (D["min_copy"] >= a.min_copy)]
    closed_tar = D[D["closed"] & (D["min_copy"] < a.min_copy)]
    # heredity of closed candidates: their offspring (partner after the all-x encounter) run as A against the random partners
    if len(closed_rep):
        from micro.bff import assay
        g2 = []
        for u in closed_rep["unit"]:
            t = np.frombuffer((u * (TAPE // len(u) + 1))[:TAPE].encode(), dtype=np.uint8)
            g2.append(assay(bff, t, n=32, seed=1)["gen2"])
        closed_rep = closed_rep.assign(gen2=g2)
    md = [f"# Closed self-replicators in BFF + P, wrap, {'no-halt' if a.nohalt else 'halting'} (generated)\n",
          f"- {len(D):,} periodic tilings (period ≤ {a.max_period}, alphabet `{ALPHA}`, containing P), 3 partners each (all-x, two random), 2^13 steps",
          f"- closed AND ≥ {a.min_copy:.0%} copy into every partner: **{len(closed_rep)}** (of which heritable, gen2 ≥ 0.3: {int((closed_rep['gen2'] >= 0.3).sum()) if len(closed_rep) else 0}); open replicators (≥ {a.min_copy:.0%} copy, pointer enters): {len(open_rep)}; closed non-replicators (immune): {len(closed_tar)}",
          ""]
    if len(closed_rep):
        md.append(closed_rep.sort_values(["period", "min_copy"], ascending=[True, False]).head(30).to_markdown(index=False, floatfmt=".2f") + "\n")
    md.append("open replicators by period (count):\n" + open_rep.groupby("period").size().to_frame("n").T.to_markdown() + "\n")
    with open(os.path.join(a.out, "NUMBERS_CLOSED_SEARCH.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md[:4]))
    if len(closed_rep):
        print(closed_rep.sort_values(["period", "min_copy"], ascending=[True, False]).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
