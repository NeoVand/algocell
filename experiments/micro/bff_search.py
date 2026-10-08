"""Constructive search for an open (straight-line) full replicator in wrap-BFF (THEORY.md P1(b), pre-stated 2026-10-08).

    python -m micro.bff_search [--out runs/bff_search] [--max-prefix 3] [--max-period 5]

Standard BFF needs no search: a straight-line program executes at most 128 instruction bytes (one pass over the pair),
and copying 64 bytes into the partner needs ≥ 64 writes plus ≥ 64 moves of each head (192 instructions), so no
straight-line program can copy itself in one encounter — the write-bandwidth criterion of THEORY.md, as a count.
With a wrapping pointer the budget is 2^13 steps and the question is open; this script enumerates every straight-line
program of the form  prefix (0–3 instructions)  +  tiling unit (period 2–5, at least one copy instruction `.`/`,`)
repeated to 64 bytes, over the eight non-bracket instructions, and runs each as A against random partners.
Stage 1: 2 random partners; a candidate passes the alphabet filter if ≥ 75% of each partner's bytes afterwards lie in the
candidate's own byte alphabet (a cheap necessary condition for a ≥ 75% copy), and survives if both partners are then
≥ 50% copies at the best cyclic shift (forwards or reversed). Stage 2: survivors get the full culture test (32
partners; copies at best cyclic shift forwards or reversed, gen2, self-damage, pointer-entered-partner).
Writes candidates.csv (survivors with their assay), NUMBERS_SEARCH.md.
"""

from __future__ import annotations

import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

from micro.bff import BFF, PAIR, TAPE, assay
from micro.bff_soup import batch_best_similarity

INS = "<>{}+-.,"


def build(prefix: str, unit: str) -> np.ndarray:
    body = prefix + (unit * (TAPE // len(unit) + 1))
    return np.frombuffer(body[:TAPE].encode(), dtype=np.uint8).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/bff_search")
    ap.add_argument("--max-prefix", type=int, default=3)
    ap.add_argument("--max-period", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    units = [u for p in range(2, a.max_period + 1) for u in map("".join, itertools.product(INS, repeat=p)) if ("." in u or "," in u)]
    prefixes = [""] + [p for k in range(1, a.max_prefix + 1) for p in map("".join, itertools.product(INS, repeat=k))]
    print(f"{len(units)} units × {len(prefixes)} prefixes = {len(units) * len(prefixes):,} candidates", flush=True)
    bff = BFF(max_pairs=a.batch * 2, steps=1 << 13, ip_wrap=True)
    t0 = time.time()
    survivors = []
    n_done = 0
    n_alpha = 0
    cands = ((p, u) for p in prefixes for u in units)
    while True:
        chunk = list(itertools.islice(cands, a.batch))
        if not chunk:
            break
        A = np.stack([build(p, u) for p, u in chunk])
        alph = np.zeros((len(chunk), 256), dtype=bool)
        for i, row in enumerate(A):
            alph[i, np.unique(row)] = True
        partners = rng.integers(0, 256, size=(len(chunk), 2, TAPE), dtype=np.uint8)
        pairs = np.concatenate([np.repeat(A[:, None, :], 2, axis=1), partners], axis=2).reshape(-1, PAIR)
        mem, out = bff.execute(pairs)
        B = mem[:, TAPE:].reshape(len(chunk), 2, TAPE)
        inalph = alph[np.arange(len(chunk))[:, None, None], B].mean(2)  # (n, 2)
        pre = np.flatnonzero((inalph >= 0.75).all(1))
        n_alpha += len(pre)
        if len(pre):
            best, _, _ = batch_best_similarity(np.repeat(A[pre], 2, axis=0), B[pre].reshape(-1, TAPE))
            best = best.reshape(-1, 2)
            for j, i in enumerate(pre):
                if best[j].min() >= 0.5:
                    survivors.append((chunk[i][0], chunk[i][1], float(inalph[i].min()), float(best[j].min()), float(best[j].mean())))
        n_done += len(chunk)
        if (n_done // a.batch) % 50 == 0:
            print(f"  {n_done:,} candidates, {n_alpha} alphabet survivors, {len(survivors)} with both partners ≥ 0.5 copied, {time.time() - t0:.0f}s", flush=True)
    print(f"stage 1 done: {n_done:,} candidates, {n_alpha} alphabet survivors, {len(survivors)} with both partners ≥ 0.5 copied, {time.time() - t0:.0f}s", flush=True)
    rows = []
    bff2 = BFF(max_pairs=64, steps=1 << 13, ip_wrap=True)
    for p, u, frac, smin, smean in survivors:
        t = build(p, u)
        r = assay(bff2, t, n=32, seed=a.seed)
        rows.append({"prefix": p, "unit": u, "stage1_min_in_alphabet": frac, "stage1_min_similarity": smin, "stage1_mean_similarity": smean, **r, "program": t.tobytes().decode()})
    S = pd.DataFrame(rows).sort_values("copies", ascending=False) if rows else pd.DataFrame(columns=["prefix", "unit", "copies", "gen2", "self_damage", "entered"])
    S.to_csv(os.path.join(a.out, "candidates.csv"), index=False)
    best = S.iloc[0] if len(S) else None
    md = ["# Search for an open full replicator in wrap-BFF (generated)\n",
          f"- enumerated: {n_done:,} straight-line programs (prefix ≤ {a.max_prefix} instructions over `{INS}`, tiling unit of period 2–{a.max_period} with a copy instruction, repeated to 64 bytes); "
          f"2 random partners each, 2^13 steps, wrapping pointer; alphabet filter passed (≥ 75% of each partner's bytes in the program's alphabet): {n_alpha:,}; both partners ≥ 50% copied: {len(survivors)}",
          f"- stage 2 (32 random partners): programs with copies ≥ 0.5: {int((S['copies'] >= 0.5).sum()) if len(S) else 0}; with copies ≥ 0.5 and gen2 ≥ 0.3: {int(((S['copies'] >= 0.5) & (S['gen2'] >= 0.3)).sum()) if len(S) else 0}; "
          f"best copies: {best['copies']:.2f} (`{best['prefix']}` + `{best['unit']}`, self-damage {best['self_damage']:.2f}, gen2 {best['gen2']:.2f})" if best is not None else "- stage 2: no survivors",
          f"- wall time {time.time() - t0:.0f}s\n"]
    if len(S):
        md.append(S.head(20)[["prefix", "unit", "copies", "score", "gen2", "self_damage", "entered", "executed"]].to_markdown(index=False, floatfmt=".2f") + "\n")
    with open(os.path.join(a.out, "NUMBERS_SEARCH.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
