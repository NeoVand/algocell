"""DZ (REVISION_PREREG): the lethality dial in the Z80, L = 16. World runner (run_world) and analysis (--analyse).

    .venv/bin/modal run modal_dial.py                      # all worlds on Modal (outputs under /runs/dial)
    .venv/bin/python dial_soups.py --analyse --runs runs/dial/dial
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

L = 16
DIAL = {"lp0": 0.0, "lp001": 0.01, "lp003": 0.03, "lp01": 0.1, "lp03": 0.3, "lp1": 1.0}
SEEDS = range(9001, 9011)
SNAPS = (2000, 20000, 100000, 300000)


def hx(t):
    return " ".join(f"{b:02x}" for b in t)


def schedule(step):
    return 50 if step < 2000 else (250 if step < 20000 else 2500)


def run_world(v: str, seed: int, steps: int, outdir: str) -> dict:
    from algocell_exp.soup import Soup
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{v}_s{seed}.jsonl")
    soup = Soup(160, 125, "square", L, seed, 8192, 128, 4, [], shader_variant=v)
    step, t0 = 0, time.time()
    with open(path, "w") as f:
        f.write(json.dumps({"kind": "condition", "variant": v, "p": DIAL.get(v), "seed": seed, "steps": steps}) + "\n")
        while step <= steps:
            s = soup.read_soup()
            if step in SNAPS or step == steps:
                np.save(os.path.join(outdir, f"{v}_s{seed}_t{step}.npy"), s)
            u, n = np.unique(s, axis=0, return_counts=True)
            o = np.argsort(-n)[:3]
            f.write(json.dumps({"step": step, "top3": [hx(u[i]) for i in o], "shares": [float(n[i] / len(s)) for i in o], "zero_frac": float((s == 0).mean())}) + "\n")
            if step >= steps:
                break
            dt = min(schedule(step), steps - step)
            soup.step(dt)
            step += dt
    return {"variant": v, "seed": seed, "steps": steps, "wall_s": round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------------------------ analysis
def _patch(v):
    """Route the standard executors (assay, exectrace) through the world's own variant, with a fresh seed per call."""
    from algocell_exp import assay as A
    from algocell_exp import conv
    from algocell_exp import exectrace as X
    ctr = {"k": 0}

    def nxt():
        ctr["k"] += 1
        return zlib.crc32(f"{v}:{ctr['k']}".encode()) & 0x7FFFFFFF

    def ex(pairs, tape_length, z80_steps, suppress=(), mem_length=None, zero_halts=False):
        return conv.execute(pairs, tape_length, z80_steps, v, nxt(), suppress)

    def ext(pairs, tape_length, steps, suppress=(), zero_halts=False, **kw):
        return conv.execute(pairs, tape_length, steps, v, nxt(), suppress, traced=True)
    A.execute_pairs = ex
    X.execute_pairs_traced = ext


def analyse(runs: str, out: str, variants=None, seeds=None, snap=300000) -> None:
    from algocell_exp import assay as A
    from algocell_exp import exectrace as X
    import population_classes as PC
    from threshold_grid import is_loadpush
    rows = []
    variants = variants or list(DIAL)
    for v in variants:
        _patch(v)
        for seed in (seeds or SEEDS):
            path = os.path.join(runs, f"{v}_s{seed}.jsonl")
            if not os.path.exists(path):
                continue
            recs = [json.loads(x) for x in open(path)][1:]
            rng = np.random.default_rng([seed, int(DIAL.get(v, 0) * 1000), 11])
            cache = {}

            def judge(t):
                if t not in cache:
                    tape = np.array([int(b, 16) for b in t.split()], np.uint8)
                    r = A.assay_many(tape[None], n=64, seed=int(rng.integers(1 << 31)))[0]
                    P = rng.integers(0, 256, size=(16, L), dtype=np.uint8)
                    _, m = X.execute_pairs_traced(np.concatenate([np.repeat(tape[None], 16, 0), P], 1), L, 128)
                    ent = X.exec_positions(m, 2 * L)[:, L:].any(axis=1)
                    cache[t] = (bool(r["is_replicator"]), float(ent.mean()), bool(is_loadpush(tape)))
                return cache[t]
            t_rep = t_closed = None
            first = None
            for r in recs:
                for t, sh in zip(r["top3"], r["shares"]):
                    if sh < 0.005:
                        continue
                    her, ent, lp = judge(t)
                    if her and t_rep is None:
                        t_rep, first = r["step"], (t, ent, lp)
                    if her and ent == 0.0 and t_closed is None:
                        t_closed = r["step"]
                if t_rep is not None and t_closed is not None:
                    break
            row = {"variant": v, "p": DIAL.get(v), "seed": seed, "horizon": recs[-1]["step"], "t_rep": t_rep, "t_closed": t_closed,
                   "first_tape": first[0] if first else None, "first_open": (first[1] >= 0.5) if first else None, "first_loadpush": first[2] if first else None}
            f = os.path.join(runs, f"{v}_s{seed}_t{snap}.npy")
            if os.path.exists(f):
                soup = np.load(f)
                snapd, _ = PC.classify(soup, L, [], False, np.random.default_rng([seed, 23]))
                row.update({k: snapd[k] for k in ("frac_heritable", "frac_confined_of_heritable", "frac_regenerator_of_heritable", "frac_intermediate_of_heritable", "frac_transmitter_of_heritable", "median_unexec_sites_transmitters")})
            rows.append(row)
            print(row, flush=True)
    D = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    D.to_csv(os.path.join(out, "dial_worlds.csv"), index=False)
    report(D, out)


def _km_median(t, horizon):
    t = np.array([x if x == x and x is not None else np.inf for x in t], float)
    n = len(t)
    for k, x in enumerate(np.sort(t)):
        if (k + 1) / n >= 0.5:
            return x if np.isfinite(x) else np.inf
    return np.inf


def _perm_spearman(x, y, n=20000, seed=0):
    from pandas import Series
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rx, ry = Series(x).rank().values, Series(y).rank().values
    r = np.corrcoef(rx, ry)[0, 1]
    if not np.isfinite(r):
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n)])
    return r, float((null >= r).mean())


def report(D, out):
    from stats_tests import cochran_armitage
    lines = ["# DZ — the lethality dial in the Z80, L = 16 (generated by `dial_soups.py --analyse`)", "",
             "| p | worlds | with a replicator | first replicator open | first is load–push | KM median t_rep | worlds closed (top tape) | KM median t_closed | heritable at 300,000 (median) | transmitters of heritable (median) | regenerators of heritable (median) | transmitter-majority worlds |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ranks = {v: i for i, v in enumerate(DIAL)}
    for v, d in D.groupby("variant", sort=False):
        hz = d.horizon.max()
        tr = d.get("frac_transmitter_of_heritable", pd.Series(dtype=float))
        rg = d.get("frac_regenerator_of_heritable", pd.Series(dtype=float))
        lines.append(f"| {DIAL[v]} | {len(d)} | {int(d.t_rep.notna().sum())} | {int(d.first_open.fillna(False).astype(bool).sum())} | {int(d.first_loadpush.fillna(False).astype(bool).sum())} | "
                     f"{_km_median(d.t_rep.tolist(), hz):,.0f} | {int(d.t_closed.notna().sum())} | {_km_median(d.t_closed.tolist(), hz):,.0f} | "
                     f"{d.get('frac_heritable', pd.Series(dtype=float)).median():.2f} | {tr.median():.2f} | {rg.median():.2f} | {int((tr > 0.5).sum())} |")
    D = D.copy()
    D["rank"] = D.variant.map(ranks)
    g = D.groupby("rank")
    z, p2 = cochran_armitage(g.first_open.apply(lambda s: int(s.fillna(False).astype(bool).sum())).values, g.size().values, g.size().index.values)
    tc = D.t_closed.fillna(D.horizon).astype(float)
    r2, p2s = _perm_spearman(D["rank"], tc)
    r3, p3 = _perm_spearman(D["rank"], D.get("frac_transmitter_of_heritable", pd.Series(np.nan, index=D.index)))
    d0, d1 = D[D.variant == "lp0"], D[D.variant == "lp1"]
    lines += ["",
              f"- DZ1: open-first at p = 0 {int(d0.first_open.fillna(False).astype(bool).sum())} of {len(d0)} (needs ≥ 8), at p = 1 {int(d1.first_open.fillna(False).astype(bool).sum())} of {len(d1)} (needs ≤ 2); trend z = {z:.2f}, one-sided P = {p2 / 2 if z < 0 else 1 - p2 / 2:.4f} (falling with p needs z < 0)",
              f"- DZ2: Spearman(rank of p, t_closed censored at the horizon) = {r2:.2f}, permutation P = {p2s:.4f} (needs > 0, P < 0.05)",
              f"- DZ3: Spearman(rank of p, transmitter share of heritable cells at 300,000) = {r3:.2f}, permutation P = {p3:.4f} (needs > 0, P < 0.05); transmitter-majority worlds at p = 0: {int((d0.get('frac_transmitter_of_heritable', pd.Series(dtype=float)) > 0.5).sum())} (needs ≤ 3), at p = 1: {int((d1.get('frac_transmitter_of_heritable', pd.Series(dtype=float)) > 0.5).sum())} (needs ≥ 7)"]
    open(os.path.join(out, "DIAL.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--runs", default=os.path.join(HERE, "runs", "dial", "dial"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "dial"))
    ap.add_argument("--smoke", action="store_true", help="local smoke: lp0 and lp1, seed 9001, 2,000 steps")
    a = ap.parse_args()
    if a.smoke:
        for v in ("lp0", "lp1"):
            print(run_world(v, 9001, 2000, os.path.join(HERE, "runs", "dial_smoke")))
        return
    if a.analyse:
        analyse(a.runs, a.out)


if __name__ == "__main__":
    main()
