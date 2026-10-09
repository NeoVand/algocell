"""C — convention-test soups (registered: REVISION_PREREG.md R3/C, 2026-10-09).

    .venv/bin/python conv_soups.py --yes [--variants randreg,randsp] [--seeds 8001-8020] [--steps 300000]
    .venv/bin/python conv_soups.py --analyse

Stage G conditions at L = 16 (160 x 125 square lattice, 8,192 pairs, 128 instructions, mutation 1/16) with every encounter
under a derived shader: random initial registers (randreg) or a random initial stack pointer (randsp). Records the three
most common tapes and their shares every 250 steps to 20,000 and every 2,500 after (runs/conv/<v>_s<seed>.jsonl) and soup
snapshots at 20,000, 100,000 and 300,000 steps (.npy). --analyse dates the first replicator (first record at which a top-3
tape with >= 0.5% of cells passes the culture test under the world's convention), classifies it (load-push word; open:
pointer enters the partner in >= half of 16 encounters), and classifies 64 random cells of each snapshot (heritable,
confined in 16 of 16 encounters) under the convention. Output: results/conv/conv_worlds.csv, REPORT.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from algocell_exp import conv  # noqa: E402
from algocell_exp import exectrace as X  # noqa: E402
from algocell_exp.soup import Soup  # noqa: E402
from threshold_grid import is_loadpush  # noqa: E402

L = 16
RUNS = os.path.join(HERE, "runs", "conv")
OUT = os.path.join(HERE, "results", "conv")
SNAPS = (20000, 100000, 300000)


def hx(t):
    return " ".join(f"{b:02x}" for b in t)


def run_world(v: str, seed: int, steps: int) -> None:
    os.makedirs(RUNS, exist_ok=True)
    path = os.path.join(RUNS, f"{v}_s{seed}.jsonl")
    soup = Soup(160, 125, "square", L, seed, 8192, 128, 4, [], shader_variant=v)
    step, t0 = 0, time.time()
    with open(path, "w") as f:
        while step <= steps:
            if step in SNAPS or step == steps:
                np.save(os.path.join(RUNS, f"{v}_s{seed}_t{step}.npy"), soup.read_soup())
            s = soup.read_soup()
            u, n = np.unique(s, axis=0, return_counts=True)
            o = np.argsort(-n)[:3]
            f.write(json.dumps({"step": step, "top3": [hx(u[i]) for i in o], "shares": [float(n[i] / len(s)) for i in o], "zero_frac": float((s == 0).mean())}) + "\n")
            dt = 250 if step < 20000 else 2500
            soup.step(dt)
            step += dt
    print(f"{v} seed {seed}: {steps} steps in {time.time() - t0:.0f} s", flush=True)


def classify_cells(cells, v, rng):
    res = conv.assay_many_conv(cells, v, n=32, seed=int(rng.integers(1 << 31)))
    herit = np.array([r["is_replicator"] for r in res])
    P = rng.integers(0, 256, size=(len(cells) * 16, L), dtype=np.uint8)
    _, masks = conv.execute(np.concatenate([np.repeat(cells, 16, 0), P], 1), L, 128, v, int(rng.integers(1 << 31)), traced=True)
    conf = ~X.exec_positions(masks, 2 * L)[:, L:].any(axis=1).reshape(len(cells), 16).any(axis=1)
    return herit, conf


def analyse() -> None:
    rows = []
    for v in ("randreg", "randsp"):
        for seed in range(8001, 8021):
            path = os.path.join(RUNS, f"{v}_s{seed}.jsonl")
            if not os.path.exists(path):
                continue
            recs = [json.loads(l) for l in open(path)]
            rng = np.random.default_rng([seed, 1 if v == "randreg" else 2])
            cache = {}
            first = None
            for r in recs:
                for t, sh in zip(r["top3"], r["shares"]):
                    if sh < 0.005:
                        continue
                    if t not in cache:
                        tape = np.array([int(b, 16) for b in t.split()], np.uint8)[None]
                        cache[t] = conv.assay_many_conv(tape, v, n=64, seed=int(rng.integers(1 << 31)))[0]["is_replicator"]
                    if cache[t]:
                        first = (r["step"], t)
                        break
                if first:
                    break
            row = {"variant": v, "seed": seed, "horizon": recs[-1]["step"]}
            if first:
                tape = np.array([int(b, 16) for b in first[1].split()], np.uint8)
                P = rng.integers(0, 256, size=(16, L), dtype=np.uint8)
                _, m = conv.execute(np.concatenate([np.repeat(tape[None], 16, 0), P], 1), L, 128, v, int(rng.integers(1 << 31)), traced=True)
                row.update({"t_first": first[0], "first_tape": first[1], "first_loadpush": is_loadpush(tape),
                            "first_open": float(X.exec_positions(m, 2 * L)[:, L:].any(axis=1).mean()) >= 0.5})
            for st in SNAPS:
                f = os.path.join(RUNS, f"{v}_s{seed}_t{st}.npy")
                if os.path.exists(f):
                    soup = np.load(f)
                    cells = soup[rng.choice(len(soup), size=64, replace=False)]
                    herit, conf = classify_cells(cells, v, rng)
                    u, n = np.unique(soup, axis=0, return_counts=True)
                    row.update({f"herit_t{st}": float(herit.mean()), f"conf_given_herit_t{st}": float(conf[herit].mean()) if herit.any() else np.nan,
                                f"modal_t{st}": hx(u[int(np.argmax(n))]), f"modal_share_t{st}": float(n.max() / len(soup))})
            rows.append(row)
            print(row, flush=True)
    os.makedirs(OUT, exist_ok=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "conv_worlds.csv"), index=False)
    lines = ["# C — convention-test soups (generated by `conv_soups.py --analyse`)", ""]
    for v, d in D.groupby("variant"):
        n = len(d)
        hasr = d.t_first.notna() if "t_first" in d else pd.Series(False, index=d.index)
        lp = int(d.get("first_loadpush", pd.Series(False)).fillna(False).astype(bool).sum())
        op = int(d.get("first_open", pd.Series(False)).fillna(False).astype(bool).sum())
        fin = d.get("conf_given_herit_t300000")
        closed = int((fin > 0.5).sum()) if fin is not None else 0
        lines.append(f"- {v}: {n} worlds; with a replicator {int(hasr.sum())}; first replicator a load–push word in {lp}, open in {op}; "
                     f"first replicator step median {d.t_first.median() if hasr.any() else float('nan'):.0f}; population closed (most heritable cells confined) at 300,000 steps in {closed} of {int(fin.notna().sum()) if fin is not None else 0}; "
                     f"heritable share at 300,000 median {d.get('herit_t300000', pd.Series(dtype=float)).median():.2f}")
    if "randreg" in set(D.variant):
        d = D[D.variant == "randreg"]
        lines.append(f"- C1 (load–push first in >= 18 of 20 under randreg): {'met' if int(d.first_loadpush.fillna(False).astype(bool).sum()) >= 18 else 'NOT met'}")
    if "randsp" in set(D.variant):
        d = D[D.variant == "randsp"]
        nr = int(d.t_first.notna().sum())
        lines.append(f"- C2 (open first in >= 15 of 20 worlds with a replicator under randsp): {'met' if int(d.first_open.fillna(False).astype(bool).sum()) >= 15 else 'NOT met'} ({int(d.first_open.fillna(False).astype(bool).sum())} of {nr})")
    open(os.path.join(OUT, "REPORT.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="randreg,randsp")
    ap.add_argument("--seeds", default="8001-8020")
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--runs", default="", help="directory of the run files (default runs/conv; Modal outputs: runs/conv_modal/conv)")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    global RUNS
    if a.runs:
        RUNS = a.runs
    if a.analyse:
        analyse()
        return
    if not a.yes:
        sys.exit("refusing to run: pass --yes after pausing the browser simulation (local GPU)")
    lo, hi = (int(x) for x in a.seeds.split("-"))
    for seed in range(lo, hi + 1):
        for v in a.variants.split(","):
            if os.path.exists(os.path.join(RUNS, f"{v}_s{seed}_t{a.steps}.npy")):
                continue
            run_world(v, seed, a.steps)


if __name__ == "__main__":
    main()
