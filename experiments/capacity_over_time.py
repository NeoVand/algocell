"""Capacity for inherited variation, executed set and confinement of the dominant tape at every snapshot of every world
(NATURE_PLAN Move 2b; REVISION_PREREG L2/L3 use the same measures on Stage L).

    .venv/bin/python capacity_over_time.py --yes [--stages G,K,I] [--out results/capacity_time]

Per snapshot: the modal tape (most common exact tape), its share, mutational scan (32 values per position, 32 partners),
traced execution against 64 partners (entered fraction, executed union), and the number of transmissible sites that are
never executed. Output: <out>/capacity_over_time.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "manuscript", "figures"))
import soup_stills as ss  # noqa: E402
from algocell_exp import exectrace as X  # noqa: E402
from mutscan import scan_tape, summarise  # noqa: E402

R = os.path.join(HERE, "results")
STAGES = {"G": (os.path.join(HERE, "runs", "stageG"), os.path.join(R, "stageG", "stageG", "stage_g_runs.csv"), False),
          "K": (os.path.join(HERE, "runs", "stageK", "stageK"), os.path.join(R, "stageK", "stageK", "stage_g_runs.csv"), False),
          "I": (os.path.join(HERE, "runs", "stageI"), os.path.join(R, "stageI", "stageI", "stage_g_runs.csv"), True),
          "L": (os.path.join(HERE, "runs", "stageL", "stageL"), os.path.join(R, "stageL", "stageL", "stage_g_runs.csv"), False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="G,K,I")
    ap.add_argument("--out", default=os.path.join(R, "capacity_time"))
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if not a.yes:
        sys.exit("refusing to run: pass --yes after pausing the browser simulation (local GPU)")
    os.makedirs(a.out, exist_ok=True)
    out_csv = os.path.join(a.out, "capacity_over_time.csv")
    rows = []
    t0 = time.time()
    for stage in a.stages.split(","):
        runs_dir, csv, zh = STAGES[stage]
        g = pd.read_csv(csv)
        for _, w in g.iterrows():
            L, seed, label = int(w["L"]), int(w["seed"]), str(w["label"])
            stem = os.path.join(runs_dir, f"{label}_L{L}_st128_k4_s{seed}")
            snaps = [(n, st, f) for n, st, f in ss.snapshot_files(stem) if st is not None]
            if not snaps:
                print("no snapshots for", stem, flush=True)
                continue
            for name, step, f in snaps:
                soup = ss.load(f, L)
                uniq, counts = np.unique(soup, axis=0, return_counts=True)
                i = int(np.argmax(counts))
                tape = uniq[i]
                share = counts[i] / len(soup)
                if not tape.any():
                    rows.append({"stage": stage, "L": L, "seed": seed, "step": step, "tape": "all-zero", "share": share})
                    continue
                K = 255 if L <= 20 else 32
                d = scan_tape(tape, K, 32, 128, seed=20261010 + L * 7 + seed, zero_halts=zh)
                row, site = summarise(d, L)
                rng = np.random.default_rng([20261010, L, seed, step])
                Rp = rng.integers(0, 256, size=(64, L), dtype=np.uint8)
                res, masks = X.execute_pairs_traced(np.concatenate([np.repeat(tape[None, :], 64, 0), Rp], 1), L, 128, zero_halts=zh)
                ex = X.exec_positions(masks, 2 * L)
                union = ex[:, :L].any(axis=0)
                tr = site["transmissible"].values.astype(bool)
                rows.append({"stage": stage, "L": L, "seed": seed, "step": step, "tape": " ".join(f"{b:02x}" for b in tape), "share": float(share),
                             "ctrl_herit": d["ctrl"]["herit"], "ctrl_gen2": d["ctrl"]["gen2"], "robustness_h": row["robustness_h"], "n_sites": row["n_sites"],
                             "capacity_bits": row["capacity_bits"], "entered_frac": float(ex[:, L:].any(axis=1).mean()), "exec_union": int(union.sum()),
                             "n_sites_unexecuted": int((tr & ~union).sum())})
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            last = rows[-1]
            print(f"{stage} L{L} s{seed}: {len(snaps)} snapshots; last step {last['step']:,} tape {last['tape'][:11]} share {last['share']:.3f} sites {last.get('n_sites', '-')} unexec {last.get('n_sites_unexecuted', '-')} entered {last.get('entered_frac', float('nan')):.2f} ({time.time() - t0:.0f} s)", flush=True)
    print("done ->", out_csv)


if __name__ == "__main__":
    main()
