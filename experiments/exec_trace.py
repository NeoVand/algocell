"""Pointer confinement and the executed set for every Z80 replicator (NATURE_PLAN Move 2a; REVISION_PREREG P).

    .venv/bin/python exec_trace.py --yes [--out results/exectrace]

For the first replicator and final dominant of every Stage G, K and I world: 256 seeded random partners, traced execution
(algocell_exp.exectrace). Per tape: entered_frac (encounters in which any partner address was fetched as instruction
stream), exec_union (organism positions fetched in any encounter), exec_mean (per encounter), partner_exec_mean (partner
positions fetched per encounter), and joins: inflow H_bits (results/biology/individuality or recomputed here as the plug-in
entropy of the offspring), transmissible sites from the mutational scan and how many of them are never executed (the
genotype segment G = transmissible and unexecuted). Writes per_replicator.csv and AGREEMENT.md.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from algocell_exp import assay as A  # noqa: E402
from algocell_exp import exectrace as X  # noqa: E402

R = os.path.join(HERE, "results")
SOURCES = [("G", os.path.join(R, "stageG", "stageG", "stage_g_runs.csv"), False, os.path.join(R, "mutscan", "mutscan_sites.csv")),
           ("K", os.path.join(R, "stageK", "stageK", "stage_g_runs.csv"), False, os.path.join(R, "mutscan_K", "mutscan_sites.csv")),
           ("I", os.path.join(R, "stageI", "stageI", "stage_g_runs.csv"), True, os.path.join(R, "mutscan_I", "mutscan_sites.csv"))]


def parse(s):
    return np.array([int(b, 16) for b in str(s).split()], dtype=np.uint8)


def entropy_bits(rows: np.ndarray) -> float:
    _, c = np.unique(rows, axis=0, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R, "exectrace"))
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if not a.yes:
        sys.exit("refusing to run: pass --yes after pausing the browser simulation (local GPU)")
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for stage, path, zh, sites_path in SOURCES:
        g = pd.read_csv(path)
        sites = pd.read_csv(sites_path) if os.path.exists(sites_path) else None
        for _, w in g.iterrows():
            L, seed = int(w["L"]), int(w["seed"])
            rng = np.random.default_rng([20261009, L, seed])
            Rp = rng.integers(0, 256, size=(a.n, L), dtype=np.uint8)
            for which in ("first", "final"):
                t = parse(w[f"{which}_tape"])
                pairs = np.concatenate([np.repeat(t[None, :], a.n, 0), Rp], axis=1)
                res, masks = X.execute_pairs_traced(pairs, L, 128, zero_halts=zh)
                ex = X.exec_positions(masks, 2 * L)
                offspring = res[:, L:]
                after, shifts = A._best_shift_rows(offspring, np.repeat(t[None, :], a.n, 0))
                copies = after >= 0.75
                exec_union = ex[:, :L].any(axis=0)
                rec = {"stage": stage, "label": f"L{L}_s{seed}_{which}", "L": L, "seed": seed, "which": which, "zero_halts": zh,
                       "has_loop": bool(w[f"{which}_has_cf"]) or bool(w[f"{which}_has_block"]), "tape": w[f"{which}_tape"],
                       "entered_frac": float(ex[:, L:].any(axis=1).mean()), "exec_mean": float(ex[:, :L].sum(axis=1).mean()),
                       "exec_union": int(exec_union.sum()), "partner_exec_mean": float(ex[:, L:].sum(axis=1).mean()),
                       "copied_frac": float(copies.mean()), "H_bits": entropy_bits(offspring), "n_unique_offspring": int(len(np.unique(offspring, axis=0)))}
                if sites is not None:
                    s = sites[(sites.L == L) & (sites.seed == seed) & (sites.which == which)].sort_values("pos")
                    if len(s) == L:
                        tr = s["transmissible"].astype(bool).values
                        rec["n_sites"] = int(tr.sum())
                        rec["n_sites_unexecuted"] = int((tr & ~exec_union).sum())
                        rec["n_unexecuted"] = int((~exec_union).sum())
                rows.append(rec)
                print(f"{stage} {rec['label']:>18} loop={int(rec['has_loop'])} entered {rec['entered_frac']:.2f} exec {rec['exec_mean']:4.1f}/{L} union {rec['exec_union']:3d} H {rec['H_bits']:.2f} sites {rec.get('n_sites', '-')} unexec-sites {rec.get('n_sites_unexecuted', '-')}", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(a.out, "per_replicator.csv"), index=False)
    lines = ["# Executed-address traces: pointer confinement, the executed set and the genotype segment (generated)", ""]
    for stage, d in T.groupby("stage"):
        for which, e in d.groupby("which"):
            pc = e.entered_frac <= 0.05
            po = e.entered_frac >= 0.5
            hz = e.H_bits == 0
            lines.append(f"- Stage {stage}, {which} (n = {len(e)}): pointer-closed (entered ≤ 0.05) {int(pc.sum())}, pointer-open (≥ 0.5) {int(po.sum())}, in between {int((~pc & ~po).sum())}; "
                         f"inflow zero {int(hz.sum())}; agreement pointer-closed ∧ H = 0: {int((pc & hz).sum())}, pointer-closed ∧ H > 0: {int((pc & ~hz).sum())}, pointer-open ∧ H = 0: {int((po & hz).sum())}; "
                         f"executed positions per encounter median {e.exec_mean.median():.1f} of L, union median {e.exec_union.median():.0f}" +
                         (f"; transmissible sites median {e.n_sites.median():.0f}, of which unexecuted {e.n_sites_unexecuted.median():.0f} (sum {int(e.n_sites_unexecuted.sum())} of {int(e.n_sites.sum())})" if "n_sites" in e and e.n_sites.notna().any() else ""))
    out = "\n".join(lines)
    print(out)
    open(os.path.join(a.out, "AGREEMENT.md"), "w").write(out + "\n")


if __name__ == "__main__":
    main()
