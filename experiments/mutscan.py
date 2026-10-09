"""Single-byte mutational scan of the Stage G first replicators and final dominants (pre-registered: REVISION_PREREG.md, M).

    .venv/bin/python mutscan.py --yes [--L 16,20,50,64] [--n 32] [--out results/mutscan]
    .venv/bin/python mutscan.py --report            # predictions M1–M3 from the written tables

For every tape and every position i, each alternative byte v (all 255 at L <= 20, 32 seeded values at L >= 50) gives the
mutant x[i <- v]; it runs as A against n shared random partners (assay.execute_pairs, 128 instructions), its offspring
against n fresh partners (gen2). Per mutant: gen2 (heritable iff >= GEN2_MIN), faithful, and the transmission rate: among
offspring that are >= 75% copies at the best cyclic shift, the fraction that carry v at the aligned position of i.
Outputs: <out>/mutscan_tapes.csv (one row per tape), <out>/mutscan_sites.csv (one row per tape x position),
runs/mutscan/<label>.npz (per-mutant arrays). Local GPU; pass --yes after pausing the browser simulation.
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
from algocell_exp import assay as A  # noqa: E402

R = os.path.join(HERE, "results")


def parse_tape(s: str) -> np.ndarray:
    return np.array([int(b, 16) for b in s.split()], dtype=np.uint8)


def scan_tape(tape: np.ndarray, K: int, n: int, steps: int, seed: int) -> dict:
    L = tape.size
    rng = np.random.default_rng(seed)
    vals = []
    for i in range(L):
        alt = np.array([v for v in range(256) if v != tape[i]], dtype=np.uint8)
        vals.append(alt if K >= 255 else rng.choice(alt, size=K, replace=False))
    pos = np.concatenate([np.full(len(vals[i]), i, dtype=int) for i in range(L)])
    val = np.concatenate(vals).astype(np.uint8)
    M = len(pos)
    mut = np.tile(tape, (M, 1))
    mut[np.arange(M), pos] = val
    T = np.concatenate([tape[None, :], mut])                     # row 0 is the unmutated control
    M1 = M + 1
    Rp = rng.integers(0, 256, size=(n, L), dtype=np.uint8)
    Rp2 = rng.integers(0, 256, size=(n, L), dtype=np.uint8)
    TT = np.repeat(T, n, axis=0)
    RR = np.tile(Rp, (M1, 1))
    res = A.execute_pairs(np.concatenate([TT, RR], axis=1), L, steps)
    before = A._best_shift_match_rows(RR, TT)
    after, shifts = A._best_shift_rows(res[:, L:], TT)
    score = A._norm_gain_rows(before, after, M1, n)
    offspring = res[:, L:]
    copies = after >= 0.75
    rows = np.arange(M1 * n)
    tape_idx = rows // n
    posr = np.concatenate([[0], pos])[tape_idx]
    valr = np.concatenate([[0], val])[tape_idx]
    is_mut = tape_idx > 0
    # target[j] == tape[(j - s) % L]  ->  byte i of the organism lands at offspring position (i + s) % L
    carried = (offspring[rows, (posr + shifts) % L] == valr) & is_mut
    cp = copies.reshape(M1, n)
    ca = (carried & copies).reshape(M1, n)
    ncp = cp.sum(axis=1)
    trans = np.where(ncp > 0, ca.sum(axis=1) / np.maximum(ncp, 1), np.nan)
    RR2 = np.tile(Rp2, (M1, 1))
    res2 = A.execute_pairs(np.concatenate([offspring, RR2], axis=1), L, steps)
    before2 = A._best_shift_match_rows(RR2, TT)
    after2 = A._best_shift_match_rows(res2[:, L:], TT)
    gen2 = A._norm_gain_rows(before2, after2, M1, n)
    q75 = cp.mean(axis=1)
    herit = np.isfinite(gen2) & (gen2 >= A.GEN2_MIN)
    faith = herit & (q75 >= A.FAITHFUL_MIN)
    return {"pos": pos, "val": val, "score": score[1:], "gen2": gen2[1:], "q75": q75[1:], "herit": herit[1:], "faith": faith[1:], "trans": trans[1:],
            "ctrl": {"score": float(score[0]), "gen2": float(gen2[0]), "q75": float(q75[0]), "herit": bool(herit[0]), "faith": bool(faith[0])}}


def summarise(d: dict, L: int) -> tuple[dict, pd.DataFrame]:
    tr_ok = np.nan_to_num(d["trans"], nan=0.0) >= 0.5
    both = d["herit"] & tr_ok
    S = pd.DataFrame({"pos": d["pos"], "herit": d["herit"], "faith": d["faith"], "trans_ok": tr_ok, "both": both})
    site = S.groupby("pos").agg(frac_herit=("herit", "mean"), frac_faith=("faith", "mean"), frac_trans=("trans_ok", "mean"), frac_both=("both", "mean"), n_vals=("herit", "size")).reindex(range(L)).fillna(0.0)
    site["transmissible"] = site["frac_both"] >= 0.5
    cap = float(np.log2(1.0 + 255.0 * site["frac_both"].values).sum())
    tape_row = {"robustness_h": float(d["herit"].mean()), "robustness_f": float(d["faith"].mean()),
                "trans_mean_heritable": float(np.nanmean(d["trans"][d["herit"]])) if d["herit"].any() else float("nan"),
                "frac_both": float(both.mean()), "n_sites": int(site["transmissible"].sum()), "capacity_bits": cap, "n_mutants": int(len(d["pos"]))}
    return tape_row, site


def run(a) -> None:
    g = pd.read_csv(os.path.join(R, "stageG", "stageG", "stage_g_runs.csv"))
    Ls = [int(x) for x in a.L.split(",")]
    g = g[g["L"].isin(Ls)].reset_index(drop=True)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(HERE, "runs", "mutscan"), exist_ok=True)
    tape_rows, site_rows = [], []
    tp, sp = os.path.join(a.out, "mutscan_tapes.csv"), os.path.join(a.out, "mutscan_sites.csv")
    if os.path.exists(tp) and not a.fresh:
        old = pd.read_csv(tp)
        olds = pd.read_csv(sp)
        old = old[~old["L"].isin(Ls)]
        olds = olds[~olds["L"].isin(Ls)]
        tape_rows, site_rows = old.to_dict("records"), [olds]
    t_all = time.time()
    for _, w in g.iterrows():
        L, seed = int(w["L"]), int(w["seed"])
        K = 255 if L <= 20 else a.K
        for which in ("first", "final"):
            tape = parse_tape(w[f"{which}_tape"])
            loop = bool(w[f"{which}_has_cf"]) or bool(w[f"{which}_has_block"])
            label = f"L{L}_s{seed}_{which}"
            t0 = time.time()
            d = scan_tape(tape, K, a.n, 128, seed=20261009 + 1000 * L + seed + (7 if which == "final" else 0))
            row, site = summarise(d, L)
            row.update({"label": label, "L": L, "seed": seed, "which": which, "has_loop": loop, "tape": w[f"{which}_tape"], "K": K, "n_partners": a.n,
                        **{f"ctrl_{k}": v for k, v in d["ctrl"].items()}, "wall_s": round(time.time() - t0, 1)})
            tape_rows.append(row)
            site = site.reset_index().rename(columns={"index": "pos"})
            site.insert(0, "label", label)
            site.insert(1, "L", L)
            site.insert(2, "seed", seed)
            site.insert(3, "which", which)
            site["byte"] = [f"{b:02x}" for b in tape]
            site_rows.append(site)
            np.savez_compressed(os.path.join(HERE, "runs", "mutscan", label + ".npz"), **{k: v for k, v in d.items() if k != "ctrl"})
            print(f"{label:>18} loop={int(loop)} ctrl gen2 {d['ctrl']['gen2']:.2f}  robust {row['robustness_h']:.3f}  sites {row['n_sites']:2d}  cap {row['capacity_bits']:5.1f} bits  trans {row['trans_mean_heritable']:.2f}  ({row['wall_s']} s)", flush=True)
            pd.DataFrame(tape_rows).to_csv(tp, index=False)
            pd.concat(site_rows, ignore_index=True).to_csv(sp, index=False)
    print(f"done in {time.time() - t_all:.0f} s -> {tp}")


def report(a) -> None:
    T = pd.read_csv(os.path.join(a.out, "mutscan_tapes.csv"))
    lines = ["# Mutational scan: predictions M1–M3 against the data", ""]
    for L, d in T.groupby("L"):
        f = d[d.which == "first"].set_index("seed")
        n = d[d.which == "final"].set_index("seed")
        idx = f.index.intersection(n.index)
        more_robust = int((n.loc[idx, "robustness_h"] > f.loc[idx, "robustness_h"]).sum())
        loopn = n.loc[idx][n.loc[idx, "has_loop"]]
        lines.append(f"L = {L}: worlds {len(idx)}; final more mutation-robust than first in {more_robust}/{len(idx)} (M1 needs >= 15); "
                     f"robustness median first {f.loc[idx, 'robustness_h'].median():.3f} vs final {n.loc[idx, 'robustness_h'].median():.3f}; "
                     f"transmissible sites median first {f.loc[idx, 'n_sites'].median():.1f} vs loop-bearing finals {loopn['n_sites'].median() if len(loopn) else float('nan'):.1f} (n = {len(loopn)}); "
                     f"capacity median first {f.loc[idx, 'capacity_bits'].median():.1f} vs loop-bearing finals {loopn['capacity_bits'].median() if len(loopn) else float('nan'):.1f} bits")
    m = T[(T.L == 50) & (T.which == "final") & T.tape.str.startswith("01 c5") & T.tape.str.contains("20 f0")]
    if len(m):
        f50 = T[(T.L == 50) & (T.which == "first")].set_index("seed")
        wins = sum(int(r.n_sites > f50.loc[r.seed, "n_sites"]) for r in m.itertuples())
        capw = sum(int(r.capacity_bits > f50.loc[r.seed, "capacity_bits"]) for r in m.itertuples())
        lines.append(f"M3 matched pairs at L = 50 (pusher + JR vs pusher): {len(m)} worlds; final has more sites in {wins}, higher capacity in {capw}; "
                     f"sites first median {f50.loc[m.seed, 'n_sites'].median():.1f} vs final {m.n_sites.median():.1f}; capacity {f50.loc[m.seed, 'capacity_bits'].median():.1f} vs {m.capacity_bits.median():.1f} bits")
    ctrl_bad = T[~T.ctrl_herit.astype(bool)]
    lines.append(f"controls: {len(T) - len(ctrl_bad)}/{len(T)} unmutated tapes heritable in the scan's own partner draw" + (f"; not: {ctrl_bad.label.tolist()}" if len(ctrl_bad) else ""))
    out = "\n".join(lines)
    print(out)
    open(os.path.join(a.out, "REPORT.md"), "w").write(out + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", default="16,20,50,64")
    ap.add_argument("--K", type=int, default=32, help="alternative values per position at L >= 50 (all 255 at L <= 20)")
    ap.add_argument("--n", type=int, default=32, help="partners per generation")
    ap.add_argument("--out", default=os.path.join(R, "mutscan"))
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--yes", action="store_true", help="confirm that the browser simulation is paused (local GPU)")
    a = ap.parse_args()
    if a.report:
        report(a)
        return
    if not a.yes:
        sys.exit("refusing to run: pass --yes after pausing the browser simulation (local GPU)")
    run(a)


if __name__ == "__main__":
    main()
