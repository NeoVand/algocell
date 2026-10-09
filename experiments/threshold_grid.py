"""Threshold sensitivity of the first replicator (round-2 review; analysis of recorded runs, no new soups).

    .venv/bin/python threshold_grid.py --yes        # -> results/review_r2/threshold_grid.csv, THRESHOLD_GRID.md

For every world of Stages G, H, I, K, L and M, the first replicator is re-dated under a grid of heredity thresholds: the
first sample (time order) at which one of the three most common tapes holding >= 0.5% of cells has gen2 >= theta_g, with
gen2 computed over partners whose best-shift match to the tape is below theta_b before the encounter (the paper's choice:
theta_g = 0.3, theta_b = 0.75). Each candidate tape is assayed once (64 partners, the run's own rule) and its gen2 kept
for every theta_b. The first replicator is then classified: load-push word (best-shift match >= 0.875 to a tiling of
01 c5, 11 d5, 21 e5, 2a e5 or e5 2a) and open (the pointer fetches a partner byte in at least half of 16 encounters).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from algocell_exp import assay as A  # noqa: E402
from algocell_exp import exectrace as X  # noqa: E402

R = os.path.join(HERE, "results")
STAGES = {"G": ("runs/stageG", "results/stageG/stageG/stage_g_runs.csv", False, ""),
          "H": ("runs/stageH", "results/stageH/stageH/stage_g_runs.csv", False, ""),
          "I": ("runs/stageI", "results/stageI/stageI/stage_g_runs.csv", True, ""),
          "K": ("runs/stageK/stageK", "results/stageK/stageK/stage_g_runs.csv", False, ""),
          "L": ("runs/stageL/stageL", "results/stageL/stageL/stage_g_runs.csv", False, ""),
          "M": ("runs/stageM/stageM", "results/stageM/stageM/stage_g_runs.csv", False, "i8080")}
THETA_G = (0.2, 0.3, 0.4, 0.5, 0.6)
THETA_B = (0.65, 0.75, 0.85)
WORDS = ("01 c5", "11 d5", "21 e5", "2a e5", "e5 2a")


def gen2_grid(T: np.ndarray, sup, zh, seed: int, n: int = 64) -> np.ndarray:
    """(M, len(THETA_B)) gen2 of each tape for each informative-partner threshold."""
    M, L = T.shape
    rng = np.random.default_rng(seed)
    R1 = rng.integers(0, 256, size=(n, L), dtype=np.uint8)
    R2 = rng.integers(0, 256, size=(n, L), dtype=np.uint8)
    TT = np.repeat(T, n, axis=0)
    res = A.execute_pairs(np.concatenate([TT, np.tile(R1, (M, 1))], 1), L, 128, sup, None, zh)
    off = res[:, L:]
    RR2 = np.tile(R2, (M, 1))
    res2 = A.execute_pairs(np.concatenate([off, RR2], 1), L, 128, sup, None, zh)
    before2 = A._best_shift_match_rows(RR2, TT)
    after2 = A._best_shift_match_rows(res2[:, L:], TT)
    return np.stack([A._norm_gain_rows(before2, after2, M, n, max_before=b) for b in THETA_B], axis=1)


def is_loadpush(t: np.ndarray) -> bool:
    L = t.size
    for w in WORDS:
        u = np.array([int(b, 16) for b in w.split()], np.uint8)
        tile = np.resize(u, L)
        if max((t == np.roll(tile, s)).mean() for s in range(2)) >= 0.875:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="G,H,I,K,L,M")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if not a.yes:
        sys.exit("refusing to run: pass --yes after pausing the browser simulation (local GPU)")
    from make_conds import ABLATIONS
    rows = []
    for st in a.stages.split(","):
        rdir, csv, zh, abl = STAGES[st]
        sup = list(ABLATIONS[abl]) if abl else []
        g = pd.read_csv(os.path.join(HERE, csv))
        for _, w in g.iterrows():
            L, seed, label = int(w["L"]), int(w["seed"]), str(w["label"])
            path = os.path.join(HERE, rdir, f"{label}_L{L}_st128_k4_s{seed}.jsonl")
            cands = []      # (step, tape) in time order, first occurrence of each tape
            seen = set()
            for line in open(path):
                d = json.loads(line)
                if d.get("kind") != "sample":
                    continue
                shares = d.get("top3_shares") or []
                for rank, ex in enumerate(d.get("exemplars") or []):
                    if rank < len(shares) and shares[rank] >= 0.005 and ex["tape"] not in seen:
                        seen.add(ex["tape"])
                        cands.append((d["step"], ex["tape"]))
            first = {}
            todo = {(tg, tb) for tg in THETA_G for tb in THETA_B}
            for c0 in range(0, len(cands), 128):
                chunk = cands[c0:c0 + 128]
                T = np.array([[int(b, 16) for b in t.split()] for _, t in chunk], np.uint8)
                G2 = gen2_grid(T, sup, zh, seed=20261014 + seed)
                for (step, tape), gg in zip(chunk, G2):
                    for j, tb in enumerate(THETA_B):
                        for tg in THETA_G:
                            if (tg, tb) in todo and np.isfinite(gg[j]) and gg[j] >= tg:
                                first[(tg, tb)] = (step, tape)
                                todo.discard((tg, tb))
                if not todo:
                    break
            cls = {}
            for key, (step, tape) in first.items():
                if tape not in cls:
                    t = np.array([int(b, 16) for b in tape.split()], np.uint8)
                    P = np.random.default_rng([seed, L]).integers(0, 256, size=(16, L), dtype=np.uint8)
                    _, masks = X.execute_pairs_traced(np.concatenate([np.repeat(t[None], 16, 0), P], 1), L, 128, suppress=sup, zero_halts=zh)
                    cls[tape] = (is_loadpush(t), float(X.exec_positions(masks, 2 * L)[:, L:].any(axis=1).mean()) >= 0.5)
            for tg in THETA_G:
                for tb in THETA_B:
                    if (tg, tb) in first:
                        step, tape = first[(tg, tb)]
                        lp, op = cls[tape]
                        rows.append({"stage": st, "L": L, "seed": seed, "theta_g": tg, "theta_b": tb, "t_first": step, "tape": tape, "loadpush": lp, "open": op})
                    else:
                        rows.append({"stage": st, "L": L, "seed": seed, "theta_g": tg, "theta_b": tb, "t_first": np.nan, "tape": "", "loadpush": False, "open": False})
            r = [x for x in rows if x["seed"] == seed and x["stage"] == st and x["theta_g"] == 0.3 and x["theta_b"] == 0.75][0]
            print(f"{st} L{L} s{seed}: {len(cands)} candidates; default first {r['t_first']} {r['tape'][:11]} loadpush {r['loadpush']} open {r['open']}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(R, "review_r2", "threshold_grid.csv"), index=False)
    report()


def report():
    D = pd.read_csv(os.path.join(R, "review_r2", "threshold_grid.csv"))
    lines = ["# Threshold sensitivity of the first replicator (generated by `threshold_grid.py`)", "",
             "Per stage and length: worlds whose first replicator (under the thresholds) is a load–push word / is open / exists, for heredity threshold θ_g (rows) and informative-partner threshold θ_b (columns). The paper uses θ_g = 0.3, θ_b = 0.75.", ""]
    for (st, L), d in D.groupby(["stage", "L"]):
        n = d.seed.nunique()
        lines.append(f"**Stage {st}, L = {L}** ({n} worlds)")
        lines.append("")
        lines.append("| θ_g \\ θ_b | " + " | ".join(f"{b}" for b in THETA_B) + " |")
        lines.append("|---|" + "---|" * len(THETA_B))
        for tg in THETA_G:
            cells = []
            for tb in THETA_B:
                e = d[(d.theta_g == tg) & (d.theta_b == tb)]
                cells.append(f"{int(e.loadpush.sum())} / {int(e.open.sum())} / {int(e.t_first.notna().sum())}")
            lines.append(f"| {tg} | " + " | ".join(cells) + " |")
        lines.append("")
    z = D[(D.L >= 16) & (D.L <= 64) & (D.stage != "I")]
    for tg in THETA_G:
        for tb in THETA_B:
            e = z[(z.theta_g == tg) & (z.theta_b == tb) & z.t_first.notna()]
            lines.append(f"- benign tar, L 16–64, θ_g = {tg}, θ_b = {tb}: load–push first {int(e.loadpush.sum())} of {len(e)} worlds with a replicator; open first {int(e.open.sum())} of {len(e)}")
    txt = "\n".join(lines)
    open(os.path.join(R, "review_r2", "THRESHOLD_GRID.md"), "w").write(txt + "\n")
    print(txt[-3000:])


if __name__ == "__main__":
    main()
