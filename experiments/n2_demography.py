"""N2 (REVISION_PREREG): in-situ demography of regenerators (R) and transmitters (T) at L = 32, composition x rule.

    .venv/bin/python n2_demography.py --run        # runs/n2/*.npz (tallies)
    .venv/bin/python n2_demography.py --analyse    # results/n2/DEMOGRAPHY.md, demography.csv
    .venv/bin/python n2_demography.py --backup     # N2-2: results/n2/BACKUP.md
Classes by the first four bytes: O (no core) 0, M (core, other first byte) 1, R 2, T 3.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from algocell_exp import exectrace as X  # noqa: E402
from algocell_exp.soup import Soup  # noqa: E402
from offset_switch import classify  # noqa: E402

OFF = os.path.join(HERE, "runs", "offset", "offset")
RUNS = os.path.join(HERE, "runs", "n2")
OUT = os.path.join(HERE, "results", "n2")
NAMES = ["O", "M", "R", "T"]
L = 32


def cls(s: np.ndarray) -> np.ndarray:
    core, isR, isT = classify(s, L)
    c = np.zeros(len(s), np.int8)
    c[core] = 1
    c[isR] = 2
    c[isT] = 3
    return c


def run_one(comp: str, rule: str, seed: int, steps: int = 2000):
    f = os.path.join(OFF, f"L32_{comp}_muton_mix50_s{seed}_final.npy")
    s0 = np.load(f)
    soup = Soup(160, 125, "square", L, 41000 + seed + 100 * (comp == "lethal") + 1000 * (rule == "lethal"), 8192, 128, 4, [], zero_halts=(rule == "lethal"))
    soup.write_soup(s0)
    pre = soup.read_soup()
    cp = cls(pre)
    enc = np.zeros((4, 4, 4), np.int64)     # [executor class, partner class before, partner class after encounter]
    exe = np.zeros((4, 4), np.int64)        # [executor class before, executor class after encounter]
    mut = np.zeros((4, 4), np.int64)        # [class after encounter, class after mutation] for mutated cells
    occ = np.zeros(4, np.int64)             # cell-steps per class (before each step)
    counts = []
    for t in range(steps):
        occ += np.bincount(cp, minlength=4)
        soup.step(1)
        inter = soup.read_interactions()
        pm = soup.read_pair_memory()
        post = soup.read_soup()
        k = np.where(inter["active"])[0]
        I = inter["pairs"][k, 0].astype(np.int64)
        J = inter["pairs"][k, 1].astype(np.int64)
        mid = pre.copy()
        mid[I] = pm[k, 0]
        mid[J] = pm[k, 1]
        cm = cp.copy()
        cm[I] = cls(pm[k, 0])
        cm[J] = cls(pm[k, 1])
        np.add.at(enc, (cp[I], cp[J], cm[J]), 1)
        np.add.at(exe, (cp[I], cm[I]), 1)
        mrows = np.where((post != mid).any(1))[0]
        cpost = cls(post)
        np.add.at(mut, (cm[mrows], cpost[mrows]), 1)
        pre, cp = post, cpost
        if t % 250 == 0 or t == steps - 1:
            counts.append(np.bincount(cp, minlength=4))
    os.makedirs(RUNS, exist_ok=True)
    np.savez(os.path.join(RUNS, f"{comp}_soup_{rule}_rule_s{seed}.npz"), enc=enc, exe=exe, mut=mut, occ=occ, counts=np.array(counts), steps=steps)


def run_all():
    t0 = time.time()
    for comp in ("benign", "lethal"):
        for rule in ("benign", "lethal"):
            for seed in range(1, 11):
                p = os.path.join(RUNS, f"{comp}_soup_{rule}_rule_s{seed}.npz")
                if os.path.exists(p):
                    continue
                run_one(comp, rule, seed)
                print(f"{comp} soup, {rule} rule, seed {seed}: {time.time() - t0:.0f} s", flush=True)


def rates(z) -> dict:
    enc, exe, mut, occ = z["enc"], z["exe"], z["mut"], z["occ"].astype(float)
    out = {}
    for c in (2, 3):
        n = occ[c]
        births = enc[c, :, c].sum() - enc[c, c, c]                               # executor c turns a non-c partner into c
        conv_lost = sum(enc[d, c, d] for d in range(4) if d != c)                # partner c turned into the executor's class d
        dmg_partner = enc[:, c, :].sum() - enc[:, c, c].sum() - conv_lost         # partner c changed class, not into its executor's class
        dmg_self = exe[c, :].sum() - exe[c, c]                                   # executor c changed its own class
        mut_lost = mut[c, :].sum() - mut[c, c]
        mut_gain = mut[:, c].sum() - mut[c, c]
        other_gain = sum(enc[d, b, c] for d in range(4) for b in range(4) if d != c and b != c) + sum(exe[b, c] for b in range(4) if b != c)
        out[NAMES[c]] = {"births": births / n, "conv_lost": conv_lost / n, "dmg_partner": dmg_partner / n, "dmg_self": dmg_self / n,
                         "mut_lost": mut_lost / n, "mut_gain": mut_gain / n, "other_gain": other_gain / n,
                         "net": (births + mut_gain + other_gain - conv_lost - dmg_partner - dmg_self - mut_lost) / n, "cell_steps": n}
    return out


def analyse():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(RUNS, "*.npz"))):
        b = os.path.basename(f)[:-4]
        comp, _, rule, _, sd = b.split("_")
        z = np.load(f)
        r = rates(z)
        cnt = z["counts"]
        for c in ("R", "T"):
            rows.append({"comp": comp, "rule": rule, "seed": int(sd[1:]), "class": c, **r[c],
                         "T_share_start": cnt[0][3] / max(cnt[0][2] + cnt[0][3], 1), "T_share_end": cnt[-1][3] / max(cnt[-1][2] + cnt[-1][3], 1)})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "demography.csv"), index=False)
    comps = ["births", "conv_lost", "dmg_partner", "dmg_self", "mut_lost", "mut_gain", "other_gain", "net"]
    lines = ["# N2: in-situ demography at L = 32 (generated by `n2_demography.py --analyse`)", "",
             "Per-capita rates per cell-step (mean over the ten soups of each cell; 2,000 steps each). Loss terms are subtracted in `net`.", "",
             "| soup | rule | class | " + " | ".join(comps) + " | T share start → end |", "|---|---|---|" + "---|" * len(comps) + "---|"]
    for (comp, rule, c), e in D.groupby(["comp", "rule", "class"]):
        lines.append(f"| {comp} | {rule} | {c} | " + " | ".join(f"{e[k].mean():.5f}" for k in comps) + f" | {e.T_share_start.mean():.3f} → {e.T_share_end.mean():.3f} |")
    lines += ["", "## g_T − g_R by component (mean over soups; positive favours transmitters)", "", "| soup | rule | " + " | ".join(comps) + " |", "|---|---|" + "---|" * len(comps)]
    sign = {"births": 1, "conv_lost": -1, "dmg_partner": -1, "dmg_self": -1, "mut_lost": -1, "mut_gain": 1, "other_gain": 1, "net": 1}
    for (comp, rule), e in D.groupby(["comp", "rule"]):
        R, T = e[e["class"] == "R"].set_index("seed"), e[e["class"] == "T"].set_index("seed")
        lines.append(f"| {comp} | {rule} | " + " | ".join(f"{sign[k] * (T[k] - R[k]).mean():+.5f}" for k in comps) + " |")
    lines += ["", "## N2-1: R's net loss to O (damage + mutation − mutational gain from O/M) against T's, by rule", ""]
    for (comp, rule), e in D.groupby(["comp", "rule"]):
        R, T = e[e["class"] == "R"].set_index("seed"), e[e["class"] == "T"].set_index("seed")
        lossR = R.dmg_partner + R.dmg_self + R.mut_lost - R.mut_gain
        lossT = T.dmg_partner + T.dmg_self + T.mut_lost - T.mut_gain
        lines.append(f"- {comp} soup, {rule} rule: R {lossR.mean():.5f}, T {lossT.mean():.5f}; R lower in {(lossR < lossT).sum()} of {len(lossR)} soups")
    open(os.path.join(OUT, "DEMOGRAPHY.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def backup(n=4096):
    """N2-2: zero at one of positions 0-3 of R (`04 5e ed b0` x 8) and T (`a0 5e ed b0` + 28 random bytes)."""
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(11)
    soup = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(OFF, "L32_benign_muton_mix50_s*_final.npy")))])
    P = soup[rng.integers(0, len(soup), n)]
    R = np.array([0x04, 0x5e, 0xed, 0xb0] * 8, np.uint8)
    lines = ["# N2-2: a zero in the first core (generated by `n2_demography.py --backup`)", "",
             f"{n} partners drawn from the benign L = 32 final soups; T tails random per encounter. 'copies' = the partner becomes an exact copy of the mutant; 'functional' = the partner's first four bytes are a core of the same class (R or T).", "",
             "| tape | zero at | rule | copies itself | functional offspring | executor entered partner |", "|---|---|---|---|---|---|"]
    rows = []
    for name in ("R", "T"):
        for pos in (None, 0, 1, 2, 3):
            if name == "R":
                A = np.tile(R, (n, 1))
            else:
                A = np.concatenate([np.tile(np.array([0xa0, 0x5e, 0xed, 0xb0], np.uint8), (n, 1)), rng.integers(1, 256, (n, 28), dtype=np.uint8)], 1)
            if pos is not None:
                A[:, pos] = 0
            for rule in ("benign", "lethal"):
                res, m = X.execute_pairs_traced(np.concatenate([A, P], 1), L, 128, zero_halts=(rule == "lethal"))
                B = res[:, L:]
                copies = (B == A).all(1).mean()
                c = cls(B)
                func = (c == (2 if name == "R" else 3)).mean()
                ent = X.exec_positions(m, 2 * L)[:, L:].any(1).mean()
                rows.append({"tape": name, "zero_at": pos, "rule": rule, "copies": copies, "functional": func, "entered": ent})
                lines.append(f"| {name} | {'none' if pos is None else pos} | {rule} | {copies:.3f} | {func:.3f} | {ent:.3f} |")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "backup.csv"), index=False)
    open(os.path.join(OUT, "BACKUP.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--backup", action="store_true")
    a = ap.parse_args()
    if a.backup:
        backup()
    if a.run:
        run_all()
    if a.analyse:
        analyse()
