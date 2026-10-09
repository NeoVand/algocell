"""A — accessibility or selection (pre-registered: REVISION_PREREG.md, R2/A, 2026-10-09).

    .venv/bin/python invasion_closer.py --yes --exp A1 [--seeds 5] [--controls 3]     # 8080 subset, 20,000 steps
    .venv/bin/python invasion_closer.py --yes --exp A2                                 # full Z80, 50,000 steps

A1: the constructed 8080 closer (20-byte core + payload 1 of serial_retention.py) against the 8080 pusher `01 c5`x16, under
the i8080 suppression. A2: the same closer against the evolved LDIR closer `04 5e ed b0`x8, full Z80. L = 32, Stage M
dynamics (160 x 125 square lattice, 8,192 pairs, 128 instructions, mutation 1/16 per pair). Resident = every cell one tape;
invader = 1% of cells the other tape at step 0; both directions; unseeded controls. Recorded every 10 steps to 500, then
every 250: share of cells carrying the core at any cyclic shift; share within Hamming 8 (any shift) of the other tape;
among core carriers, the share with the seeded payload and the number of distinct payloads.
Post hoc (added 2026-10-09 after the first A1 run, labelled as such in the report): the share of cells carrying the
closer's loop body `2b 46 2b 4e c5 15 c2` (DEC HL; LD B,(HL); DEC HL; LD C,(HL); PUSH BC; DEC D; JP NZ) at any cyclic shift,
because the seeded core mutates within ~2,000 steps while its descendants keep the loop; the final soup of every run is
saved to runs/invasion_closer/ for analysis.
Output: results/invasion_closer/<exp>.csv and <exp>_REPORT.md.
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
from algocell_exp.soup import Soup  # noqa: E402
from serial_retention import CORE_8080, parse  # noqa: E402

L = 32
CORE = parse(CORE_8080)
NC = CORE.size
LOOP = parse("2b 46 2b 4e c5 15 c2")


def loop_share(s: np.ndarray) -> np.ndarray:
    d = np.concatenate([s, s], axis=1)
    has = np.zeros(len(s), bool)
    for sh in range(L):
        has |= (d[:, sh:sh + LOOP.size] == LOOP).all(axis=1)
    return has


def payload1() -> np.ndarray:
    rng = np.random.default_rng(20261010)
    return rng.integers(0, 256, size=12, dtype=np.uint8)


def core_hits(s: np.ndarray):
    """Per cell: carries the core at some cyclic shift; payload at the first such shift (or None)."""
    d = np.concatenate([s, s], axis=1)
    has = np.zeros(len(s), bool)
    pay = np.zeros((len(s), L - NC), np.uint8)
    for sh in range(L):
        m = (d[:, sh:sh + NC] == CORE).all(axis=1) & ~has
        if m.any():
            pay[m] = d[m, sh + NC:sh + L]
            has |= m
    return has, pay


def min_hamming(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    best = np.full(len(s), L, dtype=int)
    for sh in range(L):
        best = np.minimum(best, (s != np.roll(t, sh)).sum(axis=1))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=("A1", "A2"), required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--controls", type=int, default=3)
    ap.add_argument("--steps", type=int, default=0, help="default 20,000 (A1) or 50,000 (A2)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "invasion_closer"))
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if not a.yes:
        sys.exit("refusing to run: pass --yes after pausing the browser simulation (local GPU)")
    from make_conds import ABLATIONS
    steps = a.steps or (20000 if a.exp == "A1" else 50000)
    closer = np.concatenate([CORE, payload1()])
    if a.exp == "A1":
        other, other_name, sup = parse(" ".join(["01 c5"] * 16)), "pusher", list(ABLATIONS["i8080"])
    else:
        other, other_name, sup = parse(" ".join(["04 5e ed b0"] * 8)), "ldir", []
    tapes = {"closer": closer, other_name: other}
    os.makedirs(a.out, exist_ok=True)
    plan = [(other_name, "closer", sd) for sd in range(1, a.seeds + 1)] + [("closer", other_name, sd) for sd in range(1, a.seeds + 1)]
    plan += [(other_name, "none", sd) for sd in range(1, a.controls + 1)] + [("closer", "none", sd) for sd in range(1, a.controls + 1)]
    rows = []
    t_all = time.time()
    p1 = payload1()
    for resident, invader, seed in plan:
        soup = Soup(160, 125, "square", L, 300 + seed, 8192, 128, 4, sup)
        rng = np.random.default_rng([seed, L, 1 if resident == "closer" else 2, 7 if a.exp == "A2" else 8])
        cells = soup.read_soup()
        cells[:] = tapes[resident][None, :]
        if invader != "none":
            idx = rng.choice(len(cells), size=int(0.01 * len(cells)), replace=False)
            cells[idx] = tapes[invader][None, :]
        soup.write_soup(cells)
        step, t0 = 0, time.time()
        while step <= steps:
            s = soup.read_soup()
            has, pay = core_hits(s)
            hd = min_hamming(s, other)
            rec = {"exp": a.exp, "resident": resident, "invader": invader, "seed": seed, "step": step, "core_share": float(has.mean()),
                   f"{other_name}_class_share": float((hd <= 8).mean()), "zero_frac": float((s == 0).mean()),
                   "loop_share_posthoc": float(loop_share(s).mean())}
            if has.any():
                P = pay[has]
                rec["payload_seeded_share"] = float((P == p1).all(axis=1).mean())
                rec["payload_distinct"] = int(len(np.unique(P, axis=0)))
                rec["payload_mean_hamming"] = float((P != p1).sum(axis=1).mean())
            rows.append(rec)
            dt = 10 if step < 500 else 250
            soup.step(dt)
            step += dt
        os.makedirs(os.path.join(HERE, "runs", "invasion_closer"), exist_ok=True)
        np.save(os.path.join(HERE, "runs", "invasion_closer", f"{a.exp}_{resident}_{invader}_s{seed}_final.npy"), soup.read_soup())
        last = rows[-1]
        print(f"loop {last['loop_share_posthoc']:.3f} | {a.exp} {invader:>7} into {resident:<7} seed {seed}: core {last['core_share']:.3f} {other_name}-class {last[f'{other_name}_class_share']:.3f} "
              f"distinct payloads {last.get('payload_distinct', 0)} ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{a.exp}.csv"), index=False)
    report(a.exp, a.out, other_name)
    print(f"done in {time.time() - t_all:.0f} s")


def report(exp: str, out: str, other_name: str) -> None:
    df = pd.read_csv(os.path.join(out, f"{exp}.csv"))
    fin = df[df.step == df.step.max()]
    lines = [f"# {exp} — constructed closer against the {other_name} (generated by `invasion_closer.py`)", ""]
    for (res, inv), d in fin.groupby(["resident", "invader"]):
        share = d["core_share"] if inv == "closer" else d[f"{other_name}_class_share"]
        what = "core share of the closer" if inv == "closer" else f"{other_name}-class share (Hamming <= 8)"
        if inv == "none":
            share = d["core_share"] if res == "closer" else d[f"{other_name}_class_share"]
            what = f"resident's own share ({'core' if res == 'closer' else other_name + ' class'})"
        lines.append(f"- {inv} into {res}-filled world, step {int(d.step.iloc[0]):,}: {what} " + ", ".join(f"{v:.3f}" for v in share)
                     + f" (> 50% in {int((share > 0.5).sum())}/{len(d)}; < 5% in {int((share < 0.05).sum())}/{len(d)})"
                     + (f"; distinct payloads among core carriers " + ", ".join(str(int(v)) for v in d.payload_distinct.fillna(0)) if "payload_distinct" in d else ""))
    lines.append("")
    lines.append("Post hoc (not registered): share of cells carrying the closer's loop body `2b 46 2b 4e c5 15 c2` at the last step")
    for (res, inv), d in fin.groupby(["resident", "invader"]):
        lines.append(f"- {inv} into {res}: " + ", ".join(f"{v:.3f}" for v in d.loop_share_posthoc))
    for seed, d in df[(df.invader == "closer")].groupby("seed"):
        t50l = d[d.loop_share_posthoc > 0.5].step.min()
        lines.append(f"- closer into {other_name}, seed {seed}: loop-body share first above 50% at step {t50l if t50l == t50l else 'never'} (post hoc)")
    for seed, d in df[(df.invader == "closer")].groupby("seed"):
        t50 = d[d.core_share > 0.5].step.min()
        lines.append(f"- closer into {other_name}, seed {seed}: core share first above 50% at step {t50 if t50 == t50 else 'never'}")
    txt = "\n".join(lines)
    open(os.path.join(out, f"{exp}_REPORT.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
