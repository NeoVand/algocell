"""E — the copy-offset switch: regenerator against transmitter with the same 4-byte core (REVISION_PREREG E).

    .venv/bin/modal run modal_offset.py                    # all worlds on Modal (outputs under /runs/offset)
    .venv/bin/python offset_switch.py --analyse            # after: modal volume get algocell-atlas-runs offset runs/

World runner: run_world(cond, outdir). cond = {L, tar ('benign'|'lethal'), mut ('on'|'off'), start ('mix50'|'R_into_T'|
'T_into_R'), seed, steps}. Every 50 steps to 5,000, then every 500: counts of cells by their first four bytes (core
`5e ed b0` at bytes 1-3; offset class of byte 0 from offset_census: R (regenerating), T (transmitting), X (broken)) and,
for transmitters, the number of distinct tails and the mean Hamming distance of 200 sampled tails to the cell-wise
initial tails' population (diversity). The final soup is saved.
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

R_VALS = {16: {0x04, 0x24, 0x44, 0x64, 0x84, 0xa4, 0x08, 0x48, 0x68, 0x88, 0xa8, 0xc8, 0xe8}, 32: {0x04, 0x44, 0x84, 0x08, 0x48, 0x88, 0xc8, 0x50, 0x90}}
T_VALS = {16: {0x50, 0x90, 0xb0}, 32: {0x60, 0xa0}}
R_TAPE = {16: [0x44, 0x5e, 0xed, 0xb0] * 4, 32: [0x04, 0x5e, 0xed, 0xb0] * 8}
T_BYTE = {16: 0xb0, 32: 0xa0}


def conditions():
    out = []
    for L, tar in ((16, "benign"), (16, "lethal"), (32, "benign")):
        for s in range(1, 11):
            out.append({"L": L, "tar": tar, "mut": "off", "start": "mix50", "seed": s, "steps": 100000})
            out.append({"L": L, "tar": tar, "mut": "on", "start": "mix50", "seed": s, "steps": 300000})
        for s in range(1, 6):
            out.append({"L": L, "tar": tar, "mut": "on", "start": "R_into_T", "seed": s, "steps": 300000})
            out.append({"L": L, "tar": tar, "mut": "on", "start": "T_into_R", "seed": s, "steps": 300000})
    return out


def conditions_l32lethal():
    """E-L32L (REVISION_PREREG): the matched pair at L = 32 under lethal tar."""
    out = [{"L": 32, "tar": "lethal", "mut": "on", "start": "mix50", "seed": s, "steps": 300000} for s in range(1, 11)]
    for s in range(1, 6):
        out.append({"L": 32, "tar": "lethal", "mut": "on", "start": "R_into_T", "seed": s, "steps": 300000})
        out.append({"L": 32, "tar": "lethal", "mut": "on", "start": "T_into_R", "seed": s, "steps": 300000})
    return out


def stem(c):
    return f"L{c['L']}_{c['tar']}_mut{c['mut']}_{c['start']}_s{c['seed']}"


def classify(s, L):
    core = (s[:, 1] == 0x5e) & (s[:, 2] == 0xed) & (s[:, 3] == 0xb0)
    b0 = s[:, 0]
    isR = core & np.isin(b0, list(R_VALS[L]))
    isT = core & np.isin(b0, list(T_VALS[L]))
    return core, isR, isT


def run_world(c: dict, outdir: str) -> dict:
    from algocell_exp.soup import Soup
    L = c["L"]
    os.makedirs(outdir, exist_ok=True)
    seed = 7000 + 100 * (L == 32) + 50 * (c["tar"] == "lethal") + c["seed"] + 1000 * ["mix50", "R_into_T", "T_into_R"].index(c["start"]) + 20000 * (c["mut"] == "off")
    soup = Soup(160, 125, "square", L, seed, 8192, 128, 4, [], zero_halts=(c["tar"] == "lethal"), mutations_per_step=(0 if c["mut"] == "off" else None))
    rng = np.random.default_rng([seed, 99])
    N = soup.cell_count
    Rt = np.array(R_TAPE[L], np.uint8)
    T = np.concatenate([np.full((N, 1), T_BYTE[L], np.uint8), np.tile(np.array([0x5e, 0xed, 0xb0], np.uint8), (N, 1)), rng.integers(0, 256, size=(N, L - 4), dtype=np.uint8)], 1)
    if c["start"] == "mix50":
        isT0 = rng.random(N) < 0.5
    elif c["start"] == "R_into_T":
        isT0 = np.ones(N, bool)
        isT0[rng.choice(N, size=N // 100, replace=False)] = False
    else:
        isT0 = np.zeros(N, bool)
        isT0[rng.choice(N, size=N // 100, replace=False)] = True
    cells = np.where(isT0[:, None], T, Rt[None, :])
    soup.write_soup(cells.astype(np.uint8))
    path = os.path.join(outdir, stem(c) + ".jsonl")
    step, t0 = 0, time.time()
    with open(path, "w") as f:
        f.write(json.dumps({"kind": "condition", **c, "soup_seed": seed}) + "\n")
        while step <= c["steps"]:
            s = soup.read_soup()
            core, isR, isT = classify(s, L)
            tails = s[isT][:, 4:]
            nd = int(len(np.unique(tails, axis=0))) if len(tails) else 0
            f.write(json.dumps({"step": step, "core": int(core.sum()), "R": int(isR.sum()), "T": int(isT.sum()), "T_distinct_tails": nd,
                                "zero_frac": float((s == 0).mean())}) + "\n")
            dt = 50 if step < 5000 else 500
            soup.step(dt)
            step += dt
    np.save(os.path.join(outdir, stem(c) + "_final.npy"), soup.read_soup())
    return {**c, "wall_s": round(time.time() - t0, 1)}


def analyse(indir: str, out: str) -> None:
    rows = []
    for f in sorted(os.listdir(indir)):
        if not f.endswith(".jsonl"):
            continue
        recs = [json.loads(l) for l in open(os.path.join(indir, f))]
        c = recs[0]
        S = pd.DataFrame(recs[1:])
        S["Tshare"] = S["T"] / (S["R"] + S["T"]).replace(0, np.nan)
        last = S.iloc[-1]
        # selection coefficient: slope of logit(T share) per 1,000 steps over the run (shares clipped away from 0 and 1)
        p = S.Tshare.clip(1e-3, 1 - 1e-3)
        ok = S.Tshare.notna() & (S.step > 0)
        slope = float(np.polyfit(S.step[ok] / 1000.0, np.log(p[ok] / (1 - p[ok])), 1)[0]) if ok.sum() > 5 else np.nan
        rows.append({**{k: c[k] for k in ("L", "tar", "mut", "start", "seed", "steps")}, "T_share_end": float(last.Tshare), "core_share_end": float(last.core / 20000),
                     "T_share_start": float(S.Tshare.iloc[0]), "logit_slope_per_1000": slope, "T_distinct_tails_end": int(last.T_distinct_tails)})
    D = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    D.to_csv(os.path.join(out, "offset_switch.csv"), index=False)
    lines = ["# E — the copy-offset switch (generated by `offset_switch.py --analyse`)", "",
             "Transmitting share among core-carrying cells (T / (R + T)) at the end; logit slope per 1,000 steps (positive favours transmitters).", "",
             "| L | tar | mutation | start | worlds | T share at start | T share at end (median, range) | logit slope per 1,000 steps (median, range) | core-carrying cells at end (median) | distinct transmitter tails at end (median) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for (L, tar, mut, start), d in D.groupby(["L", "tar", "mut", "start"]):
        lines.append(f"| {L} | {tar} | {mut} | {start} | {len(d)} | {d.T_share_start.median():.2f} | {d.T_share_end.median():.2f} ({d.T_share_end.min():.2f}–{d.T_share_end.max():.2f}) | "
                     f"{d.logit_slope_per_1000.median():+.3f} ({d.logit_slope_per_1000.min():+.3f} to {d.logit_slope_per_1000.max():+.3f}) | {d.core_share_end.median():.2f} | {int(d.T_distinct_tails_end.median())} |")
    lines.append("")
    for (L, tar), d in D[D.start == "mix50"].groupby(["L", "tar"]):
        off = d[d.mut == "off"]
        on = d[d.mut == "on"]
        e1 = int(((off.T_share_end >= 0.3) & (off.T_share_end <= 0.7)).sum())
        e2 = int(((on.T_share_end >= 0.05) & (on.T_share_end <= 0.4)).sum())
        tag = "E3 (lethal, descriptive)" if tar == "lethal" else "E1/E2"
        lines.append(f"- L = {L}, {tar} tar [{tag}]: mutation off, T share in 0.3–0.7 at the end in {e1} of {len(off)} (E1 needs ≥ 7); mutation on, T share in 0.05–0.4 in {e2} of {len(on)} (E2 needs ≥ 7)")
    open(os.path.join(out, "REPORT.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--indir", default=os.path.join(HERE, "runs", "offset"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "offset"))
    ap.add_argument("--local-smoke", action="store_true")
    a = ap.parse_args()
    if a.analyse:
        analyse(a.indir, a.out)
    elif a.local_smoke:
        print(run_world({"L": 16, "tar": "benign", "mut": "off", "start": "mix50", "seed": 99, "steps": 2000}, os.path.join(HERE, "runs", "offset_smoke")))


if __name__ == "__main__":
    main()
