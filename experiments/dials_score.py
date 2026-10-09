"""Score the BFF dial batches (REVISION_PREREG D1, D2) from the fetched runs.

    .venv/bin/python dials_score.py [--root runs/bff_modal] [--out results/bff_dials]

D1 (batches bff_dial_hp*, plus the existing wraplit p = 1 and wraplitnh p = 0): per soup, the all-`P` share over epochs
(from samples.jsonl), persistence at the last epoch (share >= 0.10), and the first epoch at which the share falls below
0.01 after having reached 0.10. D2 (bff_dial_r*, plus lit r = 1): the first replicator (bff_analysis runs.csv, first_tape)
executed against 256 random partners under the batch's flags: plug-in entropy of the offspring (bits), pointer entry.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from micro.bff import BFF, PAIR, TAPE  # noqa: E402

R = os.path.join(HERE, "results")


def allp_share(sample: dict) -> float:
    for t in sample.get("top", []):
        tp = t.get("tape", "")
        if tp and set(tp[i:i + 2] for i in range(0, len(tp), 2)) == {"50"}:
            return float(t["share"])
    return 0.0


def soup_dirs(root: str, batch: str) -> list[str]:
    cands = glob.glob(os.path.join(root, batch, batch, "*_s*")) + glob.glob(os.path.join(root, batch, "*_s*"))
    return sorted(d for d in cands if os.path.isdir(d) and os.path.exists(os.path.join(d, "samples.jsonl")))


def d1_rows(root: str) -> list[dict]:
    rows = []
    batches = {"bff": ("wraplit", 1.0), "bff": ("wraplitnh", 0.0)}
    specs = [("bff", "wraplitnh_s", 0.0), ("bff", "wraplit_s", 1.0)] + [(f"bff_dial_hp{p}", f"wraplit_hp{p}_s", p) for p in ("0.01", "0.03", "0.1", "0.3")]
    for batch, prefix, p in specs:
        for d in soup_dirs(root, batch):
            if not os.path.basename(d).startswith(prefix):
                continue
            S = [json.loads(l) for l in open(os.path.join(d, "samples.jsonl"))]
            ep = np.array([s["epoch"] for s in S])
            sh = np.array([allp_share(s) for s in S])
            reached = np.where(sh >= 0.10)[0]
            t_fall = None
            if len(reached):
                after = np.where((ep > ep[reached[0]]) & (sh < 0.01))[0]
                t_fall = int(ep[after[0]]) if len(after) else None
            rows.append({"batch": batch, "soup": os.path.basename(d), "halt_p": float(p), "last_epoch": int(ep[-1]), "share_last": float(sh[-1]), "share_max": float(sh.max()),
                         "persists": bool(sh[-1] >= 0.10), "epoch_fall_below_1pct": t_fall})
    return rows


def d2_rows(root: str, n_partners: int = 256) -> list[dict]:
    rows = []
    specs = [("bff", "stdlit_s", 1), ("bff_dial_r2", "stdlit_r2_s", 2), ("bff_dial_r3", "stdlit_r3_s", 3)]
    rng = np.random.default_rng(20261010)
    partners = rng.integers(0, 256, size=(n_partners, TAPE), dtype=np.uint8)
    for batch, prefix, r in specs:
        runs_csv = os.path.join(R, "bff" if batch == "bff" else batch, "runs.csv")
        if not os.path.exists(runs_csv):
            print("no runs.csv for", batch, "(score with micro/bff_analysis.py first)")
            continue
        T = pd.read_csv(runs_csv, dtype={"first_tape": str, "final_tape": str})  # all-digit hex tapes (50 50 ...) must stay strings
        T = T[T["run"].astype(str).str.startswith(prefix)] if "run" in T else T
        bff = BFF(max_pairs=n_partners, steps=1 << 13, ip_wrap=False, literal=True, nohalt=False, halt_p=1.0, lit_rep=r)
        for _, w in T.iterrows():
            if not isinstance(w.get("first_tape"), str) or not w["first_tape"]:
                rows.append({"batch": batch, "run": w["run"], "lit_rep": r, "transition": False})
                continue
            x = np.frombuffer(bytes.fromhex(w["first_tape"].replace(" ", "")), dtype=np.uint8)
            if x.size != TAPE:
                continue
            pairs = np.concatenate([np.repeat(x[None, :], n_partners, 0), partners], axis=1)
            mem, out = bff.execute(pairs, roll_seed=7)
            off = mem[:, TAPE:]
            _, c = np.unique(off, axis=0, return_counts=True)
            pr = c / c.sum()
            H = float(-(pr * np.log2(pr)).sum())
            rows.append({"batch": batch, "run": w["run"], "lit_rep": r, "transition": True, "first_tape": w["first_tape"][:24], "first_loop": bool(w.get("first_loop", False)),
                         "H_bits": H, "entered_frac": float(out[:, 1].mean()), "n_unique_offspring": int(len(c)), "t_top": w.get("t_top"), "share_e1024": w.get("allp_e1024", np.nan)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(HERE, "runs", "bff_modal"))
    ap.add_argument("--out", default=os.path.join(R, "bff_dials"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D1 = pd.DataFrame(d1_rows(a.root))
    D2 = pd.DataFrame(d2_rows(a.root))
    D1.to_csv(os.path.join(a.out, "d1_lethality.csv"), index=False)
    D2.to_csv(os.path.join(a.out, "d2_bandwidth.csv"), index=False)
    lines = ["# BFF dials: pre-registered scoring (generated)", "", "## D1 tar lethality (wraplit, all-P share)"]
    if len(D1):
        for p, d in D1.groupby("halt_p"):
            falls = d.epoch_fall_below_1pct.dropna()
            lines.append(f"- p = {p:g}: n = {len(d)}; persists to the last epoch (share ≥ 0.10) in {int(d.persists.sum())}/{len(d)}; share at last epoch median {d.share_last.median():.3f}; "
                         f"fell below 1% in {len(falls)}/{len(d)} (median epoch {falls.median():.0f})" if len(falls) else
                         f"- p = {p:g}: n = {len(d)}; persists in {int(d.persists.sum())}/{len(d)}; share at last epoch median {d.share_last.median():.3f}; never fell below 1%")
    lines += ["", "## D2 literal bandwidth (lit, no wrap; first replicator against 256 random partners)"]
    if len(D2):
        for r, d in D2.groupby("lit_rep"):
            t = d[d.transition == True]  # noqa: E712
            lines.append(f"- r = {r} (write ratio {2 * r / 3:.2f}): soups {len(d)}, transitions {len(t)}; first replicator inflow median {t.H_bits.median() if len(t) else float('nan'):.2f} bits (max {t.H_bits.max() if len(t) else float('nan'):.2f}); "
                         f"loop-free first replicators {int((~t.first_loop).sum()) if len(t) else 0}/{len(t)}; pointer entry median {t.entered_frac.median() if len(t) else float('nan'):.2f}")
    out = "\n".join(lines)
    print(out)
    open(os.path.join(a.out, "SCORING.md"), "w").write(out + "\n")


if __name__ == "__main__":
    main()
