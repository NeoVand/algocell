"""Nascent copiers — do LDIR-bearing tapes appear in the top-10 before emergence and then vanish? (CPU only)

    python nascent.py runs/stageD --labels none,stack-write-only,stack-read-only,push,call-rst-write [--steps 128] [--k 4]

For every run, every sample's top-10 exemplar tapes are scanned for a block-copy opcode (ED B0/B8/A0/A8). Per run:
first step an LDIR tape enters the top-10; number of episodes (an LDIR tape present in the top-10, then absent at the
next sample) before the run's heritable event (or over the whole run if none); maximum top-10 share held by an LDIR
tape before the event; whether the first replicator (t_rep) is LDIR-based. Per arm: medians and counts.
Distinguishes "nascent copiers assemble and are destroyed" (many short episodes, no take-over) from "copiers never
assemble" (no episodes). Pre-registered question from results/stageD/FINDINGS.md.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from algocell_exp.batch import select_summaries

LDIR_WORDS = ("ed b0", "ed b8", "ed a0", "ed a8")


def has_ldir(tape_hex: str) -> bool:
    t = tape_hex.lower()
    return any(w in t for w in LDIR_WORDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    labels = a.labels.split(",")
    out = a.out or os.path.join(a.dir, "analysis", "census")
    os.makedirs(out, exist_ok=True)
    asy = pd.read_csv(os.path.join(a.dir, "analysis", "assays.csv")).set_index("file")
    rows = []
    for f in select_summaries(a.dir):
        s = json.load(open(f))
        if s["label"] not in labels or s["z80_steps"] != a.steps or s["noise_exp"] != a.k:
            continue
        stem = f[: -len(".summary.json")]
        base = os.path.basename(f)
        t_rep = float(asy.loc[base, "t_rep"]) if base in asy.index else -1
        mechs = str(asy.loc[base, "trep_mechs"]) if base in asy.index else ""
        present_prev = False
        episodes = 0
        first = -1
        max_share = 0.0
        n_samples_pre = 0
        n_present_pre = 0
        for line in open(stem + ".jsonl"):
            if '"kind": "sample"' not in line:
                continue
            r = json.loads(line)
            pre = not (t_rep > 0 and r["step"] >= t_rep)
            ex = r.get("exemplars") or []
            shares = r.get("top10_shares") or r.get("top3_shares") or []
            present = False
            for i, e in enumerate(ex):
                if has_ldir(e["tape"]):
                    present = True
                    if pre and i < len(shares):
                        max_share = max(max_share, float(shares[i]))
            if pre:
                n_samples_pre += 1
                n_present_pre += int(present)
                if present and first < 0:
                    first = r["step"]
                if present_prev and not present:
                    episodes += 1
            present_prev = present
        rows.append({"label": s["label"], "seed": s["seed"], "t_rep": t_rep, "emerged": t_rep > 0, "ldir_first": "block-copy" in mechs,
                     "first_ldir_in_top10": first, "episodes_pre": episodes, "max_ldir_share_pre": max_share, "frac_samples_with_ldir_pre": n_present_pre / n_samples_pre if n_samples_pre else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, f"nascent_st{a.steps}_k{a.k}.csv"), index=False)
    agg = df.groupby("label").agg(n=("seed", "size"), emerged=("emerged", "sum"), runs_with_ldir_in_top10_pre=("first_ldir_in_top10", lambda x: int((x > 0).sum())),
                                  first_ldir_median=("first_ldir_in_top10", lambda x: float(np.median(x[x > 0])) if (x > 0).any() else np.nan),
                                  episodes_median=("episodes_pre", "median"), episodes_max=("episodes_pre", "max"),
                                  max_share_median=("max_ldir_share_pre", "median"), frac_samples_median=("frac_samples_with_ldir_pre", "median")).reindex(labels)
    md = [f"# Nascent LDIR copiers in the top-10 before emergence (generated) — {os.path.basename(a.dir)}, {a.steps} steps, 1/{2**a.k}\n", agg.round(3).to_markdown() + "\n",
          "`runs_with_ldir_in_top10_pre`: runs in which an LDIR-bearing tape entered the top-10 before the heritable event (or ever, if none); `episodes_pre`: times such a tape was present at one sample and absent at the next, before the event; `max_share`: largest top-10 share an LDIR tape held before the event.\n"]
    with open(os.path.join(out, f"NUMBERS_NASCENT_st{a.steps}_k{a.k}.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
