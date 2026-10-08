"""Stage I analysis: does lethal tar shorten the open window in the first machine? Pre-registered in PLAN.md
(2026-10-08 night, "Stage I pre-registration") before any Stage I run.

    python stage_i_analysis.py [--runs runs/stageI] [--assays results/stageI/assays.csv] [--c4 results/stageI/c4/functional.csv]
                               [--table results/stageI/stageI/stage_g_runs.csv] [--out results/stageI]

Per world: zero fraction at step 500 (I1); t_rep and the first heritable tape (assays); heritable fraction of 16 random
cells per sampled step (c4); the first tape's class share over time (samples: top-10 exemplars and shares); the final
dominant's control flow (stage_g table). Verdicts against the pre-registered thresholds:
  I1  zero fraction ≥ 0.15 at step 500 in ≥ 7/10 worlds
  strong form  ≥ 7/10 worlds without a heritable replicator by the horizon
  I2  among worlds with a replicator: heritability < 0.3 throughout and the first tape's share < 1% within 100,000 steps of
      t_rep (open, then extinct), or the world closes; "open, then extinct" in ≥ 5/10 worlds
  I3  closed dominant (control-flow or block instruction and heritable fraction ≥ 0.7 at the last sampled step) in ≤ 6/10
      worlds (control 20/20); kill ≥ 9/10
  I4  (descriptive) closed dominants contain no zero byte
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")


def load_samples(path: str) -> list[dict]:
    out = []
    for line in open(path):
        if '"kind": "sample"' in line:
            r = json.loads(line)
            out.append({"step": r["step"], "zero": r.get("zero_frac", np.nan),
                        "shares": dict(zip(r.get("top10_hashes", []), r.get("top10_shares", []))),
                        "tapes": {e.get("hash"): e.get("tape", "") for e in (r.get("exemplars") or [])},
                        "min_top10": min(r.get("top10_shares", [1.0]) or [1.0])})
    out.sort(key=lambda r: r["step"])
    return out


def share_trajectory(samples: list[dict], tape: str) -> list[tuple[int, float]]:
    """The class share of `tape` at every sample: its recorded share when in the top ten, else an upper bound (the
    smallest top-ten share), reported as negative to mark the bound."""
    h = None
    for s in samples:
        for hh, t in s["tapes"].items():
            if t == tape:
                h = hh
                break
        if h is not None:
            break
    traj = []
    for s in samples:
        if h is not None and h in s["shares"]:
            traj.append((s["step"], float(s["shares"][h])))
        else:
            traj.append((s["step"], -float(s["min_top10"])))
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs", "stageI"))
    ap.add_argument("--assays", default=os.path.join(R, "stageI", "assays.csv"))
    ap.add_argument("--c4", default=os.path.join(R, "stageI", "c4", "functional.csv"))
    ap.add_argument("--table", default=os.path.join(R, "stageI", "stageI", "stage_g_runs.csv"))
    ap.add_argument("--out", default=os.path.join(R, "stageI"))
    ap.add_argument("--horizon", type=int, default=300000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    A = pd.read_csv(a.assays)
    C = pd.read_csv(a.c4)
    T = pd.read_csv(a.table) if os.path.exists(a.table) else None
    rows = []
    for f in sorted(glob.glob(os.path.join(a.runs, "*.jsonl"))):
        stem = os.path.basename(f)[:-6]
        seed = int(stem.rsplit("_s", 1)[1])
        S = load_samples(f)
        z500 = next((s["zero"] for s in S if s["step"] >= 500), np.nan)
        ar = A[A["seed"] == seed]
        t_rep = float(ar["t_rep"].iloc[0]) if len(ar) else np.nan
        t_rep = np.nan if (not np.isfinite(t_rep) or t_rep < 0) else t_rep
        first_tape = str(ar["trep_tape"].iloc[0]) if len(ar) and isinstance(ar["trep_tape"].iloc[0], str) else ""
        c = C[C["seed"] == seed].sort_values("step")
        h_max = float(c["frac_heritable"].max()) if len(c) else np.nan
        h_last = float(c["frac_heritable"].iloc[-1]) if len(c) else np.nan
        extinct_step = np.nan
        if first_tape and np.isfinite(t_rep):
            traj = share_trajectory(S, first_tape)
            peaked = False
            for st, sh in traj:
                if st < t_rep:
                    continue
                if sh >= 0.01:
                    peaked = True
                elif peaked and abs(sh) < 0.01 and st <= t_rep + 100000:
                    extinct_step = st
                    break
        closed = False
        no_zero = np.nan
        if T is not None:
            tr = T[T["seed"] == seed]
            if len(tr):
                has_loop = bool(tr["final_has_cf"].iloc[0]) or bool(tr["final_has_block"].iloc[0])
                closed = has_loop and (h_last >= 0.7)
                ft = str(tr["final_tape"].iloc[0])
                no_zero = "00" not in ft.split()
        open_then_extinct = np.isfinite(t_rep) and (h_max < 0.3) and np.isfinite(extinct_step)
        rows.append({"seed": seed, "zero_500": z500, "t_rep": t_rep, "first_tape": first_tape[:23], "h_max": h_max, "h_last": h_last,
                     "extinct_step": extinct_step, "open_then_extinct": bool(open_then_extinct), "closed": bool(closed), "closed_no_zero": no_zero})
    W = pd.DataFrame(rows)
    W.to_csv(os.path.join(a.out, "per_world.csv"), index=False)
    n = len(W)
    i1 = int((W["zero_500"] >= 0.15).sum())
    no_life = int(W["t_rep"].isna().sum())
    i2 = int(W["open_then_extinct"].sum())
    i3 = int(W["closed"].sum())
    pred = pd.DataFrame([
        {"prediction": "I1", "statement": "zero fraction ≥ 0.15 at step 500 in ≥ 7/10", "value": f"{i1}/{n}", "outcome": "met" if i1 >= 7 else "not met"},
        {"prediction": "strong form", "statement": "≥ 7/10 worlds without a heritable replicator by horizon", "value": f"{no_life}/{n}", "outcome": "triggered" if no_life >= 7 else "not triggered"},
        {"prediction": "I2", "statement": "open, then extinct (heritability < 0.3 throughout; first tape < 1% within 100k of t_rep) in ≥ 5/10", "value": f"{i2}/{n} (worlds with a replicator: {n - no_life})", "outcome": "met" if i2 >= 5 else "not met"},
        {"prediction": "I3", "statement": "closed dominant by horizon ≤ 6/10 (kill ≥ 9/10)", "value": f"{i3}/{n}", "outcome": "met" if i3 <= 6 else ("killed" if i3 >= 9 else "between")},
        {"prediction": "I4", "statement": "closed dominants contain no zero byte (descriptive)", "value": f"{int(W.loc[W['closed'], 'closed_no_zero'].fillna(False).astype(bool).sum())}/{i3}", "outcome": "descriptive"},
    ])
    pred.to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    lines = ["# Stage I numbers (generated by stage_i_analysis.py)", "", f"- worlds {n}; zero fraction at step 500: median {W['zero_500'].median():.3f}",
             f"- heritable replicator by horizon: {n - no_life}/{n}; t_rep median {W['t_rep'].median():.0f}" if n - no_life else f"- heritable replicator by horizon: 0/{n}",
             f"- max heritable fraction: median {W['h_max'].median():.3f}; last-step median {W['h_last'].median():.3f}",
             f"- open then extinct: {i2}; closed: {i3}", "", "## Verdicts", "", pred.to_markdown(index=False), "", "## Per world", "", W.to_markdown(index=False)]
    open(os.path.join(a.out, "NUMBERS_I.md"), "w").write("\n".join(lines))
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
