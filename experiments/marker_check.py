"""V — the L = 50 invasion marker against traced confinement (pre-registered: REVISION_PREREG.md, R2/V, 2026-10-09).

    .venv/bin/python marker_check.py --yes

Re-runs the closed-into-pusher invasion of invasion_pair.py (seeds 1-3, identical initial conditions and stepping) and at
steps 0, 50, 100, 150, 200, 300, 500, 1,000 traces 512 random cells against 16 random partners each. A cell is confined
if no partner byte is fetched in any of the 16 encounters; it carries the marker if it contains the jump word `20 f0`.
Output: results/marker_check/marker_check.csv, REPORT.md.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from algocell_exp import exectrace as X  # noqa: E402
from algocell_exp.soup import Soup  # noqa: E402

L = 50
CHECK = (0, 50, 100, 150, 200, 300, 500, 1000)


def parse(s):
    return np.array([int(b, 16) for b in s.split()], dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cells", type=int, default=512)
    ap.add_argument("--partners", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "marker_check"))
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if not a.yes:
        sys.exit("refusing to run: pass --yes after pausing the browser simulation (local GPU)")
    g = pd.read_csv(os.path.join(HERE, "results", "stageG", "stageG", "stage_g_runs.csv"))
    m = g[(g.L == 50) & g.final_tape.str.startswith("01 c5") & g.final_tape.str.contains("20 f0")]
    closed, pusher = parse(m.final_tape.mode().iloc[0]), parse(m.first_tape.mode().iloc[0])
    ref = pd.read_csv(os.path.join(HERE, "results", "invasion_pair", "invasion_pair.csv"))
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for seed in range(1, a.seeds + 1):
        soup = Soup(160, 125, "square", L, 100 + seed, 8192, 128, 4, [])
        rng = np.random.default_rng([seed, L, 1])                       # as invasion_pair.py, resident = pusher
        cells = soup.read_soup()
        cells[:] = pusher[None, :]
        idx = rng.choice(len(cells), size=int(0.01 * len(cells)), replace=False)
        cells[idx] = closed[None, :]
        soup.write_soup(cells)
        crng = np.random.default_rng([20261009, seed])
        step = 0
        while step <= max(CHECK):
            if step in CHECK:
                s = soup.read_soup()
                motif_all = ((s == 0x20) & (np.roll(s, -1, axis=1) == 0xF0)).any(axis=1)
                pick = crng.choice(len(s), size=a.cells, replace=False)
                c = s[pick]
                P = crng.integers(0, 256, size=(a.cells * a.partners, L), dtype=np.uint8)
                _, masks = X.execute_pairs_traced(np.concatenate([np.repeat(c, a.partners, 0), P], 1), L, 128)
                ent = X.exec_positions(masks, 2 * L)[:, L:].any(axis=1).reshape(a.cells, a.partners)
                confined = ~ent.any(axis=1)
                motif = motif_all[pick]
                njump = ((c == 0x20) & (np.roll(c, -1, axis=1) == 0xF0)).sum(axis=1)
                r0 = ref[(ref.invader == "closed") & (ref.resident == "pusher") & (ref.seed == seed) & (ref.step == step)]
                rows.append({"seed": seed, "step": step, "marker_share_soup": float(motif_all.mean()),
                             "marker_share_ref": float(r0.jump_share.iloc[0]) if len(r0) else np.nan,
                             "marker_share_sample": float(motif.mean()), "confined_share": float(confined.mean()),
                             "confined_given_marker": float(confined[motif].mean()) if motif.any() else np.nan,
                             "confined_given_no_marker": float(confined[~motif].mean()) if (~motif).any() else np.nan,
                             "n_marker": int(motif.sum()), "n_confined": int(confined.sum()),
                             **{f"confined_given_{k}jumps": (float(confined[njump == k].mean()) if (njump == k).any() else np.nan) for k in range(0, 7)},
                             **{f"n_{k}jumps": int((njump == k).sum()) for k in range(0, 7)}, "n_7plus": int((njump >= 7).sum())})
                r = rows[-1]
                print(f"seed {seed} step {step:5d}: marker {r['marker_share_soup']:.3f} (ref {r['marker_share_ref']:.3f}) confined {r['confined_share']:.3f} | "
                      f"P(conf|marker) {r['confined_given_marker']:.2f} P(conf|no marker) {r['confined_given_no_marker']:.2f}", flush=True)
            dt = 10 if step < 500 else 250
            soup.step(dt)
            step += dt
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "marker_check.csv"), index=False)
    gap = (df.confined_share - df.marker_share_sample).abs()
    pm = df.confined_given_marker.dropna()
    pn = df.confined_given_no_marker.dropna()
    ok = bool((gap <= 0.10).all() and (pm >= 0.90).all() and (pn <= 0.10).all())
    lines = ["# V — invasion marker against traced confinement (generated by `marker_check.py`)", "",
             f"- reproduction check: marker share in the re-run equals invasion_pair.csv at every checked step: {bool(np.allclose(df.marker_share_soup, df.marker_share_ref, atol=1e-9, equal_nan=True))}",
             f"- V1 ({'met' if ok else 'NOT met'}): max |confined − marker| = {gap.max():.3f}; P(confined | marker) min {pm.min():.2f}, median {pm.median():.2f}; "
             f"P(confined | no marker) max {pn.max():.2f}, median {pn.median():.2f}", "",
             "| seed | step | marker share (soup) | confined share (512 cells) | P(confined given marker) | P(confined given no marker) |", "|---|---|---|---|---|---|"]
    lines += [f"| {r.seed} | {r.step} | {r.marker_share_soup:.3f} | {r.confined_share:.3f} | {r.confined_given_marker:.2f} | {r.confined_given_no_marker:.2f} |" for r in df.itertuples()]
    agg = {k: (df[f"n_{k}jumps"].sum(), np.nansum(df[f"confined_given_{k}jumps"] * df[f"n_{k}jumps"])) for k in range(7)}
    lines += ["", "Post hoc: confinement by number of jump words in the cell (all checked samples pooled): " +
              "; ".join(f"{k} jumps: {int(c)}/{int(n)} confined" for k, (n, c) in agg.items() if n)]
    txt = "\n".join(lines)
    open(os.path.join(a.out, "REPORT.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
