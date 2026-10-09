"""Q — population sampling of final snapshots (pre-registered: REVISION_PREREG.md, R2/Q, 2026-10-09).

    .venv/bin/python population_sample.py --yes [--stages G,K,I,M,L]

Final snapshot of every Stage G, K, I and M world and of every Stage L world (ten million steps): 256 random cells
(seeded), each assayed by the culture test (32 partners, gen2 >= 0.3) and traced against 16 random partners (confined if
no partner byte is fetched in any of the 16). Per world: heritable share, confined share among heritable cells, share of
heritable cells identical to the modal tape at some cyclic shift, median Hamming distance of heritable cells to the modal
tape at its best shift; whether the modal tape itself is confined (16 partners). Up to 8 random heritable cells per world
get the single-mutant scan (16 values per position, 16 partners) for the distribution of transmissible sites.
Output: results/population/population_worlds.csv, population_cells.csv, REPORT.md (Q1).
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
sys.path.insert(0, os.path.join(HERE, "manuscript", "figures"))
import soup_stills as ss  # noqa: E402
from algocell_exp import assay as A  # noqa: E402
from algocell_exp import exectrace as X  # noqa: E402
from mutscan import scan_tape, summarise  # noqa: E402

R = os.path.join(HERE, "results")
STAGES = {"G": (os.path.join(HERE, "runs", "stageG"), os.path.join(R, "stageG", "stageG", "stage_g_runs.csv"), False, ""),
          "K": (os.path.join(HERE, "runs", "stageK", "stageK"), os.path.join(R, "stageK", "stageK", "stage_g_runs.csv"), False, ""),
          "I": (os.path.join(HERE, "runs", "stageI"), os.path.join(R, "stageI", "stageI", "stage_g_runs.csv"), True, ""),
          "M": (os.path.join(HERE, "runs", "stageM", "stageM"), os.path.join(R, "stageM", "stageM", "stage_g_runs.csv"), False, "i8080"),
          "L": (os.path.join(HERE, "runs", "stageL", "stageL"), os.path.join(R, "stageL", "stageL", "stage_g_runs.csv"), False, "")}
NCELL, NP_TRACE, NSCAN = 256, 16, 8


def best_shift_hamming(cells: np.ndarray, t: np.ndarray) -> np.ndarray:
    L = t.size
    best = np.full(len(cells), L, dtype=int)
    for s in range(L):
        best = np.minimum(best, (cells != np.roll(t, s)).sum(axis=1))
    return best


def confined(cells: np.ndarray, L: int, sup, zh: bool, rng) -> np.ndarray:
    P = rng.integers(0, 256, size=(len(cells) * NP_TRACE, L), dtype=np.uint8)
    _, masks = X.execute_pairs_traced(np.concatenate([np.repeat(cells, NP_TRACE, 0), P], 1), L, 128, suppress=sup, zero_halts=zh)
    ent = X.exec_positions(masks, 2 * L)[:, L:].any(axis=1).reshape(len(cells), NP_TRACE)
    return ~ent.any(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="G,K,I,M,L")
    ap.add_argument("--out", default=os.path.join(R, "population"))
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if not a.yes:
        sys.exit("refusing to run: pass --yes after pausing the browser simulation (local GPU)")
    from make_conds import ABLATIONS
    os.makedirs(a.out, exist_ok=True)
    wrows, crows = [], []
    t0 = time.time()
    for stage in a.stages.split(","):
        runs_dir, csv, zh, abl = STAGES[stage]
        sup = list(ABLATIONS[abl]) if abl else []
        g = pd.read_csv(csv)
        for _, w in g.iterrows():
            L, seed, label = int(w["L"]), int(w["seed"]), str(w["label"])
            stem = os.path.join(runs_dir, f"{label}_L{L}_st128_k4_s{seed}")
            snaps = [x for x in ss.snapshot_files(stem) if x[0] == "final"]
            if not snaps:
                print("no final snapshot for", stem, flush=True)
                continue
            _, step, f = snaps[0]
            soup = ss.load(f, L)
            rng = np.random.default_rng([20261011, L, seed])
            uniq, counts = np.unique(soup, axis=0, return_counts=True)
            modal = uniq[int(np.argmax(counts))]
            cells = soup[rng.choice(len(soup), size=NCELL, replace=False)]
            res = A.assay_many(cells, z80_steps=128, suppress=sup, n=32, seed=int(rng.integers(1 << 31)), zero_halts=zh)
            herit = np.array([r["is_replicator"] for r in res])
            conf = confined(cells, L, sup, zh, rng)
            modal_conf = bool(confined(modal[None], L, sup, zh, rng)[0]) if modal.any() else False
            ham = best_shift_hamming(cells, modal)
            sites = []
            hidx = np.flatnonzero(herit)
            for j in rng.permutation(hidx)[:NSCAN]:
                d = scan_tape(cells[j], 16, 16, 128, seed=int(rng.integers(1 << 31)), zero_halts=zh, suppress=tuple(sup))
                row, site = summarise(d, L)
                sites.append(row["n_sites"])
                crows.append({"stage": stage, "L": L, "seed": seed, "cell": int(j), "tape": " ".join(f"{b:02x}" for b in cells[j]), "heritable": True,
                              "confined": bool(conf[j]), "hamming_to_modal": int(ham[j]), "n_sites": int(row["n_sites"]), "ctrl_herit": bool(d["ctrl"]["herit"])})
            wrows.append({"stage": stage, "L": L, "seed": seed, "step": step, "modal_tape": " ".join(f"{b:02x}" for b in modal), "modal_share": float(counts.max() / len(soup)),
                          "modal_confined": modal_conf, "frac_heritable": float(herit.mean()), "frac_confined": float(conf.mean()),
                          "confined_given_heritable": float(conf[herit].mean()) if herit.any() else np.nan,
                          "identical_to_modal_given_heritable": float((ham[herit] == 0).mean()) if herit.any() else np.nan,
                          "median_hamming_given_heritable": float(np.median(ham[herit])) if herit.any() else np.nan,
                          "n_scanned": len(sites), "median_sites_scanned": float(np.median(sites)) if sites else np.nan,
                          "max_sites_scanned": int(max(sites)) if sites else -1})
            r = wrows[-1]
            print(f"{stage} L{L} s{seed}: modal share {r['modal_share']:.3f} closed {int(modal_conf)} | herit {r['frac_heritable']:.2f} conf|herit {r['confined_given_heritable']:.2f} "
                  f"ident {r['identical_to_modal_given_heritable']:.2f} ham {r['median_hamming_given_heritable']} | sites med {r['median_sites_scanned']} max {r['max_sites_scanned']} ({time.time() - t0:.0f} s)", flush=True)
            pd.DataFrame(wrows).to_csv(os.path.join(a.out, "population_worlds.csv"), index=False)
            pd.DataFrame(crows).to_csv(os.path.join(a.out, "population_cells.csv"), index=False)
    report(a.out)


def report(out: str) -> None:
    W = pd.read_csv(os.path.join(out, "population_worlds.csv"))
    C = pd.read_csv(os.path.join(out, "population_cells.csv"))
    lines = ["# Q — population sampling of final snapshots (generated by `population_sample.py`)", "",
             "| stage | L | worlds | modal tape closed | heritable share (median) | confined among heritable, closed-modal worlds (median, min) | identical to modal among heritable (median) | sites of sampled heritable cells, closed-modal worlds (median, max) |",
             "|---|---|---|---|---|---|---|---|"]
    for (st, L), d in W.groupby(["stage", "L"]):
        dc = d[d.modal_confined.astype(bool)]
        cc = C[(C.stage == st) & (C.L == L) & C.seed.isin(dc.seed)]
        lines.append(f"| {st} | {L} | {len(d)} | {int(d.modal_confined.sum())} | {d.frac_heritable.median():.2f} | "
                     + (f"{dc.confined_given_heritable.median():.2f}, {dc.confined_given_heritable.min():.2f}" if len(dc) else "–") + f" | {d.identical_to_modal_given_heritable.median():.2f} | "
                     + (f"{cc.n_sites.median():.1f}, {int(cc.n_sites.max())}" if len(cc) else "–") + " |")
    dc = W[W.modal_confined.astype(bool)]
    cc = C.merge(dc[["stage", "L", "seed"]], on=["stage", "L", "seed"])
    ok1 = (dc.confined_given_heritable >= 0.9)
    q1 = bool(ok1.all() and cc.n_sites.median() <= 2)
    lines += ["", f"- Q1 ({'met' if q1 else 'NOT met'}): closed-modal worlds {len(dc)}; confined among heritable >= 0.90 in {int(ok1.sum())} of {len(dc)} "
              f"(fails: {', '.join(f'{r.stage} L{r.L} s{r.seed} ({r.confined_given_heritable:.2f})' for r in dc[~ok1].itertuples())}); "
              f"median transmissible sites of sampled heritable cells {cc.n_sites.median():.1f} (n = {len(cc)})"]
    txt = "\n".join(lines)
    open(os.path.join(out, "REPORT.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
