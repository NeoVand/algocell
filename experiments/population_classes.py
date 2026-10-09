"""Q2 — population classes over time (registered before the run: REVISION_PREREG.md, R2b, 2026-10-09; exploratory).

    .venv/bin/python population_classes.py --yes [--stages G,I,K,L,M] [--finals-only]
    .venv/bin/python population_classes.py --yes --soups 'runs/invasion_closer/A1_*_final.npy' --L 32 --ablation i8080 --tag A1
    .venv/bin/python population_classes.py --report

Per snapshot: 64 random cells (seeded); culture test (32 partners; heritable iff gen2 >= 0.3); traced against 16 random
partners (confined iff no partner byte fetched in any). Up to 16 random heritable cells: single-mutant scan (16 values per
position, 16 partners) and the executed union of their 16 traced encounters. Classes of heritable cells: open (not
confined); regenerator (confined, <= 2 transmissible sites); intermediate (confined, 3-4); transmitter (confined, >= 5).
Classes of unscanned confined heritable cells are inferred from the scanned confined ones (their proportions).
Output: results/population/classes_snapshots.csv, classes_cells.csv (or <tag>_*.csv), REPORT_classes.md.
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
sys.path.insert(0, os.path.join(HERE, "manuscript", "figures"))
import soup_stills as ss  # noqa: E402
from algocell_exp import assay as A  # noqa: E402
from algocell_exp import exectrace as X  # noqa: E402
from mutscan import scan_tape, summarise  # noqa: E402
from population_sample import STAGES  # noqa: E402

R = os.path.join(HERE, "results")
NCELL, NP_TRACE, NSCAN = 64, 16, 16


def cls(conf: bool, sites: int) -> str:
    if not conf:
        return "open"
    return "regenerator" if sites <= 2 else ("transmitter" if sites >= 5 else "intermediate")


def classify(soup: np.ndarray, L: int, sup: list, zh: bool, rng) -> tuple[dict, list[dict]]:
    cells = soup[rng.choice(len(soup), size=min(NCELL, len(soup)), replace=False)]
    res = A.assay_many(cells, z80_steps=128, suppress=sup, n=32, seed=int(rng.integers(1 << 31)), zero_halts=zh)
    herit = np.array([r["is_replicator"] for r in res])
    P = rng.integers(0, 256, size=(len(cells) * NP_TRACE, L), dtype=np.uint8)
    _, masks = X.execute_pairs_traced(np.concatenate([np.repeat(cells, NP_TRACE, 0), P], 1), L, 128, suppress=sup, zero_halts=zh)
    ex = X.exec_positions(masks, 2 * L).reshape(len(cells), NP_TRACE, 2 * L)
    conf = ~ex[:, :, L:].any(axis=(1, 2))
    union = ex[:, :, :L].any(axis=1)
    out_cells = []
    hidx = np.flatnonzero(herit)
    scanned = rng.permutation(hidx)[:NSCAN]
    for j in scanned:
        d = scan_tape(cells[j], 16, 16, 128, seed=int(rng.integers(1 << 31)), zero_halts=zh, suppress=tuple(sup))
        row, site = summarise(d, L)
        tr = site["transmissible"].values.astype(bool)
        out_cells.append({"cell": int(j), "tape": " ".join(f"{b:02x}" for b in cells[j]), "confined": bool(conf[j]), "n_sites": int(row["n_sites"]),
                          "n_sites_unexecuted": int((tr & ~union[j]).sum()), "n_executed": int(union[j].sum()), "class": cls(bool(conf[j]), int(row["n_sites"]))})
    nh = int(herit.sum())
    nopen = int((herit & ~conf).sum())
    nconf = nh - nopen
    sc = pd.DataFrame(out_cells)
    scc = sc[sc.confined] if len(sc) else sc
    snap = {"n_cells": len(cells), "frac_heritable": float(herit.mean()), "frac_open_of_heritable": nopen / nh if nh else np.nan,
            "frac_confined_of_heritable": nconf / nh if nh else np.nan, "n_scanned": len(sc), "n_scanned_confined": len(scc)}
    for c in ("regenerator", "intermediate", "transmitter"):
        p = float((scc["class"] == c).mean()) if len(scc) else np.nan
        snap[f"frac_{c}_of_heritable"] = (nconf / nh * p) if (nh and len(scc)) else (0.0 if nh and nconf == 0 else np.nan)
    tm = scc[scc["class"] == "transmitter"] if len(scc) else scc
    snap["median_unexec_sites_transmitters"] = float(tm.n_sites_unexecuted.median()) if len(tm) else np.nan
    snap["median_sites_confined_scanned"] = float(scc.n_sites.median()) if len(scc) else np.nan
    return snap, out_cells


def run_stages(a) -> None:
    from make_conds import ABLATIONS
    os.makedirs(a.out, exist_ok=True)
    srows, crows = [], []
    t0 = time.time()
    sp, cp = os.path.join(a.out, "classes_snapshots.csv"), os.path.join(a.out, "classes_cells.csv")
    done = set()
    if os.path.exists(sp) and not a.fresh:
        old = pd.read_csv(sp)
        srows = old.to_dict("records")
        crows = pd.read_csv(cp).to_dict("records") if os.path.exists(cp) else []
        done = {(r["stage"], r["L"], r["seed"], r["snapshot"]) for r in srows}
    for stage in a.stages.split(","):
        runs_dir, csv, zh, abl = STAGES[stage]
        sup = list(ABLATIONS[abl]) if abl else []
        g = pd.read_csv(csv)
        for _, w in g.iterrows():
            L, seed, label = int(w["L"]), int(w["seed"]), str(w["label"])
            stem = os.path.join(runs_dir, f"{label}_L{L}_st128_k4_s{seed}")
            snaps = ss.snapshot_files(stem)
            if a.finals_only:
                snaps = [x for x in snaps if x[0] == "final"]
            for name, step, f in snaps:
                if (stage, L, seed, name) in done:
                    continue
                soup = ss.load(f, L)
                rng = np.random.default_rng([20261012, L, seed, step if step is not None else -1, len(name)])
                snap, cells = classify(soup, L, sup, zh, rng)
                base = {"stage": stage, "L": L, "seed": seed, "snapshot": name, "step": step}
                srows.append({**base, **snap})
                crows += [{**base, **c} for c in cells]
            pd.DataFrame(srows).to_csv(sp, index=False)
            pd.DataFrame(crows).to_csv(cp, index=False)
            last = [r for r in srows if r["stage"] == stage and r["L"] == L and r["seed"] == seed]
            if last:
                r = last[-1]
                print(f"{stage} L{L} s{seed}: {len(last)} snapshots; final herit {r['frac_heritable']:.2f} open {r['frac_open_of_heritable']:.2f} "
                      f"regen {r['frac_regenerator_of_heritable']:.2f} trans {r['frac_transmitter_of_heritable']:.2f} ({time.time() - t0:.0f} s)", flush=True)
    report(a.out)


def run_soups(a) -> None:
    from make_conds import ABLATIONS
    sup = list(ABLATIONS[a.ablation]) if a.ablation else []
    os.makedirs(a.out, exist_ok=True)
    srows, crows = [], []
    for f in sorted(glob.glob(a.soups)):
        soup = np.load(f)
        rng = np.random.default_rng([20261012, a.L, len(f), sum(map(ord, os.path.basename(f)))])
        snap, cells = classify(soup, a.L, sup, False, rng)
        u, n = np.unique(soup, axis=0, return_counts=True)
        top = u[np.argsort(-n)[:3]]
        base = {"file": os.path.basename(f), "top1": " ".join(f"{b:02x}" for b in top[0]), "top1_share": float(n.max() / len(soup)),
                "zero_frac": float((soup == 0).mean())}
        srows.append({**base, **snap})
        crows += [{**base, **c} for c in cells]
        r = srows[-1]
        print(f"{r['file']}: herit {r['frac_heritable']:.2f} open {r['frac_open_of_heritable']:.2f} regen {r['frac_regenerator_of_heritable']:.2f} "
              f"trans {r['frac_transmitter_of_heritable']:.2f} | top {r['top1']} ({r['top1_share']:.3f})", flush=True)
    pd.DataFrame(srows).to_csv(os.path.join(a.out, f"{a.tag}_classes.csv"), index=False)
    pd.DataFrame(crows).to_csv(os.path.join(a.out, f"{a.tag}_cells.csv"), index=False)


def report(out: str) -> None:
    S = pd.read_csv(os.path.join(out, "classes_snapshots.csv"))
    F = S[S.snapshot == "final"]
    lines = ["# Q2 — population classes (generated by `population_classes.py`; exploratory, R2b)", "",
             "Final snapshots. Shares are of heritable cells; worlds counted where the class is the majority of heritable cells.", "",
             "| stage | L | worlds | heritable (median) | open (median) | regenerator (median) | transmitter (median) | majority open / regenerator / transmitter / none | unexecuted sites of transmitters (median) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (st, L), d in F.groupby(["stage", "L"]):
        maj = {c: int((d[f"frac_{c}_of_heritable"] > 0.5).sum()) for c in ("open", "regenerator", "transmitter")}
        none = len(d) - sum(maj.values())
        lines.append(f"| {st} | {L} | {len(d)} | {d.frac_heritable.median():.2f} | {d.frac_open_of_heritable.median():.2f} | {d.frac_regenerator_of_heritable.median():.2f} | "
                     f"{d.frac_transmitter_of_heritable.median():.2f} | {maj['open']} / {maj['regenerator']} / {maj['transmitter']} / {none} | {d.median_unexec_sites_transmitters.median():.1f} |")
    lines += ["", "Worlds in which transmitters were ever the majority of heritable cells at some snapshot, and at the final snapshot:", ""]
    for (st, L), d in S.groupby(["stage", "L"]):
        ever = d.groupby("seed").frac_transmitter_of_heritable.max()
        fin = d[d.snapshot == "final"].set_index("seed").frac_transmitter_of_heritable
        lines.append(f"- {st} L = {L}: ever {int((ever > 0.5).sum())} of {len(ever)}; at the end {int((fin > 0.5).sum())} of {len(fin)}; "
                     f"ever ≥ 10% of heritable cells in {int((ever >= 0.1).sum())} of {len(ever)}")
    txt = "\n".join(lines)
    open(os.path.join(out, "REPORT_classes.md"), "w").write(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="G,I,K,L,M")
    ap.add_argument("--finals-only", action="store_true")
    ap.add_argument("--soups", default="")
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--ablation", default="")
    ap.add_argument("--tag", default="soups")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--out", default=os.path.join(R, "population"))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(a.out)
        return
    if not a.yes:
        sys.exit("refusing to run: pass --yes after pausing the browser simulation (local GPU)")
    if a.soups:
        run_soups(a)
    else:
        run_stages(a)


if __name__ == "__main__":
    main()
