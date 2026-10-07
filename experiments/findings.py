"""Every number quoted in a FINDINGS document, generated from the analysis tables → analysis/NUMBERS.md.

    python findings.py runs/stageB [runs/stageA] --out runs/stageB/analysis/NUMBERS.md

Review 2026-10-07: the first Stage B findings were typed from console tables and several
numbers came from the wrong row. Prose may only quote numbers that appear in this file.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import brotli
import numpy as np
import pandas as pd

from algocell_exp.batch import select_summaries
from analyze import km_median, wilson
from report import fisher, km_str


def zero_fraction_table(dirs: list[str]) -> pd.DataFrame:
    """Fraction of 0x00 bytes in the final snapshot and in the emergence snapshot (tq_10 crossing), per cell."""
    rows = []
    for d in dirs:
        for p in select_summaries(d):
            s = json.load(open(p))
            L = s.get("tape_length", 16)
            stem = p[: -len(".summary.json")]
            rec = {"label": s["label"], "tape": L, "steps": s["z80_steps"], "k": s["noise_exp"], "seed": s["seed"], "steps_run": s["steps_run"], "H0_final": s["final"].get("H0")}
            for name in ("final", "emergence"):
                f = f"{stem}.soup_{name}.u8.br"
                if os.path.exists(f) and (name == "final" or s["tq_10"] >= 0):
                    raw = np.frombuffer(brotli.decompress(open(f, "rb").read()), dtype=np.uint8)
                    rec[f"zero_{name}"] = float((raw == 0).mean())
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    asy = pd.concat([pd.read_csv(os.path.join(d, "analysis", "assays.csv")) for d in a.dirs], ignore_index=True)
    cells = pd.concat([pd.read_csv(os.path.join(d, "analysis", "cells.csv")) for d in a.dirs], ignore_index=True)
    succ = pd.concat([pd.read_csv(os.path.join(d, "analysis", "succession.csv")) for d in a.dirs if os.path.exists(os.path.join(d, "analysis", "succession.csv"))], ignore_index=True)
    out = []

    # 1. emergence by L (k = 4): tq_10 / t_rep / t_faith counts and KM medians
    out.append("## 1. Emergence by tape length (mutation 1/16)\n")
    sub = cells[cells["k"] == 4].sort_values(["label", "steps", "tape"])
    tab = sub.assign(trep=lambda d: d.apply(lambda r: f"{int(r.t_rep_n)}/{int(r.n)} ({km_str(r.t_rep_km_median)})", axis=1),
                     tfaith=lambda d: d.apply(lambda r: f"{int(r.t_faith_n)}/{int(r.n)}", axis=1),
                     tq10=lambda d: d.apply(lambda r: f"{int(r.tq_10_n)}/{int(r.n)}", axis=1),
                     stopped=lambda d: d["stopped_early"].astype(int))
    for label in tab["label"].unique():
        t = tab[tab["label"] == label].pivot_table(index="tape", columns="steps", values=["trep", "tfaith", "tq10", "stopped"], aggfunc="first")
        out.append(f"### {label}\n\n" + t.to_markdown() + "\n")

    # 2. the L = 9 contrast (post hoc), one- and two-sided
    out.append("## 2. L = 9: none vs stack-writes (post hoc; one cell of the design)\n")
    for steps in (128, 512):
        n_ = cells[(cells["label"] == "none") & (cells["tape"] == 9) & (cells["steps"] == steps) & (cells["k"] == 4)]
        s_ = cells[(cells["label"] == "stack-writes") & (cells["tape"] == 9) & (cells["steps"] == steps) & (cells["k"] == 4)]
        if len(n_) and len(s_):
            n_, s_ = n_.iloc[0], s_.iloc[0]
            for ev in ("t_rep", "t_faith", "tq_10"):
                a1, b1 = int(s_[f"{ev}_n"]), int(s_["n"] - s_[f"{ev}_n"])
                a2, b2 = int(n_[f"{ev}_n"]), int(n_["n"] - n_[f"{ev}_n"])
                out.append(f"- {steps} steps, {ev}: stack-writes {a1}/{a1+b1} vs none {a2}/{a2+b2}; Fisher two-sided p = {fisher(a1, b1, a2, b2):.3g}, one-sided (stack-writes greater) p = {fisher(a1, b1, a2, b2, 'greater'):.3g}; KM medians {km_str(s_[f'{ev}_km_median'])} vs {km_str(n_[f'{ev}_km_median'])}")
    out.append("")

    # 3. zero-byte load (from snapshots) and byte entropy
    out.append("## 3. Zero-byte fraction of the soup (from snapshots) and final byte entropy H0\n")
    z = zero_fraction_table(a.dirs)
    if len(z):
        g = z.groupby(["label", "tape", "steps"]).agg(n=("seed", "size"), zero_final_mean=("zero_final", "mean"), zero_final_min=("zero_final", "min"), zero_final_max=("zero_final", "max"),
                                                       zero_emergence_mean=("zero_emergence", "mean"), H0_final_mean=("H0_final", "mean"), steps_run_min=("steps_run", "min"))
        out.append(g.round(3).to_markdown() + "\n")

    # 4. tiling: period of the first-replicator tape; divides L / 2L; HOE by label
    out.append("## 4. Tiling: period of the first heritable replicator tape\n")
    rep = asy[(asy["t_rep"] > 0) & asy["trep_period"].notna()].copy()
    rep["tiled"] = rep["trep_period"] <= rep["tape_len"] / 2
    rep["whole_tape"] = (rep["trep_period"] == rep["tape_len"]) & (rep.get("trep_offset", pd.Series(np.nan, index=rep.index)).fillna(-1) == 0)
    # divisor statistics over TILED tapes only (an untiled tape's period L divides 2L trivially; review 2026-10-07)
    tiled = rep[rep["tiled"]].copy()
    tiled["div_L"] = (tiled["tape_len"] % tiled["trep_period"].astype(int) == 0)
    tiled["div_2L"] = ((2 * tiled["tape_len"]) % tiled["trep_period"].astype(int) == 0)
    if "trep_offset" in tiled:
        off = tiled["trep_offset"].fillna(-1).astype(int)
        tiled["gcd_law"] = [(o >= 0 and np.gcd(o if o > 0 else 2 * L, 2 * L) == p) for o, L, p in zip(off, tiled["tape_len"], tiled["trep_period"].astype(int))]
    g = rep.groupby(["label", "tape_len", "steps"]).agg(n=("seed", "size"), period_median=("trep_period", "median"), tiled=("tiled", "mean"), whole_tape=("whole_tape", "sum"),
                                                     periods=("trep_period", lambda x: ", ".join(f"{int(p)}×{c}" for p, c in x.value_counts().sort_index().items())))
    g2 = tiled.groupby(["label", "tape_len", "steps"]).agg(n_tiled=("seed", "size"), divides_L=("div_L", "mean"), divides_2L=("div_2L", "mean"), **({"gcd_law": ("gcd_law", "mean")} if "gcd_law" in tiled else {}))
    out.append(g.join(g2, how="left").round(2).to_markdown() + "\n")
    out.append("`tiled` = period ≤ L/2; `whole_tape` = period L with copy offset 0 (exact whole-tape copier); `divides_*` and `gcd_law` (period == gcd(offset, 2L)) are computed over tiled tapes only.\n")
    out.append("### Final-soup high-order entropy (bits/byte), median over seeds\n")
    out.append(asy.groupby(["label", "tape_len", "steps"])["final_hoe"].median().unstack("steps").round(2).to_markdown() + "\n")

    # 5. faithfulness and functional fraction
    out.append("## 5. Faithfulness and functional fraction of the final population\n")
    g = asy.groupby(["label", "tape_len", "steps"]).agg(n=("seed", "size"), final_replicator=("final_replicator", "sum"), final_faithful=("final_faithful", "sum"),
                                                     func_rnd_median=("final_func_rnd", "median"), func_rnd_faithful_median=("final_func_rnd_faithful", "median"),
                                                     func_insitu_median=("final_func_insitu", "median"), insitu_informative_runs=("final_func_insitu", lambda x: int(x.notna().sum())),
                                                     trep_unfaithful=("trep_faithful", lambda x: int((~x.fillna(True).astype(bool)).sum())))
    out.append(g.round(2).to_markdown() + "\n")

    # 6. succession at fixed steps
    if len(succ) and "family_300k" in succ:
        out.append("## 6. Census family at 300k steps and at the last step\n")
        g = succ.groupby(["label", "tape", "steps"]).agg(n=("seed", "size"), stopped_early=("stopped_early", lambda x: int(x.fillna(False).astype(bool).sum())),
                                                      family_300k=("family_300k", lambda x: ", ".join(f"{k}:{v}" for k, v in x.dropna().value_counts().items()) or "–"),
                                                      family_last=("final_family", lambda x: ", ".join(f"{k}:{v}" for k, v in x.value_counts().items())),
                                                      stack_takeover=("t_push_30", lambda x: f"{int((x > 0).sum())} runs, median {x[x > 0].median():.0f}" if (x > 0).any() else "0"),
                                                      ldir_takeover=("t_ldir_30", lambda x: f"{int((x > 0).sum())} runs, median {x[x > 0].median():.0f}" if (x > 0).any() else "0"))
        out.append(g.to_markdown() + "\n")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("# Numbers for the findings (generated; do not edit)\n\n" + "\n".join(out))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
