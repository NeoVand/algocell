"""Life detectors versus ground truth, time-resolved and size-resolved (CPU only; uses the per-sample statistics
recorded by every run and the assay-based ground truth).

    python detectors.py --c runs/stageC --e runs/stageE [--out runs/detectors]

Ground truths per (run, step):
  * `heritable` (Stage C only): the C4 fraction of 16 random cells with gen2 ≥ 0.3 is ≥ 0.25 at that step
    (c4/functional.csv, same steps as the snapshot schedule);
  * `event`: the run's first heritable top-3 exemplar has appeared (step ≥ t_rep > 0; assays.csv).
Detectors (per-sample fields recorded by the run; orientation fixed here, before any AUC was computed):
  higher = alive: hoe (H0 − brotli bits/byte), nonunique (1 − unique/cells), top_share, q_share, q_shift_share,
                  motif_share, ix_wb_ge_L_frac (fraction of interactions writing ≥ L bytes into the partner);
  lower  = alive: H_species, brotli_bpb, ix_silent_frac.
AUC is the Mann–Whitney statistic over all (run, step) samples of a stratum (samples of one run are correlated;
strata are reported per L and per step so that the pooled numbers can be read with that in mind).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import figstyle as fs
from algocell_exp.batch import select_summaries

STEPS = [500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 50000, 75000, 100000, 150000, 200000, 300000, 500000, 750000, 1000000]
UP = ["hoe", "nonunique", "top_share", "q_share", "q_shift_share", "motif_share", "ix_wb_ge_L_frac"]
DOWN = ["H_species", "brotli_bpb", "ix_silent_frac"]


def auc(score: np.ndarray, truth: np.ndarray) -> float:
    pos, neg = score[truth], score[~truth]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    # Mann–Whitney with ties = 0.5
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    vals = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def samples(run_dir: str) -> pd.DataFrame:
    rows = []
    for f in select_summaries(run_dir):
        s = json.load(open(f))
        stem = f[: -len(".summary.json")]
        if not os.path.exists(stem + ".jsonl"):
            continue
        want = set(STEPS)
        for line in open(stem + ".jsonl"):
            if '"kind": "sample"' not in line:
                continue
            r = json.loads(line)
            if r["step"] not in want:
                continue
            rows.append({"file": os.path.basename(f), "label": s["label"], "tape_len": s.get("tape_length", 16), "steps": s["z80_steps"], "k": s["noise_exp"], "seed": s["seed"], "cells": s["cells"], "step": r["step"],
                         "hoe": r.get("hoe"), "nonunique": 1 - r["unique"] / s["cells"], "top_share": r.get("top_share"), "q_share": r.get("q_share"), "q_shift_share": r.get("q_shift_share"),
                         "motif_share": r.get("motif_share"), "ix_wb_ge_L_frac": r.get("ix_wb_ge_L_frac"), "H_species": r.get("H_species"), "brotli_bpb": r.get("brotli_bpb"), "ix_silent_frac": r.get("ix_silent_frac"), "zero_frac": r.get("zero_frac")})
    return pd.DataFrame(rows)


def with_truth(df: pd.DataFrame, run_dir: str) -> pd.DataFrame:
    asy = pd.read_csv(os.path.join(run_dir, "analysis", "assays.csv"))[["file", "t_rep"]]
    df = df.merge(asy, on="file", how="left")
    df["event"] = (df["t_rep"] > 0) & (df["step"] >= df["t_rep"])
    c4 = os.path.join(run_dir, "analysis", "c4", "functional.csv")
    if os.path.exists(c4):
        fc = pd.read_csv(c4)[["file", "step", "frac_heritable"]]
        df = df.merge(fc, on=["file", "step"], how="left")
        df["heritable"] = df["frac_heritable"] >= 0.25
    return df


def table(df: pd.DataFrame, truth: str, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by) if by else [((), df)]:
        g = g[g[truth].notna()]
        t = g[truth].astype(bool).to_numpy()
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row.update({"n": len(g), "alive": int(t.sum())})
        for d in UP:
            row[d] = auc(g[d].astype(float).fillna(-np.inf).to_numpy(), t)
        for d in DOWN:
            row[d] = auc(-g[d].astype(float).fillna(np.inf).to_numpy(), t)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", default="runs/stageC")
    ap.add_argument("--e", default="runs/stageE")
    ap.add_argument("--out", default="runs/detectors")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    md = ["# Life detectors vs ground truth (generated)\n"]
    fs.setup()
    import matplotlib.pyplot as plt
    colors = dict(zip(UP + DOWN, ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#999999", "#7F3C8D", "#11A579"]))

    if os.path.isdir(a.c):
        dc = with_truth(samples(a.c), a.c)
        dc.to_csv(os.path.join(a.out, "samples_C.csv"), index=False)
        if "heritable" in dc:
            md.append("## Stage C, truth = C4 heritable fraction ≥ 0.25 (pooled over all runs and steps, then by L, then by step at L = 16 · 128 · 1/16)\n")
            md.append(table(dc, "heritable", []).round(3).to_markdown(index=False) + "\n")
            md.append(table(dc, "heritable", ["tape_len"]).round(3).to_markdown(index=False) + "\n")
            sub = dc[(dc["tape_len"] == 16) & (dc["steps"] == 128) & (dc["k"] == 4)]
            ts = table(sub, "heritable", ["step"])
            md.append(ts.round(3).to_markdown(index=False) + "\n")
            fig, ax = plt.subplots(figsize=(fs.SINGLE, 2.4))
            for d in UP + DOWN:
                ax.plot(ts["step"], ts[d], color=colors[d], lw=1, marker="o", ms=2.5, label=d)
            ax.axhline(0.5, color="#bbbbbb", lw=0.6, ls=":")
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("step (L = 16, 128 steps, 1/16, all C3 arms)")
            ax.set_ylabel("AUC vs C4 heritable ≥ 0.25")
            ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="lower right")
            fs.save(fig, os.path.join(a.out, "detectors_C_by_step"))
        md.append("## Stage C, truth = first heritable exemplar has appeared (event)\n")
        md.append(table(dc, "event", ["tape_len"]).round(3).to_markdown(index=False) + "\n")

    if os.path.isdir(a.e):
        de = with_truth(samples(a.e), a.e)
        de.to_csv(os.path.join(a.out, "samples_E.csv"), index=False)
        de_nom = de[de["label"].isin(["none@nominal", "stack-write-only@nominal"])]
        tl = table(de_nom, "event", ["tape_len"])
        md.append("## Stage E @nominal, truth = event, by L (both ablations pooled)\n")
        md.append(tl.round(3).to_markdown(index=False) + "\n")
        md.append("### by ablation and L\n")
        md.append(table(de_nom, "event", ["label", "tape_len"]).round(3).to_markdown(index=False) + "\n")
        fig, ax = plt.subplots(figsize=(fs.SINGLE, 2.4))
        for d in UP + DOWN:
            ax.plot(tl["tape_len"], tl[d], color=colors[d], lw=1, marker="o", ms=2.5, label=d)
        ax.axhline(0.5, color="#bbbbbb", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xticks([3, 4, 9, 16, 25, 36, 49, 64, 81, 100], ["3", "4", "9", "16", "25", "36", "49", "64", "81", "100"])
        ax.minorticks_off()
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("tape length L (Stage E @nominal, truth = heritable exemplar present)")
        ax.set_ylabel("AUC")
        ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="lower left")
        fs.save(fig, os.path.join(a.out, "detectors_E_by_L"))

    with open(os.path.join(a.out, "NUMBERS_DETECTORS.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
