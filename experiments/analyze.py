"""Aggregate a batch of run summaries into the pre-registered tables and figures.

    python analyze.py runs/stageA            # directory of *.summary.json (+ .jsonl)
    python analyze.py runs/stageA --out figs/stageA

Produces, per (label, tape, steps, k) cell: n seeds, fraction reaching tq_10 within
the horizon with a Wilson 95% interval, median tq_10 among emerged runs, the
mechanism-class distribution of the first emergent replicator, and the horizon.
Censored runs are counted as censored, never imputed. Figures: emergence
fraction grids and Kaplan–Meier curves per ablation.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def load(d: str) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.summary.json"))):
        s = json.load(open(p))
        fe = s.get("first_emergent") or {}
        rows.append(
            {
                "label": s["label"],
                "tape": s.get("tape_length", 16),
                "steps": s["z80_steps"],
                "k": s["noise_exp"],
                "seed": s["seed"],
                "horizon": s["horizon"],
                "steps_run": s["steps_run"],
                "t_02": s["t_02"],
                "t_10": s["t_10"],
                "tq_10": s["tq_10"],
                "tq_50": s["tq_50"],
                "emerged": s["tq_10"] > 0,
                "first_tape": fe.get("tape"),
                "first_mech": "+".join(fe.get("mechanisms", [])) or ("-" if fe else None),
                "final_top_share": s["final"]["top_share"],
                "final_q_share": s["final"].get("q_share", s["final"].get("q4_share")),
                "final_H": s["final"]["H_species"],
                "final_hoe": s["final"]["hoe"],
                "final_tape": s["final"]["exemplars"][0]["tape"],
                "final_mech": "+".join(s["final"]["exemplars"][0]["mechanisms"]) or "-",
                "wall_s": s["wall_s"],
                "file": os.path.basename(p),
            }
        )
    return pd.DataFrame(rows)


def km(times: np.ndarray, events: np.ndarray, horizon: int):
    """Kaplan–Meier survival of 'not yet emerged' (times = tq_10 or horizon if censored)."""
    order = np.argsort(times)
    t, e = times[order], events[order]
    n = len(t)
    at_risk = n
    S = 1.0
    xs, ys = [0], [1.0]
    for i in range(n):
        if e[i]:
            S *= 1 - 1 / at_risk
            xs.append(t[i])
            ys.append(S)
        at_risk -= 1
    xs.append(horizon)
    ys.append(S)
    return np.array(xs), np.array(ys)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    has_rep = "emerged_rep" in df
    for (label, tape, steps, k), g in df.groupby(["label", "tape", "steps", "k"]):
        n = len(g)
        ne = int(g["emerged"].sum())
        lo, hi = wilson(ne, n)
        em = g[g["emerged"]]
        mech = Counter(em["first_mech"].fillna("-"))
        row = {
            "label": label,
            "tape": tape,
            "steps": steps,
            "k": k,
            "n": n,
            "emerged": ne,
            "frac": ne / n,
            "ci_lo": lo,
            "ci_hi": hi,
            "median_tq10": float(em["tq_10"].median()) if ne else float("nan"),
            "min_tq10": int(em["tq_10"].min()) if ne else -1,
            "mechanisms": ", ".join(f"{m}:{c}" for m, c in mech.most_common()),
            "horizon": int(g["horizon"].max()),
            "censored_steps_run": int(g[~g["emerged"]]["steps_run"].min()) if ne < n else -1,
        }
        if has_rep:
            nr = int(g["emerged_rep"].sum())
            rlo, rhi = wilson(nr, n)
            er = g[g["emerged_rep"]]
            row.update(
                {
                    "rep_emerged": nr,
                    "rep_frac": nr / n,
                    "rep_ci_lo": rlo,
                    "rep_ci_hi": rhi,
                    "median_trep": float(er["t_rep"].median()) if nr else float("nan"),
                }
            )
        out.append(row)
    return pd.DataFrame(out).sort_values(["label", "tape", "steps", "k"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = load(a.dir)
    if df.empty:
        print("no summaries in", a.dir)
        return
    out = a.out or os.path.join(a.dir, "analysis")
    os.makedirs(out, exist_ok=True)
    # Assay-based emergence (t_rep) from assay_batch.py, if it has been run.
    assays_path = os.path.join(out, "assays.csv")
    if os.path.exists(assays_path):
        asy = pd.read_csv(assays_path)[["file", "t_rep", "t_rep_tape", "t_rep_gen2", "final_gen2_insitu", "final_replicator_insitu"]]
        df = df.merge(asy, on="file", how="left")
        df["emerged_rep"] = df["t_rep"] > 0
    else:
        df["t_rep"] = -1
        df["emerged_rep"] = False
    df.to_csv(os.path.join(out, "runs.csv"), index=False)
    table = summarize(df)
    table.to_csv(os.path.join(out, "cells.csv"), index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(dict.fromkeys(df["label"]))
    tapes = sorted(df["tape"].unique())
    # 1) emergence-fraction heatmaps: one panel per label (steps × k), per tape,
    #    for the pre-registered occupancy measure (tq_10) and the assay measure (t_rep)
    measures = [("frac", "q_share ≥ 10% (pre-registered)", "emergence")]
    if "rep_frac" in table:
        measures.append(("rep_frac", "heritable replicator (assay gen2 ≥ 0.3)", "emergence_rep"))
    for tape in tapes:
      for col, title, fname in measures:
        sub = table[table["tape"] == tape]
        if sub.empty:
            continue
        steps_levels = sorted(sub["steps"].unique())
        k_levels = sorted(sub["k"].unique())
        if len(steps_levels) * len(k_levels) <= 1:
            continue
        fig, axes = plt.subplots(1, len(labels), figsize=(2.6 * len(labels), 2.8), squeeze=False)
        for ax, label in zip(axes[0], labels):
            m = np.full((len(k_levels), len(steps_levels)), np.nan)
            for _, r in sub[sub["label"] == label].iterrows():
                m[k_levels.index(r["k"]), steps_levels.index(r["steps"])] = r[col]
            im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis", origin="lower")
            ax.set_xticks(range(len(steps_levels)), steps_levels)
            if ax is axes[0][0]:
                ax.set_yticks(range(len(k_levels)), [f"1/2^{k}" for k in k_levels])
                ax.set_ylabel("mutation")
            else:
                ax.set_yticks(range(len(k_levels)), [""] * len(k_levels))
            ax.set_xlabel("z80 steps")
            ax.set_title(label, fontsize=9)
            for i in range(len(k_levels)):
                for j in range(len(steps_levels)):
                    if not np.isnan(m[i, j]):
                        ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", color="w" if m[i, j] < 0.6 else "k", fontsize=8)
        fig.suptitle(f"fraction of seeds with a replicator within horizon — {title} — L={tape}")
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
        fig.savefig(os.path.join(out, f"{fname}_L{tape}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
    # 2) KM curves per label at the default (steps=128, k=4) for each tape, and per tape
    for (steps, k), g0 in df.groupby(["steps", "k"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        for label in labels:
            for tape in tapes:
                g = g0[(g0["label"] == label) & (g0["tape"] == tape)]
                if g.empty:
                    continue
                ev = g["emerged_rep"] if "emerged_rep" in g and g["t_rep"].notna().any() else g["emerged"]
                tt = g["t_rep"] if "emerged_rep" in g and g["t_rep"].notna().any() else g["tq_10"]
                times = np.where(ev, tt, g["steps_run"]).astype(float)
                xs, ys = km(times, ev.to_numpy().astype(bool), int(g["horizon"].max()))
                ax.step(xs, ys, where="post", label=f"{label}" + (f" L{tape}" if len(tapes) > 1 else ""))
        ax.set_xscale("log")
        ax.set_xlabel("simulation steps")
        ax.set_ylabel("P(no replicator yet)")
        ax.set_title(f"Kaplan–Meier (assay-based t_rep where available), steps={steps}, mutation 1/2^{k}")
        ax.legend(fontsize=7, ncol=2)
        fig.savefig(os.path.join(out, f"km_st{steps}_k{k}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
    # 3) size axis: median tq_10 vs L per label/steps when several tapes exist
    if len(tapes) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for label in labels:
            for steps in sorted(df["steps"].unique()):
                sub = table[(table["label"] == label) & (table["steps"] == steps)].sort_values("tape")
                if sub.empty:
                    continue
                ax.plot(sub["tape"], sub["frac"], marker="o", label=f"{label} st{steps}")
        ax.set_xlabel("tape length L (bytes)")
        ax.set_ylabel("fraction emerged")
        ax.legend(fontsize=7, ncol=2)
        fig.savefig(os.path.join(out, "size_axis.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
