"""Aggregate a batch of run summaries (+ assays.csv, succession.csv when present) into the
per-cell tables and the standard figures.

    python analyze.py runs/stageA
    python analyze.py runs/stageB --out runs/stageB/analysis

Per (label, tape, steps, k) cell: n, seeds reaching the pre-registered occupancy event
(tq_10) with a Wilson 95% interval, seeds with a heritable replicator (t_rep) and with a
faithful one (t_faith), Kaplan–Meier median time to each event (censored at steps_run; "NR"
when the survival curve never crosses 0.5), the conditional median among emerged seeds
(labelled as such), mechanism classes of the first replicator tapes under the condition's
suppression set, and how many runs were stopped early.

Review 2026-10-07: medians were conditional on emergence and unlabelled; mechanism labels
ignored suppression and were anchored to the first 2% exact-share crossing; the KM figures
had 32 curves per axes; two "size axis" figures showed different measures. All replaced.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd

from algocell_exp.batch import select_summaries
from algocell_exp.isa import mechanisms, parse_patterns, resolve
import figstyle as fs

EVENTS = {"tq_10": ("tq_10", "q_share ≥ 10% (pre-registered)"), "t_rep": ("t_rep", "heritable replicator (assay, gen2 ≥ 0.3)"), "t_faith": ("t_faith", "faithful replicator (assay)")}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def km_curve(times: np.ndarray, events: np.ndarray):
    """Kaplan–Meier survival of 'event not yet happened'. Ties: events before censorings
    (the censored run was still at risk when the event occurred)."""
    order = np.lexsort((~events.astype(bool), times))   # by time, events (True) first
    t, e = times[order], events[order].astype(bool)
    at_risk = len(t)
    S = 1.0
    xs, ys = [], []
    for i in range(len(t)):
        if e[i]:
            S *= 1 - 1 / at_risk
            xs.append(float(t[i]))
            ys.append(S)
        at_risk -= 1
    return np.array(xs), np.array(ys)


def km_median(times: np.ndarray, events: np.ndarray) -> float:
    xs, ys = km_curve(times, events)
    below = np.where(ys <= 0.5)[0]
    return float(xs[below[0]]) if len(below) else float("inf")


def load(d: str) -> pd.DataFrame:
    rows = []
    for p in select_summaries(d):
        s = json.load(open(p))
        patterns = s["suppress"] if isinstance(s["suppress"], list) else parse_patterns(s["suppress"])
        sets = resolve(patterns)
        fe = s.get("first_emergent") or {}
        prov = s.get("provenance") or {}
        rows.append({
            "label": s["label"], "ablation": s["label"].split("@", 1)[0], "arm": s["label"].split("@", 1)[1] if "@" in s["label"] else "nominal",
            "tape": s.get("tape_length", 16), "steps": s["z80_steps"], "k": s["noise_exp"], "seed": s["seed"], "replicate": prov.get("replicate"),
            "horizon": s["horizon"], "steps_run": s["steps_run"], "stopped_early": s["steps_run"] < s["horizon"],
            "t_02": s["t_02"], "t_10": s["t_10"], "tq_10": s["tq_10"], "tq_50": s["tq_50"],
            "t02_tape": fe.get("tape"),
            "t02_mech": "+".join(mechanisms(bytes.fromhex(fe["tape"].replace(" ", "")), sets)) or "-" if fe.get("tape") else None,
            "final_top_share": s["final"]["top_share"], "final_q_share": s["final"].get("q_share"), "final_H": s["final"]["H_species"],
            "final_hoe": s["final"]["hoe"], "final_tape_top1": s["final"]["exemplars"][0]["tape"],
            "wall_s": s["wall_s"], "file": os.path.basename(p),
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (label, tape, steps, k), g in df.groupby(["label", "tape", "steps", "k"]):
        n = len(g)
        row = {"label": label, "tape": tape, "steps": steps, "k": k, "n": n, "horizon": int(g["horizon"].max()),
               "stopped_early": int(g["stopped_early"].sum()), "steps_run_min": int(g["steps_run"].min())}
        for ev in ("tq_10", "t_rep", "t_faith"):
            if ev not in g:
                continue
            t = g[ev].astype(float)
            valid = t.notna()
            emerged = valid & (t > 0)
            ne = int(emerged.sum())
            lo, hi = wilson(ne, int(valid.sum()))
            times = np.where(emerged, t, g["steps_run"]).astype(float)[valid.to_numpy()]
            kmm = km_median(times, emerged.to_numpy()[valid.to_numpy()])
            censored_early = int((valid & ~emerged & g["stopped_early"]).sum())   # event not seen in a run the early stop cut short (informative for t_faith)
            row.update({
                f"{ev}_n": ne, f"{ev}_frac": ne / max(int(valid.sum()), 1), f"{ev}_lo": lo, f"{ev}_hi": hi, f"{ev}_censored_early": censored_early,
                f"{ev}_km_median": kmm,                                             # inf = not reached
                f"{ev}_cond_median": float(t[emerged].median()) if ne else float("nan"),  # among emerged seeds only
                f"{ev}_min": int(t[emerged].min()) if ne else -1,
            })
        if "trep_mechs" in g:
            mech = Counter(g.loc[g["t_rep"] > 0, "trep_mechs"].fillna("-"))
            row["trep_mechanisms"] = ", ".join(f"{m}:{c}" for m, c in mech.most_common())
        mech02 = Counter(g.loc[g["t_02"] > 0, "t02_mech"].dropna())
        row["t02_mechanisms"] = ", ".join(f"{m}:{c}" for m, c in mech02.most_common())
        out.append(row)
    return pd.DataFrame(out).sort_values(["tape", "label", "steps", "k"])


def fmt_km(x: float) -> str:
    return "NR" if not np.isfinite(x) else f"{x:.0f}"


def figures(df: pd.DataFrame, table: pd.DataFrame, out: str) -> None:
    import matplotlib.pyplot as plt

    fs.setup()
    labels = fs.order(df["label"].unique())
    tapes = sorted(df["tape"].unique())
    steps_levels = sorted(df["steps"].unique())
    k_levels = sorted(df["k"].unique())
    ev_cols = [ev for ev in ("tq_10", "t_rep", "t_faith") if f"{ev}_n" in table]

    # F1 — emergence atlas: one row per measure, one panel per ablation, cells = steps × k with "k/n" text.
    for tape in tapes:
        sub = table[table["tape"] == tape]
        if len(steps_levels) <= 1 or len(k_levels) <= 1 or sub.empty:
            continue  # a 1×N grid is a bar chart; size_axis.py draws those
        n_uniform = sub["n"].nunique() == 1
        fig, axes = plt.subplots(len(ev_cols), len(labels), figsize=(max(fs.DOUBLE, 1.1 * len(labels)), 1.35 * len(ev_cols) + 0.6), squeeze=False)
        for r, ev in enumerate(ev_cols):
            for c, label in enumerate(labels):
                ax = axes[r][c]
                m = np.full((len(k_levels), len(steps_levels)), np.nan)
                txt = {}
                for _, row in sub[sub["label"] == label].iterrows():
                    i, j = k_levels.index(row["k"]), steps_levels.index(row["steps"])
                    m[i, j] = row[f"{ev}_frac"]
                    txt[(i, j)] = f"{int(row[f'{ev}_n'])}" if n_uniform else f"{int(row[f'{ev}_n'])}/{int(row['n'])}"
                im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis", origin="lower", aspect="auto")
                for (i, j), s in txt.items():
                    ax.text(j, i, s, ha="center", va="center", color="w" if m[i, j] < 0.6 else "k", fontsize=5.5)
                ax.set_xticks(range(len(steps_levels)), steps_levels if r == len(ev_cols) - 1 else [""] * len(steps_levels))
                ax.set_yticks(range(len(k_levels)), [f"1/2^{k}" for k in k_levels] if c == 0 else [""] * len(k_levels))
                if r == 0:
                    ax.set_title(label, color=fs.color(label))
                if c == 0:
                    ax.set_ylabel(EVENTS[ev][1].split(" (")[0], fontsize=7)
                if r == len(ev_cols) - 1:
                    ax.set_xlabel("Z80 steps")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.01, label="fraction of seeds")
        fig.suptitle(f"Emergence atlas, L = {tape} bytes: seeds with the event within the horizon" + (f" (of n = {int(sub['n'].iloc[0])})" if n_uniform else ""), y=1.02)
        fs.save(fig, os.path.join(out, f"atlas_L{tape}"))

    # F2 — Kaplan–Meier small multiples: panel = (steps, k) [and L when several], colour = ablation, event = t_rep when available.
    ev = "t_rep" if "t_rep" in df else "tq_10"
    for tape in tapes:
        g0 = df[df["tape"] == tape]
        cells = [(s, k) for s in steps_levels for k in k_levels if not g0[(g0["steps"] == s) & (g0["k"] == k)].empty]
        if not cells:
            continue
        ncol = min(3, len(cells))
        nrow = math.ceil(len(cells) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(fs.DOUBLE if ncol > 1 else fs.SINGLE, 1.9 * nrow + 0.4), squeeze=False, sharex=True, sharey=True)
        for ax, (s, k) in zip(axes.ravel(), cells):
            g1 = g0[(g0["steps"] == s) & (g0["k"] == k)]
            for label in labels:
                g = g1[g1["label"] == label]
                if g.empty:
                    continue
                t = g[ev].astype(float)
                emerged = (t > 0).to_numpy()
                times = np.where(emerged, t, g["steps_run"]).astype(float)
                xs, ys = km_curve(times, emerged)
                x0 = float(min(times.min(), 500))
                xs_plot = np.concatenate([[x0], xs, [float(times.max())]])
                ys_plot = np.concatenate([[1.0], ys, [ys[-1] if len(ys) else 1.0]])
                ax.step(xs_plot, ys_plot, where="post", color=fs.color(label), label=label)
                cens = times[~emerged]
                if len(cens):
                    # censor ticks at the survival level reached by then
                    lev = [ys[xs <= c][-1] if (xs <= c).any() else 1.0 for c in cens]
                    ax.plot(cens, lev, "|", color=fs.color(label), markersize=4, alpha=0.8)
            ax.set_xscale("log")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{s} steps · mutation 1/2^{k}")
        for ax in axes.ravel()[len(cells):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("simulation steps")
        for ax in axes[:, 0]:
            ax.set_ylabel("P(no replicator yet)")
        handles, lab = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, lab, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False, title=f"ablation · event: {EVENTS[ev][1]}")
        fig.suptitle(f"Time to first replicator, L = {tape} bytes (Kaplan–Meier; ticks = censored at the run's last step)", y=1.02)
        fs.save(fig, os.path.join(out, f"km_L{tape}"))


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
    assays_path = os.path.join(out, "assays.csv")
    if os.path.exists(assays_path):
        asy = pd.read_csv(assays_path)
        keep = [c for c in asy.columns if c == "file" or c.startswith(("t_rep", "trep_", "t_faith", "tfaith_", "em_", "final_"))]
        assert not asy.duplicated(["label", "tape_len", "steps", "k", "seed", "replicate"]).any(), "a (label, L, steps, k, seed, replicate) appears twice in assays.csv"
        df = df.merge(asy[keep], on="file", how="left", validate="one_to_one")
        unmatched = df["t_rep"].isna().sum()
        if unmatched:
            print(f"warning: {unmatched} runs have no assay row (t_rep NaN, excluded from assay-based fractions)")
    df.to_csv(os.path.join(out, "runs.csv"), index=False)
    table = summarize(df)
    table.to_csv(os.path.join(out, "cells.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    show = ["label", "tape", "steps", "k", "n", "stopped_early", "tq_10_n", "tq_10_km_median"]
    for ev in ("t_rep", "t_faith"):
        if f"{ev}_n" in table:
            show += [f"{ev}_n", f"{ev}_km_median", f"{ev}_cond_median"]
    if "trep_mechanisms" in table:
        show.append("trep_mechanisms")
    print(table[show].to_string(index=False, float_format=lambda x: "NR" if x == float("inf") else f"{x:.0f}"))
    figures(df, table, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
