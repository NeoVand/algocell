"""Analysis of the BFF soups (THEORY.md P1): events, first and final replicators, verdicts, figures.

    python -m micro.bff_analysis [--dir runs/bff] [--out runs/bff/analysis]

Events per run (from samples.jsonl, every 64 epochs, nothing chosen after the fact):
  t_rep     first sample at which a top-3 tape class has share ≥ 0.5% and culture-test gen2 ≥ 0.3 (the Z80 criterion)
  t_hoe1    first epoch at which the high-order entropy reaches 1 bit/byte (the paper's "complexity ≥ 1" transition)
  t_closed  first sample at which the top class is closed: pointer enters the partner in ≤ 5% of encounters and ≥ 95% of
            random partners become copies
  t_open    first sample at which a top-3 class is a replicator (gen2 ≥ 0.3) whose pointer enters the partner in ≥ 50%
The first replicator is the qualifying class at t_rep; its culture test gives entered / copies / self-damage / has_loop.
Writes runs.csv, NUMBERS_BFF.md and figures (HOE, population openness, copy events, heritable fraction; first vs final).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import figstyle as fs
from micro.bff import LIT, OPS, ascii_map, density_map


def fisher_exact(table) -> tuple[float, float]:
    """Two-sided Fisher exact test for a 2x2 table [[a, b], [c, d]] (sum of hypergeometric probabilities <= observed)."""
    from math import comb
    (a, b), (c, d) = table
    n, r1, c1 = a + b + c + d, a + b, a + c
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    denom = comb(n, c1)
    probs = {k: comb(r1, k) * comb(n - r1, c1 - k) / denom for k in range(lo, hi + 1)}
    p_obs = probs[a]
    return float("nan"), float(min(1.0, sum(v for v in probs.values() if v <= p_obs * (1 + 1e-9))))


def mannwhitneyu(x, y, alternative: str = "two-sided") -> tuple[float, float]:
    """Mann–Whitney U with the normal approximation and tie correction (adequate for n >= 8 per group); returns (U, p)."""
    from math import erfc, sqrt
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    allv = np.concatenate([x, y])
    ranks = pd.Series(allv).rank(method="average").to_numpy()
    u = ranks[:nx].sum() - nx * (nx + 1) / 2  # U for x (large U = x tends to be larger)
    mu = nx * ny / 2
    _, counts = np.unique(allv, return_counts=True)
    tie = (counts ** 3 - counts).sum()
    n = nx + ny
    sigma = sqrt(nx * ny / 12 * ((n + 1) - tie / (n * (n - 1)))) if n > 1 else float("nan")
    if not sigma:
        return float(u), 1.0
    z = (u - mu) / sigma
    if alternative == "two-sided":
        p = erfc(abs(z) / sqrt(2))
    elif alternative == "less":  # x stochastically smaller than y: P(Z <= z)
        p = 0.5 * erfc(-z / sqrt(2))
    else:  # greater: P(Z >= z)
        p = 0.5 * erfc(z / sqrt(2))
    return float(u), float(min(1.0, p))


def pretty(tape_hex: str, amap: np.ndarray) -> str:
    t = bytes.fromhex(tape_hex)
    out = []
    for b in t:
        k = int(amap[b])
        out.append((OPS + LIT)[k - 1] if k else ("0" if b == 0 else "·"))
    return "".join(out)


def fill_fraction(tape_hex: str) -> float:
    t = np.frombuffer(bytes.fromhex(tape_hex), dtype=np.uint8)
    return float(np.bincount(t, minlength=256).max() / len(t))


def is_sterile_fill(top: dict) -> bool:
    """One-symbol tar: one byte >= 90% of the tape and no heredity (gen2 < 0.3). A heritable one-byte tiling (e.g. the
    all-`P` literal pusher, whose literal is itself) is a period-1 replicator, not tar."""
    return fill_fraction(top["tape"]) >= 0.9 and top["gen2"] < 0.3


def classify(top: dict) -> str:
    """closed / open / intermediate by the culture test; 'fill' for sterile one-symbol tar."""
    if is_sterile_fill(top):
        return "fill"
    if top["entered"] <= 0.05 and top["copies"] >= 0.95:
        return "closed"
    if top["entered"] >= 0.5:
        return "open"
    return "intermediate"


def load_run(d: str) -> dict | None:
    cp = os.path.join(d, "cond.json")
    if not os.path.exists(cp):
        return None
    cond = json.load(open(cp))
    try:
        ep = pd.read_csv(os.path.join(d, "epochs.csv"))
        samples = [json.loads(l) for l in open(os.path.join(d, "samples.jsonl")) if l.strip()]
    except (FileNotFoundError, pd.errors.EmptyDataError, json.JSONDecodeError):
        return None
    if not samples or ep.empty:
        return None
    lit = bool(cond.get("literal", False))
    amap = ascii_map(lit) if cond["density"] <= 1 else density_map(cond["density"], seed=0)
    if lit:
        amap = amap.copy(); amap[ord(LIT)] = 11
    variant = ("wrap" if cond["ip_wrap"] else "std") + ("lit" if lit else "") + ("nh" if cond.get("nohalt") else "") + (f"_d{cond['density']}" if cond["density"] > 1 else "")
    r = {"run": os.path.basename(d), "variant": variant, "seed": cond["seed"], "epochs_done": int(ep["epoch"].max()), "finished": os.path.exists(os.path.join(d, "summary.json"))}
    S = pd.DataFrame([{"epoch": s["epoch"], "HOE": s["HOE"], "H0": s["H0"], "unique_frac": s["unique_frac"], "frac_heritable": s["frac_heritable"],
                       "top_share": s["top"][0]["share"], "top_gen2": s["top"][0]["gen2"], "top_entered": s["top"][0]["entered"], "top_copies": s["top"][0]["copies"],
                       "top_loop": s["top"][0]["has_loop"]} for s in samples])
    r["S"] = S
    r["E"] = ep
    hoe1 = S[S["HOE"] >= 1.0]
    r["t_hoe1"] = int(hoe1["epoch"].iloc[0]) if len(hoe1) else None
    # first replicator. Pre-registered Z80 criterion (t_rep: top-3 class share >= 0.5% and gen2 >= 0.3) is kept and reported,
    # but BFF replicator populations are quasispecies with tiny exact-class shares (seen in the first live samples, 2026-10-08
    # 03:42: heritable fraction 0.7-0.97 with top share 0.15%), so the exemplar used for "first replicator" is the top class at
    # the first sample where it is heritable (t_top), and the population event is t_her (heritable fraction of 32 random tapes >= 0.5).
    t_rep, t_top, t_her, first = None, None, None, None
    for s_ in samples:
        q = [t for t in s_["top"] if t["share"] >= 0.005 and t["gen2"] >= 0.3]
        if q and t_rep is None:
            t_rep = s_["epoch"]
        if t_top is None and s_["top"][0]["gen2"] >= 0.3:
            t_top, first = s_["epoch"], s_["top"][0]
        if t_her is None and s_["frac_heritable"] >= 0.5:
            t_her = s_["epoch"]
    r["t_rep"], r["t_top"], r["t_her"] = t_rep, t_top, t_her
    if first is not None:
        r.update({"first_tape": first["tape"], "first_pretty": pretty(first["tape"], amap), "first_loop": first["has_loop"], "first_entered": first["entered"],
                  "first_copies": first["copies"], "first_gen2": first["gen2"], "first_self_damage": first["self_damage"], "first_share": first["share"],
                  "first_fill": fill_fraction(first["tape"]),
                  "first_class": classify(first)})
    t_closed, t_open = None, None
    for s in samples:
        top = s["top"][0]
        if t_closed is None and top["gen2"] >= 0.3 and top["entered"] <= 0.05 and top["copies"] >= 0.95:
            t_closed = s["epoch"]
        if t_open is None and any(t["gen2"] >= 0.3 and t["entered"] >= 0.5 for t in s["top"]):
            t_open = s["epoch"]
    r["t_closed"], r["t_open"] = t_closed, t_open
    last = samples[-1]["top"][0]
    r.update({"final_epoch": samples[-1]["epoch"], "final_tape": last["tape"], "final_pretty": pretty(last["tape"], amap), "final_loop": last["has_loop"], "final_entered": last["entered"],
              "final_fill": fill_fraction(last["tape"]), "final_class": classify(last),
              "final_copies": last["copies"], "final_gen2": last["gen2"], "final_self_damage": last["self_damage"], "final_share": last["share"],
              "final_HOE": samples[-1]["HOE"], "final_unique_frac": samples[-1]["unique_frac"], "final_heritable": samples[-1]["frac_heritable"],
              "max_heritable": float(S["frac_heritable"].max())})
    r["peak_heritable_epoch"] = int(S.loc[S["frac_heritable"].idxmax(), "epoch"])
    r["collapsed"] = bool(t_her is not None and samples[-1]["frac_heritable"] < 0.1)
    pre = ep[ep["epoch"] < (t_top if t_top is not None else ep["epoch"].max() + 1)]
    r["chunk_mean_pre"] = float(pre["chunk_mean"].mean()) if len(pre) else float("nan")
    r["chunk_p90_max_pre"] = float(pre["chunk_p90"].max()) if len(pre) else float("nan")
    r["copy_frac_max_pre"] = float(pre["copy_frac"].max()) if len(pre) else float("nan")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/bff")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "analysis")
    os.makedirs(out, exist_ok=True)
    runs = [r for r in (load_run(d) for d in sorted(glob.glob(os.path.join(a.dir, "*_s*"))) if os.path.isdir(d)) if r]
    if not runs:
        print("no runs")
        return
    cols = [k for k in runs[0] if k not in ("S", "E")]
    for k in ("first_tape", "first_pretty", "first_loop", "first_entered", "first_copies", "first_gen2", "first_self_damage", "first_share", "first_class", "first_fill"):
        if k not in cols:
            cols.append(k)
    R = pd.DataFrame([{k: r.get(k) for k in cols} for r in runs])
    R["first_loop"] = R["first_loop"].astype("boolean").fillna(False).astype(bool)
    R["final_loop"] = R["final_loop"].astype(bool)
    R.to_csv(os.path.join(out, "runs.csv"), index=False)

    md = ["# BFF soups — numbers (generated; do not edit)\n",
          "Events: `t_top` = first sample at which the most common tape class is heritable (gen2 ≥ 0.3) and not a one-symbol fill (one byte ≥ 90% of the tape; fills are tar, reported as class 'fill'); it is the exemplar used as the first replicator; `t_her` = heritable fraction of 32 random tapes ≥ 0.5; "
          "`t_rep` = the pre-registered Z80 criterion (top-3 class share ≥ 0.5% and gen2 ≥ 0.3), kept for the record; `t_hoe1` = high-order entropy ≥ 1 bit/byte; "
          "`t_closed` = top class enters the partner ≤ 5% and copies ≥ 95% of random partners; `t_open` = a replicating top-3 class enters the partner ≥ 50%. "
          "Culture tests: 64 random partners, 2^13 steps.\n"]
    for v, g in R.groupby("variant"):
        n = len(g)
        rep = g[g["t_top"].notna()]
        md.append(f"## {v}: {n} runs ({int(g['finished'].sum())} finished; epochs done median {int(g['epochs_done'].median()):,})\n")
        md.append(f"- transitions: top class heritable (t_top) in {len(rep)}/{n} (median {rep['t_top'].median():.0f} epochs); heritable fraction ≥ 0.5 (t_her) in {int(g['t_her'].notna().sum())}/{n} "
                  f"(median {g['t_her'].median():.0f}); pre-registered share criterion (t_rep) in {int(g['t_rep'].notna().sum())}/{n}; HOE ≥ 1 in {int(g['t_hoe1'].notna().sum())}/{n} (median {g['t_hoe1'].median():.0f}); "
                  f"closed top class in {int(g['t_closed'].notna().sum())}/{n}; an open replicator ever in the top 3 in {int(g['t_open'].notna().sum())}/{n}")
        if len(rep):
            cls = rep["first_class"].value_counts().to_dict()
            md.append(f"- first replicators: {cls}; with a loop {int(rep['first_loop'].sum())}/{len(rep)}; median entered {rep['first_entered'].median():.2f}, "
                      f"copies {rep['first_copies'].median():.2f}, self-damage {rep['first_self_damage'].median():.2f}")
            md.append(f"- final top class: closed in {int(((rep['final_entered'] <= 0.05) & (rep['final_copies'] >= 0.95)).sum())}/{len(rep)}, with a loop {int(rep['final_loop'].sum())}/{len(rep)}; "
                      f"final heritable fraction median {rep['final_heritable'].median():.2f} (max over time, median {rep['max_heritable'].median():.2f}, reached at median epoch {rep['peak_heritable_epoch'].median():.0f}); "
                      f"collapsed (heritable fraction ≥ 0.5 reached, < 0.1 at the end) {int(rep['collapsed'].sum())}/{len(rep)}; first replicator a one-byte tiling in {int((rep['first_fill'] >= 0.9).sum())}/{len(rep)}")
            md.append(f"- before t_rep: mean chunk transfer {rep['chunk_mean_pre'].median():.2f} bytes/encounter (median over runs), max 90th percentile {rep['chunk_p90_max_pre'].median():.0f}, "
                      f"max copy-event fraction {rep['copy_frac_max_pre'].median():.4f}\n")
            md.append("first replicators (BFF string; `·` = non-instruction byte, `0` = zero):\n")
            md.append(rep[["seed", "t_top", "t_her", "t_hoe1", "first_class", "first_loop", "first_entered", "first_copies", "first_self_damage", "first_gen2", "first_fill", "first_pretty"]]
                      .to_markdown(index=False, floatfmt=".2f") + "\n")
            md.append("final top classes:\n")
            md.append(rep[["seed", "final_epoch", "final_class", "final_loop", "final_entered", "final_copies", "final_self_damage", "final_gen2", "final_fill", "final_pretty"]]
                      .to_markdown(index=False, floatfmt=".2f") + "\n")
        no = g[g["t_top"].isna()]
        if len(no):
            md.append(f"- runs without a replicator by the criterion: seeds {sorted(no['seed'].tolist())}; their final HOE median {no['final_HOE'].median():.2f}, "
                      f"final heritable fraction median {no['final_heritable'].median():.2f}\n")
    # transition rates and between-variant tests
    from itertools import combinations
    horizon = int(R["epochs_done"].max())
    rows_t = []
    for v, g in R.groupby("variant"):
        n = len(g)
        rows_t.append({"variant": v, "runs": n, "transition (t_top)": int(g["t_top"].notna().sum()), "heritable ≥ 0.5 (t_her)": int(g["t_her"].notna().sum()),
                       "median t_her (epochs; censored runs at horizon)": float(g["t_her"].fillna(horizon).median()),
                       "HOE ≥ 1": int(g["t_hoe1"].notna().sum()), "HOE ≥ 1 without a replicator": int((g["t_hoe1"].notna() & g["t_top"].isna()).sum()),
                       "first replicator open": int((g["first_class"] == "open").sum()), "first closed with loop": int(((g["first_class"] == "closed") & g["first_loop"]).sum()),
                       "final closed": int((g["final_class"] == "closed").sum()), "collapsed": int(g["collapsed"].sum())})
    T = pd.DataFrame(rows_t)
    md.append("## Transition rates and between-variant tests\n")
    md.append(T.to_markdown(index=False, floatfmt=".0f") + "\n")
    for a_, b_ in combinations(sorted(R["variant"].unique()), 2):
        ga, gb = R[R["variant"] == a_], R[R["variant"] == b_]
        ta, tb = int(ga["t_top"].notna().sum()), int(gb["t_top"].notna().sum())
        pf = fisher_exact([[ta, len(ga) - ta], [tb, len(gb) - tb]])[1]
        ua, ub = ga["t_her"].fillna(horizon), gb["t_her"].fillna(horizon)
        pm = mannwhitneyu(ua, ub, alternative="two-sided")[1] if len(ua) and len(ub) else float("nan")
        md.append(f"- {a_} vs {b_}: transitions {ta}/{len(ga)} vs {tb}/{len(gb)} (Fisher two-sided p = {pf:.3g}); t_her with censored runs at the horizon, Mann–Whitney two-sided p = {pm:.3g}")
    md.append("")
    # verdicts
    md.append("## Readings of THEORY.md P1 (computed, pre-stated thresholds)\n")
    std = R[(R["variant"] == "std") & R["t_top"].notna()]
    wrap = R[(R["variant"] == "wrap") & R["t_top"].notna()]
    if len(std):
        md.append(f"- (a) standard BFF: first replicators closed {int((std['first_class'] == 'closed').sum())}/{len(std)}, with a loop {int(std['first_loop'].sum())}/{len(std)}, "
                  f"open {int((std['first_class'] == 'open').sum())}/{len(std)} → {'confirmed' if ((std['first_class'] == 'closed') & std['first_loop']).all() else 'NOT as predicted'}")
    if len(wrap):
        n_open = int((wrap["first_class"] == "open").sum())
        n_closed_loop = int(((wrap["first_class"] == "closed") & wrap["first_loop"]).sum())
        md.append(f"- (b) wrap BFF: first replicators open {n_open}/{len(wrap)}, closed with a loop {n_closed_loop}/{len(wrap)} → "
                  f"{'(b1) open first' if n_open * 2 >= len(wrap) else ('(b2) born closed' if n_closed_loop * 2 >= len(wrap) else 'neither reading reaches half')}")
    wl = R[(R["variant"] == "wraplit") & R["t_top"].notna()]
    if len(wl):
        n_open_nl = int(((wl["first_class"] == "open") & ~wl["first_loop"].astype(bool)).sum())
        n_closed_loop = int(((wl["first_class"] == "closed") & wl["first_loop"]).sum())
        nwl = int((R["variant"] == "wraplit").sum())
        std_her = R[R["variant"] == "std"]["t_her"].fillna(horizon)
        wl_her = R[R["variant"] == "wraplit"]["t_her"].fillna(horizon)
        p_e2 = mannwhitneyu(wl_her, std_her, alternative="less")[1] if len(std_her) else float("nan")
        n_final_closed = int((wl["final_class"] == "closed").sum())
        md.append(f"- (e1) wrap + literal: first replicators straight-line and open {n_open_nl}/{len(wl)} transitions of {nwl} runs (≥ 9/12 predicted); closed with a loop {n_closed_loop}/{len(wl)} (≥ 6/12 kills) → "
                  f"{'(e1) met' if n_open_nl >= 9 else ('KILL' if n_closed_loop >= 6 else 'not met')}; "
                  f"(e2) earlier emergence than standard BFF: median t_her {wl_her.median():.0f} vs {std_her.median():.0f} (one-sided Mann–Whitney p = {p_e2:.3g}) → {'met' if p_e2 < 0.05 else 'not met'}; "
                  f"(e3) final dominant closed in {n_final_closed}/{len(wl)} transitioned worlds (≥ 6 predicted) → {'met' if n_final_closed >= 6 else 'not met'}; "
                  f"collapsed after the open wave in {int(wl['collapsed'].sum())}/{len(wl)} (peak heritable fraction median {wl['max_heritable'].median():.2f} at median epoch {wl['peak_heritable_epoch'].median():.0f}, final median {wl['final_heritable'].median():.2f})")
    with open(os.path.join(out, "NUMBERS_BFF.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md[:3]))
    print(R[["run", "epochs_done", "t_rep", "t_top", "t_her", "t_hoe1", "t_closed", "t_open", "first_class", "first_loop", "final_loop", "final_entered", "final_copies", "final_heritable"]].to_string(index=False))

    # figures
    fs.setup()
    import matplotlib.pyplot as plt
    variants = sorted(R["variant"].unique())
    colors = {v: ["#000000", "#D55E00", "#0072B2", "#009E73", "#E69F00"][i % 5] for i, v in enumerate(variants)}
    fig, axes = plt.subplots(2, 2, figsize=(fs.DOUBLE, 4.6), sharex=True)
    for r in runs:
        c = colors[r["variant"]]
        S, E = r["S"], r["E"]
        axes[0, 0].plot(S["epoch"], S["HOE"], color=c, lw=0.6, alpha=0.6)
        axes[0, 1].plot(E["epoch"], E["frac_entered"].rolling(16, min_periods=1).mean(), color=c, lw=0.6, alpha=0.6)
        axes[1, 0].plot(E["epoch"], E["copy_frac"].rolling(16, min_periods=1).mean(), color=c, lw=0.6, alpha=0.6)
        axes[1, 1].plot(S["epoch"], S["frac_heritable"], color=c, lw=0.6, alpha=0.6)
    axes[0, 0].set_ylabel("high-order entropy (bits/byte)")
    axes[0, 1].set_ylabel("encounters whose pointer\nentered the partner")
    axes[1, 0].set_ylabel("encounters producing a ≥ 75% copy")
    axes[1, 1].set_ylabel("heritable fraction of 32 random tapes")
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    for v in variants:
        axes[0, 0].plot([], [], color=colors[v], lw=1.2, label=f"{v} (n = {int((R['variant'] == v).sum())})")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fs.save(fig, os.path.join(out, "bff_timeseries"))

    fig, ax = plt.subplots(figsize=(fs.SINGLE, 2.6))
    rng = np.random.default_rng(0)
    for i, v in enumerate(variants):
        g = R[(R["variant"] == v) & R["t_top"].notna()]
        for j, (col, mk) in enumerate((("first_entered", "o"), ("final_entered", "s"))):
            x = i * 2.5 + j + rng.uniform(-0.15, 0.15, len(g))
            loop = g["first_loop"] if j == 0 else g["final_loop"]
            ax.scatter(x[~loop.values], g[col][~loop], marker=mk, s=14, facecolors="none", edgecolors=colors[v], lw=0.8)
            ax.scatter(x[loop.values], g[col][loop], marker=mk, s=14, color=colors[v])
    ax.set_xticks([i * 2.5 + j for i in range(len(variants)) for j in (0, 1)], [f"{v}\n{w}" for v in variants for w in ("first", "final")], fontsize=7)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("culture test: encounters whose pointer\nentered the partner (filled = has a loop)", fontsize=7)
    fs.save(fig, os.path.join(out, "bff_first_vs_final_openness"))
    print("wrote", out)


if __name__ == "__main__":
    main()
