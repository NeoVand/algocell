"""Stage F — ring arithmetic (PLAN.md F1–F3) and the dead-zone ring test (F4).

    python stage_f.py runs/stageF [--ref runs/stageE] [--out runs/stageF/analysis/stage_f]

The P = 2L reference rows are Stage E's @nominal runs (same seeds, same physics): `none@nominal` and
`stack-write-only@nominal` at L ∈ {8, 10, 12, 16, 36}. Writes NUMBERS_F.md + figures:
  F1  periods of the tiled first replicators under stack-write-only per (L, P): divides P / divides 2L / divides L
  F2  prime rings (P = 37, 73, 79): tiled LDIR replicators present?; KM median vs the P = 2L reference
  F3  control: `none` KM median per P vs P = 2L (prediction: within one 50-step sample)
  F4  `none` at L = 12 (P = 28, 36), L = 10 (P = 28, 40), L = 8 (P = 32): heritable count, KM median, modal period vs Stage E
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import figstyle as fs
from analyze import km_median, wilson
from report import km_str


def cell(g: pd.DataFrame) -> dict:
    n = len(g)
    t = g["t_rep"].astype(float)
    em = t > 0
    ne = int(em.sum())
    lo, hi = wilson(ne, n)
    times = np.where(em, t, g["steps_run"]).astype(float)
    rep = g[em]
    tiled = rep[rep["trep_period"] <= rep["tape_len"] / 2]
    P = int(g["mem_length"].iloc[0]) if "mem_length" in g and g["mem_length"].notna().any() else int(2 * g["tape_len"].iloc[0])
    per = tiled["trep_period"].astype(int)
    return {"n": n, "P": P, "t_rep_n": ne, "t_rep_lo": lo, "t_rep_hi": hi, "t_rep_km": km_median(times, em.to_numpy()), "t_faith_n": int((g["t_faith"] > 0).sum()),
            "tiled": len(tiled), "div_P": float((P % per == 0).mean()) if len(per) else np.nan, "div_2L": float(((2 * tiled["tape_len"]) % per == 0).mean()) if len(per) else np.nan,
            "div_L": float((tiled["tape_len"] % per == 0).mean()) if len(per) else np.nan,
            "periods": ", ".join(f"{int(p)}×{c}" for p, c in rep["trep_period"].value_counts().sort_index().items()) or "–",
            "period_mode": float(rep["trep_period"].mode().iloc[0]) if len(rep) else np.nan,
            "ldir_first": int(rep["trep_mechs"].astype(str).str.contains("block-copy").sum()) if len(rep) else 0,
            "func_rnd_median": float(g["final_func_rnd"].median())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ref", default="runs/stageE")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "analysis", "stage_f")
    os.makedirs(out, exist_ok=True)
    f = pd.read_csv(os.path.join(a.dir, "analysis", "assays.csv"))
    if "mem_length" not in f:
        raise SystemExit("assays.csv lacks mem_length; re-run assay_batch with the ring-aware version")
    e = pd.read_csv(os.path.join(a.ref, "analysis", "assays.csv"))
    e = e[e["label"].isin(["none@nominal", "stack-write-only@nominal"]) & e["tape_len"].isin([8, 10, 12, 16, 36]) & e["replicate"].isna()].copy()
    e["mem_length"] = 2 * e["tape_len"]
    e["source"] = "stageE"
    f["source"] = "stageF"
    f["ablation"] = f["label"].str.split("@").str[0]
    e["ablation"] = e["label"].str.split("@").str[0]
    both = pd.concat([f, e], ignore_index=True)
    rows = []
    for (abl, L, P), g in both.groupby(["ablation", "tape_len", "mem_length"]):
        rows.append({"ablation": abl, "L": int(L), **cell(g), "source": g["source"].iloc[0]})
    tab = pd.DataFrame(rows).sort_values(["ablation", "L", "P"])
    tab.to_csv(os.path.join(out, "rings.csv"), index=False)
    md = ["# Stage F numbers (generated)\n", "## All ring cells (P = 2L rows from Stage E @nominal)\n\n" + tab[["ablation", "L", "P", "source", "n", "t_rep_n", "t_rep_km", "t_faith_n", "tiled", "div_P", "div_2L", "div_L", "periods", "ldir_first", "func_rnd_median"]].to_markdown(index=False, floatfmt=".2f") + "\n"]

    # F1 / F2
    sw = tab[tab["ablation"] == "stack-write-only"]
    lines = ["## F1 — periods of tiled first replicators under stack-write-only: fraction dividing P (prediction ≥ 0.9 at composite P), and the periods seen\n"]
    for _, r in sw.iterrows():
        lines.append(f"- L = {r['L']}, P = {r['P']} ({r['source']}): {r['t_rep_n']}/{r['n']} heritable, {r['tiled']} tiled; divides P {r['div_P']:.2f}, divides 2L {r['div_2L']:.2f}, divides L {r['div_L']:.2f}; periods {r['periods']}; LDIR-first {r['ldir_first']}")
    md.append("\n".join(lines) + "\n")
    lines = ["## F2 — prime rings (P = 37, 73, 79): tiled LDIR replicators and KM median vs P = 2L\n"]
    for L, primes in ((16, (37,)), (36, (73, 79))):
        ref = sw[(sw["L"] == L) & (sw["P"] == 2 * L)]
        for P in primes:
            r = sw[(sw["L"] == L) & (sw["P"] == P)]
            if len(r) and len(ref):
                r, ref0 = r.iloc[0], ref.iloc[0]
                ratio = r["t_rep_km"] / ref0["t_rep_km"] if np.isfinite(ref0["t_rep_km"]) and np.isfinite(r["t_rep_km"]) else np.nan
                lines.append(f"- L = {L}, P = {P}: {r['t_rep_n']}/{r['n']} heritable, tiled {r['tiled']}, LDIR-first {r['ldir_first']}, KM {km_str(r['t_rep_km'])} vs {km_str(ref0['t_rep_km'])} at P = {2 * L} (ratio {ratio:.1f}); periods {r['periods']}")
    md.append("\n".join(lines) + "\n")
    # F3
    nn = tab[tab["ablation"] == "none"]
    lines = ["## F3 — control: `none` KM median per P (prediction: within one 50-step sample of P = 2L)\n"]
    for L in (16, 36):
        ref = nn[(nn["L"] == L) & (nn["P"] == 2 * L)]
        for _, r in nn[nn["L"] == L].iterrows():
            d = r["t_rep_km"] - ref.iloc[0]["t_rep_km"] if len(ref) else np.nan
            lines.append(f"- L = {L}, P = {r['P']} ({r['source']}): {r['t_rep_n']}/{r['n']}, KM {km_str(r['t_rep_km'])} (Δ vs P = 2L: {d:+.0f} steps); modal period {r['period_mode']:.0f}; periods {r['periods']}")
    md.append("\n".join(lines) + "\n")
    # F4
    lines = ["## F4 — the dead zone on padded rings (`none`, L = 8, 10, 12)\n"]
    for L in (8, 10, 12):
        for _, r in nn[nn["L"] == L].iterrows():
            lines.append(f"- L = {L}, P = {r['P']} ({r['source']}): {r['t_rep_n']}/{r['n']} heritable (Wilson {r['t_rep_lo']:.2f}–{r['t_rep_hi']:.2f}), KM {km_str(r['t_rep_km'])}, faithful {r['t_faith_n']}; modal period {r['period_mode']:.0f}; periods {r['periods']}; LDIR-first {r['ldir_first']}; final random-cell replicator fraction {r['func_rnd_median']:.2f}")
    md.append("\n".join(lines) + "\n")

    fs.setup()
    import matplotlib.pyplot as plt
    # Figure F1: period vs P scatter for stack-write-only at L = 16 and 36
    fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE, 2.6))
    rng = np.random.default_rng(0)
    for ax, L in zip(axes, (16, 36)):
        g = both[(both["ablation"] == "stack-write-only") & (both["tape_len"] == L) & (both["t_rep"] > 0)]
        Ps = sorted(g["mem_length"].unique())
        for P in Ps:
            gg = g[g["mem_length"] == P]
            per = gg["trep_period"].to_numpy(float)
            x = P + rng.normal(0, 0.08, len(gg))
            divP = (P % np.maximum(per, 1).astype(int) == 0)
            ax.scatter(x[divP], per[divP], s=14, color=fs.color("stack-write-only"), lw=0, zorder=3)
            ax.scatter(x[~divP], per[~divP], s=18, facecolors="none", edgecolors=fs.color("stack-write-only"), lw=0.8, zorder=3)
            divs = [d for d in range(2, P + 1) if P % d == 0 and d <= L]
            ax.scatter([P] * len(divs), divs, marker="_", s=120, color="#bbbbbb", zorder=1)
        ax.set_xticks(Ps, [str(int(p)) for p in Ps])
        ax.set_yscale("log")
        ax.set_yticks([2, 3, 5, 10, 20, 50], ["2", "3", "5", "10", "20", "50"])
        ax.minorticks_off()
        ax.set_xlabel(f"ring length P (L = {L})")
        ax.set_title(f"stack-write-only, L = {L}", color=fs.color("stack-write-only"))
    axes[0].set_ylabel("period of first replicator (bytes)")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], marker="o", color=fs.color("stack-write-only"), ls="none", ms=4), Line2D([], [], marker="o", color=fs.color("stack-write-only"), ls="none", mfc="none", ms=4), Line2D([], [], marker="_", color="#bbbbbb", ls="none", ms=10)],
               ["period divides P", "does not divide P", "divisors of P (≤ L)"], loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False)
    fs.save(fig, os.path.join(out, "F1_periods"))
    # Figure F4: emergence fraction and KM per P at L = 8, 10, 12, 16
    fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE, 2.6))
    for L, m in ((8, "o"), (10, "s"), (12, "^"), (16, "D")):
        s = nn[nn["L"] == L].sort_values("P")
        if s.empty:
            continue
        axes[0].errorbar(s["P"] / (2 * L), s["t_rep_n"] / s["n"], yerr=[s["t_rep_n"] / s["n"] - s["t_rep_lo"], s["t_rep_hi"] - s["t_rep_n"] / s["n"]], marker=m, ls="none", capsize=2, label=f"L = {L}")
        ok = np.isfinite(s["t_rep_km"])
        axes[1].plot(s["P"][ok] / (2 * L), s["t_rep_km"][ok], marker=m, ls="none", label=f"L = {L}")
        axes[1].plot(s["P"][~ok] / (2 * L), [400_000] * int((~ok).sum()), marker="^", mfc="white", ls="none", color="k")
    axes[0].set_ylabel("seeds with a heritable replicator (of 10)")
    axes[1].set_ylabel("KM median steps (△ not reached)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("ring length P / 2L")
        ax.axvline(1, color="#bbbbbb", lw=0.6, ls=":")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False, title="`none`, 128 steps, 1/16")
    fs.save(fig, os.path.join(out, "F4_deadzone"))

    # ── mechanism trace: one pusher encounter on the nominal and padded rings, byte by byte ──
    from algocell_exp.assay import execute_pairs, assay
    rng = np.random.default_rng(0)
    trace = []
    for L, Ps in ((8, (16, 32)), (10, (20, 28, 40)), (12, (24, 28, 36)), (16, (32, 33, 34, 35, 37))):
        unit = np.resize(np.frombuffer(bytes.fromhex("01c5"), dtype=np.uint8), L)
        Bs = rng.integers(0, 256, size=(8, L), dtype=np.uint8)           # the same 8 random partners for every P
        for P in Ps:
            r = assay(unit, z80_steps=128, suppress=[], n=64, seed=0, mem_length=None if P == 2 * L else P)
            for steps in (8, 16, 32, 64, 128):
                pairs = np.concatenate([np.repeat(unit[None], 8, 0), Bs], axis=1)
                res = np.asarray(execute_pairs(pairs, L, steps, [], None if P == 2 * L else P)).reshape(8, -1)
                A2, B2 = res[:, :L], res[:, L : 2 * L]
                trace.append({"L": L, "P": P, "steps": steps, "A_intact_frac": float((A2 == unit).mean()), "B_copy_frac": float((B2 == unit).mean()),
                              "B_last_byte_copied": float((B2[:, -1] == unit[-1]).mean()), "gen2_isolated": round(r["gen2_score"], 2)})
    tr = pd.DataFrame(trace)
    tr.to_csv(os.path.join(out, "pusher_trace.csv"), index=False)
    piv = tr.pivot_table(index=["L", "P", "gen2_isolated"], columns="steps", values=["A_intact_frac", "B_copy_frac"])
    md.append("## Mechanism trace — the pusher `01 c5` tiled to L, executed as A against the same 8 random partners for 8 … 128 steps\n\n"
              "Fraction of A's bytes still intact and fraction of B's bytes equal to the parent, by step; the isolated gen2 (64 partners) in the index.\n\n"
              + piv.round(2).to_markdown() + "\n")

    with open(os.path.join(out, "NUMBERS_F.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))
    print("wrote", out)


if __name__ == "__main__":
    main()
