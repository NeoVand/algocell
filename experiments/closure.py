"""Closure — does the successor of the first replicator stop depending on its partner, and how? (post hoc, 2026-10-08)

    python closure.py [--out runs/closure]

Three measurements over the `none` runs of Stages B, C and E at 128 steps · 1/16 (every tape length):
  1. control flow: does the first heritable replicator (t_rep tape) / the final faithful dominant contain a jump,
     relative jump, DJNZ, CALL/RET or RST (linear disassembly)? Paired per run (exact McNemar).
  2. partner-independence: the modal first tape and the modal final faithful tape of each L, executed as A for 128
     steps against 256 random partners: fraction of partners that become a ≥ 75% copy (best cyclic shift), fraction of
     encounters in which A loses ≥ 25% of its own bytes; and the isolated gen2 (assay, 64 partners).
  3. convergence: how many seeds share the modal first tape and the modal final tape at each L.
Writes NUMBERS_CLOSURE.md, closure_runs.csv, closure_tapes.csv, and a figure.
"""

from __future__ import annotations

import argparse
import os
from math import comb

import numpy as np
import pandas as pd

import figstyle as fs
from algocell_exp.assay import assay, execute_pairs
from algocell_exp.isa import disassemble

CF_FAMILIES = ("jump", "jump-rel", "call-ret", "rst")


def hb(s: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(s.replace(" ", "")), dtype=np.uint8).copy()


def control_flow(tape_hex: str) -> list[str]:
    return sorted({i["mnemonic"] for i in disassemble(hb(tape_hex)) if i["family"] in CF_FAMILIES})


def partner_test(u: np.ndarray, rng: np.random.Generator, n: int = 256, steps: int = 128, zero_halts: bool = False) -> tuple[float, float, float]:
    """(copied, damaged, mean similarity) of u as A against n random partners for one encounter; `zero_halts` runs the
    encounters under the lethal-tar rule (Stage I worlds are tested under their own rule)."""
    L = len(u)
    P = rng.integers(0, 256, size=(n, L), dtype=np.uint8)
    res = np.asarray(execute_pairs(np.concatenate([np.repeat(u[None], n, 0), P], axis=1), L, steps, [], zero_halts=zero_halts)).reshape(n, 2 * L)
    A2, B2 = res[:, :L], res[:, L:]
    best = np.zeros(n)
    for s in range(L):
        best = np.maximum(best, (B2 == np.roll(u, s)).mean(1))
    return float((best >= 0.75).mean()), float(((A2 != u).mean(1) >= 0.25).mean()), float(best.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/closure")
    ap.add_argument("--stages", default="stageB,stageC,stageE")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = []
    for stage in a.stages.split(","):
        p = os.path.join("runs", stage, "analysis", "assays.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        if "replicate" in d:
            d = d[d["replicate"].isna()]
        d = d[d["label"].isin(["none", "none@nominal"]) & (d["steps"] == 128) & (d["k"] == 4)]
        for _, r in d.iterrows():
            first = r["trep_tape"] if isinstance(r["trep_tape"], str) else None
            final = r["final_tape"] if (isinstance(r["final_tape"], str) and bool(r["final_faithful"])) else None
            runs.append({"stage": stage, "L": int(r["tape_len"]), "seed": int(r["seed"]), "t_rep": r["t_rep"], "first_tape": first, "final_tape": final,
                         "first_cf": "+".join(control_flow(first)) if first else None, "final_cf": "+".join(control_flow(final)) if final else None,
                         "first_period": r["trep_period"], "final_period": r["final_period"], "first_gen2": r["trep_gen2"], "final_gen2": r["final_gen2"]})
    R = pd.DataFrame(runs)
    R.to_csv(os.path.join(a.out, "closure_runs.csv"), index=False)
    both = R.dropna(subset=["first_tape", "final_tape"])
    f_cf = both["first_cf"].astype(str).ne("") & both["first_cf"].notna()
    l_cf = both["final_cf"].astype(str).ne("") & both["final_cf"].notna()
    b = int((~f_cf & l_cf).sum())
    c = int((f_cf & ~l_cf).sum())
    p = min(1.0, 2 * sum(comb(b + c, i) for i in range(0, min(b, c) + 1)) / 2 ** (b + c)) if b + c else float("nan")
    md = ["# Closure (generated)\n",
          f"## 1. Control flow in the first replicator vs the final faithful dominant (`none`, 128 steps, 1/16; Stages {a.stages})\n",
          f"- first replicators with any control-flow instruction: {int((R['first_cf'].astype(str).ne('') & R['first_cf'].notna()).sum())} / {int(R['first_tape'].notna().sum())}",
          f"- final faithful dominants with any control-flow instruction: {int(l_cf.sum())} / {len(both)} (runs with both tapes)",
          f"- paired: gained control flow {b}, lost {c}, both {int((f_cf & l_cf).sum())}, neither {int((~f_cf & ~l_cf).sum())}; exact McNemar two-sided p = {p:.2g}\n"]
    per_L = both.assign(first_cf_b=f_cf, final_cf_b=l_cf).groupby("L").agg(n=("seed", "size"), first_cf=("first_cf_b", "sum"), final_cf=("final_cf_b", "sum"),
                                                                             final_ops=("final_cf", lambda x: ", ".join(f"{k}:{v}" for k, v in x[x.astype(str).ne("")].value_counts().head(3).items())))
    md.append(per_L.to_markdown() + "\n")

    # 2 + 3: modal tapes per L
    rng = np.random.default_rng(0)
    rows = []
    for L, g in R.groupby("L"):
        for which, col in (("first", "first_tape"), ("final", "final_tape")):
            tapes = g[col].dropna()
            if tapes.empty:
                continue
            hx = tapes.mode().iloc[0]
            u = hb(hx)
            copied, damaged, sim = partner_test(u, rng)
            r = assay(u, z80_steps=128, suppress=[], n=64, seed=0)
            rows.append({"L": int(L), "which": which, "n_tapes": len(tapes), "seeds_identical": int((tapes == hx).sum()), "distinct": int(tapes.nunique()),
                         "control_flow": "+".join(control_flow(hx)) or "-", "period": int(g.loc[g[col] == hx, f"{which}_period"].iloc[0]),
                         "partners_copied": copied, "self_damaged": damaged, "mean_partner_similarity": sim, "gen2_isolated": round(r["gen2_score"], 2),
                         "tape": hx if L <= 36 else hx[:47] + " …"})
    T = pd.DataFrame(rows).sort_values(["L", "which"], ascending=[True, False])
    T.to_csv(os.path.join(a.out, "closure_tapes.csv"), index=False)
    md.append("## 2–3. The modal first and final tapes per L: convergence, control flow and partner-independence (256 random partners, one 128-step encounter)\n")
    md.append(T[["L", "which", "n_tapes", "seeds_identical", "distinct", "period", "control_flow", "partners_copied", "self_damaged", "gen2_isolated", "tape"]].to_markdown(index=False, floatfmt=".2f") + "\n")
    with open(os.path.join(a.out, "NUMBERS_CLOSURE.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md[:6]))
    print(T[["L", "which", "seeds_identical", "n_tapes", "period", "control_flow", "partners_copied", "self_damaged", "gen2_isolated"]].to_string(index=False))

    fs.setup()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE, 2.6))
    first = T[T["which"] == "first"]
    final = T[T["which"] == "final"]
    for ax, col, yl in zip(axes, ("partners_copied", "self_damaged"), ("fraction of random partners that become a copy", "fraction of encounters that damage the parent")):
        ax.plot(first["L"], first[col], marker="o", color="#000000", lw=1, ls="none", ms=4, label="first replicator (modal tape)")
        cf = final[final["control_flow"] != "-"]
        ncf = final[final["control_flow"] == "-"]
        ax.plot(cf["L"], cf[col], marker="s", color="#D55E00", lw=0, ms=5, label="final dominant with control flow")
        ax.plot(ncf["L"], ncf[col], marker="s", color="#D55E00", mfc="none", lw=0, ms=5, label="final dominant without control flow")
        ax.set_xscale("log")
        ax.set_xticks([3, 4, 9, 16, 25, 36, 49, 64, 81, 100], ["3", "4", "9", "16", "25", "36", "49", "64", "81", "100"])
        ax.minorticks_off()
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("tape length L")
        ax.set_ylabel(yl, fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False)
    fs.save(fig, os.path.join(a.out, "closure_partner_independence"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
