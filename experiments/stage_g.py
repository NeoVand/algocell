"""Stage G — closure, confirmatory (pre-registered in PLAN.md on 2026-10-08 before the runs).

    python stage_g.py [--dir runs/stageG] [--out runs/stageG/analysis/stageG]

Per world (80 runs, seeds 2001–2020, `none`, 128 steps, 1/16): the first heritable replicator (t_rep tape) and the
final dominant tape from `analysis/assays.csv`; for each, (i) control-flow instructions by linear disassembly
(jump, relative jump, DJNZ, CALL/RET, RST — the pre-registered, syntactic definition; block-repeat instructions
LDIR/LDDR are reported separately because they loop in hardware without a jump), (ii) the executor partner test
(256 random partners, one 128-step encounter): fraction of partners that become a ≥ 75% copy and fraction of
encounters in which the organism loses ≥ 25% of its bytes. The heritable fraction of 16 random cells per sample
comes from `c4_functional.py` (`analysis/c4/functional.csv`). Writes stage_g_runs.csv, NUMBERS_G.md, two figures.
Verdicts are computed against the pre-registered thresholds and printed; nothing is chosen after the fact.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import figstyle as fs
from algocell_exp.isa import disassemble
from closure import control_flow, hb, partner_test

BLOCK = ("LDIR", "LDDR", "CPIR", "CPDR", "INIR", "INDR", "OTIR", "OTDR")


def block_repeat(tape_hex: str) -> list[str]:
    return sorted({i["mnemonic"] for i in disassemble(hb(tape_hex)) if i["mnemonic"].split()[0] in BLOCK})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/stageG")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "analysis", "stageG")
    os.makedirs(out, exist_ok=True)
    d = pd.read_csv(os.path.join(a.dir, "analysis", "assays.csv"))
    if "replicate" in d:
        d = d[d["replicate"].isna()]
    rng = np.random.default_rng(2026)
    rows = []
    for _, r in d.iterrows():
        L = int(r["tape_len"])
        first = r["trep_tape"] if isinstance(r["trep_tape"], str) else None
        final = r["final_tape"] if isinstance(r["final_tape"], str) else None
        row = {"label": r["label"], "L": L, "seed": int(r["seed"]), "horizon": int(r["horizon"]), "t_rep": r["t_rep"], "t_faith": r["t_faith"],
               "first_tape": first, "final_tape": final, "final_faithful": bool(r["final_faithful"]), "final_share": r["final_share"],
               "first_gen2": r["trep_gen2"], "final_gen2": r["final_gen2"]}
        for which, hx in (("first", first), ("final", final)):
            if hx is None:
                continue
            cf = control_flow(hx)
            br = block_repeat(hx)
            copied, damaged, sim = partner_test(hb(hx), rng)
            row.update({f"{which}_cf": "+".join(cf) or "-", f"{which}_has_cf": bool(cf), f"{which}_block": "+".join(br) or "-", f"{which}_has_block": bool(br),
                        f"{which}_copied": copied, f"{which}_damaged": damaged, f"{which}_sim": sim})
        rows.append(row)
    R = pd.DataFrame(rows).sort_values(["L", "seed"])
    R.to_csv(os.path.join(out, "stage_g_runs.csv"), index=False)

    c4p = os.path.join(a.dir, "analysis", "c4", "functional.csv")
    C4 = pd.read_csv(c4p) if os.path.exists(c4p) else None

    md = ["# Stage G — numbers (generated; do not edit)\n",
          "Per-world executor tests: 256 random partners, one 128-step encounter; `copied` = fraction of partners that become a ≥ 75% copy "
          "(best cyclic shift), `damaged` = fraction of encounters in which the organism loses ≥ 25% of its bytes. Control flow = jump, relative jump, "
          "DJNZ, CALL/RET, RST by linear disassembly (pre-registered); `block` = LDIR/LDDR-type repeat instructions (reported separately).\n"]
    verdicts = []
    for L, g in R.groupby("L"):
        n = len(g)
        first_cf = int(g["first_has_cf"].sum())
        first_open = int((g["first_copied"] < 0.90).sum())
        final_cf = int((g["final_faithful"] & g["final_has_cf"]).sum())
        final_closed = int((g["final_faithful"] & (g["final_copied"] >= 0.95)).sum())
        final_closed_cf = int((g["final_faithful"] & (g["final_copied"] >= 0.95) & g["final_has_cf"]).sum())
        final_block = int((g["final_faithful"] & g["final_has_block"]).sum())
        final_closed_any = int((g["final_faithful"] & (g["final_copied"] >= 0.95) & (g["final_has_cf"] | g["final_has_block"])).sum())
        modal_final = g["final_tape"].mode().iloc[0]
        mf = g[g["final_tape"] == modal_final].iloc[0]
        modal_first = g["first_tape"].mode().iloc[0]
        horizon = int(g["horizon"].iloc[0])
        md.append(f"## L = {L} ({g['label'].iloc[0]}, horizon {horizon:,}, {n} worlds)\n")
        md.append(f"- first replicator: control flow in {first_cf}/{n}; copies < 90% of random partners in {first_open}/{n}; "
                  f"modal first tape `{modal_first}` in {int((g['first_tape'] == modal_first).sum())}/{n} worlds; "
                  f"median copied {g['first_copied'].median():.2f} (range {g['first_copied'].min():.2f}–{g['first_copied'].max():.2f}), "
                  f"median damaged {g['first_damaged'].median():.2f} (range {g['first_damaged'].min():.2f}–{g['first_damaged'].max():.2f})")
        md.append(f"- final dominant (faithful in {int(g['final_faithful'].sum())}/{n}): control flow in {final_cf}/{n}; block-repeat (LDIR/LDDR) in {final_block}/{n}; "
                  f"copies ≥ 95% of partners in {final_closed}/{n} (with control flow {final_closed_cf}/{n}; with control flow or block-repeat {final_closed_any}/{n}); "
                  f"modal final tape `{modal_final[:47]}{'…' if len(modal_final) > 47 else ''}` in {int((g['final_tape'] == modal_final).sum())}/{n} worlds, "
                  f"copied {mf['final_copied']:.2f}, damaged {mf['final_damaged']:.2f}, control flow {mf['final_cf']}, block {mf['final_block']}")
        md.append(f"- paired per world: gained control flow {int((~g['first_has_cf'] & g['final_has_cf']).sum())}, lost {int((g['first_has_cf'] & ~g['final_has_cf']).sum())}; "
                  f"gained closure (copied < 0.95 → ≥ 0.95) {int(((g['first_copied'] < 0.95) & (g['final_copied'] >= 0.95)).sum())}\n")
        if C4 is not None:
            c = C4[C4["tape_len"] == L]
            med = c.groupby("step")["frac_heritable"].median()
            md.append("heritable fraction of 16 random cells (median over worlds) by step:\n")
            md.append(med.to_frame("median_heritable").T.round(2).to_markdown() + "\n")
        # verdicts
        if horizon == 300_000:
            early = None
            if C4 is not None:
                c = C4[C4["tape_len"] == L].merge(g[["seed", "t_rep"]], on="seed")
                e = c[(c["step"] <= 5000) & (c["step"] >= c["t_rep"])]
                early = e.groupby("step")["frac_heritable"].median()
                late = c[c["step"] == 100_000]["frac_heritable"].median()
            ok_a = first_cf <= n - 18 and first_open >= 18
            ok_b = final_cf >= 16 and mf["final_copied"] >= 0.95 and mf["final_damaged"] <= 0.05
            ok_c = (early is not None) and bool((early < 0.20).all()) and late > 0.70
            kill = final_cf < 12 or first_cf >= 6
            verdicts.append(f"- **G1, L = {L}**: (a) first without control flow {n - first_cf}/{n} (≥ 18 needed) and partner-dependent {first_open}/{n} (≥ 18): "
                            f"{'met' if ok_a else 'NOT met'}; (b) control-flow faithful dominant {final_cf}/{n} (≥ 16), modal final copied {mf['final_copied']:.2f} (≥ 0.95), "
                            f"damaged {mf['final_damaged']:.2f} (≤ 0.05): {'met' if ok_b else 'NOT met'}; "
                            f"(c) median heritable fraction ≤ 5k after emergence {'all < 0.20' if early is not None and (early < 0.20).all() else ('n/a' if early is None else f'max {early.max():.2f}')}, "
                            f"at 100k {late:.2f} (> 0.70): {'met' if ok_c else 'NOT met'}; kill criterion {'TRIGGERED' if kill else 'not triggered'}. "
                            f"Behavioural closure (copied ≥ 0.95) in {final_closed}/{n} finals; with a loop of either kind {final_closed_any}/{n}.")
        else:
            verdicts.append(f"- **G2, L = {L}**: faithful control-flow dominant at 1M copying ≥ 95% of partners in {final_closed_cf}/{n} "
                            f"(prediction ≥ 10, alternative < 5): {'prediction' if final_closed_cf >= 10 else ('alternative' if final_closed_cf < 5 else 'between')}; "
                            f"with a loop of either kind (control flow or LDIR/LDDR) {final_closed_any}/{n}; closed by partner test regardless of syntax {final_closed}/{n}; "
                            f"finals still the open pusher (no control flow, no block-repeat, copied < 0.95): "
                            f"{int((~g['final_has_cf'] & ~g['final_has_block'] & (g['final_copied'] < 0.95)).sum())}/{n}.")
    md.append("## Pre-registered verdicts\n")
    md.extend(verdicts)
    md.append("\n## Final-tape classes per L\n")
    for L, g in R.groupby("L"):
        md.append(f"L = {L}:\n")
        t = g.groupby(["final_cf", "final_block"]).agg(n=("seed", "size"), copied=("final_copied", "median"), damaged=("final_damaged", "median")).reset_index()
        md.append(t.to_markdown(index=False, floatfmt=".2f") + "\n")
    with open(os.path.join(out, "NUMBERS_G.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(verdicts))

    # figures
    fs.setup()
    import matplotlib.pyplot as plt
    Ls = sorted(R["L"].unique())
    fig, axes = plt.subplots(1, len(Ls), figsize=(fs.DOUBLE, 2.4), sharey=True)
    for ax, L in zip(np.atleast_1d(axes), Ls):
        g = R[R["L"] == L]
        jit = rng.uniform(-0.12, 0.12, size=len(g))
        for x, which in ((0, "first"), (1, "final")):
            cf = g[f"{which}_has_cf"] | g[f"{which}_has_block"]
            ax.scatter(x + jit[~cf.values], g[f"{which}_copied"][~cf], s=12, color="#000000", facecolors="none", lw=0.8, label="no loop instruction" if (x == 0 and L == Ls[0]) else None)
            ax.scatter(x + jit[cf.values], g[f"{which}_copied"][cf], s=12, color="#D55E00", label="loop instruction (jump or LDIR/LDDR)" if (x == 1 and L == Ls[0]) else None)
        ax.set_xticks([0, 1], ["first\nreplicator", "final\ndominant"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"L = {L} ({int(g['horizon'].iloc[0]) // 1000}k steps)", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("random partners that become a copy")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False)
    fs.save(fig, os.path.join(out, "G_partner_independence"))
    if C4 is not None:
        fig, ax = plt.subplots(figsize=(fs.SINGLE, 2.4))
        for i, L in enumerate(Ls):
            c = C4[C4["tape_len"] == L].groupby("step")["frac_heritable"]
            med, lo, hi = c.median(), c.quantile(0.25), c.quantile(0.75)
            col = fs.color(f"L{L}") if hasattr(fs, "color") else None
            ax.plot(med.index, med.values, lw=1.1, label=f"L = {L}", color=col)
            ax.fill_between(med.index, lo.values, hi.values, alpha=0.15, color=col, lw=0)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("steps")
        ax.set_ylabel("heritable fraction of 16 random cells\n(median, IQR over 20 worlds)", fontsize=7)
        ax.legend(frameon=False, fontsize=7)
        fs.save(fig, os.path.join(out, "G_heritable_fraction"))
    print("wrote", out)


if __name__ == "__main__":
    main()
