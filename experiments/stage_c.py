"""Stage C — long horizons, uncensored succession, finer ablations (490 runs, L = 16 unless stated).

    python stage_c.py runs/stageC [--out runs/stageC/analysis/stage_c]

Reads analysis/assays.csv, analysis/succession.csv and the run jsonl (interaction clock); writes NUMBERS_C.md + figures:
  C1  1,000,000-step nulls (no-copy, rmw-only, all-ld): heritable / faithful counts, KM medians, horizon reached
  C2  succession without censoring (none, block-copy, ld-mem × {128, 512} × k ∈ {2, 4, 6}): census family at 5k / 50k / 300k
  C3  finer ablations at (128, k = 4) and (32, k = 2): emergence, KM medians in steps and in active interactions per cell,
      seed-paired sign test against `none`, delay ratio, tiling of the first replicator
  C5  size without censoring (L ∈ {36, 100}): periods of the first and of the final dominant replicator
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import figstyle as fs
from analyze import km_median, wilson
from report import fisher, km_str, sign_test

C3_ORDER = ["none", "stack-writes", "stack-write-only", "stack-read-only", "push-only", "ex-sp-only", "call-rst",
            "ld-imm", "ld-reg", "ld-mem", "cb-page", "ed-loads", "block-copy", "all-ld", "rmw-only", "no-copy"]
FAMILY = {"none": "comparator", "stack-writes": "stack", "stack-write-only": "stack", "stack-read-only": "stack", "push-only": "stack",
          "ex-sp-only": "stack", "call-rst": "stack", "ld-imm": "load", "ld-reg": "load", "ld-mem": "load", "cb-page": "other",
          "ed-loads": "other", "block-copy": "copy", "all-ld": "copy", "rmw-only": "copy", "no-copy": "copy"}
FAMILY_COLOR = {"comparator": "#000000", "stack": "#56B4E9", "load": "#E69F00", "other": "#999999", "copy": "#D55E00"}


def interaction_clock(files: list[str]) -> pd.Series:
    """Active interactions per cell per step, per run, from the recorded active_pairs (two cells per pair)."""
    out = {}
    for f in files:
        stem = f[: -len(".summary.json")]
        s = json.load(open(f))
        acts = []
        if os.path.exists(stem + ".jsonl"):
            for line in open(stem + ".jsonl"):
                if '"kind": "sample"' in line:
                    acts.append(json.loads(line).get("active_pairs", np.nan))
        out[os.path.basename(f)] = 2.0 * float(np.nanmean(acts)) / s["cells"] if acts else np.nan
    return pd.Series(out, name="ix_per_cell_step")


def km_curve(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan–Meier survival step function (events before censorings at equal times)."""
    order = np.lexsort((~events, times))
    t, e = times[order], events[order]
    n = len(t)
    xs, ys = [0.0], [1.0]
    S = 1.0
    at_risk = n
    i = 0
    while i < n:
        j = i
        d = 0
        while j < n and t[j] == t[i]:
            d += int(e[j])
            j += 1
        if d:
            S *= 1 - d / at_risk
            xs += [t[i], t[i]]
            ys += [ys[-1], S]
        at_risk -= j - i
        i = j
    return np.array(xs), np.array(ys)


def cell(g: pd.DataFrame) -> dict:
    n = len(g)
    row = {"n": n, "steps_run_min": int(g["steps_run"].min()), "stopped_early": int(g["stopped_early"].fillna(False).astype(bool).sum())}
    for ev in ("tq_10", "t_rep", "t_faith"):
        t = g[ev].astype(float)
        em = t > 0
        ne = int(em.sum())
        lo, hi = wilson(ne, n)
        times = np.where(em, t, g["steps_run"]).astype(float)
        row.update({f"{ev}_n": ne, f"{ev}_frac": ne / n, f"{ev}_lo": lo, f"{ev}_hi": hi, f"{ev}_km": km_median(times, em.to_numpy())})
        if "ix_per_cell_step" in g:
            row[f"{ev}_km_ix"] = km_median(times * g["ix_per_cell_step"].to_numpy(), em.to_numpy())
    rep = g[g["t_rep"] > 0]
    row["tiled_frac"] = float((rep["trep_period"] <= rep["tape_len"] / 2).mean()) if len(rep) else np.nan
    tiled = rep[rep["trep_period"] <= rep["tape_len"] / 2]
    row["div2L_tiled"] = float(((2 * tiled["tape_len"]) % tiled["trep_period"].astype(int) == 0).mean()) if len(tiled) else np.nan
    row["periods"] = ", ".join(f"{int(p)}×{c}" for p, c in rep["trep_period"].value_counts().sort_index().items()) if len(rep) else "–"
    row["mechs"] = ", ".join(f"{m}:{c}" for m, c in rep["trep_mechs"].value_counts().items()) if len(rep) else "–"
    row["final_faithful_n"] = int(g["final_faithful"].fillna(False).astype(bool).sum())
    row["func_rnd_median"] = float(g["final_func_rnd"].median())
    row["zero_final_median"] = float(g["final_zero_frac"].median()) if "final_zero_frac" in g else np.nan
    row["ix_per_cell_step"] = float(g["ix_per_cell_step"].mean()) if "ix_per_cell_step" in g else np.nan
    return row


def paired_vs_none(asy: pd.DataFrame, label: str, steps: int, k: int, tape: int = 16) -> tuple[int, int, int, float]:
    """Seed-paired t_rep comparison arm vs none at the same budget: (#arm slower, #none slower, ties, two-sided sign p). Censored = slower than any emerged."""
    sel = lambda lab: asy[(asy["label"] == lab) & (asy["tape_len"] == tape) & (asy["k"] == k) & (asy["steps"] == steps) & asy["replicate"].isna()].set_index("seed")["t_rep"]
    A, N = sel(label), sel("none")
    slow = fast = tie = 0
    for s in A.index.intersection(N.index):
        ta = np.inf if (pd.isna(A[s]) or A[s] < 0) else A[s]
        tn = np.inf if (pd.isna(N[s]) or N[s] < 0) else N[s]
        if ta > tn:
            slow += 1
        elif ta < tn:
            fast += 1
        else:
            tie += 1
    return slow, fast, tie, sign_test(slow, fast)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "analysis", "stage_c")
    os.makedirs(out, exist_ok=True)
    asy = pd.read_csv(os.path.join(a.dir, "analysis", "assays.csv"))
    clock = interaction_clock([os.path.join(a.dir, f) for f in asy["file"]])
    asy = asy.merge(clock.rename_axis("file").reset_index(), on="file", how="left")
    succ_path = os.path.join(a.dir, "analysis", "succession.csv")
    succ = pd.read_csv(succ_path) if os.path.exists(succ_path) else pd.DataFrame()
    fs.setup()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    md = ["# Stage C numbers (generated)\n"]
    md.append(f"Interaction clock: mean active interactions per cell per step over all Stage C runs = {asy['ix_per_cell_step'].mean():.4f} (min {asy['ix_per_cell_step'].min():.4f}, max {asy['ix_per_cell_step'].max():.4f}); 1,000 steps ≈ {1000 * asy['ix_per_cell_step'].mean():.0f} encounters per cell.\n")

    # ── C1: 1M-step nulls ──
    c1 = asy[asy["horizon"] >= 1_000_000]
    if len(c1):
        rows = []
        for (lab, steps, k), g in c1.groupby(["label", "steps", "k"]):
            rows.append({"label": lab, "steps": steps, "k": k, **cell(g)})
        t1 = pd.DataFrame(rows).sort_values(["steps", "label"])
        t1.to_csv(os.path.join(out, "c1_nulls.csv"), index=False)
        md.append("## C1 — 1,000,000-step runs\n\n" + t1[["label", "steps", "k", "n", "steps_run_min", "tq_10_n", "t_rep_n", "t_rep_km", "t_faith_n", "periods", "mechs", "func_rnd_median", "zero_final_median"]].to_markdown(index=False, floatfmt=".3g") + "\n")
        fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE, 2.4), sharey=True)
        for ax, (steps, k) in zip(axes, ((128, 4), (32, 2))):
            for lab in ("none", "all-ld", "rmw-only", "no-copy"):
                g = asy[(asy["label"] == lab) & (asy["steps"] == steps) & (asy["k"] == k) & (asy["tape_len"] == 16) & asy["replicate"].isna()]
                if g.empty:
                    continue
                t = g["t_rep"].astype(float)
                em = (t > 0).to_numpy()
                xs, ys = km_curve(np.where(em, t, g["steps_run"]).astype(float), em)
                xs = np.append(xs, g["steps_run"].max())
                ys = np.append(ys, ys[-1])
                ax.step(np.maximum(xs, 1), 1 - ys, where="post", color=fs.color(lab), lw=1.2, label=f"{lab} (n = {len(g)})")
            ax.set_xscale("log")
            ax.set_xlim(100, 1.2e6)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{steps} steps, mutation 1/{2**k}")
            ax.set_xlabel("steps")
            ax.axvline(300_000, color="#cccccc", lw=0.6, ls=":")
        axes[0].set_ylabel("fraction of seeds with a heritable replicator")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False, title="dotted: 300k-step horizon of C2/C3")
        fs.save(fig, os.path.join(out, "C1_nulls_km"))

    # ── C3: finer ablations at L = 16 ──
    for steps, k in ((128, 4), (32, 2)):
        g0 = asy[(asy["tape_len"] == 16) & (asy["steps"] == steps) & (asy["k"] == k) & asy["replicate"].isna()]
        if g0.empty:
            continue
        rows = []
        none_km = cell(g0[g0["label"] == "none"])["t_rep_km"] if (g0["label"] == "none").any() else np.nan
        for lab in [l for l in C3_ORDER if l in set(g0["label"])]:
            g = g0[g0["label"] == lab]
            r = {"label": lab, "family": FAMILY.get(lab, "other"), **cell(g)}
            r["km_ratio_vs_none"] = r["t_rep_km"] / none_km if np.isfinite(none_km) else np.nan
            if lab != "none":
                r["slower"], r["faster"], r["ties"], r["sign_p"] = paired_vs_none(asy, lab, steps, k)
            rows.append(r)
        t3 = pd.DataFrame(rows)
        t3.to_csv(os.path.join(out, f"c3_ablations_st{steps}_k{k}.csv"), index=False)
        md.append(f"## C3 — ablations at {steps} steps, mutation 1/{2**k}, L = 16, seeds 101–110\n\n"
                  + t3[["label", "n", "steps_run_min", "t_rep_n", "t_rep_km", "t_rep_km_ix", "km_ratio_vs_none", "slower", "faster", "ties", "sign_p", "t_faith_n", "tq_10_n", "tiled_frac", "div2L_tiled", "periods", "zero_final_median"]].to_markdown(index=False, floatfmt=".3g") + "\n")
        md.append("`slower/faster/ties`: seed-paired comparison of t_rep against `none` (a censored run counts as slower than any emerged run; two censored runs tie); `sign_p` two-sided exact sign test ignoring ties; `t_rep_km_ix` = KM median in cumulative active interactions per cell.\n")
        # forest figure
        fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE, 0.22 * len(t3) + 1.0), sharey=True, gridspec_kw={"width_ratios": [1, 1.4]})
        y = np.arange(len(t3))[::-1]
        cols = [FAMILY_COLOR[f] for f in t3["family"]]
        axes[0].errorbar(t3["t_rep_frac"], y, xerr=[t3["t_rep_frac"] - t3["t_rep_lo"], t3["t_rep_hi"] - t3["t_rep_frac"]], fmt="none", ecolor="#bbbbbb", elinewidth=0.8, capsize=0)
        axes[0].scatter(t3["t_rep_frac"], y, c=cols, s=18, zorder=3)
        axes[0].scatter(t3["t_faith_frac"], y, facecolors="none", edgecolors=cols, s=34, lw=0.8, zorder=3)
        axes[0].set_xlim(-0.05, 1.05)
        axes[0].set_xlabel("fraction of seeds (of 10)")
        axes[0].set_yticks(y, t3["label"])
        for lab_y, c in zip(axes[0].get_yticklabels(), cols):
            lab_y.set_color(c)
        km = t3["t_rep_km"].to_numpy(float)
        ok = np.isfinite(km)
        axes[1].scatter(km[ok], y[ok], c=np.array(cols)[ok], s=18, zorder=3)
        axes[1].scatter([t3["steps_run_min"].max() * 1.3] * int((~ok).sum()), y[~ok], marker=">", c=np.array(cols)[~ok], s=18, zorder=3)
        if np.isfinite(none_km):
            axes[1].axvline(none_km, color="#000000", lw=0.6, ls=":")
            for yy, lab in zip(y, t3["label"]):
                r = t3.loc[t3["label"] == lab, "km_ratio_vs_none"].iloc[0]
                if lab != "none" and np.isfinite(r):
                    axes[1].text(t3["steps_run_min"].max() * 2.2, yy, f"×{r:.3g}" if r < 10 else f"×{r:,.0f}", va="center", fontsize=6, color="#555555")
        axes[1].set_xscale("log")
        axes[1].set_xlim(100, t3["steps_run_min"].max() * 4)
        axes[1].set_xlabel("KM median steps to first heritable replicator (▶ not reached)")
        axes[0].set_title("heritable (●) and faithful (○) emergence", fontsize=8)
        axes[1].set_title(f"{steps} Z80 steps, mutation 1/{2**k}, 300,000-step horizon", fontsize=8)
        fs.save(fig, os.path.join(out, f"C3_forest_st{steps}_k{k}"))

    # ── C2: succession without censoring ──
    c2_labels = ["none", "block-copy", "ld-mem"]
    if len(succ):
        s2 = succ[succ["label"].isin(c2_labels) & (succ["tape"] == 16)]
        if len(s2):
            fam = lambda x: ", ".join(f"{k_}:{v}" for k_, v in x.dropna().value_counts().items()) or "–"
            g = s2.groupby(["label", "steps", "k"]).agg(n=("seed", "size"), family_5k=("family_5k", fam), family_50k=("family_50k", fam), family_300k=("family_300k", fam),
                                                         ldir_300k=("ldir_300k", "median"), zero8_300k=("zero8_300k", "median"), final_zero_frac=("final_zero_frac", "median"),
                                                         ldir_takeover=("t_ldir_30", lambda x: f"{int((x > 0).sum())} runs, median {x[x > 0].median():.0f}" if (x > 0).any() else "0"),
                                                         stack_takeover=("t_push_30", lambda x: f"{int((x > 0).sum())} runs, median {x[x > 0].median():.0f}" if (x > 0).any() else "0"))
            md.append("## C2 — succession without censoring (census family of the dominant unit at fixed steps; medians over seeds)\n\n" + g.reset_index().to_markdown(index=False, floatfmt=".3g") + "\n")
    rows = []
    for (lab, steps, k), g in asy[asy["label"].isin(c2_labels) & (asy["tape_len"] == 16) & asy["replicate"].isna() & (asy["horizon"] == 300_000)].groupby(["label", "steps", "k"]):
        rows.append({"label": lab, "steps": steps, "k": k, **cell(g)})
    if rows:
        t2 = pd.DataFrame(rows).sort_values(["label", "steps", "k"])
        t2.to_csv(os.path.join(out, "c2_emergence.csv"), index=False)
        md.append("### C2 emergence by mutation rate\n\n" + t2[["label", "steps", "k", "n", "t_rep_n", "t_rep_km", "t_rep_km_ix", "t_faith_n", "final_faithful_n", "func_rnd_median", "periods", "zero_final_median"]].to_markdown(index=False, floatfmt=".3g") + "\n")
        fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE, 2.4), sharey=True)
        for ax, steps in zip(axes, (128, 512)):
            for lab in c2_labels:
                s = t2[(t2["label"] == lab) & (t2["steps"] == steps)].sort_values("k")
                if s.empty:
                    continue
                rate = 8192 / 2 ** s["k"] / (20000 * 16)
                ok = np.isfinite(s["t_rep_km"])
                ax.plot(rate[ok], s["t_rep_km"][ok], marker="o", color=fs.color(lab), lw=1, label=lab)
                ax.plot(rate[~ok], [400_000] * int((~ok).sum()), marker="^", mfc="white", ls="none", color=fs.color(lab))
            ax.set_xscale("log")
            ax.set_yscale("log")
            rates = [8192 / 2 ** kk / (20000 * 16) for kk in (6, 4, 2)]
            ax.set_xticks(rates, [f"{r:.1e}\n(1/{2**kk})" for r, kk in zip(rates, (6, 4, 2))])
            ax.minorticks_off()
            ax.set_yticks([100, 200, 500, 1000, 2000, 5000], ["100", "200", "500", "1,000", "2,000", "5,000"])
            ax.set_xlabel("mutations per byte per step (mutation setting)")
            ax.set_title(f"{steps} Z80 steps")
        axes[0].set_ylabel("KM median steps to heritable replicator")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False, title="△ median not reached")
        fs.save(fig, os.path.join(out, "C2_mutation"))

    # ── C5: size without censoring ──
    c5 = asy[asy["tape_len"].isin([36, 100]) & asy["replicate"].isna()]
    if len(c5):
        rows = []
        for (lab, L, steps, k), g in c5.groupby(["label", "tape_len", "steps", "k"]):
            r = {"label": lab, "L": L, "steps": steps, "k": k, **cell(g)}
            if "final_period" in g:
                fp = g.loc[g["final_faithful"].fillna(False).astype(bool), "final_period"]
                r["final_period_le8"] = f"{int((fp <= 8).sum())}/{len(fp)}"
                r["final_periods"] = ", ".join(f"{int(p)}×{c}" for p, c in fp.value_counts().sort_index().items()) or "–"
            rp = g.loc[g["t_rep"] > 0, "trep_period"]
            r["trep_period_le8"] = f"{int((rp <= 8).sum())}/{len(rp)}"
            rows.append(r)
        t5 = pd.DataFrame(rows)
        t5.to_csv(os.path.join(out, "c5_size.csv"), index=False)
        cols5 = ["label", "L", "steps", "k", "n", "t_rep_n", "t_rep_km", "t_faith_n", "final_faithful_n", "periods", "trep_period_le8"] + [c for c in ("final_period_le8", "final_periods") if c in t5]
        md.append("## C5 — L ∈ {36, 100} without early stop (periods of the first heritable replicator and of the faithful final dominant)\n\n" + t5[cols5].to_markdown(index=False, floatfmt=".3g") + "\n")

    # ── C1 upper bound: pooled zero cells → 95% one-sided upper bound on the per-run probability of emergence within the horizon ──
    if len(c1):
        zero = c1[c1["label"].isin(["no-copy", "rmw-only"])]
        n0 = int(len(zero)); e0 = int((zero["t_rep"] > 0).sum())
        if n0 and e0 == 0:
            md.append(f"C1 pooled: no-copy + rmw-only {e0}/{n0} runs emerged within {int(zero['steps_run'].min()):,} steps → 95% upper bound on the per-run emergence probability within that horizon = {1 - 0.05 ** (1 / n0):.3f} (exact binomial, one-sided).\n")

    # ── first vs final dominant under `none`: the ancestral and the evolved unit assayed in isolation (128 steps, 64 random partners) ──
    from algocell_exp.assay import assay, execute_pairs
    from algocell_exp.metrics import tolerant_period
    def hb(h): return np.frombuffer(bytes.fromhex(h.replace(" ", "")), dtype=np.uint8).copy()
    rows = []
    sel = asy[(asy["label"] == "none") & (asy["steps"] == 128) & (asy["k"] == 4) & asy["replicate"].isna() & (asy["horizon"] == 300_000)]
    for L, g in sel.groupby("tape_len"):
        first = g.loc[g["t_rep"] > 0, "trep_tape"].dropna()
        final = g.loc[g["final_faithful"].fillna(False).astype(bool), "final_tape"].dropna()
        for kind, tapes in (("first (t_rep)", first), ("final (faithful dominant)", final)):
            if tapes.empty:
                continue
            hx = tapes.mode().iloc[0]
            t = hb(hx)
            r = assay(t, z80_steps=128, suppress=[], n=64, seed=0)
            rng = np.random.default_rng(0)
            partners = rng.integers(0, 256, size=(64, L), dtype=np.uint8)
            res = np.asarray(execute_pairs(np.concatenate([np.repeat(t[None], 64, 0), partners], axis=1), L, 128, [])).reshape(64, 2 * L)
            rows.append({"L": int(L), "which": kind, "identical_in_seeds": f"{int((tapes == hx).sum())}/{len(g)}", "period": int(tolerant_period(t)[0]),
                         "tape": hx if L <= 36 else hx[:59] + " …", "score": r["score"], "gen2": r["gen2_score"], "faithful": r["faithful"],
                         "partner_bytes_changed": float((res[:, L:] != partners).sum(1).mean()), "self_bytes_changed": float((res[:, :L] != t).sum(1).mean())})
    if rows:
        tv = pd.DataFrame(rows)
        tv.to_csv(os.path.join(out, "first_vs_final_none.csv"), index=False)
        md.append("## First replicator vs final dominant under `none` (128 steps, 1/16): the modal tapes assayed in isolation\n\n" + tv.to_markdown(index=False, floatfmt=".2f") + "\n")
        md.append("`identical_in_seeds`: seeds whose tape is byte-identical to the modal one; `partner_bytes_changed` / `self_bytes_changed`: mean bytes of the random partner / of the tape itself that differ after one 128-step encounter as program A (64 partners).\n")

    # ── pre-registered verdicts (numbers only; prose in FINDINGS) ──
    lines = ["## Pre-registered C predictions — the numbers\n"]
    def get(lab, steps, k, L=16):
        g = asy[(asy["label"] == lab) & (asy["tape_len"] == L) & (asy["steps"] == steps) & (asy["k"] == k) & asy["replicate"].isna()]
        return cell(g) if len(g) else None
    for lab in ("no-copy", "rmw-only", "all-ld"):
        for steps, k in ((128, 4), (32, 2)):
            c = get(lab, steps, k)
            if c:
                lines.append(f"- C1 {lab} @({steps}, k{k}): heritable {c['t_rep_n']}/{c['n']} at {c['steps_run_min']:,} steps (KM median {km_str(c['t_rep_km'])}); faithful {c['t_faith_n']}/{c['n']}")
    for steps, k in ((128, 4), (32, 2)):
        n0 = get("none", steps, k)
        if not n0:
            continue
        for lab in ("stack-writes", "stack-write-only", "stack-read-only", "push-only", "ex-sp-only", "call-rst", "ld-imm", "ld-reg", "ld-mem", "cb-page", "ed-loads"):
            c = get(lab, steps, k)
            if c:
                sl, fa, ti, p = paired_vs_none(asy, lab, steps, k)
                lines.append(f"- C3 {lab} @({steps}, k{k}): {c['t_rep_n']}/{c['n']} heritable, KM {km_str(c['t_rep_km'])} vs none {km_str(n0['t_rep_km'])} (ratio {c['t_rep_km'] / n0['t_rep_km'] if np.isfinite(n0['t_rep_km']) and np.isfinite(c['t_rep_km']) else float('nan'):.2f}); paired vs none: {sl} slower, {fa} faster, {ti} ties, sign p = {p:.3g}")
    md.append("\n".join(lines) + "\n")

    with open(os.path.join(out, "NUMBERS_C.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print("wrote", out)


if __name__ == "__main__":
    main()
