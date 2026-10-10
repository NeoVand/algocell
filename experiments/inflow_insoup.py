"""X2 (REVISION_PREREG.md, "X1–X3"): inflow and openness of the first replicators against partners drawn from the soup.

    .venv/bin/python inflow_insoup.py --yes [--out results/inflow_insoup] [--n 256] [--only 16:2001]

Organisms: the first heritable replicator of each of the 80 Stage G worlds (`first_tape` of
results/stageG/stageG/stage_g_runs.csv). Partners: `--n` cells (registered: 256) drawn uniformly without replacement from a
snapshot of the organism's own world (runs/stageG/<label>_L<L>_st128_k4_s<seed>.soup_<name>.u8.br, loaded with
manuscript/figures/soup_stills.py), numpy default_rng seeded with [20261010, 2, L, seed, step]:
  registered set   L = 16 and 20: step 5,000; L = 50 and 64: the fixed-schedule snapshot (soup_t*) nearest 2 × t_rep,
                   ties to the earlier snapshot;
  t10000           L = 16 also step 10,000 (registered);
  sensitivity      (not scored) the later snapshot where 2 × t_rep is equidistant from two; and, at L = 50 and 64, the
                   'emergence' snapshot (step tq_10 of summary.json) where it is strictly nearer 2 × t_rep than any soup_t*.
One encounter per partner exactly as in exec_trace.py: organism in the first half, 128 instructions, traced executor
(algocell_exp.exectrace.execute_pairs_traced, exec_positions). Measures: H(Y | X = x), the plug-in entropy in bits of the
offspring strings over the encounters (exec_trace.entropy_bits), and entered, the share of encounters in which any partner
address is fetched as instruction stream. Descriptive (not scored): copied (offspring a ≥ 0.75 copy at the best cyclic shift),
partner positions fetched per encounter, distinct partners and their plug-in entropy (the offspring is a function of the
partner, so H(Y | X = x) ≤ H(partners) in the sample), and kin, the share of partners that are a ≥ 0.75 match to the organism
at the best cyclic shift.

Uniform-partner values are read from the reports (results/exectrace/per_replicator.csv: H_bits, entered_frac;
results/biology/individuality/per_replicator.csv: H_bits, the source of the manuscript's "3.9 bits at L = 16") and
recomputed here with each report's own partner seed (256 uniform partners) as a check that this pipeline reproduces them.

Scores: X2-1, at L = 16 the median in-soup inflow (registered set, step 5,000) is at least 1 bit. X2-2, entered ≥ 0.5 for at
least 90% of the 80 first replicators (registered set). Writes per_world.csv, sensitivity.csv and INFLOW_INSOUP.md.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "manuscript", "figures"))
from algocell_exp import assay as A  # noqa: E402
from algocell_exp import exectrace as X  # noqa: E402
from exec_trace import entropy_bits, parse  # noqa: E402
import soup_stills as ss  # noqa: E402

R = os.path.join(HERE, "results")
STAGE_G = os.path.join(R, "stageG", "stageG", "stage_g_runs.csv")
EXECTRACE = os.path.join(R, "exectrace", "per_replicator.csv")
INDIV = os.path.join(R, "biology", "individuality", "per_replicator.csv")
RUNS = os.path.join(HERE, "runs", "stageG")
STEPS = 128
N_UNIFORM = 256
SEED0 = 20261010


def stem(w) -> str:
    return os.path.join(RUNS, f"{w['label']}_L{int(w['L'])}_st128_k4_s{int(w['seed'])}")


def choose_sets(w) -> tuple[list[tuple[str, str, int, str]], list[tuple[str, str, int, str]], dict]:
    """(registered sets, sensitivity sets, notes); each set is (set_name, snapshot name, step, path). Exits if a registered
    snapshot is missing: the registration is not changed silently."""
    L, t_rep = int(w["L"]), int(w["t_rep"])
    files = ss.snapshot_files(stem(w))
    fixed = {n: (st, f) for n, st, f in files if n.startswith("t")}
    reg, sens, notes = [], [], {}

    def need(name):
        if name not in fixed:
            sys.exit(f"registered snapshot {name} missing for L{L} s{int(w['seed'])} ({stem(w)}): stopping")
        return fixed[name]

    if L in (16, 20):
        st, f = need("t5000")
        reg.append(("registered", "t5000", st, f))
        if L == 16:
            st, f = need("t10000")
            reg.append(("t10000", "t10000", st, f))
    else:
        target = 2 * t_rep
        steps = sorted((st, n) for n, (st, _) in fixed.items())
        dmin = min(abs(st - target) for st, _ in steps)
        near = [(st, n) for st, n in steps if abs(st - target) == dmin]
        st, n = near[0]                                   # ties to the earlier snapshot
        reg.append(("registered", n, st, fixed[n][1]))
        notes["target_step"] = target
        notes["tie"] = len(near) > 1
        for st2, n2 in near[1:]:
            sens.append(("tie_later", n2, st2, fixed[n2][1]))
        em = [(n, s, f) for n, s, f in files if n == "emergence" and s is not None]
        if em and abs(em[0][1] - target) < dmin:
            sens.append(("emergence", "emergence", int(em[0][1]), em[0][2]))
    return reg, sens, notes


def encounter(t: np.ndarray, P: np.ndarray) -> dict:
    """One encounter of organism t against every partner row of P, as in exec_trace.py."""
    n, L = P.shape
    pairs = np.concatenate([np.repeat(t[None, :], n, 0), P], axis=1)
    res, masks = X.execute_pairs_traced(pairs, L, STEPS, zero_halts=False)
    ex = X.exec_positions(masks, 2 * L)
    off = res[:, L:]
    after, _ = A._best_shift_rows(off, np.repeat(t[None, :], n, 0))
    kin, _ = A._best_shift_rows(P, np.repeat(t[None, :], n, 0))
    return {"H_bits": max(0.0, entropy_bits(off)), "entered_frac": float(ex[:, L:].any(axis=1).mean()),
            "copied_frac": float((after >= 0.75).mean()), "partner_exec_mean": float(ex[:, L:].sum(axis=1).mean()),
            "n_unique_offspring": int(len(np.unique(off, axis=0))), "n_unique_partners": int(len(np.unique(P, axis=0))),
            "H_partners": max(0.0, entropy_bits(P)), "kin_frac": float((kin >= 0.75).mean()),
            "partner_identical_frac": float((P == t[None, :]).all(axis=1).mean())}


def draw(path: str, L: int, seed: int, step: int, n: int) -> np.ndarray:
    soup = ss.load(path, L)
    assert soup.shape == (20000, L), (path, soup.shape)
    rng = np.random.default_rng([SEED0, 2, L, seed, step])
    return soup[rng.choice(len(soup), size=n, replace=False)]


def rng_range(s: pd.Series, fmt: str = "{:.2f}") -> str:
    return f"{fmt.format(s.median())} ({fmt.format(s.min())}–{fmt.format(s.max())})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R, "inflow_insoup"))
    ap.add_argument("--n", type=int, default=256, help="in-soup partners per organism (registered: 256)")
    ap.add_argument("--only", default=None, help="L:seed, e.g. 16:2001 (smoke test)")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if not a.yes:
        sys.exit("refusing to run: pass --yes after pausing the browser simulation (local GPU)")
    os.makedirs(a.out, exist_ok=True)
    G = pd.read_csv(STAGE_G)
    ET = pd.read_csv(EXECTRACE)
    ET = ET[(ET.stage == "G") & (ET.which == "first")].set_index(["L", "seed"])
    IN = pd.read_csv(INDIV)
    IN = IN[(IN.machine == "z80") & (IN.which == "first")].copy()
    IN["L"] = IN.group.astype(int)
    IN["seed"] = IN.world.str.extract(r"_s(\d+)$")[0].astype(int)
    IN = IN.set_index(["L", "seed"])
    if a.only:
        Lo, so = (int(x) for x in a.only.split(":"))
        G = G[(G.L == Lo) & (G.seed == so)]
    rows, srows = [], []
    for _, w in G.iterrows():
        L, seed = int(w["L"]), int(w["seed"])
        t = parse(w["first_tape"])
        assert t.size == L
        et, iv = ET.loc[(L, seed)], IN.loc[(L, seed)]
        assert et["tape"] == w["first_tape"] and iv["tape_hex"] == w["first_tape"].replace(" ", "")
        # uniform partners, recomputed with each report's own seed (check)
        U_et = np.random.default_rng([20261009, L, seed]).integers(0, 256, size=(N_UNIFORM, L), dtype=np.uint8)
        U_in = np.random.default_rng([20261008, 0, L, seed, 0]).integers(0, 256, size=(N_UNIFORM, L), dtype=np.uint8)
        u_et, u_in = encounter(t, U_et), encounter(t, U_in)
        rec = {"L": L, "seed": seed, "t_rep": int(w["t_rep"]), "first_tape": w["first_tape"],
               "H_uniform": float(iv["H_bits"]), "H_uniform_exectrace": float(et["H_bits"]),
               "entered_uniform": float(et["entered_frac"]), "copied_uniform": float(et["copied_frac"]),
               "H_uniform_recomp": u_in["H_bits"], "H_uniform_exectrace_recomp": u_et["H_bits"],
               "entered_uniform_recomp": u_et["entered_frac"],
               "uniform_check_ok": bool(np.isclose(u_in["H_bits"], iv["H_bits"]) and np.isclose(u_et["H_bits"], et["H_bits"])
                                        and np.isclose(u_et["entered_frac"], et["entered_frac"])
                                        and np.isclose(u_et["copied_frac"], et["copied_frac"]))}
        reg, sens, notes = choose_sets(w)
        rec["target_step"] = notes.get("target_step")
        rec["tie"] = notes.get("tie", False)
        for name, snap, step, path in reg:
            m = encounter(t, draw(path, L, seed, step, a.n))
            pre = "soup" if name == "registered" else "soup10k"
            rec[f"{pre}_snapshot"] = snap
            rec[f"{pre}_step"] = step
            rec.update({f"{k}_{pre}": v for k, v in m.items()})
        for name, snap, step, path in sens:
            m = encounter(t, draw(path, L, seed, step, a.n))
            srows.append({"L": L, "seed": seed, "set": name, "snapshot": snap, "step": step, "target_step": notes.get("target_step"),
                          "registered_snapshot": rec["soup_snapshot"], **m})
        rows.append(rec)
        print(f"L{L:2d} s{seed} uniform H {rec['H_uniform']:.2f}/{rec['H_uniform_exectrace']:.2f} (recomp {u_in['H_bits']:.2f}/{u_et['H_bits']:.2f}, ok={rec['uniform_check_ok']}) "
              f"entered {rec['entered_uniform']:.2f} | soup {rec['soup_snapshot']} H {rec['H_bits_soup']:.2f} entered {rec['entered_frac_soup']:.2f} "
              f"kin {rec['kin_frac_soup']:.2f} uniqP {rec['n_unique_partners_soup']}"
              + (f" | t10000 H {rec['H_bits_soup10k']:.2f} entered {rec['entered_frac_soup10k']:.2f}" if L == 16 else "")
              + (f" | sens {[(s['set'], s['snapshot'], round(s['H_bits'], 2), s['entered_frac']) for s in srows if s['L'] == L and s['seed'] == seed]}" if sens else ""),
              flush=True)
    T = pd.DataFrame(rows)
    S = pd.DataFrame(srows)
    T.to_csv(os.path.join(a.out, "per_world.csv"), index=False)
    S.to_csv(os.path.join(a.out, "sensitivity.csv"), index=False)
    report(T, S, a)


def report(T: pd.DataFrame, S: pd.DataFrame, a) -> None:
    n = len(T)
    L16 = T[T.L == 16]
    lines = ["# X2 — inflow and openness against partners from the soup (generated by `inflow_insoup.py`; do not edit)", "",
             f"Registration: `REVISION_PREREG.md`, X1–X3, X2 (committed at 244616b). Organisms: the first heritable replicator of "
             f"each Stage G world (n = {n}). In-soup partners: {a.n} cells drawn uniformly without replacement from a snapshot "
             f"of the organism's own world (numpy default_rng, seed [{SEED0}, 2, L, seed, step]); step 5,000 at L = 16 and 20, the "
             f"fixed-schedule snapshot nearest 2 × t_rep at L = 50 and 64 (ties to the earlier), and step 10,000 at L = 16. One "
             f"encounter per partner, organism first, {STEPS} instructions, traced executor (`algocell_exp.exectrace`), as in "
             f"`exec_trace.py`. H = plug-in entropy (bits) of the offspring over the encounters; entered = share of encounters in "
             f"which any partner address is fetched as instruction stream. Uniform-partner values: H from "
             f"`results/biology/individuality/per_replicator.csv` (the manuscript's figures) and from "
             f"`results/exectrace/per_replicator.csv`, entered from the latter. Table: `per_world.csv`; sensitivity: `sensitivity.csv`. "
             f"Descriptive columns (not registered): copied = share of offspring that are a ≥ 0.75 copy of the organism at the best "
             f"cyclic shift; kin share = share of partners that are themselves a ≥ 0.75 match to the organism's own tape at the best "
             f"cyclic shift (a narrow, tape-level notion of kin, not a replicator class); distinct partners and H(partners), the "
             f"plug-in entropy of the partner strings, which bounds H in the sample (the offspring is a function of the partner).", ""]
    ok = T.uniform_check_ok.all()
    lines.append(f"Check: the uniform-partner H and entered recomputed by this script with each report's own partner seed "
                 f"{'reproduce the reported values in all ' + str(n) + ' worlds' if ok else 'DO NOT reproduce the reported values in ' + str(int((~T.uniform_check_ok).sum())) + ' worlds'}.")
    lines.append("")
    # scores
    lines += ["## Scores", ""]
    if len(L16):
        med = L16.H_bits_soup.median()
        med10 = L16.H_bits_soup10k.median()
        x21 = "met" if med >= 1 else "not met"
        lines.append(f"- **X2-1** (at L = 16 the median in-soup inflow is at least 1 bit): median H = {med:.3f} bits over {len(L16)} "
                     f"worlds (step 5,000) → **{x21}**. With the step-10,000 partners: median {med10:.3f} bits "
                     f"({'met' if med10 >= 1 else 'not met'} at that step).")
    k = int((T.entered_frac_soup >= 0.5).sum())
    need = int(np.ceil(0.9 * n))
    x22 = "met" if k >= 0.9 * n else "not met"
    lines.append(f"- **X2-2** (pointer enters the partner in at least half of the in-soup encounters for at least 90% of first "
                 f"replicators): {k} of {n} ({k / n:.1%}; threshold {need}) → **{x22}**.")
    if len(L16):
        alt = T.entered_frac_soup.where(T.L != 16, T.entered_frac_soup10k)
        k2 = int((alt >= 0.5).sum())
        both = T.entered_frac_soup.ge(0.5) & (T.L.ne(16) | T.entered_frac_soup10k.ge(0.5))
        lines.append(f"  With the step-10,000 partners at L = 16 in place of step 5,000: {k2} of {n}; requiring both L = 16 sets: "
                     f"{int(both.sum())} of {n}.")
    lines.append("")
    # by length
    lines += ["## By length: uniform vs in-soup (median, range min–max)", "",
              "| L | n | partners (in-soup) | H uniform (individuality) | H uniform (exectrace) | H in-soup | entered uniform | entered in-soup | copied uniform | copied in-soup | kin share of partners | distinct partners of " + str(a.n) + " | H(partners) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for L, d in T.groupby("L"):
        snaps = ", ".join(f"{s} ×{c}" for s, c in d.soup_snapshot.value_counts().sort_index().items())
        lines.append(f"| {L} | {len(d)} | {snaps} | {rng_range(d.H_uniform)} | {rng_range(d.H_uniform_exectrace)} | {rng_range(d.H_bits_soup)} | "
                     f"{rng_range(d.entered_uniform)} | {rng_range(d.entered_frac_soup)} | {rng_range(d.copied_uniform)} | {rng_range(d.copied_frac_soup)} | "
                     f"{rng_range(d.kin_frac_soup)} | {rng_range(d.n_unique_partners_soup, '{:.0f}')} | {rng_range(d.H_partners_soup)} |")
        if L == 16:
            lines.append(f"| 16 | {len(d)} | t10000 | {rng_range(d.H_uniform)} | {rng_range(d.H_uniform_exectrace)} | {rng_range(d.H_bits_soup10k)} | "
                         f"{rng_range(d.entered_uniform)} | {rng_range(d.entered_frac_soup10k)} | {rng_range(d.copied_uniform)} | {rng_range(d.copied_frac_soup10k)} | "
                         f"{rng_range(d.kin_frac_soup10k)} | {rng_range(d.n_unique_partners_soup10k, '{:.0f}')} | {rng_range(d.H_partners_soup10k)} |")
    lines.append("")
    lines.append("Per-world drop in inflow (H uniform (individuality) − H in-soup, registered set): " + "; ".join(
        f"L = {L} median {(d.H_uniform - d.H_bits_soup).median():.2f} bits" for L, d in T.groupby("L")) + ".")
    lines.append("")
    # reading
    lines += ["## Reading (as registered)", ""]
    if len(L16):
        lines.append("- X2-1 " + ("met: no restriction is triggered (the registered reading applies only if X2-1 is not met)."
                                  if L16.H_bits_soup.median() >= 1 else
                                  "not met: the text says that the partner dependence of the open replicator's offspring largely vanishes "
                                  "among its own kind, and the claim is restricted to random partners."))
    lines.append("- X2-2 " + ("met: no restriction is triggered (the registered reading applies only if X2-2 is not met)."
                              if k >= 0.9 * n else "not met: \"open\" is restricted to random partners."))
    lines.append("")
    # sensitivity
    lines += ["## Sensitivity (not scored)", ""]
    tie = T[T.tie.astype(bool)]
    lines.append(f"- Ties (2 × t_rep equidistant from two fixed-schedule snapshots; registered set takes the earlier): "
                 + (", ".join(f"L = {r.L} seed {r.seed} (2 × t_rep = {int(r.target_step)})" for r in tie.itertuples()) if len(tie) else "none") + ".")
    if len(S):
        for r in S.itertuples():
            base = T[(T.L == r.L) & (T.seed == r.seed)].iloc[0]
            if r.set == "tie_later":
                lines.append(f"  - L = {r.L} seed {r.seed}: {base.soup_snapshot} H {base.H_bits_soup:.2f}, entered {base.entered_frac_soup:.2f}; "
                             f"{r.snapshot} H {r.H_bits:.2f}, entered {r.entered_frac:.2f}.")
        em = S[S.set == "emergence"]
        if len(em):
            j = T.set_index(["L", "seed"])
            alt = T.copy()
            for r in em.itertuples():
                m = (alt.L == r.L) & (alt.seed == r.seed)
                alt.loc[m, "H_bits_soup"] = r.H_bits
                alt.loc[m, "entered_frac_soup"] = r.entered_frac
            lines.append(f"- Emergence snapshot (step tq_10 of summary.json; not a fixed-schedule snapshot) strictly nearer 2 × t_rep than "
                         f"any fixed one in {len(em)} of {int((T.L >= 50).sum())} worlds at L = 50 and 64. Substituting it there: "
                         + "; ".join(f"L = {L} H in-soup {rng_range(d.H_bits_soup)}, entered {rng_range(d.entered_frac_soup)}" for L, d in alt[alt.L >= 50].groupby("L"))
                         + f"; X2-2 count {int((alt.entered_frac_soup >= 0.5).sum())} of {n}. Per world: "
                         + "; ".join(f"L{r.L} s{r.seed} {j.loc[(r.L, r.seed), 'soup_snapshot']}→emergence@{r.step}: H {j.loc[(r.L, r.seed), 'H_bits_soup']:.2f}→{r.H_bits:.2f}, "
                                     f"entered {j.loc[(r.L, r.seed), 'entered_frac_soup']:.2f}→{r.entered_frac:.2f}" for r in em.itertuples()) + ".")
    out = "\n".join(lines)
    print(out)
    open(os.path.join(a.out, "INFLOW_INSOUP.md"), "w").write(out + "\n")


if __name__ == "__main__":
    main()
