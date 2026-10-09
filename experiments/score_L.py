"""Score Stage L (ten-million-step extensions at L = 16 and 20) against REVISION_PREREG.md §L.

.venv/bin/python score_L.py [--cap results/capacity_time_L/capacity_over_time.csv] [--runs results/stageL/stageL/stage_g_runs.csv]
                            [--out results/stageL/SCORING.md]

L1: in >= 5 of 10 worlds at each length the dominant tape at 10,000,000 steps has more transmissible sites than the closed
    dominant at the reference step (300,000 at L = 16; 1,000,000 at L = 20). If the dominant at the reference step is still
    open (pointer enters the partner), the first later snapshot whose dominant is heritable and pointer-closed is the reference.
L2: every such recovery is in a block-copy lineage (LDIR `ed b0` / LDDR `ed b8` in the tape) and its transmissible sites are
    copied-but-unexecuted positions (reported as n_sites_unexecuted of n_sites).
L3 (as registered): no return-based or push-based dominant (any heritable dominant without a block copy) has more than one
    transmissible site at any snapshot. Also reported restricted to pointer-closed non-block dominants, since the open pusher
    of L = 20 can hold a few sites before closure.
Kill: L1 failing at both lengths with the final dominant holding <= 1 transmissible site in >= 8 of 10 worlds at both lengths.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
REF = {16: 300_000, 20: 1_000_000}
FINAL = 10_000_000
KEY_STEPS = [300_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000, 7_500_000, 10_000_000]


def is_block(tape: str) -> bool:
    t = str(tape).lower()
    return "ed b0" in t or "ed b8" in t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", default=os.path.join(R, "capacity_time_L", "capacity_over_time.csv"))
    ap.add_argument("--runs", default=os.path.join(R, "stageL", "stageL", "stage_g_runs.csv"))
    ap.add_argument("--out", default=os.path.join(R, "stageL", "SCORING.md"))
    a = ap.parse_args()
    C = pd.read_csv(a.cap).drop_duplicates(["stage", "L", "seed", "step"]).sort_values(["L", "seed", "step"])
    C["herit"] = C["ctrl_herit"].fillna(False).astype(bool)
    runs = pd.read_csv(a.runs) if os.path.exists(a.runs) else None
    rows = []
    for (L, seed), d in C.groupby(["L", "seed"]):
        d = d.set_index("step")
        fin = d.loc[FINAL] if FINAL in d.index else d.iloc[-1]
        closed_steps = [s for s in d.index if d.loc[s, "herit"] and d.loc[s, "entered_frac"] < 0.5]
        ref_step = REF[int(L)]
        ref_kind = "reference step"
        if ref_step in d.index and d.loc[ref_step, "herit"] and d.loc[ref_step, "entered_frac"] < 0.5:
            ref = d.loc[ref_step]
        else:
            later = [s for s in closed_steps if s >= ref_step]
            if later:
                ref_step, ref_kind = later[0], "first closed snapshot after the reference step"
                ref = d.loc[ref_step]
            elif closed_steps:
                ref_step, ref_kind = closed_steps[0], "first closed snapshot (before the reference step)"
                ref = d.loc[ref_step]
            else:
                ref, ref_step, ref_kind = None, None, "never closed"
        first_closed = closed_steps[0] if closed_steps else None
        rec = (ref is not None) and (fin["n_sites"] > ref["n_sites"])
        traj = " ".join(f"{s // 1000}k:{int(d.loc[s, 'n_sites']) if s in d.index and pd.notna(d.loc[s, 'n_sites']) else '-'}" for s in KEY_STEPS)
        rows.append({"L": int(L), "seed": int(seed), "first_closed_step": first_closed, "ref_step": ref_step, "ref_kind": ref_kind,
                     "ref_sites": None if ref is None else int(ref["n_sites"]), "ref_tape": None if ref is None else str(ref["tape"])[:23],
                     "final_tape": str(fin["tape"])[:23], "final_share": float(fin["share"]), "final_herit": bool(fin["herit"]),
                     "final_entered": float(fin["entered_frac"]) if pd.notna(fin["entered_frac"]) else None,
                     "final_sites": int(fin["n_sites"]) if pd.notna(fin["n_sites"]) else None,
                     "final_unexec": int(fin["n_sites_unexecuted"]) if pd.notna(fin["n_sites_unexecuted"]) else None,
                     "final_capacity": float(fin["capacity_bits"]) if pd.notna(fin["capacity_bits"]) else None,
                     "final_block": is_block(fin["tape"]), "recovered": bool(rec), "sites_trajectory": traj})
    T = pd.DataFrame(rows).sort_values(["L", "seed"])
    out = ["# Stage L (ten million steps, L = 16 and 20, ten worlds each): pre-registered scoring", ""]
    verdicts = []
    for L in (16, 20):
        t = T[T.L == L]
        n_rec = int(t.recovered.sum()); n_closed = int(t.ref_step.notna().sum())
        n_le1 = int((t.final_sites.fillna(0) <= 1).sum())
        l2_ok = all((r.final_block and (r.final_unexec or 0) >= max(1, (r.final_sites or 0)) * 0.5) for r in t[t.recovered].itertuples())
        snaps = C[(C.L == L) & C.herit & C.tape.notna()]
        nonblock = snaps[~snaps.tape.astype(str).apply(is_block)]
        viol = nonblock[nonblock.n_sites > 1]
        viol_closed = viol[viol.entered_frac < 0.5]
        reopened = [int(r.seed) for r in t.itertuples() if r.ref_step is not None and r.final_entered is not None and r.final_entered >= 0.5]
        out.append(f"## L = {L}")
        out.append(f"- worlds that closed by 10M steps: {n_closed}/10 (first closed snapshot: " + ", ".join(f"{int(r.seed)}:{r.first_closed_step}" for r in t.itertuples() if r.first_closed_step is not None) + ")")
        out.append(f"- L1 recovery (more transmissible sites at 10M than the closed reference dominant): {n_rec}/10 (needs >= 5) -> {'MET' if n_rec >= 5 else 'NOT MET'}")
        out.append(f"- L2 every recovery in a block-copy lineage with unexecuted sites: {'MET' if (n_rec > 0 and l2_ok) else ('n/a (no recovery)' if n_rec == 0 else 'NOT MET')}")
        out.append(f"- L3 (as registered) no return- or push-based dominant with > 1 transmissible site at any snapshot: {'MET' if viol.empty else 'NOT MET'} "
                   f"({len(viol)} violating snapshots" + ("" if viol.empty else ": " + "; ".join(f"seed {int(r.seed)} step {int(r.step)} {'closed' if r.entered_frac < 0.5 else 'open'} sites {int(r.n_sites)}" for r in viol.head(12).itertuples())) + ")")
        out.append(f"- L3 restricted to pointer-closed non-block dominants: {'MET' if viol_closed.empty else 'NOT MET'} ({len(viol_closed)} violating snapshots)")
        out.append(f"- worlds whose dominant is open again at 10M after having closed: {reopened if reopened else 'none'}")
        out.append(f"- final dominants with <= 1 transmissible site: {n_le1}/10 (kill condition component: >= 8 at both lengths)")
        out.append("")
        verdicts.append((L, n_rec, n_le1))
    killed = all(n_rec < 5 and n_le1 >= 8 for _, n_rec, n_le1 in verdicts)
    out.append(f"**Kill criterion** (L1 fails at both lengths with <= 1 site in >= 8/10 worlds at both): {'FIRED' if killed else 'not fired'}")
    out.append("")
    out.append("## Per world")
    out.append("")
    pd.set_option("display.width", 300)
    out.append("```")
    out.append(T[["L", "seed", "first_closed_step", "ref_step", "ref_sites", "ref_tape", "final_tape", "final_share", "final_entered", "final_sites", "final_unexec", "final_capacity", "final_block", "recovered"]].to_string(index=False))
    out.append("")
    out.append("transmissible sites of the most common tape at key steps (k = thousand steps; '-' no snapshot or not heritable):")
    for r in T.itertuples():
        out.append(f"L = {r.L} seed {r.seed}: {r.sites_trajectory}")
    out.append("```")
    if runs is not None:
        out.append("")
        out.append("## Culture-test table of the final dominants (stage_g pipeline)")
        out.append("```")
        cols = [c for c in ["L", "seed", "t_rep", "final_cf", "final_block", "final_copied", "final_damaged", "final_share"] if c in runs.columns]
        out.append(runs.sort_values(["L", "seed"])[cols].to_string(index=False))
        out.append("```")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(out) + "\n")
    T.to_csv(os.path.join(os.path.dirname(a.out), "scoring_L.csv"), index=False)
    print("\n".join(out))


if __name__ == "__main__":
    main()
