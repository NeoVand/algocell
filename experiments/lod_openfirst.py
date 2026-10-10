"""X3 (REVISION_PREREG.md, "X1–X3", X3; registered at 244616b): open first along the lines of descent.

On each sampled line of every closed world (the line_cells lines of line_records.npz; functional parents, lod_v6),
every distinct tape from step 0 to the line's founder is classed against 64 random partners exactly as in N3
(`lod_traj.classify_tapes`: copier if it copies >= 0.75 at the best cyclic shift in >= half the encounters; open if its
pointer fetches a partner byte in >= half, confined otherwise). The line's first copier is the earliest record whose tape
is an open or a confined copier.

The founder on each line is located with the frozen rule of `lod_chain.py` (the first confined copier after the newest
open copier older than the final >= 90%-confined stretch) and must be one of the founders in LOD_OUT/founders2.csv; the
closed worlds are the worlds with founders in that table.

Predictions (as registered):
  X3-1: in at least 90% of closed worlds the first copier on every sampled line is an open copier.
  X3-2: in no world does a confined copier on a line precede that line's first open copier by more than the median
        founder delay (79 steps).

Paths only from the environment (definitions unchanged):
    LOD_IND  (default runs/lod_v6_modal/lod_v6)   worlds L16_benign_s*/line_records.npz, lod.json
    LOD_OUT  (default results/lod)                founders2.csv in; OPEN_FIRST.md, open_first.csv and the class cache
                                                  open_first_classes.csv out

    .venv/bin/python lod_openfirst.py                    # all closed worlds
    .venv/bin/python lod_openfirst.py --seeds 31002      # smoke test on one world (writes *_smoke outputs)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
IND = os.environ.get("LOD_IND", os.path.join(HERE, "runs", "lod_v6_modal", "lod_v6"))   # paths only
OUT = os.environ.get("LOD_OUT", os.path.join(HERE, "results", "lod"))
from lod_traj import classify_tapes  # noqa: E402

L = 16
DELAY = 79          # registered: the median founder delay (steps), N3 first founders
OPEN, CONF, NONC = "open copier", "confined copier", "non-copier"
COPIERS = (OPEN, CONF)
CACHE = os.path.join(OUT, "open_first_classes.csv")
CACHE_COLS = ["tape", "copy", "enter", "exact", "class"]


def h(a):
    return " ".join(f"{int(x):02x}" for x in a)


def load_cache() -> dict:
    if not os.path.exists(CACHE):
        return {}
    C = pd.read_csv(CACHE, dtype={"tape": str, "class": str})
    return {r.tape: {"copy": r.copy, "enter": r.enter, "exact": r.exact, "class": r[5]} for r in C.itertuples()}


def classes_for(tapes: set[str], cache: dict) -> dict:
    """classify_tapes is per tape (fixed partner set: 64 partners, seed 3), so a per-tape cache is exact."""
    new = sorted(t for t in tapes if t not in cache)
    if new:
        cache.update(classify_tapes(new))
        pd.DataFrame([{"tape": t, **{k: v[k] for k in CACHE_COLS[1:]}} for t, v in sorted(cache.items())],
                     columns=CACHE_COLS).to_csv(CACHE, index=False)
    return {t: cache[t] for t in tapes}


def founder_index(cls_line: np.ndarray) -> int | None:
    """lod_chain.py's frozen founder rule on one line (index 0 = newest record); returns fq (index of F) or None."""
    n = len(cls_line)
    conf = cls_line == CONF
    frac = np.cumsum(conf) / np.arange(1, n + 1)
    ok = np.where(frac >= 0.9)[0]
    if not len(ok):
        return None
    tr = int(ok.max())
    lo = next((q for q in range(tr, n) if cls_line[q] == OPEN), None)
    if lo is None:
        return None
    fq = next((q for q in range(lo - 1, -1, -1) if cls_line[q] == CONF), None)
    return fq


def world(seed: int, F2: pd.DataFrame, cache: dict) -> tuple[list[dict], dict]:
    d = os.path.join(IND, f"L16_benign_s{seed}")
    z = dict(np.load(os.path.join(d, "line_records.npz")))
    res = json.load(open(os.path.join(d, "lod.json")))["result"]
    tapes = [h(t) for t in z["tape"]]
    C = classes_for(set(tapes), cache)                     # every record tape on the lines (superset of the window)
    cls = np.array([C[t]["class"] for t in tapes])
    steps = z["step"]
    known = {(int(r.F_step), r.F_tape): bool(r.first) for r in F2[F2.seed == seed].itertuples()}
    rows = []
    off = 0
    for li, (cell, n) in enumerate(zip(z["line_cells"], z["line_lengths"])):
        ids = z["line_ids"][off:off + n]
        off += n
        cl = cls[ids]
        fq = founder_index(cl)
        row = {"seed": seed, "t_close": res["t_close"], "line": li, "cell": int(cell), "line_records": int(n)}
        if fq is None:
            row.update({"founder_found": False})
            rows.append(row)
            continue
        F = int(ids[fq])
        key = (int(steps[F]), tapes[F])
        win = list(range(n - 1, fq - 1, -1))                     # oldest (step 0) ... founder, inclusive
        wcls = [cl[q] for q in win]
        first = next(q for q, c in zip(win, wcls) if c in COPIERS)   # exists: F itself is a confined copier
        fo = next((q for q, c in zip(win, wcls) if c == OPEN), None)
        fc = next((q for q, c in zip(win, wcls) if c == CONF), None)
        st = lambda q: int(steps[ids[q]]) if q is not None else None  # noqa: E731
        # sensitivity (not registered): first copier when tapes at the copy threshold (copy rate exactly 0.5) or tapes
        # that never copy exactly are not counted as copiers
        s1 = next(q for q, c in zip(win, wcls) if c in COPIERS and C[tapes[ids[q]]]["copy"] > 0.5)
        s2 = next(q for q, c in zip(win, wcls) if c in COPIERS and C[tapes[ids[q]]]["exact"] > 0)
        first_cls = cl[first]
        lead = (st(fo) - st(first)) if (first_cls == CONF and fo is not None) else None
        row.update({
            "founder_found": True, "founder_in_founders2": key in known, "founder_is_first": known.get(key),
            "F_step": key[0], "F_tape": key[1], "step0_kind": int(z["kind"][ids[-1]]), "step0_step": st(n - 1),
            "window_records": len(win), "window_distinct_tapes": len({tapes[ids[q]] for q in win}),
            "window_classes": ";".join(f"{k}:{sum(1 for c in wcls if c == k)}" for k in (OPEN, CONF, NONC)),
            "first_copier_step": st(first), "first_copier_class": first_cls, "first_copier_tape": tapes[ids[first]],
            "first_copier_kind": int(z["kind"][ids[first]]), "first_copier_rec": int(ids[first]),
            "first_copier_copy": C[tapes[ids[first]]]["copy"], "first_copier_enter": C[tapes[ids[first]]]["enter"],
            "first_copier_exact": C[tapes[ids[first]]]["exact"],
            "first_open_step": st(fo), "first_open_tape": tapes[ids[fo]] if fo is not None else None,
            "first_conf_step": st(fc), "first_conf_tape": tapes[ids[fc]] if fc is not None else None,
            "first_conf_is_founder": fc == fq,
            "confined_lead_steps": lead,
            "sens_copy_gt_half_class": cl[s1], "sens_copy_gt_half_step": st(s1),
            "sens_exact_class": cl[s2], "sens_exact_step": st(s2)})
        rows.append(row)
    return rows, res


def summarise(P: pd.DataFrame) -> pd.DataFrame:
    out = []
    for seed, E in P.groupby("seed", sort=True):
        G = E[E.founder_found]
        conf_first = G[G.first_copier_class == CONF]
        out.append({"seed": seed, "t_close": E.t_close.iloc[0], "lines": len(E), "lines_with_founder": len(G),
                    "founders_on_lines": G.F_step.nunique() if len(G) else 0,
                    "founders_not_in_table": int((~G.founder_in_founders2.astype(bool)).sum()),
                    "all_first_open": bool(len(G) == len(E) and (G.first_copier_class == OPEN).all()),
                    "lines_first_confined": len(conf_first),
                    "earliest_open_step": G.first_open_step.min(), "earliest_conf_step": G.first_conf_step.min(),
                    "first_copier_records": G.first_copier_rec.nunique(),
                    "sens_copy_gt_half_all_open": bool(len(G) == len(E) and (G.sens_copy_gt_half_class == OPEN).all()),
                    "sens_exact_all_open": bool(len(G) == len(E) and (G.sens_exact_class == OPEN).all()),
                    "max_conf_lead": conf_first.confined_lead_steps.max() if len(conf_first) else None,
                    "first_copier_step_median": G.first_copier_step.median(),
                    "first_copier_tapes": " | ".join(f"{t} ({c})" for t, c in G.first_copier_tape.value_counts().head(3).items())})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="", help="comma-separated subset (smoke test; outputs get a _smoke suffix)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    F2 = pd.read_csv(os.path.join(OUT, "founders2.csv"))
    seeds = sorted(F2.seed.unique().tolist())                # closed worlds = worlds with founders
    if a.seeds:
        want = [int(s) for s in a.seeds.split(",")]
        seeds = [s for s in seeds if s in want]
    tag = "_smoke" if a.seeds else ""
    cache = load_cache()
    t0 = time.time()
    rows, wall = [], {}
    for s in seeds:
        t1 = time.time()
        r, _ = world(s, F2, cache)
        rows += r
        wall[s] = time.time() - t1
        print(f"world {s}: {len(r)} lines, {wall[s]:.1f} s", flush=True)
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(OUT, f"open_first{tag}.csv"), index=False)
    W = summarise(P)
    n = len(W)
    k1 = int(W.all_first_open.sum())
    x31 = k1 >= 0.9 * n
    bad = W[W.max_conf_lead.fillna(-1) > DELAY]
    x32 = len(bad) == 0
    nolines = P[~P.founder_found]
    md = ["# X3 — open first along the lines of descent (generated by `lod_openfirst.py`)", "",
          f"Input `{os.path.relpath(IND, HERE)}`; founders `{os.path.relpath(os.path.join(OUT, 'founders2.csv'), HERE)}`; "
          f"{n} closed worlds (worlds with founders), {len(P)} sampled lines.", "",
          "On each sampled line every distinct tape from step 0 to the line's founder is classed against 64 random partners as in N3 "
          "(`lod_traj.classify_tapes`). The first copier is the earliest record on the line whose tape is an open or a confined copier. "
          "The founder on each line is located with `lod_chain.py`'s rule and checked against `founders2.csv`.", "",
          "## Scores", "",
          f"- **X3-1** (first copier on every sampled line open in ≥ 90% of closed worlds): {k1} of {n} worlds "
          f"({k1 / n:.3f}) — **{'met' if x31 else 'not met'}**.",
          f"- **X3-2** (no world in which a confined copier on a line precedes that line's first open copier by more than {DELAY} steps): "
          f"{len(bad)} such worlds — **{'met' if x32 else 'not met'}**.", "",
          "## Checks", "",
          f"- lines with a founder by `lod_chain.py`'s rule: {int(P.founder_found.sum())} of {len(P)}"
          + (f" (without: {', '.join(f'{r.seed}/line {r.line}' for r in nolines.itertuples())})" if len(nolines) else ""),
          f"- founders on lines that are in `founders2.csv`: {int(P[P.founder_found].founder_in_founders2.sum())} of {int(P.founder_found.sum())}; "
          f"distinct founders reached by the lines: {P[P.founder_found].groupby('seed').F_step.nunique().sum()} "
          f"(founders2.csv: {len(F2[F2.seed.isin(seeds)])})",
          f"- every line starts at a step-0 record: {bool((P[P.founder_found].step0_step == 0).all())}",
          f"- distinct tapes classed (cache `{os.path.basename(CACHE)}`): {len(cache)}", "",
          "## Per world", "",
          "| world | closure step | lines (with founder) | founders on lines | first copier open on every line | lines whose first copier is confined | earliest open copier (step) | earliest confined copier (step) | largest confined lead (steps) | median step of first copier | distinct first-copier records |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in W.itertuples():
        md.append(f"| {r.seed} | {r.t_close} | {r.lines} ({r.lines_with_founder}) | {r.founders_on_lines} | {'yes' if r.all_first_open else 'no'} | "
                  f"{r.lines_first_confined} | {int(r.earliest_open_step)} | {int(r.earliest_conf_step)} | "
                  f"{'—' if pd.isna(r.max_conf_lead) else int(r.max_conf_lead)} | {r.first_copier_step_median:.0f} | {r.first_copier_records} |")
    G = P[P.founder_found]
    md += ["", "## First copiers", "",
           f"- first-copier classes over all lines: " + ", ".join(f"{k} {v}" for k, v in G.first_copier_class.value_counts().items()),
           f"- step of the first copier: median {G.first_copier_step.median():.0f} (range {G.first_copier_step.min()}–{G.first_copier_step.max()})",
           f"- steps from the first open copier to the founder: median {(G.F_step - G.first_open_step).median():.0f} "
           f"(range {(G.F_step - G.first_open_step).min()}–{(G.F_step - G.first_open_step).max()})",
           f"- lines on which the earliest confined copier is the founder itself: {int(G.first_conf_is_founder.sum())} of {len(G)}",
           "- first-copier tapes (lines; copy, enter and exact rates against the 64 partners): " + "; ".join(
               f"`{t}` ({c}; {cache[t]['copy']:.3f}, {cache[t]['enter']:.3f}, {cache[t]['exact']:.3f})" for t, c in G.first_copier_tape.value_counts().items()),
           f"- the 64 lines of a world share their first-copier record: distinct first-copier records per world, median "
           f"{W.first_copier_records.median():.0f} (range {W.first_copier_records.min()}–{W.first_copier_records.max()})"]
    md += ["", "## Sensitivity (not registered; the scores above use the registered classes only)", "",
           f"- first copier counted only if its copy rate exceeds 0.5 (excludes tapes exactly at the threshold): first copier open on every "
           f"line in {int(W.sens_copy_gt_half_all_open.sum())} of {n} worlds",
           f"- first copier counted only if it also copies itself exactly at least once: first copier open on every line in "
           f"{int(W.sens_exact_all_open.sum())} of {n} worlds"]
    CF = G[G.first_copier_class == CONF]
    if len(CF):
        md += ["", "## Lines whose first copier is confined", "",
               "| world | line | first copier (step, tape) | first open copier (step) | lead (steps) | founder step |", "|---|---|---|---|---|---|"]
        for r in CF.itertuples():
            md.append(f"| {r.seed} | {r.line} | {r.first_copier_step} `{r.first_copier_tape}` | {r.first_open_step} | {r.confined_lead_steps} | {r.F_step} |")
    md += ["", f"Runtime: {time.time() - t0:.0f} s for {n} worlds (classification cached per distinct tape)."]
    open(os.path.join(OUT, f"OPEN_FIRST{tag}.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
