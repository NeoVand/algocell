"""N3 analysis: lines of descent, the founding events C* of self-confined copying, and their exact byte provenance.

    .venv/bin/python lod_analyse.py --indir runs/lod_modal/lod      # results/lod/LOD.md, lod_worlds.csv, cstar_events.csv

Byte provenance of an encounter (exact, by differential replay): the encounter is re-run with each of the 2L input
bytes perturbed to two other values; an output byte is a copy of input byte p when it takes the perturbed value in both
re-runs. Output bytes with no such source are unchanged (equal to the same position before and not dependent on any
input in this way) or synthesised (written from register contents, e.g. zeros pushed from the stack).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from algocell_exp import exectrace as X  # noqa: E402

CONF, OPEN = 1, 2


def hx(s):
    return np.array([int(b, 16) for b in s.split()], np.uint8) if s else None


def provenance(A0: np.ndarray, B0: np.ndarray, L: int, zero_halts: bool = False):
    """Return post (2L,), src (2L,) with src[q] = input position copied into output q, -1 if none (value-transparent test)."""
    pre = np.concatenate([A0, B0])
    P = 2 * L
    batch = [pre]
    for p in range(P):
        for v in ((int(pre[p]) + 1) % 256, int(pre[p]) ^ 0x80):
            x = pre.copy()
            x[p] = v
            batch.append(x)
    batch = np.stack(batch)
    res, _ = X.execute_pairs_traced(batch, L, 128, zero_halts=zero_halts)
    post = res[0]
    src = np.full(P, -1)
    for p in range(P):
        r1, r2 = res[1 + 2 * p], res[2 + 2 * p]
        v1, v2 = batch[1 + 2 * p][p], batch[2 + 2 * p][p]
        hit = (r1 == v1) & (r2 == v2) & (post == pre[p])
        for q in np.where(hit & (src == -1))[0]:
            src[q] = p
    return post, src


def executed_positions(tape: np.ndarray, L: int, n: int = 64, seed: int = 0) -> np.ndarray:
    """Positions of `tape` fetched as instruction stream when it executes as A against n random partners (union)."""
    rng = np.random.default_rng(seed)
    P = rng.integers(0, 256, (n, L), dtype=np.uint8)
    _, m = X.execute_pairs_traced(np.concatenate([np.tile(tape, (n, 1)), P], 1), L, 128)
    ex = X.exec_positions(m, 2 * L)
    return ex[:, :L].any(0), ex[:, L:].any(1).mean()


def founding(ev: dict, L: int):
    """Origin of each byte of the C* record's tape, by alignment: from the copier (the executor's tape at the copy's
    cyclic shift, or for novel records the minor parent at its stored shift), kept from the overwritten tape, both
    (ambiguous), or neither (new: mutation or bytes synthesised from registers). Also: the positions C* itself executes
    when it runs as A against random partners (its core), and how often it enters the partner."""
    cs = ev["cstar"]
    tape = hx(cs["tape"])
    kind = cs["kind"]
    out = {"kind": kind, "step": cs["step"], "tape": cs["tape"]}
    sh = int(cs.get("shift", 0))
    if kind == "copy":
        src, old = np.roll(hx(cs["p1_tape"]), sh), hx(cs["other"])
    elif kind == "copyA":
        src, old = np.roll(hx(cs["p1_tape"]), sh), hx(cs["other"])
    elif kind == "novel":
        # p1 = major parent, p2 = minor; which one is the cell's own old tape is not stored: take the one aligned at shift 0
        t1, t2 = hx(cs["p1_tape"]), hx(cs["p2_tape"])
        a1, a2 = (t1 == tape).sum(), (t2 == tape).sum()
        old, src = (t1, np.roll(t2, sh)) if a1 >= a2 else (t2, np.roll(t1, sh))
    elif kind in ("mut", "damage"):
        par = hx(cs["p1_tape"])
        out["changed_positions"] = [int(i) for i in np.where(par != tape)[0]]
        out["parent_flag"] = cs.get("p1_flag")
        out["parent_kind"] = cs.get("p1_kind")
        src, old = None, par
    else:
        return out
    cats = []
    for q in range(L):
        fs = src is not None and tape[q] == src[q]
        fo = tape[q] == old[q]
        cats.append("both" if fs and fo else "copier" if fs else "kept" if fo else "new")
    ex_self, ent = executed_positions(tape, L)
    core = [int(i) for i in np.where(ex_self)[0]]
    out.update({"copier_tape": None if src is None else src.tobytes().hex(" "), "old_tape": old.tobytes().hex(" "), "sources": "".join(c[0] for c in cats),
                "n_copier": cats.count("copier"), "n_kept": cats.count("kept"), "n_both": cats.count("both"), "n_new": cats.count("new"),
                "core_positions": core, "core_copier": sum(cats[i] == "copier" for i in core), "core_kept": sum(cats[i] == "kept" for i in core),
                "core_both": sum(cats[i] == "both" for i in core), "core_new": sum(cats[i] == "new" for i in core), "cstar_enters_partner": float(ent)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=os.path.join(HERE, "runs", "lod_modal", "lod"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "lod"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W, E = [], []
    for f in sorted(glob.glob(os.path.join(a.indir, "L16_benign_s*", "lod.json"))):
        D = json.load(open(f))
        R = D["result"]
        L = R["L"]
        lines = D["lines"]["lines"]
        with_c = [ln for ln in lines if "cstar" in ln]
        desc = [ln for ln in with_c if ln["open_before_cstar"] > 0]
        win = [ln for ln in with_c if ln["open_within_window"] > 0]
        keys = {}
        for ln in with_c:
            k = (ln["cstar"]["step"], ln["cstar"]["cell"])
            keys.setdefault(k, []).append(ln)
        rates = pd.DataFrame(D.get("base_rates", []))
        for k, lns in keys.items():
            ev = lns[0]
            fd = founding(ev, L)
            br = rates.iloc[(rates.t - k[0]).abs().argmin()] if len(rates) else None
            E.append({"seed": R["seed"], "cstar_step": k[0], "cstar_cell": k[1], "n_lines": len(lns), "kind": ev["cstar"]["kind"],
                      "executor_flag": ev["cstar"].get("p1_flag"), "open_before": ev["open_before_cstar"], "open_within_2000": ev["open_within_window"],
                      "base_rate_open_within_2000": float(br["open_within"]) if br is not None else np.nan,
                      **{f"f_{kk}": (json.dumps(vv) if isinstance(vv, (list, dict)) else vv) for kk, vv in fd.items()}})
        W.append({"seed": R["seed"], "t_close": R["t_close"], "t_end": R["t_end"], "V2_mismatches": R["V2_mismatches"], "lines": len(lines),
                  "lines_with_cstar": len(with_c), "descent_lines": len(desc), "descent_within_2000": len(win), "distinct_cstar": len(keys),
                  "N3_1_world": len(desc) > len(with_c) / 2 if with_c else None})
    Wd, Ed = pd.DataFrame(W), pd.DataFrame(E)
    Wd.to_csv(os.path.join(a.out, "lod_worlds.csv"), index=False)
    Ed.to_csv(os.path.join(a.out, "cstar_events.csv"), index=False)
    closed = Wd[Wd.t_close.notna()]
    lines = ["# N3: lines of descent of the first self-confined copiers (generated by `lod_analyse.py`)", "",
             f"Worlds: {len(Wd)}; closed (confined share of copy events > 0.5 for 2,000 steps): {len(closed)}.", "",
             f"**N3-1** (registered: ≥ 15 closing worlds in which most sampled lines pass through an open copier before C*): "
             f"{int(closed.N3_1_world.sum())} of {len(closed)} closing worlds.", "",
             "| seed | closure step | end | lines | lines with C* | through an open copier | ... within 2,000 steps | distinct C* |", "|---|---|---|---|---|---|---|---|"]
    for r in Wd.itertuples():
        lines.append(f"| {r.seed} | {r.t_close} | {r.t_end} | {r.lines} | {r.lines_with_cstar} | {r.descent_lines} | {r.descent_within_2000} | {r.distinct_cstar} |")
    lines += ["", "## Founding events C* (one row per distinct C*; lines that share it are counted)", "",
              "| seed | step | lines | kind | executor flag | open copiers before | base rate | bytes copier / kept / both / new | core (executed) copier / kept / both / new | sources by position | tape |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in (Ed.sort_values(["seed", "n_lines"], ascending=[True, False]).itertuples() if len(Ed) else []):
        g = lambda k: getattr(r, k, "")  # noqa: E731
        lines.append(f"| {r.seed} | {r.cstar_step} | {r.n_lines} | {r.kind} | {r.executor_flag} | {r.open_before} | {r.base_rate_open_within_2000:.2f} | "
                     f"{g('f_n_copier')} / {g('f_n_kept')} / {g('f_n_both')} / {g('f_n_new')} | {g('f_core_copier')} / {g('f_core_kept')} / {g('f_core_both')} / {g('f_core_new')} | `{g('f_sources')}` | `{r.f_tape}` |")
    open(os.path.join(a.out, "LOD.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
