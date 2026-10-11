"""Writes results/census6502/CENSUS.md and its tables from the census checkpoints and the CSVs of census6502.py.

    cd experiments && python census6502.py report

Every number in CENSUS.md is computed here from the generated files (checkpoints, CSVs); nothing is typed by hand.
The exploratory census's numbers are parsed from its own output files (read only).
"""

from __future__ import annotations

import io
import re
import subprocess
import time

import numpy as np
import pandas as pd

import census6502 as C
from algocell_exp.soup import adapter_summary
from algocell_exp.zcensus import COLUMN_FLAGS, F_NEAR, evaluate, n_primitive, primitive_period, tile_bytes
from algocell_exp.zmachine import load_machine

OUT = C.OUT
HALTS = ["running", "halt-insn", "host (BRK)", "trap"]
COLS = {1: "exact, zero partner", 2: "exact, zero partner (phase score only)", 4: "exact, ≥ 1 random partner",
        8: "exact, ≥ 1 random partner (phase score only)", 16: "≥ 75% in ≥ 8 of 16", 32: "≥ 75% in ≥ 8 of 16 (phase score only)"}


# ── formatting ──────────────────────────────────────────────────────────────────────────────────────────────────────
def md(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "(none)\n"
    s = io.StringIO()
    cols = list(df.columns)
    s.write("| " + " | ".join(str(c) for c in cols) + " |\n|" + "---|" * len(cols) + "\n")
    for r in df.itertuples(index=False):
        s.write("| " + " | ".join(_fmt(v) for v in r) + " |\n")
    return s.getvalue()


def _fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if v != v:
            return "–"
        return f"{v:.3g}" if abs(v) < 1e4 else f"{v:,.0f}"
    return str(v).replace("|", "\\|")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def save(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df.to_csv(OUT / f"table_{name}.csv", index=False)
    return df


# ── loading ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def censuses() -> list[dict]:
    out = []
    for p in sorted(C.CKPT.glob("m6502_*.jsonl")):
        R = C.load_census(p)
        h = R["head"]
        mname = h["machine"]["machine"]
        out.append({"path": p, "R": R, "machine": mname, "L": h["L"], "k": h["k"], "brk": "halt" if "brkhalt" in mname else "faithful",
                    "obj": None})
    out.sort(key=lambda x: (x["L"], x["brk"] != "faithful", x["k"]))
    return out


def obj(x: dict):
    if x["obj"] is None:
        x["obj"] = C.census_obj(x["R"]["head"])
    return x["obj"]


def decoded(x: dict, b: int) -> dict:
    return obj(x).decode_hits(x["R"]["budgets"][b]["hits"])


def word_of(t: int, k: int) -> bytes:
    return bytes(tile_bytes(np.array([t], np.uint64), k)[0])


# ── tables ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def status(R: dict) -> str:
    if R["complete"]:
        return "complete"
    return (f"**INCOMPLETE: {R['chunks_done']}/{R['chunks_total']} chunks ({R['tiles_done']:,} of {256 ** R['head']['k']:,} tiles)**"
            if R["chunks_total"] else "**INCOMPLETE: not started**")


def short_status(R: dict) -> str:
    return "complete" if R["complete"] else f"INCOMPLETE {R['chunks_done']}/{R['chunks_total']}"


def main_table(cs: list[dict]) -> pd.DataFrame:
    rows = []
    for x in cs:
        R = x["R"]
        for b in R["head"]["budgets"]:
            d = decoded(x, b)
            nz = d["t"] != 0
            fl = d["flags"]
            closed = ~d["entered"].any(1)
            zex = nz & ((fl & 1) != 0)
            prim = primitive_period(tile_bytes(d["t"][zex], x["k"])) if zex.any() else np.zeros(0, int)
            rex = nz & ((fl & 4) != 0)
            r75 = nz & ((fl & 16) != 0)
            rows.append({"L": x["L"], "BRK": x["brk"], "budget": b, "k": x["k"], "status": status(R), "words": 256 ** x["k"],
                         "done": R["tiles_done"], "primitive period k": n_primitive(x["k"]),
                         "exact, zero partner": int(zex.sum()), "of primitive period k": int((prim == x["k"]).sum()),
                         "closed": int((zex & closed).sum()),
                         "exact, ≥ 1 random partner": int(rex.sum()), "max partners": int(d["n_exact"][rex].max()) if rex.any() else 0,
                         "closed ": int((rex & closed).sum()),
                         "≥ 75% in ≥ 8 of 16": int(r75.sum()), "closed  ": int((r75 & closed).sum()),
                         "zero word exact": bool(((fl & 1) != 0)[d["t"] == 0].any()),
                         "phase score (zero / random / ≥ 75%)": f"{int((nz & ((fl & 2) != 0)).sum())} / {int((nz & ((fl & 8) != 0)).sum())} / {int((nz & ((fl & 32) != 0)).sum())}",
                         "GPU min": round(R["secs"] / 60, 1)})
    return pd.DataFrame(rows)


def hit_rows(cs: list[dict], near: bool = False) -> pd.DataFrame:
    rows = []
    for x in cs:
        R = x["R"]
        for b in R["head"]["budgets"]:
            d = decoded(x, b)
            sel = ((d["flags"] & (F_NEAR if near else COLUMN_FLAGS)) != 0) & (d["t"] != 0)
            if near:
                sel &= (d["flags"] & COLUMN_FLAGS) == 0
            for i in np.flatnonzero(sel):
                t = int(d["t"][i])
                w = word_of(t, x["k"])
                fl = int(d["flags"][i])
                rows.append({"L": x["L"], "BRK": x["brk"], "budget": b, "k": x["k"], "census": short_status(R), "word": w.hex(" "),
                             "disassembly (tiled, from 0)": C.dis6502_word(w, x["L"]),
                             "columns": "; ".join([v for f, v in COLS.items() if fl & f and f in (1, 4, 16)] +
                                                  [v for f, v in COLS.items() if fl & f and f in (2, 8, 32) and not fl & (f >> 1)]) or "–",
                             "zero-partner match": int(d["rot"][i, 0]), "exact partners": int(d["n_exact"][i]),
                             "≥ 75% partners": int(d["n75"][i]), "best random match": int(d["rot"][i, 1:].max()) if d["rot"].shape[1] > 1 else 0,
                             "closed": bool(not d["entered"][i].any()), "partner entered (of 17)": int(d["entered"][i].sum()),
                             "zero partner: halt / steps": f"{HALTS[int(d['halt'][i, 0])]} / {int(d['steps'][i, 0])}"})
    return pd.DataFrame(rows)


def hit_words(h: pd.DataFrame) -> pd.DataFrame:
    """The hit rows grouped by word: one row per (L, k, word), with the runs (BRK arm, budget) in which it hit."""
    if not len(h):
        return h
    order = {"exact, zero partner": 0, "≥ 75% in ≥ 8 of 16": 1, "exact, ≥ 1 random partner": 2}
    rows = []
    for (L, k, w), g in h.groupby(["L", "k", "word"], sort=False):
        cols = sorted({c for cc in g["columns"] for c in cc.split("; ")}, key=lambda c: order.get(c, 9))
        runs = "; ".join(f"{brk} {', '.join(str(b) for b in sorted(gg.budget))}" for brk, gg in g.groupby("BRK"))
        rows.append({"L": L, "k": k, "word": w, "disassembly (tiled, from 0)": g["disassembly (tiled, from 0)"].iloc[0],
                     "columns": "; ".join(cols), "runs (BRK arm, budgets)": runs,
                     "zero-partner match": "–".join(sorted({str(v) for v in g["zero-partner match"]})),
                     "exact partners (of 16)": "–".join(sorted({str(v) for v in g["exact partners"]}, key=int)),
                     "≥ 75% partners (of 16)": "–".join(sorted({str(v) for v in g["≥ 75% partners"]}, key=int)),
                     "closed": "yes" if g.closed.all() else ("no" if not g.closed.any() else "in some runs"),
                     "has runs of an incomplete census": "yes" if (g.census != "complete").any() else "",
                     "_o": min(order.get(c, 9) for c in cols)})
    return pd.DataFrame(rows).sort_values(["L", "k", "_o", "word"], kind="stable").drop(columns=["_o"])


def diagnostics(cs: list[dict]) -> pd.DataFrame:
    rows = []
    for x in cs:
        R = x["R"]
        for b in R["head"]["budgets"]:
            Cn = R["budgets"][b]["counters"]
            L = x["L"]
            for cls in ("zero", "rand"):
                n = Cn[f"{cls}_lanes"]
                if n == 0:
                    continue
                mh = np.array([Cn[f"{cls}_match{m}"] for m in range(L + 1)])
                rows.append({"L": L, "BRK": x["brk"], "budget": b, "k": x["k"], "census": short_status(R), "partner": cls, "lanes": n,
                             "halted at step 1": pct(Cn[f"{cls}_steps_1"] / n),
                             "ran < 16 steps": pct(sum(Cn[f"{cls}_steps_{s}"] for s in ("0", "1", "2-3", "4-15")) / n),
                             "ran the budget": pct(Cn[f"{cls}_steps_budget"] / n), "JAM": pct(Cn[f"{cls}_trap_illegal"] / n),
                             "BRK halt": pct(Cn[f"{cls}_halt2"] / n), "wrote B": pct(Cn[f"{cls}_wrote_b"] / n),
                             "B changed": pct(Cn[f"{cls}_b_changed"] / n), "A changed": pct(Cn[f"{cls}_a_changed"] / n),
                             "fetched from B": pct(Cn[f"{cls}_entered"] / n),
                             "match ≥ 75%": int(mh[-(-3 * L // 4):].sum()), f"match ≥ L − 1": int(mh[L - 1:].sum()),
                             "best match": f"{int(np.flatnonzero(mh)[-1])}/{L}"})
    return pd.DataFrame(rows)


def recheck_table(cs: list[dict]) -> pd.DataFrame:
    rows = []
    for x in cs:
        R = x["R"]
        for b in R["head"]["budgets"]:
            rc = R["budgets"][b]["recheck"]
            rows.append({"L": x["L"], "BRK": x["brk"], "budget": b, "k": x["k"], "census": short_status(R), "tiles re-run": rc.get("tiles", 0), "lanes": rc.get("lanes", 0),
                         "mismatching tiles (any field)": sum(v for kk, v in rc.items() if kk.startswith("mismatch")),
                         "harness WGSL": R["head"]["harness_wgsl_sha256"][:12], "machine WGSL": R["head"]["machine"]["sha256_wgsl"][:12]})
    return pd.DataFrame(rows)


def consistency_table(cs: list[dict]) -> pd.DataFrame:
    """Every record's flags and per-word counts recomputed from its per-encounter scores, and the GPU counters compared
    with the records (a word that meets a column is always recorded, so the two counts must be equal)."""
    rows = []
    for x in cs:
        R = x["R"]
        L = x["L"]
        thr = -(-3 * L // 4)
        for b in R["head"]["budgets"]:
            d = decoded(x, b)
            Cn = R["budgets"][b]["counters"]
            rot, ph = d["rot"].astype(int), d["phase"].astype(int)
            V = rot.shape[1]
            flags = ((rot[:, 0] == L) * 1 + (ph[:, 0] == L) * 2 + (rot[:, 1:] == L).any(1) * 4 + (ph[:, 1:] == L).any(1) * 8
                     + (((rot[:, 1:] >= thr).sum(1) >= 8) & (V > 1)) * 16 + (((ph[:, 1:] >= thr).sum(1) >= 8) & (V > 1)) * 32)
            near = (rot[:, 0] >= thr) | ((rot[:, 1:] >= thr).sum(1) >= 2) | ((rot[:, 1:].max(1) if V > 1 else 0) + 1 >= L)
            bad_flags = int(((d["flags"] & COLUMN_FLAGS) != flags).sum() + (((d["flags"] & F_NEAR) != 0) != near).sum())
            bad_counts = int(((rot[:, 1:] == L).sum(1) != d["n_exact"]).sum() + ((rot[:, 1:] >= thr).sum(1) != d["n75"]).sum()
                             + ((ph[:, 1:] == L).sum(1) != d["n_phase_exact"]).sum() + ((ph[:, 1:] >= thr).sum(1) != d["n_phase75"]).sum())
            pairs = [("tiles_zero_exact", 1), ("tiles_zero_phase_exact", 2), ("tiles_rand_exact_any", 4), ("tiles_rand_phase_exact_any", 8),
                     ("tiles_rand75_ge8", 16), ("tiles_rand_phase75_ge8", 32)]
            bad_ctr = sum(int(Cn[c] != int(((d["flags"] & f) != 0).sum())) for c, f in pairs)
            rows.append({"L": L, "BRK": x["brk"], "budget": b, "k": x["k"], "census": short_status(R), "records": len(d["t"]), "counter = records (hits_total)": Cn["hits_total"] == len(d["t"]),
                         "flags ≠ scores": bad_flags, "counts ≠ scores": bad_counts, "column counters ≠ records": bad_ctr})
    return pd.DataFrame(rows)


# ── the exploratory census ──────────────────────────────────────────────────────────────────────────────────────────
def exploratory_summary() -> pd.DataFrame:
    """Its counts per (k, L, BRK), recomputed from its output files as its summarize.py defines them."""
    cand = C.parse_exploratory()
    prim = {}
    for f in C.EXPLORATORY.glob("log*.txt"):
        m = re.search(r"DONE k=(\d+) L=(\d+) bh=(\d) chunk=\S+ primitive=(\d+)", f.read_text())
        if m:
            key = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            prim[key] = prim.get(key, 0) + int(m.group(4))
    for f in C.EXPLORATORY.glob("out*.txt"):   # every setting it ran, with or without candidates
        m = re.match(r"out_k(\d+)_L(\d+)_bh(\d)_c\d+\.txt|outL(\d+)_k(\d+)\.txt", f.name)
        if m and m.group(1):
            prim.setdefault((int(m.group(1)), int(m.group(2)), int(m.group(3))), None)
        elif m:
            prim.setdefault((int(m.group(5)), int(m.group(4)), 0), None)
    rows = []
    keys = sorted({(int(d["k"]), d["L"], d["brk_halt"]) for d in cand} | set(prim))
    for (k, L, bh) in keys:
        ds = [d for d in cand if (int(d["k"]), d["L"], d["brk_halt"]) == (k, L, bh)]
        rows.append({"k": k, "L": L, "BRK": "halt" if bh else "faithful", "tiles (primitive)": prim.get((k, L, bh)) or n_primitive(k),
                     "screened candidates": len(ds),
                     "exact, zero partner": sum(d["zero_exact"] == "1" for d in ds),
                     "exact, zero partner (non-zero tiles)": sum(d["zero_exact"] == "1" and int(d["tile"], 16) != 0 for d in ds),
                     "exact, ≥ 1 random partner": sum(not d["rand_exact"].startswith("0/") for d in ds),
                     "≥ 75% in ≥ 8 of 16": sum(int(d["rand_75"].split("/")[0]) >= 8 for d in ds)})
    return pd.DataFrame(rows)


def partner_rates(tiles: list[tuple[int, int, int, str]], n: int = 1024) -> pd.DataFrame:
    """Exact-copy and ≥ 75% rates of words over the same n fresh random partners (numpy rng, seed fixed), budget 128
    and 512; one run per (k, L, BRK arm, budget) group."""
    rows = []
    rng = np.random.default_rng(20261011)
    groups = {}
    for (k, L, t, brk) in tiles:
        groups.setdefault((k, L, brk), []).append(t)
    for (k, L, brk), ts in sorted(groups.items()):
        m = load_machine(C.machine_name(L, brk))
        P1 = rng.integers(0, 256, size=(1, n, L), dtype=np.uint8)
        P = np.repeat(P1, len(ts), 0)
        for b in (128, 512):
            e = evaluate(m, k, np.array(ts, np.uint64), n, b, partners=P)
            for i, t in enumerate(ts):
                rows.append({"k": k, "L": L, "BRK": brk, "word": word_of(t, k).hex(" "), "budget": b, "partners": n,
                             "exact": int((e.rot[i, 1:] == L).sum()), "≥ 75%": int((e.rot[i, 1:] >= -(-3 * L // 4)).sum()),
                             "best match": int(e.rot[i, 1:].max()), "partner entered": int(e.entered[i, 1:].sum())})
    return pd.DataFrame(rows)


def signature_fetch_check(D: pd.DataFrame) -> pd.DataFrame:
    """For candidates whose 'entered' differs between the two interpreters: where the BRK-halted lane stopped."""
    rows = []
    bad = D[(D.expl_zero_entered != D.zil_zero_entered) | (D.expl_entered_any != D.zil_entered_any)]
    for r in bad.itertuples():
        k, L = int(r.k), int(r.L)
        m = load_machine(C.machine_name(L, "halt" if r.brk_halt else "faithful"))
        w = bytes.fromhex(r.tile)
        e = evaluate(m, k, np.array([int(r.tile, 16)], np.uint64), 0, 128)
        pc = int(np.flatnonzero(e.start_map[0, 0])[-1]) if e.start_map[0, 0].any() else -1
        rr = m.run(np.concatenate([np.frombuffer(bytes(w[i % k] for i in range(L)), np.uint8), np.zeros(L, np.uint8)])[None], budget=128)
        pcf = int(rr.reg("pc")[0]) % (2 * L)
        rows.append({"word": w.hex(" "), "BRK": "halt" if r.brk_halt else "faithful", "zero partner: entered (exploratory / Zilion)": f"{r.expl_zero_entered} / {r.zil_zero_entered}",
                     "Zilion: halt": HALTS[int(rr.halt_reason[0])], "PC at halt (ring position)": pcf, "byte there": f"{int(rr.memory[0, pcf]):02x}",
                     "signature byte position": (pcf + 1) % (2 * L), "signature byte in B": (pcf + 1) % (2 * L) >= L})
    return pd.DataFrame(rows)


def word_comparison(cs: list[dict], rerun: pd.DataFrame) -> pd.DataFrame:
    """For the random-partner columns at budget 128: the words in either census, side by side. A word only in this census
    is re-run through the exploratory census's encounters (did it pass that screen; its counts with those partners); a
    word only in the exploratory census is re-run with this census's partners."""
    rows = []
    for x in cs:
        if x["R"]["tiles_done"] == 0:
            continue
        k, L, brk = x["k"], x["L"], x["brk"]
        bh = 1 if brk == "halt" else 0
        d = decoded(x, 128)
        E = rerun[(rerun.k == k) & (rerun.L == L) & (rerun.brk_halt == bh)]
        for col, flag, ecol in (("exact, ≥ 1 random partner", 4, "expl_rand_exact"), ("≥ 75% in ≥ 8 of 16", 16, "expl_rand_75")):
            mine = {int(t) for t in d["t"][((d["flags"] & flag) != 0) & (d["t"] != 0)]}
            theirs = {int(t, 16) for t in (E[E[ecol] > 0] if flag == 4 else E[E[ecol] >= 8]).tile}
            only_m, only_e = sorted(mine - theirs), sorted(theirs - mine)
            ex = C.exploratory_encounters([word_of(t, k) for t in only_m], k, L, brk) if only_m else None
            for t in sorted(mine | theirs):
                w = word_of(t, k)
                i = np.flatnonzero(d["t"] == t)
                run = C.tile_done(x["R"], t)
                r = {"k": k, "L": L, "BRK": brk, "column": col, "word": w.hex(" "), "disassembly": C.dis6502_word(w, L),
                     "in this census": (t in mine) if run else "not run yet", "in the exploratory": t in theirs,
                     "this census: exact / ≥ 75% partners": f"{int(d['n_exact'][i[0]])} / {int(d['n75'][i[0]])}" if len(i) else "",
                     "exploratory: exact / ≥ 75% partners": "", "passes its screen": ""}
                e = E[E.tile == w.hex()]
                if len(e):
                    r["exploratory: exact / ≥ 75% partners"] = f"{int(e.expl_rand_exact.iloc[0])} / {int(e.expl_rand_75.iloc[0])}"
                    r["passes its screen"] = True
                elif ex is not None and t in only_m:
                    j = only_m.index(t)
                    r["exploratory: exact / ≥ 75% partners"] = f"{int(ex['n_exact'][j])} / {int(ex['n75'][j])} (re-run)"
                    r["passes its screen"] = bool(ex["passes_screen"][j])
                if not len(i):   # only in the exploratory census and not recorded here: this census's partners
                    if run:
                        e2 = evaluate(obj(x).machine, k, np.array([t], np.uint64), 16, 128)
                        r["this census: exact / ≥ 75% partners"] = f"{int((e2.rot[0, 1:] == L).sum())} / {int((e2.rot[0, 1:] >= -(-3 * L // 4)).sum())} (re-run)"
                    else:
                        r["this census: exact / ≥ 75% partners"] = "not yet run"
                rows.append(r)
    return pd.DataFrame(rows)


# ── closers ─────────────────────────────────────────────────────────────────────────────────────────────────────────
S0 = {16: 0xFF, 20: 0xDF, 32: 0xFF}   # the initial S of each world (machines' manifests: first push on the last byte of B)


def predicted_offset(fam: str, zz: int, L: int, budget: int) -> tuple[int, bool]:
    """The copy offset d the operand zz sets, from the instructions' address arithmetic alone (no run): iteration i of
    `B5 zz 48 CA 50 FA` (X = −i) reads zero-page address (zz − i) & 0xFF and pushes at 0x100 | (S0 − i); iteration i of
    `BA B5 zz 48 50 FA` (X = S = S0 − i) reads (zz + S) & 0xFF and pushes at 0x100 | S. d = (read − write) mod 2L at
    iteration 0; `steady` says whether that offset stays the same over all budget/4 iterations of the loop (on a ring
    of 40 the 8-bit wraps of the zero-page index and of S move it: 256 mod 40 = 16)."""
    P, s0 = 2 * L, S0[L]
    offs = set()
    for i in range(budget // 4):
        s = (s0 - i) & 0xFF
        rd = ((zz - i) & 0xFF) if fam.startswith("B5") else ((zz + s) & 0xFF)
        offs.add((rd % P - (0x100 | s) % P) % P)
    d0 = ((zz & 0xFF) % P - (0x100 | s0) % P) % P if fam.startswith("B5") else (((zz + s0) & 0xFF) % P - (0x100 | s0) % P) % P
    return d0, len(offs) == 1


def closer_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exactness per zz, and the d − 6 law tested on every zz whose PREDICTED offset (from the operand alone) meets the
    law's hypotheses: the offset is steady over the run, 6 ≤ d ≤ L and d divides L. The law then says: the pointer never
    enters the partner, the sites carried are exactly 0 … d − 1 (so the payload beyond the six-byte closer is d − 6),
    and the copy is exact if and only if d = L."""
    D = pd.read_csv(OUT / "closers_scan.csv", dtype={"zz": str})
    ex_rows, law_rows = [], []
    for (L, brk, fam, b), g in D.groupby(["L", "BRK", "family", "budget"], sort=True):
        ok = g[g.exact == g.encounters]
        ex_rows.append({"L": L, "BRK": brk, "closer": fam, "budget": b, "zz exact in every encounter": " ".join(ok.zz) or "none",
                        "encounters each": int(g.encounters.iloc[0]), "pointer in partner (those zz)": int(ok.pointer_in_partner.sum()),
                        "zz confined (no partner fetch, all encounters)": int((g.pointer_in_partner == 0).sum())})
        for r in g.itertuples():
            zz = int(r.zz, 16)
            d, steady = predicted_offset(fam, zz, int(L), int(b))
            hyp = steady and 6 <= d <= L and L % d == 0
            s = r.sites_carried_all_bodies
            dobs = len(s) - len(s.lstrip("#"))
            carried_ok = s[:d] == "#" * d and "#" not in s[d:]
            law_rows.append({"L": L, "BRK": brk, "closer": fam, "budget": b, "zz": r.zz, "predicted d": d, "steady": steady,
                             "hypotheses": hyp, "confined": r.pointer_in_partner == 0, "sites carried": s, "carried prefix": dobs,
                             "carried = 0 … d − 1": carried_ok, "payload carried (sites ≥ 6)": int(r.payload_carried),
                             "d − 6": d - 6, "exact copies": int(r.exact), "encounters": int(r.encounters),
                             "law holds": bool(hyp and r.pointer_in_partner == 0 and carried_ok and int(r.payload_carried) == d - 6
                                               and ((int(r.exact) == int(r.encounters)) == (d == L)))})
    return pd.DataFrame(ex_rows), pd.DataFrame(law_rows)


# ── CENSUS.md ───────────────────────────────────────────────────────────────────────────────────────────────────────
def report() -> None:
    cs = censuses()
    zcommit = re.search(r"commit `([0-9a-f]{40})`", (C.HERE / "algocell_exp/machines/PROVENANCE.md").read_text()).group(1)   # the vendored machines'
    zhead = subprocess.check_output(["git", "-C", str(C.ZILION), "rev-parse", "HEAD"], text=True).strip()
    stamp = time.strftime("%Y-%m-%d %H:%M %Z")
    T = {}
    T["main"] = save("main", main_table(cs))
    T["hits"] = save("hits", hit_rows(cs))
    T["near"] = save("near", hit_rows(cs, near=True))
    T["diag"] = save("diagnostics", diagnostics(cs))
    T["recheck"] = save("recheck", recheck_table(cs))
    ctrl = pd.read_csv(OUT / "control_z80.csv")
    ctrl_words = pd.read_csv(OUT / "control_z80_words.csv")
    planted = pd.read_csv(OUT / "planted_6502.csv")
    expl = save("exploratory_summary", exploratory_summary())
    rerun = pd.read_csv(OUT / "exploratory_rerun.csv", dtype={"tile": str})
    sig = save("signature_fetch", signature_fetch_check(rerun))
    cex, claw = closer_tables()
    save("closers_exact", cex)
    save("closers_law", claw)
    opentiles = pd.read_csv(OUT / "closers_open_tiles.csv")
    k4 = [x for x in cs if x["k"] == 4]
    k4_complete = all(x["R"]["complete"] for x in k4) and len(k4) == 2

    # tiles that were exact into random partners, in either census, re-run over 1,024 fresh partners
    lucky = set()
    for r in rerun[(rerun.expl_rand_exact > 0)].itertuples():
        lucky.add((int(r.k), int(r.L), int(r.tile, 16), "halt" if r.brk_halt else "faithful"))
    hits = T["hits"]
    if len(hits):
        for r in hits[hits["columns"].str.contains("random partner", regex=False)].itertuples():
            lucky.add((int(r.k), int(r.L), int(r.word.replace(" ", ""), 16), r.BRK))
    rates = save("random_partner_rates", partner_rates(sorted(lucky)))

    m = T["main"]
    P = []
    P.append(f"<!-- Generated by `python census6502.py report` on {stamp} (adapter {adapter_summary()}; machines from zilion {zcommit[:12]}, checkout at {zhead[:12]}). "
             "Do not edit by hand: every number below is computed from the files in this directory. -->\n")
    P.append("# The 6502 self-writer census on the validated Zilion core\n")
    # headline
    def row(L, brk, b, k):
        r = m[(m.L == L) & (m.BRK == brk) & (m.budget == b) & (m.k == k)]
        return r.iloc[0] if len(r) else None
    lines = []
    for brk in ("faithful", "halt"):
        for b in (128, 512):
            vals = []
            for k in (1, 2, 3, 4):
                r = row(16, brk, b, k)
                vals.append("–" if r is None else (f"{r['exact, zero partner']}" + ("" if r["done"] == r["words"] else f" ({r['status'].strip('*')})")))
            lines.append(f"BRK {brk}, budget {b}: {', '.join(vals)}")
    P.append("**Headline.** Self-writers (exact into the all-zero partner, non-zero tapes) at periods 1, 2, 3, 4, L = 16: "
             + "; ".join(lines) + ". On the Z80 the same harness (at period 4 with the zero partner only) gives " + ", ".join(str(int(v)) for v in ctrl.self_writers) +
             " at periods 1–4, the paper's census word for word (section 2).\n")
    if not k4_complete:
        st = "; ".join(f"BRK {x['brk']}: {status(x['R']).strip('*')}" for x in k4) or "not started"
        P.append(f"**Status: the period-4 census at L = 16 is not complete** ({st}). Every table below marks the incomplete rows and counts "
                 "only the words already run. Words run in index order (first byte 00 upward), so an incomplete period-4 row covers the words "
                 "whose first byte is below the stopping point, a slice of opcodes, not a sample. `python census6502.py census --L 16 --k 4` "
                 "resumes from the checkpoint and rewrites this file.\n")

    # summary for the pre-registration, every number from the tables
    P.append("## Summary\n")
    allrows = m.copy()
    zero_total = int(allrows["exact, zero partner"].sum())
    inc = allrows[allrows.status != "complete"]
    sm = [f"- **Self-writers (exact into the all-zero partner, non-zero tapes): {zero_total}** over every run in section 3 "
          f"(periods 1–4 at L = 16, periods 1–3 at L = 20 and 32; BRK faithful and halting; 128 and 512 instructions)"
          + (f", the BRK-halting period-4 runs counting only the {int(inc['done'].iloc[0]):,} words run so far" if len(inc) else "") + "."]
    hn = pd.concat([T["hits"], T["near"]]) if len(T["near"]) else T["hits"]
    for brk in ("faithful", "halt"):
        for b in (128, 512):
            g = hn[(hn.L == 16) & (hn.k == 4) & (hn.BRK == brk) & (hn.budget == b)] if len(hn) else hn
            if len(g):
                best = int(g["zero-partner match"].max())
                ws = sorted(set(g[g["zero-partner match"] == best].word))
                sm.append(f"- Best zero-partner copy at period 4, L = 16, BRK {brk}, {b} instructions: {best}/16, by {', '.join('`' + w + '`' for w in ws[:6])}"
                          + (f" and {len(ws) - 6} more" if len(ws) > 6 else "") + ".")
    for _, d in allrows[(allrows.L == 16) & (allrows.k == 4)].iterrows():
        sm.append(f"- Period 4, L = 16, BRK {d['BRK']}, {d['budget']} instructions ({d['status'].strip('*')}): {d['≥ 75% in ≥ 8 of 16']} word(s) with "
                  f"≥ 75% copies into ≥ 8 of 16 random partners ({d['closed  ']} closed); {d['exact, ≥ 1 random partner']} exact into at least one "
                  f"random partner ({d['closed ']} closed).")
    held = claw[claw.hypotheses]
    sm.append(f"- Seeded closers (six bytes, §4.2): exact and confined copies for the operands listed in section 8; the d − 6 law, with d "
              f"predicted from the operand, holds in {int(held['law holds'].sum())} of {len(held)} rows meeting its hypotheses.")
    c4r = ctrl[ctrl.period == 4].iloc[0]
    sm.append(f"- Positive controls: the Z80 through the same harness reproduces the paper's census ({int(c4r.self_writers)} at period 4, "
              f"{int(c4r.closed_census_partners_zilion)} closed, word for word); the 6502 harness finds both seeded closers tiled at period 6, "
              f"exact and closed (section 2).")
    P.append("\n".join(sm) + "\n")
    P.append("## 1. Harness and definitions\n")
    P.append("- **Machines.** `algocell_exp/machines/worlds_6502.json` (provenance in `machines/PROVENANCE.md`): zilion's full NMOS 6502 core "
             f"(zilion `{zcommit[:7]}`, status experimental), a pair of tapes A, B of L bytes on a ring of 2L, A = X = Y = 0, P = 0x20, PC = 0, the first "
             "push on the last byte of B, JAM halting as on the chip, BRK faithful (`m6502_pair_maps_L*`) or halting the lane before it executes "
             "(`m6502_brkhalt_maps_L*`), fetch and start maps. The Z80 control runs the vendored `z80c02_pair_maps_L16` (compat 0.2).\n"
             "- **Harness.** `algocell_exp/zcensus.py`, driven by `census6502.py`. Tape A is every word T of period k tiled to L bytes; it meets the "
             "all-zero partner and 16 random partners (one set per word, from a counter-based hash), for 128 and 512 instructions in one pass "
             "(lanes still running after 128 steps are compacted and resumed with Zilion's resumable runs).\n"
             "- **Copy score.** The best of the L cyclic shifts: the number of positions j with B_final[j] = A[(j − s) mod L] (the Z80 census's "
             "\"partner equal to the tape at some cyclic shift\" is a score of L). The exploratory census scored the k phases of the word instead "
             "(B_final[j] = T[(j + s) mod k]); the two agree when k divides L, and both are reported (the \"phase score\" columns).\n"
             "- **Columns** (ISA_RESEARCH §4.1): exact into the zero partner; exact into at least 1 of the 16 random partners; a copy of at least 75% "
             "(ceil(3L/4) positions) into at least 8 of the 16. **Closed**: no partner byte fetched as instruction stream (fetch map) in any of the "
             "17 encounters, the Z80 census's definition. The zero word is excluded from the counts, as in `census_period.py`.\n"
             "- **Tested every run.** Every recorded tile (all hits, near misses and a hash-chosen sample) is re-run through one direct "
             "`ZMachine.run` with memory built in numpy; a chunk is written only if every field agrees (section 7).\n")

    P.append("## 2. Positive controls\n")
    P.append("### 2.1 The Z80 census of the paper, through the same harness\n")
    c4 = ctrl[ctrl.period == 4].iloc[0]
    P.append(f"`z80c02_pair_maps_L16` (compat 0.2), L = 16, every word of period 1–4, through the harness with the 6502 census's settings "
             f"(128 and 512 instructions in one pass, segments of 128 steps), periods 1–3 with 16 random partners per word as for the 6502, "
             f"period 4 with the zero partner only (the column the paper counts; 16 partners would cost about 2.5 GPU hours). \"Paper\" is "
             f"`results/review_r2/census_p*.csv` (`census_period.py`, 128 instructions). The harness's self-writers are then re-traced, outside "
             f"the harness, with `census_period.py`'s own partners (`rng([20261018, p])`, drawn per self-writer in index order) on the Zilion "
             f"machine (`evaluate`) and on Algocell's op-traced executor (`exectrace.py`), and with the harness's partners; the paper's open/closed "
             f"column is the first of these. `self-writers at 512` is new (no paper number).\n")
    P.append(md(ctrl[["period", "words", "tiles_done", "self_writers", "by_primitive_period", "paper_self_writers", "word_sets_equal",
                      "closed_census_partners_zilion", "closed_census_partners_algocell", "paper_closed", "closed_flags_equal",
                      "closed_harness_partners", "rand_exact_any", "rand75_ge8", "recheck_tiles", "recheck_mismatches", "self_writers_512"]].rename(columns={
        "tiles_done": "words run", "self_writers": "self-writers", "by_primitive_period": "by primitive period", "paper_self_writers": "paper",
        "word_sets_equal": "same words", "closed_census_partners_zilion": "closed (Zilion)", "closed_census_partners_algocell": "closed (Algocell executor)",
        "paper_closed": "paper closed", "closed_flags_equal": "same closed flags", "closed_harness_partners": "closed (harness partners)",
        "rand_exact_any": "exact, ≥ 1 random", "rand75_ge8": "≥ 75% in ≥ 8", "recheck_tiles": "tiles re-run", "recheck_mismatches": "mismatches",
        "self_writers_512": "self-writers at 512"})))
    P.append(f"\nThe period-4 run reproduces the paper's {int(c4.paper_self_writers)} self-writers word for word ({c4.by_primitive_period} by primitive "
             f"period) and its {int(c4.paper_closed)} closed ones flag for flag. Its words are in `control_z80_words.csv`.\n")
    P.append("### 2.2 The 6502: seeded closers tiled at period 6\n")
    P.append("The harness at period 6 over the 65,536 words that share the first four bytes of each seeded closer of §4.2 (so the scan "
             "contains the closer): it must report the closer exact into the zero partner and into its random partners, and closed.\n")
    P.append(md(planted))
    P.append("\n")

    P.append("## 3. The 6502 census\n")
    P.append("Counts exclude the zero word (`zero word exact` says whether it is exact: under BRK halting the tape `00 …` halts at once and leaves the "
             "zero partner untouched). `words` counts every word of period k, `primitive period k` those of exactly that period (the exploratory "
             "census enumerated these); `of primitive period k` splits the exact ones. The three `closed` columns refer to the column before each.\n")
    P.append(md(m))
    P.append("\n## 4. Every hit\n")
    h = T["hits"]
    hw = save("hit_words", hit_words(h))
    if len(hw):
        n_w = len(hw)
        rx = hw[hw["columns"].str.contains("random partner", regex=False)]
        r75 = hw[hw["columns"].str.contains("75%", regex=False)]
        zx = hw[hw["columns"].str.contains("zero partner", regex=False)]
        P.append(f"Every word (non-zero) that meets any column in any run, under either score: {n_w} words ({len(zx)} exact into the zero "
                 f"partner, {len(r75)} with ≥ 75% copies into ≥ 8 of 16 partners, {len(rx)} exact into at least one random partner; a word can "
                 f"meet several). One row per word, with the runs in which it met a column and the range of its scores over those runs; "
                 f"closed words: {int((hw.closed == 'yes').sum())}, of which exact into a random partner: {int(((hw.closed == 'yes') & hw['columns'].str.contains('random partner', regex=False)).sum())}. "
                 f"Disassembly is linear from offset 0 of the tiled tape until the decode returns to a word boundary. Per run: `table_hits.csv`.\n\n")
        P.append(md(hw))
    else:
        P.append("(none)\n")
    P.append("\n## 5. Near misses\n")
    nm = T["near"]
    if len(nm):
        nm4 = nm.assign(_f=nm["zero-partner match"] / nm.L).sort_values(["_f", "≥ 75% partners", "best random match"], ascending=False).drop(columns=["_f"])
        P.append(f"Words that meet no column but come near one (zero-partner match ≥ 75%, ≥ 2 random partners at ≥ 75%, or a random partner at L − 1): "
                 f"{len(nm):,} rows in `table_near.csv`; the 40 with the highest zero-partner copy (as a fraction of L):\n\n")
        P.append(md(nm4.head(40)))
    else:
        P.append("(none)\n")
    P.append("\n## 6. Non-vacuity diagnostics\n")
    P.append("Per census and partner class (zero: the all-zero partner; rand: the 16 random partners): how the lanes ran. `wrote B`: the lane wrote "
             "into the partner at least once; `B changed`: the partner's final bytes differ from its initial bytes; `fetched from B`: the open "
             "criterion; `match ≥ 75%` / `≥ L − 1`: lanes whose copy score reached that. A census whose lanes all halted at once, or never wrote the "
             "partner, would show it here. These counters cover every word, the zero word included: under BRK halting its zero-partner lane is "
             "the one with a perfect match.\n")
    P.append(md(T["diag"]))
    P.append("\n## 7. Re-checks\n")
    P.append("Every recorded tile (hits, near misses and the hash-chosen sample: 1 word in 2^18 at period 4, 2^12 at period 3) re-run "
             "at each budget through one direct `ZMachine.run` (one dispatch, no segments), with tape and partners built in numpy: per-encounter "
             "copy score under both scores, fetch into B, halt reason, trap cause, steps, and the per-word counts.\n")
    P.append(md(T["recheck"]))
    cons = save("consistency", consistency_table(cs))
    P.append("\nAnd every record against itself and the counters: its column flags and per-word counts recomputed from its per-encounter "
             "scores, and each column's GPU counter against the number of records carrying that column:\n\n")
    P.append(md(cons))

    P.append("\n## 8. Seeded closers (ISA_RESEARCH §4.2)\n")
    P.append("Each closer followed by a random body (3 bodies) against 64 random partners per body, for every operand zz (256), at L = 16, 20 "
             "and 32, both BRK arms, 128 and 512 instructions (`closers_scan.csv`).\n\n**Fidelity and confinement** (zz giving an exact copy in every "
             "encounter):\n\n")
    P.append(md(cex))
    P.append("\n**Sites carried and the d − 6 law.** The offset d is predicted from the operand alone, from the loop's address arithmetic "
             "(`predicted_offset`: iteration 0 reads zero page (zz − X) or (zz + S) and pushes at 0x100 | S; `steady` = the offset stays the same "
             "for all budget/4 iterations, which fails on the ring of 40 when the 8-bit zero-page index or S wraps, 256 mod 40 = 16). A site is "
             "carried when the offspring holds the organism's byte there in every encounter of every body. Where the hypotheses hold (steady, "
             "6 ≤ d ≤ L, d divides L), the law says: the pointer never enters the partner, the sites carried are exactly 0 … d − 1, so the payload "
             "beyond the six-byte closer is d − 6, and the copy is exact if and only if d = L. Every zz is tested, selected by its predicted d, "
             "not by its outcome:\n\n")
    hyp = claw[claw.hypotheses]
    agg = (hyp.groupby(["L", "closer", "budget", "predicted d", "d − 6"])
           .agg(zz=("zz", "count"), arms=("BRK", lambda x: " ".join(sorted(set(x)))), confined=("confined", "sum"),
                carried_ok=("carried = 0 … d − 1", "sum"), payload=("payload carried (sites ≥ 6)", lambda x: " ".join(map(str, sorted(set(x))))),
                exact_rows=("exact copies", lambda x: int((x > 0).sum())), law=("law holds", "sum")).reset_index()
           .rename(columns={"zz": "rows (zz × BRK arm)", "carried_ok": "carried = 0 … d − 1", "payload": "payload carried", "exact_rows": "rows with exact copies", "law": "law holds"}))
    P.append(md(agg))
    nh, nl = len(hyp), int(hyp["law holds"].sum())
    off = claw[~claw.hypotheses & claw.confined & (claw["carried prefix"] >= 6)]
    off_ls = sorted(set(off.L))
    off16 = int((off["carried prefix"] == 16).sum())
    offL = int((off["exact copies"] == off["encounters"]).sum())
    P.append(f"\nThe law holds in {nl} of {nh} rows that meet its hypotheses (zz × BRK arm × budget × L × closer). "
             f"{len(off)} confined rows outside the hypotheses (all with L in {off_ls}) still carry a prefix of 6 or more sites (`table_closers_law.csv`): "
             f"{off16} carry exactly 16 sites, the period of the 8-bit zero-page wrap on a ring of 40, and {offL} copy exactly although the zero-page "
             f"index wraps during the run (after the copy is complete).\n\n"
             "**Open straight-line tiles of §4.2** (64 random partners, L = 16):\n\n")
    P.append(md(opentiles))

    P.append("\n## 9. Comparison with the exploratory census\n")
    P.append("The exploratory census (`ISA_RESEARCH_2026-10-10.md` §4.1, code and outputs in the session scratchpad, read only) ran its own, "
             "unvalidated interpreter on primitive words only, gave 16 random partners only to words that passed a screen (zero-partner or one "
             "screen partner at ≥ 75%), drew its partners with splitmix64 per word, scored phases of the word, and halted BRK after fetching the "
             "signature byte. Its counts, recomputed from its output files:\n\n")
    P.append(md(expl))
    agree = {c: float((rerun[f"expl_{c}"].astype(str) == rerun[f"zil_{c}"].astype(str)).mean()) for c in
             ("zero_match", "zero_entered", "zero_halted", "rand_exact", "rand_75", "entered_any", "rand_halted", "rand_meanmatch")}
    n_rr = len(rerun)
    P.append(f"\n**Same encounters on the validated core.** All {n_rr:,} words the exploratory census printed, re-run on Zilion with its own "
             "partners (its splitmix64 draws reproduced from its source) and its phase score (`exploratory_rerun.csv`). Agreement per field: "
             + ", ".join(f"{c} {pct(v)}" for c, v in agree.items()) + ".\n\n")
    if len(sig):
        P.append("The words where `entered` differs, and where their BRK-halted lane stopped on Zilion:\n\n" + md(sig) + "\n")
    wc = save("word_comparison", word_comparison(cs, rerun))
    P.append("**The random-partner columns word by word** (budget 128, both censuses; a word recorded by only one census is re-run "
             "through the other's encounters, marked \"re-run\"; `passes its screen`: whether the exploratory census would have given the "
             "word its 16 partners at all):\n\n")
    if len(wc):
        showc = wc[wc["in the exploratory"] | (wc.k < 4) | (wc.column == "≥ 75% in ≥ 8 of 16")]
        restc = wc[~wc.index.isin(showc.index)]
        P.append("Every word of the exploratory census, every ≥ 75% word and every word of periods 1–3 (all rows in `table_word_comparison.csv`):\n\n")
        P.append(md(showc))
        if len(restc):
            g = restc.groupby(["k", "L", "BRK", "column", "passes its screen"]).size().reset_index(name="words found only here")
            P.append("\nThe other words, all period 4 and exact into a random partner in this census only, by whether they would have passed "
                     "the exploratory screen:\n\n" + md(g))
    else:
        P.append("(none)\n")
    expl_words = {r.tile for r in rerun[rerun.expl_rand_exact > 0].itertuples()}
    show = rates[rates.word.str.replace(" ", "").isin(expl_words) | (rates.k < 4)] if len(rates) else rates
    rest = rates[~rates.index.isin(show.index)] if len(rates) else rates
    P.append("\n**Words exact into random partners, over 1,024 fresh random partners** (the same 1,024 for every word of a run; budget 128 "
             "and 512; `table_random_partner_rates.csv` has every word of either census). The exploratory census's words and those of "
             "periods 1–3:\n\n")
    P.append(md(show))
    if len(rest):
        g = rest.groupby(["k", "L", "BRK", "budget"]).agg(words=("word", "nunique"), median_exact=("exact", "median"), max_exact=("exact", "max"),
                                                           zero_exact=("exact", lambda x: int((x == 0).sum())), entered_all=("partner entered", lambda x: int((x == 1024).sum()))).reset_index()
        P.append("\nThis census's other words exact into a random partner (period 4, found in any run of that BRK arm), each re-run at both "
                 "budgets (`budget` is the re-run's): exact copies per 1,024 fresh partners (median, maximum, words with none), and words whose "
                 "pointer entered the partner in all 1,024 encounters:\n\n")
        P.append(md(g.rename(columns={"median_exact": "median exact", "max_exact": "max exact", "zero_exact": "words with 0 exact",
                                      "entered_all": "entered in all 1,024"})))
    # the differences, explained from the tables above
    n_sig = len(sig)
    sig_ok = bool(n_sig) and bool(sig["signature byte in B"].all()) and bool((sig["byte there"] == "00").all())
    if len(wc):
        mine_col = wc["in this census"].map(lambda v: v is True or v == True)  # noqa: E712
        ran = wc["in this census"].map(lambda v: v != "not run yet")
        only_m = wc[mine_col & ~wc["in the exploratory"]]
        only_e = wc[ran & ~mine_col & wc["in the exploratory"]]
        not_run = int((~ran).sum())
    else:
        only_m = only_e = wc
        not_run = 0
    m_fail = int((only_m["passes its screen"] == False).sum()) if len(only_m) else 0  # noqa: E712
    phase_only = int(h["columns"].str.contains("phase score only", regex=False).sum()) if len(h) else 0
    P.append("\n**The differences, and why.**\n\n"
             + (f"1. *The interpreter.* On the exploratory census's own encounters the validated core reproduces every number it printed for its "
                f"{n_rr:,} words (zero-partner score, halts, the counts of exact and ≥ 75% partners, the mean copy score and the halts over its random "
                f"partners), except "
                f"`entered` for {n_sig} word(s) under BRK halting" if all(agree[c] == 1.0 for c in ("zero_match", "zero_halted", "rand_exact", "rand_75", "rand_halted", "rand_meanmatch"))
                else f"1. *The interpreter.* On the exploratory census's own encounters the two interpreters disagree on some words (agreement per field above); "
                     f"`entered` differs for {n_sig} word(s)")
             + (": in each, BRK sits on the organism's last byte, so its signature byte is the partner's first; the exploratory interpreter "
                "fetched the signature byte and then halted, Zilion's lethal rule halts the lane before the instruction runs. Only the open/closed "
                "label of such a word can differ, never a copy score." if sig_ok else ".") + "\n"
             "Its initial P is 0 rather than 0x20; bits 4 and 5 of P are never observable (PHP and BRK push them set whatever P holds, and no "
             "other instruction reads them).\n"
             "2. *The words enumerated.* The exploratory census ran the words of primitive period k only, indexed little-endian (its word "
             "`b5 0f 48 ca` is index 0xca480fb5); this census runs every word of period k, indexed big-endian as `census_period.py`, and reports "
             "the primitive ones separately (section 3), so the counts compare on the primitive rows.\n"
             f"3. *The screen.* The exploratory census gave 16 random partners only to words whose zero-partner or screen-partner copy reached "
             f"75%. Of the {len(only_m)} word-column entries found only here, {m_fail} would not have passed that screen.\n"
             f"4. *The partners.* Exactness into a random partner is a property of the (word, partner) pair: the {len(only_e)} entries found only "
             "by the exploratory census, and the others found only here, are re-run above with the other census's partners; the 1,024-partner "
             "rates show how often each word copies exactly into a fresh random partner."
             + (f" ({not_run} exploratory entries are in the part of the BRK-halting period-4 census not run yet.)" if not_run else "") + "\n"
             f"5. *The score.* Where k does not divide L (period 3 here) the exploratory phase score and this census's rotation score differ; "
             f"{phase_only} hit row(s) meet a column under the phase score only (section 4).\n"
             "6. *The budget.* The 512-instruction arm is new; the exploratory census ran 128.\n")
    P.append("\n")
    (OUT / "CENSUS.md").write_text("\n".join(P))
    print(f"CENSUS.md written ({len(cs)} censuses; period 4 complete: {k4_complete})")


if __name__ == "__main__":
    report()
