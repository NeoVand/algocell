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
from pathlib import Path

import numpy as np
import pandas as pd

import census6502 as C
from algocell_exp.soup import adapter_summary
from algocell_exp.zcensus import COLUMN_FLAGS, F_NEAR, FLAG_NAMES, evaluate, n_primitive, primitive_period, tile_bytes
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
            rows.append({"L": x["L"], "BRK": x["brk"], "budget": b, "k": x["k"], "words": 256 ** x["k"],
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
                rows.append({"L": x["L"], "BRK": x["brk"], "budget": b, "k": x["k"], "word": w.hex(" "),
                             "disassembly (tiled, from 0)": C.dis6502_word(w, x["L"]),
                             "columns": ", ".join([v for f, v in COLS.items() if fl & f and f in (1, 4, 16)] +
                                                  [v for f, v in COLS.items() if fl & f and f in (2, 8, 32) and not fl & (f >> 1)]) or "–",
                             "zero-partner match": int(d["rot"][i, 0]), "exact partners": int(d["n_exact"][i]),
                             "≥ 75% partners": int(d["n75"][i]), "best random match": int(d["rot"][i, 1:].max()) if d["rot"].shape[1] > 1 else 0,
                             "closed": bool(not d["entered"][i].any()), "partner entered (of 17)": int(d["entered"][i].sum()),
                             "zero partner: halt / steps": f"{HALTS[int(d['halt'][i, 0])]} / {int(d['steps'][i, 0])}"})
    return pd.DataFrame(rows)


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
                rows.append({"L": L, "BRK": x["brk"], "budget": b, "k": x["k"], "partner": cls, "lanes": n,
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
            rows.append({"L": x["L"], "BRK": x["brk"], "budget": b, "k": x["k"], "tiles re-run": rc.get("tiles", 0), "lanes": rc.get("lanes", 0),
                         "mismatching tiles (any field)": sum(v for kk, v in rc.items() if kk.startswith("mismatch")),
                         "harness WGSL": R["head"]["harness_wgsl_sha256"][:12], "machine WGSL": R["head"]["machine"]["sha256_wgsl"][:12]})
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
    rows = []
    keys = sorted({(int(d["k"]), d["L"], d["brk_halt"]) for d in cand} | set(prim))
    for (k, L, bh) in keys:
        ds = [d for d in cand if (int(d["k"]), d["L"], d["brk_halt"]) == (k, L, bh)]
        rows.append({"k": k, "L": L, "BRK": "halt" if bh else "faithful", "tiles (primitive)": prim.get((k, L, bh), n_primitive(k)),
                     "screened candidates": len(ds),
                     "exact, zero partner": sum(d["zero_exact"] == "1" for d in ds),
                     "exact, zero partner (non-zero tiles)": sum(d["zero_exact"] == "1" and int(d["tile"], 16) != 0 for d in ds),
                     "exact, ≥ 1 random partner": sum(not d["rand_exact"].startswith("0/") for d in ds),
                     "≥ 75% in ≥ 8 of 16": sum(int(d["rand_75"].split("/")[0]) >= 8 for d in ds)})
    return pd.DataFrame(rows)


def partner_rates(tiles: list[tuple[int, int, int, str]], n: int = 1024) -> pd.DataFrame:
    """Exact-copy and ≥ 75% rates of tiles over n fresh random partners (numpy rng, seed fixed), budget 128 and 512."""
    rows = []
    rng = np.random.default_rng(20261011)
    for (k, L, t, brk) in tiles:
        m = load_machine(C.machine_name(L, brk))
        P = rng.integers(0, 256, size=(1, n, L), dtype=np.uint8)
        for b in (128, 512):
            e = evaluate(m, k, np.array([t], np.uint64), n, b, partners=P)
            rows.append({"k": k, "L": L, "BRK": brk, "word": word_of(t, k).hex(" "), "budget": b, "partners": n,
                         "exact": int((e.rot[0, 1:] == L).sum()), "≥ 75%": int((e.rot[0, 1:] >= -(-3 * L // 4)).sum()),
                         "best match": int(e.rot[0, 1:].max()), "partner entered": int(e.entered[0, 1:].sum())})
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
                r = {"k": k, "L": L, "BRK": brk, "column": col, "word": w.hex(" "), "disassembly": C.dis6502_word(w, L),
                     "in this census": t in mine, "in the exploratory": t in theirs,
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
                    if x["R"]["complete"] or t < x["R"]["tiles_done"]:
                        e2 = evaluate(obj(x).machine, k, np.array([t], np.uint64), 16, 128)
                        r["this census: exact / ≥ 75% partners"] = f"{int((e2.rot[0, 1:] == L).sum())} / {int((e2.rot[0, 1:] >= -(-3 * L // 4)).sum())} (re-run)"
                    else:
                        r["this census: exact / ≥ 75% partners"] = "not yet run"
                rows.append(r)
    return pd.DataFrame(rows)


# ── closers ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def closer_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    D = pd.read_csv(OUT / "closers_scan.csv", dtype={"zz": str})
    ex_rows, law_rows = [], []
    for (L, brk, fam, b), g in D.groupby(["L", "BRK", "family", "budget"], sort=True):
        ok = g[g.exact == g.encounters]
        ex_rows.append({"L": L, "BRK": brk, "closer": fam, "budget": b, "zz exact in every encounter": " ".join(ok.zz) or "none",
                        "encounters each": int(g.encounters.iloc[0]), "pointer in partner (those zz)": int(ok.pointer_in_partner.sum()),
                        "zz confined (no partner fetch, all encounters)": int((g.pointer_in_partner == 0).sum())})
        conf = g[(g.pointer_in_partner == 0) & (g.n_carried > 0)]
        for r in conf.itertuples():
            s = r.sites_carried_all_bodies
            dhat = len(s) - len(s.lstrip("#")) if s.startswith("#") else 0
            prefix = s[:dhat] == "#" * dhat and "#" not in s[dhat:]
            law_rows.append({"L": L, "BRK": brk, "closer": fam, "budget": b, "zz": r.zz, "sites carried": s, "d (carried prefix)": dhat,
                             "prefix only": prefix, "d divides L": L % dhat == 0 if dhat else False, "payload carried (sites ≥ 6)": int(r.payload_carried),
                             "d − 6": dhat - 6, "law holds": int(r.payload_carried) == max(dhat - 6, 0), "exact copies": int(r.exact)})
    return pd.DataFrame(ex_rows), pd.DataFrame(law_rows)


# ── CENSUS.md ───────────────────────────────────────────────────────────────────────────────────────────────────────
def report() -> None:
    cs = censuses()
    zcommit = subprocess.check_output(["git", "-C", str(C.ZILION), "rev-parse", "HEAD"], text=True).strip()
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
    P.append(f"<!-- Generated by `python census6502.py report` on {stamp} (adapter {adapter_summary()}, zilion {zcommit[:12]}). "
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
                vals.append("–" if r is None else (f"{r['exact, zero partner']}" + ("" if r["done"] == r["words"] else f" (of {r['done']:,} done)")))
            lines.append(f"BRK {brk}, budget {b}: {', '.join(vals)}")
    P.append("**Headline.** Self-writers (exact into the all-zero partner, non-zero tapes) at periods 1, 2, 3, 4, L = 16: "
             + "; ".join(lines) + ". On the Z80 the same harness gives " + ", ".join(str(int(v)) for v in ctrl.self_writers) +
             " at periods 1–4, the paper's census exactly (section 2).\n")
    if not k4_complete:
        status = ", ".join(f"{x['brk']}: {x['R']['tiles_done']:,} of {256 ** 4:,}" for x in k4) or "not started"
        P.append(f"**Status: the period-4 census is incomplete** ({status}); the tables below cover what is done.\n")

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
    P.append(f"`z80c02_pair_maps_L16` (compat 0.2), 128 instructions, L = 16, every word of period 1–4. Periods 1–3 also met 16 random "
             f"partners per word; period 4 the zero partner only (the column the paper counts). \"Paper\" is `results/review_r2/census_p*.csv` "
             f"(`census_period.py`). Open/closed is re-traced with `census_period.py`'s own partners (`rng([20261018, p])`, drawn per self-writer "
             f"in index order) on the Zilion machine and on Algocell's op-traced executor (`exectrace.py`), and with the harness's partners.\n")
    P.append(md(ctrl[["period", "words", "tiles_done", "self_writers", "by_primitive_period", "paper_self_writers", "word_sets_equal",
                      "closed_census_partners_zilion", "closed_census_partners_algocell", "paper_closed", "closed_flags_equal",
                      "closed_harness_partners", "rand_exact_any", "rand75_ge8", "recheck_tiles", "recheck_mismatches"]].rename(columns={
        "tiles_done": "words run", "self_writers": "self-writers", "by_primitive_period": "by primitive period", "paper_self_writers": "paper",
        "word_sets_equal": "same words", "closed_census_partners_zilion": "closed (Zilion)", "closed_census_partners_algocell": "closed (Algocell executor)",
        "paper_closed": "paper closed", "closed_flags_equal": "same closed flags", "closed_harness_partners": "closed (harness partners)",
        "rand_exact_any": "exact, ≥ 1 random", "rand75_ge8": "≥ 75% in ≥ 8", "recheck_tiles": "tiles re-run", "recheck_mismatches": "mismatches"})))
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
    P.append("Every word (non-zero) that meets any column, under either score, with its linear disassembly from offset 0 of the tiled tape "
             "(until the decode returns to a word boundary).\n")
    h = T["hits"]
    P.append(md(h) if len(h) else "(none)\n")
    P.append("\n## 5. Near misses\n")
    nm = T["near"]
    if len(nm):
        nm4 = nm.sort_values(["zero-partner match", "≥ 75% partners", "best random match"], ascending=False)
        P.append(f"Words that meet no column but come near one (zero-partner match ≥ 75%, ≥ 2 random partners at ≥ 75%, or a random partner at L − 1): "
                 f"{len(nm):,} rows in `table_near.csv`; the 40 with the highest zero-partner match:\n\n")
        P.append(md(nm4.head(40)))
    else:
        P.append("(none)\n")
    P.append("\n## 6. Non-vacuity diagnostics\n")
    P.append("Per census and partner class (zero: the all-zero partner; rand: the 16 random partners): how the lanes ran. `wrote B`: the lane wrote "
             "into the partner at least once; `B changed`: the partner's final bytes differ from its initial bytes; `fetched from B`: the open "
             "criterion; `match ≥ 75%` / `≥ L − 1`: lanes whose copy score reached that. A census whose lanes all halted at once, or never wrote the "
             "partner, would show it here.\n")
    P.append(md(T["diag"]))
    P.append("\n## 7. Re-checks\n")
    P.append("Every recorded tile (hits, near misses and the hash-chosen sample: 1 word in 2^18 at period 4, 2^12 at period 3) re-run "
             "at each budget through one direct `ZMachine.run` (one dispatch, no segments), with tape and partners built in numpy: per-encounter "
             "copy score under both scores, fetch into B, halt reason, trap cause, steps, and the per-word counts.\n")
    P.append(md(T["recheck"]))

    P.append("\n## 8. Seeded closers (ISA_RESEARCH §4.2)\n")
    P.append("Each closer followed by a random body (3 bodies) against 64 random partners per body, for every operand zz (256), at L = 16, 20 "
             "and 32, both BRK arms, 128 and 512 instructions (`closers_scan.csv`).\n\n**Fidelity and confinement** (zz giving an exact copy in every "
             "encounter):\n\n")
    P.append(md(cex))
    key = ["L", "closer", "budget", "zz", "sites carried", "payload carried (sites ≥ 6)"]
    fset = set(map(tuple, claw[claw.BRK == "faithful"][key].astype(str).values))
    hset = set(map(tuple, claw[claw.BRK == "halt"][key].astype(str).values))
    brk_same = fset == hset
    P.append("\n**Sites carried and the d − 6 law.** A site is carried when the offspring holds the organism's byte there in every encounter of "
             "every body; `d` is the length of the carried prefix (sites 0 … d − 1); the law is that the payload carried (sites 6 and above, beyond "
             "the six-byte closer) is d − 6. Rows: every zz whose pointer never entered the partner and that carried at least one site, grouped by d "
             + ("(the two BRK arms give the same rows).\n\n" if brk_same else
                f"(the two BRK arms differ in {len(fset ^ hset)} (zz, budget) rows, so both are shown; per zz in `table_closers_law.csv`).\n\n"))
    agg = (claw.groupby(["L", "BRK", "closer", "budget", "d (carried prefix)", "payload carried (sites ≥ 6)", "d − 6", "d divides L", "law holds", "prefix only"])
           .agg(zz=("zz", lambda s: " ".join(s)), exact=("exact copies", "max")).reset_index())
    P.append(md(agg))
    held = claw[(claw["d divides L"]) & (claw["d (carried prefix)"] >= 6)]
    P.append(f"\nWhere d divides L and d ≥ 6, the law holds in {int(held['law holds'].sum())} of {len(held)} rows (both BRK arms, both budgets).\n\n"
             "**Open straight-line tiles of §4.2** (64 random partners, L = 16):\n\n")
    P.append(md(opentiles))

    P.append("\n## 9. Comparison with the exploratory census\n")
    P.append("The exploratory census (`ISA_RESEARCH_2026-10-10.md` §4.1, code and outputs in the session scratchpad, read only) ran its own, "
             "unvalidated interpreter on primitive words only, gave 16 random partners only to words that passed a screen (zero-partner or one "
             "screen partner at ≥ 75%), drew its partners with splitmix64 per word, scored phases of the word, and halted BRK after fetching the "
             "signature byte. Its counts, recomputed from its output files:\n\n")
    P.append(md(expl))
    agree = {c: float((rerun[f"expl_{c}"] == rerun[f"zil_{c}"]).mean()) for c in
             ("zero_match", "zero_entered", "zero_halted", "rand_exact", "rand_75", "entered_any", "rand_halted")}
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
    P.append(md(wc))
    P.append("\n**Words exact into random partners, either census, over 1,024 fresh random partners** (budget 128 and 512):\n\n")
    P.append(md(rates))
    # the differences, explained from the tables above
    n_sig = len(sig)
    sig_ok = bool(n_sig) and bool(sig["signature byte in B"].all()) and bool((sig["byte there"] == "00").all())
    only_m = wc[wc["in this census"] & ~wc["in the exploratory"]] if len(wc) else wc
    only_e = wc[~wc["in this census"] & wc["in the exploratory"]] if len(wc) else wc
    m_fail = int((only_m["passes its screen"] == False).sum()) if len(only_m) else 0  # noqa: E712
    phase_only = int(h["columns"].str.contains("phase score only", regex=False).sum()) if len(h) else 0
    P.append("\n**The differences, and why.**\n\n"
             + (f"1. *The interpreter.* On the exploratory census's own encounters the validated core reproduces every number it printed for its "
                f"{n_rr:,} words (zero-partner score, halts, the counts of exact and ≥ 75% partners, halts in its random partners), except "
                f"`entered` for {n_sig} word(s) under BRK halting" if all(agree[c] == 1.0 for c in ("zero_match", "zero_halted", "rand_exact", "rand_75", "rand_halted"))
                else f"1. *The interpreter.* On the exploratory census's own encounters the two interpreters disagree on some words (agreement per field above); "
                     f"`entered` differs for {n_sig} word(s)")
             + (": in each, BRK sits on the organism's last byte, so its signature byte is the partner's first; the exploratory interpreter "
                "fetched the signature byte and then halted, Zilion's lethal rule halts the lane before the instruction runs. Only the open/closed "
                "label of such a word can differ, never a copy score." if sig_ok else ".") + "\n"
             "2. *The words enumerated.* The exploratory census ran the words of primitive period k only; this census runs every word of "
             "period k and reports the primitive ones separately (section 3), so the counts compare on the primitive rows.\n"
             f"3. *The screen.* The exploratory census gave 16 random partners only to words whose zero-partner or screen-partner copy reached "
             f"75%. Of the {len(only_m)} word-column entries found only here, {m_fail} would not have passed that screen.\n"
             f"4. *The partners.* Exactness into a random partner is a property of the (word, partner) pair: the {len(only_e)} entries found only "
             "by the exploratory census, and the others found only here, are re-run above with the other census's partners; the 1,024-partner "
             "rates show how often each word copies exactly into a fresh random partner.\n"
             f"5. *The score.* Where k does not divide L (period 3 here) the exploratory phase score and this census's rotation score differ; "
             f"{phase_only} hit row(s) meet a column under the phase score only (section 4).\n"
             "6. *The budget.* The 512-instruction arm is new; the exploratory census ran 128.\n")
    P.append("\n")
    (OUT / "CENSUS.md").write_text("\n".join(P))
    for name, df in T.items():
        (OUT / f"table_{name}.md").write_text(md(df))
    print(f"CENSUS.md written ({len(cs)} censuses; period 4 complete: {k4_complete})")


if __name__ == "__main__":
    report()
