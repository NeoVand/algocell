"""S — serial allele retention (pre-registered: REVISION_PREREG.md, R2/S, 2026-10-09).

    .venv/bin/python serial_retention.py --yes [--only key1,key2] [--out results/serial_retention]
    .venv/bin/python serial_retention.py --report

Every single-byte mutant of each panel genotype founds R = 16 lineages; the unmutated tape founds 256. Each generation the
lineage tape runs as A against a fresh uniform random partner (128 instructions) and the partner half becomes the next
lineage tape. A transfer is a copy if the offspring matches its parent at >= 75% of positions at the best cyclic shift; a
lineage is alive at g if transfers 1..g were all copies. The allele is present at g if the mutant's 5-byte cyclic window
centred on the mutated position occurs cyclically in the lineage tape. Mutants whose window occurs in the wild type are
not identifiable and are excluded. Background: the share of alive control lineages whose tape contains the window.
Outputs: <out>/sr_mutants.csv.gz (one row per genotype x mutant x generation), <out>/sr_sites.csv, <out>/sr_summary.csv,
<out>/REPORT.md (predictions S1–S5).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from algocell_exp import assay as A  # noqa: E402
from algocell_exp import exectrace as X  # noqa: E402

R_RES = os.path.join(HERE, "results")
G = 8
R_MUT = 16
R_CTRL = 256
W = 5                       # window length (centred: 2 bytes each side)
GENS = (1, 2, 4, 8)
CORE_8080 = "21 20 00 31 40 00 16 10 2b 46 2b 4e c5 15 c2 08 00 c3 11 00"


def hx(t: np.ndarray) -> str:
    return " ".join(f"{b:02x}" for b in t)


def parse(s: str) -> np.ndarray:
    return np.array([int(b, 16) for b in s.split()], dtype=np.uint8)


def panel() -> list[dict]:
    from make_conds import ABLATIONS
    i8080 = tuple(ABLATIONS["i8080"])
    g = pd.read_csv(os.path.join(R_RES, "stageG", "stageG", "stage_g_runs.csv"))
    gi = pd.read_csv(os.path.join(R_RES, "stageI", "stageI", "stage_g_runs.csv"))
    cl = pd.read_csv(os.path.join(R_RES, "capacity_time_L", "capacity_over_time.csv"))
    jr50 = g[(g.L == 50)].final_tape.value_counts().index[0]
    s4009 = gi[gi.seed == 4009].final_tape.iloc[0]

    def genome(L, seed, step):
        return cl[(cl.L == L) & (cl.seed == seed) & (cl.step == step)].tape.iloc[0]
    rng = np.random.default_rng(20261010)
    pay = [rng.integers(0, 256, size=12, dtype=np.uint8) for _ in range(2)]
    P = [
        ("pusher16", "pusher", " ".join(["01 c5"] * 8), (), False),
        ("pusher50", "pusher", " ".join(["01 c5"] * 25), (), False),
        ("pusher64", "pusher", " ".join(["01 c5"] * 32), (), False),
        ("ret16", "evolved closer", "ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0", (), False),
        ("ldir20", "evolved closer", " ".join(["1e 04 ed b0"] * 5), (), False),
        ("ldir32", "evolved closer", " ".join(["04 5e ed b0"] * 8), (), False),
        ("jr50", "evolved closer", jr50, (), False),
        ("lethal16_s4009", "lethal-tar closer", s4009, (), True),
        ("genome16_s6006", "transient genome", genome(16, 6006, 1_000_000), (), False),
        ("genome20_s6003", "transient genome", genome(20, 6003, 100_000), (), False),
        ("genome20_s6004", "transient genome", genome(20, 6004, 500_000), (), False),
        ("pusher32_8080", "pusher (8080)", " ".join(["01 c5"] * 16), i8080, False),
        ("closer32_8080_p1", "constructed closer (8080)", CORE_8080 + " " + hx(pay[0]), i8080, False),
        ("closer32_8080_p2", "constructed closer (8080)", CORE_8080 + " " + hx(pay[1]), i8080, False),
    ]
    return [{"key": k, "kind": kind, "tape": parse(t), "sup": sup, "zh": zh} for k, kind, t, sup, zh in P]


def window_codes(T: np.ndarray) -> np.ndarray:
    """(N, L) uint8 -> (N, L) int64: code of the cyclic W-window starting at each position."""
    L = T.shape[1]
    c = np.zeros(T.shape, dtype=np.int64)
    for k in range(W):
        c |= T[:, (np.arange(L) + k) % L].astype(np.int64) << (8 * k)
    return c


def run_genotype(gt: dict, seed: int, pool: np.ndarray | None = None) -> tuple[pd.DataFrame, dict]:
    """pool: if given (post hoc use only), partners are drawn uniformly from these tapes instead of uniform random bytes."""
    x, sup, zh = gt["tape"], list(gt["sup"]), gt["zh"]
    L = x.size
    K = 255 if L <= 32 else 32
    rng = np.random.default_rng([seed, L])
    pos, val = [], []
    for i in range(L):
        alt = np.array([v for v in range(256) if v != x[i]], dtype=np.uint8)
        vs = alt if K >= 255 else rng.choice(alt, size=K, replace=False)
        pos += [i] * len(vs)
        val += list(vs)
    pos, val = np.array(pos), np.array(val, dtype=np.uint8)
    M = pos.size
    mut = np.tile(x, (M, 1))
    mut[np.arange(M), pos] = val
    # the mutant's window: start at pos - 2
    wcode = window_codes(mut)[np.arange(M), (pos - W // 2) % L]
    wt_codes = window_codes(x[None, :])[0]
    ident = ~np.isin(wcode, wt_codes)
    # partners: lineage r, generation g; control lineages 0..15 share the mutant lineages' partners
    if pool is None:
        Pm = rng.integers(0, 256, size=(G, R_MUT, L), dtype=np.uint8)
        Pc = rng.integers(0, 256, size=(G, R_CTRL, L), dtype=np.uint8)
    else:
        Pm = pool[rng.integers(0, len(pool), size=(G, R_MUT))]
        Pc = pool[rng.integers(0, len(pool), size=(G, R_CTRL))]
    Pc[:, :R_MUT] = Pm
    # executed positions of the wild type (64 partners) for annotation
    Q = rng.integers(0, 256, size=(64, L), dtype=np.uint8)
    _, masks = X.execute_pairs_traced(np.concatenate([np.repeat(x[None], 64, 0), Q], 1), L, 128, suppress=sup, zero_halts=zh)
    executed = X.exec_positions(masks, 2 * L)[:, :L].any(axis=0)
    entered = float(X.exec_positions(masks, 2 * L)[:, L:].any(axis=1).mean())
    # lineages
    Tm = np.repeat(mut, R_MUT, axis=0)                       # (M*R, L), lineage r of mutant m at row m*R + r
    Tc = np.repeat(x[None], R_CTRL, axis=0)
    alive_m = np.ones(M * R_MUT, bool)
    alive_c = np.ones(R_CTRL, bool)
    rows = []
    ctrl = {}
    # first-generation aligned transmission as in mutscan (copies judged against the wild type)
    for gen in range(1, G + 1):
        Pg_m = np.tile(Pm[gen - 1], (M, 1))
        res = A.execute_pairs(np.concatenate([Tm, Pg_m], 1), L, 128, sup, None, zh)
        off = res[:, L:]
        copy = A._best_shift_match_rows(off, Tm) >= 0.75
        if gen == 1:
            after, shifts = A._best_shift_rows(off, np.repeat(x[None], M * R_MUT, 0))
            pr, vr = np.repeat(pos, R_MUT), np.repeat(val, R_MUT)
            carried = off[np.arange(M * R_MUT), (pr + shifts) % L] == vr
            cpy = (after >= 0.75).reshape(M, R_MUT)
            aligned = np.where(cpy.sum(1) > 0, (carried.reshape(M, R_MUT) & cpy).sum(1) / np.maximum(cpy.sum(1), 1), np.nan)
        alive_m &= copy
        Tm = off
        resc = A.execute_pairs(np.concatenate([Tc, Pc[gen - 1]], 1), L, 128, sup, None, zh)
        offc = resc[:, L:]
        alive_c &= A._best_shift_match_rows(offc, Tc) >= 0.75
        Tc = offc
        if gen in GENS:
            codes = window_codes(Tm)
            present = (codes == np.repeat(wcode, R_MUT)[:, None]).any(axis=1)
            am = alive_m.reshape(M, R_MUT)
            pm = (present & alive_m).reshape(M, R_MUT)
            cc = window_codes(Tc[alive_c]) if alive_c.any() else np.zeros((0, L), np.int64)
            bg = np.array([(cc == c).any(axis=1).mean() if len(cc) else np.nan for c in wcode]) if len(cc) else np.full(M, np.nan)
            rows.append(pd.DataFrame({"key": gt["key"], "gen": gen, "pos": pos, "val": val, "identifiable": ident,
                                      "alive": am.mean(1), "retained": pm.mean(1), "erased": (am & ~pm).mean(1), "lost": (~am).mean(1),
                                      "p_present_alive": np.where(am.sum(1) > 0, pm.sum(1) / np.maximum(am.sum(1), 1), 0.0), "bg": bg,
                                      "aligned_g1": aligned if gen == 1 else np.nan}))
            ctrl[gen] = float(alive_c.mean())
    df = pd.concat(rows, ignore_index=True)
    meta = {"key": gt["key"], "kind": gt["kind"], "L": L, "tape": hx(x), "rule": ("i8080" if gt["sup"] else "Z80") + (", lethal tar" if zh else ""),
            "n_mutants": M, "n_identifiable": int(ident.sum()), "entered_frac": entered, "executed": "".join("1" if e else "0" for e in executed),
            **{f"ctrl_alive_g{g}": ctrl[g] for g in GENS}}
    return df, meta


def sites(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    d = df[df.identifiable].copy()
    d["ok"] = (d.p_present_alive - d.bg.fillna(0.0)) >= 0.5
    d["ok_aligned"] = d.aligned_g1.fillna(0.0) >= 0.5
    s = d.groupby(["gen", "pos"]).agg(frac_ok=("ok", "mean"), n_ident=("ok", "size"), retained=("retained", "mean"), erased=("erased", "mean"), lost=("lost", "mean")).reset_index()
    s["transmissible"] = s.frac_ok >= 0.5
    ex = meta["executed"]
    s["executed"] = [ex[p] == "1" for p in s.pos]
    s.insert(0, "key", meta["key"])
    return s


def run(a) -> None:
    os.makedirs(a.out, exist_ok=True)
    P = panel()
    if a.only:
        P = [p for p in P if p["key"] in a.only.split(",")]
    allm, alls, metas = [], [], []
    t_all = time.time()
    for gt in P:
        t0 = time.time()
        df, meta = run_genotype(gt, seed=20261009)
        s = sites(df, meta)
        allm.append(df)
        alls.append(s)
        for g in GENS:
            dg = df[(df.gen == g) & df.identifiable]
            sg = s[s.gen == g]
            meta[f"retained_g{g}"] = float(dg.retained.mean())
            meta[f"erased_g{g}"] = float(dg.erased.mean())
            meta[f"lost_g{g}"] = float(dg.lost.mean())
            meta[f"sites_g{g}"] = int(sg.transmissible.sum())
            meta[f"sites_unexec_g{g}"] = int((sg.transmissible & ~sg.executed).sum())
        d1 = df[(df.gen == 1) & df.identifiable]
        meta["sites_aligned_g1"] = int((d1.assign(ok=d1.aligned_g1.fillna(0) >= 0.5).groupby("pos").ok.mean() >= 0.5).sum())
        meta["wall_s"] = round(time.time() - t0, 1)
        metas.append(meta)
        print(f"{meta['key']:>18} L={meta['L']:2d} ident {meta['n_identifiable']}/{meta['n_mutants']} ctrl alive g1/g4/g8 {meta['ctrl_alive_g1']:.2f}/{meta['ctrl_alive_g4']:.2f}/{meta['ctrl_alive_g8']:.2f} "
              f"| sites g1/g4/g8 {meta['sites_g1']}/{meta['sites_g4']}/{meta['sites_g8']} (aligned g1 {meta['sites_aligned_g1']}) | g4 lost/erased/retained {meta['lost_g4']:.2f}/{meta['erased_g4']:.2f}/{meta['retained_g4']:.2f} ({meta['wall_s']} s)", flush=True)
    sm = pd.DataFrame(metas)
    keys = [m["key"] for m in metas]
    if a.only and os.path.exists(os.path.join(a.out, "sr_summary.csv")):
        old = pd.read_csv(os.path.join(a.out, "sr_summary.csv"))
        sm = pd.concat([old[~old.key.isin(keys)], sm], ignore_index=True)
        om = pd.read_csv(os.path.join(a.out, "sr_mutants.csv.gz"))
        osi = pd.read_csv(os.path.join(a.out, "sr_sites.csv"))
        allm = [om[~om.key.isin(keys)]] + allm
        alls = [osi[~osi.key.isin(keys)]] + alls
    sm.to_csv(os.path.join(a.out, "sr_summary.csv"), index=False)
    pd.concat(allm, ignore_index=True).to_csv(os.path.join(a.out, "sr_mutants.csv.gz"), index=False)
    pd.concat(alls, ignore_index=True).to_csv(os.path.join(a.out, "sr_sites.csv"), index=False)
    print(f"done in {time.time() - t_all:.0f} s")
    report(a)


def report(a) -> None:
    sm = pd.read_csv(os.path.join(a.out, "sr_summary.csv")).set_index("key")
    st = pd.read_csv(os.path.join(a.out, "sr_sites.csv"))
    L = ["# S — serial allele retention: predictions S1–S5 against the data (generated by `serial_retention.py --report`)", ""]
    L.append("| genotype | kind | L | rule | identifiable mutants | control alive g = 1 / 4 / 8 | sites g = 1 / 2 / 4 / 8 (unexecuted at 4) | aligned sites g = 1 | lost / erased / retained, g = 1 | g = 4 | g = 8 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, r in sm.iterrows():
        L.append(f"| {k} | {r.kind} | {r.L} | {r.rule} | {r.n_identifiable} of {r.n_mutants} | {r.ctrl_alive_g1:.2f} / {r.ctrl_alive_g4:.2f} / {r.ctrl_alive_g8:.2f} | "
                 f"{r.sites_g1} / {r.sites_g2} / {r.sites_g4} / {r.sites_g8} ({r.sites_unexec_g4}) | {r.sites_aligned_g1} | "
                 f"{r.lost_g1:.2f} / {r.erased_g1:.2f} / {r.retained_g1:.2f} | {r.lost_g4:.2f} / {r.erased_g4:.2f} / {r.retained_g4:.2f} | {r.lost_g8:.2f} / {r.erased_g8:.2f} / {r.retained_g8:.2f} |")
    L.append("")

    def has(k):
        return k in sm.index
    out = []
    if has("pusher50") and has("pusher64"):
        ok = all(sm.loc[k, "sites_g4"] >= max(10, sm.loc[k, "sites_g1"] / 2) for k in ("pusher50", "pusher64"))
        out.append(f"S1 (pusher sites persist at L = 50, 64: g4 >= max(10, g1/2)): {'met' if ok else 'NOT met'} — " + "; ".join(f"{k}: g1 {sm.loc[k, 'sites_g1']}, g4 {sm.loc[k, 'sites_g4']}" for k in ("pusher50", "pusher64")))
    ev = [k for k in ("ret16", "ldir20", "ldir32", "jr50") if has(k)]
    if ev:
        ok = all(sm.loc[k, "sites_g4"] <= 2 and sm.loc[k, "erased_g1"] > sm.loc[k, "retained_g1"] for k in ev)
        out.append(f"S2 (evolved closers: <= 2 sites at g4 and erased > retained at g1): {'met' if ok else 'NOT met'} — " + "; ".join(f"{k}: sites g4 {sm.loc[k, 'sites_g4']}, g1 erased {sm.loc[k, 'erased_g1']:.2f} vs retained {sm.loc[k, 'retained_g1']:.2f}" for k in ev))
    cc = [k for k in ("closer32_8080_p1", "closer32_8080_p2") if has(k)]
    if cc:
        res = []
        okall = True
        for k in cc:
            s8 = st[(st.key == k) & (st.gen == 8) & (st.pos >= 20)]
            n = int(s8.transmissible.sum())
            ok = n >= 11 and sm.loc[k, "ctrl_alive_g8"] >= 0.99
            okall &= ok
            res.append(f"{k}: payload sites at g8 {n} of 12, control alive g8 {sm.loc[k, 'ctrl_alive_g8']:.3f}, core sites at g8 {int(st[(st.key == k) & (st.gen == 8) & (st.pos < 20)].transmissible.sum())} of 20")
        out.append(f"S3 (constructed closer keeps its payload): {'met' if okall else 'NOT met'} — " + "; ".join(res))
    gg = [k for k in ("genome16_s6006", "genome20_s6003", "genome20_s6004") if has(k)]
    if gg:
        ok = all(sm.loc[k, "sites_g4"] >= 5 for k in gg)
        out.append(f"S4 (transient genomes >= 5 sites at g4): {'met' if ok else 'NOT met'} — " + "; ".join(f"{k}: g1 {sm.loc[k, 'sites_g1']}, g4 {sm.loc[k, 'sites_g4']} ({sm.loc[k, 'sites_unexec_g4']} unexecuted)" for k in gg))
    if has("pusher16"):
        out.append(f"S5 (descriptive, pusher L = 16 <= 2 sites at g4): {'met' if sm.loc['pusher16', 'sites_g4'] <= 2 else 'NOT met'} — g4 {sm.loc['pusher16', 'sites_g4']}")
    L += [f"- {o}" for o in out]
    txt = "\n".join(L)
    open(os.path.join(a.out, "REPORT.md"), "w").write(txt + "\n")
    print(txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R_RES, "serial_retention"))
    ap.add_argument("--only", default="")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--yes", action="store_true", help="confirm that the browser simulation is paused (local GPU)")
    a = ap.parse_args()
    if a.report:
        report(a)
        return
    if not a.yes:
        sys.exit("refusing to run: pass --yes after pausing the browser simulation (local GPU)")
    run(a)


if __name__ == "__main__":
    main()
