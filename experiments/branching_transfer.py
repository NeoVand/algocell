"""X1 — branching transfer (pre-registered: REVISION_PREREG.md, "X1–X3 — three referee-driven tests", X1; committed at
244616b, 2026-10-10, before this script was written).

    .venv/bin/python branching_transfer.py [--only key1,key2] [--arms U,K] [--out results/branching]
    .venv/bin/python branching_transfer.py --report [--out results/branching]
    smoke:  .venv/bin/python branching_transfer.py --only pusher16 --arms U --gens 2 --values 2 --reps 2 --ctrl 512 --out <dir>

Design (the defaults are the registered values; any other value is flagged in the report and blocks scoring):
  genotypes   the serial-retention panel keys pusher16, pusher64, ret16, ldir32, genome20_s6003 (`serial_retention.panel`).
  mutants     every position, 32 seeded alternative values per position (x[i] excluded); 4 replicate populations per
              mutant; 64 unmutated (control) populations. Every population is founded by one tape.
  generation  every member runs as organism A (128 instructions, the genotype's own rule, `A.execute_pairs`) against
              k = 2 fresh partners; every offspring (partner half after the encounter) that matches its parent at >= 75%
              of positions at the best cyclic shift (`A._best_shift_match_rows`) joins the next generation; parents do
              not carry over; a population with more than N = 32 offspring-copies keeps a uniform random subsample of 32.
              G = 8 generations.
  partners    arm U: uniform random bytes (all genotypes). Arm K (pushers only): cells drawn uniformly from the pooled
              snapshots at steps 5,000 and 10,000 of the 20 Stage G worlds of the genotype's length.
  measures    alive at g: the population has a member. Carries the allele: at least half of its members contain the
              mutant's 5-byte window centred on the mutated position, cyclically (`serial_retention.window_codes`).
              Identifiable: the window does not occur cyclically in the wild type (others excluded and counted).
              Background b_g(i, v): share of alive control populations that carry the same window (same rule).
              ok(i, v, g): (share of alive mutant populations carrying) - b_g >= 0.5, with share = 0 if none is alive and
              b = 0 if no control is alive (as `serial_retention.sites`); evaluated in exact integer arithmetic.
              Site (branching-transmissible) at g: ok for at least half of the position's identifiable values.
  predictions X1-1: pusher16 control populations alive at g = 8 in arm U >= 0.5. X1-2 (decision): pusher16 >= 1 site at
              g = 8 in arm U. X1-3: ret16 and ldir32 <= 2 sites at g = 8, genome20_s6003 >= 5 (arm U). Arm K reported.
Outputs (<out>): branching_summary.csv (genotype x arm x generation), branching_sites.csv (genotype x arm x generation x
position), branching_mutants.csv.gz (genotype x arm x generation x mutant), BRANCHING.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from algocell_exp import assay as A  # noqa: E402
from algocell_exp import exectrace as X  # noqa: E402
from algocell_exp.soup import adapter_summary  # noqa: E402
from serial_retention import R_RES, W, hx, panel, window_codes  # noqa: E402

KEYS = ("pusher16", "pusher64", "ret16", "ldir32", "genome20_s6003")
K_KEYS = ("pusher16", "pusher64")                       # arm K: pushers only
K_SNAPS = ("t5000", "t10000")                           # open-phase snapshots, steps 5,000 and 10,000
REG = {"values": 32, "reps": 4, "ctrl": 64, "k": 2, "cap": 32, "gens": 8}
STEPS = 128
COPY = 0.75
GENS_REPORT = (1, 2, 4, 8)
SEED = 20261010
ARM_ID = {"U": 0, "K": 1}
CHUNK = 1 << 18                                         # pairs per executor dispatch (32 MiB of pair memory at L = 64)


def execute(pairs: np.ndarray, L: int, sup: list, zh: bool) -> np.ndarray:
    out = np.empty_like(pairs)
    for s in range(0, len(pairs), CHUNK):
        out[s:s + CHUNK] = A.execute_pairs(pairs[s:s + CHUNK], L, STEPS, sup, None, zh)
    return out


def kin_pool(L: int) -> tuple[np.ndarray, str]:
    """All cells of the step-5,000 and step-10,000 snapshots of the 20 Stage G worlds at length L, pooled."""
    sys.path.insert(0, os.path.join(HERE, "manuscript", "figures"))
    import soup_stills as ss
    g = pd.read_csv(os.path.join(R_RES, "stageG", "stageG", "stage_g_runs.csv"))
    w = g[g.L == L].sort_values("seed")
    if len(w) != 20:
        raise SystemExit(f"arm K: expected 20 Stage G worlds at L = {L}, found {len(w)}; stopping (registered pool unavailable)")
    tapes, stems = [], []
    for r in w.itertuples():
        stem = os.path.join(HERE, "runs", "stageG", f"{r.label}_L{L}_st128_k4_s{r.seed}")
        files = {n: f for n, st, f in ss.snapshot_files(stem)}
        for name in K_SNAPS:
            if name not in files:
                raise SystemExit(f"arm K: {os.path.basename(stem)} has no {name} snapshot; stopping (registered pool unavailable)")
            tapes.append(ss.load(files[name], L))
        stems.append(os.path.basename(stem))
    pool = np.concatenate(tapes)
    desc = f"{len(stems)} worlds ({stems[0]} .. {stems[-1]}), snapshots {', '.join(K_SNAPS)}, {len(pool)} cells"
    return pool, desc


def partners(rng: np.random.Generator, n: int, L: int, pool: np.ndarray | None) -> np.ndarray:
    if pool is None:
        return rng.integers(0, 256, size=(n, L), dtype=np.uint8)
    return pool[rng.integers(0, len(pool), size=n)]


def mutants(gt: dict, n_values: int, seed: int):
    x = gt["tape"]
    L = x.size
    rng = np.random.default_rng([seed, zlib.crc32(gt["key"].encode())])
    pos, val = [], []
    for i in range(L):
        alt = np.array([v for v in range(256) if v != x[i]], dtype=np.uint8)
        vs = rng.choice(alt, size=n_values, replace=False)
        pos += [i] * n_values
        val += list(vs)
    pos, val = np.array(pos), np.array(val, dtype=np.uint8)
    M = pos.size
    mut = np.tile(x, (M, 1))
    mut[np.arange(M), pos] = val
    wcode = window_codes(mut)[np.arange(M), (pos - W // 2) % L]
    ident = ~np.isin(wcode, window_codes(x[None, :])[0])
    return pos, val, mut, wcode, ident


def executed_positions(gt: dict, seed: int) -> tuple[np.ndarray, float]:
    """Positions of the wild type fetched as instructions against 64 uniform random partners (annotation, as in S)."""
    x = gt["tape"]
    L = x.size
    rng = np.random.default_rng([seed, zlib.crc32(gt["key"].encode()), 99])
    Q = rng.integers(0, 256, size=(64, L), dtype=np.uint8)
    _, masks = X.execute_pairs_traced(np.concatenate([np.repeat(x[None], 64, 0), Q], 1), L, STEPS, suppress=list(gt["sup"]), zero_halts=gt["zh"])
    ex = X.exec_positions(masks, 2 * L)
    return ex[:, :L].any(axis=0), float(ex[:, L:].any(axis=1).mean())


def measure(T: np.ndarray, mask: np.ndarray, MR: int, R: int, wcode: np.ndarray):
    """Mutant populations are rows 0..MR-1 (mutant m, replicate r at row m*R + r); controls are rows MR.. ."""
    NP, cap, L = T.shape
    M = MR // R
    n_mem = mask.sum(axis=1)
    alive = n_mem > 0
    carry = np.zeros(MR, bool)
    wc_pop = np.repeat(wcode, R)
    for s in range(0, MR, 1024):
        e = min(s + 1024, MR)
        codes = window_codes(T[s:e].reshape(-1, L)).reshape(e - s, cap, L)
        cont = (codes == wc_pop[s:e, None, None]).any(axis=2) & mask[s:e]
        carry[s:e] = alive[s:e] & (2 * cont.sum(axis=1) >= n_mem[s:e])
    nc = 0
    bgc = np.zeros(M, np.int64)
    for q in range(MR, NP):
        if not alive[q]:
            continue
        nc += 1
        mem = T[q][mask[q]]
        codes = np.sort(window_codes(mem), axis=1)
        first = np.ones(codes.shape, bool)
        first[:, 1:] = codes[:, 1:] != codes[:, :-1]          # each window counted once per member
        u, cnt = np.unique(codes[first], return_counts=True)
        bgc += np.isin(wcode, u[2 * cnt >= len(mem)])
    alive_cnt = alive[:MR].reshape(M, R).sum(axis=1)
    carry_cnt = carry.reshape(M, R).sum(axis=1)
    if nc > 0:
        ok = 2 * (carry_cnt * nc - bgc * alive_cnt) >= alive_cnt * nc
    else:
        ok = 2 * carry_cnt >= alive_cnt
    ok &= alive_cnt > 0
    return {"n_mem": n_mem, "alive": alive, "alive_cnt": alive_cnt, "carry_cnt": carry_cnt, "nc": nc, "bgc": bgc, "ok": ok}


def branch_step(T: np.ndarray, mask: np.ndarray, rng: np.random.Generator, pool: np.ndarray | None, k: int, sup: list, zh: bool):
    """One generation. T (NP, cap, L) members, mask (NP, cap). Every member runs as A against k fresh partners; offspring
    that are >= 75% copies of their parent (best cyclic shift) are candidates; each population keeps all candidates if
    there are at most cap, else a uniform random subsample of cap (the cap smallest iid uniform keys). Parents do not
    carry over. Returns the new (T, mask), the per-encounter copy flags and the population index of each encounter."""
    NP, cap, L = T.shape
    pi, ji = np.nonzero(mask)
    n = pi.size
    par = np.repeat(T[pi, ji], k, axis=0)                          # member e, partner kk at row e*k + kk
    B = partners(rng, n * k, L, pool)
    off = execute(np.concatenate([par, B], axis=1), L, sup, zh)[:, L:]
    cp = A._best_shift_match_rows(off, par) >= COPY
    pp = np.repeat(pi, k)
    slot = np.repeat(ji, k) * k + np.tile(np.arange(k), n)
    cand = np.zeros((NP, cap * k, L), np.uint8)
    cok = np.zeros((NP, cap * k), bool)
    cand[pp, slot] = off
    cok[pp, slot] = cp
    key = rng.random((NP, cap * k))
    key[~cok] = np.inf                                             # only copies can join
    order = np.argsort(key, axis=1, kind="stable")[:, :cap]        # the cap smallest keys: uniform subsample
    Tn = np.take_along_axis(cand, order[:, :, None], axis=1)
    mn = np.isfinite(np.take_along_axis(key, order, axis=1))
    Tn[~mn] = 0
    return Tn, mn, cp, pp


def run_arm(gt: dict, arm: str, prm: dict, pool: np.ndarray | None, executed: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    x, sup, zh = gt["tape"], list(gt["sup"]), gt["zh"]
    L = x.size
    R, C, k, cap, G = prm["reps"], prm["ctrl"], prm["k"], prm["cap"], prm["gens"]
    pos, val, mut, wcode, ident = mutants(gt, prm["values"], SEED)
    M = pos.size
    MR = M * R
    NP = MR + C
    rng = np.random.default_rng([SEED, zlib.crc32(gt["key"].encode()), ARM_ID[arm]])
    T = np.zeros((NP, cap, L), np.uint8)
    T[:MR, 0] = np.repeat(mut, R, axis=0)
    T[MR:, 0] = x
    mask = np.zeros((NP, cap), bool)
    mask[:, 0] = True
    mrows, srows = [], []
    summ = []
    for gen in range(1, G + 1):
        t0 = time.time()
        T, mask, cp, pp = branch_step(T, mask, rng, pool, k, sup, zh)
        n = pp.size // k
        ms = measure(T, mask, MR, R, wcode)
        alive_cnt, carry_cnt, nc, bgc, ok = ms["alive_cnt"], ms["carry_cnt"], ms["nc"], ms["bgc"], ms["ok"]
        n_mem = ms["n_mem"]
        msz = n_mem[:MR].reshape(M, R)
        dm = pd.DataFrame({"key": gt["key"], "arm": arm, "gen": gen, "pos": pos, "val": val, "identifiable": ident,
                           "alive": alive_cnt / R, "carry_alive": np.where(alive_cnt > 0, carry_cnt / np.maximum(alive_cnt, 1), 0.0),
                           "bg": (bgc / nc) if nc else np.full(M, np.nan), "ok": ok,
                           "retained": carry_cnt / R, "erased": (alive_cnt - carry_cnt) / R, "lost": 1 - alive_cnt / R,
                           "mean_size": np.where(alive_cnt > 0, msz.sum(1) / np.maximum(alive_cnt, 1), 0.0)})
        mrows.append(dm)
        d = dm[dm.identifiable]
        s = d.groupby("pos").agg(n_ident=("ok", "size"), n_ok=("ok", "sum"), alive=("alive", "mean"), carry_alive=("carry_alive", "mean"),
                                 bg=("bg", "mean"), retained=("retained", "mean"), erased=("erased", "mean"), lost=("lost", "mean")).reset_index()
        s["frac_ok"] = s.n_ok / s.n_ident
        s["transmissible"] = 2 * s.n_ok >= s.n_ident
        s["executed"] = executed[s.pos.to_numpy()]
        s.insert(0, "gen", gen)
        s.insert(0, "arm", arm)
        s.insert(0, "key", gt["key"])
        srows.append(s)
        ctrl_alive = ms["alive"][MR:]
        is_ctrl_enc = pp >= MR
        summ.append({"key": gt["key"], "arm": arm, "gen": gen,
                     "ctrl_alive": float(ctrl_alive.mean()), "ctrl_alive_n": int(ctrl_alive.sum()),
                     "ctrl_mean_size": float(n_mem[MR:][ctrl_alive].mean()) if ctrl_alive.any() else 0.0,
                     "ctrl_copy_rate": float(cp[is_ctrl_enc].mean()) if is_ctrl_enc.any() else np.nan,
                     "mut_alive": float(d.alive.mean()), "mut_mean_size": float(np.average(d.mean_size, weights=d.alive)) if d.alive.sum() > 0 else 0.0,
                     "mut_copy_rate": float(cp[~is_ctrl_enc].mean()) if (~is_ctrl_enc).any() else np.nan,
                     "retained": float(d.retained.mean()), "erased": float(d.erased.mean()), "lost": float(d.lost.mean()),
                     "bg_mean": float(d.bg.mean()) if nc else np.nan, "frac_ok_mutants": float(d.ok.mean()),
                     "sites": int(s.transmissible.sum()), "sites_unexec": int((s.transmissible & ~s.executed).sum()),
                     "positions_scored": int(len(s)), "encounters": int(n * k), "gen_wall_s": round(time.time() - t0, 2)})
        print(f"  {gt['key']:>15} arm {arm} g{gen}: ctrl alive {summ[-1]['ctrl_alive']:.3f} (size {summ[-1]['ctrl_mean_size']:.1f}, copy/enc {summ[-1]['ctrl_copy_rate']:.3f}) "
              f"| mut alive {summ[-1]['mut_alive']:.3f} | sites {summ[-1]['sites']} | {n * k} enc, {summ[-1]['gen_wall_s']} s", flush=True)
    return pd.concat(mrows, ignore_index=True), pd.concat(srows, ignore_index=True), summ


def run(a) -> None:
    os.makedirs(a.out, exist_ok=True)
    prm = {"values": a.values, "reps": a.reps, "ctrl": a.ctrl, "k": a.k, "cap": a.cap, "gens": a.gens}
    registered = prm == REG
    P = {p["key"]: p for p in panel()}
    keys = [k for k in (a.only.split(",") if a.only else KEYS)]
    arms = a.arms.split(",")
    pools = {}
    allm, alls, allsum = [], [], []
    t_all = time.time()
    for key in keys:
        gt = P[key]
        L = gt["tape"].size
        executed, entered = executed_positions(gt, SEED)
        n_ident = None
        for arm in arms:
            if arm == "K" and key not in K_KEYS:
                continue
            pool, pdesc = None, "uniform random bytes"
            if arm == "K":
                if L not in pools:
                    pools[L] = kin_pool(L)
                pool, pdesc = pools[L]
                rs = np.random.default_rng([SEED, L, 7]).choice(len(pool), size=min(20000, len(pool)), replace=False)
                pool_sim = float((A._best_shift_match_rows(pool[rs], np.repeat(gt["tape"][None], len(rs), 0)) >= COPY).mean())
            else:
                pool_sim = np.nan
            t0 = time.time()
            dm, ds, summ = run_arm(gt, arm, prm, pool, executed)
            wall = round(time.time() - t0, 1)
            n_ident = int(dm[dm.gen == 1].identifiable.sum())
            for r in summ:
                r.update({"kind": gt["kind"], "L": L, "tape": hx(gt["tape"]), "rule": ("i8080" if gt["sup"] else "Z80") + (", lethal tar" if gt["zh"] else ""),
                          "executor": A.executor_file(L, None, gt["zh"]), "partners": pdesc, "pool_share_copy_of_wt": pool_sim,
                          "n_mutants": int((dm.gen == 1).sum()), "n_identifiable": n_ident, "entered_frac_wt": entered,
                          "executed_wt": "".join("1" if e else "0" for e in executed), "arm_wall_s": wall, "registered_params": registered, **prm})
            allm.append(dm)
            alls.append(ds)
            allsum += summ
            print(f"{key} arm {arm}: {wall} s", flush=True)
    sm = pd.DataFrame(allsum)
    front = ["key", "kind", "L", "arm", "gen"]
    sm = sm[front + [c for c in sm.columns if c not in front]]
    sm.to_csv(os.path.join(a.out, "branching_summary.csv"), index=False)
    pd.concat(alls, ignore_index=True).to_csv(os.path.join(a.out, "branching_sites.csv"), index=False)
    pd.concat(allm, ignore_index=True).to_csv(os.path.join(a.out, "branching_mutants.csv.gz"), index=False)
    total = time.time() - t_all
    with open(os.path.join(a.out, "runtime.txt"), "w") as f:
        f.write(f"total wall {total:.1f} s on {adapter_summary()}\n")
    print(f"done in {total:.0f} s")
    report(a)


def report(a) -> None:
    sm = pd.read_csv(os.path.join(a.out, "branching_summary.csv"))
    rt = open(os.path.join(a.out, "runtime.txt")).read().strip() if os.path.exists(os.path.join(a.out, "runtime.txt")) else "n/a"
    registered = bool(sm.registered_params.all())
    p0 = sm.iloc[0]
    out = ["# X1 — branching transfer: predictions X1-1 to X1-3 against the data (generated by `branching_transfer.py --report`)", "",
           "Registration: `REVISION_PREREG.md`, \"X1–X3 — three referee-driven tests\", X1 (committed at 244616b, before this script).",
           f"Parameters: {int(p0['values'])} values per position, {int(p0.reps)} populations per mutant, {int(p0.ctrl)} control populations, "
           f"k = {int(p0.k)} partners per member per generation, cap N = {int(p0.cap)}, G = {int(p0.gens)}; copy at >= {COPY} best cyclic shift; "
           f"{STEPS} instructions; seed {SEED}. Registered values: **{'yes' if registered else 'NO — predictions not scored'}**.",
           f"Runtime: {rt}.", ""]
    out.append("| genotype | arm | L | identifiable | control alive g = 1 / 2 / 4 / 8 | mutant populations alive g = 1 / 2 / 4 / 8 | "
               "sites g = 1 / 2 / 4 / 8 (unexecuted at 8) | lost / erased / retained g = 8 | mean background g = 8 | control copy rate per encounter g = 1 | control size g = 8 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    order = {k: i for i, k in enumerate(KEYS)}
    combos = sorted(sm[["key", "arm"]].drop_duplicates().itertuples(index=False), key=lambda r: (r.arm, order.get(r.key, 99)))
    G = int(p0.gens)
    gens = [g for g in GENS_REPORT if g <= G]

    def row(key, arm, g):
        r = sm[(sm.key == key) & (sm.arm == arm) & (sm.gen == g)]
        return r.iloc[0] if len(r) else None
    for key, arm in combos:
        rs = {g: row(key, arm, g) for g in gens}
        rl = rs[gens[-1]]
        out.append(f"| {key} | {arm} | {rl.L} | {rl.n_identifiable} of {rl.n_mutants} | " + " / ".join(f"{rs[g].ctrl_alive:.3f}" for g in gens) + " | "
                   + " / ".join(f"{rs[g].mut_alive:.3f}" for g in gens) + " | " + " / ".join(str(int(rs[g].sites)) for g in gens) + f" ({int(rl.sites_unexec)}) | "
                   f"{rl.lost:.2f} / {rl.erased:.2f} / {rl.retained:.2f} | {rl.bg_mean:.3f} | {rs[gens[0]].ctrl_copy_rate:.3f} | {rl.ctrl_mean_size:.1f} |")
    out.append("")
    out.append("Sites: positions at which, for at least half of the identifiable values, the share of alive mutant populations "
               "carrying the allele exceeds the background by at least 0.5. *lost / erased / retained*: shares of all identifiable "
               "mutant populations that are extinct / alive without the allele / alive and carrying it.")
    out.append("")
    res = []
    if not registered:
        res.append("Parameters differ from the registration: predictions are not scored.")
    elif G >= 8:
        p16 = row("pusher16", "U", 8)
        x11 = x12 = None
        if p16 is not None:
            x11 = p16.ctrl_alive >= 0.5
            x12 = p16.sites >= 1
            res.append(f"**X1-1** (pusher16 control populations alive at g = 8 in arm U >= 0.5): **{'met' if x11 else 'NOT met'}** — "
                       f"{int(p16.ctrl_alive_n)} of {int(p16.ctrl)} alive ({p16.ctrl_alive:.3f}).")
            res.append(f"**X1-2** (decision; pusher16 has >= 1 branching-transmissible site at g = 8 in arm U): **{'met' if x12 else 'NOT met'}** — "
                       f"{int(p16.sites)} sites (share of identifiable mutants passing the threshold {p16.frac_ok_mutants:.3f}; mutant populations alive {p16.mut_alive:.3f}).")
        parts = {k: row(k, "U", 8) for k in ("ret16", "ldir32", "genome20_s6003")}
        if all(v is not None for v in parts.values()):
            ok3 = parts["ret16"].sites <= 2 and parts["ldir32"].sites <= 2 and parts["genome20_s6003"].sites >= 5
            res.append(f"**X1-3** (ret16 and ldir32 <= 2 sites at g = 8, genome20_s6003 >= 5; arm U): **{'met' if ok3 else 'NOT met'}** — "
                       + "; ".join(f"{k}: {int(v.sites)}" for k, v in parts.items()) + ".")
        k16 = row("pusher16", "K", 8)
        kk = [row(k, "K", 8) for k in K_KEYS]
        if any(v is not None for v in kk):
            res.append("**Arm K** (partners from the open-phase Stage G soups, reported regardless): " + "; ".join(
                f"{v.key}: control alive at g = 8 {v.ctrl_alive:.3f}, sites at g = 8 {int(v.sites)}, mutant populations alive {v.mut_alive:.3f}, "
                f"share of pool cells that already are >= 75% copies of the wild type {v.pool_share_copy_of_wt:.3f}" for v in kk if v is not None) + ".")
        if x11 is not None:
            if not x11:
                reading = "X1-1 failed: branching does not rescue the pusher's lineages, and X1-2 is read as uninformative about alleles (registered reading)."
            elif x12:
                reading = "X1-2 met: the open replicator carries heritable variation through branching lineages; the paper says heritable variation precedes closure (registered reading)."
            elif k16 is not None and k16.sites < 1:
                reading = "X1-2 not met in both arms: the paper says that persistent heritable variation arrives with closed transmitters, and the summary and conclusions change accordingly (registered reading)."
            elif k16 is not None:
                reading = (f"X1-2 not met in arm U but pusher16 has {int(k16.sites)} sites at g = 8 in arm K: the registration gives no reading for this case "
                           "(its readings cover 'met' and 'not met in both arms'); reported as found.")
            else:
                reading = "X1-2 not met in arm U; arm K for pusher16 not run."
            res.append("**Reading.** " + reading)
    out += [f"- {r}" for r in res]
    out.append("")
    out.append("Per-generation numbers: `branching_summary.csv`; per-site: `branching_sites.csv`; per-mutant: `branching_mutants.csv.gz`.")
    out.append("")
    out.append("**Implementation readings** (points the registration leaves open; fixed in the script before the full run): "
               "(1) every population is founded by one tape; (2) generations do not overlap: a member's copies replace it, the parent does "
               "not carry over (as in S, where the lineage continues in the offspring); (3) every encounter draws its own partner "
               "(no partner sequences shared across mutants, unlike S's common random numbers, which X1 does not register); arm K draws "
               "with replacement from the pooled cells; (4) the allele window is the 5-byte window centred on the mutated position, as in S, and "
               "'tested values' are the identifiable ones (S's exclusion); (5) as in S, a mutant with no alive population has carrying share 0 and "
               "the background is 0 when no control population is alive; the 0.5 thresholds are evaluated in exact integer arithmetic; "
               "(6) the copy test compares each offspring with its parent's tape as it was before the encounter; (7) the Stage G worlds are "
               "resolved from `stage_g_runs.csv` (at L = 64 their label is `none@closure1M`); all 20 worlds at L = 16 and 64 have both "
               "registered snapshots, so no substitute snapshot was used; (8) the 'unexecuted' annotation uses the wild type's executed "
               "positions against 64 uniform random partners, in both arms.")
    txt = "\n".join(out)
    open(os.path.join(a.out, "BRANCHING.md"), "w").write(txt + "\n")
    print(txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R_RES, "branching"))
    ap.add_argument("--only", default="")
    ap.add_argument("--arms", default="U,K")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--values", type=int, default=REG["values"])
    ap.add_argument("--reps", type=int, default=REG["reps"])
    ap.add_argument("--ctrl", type=int, default=REG["ctrl"])
    ap.add_argument("--k", type=int, default=REG["k"])
    ap.add_argument("--cap", type=int, default=REG["cap"])
    ap.add_argument("--gens", type=int, default=REG["gens"])
    a = ap.parse_args()
    if a.report:
        report(a)
        return
    run(a)


if __name__ == "__main__":
    main()
