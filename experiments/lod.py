"""N3 (REVISION_PREREG): the exact line of descent of the first self-confined replicators.

Every step is recorded exactly (pairs and active flags, each active pair's memories before execution, after execution
and before mutation, and the soup after mutation). Each changed cell gets a birth record (copy / damage / novel /
mutation) whose parents are the records it came from; records no living cell descends from are freed by reference
counting, so the ancestry graph of the living population is kept in memory. Every copy event is re-run on the traced
executor and its executor's record is flagged confined (no partner byte fetched) or open.

    .venv/bin/python lod.py --validate            # V1-V3 on the seeded-closer world (L = 16, 5,000 steps)
    .venv/bin/python lod.py --seed 31001          # one registered world (L = 16, benign, zero registers)
Outputs: runs/lod/<name>/steps.jsonl (per 50 steps), lod.json (sampled lines of descent, C* events), validate.json
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from algocell_exp import exectrace as X  # noqa: E402
from algocell_exp.soup import Soup  # noqa: E402

THETA = 0.75
GAIN = 0.25     # amendment (2026-10-09 night): a copy must make >= L/4 positions newly match the executor at the copy's shift
PUSHER16 = np.array([0x01, 0xc5] * 8, np.uint8)
CLOSER16 = np.array([0xad, 0xe3, 0x21, 0xe3, 0x21, 0xc0, 0xad, 0xc0] * 2, np.uint8)
CONF, OPEN = 1, 2


class Rec:
    """Birth record of a tape. p1 = major parent (copy: the executor's record; damage and mutation: the cell's own
    previous record; novel: the larger contributor), p2 = minor parent (novel only). other = the overwritten tape
    (copy and damage: the cell's own previous tape; novel: None). flag: CONF / OPEN bits set when this record, as
    executor, made a copy; fstep = step of the last such flag."""
    __slots__ = ("step", "cell", "kind", "tape", "p1", "p2_tape", "p2_flag", "other", "flag", "fstep", "tag", "shift", "nown", "nexe")

    def __init__(self, step, cell, kind, tape, p1=None, p2=None, other=None, tag=None, shift=0, nown=0, nexe=0):
        self.step, self.cell, self.kind, self.tape = step, cell, kind, tape
        # the minor parent is kept as its tape and flag only, so its ancestry can be freed (the line follows p1)
        self.p2_tape = p2.tape if p2 is not None else None
        self.p2_flag = p2.flag if p2 is not None else None
        self.p1, self.other, self.tag, self.shift = p1, other, tag, shift
        self.flag, self.fstep, self.nown, self.nexe = 0, -1, nown, nexe


def best_shift(Xs: np.ndarray, Ys: np.ndarray):
    """For each row: max over cyclic shifts s of mean(X == roll(Y, s)), and the argmax shift."""
    L = Xs.shape[1]
    sims = np.stack([(Xs == np.roll(Ys, s, axis=1)).mean(1) for s in range(L)], 1)
    sh = sims.argmax(1)
    return sims[np.arange(len(Xs)), sh], sh


def _js(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


class World:
    def __init__(self, L: int, seed: int, zero_halts: bool = False, init: np.ndarray | None = None, tags=None):
        self.L = L
        self.soup = Soup(160, 125, "square", L, seed, 8192, 128, 4, [], zero_halts=zero_halts)
        self.zh = zero_halts
        if init is not None:
            self.soup.write_soup(init)
        self.cur = self.soup.read_soup()
        N = len(self.cur)
        self.recs = [Rec(0, c, "init", self.cur[c].tobytes(), tag=(tags[c] if tags is not None else None)) for c in range(N)]
        self.t = 0
        self.acc = {"copy": 0, "conf": 0, "open": 0, "damage": 0, "novel": 0, "mut": 0, "copyA": 0, "active": 0}
        self.window = []          # (step, conf, open) per step, for the closure measure
        self.v2_bad = 0
        self.v2_checked = 0
        self.v3_max = 0
        self.tagc = {}            # executor root tag -> [confined copy events, open copy events]

    def step(self, check_replay: bool = False):
        L, soup = self.L, self.soup
        pre = self.cur
        soup.step(1)
        self.t += 1
        t = self.t
        inter = soup.read_interactions()
        pm = soup.read_pair_memory()
        post = soup.read_soup()
        k = np.where(inter["active"])[0]
        I = inter["pairs"][k, 0].astype(np.int64)
        J = inter["pairs"][k, 1].astype(np.int64)
        A0, B0, A1, B1 = pre[I], pre[J], pm[k, 0], pm[k, 1]
        if check_replay:
            res, _ = X.execute_pairs_traced(np.concatenate([A0, B0], 1), L, 128, zero_halts=self.zh)
            self.v2_bad += int((res != np.concatenate([A1, B1], 1)).any(1).sum())
            self.v2_checked += len(k)
        absorbed = pre.copy()
        absorbed[I] = A1
        absorbed[J] = B1
        mut_rows = np.where((post != absorbed).any(1))[0]
        inpair = np.zeros(len(pre), bool)
        inpair[I] = True
        inpair[J] = True
        self.v3_max = max(self.v3_max, int(((post != pre).any(1) & ~inpair).sum()))
        chB = (B1 != B0).any(1)
        chA = (A1 != A0).any(1)
        simB, shB = best_shift(B1, A0)
        selfB = (B1 == B0).mean(1)
        rA = np.stack([np.roll(A0[q], shB[q]) for q in range(len(k))]) if len(k) else A0
        gainB = simB - (B0 == rA).mean(1)
        copyB = chB & (simB >= THETA) & (gainB >= GAIN)
        damB = chB & ~copyB & (selfB >= THETA)
        novB = chB & ~copyB & ~damB
        simA, shA = best_shift(A1, B0)
        selfA = (A1 == A0).mean(1)
        rB = np.stack([np.roll(B0[q], shA[q]) for q in range(len(k))]) if len(k) else B0
        gainA = simA - (A0 == rB).mean(1)
        copyA = chA & (simA >= THETA) & (gainA >= GAIN)       # the partner's tape copied into the executor's half
        damA = chA & ~copyA & (selfA >= THETA)
        novA = chA & ~copyA & ~damA
        # confinement of every copy event's executor
        ci = np.where(copyB)[0]
        conf = np.zeros(len(k), bool)
        if len(ci):
            _, m = X.execute_pairs_traced(np.concatenate([A0[ci], B0[ci]], 1), L, 128, zero_halts=self.zh)
            entered = X.exec_positions(m, 2 * L)[:, L:].any(1)
            conf[ci] = ~entered
        old = self.recs
        new = list(old)
        for q in ci:
            r = old[I[q]]
            r.flag |= CONF if conf[q] else OPEN
            r.fstep = t
            if r.tag is not None:
                self.tagc.setdefault(r.tag, [0, 0])[0 if conf[q] else 1] += 1
        for q in np.where(chB)[0]:
            j, i = J[q], I[q]
            if copyB[q]:
                new[j] = Rec(t, j, "copy", B1[q].tobytes(), p1=old[i], other=B0[q].tobytes(), tag=old[i].tag, shift=int(shB[q]))
            elif damB[q]:
                new[j] = Rec(t, j, "damage", B1[q].tobytes(), p1=old[j], other=B0[q].tobytes(), tag=old[j].tag)
            else:
                nown = int((B1[q] == B0[q]).sum())
                nexe = int((B1[q] == np.roll(A0[q], shB[q])).sum())
                maj, mnr = (old[j], old[i]) if nown >= nexe else (old[i], old[j])
                new[j] = Rec(t, j, "novel", B1[q].tobytes(), p1=maj, p2=mnr, tag=maj.tag, shift=int(shB[q]), nown=nown, nexe=nexe)
        for q in np.where(chA)[0]:
            i, j = I[q], J[q]
            if copyA[q]:
                new[i] = Rec(t, i, "copyA", A1[q].tobytes(), p1=old[j], other=A0[q].tobytes(), tag=old[j].tag, shift=int(shA[q]))
            elif damA[q]:
                new[i] = Rec(t, i, "damage", A1[q].tobytes(), p1=old[i], other=A0[q].tobytes(), tag=old[i].tag)
            else:
                nown = int((A1[q] == A0[q]).sum())
                nexe = int((A1[q] == np.roll(B0[q], shA[q])).sum())
                maj, mnr = (old[i], old[j]) if nown >= nexe else (old[j], old[i])
                new[i] = Rec(t, i, "novel", A1[q].tobytes(), p1=maj, p2=mnr, tag=maj.tag, shift=int(shA[q]), nown=nown, nexe=nexe)
        for c in mut_rows:
            new[c] = Rec(t, int(c), "mut", post[c].tobytes(), p1=new[c], tag=new[c].tag)
        self.recs = new
        self.cur = post
        a = self.acc
        a["active"] += len(k)
        a["copy"] += int(copyB.sum())
        a["conf"] += int(conf[ci].sum())
        a["open"] += int(len(ci) - conf[ci].sum())
        a["damage"] += int(damB.sum() + damA.sum())
        a["novel"] += int(novB.sum() + novA.sum())
        a["mut"] += len(mut_rows)
        a["copyA"] += int(copyA.sum())
        self.window.append((t, int(conf[ci].sum()), int(len(ci) - conf[ci].sum())))
        if len(self.window) > 500:
            self.window.pop(0)

    def confined_share(self) -> float:
        c = sum(w[1] for w in self.window)
        o = sum(w[2] for w in self.window)
        return c / (c + o) if c + o else float("nan")

    def flush(self) -> dict:
        d = {"step": self.t, **self.acc, "conf_share_500": self.confined_share(), "rss_mb": round(rss_mb(), 1)}
        self.acc = {k: 0 for k in self.acc}
        return d


def walk(rec: Rec, max_len: int = 10 ** 7):
    """The line of descent through major parents, newest first."""
    out = []
    r = rec
    while r is not None and len(out) < max_len:
        out.append(r)
        r = r.p1
    return out


def describe(r: Rec) -> dict:
    d = {"step": r.step, "cell": r.cell, "kind": r.kind, "tape": r.tape.hex(" "), "flag": r.flag, "tag": r.tag, "shift": r.shift}
    if r.other is not None:
        d["other"] = r.other.hex(" ")
    if r.kind == "novel":
        d.update({"nown": r.nown, "nexe": r.nexe, "p2_tape": r.p2_tape.hex(" ") if r.p2_tape is not None else None, "p2_flag": r.p2_flag})
    if r.p1 is not None:
        d.update({"p1_tape": r.p1.tape.hex(" "), "p1_flag": r.p1.flag, "p1_kind": r.p1.kind, "p1_step": r.p1.step})
    return d


def base_rate(w: World, n: int = 1000, window: int = 2000, seed: int = 0) -> dict:
    """Share of n random living cells whose line of descent contains a record flagged open copier (or confined copier)
    within `window` steps before now."""
    rng = np.random.default_rng(seed)
    cells = rng.choice(len(w.recs), size=n, replace=False)
    has_open = has_conf = 0
    for c in cells:
        r = w.recs[c]
        o = cf = False
        while r is not None and r.step >= w.t - window:
            o |= bool(r.flag & OPEN)
            cf |= bool(r.flag & CONF)
            r = r.p1
        has_open += o
        has_conf += cf
    return {"t": w.t, "n": n, "open_within": has_open / n, "conf_within": has_conf / n}


KIND_CODE = {"init": 0, "copy": 1, "damage": 2, "novel": 3, "mut": 4, "copyA": 5}


def analyse_lines(w: World, n: int = 64, recent: int = 500, seed: int = 0, window_open: int = 2000) -> dict:
    """Sample n recent confined copiers; walk each line of descent back to step 0. C* = the oldest record flagged
    confined; C** = the oldest confined record newer than the newest open-flagged record (start of the final confined
    stretch). Stores compact per-line arrays and the tapes needed for figures."""
    rng = np.random.default_rng(seed)
    cand = [c for c, r in enumerate(w.recs) if (r.flag & CONF) and r.fstep >= w.t - recent]
    if not cand:
        cand = [c for c, r in enumerate(w.recs) if (r.p1 is not None and (r.p1.flag & CONF))]
    pick = rng.choice(cand, size=min(n, len(cand)), replace=False) if cand else []
    lines = []
    for c in pick:
        line = walk(w.recs[c])
        flags = np.array([r.flag for r in line])
        steps = np.array([r.step for r in line])
        kinds = [r.kind for r in line]
        conf_idx = np.where(flags & CONF)[0]
        open_idx = np.where(flags & OPEN)[0]
        cstar = int(conf_idx.max()) if len(conf_idx) else None
        newest_open = int(open_idx.min()) if len(open_idx) else None
        cands = conf_idx[conf_idx < newest_open] if newest_open is not None else conf_idx
        css = int(cands.max()) if len(cands) else None
        before = range(cstar + 1, len(line)) if cstar is not None else range(0)
        open_before = [q for q in before if line[q].flag & OPEN]
        open_recent = [q for q in open_before if steps[cstar] - steps[q] <= window_open] if cstar is not None else []
        keep = sorted(set(range(0, len(line), max(1, len(line) // 300))) | set(range(max(0, (css or 0) - 8), min(len(line), (css or 0) + 12))))
        ev = {"cell": int(c), "line_len": len(line), "root_kind": kinds[-1], "root_tag": line[-1].tag, "root_tape": line[-1].tape.hex(" "),
              "n_conf_records": int(len(conf_idx)), "n_open_records": int(len(open_idx)), "kinds": {k: kinds.count(k) for k in set(kinds)},
              "steps": steps.tolist(), "kcode": [KIND_CODE.get(k, 9) for k in kinds], "flags": flags.tolist(),
              "tapes": {str(q): line[q].tape.hex(" ") for q in keep}}
        if cstar is not None:
            ev.update({"cstar": describe(line[cstar]), "cstar_index": cstar, "open_before_cstar": len(open_before), "open_within_window": len(open_recent),
                       "last_open_before": describe(line[open_before[0]]) if open_before else None,
                       "context": [describe(line[q]) for q in range(max(0, cstar - 3), min(len(line), cstar + 6))]})
        if css is not None:
            ev.update({"cstarstar": describe(line[css]), "cstarstar_index": css,
                       "newest_open": describe(line[newest_open]) if newest_open is not None else None,
                       "open_before_cstarstar": int(sum(1 for q in range(css + 1, len(line)) if line[q].flag & OPEN)),
                       "context2": [describe(line[q]) for q in range(max(0, css - 3), min(len(line), css + 8))]})
        lines.append(ev)
    return {"t": w.t, "n_candidates": len(cand), "lines": lines}


def validate(out: str, steps: int = 5000):
    os.makedirs(out, exist_ok=True)
    L = 16
    w0 = Soup(160, 125, "square", L, 9901, 8192, 128, 4, [])
    N = w0.cell_count
    del w0
    rng = np.random.default_rng([1, 16, 1, 33])
    cells = np.tile(PUSHER16, (N, 1))
    tags = np.array(["pusher"] * N, dtype=object)
    s = rng.choice(N, size=N // 100, replace=False)
    cells[s] = CLOSER16
    tags[s] = "closer"
    w = World(L, 9901, init=cells, tags=list(tags))
    t0 = time.time()
    with open(os.path.join(out, "steps.jsonl"), "w") as f:
        for _ in range(steps):
            # tally copy-event confinement by the executor's root tag before stepping (cheap proxy: tag of executor record)
            w.step(check_replay=(w.t < 200))
            if w.t % 50 == 0:
                f.write(json.dumps(w.flush()) + "\n")
    by_tag = w.tagc
    lines = analyse_lines(w, 64)
    roots = [ln["root_tag"] for ln in lines["lines"]]
    res = {"steps": steps, "wall_s": round(time.time() - t0, 1), "V2_checked_pairs": w.v2_checked, "V2_mismatches": w.v2_bad,
           "V3_max_changed_outside_pairs": w.v3_max, "mutation_count": w.soup.mutation_count,
           "V1_sampled": len(roots), "V1_root_closer": roots.count("closer"), "V1_root_pusher": roots.count("pusher"),
           "flags_by_tag_conf_open": by_tag, "final_conf_share_500": w.confined_share(), "rss_mb": rss_mb()}
    json.dump({"result": res, "lines": lines}, open(os.path.join(out, "validate.json"), "w"), indent=1, default=_js)
    print(json.dumps(res, indent=1, default=_js))


def run(seed: int, out: str, L: int = 16, cap: int = 80000, hold: int = 2000, extra: int = 5000):
    name = f"L{L}_benign_s{seed}"
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    w = World(L, seed)
    t0 = time.time()
    above_since, t_close, snaps, rates = None, None, {}, []
    with open(os.path.join(d, "steps.jsonl"), "w") as f:
        f.write(json.dumps({"kind": "condition", "L": L, "seed": seed, "cap": cap, "hold": hold, "extra": extra}) + "\n")
        while w.t < cap:
            w.step(check_replay=(w.t < 200))
            sh = w.confined_share()
            if w.t >= 500 and sh == sh and sh > 0.5:
                above_since = above_since if above_since is not None else w.t
                if t_close is None and w.t - above_since >= hold:
                    t_close = above_since
            else:
                above_since = None
            if w.t % 50 == 0:
                f.write(json.dumps(w.flush()) + "\n")
                f.flush()
            if w.t % 500 == 0:
                rates.append(base_rate(w, seed=w.t))
            if w.t in (5000, 20000, 40000) or (t_close is not None and w.t == t_close + hold):
                snaps[w.t] = analyse_lines(w, 16, seed=w.t)
            if t_close is not None and w.t >= t_close + hold + extra:
                break
    lines = analyse_lines(w, 64, seed=seed)
    res = {"seed": seed, "L": L, "t_end": w.t, "t_close": t_close, "wall_s": round(time.time() - t0, 1), "V2_mismatches": w.v2_bad,
           "V2_checked_pairs": w.v2_checked, "V3_max_changed_outside_pairs": w.v3_max, "rss_mb": rss_mb()}
    json.dump({"result": res, "lines": lines, "snapshots": {str(k): v for k, v in snaps.items()}, "base_rates": rates}, open(os.path.join(d, "lod.json"), "w"), default=_js)
    np.save(os.path.join(d, "final.npy"), w.cur)
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--cap", type=int, default=80000)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "lod"))
    a = ap.parse_args()
    if a.validate:
        validate(os.path.join(a.out, "validate"), a.steps)
    elif a.seed is not None:
        run(a.seed, a.out, cap=a.cap)
