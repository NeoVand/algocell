"""Emulator impact study (planning/IMPACT_STUDY_DRAFT.md): the C0 equivalence proofs, the dry run, and the registered run.

    python impact_study.py equiv                 # C0 machines vs Algocell's own executors on archived pairs → results/impact_equiv/
    python impact_study.py dryrun [--n 1000]     # 1,000 pairs per corpus and condition, all three cores → results/impact_dryrun/
    python impact_study.py run --registered      # THE REGISTERED RUN (100,000 pairs per condition); refuses without the flag
    python impact_study.py conditions            # list the corpus conditions

Machines: algocell_exp/machines/worlds_impact.json (impact_worlds.py). Corpora and seed streams: impact_corpus.py.
Measures and attribution: impact.py. Corpus 3 (the assays under the three cores) is pending: see `corpus3_assays`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from algocell_exp import impact as I
from algocell_exp import impact_corpus as K
from algocell_exp.impact_worlds import CORES, WORLDS, world as get_world
from algocell_exp.isa import masks as make_masks
from algocell_exp.isa import resolve
from algocell_exp.soup import adapter_summary
from algocell_exp.zmachine import load_machine

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EQ_STREAM = 0xE0          # the equivalence proofs' own stream (C0 only; no impact data)


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def md_table(rows: list[dict], cols: list[str]) -> str:
    def f(v):
        if isinstance(v, float):
            return f"{v:.3g}" if v and (abs(v) < 1e-3 or abs(v) >= 1e4) else f"{v:.4f}".rstrip("0").rstrip(".")
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        out += "| " + " | ".join(f(r.get(c, "")) for c in cols) + " |\n"
    return out


def git_head() -> str:
    import subprocess

    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def machines_provenance() -> dict:
    prov = (HERE / "algocell_exp" / "machines" / "PROVENANCE.md").read_text()
    commit = prov.split("commit `", 1)[1].split("`", 1)[0]
    return {"zilion_commit": commit}


MORE = 3   # exit code of a chunk that left work to do (run the same command again after a pause)


def chunk_loop(parts: Path, items: list, key, work, chunk_s: float, predict) -> tuple[list[dict], int]:
    """Resumable chunks: each item's result is appended to `parts` (JSON lines) as soon as it is done; items already in
    `parts` are skipped; a chunk stops starting new items once its elapsed time plus the item's predicted time would
    exceed chunk_s (it always runs at least one). Returns (the results of every done item, in item order; items left)."""
    done: dict = {}
    if parts.exists():
        for ln in parts.read_text().splitlines():
            if ln.strip():
                rec = json.loads(ln)
                done[rec["key"]] = rec
    t0 = time.time()
    ran = 0
    for it in items:
        k = key(it)
        if k in done:
            continue
        if ran and (time.time() - t0) + predict(it) > chunk_s:
            break
        rec = {"key": k, **work(it)}
        parts.parent.mkdir(parents=True, exist_ok=True)
        with parts.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[k] = rec
        ran += 1
    left = sum(1 for it in items if key(it) not in done)
    with (parts.parent / (parts.stem + "_chunks.jsonl")).open("a") as fh:
        fh.write(json.dumps({"start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), "s": round(time.time() - t0, 1), "items": ran, "left": left}) + "\n")
    log(f"chunk: {ran} items in {time.time() - t0:.0f} s, {left} left")
    return [done[key(it)] for it in items if key(it) in done], left


def chunk_stats(parts: Path) -> dict:
    f = parts.parent / (parts.stem + "_chunks.jsonl")
    ch = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    return {"chunks": len(ch), "chunk_s_total": round(sum(c["s"] for c in ch), 1), "chunk_s_max": max((c["s"] for c in ch), default=0)}


def masks_of(suppress) -> np.ndarray | None:
    return make_masks(resolve(list(suppress))) if suppress else None


# ── C0 equivalence proofs ──────────────────────────────────────────────────────────────────────────────────────────
def cmd_equiv(a) -> None:
    out = RESULTS / "impact_equiv"
    conds = K.conditions()
    by_world: dict = defaultdict(list)
    for c in conds:
        by_world[c.world].append(c)
    worlds = [w for w in WORLDS.values() if not a.only or w.tag in a.only.split(",")]
    parts = out / "_equiv_parts.jsonl"

    def work(w):
        groups_rows, controls = [], []
        t0 = time.time()
        m0 = load_machine(w.machine("c0"))
        own = by_world.get(w.tag, [])
        proxy_note = ""
        src_conds = own
        truncate = None
        if not own:
            if w.rule in ("randreg", "randsp"):
                src_conds = by_world.get(f"L{w.L}", [])
                proxy_note = f"archived pairs of the plain L = {w.L} conditions (no {w.rule} soup at L = {w.L})"
            elif w.tag == "L19":
                src_conds = by_world.get("L20", [])
                truncate = 19
                proxy_note = "no soup ran at L = 19: archived L = 20 pairs cut to 19 bytes per tape, plus random pairs"
        groups: dict = defaultdict(list)
        cache: dict = {}
        for c in src_conds:
            rng = K.rng_for(EQ_STREAM, 4, c.cid)
            slots = K._slots(c, a.per_cond, rng)
            p = K.pairs_of(c, slots, cache)
            if truncate:
                p = np.concatenate([p[:, :truncate], p[:, c.L: c.L + truncate]], axis=1)
            budget, sup = (c.budget, c.suppress) if not proxy_note else (128, ())
            groups[(budget, sup)].append(p)
        cache.clear()
        rng = np.random.default_rng([EQ_STREAM, 5, K.h32(w.tag)])
        rand = rng.integers(0, 256, size=(a.random, 2 * w.L), dtype=np.uint8)
        rmask = np.packbits(rng.random((24, 32)) < 1 / 6, axis=1, bitorder="little").view(np.uint32).reshape(24)
        seed = int(K.h32("equiv/" + w.tag))
        tot = Counter()
        n_groups = 0
        biggest = None
        for (budget, sup), parts in sorted(groups.items()):
            pairs = np.concatenate(parts)
            r = I.compare_c0(w, m0, pairs, budget, masks_of(sup), seed)
            n_groups += 1
            groups_rows.append({"world": w.tag, "budget": budget, "suppress": ";".join(sup) or "-", "conditions": len(parts), **r})
            tot.update(r)
            if biggest is None or pairs.shape[0] > biggest[0].shape[0]:
                biggest = (pairs, budget, sup)
        for what, masks in (("random pairs, no masks", None), ("random pairs, random masks (1 in 6)", rmask)):
            r = I.compare_c0(w, m0, rand, 128, masks, seed + 7)
            groups_rows.append({"world": w.tag, "budget": 128, "suppress": what, "conditions": 0, **r})
            tot.update({k: v for k, v in r.items()})
            n_groups += 1
        # negative controls: the comparison must see a one-instruction budget difference, and (PCG worlds) a seed difference
        cp, cb, csup = biggest if biggest is not None else (rand, 128, ())
        sub = cp[: a.control]
        r = I.compare_c0(w, m0, sub, cb, masks_of(csup), seed, budget_delta=-1)
        controls.append({"world": w.tag, "control": f"machine budget {cb - 1} instead of {cb}", **r})
        if w.rule in ("randreg", "randsp") or (w.rule == "dial" and 0 < w.p < 1):   # at p = 0 and 1 no decision reads the stream
            r = _seed_control(w, m0, sub, cb, masks_of(csup), seed)
            controls.append({"world": w.tag, "control": "machine seed + 1 (PCG stream)", **r})
        row = {"world": w.tag, "machine": w.machine("c0"), "executor": w.algocell_executor()[0] + (f" ({w.algocell_executor()[1]})" if w.algocell_executor()[1] else ""),
               "conditions": len(own), "groups": n_groups, **{k: int(tot[k]) for k in ("pairs", "mem_changed", "b_written", "halted", "mem_mismatch", "wc_mismatch", "reg_mismatch")},
               "source": proxy_note or ("archived pairs of its own conditions" if own else "random pairs only"), "s": round(time.time() - t0, 1)}
        log(row)
        if row["mem_mismatch"] or row["wc_mismatch"] or row["reg_mismatch"]:
            log(f"MISMATCH in {w.tag}")
        return {"row": row, "groups": groups_rows, "controls": controls}

    done, left = chunk_loop(parts, worlds, lambda w: w.tag, work, a.chunk_s, lambda w: 12.0)
    if left:
        sys.exit(MORE)
    rows = [d["row"] for d in done]
    groups_rows = [g for d in done for g in d["groups"]]
    controls = [c for d in done for c in d["controls"]]
    cs = chunk_stats(parts)
    write_csv(out / "equiv_worlds.csv", rows)
    write_csv(out / "equiv_groups.csv", groups_rows)
    write_csv(out / "equiv_controls.csv", controls)
    bad = [r for r in rows if r["mem_mismatch"] or r["wc_mismatch"] or r["reg_mismatch"]]
    ctrl_bad = [c for c in controls if not (c["mem_mismatch"] or c["reg_mismatch"] or c["wc_mismatch"])]
    body = (f"# C0 machines against Algocell's own executors (generated by `impact_study.py equiv`)\n\n"
            f"Git {git_head()}, zilion {machines_provenance()['zilion_commit'][:12]}, adapter {adapter_summary()}, {time.strftime('%Y-%m-%d %H:%M')}; "
            f"{sum(r['s'] for r in rows):.0f} s of work in {cs['chunks']} resumable chunks (longest {cs['chunk_s_max']:.0f} s).\n\n"
            "For every world of the impact study, its C0 machine (`imp_<world>_c0`, Zilion compat '0.2') and Algocell's own "
            "single-pair executor for that world (the exported `shader/z80_test_*.wgsl`; `MEM_LENGTH` derivation for L = 19, "
            "see impact_worlds.py) run the same pairs with the same step budget, suppression masks and run seed (one dispatch "
            "each, lane = case index). Pairs: lattice-neighbour pairs (well-mixed partners for Stage H) drawn from the archived "
            f"soups of every registered condition run in that world ({a.per_cond:,} per condition, stream 0xE0, uniformly over the "
            "condition's worlds and cells), grouped by the condition's budget and suppression set; plus "
            f"{a.random:,} uniform random pairs without masks and with random masks (1 opcode in 6 on every page). Compared per "
            "pair: the final memory (2L bytes), writes into A and into B (+ padding), and the registers a f b c d e h l sp pc.\n\n"
            f"**Result: {len(rows) - len(bad)} of {len(rows)} worlds with 0 mismatches"
            f"{'' if not bad else ' — MISMATCHES in ' + ', '.join(r['world'] for r in bad)}.** Negative controls: "
            f"{len(controls) - len(ctrl_bad)} of {len(controls)} show mismatches, as they must"
            f"{'' if not ctrl_bad else ' — CONTROL FAILED: ' + ', '.join(c['world'] for c in ctrl_bad)}.\n\n"
            "## Per world\n\n" + md_table(rows, list(rows[0])) +
            "\n`mem_changed`: pairs whose memory execution changed; `b_written`: pairs with a write into B; `halted`: lanes the "
            "machine halted (non-vacuity of the comparison).\n\n## Negative controls (must show mismatches)\n\n" +
            md_table(controls, list(controls[0])) + "\n## Per budget and suppression group\n\nIn `equiv_groups.csv`.\n")
    (out / "EQUIV.md").write_text(body)
    log(f"wrote {out / 'EQUIV.md'}")
    parts.unlink()
    (parts.parent / (parts.stem + "_chunks.jsonl")).unlink()
    if bad or ctrl_bad:
        sys.exit(1)


def _seed_control(w, m0, pairs, budget, masks, seed) -> dict:
    a_mem, a_regs = I.algocell_execute(w, pairs, budget, masks, seed)
    r = m0.run(pairs, budget=budget, ablation=masks, seed=seed + 1, lane_base=0)
    wb = r.writes("B")
    return {"pairs": int(pairs.shape[0]), "mem_mismatch": int((a_mem != r.memory).any(axis=1).sum()),
            "wc_mismatch": int(((a_regs[:, 10] != r.writes("A")) | (a_regs[:, 11] != wb)).sum()),
            "reg_mismatch": int(np.any(np.stack([a_regs[:, j] != r.reg(nm) for j, nm in enumerate(I.REGS10)], axis=1), axis=1).sum())}


# ── corpora runner (dry run and registered run) ───────────────────────────────────────────────────────────────────────
ATTR_BATCH = 2048
BLOCK = 1 << 16


def run_condition(c: K.Condition, pairs: np.ndarray, seed: int, triples: dict, census: bool = False) -> tuple[dict, list[dict]]:
    """All three cores on the condition's pairs (lane = row index), the affected sets, and the attribution of every
    affected pair (census=True: every pair goes through the repaired pass, so every event is classed). Returns (summary
    row, one row per affected pair and core)."""
    tr = triples.get(c.world) or triples.setdefault(c.world, I.Triple(c.world))
    masks = masks_of(c.suppress)
    t0 = time.time()
    res = {k: [] for k in CORES}
    for s0 in range(0, pairs.shape[0], BLOCK):
        r = tr.run(pairs[s0: s0 + BLOCK], c.budget, masks, seed, lane_base=s0)
        for k in CORES:
            res[k].append(r[k])
    res = {k: _cat(v) for k, v in res.items()}
    aff = I.affected(res)
    t_exec = time.time() - t0
    n = pairs.shape[0]
    row = {"cid": c.cid, "family": c.family, "tier": c.tier, "world": c.world, "budget": c.budget, "suppress_n": len(c.suppress),
           "grid": c.grid, "snapshot": c.snapshot, "sources": c.n_sources, "pairs": n,
           "mem_changed_c0": int((res["c0"].memory != pairs).any(axis=1).sum()), "halted_c0": int((res["c0"].halt_reason != 0).sum())}
    for k in ("c1", "c2"):
        kk = int(aff[k].sum())
        lo, hi = I.clopper_pearson(kk, n)
        row.update({f"aff_{k}": kk, f"rate_{k}": kk / n, f"cp_lo_{k}": lo, f"cp_hi_{k}": hi})
    row["c1_c2_differ"] = int((res["c1"].memory != res["c2"].memory).any(axis=1).sum())
    t1 = time.time()
    per_pair = []
    Att = I.Attributor if tr.w.rule == "dial" else I.FastAttributor
    at = Att(tr, c.budget, masks, seed)
    idx = np.arange(n) if census else np.nonzero(aff["c1"] | aff["c2"])[0]
    cls_counts = {k: Counter() for k in ("c1", "c2")}
    ev_counts = {k: Counter() for k in ("c1", "c2")}
    ev_lanes = {k: 0 for k in ("c1", "c2")}
    first_not_decisive = {k: 0 for k in ("c1", "c2")}     # affected pairs whose first event is not the decisive one
    first_other_class = {k: 0 for k in ("c1", "c2")}      # ... and of another class (a first-divergence rule would misclass)
    for b0 in range(0, len(idx), ATTR_BATCH):
        sel = idx[b0: b0 + ATTR_BATCH]
        meas = {k: I._subset(res[k], sel) for k in CORES}
        out = at.attribute(pairs[sel], sel, meas, census=census)
        for k in ("c1", "c2"):
            for j, lr in zip(sel, out[k]):
                if lr.events:
                    ev_lanes[k] += 1
                    ev_counts[k].update(e.cls for e in lr.events)
                if lr.decisive_class is None:
                    continue
                cls_counts[k][lr.decisive_class] += 1
                dec = next(e for e in lr.events if e.step == lr.decisive_step and e.probe_differs)
                first = lr.events[0]
                first_not_decisive[k] += first.step != lr.decisive_step
                first_other_class[k] += first.cls != lr.decisive_class
                per_pair.append({"cid": c.cid, "core": k, "lane": int(j), "decisive_class": lr.decisive_class,
                                 "decisive_step": lr.decisive_step, "decisive_insn": dec.insn, "decisive_diff": dec.diff,
                                 "first_event_step": first.step, "first_event_class": first.cls,
                                 "propagating": "+".join(lr.propagating), "events": len(lr.events),
                                 "event_classes": "+".join(sorted({e.cls for e in lr.events})), "pair_hex": pairs[j].tobytes().hex()})
    for k in ("c1", "c2"):
        for cl in I.CLASSES:
            row[f"{k}_{cl}"] = cls_counts[k][cl]
        row[f"{k}_first_not_decisive"] = first_not_decisive[k]
        row[f"{k}_first_other_class"] = first_other_class[k]
        if census:
            row[f"{k}_lanes_with_events"] = ev_lanes[k]
            for cl in I.CLASSES:
                row[f"{k}_events_{cl}"] = ev_counts[k][cl]
    row["attr_dispatches"] = at.stats["dispatches"]
    row["t_exec_s"] = round(t_exec, 3)
    row["t_attr_s"] = round(time.time() - t1, 3)
    return row, per_pair


def _cat(rs):
    if len(rs) == 1:
        return rs[0]
    from algocell_exp.zmachine import ZResult

    return ZResult(memory=np.concatenate([r.memory for r in rs]), steps=np.concatenate([r.steps for r in rs]),
                   halt_reason=np.concatenate([r.halt_reason for r in rs]), trap_cause=np.concatenate([r.trap_cause for r in rs]),
                   write_counts=np.concatenate([r.write_counts for r in rs]), write_count_names=rs[0].write_count_names,
                   state=np.concatenate([r.state for r in rs]), state_names=rs[0].state_names, record=np.concatenate([r.record for r in rs]))


def corpus3_assays(*_a, **_k):
    """Corpus 3 of the draft (the replication assay on every tape the manuscript names, under C0, C1 and C2): PENDING.
    It needs assay.py's assays to run on Zilion machines (built on branch feat/assays). The hook: given
    `assay_on_machine(machine_name, tapes, **assay_kwargs) -> scores` from that work, run it with the three machines of
    each named tape's world (impact_worlds.World.machines()) and report, per tape, whether every score is identical under
    C0, C1 and C2 (draft measure 'assay identity')."""
    raise NotImplementedError("corpus 3 (assays under C0/C1/C2) is pending: it needs assay.py on Zilion machines (feat/assays)")


def cmd_conditions(a) -> None:
    for c in K.random_conditions() + K.conditions():
        print(f"{c.tier:6} {c.cid:70} {c.world:12} budget {c.budget:4} masks {len(c.suppress):3} {c.grid:7} sources {c.n_sources}")


def cmd_dryrun(a) -> None:
    out = RESULTS / "impact_dryrun"
    out.mkdir(parents=True, exist_ok=True)
    conds = K.random_conditions() + K.conditions()
    if a.only:
        conds = [c for c in conds if any(x in c.cid for x in a.only.split(","))]
    parts = out / "_dryrun_parts.jsonl"
    triples: dict = {}

    def work(c):
        t0 = time.time()
        if c.tier == "random":
            pairs, redrawn = K.dry_random(c, a.n, a.n_registered)
        else:
            slots, redrawn = K.dry_slots(c, a.n, a.n_registered)
            pairs = K.pairs_of(c, slots)
        t_draw = time.time() - t0
        row, per_pair = run_condition(c, pairs, K.run_seed(K.DRY_STREAM, c.cid), triples, census=True)
        row["t_draw_s"] = round(t_draw, 3)
        row["redrawn"] = redrawn
        log(f"{c.cid}: C1 {row['aff_c1']} C2 {row['aff_c2']} | draw {t_draw:.2f}s exec {row['t_exec_s']}s attr {row['t_attr_s']}s")
        if len(triples) > 6:   # keep compiled pipelines modest
            triples.pop(next(iter(triples)))
        return {"row": row, "pairs": per_pair}

    predict = lambda c: 14.0 if IWorld(c.world).rule == "dial" else 1.5 + 0.022 * c.budget  # noqa: E731  (seconds, from a first run)
    done, left = chunk_loop(parts, conds, lambda c: c.cid, work, a.chunk_s, predict)
    if left:
        sys.exit(MORE)
    rows = [d["row"] for d in done]
    pairs_rows = [x for d in done for x in d["pairs"]]
    cs = chunk_stats(parts)
    active = sum(r["t_draw_s"] + r["t_exec_s"] + r["t_attr_s"] for r in rows)
    write_csv(out / "dryrun_conditions.csv", rows)
    write_csv(out / "dryrun_affected_pairs.csv", pairs_rows)
    (out / "dryrun_meta.json").write_text(json.dumps({"n": a.n, "n_registered": a.n_registered, "work_s": round(active, 1), **cs,
                                                       "git": git_head(), **machines_provenance(), "adapter": adapter_summary(),
                                                       "dry_stream": K.DRY_STREAM, "reg_stream": K.REG_STREAM}, indent=1))
    parts.unlink()
    (parts.parent / (parts.stem + "_chunks.jsonl")).unlink()
    log(f"dry run: {len(rows)} conditions, {active:.0f} s of work in {cs['chunks']} chunks")


BENCH = (("L16", 32), ("L16", 128), ("L16", 512), ("L36", 128), ("L64", 128), ("L100", 128), ("L100", 512))


def cmd_bench(a) -> None:
    """Throughput at the registered scale, for the projection: 100,000 random pairs (dry stream; their condition ids are
    not corpus ids, so they share no stream with a registered draw) through the three cores, and one attribution batch of
    ATTR_BATCH lanes (census mode, which steps every lane: an upper bound on a batch of affected lanes). Timing only: no
    rate is reported. One item per (world, budget), resumable chunks."""
    out = RESULTS / "impact_dryrun"
    parts = out / "_bench_parts.jsonl"

    def work(item):
        tag, budget = item
        tr = I.Triple(tag)
        c = K.Condition(f"bench/{tag}_st{budget}", "bench", "random", tag, budget, L=tr.L, P=tr.P)
        pairs, _ = K.dry_random(c, a.n, 0)
        tr.run(pairs[:1000], budget, None, 1)          # warm-up (pipelines)
        t0 = time.time()
        for s0 in range(0, pairs.shape[0], BLOCK):
            tr.run(pairs[s0: s0 + BLOCK], budget, None, 1, lane_base=s0)
        t_exec = time.time() - t0
        sub = pairs[:ATTR_BATCH]
        res = tr.run(sub, budget, None, 1)
        t0 = time.time()
        I.FastAttributor(tr, budget, None, 1).attribute(sub, np.arange(len(sub)), res, census=True)
        t_attr = time.time() - t0
        row = {"world": tag, "budget": budget, "pairs": a.n, "t_exec_s": round(t_exec, 2), "pairs_per_s": round(a.n / t_exec),
               "attr_batch_lanes": len(sub), "t_attr_batch_s": round(t_attr, 2)}
        log(row)
        return {"row": row}

    done, left = chunk_loop(parts, list(BENCH), lambda it: f"{it[0]}/{it[1]}", work, a.chunk_s, lambda it: 4 + 0.06 * it[1])
    if left:
        sys.exit(MORE)
    write_csv(out / "bench.csv", [d["row"] for d in done])
    parts.unlink()
    (parts.parent / (parts.stem + "_chunks.jsonl")).unlink()


def _read_csv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _interp(points: dict, x: float) -> float:
    """Piecewise-linear interpolation over {x: y}, constant beyond the ends."""
    xs = sorted(points)
    if x <= xs[0]:
        return points[xs[0]]
    if x >= xs[-1]:
        return points[xs[-1]]
    for a, b in zip(xs, xs[1:]):
        if a <= x <= b:
            return points[a] + (points[b] - points[a]) * (x - a) / (b - a)
    raise AssertionError


def projection(rows: list[dict], bench: list[dict], n_reg: int = 100000) -> dict:
    """Projected wall time of the registered run from the dry run (per-condition load times and affected counts) and the
    bench (execution throughput and attribution batch cost at the registered scale)."""
    kexec = {}   # seconds per pair per step, by L, from the budget-128 bench rows
    kattr = {}   # seconds per attribution batch per step, by L
    for b in bench:
        L, T = int(b["world"][1:]), int(b["budget"])
        if T == 128:
            kexec[L] = float(b["t_exec_s"]) / (int(b["pairs"]) * T)
            kattr[L] = float(b["t_attr_batch_s"]) / T
    out = {}
    for name, keep in (("draft", lambda r: r["tier"] in ("random", "draft")), ("draft+added", lambda r: True)):
        tl = te = ta = ta_hi = 0.0
        n = 0
        for r in rows:
            if not keep(r):
                continue
            n += 1
            L = int(IWorld(r["world"]).L)
            T = int(r["budget"])
            tl += float(r["t_draw_s"])
            te += n_reg * T * _interp(kexec, L)
            pairs = int(r["pairs"])
            ku = max(int(r["aff_c1"]), int(r["aff_c2"]))          # lower bound of the union; the union is <= c1 + c2
            ku_hi = int(r["aff_c1"]) + int(r["aff_c2"]) + 3        # conservative: + 3 (rule of three at zero)
            for k, acc in ((ku, "lo"), (ku_hi, "hi")):
                A = k / pairs * n_reg
                batches = int(np.ceil(A / ATTR_BATCH))
                t = batches * T * _interp(kattr, L)
                if acc == "lo":
                    ta += t
                else:
                    ta_hi += t
        out[name] = {"conditions": n, "load_s": tl, "exec_s": te, "attr_s": ta, "attr_hi_s": ta_hi,
                     "total_s": tl + te + ta, "total_hi_s": tl + te + ta_hi}
    return out


def IWorld(tag):
    return get_world(tag)


def _exclusion_line(out: Path) -> str:
    ex = _read_csv(out / "exclusion.csv")
    ov = sum(int(r["overlap"]) for r in ex)
    same = sum(1 for r in ex if r["redrawn"] == r["dry_run_redrawn"])
    return (f" Checked afterwards by `impact_study.py exclusion` (`exclusion.csv`): every condition's dry-run draws re-derived and "
            f"compared with its registered draws, overlap {ov} in {len(ex)} conditions; the re-derived redraw counts equal the dry "
            f"run's in {same} of {len(ex)}.")


def cmd_report(a) -> None:
    out = RESULTS / "impact_dryrun"
    rows = _read_csv(out / "dryrun_conditions.csv")
    aff_rows = _read_csv(out / "dryrun_affected_pairs.csv") if (out / "dryrun_affected_pairs.csv").stat().st_size > 1 else []
    meta = json.loads((out / "dryrun_meta.json").read_text())
    bench = _read_csv(out / "bench.csv") if (out / "bench.csv").exists() else []
    eq = _read_csv(RESULTS / "impact_equiv" / "equiv_worlds.csv")
    ints = lambda r, k: int(r[k])  # noqa: E731
    n_pairs = sum(ints(r, "pairs") for r in rows)

    def ci(k, n):
        lo, hi = I.clopper_pearson(k, n)
        return f"{k} ({k / n:.2g}; {lo:.2g}–{hi:.2g})" if k else f"0 (0–{hi:.2g})"

    def classes(rs, core):
        cnt = Counter()
        for r in rs:
            for cl in I.CLASSES:
                cnt[cl] += ints(r, f"{core}_{cl}")
        return ", ".join(f"{cl} {v}" for cl, v in cnt.items() if v) or "–"

    body = ["# Emulator impact study: dry run (not registered)\n",
            f"Generated by `impact_study.py report` from `dryrun_conditions.csv`, `dryrun_affected_pairs.csv`, `bench.csv`. "
            f"Dry run of {time.strftime('%Y-%m-%d')} at git {meta['git']} (zilion {meta['zilion_commit'][:12]}), {meta['adapter']}; the GPU "
            "was shared with other jobs and with the user's work throughout.\n",
            "## What ran\n",
            f"- **Cores.** C0 = `imp_<world>_c0` (Zilion compat '0.2', the 0.2.0 core), C1 = `imp_<world>_c1` (compat '0.3', classic), "
            f"C2 = `imp_<world>_c2` (compat '0.3', zilog-nmos), vendored in `algocell_exp/machines/worlds_impact.json` ({len(eq)} worlds, "
            f"{3 * len(eq)} machines). The three machines of a world share its world spec and SHA-256 and its hooks. Every C0 is "
            f"bit-identical to Algocell's own executor for its world on archived pairs (`results/impact_equiv/EQUIV.md`: "
            f"{sum(ints(r, 'pairs') for r in eq):,} pairs, 0 mismatches).",
            f"- **Corpora.** {meta['n']:,} pairs per condition: corpus 1, uniform random pairs at the 12 rings of the draft (128 "
            "steps); corpus 2, lattice-neighbour pairs (well-mixed partners for Stage H) from every registered condition's archived "
            "final soups and, where archived, emergence snapshots, each condition with its own budget, suppression masks and rule. "
            f"{len(rows)} conditions, {n_pairs:,} pairs, each run under C0, C1 and C2. Tier `draft`: the archives the draft names; "
            "tier `added`: registered archives the draft's list omits (see protocol issues).",
            "- **Measures.** A pair is affected under Cx when its final pair memory differs from C0's; 95% Clopper–Pearson "
            "intervals. Every pair (affected or not) went through the repaired pass (census), so every event on every pair was "
            "classed; every affected pair was attributed to its decisive event (`algocell_exp/impact.py`).",
            "",
            "## How the dry-run pairs are excluded from the registered corpora\n",
            f"The registered corpora are drawn from seed stream {meta['reg_stream']} (the draft's seed 1); the dry run from stream "
            f"{meta['dry_stream']} (0x{meta['dry_stream']:X}). Each stream is `SeedSequence([stream, corpus, h32(condition id)])`. "
            f"On top of the separate streams, the dry run regenerated each condition's registered draw ({meta['n_registered']:,} draws; "
            "indices only, never run) and redrew any dry-run draw that coincides with a registered one: for corpus 2 the draw is the "
            "slot (source soup, cell, partner), for corpus 1 the pair's bytes. Redraws needed: "
            f"{sum(ints(r, 'redrawn') for r in rows):,} (in {sum(1 for r in rows if ints(r, 'redrawn'))} conditions, those with few "
            "cells). The registered draws do not depend on the dry run. Two different slots of an evolved soup can hold identical "
            "byte pairs (clones); exclusion is by slot, which is what a draw is."
            + (_exclusion_line(out) if (out / "exclusion.csv").exists() else "") + "\n",
            ]
    # corpus 1
    c1 = [r for r in rows if r["tier"] == "random"]
    body.append("## Results\n\n### Corpus 1: random programs\n")
    t = [{"condition": r["cid"].split("/", 1)[1], "pairs": ints(r, "pairs"), "changed by C0": ints(r, "mem_changed_c0"),
          "C1 affected (rate; 95% CI)": ci(ints(r, "aff_c1"), ints(r, "pairs")), "C2 affected (rate; 95% CI)": ci(ints(r, "aff_c2"), ints(r, "pairs")),
          "C1 classes": classes([r], "c1"), "C2 classes": classes([r], "c2")} for r in c1]
    body.append(md_table(t, list(t[0])))
    # corpus 2 by family
    fams = defaultdict(list)
    for r in rows:
        if r["tier"] != "random":
            fams[(r["tier"], r["family"])].append(r)
    t = []
    for (tier, fam), rs in sorted(fams.items(), key=lambda x: (x[0][0] != "draft", x[0][1])):
        n = sum(ints(r, "pairs") for r in rs)
        k1, k2 = sum(ints(r, "aff_c1") for r in rs), sum(ints(r, "aff_c2") for r in rs)
        t.append({"tier": tier, "family": fam, "conditions": len(rs), "pairs": n,
                  "C1 affected (pooled rate; CI)": ci(k1, n), "conds C1>0": sum(1 for r in rs if ints(r, "aff_c1")),
                  "C2 affected (pooled rate; CI)": ci(k2, n), "conds C2>0": sum(1 for r in rs if ints(r, "aff_c2")),
                  "C1 classes": classes(rs, "c1"), "C2 classes": classes(rs, "c2")})
    body.append("\n### Corpus 2: archived soups, by family\n\nPooled over the family's conditions (each condition has its own "
                f"{meta['n']:,} pairs; the registered measure is per condition, in `dryrun_conditions.csv`).\n")
    body.append(md_table(t, list(t[0])))
    hit = [r for r in rows if r["tier"] != "random" and (ints(r, "aff_c1") or ints(r, "aff_c2"))]
    hit.sort(key=lambda r: (-ints(r, "aff_c2"), -ints(r, "aff_c1")))
    body.append(f"\n### Corpus 2: conditions with an affected pair ({len(hit)} of {len(rows) - len(c1)})\n")
    if hit:
        t = [{"condition": r["cid"], "tier": r["tier"], "world": r["world"], "budget": ints(r, "budget"), "masks": ints(r, "suppress_n"),
              "C1 affected": ci(ints(r, "aff_c1"), ints(r, "pairs")), "C2 affected": ci(ints(r, "aff_c2"), ints(r, "pairs")),
              "C1 classes": classes([r], "c1"), "C2 classes": classes([r], "c2")} for r in hit[: a_limit()]]
        body.append(md_table(t, list(t[0])))
        if len(hit) > a_limit():
            body.append(f"\n({len(hit) - a_limit()} more in `dryrun_conditions.csv`.)\n")
    # attribution
    tot = {k: Counter() for k in ("c1", "c2")}
    ev = {k: Counter() for k in ("c1", "c2")}
    for r in rows:
        for k in ("c1", "c2"):
            for cl in I.CLASSES:
                tot[k][cl] += ints(r, f"{k}_{cl}")
                ev[k][cl] += ints(r, f"{k}_events_{cl}")
    insn = {k: Counter() for k in ("c1", "c2")}
    for r in aff_rows:
        b = r["decisive_insn"].split()
        key = {"block-repeat": f"ED {b[1].upper()}", "bit-hl": f"CB {b[1].upper()}", "scf-ccf": " ".join(x.upper() for x in b[:2]) if b[0] in ("dd", "fd") else b[0].upper(),
               "wasted-prefix": f"{b[0].upper()} {b[1].upper()}"}.get(r["decisive_class"], r["decisive_insn"])
        insn[r["core"]][f"{r['decisive_class']} {key}"] += 1
    body.append("\n### Attribution\n")
    t = [{"core": k.upper(), "affected pairs": sum(tot[k].values()), **{f"decisive: {cl}": tot[k][cl] for cl in I.CLASSES},
          "census events (all pairs)": sum(ev[k].values()), **{f"events: {cl}": ev[k][cl] for cl in I.CLASSES}} for k in ("c1", "c2")]
    body.append(md_table(t, list(t[0])))
    body.append("\nDecisive instructions: " + "; ".join(f"{k.upper()}: " + ", ".join(f"{x} {v}" for x, v in insn[k].most_common()) for k in ("c1", "c2") if insn[k]) + ".\n")
    body.append("\n**'Other': none.** No event of any pair, affected or not, fell outside the documented classes (an 'other' "
                "raises and stops the run; the run completed). Every consistency check held: stepping C0 reproduced its "
                "single-dispatch run on every pair; every affected pair had an event and a propagating event; the probe of every "
                "affected pair's first event reproduced Cx's single-dispatch run.\n")
    # time
    wall = meta["work_s"]
    by = Counter()
    for r in rows:
        by["draw (load soups, regenerate registered draws)"] += float(r["t_draw_s"])
        by["execute (3 cores)"] += float(r["t_exec_s"])
        by["census + attribution"] += float(r["t_attr_s"])
    body.append(f"\n## Wall time\n\nDry run: {wall / 60:.1f} min of work for {len(rows)} conditions ({n_pairs:,} pairs × 3 cores, census "
                f"of every pair): " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in by.items()) + f". It ran in {meta['chunks']} "
                f"resumable foreground chunks of at most about 60 s (longest {meta['chunk_s_max']:.0f} s; {meta['chunk_s_total'] / 60:.1f} min "
                "in all, start-up included), each followed by an equal pause, because the machine was in use.\n")
    if bench:
        body.append("\nBench at the registered scale (timing only; dry stream, no rate reported):\n\n" + md_table(bench, list(bench[0])))
        pj = projection(rows, bench)
        body.append("\n### Projected wall time of the registered run\n\n100,000 pairs per condition, attribution of affected pairs "
                    f"only (no census), batches of {ATTR_BATCH:,}. Execution time scales with pairs × budget at the bench's per-L "
                    "throughput; attribution with the expected number of affected pairs (from the dry-run counts; 'conservative' adds "
                    "3 per core and condition) at the bench's batch cost; loading as measured in the dry run.\n\n")
        t = [{"scope": k, "conditions": v["conditions"], "load (h)": round(v["load_s"] / 3600, 2), "execute (h)": round(v["exec_s"] / 3600, 2),
              "attribution (h)": round(v["attr_s"] / 3600, 2), "total (h)": round(v["total_s"] / 3600, 2),
              "total, conservative (h)": round(v["total_hi_s"] / 3600, 2)} for k, v in pj.items()]
        body.append(md_table(t, list(t[0])))
    fnd = {k: sum(ints(r, f"{k}_first_not_decisive") for r in rows) for k in ("c1", "c2")}
    foc = {k: sum(ints(r, f"{k}_first_other_class") for r in rows) for k in ("c1", "c2")}
    lanes_ev = {k: sum(ints(r, f"{k}_lanes_with_events") for r in rows) for k in ("c1", "c2")}
    shas = {}
    for w in WORLDS.values():
        for k in CORES:
            m = json.loads((HERE / "algocell_exp/machines" / f"{w.machine(k)}.manifest.json").read_text())
            shas[(k, ",".join(m["core"]["hooks"]) or "none")] = m["sha256"]["core"]
    body.append(PROTOCOL.format(
        n_added=sum(1 for r in rows if r["tier"] == "added"), n_draft=sum(1 for r in rows if r["tier"] == "draft"),
        fnd1=fnd["c1"], fnd2=fnd["c2"], foc1=foc["c1"], foc2=foc["c2"], aff1=sum(tot["c1"].values()), aff2=sum(tot["c2"].values()),
        lev1=lanes_ev["c1"], lev2=lanes_ev["c2"], npairs=n_pairs, ev1=sum(ev["c1"].values()), ev2=sum(ev["c2"].values()),
        sha_table="\n".join(f"| {k.upper()} | {h} | `{v}` |" for (k, h), v in sorted(shas.items()))))
    (out / "REPORT.md").write_text("\n".join(body) + "\n")
    log(f"wrote {out / 'REPORT.md'}")


PROTOCOL = """
## Protocol issues found while building the tooling and running the dry run (proposed fixes; the draft is unchanged)

1. **Corpus 2 omits registered archives.** The draft lists "Stages A–F, lod_v6, N6 and the X1–X3 inputs", but Stages G, H, I,
   K, L and M also archive `*.soup_final.u8.br`, and DZ, C, C4a, C4a-rep, E, the R4 invasions (C4b/C4c), N5 and the earlier N3
   runs archive their final soups as `.npy`. *Fix:* corpus 2 = every registered condition with an archived final soup in
   any format; the registration lists them (here: {n_draft} `draft` and {n_added} `added` conditions, `impact_corpus.conditions()`).
2. **Budgets and conditions.** The world paragraph says 128 instructions; Stages A, C, D and E ran 32 to 2,048 (budget arms
   and L-scaled budgets). *Fix:* each condition runs at its own budget, masks and rule; a condition is a run label without
   its seed, and its emergence snapshots are a condition of their own. State whether emergence conditions enter the C1
   decision rule.
3. **Stage H is well mixed.** "Lattice-neighbour pairs" do not exist there. *Fix:* partners as the condition's grid draws
   them (Stage H: a uniform other cell; the square lattice: one of four neighbours, reflected at the edges).
4. **"Seed 1" does not fix the draw.** *Fix:* name the stream (`SeedSequence([1, corpus, h32(condition id)])`, h32 = first
   four bytes of the SHA-256 of the id), the draw order (source, x, y, direction) and, for the PCG worlds (dial, randreg,
   randsp), the run seed and lane (lane = the pair's index in its corpus), as implemented in `impact_corpus.py`.
5. **The threshold "rate below 1e-4" is ambiguous.** With 100,000 pairs a point estimate below 1e-4 allows 9 affected
   pairs; the 95% Clopper–Pearson upper bound is below 1e-4 only for 3 or fewer. Across ~680 conditions some condition can
   exceed either by chance. *Fix:* state which (proposed: the upper bound), and whether the rule is per condition or per
   family with a multiplicity correction.
6. **"The first instruction where the cores' states diverge" misattributes.** Deviations that never reach memory are common:
   every repeating LDIR/CPIR iteration differs in X/Y under C1 (PC bits 13 and 11 against bits of A + n) and is overwritten
   by the next iteration, and a wasted prefix leaves R different for the rest of the run. In this dry run {lev1:,} of
   {npairs:,} pairs had a C1 event ({ev1:,} events) and {lev2:,} a C2 event ({ev2:,}), almost all without effect. Of the
   affected pairs, the first divergence was not the decisive event in {fnd1} of {aff1} (C1) and {fnd2} of {aff2} (C2), and was
   of another class in {foc1} and {foc2}: a first-divergence rule would have misclassed those. *Fix:* the definition
   implemented here (`impact.py`): every step starts from C0's state (the repaired pass), so each deviation is an event of its
   own, classed from identical states with its exact difference pattern; the pair is attributed to its decisive event,
   the latest event whose unrepaired continuation changes the final memory (repairing every event up to it, and none after,
   reproduces C0's memory). The startMap is not needed: the PC in the per-step state record locates each instruction.
7. **"Maps out" (before registration, item 2).** The impact machines output memory, state and write counts; attribution
   does not need the fetch or start maps. If the registration wants maps, adding them changes the world SHA-256 and the C0
   proofs must be re-run.
8. **Control flow can change under C1.** "No conditional tests F3/F5, so control flow is never directly affected" holds for
   F3/F5, but the I/O repeats (INIR, INDR, OTIR, OTDR) also change H and P/V under C1 (the draft's own table), and P/V is
   tested by JP/CALL/RET PE/PO, H by DAA. *Fix:* amend the sentence; the attribution class covers it.
9. **One core SHA-256 per hook set.** The core text depends on the hooks (the lethal rule and the dial use
   `instructionStart`). *Fix:* the registration table lists:

| core | hooks | core SHA-256 |
|---|---|---|
{sha_table}

10. **Emergence snapshots** exist only as `soup_emergence.u8.br` (Stages A–M); the line-of-descent runs keep 37 unlabelled
    snapshots in `soups.npz`. *Fix:* "emergence" = `soup_emergence.u8.br`.
11. **X1–X3 inputs.** X3 used no soup (line records and uniform random partners); X1 (arm K) and X2 drew partners from Stage G
    snapshots. *Fix:* name them: steps 5,000 and 10,000 at L = 16 and 64 (X1, X2), step 5,000 at L = 20 and the per-world
    snapshot nearest 2 · t_rep at L = 50 and 64 (X2) (here 7 conditions).
12. **P = 38 needs L = 19**, at which no soup ran and no executor was exported. The C0 proof at L = 19 uses the executor
    export-sim.ts would write (`MEM_LENGTH` and the private memory size; the same derivation regenerates every exported plain
    and ring executor byte for byte) on random pairs and archived L = 20 pairs cut to 19 bytes. *Fix:* say so, or drop P = 38.
13. **The registered draw size is part of the exclusion.** The dry run was excluded from draws of 100,000 per condition; a
    different n is a different draw. `impact_study.py run` refuses any other n.
14. **C2 includes C1.** C2's affected pairs include C1's classes. *Fix (descriptive):* also report C2 against C1 (the Q latch
    alone); the dry run records `c1_c2_differ` per condition.

## Pending

- **Corpus 3 (assay identity).** The replication assay on every named tape under C0, C1 and C2 needs `assay.py` on Zilion
  machines (being built on branch feat/assays). Hook: `impact_study.corpus3_assays` (raises NotImplementedError) takes an
  `assay_on_machine(machine, tapes, ...)` and the three machines of each tape's world (`World.machines()`). The C1 decision
  rule needs it.
"""


def a_limit() -> int:
    return 60


def cmd_exclusion(a) -> None:
    """Re-derive, for every condition, the dry run's draws and the registered draws (indices or bytes only, nothing runs)
    and check that they are disjoint and that the dry-run draws are what the dry run used (same count of redraws)."""
    out = RESULTS / "impact_dryrun"
    ran = {r["cid"]: r for r in _read_csv(out / "dryrun_conditions.csv")}
    meta = json.loads((out / "dryrun_meta.json").read_text())
    rows = []
    for c in K.random_conditions() + K.conditions():
        if c.tier == "random":
            reg = {bytes(r) for r in K.draw_random(c, meta["n_registered"], K.REG_STREAM)}
            dry, redrawn = K.dry_random(c, meta["n"], meta["n_registered"])
            overlap = len({bytes(r) for r in dry} & reg)
            distinct = len(reg)
        else:
            reg = {tuple(r) for r in K.draw_slots(c, meta["n_registered"], K.REG_STREAM).tolist()}
            dry, redrawn = K.dry_slots(c, meta["n"], meta["n_registered"])
            overlap = len({tuple(r) for r in dry.tolist()} & reg)
            distinct = len(reg)
        r = ran.get(c.cid)
        rows.append({"cid": c.cid, "registered_draws": meta["n_registered"], "registered_distinct": distinct, "dry_draws": len(dry),
                     "overlap": overlap, "redrawn": redrawn, "dry_run_redrawn": int(r["redrawn"]) if r else -1})
    bad = [r for r in rows if r["overlap"] or r["redrawn"] != r["dry_run_redrawn"]]
    write_csv(out / "exclusion.csv", rows)
    log(f"{len(rows)} conditions: overlap total {sum(r['overlap'] for r in rows)}, redraws {sum(r['redrawn'] for r in rows)}, "
        f"mismatching the dry run: {len(bad)}")
    if bad:
        sys.exit(1)


def cmd_run(a) -> None:
    if not a.registered:
        sys.exit("refusing: the registered run needs --registered (and the impact study registered in REVISION_PREREG.md "
                 "with the three cores' SHA-256s); this tooling's dry run is `dryrun`")
    prereg = (HERE / "REVISION_PREREG.md").read_text()
    shas = {json.load(open(HERE / "algocell_exp/machines" / f"{w.machine(k)}.manifest.json"))["sha256"]["core"]
            for w in WORLDS.values() for k in ("c1", "c2")}
    missing = sorted(s for s in shas if s not in prereg)
    if missing:
        sys.exit(f"refusing: REVISION_PREREG.md does not carry the C1/C2 core SHA-256s {', '.join(m[:16] for m in missing)} "
                 "(the draft's registration step; one per core and hook set)")
    meta = json.loads((RESULTS / "impact_dryrun" / "dryrun_meta.json").read_text())
    if a.n != meta["n_registered"]:
        sys.exit(f"refusing: the dry run was excluded from registered draws of {meta['n_registered']:,} pairs per condition; "
                 f"a draw of {a.n:,} is a different draw (re-run the dry run with --n-registered {a.n})")
    out = RESULTS / "impact"
    out.mkdir(parents=True, exist_ok=True)
    conds = K.random_conditions() + K.conditions(tiers=tuple(a.tiers.split(",")))
    triples: dict = {}
    rows, pairs_rows = [], []
    for k, c in enumerate(conds):
        if c.tier == "random":
            pairs = K.draw_random(c, a.n, K.REG_STREAM)
        else:
            pairs = K.pairs_of(c, K.draw_slots(c, a.n, K.REG_STREAM))
        row, per_pair = run_condition(c, pairs, K.run_seed(K.REG_STREAM, c.cid), triples)
        rows.append(row)
        pairs_rows += per_pair
        log(f"[{k + 1}/{len(conds)}] {c.cid}: C1 {row['aff_c1']} C2 {row['aff_c2']}")
        write_csv(out / "conditions.csv", rows)
        write_csv(out / "affected_pairs.csv", pairs_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("equiv")
    e.add_argument("--per-cond", type=int, default=2000)
    e.add_argument("--random", type=int, default=20000)
    e.add_argument("--control", type=int, default=5000)
    e.add_argument("--only", default="")
    e.add_argument("--chunk-s", type=float, default=60.0)
    d = sub.add_parser("dryrun")
    d.add_argument("--n", type=int, default=1000)
    d.add_argument("--n-registered", type=int, default=100000)
    d.add_argument("--only", default="")
    d.add_argument("--chunk-s", type=float, default=60.0)
    r = sub.add_parser("run")
    r.add_argument("--registered", action="store_true")
    r.add_argument("--n", type=int, default=100000)
    r.add_argument("--tiers", default="draft")
    sub.add_parser("conditions")
    b = sub.add_parser("bench")
    b.add_argument("--n", type=int, default=100000)
    b.add_argument("--chunk-s", type=float, default=60.0)
    sub.add_parser("report")
    sub.add_parser("exclusion")
    a = ap.parse_args()
    {"equiv": cmd_equiv, "dryrun": cmd_dryrun, "run": cmd_run, "conditions": cmd_conditions, "bench": cmd_bench, "report": cmd_report, "exclusion": cmd_exclusion}[a.cmd](a)


if __name__ == "__main__":
    main()
