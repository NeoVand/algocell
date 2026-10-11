"""Emulator impact study (planning/IMPACT_STUDY_DRAFT.md): the C0 equivalence proofs, the dry run, and the registered run.

    python impact_study.py equiv                 # C0 machines vs Algocell's own executors on archived pairs → results/impact_equiv/
    python impact_study.py dryrun [--n 1000]     # 1,000 pairs per corpus and condition, all three cores → results/impact_dryrun/
    python impact_study.py run --registered      # THE REGISTERED RUN (100,000 pairs per condition); refuses without the flag
    python impact_study.py conditions            # list the corpus conditions

Machines: algocell_exp/machines/worlds_impact.json (impact_worlds.py). Corpora and seed streams: impact_corpus.py.
Measures and attribution: impact.py. Corpus 3: impact_assays.py, through the executor interface.
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
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
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
    row["aff_union"] = int((aff["c1"] | aff["c2"]).sum())
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


def corpus3_assays(items=None):
    """Every output column of the replication assay, under the three imp_* cores."""
    from algocell_exp.impact_assays import assay_item, panel
    return [assay_item(item) for item in (panel() if items is None else items)]


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


def IWorld(tag):
    return get_world(tag)


def cmd_report(a) -> None:
    from impact_report import report
    report(RESULTS / "impact_dryrun")


def cmd_exclusion(a) -> None:
    from impact_modal import exclusion
    exclusion()


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
