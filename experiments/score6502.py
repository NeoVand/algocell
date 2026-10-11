"""S1 scoring with the audited sequential-PC-wrap correction.

This script was validated only on handmade inputs and archived Z80 DZ lp0 worlds
before inspecting any registered S1 output. See results/s6502/SCORER_CONTROL.md.
The DZ partner source, counts, seeds, cache and class estimator are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import brotli
import numpy as np
import pandas as pd

from algocell_exp import assay as A, executor as E, exectrace as X
from dial_soups import _km_median, hx
from population_classes import classify

HERE = Path(__file__).resolve().parent
FRACS = ("frac_heritable", "frac_confined_of_heritable", "frac_regenerator_of_heritable",
         "frac_intermediate_of_heritable", "frac_transmitter_of_heritable")
CONTROL_COLUMNS = ("t_rep", "t_closed", "first_tape", "first_open", *FRACS,
                   "median_unexec_sites_transmitters")


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, result: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")


def pc_has_loop(pc: np.ndarray, instruction_lengths: np.ndarray | None = None) -> bool:
    """PC decrease excluding sequential increments modulo 2^16.

    Omitted lengths retain the frozen Z80 control's one-byte-wrap exception.
    The 6502 scorer always supplies executed instruction lengths from replay.
    """
    pc = np.asarray(pc, dtype=np.uint32) & 0xFFFF
    before, after = pc[..., :-1], pc[..., 1:]
    lengths = 1 if instruction_lengths is None else np.asarray(instruction_lengths, dtype=np.uint32)
    if instruction_lengths is not None and lengths.shape != before.shape:
        raise ValueError("one instruction length is required per PC transition")
    return bool(((after < before) & (after != ((before + lengths) & 0xFFFF))).any())


def pair_has_loop(ex, pairs: np.ndarray, L: int, steps: int, zh: bool) -> bool:
    if ex.isa != "m6502":
        return pc_has_loop(ex.pc_trace(pairs, L, steps, zero_halts=zh))
    table = json.loads((HERE / "algocell_exp/m6502_instructions.json").read_text())
    lengths = np.array([r["length"] for r in table["instructions"]], dtype=np.uint32)
    pc, opcodes = ex.pc_trace(pairs, L, steps, zero_halts=zh, return_opcodes=True)
    return pc_has_loop(pc, lengths[opcodes])


def trace_flags(tape, partners, ex, L, steps, zh):
    pairs = np.concatenate([np.repeat(tape[None], len(partners), 0), partners], axis=1)
    _, masks = X.execute_pairs_traced(pairs, L, steps, zero_halts=zh)
    entered = X.exec_positions(masks, 2 * L)[:, L:].any(axis=1)
    return {"entered": int(entered.sum()), "open": bool(entered.sum() >= 8),
            "confined": bool(not entered.any()), "loop": pair_has_loop(ex, pairs, L, steps, zh)}


def judge_tape(tape: np.ndarray, rng, ex, L: int, steps: int, zh: bool) -> dict:
    r = A.assay_many(tape[None], z80_steps=steps, n=64,
                     seed=int(rng.integers(1 << 31)), zero_halts=zh)[0]
    partners = rng.integers(0, 256, size=(16, L), dtype=np.uint8)
    pairs = np.concatenate([np.repeat(tape[None], 16, 0), partners], axis=1)
    _, masks = X.execute_pairs_traced(pairs, L, steps, zero_halts=zh)
    entered = X.exec_positions(masks, 2 * L)[:, L:].any(axis=1)
    heritable = bool(r["is_replicator"])
    # Replays consume no random draws. Use precisely the same 16 initial pairs.
    loop = pair_has_loop(ex, pairs, L, steps, zh) if heritable else None
    return {"heritable": heritable, "entered": int(entered.sum()), "open": bool(entered.sum() >= 8),
            "confined": bool(not entered.any()), "loop": loop, "copy_offset": r["copy_offset"],
            "gen2": r["gen2_score"]}


def endpoints(samples: list[dict], judge) -> tuple[dict, list[dict]]:
    """Same chronological, share-ordered top-3 walk and early stop as DZ.

    All tapes at the closure sample are judged (as in DZ); P2 uses strict <
    t_closed. Censored closure means all recorded samples count, including the
    horizon. A cached tape always retains its original assay/trace draws.
    """
    t_rep = t_closed = None
    first = None
    p2_candidates = []
    for sample in samples:
        for tape, share in sample["top3"]:
            if share < 0.005:
                continue
            r = judge(tape)
            if r["heritable"] and t_rep is None:
                t_rep, first = sample["step"], {"tape": tape, "share": share, **r}
            if r["heritable"] and r["confined"] and t_closed is None:
                t_closed = sample["step"]
            if r["heritable"] and r["open"] and not r["loop"]:
                p2_candidates.append({"step": sample["step"], "tape": tape, "share": share})
        if t_rep is not None and t_closed is not None:
            break
    violations = [r for r in p2_candidates if t_closed is None or r["step"] < t_closed]
    return {"t_rep": t_rep, "t_closed": t_closed, "first_tape": first["tape"] if first else None,
            "first_open": first["open"] if first else None,
            "first_confined": first["confined"] if first else None,
            "first_loop": first["loop"] if first else None,
            "first_entered": first["entered"] if first else None,
            "p2_violation": bool(violations),
            "p2_first_step": violations[0]["step"] if violations else None}, violations


def read_input(path: Path, input_format: str) -> tuple[dict, list[dict], dict]:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if not records or records[0].get("kind") != "condition":
        raise ValueError(f"{path}: missing condition")
    condition = records[0]
    samples = []
    for r in records[1:]:
        if "top3" not in r:
            continue
        if input_format == "dz":
            if len(r["top3"]) != len(r["shares"]):
                raise ValueError("top3/shares length mismatch")
            tapes = [(hx(np.frombuffer(bytes.fromhex(t), dtype=np.uint8)), float(s))
                     for t, s in zip(r["top3"], r["shares"])]
        else:
            tapes = [(hx(np.frombuffer(bytes.fromhex(t["tape"]), dtype=np.uint8)), float(t["share"]))
                     for t in r["top3"]]
            for t in r["top3"]:
                if hashlib.sha256(bytes.fromhex(t["tape"])).hexdigest() != t["sha256"]:
                    raise ValueError("top3 SHA-256 mismatch")
        if not 1 <= len(tapes) <= 3 or any(not math.isfinite(s) or not 0 <= s <= 1 for _, s in tapes):
            raise ValueError("invalid top-3 shares")
        samples.append({"step": int(r["step"]), "top3": sorted(tapes, key=lambda t: -t[1])})
    steps = [r["step"] for r in samples]
    if not samples or steps != sorted(set(steps)) or steps[-1] != condition["steps"]:
        raise ValueError("samples must increase strictly and finish at the horizon")
    if input_format == "dz":
        if condition.get("variant") != "lp0" or condition.get("p") != 0:
            raise ValueError("only the p=0 DZ positive control uses the standard machine")
        snapshot = {"step": steps[-1], "file": f"{path.stem}_t{steps[-1]}.npy"}
    else:
        summaries = [r for r in records if r.get("kind") == "summary"]
        finals = [r for r in records if r.get("kind") == "snapshot" and r["step"] == condition["steps"]]
        if len(summaries) != 1 or summaries[0]["steps_run"] != condition["steps"] or len(finals) != 1:
            raise ValueError("world is incomplete or missing its final snapshot")
        snapshot = finals[0]
    if Path(snapshot["file"]).name != snapshot["file"]:
        raise ValueError("snapshot filename must be local to the run directory")
    return condition, samples, snapshot


def load_snapshot(path: Path, snapshot: dict, L: int) -> np.ndarray:
    raw = brotli.decompress((path.parent / snapshot["file"]).read_bytes())
    if hashlib.sha256(raw).hexdigest() != snapshot["sha256_raw"]:
        raise ValueError("snapshot SHA-256 mismatch")
    if len(raw) != snapshot["raw_bytes"]:
        raise ValueError("snapshot raw size mismatch")
    soup = np.frombuffer(raw, dtype=np.uint8).reshape(snapshot["shape"])
    if soup.ndim != 2 or soup.shape[1] != L:
        raise ValueError("snapshot is not cells x L uint8")
    return soup


def score_world(path: Path, machine: str, input_format: str = "s1", *, device=None,
                sensitivities: bool = False) -> dict:
    condition, samples, snapshot = read_input(path, input_format)
    seed = int(condition["seed"])
    L = 16 if input_format == "dz" else int(condition["L"])
    steps = 128 if input_format == "dz" else int(condition["budget"])
    zh = False if input_format == "dz" else condition["brk"] == "halt"
    ex = E.ZilionExecutor(machine, device=device)
    m = ex.machine_for(L, zero_halts=zh, maps=True)
    if input_format == "dz":
        if ex.isa != "z80" or m.core["compat"] != "0.2":
            raise ValueError("DZ control needs the Z80 compat 0.2 machine")
    else:
        if machine != condition["machine_name"]:
            raise ValueError("machine differs from the world's machine_name")
        for key in ("sha256_core", "sha256_world", "sha256_wgsl"):
            if m.describe()[key] != condition["machine"][key]:
                raise ValueError(f"world's vendored machine differs: {key}")
    rng = np.random.default_rng([seed, 0, 11])  # DZ lp0 stream, unchanged
    cache = {}
    culture_rows = [] if sensitivities else None
    extra = {}

    def judge(t):
        if t not in cache:
            tape = np.frombuffer(bytes.fromhex(t), dtype=np.uint8)
            if len(tape) != L:
                raise ValueError("top3 tape length differs from the world")
            cache[t] = judge_tape(tape, rng, ex, L, steps, zh)
        return cache[t]

    with E.using(ex):
        row, violations = endpoints(samples, judge)
        snap_path = path.parent / snapshot["file"]
        if input_format == "dz":
            soup = np.load(snap_path, allow_pickle=False)
        else:
            soup = load_snapshot(path, snapshot, L)
        if soup.ndim != 2 or soup.shape[1] != L or soup.dtype != np.uint8:
            raise ValueError("snapshot is not cells x L uint8")
        classes, cells = classify(soup, L, [], zh, np.random.default_rng([seed, 23]),
                                  steps=steps, include_offsets=True, culture_rows=culture_rows)
        if sensitivities:
            if input_format != "s1":
                raise ValueError("sensitivities require a registered 6502 world")
            classes64, cells64 = classify(soup, L, [], zh, np.random.default_rng([seed, 23]),
                                          steps=steps, include_offsets=True, culture_partners=64,
                                          culture_rows=culture_rows)
            extra = {"literal64": {"classes": classes64, "cells": cells64},
                     "culture_cells": culture_rows, "soup_trace": None}
            if row["t_rep"] is not None:
                snapshots = [json.loads(line) for line in path.read_text().splitlines()
                             if json.loads(line).get("kind") == "snapshot"]
                near = min((s for s in snapshots if s["step"] >= row["t_rep"]), key=lambda s: s["step"])
                near_soup = soup if near["step"] == snapshot["step"] else load_snapshot(path, near, L)
                srng = np.random.default_rng([seed, 0, 6502, 16])
                indices = srng.choice(len(near_soup), size=16, replace=False)
                tape = np.frombuffer(bytes.fromhex(row["first_tape"]), dtype=np.uint8)
                extra["soup_trace"] = {"snapshot_step": near["step"], "snapshot_file": near["file"],
                                       "snapshot_sha256": near["sha256_raw"],
                                       "rng_seed": [seed, 0, 6502, 16], "partner_indices": indices.tolist(),
                                       "partners": [t.tobytes().hex() for t in near_soup[indices]],
                                       **trace_flags(tape, near_soup[indices], ex, L, steps, zh)}
    row.update({"world": path.stem, "arm": condition.get("arm", "DZ"), "seed": seed,
                "horizon": int(condition["steps"]), "L": L, "budget": steps,
                "machine": machine, "snapshot_step": snapshot["step"], **classes})
    code_files = ("score6502.py", "population_classes.py", "mutscan.py", "dial_soups.py",
                  "algocell_exp/executor.py", "algocell_exp/assay.py", "algocell_exp/exectrace.py",
                  "algocell_exp/zmachine.py", "algocell_exp/m6502_instructions.json")
    return json_safe({"schema": "s1-score/2", "world": row, "cells": cells, **extra,
                      "p2_tapes": violations, "judged_tapes": [{"tape": t, **r} for t, r in cache.items()],
                      "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "code_sha256": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in code_files},
                      "executor": ex.describe(), "usage": E.usage()})


def arm_summary(results: list[dict]) -> dict:
    rows = [r["world"] for r in results]
    rep = [r for r in rows if r["t_rep"] is not None]
    confined_loop = sum(bool(r["first_confined"] and r["first_loop"]) for r in rep)
    open_straight = sum(bool(r["first_open"] and not r["first_loop"]) for r in rep)
    cells = [c for r in results for c in r["cells"] if c["confined"]]
    eligible = [c for c in cells if c.get("copy_offset") is not None and c["copy_offset"] >= 6]
    equal = sum(c["n_sites"] == c["copy_offset"] - 6 for c in eligible)
    horizon = max(r["horizon"] for r in rows)
    km_rep = _km_median([r["t_rep"] for r in rows], horizon)
    km_closed = _km_median([r["t_closed"] for r in rows], horizon)
    classes = {c: sum(x["class"] == c for x in cells) for c in ("transmitter", "regenerator")}
    return json_safe({"arm": rows[0]["arm"], "worlds": len(rows), "with_rep": len(rep),
                      "with_closed": sum(r["t_closed"] is not None for r in rows),
                      "first_confined_loop": confined_loop, "first_open_straight": open_straight,
                      "p2_worlds": sum(r["p2_violation"] for r in rows), "km_t_rep": km_rep,
                      "km_t_closed": km_closed, **classes, "offset_eligible": len(eligible), "offset_equal": equal})


def verdict(results: list[dict]) -> dict:
    """Only a complete primary arm receives registered tests; others descriptive."""
    summary = arm_summary(results)
    rows = [r["world"] for r in results]
    if summary["arm"] != "A":
        return {**summary, "status": "descriptive"}
    complete = (len(rows) == 20 and {r["seed"] for r in rows} == set(range(65001, 65021))
                and all(r["horizon"] == 10_000_000 and r["snapshot_step"] == 10_000_000
                        and r["L"] == 16 and r["budget"] == 128
                        and r["machine"] == "m6502_pair_maps_L16" for r in rows))
    if not complete:
        return {**summary, "status": "incomplete primary arm; registered tests not evaluated"}
    n = summary["with_rep"]
    null = n < 5
    class_ok = summary["transmitter"] > 0 and summary["regenerator"] > 0
    offset_ok = (summary["offset_equal"] / summary["offset_eligible"] >= 0.8) if summary["offset_eligible"] else None
    p3 = False if not class_ok else offset_ok
    return {**summary, "status": "complete", "null": null,
            "P1": None if null else summary["first_confined_loop"] / n >= 0.8,
            "P2": None if null else summary["p2_worlds"] == 0,
            "P3": None if null else p3,
            "P3_classes": None if null else class_ok, "P3_offset": None if null else offset_ok,
            "P4": (summary["km_t_rep"] is not None and summary["km_t_rep"] > 450) or (20 - n) > 10,
            "kill": bool(n and summary["first_open_straight"] / n >= 0.25)}


def report(results: list[dict], out: Path, *, report_name: str = "S1.md"):
    if not results:
        raise ValueError("no scored worlds")
    keys = [(r["world"]["arm"], r["world"]["seed"]) for r in results]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate arm/seed results")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r["world"] for r in results]).to_csv(out / "s1_worlds.csv", index=False)
    pd.DataFrame([{**{k: r["world"][k] for k in ("world", "arm", "seed")}, **c}
                  for r in results for c in r["cells"]], columns=("world", "arm", "seed", "cell", "tape",
                  "confined", "n_sites", "n_sites_unexecuted", "n_executed", "class", "copy_offset",
                  "sites_equals_d_minus6")).to_csv(out / "s1_cells.csv", index=False)
    pd.DataFrame([{**{k: r["world"][k] for k in ("world", "arm", "seed")}, **c}
                  for r in results for c in r["p2_tapes"]], columns=("world", "arm", "seed", "step", "tape",
                  "share")).to_csv(out / "s1_p2.csv", index=False)
    arms = sorted(set(k[0] for k in keys))
    verdicts = [verdict([r for r in results if r["world"]["arm"] == arm]) for arm in arms]
    write_json(out / "s1_verdicts.json", {"arms": verdicts})
    lines = ["# S1 — generated by score6502.py", "",
             "Uniform byte partners and counts follow the frozen DZ scorer: top tapes culture-tested over 64 partners; "
             "16 traced encounters; final 64 cells culture-tested over 32 shared partners; up to 16 heritable cells "
             "scanned (16 alternatives per site, 16 partners). Classes use the frozen population estimator. "
             "Offset equality uses the scanned heritable confined cells with measured d ≥ 6.", "",
             "Loop calls use identical-pair PC prefix replays through the world's executor. "
             "Closure is strict for P2: tapes at t_closed do not count. Null P1–P3 means not tested. "
             "KM medians follow dial_soups; 'censored' means the median was not reached.", "",
             "| arm | worlds | t_rep events | first confined + loop | first open + straight | P2 worlds | "
             "t_closed events | KM t_rep | KM t_closed | transmitters / regenerators scanned | sites = d − 6 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in verdicts:
        kr = v["km_t_rep"] if v["km_t_rep"] is not None else "censored"
        kc = v["km_t_closed"] if v["km_t_closed"] is not None else "censored"
        lines.append(f"| {v['arm']} | {v['worlds']} | {v['with_rep']} | {v['first_confined_loop']} | "
                     f"{v['first_open_straight']} | {v['p2_worlds']} | {v['with_closed']} | {kr} | {kc} | "
                     f"{v['transmitter']} / {v['regenerator']} | {v['offset_equal']} / {v['offset_eligible']} |")
    for v in verdicts:
        if v["arm"] == "A":
            lines += ["", f"Arm A: {v['status']}."]
            if v["status"] == "complete":
                def label(x):
                    return "not tested" if x is None else ("supported" if x else "not supported")
                lines += ["", f"- Null rule: {'uninformative about form (fewer than 5 t_rep events)' if v['null'] else 'does not apply'}.",
                          *[f"- {p}: {label(v[p])}." for p in ("P1", "P2", "P3", "P4")],
                          f"- P3 class occurrence: {label(v['P3_classes'])}; offset equality: {label(v['P3_offset'])}.",
                          f"- Kill rule: {'triggered; beginning claim withdrawn for 6502' if v['kill'] else 'not triggered'}."]
    lines.insert(2, "6502 loop calls exclude every sequential PC increment modulo 2^16, using the executed "
                 "opcode from each prefix's replayed memory and the Zilion descriptor length. "
                 "Z80 control loop calls retain the frozen rule.")
    (out / report_name).write_text("\n".join(lines) + "\n")


def control_report(results: list[dict], expected: Path, out: Path) -> bool:
    target = pd.read_csv(expected, float_precision="round_trip")
    target = target[target.variant == "lp0"].set_index("seed")
    got = {r["world"]["seed"]: r["world"] for r in results}
    if set(got) != set(range(9001, 9011)) or set(target.index) != set(got):
        raise ValueError("positive control requires exactly lp0 seeds 9001–9010")
    lines = ["# S1 scorer positive control (generated)", "",
             "Archived DZ lp0, seeds 9001–9010, final snapshot 300,000; vendored Z80 compat '0.2'. "
             "Exact comparisons, including round-trip parsed floats (NaN equals missing).", "",
             "| seed | " + " | ".join(CONTROL_COLUMNS) + " |",
             "|---|" + "---|" * len(CONTROL_COLUMNS)]
    differences = []
    totals = {k: 0 for k in CONTROL_COLUMNS}
    for seed in sorted(got):
        cells = []
        for k in CONTROL_COLUMNS:
            a, b = got[seed].get(k), target.loc[seed, k]
            same = (a is None and pd.isna(b)) or a == b
            totals[k] += int(same)
            cells.append("match" if same else "DIFF")
            if not same:
                differences.append(f"- seed {seed}, {k}: scorer {a!r}; registered {b!r}.")
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    lines += ["", "| column | exact matches |", "|---|---:|",
              *[f"| {k} | {totals[k]}/10 |" for k in CONTROL_COLUMNS], "",
              "PASS: every compared endpoint and class column matches exactly." if not differences
              else "FAIL: differences require a cause investigation before approval.", *differences,
              "", "The random streams are reproducible: judge RNG = [seed, 0, 11], final RNG = [seed, 23]; "
              "PC replays draw no randomness. At p = 0 the dial is the ordinary executor. "
              "All executions used the vendored machine; no shader fallback.", "",
              "No registered 6502 world was read, listed, fetched or scored during validation."]
    out.mkdir(parents=True, exist_ok=True)
    (out / "SCORER_CONTROL.md").write_text("\n".join(lines) + "\n")
    return not differences


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--machine")
    ap.add_argument("--format", choices=("s1", "dz"), default="s1")
    ap.add_argument("--worlds", help="comma-separated stems; default every jsonl in --runs")
    ap.add_argument("--gather", type=Path, help="directory of already scored result JSONs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--control", type=Path, help="registered DZ CSV to compare (all ten lp0 seeds required)")
    a = ap.parse_args()
    if a.gather:
        results = [json.loads(p.read_text()) for p in sorted(a.gather.glob("*.json"))]
    else:
        if not a.runs or not a.machine:
            ap.error("--runs and --machine are required without --gather")
        paths = ([a.runs / f"{w}.jsonl" for w in a.worlds.split(",")] if a.worlds
                 else sorted(a.runs.glob("*.jsonl")))
        results = []
        for path in paths:
            result = score_world(path, a.machine, a.format)
            write_json(a.out / "worlds" / f"{path.stem}.json", result)
            results.append(result)
            print(json.dumps(result["world"]), flush=True)
    report(results, a.out)
    if a.control and not control_report(results, a.control, a.out):
        raise SystemExit("positive control differs; see SCORER_CONTROL.md")


if __name__ == "__main__":
    main()
