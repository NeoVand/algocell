#!/usr/bin/env python3
"""EXPLORATORY archive mining only: no simulation, executor, assay, GPU, or network.

Run with the existing experiments Python. All writes stay beside this script.
CSV rows and the generated report are explicitly EXPLORATORY.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import mean, median

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "experiments/results"
OUT = Path(__file__).resolve().parent
STATUS = "EXPLORATORY"
INPUTS = {}


def track(path):
    path = Path(path).resolve()
    if path not in INPUTS:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for b in iter(lambda: f.read(1024 * 1024), b""):
                h.update(b)
        INPUTS[path] = {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}
    return path


def read_csv(relative):
    with track(RESULTS / relative).open(newline="") as f:
        return list(csv.DictReader(f))


def number(value):
    if value in (None, "", "nan", "None"):
        return None
    n = float(value)
    return n if math.isfinite(n) else None


def event(value):
    n = number(value)
    return n if n is not None and n >= 0 else None


def integer(value):
    return int(float(value))


def truth(value):
    return value is True or str(value).lower() == "true"


def km(records):
    """Grouped KM median; records = (event time or None, censoring time).

    Exact fractions avoid rounding around S(t)=1/2. Events at a censoring time
    precede censoring at that time. NR is distinct from missing/unscored.
    """
    s = Fraction(1)
    for t in sorted({e for e, c in records if e is not None}):
        risk = sum((e if e is not None else c) >= t for e, c in records)
        deaths = sum(e == t for e, c in records)
        s *= Fraction(risk - deaths, risk)
        if s <= Fraction(1, 2):
            return t
    return None


def med(values):
    values = [x for x in values if x is not None]
    return median(values) if values else None


def fmt(x):
    if x is None or x == "":
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        if x.is_integer():
            return f"{x:,.0f}"
        return f"{x:,.1f}" if abs(x) >= 1000 else f"{x:,.4g}"
    if isinstance(x, int):
        return f"{x:,}"
    return str(x).replace("|", ";")


def kmfmt(x):
    return "NR" if x is None else fmt(x)


def write_csv(name, rows):
    assert rows, name
    rows = [{"status": STATUS, **r} for r in rows]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def table(rows, columns):
    return "\n".join([
        "| " + " | ".join(title for key, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *["| " + " | ".join(fmt(r.get(key)) for key, title in columns) + " |" for r in rows],
    ])


def tally(values):
    return "; ".join(f"{k}:{v}" for k, v in sorted(Counter(values).items()))


def mine_atlas(archive):
    rows = []
    scored_closure = {(r["stage"], integer(r["L"]), integer(r["seed"])): r for r in read_csv("review_r2/first_closure.csv")}
    inflow = {(integer(r["L"]), integer(r["seed"])): r for r in read_csv("inflow_insoup/per_world.csv")}
    population_closed = defaultdict(list)
    for r in read_csv("population/classes_snapshots.csv"):
        if number(r["frac_confined_of_heritable"]) is not None and float(r["frac_confined_of_heritable"]) > 0:
            population_closed[(r["stage"], integer(r["L"]), integer(r["seed"]))].append(number(r["step"]))
    blocked = {"ld-imm", "all-ld", "no-copy", "rmw-only", "push-only", "push", "stack-write-only", "stack-writes"}
    for stage in "ABCDEFGHI":
        source = f"stage{stage}/runs.csv"
        runs = read_csv(source)
        succession = {r["file"]: r for r in read_csv(f"stage{stage}/succession.csv")}
        assert len({r["file"] for r in runs}) == len(runs)
        for r in runs:
            p = track(archive / f"stage{stage}" / r["file"])
            d = json.loads(p.read_text())
            s = succession[r["file"]]
            assert integer(r["steps_run"]) == d["steps_run"]
            assert integer(r["seed"]) == d["seed"]
            rep = event(r["t_rep"])
            pusher = rep is not None and r["trep_mechs"] == "stack" and number(r["trep_period"]) == 2
            successor = min((t for t in [event(s["t_ex_sp_30"]), event(s["t_ldir_30"])] if t is not None), default=None)
            key = (stage, integer(r["tape"]), integer(r["seed"]))
            closed = scored_closure.get(key)
            open_scored = float(inflow[key[1:]]["entered_frac_soup"]) == 1 if stage == "G" else None
            rows.append({
                "stage": stage, "label": r["label"], "ablation": r["label"].split("@")[0],
                "L": integer(r["tape"]), "budget": integer(r["steps"]), "k": integer(r["k"]),
                "seed": integer(r["seed"]), "summary": str(p), "source_table": source,
                "suppress": "; ".join(d["suppress"]), "literal_loadpush_removed": r["label"].split("@")[0] in blocked,
                "mem_length": d.get("mem_length", 2 * d["tape_length"]),
                "horizon": integer(r["horizon"]), "steps_run": integer(r["steps_run"]),
                "stopped_early": truth(r["stopped_early"]), "sample_every": d["sample_every"],
                "early_sample_every": d.get("sample_every_early"), "tq_10": event(r["tq_10"]),
                "t_rep": rep, "first_mechanisms": r["trep_mechs"], "first_period": number(r["trep_period"]),
                "first_tape": r["trep_tape"], "first_pusher_mechanism_proxy": pusher,
                "first_replicator_open_scored": open_scored,
                "pusher_status": "mechanism proxy plus archived soup-entry score" if stage == "G" else "mechanism proxy only; no archived trace in this atlas row",
                "t_push_30": event(s["t_push_30"]), "t_ex_sp_30": event(s["t_ex_sp_30"]),
                "t_ldir_30": event(s["t_ldir_30"]), "t_successor_family_30_proxy": successor,
                "closure_status": "SCORED top-3 endpoint; blank time means censored" if closed else "UNSCORED top-3 endpoint; census patterns do not establish closure",
                "t_closed": event(closed["t_closed"]) if closed else None,
                "first_closed_random_population_observed": min(population_closed[key], default=None),
                "final_family": s["final_family"], "final_tape": r["final_tape"],
            })
    write_csv("atlas_worlds.csv", rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["stage"], r["label"], r["L"], r["budget"], r["k"], r["mem_length"])].append(r)
    summaries = []
    for key, g in sorted(groups.items()):
        summaries.append({
            **dict(zip(["stage", "label", "L", "budget", "k", "mem_length"], key)),
            "n": len(g), "literal_loadpush_removed": g[0]["literal_loadpush_removed"], "suppress": g[0]["suppress"],
            "first_pusher_proxy_n": sum(r["first_pusher_mechanism_proxy"] for r in g),
            "first_mechanisms": tally(r["first_mechanisms"] or "no event" for r in g),
            "t_rep_n": sum(r["t_rep"] is not None for r in g),
            "t_rep_km": km([(r["t_rep"], r["steps_run"]) for r in g]),
            "tq_10_n": sum(r["tq_10"] is not None for r in g),
            "tq_10_km": km([(r["tq_10"], r["steps_run"]) for r in g]),
            "successor_proxy_n": sum(r["t_successor_family_30_proxy"] is not None for r in g),
            "successor_proxy_km": km([(r["t_successor_family_30_proxy"], r["steps_run"]) for r in g]),
            "ldir_proxy_n": sum(r["t_ldir_30"] is not None for r in g),
            "ldir_proxy_km": km([(r["t_ldir_30"], r["steps_run"]) for r in g]),
            "steps_run_min": min(r["steps_run"] for r in g), "steps_run_max": max(r["steps_run"] for r in g),
            "stopped_early_n": sum(r["stopped_early"] for r in g), "closure_status": "UNSCORED in atlas; use closure_worlds for scored cohorts",
            "closure_scored_n": sum(r["closure_status"].startswith("SCORED") for r in g),
            "t_closed_n": sum(r["t_closed"] is not None for r in g),
            "t_closed_km": km([(r["t_closed"], r["steps_run"]) for r in g]) if all(r["closure_status"].startswith("SCORED") for r in g) else None,
            "closed_random_population_seen_n": sum(r["first_closed_random_population_observed"] is not None for r in g),
        })
        summaries[-1]["closure_status"] = "SCORED top-3 endpoint" if summaries[-1]["closure_scored_n"] == len(g) else "UNSCORED top-3 endpoint"
    write_csv("atlas_groups.csv", summaries)
    return rows, summaries


def mine_closure():
    rows = read_csv("review_r2/first_closure.csv")
    out = []
    for r in rows:
        out.append({"stage": r["stage"], "L": integer(r["L"]), "seed": integer(r["seed"]),
                    "horizon": integer(r["horizon"]), "t_rep": event(r["t_rep"]), "t_closed": event(r["t_closed"]),
                    "closed_tape": r["closed_tape"], "endpoint": "top-3 >=0.5%, gen2 >=0.3, confined in 16/16 traces"})
    write_csv("closure_worlds.csv", out)
    groups = defaultdict(list)
    for r in out:
        groups[(r["stage"], r["L"])].append(r)
    summary = []
    for (st, L), g in sorted(groups.items()):
        summary.append({"stage": st, "L": L, "n": len(g), "closed_n": sum(r["t_closed"] is not None for r in g),
                        "t_rep_km": km([(r["t_rep"], r["horizon"]) for r in g]),
                        "t_closed_km": km([(r["t_closed"], r["horizon"]) for r in g])})
    write_csv("closure_groups.csv", summary)
    return out, summary


def mine_population():
    snaps = read_csv("population/classes_snapshots.csv")
    cells = read_csv("population/classes_cells.csv")
    # These scans, not an unexecuted-byte heuristic, define the classes.
    for r in cells:
        expected = "open" if not truth(r["confined"]) else "regenerator" if integer(r["n_sites"]) <= 2 else "transmitter" if integer(r["n_sites"]) >= 5 else "intermediate"
        assert r["class"] == expected
    inflow = {(integer(r["L"]), integer(r["seed"])): r for r in read_csv("inflow_insoup/per_world.csv")}
    first = {(integer(r["L"]), integer(r["seed"])): r for r in read_csv("stageG/stageG/stage_g_runs.csv")}
    groups = defaultdict(list)
    for r in snaps:
        groups[(r["stage"], integer(r["L"]), integer(r["seed"]))].append(r)
    worlds = []
    for (st, L, seed), g in sorted(groups.items()):
        for cutoff, name in [(0, "any"), (0.1, "share_gt_10pct"), (0.5, "majority")]:
            out = {"stage": st, "L": L, "seed": seed, "threshold": name, "threshold_strict_gt": cutoff,
                   "snapshots": len(g), "last_sample_step": max(number(r["step"]) for r in g),
                   "t_rep": event(first[(L, seed)]["t_rep"]) if st == "G" else None,
                   "first_replicator_open_scored": truth(float(inflow[(L, seed)]["entered_frac_soup"]) == 1) if st == "G" else None}
            for c in ["open", "regenerator", "intermediate", "transmitter"]:
                positive = [r for r in g if number(r[f"frac_{c}_of_heritable"]) is not None and float(r[f"frac_{c}_of_heritable"]) > cutoff]
                t = min((number(r["step"]) for r in positive), default=None)
                out[f"first_{c}_observed"] = t
                # A prior random sample's non-detection is NOT a bound on true birth time.
                out[f"previous_{c}_nondetection_sample"] = max((number(r["step"]) for r in g if t is not None and number(r["step"]) < t and number(r[f"frac_{c}_of_heritable"]) is not None), default=None)
            R, T = out["first_regenerator_observed"], out["first_transmitter_observed"]
            out["R_T_order"] = "neither" if R is None and T is None else "R only" if T is None else "T only" if R is None else "R before T" if R < T else "T before R" if T < R else "same sample"
            out["T_minus_R_observed"] = T - R if R is not None and T is not None else None
            O = out["first_open_observed"]
            C = min((out[f"first_{c}_observed"] for c in ["regenerator", "intermediate", "transmitter"] if out[f"first_{c}_observed"] is not None), default=None)
            out["first_closed_class_observed"] = C
            out["open_closed_class_order"] = "neither" if O is None and C is None else "open only" if C is None else "closed only" if O is None else "open earlier" if O < C else "closed earlier" if C < O else "same sample"
            finals = [r for r in g if r["snapshot"] == "final"]
            assert len(finals) == 1
            for c in ["open", "regenerator", "transmitter"]:
                out[f"final_frac_{c}_of_heritable"] = number(finals[0][f"frac_{c}_of_heritable"])
            worlds.append(out)
    write_csv("population_worlds.csv", worlds)
    summary = []
    grouped = defaultdict(list)
    for r in worlds:
        grouped[(r["stage"], r["L"], r["threshold"])].append(r)
    for (st, L, threshold), g in sorted(grouped.items()):
        row = {"stage": st, "L": L, "threshold": threshold, "n": len(g)}
        count = Counter(r["R_T_order"] for r in g)
        for label in ["R before T", "same sample", "T before R", "R only", "T only", "neither"]:
            row[label] = count[label]
        both = count["R before T"] + count["same sample"] + count["T before R"]
        row["both_seen_n"] = both
        row["R_before_fraction_of_both"] = count["R before T"] / both if both else None
        oc = Counter(r["open_closed_class_order"] for r in g)
        for label in ["open earlier", "same sample", "closed earlier", "open only", "closed only", "neither"]:
            row[f"open_closed_{label}"] = oc[label]
        for c in ["open", "regenerator", "transmitter"]:
            row[f"{c}_seen_n"] = sum(r[f"first_{c}_observed"] is not None for r in g)
            row[f"median_first_{c}_observed_conditional"] = med([r[f"first_{c}_observed"] for r in g])
        summary.append(row)
    write_csv("population_groups.csv", summary)
    return worlds, summary


def mine_lod(archive):
    lines, worlds = [], []
    for tag, folder in [("N3-v6", "lod"), ("N6", "lod_n6")]:
        input_dir = archive / ("lod_v6_modal/lod_v6" if folder == "lod" else "lod_n6_modal/lod_n6")
        runs = sorted(input_dir.glob("*/lod.json"))
        assert len(runs) == 20
        by_seed = {}
        for p in runs:
            # Only the small header is decoded; no replay or tape classification.
            track(p)
            with p.open() as f:
                header = f.read(8192)
            start = header.index('"result"')
            start = header.index(":", start) + 1
            result, _ = json.JSONDecoder().raw_decode(header[start:].lstrip())
            assert result["V2_mismatches"] == 0
            by_seed[result["seed"]] = result
        grouped = defaultdict(list)
        for r in read_csv(f"{folder}/open_first.csv"):
            seed = integer(r["seed"])
            grouped[seed].append(r)
            lines.append({"cohort": tag, "seed": seed, "line": integer(r["line"]),
                          "first_copier_class": r["first_copier_class"], "first_open_step": event(r["first_open_step"]),
                          "first_confined_step": event(r["first_conf_step"]), "founder_step": event(r["F_step"]),
                          "first_confined_is_founder": truth(r["first_conf_is_founder"]),
                          "regenerator_transmitter_status": "NOT SCORED: no archived site scan at these lineage events"})
        for seed, result in sorted(by_seed.items()):
            g = grouped.get(seed, [])
            if g:
                assert len(g) == 64
                assert all(integer(r["t_close"]) == result["t_close"] for r in g)
            worlds.append({"cohort": tag, "seed": seed, "t_close_lod": result["t_close"], "t_end": result["t_end"],
                           "sampled_lines": len(g), "first_copier_open_on_all_lines": all(r["first_copier_class"] == "open copier" for r in g) if g else None,
                           "earliest_open_on_sampled_lines": min((event(r["first_open_step"]) for r in g), default=None),
                           "earliest_confined_on_sampled_lines": min((event(r["first_conf_step"]) for r in g), default=None),
                           "V2_mismatches": result["V2_mismatches"], "R_T_status": "NOT SCORED"})
    write_csv("lod_lines.csv", lines)
    write_csv("lod_worlds.csv", worlds)
    summary = []
    for cohort in ["N3-v6", "N6"]:
        g = [r for r in worlds if r["cohort"] == cohort]
        ls = [r for r in lines if r["cohort"] == cohort]
        summary.append({"cohort": cohort, "worlds": len(g), "closed_worlds": sum(r["t_close_lod"] is not None for r in g),
                        "sampled_lines": len(ls), "open_first_lines": sum(r["first_copier_class"] == "open copier" for r in ls),
                        "all_lines_open_first_worlds": sum(r["first_copier_open_on_all_lines"] is True for r in g),
                        "median_earliest_open_per_closed_world": med([r["earliest_open_on_sampled_lines"] for r in g]),
                        "median_earliest_confined_per_closed_world": med([r["earliest_confined_on_sampled_lines"] for r in g]),
                        "median_t_close_conditional": med([r["t_close_lod"] for r in g]),
                        "earliest_confined_is_founder_lines": sum(r["first_confined_is_founder"] for r in ls)})
    write_csv("lod_groups.csv", summary)
    return worlds, summary


def mine_n2(archive):
    data = read_csv("n2/demography.csv")
    summaries, snapshots = [], []
    for comp in ["benign", "lethal"]:
        for rule in ["benign", "lethal"]:
            R = [r for r in data if r["comp"] == comp and r["rule"] == rule and r["class"] == "R"]
            T = [r for r in data if r["comp"] == comp and r["rule"] == rule and r["class"] == "T"]
            assert len(R) == len(T) == 10
            both = 0
            for seed in range(1, 11):
                p = track(archive / "n2" / f"{comp}_soup_{rule}_rule_s{seed}.npz")
                with np.load(p) as z:
                    counts = z["counts"]
                    assert integer(z["steps"]) == 2000 and counts.shape == (9, 4)
                    both += int(counts[0, 2] > 0 and counts[0, 3] > 0)
                    for step, count in zip([1, 251, 501, 751, 1001, 1251, 1501, 1751, 2000], counts):
                        snapshots.append({"composition": comp, "rule": rule, "seed": seed, "continuation_step": step,
                                          **dict(zip(["O", "M", "R", "T"], map(int, count))),
                                          "origin_status": "pre-existing mix50 equilibrium continuation; no de novo origin times"})
            summaries.append({"composition": comp, "rule": rule, "worlds": len(R), "both_at_first_saved_count": both,
                              "mean_T_share_start": mean(float(r["T_share_start"]) for r in R),
                              "mean_T_share_end": mean(float(r["T_share_end"]) for r in R),
                              "mean_g_R": mean(float(r["net"]) for r in R), "mean_g_T": mean(float(r["net"]) for r in T),
                              "origin_status": "NOT ADDRESSABLE: both classes pre-exist"})
    write_csv("n2_counts.csv", snapshots)
    write_csv("n2_groups.csv", summaries)
    return summaries


def mine_6502():
    worlds = read_csv("s6502/v2/s1_worlds.csv")
    cells = read_csv("s6502/v2/s1_cells.csv")
    read_csv("s6502/v2/partner_summary.csv")
    track(RESULTS / "s6502/CORRECTION.md")
    for r in cells:
        expected = "open" if not truth(r["confined"]) else "regenerator" if integer(r["n_sites"]) <= 2 else "transmitter" if integer(r["n_sites"]) >= 5 else "intermediate"
        assert r["class"] == expected
    rows, summary = [], []
    for r in worlds:
        c = [x for x in cells if x["world"] == r["world"]]
        rows.append({"world": r["world"], "arm": r["arm"], "L": integer(r["L"]), "budget": integer(r["budget"]),
                     "seed": integer(r["seed"]), "horizon": integer(r["horizon"]), "t_rep": event(r["t_rep"]), "t_closed": event(r["t_closed"]),
                     "first_open": truth(r["first_open"]) if r["first_open"] else None,
                     "first_confined": truth(r["first_confined"]) if r["first_confined"] else None,
                     "first_loop": truth(r["first_loop"]) if r["first_loop"] else None,
                     "first_tape": r["first_tape"], "snapshot_step": integer(r["snapshot_step"]),
                     "final_scanned_open": sum(x["class"] == "open" for x in c),
                     "final_scanned_R": sum(x["class"] == "regenerator" for x in c),
                     "final_scanned_M": sum(x["class"] == "intermediate" for x in c),
                     "final_scanned_T": sum(x["class"] == "transmitter" for x in c),
                     "R_T_order": "UNKNOWN: site scans only at final 10M snapshot"})
    write_csv("s6502_worlds.csv", rows)
    for arm in "ABCD":
        g = [r for r in rows if r["arm"] == arm]
        summary.append({"arm": arm, "L": g[0]["L"], "budget": g[0]["budget"], "worlds": len(g),
                        "rep_n": sum(r["t_rep"] is not None for r in g), "closed_n": sum(r["t_closed"] is not None for r in g),
                        "first_open_n": sum(r["first_open"] is True for r in g), "first_confined_n": sum(r["first_confined"] is True for r in g),
                        "first_confined_loop_n": sum(r["first_confined"] is True and r["first_loop"] is True for r in g),
                        "t_rep_km": km([(r["t_rep"], r["horizon"]) for r in g]),
                        "t_closed_km": km([(r["t_closed"], r["horizon"]) for r in g]),
                        **{f"final_scanned_{c}": sum(r[f"final_scanned_{c}"] for r in g) for c in ["open", "R", "M", "T"]},
                        **{f"final_{c}_worlds": sum(r[f"final_scanned_{c}"] > 0 for r in g) for c in ["R", "T"]},
                        "final_both_R_T_worlds": sum(r["final_scanned_R"] > 0 and r["final_scanned_T"] > 0 for r in g),
                        "R_T_order": "UNOBSERVED"})
    write_csv("s6502_groups.csv", summary)
    return rows, summary


def mine_effects(atlas, closure, s6502):
    effects = []
    control = next(r for r in atlas if r["stage"] == "C" and r["label"] == "none" and r["L"] == 16 and r["budget"] == 128 and r["k"] == 4)
    for r in atlas:
        if r["stage"] == "C" and r["L"] == 16 and r["budget"] == 128 and r["k"] == 4:
            for endpoint, time, count in [("t_rep", "t_rep_km", "t_rep_n"), ("successor family proxy", "successor_proxy_km", "successor_proxy_n")]:
                effects.append({"comparison": f"C {r['label']} / C none", "endpoint": endpoint, "arm_n": r["n"], "control_n": control["n"],
                                "arm_events": r[count], "control_events": control[count], "arm_km": r[time], "control_km": control[time],
                                "km_ratio_arm_over_control": r[time] / control[time] if r[time] is not None and control[time] else None,
                                "event_risk_difference": r[count] / r["n"] - control[count] / control["n"],
                                "caveat": "closure unscored; instruction and horizon confounds; proxy is not t_closed"})
    G = next(r for r in closure if r["stage"] == "G" and r["L"] == 16)
    I = next(r for r in closure if r["stage"] == "I" and r["L"] == 16)
    A = next(r for r in s6502 if r["arm"] == "A")
    for label, arm, nkey, countkey, caveat in [
        ("Z80 I16 / G16", I, "n", "closed_n", "lethality and seeds differ; first-replicator mechanism differs"),
        ("6502 A / Z80 G16", A, "worlds", "closed_n", "ISA, loops, addressing, horizons and sample cadence differ"),
    ]:
        effects.append({"comparison": label, "endpoint": "t_closed", "arm_n": arm[nkey], "control_n": G["n"],
                        "arm_events": arm[countkey], "control_events": G["closed_n"], "arm_km": arm["t_closed_km"], "control_km": G["t_closed_km"],
                        "km_ratio_arm_over_control": arm["t_closed_km"] / G["t_closed_km"],
                        "event_risk_difference": arm[countkey] / arm[nkey] - G["closed_n"] / G["n"], "caveat": caveat})
    # Existing dial is the closest same-ISA first-open stratification, still observational.
    dial = read_csv("dial/dial_worlds.csv")
    strata = []
    for p in sorted({float(r["p"]) for r in dial}):
        for isopen in [True, False]:
            g = [r for r in dial if float(r["p"]) == p and truth(r["first_open"]) == isopen]
            if not g:
                continue
            strata.append({"p": p, "first_open": isopen, "n": len(g), "closed_n": sum(event(r["t_closed"]) is not None for r in g),
                           "t_closed_km": km([(event(r["t_closed"]), float(r["horizon"])) for r in g]),
                           "caveat": "post-outcome stratification; same p, tiny unbalanced groups; no intervention on openness"})
        gs = [r for r in strata if r["p"] == p]
        if len(gs) == 2:
            a, b = gs
            effects.append({"comparison": f"dial p={p:g}: first-not-open / first-open", "endpoint": "t_closed",
                            "arm_n": b["n"], "control_n": a["n"], "arm_events": b["closed_n"], "control_events": a["closed_n"],
                            "arm_km": b["t_closed_km"], "control_km": a["t_closed_km"],
                            "km_ratio_arm_over_control": b["t_closed_km"] / a["t_closed_km"] if b["t_closed_km"] is not None and a["t_closed_km"] else None,
                            "event_risk_difference": b["closed_n"] / b["n"] - a["closed_n"] / a["n"], "caveat": b["caveat"]})
    write_csv("effect_sizes.csv", effects)
    write_csv("dial_strata.csv", strata)
    return effects, strata


def report(atlas_worlds, atlas, closure, popworlds, pop, lod, n2, s6502worlds, s6502, effects, dial):
    main = [r for r in atlas if r["stage"] == "C" and r["L"] == 16 and r["budget"] == 128 and r["k"] == 4]
    ctrl = next(r for r in main if r["label"] == "none")
    for r in main:
        r["rep_count"] = f"{r['t_rep_n']}/{r['n']}"
        r["pusher_count"] = f"{r['first_pusher_proxy_n']}/{r['n']}"
        r["rep_km_display"] = kmfmt(r["t_rep_km"])
        r["rep_ratio"] = r["t_rep_km"] / ctrl["t_rep_km"] if r["t_rep_km"] is not None else "NR"
        r["q_display"] = f"{r['tq_10_n']}/{r['n']}; {kmfmt(r['tq_10_km'])}"
        r["successor_display"] = f"{r['successor_proxy_n']}/{r['n']}; {kmfmt(r['successor_proxy_km'])}"
        r["proxy_ratio"] = r["successor_proxy_km"] / ctrl["successor_proxy_km"] if r["successor_proxy_km"] is not None else "NR"
    main.sort(key=lambda r: (r["label"] != "none", r["label"]))
    coverage = []
    for stage in "ABCDEFGHI":
        g = [r for r in atlas_worlds if r["stage"] == stage]
        coverage.append({"stage": stage, "worlds": len(g), "arms": ", ".join(sorted({r["label"] for r in g})),
                         "early": sum(r["stopped_early"] for r in g)})
    popany = [r for r in pop if r["threshold"] == "any"]
    G = [r for r in popany if r["stage"] == "G"]
    orders = {c: sum(r[c] for r in G) for c in ["R before T", "same sample", "T before R", "R only", "T only", "neither"]}
    both = sum(r["both_seen_n"] for r in G)
    major = [r for r in pop if r["stage"] == "G" and r["threshold"] == "majority"]
    majority_before = sum(r["R before T"] for r in major)
    majority_after = sum(r["T before R"] for r in major)
    majority_both = sum(r["both_seen_n"] for r in major)
    open_closed = {c: sum(r[f"open_closed_{c}"] for r in G) for c in ["open earlier", "same sample", "closed earlier", "open only"]}
    stage_i = next(r for r in popany if r["stage"] == "I")
    ratios = {r["label"]: r["rep_ratio"] for r in main}
    risks = {r["comparison"]: r["event_risk_difference"] for r in effects if r["endpoint"] == "t_rep"}
    conditional = []
    for r in G:
        conditional.append({"L": r["L"], "open": r["median_first_open_observed_conditional"],
                            "R": r["median_first_regenerator_observed_conditional"], "T": r["median_first_transmitter_observed_conditional"],
                            "R_n": r["regenerator_seen_n"], "T_n": r["transmitter_seen_n"]})
    d9 = [r for r in atlas if r["stage"] == "D" and r["L"] == 9 and r["budget"] == 128 and r["k"] == 4]
    for r in d9:
        r["rep_display"] = f"{r['t_rep_n']}/{r['n']}; {kmfmt(r['t_rep_km'])}"
    for r in closure:
        r["closure_display"] = f"{r['closed_n']}/{r['n']}; {kmfmt(r['t_closed_km'])}"
    for r in s6502:
        r["rep_display"] = f"{r['rep_n']}/{r['worlds']}; {kmfmt(r['t_rep_km'])}"
        r["closed_display"] = f"{r['closed_n']}/{r['worlds']}; {kmfmt(r['t_closed_km'])}"
        r["site_counts"] = f"{r['final_scanned_open']} / {r['final_scanned_R']} / {r['final_scanned_M']} / {r['final_scanned_T']}"
        r["world_counts"] = f"{r['final_R_worlds']} / {r['final_T_worlds']} / {r['final_both_R_T_worlds']}"
    exceptions = [r for r in popworlds if r["threshold"] == "any" and r["stage"] == "G" and r["R_T_order"] in ["T before R", "T only"]]
    ratio = next(r["km_ratio_arm_over_control"] for r in effects if r["comparison"] == "6502 A / Z80 G16")
    text = f"""# EXPLORATORY — existing-data testability of H-midwife and H-ladder

**EXPLORATORY throughout. No result in this report is registered.** This is a new post hoc reading of existing archives, including archives whose original questions were registered. The two designs below are proposals, **not registrations**. Generated by `build_ladder.py`; no worlds, assays, traces, site scans, CPU emulations, Modal jobs or GPU work were run.

## EXPLORATORY answers

- **H-midwife: not addressable with existing data as a causal acceleration hypothesis.** Removing the load–push route strongly delays or prevents the first replicator at the usual L=16 setting, but the ablation atlas has no matched scored closure endpoint. Its successor-family timing proxies give mixed directions. Removing instructions also changes what closers can build and use. Existing genealogies support open-before-closed ancestry; they do not identify how much faster closure would have been without that phase.
- **H-ladder: supported as a descriptive Stage G first-detection tendency, with a narrow scope.** Among the {both} Stage G worlds in which both R and T were observed, R was detected earlier in {orders['R before T']} ({orders['R before T']/both:.1%}), at the same sample in {orders['same sample']}, and later in {orders['T before R']}. This does **not** establish a typical R→T ancestor transition. The obligatory or environment-independent *detection sequence* is contradicted by recorded T-before-R/T-only histories; missed earlier regenerators remain possible, and the typical lineage version is not addressable with the archived site scans.

Here R means a heritable confined cell with **≤2 transmissible sites**; T means heritable confined with **≥5**; M means 3–4. R is limited heredity, not literally no genome: even a zero-site scan is an assay threshold statement. “Self” is the proposed interpretation of confinement, not an independently measured identity criterion.

## EXPLORATORY methods and coverage

Inputs are existing result tables in this worktree and read-only raw files under `/Users/neo/repos/algocell/experiments/runs`. `inputs.csv` hashes every file consumed. `atlas_worlds.csv` preserves every run's arm, suppression list, actual stop, first tape, scored mechanism, timing and successor-family proxies; `atlas_groups.csv` retains every stage/label/L/budget/k/ring stratum. Aggregating those strata as if they were exchangeable would confound length, mutation, budget, ring size, initial seeds and sampling.

{table(coverage, [('stage','Stage'), ('worlds','Archived worlds'), ('arms','Recorded arms'), ('early','Stopped early')])}

`t_rep` is the first archived sample with a top-3 tape of share ≥0.5% passing gen2 ≥0.3. `tq_10` is ≥10% quasispecies occupancy; it can count zero floods. Neither establishes closure. KM medians use **actual last steps** for early-stopped atlas runs. NR means the KM median was not reached within observation; a blank closure value with UNSCORED means no closure test was archived, not no closure. Event fractions are descriptive and do not correct informative early stopping. Stage A/B sample at 500-step resolution; Stage C has 50-step early sampling.

Atlas mechanisms are archived suppression-aware disassembly labels. The pusher proxy requires `trep_mechs=stack` and period 2; this is not a trace-based open classification, and zero proxy events does not rule out rarer pushers outside the top three or later open copiers. In Stage G, the independent archived inflow table supplies the stronger check: first replicators enter the partner in all tested soup encounters in all 80 worlds. No new openness is inferred by executing archived tapes.

## EXPLORATORY H-midwife: matched ablations

The strongest atlas comparison holds L=16, budget=128, mutation k=4 and seeds 101–110 fixed in Stage C. Except all-ld/no-copy/rmw-only (1M), horizons are 300k. “Successor proxy” is the earlier of `t_ex_sp_30` and `t_ldir_30`: first census excess ≥30% over the random baseline for exchange or block-copy patterns. A pattern count neither proves that it executes nor that its pointer is confined. **Every arm below has closure UNSCORED in the atlas.** Direct closure records from later cohorts appear separately below.

{table(main, [('label','Arm'), ('pusher_count','First pusher proxy'), ('rep_count','t_rep events'), ('rep_km_display','KM t_rep'), ('rep_ratio','t_rep / none'), ('q_display','tq_10 n; KM'), ('successor_display','Successor proxy n; KM'), ('proxy_ratio','Proxy / none')])}

The names describe **removals**: `push-only` removes PUSH **and POP**, while Stage D's `push` removes PUSH alone. `stack-writes` removes stack, exchange, call/return and RST; `stack-write-only` removes PUSH, EX(SP), CALL and RST. `ld-imm` removes 8/16-bit immediates (including LD rr,nn); `all-ld` removes those plus register and memory loads. These arms interrupt the literal load–push route. `ld-reg` leaves immediate loads and PUSH available, and it still has ten pusher proxies. `stack-read-only` and `ex-sp-only` also leave that route available, while directly removing return/exchange routes used by evolved closers. `ld-mem`, `ed-loads`, `cb-page`, `call-rst` and `block-copy` are recorded controls, not clean interventions on the pusher phase. The i8080 restriction (Stage M) retains immediate loads/PUSH and supports open-first replication; its scored closure is included below.

The first-replicator ratios of {ratios['stack-writes']:g}× (`stack-writes`), {ratios['stack-write-only']:g}× (`stack-write-only`) and {ratios['push-only']:g}× (`push-only`) reproduce the archived Stage C numbers. Event risk differences against none are {risks['C stack-writes / C none']:.1f}, {risks['C stack-write-only / C none']:.1f} and {risks['C push-only / C none']:.1f}, respectively (`effect_sizes.csv`). `ld-imm` has only two t_rep events, so a finite population median or delay ratio cannot be estimated; its 10/10 tq_10 crossings do not rescue that inference. Mechanisms, including atypical `-`/mixed labels, are retained in the tables rather than re-labelled as closed cells.

For successor-family timing the directions differ: the broad `stack-writes` removal reaches the proxy earlier than none, while `stack-write-only` and `push-only` reach it later. The proxy can precede a scored t_rep (a common census motif can fail the culture test). Neither a 160× first-replicator delay nor a family takeover proves a 160× closure delay. The combined proxy also permits different successor mechanisms across arms. `ld-imm`, `all-ld` and the stack removals directly suppress closer construction/execution pathways, so even a future re-score of their saved tapes could not by itself isolate midwifery.

Stage D supplies the pure PUSH-removal arm, but at **L=9**, not a second matched L=16 test:

{table(d9, [('label','Stage D arm'), ('first_pusher_proxy_n','First pusher proxies'), ('rep_display','t_rep n; KM'), ('first_mechanisms','Scored first mechanisms')])}

The unablated L=9 group has no first-pusher proxies either, so it cannot compare an established open-pusher phase with its removal. Removing PUSH increases observed t_rep events here; that is evidence against equating a pusher instruction ablation with a uniform delay of “life”, not a clean counter-test of H-midwife.

All other A–F arm/settings, including stack removal across size/budget/mutation and nonstandard ring arms, are fully tabulated in `atlas_groups.csv`. A/B early stops censor closure observation; C–F family takeovers do not score closure. Unobserved closure in these arms therefore remains **unknown**, rather than being converted into a negative result.

## EXPLORATORY H-midwife: closure observations and remaining confounds

These are the existing **scored** top-3 closure cohorts (`closure_worlds.csv`). They are different runs from the A–F instruction ablations. G/H/I closure records are also joined back into `atlas_worlds.csv`; absent joins remain explicitly UNSCORED. Stage G L=50 has no row in the archived first-closure scorer; this is missing measurement, not evidence of no closed cells. The separate `first_closed_random_population_observed` column records closed random-cell detections where population classification exists; it is a different endpoint with a different detection threshold.

{table(closure, [('stage','Stage'), ('L','L'), ('n','Worlds'), ('t_rep_km','KM t_rep'), ('closure_display','Closed n; KM t_closed')])}

Z80 Stage G L=16 closes at a KM median of {fmt(next(r['t_closed_km'] for r in closure if r['stage']=='G' and r['L']==16))}; Stage I lethal tar at {fmt(next(r['t_closed_km'] for r in closure if r['stage']=='I'))}, a {next(r['km_ratio_arm_over_control'] for r in effects if r['comparison']=='Z80 I16 / G16'):.3g}× ratio. The lethal first mechanisms are block-copy rather than the benign pusher proxies, but lethality changes both the production and survival of closers. The corrected 6502 arm A closes {ratio:.3g}× later than G16, consistent with the narrative's approximate 50× comparison. That comparison additionally changes ISA, loops (including the Z80's LDIR), addressing, encounter cost, horizons and sampling. It is not a measured effect of the missing pusher phase.

The already recorded lethality dial permits an observational within-p stratification by **scored first-open status**:

{table([{**r, 'km_display':kmfmt(r['t_closed_km'])} for r in dial], [('p','p'), ('first_open','First open'), ('n','Worlds'), ('closed_n','Closed'), ('km_display','KM t_closed')])}

These groups are small and unbalanced. First-open status is an outcome, not random assignment; stratifying on it selects evolutionary histories. `effect_sizes.csv` gives the available same-p KM ratios and event risk differences without causal claims. Changing p itself also changes closer survival. None of these comparisons meets H-midwife's needed intervention.

### EXPLORATORY proposed registered design for H-midwife — NOT REGISTERED

Keep the complete Z80 ISA, L, actual mutation policy, topology, budget, partner environment and final horizon identical. Randomize paired initial worlds to a fixed early window in which replication by preclassified open copiers is allowed or selectively vetoed, with a sham arm and a matched nonspecific rewrite-reduction control. Restore identical unrestricted dynamics after the window. Predefine the classifier/veto so it leaves closed copying and non-copying rewrite events intact; validate existing regenerator and transmitter controls in both arms. Add a pusher-seeding rescue and a matched noncopying-seed control to distinguish ecological rewriting from instruction availability. Such interventions have their own ecological effects, which the flux-matched control must quantify.

Freeze the intervention, window, classifier, seed list, power/sample-size calculation, analysis and exclusion rules **before collecting new worlds**. Register t_closed from a population confinement/culture criterion that also catches heterogeneous transmitters, plus first closed birth in complete event genealogies and the secondary top-3 endpoint. Sample all arms on the same schedule; no occupancy-triggered early stop. Analyze intention to treat at the world level with survival/RMST effect sizes and confidence intervals, keeping censored worlds; do not select groups by whether a pusher happened to arise. Test the intervention's early-open manipulation and closer-control behavior separately. No part of this proposed experiment was run here.

## EXPLORATORY H-ladder: population order over time

The archived estimator samples 64 random cells per snapshot, culture-tests with 32 partners, traces with 16, and scans up to 16 heritable cells (16 alternatives/site, 16 partners). Confined R/M/T population shares extrapolate the proportions among scanned confined cells. First positive R/T shares imply at least one scanned example, not population-wide emergence. Dates below are **first detections**, not birth times; prior random-sample non-detections are not lower bounds. Irregular snapshot gaps, rare-cell misses, assay noise and the small scan can create ties or reverse apparent order. Duplicate final/dated snapshots at the same step are independent samples at that step; they are not extra worlds.

{table(popany, [('stage','Stage'), ('L','L'), ('n','Worlds'), ('R before T','R earlier'), ('same sample','Same sample'), ('T before R','T earlier'), ('R only','R only'), ('T only','T only'), ('neither','Neither')])}

For Stage G, all 80 first replicators are scored open in the archived soup-inflow analysis. First positive population-class samples have these conditional medians (different subsets for each median, **not a median evolutionary trajectory**):

{table(conditional, [('L','G length'), ('open','First open sample'), ('R','First R sample'), ('R_n','Worlds with R'), ('T','First T sample'), ('T_n','Worlds with T')])}

Considering all observed closed classes (R/M/T), the random-cell sampling detects open cells first in {open_closed['open earlier']}/80 G worlds, open and closed at the same step in {open_closed['same sample']}, and closed cells first in {open_closed['closed earlier']}; {open_closed['open only']} world has only open-class detections. All five closed-first observations occur at L=64. This is compatible with the top-3 first replicator being open: rare-cell sampling and modal-tape emergence are different measurements. It precludes claiming that these population snapshots resolve open→closed birth order in every world.

Across G: {orders['R before T']}/{both} worlds with both detected have R earlier; {orders['same sample']}/{both} tie; {orders['T before R']}/{both} have T earlier. Including T-only histories in the denominator gives {orders['R before T']}/{both+orders['T only']} ({orders['R before T']/(both+orders['T only']):.1%}) of T-observed worlds with a detected prior R. Restricting to temporally separated detections gives {orders['R before T']}/{orders['R before T']+orders['T before R']}, but that conditional fraction must not discard ties when stating a population claim. {orders['R only']} additional worlds have only R detected, {orders['T only']} only T, and {orders['neither']} neither. R-only worlds demonstrate no observed ascent to T within their horizon. The aggregate is dominated by L=50/64; it does not describe short tapes uniformly.

The G counterexamples are explicit:

{table(exceptions, [('L','L'), ('seed','Seed'), ('first_regenerator_observed','First R detection'), ('first_transmitter_observed','First T detection'), ('R_T_order','Observed order')])}

In lethal Stage I L=16, T is detected before R in {stage_i['T before R']} worlds and T appears without detected R in {stage_i['T only']} more, whereas only {stage_i['R before T']} world has R first. The recorded sequence therefore does not require a detected intermediate R phase; sparse scans cannot exclude a missed one. The benign long-horizon L/M cohorts also show no necessary eventual T: M16 has no detected T in any of 20 worlds; L16 has R-only histories in nine of ten.

The tendency depends on what “appears” means. `population_groups.csv` also repeats the analysis at **>10%** and **>50%** of heritable cells. Among G worlds where both classes ever become a majority, R-majority is earlier in {majority_before}/{majority_both}, T-majority earlier in {majority_after}/{majority_both}. Thus the first rare-cell tendency does not imply a majority succession ladder. These cutoffs are exploratory sensitivity analyses, not additional registered tests.

## EXPLORATORY H-ladder: genealogy and N2

The final v6 genealogy cohorts, rather than older registered/amended copy-rule tables, supply the matched open-first evidence. Their closure times are the genealogy pipeline's **own** event/operational endpoint, not the top-3 t_closed endpoint above:

{table(lod, [('cohort','Cohort'), ('worlds','Worlds'), ('closed_worlds','Closed worlds'), ('sampled_lines','Sampled lines'), ('open_first_lines','Open-first lines'), ('all_lines_open_first_worlds','All lines open-first worlds'), ('median_earliest_open_per_closed_world','Median earliest open/world'), ('median_earliest_confined_per_closed_world','Median earliest confined/world'), ('median_t_close_conditional','Median lod closure')])}

All 2,368 sampled lines in 37 closed worlds start with an open copier before a confined one. The 64 lines within each world are correlated, often sharing the same first-copier record; **37 worlds**, not 2,368 independent replicates. Three N3 worlds did not close by the archive's stop. The oldest confined copier is not always the founder selected after the last open segment: the archived founder-equals-earliest counts are {lod[0]['earliest_confined_is_founder_lines']}/{lod[0]['sampled_lines']} and {lod[1]['earliest_confined_is_founder_lines']}/{lod[1]['sampled_lines']}. The genealogy classes are only open copier / confined copier / non-copier. No contemporaneous transmissible-site scan labels the earliest confined cell as R or T, so the second transition is **unscored**. Periodicity or an executed-byte count cannot substitute for those scans. Read-only replay-validation metadata agree (V2 mismatch zero); no replay was performed here.

N2 is an equilibrium continuation, not an origin experiment. Its R/T classes are the predeclared first-four-byte core/offset proxies at L=32 (O and M are other/core-variant cells), not new site scans. Both classes already occur in every world's first saved count, after continuation step 1:

{table(n2, [('composition','Starting composition'), ('rule','Continuation rule'), ('worlds','Worlds'), ('both_at_first_saved_count','Both at first count'), ('mean_T_share_start','Mean T share start'), ('mean_T_share_end','Mean T share end'), ('mean_g_R','Mean g_R/cell-step'), ('mean_g_T','Mean g_T/cell-step')])}

`n2_counts.csv` preserves all nine saved counts per continuation. These data describe persistence and differential demographic rates of established R/T mixtures; neither origin order nor an R→T ancestry transition can be inferred. O in N2 is “no core”, not a scored open first replicator.

## EXPLORATORY 6502 S1 in the same terms

Use the corrected `s6502/v2` **frozen32** results (same worlds as the original, not 80 new worlds); do not pool the 64-partner sensitivity as independent observations. A: L16/128 BRK faithful; B: L32/128 faithful; C: L16/512 faithful; D: L16/128 BRK halts. All have a 10M horizon. Final site scans give:

{table(s6502, [('arm','Arm'), ('rep_display','t_rep n; KM'), ('first_open_n','First open'), ('first_confined_n','First confined'), ('first_confined_loop_n','First confined+loop'), ('closed_display','t_closed n; KM'), ('site_counts','Final scanned open/R/M/T'), ('world_counts','Final worlds R/T/both')])}

All arm A/B/D first replicators are confined; C has one open first replicator (with a loop, not an open straight-line pusher). A has 14/20 confined first replicators with a detected loop, so “all first selves loop” is not supported by this score. No open straight-line top-3 qualifying tape was recorded before closure in any arm. That does not exclude subthreshold open copiers. The narrative's width-one prediction concerns open straight-line literal patterns, not every possible open replication mechanism.

The R/T scans are **only at the final 10M snapshot**. They establish coexistence and class occurrence across worlds, but no t_R, t_T or first-replicator site class. Thus S1 does not answer whether its first closed cell was R, whether R preceded T, or whether a T descended from an R. It does show that an observed open straight-line phase is not required for scored closure under this substrate. Final R detections occur in five B worlds despite only four modal t_rep events, and in all 20 D worlds despite 18 modal closure events: a censored top-3 event is not proof that the population contains no closed heritable cells.

The correction audit records a partner-source discrepancy: preregistration says soup partners, while the frozen classification used uniform random bytes. A later soup-snapshot sensitivity retains first-replicator status but may use a snapshot after t_rep; it supplies no new historical R/T order. The audit also records a shared packed-word mutation lost-update race, so nominal mutation attempts are not guaranteed applied mutations. These limitations remain in the fixed archives. No corrected worlds were generated here.

### EXPLORATORY proposed registered design for H-ladder — NOT REGISTERED

Define the target claim before running: population order, ancestor order, or both. Register a fixed panel of benign/lethal Z80 and 6502 conditions, matched horizons/cadence, seeds and powered world counts. Define “typically” as a prespecified world-level fraction exceeding one half with a confidence interval; specify the denominator (all worlds, T-producing worlds, and a separately reported censored set). Retain same-sample and unresolved intervals rather than silently dropping them.

Record complete encounters, rewrite donors and mutation events from step zero. On sampled T lineages, classify every distinct heritable ancestral tape against prespecified uniform and contemporary soup partners, then scan transmissible sites with fixed alternatives/partners and blind thresholds. Record t_open, t_R, t_M and t_T at the birth events; distinguish a first T produced by rewrite from a descendant transition inherited through an R. Do not infer heredity from copied-but-unexecuted byte counts alone. Predefine scan uncertainty/sensitivity and sustained population occurrence thresholds so a fleeting rare R is distinguished from an established phase. Primary endpoint: fraction of worlds whose earliest established T has an earlier R in its recorded ancestry; secondary: first sustained population class order. Report skipped rungs, coexistence, reversions, censoring and ties at the world level. Freeze code and the actual mutation policy first. No experiment or assay for this design was run here.

## EXPLORATORY reproduction and provenance

```sh
cd /Users/neo/repos/algocell-wt-ladder
/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/results/ladder/build_ladder.py --archive /Users/neo/repos/algocell/experiments/runs
```

All generated CSVs contain a `status=EXPLORATORY` column. `inputs.csv` records source paths, byte sizes and SHA-256 hashes; `effect_sizes.csv` gives event fractions/risk differences and identifiable KM ratios; blanks preserve missing/unreached medians. No inferential P values were added. Source definitions: [Round-1b narrative](../../planning/narrative/CLAUDE_R1b.md), [revision preregistration](../../REVISION_PREREG.md), [numbers audit](../../manuscript/NUMBERS_INDEX.md), [population classifier](../../population_classes.py), [first-closure scorer](../../first_closure.py), [N2 definitions](../../n2_demography.py), [6502 correction](../s6502/CORRECTION.md). Consult `atlas_groups.csv` for every recorded atlas arm/setting, not just the displayed contrasts.
"""
    (OUT / "LADDER.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("/Users/neo/repos/algocell/experiments/runs"))
    a = parser.parse_args()
    assert a.archive.is_dir()
    aw, ag = mine_atlas(a.archive)
    cw, cg = mine_closure()
    pw, pg = mine_population()
    lw, lg = mine_lod(a.archive)
    ng = mine_n2(a.archive)
    sw, sg = mine_6502()
    ef, dg = mine_effects(ag, cg, sg)
    for relative in ["../../planning/narrative/CLAUDE_R1b.md", "../../REVISION_PREREG.md", "../../manuscript/NUMBERS_INDEX.md",
                     "../../population_classes.py", "../../first_closure.py", "../../n2_demography.py", "../../assay_batch.py",
                     "../../succession.py", "../../make_conds.py", "../../score6502.py"]:
        track(OUT / relative)
    report(aw, ag, cg, pw, pg, lg, ng, sw, sg, ef, dg)
    write_csv("inputs.csv", sorted(INPUTS.values(), key=lambda r: r["path"]))
    print(f"EXPLORATORY: {len(aw)} atlas worlds; {len(ag)} atlas strata; {len(pw)//3} population worlds; {len(sw)} 6502 worlds; {len(INPUTS)} hashed inputs. No simulations/assays/GPU.")


if __name__ == "__main__":
    main()
