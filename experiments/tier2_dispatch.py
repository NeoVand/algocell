"""I1-T2 dispatch manifest, completion markers and spend accounting (no scoring)."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "results/tier2"
RATE = 0.000222 + 0.0000131 + 8 * 0.00000222
CAP = 40.0
PANEL_RESERVE = 4.0
OTHER_RESERVE = 3.2  # startup $1, storage $0.20, preparation $2
FROZEN = {
    "DESIGN_FULL.json": "ebc567db3b43b25c60bbfdbfef6556ec0f420807009ed44ffe9f3f8f456d4ffb",
    "control_inputs.json": "e122e8a664787b68ddce80165969c3f64a4e567f29c5d46562bfc05591a321cd",
}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def batch_stem(c):
    return f"{c['label']}_L{c['tape']}_st{c.get('z80_steps', 128)}_k{c.get('noise_exp', 4)}_s{c['seed']}"


def jobs():
    for name, sha in FROZEN.items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Frozen registration input changed: {name}")
    result = json.loads((DATA / "DESIGN_FULL.json").read_text())
    for j in result:
        e, c = j["experiment"], j.get("condition", {})
        if e == "E" and c["L"] == 32 and c["tar"] == "lethal":
            j["experiment"] = "E-L32L"
        j["kind"] = "batch" if e in ("L", "F") else ("dial" if e == "DZ" else "offset" if e == "E" else "invasion")
    for c in json.loads((HERE / "conds/stageE.json").read_text()):
        if c["label"] == "none@nominal" and c["tape"] == 10:
            c["machine"] = "imp_L10_c1"
            result.append(dict(experiment="StageE-L10", kind="batch", tag="L10", machine="imp_L10_c1", condition=c, steps=c["horizon"]))
    controls = json.loads((DATA / "control_inputs.json").read_text())
    for gt in controls["panel"]:
        result.append(dict(experiment="Serial", kind="serial", genotype=gt, seed=20261009, machine="tier2c1_assay"))
    for key in ("pusher16", "pusher64", "ret16", "ldir32", "genome20_s6003"):
        gt = next(g for g in controls["panel"] if g["key"] == key)
        for arm in (("U", "K") if key in ("pusher16", "pusher64") else ("U",)):
            result.append(dict(experiment="Branching", kind="branching", genotype=gt, arm=arm, seed=20261010, machine="tier2c1_assay"))
    rates = {x["tag"]: min(x["steps_per_s"]) for x in json.loads((DATA / "l4.json").read_text())["fresh"]}
    for j in result:
        kind, c = j["kind"], j.get("condition", {})
        if kind == "batch":
            stem = batch_stem(c)
            req = [f"{stem}.jsonl", f"{stem}.summary.json", f"{stem}.soup_final.u8.br"]
            req += [f"{stem}.soup_t{t}.u8.br" for t in c.get("snapshot_steps", []) if 0 < t <= j["steps"]]
        elif kind == "dial":
            stem = f"{j['variant']}_s{j['seed']}"
            req = [f"{stem}.jsonl"] + [f"{stem}_t{t}.npy" for t in (2000, 20000, 100000, 300000)]
        elif kind == "offset":
            stem = f"L{c['L']}_{c['tar']}_mut{c['mut']}_{c['start']}_s{c['seed']}"
            req = [f"{stem}.jsonl", f"{stem}_final.npy"]
        elif kind == "invasion":
            stem = f"{c['variant'] or 'std'}_{c['resident']}_{c['invader']}_s{c['seed']}"
            req = [f"{stem}.jsonl", f"{stem}_final.npy"]
        else:
            stem = j["genotype"]["key"] + (f"_{j['arm']}" if kind == "branching" else "")
            prefix = "sr" if kind == "serial" else "branching"
            req = [f"{stem}/{prefix}_mutants.csv.gz", f"{stem}/{prefix}_sites.csv", f"{stem}/{prefix}_summary.csv"]
        j.update(stem=stem, required=req, id=f"{j['experiment']}/{stem}")
        j["timeout"] = 14400 if j["experiment"] in ("L", "F", "StageE-L10") else 3600 if kind == "dial" else 7200
        if kind in ("serial", "branching"):
            # Half of the existing $4 assay/scoring reserve for panel execution;
            # keep the other half for subsequent soup scoring.
            j["forecast_usd"] = 2.0 / 21
        else:
            # L10 has no dedicated timing: reserve $0.50 (above the slowest
            # measured fresh-state projection for its 3M aggregate steps).
            j["forecast_usd"] = 0.05 if j["tag"] == "L10" else j["steps"] / rates[j["tag"]] / 0.8 * 1.1 * RATE
        j["spec_sha256"] = digest(j)
    assert Counter(j["experiment"] for j in result) == {"DZ": 60, "E": 90, "E-L32L": 20, "L": 20, "R4": 32, "F": 10, "StageE-L10": 10, "Serial": 14, "Branching": 7}
    assert len({j["id"] for j in result}) == len(result)
    return result


def marker_name(j, kind):
    return f"{j['stem']}.{kind}.json"


def complete(j, names, read_json):
    """Only names and compact summary markers; never read soup arrays/tables."""
    marker = marker_name(j, "complete")
    if marker not in names or not set(j["required"]).issubset(names):
        return False
    m = read_json(marker)
    if m.get("status") != "complete" or m.get("spec_sha256") != j["spec_sha256"]:
        return False
    if j["kind"] == "batch":
        s = read_json(f"{j['stem']}.summary.json")
        used = s.get("machine")
        used = used.get("machine") if isinstance(used, dict) else used
        if s.get("steps_run") != j["steps"] or used != j["machine"]:
            return False
        if s.get("tq_10", -1) >= 0 and f"{j['stem']}.soup_emergence.u8.br" not in names:
            return False
    return True


def local_complete(j, root):
    base = Path(root) / j["experiment"]
    names = {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()}
    return complete(j, names, lambda n: json.loads((base / n).read_text()))


def projection(all_jobs, markers, now=None):
    now = now or time.time()
    estimated, remaining = 0.0, 0.0
    for j in all_jobs:
        m = markers.get(j["id"], {})
        elapsed = max(0, m.get("finished_epoch", now) - m.get("started_epoch", now))
        cost = elapsed * RATE
        estimated += cost
        if m.get("status") != "complete":
            remaining += max(0, j["forecast_usd"] - cost) if m.get("status") == "running" else j["forecast_usd"]
    # $2 panel execution is assigned above; $2 remains for later scoring.
    return dict(estimated_execution_usd=estimated, required_remaining_usd=remaining,
                reserves_usd=OTHER_RESERVE + 2.0, total_forecast_usd=estimated + remaining + OTHER_RESERVE + 2.0)


def write_spend(path, all_jobs, markers=None, billing=None):
    p = projection(all_jobs, markers or {})
    counts = Counter(j["experiment"] for j in all_jobs)
    lines = ["# I1-T2 running spend ledger", "", f"Updated UTC: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}.", "",
             "COST.md rates: L4 $0.000222/s + one CPU core $0.0000131/s + 8 GiB × $0.00000222/s = $0.00025286/container-second. No credits or automatic retries assumed.", "",
             "| Experiment | Required worlds/jobs | Forecast USD |", "|---|---:|---:|"]
    for e, count in counts.items():
        lines.append(f"| {e} | {count} | {sum(j['forecast_usd'] for j in all_jobs if j['experiment'] == e):.4f} |")
    lines += ["| Later soup scoring reserve | — | 2.0000 |", "| Startup / compilation reserve | — | 1.0000 |", "| Storage / egress reserve | — | 0.2000 |", "| Preparation reserve (prior work) | — | 2.0000 |", "",
              "Soup costs retain COST.md's slowest fresh-state rate, 20% throughput reduction and 10% runner overhead. Stage E L10 adds a conservative $0.50. Serial/branching execution uses $2 of the already registered $4 assay/scoring reserve; $2 remains reserved for subsequent soup scoring. Panels emit measurement tables only; claim reports are deferred.", "",
              f"Initial all-in forecast: **${projection(all_jobs, {})['total_forecast_usd']:.4f}**. Cap: **$40**.",
              f"Runtime estimate so far: **${p['estimated_execution_usd']:.4f}**; required remaining: **${p['required_remaining_usd']:.4f}**; other reserves: **${p['reserves_usd']:.2f}**; current all-in forecast: **${p['total_forecast_usd']:.4f}**.",
              f"Modal app-attributed metered spend: {billing or 'unavailable before launch; status script queries billing if supported' }.", "",
              "Runtime estimates use per-job start/end markers and allocated resource rates, excluding queue time. Modal billing is authoritative when available and may lag; runtime estimates are not claimed as actual charges. Each snapshot refreshes a marker; the status command can refresh this ledger with --update-ledger. Dispatch checks the remaining-work projection before launch and between batches, and containers check it before starting. If it exceeds $40, pending dispatch stops without shortening any design; incomplete work is never scored.", ""]
    Path(path).write_text("\n".join(lines))
    return p
