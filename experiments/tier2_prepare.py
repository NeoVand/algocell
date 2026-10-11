"""Generate I1-T2 inventory, condition manifest, machine hashes, and cost. Never runs a world."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/tier2"
BASE = "0970b82"
ARCHIVE = "/Users/neo/repos/algocell/experiments/runs"

COMMANDS = {
    "DZ": "$PY dial_soups.py --analyse --machine c1 --runs $R/dial --out $O/dial",
    "E": "$PY offset_switch.py --analyse --machine c1 --indir $R/offset --out $O/offset\n$PY offset_encounter.py --machine tier2c1_assay --runs $R/offset --out $O/offset/OFFSET_ENCOUNTER.md",
    "L": "$PY assay_batch.py $R/stageL --machine tier2c1_assay\n$PY c4_functional.py $R/stageL --machine tier2c1_assay\n$PY stage_g.py --dir $R/stageL --out $O/stageL --no-verdicts --machine tier2c1_assay\n$PY capacity_over_time.py --yes --stages L --machine tier2c1_assay --runs-dir $R/stageL --runs-table $O/stageL/stage_g_runs.csv --out $O/capacity_time_L\n$PY score_L.py --cap $O/capacity_time_L/capacity_over_time.csv --runs $O/stageL/stage_g_runs.csv --out $O/stageL/SCORING.md",
    "S": "$PY serial_retention.py --yes --machine tier2c1_assay --out $O/serial\n$PY serial_retention.py --report --out $O/serial",
    "X1": f"$PY branching_transfer.py --machine tier2c1_assay --runs-root {ARCHIVE} --out $O/branching\n$PY branching_transfer.py --report --out $O/branching",
    "R4": f"$PY r4_analyse.py --machine c1 --runs $R/r4_inv --seed-path-root {ARCHIVE}/r4_inv/r4_inv --out $O/r4",
    "F": "$PY assay_batch.py $R/stageF --machine tier2c1_assay\n$PY stage_f.py $R/stageF --f4c-only --machine tier2c1_assay --out $O/stageF",
}


def design():
    import dial_soups as D
    import offset_switch as E
    import invasion_generic as R
    from algocell_exp.tier2 import tag_for
    jobs = []
    for v in D.DIAL:
        for seed in D.SEEDS:
            jobs.append({"experiment": "DZ", "tag": f"{v}_L16", "variant": v, "seed": seed, "steps": 300000})
    for c in E.conditions() + E.conditions_l32lethal():
        jobs.append({"experiment": "E", "tag": tag_for(c["L"], lethal=c["tar"] == "lethal"), "condition": c, "steps": c["steps"]})
    for c in json.loads((HERE / "conds/stageL.json").read_text()):
        jobs.append({"experiment": "L", "tag": f"L{c['tape']}", "condition": c, "steps": c["horizon"]})
    for c in R.conditions():
        jobs.append({"experiment": "R4", "tag": tag_for(16, variant=c["variant"]), "condition": c, "steps": c["steps"]})
    for c in json.loads((HERE / "conds/stageF.json").read_text()):
        if c["tape"] == 8 and c["label"] == "none@ring32":
            jobs.append({"experiment": "F", "tag": "L8_P32", "condition": c, "steps": c["horizon"]})
    for j in jobs:
        j["machine"] = f"imp_{j['tag']}_c1"
        if j["experiment"] in ("L", "F"):
            j["condition"]["machine"] = j["machine"]
    return jobs


def frozen_manifest():
    names = ["dial_soups.py", "offset_switch.py", "r4_analyse.py", "offset_encounter.py", "assay_batch.py",
             "c4_functional.py", "stage_g.py", "capacity_over_time.py", "score_L.py", "stage_f.py",
             "serial_retention.py", "branching_transfer.py", "mutscan.py", "population_classes.py"]
    rows = []
    sha = lambda b: hashlib.sha256(b).hexdigest()
    for n in names:
        frozen = subprocess.check_output(["git", "show", f"{BASE}:experiments/{n}"], cwd=HERE)
        current = (HERE / n).read_bytes()
        rows.append({"script": n, "baseline_commit": BASE, "baseline_sha256": sha(frozen), "prepared_sha256": sha(current),
                     "changes": "none" if current == frozen else "explicit CLI/input/backend routing and targeted F4c reporting; registered numeric measurements/formulas retained"})
    # These numerical functions must remain byte-for-byte the same AST, independent of routing additions.
    functions = {"dial_soups.py": ["_km_median", "_perm_spearman", "report"],
                 "offset_switch.py": ["classify"], "stage_f.py": ["cell"],
                 "serial_retention.py": ["run_genotype", "sites", "report"],
                 "branching_transfer.py": ["branch_step", "run_arm", "mutants", "report"]}
    checks = []
    for n, fs in functions.items():
        sources = [subprocess.check_output(["git", "show", f"{BASE}:experiments/{n}"], cwd=HERE).decode(), (HERE / n).read_text()]
        parsed = [{x.name: ast.dump(x, include_attributes=False) for x in ast.parse(s).body if isinstance(x, ast.FunctionDef)} for s in sources]
        for f in fs:
            if f not in parsed[0]:
                raise ValueError(f"unknown frozen function {n}:{f}")
            assert parsed[0][f] == parsed[1][f], f"frozen scorer changed: {n}:{f}"
            checks.append({"script": n, "function": f, "ast_sha256": sha(parsed[0][f].encode()), "identical": True})
    return {"scripts": rows, "frozen_numerical_functions": checks}


def write_machine_hashes():
    root = HERE / "algocell_exp/machines"
    own = json.loads((root / "worlds_tier2.json").read_text())
    names = set(own) | {j["machine"] for j in design()} | {j["machine"].replace("_c1", "_c0") for j in design()}
    rows = []
    for name in sorted(names):
        p = root / f"{name}.manifest.json"
        m = json.loads(p.read_text())
        rows.append({"machine": name, "core_sha256": m["sha256"]["core"], "world_sha256": m["sha256"]["world"],
                     "wgsl_sha256": m["sha256"]["wgsl"], "manifest_sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    write_csv(OUT / "machines.csv", rows)


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cost(jobs):
    # Conservative timing: slowest measured fresh-state repetition. Mature-state rates are a sensitivity only.
    prices = {"L4": 0.000222, "L40S": 0.000542, "cpu_core": 0.0000131, "memory_gib": 0.00000222}
    reports = []
    for gpu, filename in (("L40S", "bench.json"), ("L4", "l4.json")):
        record = json.loads((OUT / filename).read_text())
        records = record["fresh"] if isinstance(record, dict) else record
        rates = {x["tag"]: min(x["steps_per_s"]) for x in records}
        for experiment in ("DZ", "E", "L", "R4", "F"):
            group = [j for j in jobs if j["experiment"] == experiment]
            seconds = sum(j["steps"] / rates[j["tag"]] for j in group)
            # One reserved physical CPU core and 8 GiB for the entire container lifespan.
            per_s = prices[gpu] + prices["cpu_core"] + 8 * prices["memory_gib"]
            reports.append({"gpu": gpu, "experiment": experiment, "worlds": len(group), "steps": sum(j["steps"] for j in group),
                            "raw_seconds": seconds, "raw_usd": seconds * per_s,
                            "budget_usd": seconds * per_s / 0.8 * 1.1})
    write_csv(OUT / "cost.csv", reports)
    lines = ["# I1-T2 cost estimate (generated)", "", "Rates: [Modal pricing](https://modal.com/pricing), fetched 2026-10-11: L4 $0.000222/s; L40S $0.000542/s; CPU $0.0000131/core/s; memory $0.00000222/GiB/s. No free credits assumed.", "",
             "`bench.json`, `mature.json`, and `l4.json` record C1 timing on Modal. Project using the **slowest fresh-state repetition per world**, reduce throughput a further 20%, then add 10% for runner sampling/serialization. Mature Stage L states are faster and receive no discount in this budget. Allocate one physical CPU core plus 8 GiB for the full container lifespan; specify these bounds in the registered launcher.", "",
             "| GPU | experiment | worlds | registered steps | budget USD |", "|---|---|---|---|---|"]
    for r in reports:
        lines.append(f"| {r['gpu']} | {r['experiment']} | {r['worlds']} | {r['steps']:,} | {r['budget_usd']:.2f} |")
    totals = {}
    for gpu in ("L4", "L40S"):
        world_cost = sum(r["budget_usd"] for r in reports if r["gpu"] == gpu)
        totals[gpu] = world_cost + 7.2
        lines += ["", f"**{gpu}: worlds ${world_cost:.2f}; complete preparation + tier-2 envelope ${totals[gpu]:.2f}.** Add $4 for all frozen scoring/serial/branching assays, $1 startup/compilation, $0.20 storage/egress, and $2 preparation smoke/proofs/benchmarks. These are reserves, not measured charges. The $4 scoring reserve covers about 4.4 L4 GPU-container hours at the allocated CPU/RAM price."]
    lines += ["", "Choose **L4** for the complete 232-world design plus the full 14-genotype serial and five-genotype/two-arm branching controls. Nothing is dropped; all six DZ p levels, all original seeds/horizons, E's mutation-off tests and L32 environment controls, and both Stage L lengths remain intact. Long Stage L containers need at least a 2-hour timeout (4 hours recommended).",
              "", "The L40S-only envelope exceeds the $40 cap; changing GPU to L4 preserves the design. Do not silently truncate seeds, p levels, or horizons to enforce the cap. Use a global spend ledger: stop dispatching if the forecast including the remaining required worlds exceeds $40; checkpoint unfinished work and request the coordinator's revised registration. A budget stop is an incomplete test, never a verdict.",
              "", "Residual uncertainty: C1 may evolve states slower than the C0 archive performance inputs; pure dispatch timing omits some CPU scoring and runner I/O. The fresh-state floor, 20% rate reduction, runner uplift and explicit reserves address this without claiming a guaranteed bill. No auto-retries are budgeted; a retry must consume the same global reserve. Raw timing was on NVIDIA GPUs; no registered world was run."]
    (OUT / "COST.md").write_text("\n".join(lines) + "\n")
    return totals


def inventory():
    rows = [
        ["DZ — ED 8b–d", "`dial_soups.py::run_world`; `modal_dial.py --part dial`", "60 = 6 p × 10; seeds 9001–9010 at each p; 300,000; L16/P32; 20,000 square-lattice cells, 8,192 pairs/step, 128 instructions; p={0,.01,.03,.1,.3,1}; raw-PCG lethal fetch hook; standard zero registers/SP-last-byte; 512 mutated bytes/step (k4)", "DZ1: endpoints ≥8 open-first at p0, ≤2 at p1 + falling Cochran–Armitage rank trend, one-sided P<.05. DZ2: KM closure medians ≤1 reversal + positive Spearman with permutation P<.05. DZ3: positive transmitter trend P<.05; transmitter-majority ≤3 worlds at p0, ≥7 at p1. `REVISION_PREREG.md` §DZ:540–558; outcome/correction:632–645,667–672", "`dial_soups.py --analyse`, world-specific mapped dial machines; unchanged numerical report/KM/permutation functions", "DZ1 met; DZ2 correlation met, monotonic medians not met (2 reversals), aggregate partly met/kill applied; DZ3 trend/p0 met, p1 not met (6/10), aggregate partly met. p1 open-first 0/9 with a replicator, 0/10 registered worlds.", f"{ARCHIVE}/dial/dial → $R/dial"],
        ["E — ED 6c/e", "`offset_switch.py::run_world`; `modal_offset.py`", "90 original: L16 benign, L16 lethal, L32 benign; each: ten mix50 mutation-off (100,000), ten mix50 on (300,000), five R_into_T on + five T_into_R on (300,000); seed IDs1–10 or1–5; actual soup seeds `7000+100*(L32)+50*(lethal)+ID+1000*start_index+20000*(off)`; P=2L, standard convention, same grid/pairs/budget/k4 or0 mutation", "E1 benign mutation-off T/(R+T) ∈ [.3,.7] in ≥7/10; E2 mutation-on end below start and ∈ [.05,.4] in ≥7/10 mixtures; E3 lethal no direction. `REVISION_PREREG.md` §E:390–405, outcome434–448; correction571–577", "`offset_switch.py --analyse` (core-independent byte classifier; `--machine c1` validates world metadata); `offset_encounter.py` with mapped C1 assays for ED6e (post hoc descriptive)", "E1 not met (3,1,2/10 as reported across benign16/lethal16/benign32; only benign arms direction-registered); E2 not met (3,0,6/10); L16 benign on core collapses, share not interpretable; lethal16 on converges to T-rich descendants. Encounter components descriptive, no binary registered direction.", f"{ARCHIVE}/offset (archived files are flat) → $R/offset"],
        ["E-L32L — environment control", "`offset_switch.py::run_world`; `modal_offset.py --part l32lethal`", "20 additional L32 lethal/P64; ten mix50 on seeds1–10; five each invasion on seeds1–5; 300,000, standard convention/k4; same seed formula and grid/pairs/budget as E", "T/(R+T)>.5 in ≥7/10; kill if <.4 in ≥7/10; `REVISION_PREREG.md` §E-L32L:607–615, outcome625–630", "`offset_switch.py --analyse` CSV; endpoint threshold read from its frozen `T_share_end` (script labels lethal arms descriptive, so register the E-L32L threshold separately)", "Met10/10; kill not fired. No C1 flag here; retained because this registered control removes length confounding for E's environmental-choice claim.", f"{ARCHIVE}/offset → $R/offset"],
        ["Stage L — Fig4, ED4", "`algocell_exp.run.run` via `algocell_exp.batch.run_to_dir`, `modal_app.py`, `conds/stageL.json`", "20 = ten per L16/P32 and L20/P40; seeds6001–6010; 10,000,000; square160×125, 8192pairs,128instructions,k4,plain,standard convention; original sampling/snapshot arrays preserved. L16 retained for joint kill and serial-genome control", "L1 ≥5/10 recover sites at10M relative closed reference300k(L16)/1M(L20); L2 all recovery block-copy with unexecuted sites; L3 no heritable non-block dominant >1 site at any snapshot; joint kill L1 fails both and ≥8/10 final ≤1site both. `REVISION_PREREG.md` §L:131–145; outcome202–207", "`assay_batch.py`, `c4_functional.py`, `stage_g.py --no-verdicts`, `capacity_over_time.py`, `score_L.py`; mapped C1 set for every assay", "L1 not met1/10,0/10; L2 not met16,n/a20; L3 not met both; closed-only L3 met16/notmet20 (unregistered restriction); kill fired; all20 closed. Frozen L2 scorer implements ≥half sites unexecuted, versus registration's all-site wording: retain/report both without changing formula.", f"{ARCHIVE}/stageL/stageL → $R/stageL"],
        ["Serial transmitter controls", "`serial_retention.py` (assay experiment, no soup runner)", "All14 original panel genotypes, including three fixed Stage L tapes; K255 alternative values/site L≤32, K32 atL50/64;16mutant lineages,256controls;8transfers,128instructions; seed20261009 (panel payload seed20261010); original genotype rule/masks, random partners; no world mutation", "S1 pushers50/64 ≥half first-generation sites and ≥10 atg4; S2 four evolved closers≤2sitesg4 and erased>retainedg1; S3 constructed8080 ≥11/12payloadsitesg8 and alive≥.99; S4 each transient genome≥5sitesg4; S5 descriptivepusher16≤2. `REVISION_PREREG.md` R2/S:235–269; outcome328–329; correction529–532", "`serial_retention.py --machine tier2c1_assay --yes`, then `--report`; full frozen panel and scoring", "S1 not met; S2,S3,S4,S5 met. Keep the originally named tapes as fixed assay inputs, not new C1 dominants substituted post hoc.", "Original results/serial_retention; no soup runs directory; outputs $O/serial"],
        ["Branching transmitter controls", "`branching_transfer.py` (assay experiment)", "Original5genotypes U, pusher16/64 also K;32seeded alternatives/site,4mutant populations,64controls;8generations,k2fresh encounters/member,cap32;seed20261010;128instructions,genotype rule/masks. K pools fixed archived20StageG worlds at5000/10000", "X1-1 pusher16 U control aliveg8≥.5; X1-2 pusher16 U≥1siteg8; X1-3 ret16/ldir32≤2,genome20_s6003≥5; K reported. `REVISION_PREREG.md` §X1:1057–1075, outcome1113–1122", "`branching_transfer.py --machine tier2c1_assay`, then `--report`; freeze inputs and RNG", "X1-1 met48/64; X1-2 not met0sitesU/K; X1-3 met0,0,9. Published reading: persistent variation arrives with closed transmitters.", f"{ARCHIVE}/stageG is read-only K partner source; outputs $O/branching"],
        ["R4 — C4b/C4c", "`invasion_generic.py::run_world`; `modal_r4.py --part inv`", "32 worlds: randreg pusher↔SI13 (5 each,50,000), standard R4↔SI13 (5 each,100,000);3unseeded controls each resident/convention; IDs1–5 or1–3; soupseed9100+ID+37*resident_index+300*randreg;L16/P32,grid/pairs/budget/k4 asDZ,13-byteSI tail random/cell; raw-PCG randreg hooks", "C4b closer>50% and most heritable confined at50k in≥4/5; reverse pusher<5% in≥4/5. C4c no direction registered. Motif shares per registered definition +64random-cell culture/traces under own convention. `REVISION_PREREG.md` §R4:423–432; outcomes494–513", "`r4_analyse.py --machine c1`; paired with original journal motif shares. Preserve legacy `len(absolute_path)` RNG salt with `--seed-path-root`. C4c descriptive—do not invent a winner verdict", "Literal SI13 motif>50% not met0/5; confinement component met5/5; reverse pusher suppression met5/5. Published interpretation uses diverse closed descendants (not motif majority). C4c neither named design persists, all populations closed; no direction/winner registered. See published_r4_endpoints.csv.", f"{ARCHIVE}/r4_inv/r4_inv → $R/r4_inv"],
        ["Stage F — ED2c ring32_L8", "`algocell_exp.run.run` via `batch.run_to_dir`, `modal_app.py`, filtered `conds/stageF.json`", "Ten seeds101–110;300,000;L8/P32 paddedring;none@ring32;standardplain;160×125/8192pairs/128instructions/k4; original sampling and snapshots", "F4c impaired pusher: heritable<10/10 OR KMmedian>2000. Registered in `PLAN.md:260–277` (F4c274–275), outcome397; **no F criteria in REVISION_PREREG**; I1 scope1213–1247 identifies study context", "`assay_batch.py --machine tier2c1_assay`, then `stage_f.py --f4c-only` using unchanged `cell` KM/Wilson/table scoring", "F4c met:2/10 heritable; KMmedian not reached. Tests only F4c, not unrelated F1–F4b.", f"{ARCHIVE}/stageF → $R/stageF"],
    ]
    lines = ["# I1-T2 inventory (generated by tier2_prepare.py)", "", "Baseline: `feat/experiments-runner` at `0970b82`. No registered tier-2 run has been launched. All paths below are relative to experiments unless absolute. $R is the future C1 runs root; $O a separate C1 scoring output root; $PY is `/Users/neo/repos/algocell/experiments/.venv/bin/python`. Scoring GPU work belongs on Modal.", "",
             "All soup worlds use the original lattice partner/collision/mutation kernels through ZSoup. All suppression masks are empty except the registered 8080 assay-panel controls. Each soup uses its impact `imp_*_c1` machine; assays use the matching mapped machine in worlds_tier2.json. Core/world/WGSL/manifest hashes are in machines.csv; original/prepared scoring-script hashes in scoring_sources.json.", "",
             "| Experiment / claim | Original runner | Frozen design | Registered criteria and references | Frozen scoring | Published verdicts | Runs directory |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Scoring commands", "", "Run from experiments/, set $R/$O to C1-only scratch/download prefixes. Do not mix C0 archives with C1-world outputs. Fixed genotype/pool controls remain the originally registered inputs; their encounters are rerun entirely with C1.", ""]
    for name, command in COMMANDS.items():
        lines += [f"### {name}", "", "```sh", command, "```", ""]
    lines += ["## Scope and registration decisions", "",
              "- Exclude all flagged Stage A and Stage C arms and random corpora. Also exclude Stage E musweep/x32k2 exploratory flags. **Additional claim-bearing flag:** `stageE/none@nominal_L10_st128_k4@emergence` supports ED2b (C1 CP upper0.000102). It is distinct from copy-offset experiment E. Recommend an optional ten-world original nominal-L10 addendum; do not quietly include it in the requested scope. It has descriptive emergence evidence, no new directional test to invent.",
              "- Include DZ3: ED8d's transmitter balance and its registered partly-met verdict would otherwise go unchecked. E-L32L and L32 benign are unflagged C1 controls but the full design keeps their registered comparisons intact.",
              "- Keep both Stage L lengths to preserve its joint kill; taking L20 alone cannot score the registered both-length kill. Full serial/branching panels preserve their multi-genotype verdicts; do not replace their registered Stage L tapes with tapes selected after C1 outcomes.",
              "- Preserve the frozen overshoot: E's final .npy is one extra iteration after its horizon log (100050 or300500); R4 final .npy is50250/100250. Endpoint motif/share counts use the horizon journal; frozen final-cell assays use the saved .npy as originally published. DZ and the generic batch runner stop at the exact horizon.",
              "- Frozen scorer/report gaps require explicit registration wording: DZ's report does not emit the KM reversal decision; E's printed E2 count omits 'below start' (use its start/end CSV columns for the registered conjunction); E-L32L needs its own >.5/<.4 endpoint threshold; R4 has no binary scoring script and its literal motif majority differs from the descendant-based prose. Stage L's L2 uses a ≥half-unexecuted scoring threshold while the text says the transmissible sites are unexecuted. Record the original script output AND the literal registered components; do not repair thresholds after C1 data.",
              "- Assay seeds: DZ's frozen crc32(variant:call_number) stream is retained; R4's path-length RNG salt must be the original absolute path, not a shorter Modal/download prefix. Frozen scoring sources and panel tables are hashed for registration."]
    (OUT / "INVENTORY.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    jobs = design()
    (OUT / "DESIGN_FULL.json").write_text(json.dumps(jobs, indent=2) + "\n")
    (OUT / "scoring_sources.json").write_text(json.dumps(frozen_manifest(), indent=2) + "\n")
    import serial_retention as S
    panel = [{**p, "tape": p["tape"].tolist(), "sup": list(p["sup"])} for p in S.panel()]
    frozen_inputs = {}
    for path in ("results/stageG/stageG/stage_g_runs.csv", "results/stageI/stageI/stage_g_runs.csv", "results/capacity_time_L/capacity_over_time.csv"):
        frozen_inputs[path] = hashlib.sha256((HERE / path).read_bytes()).hexdigest()
    import pandas as pd
    g = pd.read_csv(HERE / "results/stageG/stageG/stage_g_runs.csv")
    pools = []
    for row in g[g.L.isin([16, 64])].itertuples():
        for step in (5000, 10000):
            p = Path(ARCHIVE) / "stageG" / f"{row.label}_L{row.L}_st128_k4_s{row.seed}.soup_t{step}.u8.br"
            pools.append({"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    assert len(pools) == 80
    (OUT / "control_inputs.json").write_text(json.dumps({"panel": panel, "source_tables": frozen_inputs, "K_pool_files": pools}, indent=2) + "\n")
    write_machine_hashes()
    inventory()
    totals = cost(jobs)
    print(f"Generated {len(jobs)} worlds ({dict(Counter(j['experiment'] for j in jobs))}); cost envelopes: {totals}")


if __name__ == "__main__":
    main()
