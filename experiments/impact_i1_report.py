"""Generate the registered I1 report exclusively from checked worker/table data."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import impact_confirm as C
import impact_study as S
from algocell_exp import impact as I
from algocell_exp.impact_assays import identity
from algocell_exp.impact_worlds import CORES, WORLDS

# Claims are traced to the actual current source tables, with plot filters stated.
CLAIMS = {
    "stageA": ("Exploratory budget/mutation pilot; Methods experiment inventory (MAIN:111). No headline figure directly plots Stage A.", ["review_r2/RUN_TABLE.md"], ["I1"]),
    "stageB": ("Exploratory length scan; Methods experiment inventory (MAIN:111). No headline figure directly plots Stage B.", ["stageB/", "review_r2/RUN_TABLE.md"], ["I1"]),
    "stageC": ("Exploratory ablation study; Methods inventory (MAIN:111). Extended Data Fig. 2a and the immediate-load/stack-writer emergence claim (MAIN:23,135) use only L16, st128, k4 arms; other arms are outside that plot.", ["stageC/"], ["I1"]),
    "stageD": ("Confirmatory L9 ablation comparison; removing stack writers raises emergence (Extended Data Fig. 2d; MAIN:135).", ["stageD/"], ["I1"]),
    "stageE": ("Exploratory size scan; Fig. 1b emergence timing and Extended Data Fig. 2b short-tape emergence/pusher comparison use nominal st128 k4 arms (MAIN:119,135); other arms are outside those plots.", ["stageE/"], ["I1"]),
    "stageF": ("Ring geometry changes emergence, Extended Data Fig. 2c (MAIN:135); MAIN:111 confirmatory inventory.", ["stageF/"], ["I1"]),
    "stageG": ("Open pusher first, closure and partner dependence; Figs 2–4, Extended Data Table 1, Supplementary Table 1, X1/X2 (MAIN:23–31,119–125,151).", ["stageG/", "exectrace/", "biology/individuality/"], ["X1", "X2", "R2", "I1"]),
    "stageH": ("Well-mixed open-first/closure and kin context; Supplementary Table 1 and Supplementary Methods Controls (MAIN:23,31,105,262).", ["stageH/"], ["I1"]),
    "stageI": ("Lethal-tar first block copiers, delayed emergence/closure and inherited variation; Extended Data Fig. 9, Fig. 4, Supplementary Table 1 (MAIN:41,149).", ["stageI/", "review_r2/STATS"], ["R3", "I1"]),
    "stageK": ("Aligned L32 closure in 12/20; Extended Data Fig. 4, Figs 3/4 and robustness control (MAIN:31,99,139,246).", ["stageK/", "ROBUST_REF22"], ["K —", "R3", "I1"]),
    "stageL": ("Closure after ten million steps and transient transmitters; Extended Data Fig. 4, Fig. 4 and serial/branching transmitter controls (MAIN:31,103,139).", ["stageL/", "serial_retention/", "branching/"], ["L —", "X1", "I1"]),
    "stageM": ("8080 subset open first, L16 closure and absence of L32 closure; Extended Data Fig. 4, Supplementary Table 1 (MAIN:23,33,101,139).", ["stageM/"], ["M — the 8080", "I1"]),
    "ring_sweep": ("Exploratory ring sweep, excluded from I1 decision. I1 calls this Extended Data Fig. 3 support; the current NUMBERS_INDEX maps Fig. 3 to exectrace and the plotted ring-emergence panel to Stage F (Extended Data Fig. 2c), so no direct plotted claim can be assigned to this sweep.", ["stageF/", "exectrace/"], ["I1"]),
    "lod_v6": ("Exploratory founding genealogies: founders written through non-copiers rather than point mutation; Extended Data Fig. 7 and first-set values in MAIN:37; registered X3 open-first lineages.", ["results/lod/"], ["N3", "X3", "I1"]),
    "N6": ("Confirmatory founding genealogy replication and open-first lineages; MAIN:37, Supplementary Table 3; N6-1/2/3/5′ met, N6-4 failed; X3 replication.", ["lod_n6/"], ["N6", "X3", "I1"]),
    "X-inputs": ("Stage G partner pools for branching transfer X1 (Supplementary Table 2) and in-soup inflow X2 (MAIN:25,89,93); t10000 L16 also supports the X2 sensitivity analysis.", ["branching/", "inflow_insoup/"], ["X1", "X2", "I1"]),
    "DZ": ("Lethality-dial openness, closure timing and transmitter balance; Extended Data Fig. 8b–d and Supplementary Methods Lethality dial (MAIN:147,244), registered DZ predictions.", ["dial/"], ["DZ", "I1"]),
    "C": ("Harness conventions: open-first with random registers or stack; MAIN:23, Supplementary Methods Harness conventions (MAIN:232), registered R3/C.", ["conv/"], ["R3", "Outcome of C", "I1"]),
    "C4a": ("Closure under random registers in 8/40 worlds by three million steps; Extended Data Fig. 6g and MAIN:33; first C4a set.", ["r4/"], ["C4a", "I1"]),
    "C4a-rep": ("Closure under random registers in 8/40 worlds by three million steps; Extended Data Fig. 6g and MAIN:33; C4a replication set.", ["r4/"], ["C4a", "I1"]),
    "E": ("Copy-offset regenerator/transmitter switch depends on benign/lethal tar and mutation; Extended Data Fig. 6a–f, MAIN:43,143 and Supplementary Methods (MAIN:236); registered E1/E2 predictions failed; final soups also support encounter/flow follow-ups N1–N4.", ["offset/", "n2/", "toxin/"], ["E — the copy-offset", "N1", "N2", "N4", "I1"]),
    "R4": ("Self-initialising closer seeded into random-register pusher worlds displaces it (MAIN:33); Supplementary Methods Harness conventions; registered C4b/C4c invasion outcomes.", ["r4/"], ["C4b", "C4c", "I1"]),
    "R2": ("Constructed 8080 closer invasion and descendant transmission; Extended Data Fig. 5b,c, MAIN:33,141,254,258; A1 subset and A2 full-Z80 controls.", ["invasion_closer/"], ["R2 —", "I1"]),
    "N5": ("Registered N5 accessibility/competition invasion test; reported among revision-three tests (MAIN:268), no directly identified headline figure in NUMBERS_INDEX.", ["r4/", "invasion/"], ["N5", "I1"]),
    "random": ("Emulator accuracy/ring-difftest coverage, Methods Soup (MAIN:73); fresh random control corpus registered in I1, not an evolved-soup claim.", ["emulator/RING_DIFFTEST"], ["I1"]),
}


def claim(c):
    family = c["family"]
    if family.startswith("N3-"):
        desc, paths, sections = ("Earlier genealogy-recorder controls, superseded by lod_v6/N6; no quantitative headline claim directly uses these final soups.", ["lod/"], ["N3", "I1"])
    else:
        desc, paths, sections = CLAIMS[family]
    index = (S.HERE / "manuscript/NUMBERS_INDEX.md").read_text().splitlines()
    prereg = (S.HERE / "REVISION_PREREG.md").read_text().splitlines()
    refs = [f"manuscript/NUMBERS_INDEX.md:{i}" for i, ln in enumerate(index, 1) if any(p in ln for p in paths)]
    regs = [f"REVISION_PREREG.md:{i}" for i, ln in enumerate(prereg, 1) if ln.startswith("#") and any(p in ln for p in sections)]
    assert refs and regs, family
    if family in ("stageC", "stageE") and ("_st128_k4" not in c["cid"]):
        desc += " THIS CONDITION is outside the plotted st128/k4 arms; the flag applies to its exploratory inventory, not to the filtered figure estimates."
    return dict(claims=desc, numbers_index="; ".join(refs), registration="; ".join(regs))


def decision(hi):
    return "unaffected" if hi < 1e-4 else "flagged"


def rates_and_attributes(rows, pairs):
    counts = Counter((p["cid"], p["core"], p["decisive_class"]) for p in pairs)
    rates, attrs, flags = [], [], []
    for r in rows:
        n = int(r["pairs"])
        if n != C.N:
            raise ValueError("a confirmatory condition must contain exactly 100,000 pairs")
        scope = r["family"] != "ring_sweep"
        for core in CORES:
            k = 0 if core == "c0" else int(r[f"aff_{core}"])
            lo, hi = I.clopper_pearson(k, n)
            dec = "reference" if core == "c0" else decision(hi)
            rr = dict(cid=r["cid"], family=r["family"], core=core.upper(), world=r["world"],
                      pairs=n, affected=k, rate=k / n, confidence=.95, cp_lo=lo, cp_hi=hi,
                      interval_method="Clopper–Pearson", decision=dec, decision_scope=int(scope),
                      reference="C0 final pair memory")
            rates.append(rr)
            if core == "c0":
                continue
            if sum(int(r[f"{core}_{cls}"]) for cls in I.CLASSES) != k:
                raise ValueError("attribution summary total mismatch")
            for cls in I.CLASSES:
                value = int(r[f"{core}_{cls}"])
                if counts[(r["cid"], core, cls)] != value:
                    raise ValueError("per-pair attribution evidence mismatch")
                attrs.append(dict(cid=r["cid"], core=core.upper(), **{"class": cls}, affected_pairs=value))
            if dec == "flagged":
                flags.append({**rr, **claim(r)})
    return rates, attrs, flags


def report(out=C.OUT):
    m = json.loads((out / "input_manifest.json").read_text())
    remote = out / "remote"
    meta = json.loads((remote / "meta.json").read_text())
    if meta["code_sha256"] != m["code_sha256"]:
        raise ValueError("completed worker source mismatch")
    records = [json.loads(p.read_text()) for p in sorted((remote / "chunks").glob("*.json"))]
    expected = {sp["key"]: sp for sp in m["conditions"]}
    for rec in records:
        sp = expected[rec["key"]]
        if any(rec["row"][key] != sp[key] for key in ("draw_sha256", "pairs_sha256")) or rec["row"]["cid"] != sp["condition"]["cid"]:
            raise ValueError("executed input hash mismatch")
    if meta["status"] == "complete" and {r["key"] for r in records} != set(expected):
        raise ValueError("incomplete conditions")
    rows = [r["row"] for r in records]
    pairs = [p for r in records for p in r["pairs"]]
    S.write_csv(out / "conditions.csv", rows)
    S.write_csv(out / "affected_pairs.csv", pairs)
    rates, attrs, flags = rates_and_attributes(rows, pairs)
    S.write_csv(out / "rates.csv", rates)
    S.write_csv(out / "decisions.csv", [r for r in rates if r["core"] != "C0"])
    S.write_csv(out / "flagged_claims.csv", flags)
    S.write_csv(out / "attribution.csv", attrs)
    totals = []
    for core in ("C1", "C2"):
        r = dict(core=core, affected=sum(int(a["affected_pairs"]) for a in attrs if a["core"] == core))
        for cls in I.CLASSES:
            r[cls] = sum(int(a["affected_pairs"]) for a in attrs if a["core"] == core and a["class"] == cls)
        totals.append(r)
    S.write_csv(out / "attribution_totals.csv", totals)
    classes = [dict(code=code, meaning=meaning) for code, meaning in zip(I.CLASSES, (
        "wasted DD/FD prefix + LD A,R", "BIT n,(HL) X/Y", "SCF/CCF (Zilog NMOS Q latch)",
        "repeating block instructions, including I/O repeat H and P/V", "other (stop the study)"))]
    S.write_csv(out / "attribution_classes.csv", classes)
    S.write_csv(out / "c1_c2_differ.csv", [dict(cid=r["cid"], pairs=r["pairs"], c1_c2_differ=r["c1_c2_differ"],
                 rate=r["c1_c2_differ"] / r["pairs"]) for r in rows])
    items = {e["aid"]: e for e in m["assays"]}
    assays = [json.loads(p.read_text()) for p in sorted((remote / "assays").glob("*.json"))]
    if {a["aid"] for a in assays} != set(items):
        raise ValueError("incomplete assays")
    ai, scores = [], []
    for a in assays:
        checks = identity(a["outputs"])
        if any(checks[k] != a[k] for k in checks):
            raise ValueError("assay identity mismatch")
        item = items[a["aid"]]
        ai.append(dict(aid=a["aid"], world=item["world"], L=item["L"],
                       c1_identical=int(checks["c1_identical"]), c2_identical=int(checks["c2_identical"]),
                       all_identical=int(checks["all_identical"]),
                       differing_columns=json.dumps(checks["differing_columns"]),
                       sources=json.dumps(item["sources"])))
        for core in CORES:
            scores.append(dict(aid=a["aid"], core=core.upper(), **a["outputs"][core]))
    S.write_csv(out / "assay_identity.csv", ai)
    S.write_csv(out / "assay_scores.csv", scores)
    families = []
    for family in sorted({r["family"] for r in rows}):
        rr = [r for r in rows if r["family"] == family]
        families.append(dict(family=family, conditions=len(rr), pairs=sum(r["pairs"] for r in rr),
                         c1_affected=sum(r["aff_c1"] for r in rr), c2_affected=sum(r["aff_c2"] for r in rr),
                         c1_flagged=sum(r["core"] == "C1" and r["family"] == family for r in flags),
                         c2_flagged=sum(r["core"] == "C2" and r["family"] == family for r in flags),
                         c1_c2_differ=sum(r["c1_c2_differ"] for r in rr)))
    S.write_csv(out / "families.csv", families)
    usage = json.loads((remote / "executor_usage.json").read_text())
    if usage["shader"] or sum(usage["machine"].values()) != len(items) * 3 * 3 * 64:
        raise ValueError("assay executor routing or pair count mismatch")
    S.write_csv(out / "executor_usage.csv", [dict(backend="machine", machine=k, pairs=v) for k, v in usage["machine"].items()])
    core_rows = []
    for w in WORLDS.values():
        for core in CORES:
            man = json.loads((S.HERE / "algocell_exp/machines" / f"{w.machine(core)}.manifest.json").read_text())
            core_rows.append(dict(world=w.tag, core=core.upper(), machine=w.machine(core), variant=man["core"]["variant"],
                                  core_sha256=man["sha256"]["core"], world_sha256=man["sha256"]["world"], wgsl_sha256=man["sha256"]["wgsl"]))
    S.write_csv(out / "cores.csv", core_rows)
    other = sum(t["other"] for t in totals)
    stopped_other = "class 'other'" in meta.get("error", "") or "other attribution" in meta.get("error", "")
    outcomes = []
    for core, col in (("C1", "c1_identical"), ("C2", "c2_identical")):
        scope_flags = [r for r in flags if r["core"] == core and r["decision_scope"]]
        scope_rates = [r for r in rates if r["core"] == core and r["decision_scope"]]
        assay_differences = sum(not a[col] for a in ai)
        verdict = ("STOPPED: other attribution" if stopped_other or other else
                   "incomplete" if meta["status"] != "complete" else
                   "all registered conditions unaffected and assays identical" if not scope_flags and not assay_differences else
                   "flagged conditions and/or non-identical assays; all-unaffected criterion not met")
        outcomes.append(dict(core=core, decision_conditions=len(scope_rates), unaffected=sum(r["decision"] == "unaffected" for r in scope_rates),
                             flagged=len(scope_flags), exploratory_flags=sum(r["core"] == core and not r["decision_scope"] for r in flags),
                             assay_combinations=len(ai), assay_identical=len(ai) - assay_differences,
                             assay_differences=assay_differences, outcome=verdict))
    S.write_csv(out / "outcome.csv", outcomes)
    limits = [
        dict(limit="stochastic-world streams", statement="Dial and randreg/randsp comparisons use executor case-index streams, not archived encounter streams; no soup-trajectory equivalence is claimed.", source="REVISION_PREREG.md I1; impact_dryrun/proof_summary.csv"),
        dict(limit="sim_* lockstep coverage", statement="Single-pair C0 executor proofs cover all impact worlds. Direct sim_* lockstep coverage is plain/lethal L16/L32 only; the other soup-generating worlds lack that direct proof.", source="impact_dryrun/README.md; results/zsoup/lockstep.md"),
        dict(limit="sampling and decisions", statement="Final and labelled-emergence snapshots and frozen X input pools; scheduled intermediates excluded. Per-condition CP upper < 1e-4, no multiplicity correction; exploratory ring sweep excluded from decision.", source="REVISION_PREREG.md I1"),
        dict(limit="assay scope", statement="Complete assay.assay score dictionaries for individually named Z80 genotypes and cited genotype-table entries, n=64, seed=0, budget=128; anonymous population/offset samples are aggregate claims, not named tapes. BFF is a separate ISA. Unspecified payloads use an explicitly labelled seed-0 tail; source-fixed payloads are retained. Same undefined (NaN) scores count as identical, without numeric tolerance.", source="assay_tapes.csv; assay_scores.csv; impact_assays.py"),
        dict(limit="attribution", statement="Latest propagating event in the frozen repaired replay, F/Q repaired coherently, exact class checks and first-probe reproduction. Every affected pair is attributed; no unaffected-pair census in the confirmatory run. An other event immediately stops execution.", source="REVISION_PREREG.md I1; impact.py at bdcdcb6"),
        dict(limit="interpretation", statement="Affected encounter memory or assay changes flag dependence on core semantics; this study does not determine whether evolved-soup headline outcomes change. C2 is a vendor variant. Second-tier replication requires explicit spend approval.", source="REVISION_PREREG.md I1"),
    ]
    S.write_csv(out / "limits.csv", limits)
    billed = None
    if (out / "billing.json").exists():
        billing = json.loads((out / "billing.json").read_text())
        call = json.loads((out / "modal_call.json").read_text())
        if any(r["object_id"] != call["app_id"] for r in billing):
            raise ValueError("billing includes another app")
        billed = float(sum((Decimal(r["cost"]) for r in billing), Decimal(0)))
    costs = [dict(metric="estimated_total_usd", value=m["cost_estimate"]["estimate_usd"], basis="Dry-run upper projection plus assay/startup/storage/egress allowances"),
             dict(metric="allocation_ceiling_usd", value=m["cost_estimate"]["allocation_ceiling_usd"], basis="Timeout at L40S rate plus reserve"),
             dict(metric="modeled_worker_cost_usd", value=meta["modeled_cost_usd"], basis="Measured worker runtime times allocated resource rate; excludes startup/idle"),
             dict(metric="actual_modal_app_cost_usd", value=billed if billed is not None else "pending", basis="Modal billing report, this app ID only; billed GPU+CPU+memory"),
             dict(metric="cap_usd", value=8, basis="User-authorized ceiling")]
    S.write_csv(out / "cost.csv", costs)
    metrics = dict(status=meta["status"], conditions=len(rows), pairs_per_condition=C.N,
                   total_pairs_per_core=sum(r["pairs"] for r in rows), cores=3, cp_confidence=.95,
                   c1_c2_differ=sum(r["c1_c2_differ"] for r in rows), other_attributions=other,
                   stopped_on_other=int(stopped_other), gpu=meta["gpu"], worker_wall_s=meta["wall_s"],
                   registered_stream=m["reg_stream"], dry_stream=m["dry_stream"],
                   exclusion_overlap=sum(int(r["overlap"]) for r in S._read_csv(out / "exclusion.csv")),
                   assay_score_columns=len(assays[0]["outputs"]["c0"]), assay_combinations=len(ai))
    S.write_csv(out / "metrics.csv", [dict(metric=k, value=v) for k, v in metrics.items()])
    S.write_csv(out / "run.csv", [{**meta, "error": meta.get("error", "")}])
    M = S.md_table
    c1flags = [r for r in flags if r["core"] == "C1" and r["decision_scope"]]
    c2flags = [r for r in flags if r["core"] == "C2" and r["decision_scope"]]
    nonidentical = [a for a in ai if not a["all_identical"]]
    readme = f'''# I1 — registered emulator impact study

Generated by `impact_i1_report.py`; numerical statements and outcomes below quote the generated CSV tables. The frozen registration and draft are unchanged. `execution_source.json`, `input_manifest.json`, `archive_hashes.csv`, and `cores.csv` bind the executed source, sampled inputs, archives and three cores. Modal persists one checkpoint per assay combination and condition.

From `metrics.csv`: status **{metrics['status']}**; **{metrics['conditions']:,} conditions × {metrics['pairs_per_condition']:,} pairs**, **{metrics['total_pairs_per_core']:,} pairs per core**, on **{metrics['gpu']}**. Registered stream {metrics['registered_stream']} excludes dry stream {metrics['dry_stream']}: `exclusion.csv` records **{metrics['exclusion_overlap']} overlapping draws**. Executed pair and draw digests are in `conditions.csv`. Draws were generated once per entire condition; executor lane indexing was preserved across dispatch blocks.

`rates.csv` gives affected final-pair-memory counts and rates for every condition and all three cores with **{100 * metrics['cp_confidence']:.0f}% Clopper–Pearson intervals**. C0 is the deterministic self-reference; its zero counts also receive a CP interval for tabular consistency. `decisions.csv` applies the registered CP-upper rule; `flagged_claims.csv` names every flagged condition and its manuscript/registration references. The exploratory ring sweep is tabulated outside the paper decision.

{M(outcomes, ['core', 'decision_conditions', 'unaffected', 'flagged', 'exploratory_flags', 'assay_identical', 'assay_differences', 'outcome'])}

C1 decision-scope flags, quoted from `flagged_claims.csv`:

{M(c1flags, ['cid', 'affected', 'rate', 'cp_lo', 'cp_hi', 'claims']) if c1flags else 'None.'}

C2 decision-scope flags, quoted from the same table (the vendor-specific Q latch is reported separately):

{M(c2flags, ['cid', 'affected', 'rate', 'cp_hi', 'claims']) if c2flags else 'None.'}

Attribution totals (`attribution_totals.csv`), with condition/class counts in `attribution.csv` and individual evidence in `affected_pairs.csv`:

{M(totals, ['core', 'affected', *I.CLASSES])}

Class key (`attribution_classes.csv`):

{M(classes, ['code', 'meaning'])}

`metrics.csv` records **{metrics['other_attributions']} other attributions**, **{metrics['stopped_on_other']} stopped-on-other events**, and **{metrics['c1_c2_differ']:,} C1-versus-C2 differing pairs**. `c1_c2_differ.csv` lists those counts per condition.

`assay_tapes.csv` is the generated source catalogue; `assay_identity.csv` compares all **{metrics['assay_score_columns']} output columns** for **{metrics['assay_combinations']} tape/world/mask combinations**. `assay_scores.csv` contains every raw score without rounding. `executor_usage.csv` verifies the imp_* backend. Non-identical combinations, quoted from `assay_identity.csv`:

{M(nonidentical, ['aid', 'world', 'c1_identical', 'c2_identical', 'differing_columns', 'sources']) if nonidentical else 'None.'}

Cost (`cost.csv`; actual app charges from `billing.json`, distinct from runtime modelling):

{M(costs, ['metric', 'value', 'basis'])}

Limits (`limits.csv`):

{M(limits, ['limit', 'statement', 'source'])}
'''
    (out / "README.md").write_text(readme)
    print(M(outcomes, list(outcomes[0])))
    print("Generated", out / "README.md")


if __name__ == "__main__":
    report()
