"""Generate auditable tables and a short README for the completed Modal dry run (CPU only)."""
from __future__ import annotations

import json
import math
import shutil
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import impact_study as S
from algocell_exp import impact as I
from algocell_exp.impact_worlds import CORES, WORLDS


def projection(rows, bench, n_reg, rate):
    """Affected union and its CP upper bound; measured batch cost, with rule-specific benchmarks.

    Plain ring sizes interpolate in L. Budgets scale linearly, with measured
    plain budget arms used directly where available. This is a planning model,
    not a confidence interval for time; the upper column uses per-condition
    binomial upper bounds and is not simultaneous over conditions.
    """
    out = []
    for r in rows:
        w = WORLDS[r["world"]]
        same_rule = [b for b in bench if b.get("rule", "plain") == w.rule]
        if not same_rule:
            raise ValueError(f"missing benchmark for {w.rule}")
        exact = next((b for b in same_rule if b["world"] == w.tag and int(b["budget"]) == int(r["budget"])), None)
        if exact:
            ex_per_pair = float(exact["t_exec_s"]) / int(exact["pairs"])
            attr_per_batch = float(exact["t_attr_batch_s"])
        else:
            base = [b for b in same_rule if int(b["budget"]) == 128]
            ex = {int(b["L"]): float(b["t_exec_s"]) / int(b["pairs"]) for b in base}
            at = {int(b["L"]): float(b["t_attr_batch_s"]) for b in base}
            ex_per_pair = S._interp(ex, w.L) * int(r["budget"]) / 128
            attr_per_batch = S._interp(at, w.L) * int(r["budget"]) / 128
        n, k = int(r["pairs"]), int(r["aff_union"])
        _, upper = I.clopper_pearson(k, n)
        expected = n_reg * k / n
        upper_expected = n_reg * upper
        batches = math.ceil(expected / S.ATTR_BATCH)
        upper_batches = math.ceil(upper_expected / S.ATTR_BATCH)
        # _initial groups selected lane indices into contiguous runs. The census
        # benchmark has one reset per core/batch, whereas sparse affected lanes
        # need extra resets. Approximate their latency with this condition's
        # measured three-core execution of a whole dry batch (a deliberately
        # generous proxy for three single-lane, budget-zero reset dispatches).
        groups = max(batches, expected * (1 - k / n) + batches * k / n)
        reset = (groups - batches) * float(r["t_exec_s"])
        reset_upper = max(0, math.ceil(upper_expected) - upper_batches) * float(r["t_exec_s"])
        # CPU preparation must draw/load the registered-scale inputs: conservatively
        # scale the whole dry preparation time, and count observed compilation too.
        prep = float(r["t_draw_s"]) * n_reg / n
        compile_s = float(r["t_compile_s"])
        execution = n_reg * ex_per_pair
        attr = batches * attr_per_batch
        attr_upper = upper_batches * attr_per_batch
        seconds = prep + compile_s + execution + attr + reset
        upper_seconds = prep + compile_s + execution + attr_upper + reset_upper
        out.append({"cid": r["cid"], "tier": r["tier"], "world": w.tag, "rule": w.rule,
                    "registered_pairs": n_reg, "union_affected": k, "union_rate": k / n,
                    "union_cp_hi": upper, "expected_affected": expected, "upper_expected_affected": upper_expected,
                    "attr_batches": batches, "upper_attr_batches": upper_batches,
                    "reset_s": reset, "upper_reset_s": reset_upper,
                    "prepare_s": prep, "compile_s": compile_s, "exec_s": execution,
                    "attr_s": attr, "upper_attr_s": attr_upper, "total_s": seconds,
                    "upper_total_s": upper_seconds, "cost_usd": seconds * rate,
                    "upper_cost_usd": upper_seconds * rate})
    return out


def report(out: Path):
    rows = S._read_csv(out / "dryrun_conditions.csv")
    pairs = S._read_csv(out / "dryrun_affected_pairs.csv")
    # Preserve the completed worker's output. Its loop variable overwrote the
    # benchmark start timestamp; the independently timed phases remain valid.
    raw_bench = out / "bench_raw.csv"
    bench = S._read_csv(raw_bench if raw_bench.exists() else out / "bench.csv")
    invalid_bench_wall = sum(float(b.get("wall_s", 0)) < 0 for b in bench)
    if invalid_bench_wall:
        if not raw_bench.exists():
            shutil.copyfile(out / "bench.csv", raw_bench)
        for b in bench:
            b.pop("wall_s", None)
            b["timed_phases_s"] = float(b["t_exec_s"]) + float(b["t_attr_batch_s"])
        S.write_csv(out / "bench.csv", bench)
    ex = S._read_csv(out / "exclusion.csv")
    meta = json.loads((out / "dryrun_meta.json").read_text())
    manifest = json.loads((out / "input_manifest.json").read_text())
    cross = json.loads((out / "cross_remote.json").read_text())
    identity = json.loads((out / "identity.json").read_text())
    done = json.loads((out / "done.json").read_text())
    if meta["status"] != "complete" or not cross["identical"]:
        raise ValueError("refusing a report without a complete run and cross-platform gate")
    if done != meta or any(meta[k] != identity[k] or meta[k] != cross[k]
                           for k in ("code_sha256", "inputs_sha256")):
        raise ValueError("completed job and cross-platform identity mismatch")
    if meta["inputs_sha256"] != manifest["inputs_sha256"]:
        raise ValueError("completed job and input bundle mismatch")
    want_ids = {sp["condition"]["cid"] for sp in manifest["conditions"]}
    if len(rows) != len(want_ids) or {r["cid"] for r in rows} != want_ids or {r["cid"] for r in ex} != want_ids:
        raise ValueError("incomplete condition or exclusion tables")
    counts = Counter((p["cid"], p["core"], p["decisive_class"]) for p in pairs)
    rates, attrs = [], []
    families = defaultdict(list)
    for r in rows:
        families[(r["tier"], r["family"])].append(r)
        n = int(r["pairs"])
        rates.append({"corpus": "random" if r["tier"] == "random" else "archived", "family": r["family"],
                      "cid": r["cid"], "world": r["world"], "core": "C0", "pairs": n,
                      "affected": 0, "rate": 0, "confidence": "", "cp_lo": "", "cp_hi": "",
                      "interval_method": "deterministic reference identity",
                      "exec_all_cores_s": r["t_exec_s"], "attr_all_cores_s": r["t_attr_s"],
                      "wall_all_cores_s": r["t_wall_s"]})
        for core in ("c1", "c2"):
            k = int(r[f"aff_{core}"])
            lo, hi = I.clopper_pearson(k, n)
            if abs(lo - float(r[f"cp_lo_{core}"])) > 1e-12 or abs(hi - float(r[f"cp_hi_{core}"])) > 1e-12:
                raise ValueError("interval mismatch")
            if sum(int(r[f"{core}_{cl}"]) for cl in I.CLASSES) != k:
                raise ValueError("attribution count mismatch")
            for cl in I.CLASSES:
                if counts[(r["cid"], core, cl)] != int(r[f"{core}_{cl}"]):
                    raise ValueError("affected-pair rows mismatch")
                if cl == "other" and (int(r[f"{core}_{cl}"]) or int(r[f"{core}_events_{cl}"])):
                    raise I.AttributionError("STOP: other found in dry-run tables")
            rates.append({"corpus": "random" if r["tier"] == "random" else "archived", "family": r["family"],
                          "cid": r["cid"], "world": r["world"], "core": core.upper(), "pairs": n,
                          "affected": k, "rate": k / n, "confidence": .95, "cp_lo": lo, "cp_hi": hi,
                          "interval_method": "Clopper-Pearson",
                          "exec_all_cores_s": r["t_exec_s"], "attr_all_cores_s": r["t_attr_s"],
                          "wall_all_cores_s": r["t_wall_s"]})
            attrs.append({"cid": r["cid"], "family": r["family"], "core": core.upper(), "affected": k,
                          **{cl: int(r[f"{core}_{cl}"]) for cl in I.CLASSES},
                          **{f"events_{cl}": int(r[f"{core}_events_{cl}"]) for cl in I.CLASSES},
                          "first_not_decisive": int(r[f"{core}_first_not_decisive"]),
                          "first_other_class": int(r[f"{core}_first_other_class"])})
    if sum(int(r[f"aff_{c}"]) for r in rows for c in ("c1", "c2")) != len(pairs):
        raise ValueError("extra or missing affected-pair rows")
    if any(int(r["overlap"]) or not int(r["draw_identical"]) or not int(r["pairs_identical"]) for r in ex):
        raise ValueError("failed exclusion proof")
    S.write_csv(out / "rates.csv", rates)
    S.write_csv(out / "attribution.csv", attrs)
    family_rows = []
    for (tier, family), rs in sorted(families.items()):
        n = sum(int(r["pairs"]) for r in rs)
        f = {"tier": tier, "family": family, "conditions": len(rs), "pairs": n,
             "wall_s": sum(float(r["t_wall_s"]) for r in rs)}
        for c in ("c1", "c2"):
            k = sum(int(r[f"aff_{c}"]) for r in rs)
            lo, hi = I.clopper_pearson(k, n)
            f.update({f"{c}_affected": k, f"{c}_rate": k / n, f"{c}_cp_lo": lo, f"{c}_cp_hi": hi,
                      **{f"{c}_{cl}": sum(int(r[f"{c}_{cl}"]) for r in rs) for cl in I.CLASSES}})
        family_rows.append(f)
    S.write_csv(out / "families.csv", family_rows)
    totals = [{"core": c.upper(), "affected": sum(int(r[f"aff_{c}"]) for r in rows),
               **{cl: sum(int(r[f"{c}_{cl}"]) for r in rows) for cl in I.CLASSES},
               **{f"events_{cl}": sum(int(r[f"{c}_events_{cl}"]) for r in rows) for cl in I.CLASSES},
               "first_not_decisive": sum(int(r[f"{c}_first_not_decisive"]) for r in rows),
               "first_other_class": sum(int(r[f"{c}_first_other_class"]) for r in rows)} for c in ("c1", "c2")]
    S.write_csv(out / "attribution_totals.csv", totals)
    projected = projection(rows, bench, meta["n_registered"], meta["rate_usd_s"])
    S.write_csv(out / "projection_conditions.csv", projected)
    scopes = []
    for scope, keep in (("draft", lambda r: r["tier"] != "added"), ("draft+added", lambda r: True)):
        ps = [r for r in projected if keep(r)]
        scopes.append({"scope": scope, "conditions": len(ps), "pairs": sum(r["registered_pairs"] for r in ps),
                       **{k: sum(r[k] for r in ps) for k in ("prepare_s", "compile_s", "exec_s", "attr_s", "upper_attr_s",
                                                          "reset_s", "upper_reset_s", "total_s", "upper_total_s", "cost_usd", "upper_cost_usd")}})
    S.write_csv(out / "projection.csv", scopes)
    old_eq = S._read_csv(S.RESULTS / "impact_equiv/equiv_worlds.csv")
    old_nc = S._read_csv(S.RESULTS / "impact_equiv/equiv_controls.csv")
    new_eq = S._read_csv(out / "equiv_groups.csv")
    new_nc = S._read_csv(out / "equiv_controls.csv")
    proofs = old_eq + new_eq
    fields = ("mem_mismatch", "wc_mismatch", "reg_mismatch")
    if {r["world"] for r in proofs} != set(WORLDS) or any(int(r[f]) for r in proofs for f in fields):
        raise ValueError("missing or failed C0 equivalence")
    controls = old_nc + new_nc
    if any(not any(int(r[f]) for f in fields) for r in controls):
        raise ValueError("negative control failed")
    proof_summary = [{"scope": "original", "worlds": len(old_eq), "pairs": sum(int(r["pairs"]) for r in old_eq),
                      "mismatches": sum(int(r[f]) for r in old_eq for f in fields), "controls_caught": len(old_nc)},
                     {"scope": "ring_sweep+R2 supplement", "worlds": len({r["world"] for r in new_eq}),
                      "pairs": sum(int(r["pairs"]) for r in new_eq), "mismatches": 0, "controls_caught": len(new_nc)},
                     {"scope": "combined", "worlds": len(WORLDS), "pairs": sum(int(r["pairs"]) for r in proofs),
                      "mismatches": 0, "controls_caught": len(controls)}]
    S.write_csv(out / "proof_summary.csv", proof_summary)
    core_rows = []
    for w in WORLDS.values():
        for c in CORES:
            m = json.loads((S.HERE / "algocell_exp/machines" / f"{w.machine(c)}.manifest.json").read_text())
            core_rows.append({"world": w.tag, "machine": w.machine(c), "core": c.upper(), "version": m["core"]["coreVersion"],
                              "variant": m["core"]["variant"], "core_sha256": m["sha256"]["core"],
                              "world_sha256": m["sha256"]["world"], "wgsl_sha256": m["sha256"]["wgsl"]})
    S.write_csv(out / "cores.csv", core_rows)
    metrics = [{"metric": k, "value": v} for k, v in {
        "conditions": len(rows), "pairs_per_condition": meta["n"], "total_pairs": sum(int(r["pairs"]) for r in rows),
        "registered_pairs_per_condition": meta["n_registered"], "dry_stream": meta["dry_stream"], "reg_stream": meta["reg_stream"],
        "input_bytes": manifest["inputs_bytes"], "cross_bytes": cross["bytes"], "cross_identical": int(cross["identical"]),
        "cross_local_wall_s": cross["local_wall_s"], "cross_remote_wall_s": cross["wall_s"], "job_wall_s": meta["wall_s"],
        "dry_chunks_wall_s": meta["work_s"], "draw_cpu_wall_s": sum(float(r["t_draw_s"]) for r in rows),
        "exec_wall_s": sum(float(r["t_exec_s"]) for r in rows), "attr_wall_s": sum(float(r["t_attr_s"]) for r in rows),
        "exclusion_conditions": len(ex), "exclusion_overlap": sum(int(r["overlap"]) for r in ex),
        "exclusion_redraws": sum(int(r["redrawn"]) for r in ex), "confidence": .95,
        "invalid_raw_benchmark_wall_rows": invalid_bench_wall,
        "estimated_cost_usd": manifest["estimated_total_usd"], "compute_allowance_usd": manifest["max_modeled_usd"],
        "modeled_job_cost_usd": meta["modeled_cost_usd"], "modeled_cross_cost_usd": cross["wall_s"] * cross["rate_usd_s"],
        "modeled_total_cost_usd": meta["modeled_cost_usd"] + cross["wall_s"] * cross["rate_usd_s"],
    }.items()]
    if (out / "billing.json").exists():
        billing = json.loads((out / "billing.json").read_text())
        billed = sum((Decimal(r["cost"]) for r in billing), Decimal(0))
        metrics.append({"metric": "billed_app_cost_usd", "value": float(billed)})
    S.write_csv(out / "metrics.csv", metrics)
    values = {r["metric"]: r["value"] for r in metrics}
    billing_text = (f"The CLI billing report attributes ${values['billed_app_cost_usd']:.3f} to these impact apps (`billing.json`)."
                    if "billed_app_cost_usd" in values else "The billing report has not been fetched yet.")
    benchmark_note = ("The worker's benchmark total-wall timer had a variable collision, fixed after execution. "
                      "`bench_raw.csv` preserves that output; `bench.csv` omits the invalid column and reports the "
                      "independently measured execution and census phases. Those valid phase timings drive the "
                      "projection. Condition and whole-job wall timers were unaffected."
                      if invalid_bench_wall else "")
    table = [{"family": r["family"], "pairs": r["pairs"],
              "C1 affected": f"{r['c1_affected']} ({100 * r['c1_rate']:.4f}%)",
              "C2 affected": f"{r['c2_affected']} ({100 * r['c2_rate']:.4f}%)"} for r in family_rows]
    all_scope = scopes[-1]
    readme = f'''# Emulator impact dry run (unregistered)

Generated by `python impact_study.py report`; every numerical result below comes from the generated CSV tables. The draft is unchanged. Assays remain the `corpus3_assays` hook.

Completed on {meta['adapter']}: {values['conditions']:,} conditions × {values['pairs_per_condition']:,} pairs; {values['total_pairs']:,} pairs through each core. `cores.csv` fixes the machines and hashes; `execution_source.json` binds the completed worker to its exact committed source. All pairs underwent the attribution census. `rates.csv` has affected rates and {100 * values['confidence']:.0f}% Clopper–Pearson intervals for every condition and comparison core; C0 is the deterministic reference. `dryrun_conditions.csv` includes each condition's budget, masks, grid and wall time.

{S.md_table(table, list(table[0]))}
Family rates pool equally sampled conditions for description; the protocol's measure and decision rule are per condition. Family intervals are in `families.csv`.

{S.md_table(totals, ['core', 'affected', *I.CLASSES])}
These are decisive attribution counts. `attribution.csv` and `attribution_totals.csv` also count every census event; `dryrun_affected_pairs.csv` records each affected pair and the first and decisive events. Any “other” stops the job; none occurred.

The Metal/Vulkan cross-platform check was byte-identical across {values['cross_bytes']:,} bytes (records, memories, state, write counts and canonical attribution for the same sampled condition), in {values['cross_local_wall_s']:.2f} s locally and {values['cross_remote_wall_s']:.2f} s remotely. See `cross_local.json`, `cross_remote.json` and `metrics.csv`.

The detached job took {values['job_wall_s'] / 60:.2f} min, including supplemental C0 proofs, dry-run chunks, benchmarks and volume commits. Every chunk was committed to `algocell-atlas-runs`; `modal_call.json` identifies the call and volume path. The estimate was ${values['estimated_cost_usd']:.2f}; measured-runtime compute cost is ${values['modeled_total_cost_usd']:.3f}, against a ${values['compute_allowance_usd']:.2f} compute allowance. The model uses CLI allocation rates and excludes container startup/idle and storage. {billing_text} See `metrics.csv`.

For the full proposed scope, {all_scope['pairs']:,} pairs project to {all_scope['total_s'] / 3600:.2f} h / ${all_scope['cost_usd']:.2f}, or {all_scope['upper_total_s'] / 3600:.2f} h / ${all_scope['upper_cost_usd']:.2f} with per-condition affected-union upper bounds. `projection.csv` separates draft and added scopes; `projection_conditions.csv` gives the inputs and components. This planning model scales preparation conservatively, interpolates measured throughput by length within each rule, scales unmeasured budgets linearly, and charges one measured census batch per expected affected batch. Census benchmarks use uniform random pairs, so affected-archive batch cost remains a proxy. Extra sparse-lane reset dispatches use the dry batch execution time as a generous latency proxy; the upper scenario assumes a separate reset for every selected lane. It excludes assays and upload time; the upper scenario is not a simultaneous time confidence bound.

{benchmark_note}

Exclusion uses dry stream {values['dry_stream']} and registered stream {values['reg_stream']}. Each registered condition's {values['registered_pairs_per_condition']:,} draws were regenerated without execution. Colliding dry draws were redrawn. `exclusion.csv` proves {values['exclusion_overlap']} overlapping draws across {values['exclusion_conditions']} conditions and verifies the transported and executed draw/pair digests against the archive. Random exclusion compares pair bytes; archived exclusion compares (source, cell, partner) slots. Clonal byte pairs in different slots can coincide; exclusion is of sampled encounters.

**C0 proof scope:** `proof_summary.csv` combines the original and supplemental checks: {proof_summary[-1]['worlds']} worlds, {proof_summary[-1]['pairs']:,} pairs, {proof_summary[-1]['mismatches']} mismatches and {proof_summary[-1]['controls_caught']} negative controls caught. These compare against `z80_test_*` single-pair executors, not the `sim_*` shaders that generated the soups. The separate [ZSoup lockstep proof](../zsoup/lockstep.md) covers the `sim_*` side for plain and lethal L16/L32 worlds. Dial, randreg and randsp use the executor's random stream, seeded by case index, rather than the soup encounter stream; their proof is not encounter-stream or soup-trajectory equivalence.

See [PROTOCOL.md](PROTOCOL.md) for ambiguities and proposed registration fixes. The ring sweep is explicitly exploratory evidence for Extended Data Fig. 3; its final and emergence archives enter the added tier. R2 final soups retain their original suppression masks.
'''
    (out / "README.md").write_text(readme)
    (out / "PROTOCOL.md").write_text(PROTOCOL)
    print(f"wrote generated report tables and {out / 'README.md'}")


PROTOCOL = """# Protocol issues and proposed fixes (draft unchanged)

- Registration order: the opening says registration precedes any fixed-core corpus run, but the before-registration list requires a fixed-core dry run. Explicitly allow a reported exploratory dry run on excluded pairs before registration, and freeze the confirmatory corpus and analysis before executing the remaining registered draws.
- Scope: the draft names Stages A–F and selected lineage/X inputs while also saying every registered condition. Freeze the complete registry, including `.npy` final soups, the R2 invasions, later stages and the explicitly exploratory ring sweep supporting Extended Data Fig. 3. Final/emergence snapshots are included; scheduled intermediates are excluded. Decide whether emergence and exploratory conditions enter the decision rule.
- Dry-run size: “1,000 pairs per corpus” is ambiguous. This run interprets it as per condition (including each snapshot condition), which is required to check every world/budget/mask combination. Register this interpretation explicitly.
- Budgets: the blanket 128 conflicts with archived arms up to 2,048. Use each condition's archived budget, masks and rule. The Stage H partner is a uniform other cell, not a lattice neighbour.
- Seeds: “seed 1” does not define draw order or PCG encounter streams. Freeze SeedSequence inputs, condition ids, draw order, run seed and lane indexing. In stochastic worlds this study uses executor case-index streams, not archived soup encounter streams; state this limit.
- Exclusion: freeze 100,000 registered draws per condition. Archive exclusions are source/cell/partner encounters, not all identical clonal byte sequences. Other draw sizes change the seed sequence draw and invalidate the exclusion proof.
- Attribution: a first transient divergence need not cause the memory effect. Use the latest propagating event in the repaired pass, with exact class checks and first-probe reproduction, as implemented. Repair F/Q coherently. startMap is unnecessary when per-step PCs locate instructions; adding maps changes the machine and requires fresh proofs.
- Threshold: specify point estimate versus CP upper bound below 1e-4, which differs at 100,000 pairs (at most 9 versus 3 hits), and the policy for multiplicity across conditions. The dry run cannot settle that threshold.
- Core hashes: hooks alter emitted core text; freeze all core/hook hashes and machine/world hashes in `cores.csv`. Registration has not happened and full-study pairs have not been executed.
- C0 evidence: executor comparisons cover every world, but direct `sim_*` lockstep evidence covers only plain/lethal L16/L32. Generalize lockstep to other worlds if soup-side equivalence is required before registration; stochastic worlds also need a defined encounter-stream bridge.
- Control flow: repeating I/O fixes can change H/PV, which affect DAA and parity conditions; the statement that only X/Y changes matter is too broad. Keep the documented I/O repeat differences in the attribution class.
- Snapshot naming: define emergence by labelled `soup_emergence` archives. The unlabelled lineage snapshots are not interchangeable. X3 has no soup input; X1/X2 use specific Stage G snapshots. The P38 random ring requires L19 and a derived executor, not an archived L19 world.
- C2 inherits C1 fixes. Also report C2-versus-C1 differences (`c1_c2_differ`) to isolate Q behavior.
- Assays stay a hook pending `feat/assays`; the C1 paper decision cannot be applied without assay identity.
"""
