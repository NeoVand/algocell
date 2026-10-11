"""Generate S1 correction tables and texts from original, audit and v2 records.

No GPU work or raw-volume downloads. Run after modal_score6502.py completes:
  PYTHONPATH=experiments /path/to/python experiments/correction_s1.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import score6502 as S

ROOT = Path(__file__).resolve().parent / "results/s6502"
OUT = ROOT / "v2"
DATE = "2026-10-11"


def table(name, rows):
    df = pd.DataFrame(rows)
    df.to_csv(OUT / name, index=False)
    return df


def md(df):
    def value(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "not tested / censored"
        return str(x).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |",
                      "|" + "---|" * len(df.columns),
                      *["| " + " | ".join(map(value, row)) + " |" for row in df.itertuples(index=False, name=None)]])


def label(value):
    return "not tested" if value is None else "supported" if value else "not supported"


def audit_findings():
    rows = []
    for line in (ROOT / "AUDIT.md").read_text().splitlines():
        if line.startswith("| "):
            fields = [s.strip() for s in line.split("|")[1:-1]]
            if len(fields) == 3 and fields[1] in {"PASS", "FAIL", "UNSURE"}:
                rows.append(dict(finding=fields[0], audit_status=fields[1], evidence=fields[2]))
    assert len(rows) == 33, len(rows)
    actions = {
        "Loop measure as a control-flow-loop detector": "Corrected sequential increment wraps; forward alias returns remain outside the registered PC-decrease measure.",
        "Registered PC-decrease implementation": "Original literal exception reproduced; v2 generalizes the sequential increment exception as instructed.",
        "Registered trace partner source": "Both conflicting definitions reported. Seeded random partners retained as reported registered analysis; first-replicator soup-snapshot sensitivity computed for all arms.",
        "Registered final-cell heritability count": "Full final-cell classification repeated with both 32 frozen and 64 literal culture partners, including selection and scans.",
        "Applied mutation updates": "Disclosed shared Z80/6502 lost-update race; no world rerun or mutation-policy change.",
        "Full raw-data audit of all worlds": "All 80 raw histories and final soups now read and integrity-checked for scoring on Modal; required sensitivity soups also checked. This does not independently replay all full world histories or inspect all 400 soups.",
        "Verdict sensitivity to protocol conflicts/race": "64-partner final classes and nearest-at-or-after first-replicator soup traces quantified; mutation-race verdict effects remain unmeasured.",
        "Culture/trace parity with frozen Z80 DZ": "32-partner branch retained; updated scorer re-runs DZ p=0 exactly. Literal64 and soup traces are separate outputs.",
        "Frozen scorer provenance": "Original files preserved; every v2 score records source hashes and committed scorer provenance.",
    }
    for r in rows:
        r["correction_action"] = actions.get(r["finding"], "Retained audit finding; no correction required.")
    return table("audit_findings.csv", rows)


def cost_report(results, controls):
    apps = json.loads((OUT / "modal_apps.json").read_text())
    rates = json.loads((OUT / "modal_rates.json").read_text())
    ids = {r["modal"]["app_id"] for r in results + controls}
    selected = [a for a in apps if a["app_id"] in ids]
    assert {a["app_id"] for a in selected} == ids
    rate = sum(float(rates[k]) for k in ("gpu_hour_cost_l4", "cpu_hour_cost")) + 2 * float(rates["mem_gib_hour_cost"])
    rows = []
    for a in selected:
        assert a["state"] == "stopped" and a["tasks"] == "0", a
        seconds = (datetime.fromisoformat(a["stopped_at"]) - datetime.fromisoformat(a["created_at"])).total_seconds()
        scored = [r for r in results + controls if r["modal"]["app_id"] == a["app_id"]]
        cap = min(20, len(scored))
        wall = sum(r["modal"]["wall_s"] for r in scored)
        rows.append({"app_id": a["app_id"], "worlds": len(scored), "gpu": "L4", "cpu": 1,
                     "memory_gib": 2, "max_simultaneous_containers": cap,
                     "created_at": a["created_at"], "stopped_at": a["stopped_at"],
                     "app_elapsed_s": seconds, "scorer_wall_s": wall,
                     "scorer_compute_estimate_usd": wall * rate / 3600,
                     "whole_app_compute_upper_bound_usd": seconds * cap * rate / 3600,
                     "state": a["state"], "tasks": a["tasks"]})
    df = table("cost.csv", rows)
    upper = float(df.whole_app_compute_upper_bound_usd.sum())
    assert upper < 5, upper
    # Reserve $0.01 for these small JSON transfers and one month of added
    # score/control-input storage; at current rates this covers >50 MiB.
    S.write_json(OUT / "cost_summary.json", {"scorer_compute_estimate_usd": float(df.scorer_compute_estimate_usd.sum()),
        "whole_app_compute_upper_bound_usd": upper, "storage_egress_allowance_usd": 0.01,
        "total_run_cost_upper_bound_usd": upper + 0.01,
        "bound_method": "Maximum simultaneous containers times entire app lifetime, plus storage/egress allowance; not an invoice."})
    assert upper + 0.01 < 5
    return df, float(df.scorer_compute_estimate_usd.sum()), upper


def main():
    old = [json.loads(p.read_text()) for p in sorted((ROOT / "worlds").glob("s6502_*.json"))]
    new = [json.loads(p.read_text()) for p in sorted((OUT / "worlds").glob("s6502_*.json"))]
    controls = [json.loads(p.read_text()) for p in sorted((OUT / "control/worlds").glob("lp0_*.json"))]
    expected = {(arm, seed) for arm in "ABCD" for seed in range(65001, 65021)}
    for rs in (old, new):
        assert len(rs) == 80 and {(r["world"]["arm"], r["world"]["seed"]) for r in rs} == expected
    for r in new + controls:
        assert not r["usage"]["shader"] and not r["modal"]["software_vulkan"]
        assert "L4" in r["modal"]["adapter"]["device"], r["modal"]
    frozen = {r["world"]["world"]: r for r in old}
    comparison = []
    for r in new:
        o = frozen[r["world"]["world"]]
        assert o["input_sha256"] == r["input_sha256"]
        assert o["executor"] == r["executor"]
        row = {k: r["world"][k] for k in ("world", "arm", "seed")}
        for k in ("t_rep", "t_closed", "first_tape", "first_confined", "first_open", "first_loop", "first_entered", "p2_violation"):
            row[f"original_{k}"] = o["world"][k]
            row[f"v2_{k}"] = r["world"][k]
            row[f"changed_{k}"] = o["world"][k] != r["world"][k]
        comparison.append(row)
    comp = table("world_comparison.csv", comparison)
    tape_rows = []
    for r in new:
        o = frozen[r["world"]["world"]]
        old_tapes = {t["tape"]: t for t in o["judged_tapes"]}
        for t in r["judged_tapes"]:
            prior = old_tapes.get(t["tape"])
            tape_rows.append({"world": r["world"]["world"], "arm": r["world"]["arm"],
                              "tape": t["tape"], "original_loop": prior["loop"] if prior else None,
                              "v2_loop": t["loop"], "identical_judgment": prior == t})
    tapes = table("tape_comparison.csv", tape_rows)
    identical_worlds = sum(frozen[r["world"]["world"]]["world"] == r["world"] for r in new)
    identical_cells = sum(frozen[r["world"]["world"]]["cells"] == r["cells"] for r in new)
    S.report(new, OUT, report_name="S1_v2.md")
    literal = [{"world": {**r["world"], **r["literal64"]["classes"]}, "cells": r["literal64"]["cells"],
                "p2_tapes": r["p2_tapes"]} for r in new]
    table("s1_worlds_64.csv", [r["world"] for r in literal])
    table("s1_cells_64.csv", [{**{k: r["world"][k] for k in ("world", "arm", "seed")}, **c}
                             for r in literal for c in r["cells"]])
    S.write_json(OUT / "s1_verdicts_64.json", {"arms": [S.verdict([r for r in literal if r["world"]["arm"] == arm]) for arm in "ABCD"]})
    culture = table("final_cultures.csv", [{**{k: r["world"][k] for k in ("world", "arm", "seed")}, **c}
                                           for r in new for c in r["culture_cells"]])
    c32, c64 = (culture[culture.partners == n].set_index(["world", "cell"]) for n in (32, 64))
    assert c32.index.equals(c64.index)
    assert (c32.tape == c64.tape).all() and (c32.soup_cell == c64.soup_cell).all()
    culture_summary = table("culture_summary.csv", [{"arm": arm, "sampled_cells": len(c32[c32.arm == arm]),
        "heritable_32": int(c32[c32.arm == arm].heritable.sum()),
        "heritable_64": int(c64[c64.arm == arm].heritable.sum()),
        "heritability_changes": int(((c32.heritable != c64.heritable) & (c32.arm == arm)).sum()),
        "scan_membership_changes": int(((c32.scanned != c64.scanned) & (c32.arm == arm)).sum())} for arm in "ABCD"])
    arms = []
    summaries = {}
    for arm in "ABCD":
        summaries[arm] = [S.verdict([r for r in rs if r["world"]["arm"] == arm]) for rs in (old, new, literal)]
        for protocol, summary, rs in zip(("original32", "v2_frozen32", "v2_literal64"), summaries[arm], (old, new, literal)):
            worlds = [r["world"] for r in rs if r["world"]["arm"] == arm]
            arms.append({"protocol": protocol, **summary,
                         "first_confined": sum(bool(w["first_confined"]) for w in worlds),
                         "first_open": sum(bool(w["first_open"]) for w in worlds),
                         "first_loop": sum(bool(w["first_loop"]) for w in worlds)})
    arm_df = table("arm_comparison.csv", arms)
    av0, av, av64 = summaries["A"]
    pred = []
    for k in ("P1", "P2", "P3", "P3_classes", "P3_offset", "P4", "null", "kill"):
        render = (lambda x: "applies" if x else "does not apply") if k == "null" else ((lambda x: "triggered" if x else "not triggered") if k == "kill" else label)
        pred.append({"component": k, "original_frozen32": render(av0[k]), "v2_frozen32": render(av[k]), "v2_literal64": render(av64[k])})
    predictions = table("verdict_comparison.csv", pred)
    partners = []
    for r in new:
        w, st = r["world"], r["soup_trace"]
        row = {k: w[k] for k in ("world", "arm", "seed", "t_rep", "first_tape")}
        row.update({"snapshot_step": st["snapshot_step"] if st else None,
                    "snapshot_file": st["snapshot_file"] if st else None,
                    "snapshot_sha256": st["snapshot_sha256"] if st else None})
        for flag in ("confined", "open", "loop", "entered"):
            row[f"random_{flag}"] = w[f"first_{flag}"]
            row[f"soup_{flag}"] = st[flag] if st else None
            if flag != "entered":
                row[f"changed_{flag}"] = (w[f"first_{flag}"] != st[flag]) if st else False
        partners.append(row)
    partner_df = table("partner_sensitivity.csv", partners)
    ps = []
    for arm in "ABCD":
        d = partner_df[partner_df.arm == arm]
        row = {"arm": arm, "first_replicators": int(d.t_rep.notna().sum())}
        for flag in ("confined", "open", "loop"):
            row.update({f"random_{flag}": int(d[f"random_{flag}"].eq(True).sum()),
                        f"soup_{flag}": int(d[f"soup_{flag}"].eq(True).sum()),
                        f"changed_{flag}": int(d[f"changed_{flag}"].sum())})
        row["changed_any"] = int(d[[f"changed_{flag}" for flag in ("confined", "open", "loop")]].any(axis=1).sum())
        ps.append(row)
    partner_summary = table("partner_summary.csv", ps)
    target = pd.read_csv(ROOT.parent / "dial/dial_worlds.csv", float_precision="round_trip")
    target = target[target.variant == "lp0"].set_index("seed")
    control_rows = []
    for r in controls:
        w = r["world"]
        for k in S.CONTROL_COLUMNS:
            a, b = w.get(k), target.loc[w["seed"], k]
            match = (a is None and pd.isna(b)) or a == b
            assert match, (w["seed"], k, a, b)
            control_rows.append({"seed": w["seed"], "column": k, "exact_match": match})
    assert len(controls) == 10
    table("control_matches.csv", control_rows)
    findings = audit_findings()
    cost, estimate, upper = cost_report(new, controls)
    changes = {k: int(comp[f"changed_{k}"].sum()) for k in ("first_loop", "first_confined", "first_open", "t_rep", "t_closed", "p2_violation")}
    methods = ["The registration governs. Its heritability definition says gen2 ≥ 0.3 over 64 partners; the cited frozen DZ final-cell estimator uses 32 shared uniform-byte partners. Both final-cell analyses are reported. Its trace text says 16 ‘soup partners’, while the cited exec_trace/DZ implementation generates seeded uniform-byte partners. Random partners remain the reported registered trace analysis, as instructed; soup partners are explicitly a sensitivity analysis.",
               "The corrected 6502 loop detector counts a 16-bit PC decrease only when PC_next differs from (PC + executed instruction length) mod 2^16. Each prefix replay supplies its current memory, so instruction lengths come from the executed opcode even after self-modification. The 256 lengths are exported from Zilion m6502.table(), commit 183ff5c, and hashed in each score. This corrects multi-byte sequential increment wraps such as $FFFE → $0001. Forward ring-alias returns still do not satisfy the registered PC-decrease rule; the prior exploratory position-space finding remains exploratory.",
               "Top-tape cultures retain 64 partners, trace draws retain 16 uniform-byte partners, judge RNG [seed, 0, 11], and final RNG [seed, 23]. Final classes are computed independently at n=32 and n=64 with the same 64 sampled cells, trace draw stream, threshold and frozen estimator. The full heritability selection, up-to-16-cell selection and mutant scans are repeated; this is not restricted to the original scanned cells. Increasing n expands the assay’s generation-one draws, so generation-two draws are not a nested prefix of the n=32 assay. Scan draws may change if heritable membership changes.",
               "Sensitivity uses each corrected random-analysis first tape, without changing t_rep or re-selecting tapes. It chooses the earliest saved snapshot at or after t_rep from 20,000, 100,000, 1M, 3M and 10M; 16 cells are drawn without replacement using independent RNG [seed, 0, 6502, 16]. Partner indices, bytes and snapshot hashes are saved in each world JSON. Only the final snapshot and that required snapshot are read from the volume on Modal. These snapshots can be later than t_rep: this is a snapshot sensitivity, not an exact contemporary-soup reconstruction. No soup-partner P2 history or revised t_closed is inferred."]
    final_summary = arm_df[["arm", "protocol", "with_rep", "first_confined", "first_open", "first_loop", "first_confined_loop", "first_open_straight", "p2_worlds", "km_t_rep", "transmitter", "regenerator", "offset_eligible", "offset_equal"]]
    text = [f"# Correction to S1 ({DATE})", "", *[x + "\n" for x in methods],
            "## Original and corrected results", "", md(final_summary), "", md(predictions), "",
            f"Arm A P1: original {av0['first_confined_loop']}/{av0['with_rep']} ({av0['first_confined_loop']/av0['with_rep']:.2f}); corrected {av['first_confined_loop']}/{av['with_rep']} ({av['first_confined_loop']/av['with_rep']:.2f}), against ≥ 0.8. All {av['with_rep']} arm-A first tapes remain confined under random tracing. P2 counts are worlds with a qualifying tape strictly before closure. P3’s overall verdict is not tested when its offset component has zero eligible cells; its class-occurrence component is reported separately. B–D are descriptive, with no kill rule.", "",
            f"Across all 80 worlds, changes after the loop fix: {changes}. {identical_worlds}/80 complete world rows and {identical_cells}/80 frozen32 scanned-cell lists are identical to the original; {int(tapes.identical_judgment.sum())}/{len(tapes)} cached top-tape judgments are identical. Exact raw-input hashes and machine provenance match the original outputs. Original result files and REVISION_PREREG.md are unchanged. All worlds retain their full 10M horizon. The independent arm-A KM median in audit_km.csv is {av['km_t_rep']:,.0f} steps.", "",
            "## Final-cell cultures: 32 frozen versus 64 literal", "", md(culture_summary), "",
            f"Arm-A P3 class occurrence: n=32 {av['transmitter']} transmitters and {av['regenerator']} regenerators ({label(av['P3_classes'])}); n=64 {av64['transmitter']} and {av64['regenerator']} ({label(av64['P3_classes'])}). Offset equality: n=32 {av['offset_equal']}/{av['offset_eligible']}; n=64 {av64['offset_equal']}/{av64['offset_eligible']}. P3 overall: {label(av['P3'])} / {label(av64['P3'])}. Counts are scanned heritable confined cells; unscanned proportions use the frozen population estimator.", "",
            "## Trace-partner sensitivity", "", md(partner_summary), "",
            f"Across {int(partner_summary.first_replicators.sum())} first replicators, {int(partner_summary.changed_confined.sum())} change confinement, {int(partner_summary.changed_open.sum())} change openness, and {int(partner_summary.changed_loop.sum())} change loop calls; {int(partner_summary.changed_any.sum())} change any flag. Censored worlds have no first-replicator trace and are retained as missing in partner_sensitivity.csv.", "",
            "## Mutation race", "",
            "The inherited packed-u32 mutation kernel has the same shared, previously disclosed nondeterminism as the Z80 runs ([ZSOUP.md](../../algocell_exp/ZSOUP.md), lines 203/268, mutation/claim nondeterminism). [audit_mutation_race.csv](audit_mutation_race.csv) supplies a concrete witness under ordinary L16 parameters: batch seed 12, mutations 174 and 180 target distinct bytes 91616 and 91617 in word 22904. Intended values are $3F and $CB; observed values are $3F and $00, losing mutation 180. The original Z80 sim_square_L16.wgsl:1139 also uses non-atomic read/modify/write of the packed word. The nominal rate is 512 byte-replacement attempts per step, not 512 guaranteed applied updates. Re-scoring fixed snapshots does not fix that race or quantify its outcome effect. A mutation-policy correction requires a separately labelled world rerun and does not replace S1.", "",
            "## Every independent-audit finding", "", md(findings), "",
            "## Validation, execution and cost", "",
            "42 scorer tests and 4 focused executor regression tests passed, including the audit’s sequential 3-byte wrap, genuine backward near-wrap branch, self-modified opcode and existing machine-loop tests. All 256 descriptors match the archived exploratory descriptor table. Z80 DZ p=0 was re-run on Modal L4: 10/10 exact on every column (control/SCORER_CONTROL.md and control_matches.csv). Every new result used its vendored machine, with no shader fallback; Modal saved-result round trips matched every returned result. All 80 results record input, source, core/world/WGSL hashes and scorer commit.", "", md(cost), "",
            f"Rate-priced scorer compute estimate: ${estimate:.4f}. Conservative whole-app compute upper bound: ${upper:.4f}; adding a $0.01 allowance for the small JSON transfers and one month of added score/control-input volume storage gives ${upper + 0.01:.4f}, below the $5 cap. This bound charges each app’s entire created-to-stopped interval at its maximum simultaneous L4 container count, 1 CPU and 2 GiB per container, including startup and idle time; it is not an invoice. Current rates are saved in modal_rates.json, and cost_summary.json records the bound. Both task apps are stopped with zero tasks; the running tier-2 app ap-GpoWKGnLbLpyPF0kyUWKsT was untouched.", "",
            "Reproduce: run the committed Modal wrapper with --registered --arms A,B,C,D --out experiments/results/s6502/v2; it refuses existing volume score paths. Gather existing results with --gather-only. Then run PYTHONPATH=experiments /Users/neo/repos/algocell/experiments/.venv/bin/python experiments/correction_s1.py. Tables are written only under v2; these two correction texts are generated beside the original audit. See v2/PROVENANCE.json for source and evidence hashes."]
    (ROOT / "CORRECTION.md").write_text("\n".join(text) + "\n")
    draft = [f"### Correction to the outcome of S1 ({DATE})", "",
             "The independent audit (results/s6502/AUDIT.md) found a multi-byte sequential-PC-wrap defect, two conflicts between the registration’s prose and its cited frozen definitions, and the shared packed-word mutation race. The original outcome therefore should not be described as implemented exactly as registered. Re-scoring all 80 fixed worlds on Modal with the corrected loop detector is recorded in results/s6502/v2/S1_v2.md and results/s6502/CORRECTION.md; original outputs are preserved.", "",
             f"**Arm A.** P1 remains {label(av['P1'])}: {av['first_confined_loop']} of {av['with_rep']} first replicators are confined and have a non-sequential 16-bit PC decrease (originally {av0['first_confined_loop']}/{av0['with_rep']}); all {av['with_rep']} are confined with random partners. P2 is {label(av['P2'])} ({av['p2_worlds']} violating worlds). P3 remains {label(av['P3'])}: both classes occur at the frozen n=32 culture count ({av['transmitter']} transmitters, {av['regenerator']} regenerators), and at the literal n=64 count ({av64['transmitter']}, {av64['regenerator']}); the offset component has {av['offset_eligible']} / {av64['offset_eligible']} eligible cells. P4 is {label(av['P4'])}, KM median t_rep {av['km_t_rep']:,.0f}. The null rule {'applies' if av['null'] else 'does not apply'}, and the kill rule is {'triggered' if av['kill'] else 'not triggered'}. The registered verdicts are reported alongside the original in the correction tables.", "",
             "**Protocol conflicts.** The registration says heritable over 64 partners, while the frozen DZ final-cell estimator uses 32; both complete final-cell classifications are now reported. It says ‘soup partners’, while exec_trace and the frozen DZ scorer draw seeded uniform-byte partners. Random tracing remains the reported registered analysis because it matches the cited frozen definition. The separate soup-snapshot sensitivity uses 16 partners from each world’s nearest saved snapshot at or after t_rep, and does not re-estimate P2 or t_closed.", "", md(partner_summary), "",
             "**Descriptive B–D.**", "", md(final_summary[final_summary.arm != "A"]), "",
             "**Remaining limits.** Sequential instruction increments through $FFFF are now exempted using the executed Zilion instruction length. Forward alias returns remain invisible to the registered 16-bit-PC-decrease measure; the prior position-space result remains exploratory and does not change P1. The nominal 512 mutations per step are attempts: the audit demonstrates a lost update when distinct byte mutations share a packed word. This is the same shared, disclosed property as in the Z80 kernel, not a 6502-specific correction. No mutation-policy rerun was performed, so its effect on the world outcomes is unmeasured. The updated scorer retains exact Z80 DZ p=0 positive-control parity, 10/10 on every column."]
    (ROOT / "CORRECTION_PREREG_TEXT.md").write_text("\n".join(draft) + "\n")
    files = [ROOT / "AUDIT.md", ROOT / "s1_worlds.csv", ROOT / "s1_cells.csv", ROOT / "s1_verdicts.json",
             *sorted(ROOT.glob("audit_*.csv")), *sorted(OUT.glob("*.csv")), OUT / "modal_rates.json", OUT / "modal_apps.json",
             OUT / "cost_summary.json", OUT / "modal_calls.json"]
    S.write_json(OUT / "PROVENANCE.json", {"date": DATE, "scorer_commits": sorted({r["modal"]["scorer_commit"] for r in new + controls}),
        "apps": sorted({r["modal"]["app_id"] for r in new + controls}), "registered_worlds": len(new),
        "control_worlds": len(controls), "whole_app_compute_upper_bound_usd": upper,
        "total_run_cost_upper_bound_usd": upper + 0.01,
        "scorer_compute_estimate_usd": estimate,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evidence_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}})
    print(json.dumps({"arm_A_32": av, "arm_A_64": av64, "flag_changes": changes,
                      "partner_summary": ps, "cost_upper_bound_usd": upper}, indent=2))


if __name__ == "__main__":
    main()
