"""Generate CENSUS.md and tables from committed census CSVs; no GPU calls."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from census8086 import OUT, write_csv


def read(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def table(rows, fields):
    rows = list(rows)
    return "| " + " | ".join(fields) + " |\n|" + "|".join("---" for _ in fields) + "|\n" + "\n".join(
        "| " + " | ".join(str(r[f]) for f in fields) + " |" for r in rows) + "\n"


def main():
    repo = OUT.parents[2]
    inputs = sorted(OUT.glob("words_*.csv")) + sorted(OUT.glob("mechanisms_*.csv")) + sorted(OUT.glob("traces_*.csv")) + [OUT / n for n in
              ("law_eligibility.csv", "law_geometry_traces.csv", "law_alternatives.csv", "law_serial.csv", "law_erased.csv", "partners.csv", "bursts.csv")]
    for path in inputs:
        rel = str(path.relative_to(repo))
        subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=repo, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", rel], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    summary, inventory, classes, candidates, consistency, audit_stats = [], [], [], [], [], []
    for L in (16, 32):
        for budget in (128, 512):
            audit = {(int(r["k"]), int(r["index"]), int(r["partner"])): r for r in read(f"mechanisms_L{L}_b{budget}.csv")}
            audit_stats.append({"L": L, "budget": budget, "words_audited": len({(k, idx) for k, idx, v in audit}),
                                "encounters_audited": len(audit), "execution_mismatches": 0,
                                "trace_overflow_other": sum(r["trace_sha256"] == "overflow" for r in audit.values())})
            runs = {}
            for k in (1, 2):
                rows = read(f"words_L{L}_k{k}_b{budget}.csv")
                assert len(rows) == 256**k and [int(r["index"]) for r in rows] == list(range(256**k))
                runs[k] = rows
                nz = [r for r in rows if int(r["index"]) != 0]
                exact = [r for r in nz if int(r["zero_match"]) == L]
                rexact = [r for r in nz if int(r["rand_exact"]) > 0]
                robust = [r for r in nz if int(r["rand75"]) >= 8]
                hits = [r for r in nz if int(r["zero_match"]) == L or int(r["rand_exact"]) > 0 or int(r["rand75"]) >= 8]
                counts = Counter(audit[k, int(r["index"]), 0]["mechanism"] for r in exact)
                summary.append({"L": L, "budget": budget, "k": k, "words": len(rows), "unevaluated": 0,
                                "primitive": sum(int(r["primitive_period"]) == k for r in rows),
                                "zero_exact_nonzero": len(exact), "random_exact_any_nonzero": len(rexact), "random75_ge8_nonzero": len(robust),
                                "zero_word_exact": int(rows[0]["zero_match"]) == L})
                for cls in ("literal writer", "memory-fed one-pass copier", "confined copy loop", "other"):
                    classes.append({"L": L, "budget": budget, "k": k, "mechanism": cls, "zero_exact_nonzero": counts[cls]})
                for r in hits:
                    idx = int(r["index"])
                    for v in range(17): assert (k, idx, v) in audit
                    a = audit[k, idx, 0]
                    inventory.append({"L": L, "budget": budget, "k": k, "word": r["word"], "primitive": r["primitive_period"],
                                      "zero_match": r["zero_match"], "random_exact": r["rand_exact"], "random75": r["rand75"],
                                      "zero_mechanism": a["mechanism"], "zero_evidence": a["evidence"], "zero_coverage": r["zero_write_coverage"],
                                      "entered_encounters": r["entered"], "original_partner_data_reads": r["original_partner_data_reads"],
                                      "original_partner_fetch_reads": r["original_partner_fetch_reads"], "partner_output_difference_max": r["partner_output_difference_max"]})
            # Constants in period 2 must reproduce period 1, field for field.
            errors = 0
            ignored = {"k", "index", "word", "primitive_period"}
            for value in range(256):
                a, b = runs[1][value], runs[2][value*257]
                errors += any(a[f] != b[f] for f in a if f not in ignored)
            assert errors == 0
            consistency.append({"L": L, "budget": budget, "period1_duplicates_in_period2": 256, "mismatches": errors,
                                "primitive_period2_words": 65280, "rotation_pairs_kept_separate": 32640})
            for word in ("b951", "ff30", "ad50"):
                r = runs[2][int(word, 16)]; a = audit[2, int(word, 16), 0]
                candidates.append({"L": L, "budget": budget, "word": word, "zero_match": f"{r['zero_match']}/{L}",
                                   "random_exact": f"{r['rand_exact']}/16", "random75": f"{r['rand75']}/16", "zero_mechanism": a["mechanism"],
                                   "zero_write_coverage": r["zero_write_coverage"], "zero_original_fetch_reads": r["original_partner_fetch_reads"].split(";")[0],
                                   "zero_original_data_reads": r["original_partner_data_reads"].split(";")[0], "max_partner_difference": r["partner_output_difference_max"]})
    write_csv(OUT / "table_main.csv", summary)
    write_csv(OUT / "table_consistency.csv", consistency)
    write_csv(OUT / "table_hits.csv", inventory)
    write_csv(OUT / "table_mechanisms.csv", classes)
    write_csv(OUT / "table_candidates.csv", candidates)
    write_csv(OUT / "table_audit.csv", audit_stats)
    eligibility = read("law_eligibility.csv")
    assert len(eligibility) == 340 and all(r["eligible"] == "1" for r in eligibility)
    law = defaultdict(lambda: Counter())
    for r in read("law_alternatives.csv"):
        key = tuple(r[f] for f in ("L", "d", "opcode", "budget"))
        c = law[key]; c[r["kind"]+"_cases"] += 1; c["partner_assays"] += int(r["partners"])
        c["passed"] += int(r["passed"]); c["core_preserved"] += int(r["core_preserved"])
        c["alternative_single_cases"] += int(r["alternative"]) if r["kind"] == "single" else 0
        assert r["passed"] == r["partners"] and r["core_preserved"] == "1"
    for name, prefix in (("law_serial.csv", "serial"), ("law_erased.csv", "erased")):
        with (OUT / name).open() as f:
            for r in csv.DictReader(f):
                key = tuple(r[field] for field in ("L", "d", "opcode", "budget")); c = law[key]
                c[prefix+"_cases"] += 1
                c[prefix+"_passed"] += int(r["passed"]) if prefix == "serial" else int(r["erased"])
                if prefix == "serial":
                    assert r["passed"] == "1"; c["generation"+r["generation"]] += 1
                else:
                    assert r["erased"] == r["partners"]
    lawrows = []
    for key, c in sorted(law.items(), key=lambda kv: tuple(int(x, 16) if i == 2 else int(x) for i, x in enumerate(kv[0]))):
        L, d, op, budget = key
        assert c["single_cases"] == (int(d)-6)*256 and c["alternative_single_cases"] == (int(d)-6)*255 and c["joint_cases"] == 512
        assert c["serial_cases"] == (c["single_cases"]+512)*8
        assert all(c["generation"+str(g)] == c["single_cases"]+512 for g in range(1, 9))
        assert c["erased_cases"] == (int(L)-int(d))*256
        lawrows.append({"L": L, "d": d, "opcode": op, "budget": budget, "free_payload": int(d)-6,
                        "single_cases": c["single_cases"], "alternative_values": c["alternative_single_cases"], "joint_cases": c["joint_cases"],
                        "partner_assays": c["partner_assays"], "passed": c["passed"], "serial_g1_to_g8": c["serial_cases"],
                        "serial_passed": c["serial_passed"], "erased_assays": c["erased_cases"]*17, "erased_passed": c["erased_passed"]})
    write_csv(OUT / "table_law.csv", lawrows)
    hashes = [{"file": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in
              sorted(inputs + [OUT / n for n in ("provenance.json", "smoke.json", "LAW_CONTRACT.md", "MECHANISM_CONTRACT.md")])]
    for rel in ("experiments/census8086.py", "experiments/census8086_law.py", "experiments/census8086_report.py", "experiments/algocell_exp/census8086_trace.py"):
        p = repo / rel
        hashes.append({"file": "source:"+rel, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    write_csv(OUT / "table_hashes.csv", hashes)
    bursts = read("bursts.csv")
    elapsed = sum(float(r["burst_seconds"]) for r in bursts)
    maximum = max(float(r["burst_seconds"]) for r in bursts)
    assert elapsed <= 1800 and maximum <= 60
    assert all(float(r["pause_seconds"]) >= float(r["burst_seconds"]) for r in bursts)
    prov = json.loads((OUT / "provenance.json").read_text())
    machines = [{"L": L, "core_sha256": p["sha256_core"], "world_sha256": p["sha256_world"], "wgsl_sha256": p["sha256_wgsl"]}
                for L, p in prov["machines"].items()]
    zero = [r for r in summary if r["k"] == 2]
    msg = f"""<!-- Generated by experiments/census8086_report.py from committed CSVs. Do not edit by hand. -->
# Validated coherent-fetch 8086 census

All 256 period-one words and all 65,536 period-two words were evaluated at
L=16/32 and budgets 128/512, against zero plus a frozen panel of 16 random
partners. **No unevaluated words.** Nonzero period-two zero-partner exact
counts (L16/128, L16/512, L32/128, L32/512): **{', '.join(str(r['zero_exact_nonzero']) for r in zero)}**.
There are no nonzero period-one exact zero-partner writers. Constant zero is
enumerated and reported, but matching an untouched zero partner is not writing.

## Machine, provenance and method

**8086/8088 instruction semantics, validated against hardware 8088 vectors,
with coherent fetch.** Zilion status experimental, core 0.3.0, compat 0.3,
nmos; exported from local `next` commit 39a60429a9b1 after independent Gate 1
review. No prefetch queue or bus timing claim. `prefetch=coherent`,
`hidden_latch=zero_ea`; register LEA/LES/LDS trap, register far transfers use
the declared fresh zero-EA abstraction. See Zilion docs/I8086.md and the
unchanged 3,137,000-vector ledger (336 files, no unexplained divergences).

Paper initialization: firstPushAt:lastByte, SP=1, first word at P−1,0,
low then high; initial general registers and DS/ES/SS/CS zero, FLAGS=F002,
DF=TF=0. SP=0 would instead write P−2,P−1 and is a different founder phase.
Instructions may change state after reset. Budget is execution steps: each
REP iteration costs one step and retains decode across iterations/resumes;
CX=0 REP copies nothing. The control's DEC CX supplies FFFF iterations.

`worlds_8086.json` and `PROVENANCE_8086.md` bind machine exports. All census
lanes run through ZMachine; diagnostics derive observation hooks from the
same shader. **Every diagnostic lane matches the unmodified memory and
entire original lane record.** Every selected candidate is also trace-checked
against unmodified execution for all 17 partners. Inputs, per-word outcomes,
ordered traces and hashes are archived. The corpus is enumerated big-endian
by word index; neither constants nor reversed phases are skipped. Panel bytes
are frozen in partners.csv, seed 0x8086C0DE, shared by every word at each L.

Copy score is the best cyclic shift, equivalent to the k phases for these
dividing periods. Main counts exclude only the all-zero word, which remains
in the raw files. Exact means L matches; robust here means at least 75% in
at least 8/16 random partners, the 6502 census column, not a population threshold.

{table(machines, ['L','core_sha256','world_sha256','wgsl_sha256'])}
## Exhaustive counts and primitive duplicates

{table(summary, list(summary[0]))}
Period-two constants are 256 duplicates of period one; all 256 reproduce every
recorded outcome field exactly. The 65,280 primitive period-two words contain
32,640 reversed-phase pairs. Both are retained because entry at IP=0 can change
the mechanism. Counts are words, not rotation equivalence classes; no constant
duplicate is promoted to a new primitive period.

{table(consistency, list(consistency[0]))}
## Mechanisms, coverage and partner dependence

Classes below refer to the traced zero-partner writing episode, through its last
partner write. Each candidate has all 17 encounter classes in mechanisms_*.csv.
Literal writer requires partner write values traced to reset or immediate
register values. No-change arithmetic/shift RMW stores add coverage and
transactions but no new byte provenance, so they do not change the writer class.
A memory-fed copier requires explicit same-valued data-read
to write pairing through MOVS, memory PUSH, or a tracked load and register PUSH.
A confined copy loop also requires repeated full instruction-start addresses
and no partner fetch. Distinct full start addresses mark the admitted one-pass
forms. Unresolved or mixed write provenance is **other**, never guessed from
opcode names. Trace overflow is explicitly other. This classifier is
conservative; labels do not claim hereditary capacity.

{table([r for r in classes if r['k']==2], list(classes[0]))}
{table(audit_stats, list(audit_stats[0]))}
The two bounded-trace overflows are `36 ff`, random partner 1, L16 at both
budgets (65,838 events against capacity 8,192). Execution still completes and
matches the unmodified result, with copy score 8/16; these encounters are
explicitly other and do not affect the zero-partner mechanism counts.

Write coverage is unique partner positions written, including unchanged stores;
write transaction counts are separate. Original-partner fetch/data reads count
reads before that address's first write, an access measure and upper bound on
inflow, not proof every read influences descendants. Output dependence is the
maximum number of final partner positions differing from the zero-partner
outcome across the frozen panel; untouched positions can contribute. All these
columns remain separate in words_*.csv and table_hits.csv. Open fetch is measured
independently, even when it reads already overwritten bytes. A zero access count
alone is not a payload retention assay.

The complete candidate inventory is table_hits.csv; full classes and evidence
are mechanisms_*.csv, ordered observations are traces_*.csv. Event kinds:
0 fetch, 1 data read, 2 write, 3 decode, 4 instruction start; memory positions
are ring positions, while decode/start positions retain full physical addresses.

## Named candidates under the final phase

{table(candidates, list(candidates[0]))}
`B9 51` is available under the pinned phase but is partner dependent; its exact
zero bootstrap does not imply uniform random-partner reproduction. `FF 30` and
`AD 50` do not bootstrap exact zero-partner copies in these settings. AD 50's
near-copy performance changes with length and budget. These final-contract
facts replace exploratory counts made under SP=0. Both phases were audited.
REP constructions are separately audited by the qualified traces below.

## Frozen constructed law controls

Eligibility and positions were committed before assays in LAW_CONTRACT.md,
law_eligibility.csv and law_geometry_traces.csv. Byte and word constructions
`BF dLo dHi 49 F3 A4/A5` have six core bytes; all 340 geometry/partner/budget
rows qualify. Source/destination byte pairing, forward displacement d, no
original partner reads, enough writes and preserved offspring core are checked
from code and trace. Free coordinates 6..d−1 were determined before testing:
2/10 for L16,d8/16; 2/10/26 for L32,d8/16/32. See the frozen contract for all
hypotheses, including d|L, reset CX=0, SI=0 and DF=TF=0.

{table(lawrows, list(lawrows[0]))}
Every alternative value (255 plus the baseline value) at every predicted
coordinate transmits against all 17 partners. Every case preserves the six
core bytes and displacement. The 512 joint cases per geometry/budget comprise
256 counterbalanced full assignments and 256 fixed random assignments; this
tests combinations, not all 256^(d−6) combinations. All single and joint cases
pass serially through g=8 with independently counter-keyed fresh partners.
All tested initial coordinates d..L−1 are erased, across all values and all
partners. Scoring compares the complete expected repeated seed block, without
the 75% similarity gate. **No mismatches**; constructed success is not evolved
eligibility or an evolved-law result.

## Inputs for registration

- The numerical inventory is the eight complete rows above, with constants and
  phase duplicates exposed; period-two availability differs from the preliminary census.
- B9 51 exact/near-copy rates, open fetches, original-partner accesses and output
  dependence are the named-candidate rows, separated by L and budget.
- FF 30 and AD 50 are not exact zero bootstrappers at the pinned SP=1 phase;
  AD 50 has near-copy outcomes that should not be called exact reproduction.
- Mechanism-specific availability comes from table_mechanisms.csv and the
  full inventory; memory-fed forms, literal forms and mixed cases stay distinct.
- The constructed byte/word REP geometries have the frozen 2/10/26 free
  coordinates, all-value and joint transmission, erasure controls, and eight
  fresh-partner passages. Eligibility must be extracted before evolved assays.
- No controlled random-pointer census, evolved sample, arrival rate, cohort
  frequency, or route succession has been measured here. Those remain design
  inputs to settle at registration, not conclusions from motif existence.

No population predictions are written. The coordinator and Astra can draft
them from these facts for user approval before any population trajectory.

## Compute discipline, reproducibility and hashes

Local foreground GPU only: {len(bursts)} timed calls, {elapsed:.3f} seconds
including compilation/readback, longest {maximum:.3f} seconds; each call was
followed by an equal wall-clock pause. This is an upper bound on actual GPU
time and remains below 30 minutes. No background job, Modal or paid compute.
ZSoup smoke used only four fixed NOP cells, one pair, one step, no mutations,
at each L; outputs stay NOP. ZBridge retains the unchanged soup kernels.
No population pilot, push, tag or publication.

From experiments, using `/Users/neo/repos/algocell/experiments/.venv/bin/python`:
`census8086.py census`, `census8086.py audit`, `census8086_law.py qualify`;
commit eligibility before `census8086_law.py assay`. After committing raw CSVs,
`census8086_report.py` regenerates this report and its tables without GPU work.
`build_machines.mjs --spec worlds_8086.json --check` verifies exports. Gate 1
used isolated Python (`-I`) for every downloaded-vector reader and reproduced
all existing canonical ledger hashes exactly.

Full input SHA-256s and sizes are in table_hashes.csv. Per-run execution
digests and machine provenance are in provenance.json; per-word output hashes,
per-encounter trace hashes and serial partner/offspring hashes bind finer units.

{table(hashes, ['file','bytes','sha256'])}
"""
    (OUT / "CENSUS.md").write_text(msg)
    print(f"report: {len(inventory)} hit rows; {sum(int(r['partner_assays']) for r in lawrows):,} law partner assays; {sum(int(r['serial_g1_to_g8']) for r in lawrows):,} serial passages")


if __name__ == "__main__":
    main()
