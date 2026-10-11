"""Reconcile the compressed trial ledger with every generated count table (CPU only)."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from algocell_exp.py65_ring import IMPLEMENTED, JAM

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/py65_difftest"
COUNT_FIELDS = ("total", "compared", "matched", "out_of_scope", "explained", "unexplained")


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    counts, groups, depth, reasons, unsupported, scopes = (defaultdict(Counter) for _ in range(6))
    available_worlds, census_words = set(), set()
    with gzip.open(OUT / "trials.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            corpus, status = row["corpus"], row["status"]
            assert status in COUNT_FIELDS[2:]
            if row["input_hex"]:
                assert len(bytes.fromhex(row["input_hex"])) == int(row["P"])
            group = (corpus, row["group"], int(row["P"]))
            for counter in (counts[corpus], groups[group]):
                counter["total"] += 1
                counter[status] += 1
                if status != "out_of_scope":
                    counter["compared"] += 1
            if corpus == "s1":
                available_worlds.add(row["group"])
            elif corpus == "census":
                census_words.add(row["group"])
            if status == "out_of_scope":
                witness = json.loads(row["unsupported"])
                assert witness["opcode"] not in IMPLEMENTED | JAM
                assert 1 <= witness["step"] <= summary["parameters"]["budget"]
                assert 0 <= witness["pc"] <= 0xFFFF
                unsupported[corpus][witness["opcode"]] += 1
                scopes[corpus][witness.get("stream", "raw-py65")] += 1
                assert not row["difference"]
                continue
            assert not row["unsupported"]
            n, halt = int(row["gpu_steps"]), int(row["gpu_halt_reason"])
            depth[corpus]["one_instruction" if n == 1 else "multiple_instructions"] += 1
            depth[corpus]["ran_to_budget" if halt == 0 else "jam_halted"] += 1
            assert halt in (0, 3)
            assert int(row["gpu_trap_cause"]) == (1 if halt == 3 else 0)
            if status == "matched":
                assert not row["difference"] and not row["reasons"]
            else:
                assert row["difference"] and row["input_hex"]
            if status == "explained":
                assert row["reasons"]
                for reason in row["reasons"].split(";"):
                    assert reason in ("status-b-bit", "jsr-low-read-order")
                    reasons[corpus][reason] += 1
    for corpus, counter in counts.items():
        assert counter["compared"] == counter["matched"] + counter["explained"] + counter["unexplained"]
        assert counter["total"] == counter["compared"] + counter["out_of_scope"]
        for field in COUNT_FIELDS:
            assert counter[field] == summary["counts"][corpus].get(field, 0), (corpus, field)
        for field in ("one_instruction", "multiple_instructions", "ran_to_budget", "jam_halted"):
            assert depth[corpus][field] == summary["execution_depth"][corpus].get(field, 0)
        assert dict(reasons[corpus]) == summary["reasons"].get(corpus, {})
        assert dict(scopes[corpus]) == summary["scope_counts"].get(corpus, {})

    with (OUT / "counts.csv").open() as f:
        for row in csv.DictReader(f):
            assert all(counts[row["corpus"]][field] == int(row[field]) for field in COUNT_FIELDS)
    with (OUT / "groups.csv").open() as f:
        for row in csv.DictReader(f):
            counter = groups[(row["corpus"], row["group"], int(row["P"]))]
            assert all(counter[field] == int(row[field]) for field in COUNT_FIELDS)
    with (OUT / "unsupported.csv").open() as f:
        for row in csv.DictReader(f):
            assert unsupported[row["corpus"]][int(row["opcode"], 16)] == int(row["count"])
    with (OUT / "explained.csv").open() as f:
        for row in csv.DictReader(f):
            assert reasons[row["corpus"]][row["reason"]] == int(row["count"])
    with (OUT / "scope_counts.csv").open() as f:
        for row in csv.DictReader(f):
            assert scopes[row["corpus"]][row["stream"]] == int(row["count"])
    with (OUT / "execution_depth.csv").open() as f:
        for row in csv.DictReader(f):
            assert all(depth[row["corpus"]][field] == int(row[field]) for field in row if field != "corpus")
    for relative, expected in {**summary["input_source_sha256"], **summary["harness_sha256"]}.items():
        assert hashlib.sha256((HERE / relative).read_bytes()).hexdigest() == expected, relative
    for ring, machine in summary["machines"].items():
        wgsl = HERE / f"algocell_exp/machines/{machine['machine']}.wgsl"
        assert hashlib.sha256(wgsl.read_bytes()).hexdigest() == machine["sha256_wgsl"], ring
    examples = json.loads((OUT / "examples.json").read_text())
    for key, example in examples.items():
        assert example["first_raw_divergence"], key
        assert example["diagnostic_final_matches"], key
    if not summary["stopped_early"] and not summary["failure"]:
        assert len(available_worlds) == summary["s1_available_first_tapes"] == 62
        if not summary["parameters"]["skip_census"]:
            assert len(census_words) == 300
        expected_random = summary["parameters"]["random"] * len(summary["parameters"]["random_seeds"])
        assert counts["random_P32"]["total"] == counts["random_P64"]["total"] == expected_random
    result = {
        "finished": datetime.now().astimezone().isoformat(),
        "command": "/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/check_py65_difftest.py",
        "status": "passed", "ledger_rows_reconciled": sum(c["total"] for c in counts.values()),
        "s1_worlds_tested": len(available_worlds), "census_words_tested": len(census_words),
        "source_hashes_checked": True, "machine_hashes_checked": True,
        "witness_categories": len(examples),
        "ledger_sha256": hashlib.sha256((OUT / "trials.csv.gz").read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
