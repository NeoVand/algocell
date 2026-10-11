"""Every source-defined unit, length and budget is represented exactly once."""
import csv
import json

import impact_assay_supplement as U
from algocell_exp.impact_assays import panel


def test_all_126_named_unit_table_cases_are_covered():
    primary = panel()
    extra, aliases = U.catalogue(primary)
    assert len(extra) + sum(len(v) for v in aliases.values()) == 126
    assert not {U.key(e) for e in primary} & {U.key(e) for e in extra}
    assert len(extra) == len({U.key(e) for e in extra})
    assert len({e["aid"] for e in primary + extra}) == len(primary + extra)
    assert {e["budget"] for e in extra} == {32, 128}
    assert any(e["L"] == 9 and e["tape"] == " ".join(["1d ed b0"] * 3) and
               any("Theorem 6" in r for r in e["references"]) for e in extra)
    for e in extra:
        assert len(bytes.fromhex(e["tape"])) == e["L"]
        assert e["n"] == 64 and e["seed"] == 0 and e["sources"] and e["references"]
