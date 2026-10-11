"""Registered decision threshold and independent attribution reconciliation."""
import pytest

import impact_i1_report as R
from algocell_exp import impact as I


def test_decision_is_strict_cp_upper_not_point_rate():
    assert R.decision(I.clopper_pearson(3, 100000)[1]) == "unaffected"
    assert R.decision(I.clopper_pearson(4, 100000)[1]) == "flagged"
    assert R.decision(1e-4) == "flagged"


def row(cid="random/test", family="random", k=4):
    r = dict(cid=cid, family=family, world="L16", pairs=100000, aff_c1=k, aff_c2=k)
    for core in ("c1", "c2"):
        for cls in I.CLASSES:
            r[f"{core}_{cls}"] = k if cls == "bit-hl" else 0
    return r


def pairs(r):
    return [dict(cid=r["cid"], core=c, decisive_class="bit-hl") for c in ("c1", "c2") for _ in range(r[f"aff_{c}"])]


def test_ring_sweep_is_reported_outside_decision_scope():
    r = row("ring_sweep/test", "ring_sweep")
    rates, attrs, flags = R.rates_and_attributes([r], pairs(r))
    assert len(rates) == 3 and len(attrs) == 10 and len(flags) == 2
    assert all(x["decision_scope"] == 0 for x in rates)
    assert all(x["decision"] == "flagged" for x in flags)


def test_attribution_records_must_match_counts_and_sample_size():
    r = row()
    with pytest.raises(ValueError, match="evidence"):
        R.rates_and_attributes([r], pairs(r)[:-1])
    r["pairs"] = 1000
    with pytest.raises(ValueError, match="100,000"):
        R.rates_and_attributes([r], pairs(r))


def test_unplotted_exploratory_arm_is_not_assigned_to_filtered_figure():
    c = R.claim(dict(family="stageC", cid="stageC/all-ld_L16_st32_k2@emergence"))
    assert "THIS CONDITION is outside" in c["claims"]
    assert "NUMBERS_INDEX" in c["numbers_index"] and "REVISION_PREREG" in c["registration"]
