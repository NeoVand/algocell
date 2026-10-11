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


def test_l10_nominal_supports_size_panel_but_not_six_length_km_panel():
    c = R.claim(dict(family="stageE", cid="stageE/none@nominal_L10_st128_k4@emergence", world="L10"))
    assert "Extended Data Fig. 2b" in c["claims"]
    assert "outside Fig. 1b's six" in c["claims"]
    assert "Also the six-length" not in c["claims"]


def test_copy_offset_benign_l16_is_an_unplotted_control():
    c = R.claim(dict(family="E", cid="E/L16_benign_muton_R_into_T@final", world="L16"))
    assert "core-collapse control" in c["claims"] and "omitted" in c["claims"]
    assert "Extended Data Fig. 6a" not in c["claims"]


def test_standard_register_invasion_maps_to_c4c_and_its_actual_outcome():
    c = R.claim(dict(family="R4", cid="R4/std_R4_SI13@final", world="L16"))
    assert "neither named design persists" in c["claims"]
    lines = (R.S.HERE / "REVISION_PREREG.md").read_text().splitlines()
    referenced = [lines[int(r.split(":")[1]) - 1] for r in c["registration"].split("; ")]
    assert any(r.startswith("**C4c") for r in referenced)
