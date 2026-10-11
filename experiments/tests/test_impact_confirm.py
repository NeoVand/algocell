"""I1 corpus coverage, frozen draw/analysis code, and all-column assay identity."""
import ast
import inspect
import json
import subprocess

import pytest

import impact_confirm as C
import impact_study as S
from algocell_exp import executor as E
from algocell_exp.impact_assays import ImpactExecutor, assay_item, identity, panel


def test_every_assay_column_is_compared_exactly():
    base = dict(score=.5, gen2_cond=float("nan"), copy_offset=None, faithful=True)
    outputs = {c: dict(base) for c in ("c0", "c1", "c2")}
    assert identity(outputs)["all_identical"]
    for column in base:
        changed = {c: dict(base) for c in outputs}
        changed["c2"][column] = "changed"
        assert identity(changed)["differing_columns"]["c2"] == [column]
    outputs["c1"]["score"] += 1e-15
    assert not identity(outputs)["c1_identical"]


def test_assay_missing_column_is_an_error():
    with pytest.raises(ValueError, match="columns"):
        identity(dict(c0=dict(score=0), c1={}, c2=dict(score=0)))


def test_panel_has_full_sources_rules_and_named_controls():
    items = panel()
    assert len(items) == len({x["aid"] for x in items})
    assert {x["world"] for x in items} >= {"L16", "L20", "L32", "L50", "L64", "lethal_L16", "randreg_L16", "randsp_L16"}
    assert any(x["suppress"] for x in items)
    assert any(x["tape"] == " ".join(["00 00"] * 8) for x in items)
    for item in items:
        assert len(bytes.fromhex(item["tape"])) == item["L"]
        assert item["sources"] and item["references"]
        assert item["n"] == 64 and item["seed"] == 0 and item["budget"] == 128
    assert sum(any("Extended Data Table 1" in r for r in x["references"]) for x in items) >= 13


def test_frozen_draw_and_attribution_modules_unchanged():
    for file in ("impact.py", "impact_corpus.py", "impact_worlds.py"):
        rel = f"experiments/algocell_exp/{file}"
        old = subprocess.check_output(["git", "show", f"bdcdcb6:{rel}"], cwd=S.HERE.parent)
        assert (S.HERE / "algocell_exp" / file).read_bytes() == old
    old = subprocess.check_output(["git", "show", "bdcdcb6:experiments/impact_study.py"], cwd=S.HERE.parent).decode()
    defs = {f.name: f for f in ast.parse(old).body if isinstance(f, ast.FunctionDef)}
    for fn in (S.run_condition, S._cat, S.masks_of):
        assert ast.dump(ast.parse(inspect.getsource(fn)).body[0]) == ast.dump(defs[fn.__name__])


def test_cost_allocation_is_within_cap():
    assert C.ALLOCATION_SECONDS * C.M.RATES["L40S"] + C.RESERVE_USD < 8
    assert C.MAX_SECONDS < C.ALLOCATION_SECONDS


@pytest.mark.parametrize("world", ["L16", "lethal_L16", "randreg_L16", "randsp_L16", "lp003_L16"])
def test_imp_assay_uses_only_machines_and_all_columns(world):
    item = dict(aid="test", tape=" ".join(["01 c5"] * 8), world=world, L=16,
                suppress=[], n=16, seed=0, budget=128)
    before = E.usage()
    result = assay_item(item)
    assert E.active() is None
    assert len(result["outputs"]["c0"]) == 14
    after = E.usage()
    assert after["shader"] == before["shader"]
    for core in ("c0", "c1", "c2"):
        machine = C.K.WORLDS[world].machine(core)
        assert after["machine"][machine] - before["machine"].get(machine, 0) == 3 * 16
