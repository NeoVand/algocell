"""Projection must use the affected union and the stochastic world's measured cost."""
import math
from types import SimpleNamespace
import numpy as np
import pytest

from algocell_exp import impact as I
from impact_report import projection
import impact_modal as M


def test_benchmark_wall_timer_survives_multiple_execution_blocks(monkeypatch):
    # CPU-only fake dispatches isolate the clock and chunk-index regression.
    times = iter((10, 12, 15, 17, 21, 22))
    monkeypatch.setattr(M.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(M, "NREG", 5)
    monkeypatch.setattr(M.S, "BLOCK", 2)
    monkeypatch.setattr(M.K, "draw_random", lambda *args: np.zeros((5, 32), np.uint8))
    lanes = []
    def run(*args, lane_base=0):
        lanes.append(lane_base)
    fake = SimpleNamespace(L=16, P=32, w=SimpleNamespace(rule="plain"), run=run)
    monkeypatch.setattr(M.I, "Triple", lambda tag: fake)
    monkeypatch.setattr(M.I, "FastAttributor", lambda *args: SimpleNamespace(attribute=lambda *a, **kw: None))
    b = M.bench_item("L16", 128)
    assert lanes == [0, 0, 2, 4, 0]
    assert b["t_exec_s"] == 3 and b["t_attr_batch_s"] == 4
    assert b["wall_s"] == 12


def test_projection_uses_union_and_rule_specific_benchmark():
    row = dict(cid="DZ/example", tier="added", world="lp003_L16", budget=128, pairs=1000,
               aff_union=25, aff_c1=10, aff_c2=20, t_draw_s=.1, t_compile_s=3, t_exec_s=.003)
    bench = [dict(world="L16", rule="plain", L=16, budget=128, pairs=100000, t_exec_s=1, t_attr_batch_s=2),
             dict(world="lp003_L16", rule="dial", L=16, budget=128, pairs=100000, t_exec_s=7, t_attr_batch_s=19)]
    p = projection([row], bench, 100000, .001)[0]
    assert p["expected_affected"] == 2500  # union, not max(C1,C2) or their sum
    assert p["attr_batches"] == 2 and p["attr_s"] == 38
    assert p["exec_s"] == pytest.approx(7) and p["prepare_s"] == 10 and p["compile_s"] == 3
    _, upper = I.clopper_pearson(25, 1000)
    assert p["upper_attr_batches"] == math.ceil(100000 * upper / 2048)
    assert p["reset_s"] == pytest.approx((2500 * .975 + 2 * .025 - 2) * .003)
    assert p["total_s"] == pytest.approx(58 + p["reset_s"])
    assert p["cost_usd"] == pytest.approx(p["total_s"] * .001)


def test_projection_zero_hits_still_has_an_upper_attribution_cost():
    row = dict(cid="random/example", tier="random", world="L16", budget=256, pairs=1000,
               aff_union=0, t_draw_s=0, t_compile_s=0, t_exec_s=.003)
    bench = [dict(world="L16", rule="plain", L=16, budget=128, pairs=100000, t_exec_s=1, t_attr_batch_s=2)]
    p = projection([row], bench, 100000, .001)[0]
    assert p["attr_batches"] == 0 and p["attr_s"] == 0
    assert p["upper_attr_batches"] == 1 and p["upper_attr_s"] == 4
    assert p["exec_s"] == 2  # budget scales linearly when no exact benchmark exists
    assert p["reset_s"] == 0 and p["upper_reset_s"] > 0
