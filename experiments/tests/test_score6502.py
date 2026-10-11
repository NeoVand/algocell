"""S1 scorer tests: handmade tapes/records only; no registered-world access."""
import copy
import hashlib
import json

import brotli
import numpy as np
import pandas as pd
import pytest

from algocell_exp import executor as E
from score6502 import endpoints, pc_has_loop, read_input, score_world, verdict, report
from modal_score6502 import locations, parse_seeds, world_cpu_test


@pytest.mark.parametrize("sequence,expected", [
    ([0, 1, 2, 31, 32, 63, 64], False),  # laps of a memory ring are not PC decreases
    ([0, 3, 5, 3], True),
    ([65534, 65535, 0, 1], False),
    ([65535, 2], True),  # only the exact registered increment wrap is exempt
    ([[0, 1, 2], [0, 2, 1]], True),
])
def test_loop_definition(sequence, expected):
    assert pc_has_loop(sequence) is expected


@pytest.mark.parametrize("machine,hexcode,expected", [
    ("z80c02_assay_L16", "01 c5" * 8, False),
    ("z80c02_assay_L16", "00 18 fd" + " 00" * 13, True),
    ("z80c02_assay_L16", "06 08 00 10 fd" + " 76" * 11, True),
    ("z80c02_assay_L16", "c3 ff ff" + " 00" * 13, False),
    ("m6502_pair_maps_L16", "b5 0f 48 ca 50 fa", True),
    ("m6502_pair_maps_L16", "ba b5 10 48 50 fa", True),
    ("m6502_pair_maps_L16", "ba b5 0f 48", False),
])
def test_handmade_machine_traces(machine, hexcode, expected):
    code = bytes.fromhex(hexcode)
    tape = np.resize(np.frombuffer(code, dtype=np.uint8), 16)
    # Handwritten NOP partners keep open code from encountering incidental
    # random branches or (on 6502) BRK vectors. Real scoring uses DZ's random
    # partners, and correctly counts those decreases too.
    partners = np.full((16, 16), 0 if machine.startswith("z80") else 0xEA, dtype=np.uint8)
    pairs = np.concatenate([np.tile(tape, (16, 1)), partners], axis=1)
    ex = E.ZilionExecutor(machine)
    pc = ex.pc_trace(pairs, 16, 128)
    assert pc.shape == (16, 129)
    assert pc_has_loop(pc) is expected
    # Each prefix really runs the same machine, with the same initial memory.
    m = ex.machine_for(16)
    for k in (0, 1, 2, 8, 128):
        assert np.array_equal(pc[:, k], m.run(pairs, budget=k).reg("pc") & 0xFFFF)


@pytest.mark.parametrize("machine,L,zh", [
    ("m6502_pair_maps_L16", 16, False), ("m6502_pair_maps_L32", 32, False),
    ("m6502_brkhalt_maps_L16", 16, True),
])
def test_registered_machine_interface_on_handmade_pairs(machine, L, zh):
    pairs = np.full((16, 2 * L), 0xEA, dtype=np.uint8)
    ex = E.ZilionExecutor(machine)
    r = ex.run(pairs, L, 512, zero_halts=zh, fetch_map=True)
    assert r.steps.tolist() == [512] * 16 and r.fetch_map.shape == (16, 2 * L)
    assert not pc_has_loop(ex.pc_trace(pairs, L, 512, zero_halts=zh))


def test_exact_machine_cache_retains_request_validation():
    ex = E.ZilionExecutor("m6502_pair_maps_L16")
    ex.machine_for(16)
    with pytest.raises(ValueError, match="halting.lethal"):
        ex.machine_for(16, zero_halts=True)
    with pytest.raises(ValueError, match="segments"):
        ex.machine_for(32)


def call(heritable=True, confined=False, loop=False, opened=True):
    return dict(heritable=heritable, confined=confined, loop=loop, open=opened, entered=16 if opened else 0)


def test_threshold_share_order_and_p2_strict_closure():
    calls = {"small": call(), "open": call(), "closed": call(confined=True, loop=True, opened=False)}
    samples = [dict(step=50, top3=[("small", .00499)]),
               dict(step=100, top3=[("open", .01)]),
               dict(step=150, top3=[("closed", .02), ("open", .005)])]
    row, violations = endpoints(samples, calls.__getitem__)
    assert row["t_rep"] == 100 and row["t_closed"] == 150 and row["first_tape"] == "open"
    assert row["first_open"] and not row["first_loop"] and row["p2_violation"]
    assert [v["step"] for v in violations] == [100]
    row, violations = endpoints([samples[-1]], calls.__getitem__)
    assert row["t_rep"] == row["t_closed"] == 150 and not violations


def test_no_closure_censoring_and_partial_open():
    row, violations = endpoints([dict(step=10, top3=[("x", .005)])], lambda _: call(opened=False))
    assert row["t_rep"] == 10 and row["t_closed"] is None and not row["first_confined"]
    assert not row["first_open"] and not violations
    row, violations = endpoints([dict(step=10, top3=[("x", .005)])], lambda _: call())
    assert len(violations) == 1


def primary(nrep=20, open_count=0):
    results = []
    for i in range(20):
        row = dict(world=f"handmade{i}", arm="A", seed=65001 + i, horizon=10_000_000,
                   snapshot_step=10_000_000, L=16, budget=128, machine="m6502_pair_maps_L16",
                   t_rep=500 if i < nrep else None, t_closed=1000 if i < nrep else None,
                   first_confined=i >= open_count, first_loop=i >= open_count,
                   first_open=i < open_count, p2_violation=i < open_count)
        cells = [dict(confined=True, n_sites=5, copy_offset=11, **{"class": "transmitter"}),
                 dict(confined=True, n_sites=0, copy_offset=6, **{"class": "regenerator"})]
        results.append(dict(world=row, cells=cells, p2_tapes=[]))
    return results


def test_registered_predictions_null_kill_and_description(tmp_path):
    results = primary()
    v = verdict(results)
    assert all(v[p] for p in ("P1", "P2", "P3", "P4")) and not v["kill"] and not v["null"]
    v = verdict(primary(open_count=5))
    assert not v["P1"] and not v["P2"] and v["kill"]
    v = verdict(primary(nrep=4, open_count=1))
    assert v["null"] and all(v[p] is None for p in ("P1", "P2", "P3")) and v["P4"] and v["kill"]
    assert verdict(results[:19])["status"].startswith("incomplete")
    other = copy.deepcopy(results)
    for r in other:
        r["world"]["arm"] = "B"
    assert "kill" not in verdict(other)
    report(results, tmp_path)
    assert "P3: supported" in (tmp_path / "S1.md").read_text()
    assert pd.read_csv(tmp_path / "s1_p2.csv").empty
    for r in results:
        r["cells"] = []
    report(results, tmp_path)
    assert pd.read_csv(tmp_path / "s1_cells.csv").empty


def test_offset_denominator_and_km_boundary():
    results = primary()
    for r in results:
        r["world"]["t_rep"] = 450
        r["cells"][0]["copy_offset"] = None
        r["cells"].append(dict(confined=True, copy_offset=5, n_sites=10, **{"class": "intermediate"}))
    v = verdict(results)
    assert v["offset_eligible"] == 20 and v["offset_equal"] == 20 and not v["P4"]
    for r in results[:4]:
        r["cells"][1]["n_sites"] = 1
    assert verdict(results)["P3_offset"]  # equality exactly 0.8
    results[4]["cells"][1]["n_sites"] = 1
    assert not verdict(results)["P3_offset"]


def test_handmade_run_format_and_integrity(tmp_path):
    tape = "ea" * 16
    raw = bytes.fromhex(tape) * 64
    snapshot = "handmade.t50.u8.br"
    (tmp_path / snapshot).write_bytes(brotli.compress(raw))
    records = [dict(kind="condition", steps=50),
               dict(kind="sample", step=50, top3=[dict(tape=tape, share=1,
                    sha256=hashlib.sha256(bytes.fromhex(tape)).hexdigest())]),
               dict(kind="snapshot", step=50, file=snapshot, raw_bytes=len(raw), shape=[64, 16],
                    sha256_raw=hashlib.sha256(raw).hexdigest()),
               dict(kind="summary", steps_run=50)]
    path = tmp_path / "handmade.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    _, samples, snap = read_input(path, "s1")
    assert samples[0]["top3"][0][1] == 1 and snap["file"] == snapshot
    records[1]["top3"][0]["sha256"] = "bad"
    path.write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(ValueError, match="SHA-256"):
        read_input(path, "s1")


@pytest.mark.parametrize("machine,budget,brk,L", [
    ("m6502_pair_maps_L16", 512, "faithful", 16),
    ("m6502_pair_maps_L32", 128, "faithful", 32),
    ("m6502_brkhalt_maps_L16", 128, "halt", 16),
])
def test_full_scorer_on_handmade_records(tmp_path, machine, budget, brk, L):
    ex = E.ZilionExecutor(machine)
    provenance = ex.machine_for(L, zero_halts=brk == "halt", maps=True).describe()
    tape = "ea" * L
    raw = bytes.fromhex(tape) * 64
    filename = "handmade.t50.u8.br"
    (tmp_path / filename).write_bytes(brotli.compress(raw))
    records = [dict(kind="condition", steps=50, seed=42, L=L, budget=budget, brk=brk,
                    arm="handmade", machine_name=machine, machine=provenance),
               dict(kind="sample", step=50, top3=[dict(tape=tape, share=1,
                    sha256=hashlib.sha256(bytes.fromhex(tape)).hexdigest())]),
               dict(kind="snapshot", step=50, file=filename, raw_bytes=len(raw), shape=[64, L],
                    sha256_raw=hashlib.sha256(raw).hexdigest()),
               dict(kind="summary", steps_run=50)]
    path = tmp_path / "handmade.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    result = score_world(path, machine)
    assert result["world"]["budget"] == budget and result["world"]["n_cells"] == 64
    assert result["world"]["t_rep"] is None and result["world"]["frac_heritable"] == 0
    assert machine in result["executor"] and not result["usage"]["shader"]
    (tmp_path / filename).write_bytes(brotli.compress(raw[:-1] + b"\x00"))
    with pytest.raises(ValueError, match="snapshot SHA-256"):
        score_world(path, machine)


@pytest.mark.parametrize("prefix,arm,seed", [
    ("s6502", "DZ", 9001), ("s6502/A", "DZ", 9001),
    ("s6502_scorer_test/../s6502", "DZ", 9001),
    ("s6502_scorer_test/test", "A", 9001), ("s6502_scorer_test/test", "DZ", 65001),
])
def test_modal_test_mode_refuses_registered_inputs(prefix, arm, seed):
    with pytest.raises(ValueError):
        locations(prefix, arm, seed, False)


def test_modal_scratch_selection():
    source, output, name, machine, fmt = locations("s6502_scorer_test/test", "DZ", 9001, False)
    assert source == "s6502_scorer_test/test" and output.endswith("scored/DZ")
    assert name == "lp0_s9001" and machine == "z80c02_assay_L16" and fmt == "dz"
    assert parse_seeds("9001-9003,9010") == [9001, 9002, 9003, 9010]


def test_modal_software_vulkan_refuses_registered_before_volume_access():
    with pytest.warns(UserWarning, match="executing locally"):
        with pytest.raises(ValueError, match="cannot access registered"):
            world_cpu_test.local("s6502", "A", 65001, True, "a" * 40)
