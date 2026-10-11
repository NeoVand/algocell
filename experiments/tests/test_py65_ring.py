"""Oracle boundary tests and small GPU cases that expose ring-only bus effects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("py65")

from algocell_exp.py65_ring import FIELDS, IMPLEMENTED, JAM, RingMemory, run_oracle, stack_pointer
from algocell_exp.zmachine import load_machine


def resumed_one_step(machine, memory, initial):
    record = np.zeros((1, machine.record_words), np.uint32)
    offset = machine.fields["state"]["offset"]
    for key in FIELDS:
        record[0, offset + machine.fields["state"]["items"].index(key)] = initial[key]
    return machine.run(np.array([list(memory)], np.uint8), budget=1, resume=record)


def test_16_bit_seam_precedes_ring_mapping():
    memory = RingMemory(range(33))
    assert memory[0xFFFF] == 30
    assert memory[0x10000] == 0  # not 65536 % 33 = 31
    assert memory[-1] == 30
    memory[0x10000] = 0x1FF
    assert memory[0] == 255
    assert memory[33] == 255


@pytest.mark.parametrize("ring,s", [(32, 255), (64, 255), (40, 223), (33, 238)])
def test_stack_uses_largest_page_one_alias(ring, s):
    assert stack_pointer(ring, ring - 1) == s
    memory = bytes([0x48]) + bytes(ring - 1)  # PHA
    result = run_oracle(memory, 1, initial_state={"a": 0xA5})
    assert result.memory[-1] == 0xA5
    assert result.state["s"] == (s - 1) & 255


def test_coverage_is_detected_from_installed_handlers():
    assert len(IMPLEMENTED) == 151
    assert not JAM & IMPLEMENTED
    assert 0xEA in IMPLEMENTED
    assert 0xEB not in IMPLEMENTED  # undocumented SBC immediate alias


@pytest.mark.parametrize("opcode", sorted(JAM))
def test_jam_parks_pc_and_counts_halt_instruction(opcode):
    result = run_oracle(bytes([opcode]) + bytes(31))
    assert result.halted and result.steps == 1
    assert result.state == dict(a=0, x=0, y=0, s=255, p=32, pc=0)
    assert result.unsupported is None


def test_unsupported_operand_does_not_exclude_a_program():
    result = run_oracle(bytes.fromhex("a9 eb 02") + bytes(29))
    assert result.unsupported is None
    assert result.halted and result.steps == 2 and result.state["a"] == 0xEB


def test_self_modified_unsupported_instruction_is_detected_before_execution():
    result = run_oracle(bytes.fromhex("a9 03 85 04 02") + bytes(27))
    assert result.steps == 2
    assert result.unsupported == {"step": 3, "pc": 4, "opcode": 3}
    assert result.state["pc"] == 4
    assert result.memory[4] == 3


def test_budget_edge_does_not_read_next_opcode():
    tape = bytes.fromhex("ea 03") + bytes(30)
    assert run_oracle(tape, 1).unsupported is None
    assert run_oracle(tape, 2).unsupported["opcode"] == 3


@pytest.mark.parametrize("ring", [32, 40, 64])
def test_raw_brk_matches_memory_but_diagnostic_explains_b_flag(ring):
    tape = bytes(ring)
    machine = load_machine(f"m6502_pair_maps_L{ring // 2}")
    gpu = machine.run(np.array([list(tape)], np.uint8), budget=1)
    raw, diagnostic = run_oracle(tape, 1), run_oracle(tape, 1, diagnostic=True)
    assert raw.memory == diagnostic.memory == gpu.memory[0].tobytes()
    assert raw.state["p"] ^ int(gpu.reg("p")[0]) == 0x10
    for key in FIELDS:
        assert diagnostic.state[key] == int(gpu.reg(key)[0])
    assert {event["kind"] for event in diagnostic.events} == {"status-b-bit"}


@pytest.mark.parametrize("ring", [32, 64])
def test_jsr_low_operand_is_read_before_aliased_stack_write(ring):
    tape = bytearray(ring)
    tape[0] = 0xAA  # high operand, read after pushes
    tape[-2:] = bytes([0x20, 7])  # JSR at P-2; low operand aliases first push
    initial = dict(a=0, x=0, y=0, s=255, p=32, pc=ring - 2)
    raw = run_oracle(tape, 1, initial_state=initial)
    diagnostic = run_oracle(tape, 1, diagnostic=True, initial_state=initial)
    machine = load_machine(f"m6502_pair_maps_L{ring // 2}")
    gpu = resumed_one_step(machine, tape, initial)
    assert raw.state["pc"] == 0xAA00
    assert diagnostic.state["pc"] == 0xAA07
    assert diagnostic.memory == gpu.memory[0].tobytes()
    for key in FIELDS:
        assert diagnostic.state[key] == int(gpu.reg(key)[0])
    assert diagnostic.events[0]["kind"] == "jsr-low-read-order"


@pytest.mark.parametrize("ring", [32, 64])
@pytest.mark.parametrize("program", [
    "f8 a9 99 18 69 01 02",  # SED; 99 + 01 -> 00 with carry (NMOS flags)
    "f8 a9 00 38 e9 01 02",  # 00 - 01 -> 99
    "f8 a9 ff 38 69 ff 02",  # invalid BCD input
    "f8 a9 0f 18 e9 fa 02",  # invalid BCD subtraction
    "a2 ff a9 42 95 01 b5 01 02",  # zero-page index wraps before mapping
    "6c ff 00",  # JMP ($00ff) has NMOS page wrap
    "08 28 02",  # PHP/PLP: only B differs
    "a9 01 85 0f e6 0f a5 0f 02",  # self-writing documented RMW
])
def test_targeted_multistep_programs_remain_independent_py65(ring, program):
    tape = bytes.fromhex(program)
    tape += bytes([0x02]) * (ring - len(tape))
    raw = run_oracle(tape)
    diagnostic = run_oracle(tape, diagnostic=True)
    assert raw.unsupported is None and diagnostic.unsupported is None
    machine = load_machine(f"m6502_pair_maps_L{ring // 2}")
    gpu = machine.run(np.array([list(tape)], np.uint8), budget=128)
    assert diagnostic.memory == gpu.memory[0].tobytes()
    for key in FIELDS:
        assert diagnostic.state[key] == int(gpu.reg(key)[0])
    assert not any(event["kind"].startswith("decimal") for event in diagnostic.events)


def test_saved_first_divergence_witnesses_replay_in_one_instruction():
    path = Path(__file__).resolve().parents[1] / "results/py65_difftest/examples.json"
    if not path.exists():
        pytest.skip("generated examples are absent")
    examples = json.loads(path.read_text())
    assert examples
    for key, example in examples.items():
        witness = example["first_raw_divergence"]
        tape = bytes.fromhex(witness["before_memory_hex"])
        initial = witness["before_state"]
        raw = run_oracle(tape, 1, initial_state=initial)
        assert raw.state == witness["oracle_after"], key
        machine = load_machine(f"m6502_pair_maps_L{len(tape) // 2}")
        gpu = resumed_one_step(machine, tape, initial)
        for field in FIELDS:
            assert int(gpu.reg(field)[0]) == witness["zilion_after"][field], (key, field)
        diagnostic = run_oracle(tape, 1, diagnostic=True, initial_state=initial)
        assert diagnostic.memory == gpu.memory[0].tobytes()
        for field in FIELDS:
            assert diagnostic.state[field] == int(gpu.reg(field)[0]), (key, field)


def test_comparator_detects_memory_register_and_halt_negative_controls():
    # Import the experiment script only for its comparator, not its CLI.
    from py65_difftest import differences

    tape = bytes([0x02]) + bytes(31)
    oracle = run_oracle(tape)
    machine = load_machine("m6502_pair_maps_L16")
    gpu = machine.run(np.array([list(tape)], np.uint8), budget=128)
    assert not differences(oracle, gpu, 0)
    gpu.state[0, gpu.state_names.index("a")] ^= 1
    assert differences(oracle, gpu, 0)["registers"]["a"] == {"py65": 0, "zilion": 1}
    gpu.memory[0, 17] ^= 1
    assert differences(oracle, gpu, 0)["memory"] == [{"position": 17, "py65": 0, "zilion": 1}]
    gpu.steps[0] += 1
    assert differences(oracle, gpu, 0)["halt"]["steps"] == [1, 2]
