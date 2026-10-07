"""Checks that the Python host matches the TypeScript app where it must:
PRNG stream, suppression-mask resolution (golden vectors exported from
z80-opcodes.ts), and the disassembler on known replicators."""

import json
from pathlib import Path

import numpy as np

from algocell_exp.isa import disassemble, masks, mechanisms, resolve
from algocell_exp.prng import SplitMix64

ROOT = Path(__file__).resolve().parents[1]


def test_prng_matches_ts():
    # From `npx tsx -e` against src/lib/sim/prng.ts
    r = SplitMix64(6)
    assert [r.next_u32() for _ in range(6)] == EXPECTED_SEED6
    r = SplitMix64(123456789)
    assert [r.next_u32() for _ in range(3)] == EXPECTED_SEED123456789


def test_masks_match_golden():
    isa = json.load(open(ROOT / "algocell_exp" / "shader" / "isa.json"))
    for g in isa["golden"]:
        got = masks(resolve(g["patterns"]))
        assert got.tolist() == g["masks"], g["patterns"]


def test_disassembly_and_mechanisms():
    classic = bytes.fromhex("21e3" * 8)  # LD HL,nn ; EX (SP),HL
    d = disassemble(classic)
    assert d[0]["mnemonic"] == "LD HL,nn" and d[1]["mnemonic"] == "EX (SP),HL"
    assert mechanisms(classic) == ["stack"]
    assert mechanisms(bytes.fromhex("01c5" * 8)) == ["stack"]  # LD BC,nn ; PUSH BC
    ldir = bytes.fromhex("1e20edb0") + bytes(12)
    assert [x["mnemonic"] for x in disassemble(ldir)[:2]] == ["LD E,n", "LDIR"]
    assert mechanisms(ldir) == ["block-copy"]
    assert mechanisms(bytes.fromhex("3634" + "00" * 14)) == ["ld-mem"]  # LD (HL),n
    assert mechanisms(bytes.fromhex("cbc6" + "00" * 14)) == ["rmw"]  # SET 0,(HL)
    assert mechanisms(bytes(16)) == []


EXPECTED_SEED6 = [2918178816, 961666969, 1923755846, 3989887632, 3629453095, 3608619632]
EXPECTED_SEED123456789 = [1038841465, 963767854, 1083899365]
