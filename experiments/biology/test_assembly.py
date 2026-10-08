"""Tests for the Re-Pair assembly-index bound and the assembly sum (BIOLOGY_PREREG.md, section A).

    .venv/bin/python -m pytest biology/test_assembly.py -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biology.assembly import a_hex, assembly, repair_index, sample_assembly, self_test  # noqa: E402


def test_prereg_worked_examples():
    assert repair_index(bytes.fromhex("01 c5" * 8)) == 4
    assert repair_index(bytes(16)) == 4
    assert repair_index(bytes(range(16))) == 15


def test_self_test_runs():
    self_test()


def test_edge_cases_and_small_strings():
    assert repair_index(b"") == 0
    assert repair_index(b"a") == 0
    assert repair_index(b"ab") == 1
    assert repair_index(b"aa") == 1          # one pair occurring once: no rule, two symbols
    assert repair_index(b"aaa") == 2
    assert repair_index(b"aaaa") == 2        # aa -> X, XX -> Y
    assert repair_index(b"abab") == 2
    assert repair_index(b"abcabc") == 3      # ab -> X, Xc -> Y, YY -> Z


def test_periodic_tapes_of_stage_g_lengths():
    # period-2 tapes: a = 1 + ceil-ish doubling chain; values checked by hand
    assert repair_index(bytes.fromhex("01 c5" * 10)) == 5     # L = 20: X x10 -> Y x5 -> Z Z Y
    assert repair_index(bytes.fromhex("01 c5" * 25)) == 7     # L = 50
    assert repair_index(bytes.fromhex("01 c5" * 32)) == 6     # L = 64: pure doubling to one symbol
    assert repair_index(bytes(20)) == 5
    assert repair_index(bytes(64)) == 6


def test_upper_bound_never_exceeds_trivial_bound_and_is_monotone_in_randomness():
    rng = np.random.default_rng(1)
    for L in (16, 20, 50, 64, 100):
        for _ in range(20):
            s = bytes(rng.integers(0, 256, L, dtype=np.uint8))
            a = repair_index(s)
            assert 0 <= a <= L - 1
        per = bytes([1, 2] * (L // 2))
        assert repair_index(per) < repair_index(bytes(rng.integers(0, 256, L, dtype=np.uint8))) or L < 8


def test_tie_break_first_occurrence():
    # (a,b) and (c,d) both occur twice; the earliest first occurrence (a,b) is replaced first.  The final index is
    # the same either way (2 rules + 3 symbols), so only check the value is as expected.
    # recomputed by hand: "abcdabcd": pairs ab(2) bc(2) cd(2) da(1); ab -> X: X c d X c d; Xc(2) -> Y: Y d Y d;
    # Yd(2) -> Z: Z Z; ZZ(1) -> stop? count of (Z,Z) is 1 -> stop: rules 3, length 2 -> 4.
    assert repair_index(b"abcdabcd") == 4


def test_assembly_sum():
    assert assembly([4], [1], 20000) == 0.0
    assert abs(assembly([4, 9], [3, 2], 20000) - (np.exp(4) * 2 + np.exp(9)) / 20000) < 1e-12
    r = {"top10_shares": [0.5, 0.25, 0.00005], "exemplars": [{"tape": "01 c5 " * 7 + "01 c5"}, {"tape": "00 " * 15 + "00"}, {"tape": " ".join(f"{i:02x}" for i in range(16))}]}
    A, a_modal, n_ge2, top = sample_assembly(r, 20000)
    assert a_modal == 4 and n_ge2 == 2 and top == 0.5
    assert abs(A - (np.exp(4) * 9999 + np.exp(4) * 4999) / 20000) < 1e-9
    assert a_hex("01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5") == 4
