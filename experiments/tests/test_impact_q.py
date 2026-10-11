"""Regression for repairing F while retaining the Zilog NMOS Q latch."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from algocell_exp import impact as I
from algocell_exp.zmachine import MACHINES_DIR


def _record_view(core):
    """Read layouts without creating a GPU device."""
    manifest = json.loads((MACHINES_DIR / f"imp_L16_{core}.manifest.json").read_text())
    fields = {f["name"]: f for f in manifest["record"]["fields"]}
    return I.Rec(SimpleNamespace(fields=fields)), manifest["record"]["wordsPerLane"]


@pytest.mark.parametrize("old_f,old_q,new_f,want_q", [
    (0x54, 0x54, 0x7C, 0x7C),  # BIT wrote F: repairing F must also repair Q
    (0x00, 0x00, 0x28, 0x28),  # a flags-writing instruction can produce zero F
    (0x7C, 0x00, 0x7C, 0x00),  # a non-flags-writing instruction leaves Q clear
])
def test_q_repair_preserves_flag_write_relation(old_f, old_q, new_f, want_q):
    rv0, words0 = _record_view("c0")
    rv2, words2 = _record_view("c2")
    r0 = np.zeros((1, words0), np.uint32)
    r2 = np.zeros((1, words2), np.uint32)
    r0[0, rv0.off["f"]] = new_f
    r2[0, rv2.off["f"]] = old_f
    r2[0, rv2.off["q"]] = old_q
    repaired = rv2.set_common(r2, rv0, r0)
    assert repaired[0, rv2.off["q"]] == want_q
    assert repaired[0, rv2.off["f"]] == new_f
    assert r2[0, rv2.off["q"]] == old_q  # input record remains available to probes


@pytest.mark.parametrize("attributor", [I.Attributor, I.FastAttributor])
def test_q_repair_does_not_invent_scf_ccf_events(attributor):
    """BIT immediately followed by SCF/CCF clears its X/Y difference in both cores.

    In a pair with an earlier wasted prefix, a stale Q creates a false later
    propagating event that steals the attribution from that real cause.
    Exercise both the CPU repair and the GPU repair kernel, plus census events
    for unaffected pairs and a genuine Q-latch event that must remain visible.
    """
    pairs = []
    for opcode in (0x37, 0x3F):
        for prefix in ([], [0xDD, 0xFD, 0x00]):
            pair = np.zeros(32, np.uint8)
            code = prefix + [0x21, 0x18, 0x00, 0xCB, 0x46, opcode, 0xF5,
                             0xED, 0x5F, 0x32, 0x1B, 0x00, 0x76]
            pair[:len(code)] = code
            pair[24] = 0x28
            pairs.append(pair)
    # OR A writes F/Q; LD A clears Q, so the following SCF really differs.
    genuine = np.zeros(32, np.uint8)
    genuine[:9] = [0x3E, 0x28, 0xB7, 0x3E, 0x00, 0x37, 0xF5, 0x76, 0x00]
    pairs.append(genuine)
    pairs = np.stack(pairs)
    tr = I.Triple("L16")
    budget = 12
    measured = tr.run(pairs, budget)
    assert I.affected(measured)["c2"].tolist() == [False, True, False, True, True]
    out = attributor(tr, budget).attribute(pairs, np.arange(len(pairs)), measured, census=True)["c2"]
    for i in (1, 3, 0, 2):
        expected = ("wasted-prefix", 1) if i % 2 else (None, None)
        assert (out[i].decisive_class, out[i].decisive_step) == expected
        assert all(e.cls != "scf-ccf" for e in out[i].events), (i, out[i])
    assert (out[4].decisive_class, out[4].decisive_step) == ("scf-ccf", 4)
