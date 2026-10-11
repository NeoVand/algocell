"""The GPU tile census (algocell_exp/zcensus.py). Needs the local GPU; under a minute.

(a) the Z80 compat 0.2 machine reproduces the paper's period-2 census (Proposition 3: `01 c5`, `11 d5`, `21 e5`, `2a e5`,
    `e5 2a`, plus the zero word), all open;
(b) the harness finds a seeded 6502 closed copier tiled at period 6, exact into the zero partner and every random
    partner, and closed;
(c) compacted, resumed segments (32 steps) give, at 128 and at 512 steps, what one direct ZMachine.run gives, on
    every recorded tile (records include a hash-chosen sample), on both 6502 arms;
(d) the generator's tile placement and partners: tape A is the tiled word, B the zero or hash partner (numpy mirror);
(e) counters: every lane and tile is counted once;
(f) census6502.run_census: a run stopped after one chunk and resumed equals one uninterrupted run.
"""

from __future__ import annotations

import numpy as np
import pytest

from algocell_exp.zcensus import TileCensus, evaluate, partner_bytes, tile_tapes
from algocell_exp.zmachine import load_machine


def _exact_words(c: TileCensus, out: dict, b: int) -> set[str]:
    d = c.decode_hits(out[b]["hits"])
    return {f"{int(t):0{2 * c.k}x}" for t in d["t"][(d["flags"] & 1) != 0]}


def test_z80_period2_census():
    c = TileCensus("z80c02_pair_maps_L16", k=2, nrand=16, budgets=(128,), seg=128, sample_bits=8)
    out = c.run_range(0, 1 << 16)
    assert _exact_words(c, out, 128) == {"0000", "01c5", "11d5", "21e5", "2ae5", "e52a"}
    d = c.decode_hits(out[128]["hits"])
    sel = np.isin(d["t"], [0x01C5, 0x11D5, 0x21E5, 0x2AE5, 0xE52A])
    assert d["entered"][sel].any(1).all()          # all five open
    rc = c.recheck(out[128]["hits"], 128)
    assert rc["tiles"] > 200 and sum(v for k, v in rc.items() if k.startswith("mismatch")) == 0


@pytest.mark.parametrize("machine", ["m6502_pair_maps_L16", "m6502_brkhalt_maps_L16"])
def test_6502_planted_closer(machine):
    c = TileCensus(machine, k=6, nrand=16, budgets=(128, 512), seg=64, sample_bits=10)
    t = 0xB50F48CA50FA                      # LDA $0F,X; PHA; DEX; BVC -6
    out = c.run_range(t - 2000, t + 2000)
    for b in (128, 512):
        d = c.decode_hits(out[b]["hits"])
        i = np.flatnonzero(d["t"] == t)
        assert len(i) == 1
        assert d["flags"][i[0]] & 1 and d["n_exact"][i[0]] == 16 and not d["entered"][i[0]].any()


@pytest.mark.parametrize("machine", ["m6502_pair_maps_L16", "m6502_brkhalt_maps_L16"])
def test_segments_equal_direct_runs(machine):
    c = TileCensus(machine, k=2, nrand=16, budgets=(128, 512), seg=32, sample_bits=4)
    out = c.run_range(0, 1 << 16)
    for b in (128, 512):
        C = out[b]["counters"]
        assert C["tiles"] == 1 << 16 and C["zero_lanes"] == 1 << 16 and C["rand_lanes"] == 16 << 16
        assert sum(C[f"zero_match{m}"] for m in range(17)) == 1 << 16
        rc = c.recheck(out[b]["hits"], b)
        assert rc["tiles"] > 3000, rc
        assert sum(v for k, v in rc.items() if k.startswith("mismatch")) == 0, rc


def test_generator_placement():
    """The checker's 'B changed' counter: with the zero partner, a tile that never writes leaves B zero; the CPU
    mirror of tape A and the partners is what evaluate() runs, and its scores agree with the GPU (test above)."""
    m = load_machine("m6502_pair_maps_L16")
    t = np.array([0xEA4C0000], np.uint64)   # NOP; JMP $0000: loops at 0-3 for 128 steps, writes nothing
    e = evaluate(m, 4, t, 4, 128)
    assert (e.b_final[0, 0] == 0).all() and (e.a_final[0] == tile_tapes(t, 4, 16)[0]).all()
    for v in range(1, 5):
        assert (e.b_final[0, v] == partner_bytes(t, v, 16, panel=False)[0]).all()
    assert (e.steps == 128).all() and not e.entered.any()
    c = TileCensus(m, k=4, nrand=4, budgets=(128,), seg=128, sample_bits=0)
    out = c.run_range(0xEA4C0000, 0xEA4C0001)
    C = out[128]["counters"]
    assert C["zero_b_changed"] == 0 and C["rand_b_changed"] == 0 and C["zero_steps_budget"] == 1 and C["rand_steps_budget"] == 4
    d = c.decode_hits(out[128]["hits"])
    assert d["rot"][0, 0] == 8 and not d["entered"].any()   # the untouched zero partner matches A's eight zero bytes


def test_checkpoint_resume(tmp_path, monkeypatch):
    """census6502.run_census stopped after its first chunk and resumed gives the same totals and records as one
    uninterrupted run (and refuses a different chunk size)."""
    import census6502 as C6

    monkeypatch.setattr(C6, "CKPT", tmp_path / "a")
    p = C6.run_census("m6502_brkhalt_maps_L16", 2, chunk_bits=12, max_minutes=1e-9, pause=0)
    part = C6.load_census(p)
    assert part["chunks_done"] == 1 and not part["complete"]
    with pytest.raises(RuntimeError):
        C6.run_census("m6502_brkhalt_maps_L16", 2, chunk_bits=13, pause=0)
    C6.run_census("m6502_brkhalt_maps_L16", 2, chunk_bits=12, pause=0)
    resumed = C6.load_census(p)
    monkeypatch.setattr(C6, "CKPT", tmp_path / "b")
    whole = C6.load_census(C6.run_census("m6502_brkhalt_maps_L16", 2, chunk_bits=12, pause=0))
    assert resumed["complete"] and resumed["chunks_done"] == 16
    for b in (128, 512):
        assert resumed["budgets"][b]["counters"] == whole["budgets"][b]["counters"]
        key = lambda H: sorted(map(tuple, H.tolist()))  # noqa: E731
        assert key(resumed["budgets"][b]["hits"]) == key(whole["budgets"][b]["hits"])
