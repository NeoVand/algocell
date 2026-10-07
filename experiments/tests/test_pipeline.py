"""Regression tests for failure classes that have bitten the sweep pipeline.

Each test names the incident it guards against. They need the local GPU
(wgpu) and take ~30 s in total; run with `.venv/bin/python -m pytest tests -q`.
"""

import itertools
import json
import re
from pathlib import Path

import numpy as np
import pytest

from algocell_exp.assay import assay
from algocell_exp.isa import parse_patterns, resolve
from algocell_exp.metrics import minimal_period
from algocell_exp.run import run
from algocell_exp.soup import SHADER_DIR, Soup

ROOT = Path(__file__).resolve().parents[1]
TAPES = (4, 9, 16, 25, 36, 49, 64, 81, 100)


# ── Incident: Stage C stopped at 2,500 steps because `stop_share: -1` was treated as a share ──

def test_stop_share_nonpositive_means_never():
    for ss in (-1, 0, None):
        s = run(tape=16, seed=1, horizon=1000, sample_every=500, stop_share=ss, quiet=True)
        assert s["steps_run"] == 1000, (ss, s["steps_run"])


def test_stop_share_positive_stops_after_stop_after_samples():
    # Any q_share >= 1e-9 triggers at the first sample (step 500); stop 4 samples later = 2500.
    s = run(tape=16, seed=1, horizon=4000, sample_every=500, stop_share=1e-9, stop_after=4, quiet=True)
    assert s["steps_run"] == 2500, s["steps_run"]


# ── Incident: assay executor had a fixed 40-byte memory; every assay for L >= 25 was garbage ──

def _load_push(L: int) -> bytes:
    # LD BC,$C501 ; PUSH BC  → writes `01 c5` pairs; tiles any L.
    return (bytes.fromhex("01c5") * (L // 2 + 1))[:L]


def _ldir(L: int) -> bytes:
    """A tiled LDIR shift-copier whose period divides both L and 2L (phase-consistent self-overlap).
    L = 9 uses the evolved 3-byte unit `DEC E ; LDIR` (DE = 0x00FF ≡ 3 mod 18) because no 4-byte unit tiles 9."""
    if L == 9:
        return bytes.fromhex("1dedb0") * 3
    off = next(o for o in range(4, L + 1) if L % o == 0)
    unit = bytes.fromhex(f"1e{off:02x}edb0") + bytes(off - 4)
    return (unit * (L // off + 1))[:L]


@pytest.mark.parametrize("L", TAPES)
def test_ldir_copier_scores_one_at_every_tape_length(L):
    steps = 512 if L > 36 else 128  # one LDIR iteration per step; a 100-byte partner needs > 128
    r = assay(_ldir(L), z80_steps=steps, n=32, seed=0)
    assert r["score"] > 0.9 and r["gen2_score"] > 0.9 and r["offspring_within_q"] > 0.9, (L, r)


@pytest.mark.parametrize("L", (16, 25, 36, 49, 64, 81, 100))
def test_load_push_copier_is_heritable_for_L_ge_16(L):
    # `01 c5` tiles copy 2 bytes per PUSH into the partner's tail and keep pushing as execution wraps the
    # ring; measured 2026-10-07: score 0.70–0.91, gen2 0.39–0.76. At L = 4 and 9 the same tape is a poor
    # copier (gen2 0.39 / 0.27: too few pushes before the partner's random head executes), matching the
    # absence of the stack family at those sizes in Stage B — so those sizes are deliberately not asserted.
    steps = 512 if L > 36 else 128
    r = assay(_load_push(L), z80_steps=steps, n=32, seed=0)
    assert r["score"] > 0.6 and r["gen2_score"] > 0.35, (L, r)


@pytest.mark.parametrize("L", (4, 16, 100))
def test_random_and_constant_tapes_do_not_score(L):
    rng = np.random.default_rng(3)
    for _ in range(3):
        r = assay(rng.integers(0, 256, L, dtype=np.uint8).tobytes(), z80_steps=128, n=32, seed=0)
        assert abs(r["score"]) < 0.15 and abs(r["gen2_score"]) < 0.15, (L, r)
    # The all-NOP tape is the assay's worst-case null: random partners push their zeroed registers, so the
    # partner drifts towards zeros by itself. Measured gen2 0.03–0.17 across L (2026-10-07); the heritability
    # threshold 0.3 must stay clear of it.
    r = assay(bytes(L), z80_steps=128, n=32, seed=0)
    assert r["score"] < 0.2 and r["gen2_score"] < 0.25, r


def test_executor_exists_for_every_tape_length():
    for L in TAPES:
        assert (SHADER_DIR / f"z80_test_L{L}.wgsl").exists(), L
        assert (SHADER_DIR / f"sim_square_L{L}.wgsl").exists(), L


# ── Params uniform layout: host writes must match the shader struct field order ──

def test_params_layout_matches_shader_struct():
    wgsl = (SHADER_DIR / "sim_square_L16.wgsl").read_text()
    struct = re.search(r"struct Params \{(.*?)\n\}", wgsl, re.S).group(1)
    fields = re.findall(r"^\s*(\w+):", struct, re.M)
    assert fields == ["soup_width", "soup_height", "tape_length", "pair_length", "pair_count", "mutation_count", "z80_steps", "batch_seed", "suppress"], fields
    s = Soup(width=16, height=8, tape_length=16, seed=1, pair_count=64, z80_steps=7, noise_exp=2, suppress="family:block-copy")
    p = s._params(0xDEADBEEF)
    assert p.nbytes == 128
    assert p[:8].tolist() == [16, 8, 16, 32, 64, 64 // 4, 7, 0xDEADBEEF]
    assert p[8:].any()  # masks present
    # ED-page mask (words 16..23 of the 24 mask words) must carry LDI/LDD/LDIR/LDDR = ED A0/A8/B0/B8
    ed = p[8 + 16 : 8 + 24]
    for op in (0xA0, 0xA8, 0xB0, 0xB8):
        assert (ed[op >> 5] >> (op & 31)) & 1, hex(op)


# ── Condition files: counts, uniqueness, resolvable suppression, pre-registered horizons ──

@pytest.mark.parametrize("stage,count,horizons,stop", [
    ("stageA", 630, {300_000}, {0.5}),
    ("stageB", 640, {300_000}, {0.5}),
    ("stageC", 420, {300_000, 1_000_000}, {-1}),
    ("stageD", 160, {300_000}, {-1}),
])
def test_condition_files(stage, count, horizons, stop):
    conds = json.load(open(ROOT / "conds" / f"{stage}.json"))
    assert len(conds) == count
    keys = [(c["label"], c["tape"], c["z80_steps"], c["noise_exp"], c["seed"]) for c in conds]
    assert len(set(keys)) == len(keys), "duplicate conditions"
    assert {c["horizon"] for c in conds} == horizons
    assert {c["stop_share"] for c in conds} == stop
    for c in conds:
        resolve(parse_patterns(c["suppress"]))  # must not raise
        assert c["tape"] in TAPES
        if stop == {-1}:
            assert c["random_tapes"] == 8


def test_stage_d_matches_plan():
    conds = json.load(open(ROOT / "conds" / "stageD.json"))
    assert {c["label"] for c in conds} == {"none", "stack-writes", "push-only", "call-rst"}
    assert {c["tape"] for c in conds} == {9}
    assert {c["z80_steps"] for c in conds} == {128, 512}
    assert {c["noise_exp"] for c in conds} == {4}
    assert {c["seed"] for c in conds} == set(range(1, 21))


# ── Metrics ──

def test_minimal_period_is_non_cyclic():
    assert minimal_period(bytes.fromhex("01c5" * 8)) == 2
    assert minimal_period(bytes.fromhex("01c5" * 4 + "01")) == 2  # odd length, still period 2
    assert minimal_period(bytes.fromhex("01c5" * 7 + "0000")) == 16  # end defect → aperiodic
    assert minimal_period(bytes(5)) == 1


# ── Instrumented run loop (review 2026-10-07): fine early sampling, byte histogram, active pairs, provenance ──

def test_run_loop_schedule_and_fields():
    import io

    buf = io.StringIO()
    s = run(tape=16, seed=2, horizon=1000, sample_every=250, sample_every_early=50, early_until=500,
            stop_share=-1, random_tapes=8, mutations_per_step=None, quiet=True, out=buf,
            provenance={"git_commit": "test"})
    lines = [json.loads(l) for l in buf.getvalue().splitlines()]
    cond = lines[0]
    samples = [l for l in lines if l["kind"] == "sample"]
    assert cond["kind"] == "condition" and cond["provenance"]["git_commit"] == "test"
    assert len(cond["provenance"]["shader_sha256_16"]) == 16 and cond["provenance"]["versions"]["wgpu"]
    assert [x["step"] for x in samples] == list(range(50, 501, 50)) + [750, 1000]
    for x in samples:
        assert sum(x["byte_hist"]) == 20000 * 16
        assert 0.0 <= x["zero_frac"] <= 1.0 and abs(x["zero_frac"] - x["byte_hist"][0] / (20000 * 16)) < 1e-9
        assert 3600 < x["active_pairs"] < 4400          # measured on GPU: 3982 ± 35 of 8192 drawn pairs survive the parallel collision claim (48.6%)
        assert 0.0 <= x["q_shift_share"] <= 1.0 and x["q_shift_n"] == 2000
        assert len(x["random_tapes"]) == 8
        for ex in x["exemplars"]:
            assert "mechanisms" in ex
    assert s["steps_run"] == 1000 and s["samples"] == len(samples) == 12
    assert sum(s["final"]["byte_hist"]) == 20000 * 16 and s["final"]["q_shift_n"] == 20000


def test_mutation_override_is_recorded_and_applied():
    s = run(tape=16, seed=2, horizon=100, sample_every=100, stop_share=-1, quiet=True, mutations_per_step=4096)
    assert s["mutations_per_step"] == 4096 and s["provenance"]["mutations_per_step_override"] == 4096
    s0 = run(tape=16, seed=2, horizon=100, sample_every=100, stop_share=-1, quiet=True)
    assert s0["mutations_per_step"] == 512 and s0["provenance"]["mutations_per_step_override"] is None


def test_assay_reports_faithfulness_and_vulnerability():
    r = assay(_ldir(16), z80_steps=128, n=32, seed=0)
    assert r["is_replicator"] and r["faithful"] and r["gen2_cond"] > 0.9 and r["n_informative"] == 32
    assert 0.0 <= r["self_preserved_as_B"] <= 1.0
    smear = assay(bytes.fromhex("ff" + "4100" * 7 + "41"), z80_steps=128, n=64, seed=0)  # RST return-address smear
    assert not smear["is_replicator"] and not smear["faithful"] and smear["offspring_within_q"] > 0.5
    # saturated in-situ soup: every partner already a copy -> NaN, never 0
    T = _ldir(16)
    sat = np.repeat(np.frombuffer(T, dtype=np.uint8)[None, :], 200, axis=0)
    r = assay(T, z80_steps=128, n=16, seed=0, neighbors=sat)
    assert r["n_informative"] == 0 and np.isnan(r["score"]) and np.isnan(r["gen2_score"])


def test_tolerant_period_and_shift_occupancy():
    from algocell_exp.metrics import shift_occupancy, tolerant_period
    assert tolerant_period(bytes.fromhex("01c5" * 40 + "01"))[0] == 2
    assert tolerant_period(bytes.fromhex("01c5" * 39 + "000001"))[0] == 2      # end defect tolerated
    assert tolerant_period(bytes(range(50)))[0] == 50                            # aperiodic
    T = np.frombuffer(bytes.fromhex("01c5" * 8), dtype=np.uint8)
    soup = np.concatenate([np.tile(T, (100, 1)), np.tile(np.roll(T, 1), (100, 1)), np.random.default_rng(0).integers(0, 256, (200, 16), dtype=np.uint8)])
    assert abs(shift_occupancy(soup, T)["q_shift_share"] - 0.5) < 0.01       # both phases count
