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
    for L in ALL_TAPES:
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

ALL_TAPES = TAPES + (3, 5, 6, 7, 8, 10, 12, 18, 20, 24, 32, 50)


@pytest.mark.parametrize("stage,count,horizons,stop", [
    ("stageA", 630, {300_000}, {0.5}),
    ("stageB", 640, {300_000}, {0.5}),
    ("stageC", 490, {300_000, 1_000_000}, {-1}),
    ("stageD", 280, {300_000}, {-1}),
    ("stageE", 1230, {300_000}, {-1}),
])
def test_condition_files(stage, count, horizons, stop):
    from algocell_exp.batch import run_stem
    from make_conds import ABLATIONS, ablation_of

    conds = json.load(open(ROOT / "conds" / f"{stage}.json"))
    assert len(conds) == count
    stems = [run_stem(c) for c in conds]
    assert len(set(stems)) == len(stems), "duplicate stems"
    assert {c["horizon"] for c in conds} == horizons
    assert {c["stop_share"] for c in conds} == stop
    for c in conds:
        resolve(parse_patterns(c["suppress"]))  # must not raise
        assert c["tape"] in ALL_TAPES
        assert ablation_of(c["label"]) in ABLATIONS
        assert c["suppress"] == ";".join(ABLATIONS[ablation_of(c["label"])])
        if stop == {-1}:
            assert c["random_tapes"] == 16 and c["sample_every_early"] == 50 and c["early_until"] == 5000
            assert c["sample_steps"] == [1, 2, 3, 5, 8, 13, 21, 34] and c["snapshot_steps"][0] == 500 and c["snapshot_steps"][-1] == c["horizon"]
            assert c["seed"] >= 101 or c.get("replicate") is not None, "C/D/E must not reuse Stage A/B seeds 1-10 (except the within-seed variance arm)"
            if c["horizon"] == 1_000_000:
                assert c["sample_every"] == 1000
    # exact seed sets and byte-identity with the generator
    from make_conds import STAGES
    import json as _json
    assert _json.dumps(STAGES[stage](), indent=0) == open(ROOT / "conds" / f"{stage}.json").read(), "condition file differs from make_conds output"
    if stage in ("stageC", "stageE"):
        assert {c["seed"] for c in conds if c.get("replicate") is None and "@x32k2" not in c["label"]} == set(range(101, 111))
    if stage == "stageC":
        from algocell_exp.isa import resolve as _r, parse_patterns as _pp
        w = _r(_pp(ABLATIONS["stack-write-only"] and ";".join(ABLATIONS["stack-write-only"]))); r = _r(_pp(";".join(ABLATIONS["stack-read-only"]))); u = _r(_pp(";".join(ABLATIONS["stack-writes"])))
        for page in ("base", "cb", "ed"):
            assert w.get(page, set()) | r.get(page, set()) == u.get(page, set()) and not (w.get(page, set()) & r.get(page, set())), page


def test_stage_d_matches_plan():
    conds = json.load(open(ROOT / "conds" / "stageD.json"))
    assert {c["label"] for c in conds} == {"none", "stack-writes", "stack-write-only", "stack-read-only", "push", "call-rst-write"}
    assert {c["tape"] for c in conds} == {9}
    assert {c["seed"] for c in conds} == set(range(1001, 1021))
    main = [c for c in conds if c["z80_steps"] in (128, 512) and c["noise_exp"] == 4]
    assert len(main) == 6 * 2 * 20
    assert len([c for c in conds if c["z80_steps"] == 32]) == 20 and len([c for c in conds if c["noise_exp"] == 6]) == 20
    from algocell_exp.isa import count
    sizes = {lab: count(resolve(parse_patterns(next(c["suppress"] for c in conds if c["label"] == lab)))) for lab in ("stack-writes", "stack-write-only", "stack-read-only", "push", "call-rst-write")}
    assert sizes == {"stack-writes": 46, "stack-write-only": 22, "stack-read-only": 24, "push": 4, "call-rst-write": 17}


def test_stage_e_control_arms():
    conds = json.load(open(ROOT / "conds" / "stageE.json"))
    by = {}
    for c in conds:
        by.setdefault(c["label"].split("@")[1] if "@" in c["label"] else "", []).append(c)
    for c in by["mubyte"]:
        assert c["mutations_per_step"] == 32 * c["tape"]            # per-byte rate pinned to the L = 16 value
    for c in by["bytes"]:
        cells = c["width"] * c["height"]
        assert abs(cells * c["tape"] - 320_000) / 320_000 < 0.02      # constant total soup bytes
        assert abs(c["pairs"] / cells - 8192 / 20000) < 0.01           # constant drawn pairs per cell
        assert c["pairs"] <= 32768
    for c in by["steps8L"]:
        assert c["z80_steps"] == 8 * c["tape"]
    assert {c["tape"] for c in by["nominal"]} == set(ALL_TAPES)
    assert {c["noise_exp"] for c in by["musweep"]} == {1, 2, 3, 5, 8} and {c["tape"] for c in by["musweep"]} == {100}
    assert {c["z80_steps"] for c in by["budget"]} == {32, 64, 256, 1024, 2048} and {c["tape"] for c in by["budget"]} == {100}
    assert len(by["var"]) == 30 and {c["replicate"] for c in by["var"]} == set(range(1, 11))
    assert {c["seed"] for c in by["x32k2"]} == set(range(1001, 1021)) and {c["z80_steps"] for c in by["x32k2"]} == {32}


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
        assert 3000 < x["active_pairs"] < 6500          # 48-56% of 8192 drawn pairs survive the parallel collision claim; the fraction depends on GPU scheduling (load, grid size), which is why it is recorded per sample
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


def test_assay_many_matches_single_assay_classification():
    from algocell_exp.assay import assay_many
    rng = np.random.default_rng(5)
    tapes = np.stack([np.frombuffer(_ldir(16), dtype=np.uint8), np.frombuffer(_load_push(16), dtype=np.uint8),
                      np.zeros(16, dtype=np.uint8), rng.integers(0, 256, 16, dtype=np.uint8)])
    many = assay_many(tapes, z80_steps=128, n=32, seed=0)
    single = [assay(t.tobytes(), z80_steps=128, n=32, seed=0) for t in tapes]
    for m, s in zip(many, single):
        assert m["is_replicator"] == s["is_replicator"] and m["faithful"] == s["faithful"], (m, s)
        # identical partner draws; assay_many is A-role only, so compare with the A-role gain, not max(A, B)
        assert abs(m["gen2_score"] - s["gen2_score"]) < 1e-9 and abs(m["score"] - s["copy_into_neighbor_as_A"]) < 1e-9 and abs(m["offspring_within_q"] - s["offspring_within_q"]) < 1e-9
    assert [m["faithful"] for m in many] == [True, True, False, False]


# ── Statistics helpers (review 2026-10-07: KM tie order, medians, exact tests) ──

def test_km_and_exact_tests():
    from analyze import km_curve, km_median
    from report import fisher, sign_test
    # 10 seeds: events at 500 (x6), one at 2000, three censored at 300000
    t = np.array([500] * 6 + [2000] + [300000] * 3, dtype=float)
    e = np.array([True] * 7 + [False] * 3)
    xs, ys = km_curve(t, e)
    assert xs.tolist()[:1] == [500.0] and abs(ys[5] - 0.4) < 1e-9        # S(500) = 1 - 6/10
    assert abs(ys[6] - 0.4 * (1 - 1 / 4)) < 1e-9                          # S(2000): 4 at risk, 1 event
    assert km_median(t, e) == 500.0
    assert km_median(np.array([300000.0] * 10), np.zeros(10, bool)) == float("inf")   # NR
    # tie: a censoring at the same time as an event must still be at risk for that event
    t2 = np.array([1000.0, 1000.0]); e2 = np.array([True, False])
    assert abs(km_curve(t2, e2)[1][0] - 0.5) < 1e-9
    # Fisher: 1/10 vs 10/10 two-sided and one-sided (L = 9, 512 steps contrast)
    assert abs(fisher(10, 0, 1, 9) - 0.000119) < 2e-5
    assert abs(fisher(10, 0, 1, 9, "greater") - 0.0000595) < 1e-5
    assert abs(fisher(9, 1, 4, 6) - 0.0573) < 1e-3
    assert abs(sign_test(10, 0) - 2 / 1024) < 1e-9 and sign_test(5, 5) == 1.0



# ── Review 2 (2026-10-07): schedule with explicit early steps, interaction readbacks, Fisher at scale, long periods ──

def test_sample_schedule_merges_explicit_steps():
    from algocell_exp.run import sample_schedule
    sch = sample_schedule(1200, 500, 50, 600, (1, 2, 3, 5, 8, 13, 21, 34))
    assert sch[:8] == [1, 2, 3, 5, 8, 13, 21, 34] and sch[8:] == [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 1000, 1200]
    assert sample_schedule(1000, 250) == [250, 500, 750, 1000]
    assert sample_schedule(300, 500, 50, 5000, ()) == [50, 100, 150, 200, 250, 300]


def test_interaction_readbacks_match_the_soup():
    s = Soup(tape_length=16, seed=4)
    s.step(200)
    before = s.read_soup()
    s.step(1)
    inter = s.read_interactions()
    mem = s.read_pair_memory()
    after = s.read_soup()
    a = inter["active"]
    i, j = inter["pairs"][a, 0], inter["pairs"][a, 1]
    assert 3600 < a.sum() < 4400
    # pair memory is the post-execution state; the soup differs from it only by that step's mutations (512 bytes)
    assert int((after[i] != mem[a, 0]).sum() + (after[j] != mem[a, 1]).sum()) < 600
    # the shader's write counters are zero exactly when nothing changed in that half (a write of the same value is counted but invisible, so >= holds, not ==)
    wc = inter["write_counts"][a]
    changed_b = (mem[a, 1] != before[j]).sum(1)
    assert ((changed_b > 0) <= (wc[:, 1] > 0)).all()
    assert (wc[:, 1] >= changed_b).all()


def test_fisher_relative_tolerance_at_scale():
    from report import fisher
    assert fisher(40, 0, 0, 40) < 1e-20 and fisher(80, 0, 0, 80) < 1e-40
    assert abs(fisher(18, 2, 8, 12) - 0.0022) < 5e-4


def test_tolerant_period_reports_long_periods():
    from algocell_exp.metrics import tolerant_period
    unit = bytes(range(24))
    assert tolerant_period(unit + unit[:12])[0] == 24            # period 24 of a 36-byte tape, not "aperiodic"
    assert tolerant_period(bytes(range(50)))[0] == 50              # truly aperiodic -> L


def test_run_records_census_and_snapshots():
    import io

    buf = io.StringIO()
    snaps = {}
    s = run(tape=16, seed=5, horizon=600, sample_every=250, sample_every_early=50, early_until=300, stop_share=-1,
            sample_steps=[1, 2, 3], snapshot_steps=[300, 600], census_pairs=8, exemplar_count=5, quiet=True, out=buf,
            on_snapshot=lambda name, soup: snaps.__setitem__(name, soup.shape))
    recs = [json.loads(l) for l in buf.getvalue().splitlines() if '"kind": "sample"' in l]
    assert [r["step"] for r in recs] == [1, 2, 3, 50, 100, 150, 200, 250, 300, 500, 600]
    assert "ix_active" in recs[0] and len(recs[-1]["exemplars"]) == 5
    c = recs[-1]["census"]
    assert c["n"] > 3000 and len(c["pairs"]) == 8 and all(len(p) == 6 for p in c["pairs"])
    assert set(snaps) == {"t300", "t600", "final"} or set(snaps) == {"t300", "t600", "final", "emergence"}
