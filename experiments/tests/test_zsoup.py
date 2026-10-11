"""ZSoup and the generic Zilion host (zmachine). Needs the local GPU; about a minute.

(a) extraction: every item ZSoup keeps from the exported simulation shader (square and lethal, L = 16 and 32) is the
    source text byte for byte, unique in the source; the parser accounts for every byte of code; nothing of the Z80 is
    kept; the ZSoup shader is exactly the kept items plus the bridge;
(b) zmachine against Algocell's single-pair executors on random pairs: plain and lethal, L = 16 and 32, random
    suppression masks, per-lane budgets (0 = untouched), fetch and start maps against the op-traced executor;
(c) the vendored machines are what build_machines.mjs generates from the Zilion checkout (skipped without one);
(d) ZSoup has Soup's public API with the same signatures, the same initial soup, readbacks and write_soup;
(e) ZSoup's own pipeline: after each step its pair memory, write counts and lane records equal a direct run of the
    machine on the pairs the step drew (Z80 and the full 6502 core);
(f) the step-locked harness: 0 mismatches over 300 steps (plain and lethal), CPU recount = comparator, and a
    one-instruction budget difference is detected;
(g) refusals: suppression masks on a machine without an ablation table, a non-square grid, run.py disagreements;
    run.py with --machine records the machine.
"""

from __future__ import annotations

import inspect
import io
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from algocell_exp import zsoup_check as C
from algocell_exp.isa import masks as make_masks
from algocell_exp.isa import resolve
from algocell_exp.soup import SHADER_DIR, Soup
from algocell_exp.zmachine import MACHINES_DIR, load_machine
from algocell_exp.zsoup import ZBridge, ZSoup, bridge_wgsl, extract_soup_kernels, parse_wgsl_items

Z80_NAMES = re.compile(r"\b(cpu_\w+|z80_(?!steps\b)\w+|mem_read|mem_write|MEM_LENGTH|load_pair_mem|store_pair_mem|sp_init|on_fetch_opcode)\b")


# ── (a) extraction ──────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["sim_square_L16.wgsl", "sim_lethal_L16.wgsl", "sim_square_L32.wgsl", "sim_lethal_L32.wgsl"])
def test_kept_items_are_verbatim(name):
    src_path = SHADER_DIR / name
    composed, kept, src = extract_soup_kernels(src_path)
    assert src == src_path.read_text()
    names = [it.name for it in kept]
    for k in ("clear_collision", "prepare_batch", "absorb_results", "mutate_soup", "count_bytes", "clear_byte_counts", "hash_cells"):
        assert k in names
    assert sorted(it.binding for it in kept if it.kind == "binding") == list(range(10))
    for it in kept:
        assert src[it.start: it.end] == it.text, it.name
        assert src.count(it.text) == 1, it.name
        assert it.text in composed, it.name
        assert not Z80_NAMES.search(re.sub(r"//[^\n]*", "", it.text)), f"{it.name} references Z80 code"
    assert "z80_execute_batch" not in names and "z80_step" not in names
    assert set(names) == {"Params", "soup", "pairs", "pair_data", "write_counts", "rng_states", "params", "pair_active", "byte_counts",
                          "collision_mask", "cell_hashes", "pcg", "rng", "rand", "rand_bounded", "clear_collision", "prepare_batch",
                          "absorb_results", "mutate_soup", "count_bytes", "clear_byte_counts", "hash_cells"}
    # the composed text is the kept items, in source order, and nothing else (after its comment header)
    body = composed[composed.index("\n\n") + 2:]
    assert body == "\n\n".join(it.text for it in kept) + "\n"
    assert [it.start for it in kept] == sorted(it.start for it in kept)


@pytest.mark.parametrize("name", ["sim_square_L16.wgsl", "sim_lethal_L32.wgsl"])
def test_parser_accounts_for_every_byte(name):
    src = (SHADER_DIR / name).read_text()
    items = parse_wgsl_items(src)
    pos = 0
    for it in items:
        gap = re.sub(r"//[^\n]*", "", src[pos: it.start])
        assert gap.strip() == "", f"code outside any item before {it.name}: {gap.strip()[:60]!r}"
        pos = it.end
    assert re.sub(r"//[^\n]*", "", src[pos:]).strip() == ""


def test_square_and_lethal_kernels_identical():
    t = lambda n: [it.text for it in extract_soup_kernels(SHADER_DIR / n)[1]]  # noqa: E731
    assert t("sim_square_L16.wgsl") == t("sim_lethal_L16.wgsl") == t("sim_square_L32.wgsl") == t("sim_lethal_L32.wgsl")


def test_zsoup_shader_is_kept_items_plus_bridge():
    m = load_machine("z80c02_pair_L16")
    br = ZBridge(m)
    kept_text, _, _ = extract_soup_kernels(br.source)
    assert br.source.name == "sim_square_L16.wgsl"
    assert br.wgsl == kept_text + bridge_wgsl(m)
    assert ZBridge(load_machine("z80c02_lethal_L16")).source.name == "sim_lethal_L16.wgsl"


# ── (b) zmachine against Algocell's executors ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("L,lethal", [(16, False), (32, False), (16, True), (32, True)])
def test_zmachine_matches_executor_random(L, lethal):
    rng = np.random.default_rng(L * 10 + lethal)
    pairs = rng.integers(0, 256, size=(4096, 2 * L), dtype=np.uint8)
    pairs[::7, :8] = 0   # some zero runs, so the lethal rule fires
    m = load_machine(f"z80c02_{'lethal' if lethal else 'pair'}_L{L}")
    for sup in ([], ["family:block-copy", "PUSH"]):
        r = C.compare_pairs(pairs, L, 128, sup, lethal, m)
        assert r["mem_changed"] > 1000 and r["b_written"] > 1000
        assert (r["mem_mismatch"], r["wc_mismatch"], r["reg_mismatch"]) == (0, 0, 0), r
    if lethal:
        assert r["zil_halted"] > 500


def test_zmachine_random_masks_and_per_lane_budgets():
    L = 16
    rng = np.random.default_rng(5)
    pairs = rng.integers(0, 256, size=(2048, 2 * L), dtype=np.uint8)
    masks = np.packbits(rng.random((24, 32)) < 1 / 6, axis=1, bitorder="little").view(np.uint32).reshape(24)
    m = load_machine("z80c02_pair_L16")
    a_mem, a_regs = C.algocell_execute(pairs, L, 128, masks, False)
    budgets = np.where(np.arange(2048) % 3 == 0, 0, 128)
    r = m.run(pairs, budget=budgets, ablation=masks)
    run = budgets > 0
    assert np.array_equal(r.memory[run], a_mem[run])
    assert np.array_equal(r.writes("A")[run], a_regs[run, 10]) and np.array_equal(r.writes("B")[run], a_regs[run, 11])
    assert np.array_equal(r.memory[~run], pairs[~run]) and (r.steps[~run] == 0).all() and (r.write_counts[~run] == 0).all()
    assert (r.steps[run] > 0).all()
    with pytest.raises(ValueError):
        m.run(pairs)   # a per-lane-budget machine has no default budget


def test_zmachine_maps_match_optraced_executor():
    L = 16
    rng = np.random.default_rng(9)
    pairs = rng.integers(0, 256, size=(4096, 2 * L), dtype=np.uint8)
    r = C.compare_pairs(pairs, L, 128, [], False, load_machine("z80c02_pair_maps_L16"), traced=True)
    assert (r["mem_mismatch"], r["wc_mismatch"], r["reg_mismatch"], r["fetch_map_mismatch"], r["start_map_mismatch"]) == (0, 0, 0, 0, 0), r
    assert r["fetched_positions_mean"] > 4


def test_zmachine_record_decoding():
    m = load_machine("z80c02_pair_L16")
    pairs = np.zeros((2, 32), dtype=np.uint8)
    pairs[0, 0] = 0x76          # HALT at 0
    pairs[1, :3] = (0x3E, 0x42, 0x76)   # LD A,0x42; HALT
    r = m.run(pairs, budget=128)
    assert list(r.steps) == [1, 2] and list(r.halt_reason) == [1, 1]
    assert list(r.trap_cause) == [m.m["trapCauses"]["halt-insn"]] * 2
    assert r.reg("a")[1] == 0x42 and list(r.reg("pc")) == [0, 2] and list(r.reg("sp")) == [0xFFFF, 0xFFFF]
    assert r.fetch_map is None and r.start_map is None and r.write_count_names == ("A", "B")


# ── (c) vendored machines ───────────────────────────────────────────────────────────────────────────────────────────
def test_vendored_machines_regenerate():
    zdir = Path(__file__).resolve().parents[3] / "zilion"
    if not (zdir / "dist" / "index.js").exists() or shutil.which("node") is None:
        pytest.skip("no Zilion build beside the repository")
    recorded = re.search(r"commit `([0-9a-f]{40})`", (MACHINES_DIR / "PROVENANCE.md").read_text()).group(1)
    head = subprocess.run(["git", "-C", str(zdir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != recorded:
        pytest.skip(f"Zilion checkout at {head[:12]}, machines vendored from {recorded[:12]} (run build_machines.mjs --check by hand)")
    p = subprocess.run(["node", str(MACHINES_DIR / "build_machines.mjs"), "--check", "--zilion", str(zdir)], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


@pytest.mark.parametrize("man", sorted(MACHINES_DIR.glob("*.manifest.json")), ids=lambda p: p.stem)
def test_vendored_manifests_match_wgsl(man):
    # Separate cases keep GPU verification splittable as the machine registry grows.
    load_machine(man.name[: -len(".manifest.json")])   # checks sha256 and compiles


# ── (d) API ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_api_parity():
    public = [n for n, v in inspect.getmembers(Soup) if not n.startswith("_") and callable(v)]
    for n in public:
        assert hasattr(ZSoup, n), n
        if n in ("step", "set_suppress"):   # overridden with the same signature
            assert inspect.signature(getattr(ZSoup, n)) == inspect.signature(getattr(Soup, n)), n
        else:                               # every other public method is Soup's own code
            assert getattr(ZSoup, n) is getattr(Soup, n), f"{n} is overridden"
    assert set(public) >= {"reset", "step", "read_soup", "write_soup", "read_interactions", "read_pair_memory", "read_byte_counts", "read_hashes", "read_active_pairs"}


def test_same_initial_soup_and_readbacks():
    s, z = Soup(seed=11), ZSoup("z80c02_pair_L16", seed=11)
    assert np.array_equal(s.read_soup(), z.read_soup())
    # The statistics kernels read the soup dimensions from params ring slot 0, which Soup fills only when it steps
    # (before the first step both return zeros); fill it the way step() does, so the comparison is not of zeros.
    for x in (s, z):
        x.device.queue.write_buffer(x.params_bufs[0], 0, x._params(0))
    bc = z.read_byte_counts()
    assert int(bc.sum()) == z.cell_count * 16 and np.array_equal(s.read_byte_counts(), bc)
    h = z.read_hashes()
    assert len(np.unique(h)) > z.cell_count // 2 and np.array_equal(s.read_hashes(), h)
    cells = np.random.default_rng(1).integers(0, 256, size=(z.cell_count, 16), dtype=np.uint8)
    z.write_soup(cells)
    s.write_soup(cells)
    assert np.array_equal(z.read_soup(), cells) and np.array_equal(s.read_hashes(), z.read_hashes())
    assert not np.array_equal(h, z.read_hashes())
    z.reset(11)
    assert np.array_equal(z.read_soup(), Soup(seed=11).read_soup())
    for attr in ("tape_length", "mem_length", "pair_length", "words_per_cell", "cell_count", "pair_count", "z80_steps", "noise_exp", "zero_halts", "mutation_count", "grid"):
        assert getattr(s, attr) == getattr(z, attr), attr


@pytest.mark.parametrize("machine,L,lethal", [("z80c02_pair_L16", 16, False), ("z80c02_lethal_L16", 16, True), ("z80c02_pair_L32", 32, False)])
def test_zsoup_draws_the_same_pairs_as_soup(machine, L, lethal):
    """The drawn pairs are a pure function of the batch seed, so a ZSoup and a Soup with the same seed draw the same
    pairs at every step if ZSoup consumes the CPU stream as Soup does and fills its params ring correctly (40 steps =
    one submit of 32 and one of 8)."""
    s, z = Soup(tape_length=L, seed=13, zero_halts=lethal), ZSoup(machine, seed=13)
    for n in (1, 5, 40):
        s.step(n)
        z.step(n)
        assert np.array_equal(s.read_interactions()["pairs"], z.read_interactions()["pairs"]), n
        assert s.batch_index == z.batch_index and s.cpu_rng.state == z.cpu_rng.state


# ── (e) ZSoup's pipeline ────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("machine", ["z80c02_pair_L16", "z80c02_lethal_L16", "m6502skel_pair_L16"])
def test_zsoup_step_equals_direct_run(machine):
    r = C.plumbing_check(machine, steps=15)
    assert r["active_pairs"] > 15 * 3000
    assert (r["mem_mismatch"], r["wc_mismatch"], r["record_mismatch"], r["inactive_wc_nonzero"]) == (0, 0, 0, 0), r
    assert r["pairs_changed"] > 0 and r["writes_B"] > 0


# ── (f) the step-locked harness ─────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("lethal", [False, True])
def test_lockstep_short(lethal):
    ls = C.Lockstep(L=16, zero_halts=lethal, seed=21, steps_capacity=300)
    s = ls.run(300, dense_until=20, every=50, checkpoint_every=300)
    assert s["steps"] == 300 and s["steps_with_mismatch"] == 0 and s["cpu_gpu_disagree"] == 0 and s["cpu_mismatches"] == 0
    assert s["active_pairs"] > 300 * 3000 and s["active_changed"] > 10000 and s["zil_steps"] > 10**7
    assert s["inactive_changed"] == 0 and s["zil_inactive_steps"] == 0


def test_lockstep_detects_a_one_instruction_difference():
    ls = C.Lockstep(L=16, seed=21, steps_capacity=20, zil_budget_delta=-1)
    s = ls.run(20, dense_until=20, every=10, checkpoint_every=20)
    assert s["steps_with_mismatch"] == 20 and s["cpu_gpu_disagree"] == 0 and s["cpu_mismatches"] > 0


# ── (g) refusals and run.py ─────────────────────────────────────────────────────────────────────────────────────────
def test_refusals():
    with pytest.raises(ValueError):
        ZSoup("m6502skel_pair_L16", suppress="family:block-copy")
    z = ZSoup("m6502skel_pair_L16")
    with pytest.raises(ValueError):
        z.set_suppress("PUSH")
    assert z.patterns == [] and not z.masks.any()   # a refused set leaves the soup unchanged
    z.step(1)
    with pytest.raises(ValueError):
        ZSoup("z80c02_pair_L16", grid="hex")
    from algocell_exp.run import run

    with pytest.raises(ValueError):
        run(machine="z80c02_pair_L16", tape=32, horizon=1, quiet=True)
    with pytest.raises(ValueError):
        run(machine="z80c02_pair_L16", zero_halts=True, horizon=1, quiet=True)
    with pytest.raises(ValueError):
        run(machine="m6502skel_pair_L16", horizon=1, quiet=True)   # run.py's analyses are Z80 disassembly


def test_ablation_mapping():
    """The soup's masks go to a Z80 'coarse' table word for word; an empty set needs no table; any other table or ISA
    refuses a non-empty set (instead of crashing at the first step on a size mismatch)."""
    from types import SimpleNamespace

    from algocell_exp.zsoup import ablation_for

    masks = make_masks(resolve(["PUSH"]))
    stub = lambda isa, abl: SimpleNamespace(name="stub", core={"id": isa}, m={"ablation": abl})  # noqa: E731
    coarse = {"kind": "uniform", "layout": "coarse", "words": 24}
    assert ablation_for(stub("z80", coarse), masks) is masks
    for isa, abl in (("z80", coarse), ("z80", {"kind": "uniform", "layout": "fine", "words": 56}), ("m6502", {"kind": "uniform", "layout": "opcode", "words": 8}), ("m6502", {"kind": "none"})):
        assert ablation_for(stub(isa, abl), np.zeros(24, dtype=np.uint32)) is None
    for isa, abl in (("z80", {"kind": "uniform", "layout": "fine", "words": 56}), ("m6502", {"kind": "uniform", "layout": "opcode", "words": 8}), ("z80", {"kind": "none"})):
        with pytest.raises(ValueError):
            ablation_for(stub(isa, abl), masks)


def test_run_with_machine_flag():
    from algocell_exp.run import run

    buf = io.StringIO()
    s = run(machine="z80c02_lethal_L16", seed=3, horizon=40, sample_every=20, stop_share=None, quiet=True, out=buf)
    cond = json.loads(buf.getvalue().splitlines()[0])
    assert cond["machine"] == "z80c02_lethal_L16" and cond["zero_halts"] is True and cond["tape_length"] == 16
    prov = cond["provenance"]
    assert prov["machine"]["machine"] == "z80c02_lethal_L16" and prov["shader_file"] == "sim_lethal_L16.wgsl"
    assert s["steps_run"] == 40


def test_masks_reach_the_machine():
    """Suppression in a ZSoup is the machine's ablation table: with every suppressible opcode of the three pages
    suppressed nothing that writes can run, so no active pair writes anything (without suppression many do)."""
    every = [f"{pg}:{b:02x}" for pg in ("base", "cb", "ed") for b in range(256)]
    z = ZSoup("z80c02_pair_L16", seed=4, suppress=every)
    assert np.array_equal(z.masks, make_masks(resolve(every))) and np.any(z.masks)
    z.step(3)
    inter = z.read_interactions()
    assert inter["active"].sum() > 3000 and (inter["write_counts"] == 0).all()
    z.set_suppress(None)
    z.step(1)
    inter = z.read_interactions()
    assert (inter["write_counts"][inter["active"]].sum(axis=1) > 0).sum() > 1000


# ── Soup's own readback tests (tests/test_pipeline.py), on a ZSoup ──────────────────────────────────────────────────
def test_zsoup_interaction_readbacks_match_the_soup():
    s = ZSoup("z80c02_pair_L16", seed=4)
    s.step(200)
    before = s.read_soup()
    s.step(1)
    inter = s.read_interactions()
    mem = s.read_pair_memory()
    after = s.read_soup()
    a = inter["active"]
    i, j = inter["pairs"][a, 0], inter["pairs"][a, 1]
    assert 3000 < a.sum() < 6000 and s.read_active_pairs() == a.sum()
    assert int((after[i] != mem[a, 0]).sum() + (after[j] != mem[a, 1]).sum()) < 600
    wc = inter["write_counts"][a]
    changed_b = (mem[a, 1] != before[j]).sum(1)
    assert ((changed_b > 0) <= (wc[:, 1] > 0)).all()
    assert (wc[:, 1] >= changed_b).all()
    pairs, active = s.read_pairs()
    assert np.array_equal(pairs, inter["pairs"]) and np.array_equal(active, a)


def test_zsoup_write_soup_round_trip_and_execution():
    s = ZSoup("z80c02_pair_L16", seed=5)
    cells = s.read_soup()
    unit = np.resize(np.frombuffer(bytes.fromhex("01c5"), dtype=np.uint8), 16)
    cells[:200] = unit
    s.write_soup(cells)
    back = s.read_soup()
    assert back.shape == cells.shape and (back == cells).all()
    s.step(200)
    after = s.read_soup()
    near = ((after != unit).sum(1) <= 4).mean()
    assert near > 0.02, near


def test_zmachine_resume_equals_one_run():
    """A run split into two resumed dispatches (record + memory carried over, step base advanced) equals one run."""
    m = load_machine("z80c02_pair_L16")
    pairs = np.random.default_rng(3).integers(0, 256, size=(2048, 32), dtype=np.uint8)
    one = m.run(pairs, budget=128)
    a = m.run(pairs, budget=50)
    b = m.run(a.memory, budget=78, step_base=50, resume=a.record)
    assert np.array_equal(one.memory, b.memory) and np.array_equal(one.record, b.record)
    assert not np.array_equal(a.memory, one.memory)
