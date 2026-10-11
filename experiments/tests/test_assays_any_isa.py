"""The executor interface (algocell_exp.executor): Algocell's shaders by default, a vendored Zilion machine on request.
Needs the local GPU; under a minute.

(a) the default path: nothing selected, the frozen host functions run the shaders and count them; executor_file names
    the shader;
(b) every assay machine reproduces its shader on random pairs (memory, writes, fetch and start maps), with suppression
    and under the lethal rule, through the interface and through the frozen host functions;
(c) non-vacuity: a selected machine makes the shader pipelines refuse; a machine one instruction short differs;
(d) resolution and refusals: missing lengths, mismatched rules, maps, Z80 suppression on a machine without a Z80 table,
    non-Z80 machines in the Z80 scripts;
(e) the vendored assay machines are what build_machines.mjs generates (skipped unless the Zilion checkout is at the
    commit PROVENANCE_assays.md records); the 6502 machine's initial state is the one its world states.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from algocell_exp import assay as A
from algocell_exp import executor as E
from algocell_exp import exectrace as X
from algocell_exp import optrace as O
from algocell_exp.zmachine import MACHINES_DIR, load_machine

SET = "z80c02_assay"


@pytest.fixture(autouse=True)
def _nothing_selected():
    E.select(None)
    yield
    E.select(None)


def rand_pairs(n, L, seed):
    return np.random.default_rng(seed).integers(0, 256, size=(n, 2 * L), dtype=np.uint8)


# ── (a) the default path ────────────────────────────────────────────────────────────────────────────────────────────
def test_default_is_the_shader():
    assert E.active() is None and isinstance(E.current(), E.AlgocellShader)
    pairs = rand_pairs(512, 16, 1)
    before = E.usage()["shader"].get("z80_test_L16.wgsl", 0)
    mem = A.execute_pairs(pairs, 16, 128)
    assert E.usage()["shader"]["z80_test_L16.wgsl"] == before + 512
    assert (mem == A._shader_execute_pairs(pairs, 16, 128)).all()
    res, masks = X.execute_pairs_traced(pairs, 16, 128)
    r2, m2 = X._shader_execute_pairs_traced(pairs, 16, 128)
    assert (res == mem).all() and (res == r2).all() and (masks == m2).all()
    assert A.executor_file(16) == "z80_test_L16.wgsl" and A.executor_file(16, None, True) == "z80_test_lethal_L16.wgsl"


def test_return_regs_does_not_change_memory():
    pairs = rand_pairs(256, 20, 2)
    mem, regs = A._shader_execute_pairs(pairs, 20, 128, return_regs=True)
    assert (mem == A._shader_execute_pairs(pairs, 20, 128)).all() and regs.shape == (256, 12)
    res, masks, regs2 = X._shader_execute_pairs_traced(pairs, 20, 128, return_regs=True)
    assert (res == mem).all() and (regs2[:, :12] == regs).all() and (regs2[:, 12:20] == masks).all()


# ── (b) machines reproduce the shaders ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("L,zh", [(16, False), (20, False), (32, False), (50, False), (64, False), (16, True), (32, True)])
def test_machine_equals_shader(L, zh):
    pairs = rand_pairs(4096, L, 10 + L + zh)
    sh, zx = E.AlgocellShader(), E.ZilionExecutor(SET)
    for sup in ([], ["family:stack", "family:block-copy"]):
        a = sh.run(pairs, L, 128, sup, zero_halts=zh, fetch_map=True)
        b = zx.run(pairs, L, 128, sup, zero_halts=zh, fetch_map=True)
        assert (a.memory == b.memory).all() and (a.writes == b.writes).all() and (a.fetch_map == b.fetch_map).all()
        assert b.executor == f"{SET}{'_lethal' if zh else ''}_L{L}" and a.steps is None and b.steps is not None
    a = sh.run(pairs, L, 128, zero_halts=zh, start_map=True)
    b = zx.run(pairs, L, 128, zero_halts=zh, fetch_map=True, start_map=True)
    assert (a.start_map == b.start_map).all() and (a.fetch_map == b.fetch_map).all() and (a.memory == b.memory).all()


def test_frozen_functions_on_a_machine():
    pairs = rand_pairs(2048, 16, 3)
    ref = A._shader_execute_pairs(pairs, 16, 128, ["family:ex"])
    ref_t, ref_m = X._shader_execute_pairs_traced(pairs, 16, 128, zero_halts=True)
    ref_o = O._shader_execute_pairs_optrace(pairs, 16, 128)
    with E.using(SET):
        assert (A.execute_pairs(pairs, 16, 128, ["family:ex"]) == ref).all()
        t, m = X.execute_pairs_traced(pairs, 16, 128, zero_halts=True)
        assert (t == ref_t).all() and (m == ref_m).all() and m.dtype == np.uint32 and m.shape == (2048, 8)
        o = O.execute_pairs_optrace(pairs, 16, 128)
        assert all((x == y).all() for x, y in zip(o, ref_o))
        assert A.executor_file(16) == "z80c02_assay_L16" and A.executor_file(32, None, True) == "z80c02_assay_lethal_L32"
        r = A.assay(bytes.fromhex("01c5" * 8), z80_steps=128, n=32, seed=0)
    assert E.active() is None
    assert json.dumps(r, sort_keys=True) == json.dumps(A.assay(bytes.fromhex("01c5" * 8), z80_steps=128, n=32, seed=0), sort_keys=True)


def test_pack_fetch_masks_roundtrip():
    f = np.random.default_rng(4).random((50, 100)) < 0.3
    assert (X.exec_positions(E.pack_fetch_masks(f), 100) == f).all()
    with pytest.raises(ValueError):
        E.pack_fetch_masks(np.zeros((1, 257), bool))


# ── (c) non-vacuity ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_shader_refuses_while_a_machine_is_selected():
    pairs = rand_pairs(64, 16, 5)
    E.select(SET)
    for f in (lambda: A._pipeline(16), lambda: X._pipeline(16, None, False), lambda: O._pipeline(16, False)):
        with pytest.raises(RuntimeError, match="was requested while machine"):
            f()
    # the explicit path (a) object still runs, and only the shader
    r = E.AlgocellShader().run(pairs, 16, 128)
    assert r.executor == "z80_test_L16.wgsl"


def test_one_instruction_short_differs():
    pairs = rand_pairs(4096, 16, 6)
    a = E.AlgocellShader().run(pairs, 16, 128)
    b = E.ZilionExecutor(SET, budget_delta=-1).run(pairs, 16, 128)
    assert (a.memory != b.memory).any(1).sum() > 0 and (a.writes != b.writes).any(1).sum() > 0


# ── (d) resolution and refusals ─────────────────────────────────────────────────────────────────────────────────────
def test_refusals():
    zx = E.ZilionExecutor(SET)
    with pytest.raises(ValueError, match="no machine for L = 24"):
        zx.run(rand_pairs(4, 24, 0), 24, 128)
    with pytest.raises(ValueError, match="no machine for L = 16, ring 34"):
        zx.run(rand_pairs(4, 16, 0), 16, 128, mem_length=34)
    with pytest.raises(ValueError, match="halting.lethal"):
        E.ZilionExecutor("z80c02_assay_L16").run(rand_pairs(4, 16, 0), 16, 128, zero_halts=True)
    with pytest.raises(ValueError, match="segments"):
        E.ZilionExecutor("z80c02_assay_L16").run(rand_pairs(4, 32, 0), 32, 128)
    with pytest.raises(ValueError, match="fetch and start maps"):
        E.ZilionExecutor("z80c02_pair_L16").run(rand_pairs(4, 16, 0), 16, 128, fetch_map=True)
    with pytest.raises(ValueError, match="Z80 'coarse' ablation table"):
        E.ZilionExecutor("m6502_assay").run(rand_pairs(4, 16, 0), 16, 128, ["family:stack"])
    with pytest.raises(ValueError, match="no vendored machine or machine set"):
        E.ZilionExecutor("no_such_machine")
    # a set is <prefix>[_lethal]_L<L>[_P<P>] only: ZSoup's z80c02_pair_L16 is not a member of the set 'z80c02'
    z = E.ZilionExecutor("z80c02")
    assert "z80c02_pair_L16" not in z.members and "z80c02_lethal_L16" in z.members
    with pytest.raises(ValueError, match="has no machine for L = 16, ring 32, plain rule"):
        z.run(rand_pairs(4, 16, 0), 16, 128)
    for isas, spec, ok in ((("z80",), "m6502_assay", False), (("z80",), SET, True), (None, "m6502_assay", True)):
        a = argparse.Namespace(machine=spec)
        if ok:
            E.from_args(a, isas=isas)
            assert E.active() is not None and E.active().spec == spec
        else:
            with pytest.raises(SystemExit, match="runs z80 machines only"):
                E.from_args(a, isas=isas)
        E.select(None)
    assert E.from_args(argparse.Namespace(machine=None)) == "algocell shaders" and E.active() is None


@pytest.mark.parametrize("script", ["assay_batch.py", "c4_functional.py", "stage_g.py", "closure.py", "exec_trace.py", "inflow_insoup.py",
                                    "mutscan.py", "serial_retention.py", "branching_transfer.py", "population_sample.py",
                                    "population_classes.py", "biology/individuality.py"])
def test_scripts_take_machine(script):
    exp = Path(__file__).resolve().parents[1]
    p = subprocess.run([sys.executable, str(exp / script), "--help"], cwd=exp, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-500:]
    assert "--machine" in p.stdout


# ── (e) vendored machines ───────────────────────────────────────────────────────────────────────────────────────────
def test_assay_machines_regenerate():
    zdir = Path(__file__).resolve().parents[3] / "zilion"
    if not (zdir / "dist" / "index.js").exists() or shutil.which("node") is None:
        pytest.skip("no Zilion build beside the repository")
    recorded = re.search(r"commit `([0-9a-f]{40})`", (MACHINES_DIR / "PROVENANCE_assays.md").read_text()).group(1)
    head = subprocess.run(["git", "-C", str(zdir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != recorded:
        pytest.skip(f"Zilion checkout at {head[:12]}, assay machines vendored from {recorded[:12]}")
    p = subprocess.run(["node", str(MACHINES_DIR / "build_machines.mjs"), "--check", "--spec", "worlds_assays.json", "--zilion", str(zdir)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def test_m6502_world_state():
    m = load_machine("m6502_assay_L16")
    r = m.run(np.zeros((1, 32), np.uint8), budget=0)   # no step: the state after the world's reset and init
    assert r.reg("s") == 0xFF and r.reg("p") == 0x20 and r.reg("pc") == 0 and r.reg("a") == 0 and r.reg("x") == 0 and r.reg("y") == 0
    assert int(r.steps[0]) == 0 and m.m["ablation"]["kind"] == "none"
    assert m.m["core"]["id"] == "m6502" and m.m["core"]["status"] != "skeleton"
