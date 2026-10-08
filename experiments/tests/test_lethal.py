"""Lethal tar (Stage I control): the `zero_halts` rule. Needs the local GPU; ≈ 30 s.

The rule: a zero byte (NOP, 0x00) fetched as the FIRST byte of an instruction halts the pair's execution for the rest of
the encounter — the remaining budget is forfeited and the memories are written back as they are. It is judged on the raw
fetched byte in `z80_step()`, before the DD/FD prefix resolution and before the suppression hook, so prefixed opcodes
(`DD 00`, `FD 00`, `ED 00`, `CB 00`) and operand bytes never trigger it (`algocell_exp.gen_lethal_shader`).

(a) the pusher `01 c5` × 8 against an all-zero partner, 128 steps: normal rule → the partner is tiled with the pusher's
    word; zero_halts → the pointer leaves the organism at cell 16, fetches a zero and halts, so exactly the 8 bytes of the
    four pushes made while executing the organism's own 16 cells are written (partner cells 7–14 = `c5 01` × 4) and the
    other 8 stay zero;
(b) outputs are bitwise identical under both rules whenever no zero byte is fetched as an opcode (the pusher against
    random partners without zero bytes; the executor's final PC tells whether a zero was fetched), and identical by
    construction against a partner of `01` × 16, in whose ring no zero byte can ever appear;
(c) an organism whose first byte is 0x00 writes nothing under zero_halts (both halves unchanged for 20 random partners)
    while the normal rule changes the partner;
(d) prefixed and operand zeros do not halt; `assay`/`assay_many` accept the flag and agree with each other under it;
(e) a 300-step soup smoke run at the Stage G settings records the flag and the derived shader; tar still forms;
(f) the derived shaders differ from their sources only by the hunk; Soup refuses the flag off the square grid;
(g) the Stage I condition file: 10 worlds, seeds 4001–4010, zero_halts, every other field identical to Stage G L = 16.
Run with `-s` to see the measured numbers.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
import wgpu

from algocell_exp import assay as A
from algocell_exp.assay import assay, assay_many, execute_pairs, executor_file
from algocell_exp.gen_lethal_shader import ANCHOR, HUNK, LOOP_BREAK, lethal_source
from algocell_exp.gen_lethal_shader import pairs as shader_pairs
from algocell_exp.run import run
from algocell_exp.soup import SHADER_DIR, Soup

ROOT = Path(__file__).resolve().parents[1]
W, H, L = 160, 125, 16
STAGE_G = dict(width=W, height=H, tape_length=L, pair_count=8192, z80_steps=128, noise_exp=4)   # Stage G L = 16 settings
STEPS = 128
PUSHER = np.frombuffer(bytes.fromhex("01c5" * 8), dtype=np.uint8)   # LD BC,$01c5 ; PUSH BC — the first replicator of Stage G L = 16


def _run(pairs: np.ndarray, zero_halts: bool, steps: int = STEPS) -> np.ndarray:
    return execute_pairs(np.ascontiguousarray(pairs, dtype=np.uint8), L, steps, (), None, zero_halts)


def _run_with_regs(pairs: np.ndarray, zero_halts: bool, steps: int = STEPS) -> tuple[np.ndarray, np.ndarray]:
    """execute_pairs plus the executor's register block per pair: (a, f, b, c, d, e, h, l, sp, pc, writes_a, writes_b)."""
    pairs = np.ascontiguousarray(pairs, dtype=np.uint8)
    N = pairs.shape[0]
    dev = A.get_device()
    pipe, layout = A._pipeline(L, None, zero_halts)
    wpc = (L + 3) // 4
    words = np.zeros((N, 2 * wpc * 4), dtype=np.uint8)
    words[:, :L] = pairs[:, :L]
    words[:, 4 * wpc: 4 * wpc + L] = pairs[:, L:]
    io_data = np.ascontiguousarray(words).view(np.uint32)
    params = np.zeros(32, dtype=np.uint32)
    params[2], params[3], params[4], params[6] = L, 2 * L, N, steps
    params[8:] = A.make_masks(A.resolve([]))
    B = wgpu.BufferUsage
    io_buf = dev.create_buffer(size=io_data.nbytes, usage=B.STORAGE | B.COPY_SRC | B.COPY_DST)
    dev.queue.write_buffer(io_buf, 0, io_data)
    regs_buf = dev.create_buffer(size=N * 12 * 4, usage=B.STORAGE | B.COPY_SRC)
    p_buf = dev.create_buffer(size=128, usage=B.UNIFORM | B.COPY_DST)
    dev.queue.write_buffer(p_buf, 0, params)
    bg = dev.create_bind_group(layout=layout, entries=[
        {"binding": 0, "resource": {"buffer": p_buf, "offset": 0, "size": 128}},
        {"binding": 1, "resource": {"buffer": io_buf, "offset": 0, "size": io_buf.size}},
        {"binding": 2, "resource": {"buffer": regs_buf, "offset": 0, "size": regs_buf.size}}])
    enc = dev.create_command_encoder()
    ps = enc.begin_compute_pass()
    ps.set_pipeline(pipe)
    ps.set_bind_group(0, bg)
    ps.dispatch_workgroups(-(-N // 64))
    ps.end()
    dev.queue.submit([enc.finish()])
    out = np.frombuffer(dev.queue.read_buffer(io_buf), dtype=np.uint8).reshape(N, 2 * wpc * 4)
    regs = np.frombuffer(dev.queue.read_buffer(regs_buf), dtype=np.uint32).reshape(N, 12).copy()
    res = np.empty((N, 2 * L), dtype=np.uint8)
    res[:, :L] = out[:, :L]
    res[:, L:] = out[:, 4 * wpc: 4 * wpc + L]
    for b in (io_buf, regs_buf, p_buf):
        b.destroy()
    return res, regs


# ── (a) the pusher against tar ──

def test_pusher_against_zero_partner_writes_eight_bytes_then_halts():
    pair = np.concatenate([PUSHER, np.zeros(L, np.uint8)])[None]
    normal = _run(pair, False)[0]
    lethal, regs = _run_with_regs(pair, True)
    lethal, regs = lethal[0], regs[0]
    assert (_run(pair, True)[0] == lethal).all()                       # the register-reading helper agrees with execute_pairs
    # Normal rule: the pointer leaves the organism at cell 16, runs through the partner's zeros as NOPs, meets the words it has
    # pushed, keeps loading and pushing around the 32-byte ring, and the partner ends up tiled with the pusher's word.
    A_n, B_n = normal[:L], normal[L:]
    assert (A_n == PUSHER).all()
    assert int((B_n == PUSHER).sum()) >= 12, B_n.tobytes().hex(" ")
    assert A._best_shift_match(B_n[None], PUSHER)[0] >= 0.75           # the assay helper calls the partner a copy
    # zero_halts: LD BC,$01c5 ; PUSH BC four times over the organism's own 16 cells. SP starts at 0xffff ≡ 31 and pre-decrements,
    # so the pushes write `c5 01` at partner cells 13–14, 11–12, 9–10, 7–8 (hi byte 01 at the even ring address, lo byte c5
    # below it). Then the pointer fetches partner cell 0 = 0x00 at ring address 16 and halts: 8 bytes written, 8 untouched.
    expected_B = np.array([0] * 7 + [0xC5, 0x01] * 4 + [0], dtype=np.uint8)
    A_l, B_l = lethal[:L], lethal[L:]
    assert (A_l == PUSHER).all()
    assert B_l.tolist() == expected_B.tolist(), B_l.tobytes().hex(" ")
    assert int((B_l != 0).sum()) == 8 and (B_l[:7] == 0).all() and B_l[15] == 0 and (B_l[7:15] == np.array([0xC5, 0x01] * 4)).all()
    assert int(regs[11]) == 8 and int(regs[10]) == 0                   # shader write counters: 8 bytes into B, none into A
    assert int(regs[9]) % (2 * L) == 16                                # PC backed up onto the lethal byte: partner cell 0 = ring address 16
    assert int(regs[8]) % (2 * L) == 23                                # SP after four pushes: 31 - 8
    print(f"\n[a] pusher vs zeros, 128 steps — normal: B = {B_n.tobytes().hex(' ')} ({int((B_n == PUSHER).sum())}/16 bytes = pusher); "
          f"zero_halts: B = {B_l.tobytes().hex(' ')} (8 written at cells 7–14, PC {int(regs[9]) % 32}, SP {int(regs[8]) % 32}, writes_b {int(regs[11])})")


# ── (b) identical outputs when no zero byte is fetched as an opcode ──

def test_identical_outputs_when_no_zero_is_fetched_as_an_opcode():
    rng = np.random.default_rng(2026)
    n = 256
    partners = rng.integers(1, 256, size=(n, L), dtype=np.uint8)       # random partners WITHOUT zero bytes
    assert (partners != 0).all()
    pairs = np.concatenate([np.repeat(PUSHER[None], n, 0), partners], axis=1)
    normal = _run(pairs, False)
    lethal, regs = _run_with_regs(pairs, True)
    pc = regs[:, 9] % (2 * L)
    # After a lethal halt the PC sits on the zero byte that was fetched (backed up as HALT does) and nothing executes afterwards,
    # so a non-zero byte under the final PC proves that no zero was ever fetched as an opcode in that encounter: those
    # encounters must be bitwise identical under both rules, and every difference must come with a zero under the PC.
    halted_on_zero = lethal[np.arange(n), pc] == 0
    same = (normal == lethal).all(axis=1)
    assert same[~halted_on_zero].all(), np.where(~same & ~halted_on_zero)[0]
    assert (~same <= halted_on_zero).all()
    assert int((~halted_on_zero).sum()) >= 8, int((~halted_on_zero).sum())
    # The partner's own random code can WRITE zeros that the pointer later fetches (PUSH of an empty register pair, CALL/RST pushing
    # a PC whose high byte is 0, LD (HL),A with A = 0 …): those encounters legitimately differ — printed, not asserted.
    print(f"\n[b] pusher vs {n} zero-free random partners: identical in {int(same.sum())}/{n}; no zero fetched as an opcode in "
          f"{int((~halted_on_zero).sum())}/{n} (all identical); a run-time zero was fetched in {int(halted_on_zero.sum())}/{n}, "
          f"of which {int((halted_on_zero & ~same).sum())} differ")
    # By construction: against a partner of 01 × 16 every byte the pointer meets is 01 or c5 and BC only ever holds those bytes,
    # so no zero can appear anywhere in the ring; likewise the pusher against itself.
    ones = np.concatenate([PUSHER, np.full(L, 1, np.uint8)])[None]
    assert (_run(ones, False) == _run(ones, True)).all()
    twin = np.concatenate([PUSHER, PUSHER])[None]
    assert (_run(twin, False) == _run(twin, True)).all() and (_run(twin, True)[0] == np.concatenate([PUSHER, PUSHER])).all()


# ── (c) a zero first byte writes nothing ──

def test_zero_first_byte_writes_nothing_under_zero_halts():
    org = np.frombuffer(bytes.fromhex("00" + "01c5" * 7 + "01"), dtype=np.uint8)   # NOP, then the pusher shifted by one cell
    rng = np.random.default_rng(7)
    partners = rng.integers(0, 256, size=(20, L), dtype=np.uint8)
    pairs = np.concatenate([np.repeat(org[None], 20, 0), partners], axis=1)
    lethal, regs = _run_with_regs(pairs, True)
    assert (lethal == pairs).all()                                     # both halves unchanged in all 20 encounters
    assert (regs[:, 10] == 0).all() and (regs[:, 11] == 0).all()       # no byte written anywhere
    assert (regs[:, 9] == 0).all()                                     # PC backed up onto cell 0, the lethal byte
    normal = _run(pairs, False)
    changed = (normal != pairs).any(axis=1)
    changed_b = (normal[:, L:] != partners).any(axis=1)
    assert changed.any() and changed_b.any()
    print(f"\n[c] organism 00 + pusher: zero_halts changed 0/20 pairs; normal rule changed {int(changed.sum())}/20 pairs ({int(changed_b.sum())}/20 partners)")


# ── (d) prefixed and operand zeros are not lethal; the assay API carries the flag ──

def test_prefixed_and_operand_zeros_do_not_halt_and_assays_take_the_flag():
    ones = np.full(L, 1, np.uint8)
    prefixed = {
        "DD 00": bytes.fromhex("dd00" + "01c5" * 7),     # IX form of NOP
        "FD 00": bytes.fromhex("fd00" + "01c5" * 7),     # IY form of NOP
        "ED 00": bytes.fromhex("ed00" + "01c5" * 7),     # undefined ED opcode: 2-byte NOP
        "CB 00": bytes.fromhex("cb00" + "01c5" * 7),     # RLC B
    }
    for name, tape in prefixed.items():
        pair = np.concatenate([np.frombuffer(tape, dtype=np.uint8), ones])[None]
        # Budget of 3 instructions: prefixed zero (2 bytes), LD BC,$01c5 (3 bytes), PUSH BC. Had the prefixed zero halted the
        # pair, nothing would be written and PC would sit on cell 1; instead the push lands (2 bytes into B) and PC = 6, under
        # both rules identically.
        lethal, regs = _run_with_regs(pair, True, steps=3)
        assert int(regs[0, 11]) == 2 and int(regs[0, 9]) == 6 and int(regs[0, 8]) % (2 * L) == 29, (name, regs[0])
        assert (lethal == _run(pair, False, steps=3)).all(), name
        assert lethal[0, L + 13] == 0xC5 and lethal[0, L + 14] == 0x01, (name, lethal[0, L:].tobytes().hex(" "))
        # over the full budget the pushes keep landing too (not halted at the prefixed zero); the two rules may legitimately
        # diverge later because the organism's own zero at cell 1 can be loaded as an operand, pushed, and fetched as an opcode
        _, regs_full = _run_with_regs(pair, True)
        assert int(regs_full[0, 11]) >= 2, (name, regs_full[0])
    # an operand zero: LD BC,$0000 (01 00 00) ; PUSH BC — the zeros are operands, fetched inside the instruction, not opcodes;
    # with a budget of 2 instructions the push writes them (00 00 at partner cells 13–14) under both rules identically
    operand = np.frombuffer(bytes.fromhex("010000c5" + "01c5" * 6), dtype=np.uint8)
    pair = np.concatenate([operand, ones])[None]
    lethal, regs = _run_with_regs(pair, True, steps=2)
    assert int(regs[0, 11]) == 2 and int(regs[0, 9]) == 4 and lethal[0, L + 13] == 0 and lethal[0, L + 14] == 0, regs[0]
    assert (lethal == _run(pair, False, steps=2)).all()
    # the assay API: assay() and assay_many() under the lethal rule share partner draws (same seed) and agree
    for zh in (False, True):
        one = assay(PUSHER.tobytes(), z80_steps=STEPS, n=32, seed=0, zero_halts=zh)
        many = assay_many(PUSHER[None], z80_steps=STEPS, n=32, seed=0, zero_halts=zh)[0]
        assert abs(many["gen2_score"] - one["gen2_score"]) < 1e-9 and abs(many["score"] - one["copy_into_neighbor_as_A"]) < 1e-9
        assert many["is_replicator"] == one["is_replicator"] and many["faithful"] == one["faithful"]
        print(f"\n[d] pusher culture test (32 random partners), zero_halts={zh}: as-A gain {one['copy_into_neighbor_as_A']:.2f}, gen2 {one['gen2_score']:.2f}, "
              f"copies ≥ 75% {one['offspring_within_q']:.2f}, heritable {one['is_replicator']}, faithful {one['faithful']}")


# ── (e) smoke run: 300 steps at the Stage G settings under zero_halts ──

def test_lethal_smoke_run_records_rule_and_forms_tar():
    buf = io.StringIO()
    s = run(grid="square", zero_halts=True, tape=L, width=W, height=H, seed=4001, pairs=8192, z80_steps=128, noise_exp=4,
            horizon=300, sample_every=50, stop_share=-1, sample_steps=[1], quiet=True, out=buf)
    recs = [json.loads(l) for l in buf.getvalue().splitlines()]
    cond, samples = recs[0], [r for r in recs if r["kind"] == "sample"]
    assert cond["kind"] == "condition" and cond["zero_halts"] is True and cond["grid"] == "square"
    assert cond["provenance"]["shader_file"] == "sim_lethal_L16.wgsl"
    assert s["zero_halts"] is True and s["grid"] == "square" and s["steps_run"] == 300 and s["provenance"]["shader_file"] == "sim_lethal_L16.wgsl"
    assert [r["step"] for r in samples] == [1, 50, 100, 150, 200, 250, 300]
    assert all(3000 < r["active_pairs"] < 6000 for r in samples)
    zf = {r["step"]: r["zero_frac"] for r in samples}
    # Tar still forms: zeros are written by pushes of empty registers whatever executing a zero does (Stage I prediction I1),
    # so the zero fraction rises from the random soup's 1/256 within the first step and stays well above it. Measured
    # 2026-10-08, seed 4001: 0.035 at step 1, 0.170 at 50, 0.178 at 100, 0.182 at 200, 0.184 at 300 — a plateau near 0.18,
    # about half the benign lattice soup's (0.22 by step 8, 0.34 by 50, ≈ 0.37 to step 1000): a pair that fetches a zero halts
    # and pushes nothing more, so lethal tar grows more slowly and saturates lower. Asserted with margin; the Stage I
    # prediction itself (I1: ≥ 0.15 at step 500 in ≥ 7/10 worlds) is scored on the real runs, not here.
    assert zf[1] > 1 / 256, zf
    assert zf[300] > 0.1 and zf[300] > zf[1], zf
    top = {r["step"]: (r["top_share"], r["q_share"]) for r in samples}
    print("\n[e] zero_halts smoke, zero_frac by step: " + ", ".join(f"{t}: {zf[t]:.3f}" for t in (1, 50, 100, 200, 300))
          + "; top class share (exact / q): " + ", ".join(f"{t}: {top[t][0]:.3f}/{top[t][1]:.3f}" for t in (1, 50, 100, 200, 300))
          + f"; active pairs at 300: {samples[-1]['active_pairs']}; wall {s['wall_s']} s for 300 steps + 7 samples")


# ── (f) derived shaders and the Soup / executor selection ──

def test_lethal_shaders_differ_only_by_the_hunk_and_are_selected_by_the_flag():
    for src_path, dst_path in shader_pairs(16, None):
        src, dst = src_path.read_text(), dst_path.read_text()
        assert dst == lethal_source(src), dst_path.name
        assert src.count(ANCHOR) == 1 and HUNK not in src
        at = src.index(ANCHOR) + len(ANCHOR)
        assert dst == src[:at] + HUNK + src[at:] and dst.count(HUNK) == 1
        assert src[:at].rfind("\nfn ") == src[:at].rfind("\nfn z80_step() {")      # the hunk sits in z80_step(), after the opcode fetch
        assert src.index("if (op == 0xddu || op == 0xfdu) {", at) < src.index("on_fetch_opcode(pfx, op)", at)   # before prefix resolution and the hook
        assert LOOP_BREAK in src                                                # both entry points stop their loop on cpu_halted
    assert HUNK.rstrip().splitlines()[-1].strip() == "if (op == 0u) { cpu_halted = 1u; cpu_pc = (cpu_pc - 1u) & 0xffffu; return; }"
    soup = Soup(grid="square", seed=1, zero_halts=True, **STAGE_G)
    assert soup.shader_file.name == "sim_lethal_L16.wgsl" and soup.zero_halts is True and soup.grid == "square"
    plain = Soup(grid="square", seed=1, **STAGE_G)
    assert plain.shader_file.name == "sim_square_L16.wgsl" and plain.zero_halts is False
    with pytest.raises(ValueError):
        Soup(grid="mixed", seed=1, zero_halts=True, **STAGE_G)
    with pytest.raises(ValueError):
        Soup(grid="hex", seed=1, zero_halts=True)
    with pytest.raises(ValueError):
        Soup(grid="square", seed=1, tape_length=50, zero_halts=True)          # not derived for L = 50
    assert executor_file(16, None, True) == "z80_test_lethal_L16.wgsl" and executor_file(16, None, False) == "z80_test_L16.wgsl"
    with pytest.raises(ValueError):
        executor_file(50, None, True)
    assert A._pipeline(16, None, True) is not A._pipeline(16, None, False)   # separate cached pipelines per rule


# ── (g) the Stage I condition file ──

def test_stage_i_conditions_match_stage_g_L16():
    from algocell_exp.batch import run_stem
    from make_conds import ABLATIONS, STAGES, ablation_of, stage_i

    text = open(ROOT / "conds" / "stageI.json").read()
    conds = json.loads(text)
    assert json.dumps(stage_i(), indent=0) == text and STAGES["stageI"] is stage_i, "condition file differs from make_conds output"
    assert len(conds) == 10 and [c["seed"] for c in conds] == list(range(4001, 4011))
    for c in conds:
        assert c["label"] == "lethal@closure" and c["zero_halts"] is True and c.get("grid", "square") == "square"
        assert c["tape"] == 16 and c["z80_steps"] == 128 and c["noise_exp"] == 4 and c["horizon"] == 300_000 and c["stop_share"] == -1 and c["suppress"] == ""
    g16 = [c for c in json.load(open(ROOT / "conds" / "stageG.json")) if c["tape"] == 16]
    assert len(g16) == 20
    ref = {k: v for k, v in g16[0].items() if k not in ("label", "seed")}
    for c in conds:
        assert {k: v for k, v in c.items() if k not in ("label", "seed", "zero_halts")} == ref   # sample schedule, snapshots, random tapes, census, stop_share …
    stems = [run_stem(c) for c in conds]
    assert len(set(stems)) == 10 and stems[0] == "lethal@closure_L16_st128_k4_s4001"
    assert ablation_of("lethal@closure") in ABLATIONS and ABLATIONS["lethal"] == []   # preflight / zoo resolve the label; nothing suppressed
    assert (SHADER_DIR / "sim_lethal_L16.wgsl").exists() and (SHADER_DIR / "z80_test_lethal_L16.wgsl").exists()
    assert json.dumps(STAGES["stageH"](), indent=0) == open(ROOT / "conds" / "stageH.json").read()   # Stage H's file is untouched
    import stage_h_local as runner
    assert os.path.abspath(ROOT / "runs" / "stageI") in runner.REAL_OUT_DIRS and os.path.abspath(ROOT / "runs" / "stageH") in runner.REAL_OUT_DIRS
    assert runner.STAGES["I"]["conds"].endswith("stageI.json") and runner.STAGES["I"]["provenance"]["stage"] == "I"
