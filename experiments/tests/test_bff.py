"""Tests of the BFF GPU kernel (micro/bff.py) against its CPU reference and against the cubff validation replicator."""
import numpy as np
import pytest

from micro.bff import BFF, PAIR, TAPE, ascii_map, assay, density_map, has_loop, reference_execute, similarity


def enc(s: str, n: int = TAPE, fill: int = 0) -> np.ndarray:
    t = np.full(n, fill, dtype=np.uint8)
    b = s.encode()
    t[: len(b)] = np.frombuffer(b, dtype=np.uint8)
    return t


MIRROR = "<[[[[[,,.[.[[}<,]],<}[,<"  # cubff validation replicator (bff.inc.h)


@pytest.fixture(scope="module")
def bff():
    return BFF(max_pairs=1024)


@pytest.mark.parametrize("ip_wrap", [False, True])
@pytest.mark.parametrize("dens", [1, 8])
def test_kernel_matches_reference(ip_wrap, dens):
    rng = np.random.default_rng(1)
    amap = ascii_map() if dens == 1 else density_map(dens, seed=3)
    b = BFF(max_pairs=256, steps=2048, ip_wrap=ip_wrap, alphabet=amap)
    pairs = rng.integers(0, 256, size=(256, PAIR), dtype=np.uint8)
    # salt half the pairs with real programs so that brackets and copies actually run
    for i in range(0, 256, 2):
        prog = "".join(rng.choice(list("<>{}+-.,[]") + [" "] * 6, size=rng.integers(4, 40)))
        pairs[i, : len(prog)] = np.frombuffer(prog.encode(), dtype=np.uint8) if dens == 1 else pairs[i, : len(prog)]
    mem, out = b.execute(pairs)
    for i in range(256):
        ref_mem, ref = reference_execute(pairs[i], 2048, ip_wrap=ip_wrap, alphabet=amap)
        assert np.array_equal(mem[i], ref_mem), f"pair {i}: memory differs"
        assert tuple(out[i]) == (ref["executed"], ref["entered"], ref["max_pc"], ref["writesB"]), f"pair {i}: stats differ {out[i]} vs {ref}"


def test_mirror_replicator_copies_itself_reversed(bff):
    rng = np.random.default_rng(7)
    A = enc(MIRROR)
    # partner byte 127 must be non-zero for the leading [ tests; use random partners and keep the ones that qualify
    partners = rng.integers(1, 256, size=(64, TAPE), dtype=np.uint8)
    mem, out = bff.execute(np.concatenate([np.repeat(A[None], 64, 0), partners], axis=1))
    B = mem[:, TAPE:]
    n = len(MIRROR)
    ok = 0
    for b, o in zip(B, out):
        # the loop }<, writes tape[127-i] = tape[i] until it copies the first zero of A (position n)
        if all(b[TAPE - 1 - i] == A[i] for i in range(n + 1)):
            ok += 1
    assert ok >= 60, f"mirror copy present in only {ok}/64 partners"
    sim, how = similarity(A, B[0])
    assert how == "rev"
    assert has_loop(A)


def test_ip_wrap_switch():
    t = enc(">" * TAPE)  # straight line of head moves
    pair = np.concatenate([t, np.full(TAPE, ord(">"), dtype=np.uint8)])
    _, o_std = BFF(max_pairs=4, steps=1000, ip_wrap=False).execute(pair[None])
    _, o_wrap = BFF(max_pairs=4, steps=1000, ip_wrap=True).execute(pair[None])
    assert o_std[0, 0] == PAIR  # one pass, then the pointer leaves the tape
    assert o_wrap[0, 0] == 1000  # laps the tape
    assert o_std[0, 1] == 1 and o_wrap[0, 1] == 1  # both entered the partner (straight-line execution runs into it)


def test_straight_line_copier_cannot_replicate_in_either_pointer_mode(bff):
    """THEORY.md P1 (as first written) claimed the straight-line tiling `.{>` becomes a full replicator once the
    instruction pointer wraps. Harness test, 2026-10-08, before any soup ran: it does not. One pass (standard BFF)
    writes ~21 bytes; with a wrapping pointer the two heads cross, the copy is re-read as source, and the
    drift between read and write heads scrambles both tapes. Recorded here so the correction stays testable."""
    rng = np.random.default_rng(3)
    partners = rng.integers(0, 256, size=(32, TAPE), dtype=np.uint8)
    for prog in (".{>" * 21 + ".", "{.>" * 21 + ".", "." + "{.>" * 21):
        t = enc(prog, fill=ord("."))
        mem, out = bff.execute(np.concatenate([np.repeat(t[None], 32, 0), partners], axis=1))
        changed = (mem[:, TAPE:] != partners).sum(1)
        # 21 bytes from the organism plus a few writes by instruction bytes of the partner that the pointer runs into
        assert 20 <= changed.min() and changed.max() <= 40, changed
        bw = BFF(max_pairs=64, steps=8192, ip_wrap=True)
        mem2, out2 = bw.execute(np.concatenate([np.repeat(t[None], 32, 0), partners], axis=1))
        sims = np.array([similarity(t, b)[0] for b in mem2[:, TAPE:]])
        self_damage = (mem2[:, :TAPE] != t).mean(1)
        # no partner becomes a copy; the organism itself is scrambled in most encounters (a few end early on a stray `]`)
        assert sims.max() < 0.75 and np.median(self_damage) > 0.5, (prog, sims, self_damage)
        assert out2[:, 1].all()  # open: the pointer runs into the partner every time


def test_density_map_density():
    m = density_map(8, seed=0)
    assert m[0] == 0
    counts = np.bincount(m, minlength=11)
    assert all(counts[1:] == 8)
    assert np.array_equal(density_map(1), ascii_map())


def test_assay_of_random_tape_is_sterile(bff):
    rng = np.random.default_rng(5)
    r = assay(bff, rng.integers(0, 256, size=TAPE, dtype=np.uint8), n=32)
    assert r["copies"] < 0.1 and r["gen2"] < 0.1
