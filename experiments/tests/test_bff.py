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


@pytest.mark.parametrize("ip_wrap", [False, True])
def test_literal_push_kernel_matches_reference(ip_wrap):
    rng = np.random.default_rng(11)
    b = BFF(max_pairs=256, steps=2048, ip_wrap=ip_wrap, literal=True)
    pairs = rng.integers(0, 256, size=(256, PAIR), dtype=np.uint8)
    for i in range(0, 256, 2):
        prog = "".join(rng.choice(list("<>{}+-.,[]P") + ["x"] * 5, size=rng.integers(4, 40)))
        pairs[i, : len(prog)] = np.frombuffer(prog.encode(), dtype=np.uint8)
    mem, out = b.execute(pairs)
    for i in range(256):
        ref_mem, ref = reference_execute(pairs[i], 2048, ip_wrap=ip_wrap, literal=True)
        assert np.array_equal(mem[i], ref_mem), f"pair {i}"
        assert tuple(out[i]) == (ref["executed"], ref["entered"], ref["max_pc"], ref["writesB"]), f"pair {i}: {out[i]} vs {ref}"


def test_literal_push_tiling_is_an_open_replicator_only_with_a_wrapping_pointer():
    """`P x` tiled: each P writes its two literal bytes (x, P) below head1 — the BFF analogue of `LD rr,nn ; PUSH rr`.
    Each push costs 4 bytes of code for 2 bytes written, so one pass over the pair copies 48 of 64 bytes (its own copy
    executes 8 more pushes): the one-pass bandwidth bound of THEORY.md. With a wrapping pointer the tiling laps and
    copies itself completely, then runs on into the partner (open)."""
    t = enc("Px" * 32)
    rng = np.random.default_rng(2)
    quiet = np.full((32, TAPE), ord("x"), dtype=np.uint8)  # no-op partners
    b = BFF(max_pairs=64, steps=8192, ip_wrap=False, literal=True)
    mem, out = b.execute(np.concatenate([np.repeat(t[None], 32, 0), quiet], axis=1))
    assert (out[:, 3] == 48).all(), out[:, 3]  # 48 bytes written into the partner (half of them equal the quiet byte)
    assert (out[:, 0] == 80).all() and out[:, 1].all()
    bw = BFF(max_pairs=64, steps=8192, ip_wrap=True, literal=True)
    mem, out = bw.execute(np.concatenate([np.repeat(t[None], 32, 0), quiet], axis=1))
    sims = np.array([similarity(t, m)[0] for m in mem[:, TAPE:]])
    assert sims.min() >= 0.95, sims
    assert out[:, 1].all()  # open: the pointer runs into the partner
    assert (mem[:, :TAPE] == t).all()  # and the organism is intact (phase-consistent overwrite)
    random_p = rng.integers(0, 256, size=(64, TAPE), dtype=np.uint8)
    mem2, out2 = bw.execute(np.concatenate([np.repeat(t[None], 64, 0), random_p], axis=1))
    sims2 = np.array([similarity(t, m)[0] for m in mem2[:, TAPE:]])
    frac = (sims2 >= 0.75).mean()
    assert 0.3 < frac < 1.0, frac  # partner-dependent, like the Z80 pusher
    # without the switch the byte P is a no-op and nothing is copied
    b0 = BFF(max_pairs=64, steps=8192, ip_wrap=True, literal=False)
    mem0, _ = b0.execute(np.concatenate([np.repeat(t[None], 32, 0), quiet], axis=1))
    assert np.array_equal(mem0[:, TAPE:], quiet)


@pytest.mark.parametrize("ip_wrap", [False, True])
def test_nohalt_kernel_matches_reference(ip_wrap):
    rng = np.random.default_rng(23)
    b = BFF(max_pairs=256, steps=2048, ip_wrap=ip_wrap, literal=True, nohalt=True)
    pairs = rng.integers(0, 256, size=(256, PAIR), dtype=np.uint8)
    for i in range(0, 256, 2):
        prog = "".join(rng.choice(list("<>{}+-.,[]P") + ["x"] * 3, size=rng.integers(4, 40)))
        pairs[i, : len(prog)] = np.frombuffer(prog.encode(), dtype=np.uint8)
    mem, out = b.execute(pairs)
    for i in range(256):
        ref_mem, ref = reference_execute(pairs[i], 2048, ip_wrap=ip_wrap, literal=True, nohalt=True)
        assert np.array_equal(mem[i], ref_mem), f"pair {i}"
        assert tuple(out[i]) == (ref["executed"], ref["entered"], ref["max_pc"], ref["writesB"]), f"pair {i}: {out[i]} vs {ref}"


def test_nohalt_makes_unmatched_brackets_noops():
    t = enc("]" * 8 + "x" * 56)  # eight unmatched ] : halts at the first one normally, runs through under nohalt
    pair = np.concatenate([t, np.full(TAPE, ord("x"), dtype=np.uint8)])
    _, o = BFF(max_pairs=4, steps=1000, ip_wrap=False).execute(pair[None])
    assert o[0, 0] == 1 and o[0, 1] == 0  # one instruction, never reached the partner
    _, o2 = BFF(max_pairs=4, steps=1000, ip_wrap=False, nohalt=True).execute(pair[None])
    assert o2[0, 0] == PAIR and o2[0, 1] == 1  # one full pass, entered the partner
    # a matched loop still loops under nohalt (tape[h0] = '[' is non-zero)
    t2 = enc("[]" + "x" * 62)
    _, o3 = BFF(max_pairs=4, steps=1000, ip_wrap=False, nohalt=True).execute(np.concatenate([t2, np.full(TAPE, ord("x"), dtype=np.uint8)])[None])
    assert o3[0, 0] == 1000 and o3[0, 1] == 0
