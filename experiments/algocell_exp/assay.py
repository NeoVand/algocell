"""Replication assay: does a tape copy itself when executed against random neighbours?

Measurement semantics (fixed 2026-10-07 after review; see PLAN change log):
  score      — headroom-normalised gain in best-cyclic-shift similarity of the partner to T
               after T runs as A (or as B), over partners that were not already copies.
  gen2_score — the SAME gain for the partners T produced (offspring) when they run as A
               against fresh partners. It is a two-generation YIELD: offspring T failed to
               convert are included, so gen2 ≈ P(copy) × P(offspring copies).
  gen2_cond  — gen2 restricted to offspring that are >= 75% copies: heritability given a
               copy was made.
  offspring_within_q — share of partners that became >= 75% copies (faithfulness).
  A replicator is gen2_score >= 0.3 (pre-registered); it is FAITHFUL when additionally
  offspring_within_q >= 0.5 (a return-address smear passes the second alone, a drifting
  CALL chain passes the first alone; neither passes both).
  self_preserved_as_B — fraction of T's bytes intact after a random partner runs first
               (vulnerability: small copiers can be overwritten before they act).
  In-situ partners: `n_informative` partners had prior similarity < 0.75; when a soup is
  saturated with copies the gains are NaN, never 0 (0 used to mean "sterile").

The dominant tape T is run as program A with N random neighbours as B (and as
program B with random A), for a given step budget and suppression set, using
the single-pair executor shader (the same core + host bits as the simulation).
Score = mean over neighbours of the best shift-aligned fraction of T's bytes
present in the neighbour's tape afterwards, minus the same quantity before
(so a tape that merely resembles random noise scores ~0). A faithful copier
scores near 1; a byte-pattern smear scores low because it does not reproduce
the instruction that writes it.
"""

from __future__ import annotations

import numpy as np
import wgpu

from .isa import masks as make_masks
from .isa import resolve
from .soup import SHADER_DIR, get_device

_PIPE: dict = {}
GEN2_MIN = 0.3        # pre-registered heritability threshold (PLAN, 2026-10-07, before any sweep)
FAITHFUL_MIN = 0.5    # share of partners that became >= 75% copies (PLAN change log, 2026-10-07)


def _pipeline(tape_length: int = 16):
    """Pipeline for the single-pair executor sized for this tape length (cached per L)."""
    if tape_length not in _PIPE:
        dev = get_device()
        path = SHADER_DIR / f"z80_test_L{tape_length}.wgsl"
        if not path.exists():
            if tape_length <= 20:
                path = SHADER_DIR / "z80_test.wgsl"  # 40-byte memory: fits pairs up to 40 bytes
            else:
                raise ValueError(f"no exported executor for tape length {tape_length} (run `npm run export:sim`)")
        module = dev.create_shader_module(code=path.read_text())
        storage = {"type": wgpu.BufferBindingType.storage}
        layout = dev.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": storage},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": storage},
            ]
        )
        pl = dev.create_pipeline_layout(bind_group_layouts=[layout])
        _PIPE[tape_length] = (dev.create_compute_pipeline(layout=pl, compute={"module": module, "entry_point": "z80_test"}), layout)
    return _PIPE[tape_length]


def execute_pairs(pairs: np.ndarray, tape_length: int, z80_steps: int, suppress=()) -> np.ndarray:
    """Run many (A,B) pairs independently. pairs: (N, 2L) uint8 → final memories (N, 2L)."""
    N = pairs.shape[0]
    L = tape_length
    assert pairs.shape[1] == 2 * L
    dev = get_device()
    pipe, layout = _pipeline(L)
    wpc = (L + 3) // 4
    words = np.zeros((N, 2 * wpc * 4), dtype=np.uint8)
    words[:, : 4 * wpc][:, :L] = pairs[:, :L]
    words[:, 4 * wpc : 4 * wpc + L] = pairs[:, L:]
    io_data = np.ascontiguousarray(words).view(np.uint32)
    params = np.zeros(32, dtype=np.uint32)
    params[2], params[3], params[4], params[6] = L, 2 * L, N, z80_steps
    params[8:] = make_masks(resolve(list(suppress)))
    B = wgpu.BufferUsage
    io_buf = dev.create_buffer(size=io_data.nbytes, usage=B.STORAGE | B.COPY_SRC | B.COPY_DST)
    dev.queue.write_buffer(io_buf, 0, io_data)
    regs_buf = dev.create_buffer(size=N * 12 * 4, usage=B.STORAGE | B.COPY_SRC)
    p_buf = dev.create_buffer(size=128, usage=B.UNIFORM | B.COPY_DST)
    dev.queue.write_buffer(p_buf, 0, params)
    bg = dev.create_bind_group(
        layout=layout,
        entries=[
            {"binding": 0, "resource": {"buffer": p_buf, "offset": 0, "size": 128}},
            {"binding": 1, "resource": {"buffer": io_buf, "offset": 0, "size": io_buf.size}},
            {"binding": 2, "resource": {"buffer": regs_buf, "offset": 0, "size": regs_buf.size}},
        ],
    )
    enc = dev.create_command_encoder()
    p = enc.begin_compute_pass()
    p.set_pipeline(pipe)
    p.set_bind_group(0, bg)
    p.dispatch_workgroups(-(-N // 64))
    p.end()
    dev.queue.submit([enc.finish()])
    out = np.frombuffer(dev.queue.read_buffer(io_buf), dtype=np.uint8).reshape(N, 2 * wpc * 4)
    res = np.empty((N, 2 * L), dtype=np.uint8)
    res[:, :L] = out[:, :L]
    res[:, L:] = out[:, 4 * wpc : 4 * wpc + L]
    for b in (io_buf, regs_buf, p_buf):
        b.destroy()
    return res


def _best_shift_match(target: np.ndarray, tape: np.ndarray) -> np.ndarray:
    """For each row of `target` (N, L): max over cyclic shifts of fraction of bytes equal to `tape`."""
    L = tape.size
    best = np.zeros(target.shape[0])
    for s in range(L):
        best = np.maximum(best, (target == np.roll(tape, s)[None, :]).mean(axis=1))
    return best


def assay(
    tape: bytes | np.ndarray,
    z80_steps: int = 128,
    suppress=(),
    n: int = 64,
    seed: int = 0,
    neighbors: np.ndarray | None = None,
    min_informative: int = 16,
) -> dict:
    """Replication score of a tape under a given budget and suppression set.

    `neighbors` (M, L) uint8 — if given, partners are drawn from it instead of
    uniform random bytes ("in situ" assay against the actual soup). Members of
    a replicator cloud can depend on their partners' bytes (e.g. POP DE reads
    the pointer from the neighbour's tail), so the random-neighbour score is a
    lower bound and the in-situ score is the ecological truth."""
    T = np.frombuffer(bytes(tape), dtype=np.uint8) if not isinstance(tape, np.ndarray) else tape.astype(np.uint8)
    L = T.size
    rng = np.random.default_rng(seed)
    if neighbors is not None:
        # Draw until at least `min_informative` partners are not already copies of T (or give up
        # after 8 rounds): in a soup saturated with copies a copier has no headroom to show.
        R = np.ascontiguousarray(neighbors[rng.integers(0, neighbors.shape[0], size=n)]).astype(np.uint8)
        for _ in range(8):
            if int((_best_shift_match(R, T) < 0.75).sum()) >= min(min_informative, n):
                break
            more = np.ascontiguousarray(neighbors[rng.integers(0, neighbors.shape[0], size=n)]).astype(np.uint8)
            keep = R[_best_shift_match(R, T) < 0.75]
            R = np.concatenate([keep, more], axis=0)[:n]
    else:
        R = rng.integers(0, 256, size=(n, L), dtype=np.uint8)
    # T as A, random B
    pairs_a = np.concatenate([np.repeat(T[None, :], n, axis=0), R], axis=1)
    # random A, T as B
    pairs_b = np.concatenate([R, np.repeat(T[None, :], n, axis=0)], axis=1)
    res_a = execute_pairs(pairs_a, L, z80_steps, suppress)
    res_b = execute_pairs(pairs_b, L, z80_steps, suppress)
    before_i = _best_shift_match(R, T)
    sim_b = _best_shift_match(res_a[:, L:], T)          # T (as A) wrote itself into B?
    sim_a = _best_shift_match(res_b[:, :L], T)          # T (as B) wrote itself into A?
    # Gain normalised by headroom, over partners that were not already copies.
    # In a soup saturated with copies a copier cannot raise its partner's
    # similarity, so a raw (after - before) would call it sterile.
    into_b = _norm_gain(before_i, sim_b)
    into_a = _norm_gain(before_i, sim_a)
    self_kept_a = float((res_a[:, :L] == T[None, :]).mean())  # T survives its own execution as A
    self_kept_b = float((res_b[:, L:] == T[None, :]).mean())  # T survives when the partner runs first (vulnerability)
    n_informative = int((before_i < 0.75).sum())
    # Heritability: do the offspring (the B tapes T produced) themselves copy
    # T-like material into fresh neighbours? A byte-pattern smear can score
    # well in one generation (it writes its periodic payload) but its offspring
    # lack the instruction that produced it, so generation 2 collapses.
    offspring = res_a[:, L:]
    if neighbors is not None:
        R2 = np.ascontiguousarray(neighbors[rng.integers(0, neighbors.shape[0], size=n)]).astype(np.uint8)
    else:
        R2 = rng.integers(0, 256, size=(n, L), dtype=np.uint8)
    res_g2 = execute_pairs(np.concatenate([offspring, R2], axis=1), L, z80_steps, suppress)
    before2 = _best_shift_match(R2, T)
    after2 = _best_shift_match(res_g2[:, L:], T)
    gen2 = _norm_gain(before2, after2)
    copies = sim_b >= 0.75
    gen2_cond = _norm_gain(before2[copies], after2[copies]) if copies.any() else float("nan")
    return {
        "copy_into_neighbor_as_A": float(into_b),
        "copy_into_neighbor_as_B": float(into_a),
        "baseline_similarity": float(before_i.mean()),
        "n_informative": n_informative,
        "self_preserved_as_A": self_kept_a,
        "self_preserved_as_B": self_kept_b,
        "offspring_within_q": float(copies.mean()),  # share of partners that became >= 75% copies (faithfulness)
        "gen2_score": float(gen2),
        "gen2_cond": float(gen2_cond),
        "score": float(max(into_b, into_a)),
        "is_replicator": bool(gen2 >= GEN2_MIN),
        "faithful": bool(gen2 >= GEN2_MIN and copies.mean() >= FAITHFUL_MIN),
    }


def _norm_gain(before: np.ndarray, after: np.ndarray, max_before: float = 0.75) -> float:
    """Mean of (after - before) / (1 - before) over partners with before < max_before.
    NaN when every partner was already a copy (no headroom to measure; it used to return 0,
    which read as "sterile" in saturated soups)."""
    m = before < max_before
    if not m.any():
        return float("nan")
    return float(((after[m] - before[m]) / (1.0 - before[m])).mean())


if __name__ == "__main__":
    import sys

    cases = {
        "load-push 01 c5": bytes.fromhex("01c5" * 8),
        "ex-sp 21 e3": bytes.fromhex("21e3" * 8),
        "ldir tiled 1e 04 ed b0": bytes.fromhex("1e04edb0" * 4),
        "ldir whole-tape 1e 10 ed b0": bytes.fromhex("1e10edb0") + bytes(12),  # DE = 16 = B (DE = 32 would alias A)
        "rst smear ff 41 00 41": bytes.fromhex("ff" + "4100" * 7 + "41"),
        "zeros": bytes(16),
        "random": bytes([37, 201, 14, 99, 180, 7, 66, 250, 121, 3, 90, 44, 210, 155, 18, 77]),
    }
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    for name, t in cases.items():
        r = assay(t, z80_steps=steps)
        print(f"{name:28s} score {r['score']:.2f} gen2 {r['gen2_score']:.2f} copies≥75% {r['offspring_within_q']:.2f} | asA {r['copy_into_neighbor_as_A']:.2f} asB {r['copy_into_neighbor_as_B']:.2f} self {r['self_preserved_as_A']:.2f}")
    print("-- stack-writes suppressed (PUSH/EX/CALL/RST removed): load-push should fail, LDIR should still copy")
    sup = ["family:stack", "family:ex", "family:call-ret", "family:rst"]
    for name in ("load-push 01 c5", "ldir whole-tape 1e 10 ed b0"):
        r = assay(cases[name], z80_steps=steps, suppress=sup)
        print(f"{name:28s} score {r['score']:.2f} gen2 {r['gen2_score']:.2f}")
