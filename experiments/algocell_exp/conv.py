"""Executors and the culture test under the convention-test rules (REVISION_PREREG R3/C): `randreg` (random initial
registers) and `randsp` (random initial stack pointer), with `None` = the standard executor. Each call draws fresh
registers per case from the call seed; pass different seeds to different calls.

    res = execute(pairs, L, steps, variant, seed)                       # (N, 2L) final memories
    res, masks = execute(pairs, L, steps, variant, seed, traced=True)   # plus fetched-address bitmaps (exectrace format)
    rows = assay_many_conv(tapes, variant, n=32, seed=0)                # assay_many() under the variant
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import assay as A
from .gen_trace_shader import ensure

SHADER_DIR = Path(__file__).parent / "shader"
_PIPE: dict = {}


def _path(L: int, variant: str | None, traced: bool) -> Path:
    p = SHADER_DIR / (f"z80_test_{variant}_L{L}.wgsl" if variant else f"z80_test_L{L}.wgsl")
    if not p.exists():
        raise ValueError(f"missing executor {p.name} (python -m algocell_exp.gen_conv_shader)")
    return ensure(p) if traced else p


def _pipeline(path: Path):
    if path not in _PIPE:
        dev = A.get_device()
        module = dev.create_shader_module(code=path.read_text())
        storage = {"type": A.wgpu.BufferBindingType.storage}
        layout = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": A.wgpu.ShaderStage.COMPUTE, "buffer": {"type": A.wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": A.wgpu.ShaderStage.COMPUTE, "buffer": storage},
            {"binding": 2, "visibility": A.wgpu.ShaderStage.COMPUTE, "buffer": storage}])
        pl = dev.create_pipeline_layout(bind_group_layouts=[layout])
        _PIPE[path] = (dev.create_compute_pipeline(layout=pl, compute={"module": module, "entry_point": "z80_test"}), layout)
    return _PIPE[path]


def execute(pairs: np.ndarray, L: int, steps: int, variant: str | None, seed: int, suppress=(), traced: bool = False):
    N = pairs.shape[0]
    assert pairs.shape[1] == 2 * L
    dev = A.get_device()
    pipe, layout = _pipeline(_path(L, variant, traced))
    wpc = (L + 3) // 4
    words = np.zeros((N, 2 * wpc * 4), dtype=np.uint8)
    words[:, :L] = pairs[:, :L]
    words[:, 4 * wpc: 4 * wpc + L] = pairs[:, L:]
    io_data = np.ascontiguousarray(words).view(np.uint32)
    params = np.zeros(32, dtype=np.uint32)
    params[2], params[3], params[4], params[6], params[7] = L, 2 * L, N, steps, np.uint32(seed & 0xFFFFFFFF)
    params[8:] = A.make_masks(A.resolve(list(suppress)))
    nreg = 20 if traced else 12
    B = A.wgpu.BufferUsage
    io_buf = dev.create_buffer(size=io_data.nbytes, usage=B.STORAGE | B.COPY_SRC | B.COPY_DST)
    dev.queue.write_buffer(io_buf, 0, io_data)
    regs_buf = dev.create_buffer(size=N * nreg * 4, usage=B.STORAGE | B.COPY_SRC)
    p_buf = dev.create_buffer(size=128, usage=B.UNIFORM | B.COPY_DST)
    dev.queue.write_buffer(p_buf, 0, params)
    bg = dev.create_bind_group(layout=layout, entries=[
        {"binding": 0, "resource": {"buffer": p_buf, "offset": 0, "size": 128}},
        {"binding": 1, "resource": {"buffer": io_buf, "offset": 0, "size": io_buf.size}},
        {"binding": 2, "resource": {"buffer": regs_buf, "offset": 0, "size": regs_buf.size}}])
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
    res[:, L:] = out[:, 4 * wpc: 4 * wpc + L]
    masks = None
    if traced:
        regs = np.frombuffer(dev.queue.read_buffer(regs_buf), dtype=np.uint32).reshape(N, 20)
        masks = regs[:, 12:20].copy()
    for b in (io_buf, regs_buf, p_buf):
        b.destroy()
    return (res, masks) if traced else res


def assay_many_conv(tapes: np.ndarray, variant: str | None, n: int = 32, seed: int = 0, suppress=()) -> list[dict]:
    """assay_many() (culture test as organism A, gen2, faithfulness) with every encounter under the variant."""
    T = np.ascontiguousarray(tapes).astype(np.uint8)
    M, L = T.shape
    rng = np.random.default_rng(seed)
    R1 = rng.integers(0, 256, size=(n, L), dtype=np.uint8)
    R2 = rng.integers(0, 256, size=(n, L), dtype=np.uint8)
    s1, s2 = (int(x) for x in rng.integers(1, 2**31, size=2))
    TT = np.repeat(T, n, axis=0)
    RR = np.tile(R1, (M, 1))
    res = execute(np.concatenate([TT, RR], 1), L, 128, variant, s1, suppress)
    before = A._best_shift_match_rows(RR, TT)
    after, _ = A._best_shift_rows(res[:, L:], TT)
    score = A._norm_gain_rows(before, after, M, n)
    RR2 = np.tile(R2, (M, 1))
    res2 = execute(np.concatenate([res[:, L:], RR2], 1), L, 128, variant, s2, suppress)
    before2 = A._best_shift_match_rows(RR2, TT)
    after2 = A._best_shift_match_rows(res2[:, L:], TT)
    gen2 = A._norm_gain_rows(before2, after2, M, n)
    copies = (after >= 0.75).reshape(M, n)
    out = []
    for i in range(M):
        g2 = float(gen2[i])
        q75 = float(copies[i].mean())
        out.append({"score": float(score[i]), "gen2_score": g2, "offspring_within_q": q75,
                    "is_replicator": bool(np.isfinite(g2) and g2 >= A.GEN2_MIN), "faithful": bool(np.isfinite(g2) and g2 >= A.GEN2_MIN and q75 >= A.FAITHFUL_MIN)})
    return out
