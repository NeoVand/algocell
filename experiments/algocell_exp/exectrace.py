"""Traced execution of (A, B) pairs: final memories plus, per pair, the set of addresses fetched as instruction stream.

    res, masks = execute_pairs_traced(pairs, L, steps, zero_halts=False)   # masks: (N, 8) uint32, bit a = address a fetched
    ex = exec_positions(masks, 2 * L)                                       # (N, 2L) bool

Pipelines are compiled from the traced executor derived by gen_trace_shader from the executor assay.py would use.
"""
from __future__ import annotations

import numpy as np

from . import assay as A
from .gen_trace_shader import ensure

_PIPE: dict = {}


def _pipeline(L: int, mem_length: int | None, zero_halts: bool):
    P = int(mem_length) if mem_length else 2 * L
    key = (L, P, bool(zero_halts))
    if key not in _PIPE:
        dev = A.get_device()
        src = A._executor_path(L, mem_length, bool(zero_halts))
        path = ensure(src)
        module = dev.create_shader_module(code=path.read_text())
        storage = {"type": A.wgpu.BufferBindingType.storage}
        layout = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": A.wgpu.ShaderStage.COMPUTE, "buffer": {"type": A.wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": A.wgpu.ShaderStage.COMPUTE, "buffer": storage},
            {"binding": 2, "visibility": A.wgpu.ShaderStage.COMPUTE, "buffer": storage}])
        pl = dev.create_pipeline_layout(bind_group_layouts=[layout])
        _PIPE[key] = (dev.create_compute_pipeline(layout=pl, compute={"module": module, "entry_point": "z80_test"}), layout)
    return _PIPE[key]


def execute_pairs_traced(pairs: np.ndarray, L: int, steps: int, suppress=(), mem_length: int | None = None, zero_halts: bool = False):
    N = pairs.shape[0]
    assert pairs.shape[1] == 2 * L
    dev = A.get_device()
    pipe, layout = _pipeline(L, mem_length, zero_halts)
    wpc = (L + 3) // 4
    words = np.zeros((N, 2 * wpc * 4), dtype=np.uint8)
    words[:, :L] = pairs[:, :L]
    words[:, 4 * wpc: 4 * wpc + L] = pairs[:, L:]
    io_data = np.ascontiguousarray(words).view(np.uint32)
    params = np.zeros(32, dtype=np.uint32)
    params[2], params[3], params[4], params[6] = L, 2 * L, N, steps
    params[8:] = A.make_masks(A.resolve(list(suppress)))
    B = A.wgpu.BufferUsage
    io_buf = dev.create_buffer(size=io_data.nbytes, usage=B.STORAGE | B.COPY_SRC | B.COPY_DST)
    dev.queue.write_buffer(io_buf, 0, io_data)
    regs_buf = dev.create_buffer(size=N * 20 * 4, usage=B.STORAGE | B.COPY_SRC)
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
    regs = np.frombuffer(dev.queue.read_buffer(regs_buf), dtype=np.uint32).reshape(N, 20)
    res = np.empty((N, 2 * L), dtype=np.uint8)
    res[:, :L] = out[:, :L]
    res[:, L:] = out[:, 4 * wpc: 4 * wpc + L]
    masks = regs[:, 12:20].copy()
    for b in (io_buf, regs_buf, p_buf):
        b.destroy()
    return res, masks


def exec_positions(masks: np.ndarray, P: int) -> np.ndarray:
    """(N, P) bool: address a (mod P) was fetched as instruction stream."""
    bits = np.unpackbits(masks.view(np.uint8), axis=1, bitorder="little")   # (N, 256)
    return bits[:, :P].astype(bool)
