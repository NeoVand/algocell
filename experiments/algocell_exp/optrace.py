"""Traced execution that also records instruction starts: per pair, a bitmap of every address fetched as instruction
stream (as exectrace) and a second bitmap of the addresses at which an instruction began (the opcode or prefix fetch of
z80_step). Derived from the traced executor by three more hunks, each anchored on a line that occurs exactly once.

    res, fetched, starts = execute_pairs_optrace(pairs, L, steps, zero_halts=False)     # bool (N, 2L) each
"""
from __future__ import annotations

import numpy as np

from . import assay as A
from .gen_trace_shader import ensure

ANCHOR_OP = "    var op = z80_fetch();\n    r_inc(); // M1: opcode (or prefix) fetch\n"
HUNKS = [
    ("var<private> exec_mask: array<u32, 8>;", "var<private> exec_mask: array<u32, 8>;\nvar<private> op_mask: array<u32, 8>;   // instruction starts (mod MEM_LENGTH)"),
    (ANCHOR_OP, ANCHOR_OP + "    { let oa = ((cpu_pc + 0xffffu) & 0xffffu) % MEM_LENGTH; op_mask[oa >> 5u] |= (1u << (oa & 31u)); }\n"),
    ("\tfor (var i = 0u; i < 8u; i++) { exec_mask[i] = 0u; }", "\tfor (var i = 0u; i < 8u; i++) { exec_mask[i] = 0u; op_mask[i] = 0u; }"),
    ("\tlet rbase = case_id * 20u;", "\tlet rbase = case_id * 28u;"),
    ("\tfor (var i = 0u; i < 8u; i++) { regs[rbase+12u+i] = exec_mask[i]; }", "\tfor (var i = 0u; i < 8u; i++) { regs[rbase+12u+i] = exec_mask[i]; regs[rbase+20u+i] = op_mask[i]; }"),
]
_PIPE: dict = {}


def _source(L: int, zero_halts: bool) -> str:
    text = ensure(A._executor_path(L, None, zero_halts)).read_text()
    for old, new in HUNKS:
        assert text.count(old) == 1, f"anchor not unique: {old[:50]!r} ({text.count(old)})"
        text = text.replace(old, new)
    return text


def _pipeline(L: int, zero_halts: bool):
    key = (L, bool(zero_halts))
    if key not in _PIPE:
        dev = A.get_device()
        module = dev.create_shader_module(code=_source(L, zero_halts))
        storage = {"type": A.wgpu.BufferBindingType.storage}
        layout = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": A.wgpu.ShaderStage.COMPUTE, "buffer": {"type": A.wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": A.wgpu.ShaderStage.COMPUTE, "buffer": storage},
            {"binding": 2, "visibility": A.wgpu.ShaderStage.COMPUTE, "buffer": storage}])
        pl = dev.create_pipeline_layout(bind_group_layouts=[layout])
        _PIPE[key] = (dev.create_compute_pipeline(layout=pl, compute={"module": module, "entry_point": "z80_test"}), layout)
    return _PIPE[key]


def execute_pairs_optrace(pairs: np.ndarray, L: int, steps: int = 128, zero_halts: bool = False):
    N = pairs.shape[0]
    dev = A.get_device()
    pipe, layout = _pipeline(L, zero_halts)
    wpc = (L + 3) // 4
    words = np.zeros((N, 2 * wpc * 4), dtype=np.uint8)
    words[:, :L] = pairs[:, :L]
    words[:, 4 * wpc: 4 * wpc + L] = pairs[:, L:]
    io_data = np.ascontiguousarray(words).view(np.uint32)
    params = np.zeros(32, dtype=np.uint32)
    params[2], params[3], params[4], params[6] = L, 2 * L, N, steps
    params[8:] = A.make_masks(A.resolve([]))
    B = A.wgpu.BufferUsage
    io_buf = dev.create_buffer(size=io_data.nbytes, usage=B.STORAGE | B.COPY_SRC | B.COPY_DST)
    dev.queue.write_buffer(io_buf, 0, io_data)
    regs_buf = dev.create_buffer(size=N * 28 * 4, usage=B.STORAGE | B.COPY_SRC)
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
    regs = np.frombuffer(dev.queue.read_buffer(regs_buf), dtype=np.uint32).reshape(N, 28)
    res = np.empty((N, 2 * L), dtype=np.uint8)
    res[:, :L] = out[:, :L]
    res[:, L:] = out[:, 4 * wpc: 4 * wpc + L]
    unpack = lambda w: np.unpackbits(np.ascontiguousarray(w).view(np.uint8), axis=1, bitorder="little")[:, : 2 * L].astype(bool)  # noqa: E731
    for b in (io_buf, regs_buf, p_buf):
        b.destroy()
    return res, unpack(regs[:, 12:20]), unpack(regs[:, 20:28])
