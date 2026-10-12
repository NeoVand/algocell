"""EXPLORATORY: observational roles on the frozen X1 executor, no ISA reimplementation.

Bits are opcode/prefix=1, immediate/displacement=2, data read=4, write=8.
Roles refer to addresses over the encounter, not the ancestry of a value after
self modification. In particular a data read is not evidence of control use.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from algocell_exp import assay as A
from z80_ring_gpu import build

PIPE = {}


def source(L, zh=False):
    path = A._executor_path(L, None, zh)
    code = path.read_text()
    original_hash = hashlib.sha256(code.encode()).hexdigest()
    def replace(old, new, count=1):
        nonlocal code
        assert code.count(old) == count, (path, old, code.count(old))
        code = code.replace(old, new)
    replace('fn mem_read(addr: u32) -> u32 { return mem[addr % MEM_LENGTH]; }',
            'var<private> role_mask: array<u32, MEM_LENGTH>;\n'
            'var<private> fetch_role: u32;\n'
            'fn mem_read(addr: u32) -> u32 { let a = addr % MEM_LENGTH; role_mask[a] |= 4u; return mem[a]; }')
    replace('\tmem[a] = val & 0xffu;', '\trole_mask[a] |= 8u;\n\tmem[a] = val & 0xffu;')
    replace('    let val = mem_read(cpu_pc);',
            '    let a = cpu_pc % MEM_LENGTH;\n    role_mask[a] |= fetch_role;\n    let val = mem[a];')
    replace('fn z80_fetch_word() -> u32 {',
            'fn crystal_opcode_fetch() -> u32 {\n    fetch_role = 1u;\n'
            '    let v = z80_fetch();\n    fetch_role = 2u;\n    return v;\n}\n\nfn z80_fetch_word() -> u32 {')
    replace('let op = z80_fetch();', 'let op = crystal_opcode_fetch();', 2)
    for old in ('var op = z80_fetch();', 'let next = z80_fetch();', 'let cbop = z80_fetch();'):
        replace(old, old.replace('z80_fetch()', 'crystal_opcode_fetch()'))
    replace('cpu_halted=0u; cpu_iff1=0u; cpu_iff2=0u; cpu_writes_a=0u; cpu_writes_b=0u;',
            'cpu_halted=0u; cpu_iff1=0u; cpu_iff2=0u; cpu_writes_a=0u; cpu_writes_b=0u;\n'
            '\tfetch_role = 2u;\n\tfor (var i=0u; i<MEM_LENGTH; i++) { role_mask[i]=0u; }')
    replace('\tlet rbase = case_id * 12u;', '\tlet rbase = case_id * (12u + MEM_LENGTH);')
    anchor = 'regs[rbase+8u]=cpu_sp; regs[rbase+9u]=cpu_pc; regs[rbase+10u]=cpu_writes_a; regs[rbase+11u]=cpu_writes_b;'
    replace(anchor, anchor+'\n\tfor (var i=0u; i<MEM_LENGTH; i++) { regs[rbase+12u+i]=role_mask[i]; }')
    return code, original_hash, hashlib.sha256(code.encode()).hexdigest()


def run(pairs, L, zh=False, steps=128):
    """The assay's buffer layout and shader, with additional readback only."""
    t0 = time.monotonic()
    dev = A.get_device()
    if (L, zh) not in PIPE:
        code, _, _ = source(L, zh)
        PIPE[L, zh] = build(dev, code)
    pipe, layout = PIPE[L, zh]
    n = len(pairs)
    wpc = (L+3)//4
    packed = np.zeros((n, 8*wpc), np.uint8)
    packed[:, :L] = pairs[:, :L]
    packed[:, 4*wpc:4*wpc+L] = pairs[:, L:]
    params = np.zeros(32, np.uint32)
    params[2], params[3], params[4], params[6] = L, 2*L, n, steps
    B = A.wgpu.BufferUsage
    io = dev.create_buffer(size=packed.nbytes, usage=B.STORAGE|B.COPY_SRC|B.COPY_DST)
    reg = dev.create_buffer(size=n*(12+2*L)*4, usage=B.STORAGE|B.COPY_SRC)
    par = dev.create_buffer(size=params.nbytes, usage=B.UNIFORM|B.COPY_DST)
    dev.queue.write_buffer(io, 0, packed)
    dev.queue.write_buffer(par, 0, params)
    bg = dev.create_bind_group(layout=layout, entries=[
        {'binding': i, 'resource': {'buffer': b, 'offset': 0, 'size': b.size}}
        for i,b in enumerate((par,io,reg))])
    enc = dev.create_command_encoder()
    p = enc.begin_compute_pass()
    p.set_pipeline(pipe)
    p.set_bind_group(0,bg)
    p.dispatch_workgroups((n+63)//64)
    p.end()
    dev.queue.submit([enc.finish()])
    out = np.frombuffer(dev.queue.read_buffer(io), np.uint8).reshape(n,8*wpc)
    regs = np.frombuffer(dev.queue.read_buffer(reg), np.uint32).reshape(n,12+2*L).copy()
    mem = np.concatenate((out[:,:L],out[:,4*wpc:4*wpc+L]),1).copy()
    for b in (io,reg,par):
        b.destroy()
    elapsed = time.monotonic()-t0
    assert elapsed < 60, f'EXPLORATORY: dispatch exceeded 60 s: {elapsed}'
    return mem, regs[:,12:].astype(np.uint8)
