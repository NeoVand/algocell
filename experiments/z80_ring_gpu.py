"""GPU half of the ring-size Z80 differential test (DEV ONLY).

Driven by src/lib/dev/z80difftest_ring.node.ts, which generates the cases, the executor WGSL
(createZ80TestShader(L, P) from src/lib/gpu/shaders.ts: the zilion core inside Algocell's memory model,
addresses taken modulo the P-byte ring, SP initialised by sp_init) and compares the result with the
z80-emulator reference. This script only dispatches the shader headlessly through wgpu-py and writes back
what the GPU produced; it contains no Z80 logic and no reference.

Input  (--inp): N cases of P = 2L bytes each, uint8, row-major (tape A then tape B).
Output (--out): for each requested step budget k (one budget, or 1..steps with --trace):
                N×P uint8 final pair memories, then N×12 uint32 registers
                (a, f, b, c, d, e, h, l, sp, pc, writes_a, writes_b), little-endian.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import wgpu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from algocell_exp.soup import adapter_summary, get_device  # noqa: E402  (same adapter policy as the runs)

REGS_PER_CASE = 12


def build(dev, code: str):
    module = dev.create_shader_module(code=code)
    storage = {"type": wgpu.BufferBindingType.storage}
    layout = dev.create_bind_group_layout(
        entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": storage},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": storage},
        ]
    )
    pl = dev.create_pipeline_layout(bind_group_layouts=[layout])
    pipe = dev.create_compute_pipeline(layout=pl, compute={"module": module, "entry_point": "z80_test"})
    return pipe, layout


def run(dev, pipe, layout, pairs: np.ndarray, L: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Same packing and Params layout as z80difftest.ts / algocell_exp.assay.execute_pairs (no suppression)."""
    N = pairs.shape[0]
    wpc = (L + 3) // 4
    words = np.zeros((N, 2 * wpc * 4), dtype=np.uint8)
    words[:, :L] = pairs[:, :L]
    words[:, 4 * wpc : 4 * wpc + L] = pairs[:, L:]
    io_data = np.ascontiguousarray(words).view(np.uint32)
    params = np.zeros(32, dtype=np.uint32)
    params[2], params[3], params[4], params[6] = L, 2 * L, N, steps  # tape_length, pair_length, count, z80_steps
    B = wgpu.BufferUsage
    io_buf = dev.create_buffer(size=io_data.nbytes, usage=B.STORAGE | B.COPY_SRC | B.COPY_DST)
    dev.queue.write_buffer(io_buf, 0, io_data)
    regs_buf = dev.create_buffer(size=N * REGS_PER_CASE * 4, usage=B.STORAGE | B.COPY_SRC)
    p_buf = dev.create_buffer(size=params.nbytes, usage=B.UNIFORM | B.COPY_DST)
    dev.queue.write_buffer(p_buf, 0, params)
    bg = dev.create_bind_group(
        layout=layout,
        entries=[
            {"binding": 0, "resource": {"buffer": p_buf, "offset": 0, "size": p_buf.size}},
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
    regs = np.frombuffer(dev.queue.read_buffer(regs_buf), dtype=np.uint32).reshape(N, REGS_PER_CASE).copy()
    mem = np.empty((N, 2 * L), dtype=np.uint8)
    mem[:, :L] = out[:, :L]
    mem[:, L:] = out[:, 4 * wpc : 4 * wpc + L]
    for b in (io_buf, regs_buf, p_buf):
        b.destroy()
    return mem, regs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shader", required=True, help="executor WGSL (createZ80TestShader(L, P))")
    ap.add_argument("--tape", type=int, required=True, help="tape length L (pair memory P = 2L)")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", action="store_true", help="run every step budget 1..steps (first-divergence tracing)")
    ap.add_argument("--chunk", type=int, default=262144, help="cases per dispatch")
    a = ap.parse_args()

    L = a.tape
    pairs = np.fromfile(a.inp, dtype=np.uint8)
    assert pairs.size % (2 * L) == 0, "input is not a whole number of 2L-byte pairs"
    pairs = pairs.reshape(-1, 2 * L)
    dev = get_device()
    pipe, layout = build(dev, Path(a.shader).read_text())
    budgets = range(1, a.steps + 1) if a.trace else [a.steps]
    with open(a.out, "wb") as f:
        for k in budgets:
            mems, regs = [], []
            for s in range(0, pairs.shape[0], a.chunk):
                m, r = run(dev, pipe, layout, pairs[s : s + a.chunk], L, k)
                mems.append(m)
                regs.append(r)
            f.write(np.concatenate(mems).tobytes())
            f.write(np.concatenate(regs).astype("<u4").tobytes())
    print(f"[z80_ring_gpu] L={L} P={2 * L} cases={pairs.shape[0]} budgets={len(budgets)} on {adapter_summary()}", file=sys.stderr)


if __name__ == "__main__":
    main()
