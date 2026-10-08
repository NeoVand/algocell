"""Exact pointer traces on the GPU pair executor (the same kernel the soup runs): registers after k steps, k = 0..K.

    python trace_z80.py  →  results/concept/traces.json  {name: {"L", "tape", "pc", "sp", "writes_b"}}

Used by manuscript/figures/concept.py (Fig. 2d cycles, Fig. 5a trajectories). The partner is all zeros (NOPs) so the
trace shows the organism's own control flow; the reported pc/sp are reduced modulo the ring length 2L.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from algocell_exp import assay as A  # noqa: E402


def regs_after(pair: bytes, L: int, steps: int) -> dict:
    wg = A.wgpu
    dev = A.get_device()
    pipe, layout = A._pipeline(L, None)
    wpc = (L + 3) // 4
    words = np.zeros((1, 2 * wpc * 4), dtype=np.uint8)
    p = np.frombuffer(pair, dtype=np.uint8)
    words[0, :L] = p[:L]
    words[0, 4 * wpc: 4 * wpc + L] = p[L:]
    io = np.ascontiguousarray(words).view(np.uint32)
    params = np.zeros(32, dtype=np.uint32)
    params[2], params[3], params[4], params[6] = L, 2 * L, 1, steps
    params[8:] = A.make_masks(A.resolve([]))
    B = wg.BufferUsage
    io_buf = dev.create_buffer(size=io.nbytes, usage=B.STORAGE | B.COPY_SRC | B.COPY_DST)
    dev.queue.write_buffer(io_buf, 0, io)
    regs_buf = dev.create_buffer(size=12 * 4, usage=B.STORAGE | B.COPY_SRC)
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
    ps.dispatch_workgroups(1)
    ps.end()
    dev.queue.submit([enc.finish()])
    regs = np.frombuffer(dev.queue.read_buffer(regs_buf), dtype=np.uint32).copy()
    for b in (io_buf, regs_buf, p_buf):
        b.destroy()
    return {"pc": int(regs[9]) % (2 * L), "sp": int(regs[8]) % (2 * L), "writes_b": int(regs[11])}


def trace(tape: bytes, K: int) -> dict:
    L = len(tape)
    pair = tape + bytes(L)
    rows = [regs_after(pair, L, k) for k in range(K + 1)]
    return {"L": L, "tape": tape.hex(" "), "pc": [r["pc"] for r in rows], "sp": [r["sp"] for r in rows], "writes_b": [r["writes_b"] for r in rows]}


def modal_final(g: pd.DataFrame, L: int, prefix: str) -> bytes:
    d = g[g["L"] == L]["final_tape"]
    d = d[d.str.replace(" ", "").str.startswith(prefix)]
    return bytes.fromhex(d.mode().iloc[0].replace(" ", ""))


def main() -> None:
    g = pd.read_csv(os.path.join(HERE, "results", "stageG", "stageG", "stage_g_runs.csv"))
    out = {
        "pusher_L16": trace(bytes.fromhex("01c5" * 8), 24),
        "pusher_L20": trace(bytes.fromhex("01c5" * 10), 32),
        "retnz_L16": trace(bytes.fromhex("ade321e321c0adc0" * 2), 48),
        "jrnz_L50": trace(modal_final(g, 50, "21e521e5"), 48),
        "djnz_L20": trace(modal_final(g, 20, "21e521e5214e10"), 48),
        "ldir_L20": trace(modal_final(g, 20, "1ea4edb0"), 40),
    }
    path = os.path.join(HERE, "results", "concept", "traces.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in out.items():
        print(f"{k:>12}: L={v['L']} pc[:20]={v['pc'][:20]}")
    print("wrote", path)


if __name__ == "__main__":
    main()
