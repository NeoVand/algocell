"""BFF (the Brainfuck family of Agüera y Arcas et al. 2024, arXiv 2406.19108) on the GPU, with two switches.

Semantics follow cubff `bff.inc.h` (checked 2026-10-08): two 64-byte tapes concatenated into 128 bytes; instruction
pointer, head0 and head1 start at 0; heads wrap modulo 128; ten instruction bytes
    <  >  head0 -/+ 1      {  }  head1 -/+ 1      +  -  byte at head0 +/- 1
    .  tape[head1] = tape[head0]      ,  tape[head0] = tape[head1]
    [  if tape[head0] == 0: jump past the matching ]   (terminate if none)
    ]  if tape[head0] != 0: jump back to the matching [ (terminate if none); execution resumes after the [
every other byte is a no-op that still costs a step; 2^13 steps per encounter; execution ENDS when the instruction
pointer leaves [0, 128) — unless `ip_wrap` is set (our variant: the pointer wraps modulo 128, so straight-line code
can lap the tape as the Z80 ring does). `alphabet` maps each of the 256 byte values to an opcode id (0 = no-op,
1..10 = the ten instructions in the order above); the default is the ASCII map; `density_map(k)` assigns k byte
values to each instruction (noisier random code without changing the language). Byte 0 is the loop-test null as in
cubff regardless of the map.

Switch `nohalt` (the benign-tar variant): an unmatched bracket is a no-op instead of ending the encounter.

Per encounter the kernel records: steps executed, whether the instruction pointer ever entered the partner's half
(closure, Definition 2 of THEORY.md), the maximum pointer position, and the number of writes into the partner.
"""

from __future__ import annotations

import numpy as np
import wgpu

TAPE = 64
PAIR = 2 * TAPE
OPS = "<>{}+-.,[]"           # opcode ids 1..10
LIT = "P"                    # opcode id 11, the literal-push switch (byte 0x50 in the ASCII map)
WORDS_PER_PAIR = PAIR // 4

WGSL = """
struct Params { n_pairs: u32, steps: u32, ip_wrap: u32, tape_len: u32, literal: u32, nohalt: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read_write> mem: array<u32>;
@group(0) @binding(1) var<storage, read> amap: array<u32>;
@group(0) @binding(2) var<storage, read_write> outp: array<u32>;
@group(0) @binding(3) var<uniform> P: Params;

fn rd(base: u32, i: u32) -> u32 { let w = mem[base + (i >> 2u)]; return (w >> ((i & 3u) * 8u)) & 0xffu; }
fn wr(base: u32, i: u32, v: u32) {
  let idx = base + (i >> 2u); let sh = (i & 3u) * 8u;
  mem[idx] = (mem[idx] & ~(0xffu << sh)) | ((v & 0xffu) << sh);
}

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= P.n_pairs) { return; }
  let base = p * (2u * P.tape_len / 4u);
  let T = 2u * P.tape_len;
  var pc: i32 = 0;
  var h0: u32 = 0u;
  var h1: u32 = 0u;
  var entered: u32 = 0u;
  var maxpc: u32 = 0u;
  var writesB: u32 = 0u;
  var executed: u32 = 0u;
  for (var s: u32 = 0u; s < P.steps; s = s + 1u) {
    if (pc < 0 || pc >= i32(T)) {
      if (P.ip_wrap == 1u) { pc = ((pc % i32(T)) + i32(T)) % i32(T); } else { break; }
    }
    let upc = u32(pc);
    if (upc > maxpc) { maxpc = upc; }
    if (upc >= P.tape_len) { entered = 1u; }
    let op = amap[rd(base, upc)];
    var stop = false;
    switch op {
      case 1u: { h0 = (h0 + T - 1u) % T; }
      case 2u: { h0 = (h0 + 1u) % T; }
      case 3u: { h1 = (h1 + T - 1u) % T; }
      case 4u: { h1 = (h1 + 1u) % T; }
      case 5u: { wr(base, h0, rd(base, h0) + 1u); if (h0 >= P.tape_len) { writesB = writesB + 1u; } }
      case 6u: { wr(base, h0, rd(base, h0) + 255u); if (h0 >= P.tape_len) { writesB = writesB + 1u; } }
      case 7u: { wr(base, h1, rd(base, h0)); if (h1 >= P.tape_len) { writesB = writesB + 1u; } }
      case 8u: { wr(base, h0, rd(base, h1)); if (h0 >= P.tape_len) { writesB = writesB + 1u; } }
      case 9u: {
        if (rd(base, h0) == 0u) {
          var depth: i32 = 1; var q: u32 = upc + 1u; var found = false;
          loop {
            if (q >= T) { break; }
            let o = amap[rd(base, q)];
            if (o == 9u) { depth = depth + 1; } else if (o == 10u) { depth = depth - 1; if (depth == 0) { found = true; break; } }
            q = q + 1u;
          }
          if (!found) { if (P.nohalt == 0u) { stop = true; } } else { pc = i32(q); }
        }
      }
      case 11u: {
        // literal push (variant switch): write the two following code bytes below head1, as the Z80 pusher does
        if (P.literal == 1u) {
          let a = rd(base, (upc + 1u) % T); let b = rd(base, (upc + 2u) % T);
          h1 = (h1 + T - 2u) % T;
          wr(base, (h1 + 1u) % T, a); wr(base, h1, b);
          if (((h1 + 1u) % T) >= P.tape_len) { writesB = writesB + 1u; }
          if (h1 >= P.tape_len) { writesB = writesB + 1u; }
          pc = pc + 2;
        }
      }
      case 10u: {
        if (rd(base, h0) != 0u) {
          var depth: i32 = 1; var q: i32 = pc - 1; var found = false;
          loop {
            if (q < 0) { break; }
            let o = amap[rd(base, u32(q))];
            if (o == 10u) { depth = depth + 1; } else if (o == 9u) { depth = depth - 1; if (depth == 0) { found = true; break; } }
            q = q - 1;
          }
          if (!found) { if (P.nohalt == 0u) { stop = true; } } else { pc = q; }
        }
      }
      default: {}
    }
    executed = executed + 1u;
    if (stop) { break; }
    pc = pc + 1;
  }
  outp[p * 4u] = executed; outp[p * 4u + 1u] = entered; outp[p * 4u + 2u] = maxpc; outp[p * 4u + 3u] = writesB;
}
"""

_DEVICE = None


def get_device():
    global _DEVICE
    if _DEVICE is None:
        from algocell_exp.soup import get_device as g
        _DEVICE = g()
    return _DEVICE


def ascii_map(literal: bool = False) -> np.ndarray:
    m = np.zeros(256, dtype=np.uint32)
    for i, c in enumerate(OPS, start=1):
        m[ord(c)] = i
    if literal:
        m[ord(LIT)] = 11
    return m


def density_map(k: int, seed: int = 0) -> np.ndarray:
    """k byte values per instruction (k = 1 is the ASCII map's density, 10/256). Byte 0 stays a no-op (the loop null)."""
    rng = np.random.default_rng(seed)
    m = np.zeros(256, dtype=np.uint32)
    for i, c in enumerate(OPS, start=1):
        m[ord(c)] = i
    pool = [b for b in range(1, 256) if m[b] == 0 and b != ord(LIT)]
    rng.shuffle(pool)
    extra = k - 1
    for i in range(1, 11):
        for _ in range(extra):
            if pool:
                m[pool.pop()] = i
    return m


class BFF:
    """Executes many 128-byte pairs for `steps` instructions on the GPU."""

    def __init__(self, max_pairs: int = 1 << 16, steps: int = 1 << 13, ip_wrap: bool = False, alphabet: np.ndarray | None = None, literal: bool = False, nohalt: bool = False):
        self.dev = get_device()
        self.max_pairs = max_pairs
        self.steps = steps
        self.ip_wrap = ip_wrap
        self.literal = literal
        self.nohalt = nohalt
        self.alphabet = ascii_map(literal) if alphabet is None else np.asarray(alphabet, dtype=np.uint32)
        B = wgpu.BufferUsage
        self.mem_buf = self.dev.create_buffer(size=max_pairs * PAIR, usage=B.STORAGE | B.COPY_DST | B.COPY_SRC)
        self.amap_buf = self.dev.create_buffer(size=256 * 4, usage=B.STORAGE | B.COPY_DST)
        self.out_buf = self.dev.create_buffer(size=max_pairs * 16, usage=B.STORAGE | B.COPY_DST | B.COPY_SRC)
        self.params_buf = self.dev.create_buffer(size=32, usage=B.UNIFORM | B.COPY_DST)
        self.dev.queue.write_buffer(self.amap_buf, 0, self.alphabet.tobytes())
        module = self.dev.create_shader_module(code=WGSL)
        storage = {"type": wgpu.BufferBindingType.storage}
        ro = {"type": wgpu.BufferBindingType.read_only_storage}
        uniform = {"type": wgpu.BufferBindingType.uniform}
        layout = self.dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": storage},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": ro},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": storage},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": uniform},
        ])
        pl = self.dev.create_pipeline_layout(bind_group_layouts=[layout])
        self.pipeline = self.dev.create_compute_pipeline(layout=pl, compute={"module": module, "entry_point": "run"})
        res = lambda b: {"buffer": b, "offset": 0, "size": b.size}
        self.bind = self.dev.create_bind_group(layout=layout, entries=[
            {"binding": 0, "resource": res(self.mem_buf)}, {"binding": 1, "resource": res(self.amap_buf)},
            {"binding": 2, "resource": res(self.out_buf)}, {"binding": 3, "resource": res(self.params_buf)}])

    def execute(self, pairs: np.ndarray, steps: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """pairs: (n, 128) uint8 → (memory after, (n, 4) uint32 [executed, entered_partner, max_pc, writes_into_partner])."""
        pairs = np.ascontiguousarray(pairs, dtype=np.uint8)
        n = pairs.shape[0]
        assert pairs.shape[1] == PAIR and n <= self.max_pairs
        steps = self.steps if steps is None else steps
        self.dev.queue.write_buffer(self.mem_buf, 0, pairs.tobytes())
        self.dev.queue.write_buffer(self.params_buf, 0, np.array([n, steps, int(self.ip_wrap), TAPE, int(self.literal), int(self.nohalt), 0, 0], dtype=np.uint32).tobytes())
        enc = self.dev.create_command_encoder()
        p = enc.begin_compute_pass()
        p.set_pipeline(self.pipeline)
        p.set_bind_group(0, self.bind)
        p.dispatch_workgroups((n + 63) // 64)
        p.end()
        self.dev.queue.submit([enc.finish()])
        mem = np.frombuffer(self.dev.queue.read_buffer(self.mem_buf, size=n * PAIR), dtype=np.uint8).reshape(n, PAIR).copy()
        out = np.frombuffer(self.dev.queue.read_buffer(self.out_buf, size=n * 16), dtype=np.uint32).reshape(n, 4).copy()
        return mem, out


def reference_execute(pair: np.ndarray, steps: int, ip_wrap: bool = False, alphabet: np.ndarray | None = None, literal: bool = False, nohalt: bool = False) -> tuple[np.ndarray, dict]:
    """CPU reference of the kernel, for tests."""
    amap = ascii_map(literal) if alphabet is None else alphabet
    t = pair.astype(np.int32).copy()
    T = len(t)
    half = T // 2
    pc, h0, h1 = 0, 0, 0
    entered = False
    maxpc = 0
    writesB = 0
    executed = 0
    for _ in range(steps):
        if pc < 0 or pc >= T:
            if ip_wrap:
                pc %= T
            else:
                break
        maxpc = max(maxpc, pc)
        entered |= pc >= half
        op = int(amap[t[pc]])
        stop = False
        if op == 1:
            h0 = (h0 - 1) % T
        elif op == 2:
            h0 = (h0 + 1) % T
        elif op == 3:
            h1 = (h1 - 1) % T
        elif op == 4:
            h1 = (h1 + 1) % T
        elif op == 5:
            t[h0] = (t[h0] + 1) & 255
            writesB += h0 >= half
        elif op == 6:
            t[h0] = (t[h0] - 1) & 255
            writesB += h0 >= half
        elif op == 7:
            t[h1] = t[h0]
            writesB += h1 >= half
        elif op == 8:
            t[h0] = t[h1]
            writesB += h0 >= half
        elif op == 9:
            if t[h0] == 0:
                depth, q, found = 1, pc + 1, False
                while q < T:
                    o = int(amap[t[q]])
                    if o == 9:
                        depth += 1
                    elif o == 10:
                        depth -= 1
                        if depth == 0:
                            found = True
                            break
                    q += 1
                if not found:
                    if not nohalt:
                        stop = True
                else:
                    pc = q
        elif op == 11:
            if literal:
                a_, b_ = int(t[(pc + 1) % T]), int(t[(pc + 2) % T])
                h1 = (h1 - 2) % T
                t[(h1 + 1) % T] = a_
                t[h1] = b_
                writesB += ((h1 + 1) % T) >= half
                writesB += h1 >= half
                pc += 2
        elif op == 10:
            if t[h0] != 0:
                depth, q, found = 1, pc - 1, False
                while q >= 0:
                    o = int(amap[t[q]])
                    if o == 10:
                        depth += 1
                    elif o == 9:
                        depth -= 1
                        if depth == 0:
                            found = True
                            break
                    q -= 1
                if not found:
                    if not nohalt:
                        stop = True
                else:
                    pc = q
        executed += 1
        if stop:
            break
        pc += 1
    return t.astype(np.uint8), {"executed": executed, "entered": int(entered), "max_pc": maxpc, "writesB": writesB}


def similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """Best match of b against a over cyclic shifts, forwards and reversed: (fraction equal, 'fwd'|'rev')."""
    best, how = 0.0, "fwd"
    for arr, tag in ((a, "fwd"), (a[::-1], "rev")):
        for s in range(len(a)):
            v = float((b == np.roll(arr, s)).mean())
            if v > best:
                best, how = v, tag
    return best, how


def assay(bff: BFF, tape: np.ndarray, n: int = 64, seed: int = 0, thresh: float = 0.75) -> dict:
    """Culture test for one 64-byte BFF tape: run it as A against n random partners; score = mean best similarity of the
    partner after to the tape (forwards or reversed); copies = fraction ≥ thresh; gen2 = the same for the offspring (the
    produced partners, run as A against fresh random partners); entered = fraction of encounters whose instruction
    pointer entered the partner (openness); self_damage = fraction of encounters in which A lost ≥ 25% of its bytes."""
    rng = np.random.default_rng(seed)
    partners = rng.integers(0, 256, size=(n, TAPE), dtype=np.uint8)
    mem, out = bff.execute(np.concatenate([np.repeat(tape[None], n, 0), partners], axis=1))
    A2, B2 = mem[:, :TAPE], mem[:, TAPE:]
    sims = np.array([similarity(tape, b)[0] for b in B2])
    copies = sims >= thresh
    kids = B2[copies] if copies.any() else B2[:0]
    gen2 = 0.0
    if len(kids):
        kids = kids[: min(len(kids), 32)]
        partners2 = rng.integers(0, 256, size=(len(kids), TAPE), dtype=np.uint8)
        mem2, _ = bff.execute(np.concatenate([kids, partners2], axis=1))
        gen2 = float(np.mean([similarity(k, g)[0] >= thresh for k, g in zip(kids, mem2[:, TAPE:])]))
    return {"score": float(sims.mean()), "copies": float(copies.mean()), "gen2": gen2 * float(copies.mean()), "gen2_cond": gen2,
            "entered": float(out[:, 1].mean()), "self_damage": float(((A2 != tape).mean(1) >= 0.25).mean()),
            "writesB": float(out[:, 3].mean()), "executed": float(out[:, 0].mean())}


def has_loop(tape: np.ndarray, alphabet: np.ndarray | None = None) -> bool:
    amap = ascii_map(True) if alphabet is None else alphabet
    ops = amap[tape]
    return bool((ops == 9).any() and (ops == 10).any())
