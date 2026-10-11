"""GPU census of periodic tiles on any vendored Zilion pair machine (the Proposition 3 census, ISA-agnostic).

    from algocell_exp.zcensus import TileCensus
    c = TileCensus("m6502_pair_maps_L16", k=4, nrand=16, budgets=(128, 512), seg=128)
    out = c.run_range(0, 1 << 20)        # {budget: {'counters': {...}, 'hits': records}}
    c.recheck(out[512]["hits"], 512)     # the recorded tiles re-run through one direct ZMachine.run

The census is the one of `census_period.py` (registered R3/P, the paper's Proposition 3 and Fig. 6b), extended by the
random-partner columns of `planning/ISA_RESEARCH_2026-10-10.md` §4.1:

- **Tile.** Every word T of period k (index t, bytes big-endian: T[0] is the most significant byte of t, so t = 0x01c5
  is the word `01 c5`, as `census_period.py` enumerates them) tiled to L bytes is tape A: A[j] = T[j mod k].
- **Encounters.** Variant 0 is the all-zero partner B; variants 1..nrand are random partners, B's word w (little-endian
  bytes) = mix(mix(mix(SEED ^ t_lo) ^ t_hi) ^ (v << 8 | w)) with mix = lowbias32 (`partner_bytes` mirrors it in numpy).
  Each encounter is one lane of the machine: memory A then B (the machine's segments A, B), the world's own initial
  state, `budget` steps.
- **Copy score.** match = max over the L cyclic shifts s of #{j : B_final[j] == A_initial[(j - s) mod L]} (the Z80
  census: "the partner equal to the tape at some cyclic shift" is match == L). `phase` is the exploratory census's
  score: max over the k phases s of #{j : B_final[j] == T[(j + s) mod k]}. The two coincide when k divides L.
- **Columns.** exact into the zero partner (match == L in variant 0); exact into random partners (match == L in at
  least one of them; the number of such partners is kept); ≥ 75% copy (match ≥ ceil(3L/4)) in ≥ 8 of 16 partners.
- **Records.** A tile is recorded (all its per-encounter scores, halt reasons and steps) when it meets any column,
  under either score; when it comes near one (zero-partner match ≥ ceil(3L/4), ≥ 2 random partners at ≥ 75%, or a
  random partner at L − 1 or better); or when it is in the hash-chosen sample.
- **Open / closed.** A lane entered the partner if its fetch map has a bit in B (positions L..P-1: instruction-stream
  bytes, opcode or operand); a tile is closed if no lane of its 1 + nrand encounters entered (the Z80 census: "closed
  if no partner byte is fetched in any of the 17 encounters").

Everything runs on the GPU: a generator kernel writes the lanes' memory from (t, variant), the machine runs (the
vendored WGSL, unchanged, bound to the harness's buffers through `ZMachine.bind_group`; lanes still running after each
segment of `seg` steps are compacted and resumed with Zilion's resumable runs, so one pass gives several budgets),
and a checker kernel scores
every tile, accumulates diagnostic counters (halt reasons, step histograms, partner entered, writes into B, B and A
changed, match histograms) and appends a record for every tile that meets any column, plus a hash-chosen sample of
tiles (1 in 2^sample_bits), which `recheck` re-runs through `ZMachine.run` with memory built in numpy and scores in
numpy, so the generator's placement of A and B, the partners and the checker are tested on every run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import wgpu

from .zmachine import ZMachine, load_machine

SEED = 0x6502C0DE           # partner stream seed (the same for every machine, period and L)
SAMPLE_SALT = 0x5A5A1234
STEP_BINS = ("0", "1", "2-3", "4-15", "16-63", "64-budget-1", "budget")
HIT_WORDS = 48
F_ZEXACT, F_ZPHASE, F_REXACT, F_RPHASE, F_R75, F_R75PHASE, F_SAMPLE, F_NEAR = (1 << i for i in range(8))
FLAG_NAMES = {F_ZEXACT: "zero_exact", F_ZPHASE: "zero_phase_exact", F_REXACT: "rand_exact_any", F_RPHASE: "rand_phase_exact_any",
              F_R75: "rand_75_ge8", F_R75PHASE: "rand_phase75_ge8", F_SAMPLE: "sample", F_NEAR: "near"}
COLUMN_FLAGS = F_ZEXACT | F_ZPHASE | F_REXACT | F_RPHASE | F_R75 | F_R75PHASE   # the census columns (not sample, not near)
M32 = np.uint32(0xFFFFFFFF)


def mix32(x):
    """lowbias32 (C. Wellons) on uint32 numpy values; the WGSL cen_mix."""
    x = np.asarray(x, dtype=np.uint32).copy()
    with np.errstate(over="ignore"):
        x ^= x >> np.uint32(16)
        x *= np.uint32(0x7FEB352D)
        x ^= x >> np.uint32(15)
        x *= np.uint32(0x846CA68B)
        x ^= x >> np.uint32(16)
    return x


def tile_bytes(t: np.ndarray, k: int) -> np.ndarray:
    """(n,) tile indices (uint64) → (n, k) words, big-endian (T[0] = most significant byte)."""
    t = np.asarray(t, dtype=np.uint64)
    return np.stack([((t >> np.uint64(8 * (k - 1 - i))) & np.uint64(0xFF)).astype(np.uint8) for i in range(k)], axis=1)


def tile_tapes(t: np.ndarray, k: int, L: int) -> np.ndarray:
    w = tile_bytes(t, k)
    return np.tile(w, (1, -(-L // k)))[:, :L]


def partner_bytes(t: np.ndarray, v: int, L: int, seed: int = SEED, panel: bool = False) -> np.ndarray:
    """The random partner of variant v ≥ 1 for each tile (mirror of the generator kernel), (n, L) uint8. With a panel
    every tile meets the same partners (those of tile 0); without (the default, the census's setting) each tile has its
    own."""
    t = np.asarray(t, dtype=np.uint64)
    if panel:
        t = np.zeros_like(t)
    lo = (t & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    hi = (t >> np.uint64(32)).astype(np.uint32)
    h0 = mix32(mix32(np.uint32(seed) ^ lo) ^ hi)
    wpc = -(-L // 4)
    words = np.stack([mix32(h0 ^ np.uint32((v << 8) | w)) for w in range(wpc)], axis=1)
    return np.ascontiguousarray(words).view(np.uint8).reshape(len(t), 4 * wpc)[:, :L]


def scores(B: np.ndarray, A0: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rotation match, phase match) per row: B final (n, L), A initial (n, L), tile words (n, k)."""
    n, L = B.shape
    k = T.shape[1]
    rot = np.max(np.stack([(B == np.roll(A0, s, axis=1)).sum(1) for s in range(L)], 1), 1)
    j = np.arange(L)
    phase = np.max(np.stack([(B == T[:, (j + s) % k]).sum(1) for s in range(k)], 1), 1)
    return rot, phase


@dataclass
class Evaluation:
    """CPU (ZMachine.run) evaluation of tiles × variants: arrays (n, V)."""
    rot: np.ndarray
    phase: np.ndarray
    entered: np.ndarray
    halt: np.ndarray
    trap: np.ndarray
    steps: np.ndarray
    writes_b: np.ndarray
    b_final: np.ndarray        # (n, V, L)
    a_final: np.ndarray        # (n, V, L)
    fetch_map: np.ndarray      # (n, V, P)
    start_map: np.ndarray      # (n, V, P)


def evaluate(machine: ZMachine, k: int, tiles: np.ndarray, nrand: int, budget: int, seed: int = SEED, partners: np.ndarray | None = None,
             panel: bool = False) -> Evaluation:
    """Run each tile against the zero partner and its nrand partners through ZMachine.run (memory built in numpy) and
    score it as the GPU checker does. `partners` (n, nrand, L) overrides the hash partners (e.g. a census's own rng)."""
    tiles = np.asarray(tiles, dtype=np.uint64)
    L = machine.segment("A")["bytes"]
    P = machine.ring_bytes
    n, V = len(tiles), 1 + nrand
    A0 = tile_tapes(tiles, k, L)
    T = tile_bytes(tiles, k)
    mem = np.zeros((n, V, 2 * L), np.uint8)
    mem[:, :, :L] = A0[:, None, :]
    for v in range(1, V):
        mem[:, v, L:] = partner_bytes(tiles, v, L, seed, panel) if partners is None else partners[:, v - 1]
    r = machine.run(mem.reshape(n * V, 2 * L), budget=budget)
    out = r.memory.reshape(n, V, 2 * L)
    rot, phase = scores(out[:, :, L:].reshape(n * V, L), np.repeat(A0, V, 0), np.repeat(T, V, 0))
    fm = r.fetch_map.reshape(n, V, P) if r.fetch_map is not None else np.zeros((n, V, P), bool)
    sm = r.start_map.reshape(n, V, P) if r.start_map is not None else np.zeros((n, V, P), bool)
    wb = r.writes("B") + (r.writes("padding") if "padding" in r.write_count_names else 0)
    return Evaluation(rot.reshape(n, V), phase.reshape(n, V), fm[:, :, L:].any(2), r.halt_reason.reshape(n, V), r.trap_cause.reshape(n, V),
                      r.steps.reshape(n, V), wb.reshape(n, V), out[:, :, L:], out[:, :, :L], fm, sm)


class TileCensus:
    """The census engine for one machine, period k and partner count, at one or more budgets in one pass.

    Lanes run in segments of `seg` steps. After each segment the lanes still running are compacted (gathered into
    dense buffers, their record and memory carried over) and resumed (Zilion's resumable runs: `resume` = 1, the record
    and memory of the previous segment), then scattered back; at each budget in `budgets` the checker scores every
    tile. A lane that halts costs nothing afterwards, so one pass gives the census at 128 and at 512 steps for about
    half the cost of a plain 512-step run. `recheck` compares every recorded tile with a single direct
    `ZMachine.run` at that budget, so the segmenting is tested on every run."""

    def __init__(self, machine: str | ZMachine, k: int, nrand: int = 16, budgets=(128,), seg: int = 32, seed: int = SEED,
                 tiles_per_dispatch: int | None = None, dispatches_per_submit: int = 2, hit_cap: int = 1 << 15,
                 sample_bits: int = 16, panel: bool = False, variant_major: bool = True, device: wgpu.GPUDevice | None = None) -> None:
        self.machine = machine if isinstance(machine, ZMachine) else load_machine(machine, device=device)
        m = self.machine
        self.dev = m.dev
        segs = [s["name"] for s in m.segments]
        if segs != ["A", "B"] or m.segment("A")["bytes"] != m.segment("B")["bytes"]:
            raise ValueError(f"{m.name}: not a pair of equal tapes A, B")
        if not m.budget_per_lane or m.m["budget"]["unit"] != "steps":
            raise ValueError(f"{m.name}: the census needs a per-lane step budget")
        if not m.m.get("resumable"):
            raise ValueError(f"{m.name}: the census resumes lanes; the machine is not resumable")
        if "fetchMap" not in m.fields:
            raise ValueError(f"{m.name}: the census needs the fetch map (open/closed)")
        if m.workgroup_size != 64 or m.lanes_per_invocation != 1:
            raise ValueError(f"{m.name}: the harness dispatches 64 lanes per workgroup, one lane per invocation "
                             f"(the machine has {m.workgroup_size} and {m.lanes_per_invocation})")
        if not (1 <= k <= 8):
            raise ValueError("period k must be 1..8")
        self.budgets = tuple(sorted(int(b) for b in budgets))
        self.seg = min(int(seg), self.budgets[-1])
        if any(b % self.seg for b in self.budgets):
            raise ValueError(f"every budget must be a multiple of the segment ({self.seg})")
        self.k, self.nrand, self.V, self.seed = k, nrand, 1 + nrand, int(seed) & 0xFFFFFFFF
        self.panel, self.variant_major = bool(panel), bool(variant_major)
        self.L = m.segment("A")["bytes"]
        self.P = m.ring_bytes
        if self.L > 64 or self.V > 21:
            raise ValueError("L ≤ 64 and at most 20 random partners (hit record layout)")
        self.thr = -(-3 * self.L // 4)
        self.n_tiles_total = 256 ** k
        cap = m.max_lanes_per_dispatch() // self.V
        tpd = tiles_per_dispatch or (1 << 17 if self.V > 1 else 1 << 21)
        while tpd > cap:
            tpd //= 2
        self.tpd = tpd
        self.dps = dispatches_per_submit
        self.hit_cap = hit_cap
        self.sample_mask = (1 << sample_bits) - 1 if sample_bits < 32 else 0xFFFFFFFF
        self._layout_counters()
        self.wgsl = self._wgsl()
        self.wgsl_sha256 = hashlib.sha256(self.wgsl.encode()).hexdigest()
        self._build()

    @property
    def budget(self) -> int:
        return self.budgets[-1]

    def describe(self) -> dict:
        return {"machine": self.machine.describe(), "k": self.k, "L": self.L, "P": self.P, "nrand": self.nrand, "budgets": list(self.budgets),
                "segment": self.seg, "seed": self.seed, "panel": self.panel, "variant_major": self.variant_major, "thr75": self.thr,
                "sample_mask": self.sample_mask, "harness_wgsl_sha256": self.wgsl_sha256}

    # ── counters ────────────────────────────────────────────────────────────────────────────────────────────────────
    def _layout_counters(self) -> None:
        names = ["tiles", "hits_total"]
        for cls in ("zero", "rand"):
            names += [f"{cls}_lanes", *(f"{cls}_halt{h}" for h in range(4)), *(f"{cls}_steps_{b}" for b in STEP_BINS),
                      f"{cls}_entered", f"{cls}_wrote_b", f"{cls}_b_changed", f"{cls}_a_changed", f"{cls}_trap_illegal"]
            names += [f"{cls}_match{m}" for m in range(self.L + 1)]
            names += [f"{cls}_phase{m}" for m in range(self.L + 1)]
        names += [f"n_exact{i}" for i in range(self.V)] + [f"n75_{i}" for i in range(self.V)]
        names += [f"n_phase_exact{i}" for i in range(self.V)] + [f"n_phase75_{i}" for i in range(self.V)]
        names += ["tiles_zero_exact", "tiles_zero_phase_exact", "tiles_rand_exact_any", "tiles_rand_phase_exact_any",
                  "tiles_rand75_ge8", "tiles_rand_phase75_ge8", "tiles_zero_exact_closed", "tiles_closed", "tiles_zero_ge_thr"]
        self.counter_names = names
        self.ci = {n: i for i, n in enumerate(names)}

    # ── WGSL ────────────────────────────────────────────────────────────────────────────────────────────────────────
    def _wgsl(self) -> str:
        m, L, P, V, k = self.machine, self.L, self.P, self.V, self.k
        a, b = m.segment("A"), m.segment("B")
        f = m.fields
        wc = f["writeCounts"]
        wb = wc["offset"] + list(wc["items"]).index("B")
        wpad = wc["offset"] + list(wc["items"]).index("padding") if "padding" in wc["items"] else None
        ci = self.ci
        C = lambda n: f"{ci[n]}u"  # noqa: E731
        return f"""// Tile census harness (algocell_exp/zcensus.py) for machine {m.name} (wgsl sha256 {m.wgsl_sha256}).
const L: u32 = {L}u;
const P: u32 = {P}u;
const K: u32 = {k}u;
const V: u32 = {V}u;
const THR: u32 = {self.thr}u;
const SEED: u32 = {self.seed}u;
const SAMPLE_SALT: u32 = {SAMPLE_SALT}u;
const SAMPLE_MASK: u32 = {self.sample_mask}u;
const MEM_WORDS: u32 = {m.storage_words}u;
const A_WORD: u32 = {a['storageWord']}u;
const B_WORD: u32 = {b['storageWord']}u;
const SEG_WORDS: u32 = {(L + 3) // 4}u;
const REC_WORDS: u32 = {m.record_words}u;
const R_STEPS: u32 = {f['steps']['offset']}u;
const R_HALT: u32 = {f['haltReason']['offset']}u;
const R_TRAP: u32 = {f['trapCause']['offset']}u;
const R_WB: u32 = {wb}u;
const R_FETCH: u32 = {f['fetchMap']['offset']}u;
const HIT_WORDS: u32 = {HIT_WORDS}u;
const HIT_CAP: u32 = {self.hit_cap}u;
const VARIANT_MAJOR: bool = {str(self.variant_major).lower()};
const PANEL: bool = {str(self.panel).lower()};

struct CParams {{ base_lo: u32, base_hi: u32, n_tiles: u32, budget: u32, seg: u32, n_lanes: u32, _p0: u32, _p1: u32 }};
@group(0) @binding(0) var<uniform> cp: CParams;
@group(0) @binding(1) var<storage, read_write> mem: array<u32>;
@group(0) @binding(2) var<storage, read_write> rec: array<u32>;
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> hits: array<u32>;
@group(0) @binding(5) var<storage, read_write> cmem: array<u32>;
@group(0) @binding(6) var<storage, read_write> crec: array<u32>;
@group(0) @binding(7) var<storage, read_write> cbudget: array<u32>;
@group(0) @binding(8) var<storage, read_write> cidx: array<u32>;
@group(0) @binding(9) var<storage, read_write> ctl: array<atomic<u32>>;   // [0] live count, [1..3] indirect dispatch args

fn cen_mix(x0: u32) -> u32 {{
    var x = x0;
    x = x ^ (x >> 16u); x = x * 0x7feb352du; x = x ^ (x >> 15u); x = x * 0x846ca68bu; x = x ^ (x >> 16u);
    return x;
}}
fn tile_id(tl: u32) -> vec2<u32> {{
    let lo = cp.base_lo + tl;
    return vec2<u32>(lo, cp.base_hi + select(0u, 1u, lo < cp.base_lo));
}}
fn tile_byte(t: vec2<u32>, i: u32) -> u32 {{   // T[i], big-endian over the k bytes of t
    let sh = 8u * (K - 1u - i);
    if (sh < 32u) {{ return (t.x >> sh) & 0xffu; }}
    return (t.y >> (sh - 32u)) & 0xffu;
}}
fn partner_h0(t: vec2<u32>) -> u32 {{   // a panel: the same partners for every tile (those of tile 0); else per tile
    if (PANEL) {{ return cen_mix(cen_mix(SEED)); }}
    return cen_mix(cen_mix(SEED ^ t.x) ^ t.y);
}}
fn lane_of(tl: u32, v: u32) -> u32 {{
    if (VARIANT_MAJOR) {{ return v * cp.n_tiles + tl; }}
    return tl * V + v;
}}

@compute @workgroup_size(64)
fn gen(@builtin(global_invocation_id) gid: vec3u) {{
    let lane = gid.x;
    if (lane >= cp.n_tiles * V) {{ return; }}
    var tl = lane / V;
    var v = lane % V;
    if (VARIANT_MAJOR) {{ tl = lane % cp.n_tiles; v = lane / cp.n_tiles; }}
    let t = tile_id(tl);
    let base = lane * MEM_WORDS;
    for (var w = 0u; w < MEM_WORDS; w++) {{ mem[base + w] = 0u; }}
    for (var w = 0u; w < SEG_WORDS; w++) {{
        var word = 0u;
        for (var q = 0u; q < 4u; q++) {{
            let j = 4u * w + q;
            if (j < L) {{ word = word | (tile_byte(t, j % K) << (8u * q)); }}
        }}
        mem[base + A_WORD + w] = word;
    }}
    if (v > 0u) {{
        let h0 = partner_h0(t);
        for (var w = 0u; w < SEG_WORDS; w++) {{
            var word = cen_mix(h0 ^ ((v << 8u) | w));
            let nb = min(4u, L - 4u * w);
            if (nb < 4u) {{ word = word & ((1u << (8u * nb)) - 1u); }}
            mem[base + B_WORD + w] = word;
        }}
    }}
}}

// ── segments: compact the running lanes, resume them, scatter them back ──
@compute @workgroup_size(1)
fn ctl_reset() {{ atomicStore(&ctl[0], 0u); }}

var<workgroup> wg_live: array<u32, 64>;
var<workgroup> wg_base: u32;
// The running lanes of each block of 64 keep their order (a local prefix sum) and take one contiguous range of the
// compact buffers (one atomic per block), so lanes that ran similar code stay together in a SIMD group.
@compute @workgroup_size(64)
fn compact(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {{
    let i = gid.x;
    var live = 0u;
    if (i < cp.n_lanes) {{ live = select(0u, 1u, rec[i * REC_WORDS + R_HALT] == 0u); }}   // halted lanes have nothing more to run
    wg_live[li] = live;
    workgroupBarrier();
    if (li == 0u) {{
        var total = 0u;
        for (var q = 0u; q < 64u; q++) {{ let x = wg_live[q]; wg_live[q] = total; total += x; }}
        wg_base = atomicAdd(&ctl[0], total);
    }}
    workgroupBarrier();
    if (live == 0u) {{ return; }}
    let j = wg_base + wg_live[li];
    cidx[j] = i;
    for (var w = 0u; w < MEM_WORDS; w++) {{ cmem[j * MEM_WORDS + w] = mem[i * MEM_WORDS + w]; }}
    for (var w = 0u; w < REC_WORDS; w++) {{ crec[j * REC_WORDS + w] = rec[i * REC_WORDS + w]; }}
}}

@compute @workgroup_size(1)
fn ctl_args() {{
    let c = atomicLoad(&ctl[0]);
    atomicStore(&ctl[1], (c + 63u) / 64u);
    atomicStore(&ctl[2], 1u);
    atomicStore(&ctl[3], 1u);
}}

@compute @workgroup_size(64)
fn cbudget_fill(@builtin(global_invocation_id) gid: vec3u) {{   // indirect: 64 * ceil(count / 64) invocations
    let j = gid.x;
    cbudget[j] = select(0u, cp.seg, j < atomicLoad(&ctl[0]));   // lanes past the count run no step
}}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3u) {{   // indirect
    let j = gid.x;
    if (j >= atomicLoad(&ctl[0])) {{ return; }}
    let i = cidx[j];
    for (var w = 0u; w < MEM_WORDS; w++) {{ mem[i * MEM_WORDS + w] = cmem[j * MEM_WORDS + w]; }}
    for (var w = 0u; w < REC_WORDS; w++) {{ rec[i * REC_WORDS + w] = crec[j * REC_WORDS + w]; }}
}}

// ── scoring ──
fn mem_byte(lane: u32, word0: u32, j: u32) -> u32 {{ return (mem[lane * MEM_WORDS + word0 + (j >> 2u)] >> (8u * (j & 3u))) & 0xffu; }}
fn count(i: u32) {{ atomicAdd(&counters[i], 1u); }}

@compute @workgroup_size(64)
fn check(@builtin(global_invocation_id) gid: vec3u) {{
    let tl = gid.x;
    if (tl >= cp.n_tiles) {{ return; }}
    let t = tile_id(tl);
    var a0: array<u32, {L}>;
    var tw: array<u32, {k}>;
    for (var i = 0u; i < K; i++) {{ tw[i] = tile_byte(t, i); }}
    for (var j = 0u; j < L; j++) {{ a0[j] = tw[j % K]; }}
    let h0 = partner_h0(t);
    var bb: array<u32, {L}>;
    var rots: array<u32, {V}>;
    var phs: array<u32, {V}>;
    var hs: array<u32, {V}>;
    var entered_mask = 0u;
    var n_exact = 0u; var n75 = 0u; var n_pexact = 0u; var n_p75 = 0u;
    for (var v = 0u; v < V; v++) {{
        let lane = lane_of(tl, v);
        let cls = select({C('rand_lanes')}, {C('zero_lanes')}, v == 0u);   // base index of the class's counters
        let o = lane * REC_WORDS;
        let steps = rec[o + R_STEPS];
        let halt = rec[o + R_HALT];
        let trap = rec[o + R_TRAP];
        let wrote_b = rec[o + R_WB]{f' + rec[o + {wpad}u]' if wpad is not None else ''};
        var entered = false;
        for (var p = L; p < P; p++) {{
            if (((rec[o + R_FETCH + (p >> 5u)] >> (p & 31u)) & 1u) != 0u) {{ entered = true; }}
        }}
        var b_changed = false;
        var a_changed = false;
        for (var j = 0u; j < L; j++) {{
            bb[j] = mem_byte(lane, B_WORD, j);
            if (mem_byte(lane, A_WORD, j) != a0[j]) {{ a_changed = true; }}
            var b_init = 0u;
            if (v > 0u) {{ b_init = (cen_mix(h0 ^ ((v << 8u) | (j >> 2u))) >> (8u * (j & 3u))) & 0xffu; }}
            if (bb[j] != b_init) {{ b_changed = true; }}
        }}
        var best = 0u;
        for (var s = 0u; s < L; s++) {{
            var c = 0u;
            for (var j = 0u; j < L; j++) {{
                var q = j + L - s;
                if (q >= L) {{ q = q - L; }}
                c += select(0u, 1u, bb[j] == a0[q]);
            }}
            best = max(best, c);
        }}
        var pbest = 0u;
        for (var s = 0u; s < K; s++) {{
            var c = 0u;
            for (var j = 0u; j < L; j++) {{ c += select(0u, 1u, bb[j] == tw[(j + s) % K]); }}
            pbest = max(pbest, c);
        }}
        rots[v] = best;
        phs[v] = pbest;
        hs[v] = (halt << 28u) | ((trap & 0xfu) << 24u) | min(steps, 0xffffffu);
        if (entered) {{ entered_mask = entered_mask | (1u << v); }}
        var bin = 6u;
        if (steps == 0u) {{ bin = 0u; }} else if (steps == 1u) {{ bin = 1u; }} else if (steps < 4u) {{ bin = 2u; }}
        else if (steps < 16u) {{ bin = 3u; }} else if (steps < 64u) {{ bin = 4u; }} else if (steps < cp.budget) {{ bin = 5u; }}
        count(cls);
        count(cls + 1u + min(halt, 3u));
        count(cls + 5u + bin);
        if (entered) {{ count(cls + 12u); }}
        if (wrote_b > 0u) {{ count(cls + 13u); }}
        if (b_changed) {{ count(cls + 14u); }}
        if (a_changed) {{ count(cls + 15u); }}
        if (halt == 3u && trap == 1u) {{ count(cls + 16u); }}
        count(cls + 17u + best);
        count(cls + 18u + L + pbest);
        if (v > 0u) {{
            if (best == L) {{ n_exact++; }}
            if (best >= THR) {{ n75++; }}
            if (pbest == L) {{ n_pexact++; }}
            if (pbest >= THR) {{ n_p75++; }}
        }}
    }}
    count({C('tiles')});
    count({C('n_exact0')} + n_exact);
    count({C('n75_0')} + n75);
    count({C('n_phase_exact0')} + n_pexact);
    count({C('n_phase75_0')} + n_p75);
    var flags = 0u;
    if (rots[0] == L) {{ flags |= {F_ZEXACT}u; count({C('tiles_zero_exact')}); if (entered_mask == 0u) {{ count({C('tiles_zero_exact_closed')}); }} }}
    if (phs[0] == L) {{ flags |= {F_ZPHASE}u; count({C('tiles_zero_phase_exact')}); }}
    if (n_exact > 0u) {{ flags |= {F_REXACT}u; count({C('tiles_rand_exact_any')}); }}
    if (n_pexact > 0u) {{ flags |= {F_RPHASE}u; count({C('tiles_rand_phase_exact_any')}); }}
    if (V > 1u && n75 >= 8u) {{ flags |= {F_R75}u; count({C('tiles_rand75_ge8')}); }}
    if (V > 1u && n_p75 >= 8u) {{ flags |= {F_R75PHASE}u; count({C('tiles_rand_phase75_ge8')}); }}
    if (entered_mask == 0u) {{ count({C('tiles_closed')}); }}
    if (rots[0] >= THR) {{ count({C('tiles_zero_ge_thr')}); }}
    if ((cen_mix(t.x ^ SAMPLE_SALT ^ cen_mix(t.y)) & SAMPLE_MASK) == 0u) {{ flags |= {F_SAMPLE}u; }}
    var best_rand = 0u;
    for (var v = 1u; v < V; v++) {{ best_rand = max(best_rand, rots[v]); }}
    if (rots[0] >= THR || n75 >= 2u || best_rand + 1u >= L) {{ flags |= {F_NEAR}u; }}   // near misses, recorded for the report
    if (flags == 0u) {{ return; }}
    let idx = atomicAdd(&counters[{C('hits_total')}], 1u);
    if (idx >= HIT_CAP) {{ return; }}
    let h = idx * HIT_WORDS;
    hits[h + 0u] = t.x;
    hits[h + 1u] = t.y;
    hits[h + 2u] = flags;
    hits[h + 3u] = entered_mask;
    hits[h + 4u] = n_exact | (n75 << 8u) | (n_pexact << 16u) | (n_p75 << 24u);
    for (var v = 0u; v < V; v++) {{
        hits[h + 5u + v] = rots[v] | (phs[v] << 8u);
        hits[h + 26u + v] = hs[v];
    }}
}}
"""

    # ── buffers and pipelines ───────────────────────────────────────────────────────────────────────────────────────
    def _build(self) -> None:
        dev, m = self.dev, self.machine
        B = wgpu.BufferUsage
        N = self.tpd * self.V
        self.n_lanes = N
        self.nseg = self.budget // self.seg
        mk = lambda size, extra=0: dev.create_buffer(size=max(size, 16), usage=B.STORAGE | B.COPY_SRC | B.COPY_DST | extra)  # noqa: E731
        self.mem_buf = mk(N * m.storage_words * 4)
        self.rec_buf = mk(N * m.record_words * 4)
        self.budget_buf = mk(N * 4)
        dev.queue.write_buffer(self.budget_buf, 0, np.full(N, self.seg, np.uint32).tobytes())
        self.compact = self.nseg > 1
        if self.compact:
            self.cmem_buf = mk(N * m.storage_words * 4)
            self.crec_buf = mk(N * m.record_words * 4)
            self.cbudget_buf = mk((N + 64) * 4)
            self.cidx_buf = mk(N * 4)
        else:   # unused bindings
            self.cmem_buf = self.crec_buf = self.cbudget_buf = self.cidx_buf = mk(16)
        self.ctl_buf = mk(16)
        self.args_buf = dev.create_buffer(size=16, usage=B.INDIRECT | B.COPY_DST)   # copied from ctl[1..3]: never bound as storage
        nc = len(self.counter_names)
        self.counter_bufs = {b: mk(nc * 4) for b in self.budgets}
        self.hit_bufs = {b: mk(self.hit_cap * HIT_WORDS * 4) for b in self.budgets}
        module = dev.create_shader_module(code=self.wgsl)
        st = {"type": wgpu.BufferBindingType.storage}
        self.layout = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
            *({"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": st} for i in range(1, 10)),
        ])
        pl = dev.create_pipeline_layout(bind_group_layouts=[self.layout])
        self.pipe = {ep: dev.create_compute_pipeline(layout=pl, compute={"module": module, "entry_point": ep})
                     for ep in ("gen", "check", "ctl_reset", "compact", "ctl_args", "cbudget_fill", "scatter")}
        whole = lambda buf: {"buffer": buf, "offset": 0, "size": buf.size}  # noqa: E731
        self.cp_bufs = [{b: dev.create_buffer(size=32, usage=B.UNIFORM | B.COPY_DST) for b in self.budgets} for _ in range(self.dps)]
        self.bgs = [{b: dev.create_bind_group(layout=self.layout, entries=[
            {"binding": 0, "resource": whole(self.cp_bufs[i][b])},
            {"binding": 1, "resource": whole(self.mem_buf)},
            {"binding": 2, "resource": whole(self.rec_buf)},
            {"binding": 3, "resource": whole(self.counter_bufs[b])},
            {"binding": 4, "resource": whole(self.hit_bufs[b])},
            {"binding": 5, "resource": whole(self.cmem_buf)},
            {"binding": 6, "resource": whole(self.crec_buf)},
            {"binding": 7, "resource": whole(self.cbudget_buf)},
            {"binding": 8, "resource": whole(self.cidx_buf)},
            {"binding": 9, "resource": whole(self.ctl_buf)},
        ]) for b in self.budgets} for i in range(self.dps)]
        # machine params per (slot, segment): segment 0 begins the lanes, the others resume them
        self.mp_bufs = [[dev.create_buffer(size=m.params_bytes, usage=B.UNIFORM | B.COPY_DST) for _ in range(self.nseg)] for _ in range(self.dps)]
        self.mbg_first = [m.bind_group(self.mp_bufs[i][0], self.mem_buf, self.rec_buf, self.budget_buf) for i in range(self.dps)]
        self.mbg_resume = [[m.bind_group(self.mp_bufs[i][s], self.cmem_buf, self.crec_buf, self.cbudget_buf) for s in range(self.nseg)]
                           for i in range(self.dps)] if self.compact else None

    def _pass(self, enc, pipeline, bg, n: int | None = None, indirect: bool = False) -> None:
        p = enc.begin_compute_pass()
        p.set_pipeline(pipeline)
        p.set_bind_group(0, bg)
        if indirect:
            p.dispatch_workgroups_indirect(self.args_buf, 0)
        else:
            p.dispatch_workgroups(max(1, -(-n // 64)) if n is not None else 1)
        p.end()

    def _encode(self, enc, i: int, t: int, n: int) -> None:
        q, m, N = self.dev.queue, self.machine, n * self.V
        for b in self.budgets:
            q.write_buffer(self.cp_bufs[i][b], 0, np.array([t & 0xFFFFFFFF, t >> 32, n, b, self.seg, N, 0, 0], np.uint32).tobytes())
        q.write_buffer(self.mp_bufs[i][0], 0, m.params(N, 0, 0, 0).tobytes())
        for s in range(1, self.nseg):
            q.write_buffer(self.mp_bufs[i][s], 0, m.params(N, 0, 0, 0, step_base=s * self.seg, resume=True).tobytes())
        bg0 = self.bgs[i][self.budgets[0]]
        self._pass(enc, self.pipe["gen"], bg0, N)
        self._pass(enc, m.pipeline, self.mbg_first[i], N)
        done = self.seg
        for s in range(1, self.nseg + 1):
            if done in self.budgets:
                self._pass(enc, self.pipe["check"], self.bgs[i][done], n)
            if s == self.nseg:
                break
            self._pass(enc, self.pipe["ctl_reset"], bg0)
            self._pass(enc, self.pipe["compact"], bg0, N)
            self._pass(enc, self.pipe["ctl_args"], bg0)
            enc.copy_buffer_to_buffer(self.ctl_buf, 4, self.args_buf, 0, 12)
            self._pass(enc, self.pipe["cbudget_fill"], bg0, indirect=True)
            self._pass(enc, m.pipeline, self.mbg_resume[i][s], indirect=True)
            self._pass(enc, self.pipe["scatter"], bg0, indirect=True)
            done += self.seg

    # ── running ─────────────────────────────────────────────────────────────────────────────────────────────────────
    def run_range(self, t0: int, t1: int) -> dict:
        """Census of tiles t0 ≤ t < t1. Returns {budget: {'counters': {name: int}, 'hits': (n, HIT_WORDS) uint32}}."""
        if not (0 <= t0 <= t1 <= self.n_tiles_total):
            raise ValueError(f"tile range [{t0}, {t1}) outside 0..{self.n_tiles_total}")
        dev, q = self.dev, self.dev.queue
        nc = len(self.counter_names)
        totals = {b: np.zeros(nc, np.int64) for b in self.budgets}
        hits = {b: [] for b in self.budgets}
        t = t0
        while t < t1:
            for b in self.budgets:
                q.write_buffer(self.counter_bufs[b], 0, np.zeros(nc, np.uint32).tobytes())
            enc = dev.create_command_encoder()
            planned = 0
            for i in range(self.dps):
                if t >= t1:
                    break
                n = min(self.tpd, t1 - t)
                self._encode(enc, i, t, n)
                t += n
                planned += n
            q.submit([enc.finish()])
            for b in self.budgets:
                c = np.frombuffer(q.read_buffer(self.counter_bufs[b]), np.uint32).astype(np.int64)
                if c[self.ci["tiles"]] != planned:
                    raise RuntimeError(f"budget {b}: checker saw {c[self.ci['tiles']]} tiles, {planned} were dispatched")
                if c[self.ci["zero_lanes"]] + c[self.ci["rand_lanes"]] != planned * self.V:
                    raise RuntimeError(f"budget {b}: checker saw {c[self.ci['zero_lanes']] + c[self.ci['rand_lanes']]} lanes, expected {planned * self.V}")
                nh = int(c[self.ci["hits_total"]])
                if nh > self.hit_cap:
                    raise RuntimeError(f"{nh} hit records in one submit exceed the capacity {self.hit_cap}; lower dispatches_per_submit")
                if nh:
                    hits[b].append(np.frombuffer(q.read_buffer(self.hit_bufs[b], 0, nh * HIT_WORDS * 4), np.uint32).reshape(nh, HIT_WORDS).copy())
                totals[b] += c
        return {b: {"counters": {nm: int(totals[b][i]) for i, nm in enumerate(self.counter_names)},
                    "hits": np.concatenate(hits[b]) if hits[b] else np.zeros((0, HIT_WORDS), np.uint32)} for b in self.budgets}

    # ── decoding hit records ───────────────────────────────────────────────────────────────────────────────────────
    def decode_hits(self, H: np.ndarray) -> dict:
        H = np.asarray(H, np.uint32).reshape(-1, HIT_WORDS)
        V = self.V
        t = H[:, 0].astype(np.uint64) | (H[:, 1].astype(np.uint64) << np.uint64(32))
        order = np.argsort(t, kind="stable")
        H, t = H[order], t[order]
        rp = H[:, 5:5 + V]
        hs = H[:, 26:26 + V]
        return {"t": t, "flags": H[:, 2], "entered": ((H[:, 3:4] >> np.arange(V, dtype=np.uint32)) & 1).astype(bool),
                "n_exact": H[:, 4] & 0xFF, "n75": (H[:, 4] >> 8) & 0xFF, "n_phase_exact": (H[:, 4] >> 16) & 0xFF, "n_phase75": H[:, 4] >> 24,
                "rot": rp & 0xFF, "phase": (rp >> 8) & 0xFF, "halt": hs >> 28, "trap": (hs >> 24) & 0xF, "steps": hs & 0xFFFFFF}

    def recheck(self, H: np.ndarray, budget: int, max_tiles: int | None = None) -> dict:
        """Re-run recorded tiles through one direct ZMachine.run at `budget` (memory and partners built in numpy) and
        compare every recorded per-variant field with the GPU checker's (which ran in compacted, resumed segments).
        Returns the numbers of tiles and lanes checked and of tiles with a mismatch, per field."""
        d = self.decode_hits(H)
        if max_tiles is not None and len(d["t"]) > max_tiles:
            sel = np.linspace(0, len(d["t"]) - 1, max_tiles).astype(int)
            d = {k: v[sel] for k, v in d.items()}
        n = len(d["t"])
        if n == 0:
            return {"tiles": 0, "lanes": 0}
        e = evaluate(self.machine, self.k, d["t"], self.nrand, budget, self.seed, panel=self.panel)
        mism = {
            "rot": int((e.rot != d["rot"]).any(1).sum()),
            "phase": int((e.phase != d["phase"]).any(1).sum()),
            "entered": int((e.entered != d["entered"]).any(1).sum()),
            "halt": int((e.halt != d["halt"]).any(1).sum()),
            "trap": int(((e.trap & 0xF) != d["trap"]).any(1).sum()),
            "steps": int((np.minimum(e.steps, 0xFFFFFF) != d["steps"]).any(1).sum()),
            "n_exact": int(((e.rot[:, 1:] == self.L).sum(1) != d["n_exact"]).sum()),
            "n75": int(((e.rot[:, 1:] >= self.thr).sum(1) != d["n75"]).sum()),
        }
        return {"tiles": n, "lanes": n * self.V, **{f"mismatch_{k}": v for k, v in mism.items()}}


def primitive_period(words: np.ndarray) -> np.ndarray:
    """(n, k) words → the smallest q dividing k with words periodic of period q."""
    n, k = words.shape
    out = np.full(n, k, np.int64)
    for q in sorted((q for q in range(1, k) if k % q == 0), reverse=True):
        ok = (words == np.tile(words[:, :q], (1, k // q))).all(1)
        out[ok] = q
    return out


def n_primitive(k: int) -> int:
    """Number of words of primitive period exactly k (Möbius sum)."""
    def mu(n):
        r, p, m = 1, 2, n
        while p * p <= m:
            if m % p == 0:
                m //= p
                if m % p == 0:
                    return 0
                r = -r
            p += 1
        return -r if m > 1 else r
    return sum(mu(k // d) * 256 ** d for d in range(1, k + 1) if k % d == 0)


__all__ = ["TileCensus", "evaluate", "Evaluation", "tile_bytes", "tile_tapes", "partner_bytes", "scores", "mix32", "primitive_period", "n_primitive", "SEED", "FLAG_NAMES", "COLUMN_FLAGS"]
