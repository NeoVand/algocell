"""Validation of the Zilion bridge against Algocell's own shaders (used by experiments/zsoup_validate.py and the tests).

1. `pair_equivalence`: lattice-neighbour pairs sampled from archived final soups run through Algocell's single-pair
   executor (shader/z80_test[_lethal]_L<L>.wgsl, as assay.py compiles it) and through a vendored Zilion machine, with
   the condition's own step budget and suppression masks; final memory, write counts and registers compared.
2. `Lockstep`: the original Soup runs; at every step, after prepare_batch, pair_data and write_counts are copied, the
   original z80_execute_batch runs on the soup's buffers and the Zilion machine (through ZSoup's own bridge code) on the
   copies, and a comparator kernel counts, per step, the pairs whose pair_data or write counts differ, with
   non-vacuity counters. Selected steps are re-checked on the CPU from the raw buffers, and the CPU counts must equal
   the comparator's. The soup's trajectory is the original pipeline's.
3. `nondeterminism`: two original Soups with the same seed, stepped in lockstep; the first step where they differ.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import brotli
import numpy as np
import wgpu

from . import assay as A
from .isa import masks as make_masks
from .isa import resolve
from .metrics import quasispecies_share, shift_occupancy, species_stats
from .soup import MAX_PAIRS, Soup, get_device
from .zmachine import ZMachine, load_machine
from .zsoup import ZBridge

RUNS = Path(os.environ.get("ALGOCELL_RUNS", "/Users/neo/repos/algocell/experiments/runs"))   # archived runs (read-only)
STAGES = ("stageA", "stageB", "stageC", "stageD", "stageE", "stageF", "stageG", "stageH", "stageI")
LOD_DIRS = ("lod", "lod_local", "lod_modal", "lod_smoke", "lod_v2", "lod_v2_modal", "lod_v3_modal", "lod_v6", "lod_v6_modal", "lod_n6_modal")
W, H = 160, 125
REGS = ("a", "f", "b", "c", "d", "e", "h", "l", "sp", "pc")   # Algocell executor register words 0-9 (10, 11: writes A, B)


# ── archived soups ──────────────────────────────────────────────────────────────────────────────────────────────────
def archived_soups(runs: Path = RUNS) -> list[dict]:
    """Every archived final soup of Stages A–I and the line-of-descent runs, with its condition."""
    out = []
    for st in STAGES:
        for sp in sorted((runs / st).glob("*.summary.json")):
            stem = sp.name[: -len(".summary.json")]
            soup = sp.with_name(f"{stem}.soup_final.u8.br")
            if not soup.exists():
                continue
            s = json.loads(sp.read_text())
            L = int(s["tape_length"])
            out.append({
                "stage": st, "path": soup, "L": L, "P": int(s.get("mem_length") or 2 * L), "zero_halts": bool(s.get("zero_halts") or False),
                "grid": s.get("grid", "square"), "z80_steps": int(s["z80_steps"]), "suppress": list(s.get("suppress") or []),
                "width": int(s.get("width", W)), "height": int(s.get("height", H)),
            })
    for d in LOD_DIRS:
        for f in sorted((runs / d).glob("**/final.npy")):
            meta = f.with_name("lod.json")
            L = int(json.loads(meta.read_text())["result"]["L"]) if meta.exists() else 16
            out.append({"stage": "lod", "path": f, "L": L, "P": 2 * L, "zero_halts": False, "grid": "square", "z80_steps": 128,
                        "suppress": [], "width": W, "height": H})
    return out


def load_soup(rec: dict) -> np.ndarray:
    p = Path(rec["path"])
    if p.suffix == ".npy":
        a = np.load(p)
    else:
        a = np.frombuffer(brotli.decompress(p.read_bytes()), dtype=np.uint8)
    return np.ascontiguousarray(a, dtype=np.uint8).reshape(rec["width"] * rec["height"], rec["L"])


def neighbour_pairs(soup: np.ndarray, n: int, rng: np.random.Generator, w: int = W, h: int = H) -> np.ndarray:
    """n lattice-neighbour pairs (A = cell i, B = one of its 4 neighbours, reflected at the edges, as prepare_batch
    draws them), as (n, 2L) uint8."""
    x = rng.integers(0, w, size=n)
    y = rng.integers(0, h, size=n)
    d = rng.integers(0, 4, size=n)
    nx, ny = x.copy(), y.copy()
    nx = np.where(d == 0, np.where(x + 1 < w, x + 1, x - 1), nx)
    ny = np.where(d == 1, np.where(y + 1 < h, y + 1, y - 1), ny)
    nx = np.where(d == 2, np.where(x > 0, x - 1, x + 1), nx)
    ny = np.where(d == 3, np.where(y > 0, y - 1, y + 1), ny)
    i, j = y * w + x, ny * w + nx
    return np.concatenate([soup[i], soup[j]], axis=1)


def path_seed(p: Path) -> int:
    return int.from_bytes(hashlib.sha256(str(Path(p).relative_to(RUNS) if Path(p).is_relative_to(RUNS) else p).encode()).digest()[:4], "little")


# ── Algocell's executor ─────────────────────────────────────────────────────────────────────────────────────────────
def algocell_execute(pairs: np.ndarray, L: int, steps: int, masks: np.ndarray, zero_halts: bool, traced: bool = False,
                     chunk: int = 1 << 18) -> tuple[np.ndarray, np.ndarray]:
    """Algocell's single-pair executor (assay._pipeline: shader/z80_test[_lethal]_L<L>.wgsl), or with traced=True the
    op-traced executor (optrace: + fetch and start bitmaps). Returns final memory (n, 2L) and the register words
    (n, 12 or 28): a f b c d e h l sp pc writes_a writes_b [exec_mask x8, op_mask x8]."""
    from . import optrace

    dev = get_device()
    pipe, layout = optrace._pipeline(L, zero_halts) if traced else A._pipeline(L, None, zero_halts)
    rw = 28 if traced else 12
    n = pairs.shape[0]
    wpc = (L + 3) // 4
    mem = np.empty((n, 2 * L), dtype=np.uint8)
    regs = np.empty((n, rw), dtype=np.uint32)
    B = wgpu.BufferUsage
    for s0 in range(0, n, chunk):
        c = min(chunk, n - s0)
        words = np.zeros((c, 2 * wpc * 4), dtype=np.uint8)
        words[:, :L] = pairs[s0: s0 + c, :L]
        words[:, 4 * wpc: 4 * wpc + L] = pairs[s0: s0 + c, L:]
        params = np.zeros(32, dtype=np.uint32)
        params[2], params[3], params[4], params[6] = L, 2 * L, c, steps
        params[8:] = masks
        io = dev.create_buffer(size=words.nbytes, usage=B.STORAGE | B.COPY_SRC | B.COPY_DST)
        dev.queue.write_buffer(io, 0, np.ascontiguousarray(words).tobytes())
        rb = dev.create_buffer(size=c * rw * 4, usage=B.STORAGE | B.COPY_SRC)
        pb = dev.create_buffer(size=128, usage=B.UNIFORM | B.COPY_DST)
        dev.queue.write_buffer(pb, 0, params.tobytes())
        bg = dev.create_bind_group(layout=layout, entries=[
            {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate((pb, io, rb))])
        enc = dev.create_command_encoder()
        p = enc.begin_compute_pass()
        p.set_pipeline(pipe)
        p.set_bind_group(0, bg)
        p.dispatch_workgroups(-(-c // 64))
        p.end()
        dev.queue.submit([enc.finish()])
        out = np.frombuffer(dev.queue.read_buffer(io), dtype=np.uint8).reshape(c, 2 * wpc * 4)
        mem[s0: s0 + c, :L] = out[:, :L]
        mem[s0: s0 + c, L:] = out[:, 4 * wpc: 4 * wpc + L]
        regs[s0: s0 + c] = np.frombuffer(dev.queue.read_buffer(rb), dtype=np.uint32).reshape(c, rw)
        for b in (io, rb, pb):
            b.destroy()
    return mem, regs


def compare_pairs(pairs: np.ndarray, L: int, steps: int, suppress, zero_halts: bool, machine: ZMachine, traced: bool = False,
                  zil_budget_delta: int = 0) -> dict:
    """Run `pairs` through Algocell's executor and through `machine` (budget = steps + zil_budget_delta: a negative
    control when non-zero); count the pairs that differ in each output."""
    masks = make_masks(resolve(list(suppress)))
    a_mem, a_regs = algocell_execute(pairs, L, steps, masks, zero_halts, traced)
    r = machine.run(pairs, budget=steps + zil_budget_delta, ablation=masks)
    assert r.memory is not a_mem and r.memory.shape == a_mem.shape
    wb = r.writes("B") + (r.writes("padding") if "padding" in r.write_count_names else 0)
    out = {
        "pairs": int(pairs.shape[0]),
        "distinct_pairs": int(np.unique(pairs, axis=0).shape[0]),
        "a_equals_b": int((pairs[:, :L] == pairs[:, L:]).all(axis=1).sum()),
        "mem_changed": int((a_mem != pairs).any(axis=1).sum()),            # non-vacuity: execution changed the pair
        "b_written": int((a_regs[:, 11] > 0).sum()),
        "writes_total": int(a_regs[:, 10].astype(np.int64).sum() + a_regs[:, 11].astype(np.int64).sum()),
        "zil_steps_mean": float(r.steps.mean()) if len(r.steps) else 0.0,
        "zil_halted": int((r.halt_reason != 0).sum()),
        "mem_mismatch": int((a_mem != r.memory).any(axis=1).sum()),
        "wc_mismatch": int(((a_regs[:, 10] != r.writes("A")) | (a_regs[:, 11] != wb)).sum()),
        "reg_mismatch": int(np.any(np.stack([a_regs[:, k] != r.reg(n) for k, n in enumerate(REGS)], axis=1), axis=1).sum()),
    }
    if traced:
        unpack = lambda w: np.unpackbits(np.ascontiguousarray(w).view(np.uint8), axis=1, bitorder="little")[:, : 2 * L].astype(bool)  # noqa: E731
        out["fetch_map_mismatch"] = int((unpack(a_regs[:, 12:20]) != r.fetch_map).any(axis=1).sum())
        out["start_map_mismatch"] = int((unpack(a_regs[:, 20:28]) != r.start_map).any(axis=1).sum())
        out["fetched_positions_mean"] = float(r.fetch_map.sum(axis=1).mean())
    return out


# ── step-locked differential test ───────────────────────────────────────────────────────────────────────────────────
STATS = (
    "mem_mismatch",        # pair slots (< pair_count) whose pair_data differs: original vs Zilion
    "wc_mismatch",         # pair slots whose write counts differ
    "active",              # pairs that won the collision claim
    "active_changed",      # active pairs whose memory execution changed (pair_data vs its copy before execution)
    "inactive_changed",    # inactive pair slots whose memory the ORIGINAL executor changed (must be 0; the Zilion side's
                           # inactive slots are covered by mem_mismatch, which compares every slot, and zil_inactive_steps)
    "writes",              # total writes (A + B) counted by the original executor
    "b_written",           # pairs with at least one write into B
    "zil_steps",           # instructions the Zilion lanes executed (record field steps), summed
    "zil_inactive_steps",  # Zilion lanes of inactive pairs that executed a step (must be 0)
)
K = len(STATS)

_CMP_WGSL = """
struct CmpParams { pair_count: u32, words_per_pair: u32, slot: u32, rec_words: u32, rec_steps: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<storage, read> pd_orig: array<u32>;
@group(0) @binding(1) var<storage, read> pd_zil: array<u32>;
@group(0) @binding(2) var<storage, read> pd_pre: array<u32>;
@group(0) @binding(3) var<storage, read> wc_orig: array<u32>;
@group(0) @binding(4) var<storage, read> wc_zil: array<u32>;
@group(0) @binding(5) var<storage, read> act_flags: array<u32>;
@group(0) @binding(6) var<storage, read> zrec: array<u32>;
@group(0) @binding(7) var<storage, read_write> stats: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> cp: CmpParams;

@compute @workgroup_size(64)
fn compare(@builtin(global_invocation_id) id: vec3u) {
    let p = id.x;
    if (p >= cp.pair_count) { return; }
    var differs = false;
    var changed = false;
    for (var w = 0u; w < cp.words_per_pair; w++) {
        let k = p * cp.words_per_pair + w;
        if (pd_orig[k] != pd_zil[k]) { differs = true; }
        if (pd_orig[k] != pd_pre[k]) { changed = true; }
    }
    let base = cp.slot * __K__u;
    let act = act_flags[p] != 0u;
    if (differs) { atomicAdd(&stats[base + 0u], 1u); }
    if (wc_orig[2u * p] != wc_zil[2u * p] || wc_orig[2u * p + 1u] != wc_zil[2u * p + 1u]) { atomicAdd(&stats[base + 1u], 1u); }
    if (act) { atomicAdd(&stats[base + 2u], 1u); }
    if (act && changed) { atomicAdd(&stats[base + 3u], 1u); }
    if (!act && changed) { atomicAdd(&stats[base + 4u], 1u); }
    atomicAdd(&stats[base + 5u], wc_orig[2u * p] + wc_orig[2u * p + 1u]);
    if (wc_orig[2u * p + 1u] != 0u) { atomicAdd(&stats[base + 6u], 1u); }
    let st = zrec[p * cp.rec_words + cp.rec_steps];
    if (act) { atomicAdd(&stats[base + 7u], st); }
    if (!act && st != 0u) { atomicAdd(&stats[base + 8u], 1u); }
}
""".replace("__K__", str(K))


class Lockstep:
    """The original Soup with a Zilion machine executing a copy of every step's pairs beside it (see module doc)."""

    def __init__(self, L: int = 16, zero_halts: bool = False, seed: int = 1, machine: str | None = None, suppress=None,
                 steps_capacity: int = 20000, zil_budget_delta: int = 0, z80_steps: int = 128) -> None:
        self.soup = s = Soup(tape_length=L, seed=seed, zero_halts=zero_halts, suppress=suppress, z80_steps=z80_steps)
        self.dev = dev = s.device
        name = machine or f"z80c02_{'lethal' if zero_halts else 'pair'}_L{L}"
        self.machine = load_machine(name, device=dev)
        self.bridge = br = ZBridge(self.machine, device=dev)
        assert (br.L, br.P) == (s.tape_length, s.mem_length), (br.L, br.P, s.tape_length, s.mem_length)
        self.L, self.seed, self.zero_halts, self.name = L, seed, zero_halts, name
        self.delta = zil_budget_delta
        B = wgpu.BufferUsage
        self.wpp = 2 * s.words_per_cell
        self.pd_bytes = s.pair_count * self.wpp * 4
        self.wc_bytes = s.pair_count * 2 * 4
        mk = lambda size: dev.create_buffer(size=size, usage=B.STORAGE | B.COPY_SRC | B.COPY_DST)  # noqa: E731
        self.pd_zil = mk(MAX_PAIRS * self.wpp * 4)
        self.pd_pre = mk(MAX_PAIRS * self.wpp * 4)
        self.wc_zil = mk(MAX_PAIRS * 2 * 4)
        self.capacity = steps_capacity
        self.stats_buf = mk(steps_capacity * K * 4)
        self.zparams = [dev.create_buffer(size=self.machine.params_bytes, usage=B.UNIFORM | B.COPY_DST) for _ in range(s.ring)]
        # the bridge reads params.z80_steps: a separate params ring lets the negative control give the machine a different budget
        self.bparams = [dev.create_buffer(size=128, usage=B.UNIFORM | B.COPY_DST) for _ in range(s.ring)] if zil_budget_delta else s.params_bufs
        self.cparams = [dev.create_buffer(size=32, usage=B.UNIFORM | B.COPY_DST) for _ in range(s.ring)]
        bufs = lambda k: {0: s.soup_buf, 1: s.pairs_buf, 2: self.pd_zil, 3: self.wc_zil, 4: s.rng_states_buf, 5: self.bparams[k],  # noqa: E731
                          6: s.pair_active_buf, 7: s.byte_counts_buf, 8: s.collision_buf, 9: s.hash_buf}
        self.bridge_bgs = [br.soup_bind_group(bufs(k)) for k in range(s.ring)]
        self.machine_bgs = [br.machine_bind_group(self.zparams[k], self.pd_zil) for k in range(s.ring)]
        cmp_mod = dev.create_shader_module(code=_CMP_WGSL)
        self.cmp_pipe = dev.create_compute_pipeline(layout="auto", compute={"module": cmp_mod, "entry_point": "compare"})
        lay = self.cmp_pipe.get_bind_group_layout(0)
        cmp_bufs = [s.pair_data_buf, self.pd_zil, self.pd_pre, s.write_counts_buf, self.wc_zil, s.pair_active_buf, br.record_buf, self.stats_buf]
        assert len({id(b) for b in cmp_bufs}) == len(cmp_bufs), "the comparator must read distinct buffers"
        self.cmp_bgs = [dev.create_bind_group(layout=lay, entries=[
            *({"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(cmp_bufs)),
            {"binding": 8, "resource": {"buffer": self.cparams[k], "offset": 0, "size": 32}}]) for k in range(s.ring)]
        self.rec_steps = self.machine.fields["steps"]["offset"]
        self.step_index = 0
        self.cpu_checks: list[dict] = []
        self.trajectory: list[dict] = []

    def _encode_step(self, enc, k: int) -> None:
        s, br = self.soup, self.bridge
        pairs_wg = -(-s.pair_count // 64)
        disp = lambda ep, wg: _soup_dispatch(s, enc, ep, k, wg)  # noqa: E731
        disp("clear_collision", -(-s.cell_count // 256))
        disp("prepare_batch", pairs_wg)
        enc.copy_buffer_to_buffer(s.pair_data_buf, 0, self.pd_zil, 0, self.pd_bytes)
        enc.copy_buffer_to_buffer(s.pair_data_buf, 0, self.pd_pre, 0, self.pd_bytes)
        enc.copy_buffer_to_buffer(s.write_counts_buf, 0, self.wc_zil, 0, self.wc_bytes)
        disp("z80_execute_batch", pairs_wg)                                     # the original executor, in place
        br.encode_execute(enc, self.bridge_bgs[k], self.machine_bgs[k], s.pair_count)   # ZSoup's path, on the copies
        p = enc.begin_compute_pass()
        p.set_pipeline(self.cmp_pipe)
        p.set_bind_group(0, self.cmp_bgs[k])
        p.dispatch_workgroups(pairs_wg)
        p.end()
        disp("absorb_results", pairs_wg)
        disp("mutate_soup", -(-s.mutation_count // 64))

    def run_steps(self, n: int) -> None:
        """n more steps, as few submits as the params ring allows."""
        s, q = self.soup, self.dev.queue
        done = 0
        while done < n:
            ks = min(s.ring, n - done)
            for k in range(ks):
                if self.step_index + k >= self.capacity:
                    raise ValueError("stats capacity exceeded")
                seed = s.cpu_rng.next_u32()
                params = s._params(seed)
                q.write_buffer(s.params_bufs[k], 0, params)
                if self.delta:
                    bp = params.copy()
                    bp[6] = s.z80_steps + self.delta
                    q.write_buffer(self.bparams[k], 0, bp)
                q.write_buffer(self.zparams[k], 0, self.bridge.machine_params(s.pair_count, seed, s.masks))
                q.write_buffer(self.cparams[k], 0, np.array([s.pair_count, self.wpp, self.step_index + k, self.machine.record_words, self.rec_steps, 0, 0, 0], dtype=np.uint32))
            enc = self.dev.create_command_encoder()
            for k in range(ks):
                self._encode_step(enc, k)
            q.submit([enc.finish()])
            done += ks
            self.step_index += ks
            s.batch_index += ks

    def _read(self, buf, nbytes) -> np.ndarray:
        return np.frombuffer(self.dev.queue.read_buffer(buf, 0, nbytes), dtype=np.uint32).copy()

    def _read_many(self, parts: list) -> list[np.ndarray]:
        """Several buffer ranges [(buffer, bytes from offset 0)] copied into one staging buffer and mapped once (the
        same bytes as one read_buffer per range, with one map instead of seven)."""
        total = sum(nb for _, nb in parts)
        staging = self.dev.create_buffer(size=total, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
        enc = self.dev.create_command_encoder()
        off = 0
        for buf, nb in parts:
            enc.copy_buffer_to_buffer(buf, 0, staging, off, nb)
            off += nb
        self.dev.queue.submit([enc.finish()])
        staging.map_sync(wgpu.MapMode.READ)
        raw = np.frombuffer(staging.read_mapped(), dtype=np.uint32).copy()
        staging.unmap()
        staging.destroy()
        out, off = [], 0
        for _, nb in parts:
            out.append(raw[off // 4: (off + nb) // 4])
            off += nb
        return out

    def cpu_check(self) -> dict:
        """Re-derive the last step's counters from the raw buffers on the CPU (they must equal the comparator's)."""
        s = self.soup
        n = s.pair_count
        assert len({id(b) for b in (s.pair_data_buf, self.pd_zil, self.pd_pre, s.write_counts_buf, self.wc_zil)}) == 5
        po, pz, pp, wo, wz, act, rec = self._read_many([
            (s.pair_data_buf, self.pd_bytes), (self.pd_zil, self.pd_bytes), (self.pd_pre, self.pd_bytes),
            (s.write_counts_buf, self.wc_bytes), (self.wc_zil, self.wc_bytes), (s.pair_active_buf, n * 4),
            (self.bridge.record_buf, n * self.machine.record_words * 4)])
        po, pz, pp = po.reshape(n, self.wpp), pz.reshape(n, self.wpp), pp.reshape(n, self.wpp)
        wo, wz, act, rec = wo.reshape(n, 2), wz.reshape(n, 2), act.astype(bool), rec.reshape(n, -1)
        changed = (po != pp).any(axis=1)
        st = rec[:, self.rec_steps].astype(np.int64)
        cpu = {
            "mem_mismatch": int((po != pz).any(axis=1).sum()), "wc_mismatch": int((wo != wz).any(axis=1).sum()),
            "active": int(act.sum()), "active_changed": int((act & changed).sum()), "inactive_changed": int((~act & changed).sum()),
            "writes": int(wo.astype(np.int64).sum()), "b_written": int((wo[:, 1] > 0).sum()),
            "zil_steps": int(st[act].sum()), "zil_inactive_steps": int(((~act) & (st != 0)).sum()),
        }
        t = self.step_index - 1
        gpu = np.frombuffer(self.dev.queue.read_buffer(self.stats_buf, t * K * 4, K * 4), dtype=np.uint32)
        rec_ = {"step": self.step_index, **{f"cpu_{k}": v for k, v in cpu.items()}, "gpu_agrees": all(int(gpu[i]) == cpu[k] for i, k in enumerate(STATS))}
        self.cpu_checks.append(rec_)
        return rec_

    def stats(self, upto: int | None = None) -> np.ndarray:
        upto = self.step_index if upto is None else upto
        return self._read(self.stats_buf, upto * K * 4).reshape(upto, K)

    def checkpoint(self) -> dict:
        """Population state as run.py measures it: exact-hash top share and the quasispecies occupancy (Hamming radius
        L/4 around the dominant tape, q_share; with cyclic shifts, q_shift_share)."""
        hashes = self.soup.read_hashes()
        sp = species_stats(hashes)
        cells = self.soup.read_soup()
        top = cells[int(np.argmax(hashes == np.uint32(sp["top_hash"])))]
        qs = quasispecies_share(cells, top)
        qs.update(shift_occupancy(cells, top))
        rec = {"step": self.step_index, "top_share": sp["top_share"], "unique": sp["unique"], "q_share": qs["q_share"],
               "q_shift_share": qs["q_shift_share"], "top_tape": top.tobytes().hex(" ")}
        self.trajectory.append(rec)
        return rec

    def run(self, steps: int, dense_until: int = 2000, every: int = 100, checkpoint_every: int = 1000, log=None) -> dict:
        """Run `steps` steps: CPU re-check after every step up to `dense_until`, then every `every`-th; a species
        checkpoint every `checkpoint_every` steps. Returns the summary (the comparator covers every step)."""
        while self.step_index < steps:
            t = self.step_index
            nxt = t + 1 if t + 1 <= dense_until else min(((t // every) + 1) * every, steps)
            nxt = min(nxt, ((t // checkpoint_every) + 1) * checkpoint_every, steps)
            self.run_steps(nxt - t)
            if nxt <= dense_until or nxt % every == 0 or nxt == steps:
                self.cpu_check()
            if nxt % checkpoint_every == 0 or nxt == steps:
                c = self.checkpoint()
                if log:
                    log(f"[lockstep {self.name} s={self.seed}{' delta=' + str(self.delta) if self.delta else ''}] step {nxt}: "
                        f"mismatching steps so far {int((self.stats()[:, :2].sum(axis=1) > 0).sum())}, top share {c['top_share']:.4f}, q {c['q_share']:.3f}")
        return self.summary()

    def windows(self, size: int = 1000) -> list[dict]:
        """The per-step counters summed over consecutive windows of `size` steps (non-vacuity over time: a soup that
        degenerates late would show it here, not in the run totals)."""
        st = self.stats().astype(np.int64)
        return [{"machine": self.name, "seed": self.seed, "suppress": ";".join(self.soup.patterns), "zil_budget_delta": self.delta,
                 "from_step": w + 1, "to_step": min(w + size, len(st)), **{k: int(st[w: w + size, i].sum()) for i, k in enumerate(STATS)}}
                for w in range(0, len(st), size)]

    def summary(self) -> dict:
        st = self.stats().astype(np.int64)
        bad = (st[:, 0] + st[:, 1]) > 0
        win = self.windows()
        return {
            "machine": self.name, "L": self.L, "lethal": self.zero_halts, "seed": self.seed, "suppress": ";".join(self.soup.patterns),
            "zil_budget_delta": self.delta, "steps": int(st.shape[0]),
            "steps_with_mismatch": int(bad.sum()), "first_mismatch_step": int(np.argmax(bad)) + 1 if bad.any() else -1,
            "pair_mem_mismatches": int(st[:, 0].sum()), "pair_wc_mismatches": int(st[:, 1].sum()),
            "pairs_compared": int(st.shape[0]) * self.soup.pair_count, "active_pairs": int(st[:, 2].sum()),
            "active_changed": int(st[:, 3].sum()), "inactive_changed": int(st[:, 4].sum()), "writes": int(st[:, 5].sum()),
            "b_written": int(st[:, 6].sum()), "zil_steps": int(st[:, 7].sum()), "zil_inactive_steps": int(st[:, 8].sum()),
            "min_window_active_changed": min(w["active_changed"] for w in win), "last_window_active_changed": win[-1]["active_changed"],
            "last_window_b_written": win[-1]["b_written"],
            "cpu_checks": len(self.cpu_checks), "cpu_gpu_disagree": sum(not c["gpu_agrees"] for c in self.cpu_checks),
            "cpu_mismatches": sum(c["cpu_mem_mismatch"] + c["cpu_wc_mismatch"] for c in self.cpu_checks),
            "first_checkpoint_t02": next((c["step"] for c in self.trajectory if c["top_share"] >= 0.02), -1),
            "top_share_max": max((c["top_share"] for c in self.trajectory), default=None),
            "top_tape_at_max": max(self.trajectory, key=lambda c: c["top_share"])["top_tape"][:23] if self.trajectory else None,
            "q_share_max": max((c["q_share"] for c in self.trajectory), default=None),
            "q_shift_share_max": max((c["q_shift_share"] for c in self.trajectory), default=None),
            "first_checkpoint_q10": next((c["step"] for c in self.trajectory if c["q_share"] >= 0.10), -1),
            "q_share_final": self.trajectory[-1]["q_share"] if self.trajectory else None,
        }


def _soup_dispatch(soup: Soup, enc, ep: str, k: int, wg: int) -> None:
    """Encode one of the original Soup's kernels (its own pipelines and bind group of params ring slot k) into `enc`,
    one pass per dispatch, as Soup.step does."""
    if wg == 0:
        return
    p = enc.begin_compute_pass()
    p.set_pipeline(soup.pipelines[ep])
    p.set_bind_group(0, soup.bind_groups[k])
    p.dispatch_workgroups(wg)
    p.end()


# ── run-to-run nondeterminism of the original pipeline ──────────────────────────────────────────────────────────────
def nondeterminism(L: int = 16, zero_halts: bool = False, seed: int = 1, max_steps: int = 50) -> dict:
    """Two runs of the original Soup with the same seed, one after the other in one process, each recording after
    every step the soup, the drawn pairs and the pair_active flags; the first step after which the runs differ, and
    the size of the difference at step 1."""

    def one_run():
        s = Soup(tape_length=L, seed=seed, zero_halts=zero_halts)
        recs = [(s.read_soup(), None, None)]
        for _ in range(max_steps):
            s.step(1)
            inter = s.read_interactions()
            recs.append((s.read_soup(), inter["pairs"], inter["active"]))
        del s
        return recs

    r1, r2 = one_run(), one_run()
    assert np.array_equal(r1[0][0], r2[0][0]), "the initial soups are seeded and must agree"
    first_soup = first_active = -1
    for t in range(1, max_steps + 1):
        assert np.array_equal(r1[t][1], r2[t][1]), "the drawn pairs are seeded and must agree"
        if first_active < 0 and not np.array_equal(r1[t][2], r2[t][2]):
            first_active = t
        if first_soup < 0 and not np.array_equal(r1[t][0], r2[t][0]):
            first_soup = t
    return {"L": L, "lethal": zero_halts, "seed": seed, "steps_compared": max_steps,
            "first_step_active_differ": first_active, "first_step_soup_differ": first_soup,
            "step1_active_run1": int(r1[1][2].sum()), "step1_active_run2": int(r2[1][2].sum()),
            "step1_active_flags_differ": int((r1[1][2] != r2[1][2]).sum()),
            "step1_soup_bytes_differ": int((r1[1][0] != r2[1][0]).sum()),
            "steps_with_identical_soups": int(sum(np.array_equal(r1[t][0], r2[t][0]) for t in range(1, max_steps + 1)))}


# ── a machine through ZSoup's whole pipeline vs the same machine run directly ───────────────────────────────────────
def plumbing_check(machine: str, steps: int = 200, seed: int = 7) -> dict:
    """ZSoup on `machine` for `steps` steps; after each, the soup's pair memory, write counts and the machine's lane
    records must equal a direct ZMachine.run of the same machine on the pairs the step drew, read from the soup before
    the step (all pair_count lanes, inactive ones with budget 0, with the step's batch seed, so lane indices and seeds
    match). Works for any pair-world machine (used on the 6502 skeleton to show the plumbing on a second ISA)."""
    from .prng import SplitMix64
    from .zsoup import ZSoup

    zs = ZSoup(machine, seed=seed)
    m = zs.zmachine
    initial = zs.read_soup()
    n = zs.pair_count
    tot = Counter()
    halts = Counter()
    for _ in range(steps):
        before = zs.read_soup()
        peek = SplitMix64(0)
        peek.state = zs.cpu_rng.state
        batch_seed = peek.next_u32()
        zs.step(1)
        inter = zs.read_interactions()
        mem = zs.read_pair_memory().reshape(n, -1)
        rec = zs.read_records()
        act = inter["active"]
        i, j = inter["pairs"][:, 0], inter["pairs"][:, 1]
        lanes = np.concatenate([before[i], before[j]], axis=1)
        lanes[~act] = 0
        r = m.run(lanes, budget=np.where(act, zs.z80_steps, 0), seed=batch_seed, ablation=zs.masks if m.ablation_words else None)
        wb = r.writes("B") + (r.writes("padding") if "padding" in r.write_count_names else 0)
        tot["checked_steps"] += 1
        tot["active_pairs"] += int(act.sum())
        tot["pairs_changed"] += int((mem[act] != lanes[act]).any(axis=1).sum())
        tot["writes_A"] += int(inter["write_counts"][act, 0].sum())
        tot["writes_B"] += int(inter["write_counts"][act, 1].sum())
        tot["mem_mismatch"] += int((mem[act] != r.memory[act]).any(axis=1).sum())
        tot["wc_mismatch"] += int(((inter["write_counts"][act, 0] != r.writes("A")[act]) | (inter["write_counts"][act, 1] != wb[act])).sum())
        tot["record_mismatch"] += int((rec[act] != r.record[act]).any(axis=1).sum())
        tot["inactive_wc_nonzero"] += int((inter["write_counts"][~act] != 0).any(axis=1).sum())
        dec = m.decode(rec[act])
        for reason, cause in zip(dec.halt_reason, dec.trap_cause):
            if reason:
                halts[f"{reason}/{cause}"] += 1
    names = {v: k for k, v in m.m["haltReasons"].items()}
    causes = {v: k for k, v in m.m["trapCauses"].items()}
    halt_txt = ", ".join(f"{names.get(int(k.split('/')[0]), k)}{'/' + causes.get(int(k.split('/')[1]), '') if k.split('/')[1] != '0' else ''}: {v:,}"
                         for k, v in sorted(halts.items(), key=lambda kv: -kv[1]))
    return {"machine": machine, "core": m.describe()["core"], "init_stack": json.dumps(m.m["init"].get("stack")), **dict(tot),
            "soup_bytes_changed": int((zs.read_soup() != initial).sum()), "halted_lanes (reason/cause)": halt_txt or "-"}
