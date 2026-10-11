"""Generic host for exported Zilion machines (`zilion.manifest/2`): run any machine, of any ISA, on many lanes.

    from algocell_exp.zmachine import load_machine
    m = load_machine("z80c02_pair_L16")                    # machines/<name>.wgsl + machines/<name>.manifest.json
    r = m.run(pairs, budget=128, ablation=masks)           # pairs: (n, 2L) uint8, tape A then tape B
    r.memory, r.steps, r.halt_reason, r.trap_cause, r.write_counts, r.fetch_map, r.start_map, r.state

Everything about the machine (bindings, the params struct, the memory and record layouts, the state fields) is read
from its manifest; this file names no register, binding number or params offset. A Zilion machine is generated in
TypeScript (`machines/build_machines.mjs`) and vendored with its manifest.

Provenance: adapted from zilion's `test/world/gpu_runner.py` (MIT licence, same author, NeoVand; zilion `next` at
81c88dc01083ebd940264d4e4a0708bf97f186f7; the runner is unchanged since c9aba93). Changes: the Algocell adapter policy and shared device
(`soup.get_device`); per-lane or uniform budgets on one call; typed results (`ZResult`); `bind_group`/`encode` so a
host can run a machine on buffers it owns (ZSoup runs it on the soup's `pair_data`); a check of the WGSL against the
manifest's SHA-256 on load.

Memory layouts. Host layout: the segments back to back, `manifest.memory.hostBytesPerLane` bytes per lane (for the
Algocell pair worlds: tape A then tape B, 2L bytes). Storage layout: each segment word-padded (ceil(bytes/4) u32,
little-endian bytes), which is Algocell's `pair_data` layout; ring bytes after the last segment (padding) are zeroed
at the start of every run and never stored.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import wgpu

MACHINES_DIR = Path(__file__).parent / "machines"
SCHEMA = "zilion.manifest/2"
_BTYPE = {
    "uniform": wgpu.BufferBindingType.uniform,
    "storage": wgpu.BufferBindingType.storage,
    "read-only-storage": wgpu.BufferBindingType.read_only_storage,
}


@dataclass
class ZResult:
    """Outputs of one run, one row per lane. Absent outputs (not in the machine's record) are None."""

    memory: np.ndarray | None          # (n, hostBytes) uint8, host layout (segments back to back); None if not stored
    steps: np.ndarray                  # (n,) steps executed (including the one that halted)
    halt_reason: np.ndarray            # (n,) 0 running/budget, 1 halt-insn, 2 host, 3 trap (manifest.haltReasons)
    trap_cause: np.ndarray             # (n,) manifest.trapCauses code of the trap that halted the lane, else 0
    write_counts: np.ndarray | None    # (n, k): writes per segment, then into padding (names: write_count_names)
    write_count_names: tuple = ()
    fetch_map: np.ndarray | None = None   # (n, P) bool: positions fetched as instruction stream
    start_map: np.ndarray | None = None   # (n, P) bool: positions where an instruction started
    state: np.ndarray | None = None       # (n, fields) final state record (names: state_names)
    state_names: tuple = ()
    record: np.ndarray = field(default=None, repr=False)   # (n, record words) raw lane records

    def reg(self, name: str) -> np.ndarray:
        """One state field per lane (e.g. 'pc', 'sp', 'a')."""
        return self.state[:, self.state_names.index(name)]

    def writes(self, name: str) -> np.ndarray:
        """Writes into one segment per lane (e.g. 'A', 'B', 'padding')."""
        return self.write_counts[:, self.write_count_names.index(name)]


class ZMachine:
    """One compiled machine, driven by its manifest."""

    def __init__(self, wgsl: str, manifest: dict, name: str | None = None, device: wgpu.GPUDevice | None = None) -> None:
        if manifest.get("schema") != SCHEMA:
            raise ValueError(f"manifest schema {manifest.get('schema')!r}; this host reads {SCHEMA}")
        if manifest.get("entryPoint") is None:
            raise ValueError("library-mode manifests have no entry point; embed them in a host shader")
        sha = hashlib.sha256(wgsl.encode()).hexdigest()
        if sha != manifest["sha256"]["wgsl"]:
            raise ValueError(f"{name or 'machine'}: WGSL SHA-256 {sha[:16]}… does not match its manifest ({manifest['sha256']['wgsl'][:16]}…)")
        if device is None:
            from .soup import get_device

            device = get_device()
        self.name = name
        self.m = manifest
        self.wgsl = wgsl
        self.wgsl_sha256 = sha
        self.dev = device
        self.module = device.create_shader_module(code=wgsl)
        self.bindings = sorted(manifest["bindings"], key=lambda b: b["binding"])
        self.layout = device.create_bind_group_layout(
            entries=[
                {"binding": b["binding"], "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": _BTYPE[b["type"]]}}
                for b in self.bindings
            ]
        )
        self.pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.layout]),
            compute={"module": self.module, "entry_point": manifest["entryPoint"]},
        )
        mem = manifest["memory"]
        self.ring_bytes = int(mem["ringBytes"])
        self.host_bytes = int(mem["hostBytesPerLane"])
        self.storage_words = int(mem["storageWordsPerLane"])
        self.segments = list(mem["segments"])
        self.padding = mem["padding"]
        self.record_words = int(manifest["record"]["wordsPerLane"])
        self.fields = {f["name"]: f for f in manifest["record"]["fields"]}
        self.params_bytes = max(int(manifest["params"]["bytes"]), 16)
        self.workgroup_size = int(manifest["workgroupSize"])
        self.lanes_per_invocation = int(manifest.get("lanesPerInvocation", 1))

    # --- description ----------------------------------------------------------------------------------------------
    @property
    def core(self) -> dict:
        return self.m["core"]

    @property
    def world(self) -> dict:
        return self.m["world"]

    @property
    def budget_per_lane(self) -> bool:
        return bool(self.m["budget"]["perLane"])

    @property
    def ablation_words(self) -> int:
        return int(self.m["ablation"].get("words", 0)) if self.m["ablation"]["kind"] == "uniform" else 0

    def role(self, r: str) -> dict | None:
        return next((b for b in self.bindings if b["role"] == r), None)

    def segment(self, name: str) -> dict:
        return next(s for s in self.segments if s["name"] == name)

    def describe(self) -> dict:
        """Provenance of this machine for run records."""
        c = self.core
        return {
            "machine": self.name,
            "zilion": self.m["zilion"],
            "core": f"{c['id']} {c['coreVersion']} ({c['status']}; variant {c['variant']}; compat {c['compat']}; hooks {','.join(c['hooks']) or 'none'})",
            "sha256_core": self.m["sha256"]["core"],
            "sha256_world": self.m["sha256"]["world"],
            "sha256_wgsl": self.wgsl_sha256,
            "ring": self.ring_bytes,
            "segments": [(s["name"], s["bytes"]) for s in self.segments],
            "init": self.m["init"],
            "budget": self.m["budget"],
            "halting": self.world.get("halting"),
            "ablation": self.m["ablation"].get("layout") if self.m["ablation"]["kind"] == "uniform" else self.m["ablation"]["kind"],
        }

    # --- memory layout --------------------------------------------------------------------------------------------
    def pack(self, host: np.ndarray) -> np.ndarray:
        """(n, hostBytes) uint8 → (n, storageWords) uint32 (each segment word-padded, pad bytes zero)."""
        host = np.ascontiguousarray(host, dtype=np.uint8).reshape(-1, self.host_bytes)
        st = np.zeros((host.shape[0], self.storage_words * 4), dtype=np.uint8)
        for s in self.segments:
            st[:, 4 * s["storageWord"]: 4 * s["storageWord"] + s["bytes"]] = host[:, s["hostOffset"]: s["hostOffset"] + s["bytes"]]
        return st.view(np.uint32)

    def unpack(self, storage: np.ndarray) -> np.ndarray:
        """(n, storageWords) uint32 → (n, hostBytes) uint8."""
        st = np.ascontiguousarray(storage, dtype=np.uint32).view(np.uint8).reshape(-1, self.storage_words * 4)
        host = np.zeros((st.shape[0], self.host_bytes), dtype=np.uint8)
        for s in self.segments:
            host[:, s["hostOffset"]: s["hostOffset"] + s["bytes"]] = st[:, 4 * s["storageWord"]: 4 * s["storageWord"] + s["bytes"]]
        return host

    # --- params ---------------------------------------------------------------------------------------------------
    def params(self, n: int, lane_base: int = 0, seed: int = 0, budget: int = 0, ablation=None, step_base: int = 0, resume: bool = False) -> np.ndarray:
        """The params uniform as u32 words. `budget` is ignored by a per-lane-budget machine (it reads its buffer)."""
        p = np.zeros(self.params_bytes // 4, dtype=np.uint32)
        roles = {f["role"] for f in self.m["params"]["fields"]}
        if resume and "resume" not in roles:
            raise ValueError("this machine cannot resume lanes (manifest.resumable is false)")
        if step_base and "stepBase" not in roles:
            raise ValueError("this machine has no step base (params role stepBase)")
        if not (0 <= step_base < 2**64):
            raise ValueError(f"step_base {step_base} is not a 64-bit step index")
        abl = None
        if ablation is not None:
            abl = np.asarray(ablation, dtype=np.uint32).reshape(-1)
            if "ablation" not in roles:
                if abl.any():
                    raise ValueError(f"{self.name}: this machine has no ablation table, but a non-empty one was given")
                abl = None
        for f in self.m["params"]["fields"]:
            o, r = f["offset"] // 4, f["role"]
            if r == "laneCount":
                p[o] = n
            elif r == "laneBase":
                p[o] = lane_base
            elif r == "budget":
                p[o] = budget
            elif r == "seed":
                p[o] = seed & 0xFFFFFFFF
            elif r == "stepBase":
                p[o] = step_base & 0xFFFFFFFF
                p[o + 1] = step_base >> 32
            elif r == "resume":
                p[o] = 1 if resume else 0
            elif r == "constant":
                p[o] = f["value"]
            elif r == "ablation" and abl is not None:
                if abl.size != f["words"]:
                    raise ValueError(f"{self.name}: ablation table of {abl.size} words, the machine's has {f['words']}")
                p[o: o + f["words"]] = abl
        return p

    # --- running on host-owned buffers ----------------------------------------------------------------------------
    def bind_group(self, params, memory, record, budget=None, state=None) -> wgpu.GPUBindGroup:
        """A bind group over buffers the caller owns. Each argument is a GPUBuffer or (buffer, offset, size)."""
        given = {"params": params, "memory": memory, "record": record, "budgetIn": budget, "stateIn": state}
        entries = []
        for b in self.bindings:
            res = given.get(b["role"])
            if res is None:
                raise ValueError(f"{self.name}: binding {b['binding']} ({b['role']}) needs a buffer")
            buf, off, size = (res, 0, res.size) if isinstance(res, wgpu.GPUBuffer) else res
            entries.append({"binding": b["binding"], "resource": {"buffer": buf, "offset": off, "size": size}})
        return self.dev.create_bind_group(layout=self.layout, entries=entries)

    def encode(self, encoder: wgpu.GPUCommandEncoder, bind_group: wgpu.GPUBindGroup, lanes: int) -> None:
        """One dispatch of `lanes` lanes (the params' laneCount must be `lanes`)."""
        if lanes <= 0:
            return
        p = encoder.begin_compute_pass()
        p.set_pipeline(self.pipeline)
        p.set_bind_group(0, bind_group)
        p.dispatch_workgroups(-(-lanes // (self.workgroup_size * self.lanes_per_invocation)))
        p.end()

    def max_lanes_per_dispatch(self) -> int:
        """Largest lane count of one dispatch: the workgroup-count limit, and every per-lane binding within the storage
        binding size (and buffer size) limits."""
        lim = self.dev.limits
        lanes = int(lim.get("max-compute-workgroups-per-dimension", 65535)) * self.workgroup_size * self.lanes_per_invocation
        cap = min(int(lim.get("max-storage-buffer-binding-size", 1 << 27)), int(lim.get("max-buffer-size", 1 << 28)))
        for words in (self.storage_words, self.record_words, *(b.get("wordsPerLane", 0) for b in self.bindings)):
            if words:
                lanes = min(lanes, cap // (4 * words))
        return lanes

    # --- running from the host ------------------------------------------------------------------------------------
    def run(self, memory: np.ndarray, budget=None, seed: int = 0, ablation=None, step_base: int = 0, lane_base: int = 0,
            state=None, resume=None, chunk: int | None = None) -> ZResult:
        """Run every lane once. memory: (n, hostBytes) uint8 (host layout). budget: an int for every lane, or an (n,)
        array on a per-lane-budget machine (0 = the lane takes no step); default the manifest's. ablation: the table's
        u32 words (e.g. Algocell's 24 suppression words on a Z80 'coarse' table). step_base: the dial index of the
        first step. resume: previous records (n, record words) to continue from, with `memory` their memory."""
        memory = np.ascontiguousarray(memory, dtype=np.uint8).reshape(-1, self.host_bytes)
        n = memory.shape[0]
        if self.budget_per_lane:
            if budget is None:
                raise ValueError(f"{self.name}: a per-lane-budget machine needs a budget (an int or an (n,) array)")
            budgets = np.broadcast_to(np.asarray(budget, dtype=np.int64), (n,))
            if budgets.min(initial=0) < 0 or budgets.max(initial=0) > 0xFFFFFFFF:
                raise ValueError("budgets must be u32")
            budgets = np.ascontiguousarray(budgets, dtype=np.uint32)
            uniform_budget = 0
        else:
            if budget is None:
                budget = self.m["budget"]["default"]
            if np.ndim(budget) != 0:
                raise ValueError(f"{self.name}: per-lane budgets need a machine whose world has budget.value 'perLane'")
            budgets, uniform_budget = None, int(budget)
        if resume is not None:
            resume = np.ascontiguousarray(np.asarray(resume, dtype=np.uint32).reshape(n, self.record_words))
        st_role = self.role("stateIn")
        if st_role is not None and state is None and resume is None:
            raise ValueError(f"{self.name}: this machine loads the state per lane (init.state 'perLane'): pass state")
        chunk = min(chunk or n, self.max_lanes_per_dispatch()) if n else 1
        mem_out = np.zeros((n, self.host_bytes), dtype=np.uint8)
        rec_out = np.zeros((n, self.record_words), dtype=np.uint32)
        B = wgpu.BufferUsage
        dev = self.dev
        for s0 in range(0, n, chunk):
            c = min(chunk, n - s0)
            params = self.params(c, lane_base + s0, seed, uniform_budget, ablation, step_base, resume is not None)
            pbuf = dev.create_buffer(size=params.nbytes, usage=B.UNIFORM | B.COPY_DST)
            dev.queue.write_buffer(pbuf, 0, params.tobytes())
            mbuf = dev.create_buffer(size=max(c * self.storage_words * 4, 4), usage=B.STORAGE | B.COPY_SRC | B.COPY_DST)
            dev.queue.write_buffer(mbuf, 0, self.pack(memory[s0: s0 + c]).tobytes())
            rbuf = dev.create_buffer(size=max(c * self.record_words * 4, 4), usage=B.STORAGE | B.COPY_SRC | B.COPY_DST)
            if resume is not None:
                dev.queue.write_buffer(rbuf, 0, resume[s0: s0 + c].tobytes())
            bbuf = sbuf = None
            if budgets is not None:
                bbuf = dev.create_buffer(size=max(c * 4, 4), usage=B.STORAGE | B.COPY_DST)
                dev.queue.write_buffer(bbuf, 0, budgets[s0: s0 + c].tobytes())
            if st_role is not None:
                words = st_role["wordsPerLane"]
                data = (np.zeros((c, words), dtype=np.uint32) if state is None
                        else np.ascontiguousarray(np.asarray(state, dtype=np.uint32).reshape(n, words)[s0: s0 + c]))
                sbuf = dev.create_buffer(size=max(data.nbytes, 4), usage=B.STORAGE | B.COPY_DST)
                dev.queue.write_buffer(sbuf, 0, data.tobytes())
            bg = self.bind_group(pbuf, mbuf, rbuf, bbuf, sbuf)
            enc = dev.create_command_encoder()
            self.encode(enc, bg, c)
            dev.queue.submit([enc.finish()])
            st = np.frombuffer(dev.queue.read_buffer(mbuf), dtype=np.uint32)[: c * self.storage_words]
            mem_out[s0: s0 + c] = self.unpack(st.reshape(c, self.storage_words))
            rec_out[s0: s0 + c] = np.frombuffer(dev.queue.read_buffer(rbuf), dtype=np.uint32)[: c * self.record_words].reshape(c, self.record_words)
            for b in (pbuf, mbuf, rbuf, bbuf, sbuf):
                if b is not None:
                    b.destroy()
        res = self.decode(rec_out)
        res.memory = mem_out if self.m["memory"]["stored"] else None
        return res

    def decode(self, record: np.ndarray) -> ZResult:
        """Typed view of raw lane records (n, record words)."""
        rec = np.asarray(record, dtype=np.uint32).reshape(-1, self.record_words)
        f = self.fields

        def col(name):
            return rec[:, f[name]["offset"]].copy()

        def block(name):
            if name not in f:
                return None
            return rec[:, f[name]["offset"]: f[name]["offset"] + f[name]["words"]].copy()

        def bitmap(name):
            b = block(name)
            if b is None:
                return None
            return np.unpackbits(np.ascontiguousarray(b).view(np.uint8), axis=1, bitorder="little")[:, : self.ring_bytes].astype(bool)

        return ZResult(
            memory=None,
            steps=col("steps"),
            halt_reason=col("haltReason"),
            trap_cause=col("trapCause"),
            write_counts=block("writeCounts"),
            write_count_names=tuple(f["writeCounts"]["items"]) if "writeCounts" in f else (),
            fetch_map=bitmap("fetchMap"),
            start_map=bitmap("startMap"),
            state=block("state"),
            state_names=tuple(f["state"]["items"]) if "state" in f else (),
            record=rec,
        )


_CACHE: dict = {}


def machine_paths(name: str, machines_dir: Path | str = MACHINES_DIR) -> tuple[Path, Path]:
    d = Path(machines_dir)
    return d / f"{name}.wgsl", d / f"{name}.manifest.json"


def available(machines_dir: Path | str = MACHINES_DIR) -> list[str]:
    return sorted(p.name[: -len(".manifest.json")] for p in Path(machines_dir).glob("*.manifest.json"))


def load_machine(name: str, machines_dir: Path | str = MACHINES_DIR, device: wgpu.GPUDevice | None = None) -> ZMachine:
    """The vendored machine `name` (machines/<name>.wgsl + .manifest.json), compiled once per device."""
    if device is None:
        from .soup import get_device

        device = get_device()
    wgsl_path, man_path = machine_paths(name, machines_dir)
    if not man_path.exists():
        raise ValueError(f"no machine {name!r} in {machines_dir} (have: {', '.join(available(machines_dir))})")
    key = (str(wgsl_path.resolve()), id(device))
    if key not in _CACHE:
        _CACHE[key] = ZMachine(wgsl_path.read_text(), json.loads(man_path.read_text()), name=name, device=device)
    return _CACHE[key]
