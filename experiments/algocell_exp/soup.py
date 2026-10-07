"""wgpu host for the exported simulation shader. Mirrors GPUEngine (src/lib/gpu/engine.ts):
same buffers, bind group layout, dispatch sizes, params layout, PRNG and
initialisation, minus rendering. Several steps are encoded per submit via a
ring of params buffers so Python overhead stays small."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import wgpu

from .isa import masks as make_masks
from .isa import resolve, parse_patterns
from .prng import SplitMix64

SHADER_DIR = Path(__file__).parent / "shader"
MAX_PAIRS = 32768   # buffer capacity; the drawn pair count per step is params.pair_count (8192 by default)
ENTRY_POINTS = (
    "clear_collision",
    "prepare_batch",
    "z80_execute_batch",
    "absorb_results",
    "mutate_soup",
    "count_bytes",
    "clear_byte_counts",
    "hash_cells",
)

_DEVICE: wgpu.GPUDevice | None = None
_ADAPTER_INFO: dict = {}


def pick_adapter() -> wgpu.GPUAdapter:
    """Prefer a real GPU; never silently run on a CPU rasterizer (llvmpipe)."""
    adapters = list(wgpu.gpu.enumerate_adapters_sync())
    rank = {"DiscreteGPU": 0, "IntegratedGPU": 1, "VirtualGPU": 2, "CPU": 3, "Unknown": 4}
    # Same physical GPU can be enumerated through several backends (Vulkan, OpenGL/EGL); WGPU_BACKEND_TYPE
    # is not honoured by enumerate_adapters, so prefer Vulkan/Metal explicitly and never a CPU or unknown adapter.
    backend_rank = {"Vulkan": 0, "Metal": 0, "D3D12": 1, "OpenGL": 2}
    adapters.sort(key=lambda a: (rank.get(str(a.info.get("adapter_type", "Unknown")), 5), backend_rank.get(str(a.info.get("backend_type")), 3)))
    if not adapters:
        raise RuntimeError("no WebGPU adapter")
    chosen = adapters[0]
    if str(chosen.info.get("adapter_type")) in ("CPU", "Unknown"):
        raise RuntimeError(f"no usable GPU adapter (best was {chosen.info})")
    return chosen


def get_device() -> wgpu.GPUDevice:
    global _DEVICE, _ADAPTER_INFO
    if _DEVICE is None:
        adapter = pick_adapter()
        _ADAPTER_INFO = dict(adapter.info)
        _DEVICE = adapter.request_device_sync()
    return _DEVICE


def adapter_summary() -> str:
    get_device()
    return f"{_ADAPTER_INFO.get('device', '?')} ({_ADAPTER_INFO.get('backend_type', '?')}, {_ADAPTER_INFO.get('adapter_type', '?')})"


class Soup:
    def __init__(
        self,
        width: int = 160,
        height: int = 125,
        grid: str = "square",
        tape_length: int | None = None,
        seed: int = 6,
        pair_count: int = 8192,
        z80_steps: int = 128,
        noise_exp: int = 4,
        suppress: str | list[str] | None = None,
        ring: int = 32,
        device: wgpu.GPUDevice | None = None,
        mutations_per_step: int | None = None,
    ) -> None:
        assert grid in ("square", "hex")
        assert 1 <= pair_count <= MAX_PAIRS
        self.device = device or get_device()
        self.width, self.height, self.grid = width, height, grid
        if grid == "hex":
            self.tape_length = 19
            self.shader_file = SHADER_DIR / "sim_hex.wgsl"
        else:
            self.tape_length = tape_length or 16
            self.shader_file = SHADER_DIR / f"sim_square_L{self.tape_length}.wgsl"
            if not self.shader_file.exists():
                raise ValueError(f"no exported shader for tape length {self.tape_length} (run `npm run export:sim`)")
        self.pair_length = self.tape_length * 2
        self.words_per_cell = (self.tape_length + 3) // 4
        self.cell_count = width * height
        self.pair_count = pair_count
        self.z80_steps = z80_steps
        self.noise_exp = noise_exp
        # Default: pair_count / 2^noise_exp byte replacements per step (constant per cell, ∝ 1/L per byte).
        # The override exists for the per-byte-constant mutation control arm.
        self.mutations_per_step = mutations_per_step
        self.ring = ring
        self.patterns = parse_patterns(suppress) if isinstance(suppress, str) else list(suppress or [])
        self.sets = resolve(self.patterns)
        self.masks = make_masks(self.sets)
        self.batch_index = 0
        self._build()
        self.reset(seed)

    # ── GPU setup ────────────────────────────────────────────────────────────
    def _build(self) -> None:
        dev = self.device
        B = wgpu.BufferUsage
        wgsl = self.shader_file.read_text()
        self.shader_sha256_16 = hashlib.sha256(wgsl.encode()).hexdigest()[:16]
        module = dev.create_shader_module(code=wgsl)
        soup_bytes = self.cell_count * self.words_per_cell * 4
        words_per_pair = self.words_per_cell * 2
        mk = lambda size, extra=0: dev.create_buffer(size=size, usage=B.STORAGE | B.COPY_DST | extra)  # noqa: E731
        self.soup_buf = mk(soup_bytes, B.COPY_SRC)
        self.pairs_buf = mk(MAX_PAIRS * 2 * 4, B.COPY_SRC)
        self.pair_data_buf = mk(MAX_PAIRS * words_per_pair * 4, B.COPY_SRC)
        self.write_counts_buf = mk(MAX_PAIRS * 2 * 4, B.COPY_SRC)
        self.rng_states_buf = mk(MAX_PAIRS * 4)
        self.pair_active_buf = mk(MAX_PAIRS * 4, B.COPY_SRC)
        self.byte_counts_buf = mk(256 * 4, B.COPY_SRC)
        self.collision_buf = mk(self.cell_count * 4)
        self.hash_buf = mk(self.cell_count * 4, B.COPY_SRC)
        self.params_bufs = [dev.create_buffer(size=128, usage=B.UNIFORM | B.COPY_DST) for _ in range(self.ring)]

        storage = {"type": wgpu.BufferBindingType.storage}
        uniform = {"type": wgpu.BufferBindingType.uniform}
        layout = dev.create_bind_group_layout(
            entries=[
                {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": (uniform if i == 5 else storage)}
                for i in range(10)
            ]
        )
        pl = dev.create_pipeline_layout(bind_group_layouts=[layout])
        self.pipelines = {
            ep: dev.create_compute_pipeline(layout=pl, compute={"module": module, "entry_point": ep}) for ep in ENTRY_POINTS
        }

        def res(buf):
            return {"buffer": buf, "offset": 0, "size": buf.size}

        self.bind_groups = []
        for k in range(self.ring):
            self.bind_groups.append(
                dev.create_bind_group(
                    layout=layout,
                    entries=[
                        {"binding": 0, "resource": res(self.soup_buf)},
                        {"binding": 1, "resource": res(self.pairs_buf)},
                        {"binding": 2, "resource": res(self.pair_data_buf)},
                        {"binding": 3, "resource": res(self.write_counts_buf)},
                        {"binding": 4, "resource": res(self.rng_states_buf)},
                        {"binding": 5, "resource": res(self.params_bufs[k])},
                        {"binding": 6, "resource": res(self.pair_active_buf)},
                        {"binding": 7, "resource": res(self.byte_counts_buf)},
                        {"binding": 8, "resource": res(self.collision_buf)},
                        {"binding": 9, "resource": res(self.hash_buf)},
                    ],
                )
            )

    # ── State ────────────────────────────────────────────────────────────────
    def initial_soup(self, seed: int) -> np.ndarray:
        """Byte-identical to GPUEngine.initSoup(): SplitMix64 words, little-endian, padding zeroed."""
        size = self.cell_count * self.words_per_cell * 4
        rng = SplitMix64(seed)
        words = np.fromiter((rng.next_u32() for _ in range(size // 4)), dtype=np.uint32, count=size // 4)
        data = words.view(np.uint8).copy()  # little-endian on all supported hosts
        stride = self.words_per_cell * 4
        if stride != self.tape_length:
            cells = data.reshape(self.cell_count, stride)
            cells[:, self.tape_length :] = 0
            data = cells.reshape(-1)
        return data

    def reset(self, seed: int) -> None:
        self.seed = seed
        self.cpu_rng = SplitMix64(seed)
        self.device.queue.write_buffer(self.soup_buf, 0, self.initial_soup(seed))
        self.batch_index = 0

    def set_suppress(self, patterns: str | list[str] | None) -> None:
        self.patterns = parse_patterns(patterns) if isinstance(patterns, str) else list(patterns or [])
        self.sets = resolve(self.patterns)
        self.masks = make_masks(self.sets)

    @property
    def mutation_count(self) -> int:
        if self.mutations_per_step is not None:
            return int(self.mutations_per_step)
        return self.pair_count // (2**self.noise_exp)

    def _params(self, batch_seed: int) -> np.ndarray:
        p = np.zeros(32, dtype=np.uint32)
        p[:8] = (
            self.width,
            self.height,
            self.tape_length,
            self.pair_length,
            self.pair_count,
            self.mutation_count,
            self.z80_steps,
            batch_seed,
        )
        p[8:] = self.masks
        return p

    # ── Stepping ─────────────────────────────────────────────────────────────
    def step(self, n: int = 1) -> None:
        """Run n simulation steps (each = clear → prepare → execute → absorb → mutate)."""
        dev = self.device
        q = dev.queue
        pairs_wg = -(-self.pair_count // 64)
        cells_wg = -(-self.cell_count // 256)
        mut = self.mutation_count
        mut_wg = -(-mut // 64)
        done = 0
        while done < n:
            k_steps = min(self.ring, n - done)
            for k in range(k_steps):
                q.write_buffer(self.params_bufs[k], 0, self._params(self.cpu_rng.next_u32()))
            enc = dev.create_command_encoder()
            for k in range(k_steps):
                bg = self.bind_groups[k]
                for ep, wg in (
                    ("clear_collision", cells_wg),
                    ("prepare_batch", pairs_wg),
                    ("z80_execute_batch", pairs_wg),
                    ("absorb_results", pairs_wg),
                    ("mutate_soup", mut_wg),
                ):
                    if wg == 0:
                        continue
                    p = enc.begin_compute_pass()
                    p.set_pipeline(self.pipelines[ep])
                    p.set_bind_group(0, bg)
                    p.dispatch_workgroups(wg)
                    p.end()
            q.submit([enc.finish()])
            done += k_steps
            self.batch_index += k_steps

    # ── Readbacks ────────────────────────────────────────────────────────────
    def _dispatch(self, ep: str, wg: int) -> None:
        enc = self.device.create_command_encoder()
        p = enc.begin_compute_pass()
        p.set_pipeline(self.pipelines[ep])
        p.set_bind_group(0, self.bind_groups[0])
        p.dispatch_workgroups(wg)
        p.end()
        self.device.queue.submit([enc.finish()])

    def read_byte_counts(self) -> np.ndarray:
        self._dispatch("clear_byte_counts", 1)
        self._dispatch("count_bytes", -(-(self.cell_count * self.words_per_cell) // 256))
        return np.frombuffer(self.device.queue.read_buffer(self.byte_counts_buf), dtype=np.uint32).copy()

    def read_hashes(self) -> np.ndarray:
        self._dispatch("hash_cells", -(-self.cell_count // 256))
        return np.frombuffer(self.device.queue.read_buffer(self.hash_buf), dtype=np.uint32).copy()

    def read_active_pairs(self) -> int:
        """Pairs that survived the collision claim in the LAST executed step (the kernel draws
        pair_count pairs; ≈57% are active on the 160×125 grid). Needed to express time in
        effective interactions per cell."""
        a = np.frombuffer(self.device.queue.read_buffer(self.pair_active_buf), dtype=np.uint32)
        return int(a[: self.pair_count].sum())

    def read_interactions(self) -> dict:
        """Per-pair data of the LAST executed step, computed by the shader anyway: the two cell indices,
        the active flag, and the number of byte writes each program made into A and into B
        (write_counts[pair, 0] = writes landing in A's half, [pair, 1] = in B's half; repeated writes to
        one address count each time). ≈ 200 KB of readback."""
        n = self.pair_count
        q = self.device.queue
        pairs = np.frombuffer(q.read_buffer(self.pairs_buf), dtype=np.uint32)[: 2 * n].reshape(n, 2).copy()
        active = np.frombuffer(q.read_buffer(self.pair_active_buf), dtype=np.uint32)[:n].astype(bool)
        wc = np.frombuffer(q.read_buffer(self.write_counts_buf), dtype=np.uint32)[: 2 * n].reshape(n, 2).copy()
        return {"pairs": pairs, "active": active, "write_counts": wc}

    def read_pair_memory(self) -> np.ndarray:
        """(pair_count, 2, L) post-execution, PRE-mutation memories of the last step's pairs (padding stripped)."""
        wpc = self.words_per_cell
        raw = np.frombuffer(self.device.queue.read_buffer(self.pair_data_buf), dtype=np.uint8)[: self.pair_count * 2 * wpc * 4]
        return raw.reshape(self.pair_count, 2, wpc * 4)[:, :, : self.tape_length].copy()

    def read_soup(self) -> np.ndarray:
        """(cell_count, tape_length) uint8, padding stripped."""
        raw = np.frombuffer(self.device.queue.read_buffer(self.soup_buf), dtype=np.uint8)
        return raw.reshape(self.cell_count, self.words_per_cell * 4)[:, : self.tape_length].copy()

    def sync(self) -> None:
        """Block until all submitted work is done (a tiny readback)."""
        self.device.queue.read_buffer(self.byte_counts_buf, size=4)
