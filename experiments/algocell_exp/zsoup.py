"""ZSoup: Algocell's soup with the execution of each step's pairs delegated to a Zilion machine.

    from algocell_exp.zsoup import ZSoup
    soup = ZSoup("z80c02_pair_L16", seed=6)          # any vendored machine whose world is a pair of tapes A, B
    soup.step(1000); soup.read_soup(); soup.read_interactions()   # the Soup API, unchanged

What is kept, and what is new:

- **The soup's kernels are the original ones, verbatim.** `extract_soup_kernels` parses the exported simulation shader
  (`shader/sim_square_L<L>.wgsl`, or `sim_lethal_L<L>.wgsl` for a lethal machine) into its top-level items and keeps the
  ten bindings (0-9) and the kernels clear_collision, prepare_batch, absorb_results, mutate_soup, count_bytes,
  clear_byte_counts and hash_cells with everything they reference (Params, the PCG stream). Each kept item is the exact
  source text (`tests/test_zsoup.py` checks every one byte for byte against the source). Nothing of the Z80 is kept:
  z80_execute_batch and the core it calls are dropped.
- **Execution is the Zilion machine's.** Its memory binding is the soup's own `pair_data` buffer (the machine's
  segments A and B are stored word-padded exactly as `pair_data` lays out a pair), one lane per drawn pair slot.
- **Two bridge kernels** (the only new WGSL): `zsoup_budget_from_active` gives each lane the budget `params.z80_steps`
  if its pair won the collision claim and 0 otherwise (a lane with budget 0 takes no step: the machine loads and stores
  its memory unchanged), and `zsoup_write_counts` copies the machine's per-segment write counts of each active pair into
  `write_counts` (writes into ring padding count as B's, as Algocell counts them), so `read_interactions()` is unchanged.

One step: clear_collision → prepare_batch → zsoup_budget_from_active → machine → zsoup_write_counts → absorb_results →
mutate_soup (Soup's step with z80_execute_batch replaced by the middle three dispatches).

ISA-agnostic: the world's conventions (initial stack, entry point, lethal rule, ablation layout) are the machine's,
compiled into it by Zilion and listed in its manifest; ZSoup only checks that the memory layout is the soup's pair.
The machine's run seed is the step's batch seed (only machines with dials or random initial state read it); the
opcode-suppression masks of the soup go to the machine's ablation table (a Z80 'coarse' table is Algocell's
params.suppress layout); a machine without a table refuses a non-empty suppression set.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wgpu

from .isa import masks as make_masks
from .isa import parse_patterns, resolve
from .soup import MAX_PAIRS, SHADER_DIR, Soup, get_device
from .zmachine import MACHINES_DIR, ZMachine, load_machine

SOUP_KERNELS = (
    "clear_collision",
    "prepare_batch",
    "absorb_results",
    "mutate_soup",
    "count_bytes",
    "clear_byte_counts",
    "hash_cells",
)
BRIDGE_KERNELS = ("zsoup_budget_from_active", "zsoup_write_counts")
SOUP_BINDINGS = tuple(range(10))   # the soup's bind group: soup, pairs, pair_data, write_counts, rng_states, params, pair_active, byte_counts, collision_mask, cell_hashes
BUDGET_BINDING, RECORD_BINDING = 10, 11

# ── WGSL top-level items ─────────────────────────────────────────────────────────────────────────────────────────────
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class WgslItem:
    kind: str                 # 'fn', 'struct', 'binding', 'var', 'const', 'override', 'alias'
    name: str
    start: int                # offsets in the source: text == source[start:end]
    end: int
    text: str
    binding: int | None = None


def _blank_comments(src: str) -> str:
    """The source with every // comment replaced by spaces (same length, so offsets carry over). The exported shaders
    have no block comments (checked)."""
    if "/*" in src:
        raise ValueError("block comments are not supported by this parser")
    return re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), src)


def _match_brace(code: str, open_at: int) -> int:
    depth = 0
    for k in range(open_at, len(code)):
        if code[k] == "{":
            depth += 1
        elif code[k] == "}":
            depth -= 1
            if depth == 0:
                return k + 1
    raise ValueError(f"unbalanced braces from offset {open_at}")


def parse_wgsl_items(src: str) -> list[WgslItem]:
    """Top-level declarations of a WGSL module, each with its exact source text (attributes included)."""
    code = _blank_comments(src)
    items: list[WgslItem] = []
    i, n = 0, len(code)
    while i < n:
        if code[i].isspace():
            i += 1
            continue
        start = i
        attrs = []
        while code.startswith("@", i):
            m = _IDENT.match(code, i + 1)
            j = m.end()
            if code.startswith("(", j):
                j = code.index(")", j) + 1
            attrs.append(code[i:j])
            i = j
            while i < n and code[i].isspace():
                i += 1
        m = _IDENT.match(code, i)
        if not m:
            raise ValueError(f"unexpected text at offset {i}: {src[i:i + 40]!r}")
        kw = m.group(0)
        if kw in ("fn", "struct"):
            j = m.end()
            while code[j].isspace():
                j += 1
            name = _IDENT.match(code, j).group(0)
            end = _match_brace(code, code.index("{", m.end()))
            if kw == "struct" and code.startswith(";", end):
                end += 1
        elif kw in ("var", "const", "override", "alias"):
            j = m.end()
            if kw == "var" and code.startswith("<", j):
                j = code.index(">", j) + 1
            name = _IDENT.search(code, j).group(0)
            end = code.index(";", j) + 1
        else:
            raise ValueError(f"unexpected top-level keyword {kw!r} at offset {i}")
        binding = None
        for a in attrs:
            bm = re.fullmatch(r"@binding\((\d+)\)", a)
            if bm:
                binding = int(bm.group(1))
        kind = "binding" if binding is not None else kw
        items.append(WgslItem(kind, name, start, end, src[start:end], binding))
        i = end
    return items


def _references(item: WgslItem) -> set[str]:
    return set(_IDENT.findall(_blank_comments(item.text))) - {item.name}


def extract_soup_kernels(source: Path | str) -> tuple[str, list[WgslItem], str]:
    """From an exported simulation shader, the soup's bindings (0-9) and kernels with their dependency closure, as
    (composed text, kept items in source order, source text). Each kept item's text is source[item.start:item.end]."""
    src = Path(source).read_text()
    items = parse_wgsl_items(src)
    by_name = {}
    for it in items:
        if it.name in by_name:
            raise ValueError(f"{Path(source).name}: duplicate top-level name {it.name!r}")
        by_name[it.name] = it
    missing = [k for k in SOUP_KERNELS if k not in by_name]
    if missing:
        raise ValueError(f"{Path(source).name}: kernels not found: {missing}")
    keep = set(SOUP_KERNELS) | {it.name for it in items if it.kind == "binding" and it.binding in SOUP_BINDINGS}
    stack = list(keep)
    while stack:
        for ref in _references(by_name[stack.pop()]):
            if ref in by_name and ref not in keep:
                keep.add(ref)
                stack.append(ref)
    kept = [it for it in items if it.name in keep]
    if sorted(it.binding for it in kept if it.kind == "binding") != list(SOUP_BINDINGS):
        raise ValueError(f"{Path(source).name}: the soup bindings are not exactly 0-9")
    sha = hashlib.sha256(src.encode()).hexdigest()
    head = (
        f"// ZSoup soup shader. Kept verbatim from {Path(source).name} (sha256 {sha}): {len(kept)} top-level items, in\n"
        f"// source order (bindings 0-9, the soup kernels and what they reference). No Z80 code is kept.\n\n"
    )
    return head + "\n\n".join(it.text for it in kept) + "\n", kept, src


def bridge_wgsl(machine: ZMachine) -> str:
    """The two bridge kernels for this machine's record layout (constants from its manifest)."""
    wc = machine.fields["writeCounts"]
    items = list(wc["items"])
    a = wc["offset"] + items.index("A")
    b = wc["offset"] + items.index("B")
    pad = wc["offset"] + items.index("padding") if "padding" in items else None
    b_expr = "zsoup_record[o + ZSOUP_WRITES_B]" + (" + zsoup_record[o + ZSOUP_WRITES_PADDING]" if pad is not None else "")
    return f"""
// === ZSoup bridge (new; not from the source shader) ===
// The Zilion machine `{machine.name}` (core sha256 {machine.m['sha256']['core']}) executes the drawn pairs in place
// in pair_data, one lane per pair slot. A pair that lost the collision claim gets budget 0: the machine loads and stores
// its memory unchanged and takes no step. Then the per-segment write counts of each active pair go to write_counts.
@group(0) @binding({BUDGET_BINDING}) var<storage, read_write> zsoup_budget: array<u32>;   // the machine's per-lane budgets
@group(0) @binding({RECORD_BINDING}) var<storage, read_write> zsoup_record: array<u32>;   // the machine's lane records
const ZSOUP_RECORD_WORDS: u32 = {machine.record_words}u;
const ZSOUP_WRITES_A: u32 = {a}u;   // record word: writes into segment A
const ZSOUP_WRITES_B: u32 = {b}u;   // record word: writes into segment B
{f"const ZSOUP_WRITES_PADDING: u32 = {pad}u;   // record word: writes into ring padding (counted as B's)" if pad is not None else ""}

@compute @workgroup_size(64)
fn zsoup_budget_from_active(@builtin(global_invocation_id) id: vec3u) {{
    let pair_id = id.x;
    if (pair_id >= params.pair_count) {{ return; }}
    zsoup_budget[pair_id] = select(0u, params.z80_steps, pair_active[pair_id] != 0u);
}}

@compute @workgroup_size(64)
fn zsoup_write_counts(@builtin(global_invocation_id) id: vec3u) {{
    let pair_id = id.x;
    if (pair_id >= params.pair_count) {{ return; }}
    if (pair_active[pair_id] == 0u) {{ return; }}   // as z80_execute_batch: inactive pairs keep prepare_batch's zeros
    let o = pair_id * ZSOUP_RECORD_WORDS;
    write_counts[pair_id * 2u] = zsoup_record[o + ZSOUP_WRITES_A];
    write_counts[pair_id * 2u + 1u] = {b_expr};
}}
"""


def check_pair_layout(machine: ZMachine) -> tuple[int, int]:
    """(L, P) of a machine whose memory is Algocell's pair (segments A, B of L bytes, word-padded back to back on a ring
    of P >= 2L); raises otherwise. Also requires a per-lane step budget and write counts in the record."""
    mem = machine.m["memory"]
    segs = mem["segments"]
    if mem["model"] != "segments" or [s["name"] for s in segs] != ["A", "B"] or segs[0]["bytes"] != segs[1]["bytes"]:
        raise ValueError(f"{machine.name}: memory is not a pair of equal tapes A, B (segments {[(s['name'], s['bytes']) for s in segs]})")
    L = int(segs[0]["bytes"])
    wpc = (L + 3) // 4
    if (segs[0]["position"], segs[0]["storageWord"], segs[1]["position"], segs[1]["storageWord"], mem["storageWordsPerLane"]) != (0, 0, L, wpc, 2 * wpc):
        raise ValueError(f"{machine.name}: segment layout differs from pair_data's (A at word 0, B at word {wpc}, {2 * wpc} words per pair)")
    if not mem["stored"]:
        raise ValueError(f"{machine.name}: the machine does not store memory")
    b = machine.m["budget"]
    if b["unit"] != "steps" or not b["perLane"]:
        raise ValueError(f"{machine.name}: ZSoup needs a per-lane step budget (budget {b})")
    if "writeCounts" not in machine.fields:
        raise ValueError(f"{machine.name}: the record has no write counts")
    if machine.role("stateIn") is not None:
        raise ValueError(f"{machine.name}: per-lane initial state is not supported in a soup")
    return L, int(mem["ringBytes"])


def is_lethal_tar(machine: ZMachine) -> bool:
    """Whether the machine's world has Algocell's lethal-tar rule (Stage I, zero_halts): on the Z80, a zero byte fetched as
    the first byte of an instruction always halts the lane."""
    lethal = machine.world.get("halting", {}).get("lethal")
    return bool(machine.core["id"] == "z80" and lethal is not None and lethal.get("firstUnit") == 0 and lethal.get("probability", 1) >= 1)


def ablation_for(machine: ZMachine, masks: np.ndarray) -> np.ndarray | None:
    """The machine's ablation table for the soup's suppression masks: the masks themselves on a Z80 'coarse' table
    (Algocell's params.suppress layout, word for word); no table (all zero) when nothing is suppressed; otherwise an
    error, since Algocell's Z80 opcode masks mean nothing in another layout or on another ISA."""
    masks = np.asarray(masks, dtype=np.uint32)
    if not masks.any():
        return None
    abl = machine.m["ablation"]
    if machine.core["id"] == "z80" and abl["kind"] == "uniform" and abl.get("layout") == "coarse" and abl.get("words") == masks.size:
        return masks
    raise ValueError(f"{machine.name}: Algocell's Z80 suppression masks need a Z80 machine with the 'coarse' ablation table "
                     f"(this one: core {machine.core['id']}, ablation {abl.get('layout', abl['kind'])})")


def soup_source_for(machine: ZMachine) -> Path:
    """The exported simulation shader whose soup kernels ZSoup keeps: the lethal export for a lethal-tar machine (its
    soup kernels are the square ones; Soup(zero_halts=True) compiles this file), else the square export, for the
    machine's L and ring P."""
    L, P = check_pair_layout(machine)
    kind = "lethal" if is_lethal_tar(machine) else "square"
    suffix = "" if P == 2 * L else f"_P{P}"
    path = SHADER_DIR / f"sim_{kind}_L{L}{suffix}.wgsl"
    if not path.exists():
        raise ValueError(f"no exported simulation shader {path.name} for {machine.name} (L = {L}, P = {P})")
    return path


class ZBridge:
    """The soup kernels (verbatim) + the bridge kernels in one module with a 12-binding layout, the machine, and the
    machine's budget and record buffers. ZSoup runs on it; so does the step-locked differential test, which binds the
    original Soup's buffers with pair_data and write_counts replaced by copies."""

    def __init__(self, machine: ZMachine, source: Path | None = None, device: wgpu.GPUDevice | None = None) -> None:
        self.dev = device or machine.dev
        self.machine = machine
        self.L, self.P = check_pair_layout(machine)
        self.source = Path(source) if source else soup_source_for(machine)
        kept_text, self.kept, self.source_text = extract_soup_kernels(self.source)
        self.wgsl = kept_text + bridge_wgsl(machine)
        self.sha256 = hashlib.sha256(self.wgsl.encode()).hexdigest()
        module = self.dev.create_shader_module(code=self.wgsl)
        storage = {"type": wgpu.BufferBindingType.storage}
        uniform = {"type": wgpu.BufferBindingType.uniform}
        self.layout = self.dev.create_bind_group_layout(entries=[
            {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": (uniform if i == 5 else storage)}
            for i in range(12)
        ])
        pl = self.dev.create_pipeline_layout(bind_group_layouts=[self.layout])
        self.pipelines = {ep: self.dev.create_compute_pipeline(layout=pl, compute={"module": module, "entry_point": ep})
                          for ep in SOUP_KERNELS + BRIDGE_KERNELS}
        B = wgpu.BufferUsage
        self.budget_buf = self.dev.create_buffer(size=MAX_PAIRS * 4, usage=B.STORAGE | B.COPY_SRC | B.COPY_DST)
        self.record_buf = self.dev.create_buffer(size=MAX_PAIRS * machine.record_words * 4, usage=B.STORAGE | B.COPY_SRC | B.COPY_DST)

    def soup_bind_group(self, bufs: dict[int, wgpu.GPUBuffer]) -> wgpu.GPUBindGroup:
        """Bindings 0-9 from `bufs` (the soup's buffers by binding number), 10-11 the machine's budget and record."""
        full = {**bufs, BUDGET_BINDING: self.budget_buf, RECORD_BINDING: self.record_buf}
        return self.dev.create_bind_group(layout=self.layout, entries=[
            {"binding": i, "resource": {"buffer": full[i], "offset": 0, "size": full[i].size}} for i in range(12)])

    def machine_bind_group(self, params_buf: wgpu.GPUBuffer, pair_data_buf: wgpu.GPUBuffer) -> wgpu.GPUBindGroup:
        return self.machine.bind_group(params_buf, pair_data_buf, self.record_buf, self.budget_buf)

    def machine_params(self, pair_count: int, batch_seed: int, masks: np.ndarray) -> np.ndarray:
        """lane_count = pair_count, lane_base 0, seed = the step's batch seed, step base 0, ablation = the soup masks
        (see ablation_for)."""
        return self.machine.params(pair_count, 0, batch_seed, 0, ablation_for(self.machine, masks))

    def dispatch(self, enc: wgpu.GPUCommandEncoder, ep: str, bg: wgpu.GPUBindGroup, wg: int) -> None:
        if wg == 0:
            return
        p = enc.begin_compute_pass()
        p.set_pipeline(self.pipelines[ep])
        p.set_bind_group(0, bg)
        p.dispatch_workgroups(wg)
        p.end()

    def encode_execute(self, enc: wgpu.GPUCommandEncoder, soup_bg, machine_bg, pair_count: int) -> None:
        """What replaces z80_execute_batch: budgets from pair_active, the machine on pair_data, write counts back."""
        wg = -(-pair_count // 64)
        self.dispatch(enc, "zsoup_budget_from_active", soup_bg, wg)
        self.machine.encode(enc, machine_bg, pair_count)
        self.dispatch(enc, "zsoup_write_counts", soup_bg, wg)


class ZSoup(Soup):
    """The Soup with execution delegated to a Zilion machine; the public API (reset, step, read_soup, write_soup,
    read_interactions, read_pair_memory, read_byte_counts, read_hashes, read_active_pairs, read_pairs, set_suppress,
    sync) is Soup's own code, inherited. Only the build and the step differ."""

    def __init__(
        self,
        machine: str | ZMachine = "z80c02_pair_L16",
        width: int = 160,
        height: int = 125,
        seed: int = 6,
        pair_count: int = 8192,
        z80_steps: int = 128,
        noise_exp: int = 4,
        suppress: str | list[str] | None = None,
        ring: int = 32,
        device: wgpu.GPUDevice | None = None,
        mutations_per_step: int | None = None,
        grid: str = "square",
        machines_dir: Path | str = MACHINES_DIR,
    ) -> None:
        if grid != "square":
            raise ValueError(f"ZSoup keeps the square-lattice soup kernels only (grid={grid!r})")
        assert 1 <= pair_count <= MAX_PAIRS
        self.device = device or get_device()
        self.zmachine = machine if isinstance(machine, ZMachine) else load_machine(machine, machines_dir, self.device)
        L, P = check_pair_layout(self.zmachine)
        self.width, self.height, self.grid = width, height, grid
        self.tape_length, self.mem_length = L, P
        self.zero_halts = is_lethal_tar(self.zmachine)
        self.pair_length = 2 * L
        self.words_per_cell = (L + 3) // 4
        self.cell_count = width * height
        self.pair_count = pair_count
        self.z80_steps = z80_steps   # the per-encounter step budget (Params.z80_steps), whatever the ISA
        self.noise_exp = noise_exp
        self.mutations_per_step = mutations_per_step
        self.ring = ring
        self.patterns = parse_patterns(suppress) if isinstance(suppress, str) else list(suppress or [])
        self.sets = resolve(self.patterns)
        self.masks = make_masks(self.sets)
        ablation_for(self.zmachine, self.masks)   # refuses masks the machine cannot take
        self.batch_index = 0
        self._build()
        self.reset(seed)

    def set_suppress(self, patterns: str | list[str] | None) -> None:
        """Soup.set_suppress, refusing (and leaving the soup unchanged) a set the machine cannot take."""
        new = parse_patterns(patterns) if isinstance(patterns, str) else list(patterns or [])
        sets = resolve(new)
        masks = make_masks(sets)
        ablation_for(self.zmachine, masks)
        self.patterns, self.sets, self.masks = new, sets, masks

    @property
    def machine_provenance(self) -> dict:
        return {**self.zmachine.describe(), "soup_kernels_from": self.shader_file.name,
                "soup_source_sha256": hashlib.sha256(self.bridge.source_text.encode()).hexdigest(),
                "zsoup_shader_sha256": self.bridge.sha256}

    def _build(self) -> None:
        dev = self.device
        B = wgpu.BufferUsage
        self.bridge = ZBridge(self.zmachine, device=dev)
        # As for Soup: shader_file and the hash of that file. The soup kernels are compiled from it verbatim; the shader
        # actually compiled (kept items + bridge) and the machine are in machine_provenance.
        self.shader_file = self.bridge.source
        self.shader_sha256_16 = hashlib.sha256(self.bridge.source_text.encode()).hexdigest()[:16]
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
        self.zparams_bufs = [dev.create_buffer(size=self.zmachine.params_bytes, usage=B.UNIFORM | B.COPY_DST) for _ in range(self.ring)]
        self.pipelines = self.bridge.pipelines
        self.bind_groups = [self.bridge.soup_bind_group(self.soup_buffers(k)) for k in range(self.ring)]
        self.machine_bind_groups = [self.bridge.machine_bind_group(self.zparams_bufs[k], self.pair_data_buf) for k in range(self.ring)]

    def soup_buffers(self, k: int = 0) -> dict[int, wgpu.GPUBuffer]:
        """The soup's buffers by binding number (as Soup binds them), with params ring slot k."""
        return {0: self.soup_buf, 1: self.pairs_buf, 2: self.pair_data_buf, 3: self.write_counts_buf, 4: self.rng_states_buf,
                5: self.params_bufs[k], 6: self.pair_active_buf, 7: self.byte_counts_buf, 8: self.collision_buf, 9: self.hash_buf}

    def step(self, n: int = 1) -> None:
        """Run n simulation steps (each = clear → prepare → [budget → machine → write counts] → absorb → mutate)."""
        q = self.device.queue
        pairs_wg = -(-self.pair_count // 64)
        cells_wg = -(-self.cell_count // 256)
        mut_wg = -(-self.mutation_count // 64)
        br = self.bridge
        done = 0
        while done < n:
            k_steps = min(self.ring, n - done)
            for k in range(k_steps):
                batch_seed = self.cpu_rng.next_u32()
                q.write_buffer(self.params_bufs[k], 0, self._params(batch_seed))
                q.write_buffer(self.zparams_bufs[k], 0, br.machine_params(self.pair_count, batch_seed, self.masks))
            enc = self.device.create_command_encoder()
            for k in range(k_steps):
                bg = self.bind_groups[k]
                br.dispatch(enc, "clear_collision", bg, cells_wg)
                br.dispatch(enc, "prepare_batch", bg, pairs_wg)
                br.encode_execute(enc, bg, self.machine_bind_groups[k], self.pair_count)
                br.dispatch(enc, "absorb_results", bg, pairs_wg)
                br.dispatch(enc, "mutate_soup", bg, mut_wg)
            q.submit([enc.finish()])
            done += k_steps
            self.batch_index += k_steps

    def read_records(self) -> np.ndarray:
        """The machine's lane records of the LAST executed step, (pair_count, record words); decode with
        self.zmachine.decode (steps, halt reason, trap cause, state, write counts per segment)."""
        raw = np.frombuffer(self.device.queue.read_buffer(self.bridge.record_buf), dtype=np.uint32)
        return raw[: self.pair_count * self.zmachine.record_words].reshape(self.pair_count, -1).copy()


__all__ = ["ZSoup", "ZBridge", "extract_soup_kernels", "parse_wgsl_items", "bridge_wgsl", "check_pair_layout", "soup_source_for", "ablation_for", "is_lethal_tar"]
