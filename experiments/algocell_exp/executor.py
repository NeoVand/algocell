"""One executor interface for the assays: run (A, B) pairs, get back memory, write counts, fetch and start maps, steps and
halt reasons.

    from algocell_exp import executor as E
    ex = E.get("z80c02_assay")                       # a vendored Zilion machine set (or one machine by its full name)
    r = ex.run(pairs, L=16, steps=128, suppress=(), zero_halts=False, fetch_map=True, start_map=True)
    r.memory, r.writes, r.fetch_map, r.start_map, r.steps, r.halt_reason, r.executor

    E.select("z80c02_assay")                         # process-wide: every assay call below runs on the machine
    with E.using("m6502_assay"): ...                 # or for one block

Two implementations:

(a) `AlgocellShader`, the default: Algocell's own executor shaders (`shader/z80_test[_lethal]_L<L>[_P<P>].wgsl` and the
    traced and op-traced executors derived from them), run by the host functions the frozen assays have always called:
    `assay.execute_pairs`, `exectrace.execute_pairs_traced` and `optrace.execute_pairs_optrace`. Their code is unchanged;
    each gained only a dispatch at the top (to the selected machine, when one is selected) and an optional read-back of
    the register words it already wrote (write counts are words 10 and 11). The shaders record no step count and no
    halt reason, so `steps` and `halt_reason` are None on this path.
(b) `ZilionExecutor`: a vendored Zilion machine (`machines/<name>.wgsl` + `.manifest.json`) through `zmachine.py`. A
    machine set is a name prefix: the request (L, ring P, rule) resolves to `<prefix>[_lethal]_L<L>[_P<P>]`, and the
    machine's manifest is checked against the request (segments A and B of L bytes, ring P, the lethal rule exactly when
    `zero_halts`, a step budget, the maps when asked for, and, on the Z80, Algocell's stack convention). A full machine
    name is checked the same way. The Z80 suppression patterns become the machine's ablation table only on a Z80
    `coarse` table (Algocell's `params.suppress` words); any other machine refuses a non-empty suppression set.

Nothing is selected by default, so every frozen script runs path (a) exactly as before unless it is given `--machine`.
While a machine is selected, the shader pipelines refuse to compile (`shader_guard`), so a run that asked for a machine
cannot silently use the shaders. Every call is counted per backend (`usage()`); with ALGOCELL_EXECUTOR_LOG set, the
counts are written to that file at exit (the regression report uses them to show which backend ran).
"""

from __future__ import annotations

import atexit
import json
import os
import re
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

DEFAULT = None             # the default executor: Algocell's shaders (path a)
_ACTIVE: "ZilionExecutor | None" = None
_SHADER_OK = 0             # > 0 while AlgocellShader.run executes explicitly (allowed even with a machine selected)
_USAGE: Counter = Counter()


# ── results ─────────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class PairRun:
    """Outputs of one run of N pairs. A field the backend does not produce is None."""

    memory: np.ndarray                    # (N, 2L) uint8: tape A then tape B after the encounter
    writes: np.ndarray | None             # (N, 2) uint32: writes into A, into B (writes into ring padding count as B's)
    fetch_map: np.ndarray | None          # (N, P) bool: ring positions fetched as instruction stream
    start_map: np.ndarray | None          # (N, P) bool: ring positions where an instruction started
    steps: np.ndarray | None              # (N,) instructions executed (path b only)
    halt_reason: np.ndarray | None        # (N,) Zilion haltReasons: 0 budget, 1 halt instruction, 2 host (lethal), 3 trap
    trap_cause: np.ndarray | None         # (N,) Zilion trapCauses code of the trap that halted the lane, else 0
    executor: str                         # the shader file or the machine that ran


def pack_fetch_masks(fetch_map: np.ndarray) -> np.ndarray:
    """(N, P) bool → (N, 8) uint32, the traced shader's layout (bit a of the 256-bit mask = ring position a)."""
    n, P = fetch_map.shape
    if P > 256:
        raise ValueError(f"a ring of {P} bytes does not fit the traced executor's 256-bit mask")
    bits = np.zeros((n, 256), dtype=np.uint8)
    bits[:, :P] = fetch_map
    return np.ascontiguousarray(np.packbits(bits, axis=1, bitorder="little")).view("<u4").astype(np.uint32)


# ── (a) Algocell's shaders ──────────────────────────────────────────────────────────────────────────────────────────
class AlgocellShader:
    """Path (a): Algocell's executor shaders through the frozen host functions."""

    name = "algocell-shaders"
    isa = "z80"
    ablation = "z80-coarse"

    def file(self, L: int, mem_length: int | None = None, zero_halts: bool = False) -> str:
        from . import assay

        return assay._executor_path(L, mem_length, bool(zero_halts)).name

    def run(self, pairs: np.ndarray, L: int, steps: int, suppress=(), *, mem_length: int | None = None, zero_halts: bool = False,
            fetch_map: bool = False, start_map: bool = False) -> PairRun:
        global _SHADER_OK
        from . import assay, exectrace, optrace

        pairs = np.ascontiguousarray(pairs, dtype=np.uint8)
        P = int(mem_length) if mem_length else 2 * L
        f = s = None
        _SHADER_OK += 1
        try:
            if start_map:
                if list(suppress) or (mem_length and int(mem_length) != 2 * L):
                    raise NotImplementedError("the op-traced shader (start map) runs unsuppressed on the ring 2L only")
                mem, f, s, regs = optrace._shader_execute_pairs_optrace(pairs, L, steps, zero_halts, return_regs=True)
                name = "optrace(" + self.file(L, None, zero_halts) + ")"
            elif fetch_map:
                mem, masks, regs = exectrace._shader_execute_pairs_traced(pairs, L, steps, suppress, mem_length, zero_halts, return_regs=True)
                f = exectrace.exec_positions(masks, P)
                name = "trace(" + self.file(L, mem_length, zero_halts) + ")"
            else:
                mem, regs = assay._shader_execute_pairs(pairs, L, steps, suppress, mem_length, zero_halts, return_regs=True)
                name = self.file(L, mem_length, zero_halts)
        finally:
            _SHADER_OK -= 1
        _USAGE[("shader", name)] += len(pairs)
        return PairRun(memory=mem, writes=regs[:, 10:12].copy(), fetch_map=f, start_map=s, steps=None, halt_reason=None,
                       trap_cause=None, executor=name)


# ── (b) a vendored Zilion machine ───────────────────────────────────────────────────────────────────────────────────
class ZilionExecutor:
    """Path (b): a vendored Zilion machine, or a set of them (a name prefix), through zmachine.py.

    `budget_delta` adds to every budget; it exists for negative controls only (a machine one instruction short must
    not reproduce the shaders) and is 0 for every real use."""

    def __init__(self, spec: str, machines_dir=None, budget_delta: int = 0) -> None:
        from . import zmachine

        self.spec = spec
        self.machines_dir = machines_dir or zmachine.MACHINES_DIR
        self.budget_delta = int(budget_delta)
        names = zmachine.available(self.machines_dir)
        self.exact = spec in names
        # a set is the machines named <spec>[_lethal]_L<L>[_P<P>], nothing else that merely shares the prefix
        pat = re.compile(re.escape(spec) + r"(_lethal)?_L\d+(_P\d+)?")
        self.members = [spec] if self.exact else sorted(n for n in names if pat.fullmatch(n))
        if not self.members:
            raise ValueError(f"no vendored machine or machine set {spec!r} in {self.machines_dir} (have: {', '.join(names)})")
        cores = set()
        for n in self.members:
            c = json.loads(zmachine.machine_paths(n, self.machines_dir)[1].read_text())["core"]
            cores.add((c["id"], c["compat"], c["variant"], c["coreVersion"]))
        if len(cores) != 1:
            raise ValueError(f"machine set {spec!r} mixes cores {sorted(cores)}; a set must run one core")
        self._cache: dict = {}
        self.name = spec

    # description of the set: its (single) core
    def _any_manifest(self) -> dict:
        from . import zmachine

        _, man = zmachine.machine_paths(self.members[0], self.machines_dir)
        return json.loads(man.read_text())

    @property
    def isa(self) -> str:
        return self._any_manifest()["core"]["id"]

    @property
    def ablation(self) -> str:
        m = self._any_manifest()
        a = m["ablation"]
        return f"{m['core']['id']}-{a.get('layout')}" if a["kind"] == "uniform" else a["kind"]

    def machine_name(self, L: int, P: int, zero_halts: bool) -> str:
        if self.exact:
            return self.spec
        return f"{self.spec}{'_lethal' if zero_halts else ''}_L{L}{'' if P == 2 * L else f'_P{P}'}"

    def machine_for(self, L: int, mem_length: int | None = None, zero_halts: bool = False, maps: bool = False):
        """The machine for this request, its manifest checked against it (raises with the reason otherwise)."""
        from . import zmachine

        P = int(mem_length) if mem_length else 2 * L
        name = self.machine_name(L, P, bool(zero_halts))
        key = (name, maps)
        if key in self._cache:
            return self._cache[key]
        if name not in self.members:
            raise ValueError(f"machine set {self.spec!r} has no machine for L = {L}, ring {P}, {'lethal tar' if zero_halts else 'plain rule'} "
                             f"(expected {name!r}); vendor it (machines/worlds_*.json) and prove it against the shader first")
        m = zmachine.load_machine(name, self.machines_dir)
        man, world = m.m, m.m["world"]
        why = []
        segs = [(s["name"], int(s["bytes"])) for s in m.segments]
        if segs != [("A", L), ("B", L)]:
            why.append(f"segments {segs}, not A and B of {L} bytes")
        if m.ring_bytes != P:
            why.append(f"ring {m.ring_bytes}, not {P}")
        lethal = (world.get("halting") or {}).get("lethal")
        if bool(zero_halts) != (lethal is not None) or (lethal is not None and lethal != {"firstUnit": 0}):
            why.append(f"halting.lethal {lethal!r} does not match zero_halts={bool(zero_halts)} (Algocell's rule is {{'firstUnit': 0}})")
        if set(world.get("halting") or {}) - {"lethal"}:
            why.append(f"halting rules {sorted(set(world['halting']) - {'lethal'})} that no Algocell executor has")
        if world.get("hooks"):
            why.append("raw hooks (the dial worlds are not covered here)")
        if (world.get("entry") or {"position": 0}) != {"position": 0}:
            why.append(f"entry {world['entry']!r}: Algocell's encounter starts at position 0")
        init = world.get("init") or {}
        if init.get("random") or init.get("state") == "perLane":
            why.append("random or per-lane initial state (the randreg/randsp conventions are not covered)")
        if man["core"]["id"] == "z80" and init != {"stack": {"topAliases": "lastByte"}}:
            why.append(f"Z80 init {init!r}, not Algocell's (registers zero, SP aliasing the last byte: {{'stack': {{'topAliases': 'lastByte'}}}})")
        if world["budget"]["unit"] != "steps":
            why.append(f"budget unit {world['budget']['unit']!r}, not instructions ('steps')")
        if not m.budget_per_lane and "budget" not in {f["role"] for f in man["params"]["fields"]}:
            why.append("a budget fixed in the shader (the caller's budget would be ignored)")
        if not man["memory"]["stored"]:
            why.append("memory is not stored")
        if maps and (m.fields.get("fetchMap") is None or m.fields.get("startMap") is None):
            why.append("no fetch and start maps in the record (a traced assay needs them)")
        if why:
            raise ValueError(f"machine {name!r} does not fit the request (L = {L}, ring {P}, zero_halts={bool(zero_halts)}): " + "; ".join(why))
        self._cache[key] = m
        return m

    def file(self, L: int, mem_length: int | None = None, zero_halts: bool = False) -> str:
        P = int(mem_length) if mem_length else 2 * L
        return self.machine_name(L, P, bool(zero_halts))

    def ablation_words(self, m, suppress) -> np.ndarray | None:
        pats = list(suppress or ())
        if not pats:
            return None
        a = m.m["ablation"]
        if not (m.core["id"] == "z80" and a["kind"] == "uniform" and a.get("layout") == "coarse"):
            raise ValueError(f"{m.name}: suppression {pats!r} needs a Z80 'coarse' ablation table (Algocell's masks are Z80 opcode pages); "
                             f"this machine has {a['kind']}{' ' + a['layout'] if a.get('layout') else ''} ({m.core['id']})")
        from .isa import masks, resolve

        return masks(resolve(pats))

    def run(self, pairs: np.ndarray, L: int, steps: int, suppress=(), *, mem_length: int | None = None, zero_halts: bool = False,
            fetch_map: bool = False, start_map: bool = False) -> PairRun:
        pairs = np.ascontiguousarray(pairs, dtype=np.uint8)
        if pairs.ndim != 2 or pairs.shape[1] != 2 * L:
            raise ValueError(f"pairs of shape {pairs.shape}, expected (N, {2 * L})")
        m = self.machine_for(L, mem_length, zero_halts, maps=fetch_map or start_map)
        abl = self.ablation_words(m, suppress)
        r = m.run(pairs, budget=int(steps) + self.budget_delta, ablation=abl)
        writes = None
        if r.write_counts is not None:
            names = r.write_count_names
            wb = r.writes("B") + (r.writes("padding") if "padding" in names else 0)
            writes = np.stack([r.writes("A"), wb], axis=1).astype(np.uint32)
        _USAGE[("machine", m.name + (f" (budget {self.budget_delta:+d})" if self.budget_delta else ""))] += len(pairs)
        return PairRun(memory=r.memory, writes=writes, fetch_map=r.fetch_map if fetch_map else None, start_map=r.start_map if start_map else None,
                       steps=r.steps, halt_reason=r.halt_reason, trap_cause=r.trap_cause, executor=m.name)

    def describe(self) -> dict:
        """Provenance of every machine of the set that this process used."""
        return {name: m.describe() for (name, _), m in self._cache.items()}


# ── selection ───────────────────────────────────────────────────────────────────────────────────────────────────────
def get(spec: str | None):
    """The executor for `spec`: None or 'algocell' → Algocell's shaders; anything else → a vendored machine (set)."""
    if spec is None or spec in ("", "algocell", "algocell-shaders"):
        return AlgocellShader()
    return ZilionExecutor(spec)


def active() -> "ZilionExecutor | None":
    """The selected machine executor, or None when the assays run on Algocell's shaders (the default)."""
    return _ACTIVE


def current():
    """The executor the assays use now (an object with .run, .file, .isa, .name)."""
    return _ACTIVE if _ACTIVE is not None else AlgocellShader()


def select(spec) -> None:
    """Select a machine (set) for every assay call in this process; None returns to Algocell's shaders."""
    global _ACTIVE
    if spec is None or isinstance(spec, str) and spec in ("", "algocell", "algocell-shaders"):
        _ACTIVE = None
    elif isinstance(spec, ZilionExecutor):
        _ACTIVE = spec
    else:
        _ACTIVE = ZilionExecutor(spec)


@contextmanager
def using(spec):
    """Run a block on a machine (set), then restore the previous selection."""
    global _ACTIVE
    prev = _ACTIVE
    select(spec)
    try:
        yield _ACTIVE
    finally:
        _ACTIVE = prev


def shader_guard(what: str) -> None:
    """Called by the shader pipelines: refuse while a machine is selected, unless AlgocellShader.run asked."""
    if _ACTIVE is not None and _SHADER_OK == 0:
        raise RuntimeError(f"{what}: Algocell's executor shader was requested while machine {_ACTIVE.spec!r} is selected")


def count_shader(name: str, n: int) -> None:
    _USAGE[("shader", name)] += int(n)


def usage() -> dict:
    """Pairs executed per backend in this process: {'shader': {file: n}, 'machine': {name: n}}."""
    out: dict = {"shader": {}, "machine": {}}
    for (kind, name), n in sorted(_USAGE.items()):
        out[kind][name] = n
    return out


def _write_log() -> None:
    path = os.environ.get("ALGOCELL_EXECUTOR_LOG")
    if not path:
        return
    rec = {"usage": usage(), "selected": None if _ACTIVE is None else _ACTIVE.spec}
    if _ACTIVE is not None:
        rec["machines"] = {k: {kk: v[kk] for kk in ("machine", "core", "sha256_wgsl", "ring", "halting")} for k, v in _ACTIVE.describe().items()}
    with open(path, "w") as f:
        json.dump(rec, f, indent=1, default=str)


atexit.register(_write_log)


# ── command-line helpers for the frozen scripts ─────────────────────────────────────────────────────────────────────
MACHINE_HELP = ("run every encounter on this vendored Zilion machine set (e.g. z80c02_assay) or machine instead of Algocell's "
                "executor shaders (default: the shaders, as registered)")


def add_argument(ap) -> None:
    ap.add_argument("--machine", default=None, help=MACHINE_HELP)


def from_args(a, isas=("z80",), why: str = "") -> str:
    """Select the machine of `--machine` (if any). `isas`: the ISAs the script's own logic supports (its inputs and
    labels are Z80 for every frozen script); a machine of another ISA is refused with `why`. Returns a description."""
    spec = getattr(a, "machine", None)
    if spec is None:
        return "algocell shaders"
    ex = ZilionExecutor(spec)
    if isas is not None and ex.isa not in isas:
        raise SystemExit(f"--machine {spec}: this script runs {', '.join(isas)} machines only ({ex.isa} given){': ' + why if why else ''}")
    select(ex)
    return f"machine {spec}"
