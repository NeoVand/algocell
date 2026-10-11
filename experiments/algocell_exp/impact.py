"""Emulator impact study (planning/IMPACT_STUDY_DRAFT.md): one runner for the three cores, the affected-pair rate and
the attribution of every affected pair to the instruction that caused it.

Cores (machines/worlds_impact.json, see impact_worlds.py): C0 = Zilion compat '0.2' (the 0.2.0 core of every registered
result), C1 = compat '0.3' classic, C2 = compat '0.3' zilog-nmos. The three machines of a world share its world spec.

Affected pair. A pair is affected under core x when its final pair memory (both tapes) differs from C0's.

Attribution. For an affected pair, both cores are stepped one instruction at a time (resumable machines, budget 1 per
dispatch, the record and memory carried over), from the same initial state:

1. Repaired pass. Before every step, x's common state (the 25 state fields of the 0.2.0 record, the memory, the ring
   padding, the step count, write counts and halt fields) is set to C0's; x keeps only its own internal latches (WZ, IM,
   and with zilog-nmos Q), which C0 does not have. A step after which the two differ is an *event*: one instruction,
   executed from identical common states, gave different results. Each event is classed from the instruction bytes at
   PC and the fields that differ (`classify`); the expected difference of each documented class is checked exactly.
   Because every step starts from repaired state, every deviation along the run is seen as an event of its own: one that
   fits no documented class is "other", which is a bug and stops the study (`AttributionError`).
2. Probes. For each event at step t, x is resumed from the repaired state before t and run unrepaired to the end. The
   probe's final memory differs from C0's exactly when that event, with everything its divergence led to, changes the
   final memory. The *decisive event* is the latest event whose probe differs: from the state just after it, x
   reproduces C0's final memory, and from the state just before it, it does not. The pair is attributed to the class of
   that event; the classes of all events whose probes differ are kept as well.
3. Checks (each failure raises AttributionError): stepping C0 reproduces C0's single-dispatch final memory and registers;
   an affected pair has at least one event; the probe of the first event (nothing repaired before it) reproduces x's
   single-dispatch final memory and registers; at least one probe differs.

Classes (`classify`): `wasted-prefix` (a DD/FD prefix followed by DD, FD or ED; only R differs; this is how R reaches
memory, through LD A,R), `bit-hl` (unprefixed CB 46+8n, BIT n,(HL); only F3/F5 differ), `scf-ccf` (SCF or CCF, optionally
behind DD/FD; only F3/F5 differ; C2 only), `block-repeat` (ED B0/B1/B8/B9 or the I/O repeats ED B2/B3/BA/BB, an iteration
that repeats; only F3/F5, and for the I/O repeats H and P/V, differ), and `other`.

Worlds whose hooks declare private state (the dial and randreg/randsp PCG streams) are not resumable; they are stepped
through a twin: the plain world of the same ring and core (impact_worlds.World.resumable_twin). randreg/randsp: the
twin is resumed from the real machine's record after reset (budget 0), which holds the drawn registers; the hooks act at
reset only. Dial: the twin is stepped with the dial's halting decision made on the CPU before every step, from
Algocell's PCG stream (gen_dial_shader.py: seeded per lane, one draw per instruction whose first byte is zero). Check 3
then also proves the twin: it must reproduce the real machines' outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .impact_worlds import CORES, World, world as get_world
from .zmachine import ZMachine, ZResult, load_machine

COMMON = ("a", "f", "b", "c", "d", "e", "h", "l", "a_", "f_", "b_", "c_", "d_", "e_", "h_", "l_", "ix", "iy", "sp", "pc", "i", "r",
          "iff1", "iff2", "halted")
HIDDEN = {"c0": (), "c1": ("wz", "im"), "c2": ("wz", "im", "q")}
REGS10 = ("a", "f", "b", "c", "d", "e", "h", "l", "sp", "pc")       # Algocell's executor register words 0-9
CLASSES = ("wasted-prefix", "bit-hl", "scf-ccf", "block-repeat", "other")
BLOCK_MEM = (0xB0, 0xB1, 0xB8, 0xB9)                                  # LDIR CPIR LDDR CPDR
BLOCK_IO = (0xB2, 0xB3, 0xBA, 0xBB)                                   # INIR OTIR INDR OTDR


class AttributionError(RuntimeError):
    """An 'other' event or a failed consistency check: a bug, which stops the study."""


# ── statistics ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta (modified Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            return h
    raise ArithmeticError("incomplete beta did not converge")


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def _beta_ppf(q: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper–Pearson) two-sided interval for k successes in n trials."""
    if n <= 0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - conf
    lo = 0.0 if k == 0 else _beta_ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta_ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


# ── the three machines of a world ──────────────────────────────────────────────────────────────────────────────────
class Rec:
    """Named access to a machine's lane records (n, words)."""

    def __init__(self, m: ZMachine) -> None:
        self.m = m
        f = m.fields
        self.state_off = f["state"]["offset"]
        self.state_names = tuple(f["state"]["items"])
        self.off = {n: self.state_off + k for k, n in enumerate(self.state_names)}
        for n in ("steps", "haltReason", "trapCause"):
            self.off[n] = f[n]["offset"]
        self.wc = (f["writeCounts"]["offset"], f["writeCounts"]["words"]) if "writeCounts" in f else None
        self.pad = (f["padding"]["offset"], f["padding"]["words"]) if "padding" in f else None

    def get(self, rec: np.ndarray, name: str) -> np.ndarray:
        return rec[:, self.off[name]]

    def common(self, rec: np.ndarray) -> dict[str, np.ndarray]:
        """Every record part C0 and the fixed cores share (not the internal latches)."""
        out = {n: rec[:, self.off[n]] for n in ("steps", "haltReason", "trapCause") + COMMON}
        if self.wc:
            out["writeCounts"] = rec[:, self.wc[0]: self.wc[0] + self.wc[1]]
        if self.pad:
            out["padding"] = rec[:, self.pad[0]: self.pad[0] + self.pad[1]]
        return out

    def set_common(self, rec: np.ndarray, src: "Rec", src_rec: np.ndarray, rows=None) -> np.ndarray:
        """rec with every common part copied from src_rec (another machine's records, mapped by name)."""
        out = rec.copy()
        rows = slice(None) if rows is None else rows
        for n in ("steps", "haltReason", "trapCause") + COMMON:
            out[rows, self.off[n]] = src_rec[rows, src.off[n]]
        if self.wc:
            assert src.wc[1] == self.wc[1]
            out[rows, self.wc[0]: self.wc[0] + self.wc[1]] = src_rec[rows, src.wc[0]: src.wc[0] + src.wc[1]]
        if self.pad:
            assert src.pad and src.pad[1] == self.pad[1]
            out[rows, self.pad[0]: self.pad[0] + self.pad[1]] = src_rec[rows, src.pad[0]: src.pad[0] + src.pad[1]]
        return out

    def padding_bytes(self, rec: np.ndarray, nbytes: int) -> np.ndarray:
        if nbytes == 0:
            return np.zeros((rec.shape[0], 0), dtype=np.uint8)
        assert self.pad, "a padded world's record must carry its padding (resumable machines do)"
        w = np.ascontiguousarray(rec[:, self.pad[0]: self.pad[0] + self.pad[1]], dtype=np.uint32)
        return w.view(np.uint8).reshape(rec.shape[0], -1)[:, :nbytes]


CORE_DEPENDENT = {"core", "record", "state"}   # manifest parts a core selection may change (plus the checks below)


def manifest_core_diff(a: dict, b: dict) -> set[str]:
    """Manifest parts of two machines that differ beyond what their core selection may change. Allowed to differ: the
    core description, the record and state layouts (the fixed cores append WZ, IM and Q), the record binding's size, the
    init's state word count, and the core/WGSL/options hashes. Everything else (world spec and hash, memory, params,
    ablation, budget, init conventions, halt reasons, trap causes, host contract, entry point, mode, resumability, schema,
    workgroup size, generator) must be equal."""
    bad = set()
    for k in set(a) | set(b):
        if k in CORE_DEPENDENT:
            continue
        va, vb = a.get(k), b.get(k)
        if k == "sha256":
            if va.get("world") != vb.get("world"):
                bad.add("sha256.world")
            continue
        if k == "init":
            va, vb = {**va, "stateWords": None}, {**vb, "stateWords": None}
        if k == "bindings":
            strip = lambda bs: [{**x, "wordsPerLane": None} if x.get("role") == "record" else x for x in bs]  # noqa: E731
            va, vb = strip(va), strip(vb)
        if va != vb:
            bad.add(k)
    return bad


class Triple:
    """The C0, C1, C2 machines of one world, checked to differ only in their core."""

    def __init__(self, w: World | str, device=None) -> None:
        self.w = w = get_world(w) if isinstance(w, str) else w
        self.m = {c: load_machine(w.machine(c), device=device) for c in CORES}
        self.rec = {c: Rec(m) for c, m in self.m.items()}
        m0 = self.m["c0"]
        for c, m in self.m.items():
            mf, mf0 = m.m, m0.m
            assert mf["sha256"]["world"] == mf0["sha256"]["world"], f"{w.tag}: world SHA-256 differs between c0 and {c}"
            assert mf["world"] == mf0["world"], f"{w.tag}: world spec differs between c0 and {c}"
            assert mf["core"]["hooks"] == mf0["core"]["hooks"], f"{w.tag}: hooks differ between c0 and {c}"
            assert manifest_core_diff(mf0, mf) == set(), (w.tag, c, manifest_core_diff(mf0, mf))
            sel = CORES[c]
            assert mf["core"]["compat"] == sel["compat"] and mf["core"]["variant"] == sel.get("variant", "classic"), c
            names = self.rec[c].state_names
            assert names[: len(COMMON)] == COMMON and names[len(COMMON):] == HIDDEN[c], (c, names)
        self.L, self.P = w.L, w.P
        assert m0.ring_bytes == self.P and m0.host_bytes == 2 * self.L
        self._twin = None

    def run(self, pairs: np.ndarray, budget, masks=None, seed: int = 0, lane_base: int = 0) -> dict[str, ZResult]:
        return {c: m.run(pairs, budget=budget, ablation=masks, seed=seed, lane_base=lane_base) for c, m in self.m.items()}

    @property
    def twin(self) -> "Triple":
        if self.w.resumable_twin is None:
            return self
        if self._twin is None:
            self._twin = Triple(self.w.resumable_twin)
            assert (self._twin.L, self._twin.P) == (self.L, self.P)
        return self._twin


def affected(res: dict[str, ZResult]) -> dict[str, np.ndarray]:
    """Per core x: pairs whose final pair memory differs from C0's."""
    m0 = res["c0"].memory
    return {c: (r.memory != m0).any(axis=1) for c, r in res.items() if c != "c0"}


# ── Algocell's dial stream on the CPU (gen_dial_shader.py) ─────────────────────────────────────────────────────────
M32 = 0xFFFFFFFF


def pcg_draw(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(new state, output) of gen_dial_shader's lethal_draw on uint64 arrays holding u32 values."""
    s = state & M32
    new = (s * 747796405 + 2891336453) & M32
    w = (((s >> ((s >> 28) + 4)) ^ s) * 277803737) & M32
    return new, ((w >> 22) ^ w) & M32


def dial_init(seed: int, lanes: np.ndarray) -> np.ndarray:
    s = ((seed & M32) * 2654435761 + (lanes.astype(np.uint64) & M32) * 2246822519 + 0x165667B1) & M32
    s = s.astype(np.uint64)
    s, _ = pcg_draw(s)
    return s


# ── classification ─────────────────────────────────────────────────────────────────────────────────────────────────
def ring_of(tr: Triple, rec_view: Rec, rec: np.ndarray, memory: np.ndarray) -> np.ndarray:
    """(n, P) ring bytes: tape A, tape B, then the padding bytes (resumable padded worlds carry them in the record)."""
    pad = tr.P - 2 * tr.L
    return np.concatenate([memory, rec_view.padding_bytes(rec, pad)], axis=1) if pad else memory


def insn_bytes(ring: np.ndarray, pc: int, P: int, k: int = 4) -> tuple[int, ...]:
    return tuple(int(ring[((pc + i) & 0xFFFF) % P]) for i in range(k))


def classify(core: str, b: tuple[int, ...], diff: dict[str, int], pc_before: int, pc_after: int) -> str:
    """Class of an event: the instruction (bytes b at PC, executed from identical common states) and the fields whose values
    differ after it (name → XOR of the two values; 'memory' for any memory byte). Any mismatch with the documented
    pattern of the instruction's class is 'other'."""
    names = set(diff)
    b0, b1 = b[0], b[1]
    if b0 in (0xDD, 0xFD) and b1 in (0xDD, 0xFD, 0xED):
        return "wasted-prefix" if names == {"r"} and pc_after == ((pc_before + 1) & 0xFFFF) else "other"
    f_only = names == {"f"}
    fx = diff.get("f", 0)
    if b0 == 0xCB and (b1 & 0xC7) == 0x46:
        return "bit-hl" if f_only and (fx & ~0x28) == 0 else "other"
    op = b1 if b0 in (0xDD, 0xFD) else b0
    if op in (0x37, 0x3F):
        return "scf-ccf" if core == "c2" and f_only and (fx & ~0x28) == 0 else "other"
    if b0 == 0xED and b1 in BLOCK_MEM + BLOCK_IO:
        allowed = 0x28 if b1 in BLOCK_MEM else (0x28 | 0x10 | 0x04)
        repeats = pc_after == pc_before
        return "block-repeat" if repeats and f_only and (fx & ~allowed) == 0 else "other"
    return "other"


# ── attribution ────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Event:
    lane: int          # index into the batch
    step: int          # 1-based step at which the event happened
    cls: str
    pc: int
    insn: str          # the first four bytes at PC, hex
    diff: str          # differing fields (f^x for flags)
    probe_differs: bool | None = None


@dataclass
class LaneResult:
    core: str
    decisive_class: str | None = None
    decisive_step: int | None = None
    propagating: tuple = ()
    events: list = field(default_factory=list)


def _diff_fields(rv0: Rec, r0: np.ndarray, m0: np.ndarray, rvx: Rec, rx: np.ndarray, mx: np.ndarray) -> list[dict]:
    """Per lane: {field: xor} of every common part that differs (memory and padding by byte count)."""
    c0, cx = rv0.common(r0), rvx.common(rx)
    n = r0.shape[0]
    out = [dict() for _ in range(n)]
    for k, v0 in c0.items():
        vx = cx[k]
        if v0.ndim == 1:
            idx = np.nonzero(v0 != vx)[0]
            for i in idx:
                out[i][k] = int(v0[i] ^ vx[i])
        else:
            idx = np.nonzero((v0 != vx).any(axis=1))[0]
            for i in idx:
                out[i][k] = int((v0[i] != vx[i]).sum())
    idx = np.nonzero((m0 != mx).any(axis=1))[0]
    for i in idx:
        out[i]["memory"] = int((m0[i] != mx[i]).sum())
    return out


def _fmt_diff(d: dict) -> str:
    return " ".join(f"{k}^{v:02x}" if k in COMMON else f"{k}:{v}" for k, v in sorted(d.items()))


class Attributor:
    """Attribution of affected pairs of one condition (world, budget, masks, seed)."""

    def __init__(self, tr: Triple, budget: int, masks=None, seed: int = 0) -> None:
        self.tr, self.budget, self.masks, self.seed = tr, int(budget), masks, int(seed)
        self.tw = tr.twin
        self.dial = tr.w.rule == "dial"
        self.p = tr.w.p
        self.stats = {"dispatches": 0}

    # one machine dispatch of the stepping machine of core c
    def _step(self, c: str, rec: np.ndarray, mem: np.ndarray, budgets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.tw.m[c].run(mem, budget=budgets, ablation=self.masks, seed=self.seed, resume=rec)
        self.stats["dispatches"] += 1
        return r.record, r.memory

    def _initial(self, c: str, pairs: np.ndarray, lanes: np.ndarray) -> np.ndarray:
        """The record after reset (budget 0) in the stepping machine's layout, each lane at its own lane index."""
        m = self.tr.m[c]
        n = pairs.shape[0]
        out = np.zeros((n, self.tw.m[c].record_words), dtype=np.uint32)
        # lanes are scattered: one dispatch per contiguous run of lane indices
        order = np.argsort(lanes, kind="stable")
        ls = lanes[order]
        starts = np.r_[0, np.nonzero(np.diff(ls) != 1)[0] + 1, n]
        for a, b in zip(starts[:-1], starts[1:]):
            sel = order[a:b]
            r = m.run(pairs[sel], budget=0, ablation=self.masks, seed=self.seed, lane_base=int(ls[a]))
            self.stats["dispatches"] += 1
            if self.tw is self.tr:
                out[sel] = r.record
            else:
                out[sel] = self.tw.rec[c].set_common(out[sel], self.tr.rec[c], r.record)
                for h in HIDDEN[c]:
                    out[sel, self.tw.rec[c].off[h]] = r.record[:, self.tr.rec[c].off[h]]
        return out

    def _ring(self, c: str, rec, mem):
        return ring_of(self.tw, self.tw.rec[c], rec, mem)

    def _dial_halts(self, state: np.ndarray, ring: np.ndarray, pc: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The dial's decision before a step: (new PCG states, lanes halted now)."""
        P = self.tr.P
        b0 = ring[np.arange(ring.shape[0]), (pc & 0xFFFF) % P]
        zero = active & (b0 == 0)
        halt = np.zeros_like(active)
        if self.p >= 1:
            return state, zero
        if zero.any():
            new, out = pcg_draw(state[zero])
            state = state.copy()
            state[zero] = new
            halt[zero] = out < np.uint64(int(min(int(self.p * 2 ** 32), 2 ** 32 - 1)))
        return state, halt

    def attribute(self, pairs: np.ndarray, lanes: np.ndarray, measured: dict[str, ZResult], cores=("c1", "c2"), census: bool = False) -> dict[str, list[LaneResult]]:
        """Attribute the given pairs (rows of `pairs`, global lane indices `lanes`; `measured`: the single-dispatch results
        of these lanes for every core) for each core in `cores`. Pairs not affected under a core get no decisive event. The
        repaired pass covers the lanes affected under any of `cores`, or with census=True every lane passed, so that every
        event of every such lane is classed (an 'other' anywhere stops the study); probes are run for affected lanes."""
        pairs = np.ascontiguousarray(pairs, dtype=np.uint8)
        lanes = np.asarray(lanes, dtype=np.int64)
        n, T = pairs.shape[0], self.budget
        if n == 0:
            return {c: [] for c in cores}
        aff_any = np.logical_or.reduce([(measured[c].memory != measured["c0"].memory).any(axis=1) for c in cores])
        if not census and not aff_any.any():
            return {c: [LaneResult(c) for _ in range(n)] for c in cores}
        rv0 = self.tw.rec["c0"]
        # C0 trajectory
        recs0 = [self._initial("c0", pairs, lanes)]
        mems0 = [pairs.copy()]
        dial_state = [dial_init(self.seed, lanes)] if self.dial else None
        dial_halted = np.zeros(n, dtype=bool)
        actives = []
        for t in range(1, T + 1):
            r, m = recs0[-1], mems0[-1]
            active = ~dial_halted
            if self.dial:
                st, h = self._dial_halts(dial_state[-1], self._ring("c0", r, m), rv0.get(r, "pc"), active & (rv0.get(r, "haltReason") == 0))
                dial_state.append(st)
                dial_halted |= h
                active = active & ~h
            actives.append(active)
            r1, m1 = self._step("c0", r, m, active.astype(np.uint32))
            recs0.append(r1)
            mems0.append(m1)
        self._check_reproduces("c0", recs0[-1], mems0[-1], measured["c0"], dial_halted if self.dial else None, "stepped C0")
        out: dict[str, list[LaneResult]] = {}
        for c in cores:
            aff = (measured[c].memory != measured["c0"].memory).any(axis=1)
            res = [LaneResult(c) for _ in range(n)]
            out[c] = res
            idx = np.arange(n) if census else np.nonzero(aff_any)[0]
            rvx = self.tw.rec[c]
            sub = lambda a: a[idx]  # noqa: E731
            rx = self._initial(c, pairs[idx], lanes[idx])
            pre = []            # repaired x records before each step (for the probes)
            events: list[Event] = []
            for t in range(1, T + 1):
                r0p, m0p, r0, m0 = sub(recs0[t - 1]), sub(mems0[t - 1]), sub(recs0[t]), sub(mems0[t])
                rx = rvx.set_common(rx, rv0, r0p)          # repair: common state = C0's before step t
                pre.append(rx)
                xr, xm = self._step(c, rx, m0p.copy(), sub(actives[t - 1]).astype(np.uint32))
                diffs = _diff_fields(rv0, r0, m0, rvx, xr, xm)
                ring = None
                for j, d in enumerate(diffs):
                    if not d:
                        continue
                    if ring is None:
                        ring = self._ring("c0", r0p, m0p)
                    pc0 = int(rv0.get(r0p, "pc")[j])
                    b = insn_bytes(ring[j], pc0, self.tr.P)
                    cls = classify(c, b, d, pc0, int(rv0.get(r0, "pc")[j]))
                    events.append(Event(int(idx[j]), t, cls, pc0, " ".join(f"{v:02x}" for v in b), _fmt_diff(d)))
                rx = xr
            bad = [e for e in events if e.cls == "other"]
            if bad:
                e = bad[0]
                raise AttributionError(f"{c}: event of class 'other' at step {e.step}, PC {e.pc:04x}, bytes {e.insn}, diff {e.diff} "
                                       f"(lane {lanes[e.lane]}): an undocumented deviation, or a bug")
            ev_by_lane: dict[int, list[Event]] = {}
            for e in events:
                ev_by_lane.setdefault(e.lane, []).append(e)
            for i, evs in ev_by_lane.items():
                res[i].events = evs
            aidx = np.nonzero(aff)[0]
            missing = [int(i) for i in aidx if int(i) not in ev_by_lane]
            if missing:
                raise AttributionError(f"{c}: affected pairs with no event (lanes {missing[:5]}): attribution machinery is wrong")
            if aidx.size == 0:
                continue
            # probes: one per event of an affected lane, unrepaired from the repaired state before it
            pos = {int(i): k for k, i in enumerate(idx)}
            pevents = [e for i in aidx for e in ev_by_lane[int(i)]]
            pr_rec = np.stack([pre[e.step - 1][pos[e.lane]] for e in pevents])
            pr_mem = np.stack([mems0[e.step - 1][e.lane] for e in pevents])
            pr_lane = np.array([e.lane for e in pevents])
            pr_step = np.array([e.step for e in pevents])
            fr, fm = self._probe(c, pr_rec, pr_mem, pr_lane, pr_step, dial_state)
            fin0 = mems0[-1][pr_lane]
            differs = (fm != fin0).any(axis=1)
            for e, dflag in zip(pevents, differs):
                e.probe_differs = bool(dflag)
            # the probe of each lane's first event is x's own run: it must reproduce the single-dispatch result
            first = {}
            for k, e in enumerate(pevents):
                first.setdefault(e.lane, k)
            fk = np.array([first[int(i)] for i in aidx])
            self._check_reproduces(c, fr[fk], fm[fk], _subset(measured[c], aidx), None if not self.dial else self._probe_dial_halted[fk], "first-event probe")
            for i in aidx:
                evs = ev_by_lane[int(i)]
                prop = [e for e in evs if e.probe_differs]
                if not prop:
                    raise AttributionError(f"{c}: affected lane {lanes[i]} with no propagating event")
                dec = max(prop, key=lambda e: e.step)
                res[i].decisive_class, res[i].decisive_step = dec.cls, dec.step
                res[i].propagating = tuple(sorted({e.cls for e in prop}))
        return out

    _probe_dial_halted = None

    def verify_stepping(self, pairs: np.ndarray, lanes: np.ndarray, measured: dict[str, ZResult]) -> int:
        """Step every core one instruction at a time, unrepaired, through the stepping machine (the world's own or its
        twin, with the dial's decisions on each core's own trajectory) and check that the final memory, registers a..pc
        and write counts equal the single-dispatch results for every lane. Returns the number of lanes checked."""
        pairs = np.ascontiguousarray(pairs, dtype=np.uint8)
        lanes = np.asarray(lanes, dtype=np.int64)
        if pairs.shape[0] == 0:
            return 0
        for c in CORES:
            rec, mem = self._initial(c, pairs, lanes), pairs.copy()
            st = dial_init(self.seed, lanes) if self.dial else None
            rv = self.tw.rec[c]
            halted = np.zeros(pairs.shape[0], dtype=bool)
            for _ in range(self.budget):
                active = ~halted
                if self.dial:
                    st, h = self._dial_halts(st, self._ring(c, rec, mem), rv.get(rec, "pc"), active & (rv.get(rec, "haltReason") == 0))
                    halted |= h
                    active &= ~h
                rec, mem = self._step(c, rec, mem, active.astype(np.uint32))
            self._check_reproduces(c, rec, mem, measured[c], halted if self.dial else None, f"unrepaired stepping of {c}")
        return pairs.shape[0]

    def _probe(self, c: str, rec: np.ndarray, mem: np.ndarray, lane: np.ndarray, step: np.ndarray, dial_state) -> tuple[np.ndarray, np.ndarray]:
        """Run core c unrepaired from (rec, mem) — the state before step `step` — to the end of the budget."""
        remaining = (self.budget - (step - 1)).astype(np.uint32)
        if not self.dial:
            r = self.tw.m[c].run(mem, budget=remaining, ablation=self.masks, seed=self.seed, resume=rec)
            self.stats["dispatches"] += 1
            return r.record, r.memory
        # dial twin: step with the dial's decision on the probe's own trajectory, from the stream state before `step`
        rv = self.tw.rec[c]
        st = np.stack([dial_state[s - 1][l] for s, l in zip(step, lane)])
        halted = np.zeros(len(step), dtype=bool)
        t = step.copy()
        while True:
            active = (t <= self.budget) & ~halted & (rv.get(rec, "haltReason") == 0)
            if not active.any():
                break
            st, h = self._dial_halts(st, self._ring(c, rec, mem), rv.get(rec, "pc"), active)
            halted |= h
            active &= ~h
            rec, mem = self._step(c, rec, mem, active.astype(np.uint32))
            t = t + 1
        self._probe_dial_halted = halted
        return rec, mem

    def _check_reproduces(self, c: str, rec: np.ndarray, mem: np.ndarray, measured: ZResult, dial_halted, what: str) -> None:
        rv = self.tw.rec[c]
        bad = (mem != measured.memory).any(axis=1)
        for name in REGS10:
            bad |= rv.get(rec, name) != measured.reg(name)
        if measured.write_counts is not None and rv.wc:
            bad |= (rec[:, rv.wc[0]: rv.wc[0] + rv.wc[1]] != measured.write_counts).any(axis=1)
        if dial_halted is not None:
            bad |= dial_halted != (measured.halt_reason == 2)
        if bad.any():
            raise AttributionError(f"{c}: {what} does not reproduce the single-dispatch run in {int(bad.sum())} of {len(bad)} lanes")


def _subset(r: ZResult, idx) -> ZResult:
    return ZResult(memory=r.memory[idx], steps=r.steps[idx], halt_reason=r.halt_reason[idx], trap_cause=r.trap_cause[idx],
                   write_counts=None if r.write_counts is None else r.write_counts[idx], write_count_names=r.write_count_names,
                   state=r.state[idx], state_names=r.state_names, record=r.record[idx])


# ── Algocell's own single-pair executors (the reference for the C0 equivalence proofs) ──────────────────────────────
_ALG_PIPES: dict = {}


def derive_executor(base: str, L: int, P: int) -> str:
    """Algocell's executor for (L, P) from another exported executor: createZ80TestShader(L, P) (src/lib/gpu/shaders.ts)
    differs between rings only in `const MEM_LENGTH: u32 = P;` and `var<private> mem: array<u32, max(40, ceil(L/4)·8, P)>;`
    (the tests regenerate every exported plain and ring executor this way)."""
    words = max(40, -(-L // 4) * 2 * 4, P)
    out = []
    hits = [0, 0]
    for ln in base.splitlines(keepends=True):
        if ln.startswith("const MEM_LENGTH: u32 = "):
            ln, hits[0] = f"const MEM_LENGTH: u32 = {P}u;\n", hits[0] + 1
        elif ln.startswith("var<private> mem: array<u32, "):
            ln, hits[1] = f"var<private> mem: array<u32, {words}>;\n", hits[1] + 1
        out.append(ln)
    assert hits == [1, 1], "MEM_LENGTH and the private memory must each be declared exactly once"
    return "".join(out)


def algocell_executor_text(w: World) -> str:
    """The text of Algocell's executor for world w (impact_worlds.World.algocell_executor), derived where needed."""
    from .impact_worlds import SHADER_DIR

    fname, derive = w.algocell_executor()
    text = (SHADER_DIR / fname).read_text()
    if derive:
        key, val = derive.split("=")
        assert key == "MEM_LENGTH" and int(val) == w.P
        text = derive_executor(text, w.L, w.P)
    return text


def algocell_execute(w: World, pairs: np.ndarray, budget: int, masks=None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Run pairs through Algocell's executor for world w in ONE dispatch (so case_id = row index, as the machine's lane
    with lane_base 0). Returns final memory (n, 2L) and the register words (n, 12): a f b c d e h l sp pc writes_a
    writes_b. Params (createZ80TestShader): tape_length@2, pair_length@3, pair_count@4, z80_steps@6, batch_seed@7,
    suppress@8..31."""
    import wgpu

    from .soup import get_device

    dev = get_device()
    key = (w.algocell_executor(), id(dev))
    if key not in _ALG_PIPES:
        module = dev.create_shader_module(code=algocell_executor_text(w))
        storage = {"type": wgpu.BufferBindingType.storage}
        layout = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": storage},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": storage}])
        pipe = dev.create_compute_pipeline(layout=dev.create_pipeline_layout(bind_group_layouts=[layout]),
                                           compute={"module": module, "entry_point": "z80_test"})
        _ALG_PIPES[key] = (pipe, layout)
    pipe, layout = _ALG_PIPES[key]
    L = w.L
    n = pairs.shape[0]
    assert 0 < n <= 65535 * 64
    wpc = (L + 3) // 4
    words = np.zeros((n, 2 * wpc * 4), dtype=np.uint8)
    words[:, :L] = pairs[:, :L]
    words[:, 4 * wpc: 4 * wpc + L] = pairs[:, L:]
    params = np.zeros(32, dtype=np.uint32)
    params[2], params[3], params[4], params[6], params[7] = L, 2 * L, n, budget, seed & 0xFFFFFFFF
    if masks is not None:
        params[8:] = np.asarray(masks, dtype=np.uint32).reshape(24)
    B = wgpu.BufferUsage
    io = dev.create_buffer(size=words.nbytes, usage=B.STORAGE | B.COPY_SRC | B.COPY_DST)
    dev.queue.write_buffer(io, 0, np.ascontiguousarray(words).tobytes())
    rb = dev.create_buffer(size=n * 12 * 4, usage=B.STORAGE | B.COPY_SRC)
    pb = dev.create_buffer(size=128, usage=B.UNIFORM | B.COPY_DST)
    dev.queue.write_buffer(pb, 0, params.tobytes())
    bg = dev.create_bind_group(layout=layout, entries=[
        {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate((pb, io, rb))])
    enc = dev.create_command_encoder()
    p = enc.begin_compute_pass()
    p.set_pipeline(pipe)
    p.set_bind_group(0, bg)
    p.dispatch_workgroups(-(-n // 64))
    p.end()
    dev.queue.submit([enc.finish()])
    out = np.frombuffer(dev.queue.read_buffer(io), dtype=np.uint8).reshape(n, 2 * wpc * 4)
    regs = np.frombuffer(dev.queue.read_buffer(rb), dtype=np.uint32).reshape(n, 12).copy()
    mem = np.concatenate([out[:, :L], out[:, 4 * wpc: 4 * wpc + L]], axis=1)
    for b in (io, rb, pb):
        b.destroy()
    return mem, regs


def compare_c0(w: World, m: ZMachine, pairs: np.ndarray, budget: int, masks=None, seed: int = 0, budget_delta: int = 0) -> dict:
    """Algocell's executor against machine m on the same pairs, one dispatch each (chunks of 65,536 pairs, each with its own
    seed = seed + chunk index, lanes from 0). Counts of pairs differing in memory, write counts and registers a..pc."""
    tot = {"pairs": 0, "mem_changed": 0, "b_written": 0, "halted": 0, "mem_mismatch": 0, "wc_mismatch": 0, "reg_mismatch": 0}
    for k, s0 in enumerate(range(0, pairs.shape[0], 65536)):
        sub = np.ascontiguousarray(pairs[s0: s0 + 65536])
        a_mem, a_regs = algocell_execute(w, sub, budget, masks, seed + k)
        r = m.run(sub, budget=budget + budget_delta, ablation=masks, seed=seed + k, lane_base=0)
        wb = r.writes("B") + (r.writes("padding") if "padding" in r.write_count_names else 0)
        tot["pairs"] += sub.shape[0]
        tot["mem_changed"] += int((a_mem != sub).any(axis=1).sum())
        tot["b_written"] += int((a_regs[:, 11] > 0).sum())
        tot["halted"] += int((r.halt_reason != 0).sum())
        tot["mem_mismatch"] += int((a_mem != r.memory).any(axis=1).sum())
        tot["wc_mismatch"] += int(((a_regs[:, 10] != r.writes("A")) | (a_regs[:, 11] != wb)).sum())
        tot["reg_mismatch"] += int(np.any(np.stack([a_regs[:, j] != r.reg(nm) for j, nm in enumerate(REGS10)], axis=1), axis=1).sum())
    return tot


# ── GPU-resident stepping (the fast path of the attribution; the CPU-stepped Attributor above is its reference) ───────
_REPAIR_WGSL = """
struct RP { n: u32, src_words: u32, dst_words: u32, count: u32 }
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read> fmap: array<u32>;
@group(0) @binding(3) var<uniform> rp: RP;
@compute @workgroup_size(64)
fn repair(@builtin(global_invocation_id) id: vec3u) {
    let lane = id.x;
    if (lane >= rp.n) { return; }
    for (var k = 0u; k < rp.count; k++) {
        dst[lane * rp.dst_words + fmap[2u * k + 1u]] = src[lane * rp.src_words + fmap[2u * k]];
    }
}
"""


def _common_offsets(rv: Rec) -> list[int]:
    offs = [rv.off[n] for n in ("steps", "haltReason", "trapCause") + COMMON]
    if rv.wc:
        offs += list(range(rv.wc[0], rv.wc[0] + rv.wc[1]))
    if rv.pad:
        offs += list(range(rv.pad[0], rv.pad[0] + rv.pad[1]))
    return offs


class FastAttributor(Attributor):
    """The same attribution as Attributor, with every core stepped on the GPU: per step, one encoder records the repair of
    C1 and C2 from C0's pre-step state (a copy of the memory and a kernel that copies the common record words), one
    single-step dispatch per core, and copies of the records and memories into history buffers; the CPU reads the
    histories once per chunk of steps and finds and classes the events. Probes are one dispatch per core. Not for the dial
    (its halting decision is made on the CPU between steps): Attributor handles those worlds."""

    HIST_BYTES = 96 << 20

    def __init__(self, tr: Triple, budget: int, masks=None, seed: int = 0) -> None:
        super().__init__(tr, budget, masks, seed)
        assert not self.dial, "the dial's decision is made on the CPU: use Attributor"
        import wgpu

        self.wgpu = wgpu
        self.dev = self.tw.m["c0"].dev
        mod = self.dev.create_shader_module(code=_REPAIR_WGSL)
        self.rpipe = self.dev.create_compute_pipeline(layout="auto", compute={"module": mod, "entry_point": "repair"})

    def attribute(self, pairs: np.ndarray, lanes: np.ndarray, measured: dict[str, ZResult], cores=("c1", "c2"), census: bool = False) -> dict[str, list[LaneResult]]:
        """As Attributor.attribute. With census=True every lane passed is put through the repaired pass (not only the
        affected ones), so every event of every lane is classed and an 'other' anywhere stops the study; the results
        then also carry the events of unaffected lanes (decisive_class None)."""
        wgpu = self.wgpu
        pairs = np.ascontiguousarray(pairs, dtype=np.uint8)
        lanes = np.asarray(lanes, dtype=np.int64)
        n, T = pairs.shape[0], self.budget
        out = {c: [LaneResult(c) for _ in range(n)] for c in cores}
        if n == 0:
            return out
        aff = {c: (measured[c].memory != measured["c0"].memory).any(axis=1) for c in cores}
        sel = np.arange(n) if census else np.nonzero(np.logical_or.reduce([aff[c] for c in cores]))[0]
        if sel.size == 0:
            return out
        dev, B = self.dev, wgpu.BufferUsage
        ks = ("c0",) + tuple(cores)
        ms = {c: self.tw.m[c] for c in ks}
        rvs = {c: self.tw.rec[c] for c in ks}
        m = sel.size
        sw = ms["c0"].storage_words
        rw = {c: ms[c].record_words for c in ks}
        mk = lambda size, extra=0: dev.create_buffer(size=max(int(size), 4), usage=B.STORAGE | B.COPY_SRC | B.COPY_DST | extra)  # noqa: E731
        mem = {c: mk(m * sw * 4) for c in ks}
        rec = {c: mk(m * rw[c] * 4) for c in ks}
        ones = mk(m * 4)
        dev.queue.write_buffer(ones, 0, np.ones(m, dtype=np.uint32).tobytes())
        params = {}
        for c in ks:
            p = ms[c].params(m, 0, self.seed, 0, self.masks, 0, True)
            params[c] = dev.create_buffer(size=max(p.nbytes, 16), usage=B.UNIFORM | B.COPY_DST)
            dev.queue.write_buffer(params[c], 0, p.tobytes())
        init = {c: self._initial(c, pairs[sel], lanes[sel]) for c in ks}
        for c in ks:
            dev.queue.write_buffer(mem[c], 0, ms["c0"].pack(pairs[sel]).tobytes())
            dev.queue.write_buffer(rec[c], 0, np.ascontiguousarray(init[c]).tobytes())
        bgs = {c: ms[c].bind_group(params[c], mem[c], rec[c], ones) for c in ks}
        # repair kernels: Cx common record words ← C0's
        off0 = _common_offsets(rvs["c0"])
        rbg = {}
        keep = []
        for c in cores:
            offx = _common_offsets(rvs[c])
            assert len(offx) == len(off0)
            fmap = np.array([v for pr in zip(off0, offx) for v in pr], dtype=np.uint32)
            fbuf = mk(fmap.nbytes)
            dev.queue.write_buffer(fbuf, 0, fmap.tobytes())
            ubuf = dev.create_buffer(size=16, usage=B.UNIFORM | B.COPY_DST)
            dev.queue.write_buffer(ubuf, 0, np.array([m, rw["c0"], rw[c], len(off0)], dtype=np.uint32).tobytes())
            rbg[c] = dev.create_bind_group(layout=self.rpipe.get_bind_group_layout(0), entries=[
                {"binding": 0, "resource": {"buffer": rec["c0"], "offset": 0, "size": rec["c0"].size}},
                {"binding": 1, "resource": {"buffer": rec[c], "offset": 0, "size": rec[c].size}},
                {"binding": 2, "resource": {"buffer": fbuf, "offset": 0, "size": fbuf.size}},
                {"binding": 3, "resource": {"buffer": ubuf, "offset": 0, "size": 16}}])
            keep += [fbuf, ubuf]
        per_step = m * 4 * (2 * (rw["c0"] + sw) + sum(2 * rw[c] + sw for c in cores))
        K = max(1, min(T, self.HIST_BYTES // max(per_step, 1)))
        events: dict[str, list[Event]] = {c: [] for c in cores}
        pre_rec: dict[str, list] = {c: [] for c in cores}     # per event: the repaired x record before its step
        pre_mem: dict[str, list] = {c: [] for c in cores}     # per event: C0's memory before its step
        t0 = 1
        while t0 <= T:
            k = min(K, T - t0 + 1)
            h0r, h0m = mk(k * m * rw["c0"] * 4), mk(k * m * sw * 4)
            h0r_pre, h0m_pre = mk(k * m * rw["c0"] * 4), mk(k * m * sw * 4)
            hxr_pre = {c: mk(k * m * rw[c] * 4) for c in cores}
            hxr = {c: mk(k * m * rw[c] * 4) for c in cores}
            hxm = {c: mk(k * m * sw * 4) for c in cores}
            enc = dev.create_command_encoder()
            for j in range(k):
                for c in cores:     # repair from C0's pre-step state
                    enc.copy_buffer_to_buffer(mem["c0"], 0, mem[c], 0, m * sw * 4)
                    p = enc.begin_compute_pass()
                    p.set_pipeline(self.rpipe)
                    p.set_bind_group(0, rbg[c])
                    p.dispatch_workgroups(-(-m // 64))
                    p.end()
                    enc.copy_buffer_to_buffer(rec[c], 0, hxr_pre[c], j * m * rw[c] * 4, m * rw[c] * 4)
                enc.copy_buffer_to_buffer(rec["c0"], 0, h0r_pre, j * m * rw["c0"] * 4, m * rw["c0"] * 4)
                enc.copy_buffer_to_buffer(mem["c0"], 0, h0m_pre, j * m * sw * 4, m * sw * 4)
                for c in ks:
                    ms[c].encode(enc, bgs[c], m)
                enc.copy_buffer_to_buffer(rec["c0"], 0, h0r, j * m * rw["c0"] * 4, m * rw["c0"] * 4)
                enc.copy_buffer_to_buffer(mem["c0"], 0, h0m, j * m * sw * 4, m * sw * 4)
                for c in cores:
                    enc.copy_buffer_to_buffer(rec[c], 0, hxr[c], j * m * rw[c] * 4, m * rw[c] * 4)
                    enc.copy_buffer_to_buffer(mem[c], 0, hxm[c], j * m * sw * 4, m * sw * 4)
            dev.queue.submit([enc.finish()])
            self.stats["dispatches"] += k * len(ks)
            rd = lambda b, w: np.frombuffer(dev.queue.read_buffer(b), dtype=np.uint32).reshape(k, m, w)  # noqa: E731
            R0pre, R0 = rd(h0r_pre, rw["c0"]), rd(h0r, rw["c0"])
            M0pre = np.stack([ms["c0"].unpack(x) for x in rd(h0m_pre, sw)])
            M0 = np.stack([ms["c0"].unpack(x) for x in rd(h0m, sw)])
            for c in cores:
                Rxp, Rx = rd(hxr_pre[c], rw[c]), rd(hxr[c], rw[c])
                Mx = np.stack([ms["c0"].unpack(x) for x in rd(hxm[c], sw)])
                for j in range(k):
                    t = t0 + j
                    diffs = _diff_fields(rvs["c0"], R0[j], M0[j], rvs[c], Rx[j], Mx[j])
                    if not any(diffs):
                        continue
                    ring = ring_of(self.tw, rvs["c0"], R0pre[j], M0pre[j])
                    pcs0, pcs1 = rvs["c0"].get(R0pre[j], "pc"), rvs["c0"].get(R0[j], "pc")
                    for q, d in enumerate(diffs):
                        if not d:
                            continue
                        pc0 = int(pcs0[q])
                        b = insn_bytes(ring[q], pc0, self.tr.P)
                        events[c].append(Event(int(sel[q]), t, classify(c, b, d, pc0, int(pcs1[q])), pc0, " ".join(f"{v:02x}" for v in b), _fmt_diff(d)))
                        pre_rec[c].append(Rxp[j, q].copy())
                        pre_mem[c].append(M0pre[j, q].copy())
            if t0 + k - 1 == T:
                final0_rec, final0_mem = R0[-1].copy(), M0[-1].copy()
            for b in [h0r, h0m, h0r_pre, h0m_pre, *hxr_pre.values(), *hxr.values(), *hxm.values()]:
                b.destroy()
            t0 += k
        for b in [*mem.values(), *rec.values(), ones, *keep]:
            b.destroy()
        self._check_reproduces("c0", final0_rec, final0_mem, _subset(measured["c0"], sel), None, "GPU-stepped C0")
        for c in cores:
            evs = events[c]
            bad = [e for e in evs if e.cls == "other"]
            if bad:
                e = bad[0]
                raise AttributionError(f"{c}: event of class 'other' at step {e.step}, PC {e.pc:04x}, bytes {e.insn}, diff {e.diff} "
                                       f"(lane {lanes[e.lane]}): an undocumented deviation, or a bug")
            by_lane: dict[int, list[int]] = {}
            for k_, e in enumerate(evs):
                by_lane.setdefault(e.lane, []).append(k_)
            idx = np.nonzero(aff[c])[0]
            missing = [int(i) for i in idx if int(i) not in by_lane]
            if missing:
                raise AttributionError(f"{c}: affected pairs with no event (lanes {missing[:5]}): attribution machinery is wrong")
            for i, ks_ in by_lane.items():
                out[c][i].events = [evs[q] for q in ks_]
            if idx.size == 0:
                continue
            pk = [q for i in idx for q in by_lane[int(i)]]       # probes for the events of affected lanes
            pr_rec = np.stack([pre_rec[c][q] for q in pk])
            pr_mem = np.stack([pre_mem[c][q] for q in pk])
            steps = np.array([evs[q].step for q in pk])
            r = self.tw.m[c].run(pr_mem, budget=(T - (steps - 1)).astype(np.uint32), ablation=self.masks, seed=self.seed, resume=pr_rec)
            self.stats["dispatches"] += 1
            fin0 = measured["c0"].memory[[evs[q].lane for q in pk]]
            differs = (r.memory != fin0).any(axis=1)
            for q, dflag in zip(pk, differs):
                evs[q].probe_differs = bool(dflag)
            where = {q: z for z, q in enumerate(pk)}
            firsts = [where[by_lane[int(i)][0]] for i in idx]
            self._check_reproduces(c, r.record[firsts], r.memory[firsts], _subset(measured[c], idx), None, "first-event probe")
            for i in idx:
                lr = out[c][i]
                prop = [e for e in lr.events if e.probe_differs]
                if not prop:
                    raise AttributionError(f"{c}: affected lane {lanes[i]} with no propagating event")
                dec = max(prop, key=lambda e: e.step)
                lr.decisive_class, lr.decisive_step = dec.cls, dec.step
                lr.propagating = tuple(sorted({e.cls for e in prop}))
        return out
