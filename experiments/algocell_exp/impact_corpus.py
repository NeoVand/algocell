"""Corpora of the emulator impact study (planning/IMPACT_STUDY_DRAFT.md): the registered conditions and their archived
soups, and the pair draws of corpus 1 (random programs) and corpus 2 (archived soups), with fixed seed streams.

Seed streams. Every draw comes from `np.random.default_rng(SeedSequence([stream, corpus, h32(condition id)]))`, where h32
is the first four bytes (little-endian) of the SHA-256 of the condition id. The registered corpora use stream 1 (the
draft's "seed 1"); the dry run uses stream DRY_STREAM. The machine run seed of a condition (read only by the dial and
randreg/randsp hooks, together with the lane = the pair's index in its corpus) is the first u32 of
`SeedSequence([stream, 3, h32(id)])`.

Dry-run exclusion. The dry run never touches a registered draw:
  corpus 1: the dry-run pairs are drawn from their own stream and any pair whose bytes equal a registered pair of the
            same condition is redrawn (the registered pairs are regenerated, never run, for this check);
  corpus 2: a draw is a slot (source soup, cell i, partner j); the registered slots are regenerated (indices only, no soup
            is read for this) and any dry-run slot among them is redrawn.
The registered corpora are drawn exactly as the draft says, independently of the dry run.

Condition registry (`conditions()`). Tier 'draft': the archives the draft names (Stages A–F final and emergence soups,
the line-of-descent runs lod_v6 and N6, the Stage G open-phase snapshots that X1 and X2 drew partners from). Tier
'added': registered conditions with archived final soups that the draft's list does not name (Stages G, H, I, K, L, M;
the earlier N3 line-of-descent runs; DZ; C; C4a; C4a-rep; E; R4 invasions; N5). See IMPACT report, protocol issues.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import brotli
import numpy as np

from .impact_worlds import WORLDS, plain_world

RUNS = Path(os.environ.get("ALGOCELL_RUNS", "/Users/neo/repos/algocell/experiments/runs"))   # read-only archive
RESULTS = Path(__file__).resolve().parent.parent / "results"
REG_STREAM = 1
DRY_STREAM = 0xD2  # 210: the dry run's stream
CORPUS_RANDOM, CORPUS_SOUPS, RUN_SEED = 1, 2, 3
W, H = 160, 125

# corpus 1: the ring-difftest sizes (P, L) and Stage F's odd and padded rings (P, L as in Stage F)
RANDOM_RINGS = ((32, 16), (38, 19), (40, 20), (64, 32), (100, 50), (128, 64), (33, 16), (35, 16), (37, 16), (73, 36), (75, 36), (79, 36))
RANDOM_BUDGET = 128

STAGES_DRAFT = ("stageA", "stageB", "stageC", "stageD", "stageE", "stageF")
STAGES_ADDED = ("stageG", "stageH", "stageI", "stageK", "stageL", "stageM")


def h32(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "little")


def rng_for(stream: int, corpus: int, cid: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([stream, corpus, h32(cid)]))


def run_seed(stream: int, cid: str) -> int:
    return int(np.random.SeedSequence([stream, RUN_SEED, h32(cid)]).generate_state(1, dtype=np.uint32)[0])


# ── sources and conditions ─────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Source:
    path: Path
    L: int
    width: int = W
    height: int = H

    def load(self) -> np.ndarray:
        p = self.path
        a = np.load(p) if p.suffix == ".npy" else np.frombuffer(brotli.decompress(p.read_bytes()), dtype=np.uint8)
        a = np.ascontiguousarray(a, dtype=np.uint8).reshape(self.width * self.height, self.L)
        return a

    @property
    def rel(self) -> str:
        return str(self.path.relative_to(RUNS)) if self.path.is_relative_to(RUNS) else str(self.path)


@dataclass
class Condition:
    cid: str
    family: str
    tier: str              # 'draft' | 'added' | 'random'
    world: str             # impact_worlds tag
    budget: int
    suppress: tuple = ()
    grid: str = "square"   # 'square' (lattice neighbours) | 'mixed' (partner uniform over the other cells) | 'random'
    snapshot: str = "final"
    sources: list = field(default_factory=list)
    L: int = 16
    P: int = 32
    note: str = ""

    @property
    def n_sources(self) -> int:
        return len(self.sources)


def _world_tag(L: int, P: int, lethal: bool) -> str:
    w = plain_world(L, P, lethal)
    if w is None:
        raise KeyError(f"no impact world for L={L} P={P} lethal={lethal}")
    return w.tag


_SEED_RE = re.compile(r"^(?P<label>.+)_s(?P<seed>\d+)$")


def _stage_conditions(stage: str, tier: str) -> list[Condition]:
    groups: dict = defaultdict(lambda: {"final": [], "emergence": []})
    meta: dict = {}
    for sp in sorted((RUNS / stage).rglob("*.summary.json")):
        stem = sp.name[: -len(".summary.json")]
        m = _SEED_RE.match(stem)
        if not m:
            continue
        s = json.loads(sp.read_text())
        L = int(s["tape_length"])
        P = int(s.get("mem_length") or 2 * L)
        key = (stage, m["label"])
        cond = {"L": L, "P": P, "zero_halts": bool(s.get("zero_halts") or False), "grid": s.get("grid", "square"),
                "budget": int(s["z80_steps"]), "suppress": tuple(s.get("suppress") or ()),
                "width": int(s.get("width", W)), "height": int(s.get("height", H))}
        if key in meta:
            assert meta[key] == cond, f"{key}: worlds of one condition disagree: {meta[key]} vs {cond}"
        meta[key] = cond
        for snap in ("final", "emergence"):
            f = sp.with_name(f"{stem}.soup_{snap}.u8.br")
            if f.exists():
                groups[key][snap].append(Source(f, L, cond["width"], cond["height"]))
    out = []
    for (st, label), snaps in sorted(groups.items()):
        c = meta[(st, label)]
        for snap, srcs in snaps.items():
            if srcs:
                out.append(Condition(f"{st}/{label}@{snap}", st, tier, _world_tag(c["L"], c["P"], c["zero_halts"]), c["budget"],
                                     c["suppress"], c["grid"], snap, srcs, c["L"], c["P"]))
    return out


def _npy_conditions(family: str, tier: str, pattern: str, world_of, budget: int = 128, note: str = "", label_of=None) -> list[Condition]:
    """Archives of one .npy per world (20,000 × L, the 160 × 125 square lattice). `world_of(path) -> (world tag, L)`."""
    groups: dict = defaultdict(list)
    for f in sorted(RUNS.glob(pattern)):
        tag, L = world_of(f)
        label = label_of(f) if label_of else re.sub(r"_s\d+(_final|_t\d+)?$", "", f.stem)
        groups[(label, tag, L)].append(Source(f, L))
    return [Condition(f"{family}/{label}@final", family, tier, tag, budget, (), "square", "final", srcs, L, 2 * L, note)
            for (label, tag, L), srcs in sorted(groups.items())]


def _x_inputs() -> list[Condition]:
    """The soups X1 (arm K: the step-5,000 and 10,000 snapshots of the Stage G worlds at L = 16 and 64) and X2 (each Stage G
    world's registered in-soup snapshot, and step 10,000 at L = 16) drew partners from."""
    sets: dict = defaultdict(set)
    perw = RESULTS / "inflow_insoup" / "per_world.csv"
    stem = lambda L, seed: next((RUNS / "stageG").glob(f"none@closure*_L{L}_st128_k4_s{seed}.summary.json")).name[: -len(".summary.json")]  # noqa: E731
    with perw.open() as fh:
        for r in csv.DictReader(fh):
            L, seed = int(r["L"]), int(r["seed"])
            name = r["soup_snapshot"]          # step 5,000 at L = 16 and 20; the snapshot nearest 2 × t_rep at L = 50 and 64
            sets[(L, "t5000" if L in (16, 20) else "2t_rep")].add((seed, name))
            if r.get("soup10k_snapshot"):
                sets[(L, r["soup10k_snapshot"])].add((seed, r["soup10k_snapshot"]))
    for L in (16, 64):
        for sp in sorted((RUNS / "stageG").glob(f"none@closure*_L{L}_st128_k4_s*.summary.json")):
            seed = int(_SEED_RE.match(sp.name[: -len(".summary.json")])["seed"])
            for name in ("t5000", "t10000"):
                sets[(L, name)].add((seed, name))
    out = []
    for (L, setname), members in sorted(sets.items()):
        srcs = [Source(RUNS / "stageG" / f"{stem(L, seed)}.soup_{name}.u8.br", L) for seed, name in sorted(members)]
        missing = [s.rel for s in srcs if not s.path.exists()]
        assert not missing, f"X inputs missing: {missing[:3]}"
        out.append(Condition(f"X-inputs/L{L}@{setname}", "X-inputs", "draft", f"L{L}", 128, (), "square", setname, srcs, L, 2 * L,
                             "Stage G open-phase snapshots that X1 (arm K) and X2 drew partners from"))
    return out


def _offset_world(f: Path):
    m = re.match(r"L(\d+)_(benign|lethal)_", f.name)
    L = int(m[1])
    return (f"lethal_L{L}" if m[2] == "lethal" else f"L{L}"), L


def _r4_world(f: Path):
    return ("randreg_L16" if f.name.startswith("randreg") else "L16"), 16


def _dial_world(f: Path):
    return f"{f.name.split('_')[0]}_L16", 16


def _conv_world(f: Path):
    return f"{f.name.split('_')[0]}_L16", 16


def conditions(tiers=("draft", "added")) -> list[Condition]:
    """Every corpus-2 condition, in a fixed order."""
    out: list[Condition] = []
    if "draft" in tiers:
        for st in STAGES_DRAFT:
            out += _stage_conditions(st, "draft")
        out += _npy_conditions("lod_v6", "draft", "lod_v6_modal/lod_v6/*/final.npy", lambda f: ("L16", 16), label_of=lambda f: "L16_benign",
                               note="N3 line-of-descent worlds (seeds 31001–31020)")
        out += _npy_conditions("N6", "draft", "lod_n6_modal/lod_n6/*/final.npy", lambda f: ("L16", 16), label_of=lambda f: "L16_benign",
                               note="N6 replication worlds (seeds 32001–32020)")
        out += _x_inputs()
    if "added" in tiers:
        for st in STAGES_ADDED:
            out += _stage_conditions(st, "added")
        for d in ("lod_modal", "lod_v2_modal", "lod_v3_modal"):
            out += _npy_conditions(f"N3-{d}", "added", f"{d}/**/final.npy", lambda f: ("L16", 16), label_of=lambda f: "L16_benign",
                                   note="N3 line-of-descent worlds under the earlier recorder rules")
        out += _npy_conditions("DZ", "added", "dial/dial/lp*_s*_t300000.npy", _dial_world, note="lethality dial, final = step 300,000")
        out += _npy_conditions("C", "added", "conv_modal/conv/rand*_s80[0-9][0-9]_t300000.npy", _conv_world, note="conventions, final = step 300,000")
        out += _npy_conditions("C4a", "added", "conv_long/conv_long/randreg_s81*_t3000000.npy", _conv_world, note="final = step 3,000,000")
        out += _npy_conditions("C4a-rep", "added", "conv_rep/conv_rep/randreg_s82*_t3000000.npy", _conv_world, note="final = step 3,000,000")
        out += _npy_conditions("E", "added", "offset/offset/*_final.npy", _offset_world, note="copy-offset switch")
        out += _npy_conditions("R4", "added", "r4_inv/r4_inv/*_final.npy", _r4_world, note="C4b/C4c invasions")
        out += _npy_conditions("N5", "added", "n5/*.npy", lambda f: ("L16", 16), note="N5 invasions")
    ids = [c.cid for c in out]
    assert len(ids) == len(set(ids)), "condition ids must be unique"
    for c in out:
        assert c.world in WORLDS, c
        assert all(s.path.exists() for s in c.sources), c.cid
    return out


def random_conditions() -> list[Condition]:
    return [Condition(f"random/P{P}_L{L}", "random", "random", (f"L{L}" if P == 2 * L else f"L{L}_P{P}"), RANDOM_BUDGET, (), "random",
                      "-", [], L, P) for P, L in RANDOM_RINGS]


# ── draws ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def draw_random(c: Condition, n: int, stream: int) -> np.ndarray:
    return rng_for(stream, CORPUS_RANDOM, c.cid).integers(0, 256, size=(n, 2 * c.L), dtype=np.uint8)


def draw_slots(c: Condition, n: int, stream: int) -> np.ndarray:
    """n slots (source index, cell i, partner j), uniform with replacement over the condition's worlds and cells; the
    partner is a lattice neighbour (reflected at the edges, as prepare_batch draws it) or, on the well-mixed grid, a
    uniform other cell."""
    rng = rng_for(stream, CORPUS_SOUPS, c.cid)
    return _slots(c, n, rng)


def _slots(c: Condition, n: int, rng: np.random.Generator) -> np.ndarray:
    dims = {(s.width, s.height) for s in c.sources}
    assert len(dims) == 1, f"{c.cid}: sources of different lattice sizes"
    w, h = dims.pop()
    src = rng.integers(0, len(c.sources), size=n)
    x = rng.integers(0, w, size=n)
    y = rng.integers(0, h, size=n)
    i = y * w + x
    if c.grid == "mixed":
        j = rng.integers(0, w * h - 1, size=n)
        j = j + (j >= i)
    else:
        assert c.grid == "square", c.grid
        d = rng.integers(0, 4, size=n)
        nx, ny = x.copy(), y.copy()
        nx = np.where(d == 0, np.where(x + 1 < w, x + 1, x - 1), nx)
        ny = np.where(d == 1, np.where(y + 1 < h, y + 1, y - 1), ny)
        nx = np.where(d == 2, np.where(x > 0, x - 1, x + 1), nx)
        ny = np.where(d == 3, np.where(y > 0, y - 1, y + 1), ny)
        j = ny * w + nx
    return np.stack([src, i, j], axis=1).astype(np.int64)


def dry_slots(c: Condition, n: int, n_registered: int) -> tuple[np.ndarray, int]:
    """Dry-run slots, none of which is a registered slot (the registered draw of n_registered is regenerated to check).
    Returns (slots, number of redraws)."""
    reg = {tuple(r) for r in draw_slots(c, n_registered, REG_STREAM).tolist()}
    rng = rng_for(DRY_STREAM, CORPUS_SOUPS, c.cid)
    out, redrawn = [], 0
    while len(out) < n:
        for r in _slots(c, n - len(out), rng).tolist():
            if tuple(r) in reg:
                redrawn += 1
            else:
                out.append(r)
    return np.array(out, dtype=np.int64), redrawn


def dry_random(c: Condition, n: int, n_registered: int) -> tuple[np.ndarray, int]:
    reg = {bytes(r) for r in draw_random(c, n_registered, REG_STREAM)}
    rng = rng_for(DRY_STREAM, CORPUS_RANDOM, c.cid)
    out, redrawn = [], 0
    while len(out) < n:
        for r in rng.integers(0, 256, size=(n - len(out), 2 * c.L), dtype=np.uint8):
            if bytes(r) in reg:
                redrawn += 1
            else:
                out.append(r)
    return np.stack(out), redrawn


def pairs_of(c: Condition, slots: np.ndarray, cache: dict | None = None) -> np.ndarray:
    """(n, 2L) pairs for the slots: tape A = cell i, tape B = cell j of the slot's source soup."""
    out = np.empty((slots.shape[0], 2 * c.L), dtype=np.uint8)
    for s in np.unique(slots[:, 0]):
        sel = slots[:, 0] == s
        src = c.sources[int(s)]
        soup = cache.get(src.path) if cache is not None else None
        if soup is None:
            soup = src.load()
            if cache is not None:
                cache[src.path] = soup
        out[sel, : c.L] = soup[slots[sel, 1]]
        out[sel, c.L:] = soup[slots[sel, 2]]
    return out
