"""Emulator impact study tooling (impact_worlds, impact, impact_corpus). Needs the local GPU; a few minutes.

(a) machines: worlds_impact.json is what impact_worlds writes; every world's three machines share the world spec and its
    SHA-256, the hooks and every manifest part a core cannot change, and name the core they claim; C0 machines without
    hooks emit the 0.2.0 core text; the vendored files are what build_machines.mjs generates (skipped without a Zilion
    checkout at the recorded commit);
(b) Algocell's executors differ between rings only in their MEM_LENGTH line, so the derived L = 19, P = 38 executor is
    what export-sim.ts would write;
(c) the classifier on every class, with the exact difference patterns, and 'other' for everything else;
(d) attribution on constructed programs, one per class, under C1 and C2, in a plain, a padded and a lethal world and
    through the twins of the PCG worlds (dial, randreg, randsp); the decisive event when a transient event precedes the
    cause; the stop on an 'other' event and on a failed consistency check;
(e) stepping through resume (and the twins) reproduces the single-dispatch runs on random pairs;
(f) Clopper–Pearson against closed forms and a tabulated value;
(g) corpora: deterministic draws, lattice and well-mixed partners, and the dry run's exclusion from the registered draws.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from algocell_exp import impact as I
from algocell_exp import impact_corpus as K
from algocell_exp import impact_worlds as IW
from algocell_exp.zmachine import MACHINES_DIR, load_machine

L = 16


def tape(a, extra=(), L=L):
    t = np.zeros(2 * L, np.uint8)
    t[: len(a)] = a
    for k, v in extra:
        t[k] = v
    return t


# one program per class; each stores the differing value (A from R, or F through PUSH AF) into tape B
PROGRAMS = {
    # DD FD: a wasted DD prefix (R + 2 in C0, R + 1 in C1/C2); FD 00; LD A,R; LD (001E),A; HALT
    "wasted-prefix": (tape([0xDD, 0xFD, 0x00, 0xED, 0x5F, 0x32, 0x1E, 0x00, 0x76]), {"c1": ("wasted-prefix", 1), "c2": ("wasted-prefix", 1)}),
    # LD HL,0010; BIT 0,(HL) with (HL) = 28: X/Y from the operand in C0, from WZ = 0 in C1/C2; PUSH AF; HALT
    "bit-hl": (tape([0x21, 0x10, 0x00, 0xCB, 0x46, 0xF5, 0x76], [(16, 0x28)]), {"c1": ("bit-hl", 2), "c2": ("bit-hl", 2)}),
    # LD A,28; OR A (F = 2C, Q = 2C); LD A,0 (Q = 0); SCF: X/Y = ((Q^F)|A) & 28 = 28 in C2, A & 28 = 0 otherwise; PUSH AF
    "scf-ccf": (tape([0x3E, 0x28, 0xB7, 0x3E, 0x00, 0x37, 0xF5, 0x76]), {"c2": ("scf-ccf", 4)}),
    # LD A,2; LD HL,0014; LD DE,000B; LD BC,5; LDIR at 000B copies F5 onto its own ED: the repeating iteration's X/Y
    # (from A + F5 in C0, from PC in C1/C2) survives into the PUSH AF fetched next
    "block-repeat": (tape([0x3E, 0x02, 0x21, 0x14, 0x00, 0x11, 0x0B, 0x00, 0x01, 0x05, 0x00, 0xED, 0xB0, 0x76], [(20, 0xF5)]),
                     {"c1": ("block-repeat", 5), "c2": ("block-repeat", 5)}),
    # a BIT n,(HL) event whose F is overwritten (XOR A) before the SCF that reaches memory: in C2 the decisive event is the
    # SCF; in C1 the pair is not affected
    "bit-then-scf": (tape([0x21, 0x10, 0x00, 0xCB, 0x46, 0xAF, 0x3E, 0x28, 0xB7, 0x3E, 0x00, 0x37, 0xF5, 0x76], [(16, 0x28)]),
                     {"c2": ("scf-ccf", 7)}),
    # SCF behind a DD prefix: the prefix clears Q in C2
    "dd-scf": (tape([0x3E, 0x28, 0xB7, 0x3E, 0x00, 0xDD, 0x37, 0xF5, 0x76]), {"c2": ("scf-ccf", 4)}),
    # a CPIR that repeats once, then stops: its repeating iteration's flags are overwritten by the last one: no effect
    "cpir-transient": (tape([0x3E, 0x77, 0x21, 0x10, 0x00, 0x01, 0x02, 0x00, 0xED, 0xB1, 0xF5, 0x76], [(16, 0x28), (17, 0x05)]), {}),
}


def _attribute(tag, programs, budget=128, seed=99, masks=None, lanes=None):
    tr = I.Triple(tag)
    pairs = np.stack([p for p, _ in programs.values()])
    lanes = np.arange(len(pairs)) if lanes is None else lanes
    res = _run_lanes(tr, pairs, lanes, budget, masks, seed)
    at = I.Attributor(tr, budget, masks, seed)
    return tr, res, at.attribute(pairs, lanes, res), at


def _run_lanes(tr, pairs, lanes, budget, masks, seed):
    """Single-dispatch results with each pair at its own lane index."""
    out = {k: [] for k in IW.CORES}
    for p, ln in zip(pairs, lanes):
        r = tr.run(p[None], budget, masks, seed, lane_base=int(ln))
        for k in IW.CORES:
            out[k].append(r[k])
    from impact_study import _cat
    return {k: _cat(v) for k, v in out.items()}


# ── (a) machines ────────────────────────────────────────────────────────────────────────────────────────────────────
def test_spec_file_is_generated():
    assert IW.SPEC_FILE.read_text() == IW.spec_json()


@pytest.mark.parametrize("tag", sorted(IW.WORLDS))
def test_triple_differs_only_in_core(tag):
    tr = I.Triple(tag)          # asserts world spec, world SHA-256, hooks and every core-independent manifest part
    w = IW.world(tag)
    m = {k: json.loads((MACHINES_DIR / f"{w.machine(k)}.manifest.json").read_text()) for k in IW.CORES}
    assert len({x["sha256"]["world"] for x in m.values()}) == 1
    assert len({json.dumps(x["world"], sort_keys=True) for x in m.values()}) == 1
    assert len({tuple(x["core"]["hooks"]) for x in m.values()}) == 1
    assert len({x["sha256"]["core"] for x in m.values()}) == 3      # three different cores
    assert [(x["core"]["compat"], x["core"]["variant"], x["core"]["coreVersion"]) for x in m.values()] == [
        ("0.2", "classic", "0.2.0"), ("0.3", "classic", "0.3.0"), ("0.3", "zilog-nmos", "0.3.0")]
    spec = json.loads(IW.SPEC_FILE.read_text())
    assert spec[w.machine("c0")]["world"] == spec[w.machine("c1")]["world"] == spec[w.machine("c2")]["world"] == w.spec()
    assert tr.m["c0"].m["resumable"] == (w.resumable_twin is None)
    if w.resumable_twin:
        tw = tr.twin
        assert tw.m["c0"].m["resumable"] and (tw.L, tw.P) == (w.L, w.P)


def test_c0_without_hooks_is_the_020_core():
    prov = (MACHINES_DIR / "PROVENANCE.md").read_text()
    z80 = prov.split("SHA-256 `", 1)[1].split("`", 1)[0]
    for w in IW.WORLDS.values():
        m = json.loads((MACHINES_DIR / f"{w.machine('c0')}.manifest.json").read_text())
        if not m["core"]["hooks"]:
            assert m["sha256"]["core"] == z80, w.tag


def test_existing_proven_machines_are_identical():
    """The four C0 worlds that zsoup_validate already proved are the same machines byte for byte."""
    for mine, old in (("L16", "z80c02_pair_L16"), ("L32", "z80c02_pair_L32"), ("lethal_L16", "z80c02_lethal_L16"), ("lethal_L32", "z80c02_lethal_L32")):
        assert (MACHINES_DIR / f"imp_{mine}_c0.wgsl").read_text() == (MACHINES_DIR / f"{old}.wgsl").read_text()


def test_vendored_machines_match_zilion():
    zdir = MACHINES_DIR.parents[2].parent / "zilion"
    node = shutil.which("node")
    if not node or not (zdir / "dist/index.js").exists():
        pytest.skip("no Zilion checkout or node")
    prov = (MACHINES_DIR / "PROVENANCE.md").read_text()
    commit = prov.split("commit `", 1)[1].split("`", 1)[0]
    head = subprocess.run(["git", "-C", str(zdir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != commit:
        pytest.skip(f"zilion checkout at {head[:12]}, machines from {commit[:12]}")
    r = subprocess.run([node, str(MACHINES_DIR / "build_machines.mjs"), "--check", "--zilion", str(zdir)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]


# ── (b) the derived executor ────────────────────────────────────────────────────────────────────────────────────────
def test_executor_derivation_regenerates_every_exported_executor():
    import re

    files = sorted(f for f in IW.SHADER_DIR.iterdir() if re.fullmatch(r"z80_test_L\d+(_P\d+)?\.wgsl", f.name))
    assert len(files) > 70
    base = (IW.SHADER_DIR / "z80_test_L18.wgsl").read_text()
    for f in files:
        m = re.fullmatch(r"z80_test_L(\d+)(?:_P(\d+))?\.wgsl", f.name)
        L = int(m[1])
        P = int(m[2] or 2 * L)
        assert I.derive_executor(base, L, P) == f.read_text(), f.name
    text = I.algocell_executor_text(IW.world("L19"))
    assert "const MEM_LENGTH: u32 = 38u;\n" in text and "var<private> mem: array<u32, 40>;\n" in text


# ── (c) classifier ──────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("core,b,diff,pc0,pc1,want", [
    ("c1", (0xDD, 0xDD, 0, 0), {"r": 3}, 5, 6, "wasted-prefix"),
    ("c1", (0xFD, 0xED, 0x5F, 0), {"r": 1}, 5, 6, "wasted-prefix"),
    ("c1", (0xDD, 0xDD, 0, 0), {"r": 3, "a": 1}, 5, 6, "other"),         # a wasted prefix changes nothing but R
    ("c1", (0xDD, 0xDD, 0, 0), {"r": 3}, 5, 7, "other"),
    ("c1", (0xCB, 0x46, 0, 0), {"f": 0x28}, 0, 2, "bit-hl"),
    ("c2", (0xCB, 0x7E, 0, 0), {"f": 0x08}, 0, 2, "bit-hl"),
    ("c1", (0xCB, 0x46, 0, 0), {"f": 0x01}, 0, 2, "other"),              # only F3/F5 may differ
    ("c1", (0xCB, 0x47, 0, 0), {"f": 0x28}, 0, 2, "other"),              # BIT 0,A is not BIT n,(HL)
    ("c1", (0xDD, 0xCB, 0x01, 0x46), {"f": 0x28}, 0, 4, "other"),        # BIT n,(IX+d) is not a documented class
    ("c2", (0x37, 0, 0, 0), {"f": 0x20}, 0, 1, "scf-ccf"),
    ("c2", (0xFD, 0x3F, 0, 0), {"f": 0x28}, 0, 2, "scf-ccf"),
    ("c1", (0x37, 0, 0, 0), {"f": 0x20}, 0, 1, "other"),                 # the Q latch is C2's only
    ("c2", (0x37, 0, 0, 0), {"f": 0x21}, 0, 1, "other"),
    ("c1", (0xED, 0xB0, 0, 0), {"f": 0x20}, 9, 9, "block-repeat"),
    ("c2", (0xED, 0xB9, 0, 0), {"f": 0x08}, 9, 9, "block-repeat"),
    ("c1", (0xED, 0xB2, 0, 0), {"f": 0x3C}, 9, 9, "block-repeat"),       # I/O repeats: H and P/V too
    ("c1", (0xED, 0xB0, 0, 0), {"f": 0x14}, 9, 9, "other"),              # LDIR: only X/Y
    ("c1", (0xED, 0xB0, 0, 0), {"f": 0x20}, 9, 11, "other"),             # the last (non-repeating) iteration
    ("c1", (0xED, 0xA0, 0, 0), {"f": 0x20}, 9, 11, "other"),             # LDI
    ("c1", (0x00, 0, 0, 0), {"memory": 1}, 0, 1, "other"),
    ("c1", (0xED, 0x5F, 0, 0), {"a": 1, "f": 0x80}, 0, 2, "other"),      # LD A,R with equal R before it: never an event
])
def test_classify(core, b, diff, pc0, pc1, want):
    assert I.classify(core, b, diff, pc0, pc1) == want


# ── (d) attribution on constructed programs ─────────────────────────────────────────────────────────────────────────
def _expect(programs, out, res):
    aff = I.affected(res)
    for k in ("c1", "c2"):
        for (name, (_, exp)), r, a in zip(programs.items(), out[k], aff[k]):
            if k in exp:
                assert a, f"{name} must be affected under {k}"
                assert (r.decisive_class, r.decisive_step) == exp[k], (name, k, r.decisive_class, r.decisive_step, [(e.step, e.cls, e.diff) for e in r.events])
                assert all(e.cls != "other" for e in r.events)
            else:
                assert not a and r.decisive_class is None, (name, k)


@pytest.mark.parametrize("tag", ["L16", "L16_P37", "lethal_L16", "lp03_L16", "lp1_L16", "randreg_L16", "randsp_L16"])
def test_attribution_constructed(tag):
    progs = dict(PROGRAMS)
    tr, res, out, at = _attribute(tag, progs)
    if tag == "randsp_L16":
        # a random SP moves the PUSH, which may land on the code and change what runs next: every event must still be of
        # a documented class, and under C1 never the Q latch
        for k in ("c1", "c2"):
            for r in out[k]:
                assert all(e.cls in I.CLASSES[:4] for e in r.events)
                assert not (k == "c1" and r.decisive_class == "scf-ccf")
        return
    _expect(progs, out, res)
    ev = {name: [(e.step, e.cls) for e in r.events] for name, r in zip(progs, out["c2"])}
    assert ev["bit-then-scf"] == [(2, "bit-hl"), (7, "scf-ccf")]           # the transient event is seen, and is not decisive
    assert out["c2"][list(progs).index("bit-then-scf")].propagating == ("bit-hl", "scf-ccf")


def test_attribution_through_the_dial():
    """A NOP before the wasted-prefix program: under p = 0.3 some lanes halt on it (not affected), the others are
    attributed one step later; the twin must agree with the real dial machines lane by lane."""
    prog = np.concatenate([[0x00], PROGRAMS["wasted-prefix"][0][:-1]]).astype(np.uint8)
    pairs = np.repeat(prog[None], 48, axis=0)
    lanes = np.arange(1000, 1048)
    tr = I.Triple("lp03_L16")
    res = _run_lanes(tr, pairs, lanes, 128, None, 4242)
    halted = res["c0"].halt_reason == 2
    assert 5 < halted.sum() < 43                       # the dial acts on some lanes only
    at = I.Attributor(tr, 128, None, 4242)
    out = at.attribute(pairs, lanes, res)
    for k in ("c1", "c2"):
        for h, r in zip(halted, out[k]):
            assert (r.decisive_class, r.decisive_step) == ((None, None) if h else ("wasted-prefix", 2))
    assert at.verify_stepping(pairs, lanes, res) == 48


def test_masks_change_the_outcome():
    """With SCF/CCF ablated (Algocell's suppression masks in the machines' table) the Q-latch pair is no longer affected."""
    from algocell_exp.isa import masks, resolve

    m = masks(resolve(["SCF", "CCF"]))
    progs = {"scf-ccf": PROGRAMS["scf-ccf"]}
    tr, res, out, _ = _attribute("L16", progs, masks=m)
    assert not I.affected(res)["c2"].any()


def test_other_stops_the_study(monkeypatch):
    monkeypatch.setattr(I, "classify", lambda *a, **k: "other")
    with pytest.raises(I.AttributionError, match="other"):
        _attribute("L16", {"bit-hl": PROGRAMS["bit-hl"]})


def test_failed_check_stops_the_study():
    tr = I.Triple("L16")
    pairs = PROGRAMS["bit-hl"][0][None]
    res = tr.run(pairs, 128)
    res["c0"].memory[:] ^= 1          # a single-dispatch result the stepping cannot reproduce
    with pytest.raises(I.AttributionError, match="reproduce"):
        I.Attributor(tr, 128).attribute(pairs, np.arange(1), res)


# ── (e) stepping reproduces single dispatches ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("tag,masked", [("L16", True), ("L16_P37", False), ("lethal_L16", False), ("lp03_L16", False), ("randreg_L16", False), ("randsp_L16", False), ("L9", False)])
def test_stepping_reproduces(tag, masked):
    tr = I.Triple(tag)
    rng = np.random.default_rng(17)
    pairs = rng.integers(0, 256, size=(256, 2 * tr.L), dtype=np.uint8)
    pairs[rng.random(pairs.shape) < 0.15] = 0
    m = np.packbits(rng.random((24, 32)) < 1 / 6, axis=1, bitorder="little").view(np.uint32).reshape(24) if masked else None
    res = tr.run(pairs, 64, m, seed=5, lane_base=0)
    at = I.Attributor(tr, 64, m, 5)
    assert at.verify_stepping(pairs, np.arange(256), res) == 256


# ── (f) statistics ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_clopper_pearson():
    lo, hi = I.clopper_pearson(0, 1000)
    assert lo == 0 and abs(hi - (1 - 0.025 ** (1 / 1000))) < 1e-9
    lo, hi = I.clopper_pearson(1000, 1000)
    assert hi == 1 and abs(lo - 0.025 ** (1 / 1000)) < 1e-9
    lo, hi = I.clopper_pearson(5, 100)
    assert abs(lo - 0.016431) < 1e-5 and abs(hi - 0.112835) < 1e-5
    lo, hi = I.clopper_pearson(3, 100000)
    assert lo < 3e-5 < hi and hi < 1e-4
    a, b = I.clopper_pearson(30, 70)
    c, d = I.clopper_pearson(40, 70)
    assert abs(a - (1 - d)) < 1e-9 and abs(b - (1 - c)) < 1e-9


# ── (g) corpora ────────────────────────────────────────────────────────────────────────────────────────────────────
def test_conditions_registry():
    cs = K.conditions()
    assert len({c.cid for c in cs}) == len(cs)
    fam = Counter(c.family for c in cs)
    for f in ("stageA", "stageB", "stageC", "stageD", "stageE", "stageF", "lod_v6", "N6", "X-inputs", "stageG", "stageI", "DZ", "C", "E"):
        assert fam[f] > 0, f
    assert {c.tier for c in cs if c.family in K.STAGES_DRAFT} == {"draft"}
    for c in cs:
        assert c.world in IW.WORLDS and IW.world(c.world).L == c.L
    h = [c for c in cs if c.family == "stageH"]
    assert h and all(c.grid == "mixed" for c in h)
    i = [c for c in cs if c.family == "stageI"]
    assert i and all(c.world == "lethal_L16" for c in i)


def test_slots_are_partners():
    c = next(c for c in K.conditions() if c.family == "stageG")
    s = K.draw_slots(c, 5000, K.REG_STREAM)
    w = c.sources[0].width
    xi, yi, xj, yj = s[:, 1] % w, s[:, 1] // w, s[:, 2] % w, s[:, 2] // w
    assert (np.abs(xi - xj) + np.abs(yi - yj) == 1).all()
    assert np.array_equal(s, K.draw_slots(c, 5000, K.REG_STREAM))        # deterministic
    m = next(c for c in K.conditions() if c.grid == "mixed")
    s = K.draw_slots(m, 5000, K.REG_STREAM)
    assert (s[:, 1] != s[:, 2]).all() and s[:, 2].max() < 20000


def test_dry_run_is_excluded_from_registered_draws():
    cs = K.conditions()
    picks = [next(c for c in cs if c.family == f) for f in ("stageA", "lod_v6", "X-inputs", "DZ")]
    picks.append(min((c for c in cs if c.family == "N5"), key=lambda c: c.n_sources))   # few slots: collisions are likely
    for c in picks:
        reg = {tuple(r) for r in K.draw_slots(c, 100000, K.REG_STREAM).tolist()}
        dry, redrawn = K.dry_slots(c, 1000, 100000)
        assert len(dry) == 1000 and not ({tuple(r) for r in dry.tolist()} & reg)
    rc = K.random_conditions()[0]
    reg = {bytes(r) for r in K.draw_random(rc, 100000, K.REG_STREAM)}
    dry, _ = K.dry_random(rc, 1000, 100000)
    assert not ({bytes(r) for r in dry} & reg)
    # the registered draws do not depend on the dry run, and the streams differ
    assert np.array_equal(K.draw_random(rc, 10, K.REG_STREAM), K.draw_random(rc, 10, K.REG_STREAM))
    assert not np.array_equal(K.draw_random(rc, 10, K.REG_STREAM), K.draw_random(rc, 10, K.DRY_STREAM))


# ── the GPU-resident path equals the reference path ─────────────────────────────────────────────────────────────────
def _summary(results, affected):
    return [(r.decisive_class, r.decisive_step, r.propagating, [(e.step, e.cls, e.insn, e.diff, e.probe_differs) for e in r.events])
            for r, a in zip(results, affected) if a]


@pytest.mark.parametrize("tag", ["L16", "L16_P37", "lethal_L16", "randreg_L16", "randsp_L16", "L36_P79"])
def test_fast_attributor_equals_reference(tag):
    tr = I.Triple(tag)
    rng = np.random.default_rng(23)
    rand = rng.integers(0, 256, size=(2000, 2 * tr.L), dtype=np.uint8)
    pairs = np.concatenate([np.stack([np.resize(p, 2 * tr.L) if tr.L != L else p for p, _ in PROGRAMS.values()]), rand])
    res = tr.run(pairs, 128, None, seed=31, lane_base=0)
    aff = I.affected(res)
    assert aff["c1"].any() and aff["c2"].any()
    lanes = np.arange(len(pairs))
    fast = I.FastAttributor(tr, 128, None, 31).attribute(pairs, lanes, res, census=True)
    idx = np.nonzero(aff["c1"] | aff["c2"])[0]
    ref = I.Attributor(tr, 128, None, 31).attribute(pairs[idx], idx, {k: I._subset(v, idx) for k, v in res.items()})
    for k in ("c1", "c2"):
        assert _summary([fast[k][i] for i in idx], aff[k][idx]) == _summary(ref[k], aff[k][idx])
        assert all(e.cls != "other" for r in fast[k] for e in r.events)      # census: every event of every lane is documented


def test_world_code_identical_outside_the_core():
    """Each machine's WGSL is (world code) + (the core text, located by its SHA-256) + (world code). Across a world's three
    machines the world code is identical except for the header, the state and record sizes, the reset, store and load of
    the fixed cores' own latches (WZ, IM, Q) and the record offsets those shift."""
    import hashlib
    import re

    zdir = MACHINES_DIR.parents[2].parent / "zilion"
    node = shutil.which("node")
    if not node or not (zdir / "dist/index.js").exists():
        pytest.skip("no Zilion checkout or node")
    script = ("import(process.argv[1]).then(Z => { const out = {}; for (const h of [[], ['instructionStart']]) for (const [k, o] of "
              "[['c0', {compat: '0.2'}], ['c1', {compat: '0.3', variant: 'classic'}], ['c2', {compat: '0.3', variant: 'zilog-nmos'}]]) "
              "out[k + ':' + h.join(',')] = Z.z80CoreText(Object.fromEntries(h.map(x => [x, true])), o); "
              "process.stdout.write(JSON.stringify(out)); });")
    cores = json.loads(subprocess.run([node, "-e", script, str(zdir / "dist/index.js")], capture_output=True, text=True, check=True).stdout)
    hidden = re.compile(r"\b(cpu_wz|cpu_im|cpu_q|z80_qprev)\b")

    def world_code(text: str) -> str:
        keep = []
        for ln in text.splitlines():
            if ln.startswith(("// core ", "// options sha256", "const ZIL_STATE_WORDS", "const ZIL_RECORD_WORDS")) or hidden.search(ln):
                continue
            keep.append(re.sub(r"zil_out\[o \+ \d+u", "zil_out[o + Nu", ln))
        return "\n".join(keep)

    for w in IW.WORLDS.values():
        parts = {}
        for k in IW.CORES:
            man = json.loads((MACHINES_DIR / f"{w.machine(k)}.manifest.json").read_text())
            core = cores[f"{k}:{','.join(man['core']['hooks'])}"]
            assert hashlib.sha256(core.encode()).hexdigest() == man["sha256"]["core"], (w.tag, k)
            text = (MACHINES_DIR / f"{w.machine(k)}.wgsl").read_text()
            assert text.count(core) == 1, (w.tag, k)
            i = text.index(core)
            parts[k] = (world_code(text[:i]), world_code(text[i + len(core):]))
        assert parts["c0"] == parts["c1"] == parts["c2"], w.tag
