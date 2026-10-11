"""Archive coverage regressions for the impact registry; CPU-only, no GPU dispatches."""

from collections import Counter
import json

import numpy as np

from algocell_exp import impact_corpus as K
from algocell_exp import impact_worlds as W
from make_conds import ABLATIONS


def test_ring_sweep_registers_every_final_and_emergence_archive():
    cs = [c for c in K.conditions() if c.family == "ring_sweep"]
    directory = K.RUNS / "ring_sweep"
    expected = set(directory.glob("*.soup_final.u8.br")) | set(directory.glob("*.soup_emergence.u8.br"))
    actual = [s.path for c in cs for s in c.sources]
    assert expected and set(actual) == expected and len(actual) == len(expected)
    assert Counter(c.snapshot for c in cs) == {"final": 52, "emergence": 45}
    assert Counter(c.snapshot for c in cs for _ in c.sources) == {"final": 156, "emergence": 130}
    assert len(list(directory.glob("*.u8.br"))) == 1690  # the other 1,404 are scheduled intermediate snapshots
    for c in cs:
        assert c.tier == "added" and c.grid == "square" and c.budget == 128 and not c.suppress
        assert "Exploratory" in c.note and "Fig. 3" in c.note
        assert c.P == W.world(c.world).P and c.L == W.world(c.world).L
    assert not any(c.family == "ring_sweep" for c in K.conditions(tiers=("draft",)))


def test_ring_sweep_worlds_have_three_cores_and_exported_executors():
    cs = [c for c in K.conditions() if c.family == "ring_sweep"]
    actual = {(c.L, c.P) for c in cs}
    assert actual == {(L, P) for L, Ps in W.RING_SWEEP_RINGS.items() for P in Ps}
    assert len(actual) == 52
    for tag in {c.world for c in cs}:
        w = W.world(tag)
        filename, derivation = w.algocell_executor()
        assert derivation is None and (W.SHADER_DIR / filename).is_file()
        manifests = [json.loads((W.MACHINES_DIR / f"{w.machine(core)}.manifest.json").read_text()) for core in W.CORES]
        assert len({m["sha256"]["world"] for m in manifests}) == 1
        assert [(m["core"]["compat"], m["core"]["variant"]) for m in manifests] == [
            ("0.2", "classic"), ("0.3", "classic"), ("0.3", "zilog-nmos")]


def test_r2_registers_all_invasion_closer_arms_with_their_original_masks():
    cs = [c for c in K.conditions() if c.family == "R2"]
    expected = set((K.RUNS / "invasion_closer").glob("*_final.npy"))
    actual = [s.path for c in cs for s in c.sources]
    assert len(expected) == 32 and set(actual) == expected and len(actual) == len(expected)
    assert len(cs) == 8 and Counter(c.n_sources for c in cs) == {3: 4, 5: 4}
    assert K.I8080_SUPPRESS == tuple(ABLATIONS["i8080"])
    for c in cs:
        assert c.world == "L32" and c.L == 32 and c.P == 64 and c.budget == 128
        assert c.grid == "square" and c.snapshot == "final" and c.tier == "added"
        want = tuple(ABLATIONS["i8080"]) if c.cid.startswith("R2/A1_") else ()
        assert c.suppress == want
        for source in c.sources:
            a = np.load(source.path, mmap_mode="r")
            assert a.shape == (20000, 32) and a.dtype == np.uint8


def test_added_registry_dry_draws_keep_registered_slots_excluded():
    cs = K.conditions()
    for family in ("ring_sweep", "R2"):
        c = next(c for c in cs if c.family == family)
        registered = {tuple(row) for row in K.draw_slots(c, 100000, K.REG_STREAM).tolist()}
        dry, _ = K.dry_slots(c, 1000, 100000)
        assert dry.shape == (1000, 3) and not registered.intersection(map(tuple, dry.tolist()))
