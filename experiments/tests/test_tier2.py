"""Preparation regression checks: frozen default output and explicit world routing (no GPU)."""
import json
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
BASE = "0970b82"


class FakeSoup:
    def __init__(self, width, height, grid, L, seed, *args, **kwargs):
        self.cell_count = width * height
        self.cells = np.random.default_rng(seed).integers(0, 256, (self.cell_count, L), dtype=np.uint8)

    def read_soup(self):
        return self.cells.copy()

    def write_soup(self, cells):
        self.cells = cells.copy()

    def step(self, n):
        self.cells[:, 0] = (self.cells[:, 0].astype(int) + n) % 256


@pytest.mark.parametrize("script,args", [
    ("dial_soups.py", ("lp01", 424242, 100)),
    ("offset_switch.py", ({"L": 16, "tar": "lethal", "mut": "on", "start": "mix50", "seed": 424242, "steps": 100},)),
    ("invasion_generic.py", ({"L": 16, "variant": "randreg", "resident": "pusher", "invader": "SI13", "seed": 424242, "steps": 100},)),
])
def test_default_output_bytes_identical(script, args, monkeypatch, tmp_path):
    import algocell_exp.soup as S
    monkeypatch.setattr(S, "Soup", FakeSoup)
    monkeypatch.setattr("time.time", lambda: 1234.5)
    sources = [subprocess.check_output(["git", "show", f"{BASE}:experiments/{script}"], cwd=HERE).decode(),
               (HERE / script).read_text()]
    outputs = []
    for i, source in enumerate(sources):
        module = types.ModuleType(f"frozen_{i}")
        module.__file__ = str(HERE / script)
        exec(compile(source, str(HERE / script), "exec"), module.__dict__)
        root = tmp_path / str(i)
        result = module.run_world(*args, str(root))
        outputs.append((result, {p.name: p.read_bytes() for p in root.iterdir()}))
    assert outputs[0] == outputs[1]


def test_complete_world_twins_and_hashes():
    import hashlib
    from algocell_exp.tier2 import machine_for
    specs = json.loads((HERE / "algocell_exp/machines/worlds_tier2.json").read_text())
    for name, spec in specs.items():
        root = HERE / "algocell_exp/machines"
        m = json.loads((root / f"{name}.manifest.json").read_text())
        assert hashlib.sha256((root / f"{name}.wgsl").read_bytes()).hexdigest() == m["sha256"]["wgsl"]
        assert spec["world"] == m["world"]
        twin = name.replace("c0", "c1") if "c0" in name else name.replace("c1", "c0")
        assert spec["world"] == specs[twin]["world"]
    for tag in ("L16", "L20", "L8_P32", "lp01_L16", "randreg_L16", "lethal_L16"):
        for core in ("c0", "c1"):
            assert machine_for(core, tag)
            assert machine_for(core, tag, maps=True)
    with pytest.raises(ValueError, match="does not match"):
        machine_for("imp_L16_c1", "randreg_L16")
    with pytest.raises(ValueError, match="scoring needs"):
        machine_for("imp_lp01_L16_c1", "lp01_L16", maps=True)


def test_full_design_preserves_registered_units():
    from tier2_prepare import design, frozen_manifest
    from collections import Counter
    jobs = design()
    assert Counter(j["experiment"] for j in jobs) == {"DZ": 60, "E": 110, "L": 20, "R4": 32, "F": 10}
    assert sum(j["steps"] for j in jobs) == 250400000
    for p in ("lp0", "lp001", "lp003", "lp01", "lp03", "lp1"):
        assert sorted(j["seed"] for j in jobs if j["experiment"] == "DZ" and j["variant"] == p) == list(range(9001, 9011))
    for L in (16, 20):
        group = [j for j in jobs if j["experiment"] == "L" and j["condition"]["tape"] == L]
        assert sorted(j["condition"]["seed"] for j in group) == list(range(6001, 6011))
        assert all(j["steps"] == 10000000 for j in group)
    assert all(r["identical"] for r in frozen_manifest()["frozen_numerical_functions"])


def test_scorer_rejects_different_world_core():
    from algocell_exp.tier2 import check_run_core
    check_run_core({"machine": {"machine": "imp_randreg_L16_c1"}}, "c1", "randreg_L16")
    with pytest.raises(ValueError, match="run metadata"):
        check_run_core({"machine": {"machine": "imp_randreg_L16_c0"}}, "c1", "randreg_L16")
