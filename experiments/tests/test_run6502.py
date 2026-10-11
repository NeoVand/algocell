"""Registered recording/refusal checks; all GPU work is mocked."""

import json
from pathlib import Path
from unittest.mock import Mock

import brotli
import numpy as np
import pytest

import modal_s6502 as M
import run6502 as R


def test_schedule_boundaries_and_count():
    expected = (list(range(50, 2001, 50)) + list(range(2250, 20001, 250))
                + list(range(22500, 1000001, 2500)) + list(range(1025000, 10000001, 25000)))
    assert R.sample_steps() == expected
    assert len(expected) == 864 and len(set(expected)) == 864
    assert R.sample_steps(30000) == expected[:116]
    assert R.sample_steps(51) == [50]


def test_exact_top3_keeps_hash_collisions_separate():
    cells = np.array([[9, 9], [1, 1], [9, 9], [2, 2], [1, 1], [9, 9]], dtype=np.uint8)
    stats, top = R.exact_tapes(cells, np.ones(6, dtype=np.uint32))
    assert stats["unique"] == 3
    assert [t["tape"] for t in stats["top3"]] == ["0909", "0101", "0202"]
    assert [t["share"] for t in stats["top3"]] == [3 / 6, 2 / 6, 1 / 6]
    assert np.array_equal(top, [9, 9])


class FakeSoup:
    def __init__(self, machine, **kwargs):
        self.width = self.height = 2
        self.cell_count = 4
        self.pair_count = 8192
        self.noise_exp = 4
        self.machine_provenance = {"machine": machine}
        L = 32 if machine.endswith("L32") else 16
        self.cells = np.array([[1] * L, [2] * L, [1] * L, [3] * L], dtype=np.uint8)
        self.step_count = 0

    def step(self, n):
        self.step_count += n

    def sync(self):
        pass

    def read_soup(self):
        return self.cells

    def read_hashes(self):
        return np.array([1, 2, 1, 3], dtype=np.uint32)

    def read_byte_counts(self):
        return np.bincount(self.cells.ravel(), minlength=256)

    def read_active_pairs(self):
        return 4000


@pytest.mark.parametrize("arm", list(R.ARMS))
def test_records_snapshots_commits_and_overwrite_refusal(tmp_path, monkeypatch, arm):
    build = Mock(side_effect=FakeSoup)
    monkeypatch.setattr(R, "ZSoup", build)
    monkeypatch.setattr(R, "adapter_summary", lambda: "mock")
    commit = Mock()
    summary = R.run_world(arm, 999, tmp_path, steps=30000, git_commit="a" * 40, commit_output=commit)
    records = [json.loads(line) for line in (tmp_path / f"{R.world_name(arm, 999)}.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records if r["kind"] == "sample"] == R.sample_steps(30000)
    assert summary["samples"] == 116 and summary["snapshots"] == 2
    assert commit.call_count == 4  # condition, two checkpoints, summary
    assert build.call_args.kwargs["z80_steps"] == R.ARMS[arm]["budget"]
    for step in (20000, 30000):
        assert len(brotli.decompress((tmp_path / f"{R.world_name(arm, 999)}.t{step}.u8.br").read_bytes())) == 4 * R.ARMS[arm]["L"]
    assert R.existing_summary(tmp_path / f"{R.world_name(arm, 999)}.jsonl") == summary
    with pytest.raises(FileExistsError):
        R.run_world(arm, 999, tmp_path, steps=30000, git_commit="a" * 40)
    assert build.call_count == 1


def test_incomplete_and_orphan_worlds_are_retained(tmp_path, monkeypatch):
    build = Mock(side_effect=AssertionError("GPU must not run"))
    monkeypatch.setattr(R, "ZSoup", build)
    path = tmp_path / "s6502_A_s999.jsonl"
    path.write_text('{"kind": "condition"}\n{"kind":')
    assert R.existing_summary(path) is None
    with pytest.raises(FileExistsError):
        R.run_world("A", 999, tmp_path, git_commit="a" * 40)
    orphan = tmp_path / "s6502_B_s999.t20000.u8.br"
    orphan.write_bytes(b"orphan")
    with pytest.raises(FileExistsError):
        R.run_world("B", 999, tmp_path, git_commit="a" * 40)
    build.assert_not_called()


def test_full_registered_horizon_has_exactly_five_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ZSoup", FakeSoup)
    monkeypatch.setattr(R, "adapter_summary", lambda: "mock")
    summary = R.run_world("C", 65001, tmp_path, git_commit="a" * 40)
    records = [json.loads(line) for line in (tmp_path / "s6502_C_s65001.jsonl").read_text().splitlines()]
    assert summary["steps_run"] == 10000000 and summary["samples"] == 864
    assert [r["step"] for r in records if r["kind"] == "snapshot"] == [20000, 100000, 1000000, 3000000, 10000000]


def test_modal_reload_skip_and_incomplete_reporting(monkeypatch):
    volume = Mock()
    run = Mock(side_effect=AssertionError("must not run existing worlds"))
    monkeypatch.setattr(M, "vol", volume)
    monkeypatch.setattr(R, "run_world", run)
    monkeypatch.setattr(R, "existing_summary", lambda path: {"kind": "summary"})
    assert M.world.local("A", 65001, "a" * 40)["status"] == "skipped"
    volume.reload.assert_called_once()
    monkeypatch.setattr(R, "existing_summary", lambda path: None)
    monkeypatch.setattr(Path, "glob", lambda *args: iter([Path("incomplete.jsonl")]))
    assert M.world.local("B", 65001, "a" * 40)["status"] == "incomplete"
    assert volume.reload.call_count == 2
    run.assert_not_called()


def test_modal_selection_separates_smoke_and_registered():
    assert M.parse_selection("A,C", "65001-65020", False) == (["A", "C"], list(R.SEEDS))
    assert M.parse_selection("A B C D", "999", True) == (list(R.ARMS), [999])
    for arms, seeds, smoke in (("A", "65001", True), ("A", "999", False),
                               ("A,A", "999", True), ("A", "999,999", True),
                               ("AB", "999", True), ("A", "65020-65001", False)):
        with pytest.raises(ValueError):
            M.parse_selection(arms, seeds, smoke)
