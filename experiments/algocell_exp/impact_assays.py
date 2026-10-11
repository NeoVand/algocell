"""I1 corpus 3: named manuscript genotypes, source rows, and exact assay identity.

The panel includes individual genotypes in the manuscript's cited genotype tables,
not the anonymous cells behind population/offset aggregates. No score is rounded.
All assays use assay.assay's complete output, 64 partners, seed 0, 128 steps.
Sources retain the published rule/mask; stochastic worlds use fresh, reproducible
case-index streams per assay call, shared by the three cores.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from . import assay as A, executor as E, impact_corpus as K
from .impact_worlds import CORES, WORLDS

HERE = Path(__file__).resolve().parent.parent
# Each is cited in NUMBERS_INDEX, or expands a named panel/table through that index.
TABLES = (
    "stageG/stageG/stage_g_runs.csv", "stageI/stageI/stage_g_runs.csv",
    "stageK/stageK/stage_g_runs.csv", "stageL/stageL/stage_g_runs.csv",
    "stageM/stageM/stage_g_runs.csv", "exectrace/per_replicator.csv",
    "mutscan/mutscan_tapes.csv", "biology/individuality/per_replicator.csv",
    "serial_retention/sr_summary.csv", "branching/branching_summary.csv",
    "pe/pe_matrix.csv", "lod/founding.csv", "lod/founders2.csv",
    "lod_n6/founding.csv", "lod_n6/founders2.csv",
    "conv/conv_worlds.csv", "dial/dial_worlds.csv",
    "r4/r4_long.csv", "r4/r4_rep.csv", "review_r2/census_p2.csv",
)
TAPE_COLUMNS = {"tape", "tape_hex", "first_tape", "final_tape", "F_tape",
                "last_open_tape", "executor", "producer_tape", "modal"}


def _hex(value):
    return bool(re.fullmatch(r"[0-9a-fA-F]{2}(?: [0-9a-fA-F]{2})+", value.strip()))


def panel():
    entries = {}
    main = (HERE / "manuscript/MAIN_nature.md").read_text().splitlines()
    index = (HERE / "manuscript/NUMBERS_INDEX.md").read_text().splitlines()

    def add(tape, world, suppress, source, name, refs):
        tape = bytes.fromhex(tape).hex(" ")
        L = len(bytes.fromhex(tape))
        assert world in WORLDS and WORLDS[world].L == L, (world, L, source)
        key = (tape, world, tuple(suppress))
        if key not in entries:
            aid = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
            entries[key] = dict(aid=aid, tape=tape, L=L, world=world, suppress=list(suppress),
                                budget=128, n=64, seed=0, sources=[], names=[], references=[])
        e = entries[key]
        for field, values in (("sources", [source]), ("names", [name]), ("references", refs)):
            e[field] = sorted(set(e[field] + values))

    for rel in TABLES:
        p = HERE / "results" / rel
        if not p.exists():
            raise FileNotFoundError(p)
        refs = [f"NUMBERS_INDEX.md:{i}" for i, line in enumerate(index, 1)
                if rel in line or "results/" + rel.split("/")[0] + "/" in line]
        assert refs, rel
        with p.open() as fh:
            for rowno, row in enumerate(csv.DictReader(fh), 2):
                if row.get("machine", "Z80") not in ("Z80", "z80"):
                    continue
                for col in sorted(TAPE_COLUMNS & row.keys()):
                    value = row[col]
                    if not _hex(value):
                        continue
                    L = len(bytes.fromhex(value))
                    lethal = row.get("zero_halts", "").lower() == "true" or "lethal" in row.get("rule", "").lower()
                    if rel.startswith("stageI/") or (rel.startswith("mutscan/") and row.get("stage") == "I"):
                        lethal = True
                    if rel.startswith("biology/") and "lethal" in row.get("world", "").lower():
                        lethal = True
                    variant = row.get("variant", "") if rel.startswith(("conv/", "dial/")) else ""
                    world = (variant + f"_L{L}" if variant else f"{'lethal_' if lethal else ''}L{L}")
                    # PE names genotypes in several environments. The standard and
                    # lethal replication assays are both retained; conventions too.
                    env = row.get("environment", "")
                    if rel.startswith("pe/"):
                        if "randreg" in env or "random registers" in env:
                            world = f"randreg_L{L}"
                        elif "randsp" in env or "random stack" in env:
                            world = f"randsp_L{L}"
                        elif "lethal" in env:
                            world = f"lethal_L{L}"
                    if rel.startswith("r4/"):
                        world = f"randreg_L{L}"
                    sup = K.I8080_SUPPRESS if "8080" in row.get("rule", "") or rel.startswith("stageM/") or row.get("stage") == "M" else ()
                    name = "/".join(row.get(x, "") for x in ("key", "parent", "seed", "which") if row.get(x)) or col
                    add(value, world, sup, f"results/{rel}:{rowno}:{col}", name, refs)

    # Full named tilings, including the null and memory-load words of Proposition 3.
    for word in ("00 00", "01 c5", "11 d5", "21 e5", "2a e5", "e5 2a", "e1 e3"):
        add(" ".join([word] * 8), "L16", (), "MAIN_nature.md:276 (Proposition 3); :232 (stack move)", word + " × 8",
            ["NUMBERS_INDEX.md (census2 and harness conventions)"])
    add(" ".join(["1e a4 ed b0"] * 5), "L20", (), "MAIN_nature.md:408 (period-four example)", "1e a4 ed b0 × 5", ["MAIN_nature.md:408"])
    # Two explicitly named hand-built designs with unspecified random tails: keep
    # the core and generate the tail deterministically, as documented in the table.
    for core in ("21 00 00 11 10 00 01 10 00 ed b0 18 fe", "2e 00 1e 10 ed b0"):
        rng = np.random.default_rng(0)
        t = bytes.fromhex(core) + rng.integers(0, 256, 16 - len(bytes.fromhex(core)), dtype=np.uint8).tobytes()
        for world in ("L16", "randreg_L16"):
            add(t.hex(" "), world, (), "MAIN_nature.md:232,:242; unspecified tail drawn with numpy seed 0", core + " + seed-0 tail",
                ["NUMBERS_INDEX.md (Parent × environment); REVISION_PREREG.md R4/C4b,C4c"])
    # Every truncated tape printed in Extended Data Table 1 resolves to a full row.
    for lineno, line in enumerate(main, 1):
        if not line.startswith("| "):
            continue
        for prefix in re.findall(r"`([0-9a-f ]+) …`", line):
            matches = [e for e in entries.values() if e["tape"].startswith(prefix + " ")]
            assert matches, (lineno, prefix)
            for e in matches:
                e["references"] = sorted(set(e["references"] + [f"MAIN_nature.md:{lineno} (Extended Data Table 1)"]))
    return sorted(entries.values(), key=lambda e: (e["world"], e["aid"]))


class ImpactExecutor(E.ZilionExecutor):
    """Explicit I1 adapter for the registered stochastic worlds, through E.using.

    Ordinary callers keep ZilionExecutor's refusals. This adapter only accepts
    exact registered imp_* machines and verifies their frozen world definition.
    """
    def __init__(self, world, core, aid):
        self.world, self.aid, self.call = WORLDS[world], aid, 0
        super().__init__(self.world.machine(core))

    def machine_for(self, L, mem_length=None, zero_halts=False, maps=False):
        if self.world.rule in ("plain", "lethal"):
            return super().machine_for(L, mem_length, zero_halts, maps)
        from .zmachine import load_machine
        assert not maps and L == self.world.L and (mem_length or 2 * L) == self.world.P and not zero_halts
        m = self._cache.setdefault((self.spec, False), load_machine(self.spec))
        expected = json.loads((HERE / "algocell_exp/machines/worlds_impact.json").read_text())
        spec = expected[self.spec]
        assert m.m["world"] == spec["world"], self.spec
        return m

    def run(self, pairs, L, steps, suppress=(), *, mem_length=None, zero_halts=False, fetch_map=False, start_map=False):
        if self.world.rule in ("plain", "lethal"):
            return super().run(pairs, L, steps, suppress, mem_length=mem_length, zero_halts=zero_halts,
                               fetch_map=fetch_map, start_map=start_map)
        pairs = np.ascontiguousarray(pairs, dtype=np.uint8)
        assert pairs.ndim == 2 and pairs.shape[1] == 2 * L
        m = self.machine_for(L, mem_length, zero_halts, fetch_map or start_map)
        self.call += 1
        seed = K.h32(f"I1-assay/{self.aid}/{self.call}")
        r = m.run(pairs, budget=int(steps), ablation=self.ablation_words(m, suppress), seed=seed, lane_base=0)
        E._USAGE[("machine", m.name)] += len(pairs)
        writes = np.stack([r.writes("A"), r.writes("B")], axis=1).astype(np.uint32)
        return E.PairRun(r.memory, writes, None, None, r.steps, r.halt_reason, r.trap_cause, m.name)


def assay_item(item):
    outputs = {}
    for core in CORES:
        ex = ImpactExecutor(item["world"], core, item["aid"])
        with E.using(ex):
            outputs[core] = A.assay(bytes.fromhex(item["tape"]), z80_steps=item["budget"],
                                   suppress=item["suppress"], n=item["n"], seed=item["seed"],
                                   mem_length=WORLDS[item["world"]].P,
                                   zero_halts=WORLDS[item["world"]].rule == "lethal")
    return {"aid": item["aid"], "outputs": outputs, **identity(outputs)}


def equal(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True  # the same undefined score, not a numeric tolerance
    return type(a) is type(b) and a == b


def identity(outputs):
    columns = set(outputs["c0"])
    if any(set(outputs[c]) != columns for c in CORES):
        raise ValueError("assay output columns differ")
    diff = {c: sorted(k for k in columns if not equal(outputs["c0"][k], outputs[c][k])) for c in ("c1", "c2")}
    return {"c1_identical": not diff["c1"], "c2_identical": not diff["c2"],
            "all_identical": not any(diff.values()), "differing_columns": diff}
