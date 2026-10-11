"""EXPLORATORY (NOT REGISTERED): reproduce S1 draws and diagnose ring returns/d.

Run with the existing experiments Python environment, from this worktree:
  /Users/neo/repos/algocell/experiments/.venv/bin/python experiments/explore_s1_6502.py
  ... experiments/explore_s1_6502.py --report-only

Only committed scorer output tables are read. The frozen scorer and registered
outputs are never written. No volume downloads or host-side ISA interpreter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import score6502 as S
from algocell_exp import assay as A, executor as E
from algocell_exp.soup import adapter_summary

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results/s6502"
LABEL = "EXPLORATORY (NOT REGISTERED)"
MACHINE = "m6502_pair_maps_L16"
L, P, STEPS = 16, 32, 128
BRANCHES = {"BCC", "BCS", "BEQ", "BMI", "BNE", "BPL", "BVC", "BVS"}
CONTROL = BRANCHES | {"JMP", "JSR", "RTS", "RTI"}
FILES = {
    "opcodes": "exploratory_opcodes.csv",
    "draws": "exploratory_draws.csv",
    "partners": "exploratory_partners.csv",
    "steps": "exploratory_steps.csv",
    "encounters": "exploratory_encounters.csv",
    "classes": "exploratory_classification.csv",
    "disassembly": "exploratory_disassembly.csv",
    "offsets": "exploratory_offsets.csv",
    "distribution": "exploratory_offset_distribution.csv",
    "provenance": "exploratory_provenance.csv",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def committed(path):
    """Reject an edited registered table/source, rather than inspecting it."""
    rel = Path(path).relative_to(ROOT)
    data = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT)
    assert data == Path(path).read_bytes(), f"not committed: {rel}"
    return data


def write_table(key, rows):
    df = pd.DataFrame(rows)
    df.insert(0, "analysis", LABEL)
    df.to_csv(OUT / FILES[key], index=False)


def load_table(key):
    df = pd.read_csv(OUT / FILES[key], keep_default_na=False)
    assert (df.analysis == LABEL).all()
    for column in ("copy_offset", "replayed_copy_offset", "pc_after", "ring_after"):
        if column in df:
            df[column] = df[column].map(lambda value: "" if value == "" else int(float(value)))
    return df


def descriptor(zilion):
    entry = zilion / "dist/cores/m6502/index.js"
    js = """import {m6502, M6502_OPCODES} from './dist/cores/m6502/index.js';
console.log(JSON.stringify(m6502.table().map(x => ({...x, mode: M6502_OPCODES[x.key].mode}))));"""
    rows = json.loads(subprocess.check_output(["node", "--input-type=module", "-e", js], cwd=zilion))
    assert len(rows) == 256 and [r["key"] for r in rows] == list(range(256))
    assert all(r["mnemonic"] for r in rows)
    write_table("opcodes", [{**r, "families": ";".join(r["families"])} for r in rows])
    return rows, entry


class Draws:
    """Log actual calls made by frozen judge_tape; add no random draws."""
    def __init__(self, seed):
        self.rng = np.random.default_rng([seed, 0, 11])
        self.assay_seed = None
        self.partners = None

    def integers(self, *args, **kwargs):
        value = self.rng.integers(*args, **kwargs)
        if np.ndim(value) == 0:
            self.assay_seed = int(value)
        else:
            assert value.shape == (16, L)
            self.partners = value.copy()
        return value


class StateReplay(E.ZilionExecutor):
    """pc_trace's identical-pair prefix replay, retaining its full ZResult.

    Same reset, machine, pairs and budgets as executor.pc_trace; memory at each
    prefix supplies the next opcode even if the tape has modified itself.
    """
    def pc_trace(self, pairs, L, steps, *, zero_halts=False):
        m = self.machine_for(L, zero_halts=zero_halts)
        assert m.budget_per_lane and L == 16 and steps == STEPS
        self.pairs = np.ascontiguousarray(pairs, dtype=np.uint8).copy()
        self.replay = m.run(np.repeat(self.pairs, steps + 1, axis=0),
                            budget=np.tile(np.arange(steps + 1), len(pairs)))
        return (self.replay.reg("pc").reshape(len(pairs), steps + 1) & 0xFFFF).astype(np.uint16)


def disassemble(op, operands, pc):
    """Formatting only; all opcode names/modes/lengths come from zilion dist."""
    name, mode = op["mnemonic"], op["mode"]
    b = operands[0] if operands else None
    word = operands[0] | (operands[1] << 8) if len(operands) == 2 else None
    byte = f"${b:02X}" if b is not None else "??"
    addr = f"${word:04X}" if word is not None else "????"
    if mode == "imp": suffix = ""
    elif mode == "acc": suffix = " A"
    elif mode == "imm": suffix = f" #{byte}"
    elif mode == "rel":
        displacement = b if b < 128 else b - 256
        target = (pc + 2 + displacement) & 0xFFFF
        suffix = f" {displacement:+d} (${target:04X}, ring {target % P})"
    elif mode in ("zp", "zpx", "zpy"):
        suffix = f" {byte}" + ({"zp": "", "zpx": ",X", "zpy": ",Y"}[mode])
    elif mode in ("abs", "abx", "aby"):
        suffix = f" {addr}" + ({"abs": "", "abx": ",X", "aby": ",Y"}[mode])
    elif mode == "ind": suffix = f" ({addr})"
    elif mode == "izx": suffix = f" ({byte},X)"
    elif mode == "izy": suffix = f" ({byte}),Y"
    else: raise ValueError(mode)
    return name + suffix


def replay(zilion):
    sources = [HERE / f for f in ("score6502.py", "population_classes.py", "run6502.py",
               "algocell_exp/executor.py", "algocell_exp/assay.py", "algocell_exp/exectrace.py",
               "algocell_exp/zmachine.py", "algocell_exp/machines/" + MACHINE + ".wgsl",
               "algocell_exp/machines/" + MACHINE + ".manifest.json")]
    inputs = [OUT / f for f in ("s1_worlds.csv", "s1_cells.csv", "S1.md")]
    inputs.append(HERE / "REVISION_PREREG.md")
    for path in sources + inputs:
        committed(path)
    worlds = pd.read_csv(OUT / "s1_worlds.csv")
    worlds = worlds[worlds.arm == "A"].sort_values("seed")
    cells = pd.read_csv(OUT / "s1_cells.csv")
    cells = cells[cells.arm == "A"].copy()
    assert list(worlds.seed) == list(range(65001, 65021))
    opcodes, entry = descriptor(zilion)
    draws, partners, steps, offsets = [], [], [], []
    ex = StateReplay(MACHINE)
    started = time.perf_counter()
    with E.using(ex):
        for world in worlds.to_dict("records"):
            seed = int(world["seed"])
            path = OUT / "worlds" / f"{world['world']}.json"
            result = json.loads(committed(path))
            inputs.append(path)
            for file, digest in result["code_sha256"].items():
                assert sha(HERE / file) == digest, f"changed scorer source: {file}"
            for key in ("first_tape", "first_loop", "first_confined", "first_entered"):
                assert result["world"][key] == world[key]
            rng = Draws(seed)
            # The committed cache is insertion-ordered, exactly as score_world
            # populated it. Replay every cache miss up to the first replicator.
            for index, cached in enumerate(result["judged_tapes"]):
                tape = np.frombuffer(bytes.fromhex(cached["tape"]), dtype=np.uint8)
                measured = S.judge_tape(tape, rng, ex, L, STEPS, False)
                for key, value in measured.items():
                    old = cached[key]
                    assert (np.isclose(value, old, atol=1e-12) if key == "gen2"
                            else value == old), (seed, index, key, value, old)
                draws.append({"seed": seed, "cache_index": index, "tape": cached["tape"],
                              "assay_seed": rng.assay_seed, "partner_bytes_sha256": hashlib.sha256(rng.partners.tobytes()).hexdigest(),
                              "first_replicator": cached["tape"] == world["first_tape"], "cached_judgment_matches": True})
                if cached["tape"] == world["first_tape"]:
                    break
            else: raise AssertionError("first tape absent from committed cache")
            assert np.array_equal(ex.pairs[:, L:], rng.partners)
            r = ex.replay
            state = r.state.reshape(16, STEPS + 1, -1)
            memory = r.memory.reshape(16, STEPS + 1, P)
            actual = r.steps.reshape(16, STEPS + 1)
            fetch = r.fetch_map.reshape(16, STEPS + 1, P)
            starts = r.start_map.reshape(16, STEPS + 1, P)
            pc_index = r.state_names.index("pc")
            assert S.pc_has_loop(state[:, :, pc_index]) == bool(world["first_loop"])
            assert not fetch[:, -1, L:].any() and not starts[:, -1, L:].any()
            m = ex.machine_for(L)
            for key in ("sha256_core", "sha256_world", "sha256_wgsl"):
                assert m.describe()[key] == result["executor"][MACHINE][key]
            for encounter in range(16):
                partners.append({"seed": seed, "encounter": encounter, "organism": world["first_tape"],
                                 "partner": " ".join(f"{b:02x}" for b in ex.pairs[encounter, L:])})
                for k in range(STEPS + 1):
                    regs = dict(zip(r.state_names, map(int, state[encounter, k])))
                    pc = regs.pop("pc") & 0xFFFF
                    pos = pc % P  # vendored zil_pos(addr) = addr % 32u
                    op = opcodes[int(memory[encounter, k, pos])]
                    executed = k < STEPS and actual[encounter, k + 1] > actual[encounter, k]
                    operands = [int(memory[encounter, k, ((pc + j) & 0xFFFF) % P]) for j in range(1, op["length"])]
                    after_pc = int(state[encounter, k + 1, pc_index]) if k < STEPS else None
                    back = executed and after_pc % P < pos and after_pc % P < L
                    decrease = executed and after_pc < pc and (pc, after_pc) != (0xFFFF, 0)
                    steps.append({"seed": seed, "encounter": encounter, "prefix": k,
                                  "executed_steps": int(actual[encounter, k]), "pc": pc,
                                  "pc_hex": f"{pc:04X}", "ring_position": pos,
                                  "opcode": op["key"], "opcode_hex": f"{op['key']:02X}",
                                  "mnemonic": op["mnemonic"], "disassembly": disassemble(op, operands, pc),
                                  "executes_next": executed, "pc_after": after_pc,
                                  "ring_after": after_pc % P if after_pc is not None else None,
                                  "registered_pc_decrease": decrease,
                                  "earlier_organism_position": back,
                                  "higher_alias_return": back and not decrease and after_pc >= pc and op["mnemonic"] in CONTROL,
                                  "brk_return": back and op["mnemonic"] == "BRK", **regs,
                                  "memory_hex": memory[encounter, k].tobytes().hex(),
                                  "fetch_positions": ";".join(map(str, np.flatnonzero(fetch[encounter, k]))),
                                  "start_positions": ";".join(map(str, np.flatnonzero(starts[encounter, k])))})
        # classify's RNG depends on soup size, not soup contents, until the assay.
        # Reproduce the 64-index selection from the registered 20,000-cell world;
        # the committed scanned tapes then share precisely its 32 assay partners.
        for seed, selected in cells.groupby("seed", sort=True):
            rng = np.random.default_rng([int(seed), 23])
            rng.choice(160 * 125, size=64, replace=False)
            assay_seed = int(rng.integers(1 << 31))
            tapes = np.array([list(bytes.fromhex(t)) for t in selected.tape], dtype=np.uint8)
            measured = A.assay_many(tapes, z80_steps=STEPS, n=32, seed=assay_seed, zero_halts=False)
            # Additional descriptive diagnostics use the same first-generation
            # partners and unchanged best-shift/75%/modal-offset definitions.
            culture_rng = np.random.default_rng(assay_seed)
            culture_partners = culture_rng.integers(0, 256, size=(32, L), dtype=np.uint8)
            tt = np.repeat(tapes, 32, axis=0)
            rr = np.tile(culture_partners, (len(tapes), 1))
            offspring = A.execute_pairs(np.concatenate([tt, rr], axis=1), L, STEPS)[:, L:]
            similarity, shifts = A._best_shift_rows(offspring, tt)
            similarity = similarity.reshape(len(tapes), 32)
            shifts = shifts.reshape(len(tapes), 32)
            for i, (cell, assay) in enumerate(zip(selected.to_dict("records"), measured)):
                old = None if pd.isna(cell["copy_offset"]) else int(cell["copy_offset"])
                assert assay["copy_offset"] == old and assay["is_replicator"]
                qualifies = similarity[i] >= 0.75
                assert A._modal_offset(shifts[i], qualifies) == old
                offsets.append({**cell, "copy_offset": old, "assay_seed": assay_seed,
                                "replayed_copy_offset": assay["copy_offset"], "matches_registered": True,
                                "gen2_score": assay["gen2_score"], "gen1_qualifying_of_32": int(qualifies.sum()),
                                "gen1_exact_of_32": int((similarity[i] == 1).sum()),
                                "qualifying_shift_counts": json.dumps(dict(sorted(Counter(map(int, shifts[i][qualifies])).items()))),
                                "p3_eligible": bool(cell["confined"] and old is not None and old >= 6)})
    gpu_seconds = time.perf_counter() - started
    assert gpu_seconds < 60, f"EXPLORATORY GPU burst exceeded limit: {gpu_seconds:.1f}s"
    # A full equal pause before any subsequent run; the report does no GPU work.
    print(f"{LABEL}: foreground replay finished in {gpu_seconds:.2f}s; equal pause", flush=True)
    time.sleep(gpu_seconds)
    for key, rows in (("draws", draws), ("partners", partners), ("steps", steps), ("offsets", offsets)):
        write_table(key, rows)
    provenance = [{"kind": "source" if path in sources else "registered_input",
                   "path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in sources + inputs]
    provenance += [{"kind": "descriptor_dist", "path": str(entry), "sha256": sha(entry)},
                   {"kind": "descriptor_dist", "path": str(entry.parent / "core.wgsl.js"), "sha256": sha(entry.parent / "core.wgsl.js")},
                   {"kind": "zilion_branch", "path": "next", "sha256": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=zilion, text=True).strip()},
                   {"kind": "adapter", "path": adapter_summary(), "sha256": ""},
                   {"kind": "foreground_gpu_seconds", "path": f"{gpu_seconds:.3f}", "sha256": ""}]
    write_table("provenance", provenance)
    analyse(worlds)
    print(f"{LABEL}: tables and report written", flush=True)


def analyse(worlds=None):
    if worlds is None:
        worlds = pd.read_csv(OUT / "s1_worlds.csv")
        worlds = worlds[worlds.arm == "A"].sort_values("seed")
    s = load_table("steps")
    opcodes = load_table("opcodes").to_dict("records")
    assert len(s) == 20 * 16 * 129
    classes, encounters, disassembly = [], [], []
    for world in worlds.to_dict("records"):
        seed = int(world["seed"])
        tape = bytes.fromhex(world["first_tape"])
        assert tape[:8] == tape[8:]
        t = s[s.seed == seed]
        returns = t[t.earlier_organism_position]
        registered = returns[returns.registered_pc_decrease]
        aliases = returns[returns.higher_alias_return]
        brk = returns[returns.brk_return]
        if len(registered): category, eligible = "i", registered
        elif len(aliases): category, eligible = "ii", aliases
        elif len(brk): category, eligible = "iii", brk
        elif not len(returns): category, eligible = "iv", returns
        else: category, eligible = "v", returns
        assert bool(t.registered_pc_decrease.any()) == bool(world["first_loop"])
        assert (category == "i") == bool(world["first_loop"])
        event = eligible.iloc[0] if len(eligible) else None
        starts = sorted({int(pos) for row in t.start_positions for pos in str(row).split(";") if pos != ""})
        short = []
        for pos in sorted({pos % 8 for pos in starts}):
            op = opcodes[tape[pos]]
            operands = list(tape[pos + 1:pos + int(op["length"])])
            short.append(f"{pos:02d}: {disassemble(op, operands, pos)}")
        for pos in range(L):
            op = opcodes[tape[pos]]
            length = int(op["length"])
            operands = list(tape[pos + 1:pos + length])
            # At the organism's end, unknown partner operands stay unknown.
            text = disassemble(op, operands, pos) if op["mode"] != "rel" or operands else op["mnemonic"] + " ??"
            disassembly.append({"seed": seed, "position": pos, "opcode_hex": f"{tape[pos]:02X}",
                                "mnemonic": op["mnemonic"], "length": length, "mode": op["mode"],
                                "disassembly": text, "instruction_started_here": pos in starts})
        for encounter, et in t.groupby("encounter", sort=True):
            last = et.iloc[-1]
            encounters.append({"seed": seed, "encounter": int(encounter), "executed_steps": last.executed_steps,
                               "registered_loop": bool(et.registered_pc_decrease.any()),
                               "position_returns": int(et.earlier_organism_position.sum()),
                               "higher_alias_returns": int(et.higher_alias_return.sum()),
                               "brk_returns": int(et.brk_return.sum()),
                               "partner_fetches": sum(int(x) >= L for x in str(last.fetch_positions).split(";") if x != "")})
        def count_encounters(mask):
            return int(t.loc[mask, "encounter"].nunique())
        classes.append({"seed": seed, "t_rep": int(world["t_rep"]), "first_tape": world["first_tape"],
                        "registered_loop": bool(world["first_loop"]), "category": category,
                        "instruction": event.mnemonic if event is not None else "",
                        "witness_encounter": int(event.encounter) if event is not None else "",
                        "witness_prefix": int(event.prefix) if event is not None else "",
                        "witness_pc": int(event.pc) if event is not None else "",
                        "witness_pc_after": int(event.pc_after) if event is not None else "",
                        "witness_ring": int(event.ring_position) if event is not None else "",
                        "witness_ring_after": int(event.ring_after) if event is not None else "",
                        "witness_disassembly": event.disassembly if event is not None else "",
                        "position_return_encounters": count_encounters(t.earlier_organism_position),
                        "pc_decrease_return_encounters": count_encounters(t.earlier_organism_position & t.registered_pc_decrease),
                        "higher_alias_return_encounters": count_encounters(t.higher_alias_return),
                        "brk_return_encounters": count_encounters(t.brk_return),
                        "short_disassembly": "; ".join(short)})
    write_table("classes", classes)
    write_table("encounters", encounters)
    write_table("disassembly", disassembly)
    offsets = load_table("offsets")
    distribution = []
    for scope, group in (("all scanned heritable", offsets), ("confined", offsets[offsets.confined]),
                         ("open", offsets[~offsets.confined])):
        for d in (0, "undefined"):
            match = group[group.copy_offset == ("" if d == "undefined" else d)]
            distribution.append({"scope": scope, "d": d, "cells": len(match),
                                 "worlds": int(match.seed.nunique()),
                                 "regenerators": int((match["class"] == "regenerator").sum()),
                                 "transmitters": int((match["class"] == "transmitter").sum()),
                                 "open": int((match["class"] == "open").sum())})
    assert sum(r["cells"] for r in distribution if r["scope"] == "all scanned heritable") == len(offsets)
    assert not offsets.p3_eligible.any() and offsets.matches_registered.all()
    write_table("distribution", distribution)
    report()


def report():
    """EXPLORATORY Markdown generated solely from the CSV analysis tables."""
    classes = load_table("classes")
    encounters = load_table("encounters")
    steps = load_table("steps")
    offsets = load_table("offsets")
    dist = load_table("distribution")
    provenance = load_table("provenance")
    draws = load_table("draws")
    counts = Counter(classes.category)
    six = classes[~classes.registered_loop]
    holds = six.category.isin(["ii", "iii"]).sum()
    lines = [f"**{LABEL}**", "", "# Exploratory S1 ring-position follow-up", "",
             "This examination implements the exploratory hypothesis declared at the end of "
             "`experiments/REVISION_PREREG.md`. Registered verdicts are final: P1 remains not supported; "
             "P3 remains not tested. All analyses and tables here are exploratory.", "",
             f"Exploratory hypothesis result: **holds for {holds} of {len(six)}** first replicators without a registered PC decrease. "
             "Every one uses a taken forward branch to a higher 16-bit alias of an earlier organism position. "
             "No BRK cycle is needed to explain these six.", "",
             "## Exploratory method and audit", "",
             f"Replayed {len(classes)} first replicators × 16 encounters = {len(encounters)} original pairs, "
             f"with reset plus budgets 1–128 ({len(steps):,} recorded prefix states). "
             "Each prefix is an independent replay of its initial pair, as `executor.pc_trace` does, using "
             f"the unchanged vendored `{MACHINE}`. All recorded registers, memory, fetch/start maps and "
             "the opcode at the pre-instruction state are saved; terminal/halted rows have `executes_next=False`.", "",
             f"Read only committed `s1_worlds.csv`, `s1_cells.csv` and the scorer's committed "
             f"`judged_tapes` tables in `worlds/*.json`. The ordered cache is replayed through frozen `score6502.judge_tape` "
             f"from `default_rng([seed, 0, 11])` through each first tape ({len(draws)} cache misses). "
             "The scorer itself consumes the assay seed and draws the uniform-byte partners. Every judgment "
             "matches its committed row; all 20 registered loop calls and all 320 confinement checks match. "
             "No raw run/soup or Modal volume file was read or fetched. The trace partners are uniform random "
             "bytes, as the scorer actually implements them.", "",
             "Position mapping comes from the vendored shader's `zil_pos`: `position = (PC & 0xFFFF) % 32`. "
             "For P = 32 this equals zilion's seam mapping: 32 divides the 65,536-address space. "
             "An exploratory position return is an executed transition to a strictly lower ring position in "
             "organism A (0–15). Prefix adjacency alone after a halt is not an instruction. "
             "Categories are exclusive, prioritising (i), then (ii), then (iii); the tables also retain "
             "all mechanisms seen in each encounter.", "",
             "Mnemonics and lengths were read by Node directly from `m6502.table()` in zilion `next`, "
             "`/Users/neo/repos/zilion/dist/cores/m6502/index.js`; addressing modes come from that module's "
             "`M6502_OPCODES`. No mnemonic is inferred from memory. The short disassembly below reads the "
             "initial bytes at observed instruction-start positions, including overlapping decodes, "
             "collapsed onto the 8-byte block that every tape repeats twice. An operand can come from "
             "the second block. Branch targets in that listing assume the shown initial PC. "
             "Witness targets use the actual PC and mutable memory. The full disassembly table also marks "
             "positions reached as instruction starts, including overlapping decodes.", "",
             "## Exploratory classification", "",
             "| Exploratory category | First replicators |", "|---|---:|",
             f"| (i) Earlier organism position via registered PC decrease | {counts['i']} |",
             f"| (ii) Earlier organism position via higher alias | {counts['ii']} |",
             f"| (iii) BRK vector cycle | {counts['iii']} |",
             f"| (iv) No earlier-position return | {counts['iv']} |",
             f"| (v) Other | {counts['v']} |", "",
             "The witness is the first eligible event in encounter order, then prefix order. Counts after "
             "the witness are the encounters (out of 16) with any earlier-position return.", "",
             "| Exploratory seed | Category | Responsible instruction | Witness PC → PC (ring → ring) | Encounters |",
             "|---|---|---|---|---:|"]
    for r in classes.to_dict("records"):
        lines.append(f"| {r['seed']} | ({r['category']}) | {r['witness_disassembly']} | "
                     f"{int(r['witness_pc']):04X} → {int(r['witness_pc_after']):04X} "
                     f"({r['witness_ring']} → {r['witness_ring_after']}) | {r['position_return_encounters']} / 16 |")
    brk_steps = int((steps.executes_next & (steps.mnemonic == "BRK")).sum())
    first_bcs = steps[(steps.seed == 65002) & (steps.encounter == 0) & (steps.prefix == 0)].iloc[0]
    taken_bcs = steps[(steps.seed == 65002) & (steps.encounter == 0) & steps.higher_alias_return].iloc[0]
    assert first_bcs.mnemonic == "BCS" and not (int(first_bcs.p) & 1) and int(taken_bcs.p) & 1
    lines += ["", f"Exploratory detail: {brk_steps} BRK instructions executed across these 320 encounters. "
              "For seed 65002, the BCS at reset PC 0 is not taken (carry clear); SEC at position 7 "
              f"sets carry, and the BCS at PC {int(taken_bcs.pc)} then jumps to "
              f"{int(taken_bcs.pc_after)} (ring {int(taken_bcs.ring_after)}). "
              "The registered-loop comparison tapes can also have higher-alias returns; category (i) "
              "retains their registered PC-decrease classification."]
    lines += ["", "### Exploratory short disassemblies of all 20 tapes", "",
              "| Exploratory seed | Initial bytes at observed starts (8-byte block repeated twice) |", "|---|---|"]
    for r in classes.to_dict("records"):
        lines.append(f"| {r['seed']} | `{r['short_disassembly']}` |")
    lines += ["", "## Exploratory P3 diagnosis", "",
              "The distribution below is the frozen `copy_offset` column, with a replay using the same "
              "culture partners. It does not reinterpret d or change eligibility.", "",
              "| Exploratory scanned population | d | Cells | Regenerators | Transmitters | Open |",
              "|---|---|---:|---:|---:|---:|"]
    for r in dist.to_dict("records"):
        lines.append(f"| {r['scope']} | {r['d']} | {r['cells']} | {r['regenerators']} | {r['transmitters']} | {r['open']} |")
    nulls = offsets[offsets.copy_offset == ""]
    defined = offsets[offsets.copy_offset != ""]
    lines += ["", f"Exploratory assay replay matched all {len(offsets)} registered scanned-cell offsets. "
              "The final-cell assay draws were reconstructed from `default_rng([seed, 23])`: "
              "the 64-cell index choice out of the registered 20,000 cells, then the assay seed. "
              "The committed scanned tapes share its same 32 uniform partners, so a soup snapshot is unnecessary.", "",
              "`copy_offset` is defined by the unchanged assay: for each first-generation partner, choose "
              "the best cyclic shift of the organism tape; retain partners with best similarity ≥ 0.75; "
              "return the modal retained shift, or `None` if none qualify. Similarity ties keep the first "
              "shift (ascending from 0); modal ties keep the smallest shift. Thus d is a cyclic alignment "
              "of offspring bytes, not a load/store address distance or the size of a control-flow jump. "
              "Measured d = 0 means alignment at zero, and alone does not mean a whole-tape exact copy.", "",
              f"Of {len(defined)} cells with a defined exploratory replay offset, every modal offset is 0. "
              "Every qualifying first-generation partner also has best shift 0 under the frozen tie rule. "
              f"The {len(nulls)} undefined cells are from seed(s) "
              f"{', '.join(map(str, sorted(nulls.seed.unique())))}. Each has "
              f"{int(nulls.gen1_qualifying_of_32.min())} qualifying first-generation partners out of 32, "
              f"while its gen2 score is {nulls.gen2_score.min():.6f}–{nulls.gen2_score.max():.6f} "
              "(heritable at ≥ 0.3). Heritability uses normalised second-generation gain and therefore "
              "does not guarantee a first-generation ≥ 75% match needed to define d.", "",
              "Exploratory diagnosis: P3's class-occurrence part was supported, but its offset-equality "
              "part had no eligible denominator. Every measured confined d is 0 (< 6), and the remaining "
              "confined values are undefined. The registered `sites = d − 6` check for measured d ≥ 6 "
              "could not be tested; no offset or eligibility criterion is redefined.", "",
              "## Exploratory artifacts and provenance", ""]
    for key, filename in FILES.items():
        lines.append(f"- Exploratory {key}: [{filename}]({filename}).")
    for kind in ("zilion_branch", "adapter", "foreground_gpu_seconds"):
        r = provenance[provenance.kind == kind].iloc[0]
        lines.append(f"- Exploratory {kind}: `{r.path}`" + (f", `{r.sha256}`." if r.sha256 else "."))
    lines += ["", "Reproduce the exploratory analysis with `experiments/explore_s1_6502.py`; "
              "`--report-only` regenerates the derived tables and this report from the saved CSVs without GPU work. "
              "The foreground replay is followed by an equal pause. Registered inputs and scorer sources "
              "are checked against git and their committed scorer hashes, and their SHA-256s are recorded."]
    (OUT / "EXPLORATORY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zilion", type=Path, default=Path("/Users/neo/repos/zilion"))
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        analyse()
        print(f"{LABEL}: report regenerated from CSV tables")
    else:
        replay(args.zilion)
