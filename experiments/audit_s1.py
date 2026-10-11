"""Independent, bounded S1 audit; writes only audit_*.csv under results/s6502.

Run with the existing experiment Python and PYTHONPATH=experiments.
--raw-dir points to the four selected downloaded JSONLs and A/65002 final soup.
GPU work is foreground with a 55-second process alarm; no packages installed.
The small-program memory oracles are literal hand-derived expectations, not
outputs of another emulator. L20 is an extra seam probe, outside registered S1.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import signal
import subprocess
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import brotli
import numpy as np

from algocell_exp import assay as A, executor as E
from algocell_exp.soup import Soup, adapter_summary
from algocell_exp.zmachine import load_machine
from algocell_exp.zsoup import ZSoup, extract_soup_kernels, parse_wgsl_items
import run6502 as R
import score6502 as S

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/s6502"
ROWS: list[dict] = []
CPU: list[dict] = []


def check(name, ok, evidence):
    ROWS.append({"check": name, "status": "PASS" if ok else "FAIL", "evidence": evidence})
    print(f"{ROWS[-1]['status']} {name}: {evidence}", flush=True)


def save(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def hx(a):
    return np.asarray(a, dtype=np.uint8).tobytes().hex()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(m, name, initial, budget, expected, regs, fetch=None, starts=None,
             writes=None, steps=None, halt=None):
    r = m.run(initial[None], budget=budget)
    ok = np.array_equal(r.memory[0], expected)
    ok &= all(int(r.reg(k)[0]) == v for k, v in regs.items())
    if fetch is not None:
        ok &= set(np.flatnonzero(r.fetch_map[0])) == set(fetch)
    if starts is not None:
        ok &= set(np.flatnonzero(r.start_map[0])) == set(starts)
    if writes is not None:
        ok &= r.write_counts[0].tolist() == writes
    if steps is not None:
        ok &= int(r.steps[0]) == steps
    if halt is not None:
        ok &= int(r.halt_reason[0]) == halt
    CPU.append({"machine": m.name, "case": name, "status": "PASS" if ok else "FAIL",
                "budget": budget, "initial_memory": hx(initial), "expected_memory": hx(expected),
                "actual_memory": hx(r.memory[0]), "expected_registers": json.dumps(regs),
                "actual_registers": json.dumps(dict(zip(r.state_names, map(int, r.state[0])))),
                "fetch_positions": ";".join(map(str, np.flatnonzero(r.fetch_map[0]))),
                "start_positions": ";".join(map(str, np.flatnonzero(r.start_map[0]))),
                "writes": json.dumps(r.write_counts[0].tolist()), "steps": int(r.steps[0]),
                "halt": int(r.halt_reason[0])})
    return r


def program(P, code):
    a = np.full(P, 0xEA, dtype=np.uint8)
    a[:len(code)] = code
    return a


def cpu_checks():
    for L in (16, 20, 32):
        m = load_machine(f"m6502_pair_maps_L{L}")
        P = 2 * L
        s = 255 - ((511 - (P - 1)) % P)
        a = program(P, [0x48])
        run_case(m, "reset", a, 0, a.copy(), {"a": 0, "x": 0, "y": 0, "p": 32, "pc": 0, "s": s},
                 fetch=[], starts=[], writes=[0, 0], steps=0)
        e = a.copy(); e[-1] = 0
        run_case(m, "PHA_first", a, 1, e, {"s": s-1, "pc": 1}, fetch=[0], starts=[0], writes=[0, 1])
        a = program(P, [0xA2, 2, 0xA9, 0x5A, 0x95, 0xFF, 0xB5, 0xFF, 0x85, 0x1E])
        e = a.copy(); e[1] = e[30] = 0x5A  # (FF + 2) & FF = 1, not 257.
        run_case(m, "zpX_8bit_wrap", a, 5, e, {"a": 0x5A, "x": 2, "pc": 10})
        a = program(P, [0xA9, 0x5A, 0x85, 0x35, 0xA5, 0x35, 0x85, 0x1E])
        e = a.copy(); e[0x35 % P] = e[30] = 0x5A
        run_case(m, "zp_alias", a, 4, e, {"a": 0x5A, "pc": 8})
        a = program(P, [0xA9, 0x5A, 0x8D, 0x34, 0x12, 0xAD, 0x34, 0x12, 0x85, 0x1E])
        e = a.copy(); e[0x1234 % P] = e[30] = 0x5A
        run_case(m, "absolute_alias", a, 4, e, {"a": 0x5A, "pc": 10})
        a = program(P, [0xA2, 0, 0x9A, 0xA9, 0x5A, 0x48, 0xA9, 1, 0x68, 0x85, 0x1E])
        e = a.copy(); e[256 % P] = e[30] = 0x5A
        run_case(m, "stack_page_and_pull", a, 7, e, {"a": 0x5A, "s": 0, "pc": 11})
        a = program(P, [0xA2, 0, 0x9A, 0, 0xEA]); a[14:16] = [0x78, 0x56]
        # LDX #0 sets Z before BRK: pushed P is 0x32, final P is 0x26.
        e = a.copy(); e[256 % P] = 0; e[511 % P] = 5; e[510 % P] = 0x32
        target = int(e[65534 % P]) | (int(e[65535 % P]) << 8)
        run_case(m, "BRK_vector_after_stack", a, 3, e, {"pc": target, "s": 253, "p": 38},
                 fetch=[0, 1, 2, 3, 4], starts=[0, 2, 3])
        a = program(P, [0, 0xEA]); a[14:16] = [0x78, 0x56]
        e = a.copy(); e[-1] = 0; e[-2] = 2; e[-3] = 0x30
        target = int(e[65534 % P]) | (int(e[65535 % P]) << 8)
        run_case(m, "BRK_first", a, 1, e, {"pc": target, "s": s-3, "p": 36},
                 fetch=[0, 1], starts=[0], writes=[0, 3], halt=0)
        a = program(P, [0xD0, 0x38]); pos = 58 % P
        a[pos:pos+4] = [0xA9, 0x5A, 0x85, 0x18]
        e = a.copy(); e[24] = 0x5A
        run_case(m, "forward_branch_alias", a, 3, e, {"pc": 62, "a": 0x5A})
        a = program(P, [0xD0, 0xFD]); a[65535 % P] = 0xA9
        run_case(m, "negative_branch_and_PC_seam", a, 2, a.copy(), {"pc": 1, "a": 0xD0},
                 fetch=[0, 1, 65535 % P], starts=[0, 65535 % P])
        a = program(P, [0x4C, 0xFE, 0xFF]); a[65534 % P] = 0x0C
        r = run_case(m, "three_byte_increment_wrap", a, 2, a.copy(), {"pc": 1},
                     fetch=[0, 1, 2, 65534 % P, 65535 % P], starts=[0, 65534 % P])
        check(f"loop definition wraps L{L}", S.pc_has_loop([0, 65534, int(r.reg('pc')[0])]),
              "JMP $FFFE; 3-byte NOP -> PC 1; literal registered definition calls the sequential wrap a loop")
    m = load_machine("m6502_pair_maps_L16")
    a = program(32, [0x4C, 0x0F, 0]); a[15:17] = [0xA9, 0x5A]
    run_case(m, "partner_operand_only", a, 2, a.copy(), {"pc": 17, "a": 0x5A},
             fetch=[0, 1, 2, 15, 16], starts=[0, 15])
    a = program(32, [2])
    for b in (1, 128):
        run_case(m, "JAM_parked", a, b, a.copy(), {"pc": 0, "halted": 3, "trap": 1},
                 fetch=[0], starts=[0], steps=1, halt=3)
    for b, x in ((128, 64), (512, 0)):
        a = program(32, [0xE8, 0x4C, 0, 0])
        run_case(m, "caller_budget", a, b, a.copy(), {"x": x, "pc": 0}, steps=b)
    m = load_machine("m6502_brkhalt_maps_L16")
    a = program(32, [0, 0xEA])
    run_case(m, "D_fetched_zero_halts", a, 128, a.copy(), {"pc": 0, "p": 32, "s": 255},
             fetch=[0], starts=[0], writes=[0, 0], steps=1, halt=2)
    a = program(32, [0xA9, 0, 0x85, 0x1E, 2]); e = a.copy(); e[30] = 0
    run_case(m, "D_operand_zero_does_not_halt", a, 128, e, {"pc": 4}, steps=3, halt=3)
    check("hand CPU probes", all(c["status"] == "PASS" for c in CPU), f"{len(CPU)} memory/state/map cases")


def hand_replicator(initial, steps):
    """Five documented instructions, written directly from the manual for seed 65002."""
    mem = initial.copy(); a = x = pc = 0; p = 0x20
    for _ in range(steps):
        op = int(mem[pc % 32]); pc = (pc + 1) & 65535
        if op == 0xB0:
            rel = int(mem[pc % 32]); pc = (pc + 1) & 65535
            if p & 1: pc = (pc + (rel if rel < 128 else rel - 256)) & 65535
        elif op == 0xCA:
            x = (x - 1) & 255; p = (p & 0x7D) | (x & 128) | (2 if x == 0 else 0)
        elif op in (0xB5, 0x95):
            z = int(mem[pc % 32]); pc = (pc + 1) & 65535; ea = ((z + x) & 255) % 32
            if op == 0xB5:
                a = int(mem[ea]); p = (p & 0x7D) | (a & 128) | (2 if a == 0 else 0)
            else: mem[ea] = a
        elif op == 0x38: p |= 1
        else: raise ValueError(f"Unexpected opcode {op:02x}; hand oracle invalid")
    return mem, {"pc": pc, "a": a, "x": x, "p": p}


def replicator_checks():
    tape = np.frombuffer(bytes.fromhex("b038cab5aa950238" * 2), dtype=np.uint8)
    partner = np.arange(0x80, 0x90, dtype=np.uint8)
    initial = np.concatenate([tape, partner]); m = load_machine("m6502_pair_maps_L16")
    for n in (1, 4, 6, 9, 89, 128):
        e, regs = hand_replicator(initial, n)
        run_case(m, f"seed65002_hand_prefix_{n}", initial, n, e, regs)
    e, _ = hand_replicator(initial, 89)
    check("65002 hand copying", np.array_equal(e[16:], tape),
          "first two writes preserve A[1],A[0], then B[15..0]; complete B by instruction 89")
    ex = E.ZilionExecutor("m6502_pair_maps_L16")
    with E.using(ex):
        r = A.assay_many(tape[None], z80_steps=128, n=64, seed=65002)[0]
        rr = ex.run(initial[None], 16, 128, fetch_map=True, start_map=True)
    check("65002 culture and fetch confinement", r["gen2_score"] == 1 and not rr.fetch_map[:, 16:].any(),
          f"gen2={r['gen2_score']}, copy_offset={r['copy_offset']}; fetches only organism positions")


def world_checks():
    for arm in "ABCD":
        cfg = R.ARMS[arm]; L = cfg["L"]; b = cfg["budget"]
        soup = ZSoup(cfg["machine"], seed=65002, z80_steps=b)
        old = object.__new__(Soup)
        for k in ("width", "height", "tape_length", "pair_length", "pair_count", "mutations_per_step", "noise_exp", "z80_steps", "masks", "cell_count", "words_per_cell"):
            setattr(old, k, getattr(soup, k))
        check(f"world params {arm}", np.array_equal(soup._params(123), old._params(123)) and soup.mutation_count == 512,
              f"160x125; 8192 pairs; 512 byte-replacement attempts; budget {b}; Soup._params identical")
        text, kept, source = extract_soup_kernels(soup.shader_file)
        original = {x.name: x.text for x in parse_wgsl_items(source)}
        check(f"verbatim soup kernels {arm}", all(x.text == original[x.name] for x in kept),
              f"{soup.shader_file.name}; {len(kept)} original declarations; no CPU executor retained")
        template = program(L, [0xA9, 0x5A, 0x8D, 0xFF, 1, 2])
        cells = np.tile(template, (20000, 1)); soup.write_soup(cells)
        sentinel = np.full((8192, 2, L), 0x77, dtype=np.uint8)
        q = soup.device.queue; q.write_buffer(soup.pair_data_buf, 0, sentinel)
        q.write_buffer(soup.params_bufs[0], 0, soup._params(123))
        q.write_buffer(soup.zparams_bufs[0], 0, soup.bridge.machine_params(8192, 123, soup.masks))
        enc = soup.device.create_command_encoder(); bg = soup.bind_groups[0]; br = soup.bridge
        br.dispatch(enc, "clear_collision", bg, 79)
        br.dispatch(enc, "prepare_batch", bg, 128)
        br.encode_execute(enc, bg, soup.machine_bind_groups[0], 8192)
        br.dispatch(enc, "absorb_results", bg, 128)
        q.submit([enc.finish()]); soup.sync()
        inter = soup.read_interactions(); active = inter["active"]
        pairs = inter["pairs"]; records = soup.zmachine.decode(soup.read_records())
        expected = cells.copy(); expected[pairs[active, 1], -1] = 0x5A
        inactive_mem = soup.read_pair_memory()[~active]
        check(f"inactive and writeback {arm}",
              np.array_equal(soup.read_soup(), expected) and (records.steps[~active] == 0).all()
              and (inter["write_counts"][~active] == 0).all() and (inactive_mem == 0x77).all()
              and (records.steps[active] == 3).all() and (inter["write_counts"][active] == [0, 1]).all()
              and len(np.unique(pairs[active])) == 2 * int(active.sum()),
              f"{int(active.sum())} active disjoint pairs; inactive budget/steps/writes 0, sentinel unchanged; exact B writeback")
        # Counter loop verifies the bridge budget, not merely ZMachine.run's caller.
        cells[:] = program(L, [0xE8, 0x4C, 0, 0]); soup.write_soup(cells)
        soup.mutations_per_step = 0; soup.step(1); soup.sync()
        active = soup.read_interactions()["active"]; r = soup.zmachine.decode(soup.read_records())
        check(f"effective bridge budget {arm}", (r.steps[active] == b).all() and (r.reg("x")[active] == (b // 2) % 256).all(),
              f"INX/JMP active records all steps={b}, X={(b // 2) % 256}")


def data_checks(raw_dir):
    expected_schedule = list(range(50, 2001, 50)) + list(range(2250, 20001, 250)) + list(range(22500, 1000001, 2500)) + list(range(1025000, 10000001, 25000))
    check("runner sample schedule", R.sample_steps() == expected_schedule, "864 steps derived independently from four registered intervals")
    # Deliberately force identical hash labels for different byte strings.
    cells = np.array([[1]*16]*4 + [[2]*16]*3 + [[3]*16]*2 + [[4]*16], np.uint8)
    got, _ = R.exact_tapes(cells, np.ones(10, dtype=np.uint32))
    check("exact tapes ignore hash collisions", [x["share"] for x in got["top3"]] == [0.4, 0.3, 0.2] and got["unique"] == 4,
          "four distinct 16-byte rows with identical GPU hash remain four species; no padding in S1 L16/L32")
    worlds = [json.loads(p.read_text()) for p in sorted((OUT / "worlds").glob("s6502_*.json"))]
    keys = {(w["world"]["arm"], w["world"]["seed"]) for w in worlds}
    check("arm/seed coverage", len(worlds) == 80 and keys == {(a, s) for a in "ABCD" for s in range(65001, 65021)},
          "80 distinct scored worlds; seeds 65001-65020 in each arm as registered")
    with (OUT / "s1_worlds.csv").open() as f: csv_worlds = list(csv.DictReader(f))
    km = []; ar = [x for x in csv_worlds if x["arm"] == "A"]; risk = 20; survival = Fraction(1)
    events = Counter(int(float(x["t_rep"])) for x in ar if x["t_rep"])
    median = None
    for t, deaths in sorted(events.items()):
        survival *= Fraction(risk-deaths, risk)
        km.append({"step": t, "at_risk": risk, "events": deaths, "survival_exact": str(survival), "survival": float(survival)})
        if survival <= Fraction(1, 2) and median is None: median = t
        risk -= deaths
    check("independent arm-A KM", median == 1125000, f"exact-rational KM survival reaches 1/2 at {median}; all 20 events uncensored")
    save("audit_km.csv", km)
    frozen = {}; provenance = []; source_matches = True
    for w in worlds:
        commit = w["modal"]["scorer_commit"]
        for name, digest in w["code_sha256"].items():
            key = (commit, name)
            if key not in frozen:
                frozen[key] = subprocess.check_output(["git", "show", f"{commit}:experiments/{name}"], cwd=HERE)
                current = (HERE / name).read_bytes()
                provenance.append({"commit": commit, "file": name, "recorded_sha256": digest,
                                   "frozen_sha256": hashlib.sha256(frozen[key]).hexdigest(),
                                   "current_sha256": hashlib.sha256(current).hexdigest()})
                if current != frozen[key]:
                    # S1 imports only hx and _km_median from dial_soups.
                    before = {n.name: ast.dump(n) for n in ast.parse(frozen[key]).body if isinstance(n, ast.FunctionDef)}
                    after = {n.name: ast.dump(n) for n in ast.parse(current).body if isinstance(n, ast.FunctionDef)}
                    source_matches &= name == "dial_soups.py" and all(before[k] == after[k] for k in ("hx", "_km_median"))
            source_matches &= hashlib.sha256(frozen[key]).hexdigest() == digest
    check("scored-source hashes", source_matches,
          "all 80 recorded hashes match scorer commit; only current dial_soups changed, imported hx/KM ASTs identical")
    save("audit_sources.csv", provenance)
    table_ok = True
    for w in worlds:
        cfg = R.ARMS[w["world"]["arm"]]
        table_ok &= w["world"]["L"] == cfg["L"] and w["world"]["budget"] == cfg["budget"]
        table_ok &= w["world"]["machine"] == cfg["machine"] and w["world"]["horizon"] == 10000000
        row = next(x for x in csv_worlds if x["world"] == w["world"]["world"])
        for k in ("t_rep", "t_closed", "first_tape", "first_loop", "first_confined", "budget", "snapshot_step"):
            v = w["world"][k]
            expected = "" if v is None else str(v)
            actual = row[k]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                table_ok &= actual != "" and float(actual) == v
            else: table_ok &= actual == expected
    check("world CSV/JSON consistency", table_ok, "all 80 endpoint/tape/loop/initial-condition rows agree")
    rawrows = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        rec = [json.loads(x) for x in path.read_text().splitlines()]; c = rec[0]; samples = [x for x in rec if x.get("kind") == "sample"]
        snaps = [x for x in rec if x.get("kind") == "snapshot"]
        saved = next(w for w in worlds if w["world"]["world"] == path.stem)
        m = json.loads((HERE / f"algocell_exp/machines/{c['machine_name']}.manifest.json").read_text())
        ok = [x["step"] for x in samples] == expected_schedule and [x["step"] for x in snaps] == [20000, 100000, 1000000, 3000000, 10000000]
        cfg = R.ARMS[c["arm"]]
        ok &= all(c[k] == cfg[k] for k in ("L", "budget", "brk")) and c["machine_name"] == cfg["machine"]
        ok &= (c["width"], c["height"], c["pair_count"], c["noise_exp"]) == (160, 125, 8192, 4)
        ok &= c["world"] == path.stem and rec[-1]["samples"] == 864 and rec[-1]["snapshots"] == 5
        ok &= rec[-1]["steps_run"] == 10000000 and saved["input_sha256"] == sha(path)
        ok &= all(c["machine"][f"sha256_{k}"] == m["sha256"][k] for k in ("core", "world", "wgsl"))
        ok &= c["script_sha256"] == sha(HERE / "run6502.py")
        for sample in samples:
            top = sample["top3"]
            ok &= len({x["tape"] for x in top}) == len(top) and [x["share"] for x in top] == sorted([x["share"] for x in top], reverse=True)
            ok &= all(len(bytes.fromhex(x["tape"])) == c["L"] and hashlib.sha256(bytes.fromhex(x["tape"])).hexdigest() == x["sha256"] for x in top)
            ok &= sum(sample["byte_hist"]) == 20000*c["L"]
        check(f"raw metadata {c['arm']}", ok, f"{path.name}: 864 samples, five snapshots, full horizon, input/runner/machine hashes checked")
        rawrows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha(path), "samples": len(samples), "snapshots": len(snaps)})
        if c["arm"] == "A":
            snap = snaps[-1]; p = raw_dir / snap["file"]; raw = brotli.decompress(p.read_bytes())
            soup = np.frombuffer(raw, np.uint8).reshape(20000, 16)
            counts = Counter(row.tobytes() for row in soup)
            top = sorted(counts, key=lambda x: (-counts[x], x))[:3]
            ok = hashlib.sha256(raw).hexdigest() == snap["sha256_raw"] and len(raw) == snap["raw_bytes"]
            ok &= [(t.hex(), counts[t]/20000) for t in top] == [(x["tape"], x["share"]) for x in samples[-1]["top3"]]
            ok &= np.bincount(soup.ravel(), minlength=256).tolist() == samples[-1]["byte_hist"]
            check("final raw soup counts", ok, "A/65002 320000 bytes, checksum, full-byte Counter top3 shares and histogram match")
            ex = E.ZilionExecutor(c["machine_name"]); rng = np.random.default_rng([c["seed"], 0, 11]); cache = {}
            _, parsed, _ = S.read_input(path, "s1")
            def judge(t):
                if t not in cache:
                    cache[t] = S.judge_tape(np.frombuffer(bytes.fromhex(t), np.uint8), rng, ex, 16, 128, False)
                return cache[t]
            with E.using(ex): row, _ = S.endpoints(parsed, judge)
            check("raw A/65002 endpoints replay", all(row[k] == saved["world"][k] for k in row)
                  and cache == {x["tape"]: {k: v for k, v in x.items() if k != "tape"} for x in saved["judged_tapes"]},
                  f"{len(cache)} cached tape judgments; t_rep=t_closed={row['t_rep']}; draws and flags match")
            rawrows.append({"file": p.name, "bytes": p.stat().st_size, "sha256": sha(p), "samples": "", "snapshots": ""})
    save("audit_inputs.csv", rawrows)


def exploratory_checks():
    with (OUT / "exploratory_steps.csv").open() as f: rows = list(csv.DictReader(f))
    partners = {(int(r["seed"]), int(r["encounter"])): r for r in rows if r["prefix"] == "0"}
    targets = {(int(r["seed"]), int(r["encounter"])): r for r in rows if r["prefix"] == "128"}
    groups = {}
    for r in rows: groups.setdefault((int(r["seed"]), int(r["encounter"])), []).append(r)
    expected_pc = {}; returns = {}
    for key, group in groups.items():
        pcs = [int(r["pc"]) for r in group]
        expected_pc[key] = any(b < a and (a, b) != (65535, 0) for a, b in zip(pcs, pcs[1:]))
        returns[key] = any(r["executes_next"] == "True" and int(float(r["ring_after"])) < int(r["ring_position"])
                           and int(float(r["ring_after"])) < 16 for r in group[:-1])
    no_pc = [s for s in range(65001, 65021) if not any(expected_pc[(s, i)] for i in range(16))]
    check("exploratory position classification", len(rows) == 41280 and len(groups) == 320 and len(no_pc) == 6
          and all(returns[(s, i)] for s in no_pc for i in range(16)),
          f"independently regrouped 41280 prefixes, 320 encounters; six no-PC-decrease seeds {no_pc} all return in position space")
    ex = E.ZilionExecutor("m6502_pair_maps_L16"); keys = sorted(partners)
    initial = np.array([list(bytes.fromhex(partners[k]["memory_hex"])) for k in keys], np.uint8)
    result = ex.machine_for(16).run(initial, budget=128)
    ok = all(hx(result.memory[i]) == targets[k]["memory_hex"] and int(result.reg("pc")[i]) == int(targets[k]["pc"])
             and ";".join(map(str, np.flatnonzero(result.fetch_map[i]))) == targets[k]["fetch_positions"]
             and ";".join(map(str, np.flatnonzero(result.start_map[i]))) == targets[k]["start_positions"] for i, k in enumerate(keys))
    check("exploratory final replay", ok, "320 saved initial pairs rerun; final memory/PC/fetch/start maps match saved prefixes")


def culture_sensitivity():
    """Hold saved cell tapes and culture seed fixed; compare the two sample sizes.

    This cannot reconstruct the unscanned 48 cells or new scan membership, so it
    measures susceptibility rather than producing corrected population classes.
    """
    rows = []; ex = E.ZilionExecutor("m6502_pair_maps_L16")
    for seed in range(65001, 65021):
        w = json.loads((OUT / f"worlds/s6502_A_s{seed}.json").read_text())
        tapes = np.array([list(bytes.fromhex(c["tape"])) for c in w["cells"]], np.uint8)
        rng = np.random.default_rng([seed, 23]); rng.choice(20000, size=64, replace=False)
        assay_seed = int(rng.integers(1 << 31))
        with E.using(ex):
            n32 = A.assay_many(tapes, z80_steps=128, n=32, seed=assay_seed)
            n64 = A.assay_many(tapes, z80_steps=128, n=64, seed=assay_seed)
        for c, a, b in zip(w["cells"], n32, n64):
            rows.append({"seed": seed, "cell": c["cell"], "class_n32": c["class"], "tape": c["tape"],
                         "assay_seed": assay_seed, "gen2_n32": a["gen2_score"], "gen2_n64": b["gen2_score"],
                         "heritable_n32": a["is_replicator"], "heritable_n64": b["is_replicator"],
                         "offset_n32": a["copy_offset"], "offset_n64": b["copy_offset"]})
    check("saved scanned-cell culture reconstruction", all(r["heritable_n32"] for r in rows),
          f"all {len(rows)} saved scanned arm-A cells remain heritable under original 32-partner culture draws")
    flips = [r for r in rows if not r["heritable_n64"]]
    print(f"SENSITIVITY {len(flips)}/{len(rows)} saved cells cease to be heritable with n=64; seeds {sorted({r['seed'] for r in flips})}", flush=True)
    save("audit_culture_sensitivity.csv", rows)


def mutation_race():
    """Probe two distinct bytes of one packed word, with ordinary soup params."""
    mask = 0xFFFFFFFF
    def pcg(state):
        new = (state * 747796405 + 2891336453) & mask
        word = (((state >> ((state >> 28) + 4)) ^ state) * 277803737) & mask
        return new, (word >> 22) ^ word
    selected = None
    for seed in range(1000):
        byword = {}
        for mid in range(512):
            state = (seed * 3266489917 + mid * 668265263 + 7) & mask
            state, _ = pcg(state); state, pos = pcg(state); _, value = pcg(state)
            pos %= 320000; value &= 255
            byword.setdefault(pos // 4, []).append((mid, pos, value))
        for word, writers in byword.items():
            if len(writers) == 2 and writers[0][0] // 32 == writers[1][0] // 32 and writers[0][1] != writers[1][1] and all(w[2] for w in writers):
                selected = seed, word, writers; break
        if selected: break
    seed, word, writers = selected
    soup = ZSoup("m6502_pair_maps_L16", seed=65002)
    soup.write_soup(np.zeros((20000, 16), np.uint8))
    soup.device.queue.write_buffer(soup.params_bufs[0], 0, soup._params(seed))
    enc = soup.device.create_command_encoder()
    soup.bridge.dispatch(enc, "mutate_soup", soup.bind_groups[0], 8)
    soup.device.queue.submit([enc.finish()]); soup.sync()
    actual = soup.read_soup().ravel(); rows = []
    for mid, pos, value in writers:
        rows.append({"batch_seed": seed, "mutation_id": mid, "word_index": word,
                     "byte_position": pos, "expected_value": value, "actual_value": int(actual[pos]),
                     "lost": int(actual[pos]) != value})
    print(f"MUTATION RACE seed={seed}, word={word}, writers={writers}; {sum(r['lost'] for r in rows)} distinct-byte updates lost", flush=True)
    save("audit_mutation_race.csv", rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw-dir", type=Path, required=True); args = ap.parse_args()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("55-second foreground audit bound")))
    signal.alarm(55); started = time.perf_counter()
    cpu_checks(); replicator_checks(); world_checks(); data_checks(args.raw_dir); exploratory_checks(); culture_sensitivity(); mutation_race()
    signal.alarm(0)
    check("foreground runtime", True, f"{time.perf_counter()-started:.3f}s including CPU, IO, shader compilation; {adapter_summary()}")
    save("audit_cpu.csv", CPU); save("audit_checks.csv", ROWS)
    if any(r["status"] == "FAIL" for r in ROWS) or any(r["status"] == "FAIL" for r in CPU): raise SystemExit(1)


if __name__ == "__main__": main()
