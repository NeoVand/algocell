"""The 6502 self-writer census on the validated Zilion 6502 core (Proposition 3 analogue), with its Z80 positive control.

    cd experiments
    python census6502.py control                    # Z80 compat 0.2: reproduce the paper's census (periods 1-4, L = 16)
    python census6502.py planted                    # 6502: the harness finds two seeded closed copiers tiled at period 6
    python census6502.py census --L 16 --k 1 2 3    # 6502 census, both BRK arms, budgets 128 and 512 (checkpointed)
    python census6502.py census --L 16 --k 4                    # the 4.29e9 period-4 tilings: resumes from the checkpoint,
                                                                # pauses as long as each chunk took, then rewrites CENSUS.md
    python census6502.py census --L 16 --k 4 --pause 0          # the same on an idle machine
    python census6502.py closers                    # seeded closers of ISA_RESEARCH §4.2 on the validated core
    python census6502.py explore                    # the exploratory census's candidates re-run on Zilion (same encounters)
    python census6502.py report                     # CENSUS.md and its tables from everything above

Definitions (all in `algocell_exp/zcensus.py`, the GPU harness): every word T of period k tiled to L bytes is tape A;
it meets the all-zero partner and 16 random partners (one per tile, from a counter-based hash); `budget` instructions
(128 and 512, in one pass); copy score = the best of the L cyclic shifts (the Z80 census's "partner equal to the tape
at some cyclic shift"); columns: exact into the zero partner; exact into ≥ 1 random partner; ≥ 75% copy in ≥ 8 of the 16
random partners; closed = no partner byte fetched as instruction stream (fetch map) in any of the 17 encounters.
Everything this script reports goes to results/census6502/ (CENSUS.md quotes only the generated tables there).
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from algocell_exp.zcensus import HIT_WORDS, TileCensus, evaluate, primitive_period, tile_bytes, tile_tapes  # noqa: E402
from algocell_exp.zmachine import load_machine  # noqa: E402

OUT = HERE / "results" / "census6502"
CKPT = OUT / "ckpt"
ZILION = Path(os.environ.get("ZILION_DIR", HERE.parent.parent / "zilion"))
EXPLORATORY = Path("/private/tmp/claude-501/-Users-neo-repos-algocell/26fc62cd-9c63-4d42-8467-b9398dac54bd/scratchpad/isa_research/census")
Z80_CENSUS = HERE / "results" / "review_r2"
BUDGETS = (128, 512)
SEGMENT = 128
NRAND = 16


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def machine_name(L: int, brk: str) -> str:
    return f"m6502_{'brkhalt' if brk == 'halt' else 'pair'}_maps_L{L}"


def sample_bits_for(k: int) -> int:
    return {1: 2, 2: 6, 3: 12}.get(k, 18)


# ── checkpointed census runs ────────────────────────────────────────────────────────────────────────────────────────
def ckpt_path(machine: str, k: int, nrand: int, budgets) -> Path:
    return CKPT / f"{machine}_k{k}_r{nrand}_b{'-'.join(map(str, budgets))}.jsonl"


def _pack(H: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(H, np.uint32).tobytes(), 9)).decode()


def _unpack(s: str) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(s)), np.uint32).reshape(-1, HIT_WORDS).copy()


def run_census(machine: str, k: int, nrand: int = NRAND, budgets=BUDGETS, seg: int = SEGMENT, chunk_bits: int = 24,
               max_minutes: float | None = None, pause: float = -1.0) -> Path:
    """Run (or continue) the census of period k on `machine`, one checkpoint line per chunk of 2^chunk_bits tiles.
    Each chunk's records (hits and the hash-chosen sample) are re-run through ZMachine.run at every budget before the
    chunk is written. Stops cleanly after max_minutes; a later call continues from the checkpoint (same chunk size).
    The GPU is shared: after each chunk (one burst, about 30-60 s at period 4) the run sleeps `pause` seconds, by
    default (pause < 0) as long as the chunk took."""
    CKPT.mkdir(parents=True, exist_ok=True)
    c = TileCensus(machine, k=k, nrand=nrand, budgets=budgets, seg=seg, sample_bits=sample_bits_for(k), panel=False, variant_major=True)
    path = ckpt_path(machine, k, nrand, c.budgets)
    head = json.loads(json.dumps({"type": "header", **c.describe(), "counter_names": c.counter_names, "hit_words": HIT_WORDS}))   # as stored
    done = set()
    if path.exists():
        lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        h0 = lines[0]
        for key in ("machine", "k", "L", "nrand", "budgets", "segment", "seed", "panel", "variant_major", "sample_mask", "harness_wgsl_sha256", "counter_names"):
            if h0.get(key) != head.get(key):
                raise RuntimeError(f"{path.name}: checkpoint header differs in {key!r}; move it away to start again")
        chunks = [x for x in lines[1:] if x["type"] == "chunk"]
        done = {x["i"] for x in chunks}
        sizes = {x["t1"] - x["t0"] for x in chunks if x["t1"] < 256 ** k}
        if sizes and sizes != {min(256 ** k, 1 << chunk_bits)}:
            raise RuntimeError(f"{path.name}: the checkpoint's chunks hold {sizes} words; resume with the same --chunk-bits")
    else:
        path.write_text(json.dumps(head) + "\n")
    total = 256 ** k
    chunk = min(total, 1 << chunk_bits)
    nchunks = -(-total // chunk)
    t_start = time.time()
    ran = 0
    for ci in range(nchunks):
        if ci in done:
            continue
        if max_minutes is not None and ran and time.time() - t_start > 60 * max_minutes:   # at least one chunk per call
            log(f"{path.name}: stopping after {max_minutes} min ({len(done)}/{nchunks} chunks done)")
            break
        t0, t1 = ci * chunk, min(total, (ci + 1) * chunk)
        s0 = time.time()
        out = c.run_range(t0, t1)
        secs = time.time() - s0
        rec = {"type": "chunk", "i": ci, "t0": t0, "t1": t1, "secs": round(secs, 2), "budgets": {}}
        for b in c.budgets:
            rc = c.recheck(out[b]["hits"], b)
            rec["budgets"][str(b)] = {"counters": [out[b]["counters"][n] for n in c.counter_names], "hits": _pack(out[b]["hits"]), "recheck": rc}
            bad = sum(v for kk, v in rc.items() if kk.startswith("mismatch"))
            if bad:
                raise RuntimeError(f"{path.name} chunk {ci} budget {b}: recheck mismatches {rc}")
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        done.add(ci)
        ran += 1
        C = out[c.budgets[-1]]["counters"]
        log(f"{path.name}: chunk {ci + 1}/{nchunks} [{t0:#x}, {t1:#x}) {secs:.1f} s; at {c.budgets[-1]}: zero-exact {C['tiles_zero_exact']}, "
            f"rand-exact {C['tiles_rand_exact_any']}, rand75≥8 {C['tiles_rand75_ge8']}; records {sum(len(out[b]['hits']) for b in c.budgets)}")
        wait = secs if pause < 0 else pause
        if wait and len(done) < nchunks:
            time.sleep(wait)
    return path


def load_census(path: Path) -> dict:
    lines = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    head = lines[0]
    names = head["counter_names"]
    chunks = [x for x in lines[1:] if x["type"] == "chunk"]
    per = {}
    for b in head["budgets"]:
        tot = np.zeros(len(names), np.int64)
        H, rc = [], {}
        for x in chunks:
            d = x["budgets"][str(b)]
            tot += np.asarray(d["counters"], np.int64)
            H.append(_unpack(d["hits"]))
            for kk, v in d["recheck"].items():
                rc[kk] = rc.get(kk, 0) + v
        per[b] = {"counters": dict(zip(names, map(int, tot))), "hits": np.concatenate(H) if H else np.zeros((0, HIT_WORDS), np.uint32), "recheck": rc}
    total = 256 ** head["k"]
    n_done = sum(x["t1"] - x["t0"] for x in chunks)
    size = max((x["t1"] - x["t0"] for x in chunks), default=0)
    return {"head": head, "budgets": per, "tiles_done": n_done, "complete": n_done == total, "secs": sum(x["secs"] for x in chunks),
            "chunks_done": len(chunks), "chunks_total": -(-total // size) if size else None,
            "ranges": sorted((x["t0"], x["t1"]) for x in chunks)}


def tile_done(R: dict, t: int) -> bool:
    """Whether word index t has been run in this checkpoint."""
    return any(t0 <= t < t1 for t0, t1 in R["ranges"])


def census_obj(head: dict) -> TileCensus:
    """A TileCensus with the header's settings (for decoding records and re-running tiles)."""
    return TileCensus(head["machine"]["machine"], k=head["k"], nrand=head["nrand"], budgets=head["budgets"], seg=head["segment"],
                      seed=head["seed"], panel=head["panel"], variant_major=head["variant_major"], tiles_per_dispatch=1 << 10)


# ── disassembly ─────────────────────────────────────────────────────────────────────────────────────────────────────
_OPS = None
LEN6502 = {"imp": 1, "acc": 1, "imm": 2, "rel": 2, "zp": 2, "zpx": 2, "zpy": 2, "izx": 2, "izy": 2, "abs": 3, "abx": 3, "aby": 3, "ind": 3}


def m6502_opcodes() -> list[dict]:
    """Zilion's 6502 opcode table (M6502_OPCODES), cached in results/census6502/m6502_opcodes.json with its provenance."""
    global _OPS
    if _OPS is None:
        p = OUT / "m6502_opcodes.json"
        if not p.exists():
            js = ("import(process.argv[1]).then(m => { const c = require('crypto'); console.log(JSON.stringify({opcodes: m.M6502_OPCODES, "
                  "jam: m.M6502_JAM_OPCODES, core_sha256: c.createHash('sha256').update(m.M6502_CORE_WGSL).digest('hex')})) })")
            res = json.loads(subprocess.check_output(["node", "-e", js, str(ZILION / "dist/cores/m6502/index.js")], text=True))
            res["zilion_commit"] = subprocess.check_output(["git", "-C", str(ZILION), "rev-parse", "HEAD"], text=True).strip()
            plain = json.loads((HERE / "algocell_exp/machines/m6502_pair_L16.manifest.json").read_text())["sha256"]["core"]
            res["matches_vendored_core"] = res["core_sha256"] == plain
            OUT.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(res, indent=0) + "\n")
        _OPS = json.loads(p.read_text())
        if not _OPS["matches_vendored_core"]:
            raise RuntimeError("m6502_opcodes.json was not dumped from the vendored core")
    return _OPS["opcodes"]


def dis6502_one(mem: bytes, pc: int) -> tuple[str, int]:
    ops = m6502_opcodes()
    op = mem[pc % len(mem)]
    mn, mode = ops[op]["mnemonic"], ops[op]["mode"]
    n = LEN6502[mode]
    if mn == "JAM":
        n = 1
    b = [mem[(pc + i) % len(mem)] for i in range(1, n)]
    lo = b[0] if b else 0
    w = (b[1] << 8 | b[0]) if n == 3 else 0
    arg = {"imp": "", "acc": " A", "imm": f" #${lo:02X}", "zp": f" ${lo:02X}", "zpx": f" ${lo:02X},X", "zpy": f" ${lo:02X},Y",
           "izx": f" (${lo:02X},X)", "izy": f" (${lo:02X}),Y", "abs": f" ${w:04X}", "abx": f" ${w:04X},X", "aby": f" ${w:04X},Y",
           "ind": f" (${w:04X})", "rel": f" {'+' if (lo ^ 0x80) - 0x80 >= 0 else ''}{(lo ^ 0x80) - 0x80}"}[mode]
    if mn == "BRK":
        arg = f" #${lo:02X}"
    return mn + arg, n


def dis6502_word(word: bytes, L: int = 16) -> str:
    """Linear disassembly of the word tiled, from offset 0, until the decode returns to a tile boundary (≤ 3 tiles)."""
    k = len(word)
    tape = bytes(word[i % k] for i in range(max(L, 3 * k + 3)))
    out, pc = [], 0
    while True:
        s, n = dis6502_one(tape, pc)
        out.append(s)
        pc += n
        if pc % k == 0 or pc >= 3 * k:
            break
    return "; ".join(out) + (" | …" if pc % k else "")


def dis_z80(word: bytes) -> str:
    """Linear Z80 disassembly of the word tiled to 16 bytes, from offset 0, until the decode returns to a tile
    boundary (at most 3 tiles); operands shown as bytes."""
    from algocell_exp.isa import disassemble

    k = len(word)
    tape = bytes(word[i % k] for i in range(16))
    out = []
    for ins in disassemble(tape):
        out.append(f"{ins['mnemonic']} [{' '.join(f'{b:02x}' for b in ins['bytes'])}]")
        end = ins["offset"] + len(ins["bytes"])
        if end % k == 0 or end >= 3 * k:
            break
    return "; ".join(out)


# ── Z80 positive control ────────────────────────────────────────────────────────────────────────────────────────────
def cmd_control(a) -> None:
    """The paper's census (census_period.py: every word of period p tiled to 16 bytes, 128 instructions, the all-zero
    partner; self-writer = the partner equals the tape at some cyclic shift; non-zero tapes) through this harness on the
    vendored Z80 compat '0.2' machine with maps. Periods 1-3 also with 16 random partners per tile; period 4 with the
    zero partner only (the column the paper reports). Then open/closed of every self-writer with census_period.py's own
    partners (rng([20261018, p]), drawn per self-writer in index order), on the same machine and on Algocell's op-traced
    executor."""
    from algocell_exp import exectrace as X

    m = "z80c02_pair_maps_L16"
    for k in (1, 2, 3, 4):   # the 6502 census's settings (budgets 128 and 512 in one pass, segments of 128); period 4 without random partners
        run_census(m, k, nrand=(NRAND if k < 4 else 0), budgets=BUDGETS, seg=SEGMENT, chunk_bits=26)
    mach = load_machine(m)
    rows, words_rows = [], []
    for k in (1, 2, 3, 4):
        R = load_census(ckpt_path(m, k, NRAND if k < 4 else 0, BUDGETS))
        c = census_obj(R["head"])
        d = c.decode_hits(R["budgets"][128]["hits"])
        ex = d["t"][(d["flags"] & 1) != 0]
        ex = ex[ex != 0]   # census_period.py: non-zero tapes
        words = tile_bytes(ex, k)
        prim = primitive_period(words) if len(ex) else np.zeros(0, int)
        # closed with census_period.py's partners: rng([20261018, p]); one draw of (16, L) per self-writer in index order
        rng = np.random.default_rng([20261018, k])
        P = np.zeros((len(ex), 16, 16), np.uint8)
        for i in range(len(ex)):
            P[i] = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
        e = evaluate(mach, k, ex, 16, 128, partners=P) if len(ex) else None
        closed_z = (~e.entered.any(1)) if len(ex) else np.zeros(0, bool)
        tapes = tile_tapes(ex, k, 16)
        if len(ex):
            pairs = np.concatenate([np.repeat(tapes, 17, 0), np.concatenate([np.zeros((len(ex), 1, 16), np.uint8), P], 1).reshape(-1, 16)], 1)
            _, masks = X.execute_pairs_traced(pairs, 16, 128)
            ent = X.exec_positions(masks, 32)[:, 16:].any(1).reshape(len(ex), 17).any(1)
            closed_a = ~ent
        else:
            closed_a = np.zeros(0, bool)
        # the harness's own partners (per-tile hash), for the self-writers
        eh = evaluate(mach, k, ex, 16, 128) if len(ex) else None
        closed_h = (~eh.entered.any(1)) if len(ex) else np.zeros(0, bool)
        # the paper's table
        paper = Z80_CENSUS / f"census_p{k}.csv"
        pw = set()
        pclosed = {}
        if paper.exists():
            import pandas as pd

            D = pd.read_csv(paper)
            if len(D):
                pw = set(D.word.str.replace(" ", ""))
                pclosed = dict(zip(D.word.str.replace(" ", ""), D.closed))
        mine = {bytes(w).hex() for w in words}
        Cn = R["budgets"][128]["counters"]
        row = {"period": k, "words": 256 ** k, "tiles_done": R["tiles_done"], "partners": R["head"]["nrand"],
               "self_writers": len(ex), "by_primitive_period": "; ".join(f"{q}: {int((prim == q).sum())}" for q in sorted(set(prim.tolist()))) or "-",
               "closed_census_partners_zilion": int(closed_z.sum()), "closed_census_partners_algocell": int(closed_a.sum()),
               "closed_harness_partners": int(closed_h.sum()),
               "paper_self_writers": len(pw) if paper.exists() else "n/a (no period-1 census)",
               "paper_closed": int(sum(pclosed.values())) if paper.exists() else "n/a",
               "word_sets_equal": (mine == pw) if paper.exists() else "n/a",
               "closed_flags_equal": (all(pclosed.get(bytes(w).hex()) == bool(cz) for w, cz in zip(words, closed_z))) if paper.exists() else "n/a",
               "rand_exact_any": Cn["tiles_rand_exact_any"] if R["head"]["nrand"] else "-", "rand75_ge8": Cn["tiles_rand75_ge8"] if R["head"]["nrand"] else "-",
               "recheck_tiles": sum(R["budgets"][b]["recheck"].get("tiles", 0) for b in BUDGETS),
               "recheck_mismatches": sum(v for b in BUDGETS for kk, v in R["budgets"][b]["recheck"].items() if kk.startswith("mismatch")),
               "self_writers_512": _self_writers(c, R, 512),
               "gpu_minutes": round(R["secs"] / 60, 1)}
        rows.append(row)
        for w, q, cz, ca, ch in zip(words, prim, closed_z, closed_a, closed_h):
            words_rows.append({"period": k, "word": bytes(w).hex(" "), "primitive_period": int(q), "closed (census partners)": bool(cz),
                               "closed (Algocell executor)": bool(ca), "closed (harness partners)": bool(ch), "in paper table": bytes(w).hex() in pw})
        log(row)
    write_csv("control_z80.csv", rows)
    write_csv("control_z80_words.csv", words_rows)
    log("control written")


def _self_writers(c: TileCensus, R: dict, b: int) -> int:
    d = c.decode_hits(R["budgets"][b]["hits"])
    return int((((d["flags"] & 1) != 0) & (d["t"] != 0)).sum())


# ── 6502 planted positive control ──────────────────────────────────────────────────────────────────────────────────
PLANTED = {"b50f48ca50fa": "LDA $0F,X; PHA; DEX; BVC -6", "bab5104850fa": "TSX; LDA $10,X; PHA; BVC -6"}


def cmd_planted(a) -> None:
    """The same harness at period 6 over the 65,536 words that share the first four bytes of each seeded closer (so
    the scan contains the closer itself): it must report the closer as exact into the zero partner and into its random
    partners, and closed."""
    rows = []
    for brk in ("faithful", "halt"):
        mname = machine_name(16, brk)
        c = TileCensus(mname, k=6, nrand=NRAND, budgets=BUDGETS, seg=SEGMENT, sample_bits=8, panel=False)
        for hexw, desc in PLANTED.items():
            t = int(hexw, 16)
            t0 = t & ~0xFFFF
            out = c.run_range(t0, t0 + 0x10000)
            for b in c.budgets:
                d = c.decode_hits(out[b]["hits"])
                rc = c.recheck(out[b]["hits"], b)
                sel = d["t"] == t
                zex = d["t"][(d["flags"] & 1) != 0]
                rows.append({"BRK": brk, "budget": b, "scan": f"{t0:012x}..{t0 + 0xFFFF:012x}", "closer": hexw, "what": desc,
                             "closer recorded": bool(sel.any()), "closer exact into zero partner": bool(sel.any() and d["flags"][sel][0] & 1),
                             "closer exact into random partners (of 16)": int(d["n_exact"][sel][0]) if sel.any() else 0,
                             "closer closed (17 encounters)": bool(sel.any() and not d["entered"][sel][0].any()),
                             "words exact into zero partner in the scan": len(zex),
                             "recheck tiles": rc["tiles"], "recheck mismatches": sum(v for kk, v in rc.items() if kk.startswith("mismatch"))})
                log(rows[-1])
    write_csv("planted_6502.csv", rows)


# ── the 6502 census ─────────────────────────────────────────────────────────────────────────────────────────────────
def cmd_census(a) -> None:
    """Run or resume the census for each period and BRK arm, then regenerate the report (CENSUS.md)."""
    pause = -1.0 if a.pause == "auto" else float(a.pause)
    for k in a.k:
        for brk in a.brk:
            run_census(machine_name(a.L, brk), k, nrand=NRAND, budgets=BUDGETS, seg=SEGMENT, chunk_bits=a.chunk_bits,
                       max_minutes=a.max_minutes, pause=pause)
    if not a.no_report:
        from census6502_report import report

        report()


# ── seeded closers (ISA_RESEARCH §4.2) ──────────────────────────────────────────────────────────────────────────────
def cmd_closers(a) -> None:
    """Closer + random body against random partners, at L = 16, 20, 32, both BRK arms, budgets 128 and 512:
    exact copies, ≥ 75% copies, pointer in the partner, and per copy-offset operand zz the sites carried (a site is
    carried when the offspring has the organism's byte there in every encounter) against the d − 6 law. Also the open
    straight-line tiles of §4.2."""
    rng = np.random.default_rng(20261010)
    rows, sites = [], []
    families = {"B5 zz 48 CA 50 FA": lambda zz: bytes([0xB5, zz, 0x48, 0xCA, 0x50, 0xFA]),
                "BA B5 zz 48 50 FA": lambda zz: bytes([0xBA, 0xB5, zz, 0x48, 0x50, 0xFA])}
    nb, npart = 3, 64
    for L in (16, 20, 32):
        for brk in ("faithful", "halt"):
            mach = load_machine(machine_name(L, brk))
            for fam, make in families.items():
                bodies = rng.integers(0, 256, size=(nb, L - 6), dtype=np.uint8)
                partners = rng.integers(0, 256, size=(nb, npart, L), dtype=np.uint8)
                # every zz at once: lanes (zz, body, partner)
                A = np.stack([np.stack([np.frombuffer(make(zz) + body.tobytes(), np.uint8) for body in bodies]) for zz in range(256)])   # (256, nb, L)
                mem = np.concatenate([np.repeat(A.reshape(-1, L), npart, 0), np.tile(partners.reshape(-1, L), (256, 1))], 1)
                for budget in BUDGETS:
                    r = mach.run(mem, budget=budget)
                    B = r.memory[:, L:].reshape(256, nb, npart, L)
                    A_rep = np.repeat(A[:, :, None, :], npart, 2)
                    exact = (B == A_rep).all(3)
                    rot = np.max(np.stack([(B == np.roll(A_rep, s, axis=3)).sum(3) for s in range(L)], 3), 3)
                    entered = r.fetch_map[:, L:].any(1).reshape(256, nb, npart)
                    halted = (r.halt_reason != 0).reshape(256, nb, npart)
                    for zz in range(256):
                        carried = (B[zz] == A_rep[zz]).all(1)          # (nb, L): site carried in every encounter of that body
                        rows.append({"L": L, "BRK": brk, "family": fam, "zz": f"{zz:02x}", "budget": budget,
                                     "exact": int(exact[zz].sum()), "encounters": nb * npart, "ge75": int((rot[zz] >= -(-3 * L // 4)).sum()),
                                     "pointer_in_partner": int(entered[zz].sum()), "halted": int(halted[zz].sum()),
                                     "sites_carried_all_bodies": "".join("#" if carried[:, i].all() else ("+" if carried[:, i].any() else ".") for i in range(L)),
                                     "n_carried": int(carried.all(0).sum()), "payload_carried": int(carried.all(0)[6:].sum())})
    write_csv("closers_scan.csv", rows)
    # the open straight-line tiles of §4.2, 64 random partners each, L = 16
    open_rows = []
    for brk in ("faithful", "halt"):
        mach = load_machine(machine_name(16, brk))
        for hexw in ("b50f48ca", "b50b48ca", "b51348ca", "bd0f0048"):
            w = bytes.fromhex(hexw)
            A = np.frombuffer(bytes(w[i % 4] for i in range(16)), np.uint8)
            P = rng.integers(0, 256, size=(64, 16), dtype=np.uint8)
            for budget in BUDGETS:
                r = mach.run(np.concatenate([np.repeat(A[None], 64, 0), P], 1), budget=budget)
                B = r.memory[:, 16:]
                rot = np.max(np.stack([(B == np.roll(A, s)[None]).sum(1) for s in range(16)], 1), 1)
                open_rows.append({"BRK": brk, "tile": hexw, "disassembly": dis6502_word(w), "budget": budget, "exact (of 64)": int((rot == 16).sum()),
                                  "≥75% (of 64)": int((rot >= 12).sum()), "pointer in partner (of 64)": int(r.fetch_map[:, 16:].any(1).sum())})
    write_csv("closers_open_tiles.csv", open_rows)
    log("closers written")


# ── the exploratory census's candidates on Zilion ──────────────────────────────────────────────────────────────────
def _splitmix_bytes(seed: int, n: int) -> list[int]:
    out = []
    x = seed & 0xFFFFFFFFFFFFFFFF
    M = 0xFFFFFFFFFFFFFFFF
    for _ in range(n):
        x = (x + 0x9E3779B97F4A7C15) & M
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
        out.append((z ^ (z >> 31)) & 0xFF)
    return out


def parse_exploratory() -> list[dict]:
    rows = []
    for f in sorted(EXPLORATORY.glob("out*.txt")):
        name = f.name
        if "8086" in name:
            continue
        if name.startswith("out_k"):
            L = 16
            bh = int(name.split("_bh")[1][0])
        else:
            L = int(name.split("L")[1].split("_")[0])
            bh = 0
        for line in f.read_text().splitlines():
            if not line.startswith("CAND"):
                continue
            try:
                d = dict(kv.split("=", 1) for kv in line.split()[1:])
                float(d["rand_meanmatch"])
            except Exception:
                continue
            d.update(L=L, brk_halt=bh)
            rows.append(d)
    return rows


def exploratory_encounters(words: list[bytes], k: int, L: int, brk: str, budget: int = 128) -> dict:
    """Run words through the exploratory census's own encounters on the validated core: the zero partner, its screen
    partner (splitmix64 seeded t·1000003 + 17) and its 16 assay partners (t·7919 + 99991, consecutive draws), t being its
    word index (T[i] = byte i of t). Scores are its phase scores. Returns per-word arrays."""
    mach = load_machine(machine_name(L, brk))
    thr = (3 * L + 3) // 4
    n = len(words)
    A = np.stack([np.frombuffer(bytes(w[i % k] for i in range(L)), np.uint8) for w in words])
    mem = np.zeros((n, 18, 2 * L), np.uint8)
    mem[:, :, :L] = A[:, None]
    for i, w in enumerate(words):
        te = int.from_bytes(w, "little")
        mem[i, 1, L:] = _splitmix_bytes(te * 1000003 + 17, L)
        mem[i, 2:, L:] = np.array(_splitmix_bytes(te * 7919 + 99991, 16 * L), np.uint8).reshape(16, L)
    r = mach.run(mem.reshape(-1, 2 * L), budget=budget)
    B = r.memory[:, L:].reshape(n, 18, L)
    T = np.stack([np.frombuffer(w, np.uint8) for w in words])
    j = np.arange(L)
    phase = np.max(np.stack([(B == T[:, None, (j + s) % k]).sum(2) for s in range(k)], 2), 2)
    ent = r.fetch_map[:, L:].any(1).reshape(n, 18)
    halted = (r.halt_reason != 0).reshape(n, 18)
    return {"phase": phase, "entered": ent, "halted": halted, "thr": thr,
            "passes_screen": (phase[:, 0] >= thr) | (phase[:, 1] >= thr),
            "n_exact": (phase[:, 2:] == L).sum(1), "n75": (phase[:, 2:] >= thr).sum(1)}


def cmd_explore(a) -> None:
    """Every candidate the exploratory census printed (its screen: zero-partner or first random partner ≥ 75%),
    re-run on the validated core with the exploratory census's own encounters (its splitmix64 partners, reproduced
    from its source), so that differences are the interpreter's, not the partners'."""
    cand = parse_exploratory()
    log(f"{len(cand)} exploratory candidates")
    rows = []
    groups = {}
    for d in cand:
        groups.setdefault((int(d["k"]), d["L"], d["brk_halt"]), []).append(d)
    for (k, L, bh), ds in sorted(groups.items()):
        L = int(L)
        words = [bytes.fromhex(d["tile"]) for d in ds]
        e = exploratory_encounters(words, k, L, "halt" if bh else "faithful")
        phase, ent, halted = e["phase"], e["entered"], e["halted"]
        for i, d in enumerate(ds):
            rows.append({"k": k, "L": L, "brk_halt": bh, "tile": d["tile"], "disassembly": dis6502_word(words[i], L),
                         "expl_zero_match": int(d["zero_match"].split("/")[0]), "zil_zero_match": int(phase[i, 0]),
                         "expl_zero_entered": int(d["zero_entered"]), "zil_zero_entered": int(ent[i, 0]),
                         "expl_zero_halted": int(d["zero_halted"]), "zil_zero_halted": int(halted[i, 0]),
                         "expl_rand_exact": int(d["rand_exact"].split("/")[0]), "zil_rand_exact": int(e["n_exact"][i]),
                         "expl_rand_75": int(d["rand_75"].split("/")[0]), "zil_rand_75": int(e["n75"][i]),
                         "expl_entered_any": int(d["entered_any"]), "zil_entered_any": int(ent[i, [0, *range(2, 18)]].any()),
                         "expl_rand_halted": int(d["rand_halted"]), "zil_rand_halted": int(halted[i, 2:].sum()),
                         "expl_rand_meanmatch": d["rand_meanmatch"], "zil_rand_meanmatch": f"{phase[i, 2:].sum() / (16 * L):.3f}",
                         "zil_screen_match": int(phase[i, 1])})
        log(k, L, bh, len(ds))
    write_csv("exploratory_rerun.csv", rows)


# ── report ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def write_csv(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("control")
    sub.add_parser("planted")
    p = sub.add_parser("census")
    p.add_argument("--L", type=int, default=16)
    p.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--brk", nargs="+", default=["faithful", "halt"], choices=["faithful", "halt"])
    p.add_argument("--chunk-bits", type=int, default=24)
    p.add_argument("--max-minutes", type=float, default=None)
    p.add_argument("--pause", default="auto", help="seconds to sleep between chunks; 'auto' (default): as long as the chunk took (the GPU is shared)")
    p.add_argument("--no-report", action="store_true", help="do not regenerate CENSUS.md afterwards")
    sub.add_parser("closers")
    sub.add_parser("explore")
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "report":
        from census6502_report import report

        report()
        return
    {"control": cmd_control, "planted": cmd_planted, "census": cmd_census, "closers": cmd_closers, "explore": cmd_explore}[a.cmd](a)


if __name__ == "__main__":
    main()
