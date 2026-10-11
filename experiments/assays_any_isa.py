"""Algocell's frozen assays on any Zilion machine: the evidence (writes results/assays_any_isa/).

    .venv/bin/python assays_any_isa.py pairs   [--n 10000]                 # PAIRS.md, pairs.csv
    .venv/bin/python assays_any_isa.py regress [--only id,id] [--work DIR]  # REGRESSION.md, regression.csv
    .venv/bin/python assays_any_isa.py smoke                                # SMOKE_6502.md

pairs    Every vendored assay machine (machines/worlds_assays.json, Z80 compat '0.2') against the Algocell shader it
         replaces, on lattice-neighbour pairs drawn from every archived final soup of its tape length and ring (each
         condition under its own step budget and suppression set) and on uniform random pairs (with random suppression
         sets): final memory, writes into A and B, the fetch map (traced executor) and, where the op-traced executor
         exists (unsuppressed, ring 2L), the start map. Negative controls: a machine one instruction short, and the lethal
         machine against the plain shader, must both mismatch.
regress  Each frozen script runs three times as a subprocess: (a) as registered (Algocell's shaders, no flag), (b) with
         `--machine z80c02_assay`, (n, some jobs) on the machine set one instruction short (a negative control, which
         must differ). Every output file of (a) is compared with (b) byte for byte; columns or lines that hold wall-clock
         times, and the `executor` provenance column (shader file under (a), machine under (b)), are compared separately
         and listed. (a) is also compared with the committed frozen table (the rows it regenerates). Each run logs the
         pairs it executed per backend (ALGOCELL_EXECUTOR_LOG), so the report shows that (a) ran only shaders and (b)
         only machines, on the same number of pairs.
smoke    The generalised assay functions on the 6502 machine (m6502_assay) with uniform random tapes: plumbing only.
         Nothing about what 6502 programs do is computed or reported (the 6502 test is not yet registered).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results" / "assays_any_isa"
ARCHIVE = Path(os.environ.get("ALGOCELL_RUNS", "/Users/neo/repos/algocell/experiments/runs"))   # archived runs (read-only)
MACHINE_SET = "z80c02_assay"
M6502 = "m6502_assay"
PY = sys.executable


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(HERE))
    except ValueError:
        return str(p)


# ══ pairs ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
def archived_conditions() -> dict:
    """{(L, P, zero_halts, steps, suppress-tuple): [soup records]} over every archived final soup (square and mixed
    grids alike: a pair is a pair) with a summary, at the tape lengths and rings of the assay machines."""
    out: dict = {}
    for sp in sorted(glob.glob(str(ARCHIVE / "**" / "*.summary.json"), recursive=True)):
        stem = sp[: -len(".summary.json")]
        soup = stem + ".soup_final.u8.br"
        if not os.path.exists(soup):
            continue
        s = json.load(open(sp))
        if "tape_length" not in s or "z80_steps" not in s:
            continue
        L = int(s["tape_length"])
        P = int(s.get("mem_length") or 2 * L)
        sup = s.get("suppress") or []
        if isinstance(sup, str):
            from algocell_exp.isa import parse_patterns
            sup = parse_patterns(sup)
        key = (L, P, bool(s.get("zero_halts") or False), int(s["z80_steps"]), tuple(sup))
        out.setdefault(key, []).append({"path": Path(soup), "L": L, "width": int(s.get("width", 160)), "height": int(s.get("height", 125)),
                                        "stage": Path(sp).relative_to(ARCHIVE).parts[0]})
    return out


def sample_pairs(recs: list[dict], n: int, seed: int) -> np.ndarray:
    from algocell_exp import zsoup_check as C

    rng = np.random.default_rng(seed)
    per = np.bincount(rng.integers(0, len(recs), size=n), minlength=len(recs))
    out = []
    for r, k in zip(recs, per):
        if k == 0:
            continue
        soup = C.load_soup(r)
        out.append(C.neighbour_pairs(soup, int(k), np.random.default_rng([seed, C.path_seed(r["path"])]), r["width"], r["height"]))
    return np.concatenate(out)


def compare_runs(ra, rb, maps: tuple) -> dict:
    d = {"mem": int((ra.memory != rb.memory).any(1).sum()), "writes": int((ra.writes != rb.writes).any(1).sum())}
    for m in maps:
        d[m] = int((getattr(ra, m) != getattr(rb, m)).any(1).sum())
    return d


def cmd_pairs(a) -> None:
    from algocell_exp import executor as E
    from algocell_exp.isa import selectable_families

    OUT.mkdir(parents=True, exist_ok=True)
    sh, zx = E.AlgocellShader(), E.ZilionExecutor(MACHINE_SET)
    neg = E.ZilionExecutor(MACHINE_SET, budget_delta=-1)
    machines = json.loads((HERE / "algocell_exp" / "machines" / "worlds_assays.json").read_text())
    z80 = {n: s for n, s in machines.items() if s["core"]["isa"] == "z80"}
    conds = archived_conditions()
    rows = []
    t0 = time.time()

    def one(machine, source, L, zh, steps, sup, pairs, n_worlds):
        maps = ("fetch_map",)
        ra = sh.run(pairs, L, steps, sup, zero_halts=zh, fetch_map=True)
        rb = zx.run(pairs, L, steps, sup, zero_halts=zh, fetch_map=True)
        assert rb.executor == machine, (rb.executor, machine)
        d = compare_runs(ra, rb, maps)
        if not sup:
            sa = sh.run(pairs, L, steps, (), zero_halts=zh, start_map=True)
            sb = zx.run(pairs, L, steps, (), zero_halts=zh, fetch_map=True, start_map=True)
            ds = compare_runs(sa, sb, ("fetch_map", "start_map"))
            d["start_map"] = ds["start_map"]
            d["mem"] += ds["mem"]
            d["fetch_map"] += ds["fetch_map"]
            d["writes"] += ds["writes"]
        else:
            d["start_map"] = None
        rows.append({"machine": machine, "source": source, "L": L, "zero_halts": zh, "steps": steps, "suppress": ";".join(sup) or "-",
                     "worlds": n_worlds, "pairs": len(pairs), "distinct": int(len(np.unique(pairs, axis=0))),
                     "changed": int((ra.memory != pairs).any(1).sum()), "entered": int(ra.fetch_map[:, L:].any(1).sum()),
                     "mm_memory": d["mem"], "mm_writes": d["writes"], "mm_fetch": d["fetch_map"], "mm_start": d["start_map"]})
        r = rows[-1]
        log(f"{machine:>24} {source:>9} st{steps:<4} sup={r['suppress'][:40]:40s} {len(pairs):>6} pairs: mismatches mem {r['mm_memory']} writes {r['mm_writes']} "
            f"fetch {r['mm_fetch']} start {r['mm_start']} ({time.time() - t0:.0f} s)")

    for machine, spec in z80.items():
        seg = spec["world"]["memory"]["segments"]
        L, P = seg[0]["bytes"], spec["world"]["memory"]["ring"]
        zh = "halting" in spec["world"]
        # archived soups of this L and ring (lethal machines: the soups of their own rule if any, else the plain soups of
        # this L run under the lethal rule: evolved tapes, another rule)
        keys = [k for k in conds if k[0] == L and k[1] == P and k[2] == zh]
        src = "archived"
        if not keys and zh:
            keys = [k for k in conds if k[0] == L and k[1] == P]
            src = "arch-plain"
        for k in sorted(keys, key=str):
            _, _, _, steps, sup = k
            recs = conds[k]
            seed = int.from_bytes(hashlib.sha256(repr((machine, k)).encode()).digest()[:4], "little")
            one(machine, src, L, zh, steps, list(sup), sample_pairs(recs, a.n, seed), len(recs))
        # uniform random pairs: no suppression, and three random suppression sets (two random selectable families each)
        rng = np.random.default_rng([1, L, int(zh)])
        fams = [f["id"] for f in selectable_families()]
        for i in range(4):
            sup = [] if i == 0 else [f"family:{f}" for f in rng.choice(fams, size=2, replace=False)]
            pairs = rng.integers(0, 256, size=(a.n, 2 * L), dtype=np.uint8)
            one(machine, "random", L, zh, 128, sup, pairs, 0)
    # negative controls (must mismatch)
    rng = np.random.default_rng(99)
    recs = conds[(16, 32, False, 128, ())]
    pairs = sample_pairs(recs, a.n, 7)
    ra = sh.run(pairs, 16, 128, fetch_map=True)
    nb = compare_runs(ra, neg.run(pairs, 16, 128, fetch_map=True), ("fetch_map",))
    nl = compare_runs(ra, E.ZilionExecutor("z80c02_assay_lethal_L16").run(pairs, 16, 128, zero_halts=True, fetch_map=True), ("fetch_map",))
    controls = [("machine budget 127 against the shader at 128", nb), ("lethal machine against the plain shader", nl)]
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "pairs.csv", index=False)
    write_pairs_md(T, controls, a)


def write_pairs_md(T: pd.DataFrame, controls, a) -> None:
    lines = ["# Assay machines against Algocell's shaders, pair by pair (generated by `assays_any_isa.py pairs`; do not edit)", "",
             f"Each vendored assay machine (`machines/worlds_assays.json`, provenance in `machines/PROVENANCE_assays.md`) runs the same "
             f"pairs as the Algocell shader it replaces (`algocell_exp.executor.AlgocellShader`: `z80_test[_lethal]_L<L>.wgsl` through the "
             f"frozen host functions). Per pair: final memory (both tapes), the writes into A and into B, and the fetch map "
             f"(traced executor, `exectrace`); for unsuppressed groups also the start map and again the fetch map (op-traced executor, "
             f"`optrace`), whose mismatches are added to the same columns. Archived pairs: lattice neighbours drawn uniformly over the "
             f"worlds of each archived condition with a final soup (up to {a.n} per condition, `zsoup_check.neighbour_pairs`), each "
             f"run under its condition's own step budget and suppression set. `arch-plain`: no archived soup has this machine's rule, so "
             f"the plain soups of its length run under the lethal rule. Random: {a.n} uniform pairs, unsuppressed and under three random "
             f"suppression sets of two selectable families. `changed`: pairs whose memory the encounter changed; `entered`: pairs in "
             f"which a partner position was fetched (shader's map).", ""]
    g = T.groupby("machine", sort=False).agg(groups=("source", "size"), pairs=("pairs", "sum"), changed=("changed", "sum"), entered=("entered", "sum"),
                                              mm_memory=("mm_memory", "sum"), mm_writes=("mm_writes", "sum"), mm_fetch=("mm_fetch", "sum"),
                                              mm_start=("mm_start", lambda x: int(np.nansum(x.astype(float)))),
                                              start_pairs=("mm_start", lambda x: int(T.loc[x.index][x.notna()]["pairs"].sum()))).reset_index()
    lines += ["| machine | groups | pairs | changed by execution | entered the partner | pairs with start map | mismatches: memory / writes / fetch map / start map |",
              "|---|---|---|---|---|---|---|"]
    for r in g.itertuples():
        lines.append(f"| `{r.machine}` | {r.groups} | {r.pairs:,} | {r.changed:,} | {r.entered:,} | {r.start_pairs:,} | {r.mm_memory} / {r.mm_writes} / {r.mm_fetch} / {r.mm_start} |")
    tot = T[["pairs", "mm_memory", "mm_writes", "mm_fetch"]].sum()
    lines += ["", f"Total: {int(tot.pairs):,} pairs; mismatches {int(tot.mm_memory)} memory, {int(tot.mm_writes)} writes, {int(tot.mm_fetch)} fetch map, "
              f"{int(np.nansum(T.mm_start.astype(float)))} start map.", "",
              "Negative controls (each must mismatch; same archived L = 16 pairs, 128 instructions, fetch maps):", ""]
    for name, d in controls:
        lines.append(f"- {name}: {d['mem']} memory, {d['writes']} writes, {d['fetch_map']} fetch-map mismatches")
    lines += ["", "Per group: `pairs.csv`.", ""]
    (OUT / "PAIRS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ══ regress ═════════════════════════════════════════════════════════════════════════════════════════════════════════
def link_stage(w: Path, stage: str, nested: bool) -> None:
    """A run directory of symlinks to the archived stage's files (assay_batch writes <dir>/analysis, never the archive)."""
    src = ARCHIVE / stage / (stage if nested else "")
    dst = w / stage / (stage if nested else "")
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not (dst / f.name).exists():
            os.symlink(f, dst / f.name)


# Each job: id, family, commands (argv after the interpreter; {w} = the path's work dir), outputs {file: committed or None},
# committed row filters, timing columns, provenance columns, volatile line patterns, setup, negative control.
R = "results"
JOBS = [
    dict(id="assay_batch_I", family="culture test", setup=("stageI", False), neg=True,
         cmds=[["assay_batch.py", "{w}/stageI"]],
         outputs={"stageI/analysis/assays.csv": f"{R}/stageI/assays.csv"}, prov=["executor"]),
    dict(id="c4_I", family="culture test", after="assay_batch_I",
         cmds=[["c4_functional.py", "{w}/stageI"]],
         outputs={"stageI/analysis/c4/functional.csv": f"{R}/stageI/c4/functional.csv",
                  "stageI/analysis/c4/functional_fixed_steps.csv": f"{R}/stageI/c4/functional_fixed_steps.csv",
                  "stageI/analysis/c4/NUMBERS_C4.md": f"{R}/stageI/c4/NUMBERS_C4.md"}),
    dict(id="stage_g_I", family="partner test", after="c4_I",
         cmds=[["stage_g.py", "--dir", "{w}/stageI", "--out", "{w}/stageI/analysis/stageI", "--no-verdicts"]],
         outputs={"stageI/analysis/stageI/stage_g_runs.csv": f"{R}/stageI/stageI/stage_g_runs.csv",
                  "stageI/analysis/stageI/NUMBERS_G.md": None}),
    dict(id="assay_batch_G", family="culture test", setup=("stageG", False),
         cmds=[["assay_batch.py", "{w}/stageG"]],
         outputs={"stageG/analysis/assays.csv": f"{R}/stageG/assays.csv"}, prov=["executor"]),
    dict(id="stage_g_G", family="partner test", after="assay_batch_G",
         cmds=[["stage_g.py", "--dir", "{w}/stageG", "--out", "{w}/stageG/analysis/stageG", "--no-verdicts"]],
         outputs={"stageG/analysis/stageG/stage_g_runs.csv": f"{R}/stageG/stageG/stage_g_runs.csv"}),
    dict(id="assay_batch_M", family="culture test", setup=("stageM", True),
         cmds=[["assay_batch.py", "{w}/stageM/stageM"]],
         outputs={"stageM/stageM/analysis/assays.csv": f"{R}/stageM/assays.csv"}, prov=["executor"]),
    dict(id="individuality", family="partner test (inflow)",
         cmds=[["biology/individuality.py", "--skip-bff", "--out", "{w}/individuality"]],
         outputs={"individuality/per_replicator.csv": f"{R}/biology/individuality/per_replicator.csv", "individuality/summary.csv": None,
                  "individuality/predictions.csv": None, "individuality/offspring_hist.json": None},
         filters={"individuality/per_replicator.csv": "machine == 'z80'"}),
    dict(id="exec_trace", family="traced execution (confinement)", neg=True,
         cmds=[["exec_trace.py", "--yes", "--out", "{w}/exectrace"]],
         outputs={"exectrace/per_replicator.csv": f"{R}/exectrace/per_replicator.csv", "exectrace/AGREEMENT.md": f"{R}/exectrace/AGREEMENT.md"}),
    dict(id="inflow_insoup", family="traced execution (inflow)",
         cmds=[["inflow_insoup.py", "--yes", "--out", "{w}/inflow_insoup"]],
         outputs={"inflow_insoup/per_world.csv": f"{R}/inflow_insoup/per_world.csv", "inflow_insoup/sensitivity.csv": f"{R}/inflow_insoup/sensitivity.csv",
                  "inflow_insoup/INFLOW_INSOUP.md": f"{R}/inflow_insoup/INFLOW_INSOUP.md"}),
    dict(id="mutscan_G", family="single-mutant scan",
         cmds=[["mutscan.py", "--yes", "--fresh", "--out", "{w}/mutscan"], ["mutscan.py", "--report", "--out", "{w}/mutscan"]],
         outputs={"mutscan/mutscan_tapes.csv": f"{R}/mutscan/mutscan_tapes.csv", "mutscan/mutscan_sites.csv": f"{R}/mutscan/mutscan_sites.csv",
                  "mutscan/REPORT.md": f"{R}/mutscan/REPORT.md"}, timing=["wall_s"],
         # the committed tables were written by mutscan.py's incremental mode (--L): the L = 16 rows, computed first, were
         # read back with pandas' default float parser (not round-trip exact) and re-written when the other lengths ran
         replay={"mutscan/mutscan_tapes.csv": "L == 16", "mutscan/mutscan_sites.csv": "L == 16"}),
    dict(id="mutscan_I", family="single-mutant scan", neg=True,
         cmds=[["mutscan.py", "--yes", "--fresh", "--runs", f"{R}/stageI/stageI/stage_g_runs.csv", "--L", "16", "--zero-halts", "--out", "{w}/mutscan_I"]],
         outputs={"mutscan_I/mutscan_tapes.csv": f"{R}/mutscan_I/mutscan_tapes.csv", "mutscan_I/mutscan_sites.csv": f"{R}/mutscan_I/mutscan_sites.csv"},
         timing=["wall_s"]),
    dict(id="mutscan_K", family="single-mutant scan",
         cmds=[["mutscan.py", "--yes", "--fresh", "--runs", f"{R}/stageK/stageK/stage_g_runs.csv", "--L", "32", "--out", "{w}/mutscan_K"]],
         outputs={"mutscan_K/mutscan_tapes.csv": f"{R}/mutscan_K/mutscan_tapes.csv", "mutscan_K/mutscan_sites.csv": f"{R}/mutscan_K/mutscan_sites.csv"},
         timing=["wall_s"]),
    dict(id="serial_retention", family="serial transfer", neg=True,
         cmds=[["serial_retention.py", "--yes", "--out", "{w}/serial_retention"]],
         outputs={"serial_retention/sr_summary.csv": f"{R}/serial_retention/sr_summary.csv", "serial_retention/sr_sites.csv": f"{R}/serial_retention/sr_sites.csv",
                  "serial_retention/sr_mutants.csv.gz": f"{R}/serial_retention/sr_mutants.csv.gz", "serial_retention/REPORT.md": f"{R}/serial_retention/REPORT.md"},
         timing=["wall_s"]),
    dict(id="branching", family="branching transfer", neg=True,
         cmds=[["branching_transfer.py", "--out", "{w}/branching"], ["branching_table.py", "--dir", "{w}/branching"]],
         outputs={"branching/branching_summary.csv": f"{R}/branching/branching_summary.csv", "branching/branching_sites.csv": f"{R}/branching/branching_sites.csv",
                  "branching/branching_mutants.csv.gz": f"{R}/branching/branching_mutants.csv.gz", "branching/BRANCHING.md": f"{R}/branching/BRANCHING.md",
                  "branching/SUPP_TABLE.md": f"{R}/branching/SUPP_TABLE.md", "branching/runtime.txt": None},
         timing=["gen_wall_s", "arm_wall_s"], prov=["executor"], lines=[r"^Runtime: ", r"^total wall "]),
    dict(id="population_classes_I", family="population classes",
         cmds=[["population_classes.py", "--yes", "--stages", "I", "--out", "{w}/population_I"]],
         outputs={"population_I/classes_snapshots.csv": f"{R}/population/classes_snapshots.csv", "population_I/classes_cells.csv": f"{R}/population/classes_cells.csv",
                  "population_I/REPORT_classes.md": None},
         filters={"population_I/classes_snapshots.csv": "stage == 'I'", "population_I/classes_cells.csv": "stage == 'I'"}),
    dict(id="population_classes_M", family="population classes",
         cmds=[["population_classes.py", "--yes", "--stages", "M", "--finals-only", "--out", "{w}/population_M"]],
         outputs={"population_M/classes_snapshots.csv": f"{R}/population/classes_snapshots.csv", "population_M/classes_cells.csv": f"{R}/population/classes_cells.csv",
                  "population_M/REPORT_classes.md": None},
         filters={"population_M/classes_snapshots.csv": "stage == 'M' and snapshot == 'final'", "population_M/classes_cells.csv": "stage == 'M' and snapshot == 'final'"}),
    dict(id="population_sample_I", family="population classes (sample)", neg=True, neg_delta=-64,   # its fractions do not move at -1
         cmds=[["population_sample.py", "--yes", "--stages", "I", "--out", "{w}/population_q_I"]],
         outputs={"population_q_I/population_worlds.csv": f"{R}/population/population_worlds.csv", "population_q_I/population_cells.csv": f"{R}/population/population_cells.csv",
                  "population_q_I/REPORT.md": None},
         filters={"population_q_I/population_worlds.csv": "stage == 'I'", "population_q_I/population_cells.csv": "stage == 'I'"}),
]

NOEXEC = {"branching_table.py"}   # formats a table from files; executes no pair and takes no --machine
NEG_WRAPPER = ("import runpy, sys; sys.path.insert(0, {here!r}); from algocell_exp import executor as E; "
               "E.select(E.ZilionExecutor({ms!r}, budget_delta={delta})); sys.argv = {argv!r}; runpy.run_path({script!r}, run_name='__main__')")


def run_job(job: dict, path: str, work: Path) -> dict:
    w = work / path
    w.mkdir(parents=True, exist_ok=True)
    if job.get("setup"):
        link_stage(w, *job["setup"])
    logf = w / f"{job['id']}.usage.json"
    if logf.exists():
        logf.unlink()
    env = dict(os.environ, ALGOCELL_EXECUTOR_LOG=str(logf))
    t0 = time.time()
    usage = {"shader": {}, "machine": {}}
    for k, c in enumerate(job["cmds"]):
        argv = [x.replace("{w}", str(w)) for x in c]
        lf = w / f"{job['id']}.{k}.usage.json"
        env["ALGOCELL_EXECUTOR_LOG"] = str(lf)
        if path == "a" or argv[0] in NOEXEC:
            cmd = [PY] + argv
        elif path == "b":
            cmd = [PY] + argv + ["--machine", MACHINE_SET]
        else:
            cmd = [PY, "-c", NEG_WRAPPER.format(here=str(HERE), ms=MACHINE_SET, argv=argv, script=str(HERE / argv[0]), delta=job.get("neg_delta", -1))]
        with open(w / f"{job['id']}.{k}.log", "w") as fh:
            p = subprocess.run(cmd, cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            raise SystemExit(f"{job['id']} ({path}) command {k} failed with {p.returncode}: see {w / (job['id'] + f'.{k}.log')}")
        if lf.exists():
            u = json.loads(lf.read_text())["usage"]
            for kind in ("shader", "machine"):
                for n, v in u[kind].items():
                    usage[kind][n] = usage[kind].get(n, 0) + v
    logf.write_text(json.dumps(usage, indent=1))
    return {"wall_s": round(time.time() - t0, 1), "usage": usage}


def read_bytes(p: Path) -> bytes:
    b = Path(p).read_bytes()
    return gzip.decompress(b) if str(p).endswith(".gz") else b


def read_table(p: Path, query: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(read_bytes(p)), dtype=str, keep_default_na=False)
    if query:
        q = pd.read_csv(io.BytesIO(read_bytes(p)), keep_default_na=True)
        keep = q.query(query).index
        df = df.loc[keep]
    return df.reset_index(drop=True)


def replay_cycle(fa: Path, cycled: str) -> Path:
    """(a)'s table with the rows matching `cycled` read back once with pandas' default float parser and the file
    re-written, as an incremental run of the script re-writes rows it computed earlier; the other rows keep their exact
    values. Written beside (a) as <name>.replayed.csv."""
    full = pd.read_csv(fa)
    raw = pd.read_csv(fa, float_precision="round_trip")
    m = full.eval(cycled)
    out = pd.concat([full[m], raw[~m]]).sort_index()
    dst = fa.with_name(fa.name.replace(".csv", ".replayed.csv"))
    out.to_csv(dst, index=False)
    return dst


def compare_files(fa: Path, fb: Path, job: dict, against_committed: bool, query: str | None = None) -> dict:
    """status: identical (raw bytes; decompressed for .gz), identical-except (only the listed volatile columns or lines
    differ), DIFFERENT, or missing. For the committed comparison `query` selects the committed rows the job regenerates."""
    if not fa.exists() or not fb.exists():
        return {"status": "missing", "detail": f"{'a' if not fa.exists() else 'b'} missing"}
    ba, bb = read_bytes(fa), read_bytes(fb)
    if ba == bb and not query:
        return {"status": "identical", "detail": ""}
    name = str(fa)
    timing = list(job.get("timing", []))
    prov = [] if against_committed else list(job.get("prov", []))
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        A_ = read_table(fa)
        B_ = read_table(fb, query)
        extra = ""
        if list(A_.columns) != list(B_.columns):
            added = [c for c in A_.columns if c not in B_.columns]
            if against_committed and [c for c in A_.columns if c in B_.columns] == list(B_.columns) and added:
                # the committed table predates columns the script writes now: compare every committed column
                A_ = A_[list(B_.columns)]
                extra = f"every committed column identical; (a) also writes {', '.join(f'`{c}`' for c in added)}, which the committed table predates"
            else:
                return {"status": "DIFFERENT", "detail": f"columns differ: {sorted(set(A_.columns) ^ set(B_.columns))}"}
        if len(A_) != len(B_):
            return {"status": "DIFFERENT", "detail": f"{len(A_)} rows vs {len(B_)}"}
        vol = [c for c in timing + prov if c in A_.columns]
        keep = [c for c in A_.columns if c not in vol]
        same = A_[keep].equals(B_[keep])
        how = ""
        if not same:
            # identical up to row order?
            sa = A_[keep].sort_values(keep).reset_index(drop=True)
            sb = B_[keep].sort_values(keep).reset_index(drop=True)
            if sa.equals(sb):
                return {"status": "identical-except", "detail": "same rows in another order", "rows": len(A_)}
            diff = (A_[keep] != B_[keep])
            cols = [c for c in keep if diff[c].any()]
            return {"status": "DIFFERENT", "detail": f"{int(diff.any(axis=1).sum())} of {len(A_)} rows differ in {len(cols)} columns (e.g. {', '.join(cols[:6])})", "rows": len(A_)}
        varied = [c for c in vol if not A_[c].equals(B_[c])]
        detail = []
        for c in varied:
            if c in prov:
                mp = sorted(set(zip(A_[c], B_[c])))
                detail.append(f"`{c}` " + ", ".join(f"{x} → {y}" for x, y in mp))
            else:
                detail.append(f"`{c}` (wall-clock time)")
        sub = f"committed rows where {query}" if query else ""
        if extra:
            detail.insert(0, extra)
        if not varied and not extra:
            if query:
                return {"status": "identical", "detail": sub + " (cell for cell)", "rows": len(A_)}
            return {"status": "identical-except", "detail": "every cell identical, the raw bytes differ", "rows": len(A_)}
        return {"status": "identical-except", "detail": "; ".join(detail) + (f" [{sub}]" if query else ""), "rows": len(A_)}
    # text: drop volatile lines
    pats = [re.compile(p) for p in job.get("lines", [])]
    la = [x for x in ba.decode().splitlines() if not any(p.search(x) for p in pats)]
    lb = [x for x in bb.decode().splitlines() if not any(p.search(x) for p in pats)]
    if la == lb:
        return {"status": "identical-except", "detail": "lines matching " + ", ".join(f"`{p.pattern}`" for p in pats)}
    nd = sum(1 for x, y in zip(la, lb) if x != y) + abs(len(la) - len(lb))
    return {"status": "DIFFERENT", "detail": f"{nd} lines differ"}


def compare_committed(fa: Path, fc: Path, job: dict, f: str) -> dict:
    """(a) against the committed frozen table; where the committed file went through a documented re-write (`replay`),
    the raw comparison is reported too and the replayed one decides."""
    r = compare_files(fa, fc, job, True, (job.get("filters") or {}).get(f))
    cyc = (job.get("replay") or {}).get(f)
    if cyc and r["status"] == "DIFFERENT" and fa.exists():
        r2 = compare_files(replay_cycle(fa, cyc), fc, job, True, (job.get("filters") or {}).get(f))
        r2["detail"] = (f"after replaying the committed file's incremental re-write on (a) (rows where {cyc} read back once with pandas' "
                        f"default float parser): {r2['status']}{': ' + r2['detail'] if r2.get('detail') else ''}; raw: {r['detail']} "
                        f"(last-digit float differences)")
        r2["status"] = "identical-except" if r2["status"] in ("identical", "identical-except") else "DIFFERENT"
        return r2
    return r


def cmd_regress(a) -> None:
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    state_p = OUT / "regression_state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() and only else {}
    for job in JOBS:
        if only and job["id"] not in only:
            continue
        paths = ["a", "b"] + (["n"] if job.get("neg") else [])
        res = {"id": job["id"], "family": job["family"], "cmds": [" ".join(c) for c in job["cmds"]], "runs": {}, "files": []}
        for p in paths:
            log(f"[{job['id']}] path {p} ...")
            res["runs"][p] = run_job(job, p, work)
            log(f"[{job['id']}] path {p} done in {res['runs'][p]['wall_s']} s: {res['runs'][p]['usage']}")
        for f, committed in job["outputs"].items():
            fa, fb = work / "a" / f, work / "b" / f
            row = {"file": f, "sha_a": sha(fa) if fa.exists() else None, "sha_b": sha(fb) if fb.exists() else None,
                   "a_vs_b": compare_files(fa, fb, job, False)}
            if job.get("neg"):
                fn = work / "n" / f
                row["a_vs_n"] = compare_files(fa, fn, job, False) if fn.exists() else {"status": "missing", "detail": ""}
            if committed:
                row["committed"] = committed
                row["a_vs_committed"] = compare_committed(fa, HERE / committed, job, f)
            res["files"].append(row)
            log(f"   {f}: a/b {row['a_vs_b']['status']} {row['a_vs_b']['detail'][:120]}" + (f" | a/committed {row['a_vs_committed']['status']} {row['a_vs_committed']['detail'][:80]}" if committed else "")
                + (f" | a/n {row['a_vs_n']['status']}" if 'a_vs_n' in row else ""))
        state[job["id"]] = res
        state_p.write_text(json.dumps(state, indent=1))
    write_regression_md(state)


def recompare(work: Path) -> None:
    """Recompute every file comparison of the recorded jobs from the outputs in `work` (no job is rerun)."""
    state_p = OUT / "regression_state.json"
    state = json.loads(state_p.read_text())
    jobs = {j["id"]: j for j in JOBS}
    for jid, res in state.items():
        job = jobs[jid]
        for row in res["files"]:
            f = row["file"]
            fa, fb = work / "a" / f, work / "b" / f
            row["sha_a"], row["sha_b"] = (sha(fa) if fa.exists() else None), (sha(fb) if fb.exists() else None)
            row["a_vs_b"] = compare_files(fa, fb, job, False)
            if "a_vs_n" in row:
                row["a_vs_n"] = compare_files(fa, work / "n" / f, job, False)
            if row.get("committed"):
                row["a_vs_committed"] = compare_committed(fa, HERE / row["committed"], job, f)
    state_p.write_text(json.dumps(state, indent=1))
    write_regression_md(state)


def write_regression_md(state: dict) -> None:
    from algocell_exp import zmachine

    order = [j["id"] for j in JOBS if j["id"] in state]
    lines = ["# Frozen assay tables on Algocell's shaders and on Zilion machines (generated by `assays_any_isa.py regress`; do not edit)", "",
             "Each frozen script was run three ways, as a subprocess from `experiments/`, writing into a scratch directory: "
             "**(a)** exactly as registered (no flag: Algocell's executor shaders); **(b)** with `--machine z80c02_assay` (every encounter on "
             "the vendored Zilion 0.2.0 Z80 machines of `machines/worlds_assays.json`); **(n)**, for some jobs, a negative control: the same machines "
             "one instruction short (`ZilionExecutor(..., budget_delta=-1)`; 64 instructions short for `population_sample_I`, whose fractions "
             "do not move at one), which must not reproduce (a): at least one of its files must differ. Run directories for the stage "
             "jobs are symlinks to the archived run files. Each run logged the pairs it executed per backend (`ALGOCELL_EXECUTOR_LOG`).", "",
             "Comparison: **identical** = the files are byte-identical (`.csv.gz` compared decompressed); **identical-except** = every byte "
             "is identical except the listed columns or lines, which hold wall-clock times or the `executor` provenance column (the shader "
             "file under (a), the machine under (b): the mapping is listed). **(a) vs committed** compares (a) with the frozen table in "
             "`experiments/results/` (where the job regenerates a subset, the committed rows selected by the filter shown).", ""]
    # summary table
    lines += ["## Summary", "", "| job | family | (a) vs (b) | (a) vs committed | (a) vs (n) | pairs (a): shader / machine | pairs (b): shader / machine |",
              "|---|---|---|---|---|---|---|"]
    tot_a = tot_b = 0
    for jid in order:
        r = state[jid]

        def st(key):
            from collections import Counter

            c = Counter(f[key]["status"] for f in r["files"] if key in f)
            if not c:
                return "–"
            if key == "a_vs_n":
                k = sum(v for s_, v in c.items() if s_ == "DIFFERENT")
                return f"{'**differs**' if k else '**DOES NOT DIFFER**'} ({k} of {sum(c.values())} files)"
            return ", ".join(f"{v} {s_}" for s_, v in sorted(c.items()))
        ua, ub = r["runs"]["a"]["usage"], r["runs"]["b"]["usage"]
        sa, ma = sum(ua["shader"].values()), sum(ua["machine"].values())
        sb, mb = sum(ub["shader"].values()), sum(ub["machine"].values())
        neg = st("a_vs_n")
        tot_a += sa
        tot_b += mb
        lines.append(f"| `{jid}` | {r['family']} | {st('a_vs_b')} | {st('a_vs_committed')} | {neg} | {sa:,} / {ma:,} | {sb:,} / {mb:,} |")
    nfiles = sum(len(state[j]["files"]) for j in order)
    bad = [(j, f["file"]) for j in order for f in state[j]["files"] if f["a_vs_b"]["status"] not in ("identical", "identical-except")]
    badc = [(j, f["file"]) for j in order for f in state[j]["files"] if f.get("a_vs_committed") and f["a_vs_committed"]["status"] not in ("identical", "identical-except")]
    lines += ["", f"{len(order)} jobs, {nfiles} files; (a) ran {tot_a:,} pairs on Algocell's shaders and (b) {tot_b:,} on the machines. "
              f"(a) vs (b): {'every file identical or identical-except' if not bad else 'DIFFERENT: ' + ', '.join(f'{j}/{f}' for j, f in bad)}. "
              f"(a) vs committed: {'every compared file identical or identical-except' if not badc else 'DIFFERENT: ' + ', '.join(f'{j}/{f}' for j, f in badc)}."]
    lines += ["", "## Per file", ""]
    for jid in order:
        r = state[jid]
        lines.append(f"### `{jid}` ({r['family']})")
        lines.append("")
        lines.append("Commands: " + "; ".join(f"`{c}`" for c in r["cmds"]) + " (path (b) appends `--machine z80c02_assay`).")
        for p, run in r["runs"].items():
            u = run["usage"]
            lines.append(f"- run ({p}), {run['wall_s']} s: shaders " + (", ".join(f"`{k}` {v:,}" for k, v in u["shader"].items()) or "none")
                         + "; machines " + (", ".join(f"`{k}` {v:,}" for k, v in u["machine"].items()) or "none"))
        lines.append("")
        lines.append("| file | SHA-256 (a), first 16 | SHA-256 (b), first 16 | (a) vs (b) | (a) vs committed | (a) vs (n) |")
        lines.append("|---|---|---|---|---|---|")
        for f in r["files"]:
            ab, ac, an = f["a_vs_b"], f.get("a_vs_committed"), f.get("a_vs_n")
            fmt = lambda d: "–" if d is None else (f"**{d['status']}**" + (f": {d['detail']}" if d.get("detail") else ""))  # noqa: E731
            lines.append(f"| `{f['file']}` | `{(f['sha_a'] or '')[:16]}` | `{(f['sha_b'] or '')[:16]}` | {fmt(ab)} | "
                         + (f"`{f['committed']}`: {fmt(ac)}" if ac else "–") + f" | {fmt(an)} |")
        lines.append("")
    lines += ["## Not regenerated here", "",
              "Frozen tables of the same assay families that were not regenerated, and why:", "",
              "- `assay_batch.py` on Stages A–F, H, K and L, and `c4_functional.py` beyond Stage I: same code path as the stages above; "
              "Stages B–F also need tape lengths or rings with no vendored machine (L = 3–12, 18, 24, 25, 36, 49, 81, 100; Stage F's P ≠ 2L).",
              "- `closure.py` (partner test of the modal tapes of Stages B, C and E over L = 3–100): lengths with no vendored machine.",
              "- `stage_g.py` on Stages H, K, L and M: the same partner test as Stages G and I above.",
              "- `population_classes.py` on Stages G, K and L, and its non-final snapshots of M; `population_sample.py` on Stages G, K, L and M: "
              "the same code as Stages I and M above, left out to keep the GPU load modest.",
              "- `pe_matrix.py`, `r4_*.py`, `conv_soups.py`, `dial_soups.py` (randreg/randsp conventions and lethality dials): no machine "
              "vendored; their executor (`conv.py`) refuses a selected machine.",
              "- `lod_*.py`, `offset_*.py`, `n2_*.py`, `invasion*.py`, `payload_toxin.py` and the other analyses that call the executor: "
              "no `--machine` flag; they route through the interface when a machine is selected in-process, untested here.", ""]
    lines += ["## Machines", "", "The machines path (b) ran (`machines/PROVENANCE_assays.md` has every SHA-256):", ""]
    used = sorted({k for jid in order for k in state[jid]["runs"]["b"]["usage"]["machine"]})
    for name in used:
        _, man = zmachine.machine_paths(name)
        m = json.loads(man.read_text())
        lines.append(f"- `{name}`: {m['core']['id']} {m['core']['coreVersion']} compat {m['core']['compat']}, hooks {', '.join(m['core']['hooks'])}; "
                     f"WGSL SHA-256 `{m['sha256']['wgsl']}`")
    (OUT / "REGRESSION.md").write_text("\n".join(lines) + "\n")
    rows = []
    for jid in order:
        for f in state[jid]["files"]:
            rows.append({"job": jid, "family": state[jid]["family"], "file": f["file"], "sha256_a": f["sha_a"], "sha256_b": f["sha_b"],
                         "a_vs_b": f["a_vs_b"]["status"], "a_vs_b_detail": f["a_vs_b"]["detail"],
                         "committed": f.get("committed"), "a_vs_committed": (f.get("a_vs_committed") or {}).get("status"),
                         "a_vs_committed_detail": (f.get("a_vs_committed") or {}).get("detail"),
                         "a_vs_n": (f.get("a_vs_n") or {}).get("status"), "a_vs_n_detail": (f.get("a_vs_n") or {}).get("detail")})
    pd.DataFrame(rows).to_csv(OUT / "regression.csv", index=False)
    print("\n".join(lines[: 12 + len(order) + 4]))


# ══ smoke ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
def cmd_smoke(a) -> None:
    """Every generalised assay function on the 6502 machine, uniform random tapes only. Records plumbing facts (calls made,
    pairs executed and on which backend, output shapes and types, invariants of the plumbing); never scores."""
    import inspect

    from algocell_exp import assay as A
    from algocell_exp import executor as E
    from algocell_exp import exectrace as X
    from algocell_exp import optrace as O
    from algocell_exp import zmachine
    import branching_transfer as BT
    import closure
    import inflow_insoup
    import mutscan
    import population_classes as PC
    import population_sample as PS
    import serial_retention as SR

    OUT.mkdir(parents=True, exist_ok=True)
    L = 16
    rng = np.random.default_rng(20261010)
    checks = []

    def shape(x):
        if isinstance(x, np.ndarray):
            return f"{x.dtype}{list(x.shape)}"
        if isinstance(x, pd.DataFrame):
            return f"DataFrame[{len(x)} × {len(x.columns)}]"
        if isinstance(x, dict):
            return f"dict({len(x)} keys)"
        if isinstance(x, (list, tuple)):
            return f"{type(x).__name__}[{len(x)}]"
        return type(x).__name__

    def record(name, fn, outputs, before, note=""):
        u = E.usage()
        n_sh = sum(u["shader"].values()) - before[0]
        n_m = sum(u["machine"].get(k, 0) for k in u["machine"] if k.startswith(M6502)) - before[1]
        checks.append({"assay": name, "function": fn, "outputs": outputs, "pairs_machine": n_m, "pairs_shader": n_sh, "note": note})
        log(f"{name}: {fn} -> {outputs}; 6502 pairs {n_m}, shader pairs {n_sh}")

    def counts():
        u = E.usage()
        return sum(u["shader"].values()), sum(u["machine"].get(k, 0) for k in u["machine"] if k.startswith(M6502))

    ex = E.ZilionExecutor(M6502)
    m = ex.machine_for(L, maps=True)
    tapes = rng.integers(0, 256, size=(32, L), dtype=np.uint8)
    partners = rng.integers(0, 256, size=(64, L), dtype=np.uint8)
    pairs = np.concatenate([np.repeat(tapes, 4, 0), np.tile(partners[:4], (32, 1))], 1)
    with E.using(ex):
        # the interface, directly
        b = counts()
        r = ex.run(pairs, L, 128, fetch_map=True, start_map=True)
        direct = m.run(pairs, budget=128)
        plain = ex.run(pairs, L, 128)
        invariants = {
            "façade memory = direct ZMachine.run memory": bool((r.memory == direct.memory).all()),
            "memory with maps = memory without maps": bool((r.memory == plain.memory).all()),
            "fetch map = direct fetch map": bool((r.fetch_map == direct.fetch_map).all()),
            "the machine executed instructions (total steps > 0)": bool(r.steps.sum() > 0),
            "some lane's memory changed": bool((r.memory != pairs).any()),
            "every lane's start map is inside its fetch map": bool((~r.start_map | r.fetch_map).all()),
            "position 0 (the entry) is fetched in every lane": bool(r.fetch_map[:, 0].all()),
        }
        record("interface", "executor.ZilionExecutor.run", f"memory {shape(r.memory)}, writes {shape(r.writes)}, fetch {shape(r.fetch_map)}, start {shape(r.start_map)}, steps {shape(r.steps)}, halt {shape(r.halt_reason)}", b)
        b = counts()
        mem = A.execute_pairs(pairs, L, 128)
        res_t, masks = X.execute_pairs_traced(pairs, L, 128)
        res_o, fo, so = O.execute_pairs_optrace(pairs, L, 128)
        invariants["assay.execute_pairs = interface memory"] = bool((mem == r.memory).all())
        invariants["traced and op-traced memory = interface memory"] = bool((res_t == r.memory).all() and (res_o == r.memory).all())
        invariants["traced mask (8 × u32) unpacks to the fetch map"] = bool((X.exec_positions(masks, 2 * L) == r.fetch_map).all())
        invariants["op-traced maps = interface maps"] = bool((fo == r.fetch_map).all() and (so == r.start_map).all())
        record("façade", "assay.execute_pairs, exectrace.execute_pairs_traced, optrace.execute_pairs_optrace", f"{shape(mem)}, ({shape(res_t)}, {shape(masks)}), ({shape(res_o)}, {shape(fo)}, {shape(so)})", b)
        # culture test
        b = counts()
        out = A.assay(tapes[0], z80_steps=128, n=64, seed=1)
        record("culture test", "assay.assay", f"{shape(out)}: {', '.join(sorted(out))}", b)
        b = counts()
        outs = A.assay_many(tapes, z80_steps=128, n=32, seed=1)
        record("culture test", "assay.assay_many", f"list[{len(outs)}] of dict({len(outs[0])} keys)", b)
        b = counts()
        outs2 = A.assay_many(tapes[:8], z80_steps=128, n=32, seed=1, neighbors=partners)
        record("culture test (in situ)", "assay.assay_many(neighbors=…)", f"list[{len(outs2)}]", b, "partners drawn from a random-tape 'soup'")
        # partner test
        b = counts()
        pt = closure.partner_test(tapes[1], np.random.default_rng(1), n=256)
        record("partner test", "closure.partner_test", f"tuple[{len(pt)}] of float", b)
        # traced execution: confinement and inflow
        b = counts()
        enc = inflow_insoup.encounter(tapes[2], partners)
        record("traced execution (inflow, entered)", "inflow_insoup.encounter", f"{shape(enc)}: {', '.join(sorted(enc))}", b)
        b = counts()
        conf = PS.confined(tapes[:8], L, [], False, np.random.default_rng(2))
        record("traced execution (confinement)", "population_sample.confined", shape(conf), b)
        # single-mutant scan
        b = counts()
        d = mutscan.scan_tape(tapes[3], 16, 16, 128, seed=3)
        row, site = mutscan.summarise(d, L)
        record("single-mutant scan", "mutscan.scan_tape + summarise", f"{len(d['pos'])} mutants; tape row dict({len(row)} keys), sites {shape(site)}", b)
        # serial transfer (the frozen R_MUT, R_CTRL and G; a random genotype)
        b = counts()
        gt = {"key": "random16", "kind": "random tape (smoke)", "tape": tapes[4], "sup": (), "zh": False}
        sdf, smeta = SR.run_genotype(gt, seed=4)
        ssites = SR.sites(sdf, smeta)
        record("serial transfer", "serial_retention.run_genotype + sites", f"mutants table {shape(sdf)}, sites {shape(ssites)}, meta dict({len(smeta)} keys)", b)
        # branching transfer (reduced parameters: 2 values per position, 2 populations, 16 controls, 2 generations)
        b = counts()
        executed, entered = BT.executed_positions(gt, BT.SEED)
        prm = {"values": 2, "reps": 2, "ctrl": 16, "k": 2, "cap": 8, "gens": 2}
        bm, bs, bsum = BT.run_arm(gt, "U", prm, None, executed)
        record("branching transfer", "branching_transfer.executed_positions + run_arm", f"mutants {shape(bm)}, sites {shape(bs)}, summary list[{len(bsum)}]", b,
               "reduced parameters (2 values, 2 populations, 16 controls, cap 8, 2 generations)")
        # population classes (a 'soup' of uniform random tapes)
        b = counts()
        soup = rng.integers(0, 256, size=(512, L), dtype=np.uint8)
        snap, cells = PC.classify(soup, L, [], False, np.random.default_rng(5))
        record("population classes", "population_classes.classify", f"snapshot dict({len(snap)} keys), cells {type(cells).__name__}", b,
               "soup of 512 uniform random tapes; its single-mutant scan of heritable cells is `mutscan.scan_tape`, exercised above")
        # refusals
        refusals = {}
        try:
            A.execute_pairs(pairs, L, 128, suppress=["family:block-copy"])
            refusals["Z80 suppression on the 6502"] = "NOT refused"
        except ValueError as e:
            refusals["Z80 suppression on the 6502"] = f"refused: {str(e)[:110]}…"
        try:
            A.execute_pairs(pairs, L, 128, zero_halts=True)
            refusals["lethal rule (no 6502 lethal machine)"] = "NOT refused"
        except ValueError as e:
            refusals["lethal rule (no 6502 lethal machine)"] = f"refused: {str(e)[:110]}…"
        try:
            A._pipeline(L)
            refusals["Algocell shader while the 6502 is selected"] = "NOT refused"
        except RuntimeError as e:
            refusals["Algocell shader while the 6502 is selected"] = f"refused: {str(e)[:110]}…"
    u = E.usage()
    man = m.m
    lines = ["# The generalised assays on the 6502 machine: plumbing smoke run (generated by `assays_any_isa.py smoke`; do not edit)", "",
             "**Plumbing only.** Uniform random tapes and partners; no evolved 6502 soup was used. The assay outputs were checked for "
             "type and shape only: no score, rate or class count of a 6502 program is computed into this report or printed, because "
             "the 6502 test is not yet registered (`planning/PREREG_6502_DRAFT.md`).", "",
             f"Machine: `{M6502}_L16` — {man['core']['name']} {man['core']['coreVersion']} ({man['core']['status']}; variant "
             f"{man['core']['variant']}; compat {man['core']['compat']}; hooks {', '.join(man['core']['hooks'])}), zilion {man['zilion']}; "
             f"ring {m.ring_bytes}; init {json.dumps(man['world']['init'])}; budget {json.dumps(man['world']['budget'])}; "
             f"ablation {man['ablation']['kind']}; WGSL SHA-256 `{man['sha256']['wgsl']}` (`machines/PROVENANCE_assays.md`).", "",
             "Every call below ran with the 6502 machine selected (`executor.using`), through the same functions the frozen Z80 "
             "scripts call; frozen parameters unless noted.", "",
             "Pair counts are not shown where they depend on what the programs do (the transfer and population assays run more "
             "encounters when copies are made); the table says only whether the 6502 machine ran and whether a shader did.", "",
             "| assay | function(s) | outputs (type and shape) | ran on the 6502 machine | pairs on Algocell's shaders | note |",
             "|---|---|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['assay']} | `{c['function']}` | {c['outputs']} | {'yes' if c['pairs_machine'] > 0 else 'NO'} | {c['pairs_shader']:,} | {c['note']} |")
    lines += ["", "Plumbing invariants (each must hold):", ""]
    lines += [f"- {k}: **{'yes' if v else 'NO'}**" for k, v in invariants.items()]
    lines += ["", "Refusals (each must refuse):", ""]
    lines += [f"- {k}: {v}" for k, v in refusals.items()]
    lines += ["", f"Backends used: machines {', '.join(f'`{k}`' for k in u['machine'])}; Algocell's shaders: "
              f"{sum(u['shader'].values()):,} pairs.", ""]
    ok = (all(invariants.values()) and all(v.startswith("refused") for v in refusals.values()) and sum(u["shader"].values()) == 0
          and all(c["pairs_machine"] > 0 for c in checks) and set(u["machine"]) == {f"{M6502}_L16"})
    lines.append(f"Result: **{'plumbing OK' if ok else 'PLUMBING FAILURE'}**.")
    (OUT / "SMOKE_6502.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not ok:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--n", type=int, default=10000)
    p = sub.add_parser("regress")
    p.add_argument("--only", default="")
    p.add_argument("--work", default=str(HERE / "runs" / "assays_any_isa"))
    p = sub.add_parser("report")
    p.add_argument("--work", default=str(HERE / "runs" / "assays_any_isa"))
    sub.add_parser("smoke")
    a = ap.parse_args()
    if a.cmd == "pairs":
        cmd_pairs(a)
    elif a.cmd == "regress":
        cmd_regress(a)
    elif a.cmd == "report":
        recompare(Path(a.work).resolve())
    else:
        cmd_smoke(a)


if __name__ == "__main__":
    main()
