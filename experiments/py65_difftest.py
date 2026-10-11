"""128-instruction NMOS ring difftest: unmodified py65 versus vendored Zilion.

Run from this worktree, using the shared environment:
  /Users/neo/repos/algocell/experiments/.venv/bin/python experiments/py65_difftest.py
GPU submissions are limited to --batch lanes; there is no Modal dependency.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import inspect
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from importlib.metadata import distribution, version
from pathlib import Path

import numpy as np
from py65.devices.mpu6502 import MPU

from algocell_exp.py65_ring import FIELDS, IMPLEMENTED, JAM, OracleResult, run_oracle
from algocell_exp.soup import adapter_summary
from algocell_exp.zmachine import load_machine

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "py65_difftest"
COUNTS = ("total", "compared", "matched", "out_of_scope", "explained", "unexplained")


def gpu_state(result, i):
    return {key: int(result.reg(key)[i]) for key in FIELDS}


def differences(oracle: OracleResult, result, i: int) -> dict:
    regs = {key: {"py65": oracle.state[key], "zilion": int(result.reg(key)[i])}
            for key in FIELDS if oracle.state[key] != int(result.reg(key)[i])}
    memory = [{"position": j, "py65": x, "zilion": int(y)}
              for j, (x, y) in enumerate(zip(oracle.memory, result.memory[i])) if x != y]
    halt = {}
    if oracle.steps != int(result.steps[i]):
        halt["steps"] = [oracle.steps, int(result.steps[i])]
    expected = (3, 1) if oracle.halted else (0, 0)
    if expected != (int(result.halt_reason[i]), int(result.trap_cause[i])):
        halt["reason_cause"] = [list(expected), [int(result.halt_reason[i]), int(result.trap_cause[i])]]
    return {key: value for key, value in (("registers", regs), ("memory", memory), ("halt", halt)) if value}


def corpus_inputs(args):
    # Separate RNG streams make corpus inputs independent of corpus order/batch.
    rng = np.random.Generator(np.random.PCG64(args.seed + 100))
    with (HERE / "results/s6502/s1_worlds.csv").open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 80:
        raise ValueError(f"expected all 80 S1 worlds, found {len(rows)}")
    for row in rows:
        tape = bytes.fromhex(row["first_tape"])
        if not tape:
            continue
        length = int(row["L"])
        if len(tape) != length or length not in (16, 32):
            raise ValueError(f"invalid first_tape in {row['world']}")
        partners = rng.integers(0, 256, (args.s1_partners, length), dtype=np.uint8)
        for i, partner in enumerate(partners):
            yield "s1", row["world"], f"{row['world']}/partner-{i}", tape + partner.tobytes()

    if not args.skip_census:
        rng = np.random.Generator(np.random.PCG64(args.seed + 200))
        with (HERE / "results/census6502/table_hit_words.csv").open() as f:
            rows = list(csv.DictReader(f))
        seen = set()
        for row in rows:
            length, word = int(row["L"]), bytes.fromhex(row["word"])
            group = f"L{length}/{word.hex()}"
            if group in seen:
                continue
            seen.add(group)
            tape = (word * ((length + len(word) - 1) // len(word)))[:length]
            partners = rng.integers(0, 256, (args.census_partners, length), dtype=np.uint8)
            # Include zero partner as well as independently sampled random partners.
            yield "census", group, f"{group}/zero", tape + bytes(length)
            for i, partner in enumerate(partners):
                yield "census", group, f"{group}/partner-{i}", tape + partner.tobytes()

    for ring in (32, 64):
        for seed in args.random_seeds:
            rng = np.random.Generator(np.random.PCG64(seed))
            for offset in range(0, args.random, args.batch):
                n = min(args.batch, args.random - offset)
                tapes = rng.integers(0, 256, (n, ring), dtype=np.uint8)
                for i, tape in enumerate(tapes):
                    yield f"random_P{ring}", f"seed-{seed}", f"seed-{seed}/tape-{offset + i}", tape.tobytes()


def coverage():
    source = Path(inspect.getfile(MPU))
    d = distribution("py65")
    license_file = next(d.locate_file(f) for f in d.files if str(f).endswith("LICENSE.txt"))
    return {
        "py65_version": version("py65"),
        "py65_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "license": "BSD-3-Clause", "license_text": license_file.read_text(),
        "implemented": sorted(IMPLEMENTED), "jam_handled_by_harness": sorted(JAM),
        "unsupported_non_jam": sorted(set(range(256)) - IMPLEMENTED - JAM),
        "detection": "MPU.instruct[opcode] != MPU.inst_not_implemented; tested at every executed instruction start",
    }


class Difftest:
    def __init__(self, args):
        self.args = args
        self.counts = defaultdict(Counter)
        self.groups = defaultdict(Counter)
        self.depth = defaultdict(Counter)
        self.scopes = defaultdict(Counter)
        self.reasons = defaultdict(Counter)
        self.scope_reasons = defaultdict(Counter)
        self.unsupported = defaultdict(Counter)
        self.examples = {}
        self.unexplained = []
        self.manifests = {}
        self.gpu_times = []
        self.started = datetime.now().astimezone().isoformat()
        self.stopped_early = False
        self.failure = None
        self.output = Path(args.out)
        self.output.mkdir(parents=True, exist_ok=True)
        with (HERE / "results/s6502/s1_worlds.csv").open() as f:
            worlds = list(csv.DictReader(f))
        self.s1_missing = [{"world": row["world"], "L": int(row["L"]), "reason": "empty first_tape in source CSV"}
                           for row in worlds if not row["first_tape"].strip()]
        (self.output / "s1_missing.json").write_text(json.dumps(self.s1_missing, indent=2) + "\n")
        self.csv_file = gzip.open(self.output / "trials.csv.gz", "wt", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=[
            "corpus", "group", "id", "P", "input_hex", "status", "reasons", "oracle_steps",
            "gpu_steps", "gpu_halt_reason", "gpu_trap_cause", "unsupported", "difference",
        ])
        self.writer.writeheader()

    def machine(self, ring):
        machine = load_machine(f"m6502_pair_maps_L{ring // 2}")
        self.manifests[ring] = machine.describe()
        convention = machine.m["init"]["stack"]["convention"]
        assert (0x100 | convention["values"]["s"]) % ring == ring - 1
        return machine

    def gpu(self, machine, tapes, budgets):
        start = time.monotonic()
        result = machine.run(np.array([list(t) for t in tapes], dtype=np.uint8), budget=budgets,
                             chunk=self.args.batch)
        elapsed = time.monotonic() - start
        self.gpu_times.append({"P": machine.ring_bytes, "lanes": len(tapes), "seconds": elapsed})
        return result

    def process(self, entries):
        ring = len(entries[0][3])
        machine = self.machine(ring)
        raw = [run_oracle(entry[3], self.args.budget) for entry in entries]
        gpu = self.gpu(machine, [entry[3] for entry in entries], self.args.budget)
        for i, (entry, oracle) in enumerate(zip(entries, raw)):
            corpus, group, identifier, tape = entry
            diff, reasons = {}, []
            unsupported = oracle.unsupported
            if unsupported:
                status = "out_of_scope"
            else:
                diff = differences(oracle, gpu, i)
                if not diff:
                    status = "matched"
                else:
                    diagnostic = run_oracle(tape, self.args.budget, diagnostic=True)
                    if not diagnostic.unsupported and not differences(diagnostic, gpu, i):
                        status = "explained"
                        reasons = sorted({e["kind"] for e in diagnostic.events})
                        assert reasons
                    elif diagnostic.unsupported:
                        # Only exclude this alternate path after a GPU prefix
                        # proves it actually executes the unsupported opcode.
                        n = diagnostic.steps
                        prefix = self.gpu(machine, [tape], n)
                        if not differences(diagnostic, prefix, 0):
                            status = "out_of_scope"
                            unsupported = {**diagnostic.unsupported, "stream": "gpu-verified-diagnostic-prefix"}
                            reasons = sorted({e["kind"] for e in diagnostic.events})
                        else:
                            status = "unexplained"
                    else:
                        status = "unexplained"
                    if status in ("explained", "unexplained"):
                        category = "+".join(reasons) if reasons else "UNEXPLAINED"
                        key = f"P{ring}/{category}"
                        example = {
                            "corpus": corpus, "id": identifier, "P": ring, "input_hex": tape.hex(),
                            "status": status, "reasons": reasons, "raw_difference": diff,
                            "raw_state": oracle.state, "zilion_state": gpu_state(gpu, i),
                            "diagnostic_state": diagnostic.state, "events": diagnostic.events,
                            "diagnostic_final_matches": not diagnostic.unsupported and not differences(diagnostic, gpu, i),
                        }
                        if key not in self.examples:
                            self.examples[key] = example
                        if status == "unexplained":
                            self.unexplained.append(example)
            count = self.counts[corpus]
            subgroup = self.groups[(corpus, group, ring)]
            for counter in (count, subgroup):
                counter["total"] += 1
                counter[status] += 1
                if status != "out_of_scope":
                    counter["compared"] += 1
            if status != "out_of_scope":
                depth = self.depth[corpus]
                depth["one_instruction" if int(gpu.steps[i]) == 1 else "multiple_instructions"] += 1
                depth["ran_to_budget" if int(gpu.halt_reason[i]) == 0 else "jam_halted"] += 1
            for reason in reasons:
                if status == "explained":
                    self.reasons[corpus][reason] += 1
                elif status == "out_of_scope":
                    self.scope_reasons[corpus][reason] += 1
            if unsupported:
                self.unsupported[corpus][unsupported["opcode"]] += 1
                self.scopes[corpus][unsupported.get("stream", "raw-py65")] += 1
            self.writer.writerow({
                "corpus": corpus, "group": group, "id": identifier, "P": ring,
                "input_hex": tape.hex() if not corpus.startswith("random_") or status in ("explained", "unexplained") else "",
                "status": status, "reasons": ";".join(reasons),
                "oracle_steps": oracle.steps, "gpu_steps": int(gpu.steps[i]),
                "gpu_halt_reason": int(gpu.halt_reason[i]), "gpu_trap_cause": int(gpu.trap_cause[i]),
                "unsupported": json.dumps(unsupported, separators=(",", ":")) if unsupported else "",
                "difference": json.dumps(diff, separators=(",", ":")) if diff and status != "out_of_scope" else "",
            })
        self.csv_file.flush()

    def first_divergence(self, example, diagnostic):
        tape = bytes.fromhex(example["input_hex"])
        machine = self.machine(len(tape))
        oracle = run_oracle(tape, self.args.budget, diagnostic=diagnostic, trace=True)
        if not oracle.trace:
            return None
        result = self.gpu(machine, [tape] * len(oracle.trace), np.arange(1, len(oracle.trace) + 1))
        for i, trace in enumerate(oracle.trace):
            snapshot = OracleResult(bytes.fromhex(trace["memory"]), trace["after"], i + 1,
                                    trace["opcode"] in JAM)
            diff = differences(snapshot, result, i)
            if diff:
                before = tape if i == 0 else bytes.fromhex(oracle.trace[i - 1]["memory"])
                return {"step": i + 1, "opcode": trace["opcode"], "before_state": trace["before"],
                        "before_memory_hex": before.hex(), "oracle_after": trace["after"],
                        "zilion_after": gpu_state(result, i), "difference": diff}
        return None

    def finish(self):
        self.csv_file.close()
        # Witnesses identify the first raw divergence, not just the final delta.
        for example in self.examples.values():
            example["first_raw_divergence"] = self.first_divergence(example, False)
        for i, example in enumerate(self.unexplained):
            # One instruction with explicit pre-state and memory is a minimal
            # step-count reproducer; it can be replayed through ZMachine.resume.
            example["minimal_reproducer"] = self.first_divergence(example, True)
            (self.output / f"unexplained_{i:04d}.json").write_text(json.dumps(example, indent=2) + "\n")
        cov = coverage()
        (self.output / "coverage.json").write_text(json.dumps(cov, indent=2) + "\n")
        (self.output / "examples.json").write_text(json.dumps(self.examples, indent=2) + "\n")
        with (self.output / "opcode_coverage.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["opcode", "mnemonic", "mode", "oracle_scope"])
            for opcode in range(256):
                mnemonic, mode = MPU.disassemble[opcode]
                w.writerow([f"{opcode:02x}", mnemonic, mode,
                            "py65" if opcode in IMPLEMENTED else "jam-halt" if opcode in JAM else "unsupported"])
        with (self.output / "counts.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["corpus", *COUNTS])
            for corpus, count in self.counts.items():
                w.writerow([corpus, *(count[k] for k in COUNTS)])
        with (self.output / "groups.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["corpus", "group", "P", *COUNTS])
            for (corpus, group, ring), count in sorted(self.groups.items()):
                w.writerow([corpus, group, ring, *(count[k] for k in COUNTS)])
        with (self.output / "unsupported.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["corpus", "opcode", "count"])
            for corpus, ops in self.unsupported.items():
                for opcode, n in sorted(ops.items()):
                    w.writerow([corpus, f"{opcode:02x}", n])
        with (self.output / "explained.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["corpus", "reason", "count"])
            for corpus, reasons in self.reasons.items():
                for reason, n in sorted(reasons.items()):
                    w.writerow([corpus, reason, n])
        with (self.output / "execution_depth.csv").open("w", newline="") as f:
            w = csv.writer(f)
            columns = ("one_instruction", "multiple_instructions", "ran_to_budget", "jam_halted")
            w.writerow(["corpus", *columns])
            for corpus, depth in self.depth.items():
                w.writerow([corpus, *(depth[k] for k in columns)])
        with (self.output / "scope_counts.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["corpus", "stream", "count"])
            for corpus, scopes in self.scopes.items():
                for stream, n in scopes.items():
                    w.writerow([corpus, stream, n])
        summary = {
            "started": self.started, "finished": datetime.now().astimezone().isoformat(),
            "stopped_early": self.stopped_early, "failure": self.failure, "parameters": vars(self.args),
            "s1_worlds_in_source": 80, "s1_available_first_tapes": 80 - len(self.s1_missing),
            "s1_missing": self.s1_missing,
            "counts": self.counts, "reasons": self.reasons, "scope_transition_reasons": self.scope_reasons,
            "execution_depth": self.depth, "scope_counts": self.scopes,
            "machines": self.manifests,
            "gpu": adapter_summary(), "gpu_calls": self.gpu_times,
            "input_source_sha256": {str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (HERE / "results/s6502/s1_worlds.csv",
                                                 HERE / "results/census6502/table_hit_words.csv")},
            "oracle": {k: v for k, v in cov.items() if k != "license_text"},
            "python": sys.version, "numpy": np.__version__,
            "harness_sha256": {str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (HERE / "py65_difftest.py", HERE / "algocell_exp/py65_ring.py")},
        }
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        self.report(summary)

    def report(self, summary):
        totals = Counter()
        for counts in self.counts.values():
            totals.update(counts)
        lines = [f"Compared: {totals['compared']}; matched: {totals['matched']}; out of scope: {totals['out_of_scope']}; "
                 f"explained: {totals['explained']}; UNEXPLAINED: {totals['unexplained']}.",
                 "", "# Independent py65 NMOS ring difftest", "",
                 "| Corpus | Total | Compared | Matched | Out of scope | Explained | UNEXPLAINED |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for corpus, counts in self.counts.items():
            lines.append(f"| {corpus} | " + " | ".join(str(counts[k]) for k in COUNTS) + " |")
        lines += ["", "Within compared cases, these are the actual GPU execution depths:", "",
                  "| Corpus | One instruction | Multiple instructions | Ran to budget | JAM halted |",
                  "|---|---:|---:|---:|---:|---:|"]
        for corpus, depth in self.depth.items():
            lines.append(f"| {corpus} | " + " | ".join(str(depth[k]) for k in
                         ("one_instruction", "multiple_instructions", "ran_to_budget", "jam_halted")) + " |")
        lines += ["", "Compared = matched + explained + unexplained; total = compared + out of scope.",
                  "Matched means exact equality against unmodified py65, including all memory bytes and A/X/Y/S/P/PC.",
                  "Steps, halt reason and trap cause are checked as well. Explained cases are raw mismatches whose",
                  "entire final output agrees after only the documented diagnostic shims. They are never counted as raw matches.",
                  "", "## Oracle and scope", "",
                  f"Installed py65 **{version('py65')}**, BSD-3-Clause, in `/Users/neo/repos/algocell/experiments/.venv`.",
                  "The environment initially lacked pip; ensurepip and reinstalling pip created `.venv/bin/pip`, then",
                  "`.venv/bin/pip install py65` installed the oracle. `requirements-py65.txt` pins the version.",
                  "", "The installed opcode table implements **151 documented opcodes, zero NMOS undocumented opcodes**.",
                  "The harness handles the 12 JAM bytes as halts, with PC parked on the opcode and the halt instruction counted.",
                  "The other **93** opcode bytes are outside oracle scope. Detection uses the installed handler table at each",
                  "instruction start, including self-modified code; unsupported bytes in operands or unexecuted memory do not exclude a trial.",
                  "The unsupported stub is never executed. `opcode_coverage.csv` lists all 256 entries; `unsupported.csv` counts actual exclusions.",
                  "", "Every memory access uses `(address & 0xffff) % P`, so address width precedes ring mapping.",
                  "A occupies positions 0..L-1, B positions L..2L-1, P=2L. Reset A=X=Y=0, status=0x20, PC=0;",
                  "S is the largest byte for which `(0x100 | S) % P == 2L-1`, checked against each vendored manifest.",
                  "S=0xff for P=32/64; any included P=40 uses S=0xdf. The seam offset is `65536 % P` (0 for 32/64, 16 for 40).",
                  f"Each trial runs up to **{self.args.budget} instructions**. BRK vectors; it does not halt.",
                  "", "## Inputs and reproducibility", "",
                  f"Uniform random bytes: NumPy PCG64 seeds {self.args.random_seeds}, {self.args.random:,} pairs per seed and ring size.",
                  f"All 80 rows of `s6502/s1_worlds.csv` were read: **{80 - len(self.s1_missing)} nonempty first_tapes**, each with {self.args.s1_partners} random partners.",
                  f"**{len(self.s1_missing)} worlds have empty first_tape and cannot supply a replicator**; listed in `s1_missing.json`.",
                  f"PCG64 seed {self.args.seed + 100}, preserving CSV row order, even when tapes repeat across worlds.",
                  f"Census: unique (L, word) rows from `census6502/table_hit_words.csv`, tiled/truncated to L,",
                  f"each with zero plus {self.args.census_partners} random partners, PCG64 seed {self.args.seed + 200}.",
                  "Only the faithful/JAM-halting machine is used, including words originally found by BRK-halting or 512-step census arms.",
                  "", "`trials.csv.gz` records every outcome, unsupported-stream witness and raw final delta.",
                  "S1/census inputs and all mismatching random inputs are included; other random inputs regenerate from the recorded seed and index.",
                  "`groups.csv` gives every S1 world / census word / random seed separately. Input source hashes, machine",
                  "core/world/WGSL hashes, installed oracle source hash, full BSD notice, GPU and timing are in the JSON records.",
                  "", "Command (from this worktree):", "", "```sh",
                  "/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/py65_difftest.py "
                  f"--random {self.args.random} --random-seeds {' '.join(map(str, self.args.random_seeds))} "
                  f"--s1-partners {self.args.s1_partners} --census-partners {self.args.census_partners} "
                  f"--seed {self.args.seed} --batch {self.args.batch} --stop-at {self.args.stop_at}", "```", "",
                  "## Explained differences and evidence", "",
                  "* **Status B bit:** py65 BRK sets stored B; PLP and RTI force stored B=1. Zilion stores B=0",
                  "  after PLP/RTI and only pushes B=1 on PHP/BRK. The diagnostic clears B after BRK/PLP/RTI;",
                  "  pushed status bytes remain identical. This can be a pure final P difference.",
                  "* **JSR bus order:** py65 pushes PC+2 before `WordAt(pc+1)`, reading both target bytes after writes.",
                  "  NMOS reads target low first, then pushes, then reads target high. A ring can alias the low operand",
                  "  with the stack. Diagnostic evidence records the low byte before/after pushes; the high byte is still read after writes.",
                  "No decimal-mode differences were observed. **All arithmetic, including decimal ADC/SBC, remains",
                  "unmodified py65 in both the raw and diagnostic runs.** No decimal-mode exception is applied.",
                  "", "Only observed, fully reconciled cases enter `explained`; definitions above are not blanket waivers.",
                  "`explained.csv` counts reasons only within explained comparisons; multiple reasons can apply to one trial.",
                  "`examples.json` contains a representative for each observed reason combination and ring size,",
                  "including the first raw divergent instruction (GPU prefix replay), pre-state, memory and final raw delta.",
                  "If a diagnostic path reaches an unsupported opcode, it is excluded only after GPU prefix equality proves",
                  "the GPU is about to execute that same unsupported byte; this is recorded as `gpu-verified-diagnostic-prefix`.",
                  "", "Primary sources: [py65 1.2.0 source](https://github.com/mnaberez/py65/blob/1.2.0/py65/devices/mpu6502.py),",
                  "[BSD license](https://github.com/mnaberez/py65/blob/main/LICENSE.txt).",
                  "Ring/stack/bus-order contract: `/Users/neo/repos/zilion/docs/ARCHITECTURE.md` and",
                  "`src/cores/m6502/index.ts`; actual machines are the Algocell vendored manifest-verified WGSL.",
                  "", "## Unexplained cases", ""]
        if not totals["unexplained"]:
            lines.append("**None in the compared scope.** This does not validate any of the 93 unsupported non-JAM opcodes.")
        else:
            lines.append(f"**{totals['unexplained']} UNEXPLAINED.** See `unexplained_*.json` for one-instruction reproducers with explicit pre-state and ring memory.")
        lines += ["", "## Execution and STATUS", "",
                  f"GPU: {summary['gpu']}. Submissions use at most {self.args.batch} lanes;",
                  f"maximum measured host GPU run call {max((t['seconds'] for t in self.gpu_times), default=0):.3f} s, "
                  f"total measured GPU run-call time {sum(t['seconds'] for t in self.gpu_times):.3f} s.",
                  "Host run-call timing includes buffer allocation, dispatch and readback; shader compilation is separate.",
                  f"Started {self.started}; finished {summary['finished']}.",
                  "STATUS: " + (f"Aborted: {self.failure}; only completed batches are counted." if self.failure else
                                "Stopped at the configured time limit; only completed batches are counted." if self.stopped_early
                                else "All requested batches completed; no worker process remains after this command exits."),
                  "Undocumented-opcode comparison and exhaustive program coverage were not performed.",
                  "No Modal jobs, server processes, pushes, or changes to the vendored emulator.", ""]
        (self.output / "DIFFTEST.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random", type=int, default=100000)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[650201, 650202])
    parser.add_argument("--s1-partners", type=int, default=64)
    parser.add_argument("--census-partners", type=int, default=16)
    parser.add_argument("--seed", type=int, default=650200)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--skip-census", action="store_true")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--stop-at", default="10:39", help="local HH:MM; stops between short batches, reserving report time")
    args = parser.parse_args()
    if args.batch < 1 or args.budget < 1:
        parser.error("batch and budget must be positive")
    test = Difftest(args)
    entries = []
    current = None
    try:
        for entry in corpus_inputs(args):
            key = (entry[0], len(entry[3]))
            if entries and (len(entries) >= args.batch or key != current):
                test.process(entries)
                print(json.dumps({"time": datetime.now().strftime("%H:%M:%S"), "counts": test.counts}), flush=True)
                entries = []
                if datetime.now().strftime("%H:%M") >= args.stop_at:
                    test.stopped_early = True
                    break
            entries.append(entry)
            current = key
        if entries and not test.stopped_early:
            test.process(entries)
    except BaseException as error:
        test.failure = repr(error)
        raise
    finally:
        test.finish()
    print(json.dumps({"finished": datetime.now().isoformat(), "counts": test.counts}), flush=True)
    return int(bool(test.unexplained))


if __name__ == "__main__":
    sys.exit(main())
