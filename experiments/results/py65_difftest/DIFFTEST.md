Compared: 241797; matched: 233265; out of scope: 1767271; explained: 8532; UNEXPLAINED: 0.

# Independent py65 NMOS ring difftest

| Corpus | Total | Compared | Matched | Out of scope | Explained | UNEXPLAINED |
|---|---:|---:|---:|---:|---:|---:|---:|
| s1 | 3968 | 3072 | 3072 | 896 | 0 | 0 |
| census | 5100 | 841 | 400 | 4259 | 441 | 0 |
| random_P32 | 1000000 | 120607 | 116196 | 879393 | 4411 | 0 |
| random_P64 | 1000000 | 117277 | 113597 | 882723 | 3680 | 0 |

Within compared cases, these are the actual GPU execution depths:

| Corpus | One instruction | Multiple instructions | Ran to budget | JAM halted |
|---|---:|---:|---:|---:|---:|
| s1 | 0 | 3072 | 3072 | 0 |
| census | 0 | 841 | 523 | 318 |
| random_P32 | 47252 | 73355 | 5723 | 114884 |
| random_P64 | 47126 | 70151 | 3100 | 114177 |

Compared = matched + explained + unexplained; total = compared + out of scope.
Matched means exact equality against unmodified py65, including all memory bytes and A/X/Y/S/P/PC.
Steps, halt reason and trap cause are checked as well. Explained cases are raw mismatches whose
entire final output agrees after only the documented diagnostic shims. They are never counted as raw matches.

## Oracle and scope

Installed py65 **1.2.0**, BSD-3-Clause, in `/Users/neo/repos/algocell/experiments/.venv`.
The environment initially lacked pip; ensurepip and reinstalling pip created `.venv/bin/pip`, then
`.venv/bin/pip install py65` installed the oracle. `requirements-py65.txt` pins the version.

The installed opcode table implements **151 documented opcodes, zero NMOS undocumented opcodes**.
The harness handles the 12 JAM bytes as halts, with PC parked on the opcode and the halt instruction counted.
The other **93** opcode bytes are outside oracle scope. Detection uses the installed handler table at each
instruction start, including self-modified code; unsupported bytes in operands or unexecuted memory do not exclude a trial.
The unsupported stub is never executed. `opcode_coverage.csv` lists all 256 entries; `unsupported.csv` counts actual exclusions.

Every memory access uses `(address & 0xffff) % P`, so address width precedes ring mapping.
A occupies positions 0..L-1, B positions L..2L-1, P=2L. Reset A=X=Y=0, status=0x20, PC=0;
S is the largest byte for which `(0x100 | S) % P == 2L-1`, checked against each vendored manifest.
S=0xff for P=32/64; any included P=40 uses S=0xdf. The seam offset is `65536 % P` (0 for 32/64, 16 for 40).
Each trial runs up to **128 instructions**. BRK vectors; it does not halt.

## Inputs and reproducibility

Uniform random bytes: NumPy PCG64 seeds [650201, 650202], 500,000 pairs per seed and ring size.
All 80 rows of `s6502/s1_worlds.csv` were read: **62 nonempty first_tapes**, each with 64 random partners.
**18 worlds have empty first_tape and cannot supply a replicator**; listed in `s1_missing.json`.
PCG64 seed 650300, preserving CSV row order, even when tapes repeat across worlds.
Census: unique (L, word) rows from `census6502/table_hit_words.csv`, tiled/truncated to L,
each with zero plus 16 random partners, PCG64 seed 650400.
Only the faithful/JAM-halting machine is used, including S1 tapes and census words originally found by BRK-halting or 512-step arms.

`trials.csv.gz` records every outcome, unsupported-stream witness and raw final delta.
S1/census inputs and all mismatching random inputs are included; other random inputs regenerate from the recorded seed and index.
`groups.csv` gives every S1 world / census word / random seed separately. Input source hashes, machine
core/world/WGSL hashes, installed oracle source hash, full BSD notice, GPU and timing are in the JSON records.

Command (from this worktree):

```sh
/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/py65_difftest.py --random 500000 --random-seeds 650201 650202 --s1-partners 64 --census-partners 16 --seed 650200 --batch 2048 --stop-at 10:38
```

## Explained differences and evidence

* **Status B bit:** py65 BRK sets stored B; PLP and RTI force stored B=1. Zilion stores B=0
  after PLP/RTI and only pushes B=1 on PHP/BRK. The diagnostic clears B after BRK/PLP/RTI;
  pushed status bytes remain identical. This can be a pure final P difference.
* **JSR bus order:** py65 pushes PC+2 before `WordAt(pc+1)`, reading both target bytes after writes.
  NMOS reads target low first, then pushes, then reads target high. A ring can alias the low operand
  with the stack. Diagnostic evidence records the low byte before/after pushes; the high byte is still read after writes.

No decimal-mode differences were observed. **All arithmetic, including decimal ADC/SBC, remains
unmodified py65 in both the raw and diagnostic runs.** No decimal-mode exception is applied.

Only observed, fully reconciled cases enter `explained`; definitions above are not blanket waivers.
`explained.csv` counts reasons only within explained comparisons; multiple reasons can apply to one trial.
`examples.json` contains a representative for each observed reason combination and ring size,
including the first raw divergent instruction (GPU prefix replay), pre-state, memory and final raw delta.
If a diagnostic path reaches an unsupported opcode, it is excluded only after GPU prefix equality proves
the GPU is about to execute that same unsupported byte; this is recorded as `gpu-verified-diagnostic-prefix`.

Primary sources: [py65 1.2.0 source](https://github.com/mnaberez/py65/blob/1.2.0/py65/devices/mpu6502.py),
[BSD license](https://github.com/mnaberez/py65/blob/main/LICENSE.txt).
Ring/stack/bus-order contract: `/Users/neo/repos/zilion/docs/ARCHITECTURE.md` and
`src/cores/m6502/index.ts`; actual machines are the Algocell vendored manifest-verified WGSL.

## Unexplained cases

**None in the compared scope.** This does not validate any of the 93 unsupported non-JAM opcodes.

## Execution and STATUS

GPU: Apple M4 (Metal, IntegratedGPU). Submissions use at most 2048 lanes;
maximum measured host GPU run call 0.093 s, total measured GPU run-call time 6.793 s.
Host run-call timing includes buffer allocation, dispatch and readback; shader compilation is separate.
Started 2026-10-11T10:25:30.059376-05:00; finished 2026-10-11T10:25:55.907146-05:00.
STATUS: Completed 1,000,000 random pairs at each of P=32/64; all 62 available S1 first tapes against 64 partners each; all 300 census hit words against zero plus 16 random partners. The other 18 S1 worlds have empty first_tape, so those tapes could not be tested. All experiment processes exited.
Undocumented-opcode comparison and exhaustive program coverage were not performed.
No Modal jobs, server processes, pushes, or changes to the vendored emulator.

## Validation

`check_py65_difftest.py` passed at 10:27:27 CDT, reconciling all **2,009,068** ledger rows with the generated corpus,
group, reason, scope and execution-depth tables, checking input/harness/machine hashes, all 62 available S1 worlds
and all 300 census words. The audit record and compressed-ledger checksum are in `validation.json`.

`cd experiments && /Users/neo/repos/algocell/experiments/.venv/bin/python -m pytest -q tests/test_py65_ring.py`:
**44 passed** (0.54 s). Cases cover the nonzero seam, stack aliases, all JAM bytes, unsupported operand versus executed
opcode, self-modification, budget boundary, BRK flags, aliased JSR, documented RMW, zero-page and indirect-JMP wraps,
valid/invalid decimal inputs, comparator negative controls, and one-instruction GPU replays of all six saved
first-divergence witness categories. No unexplained witness required minimization.
