# Algocell's assays on any Zilion machine

The frozen assays (culture test, partner tests, traced execution, single-mutant scan, serial and branching transfer,
population classes) used to call Algocell's Z80 executor shaders directly. They now call one executor interface, which
runs either those shaders (the default, unchanged) or a vendored Zilion machine. The 6502 pre-registration
(`planning/PREREG_6502_DRAFT.md`) and the emulator impact study (`planning/IMPACT_STUDY_DRAFT.md`) need exactly this.
Every measured number below is quoted from a generated report in `experiments/results/assays_any_isa/`.

## The interface (`algocell_exp/executor.py`)

```python
from algocell_exp import executor as E
r = E.get("z80c02_assay").run(pairs, L, steps, suppress, zero_halts=False, fetch_map=True, start_map=True)
r.memory, r.writes, r.fetch_map, r.start_map, r.steps, r.halt_reason, r.trap_cause, r.executor

E.select("z80c02_assay")          # every assay call in this process runs on the machine set
with E.using("m6502_assay"): ...  # or one block
```

- **(a) `AlgocellShader`, the default.** The frozen host functions `assay.execute_pairs`,
  `exectrace.execute_pairs_traced` and `optrace.execute_pairs_optrace`, whose bodies moved unchanged into
  `_shader_execute_pairs*`. Each public function gained a dispatch at the top (to the selected machine, if any) and a
  per-backend pair count; each body gained an optional read-back of the register words it already wrote (write counts
  are words 10 and 11). The shaders record no step count or halt reason, so `steps` and `halt_reason` are None here.
- **(b) `ZilionExecutor`.** A vendored machine through `zmachine.py`. A *machine set* is the machines named
  `<prefix>[_lethal]_L<L>[_P<P>]` (nothing else sharing the prefix), all of one core; `(L, ring P, rule)` resolves
  to one of them. The manifest is checked against the request before
  anything runs:
  - segments A and B of L bytes on a ring of P;
  - the lethal rule `{firstUnit: 0}` exactly when `zero_halts`, and no other halting rule;
  - no raw hooks (dials), no random or per-lane initial state, entry at position 0;
  - on the Z80, Algocell's initial state exactly (`init: {stack: {topAliases: 'lastByte'}}`);
  - a budget in instructions (`steps`); fetch and start maps when the caller needs them.

  Algocell's suppression patterns become the ablation table only on a Z80 `coarse` table (the `params.suppress`
  words); any other machine refuses a non-empty set.
- **Selection.** Nothing is selected by default. These scripts take `--machine <set or name>` (the others that run
  assays do not; see "Not covered"): (`assay_batch.py`, `c4_functional.py`, `stage_g.py`, `closure.py`, `biology/individuality.py`, `exec_trace.py`,
  `inflow_insoup.py`, `mutscan.py`, `serial_retention.py`, `branching_transfer.py`, `population_sample.py`,
  `population_classes.py`). Those whose inputs or labels are Z80 refuse a machine of another ISA; only
  `population_classes.py --soups` accepts any ISA. Tables with an `executor` column (`assay_batch.py`,
  `branching_transfer.py`) record the machine's name in it.
- **No silent fallback.** While a machine is selected, the shader pipelines (`assay`, `exectrace`, `optrace`, and the
  convention and dial executors of `conv`) refuse to compile, unless `AlgocellShader.run` is called explicitly.
  `E.usage()` counts the pairs each backend ran; with `ALGOCELL_EXECUTOR_LOG` set, the counts are written at exit.

## The machines (`machines/worlds_assays.json`, `machines/PROVENANCE_assays.md`)

Vendored from zilion `next` at `ed93b7c`. `build_machines.mjs --spec worlds_assays.json` builds or checks this file
alone and writes `PROVENANCE_assays.md`; `PROVENANCE.md` stays the provenance of `worlds.json`.

| Set | Machines | World |
|---|---|---|
| `z80c02_assay` | `_L16`, `_L20`, `_L32`, `_L50`, `_L64`, `_lethal_L16`, `_lethal_L32` | the Zilion 0.2.0 Z80 (compat `'0.2'`, core SHA-256 `e6795a6a…`, the hooked text of `z80c02_pair_maps_L16`); A, B of L bytes on 2L; SP aliasing the last byte of B; per-lane budget; the coarse ablation table; writes, fetch and start maps |
| `m6502_assay` | `_L16` | the NMOS 6502 of zilion `next` (experimental; validated per instruction against SingleStepTests 6502/v1); A, B of 16 bytes on 32; `init.stack: {firstPushAt: 'lastByte'}` (S = 0xFF), `init.set: {p: 0x20}`, every other register 0, PC 0; per-lane budget; no ablation table; writes, fetch and start maps |

## Evidence

1. **Pairs (`PAIRS.md`).** Each Z80 assay machine against the shader it replaces, on lattice-neighbour pairs from every
   archived final soup of its length and ring (each condition under its own budget and suppression set) and on uniform
   random pairs with random suppression sets: 890,000 pairs, 0 mismatches in memory, writes and fetch map; the start
   map was compared on the 170,000 pairs of the unsuppressed groups, 0 mismatches. The shader side is the traced and
   op-traced executors (whose memory equals the plain one's).
   Negative controls mismatch: a machine one instruction short (190 memory, 6,790 writes, 18 fetch-map mismatches in
   10,000 pairs), and the lethal machine against the plain shader.
2. **Frozen tables (`REGRESSION.md`).** 17 jobs covering every assay family regenerated 44 files three ways: as
   registered (a), with `--machine z80c02_assay` (b), and, for six jobs, on machines short of budget (n). Each path ran
   72,963,576 pairs; the logs show (a) only on shaders and (b) only on machines.
   - (a) vs (b): every file byte-identical, except wall-clock columns and lines and the `executor` column (shader file
     under (a), machine under (b)).
   - (a) vs the committed tables: identical in the same sense, with two stated exceptions. The Stage G tables predate
     the `zero_halts` and `executor` columns, so their committed columns are compared. The committed Stage G scan
     (`results/mutscan/`) went through `mutscan.py`'s incremental mode: its L = 16 rows were read back once with
     pandas' default float parser, which is not round-trip exact. Replaying that on (a) reproduces the committed file;
     without it, 38 rows differ in the last digits of floats.
   - The negative control differs from (a) in every job that has one. Some are weak: one instruction short moves only
     two columns of Stage I's `assays.csv`, and `population_sample_I` needed 64 instructions short. Non-vacuity rests
     mainly on the pair-level controls and the per-backend usage logs.
   - Provenance: the runs were made from the working tree before commit f61cf45 (`REGRESSION.md`, "Provenance"); the
     short jobs were rerun at the commit and reproduce the recorded outputs (same section).
3. **The 6502 (`SMOKE_6502.md`).** Every generalised assay function ran on `m6502_assay_L16` with uniform random
   tapes, and no shader ran. The plumbing invariants hold, and Z80 suppression, the lethal rule and a shader call are
   refused. Plumbing only: no score is reported.
4. **Tests.** `tests/test_assays_any_isa.py`.

## Which parts of the frozen assays are ISA-neutral

"Neutral as is": the definition refers to bytes, positions, partners and counts only. "Neutral after a stated
generalisation": the definition used a Z80 notion that has one ISA-independent reading, stated here; on the Z80
machines that reading reproduces the shaders exactly (`PAIRS.md`, `REGRESSION.md`), so no Z80 number changes.
"Z80-only": the definition is Z80 semantics or Z80 data.

| Component | Where | Status | Statement |
|---|---|---|---|
| Executing a pair | `assay.execute_pairs`, `exectrace`, `optrace` | neutral after a stated generalisation | Organism A at position 0, partner B after it, a ring of 2L, registers at the world's reset, PC 0, `steps` instructions. The initial stack is the machine's world: Algocell's Z80 has SP aliasing the last byte of B, so its first push lands on byte 2L − 2; the 6502 machine has `firstPushAt: 'lastByte'`, so its first push lands on byte 2L − 1 (open question 1 below). |
| Budget of 128 instructions | every caller | neutral as is | One instruction is one machine step on any ISA. It is a parameter of `assay`, `assay_many`, `scan_tape` and `partner_test`. It is hard-coded (128) in the functions `serial_retention.run_genotype`, `population_classes.classify`, `population_sample.confined`, `inflow_insoup.encounter` (`STEPS`) and `branching_transfer` (`STEPS`), and in the drivers `mutscan.run`, `population_sample.main`, `exec_trace.py`, `closure.main` and `individuality.py` (`Z80_STEPS`). A 512-instruction arm needs a keyword there (default 128); not done. `inflow_insoup.encounter` also hard-codes the plain rule (`zero_halts=False`). |
| Copy: ≥ 0.75 of positions at the best cyclic shift | `assay._best_shift*`, all transfers | neutral as is | Bytes. |
| Culture test: headroom-normalised gain, gen2 ≥ 0.3, faithful ≥ 0.5 of partners ≥ 0.75 copies, `n_informative`, in-situ redraws | `assay.assay`, `assay_many` | neutral as is | Bytes and partner draws (numpy, seeded). |
| `copy_offset` (modal cyclic offset of the copies) | `assay` | neutral as is | The measurement; its reading (Proposition 8, d − 4 transmitted sites for `1e NN ed b0`) is Z80. |
| Partner test: copied ≥ 0.75, self-damage ≥ 25% of bytes | `closure.partner_test`, `individuality` | neutral as is | Bytes. |
| Pointer confinement and `entered` | `exectrace` + `exec_positions`, `population_sample.confined`, `classify`, `exec_trace.py`, `inflow_insoup.py`, `lod_traj` | neutral after a stated generalisation | "Fetched as instruction stream" = every byte the core reads through its fetch path, mapped to its ring position. Z80: opcode, prefixes, displacement and operands (`z80_fetch`); 6502: opcode, operand bytes and BRK's signature byte (zilion `on_fetch`; the 6502's dummy reads are not performed and are not fetches). Confined = no partner position fetched in any of the encounters. |
| Executed set, unexecuted sites | `exec_trace.py`, `serial_retention`, `branching_transfer`, `classify` | neutral after the same generalisation | The union of fetched organism positions. |
| Inflow H (plug-in entropy of the offspring) | `individuality`, `exec_trace.py`, `inflow_insoup.py` | neutral as is | Bytes. |
| Start map (instruction starts) | `optrace` (used by `lod_origin.py`, not by the frozen assays above) | neutral after a stated generalisation | The position of the first byte of each step (Z80: the opcode or the first prefix; 6502: the opcode). |
| Write counts | `PairRun.writes` | neutral after a stated generalisation, with a caveat | Bus writes into A and into B (padding counted with B). A 6502 read-modify-write instruction writes twice (old value, then new), so it counts two; a Z80 instruction counts each byte it stores once. No frozen assay reads write counts (the soup's interaction census does). |
| Single-mutant scan: all 255 (or K seeded) alternatives, aligned transmission, site = half of the values | `mutscan` | neutral as is | Bytes. |
| Serial transfer: lineages, 5-byte window, background | `serial_retention` | neutral as is | Bytes. Its `rule` label now names the ISA that ran ("Z80" on the shaders and the Z80 machines, as before), or "i8080". |
| Branching transfer | `branching_transfer` | neutral as is | Bytes; `executor` column names the shader or machine. |
| Population classes: heritable (32 partners), confined (16 traced encounters), regenerator ≤ 2 sites, transmitter ≥ 5 | `population_classes`, `population_sample` | neutral as is | Counts of byte positions. Reading a class through the copy-offset operand (d − 4) is Z80; the 6502 analogue (d − 6) is a prediction of the draft. |
| t_rep, t_faith, exemplar share ≥ 0.5% | `assay_batch.py` | neutral as is | Read from the soup run's records; `run.py` itself refuses non-Z80 machines (its exemplar mechanisms are Z80). |
| Tolerant period | `metrics.tolerant_period` | neutral as is | Bytes. |
| Lethal tar | `gen_lethal_shader`; world `halting.lethal: {firstUnit: 0}` | neutral after a stated generalisation | "The first fetched byte of an instruction is 0x00": the same rule on any ISA, but not the same meaning (0x00 is NOP on the Z80, BRK on the 6502, where it is the draft's "BRK-halts" arm). Only Z80 lethal machines are vendored. |
| Suppression masks (ablations, `i8080`) | `isa.resolve`, `isa.masks` | Z80-only | Z80 opcode pages (base, CB, ED). Machines take them only as a Z80 coarse table. |
| Mechanism labels (`*_mechs`) | `isa.mechanisms` in `assay_batch.py` | Z80-only | Z80 disassembly. |
| Control flow and block repeat (`has_cf`, `has_block` → `has_loop`) | `closure.control_flow`, `stage_g.block_repeat`; read from `stage_g_runs.csv` by `exec_trace.py`, `mutscan.py`, `individuality.py` | Z80-only | Z80 disassembly. The 6502's "loop instruction" (draft P1) is to be defined at registration. |
| Byte-pattern census and motifs (`c_blockcopy`, `c_push2`, `c_ex_sp`, `c_rst`, `motif_share`) | `metrics.py`, soup runs | Z80-only | Z80 opcode bytes; not part of the assays. |
| Inputs: the genotype panel (`01 c5` …, `CORE_8080`), Stage tables and soups | `serial_retention.panel`, stage scripts | Z80-only | Z80 data. |
| BFF rows of the partner test | `individuality.py`, `micro/bff.py` | not affected | Their own executor. |

## Not covered

- **Rings and lengths.** Only P = 2L at L = 16, 20, 32, 50 and 64 (lethal at 16 and 32) are vendored. Stage F's padded
  and odd rings (`z80_test_L*_P*.wgsl`), the short and long tapes of Stages B–E (L = 3–12, 18, 24, 25, 36, 49, 81, 100)
  and their frozen tables (`closure.py` over L = 3–100, `assay_batch.py` on Stages A–F) are not.
- **The randreg and randsp conventions and the lethality dials** (`conv.py`, `z80_test_{randreg,randsp,lp*}_L*.wgsl`):
  no machine; `pe_matrix.py`, `r4_*`, `conv_soups.py` and `dial_soups.py` keep the shaders (and refuse a selected
  machine).
- **Scripts that run assays but have no `--machine`.** They still route through the interface when a machine is
  selected in-process, but were not regenerated: `stage_c.py`, `stage_e.py`, `stage_f.py`, `capacity_over_time.py`, `threshold_grid.py`, `first_closure.py`, `exact_copy.py`, `robust_ref22.py`, `census_period.py`, `two_byte_census.py`, `check_8080_closer.py`, `landscape.py`, `marker_check.py`, `sr_kin.py`, `sr_posthoc.py`, `stats_ci.py`, `l50_lineage.py`, `lineage_examples.py`, `invasion.py`, `invasion_aligned.py`, `invasion_pair.py`, `lod.py`, `lod_v1.py`, `lod_analyse.py`, `lod_founding.py`, `lod_origin.py`, `lod_traj.py`, `n2_composition.py`, `n2_convert.py`, `n2_demography.py`, `n4_scars.py`, `n5_path.py`, `offset_census.py`, `offset_census_enc.py`, `offset_encounter.py`, `offset_flow.py`, `offset_intruder.py`, `offset_switch.py`, `payload_toxin.py`, `r4_analyse.py`. `z80_ring_gpu.py` (the ring difftest) compiles its own
  executor. `pe_matrix.py`, `r4_*.py`, `conv_soups.py` and `dial_soups.py` use `conv.py` (above).
- **Frozen tables not regenerated here**: see `REGRESSION.md` for the list that was.

## Open questions

1. **Where the first push lands.** On Algocell's Z80 the stack pointer aliases the last byte of B, so the first pushed
   byte lands on byte 2L − 2. The 6502 machine follows the draft ("S such that the first push lands on the last byte
   of the partner", `firstPushAt: 'lastByte'`), so its first pushed byte lands on byte 2L − 1. For the 6502 both of
   Zilion's conventions give S = 0xFF. The draft calls the 6502 world "identical to the Z80 runs"; the registration
   should say which of the two it means.
2. **Budget and rule.** A 512-instruction arm needs the hard-coded 128 of five functions (and five drivers) turned into a
   keyword (default 128); a BRK-halts arm through `inflow_insoup.encounter` needs its `zero_halts=False` lifted.
3. **Loop and class definitions on the 6502.** "Loop instruction" (draft P1) and the copy-offset reading of the
   classes (d − 6) need a 6502 definition at registration. The assays themselves need none.
4. **Write counts.** A 6502 read-modify-write counts two writes; no frozen assay reads write counts, but the soup's
   interaction census would.

