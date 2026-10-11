# ZSoup: Algocell's soup on any Zilion machine

ZSoup runs Algocell's soup with the CPU replaced by an exported Zilion machine. Nothing else changes: the lattice, the
pairing, the collision claim, absorption, mutation, the statistics kernels and the Python API are Algocell's own. The
purpose is to run the same soup on a second instruction set (the NMOS 6502) with only the CPU changed, and to give the
emulator impact study and the 6502 census one runner for every core. That needs proof that the new path reproduces the
published Z80 pipeline bit for bit. The evidence is below. Every measured number in this file is quoted from a
generated table in `experiments/results/zsoup/`. Facts about the machines come from their generated manifests and
`machines/PROVENANCE.md`.

## Pieces

| File | What it is |
|---|---|
| `algocell_exp/zmachine.py` | Generic Python host for exported Zilion machines (`zilion.manifest/2`), adapted from zilion's `test/world/gpu_runner.py` (MIT, same author). It reads everything from the manifest: bindings, params, memory and record layouts, state fields. `ZMachine.run(memory, budget, seed, ablation, step_base, …)` takes n lanes of memory in the host layout (segments back to back; for a pair, tape A then tape B) and a uniform or per-lane budget. It returns a `ZResult`: memory, steps, halt reason, trap cause, write counts per segment, fetch and start maps, and the final state. `bind_group`/`encode` let a host run the machine on its own buffers. |
| `algocell_exp/machines/` | The vendored machines: `worlds.json` (named world specs, written out explicitly, `init.stack` and `budget` included), the generated `<name>.wgsl` and `<name>.manifest.json`, `PROVENANCE.md` (the zilion commit and every machine's core, world and WGSL SHA-256), and `build_machines.mjs` (regenerate, or `--check`). |
| `algocell_exp/zsoup.py` | `ZSoup`, the soup with execution delegated to a machine, and `ZBridge`, the shader and buffers it runs on. |
| `algocell_exp/zsoup_check.py` | The validation harnesses: pair executors on archived tapes, the step-locked differential test, the nondeterminism measurement, and the plumbing check. |
| `zsoup_validate.py` | Runs them and writes the tables in `results/zsoup/`. |
| `tests/test_zsoup.py` | 35 tests, under a minute on the M4. |

### The vendored machines

All Z80 machines are `compat:'0.2'`, the 0.2.0 core every registered result ran on. Their worlds are Algocell's pair:

- memory: segments A and B of L bytes on a ring of P = 2L;
- stack: `init.stack: { topAliases: 'lastByte' }`, so SP aliases the last byte of B (SP = 0xFFFF at L = 16 and 32,
  in each manifest's `init.stack`);
- budget: `{ unit: 'steps', value: 'perLane' }`, where a lane with budget 0 takes no step;
- ablation: the uniform table in the Z80 `coarse` layout, which is Algocell's `params.suppress` word for word;
- outputs: write counts per segment.

| Machine | World |
|---|---|
| `z80c02_pair_L16`, `z80c02_pair_L32` | the plain pair; no hooks, so the emitted core is the 0.2.0 core text byte for byte (`PROVENANCE.md` checks the SHA-256 against zilion's `Z80_CORE_WGSL`) |
| `z80c02_lethal_L16`, `z80c02_lethal_L32` | plus `halting.lethal: { firstUnit: 0 }`, Stage I's lethal tar |
| `z80c02_pair_maps_L16` | plus the fetch and start maps (the outputs of Algocell's op-traced executor) |
| `m6502skel_pair_L16` | the SKELETON 6502 core of zilion `next`, with `init.stack: { firstPushAt: 'lastByte' }`; for plumbing only |

Zilion's own exporter refuses skeleton cores, so `build_machines.mjs` calls `buildMachine` itself. Each machine's size
is listed in `PROVENANCE.md`.

## How ZSoup works

```
              Soup (original)                                  ZSoup
 clear_collision ─ prepare_batch ─ z80_execute_batch ─      clear_collision ─ prepare_batch ─ zsoup_budget_from_active ─
 absorb_results ─ mutate_soup                               [Zilion machine: memory = pair_data] ─ zsoup_write_counts ─
                                                            absorb_results ─ mutate_soup
```

1. **The soup kernels are the original text.** `extract_soup_kernels` parses the exported simulation shader
   (`sim_square_L<L>.wgsl`, or `sim_lethal_L<L>.wgsl` for a lethal machine, which is the file `Soup(zero_halts=True)`
   compiles) into its top-level items. It keeps the ten bindings (0–9) and the seven soup kernels (`clear_collision`,
   `prepare_batch`, `absorb_results`, `mutate_soup`, `count_bytes`, `clear_byte_counts`, `hash_cells`), plus the
   closure of what they reference: `Params` and the PCG stream (`pcg`, `rng`, `rand`, `rand_bounded`). Each kept item
   is a slice of the source, unchanged. `z80_execute_batch` and the whole Z80 are dropped.
2. **The machine executes in place in `pair_data`.** The machine's memory binding is the soup's own `pair_data`
   buffer. Its segments A and B are stored word-padded back to back, which is exactly how `prepare_batch` lays out a
   pair. There is one lane per drawn pair slot: lane count = `pair_count`, lane i = pair slot i.
3. **Inactive pairs do not execute.** `zsoup_budget_from_active` sets lane i's budget to `params.z80_steps` if pair i
   won the collision claim, and 0 if it lost. A lane with budget 0 loads and stores its memory unchanged and takes no
   step. The original executor returns early for such a pair; the result is the same.
4. **Write counts.** `zsoup_write_counts` copies each active pair's per-segment write counts from the machine's lane
   record into `write_counts`; writes into ring padding count as B's, as Algocell counts them. Inactive pairs keep the
   zeros `prepare_batch` wrote. So `read_interactions()` is unchanged.
5. **Same API.** `ZSoup` subclasses `Soup` and overrides only:
   - the constructor and the build;
   - `step` (same signature);
   - `set_suppress` (same signature), which refuses, and leaves the soup unchanged on, masks the machine cannot take.

   `reset`, `read_soup`, `write_soup`, `read_interactions`, `read_pair_memory`, `read_byte_counts`, `read_hashes`,
   `read_active_pairs`, `read_pairs` and `sync` are `Soup`'s own code.

   `run.py --machine <name>` (keyword `machine=`) runs a condition as a ZSoup. It accepts Z80 machines only, because
   its exemplar mechanisms and suppression counts are Z80 disassembly. The condition record then gains `machine`, and
   its provenance gains the machine's SHA-256s and the SHA-256 of the shader ZSoup compiles. `shader_file` and
   `shader_sha256_16` keep their meaning: the exported file the soup kernels come from, and that file's hash.
6. **ISA-agnostic.** ZSoup takes a machine name. The world's conventions come from the machine: the initial stack
   (listed in `manifest.init.stack`), the entry point, the lethal rule and the ablation layout. ZSoup itself only
   checks that the memory is the soup's pair: two equal segments A and B, word-padded back to back, a per-lane step
   budget, and write counts in the record. L and the ring P come from the manifest.
   - Each step passes its batch seed as the machine's run seed. Only machines with dials or random initial state read
     it.
   - The soup's suppression masks become the machine's ablation table, word for word, on a Z80 `coarse` table
     (`ablation_for`). An empty set needs no table. Any other table or ISA refuses a non-empty set, because Algocell's
     masks are Z80 opcode pages.

## Evidence

### 1. The kept kernels are the originals (`extraction.md`)

The parser splits each exported simulation shader into its top-level items. In all four sources
(`sim_square_L16`, `sim_lethal_L16`, `sim_square_L32`, `sim_lethal_L32`) it finds 117 items, and the text between
consecutive items is whitespace and comments only, so no code escapes the parse.

- **Kept:** 22 items per source. All 88 item checks pass: each kept item is byte-identical to its source slice, occurs
  exactly once in the source, and appears in the ZSoup shader.
- **Same in every source:** the kept text is identical in all four sources.
- **Dropped:** 95 items, `z80_execute_batch` among them.

`tests/test_zsoup.py` asserts the same, and three more things:
- the exact 22-item kept set;
- that no kept item names Z80 code;
- that the ZSoup shader is exactly the kept items followed by the bridge.

### 2. Pair executors on evolved tapes (`pairs_equivalence.md`)

`zsoup_validate.py pairs` samples lattice-neighbour pairs, uniformly with replacement and with seed 1, from every
archived final soup at L = 16 (P = 32) and at L = 32 (P = 64). It finds 3,673 archived soups. Stages B, D and F, and
the other lengths, have no vendored machine for their L and ring, so they are listed and skipped.

Each pair runs twice, with the soup's own step budget and suppression masks (39 budget-and-mask groups at L = 16):
- through Algocell's single-pair executor (`z80_test[_lethal]_L<L>.wgsl`, compiled as `assay.py` compiles it);
- through the vendored machine.

Per pair, the test compares the final memory, the writes into A and into B, and the registers a f b c d e h l sp pc.

| Condition | Soups | Pairs (distinct) | Memory changed by execution | Mismatches (memory / writes / registers) |
|---|---|---|---|---|
| L = 16 plain (lod, Stages A, C, E, G, H) | 1,343 | 1,343,000 (1,005,711) | 562,207 | 0 / 0 / 0 |
| L = 32 plain (Stage E) | 20 | 200,000 (122,891) | 118,312 | 0 / 0 / 0 |
| L = 16 lethal tar (Stage I) | 10 | 200,000 (121,913) | 122,733 | 0 / 0 / 0 |

- **Maps.** The fetch and start maps (`z80c02_pair_maps_L16`, compared against Algocell's op-traced executor) agree
  on all 233,000 pairs of the 128-step, unsuppressed L = 16 group: 0 fetch-map and 0 start-map mismatches.
- **Negative controls** on 20,000 of those pairs. Both must and do show mismatches:
  - giving the machine 127 steps instead of 128: 295 memory, 14,687 write-count and 19,849 register mismatches;
  - running the lethal machine against the plain executor: 7,415, 8,000 and 8,620.

### 3. Step-locked differential test in the soup (`lockstep.md`)

The whole soup cannot be compared run against run, because the original is not deterministic (section 5). So the
test runs the original `Soup` and, at every step, copies `pair_data` and `write_counts` after `prepare_batch`. It
then runs the original `z80_execute_batch` in place and the Zilion machine on the copies, through ZSoup's own bridge
code (`ZBridge.encode_execute`). A comparator kernel counts, for every step and every drawn pair slot, active or not,
the slots whose `pair_data` or write counts differ. The soup continues from the original executor's output.

The raw buffers are also re-read on the CPU, after every step for the first 3,000 steps and every 100th step after
that. The CPU counts must equal the comparator's.

| Run | Steps | Steps with a mismatch | CPU re-checks (disagreeing) | Active pairs whose memory changed (smallest 1,000-step window) | Emergence (500-step checkpoints) |
|---|---|---|---|---|---|
| L = 16, seed 1 | 20,000 | 0 | 3,170 (0) | 49,254,602 (899,606) | none: q_share peaked at 0.07415 |
| L = 16, seed 2 | 20,000 | 0 | 3,170 (0) | 28,387,482 (479,807) | t_02 and tq_10 at 3,000; q_share 0.8069 |
| L = 16 lethal, seed 1 | 20,000 | 0 | 3,170 (0) | 3,936,986 (187,861) | none (tar, `00 01 …`) |
| L = 16 lethal, seed 2 | 20,000 | 0 | 3,170 (0) | 16,550,699 (191,463) | none: q_share peaked at 0.0279 |
| L = 32, seed 1 | 20,000 | 0 | 3,170 (0) | 57,277,526 (1,272,901) | at 500 (the pusher `01 c5 …`) |
| L = 32, seed 2 | 20,000 | 0 | 3,170 (0) | 65,336,551 (1,429,409) | at 500 (`21 e5 …`) |
| L = 32 lethal, seed 1 | 20,000 | 0 | 3,170 (0) | 3,602,777 (169,525) | none (tar) |
| L = 32 lethal, seed 2 | 20,000 | 0 | 3,170 (0) | 3,613,563 (169,932) | none (tar) |
| L = 16, seed 3, `family:block-copy` suppressed | 5,000 | 0 | 3,020 (0) | 4,271,972 (483,559) | at 2,500 (the pusher) |

- **Scale.** Each 20,000-step run compares 163,840,000 pair slots, and the mismatch count is 0 for both memory and
  write counts in every run.
- **The machine runs, and runs only the active pairs.** In every run its active lanes execute between 674,850,090 and
  10,356,609,260 instructions, and its inactive lanes execute none (`zil_inactive_steps` = 0).
- **The original executor leaves inactive pairs alone.** It changes no inactive pair's memory (`inactive_changed` =
  0). The machine's inactive lanes are covered by the mismatch count, which compares every slot.
- **Negative controls.** Both show mismatches at every one of their 200 steps, and the CPU re-checks agree with the
  comparator:
  - a machine budget of 127: 6,949 memory and 30,242 write-count mismatches;
  - the lethal machine in a plain soup: 138,364 and 282,945.
- **Replicators.** The plain runs span the emergence of replicators in 3 of 4 runs. The lethal runs mostly sink into
  tar (`00 01 …`).
- **Coverage over time.** Every 1,000-step window still has at least 169,525 executions that changed memory. All
  windows are in `lockstep_windows.csv`, and the population trajectories are in `lockstep_trajectory.csv`.

### 4. A second ISA through ZSoup: plumbing only (`plumbing.md`)

`m6502skel_pair_L16` puts zilion's SKELETON 6502 in the same pair world, with the stack given by meaning
(`firstPushAt: 'lastByte'`, which gives S = 0xFF; the manifest lists it). ZSoup runs it with no change and no Z80
assumption.

`plumbing_check` runs it for 200 steps. After each step, the soup's own pair memory, write counts and lane records are
compared with a direct `ZMachine.run` of the same machine on the pairs that step drew.

- **Result:** 837,089 active pairs, 0 memory, 0 write-count and 0 record mismatches.
- **Activity:** 1,883 pairs were changed by execution. The skeleton traps on what it does not implement: 794,908
  `unimplemented` traps and 42,099 `illegal` (JAM) traps.
- **Z80 under the same check:** 836,895 active pairs, 0 mismatches.

This shows the plumbing only; the skeleton is not a 6502.

### 5. The original pipeline's own nondeterminism (`nondeterminism.md`)

Two runs of the original `Soup` with the same seed (run one after the other in one process) differ from step 1 in all
11 trials (L = 16 seeds 1–5, L = 16 lethal seeds 1–3, L = 32 seeds 1–3).

- **What agrees.** The initial soups and the drawn pairs agree, because both are seeded.
- **What differs.** Which drawn pairs win the collision claim differs. At step 1, 494 to 2,455 of the 8,192 drawn
  pairs have different `pair_active` flags. After step 1 the soups differ in 3,249 to 22,403 bytes.
- **Over 50 steps.** No step of any trial had identical soups.

The claim race shows at step 1 in every trial, and the unsynchronised mutation write can add to it. This is why the
equivalence is proved encounter by encounter (section 3) and never run against run.

### 6. Throughput (`throughput.md`)

Steps per second on the M4, median of 3 alternating repetitions of 3,000 steps, 32 steps per submit for both:

| Condition | Soup | ZSoup | ZSoup / Soup |
|---|---|---|---|
| L = 16 | 713.8 | 666.1 | 0.933 |
| L = 16 lethal | 1779 | 1301 | 0.731 |
| L = 32 | 1020 | 810.6 | 0.795 |

The GPU was shared with other jobs throughout, so absolute rates and single repetitions vary; `throughput.md` lists
every repetition. Where the cost comes from:
- ZSoup issues seven dispatches per step where Soup issues five;
- each lane stores its state record as well as its memory;
- lethal tar uses a hook call where Algocell's shader has an inline test.

The cost shows most where encounters are short, as in tar.

## Using it

```python
from algocell_exp.zsoup import ZSoup
soup = ZSoup("z80c02_pair_L16", seed=6, suppress="family:block-copy")   # Soup's keywords, minus tape/ring/rule
soup.step(1000)
cells = soup.read_soup(); inter = soup.read_interactions()
records = soup.zmachine.decode(soup.read_records())   # the machine's view of the last step: steps, halts, traps, state

from algocell_exp.zmachine import load_machine
m = load_machine("z80c02_pair_maps_L16")
r = m.run(pairs, budget=128, ablation=masks)            # pairs (n, 2L) uint8
r.memory, r.steps, r.halt_reason, r.trap_cause, r.writes("B"), r.fetch_map, r.start_map, r.reg("pc")
```

`python -m algocell_exp.run --machine z80c02_lethal_L16 …` runs a condition through ZSoup.

To regenerate the machines after a Zilion change, run `(cd ../zilion && npm run build)`, then
`node experiments/algocell_exp/machines/build_machines.mjs`; `--check` verifies them. `tests/test_zsoup.py` runs the
check when the Zilion checkout is at the commit in `PROVENANCE.md`.

## Limits and open questions

- **Vendored worlds.** Only P = 2L at L = 16 and 32 is vendored. Stage F's padded and odd rings, the hex grid
  (`sim_hex.wgsl`), the well-mixed and dial variants, and the randreg/randsp conventions are not. Zilion builds all of
  them (its `test:algocell` covers the rings, dials and conventions on random pairs), and adding one means a world spec
  in `worlds.json` and an equivalence run. Algocell's dial and convention shaders seed their PCG streams from
  (`params.batch_seed`, pair index), and ZSoup passes exactly these as the machine's (seed, lane). So zilion's raw-hook
  versions (`test/world/algocell.mjs`) should run in the soup unchanged, but this is untested.
- **Grids.** ZSoup keeps the square-lattice soup kernels only.
- **Pinned Zilion.** The machines were exported from zilion `next` at `81c88dc` (`PROVENANCE.md`). Every Z80 machine
  states `compat: '0.2'`, and the exporter refuses a Z80 world without one. Since the Z80 0.3 merge (`81c88dc`), the
  Z80 descriptor is version 0.3.0. Its frozen `'0.2'` compat emits the 0.2.0 core text unchanged: the core SHA-256 equals
  `Z80_CORE_WGSL`'s. All tables above were produced with these files.
- **Readbacks before the first step.** As in `Soup`, the statistics kernels (`read_byte_counts`, `read_hashes`) read
  the soup's dimensions from params ring slot 0, which only `step` fills. Before the first step they return zeros. This
  is inherited, not changed.
- **Halt codes.** Under lethal tar, Algocell's executor records a halt as `cpu_halted = 1`, the HALT value. Zilion
  records it as a host halt (2). The memory, the write counts and every register compared are identical; the halt code
  is not part of the soup's state.
- **Determinism.** ZSoup inherits the soup's nondeterminism (the claim race and the mutation race). Equivalence is
  therefore proved encounter by encounter, not run against run.
- **The 6502.** The 6502 in `machines/` is zilion's skeleton. The 6502 soup needs the validated core (zilion M2) and
  a world spec, and nothing else in ZSoup. `run.py` refuses non-Z80 machines, because its exemplar mechanisms and
  suppression counts are Z80 disassembly; they need 6502 counterparts.
- **Cost.** ZSoup issues seven dispatches per step where Soup issues five (section 6). Fusing the two bridge kernels
  into neighbouring passes would change kept kernel text, so it was not done.
