# Instruction-set ablation atlas of self-replication — pre-registered plan

Written 2026-10-07 **before** any sweep ran. Changes after this date are logged
at the bottom with reasons. The point of this file is to keep us honest: the
hypotheses, conditions, outcomes and analysis are fixed here, and we report
every cell of the grid, including the empty ones.

## Question

Which Z80 instruction families does the spontaneous emergence of self-replication
in the Algocell soup depend on, how does the step budget per interaction and the
mutation rate modulate that dependence, and what replication mechanisms appear
when the usual ones are removed?

## System under study

The deployed Algocell simulation, run headlessly with the *exact* shipping WGSL
(exported by `npm run export:sim`; sha in `algocell_exp/shader/meta.json`,
zilion 0.2.0). Square grid only (hex is out of scope for this study), 160×125 =
20,000 cells, 8,192 random neighbour pairs per step, concatenated A+B execution
with all registers zero, byte-replacement mutation of `8192 / 2^k` bytes per
step. No fitness function. GPU scheduling makes runs non-deterministic even for
a fixed seed, so each (condition, seed) is one sample of the process.

**Tape length (organism size)** is a factor: L ∈ {4, 9, 16, 25, 36, 49, 64,
81, 100} bytes (square numbers, so cells tile as √L×√L). Two conventions that
follow from this, fixed before any run:

- *Stack start.* A real Z80 resets SP to 0xFFFF, which for 32-byte pairs
  aliases to the last byte of the second program. For pair lengths that do not
  divide 65536 that alias lands at an arbitrary byte, so SP starts at the
  largest 16-bit value that aliases to the last byte of the pair (`spInit`).
  This is identical to 0xFFFF for L=16, i.e. the deployed system is unchanged.
- *Mutation.* `8192 / 2^k` byte replacements per step regardless of L, so the
  expected number of mutations per **cell** per step is independent of L
  (2.56% at k=4) while the per-byte rate falls as 1/L. We treat mutation as a
  per-organism hazard; the per-byte view is reported alongside.

Larger tapes need more instructions per interaction to be copied at all (a
PUSH-based copier needs L/2 pushes), so the step-budget axis is crossed with L.

## Hypotheses (stated so they can fail)

- **H1 (block copy is dispensable early):** removing LDI/LDD/LDIR/LDDR does not
  change the time to first replicator at the default settings. (Prediction from
  Computational Life: stack replicators come first.)
- **H2 (the stack is the early bottleneck):** removing the stack-writing
  instructions (PUSH, EX (SP),HL, CALL, RST) delays emergence by more than an
  order of magnitude at 128 steps, and the replicators that eventually appear
  are LD-loop or block-copy based.
- **H3 (loads are nearly necessary):** removing all LD families, stack and EX
  ("No-copy", 112 instructions) leaves only read-modify-write and return-address
  pushes as ways to write memory; we predict no replicator within the horizon at
  any step/mutation setting. *The user remembers seeing emergence here with
  short budgets and high mutation; we test that memory as a hypothesis, not a
  fact.*
- **H4 (step budget is the strongest knob):** cutting the step budget from 128
  to 32 delays emergence more than any single-family ablation and changes the
  winning mechanism (pilot: PUSH HL/LD HL,(nn) at 32 steps).
- **H6 (size):** time to emergence grows with L at a fixed budget, because the
  minimal replicator must copy more bytes within the same budget; at 512 steps
  large organisms emerge, and their replicators have more non-replication bytes
  ("free tape"). The user's expectation that larger organisms support more
  complex emergence is tested as: more distinct mechanism classes and a longer
  tail of coexisting quasispecies at large L. We do not assume it.
- **H5 (mutation is non-monotone):** for a fixed budget there is an
  intermediate mutation rate that minimises time to emergence; both very low
  and very high rates delay it (error threshold above).

## Design

Two pre-registered stages; a third is planned only in outline and will be
specified (and logged) after Stages A and B are read.

**Stage A — ablation × budget × mutation at L = 16.** Full factorial:

| factor | levels |
|---|---|
| ablation | `none`; `block-copy`; `stack-writes` = family:stack + family:ex + family:call-ret + family:rst (POP/RET included for simplicity); `ld-mem` = family:ld8-mem + family:ld16-mem (memory loads only); `all-ld` = every LD family; `no-copy` = all-ld + stack + ex + block-copy (the preset); `rmw-only` = family:writes-mem minus incdec-mem/rotate-mem/bit-set-mem (only read-modify-write can write memory) |
| z80 steps | 32, 128, 512 |
| mutation 1/2^k | k = 2, 4, 6 |
| seeds | 10 (seeds 1–10) |

7 × 3 × 3 × 10 = 630 runs. Horizon 300,000 steps (≈ 2.5 × 10^9 pair
interactions, ≈ 120 interactions per cell), stopping early 4 samples after
quasispecies occupancy exceeds 50%. Sampling every 500 steps.

**Stage B — organism size.** L ∈ {4, 9, 16, 25, 36, 49, 64, 81, 100} ×
ablation ∈ {`none`, `block-copy`, `stack-writes`, `no-copy`} × steps ∈
{128, 512} × k = 4 × 10 seeds = 720 runs, same horizon and stopping rule. The
L = 16 cells are shared with Stage A (not re-run).

**Stage C (outline, to be specified after A and B):** long-horizon
(≥ 1,000,000 steps) re-runs of censored cells that A/B flag as borderline, and
the 3-factor crossing L × steps × mutation for whichever ablation shows a
non-trivial mechanism change.

Compute: one run ≈ 300,000 × 0.35 ms ≈ 105 s on an H200 for L = 16 (longer
for large L); Stage A ≈ 18 GPU-hours, Stage B ≈ 30–60 GPU-hours. Runs that hit
the stopping rule finish much earlier.

## Outcomes (all recorded per sample, nothing chosen after the fact)

- `t_02`, `t_10`, `t_50`: first step at which the most common 16-byte tape holds
  ≥ 2%, 10%, 50% of cells (exact match); −1 = censored at the horizon.
- `tq_10`, `tq_50`: the same on **quasispecies occupancy** `q_share`, the
  fraction of cells within Hamming distance ⌈L/4⌉ of the dominant tape (4 for
  L = 16; `q_half_share` uses half that radius). Exact-hash
  share undercounts a replicator because mutation keeps splitting it into
  near-identical variants (local pilot, 2026-10-07, before the sweep: the
  classic replicator plateaus near 10% by exact hash while the grid is visibly
  full of it).
- The dominant tape at each threshold crossing and its **mechanism classes**
  (stack / block-copy / ld-mem / rmw / io), from a linear disassembly of the
  tape's write-capable instructions.
- `motif_share` (secondary): fraction of cells containing the dominant tape's
  most frequent 2-byte word at least 4 times; the "grid looks full of it"
  measure for replicators whose copies drift beyond Hamming 4.
- Species Shannon entropy, Simpson index, richness (≥ 0.1%), unique tapes.
- Byte entropy H0 and high-order entropy (H0 − brotli bits/byte), the
  Computational Life paper's replicator signal.

Primary outcome: `tq_10` (10% quasispecies occupancy is unambiguous
replication; 2% exact share is noisy because return-address smears reach
~0.5%). `t_10` is reported alongside for comparison with exact-species work.

## Analysis

- Per condition: fraction of seeds reaching `tq_10` within the horizon, with
  Wilson 95% intervals; median `tq_10` among those that did; Kaplan–Meier curve
  treating censoring properly (no imputation of censored runs).
- Mechanism table: for each condition, the distribution of mechanism classes of
  the first emergent replicator.
- H1: compare `none` vs `block-copy` at (128, k=4): overlapping intervals and a
  median ratio within [0.5, 2] = supported.
- H2: `stack-writes` vs `none`: median ratio > 10 or ≥ 50% censored = supported.
- H3: any `no-copy` run with `tq_10 > 0` refutes it; we then report the tape.
- H4/H5: monotonicity of median `tq_10` along the steps and mutation axes for
  `none`, with the sign test across seeds.
- H6: median `tq_10` vs L at each budget; number of mechanism classes and the
  species-entropy plateau after emergence vs L.
- Everything is plotted as the full grid (small multiples), not only the cells
  that behave.

## Rules

- No horizon, threshold or condition is changed after seeing results except by
  adding a logged entry below (additions are fine; removals are not).
- Nulls are results. A censored cell is reported as censored, never dropped.
- Single-seed observations (including the user's) are hypotheses.
- Anything that looks like a new mechanism is confirmed by re-running that seed
  and by tracing the tape against a neighbour before it is called a finding.

## Change log

- 2026-10-07, Stage A stopped after 5 of 630 runs (none read) and relaunched
  with two additions per sample: a **byte-pattern census** (fraction of cells
  containing ED-page block-copy pairs, ≥2 PUSH bytes, EX (SP),HL, RST 38,
  LD (HL),x, CB-page (HL) ops, ≥8 zero bytes; top five 4-grams) and **soup
  snapshots** (brotli-compressed, at the first `tq_10` crossing and at the
  end) for post hoc analysis. Motivation from the three pilot runs: after the
  Load–Push takeover the population became a diverse LDIR-based cloud with no
  single dominant genotype (unique tapes 19k → 7k, HOE → 5 bits/byte), which
  per-genotype measures cannot describe.
- 2026-10-07, Stage A launched (no results read yet): added a **replication
  assay** as an interpretation outcome, computed post hoc from the dominant
  tapes stored at every sample. The tape is executed as program A against N
  random neighbours (and as B) for the condition's step budget and suppression
  set, and the score is the best shift-aligned fraction of its own bytes that
  it writes into the neighbour. Motivation: the pilot's no-copy soup is
  dominated by the RST return-address smear `ff 41 00 41 …`, which spreads a
  byte pattern without copying the code that produces it; occupancy alone
  cannot tell a smear from a replicator. The assay also runs the offspring it
  produced against fresh neighbours (**gen2 score**, heritability): on known
  tapes the Load–Push replicator scores 0.81 / gen2 0.58, the RST smear 0.61 /
  gen2 0.08, and the stack-free LDIR replicator from the pilot 0.98 / gen2 0.99
  under its own ablation. A replicator is one with gen2 ≥ 0.3 (fixed here,
  before any sweep is read). The primary outcome stays `tq_10`; the assay is
  reported next to it and used to label each emergence as self-copying or not.
- 2026-10-07, before any sweep: quasispecies radius changed from a fixed 4 to
  ⌈L/4⌉ and the motif repeat count from 4 to max(2, ⌈L/4⌉), after the
  tape-length axis was added (a fixed radius 4 covers an entire 4-byte tape).
  For L = 16 nothing changes.
