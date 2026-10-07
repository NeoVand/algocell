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

*Fixed 2026-10-07 after reading Stage A and Stage B (succession and census;
Stage B assays were being recomputed after the executor fix). The exact
condition list is `make_conds.stage_c()` → `conds/stageC.json` (420 runs). All
Stage C runs have no early stop and store 8 random cell tapes per sample.*

- **C1 — slow or impossible?** `no-copy`, `rmw-only`, `all-ld` at (128, k=4)
  and (32, k=2), 10 seeds, **1,000,000 steps** (sampled every 1,000). Tests
  whether the 0/90 nulls and the rare `all-ld` emergence are horizon effects.
- **C2 — succession without censoring.** `none`, `block-copy`, `ld-mem` at
  128 and 512 steps × k ∈ {2, 4, 6}, 10 seeds, 300,000 steps.
- **C3 — finer ablations** at (128, k=4) and (32, k=2), 10 seeds, 300,000
  steps: `push-only` (PUSH/POP), `ex-sp-only` (EX (SP),HL), `call-rst`,
  `ld-imm` (immediates), `ld-reg` (register loads + LD SP,HL/I/R), `cb-page`
  (all CB), `ed-loads` (ED 16-bit memory loads).
- **C4 — functional fraction over time:** from the random tapes of every C run.
- **C5 — size without censoring:** `none` and `stack-writes` at L ∈ {36, 100},
  (128, k=4), 10 seeds, 300,000 steps, to measure whether large organisms stay
  tiled by short motifs (Stage B first look: PUSH family in 10/10 seeds for
  every L ≥ 25, final dominant tapes of small period) when the run is not
  stopped at 50% occupancy.

Predictions, stated now: C1 — `no-copy` and `rmw-only` remain null at 1M;
`all-ld` emergence fraction rises above 0.3. C3 — `ex-sp-only` and `push-only`
each delay emergence less than `stack-writes` did (redundant stack routes);
`ld-imm` is the load family whose removal costs most (Load–Push needs
immediates); `cb-page` and `ed-loads` change nothing at L = 16. C5 — dominant
tapes at L = 100 have period ≤ 8 in most seeds; no whole-tape replicator of
period > 32 appears.

**Stage D (confirmatory, fixed 2026-10-07 after reading Stage B, before any
Stage C result):** Stage B found, in one cell and post hoc, that removing the
stack-writing families at L = 9 makes replication *more* likely (see change
log). Stage D tests that as a stated hypothesis with new seeds:
`none`, `stack-writes`, `push-only`, `call-rst` at L = 9, (128, k=4) and
(512, k=4), **20 seeds each** (seeds 1–20, all new runs; Stage B's L = 9 runs
are not reused), 300,000 steps, no early stop, 8 random tapes per sample
(`make_conds.stage_d()` → `conds/stageD.json`, 160 runs).
Predictions: (D1) `stack-writes` has a higher emergence fraction than `none`
at both budgets (one-sided Fisher, α = 0.05, n = 20 per arm). (D2) the
unablated soups are zero floods (≥ 25% of bytes `00`) and the ablated ones
are not. (D3) `push-only` (PUSH/POP removed, CALL/RST kept) recovers most of
the effect, i.e. the flood is mainly `PUSH rr` of still-zero registers;
`call-rst` alone recovers less. (D4) every L = 9 replicator is LDIR-based
with period 3 or 9. If D1 fails the Stage B observation is reported as a
non-replicated single cell.

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

- 2026-10-07 (Stage C relaunched): **runner bug, Stage C aborted and
  re-run; no Stage A/B data affected.** Stage C conditions carry
  `stop_share: -1` to disable the early stop, but `run()` only treated `None`
  as "never" (the CLI converted −1, the Modal path did not), so every Stage C
  run stopped 4 samples in, at 2,500–5,000 steps (0.15 GPU-h, ≈ $0.30
  wasted). Fixed so any non-positive share means never stop, verified
  locally, and the 420 runs relaunched unchanged. The aborted batch is kept
  in `runs/stageC_aborted/` and is not used for any conclusion; its 2,500-step
  snapshots agree with Stage A's early dynamics (`ld-imm` and `push-only`
  0/10 by 2,500 steps, the other finer ablations 9–10/10 by 500–1,000).
  Stage A and B used `stop_share: 0.5` (a positive value), so their early
  stop behaved as pre-registered.
- 2026-10-07 (Stage B assays recomputed and read; Stage C finished, not yet
  read). Three post hoc additions, all reported next to the pre-registered
  measures, none replacing them:
  (a) **Faithfulness.** The no-copy soups at L = 100 contain a trace-level
  (top share 0.2%) `CALL`-chain `cd xx cd xx …` whose CALLs push their own
  16-bit return address; the high byte of that address is the CALL opcode
  itself (PC runs in the unreduced 16-bit space, so after `JP`/`CALL $xxCD`
  every return address is `$CDyy`). The neighbour is flooded with `cd yy`
  pairs (50% similar), the offspring do the same (gen2 0.49), but the operand
  drifts by +4 each generation and the lineage dies by generation 3. It passes
  the gen2 ≥ 0.3 heritability rule, so that rule alone is not enough: every
  table now also reports `offspring_within_q` (share of partners that became
  ≥ 75% copies) and a replicator is called **faithful** when it is ≥ 0.5.
  Nothing in Stages A/B changes except this one case, which is labelled
  "heritable, not faithful"; no-copy stays 0/160 by `t_rep`.
  (b) **Tiling.** Dominant tapes' minimal period relative to L (does the
  period divide L; is it ≤ L/2), and the high-order entropy of the final soup
  per family, to test H6's "free tape" clause directly.
  (c) **A reverse ablation at L = 9**, found in the data and NOT
  pre-registered: removing the stack-writing families makes replication
  *more* likely (9/10 and 10/10 vs 4/10 and 1/10; Fisher p = 0.057 at 128
  steps, 0.0001 at 512). Unablated L = 9 soups are a zero flood (34% of all
  bytes are `00`, byte entropy 6.0 bits vs 7.4 without stack writes), written
  by PUSH of still-zero registers and by CALL/RST return addresses; the
  replicator that does emerge is the period-3 `DEC E ; LDIR` tiling 9 bytes
  exactly. This is one cell of the design with n = 10 per arm and is logged
  as a hypothesis for a confirmatory run (Stage D proposal: L = 9 ×
  {none, stack-writes, push-only, call-rst} × {128, 512} × 20 seeds), not as
  a result.
- 2026-10-07 (Stage B complete, first table read): **analysis bug, no data
  affected.** The exported single-pair executor used by the assay had a fixed
  40-byte private memory (sized for the 32/38-byte differential tests), so
  every post hoc assay of a pair longer than 40 bytes (L ≥ 25) indexed out of
  bounds and returned garbage (a known `LD E,49 ; LDIR` copier scored 0.00 at
  L = 49). The executor is now exported per tape length and the Stage B assays
  are recomputed; L ≤ 16 results are unchanged. Also: census takeover
  baselines are now estimated per L from random tapes (≥ 2 PUSH bytes occurs
  in 46% of random 100-byte tapes), and the minimal period of dominant tapes
  is recorded, since the first look suggests large organisms get *tiled* by
  short motifs rather than hosting larger replicators.
- 2026-10-07 (same reading): the assay's gain is now normalised by headroom,
  (after − before)/(1 − before), averaged over partners whose prior similarity
  to the tape is < 0.75. Reason: the in-situ random-cell assay returned 0/16
  heritable cells in soups where 60% of cells carry the Load–Push pair,
  because a copier cannot raise the similarity of a partner that is already a
  copy. Random-partner scores are essentially unchanged (prior similarity
  ≈ 0.06). Threshold stays gen2 ≥ 0.3.
- 2026-10-07 (Stage A 211/630 read, interrupted by the Modal outage): two
  blind spots found, both logged for the analysis and for Stage C:
  (a) `t_rep` is built from the top-3 exact exemplars, so a maximally diverse
  replicator cloud (every member unique; no genotype ≥ 0.5%) is never assayed
  — e.g. stack-writes / 32 steps: the census shows LDIR in ≥ 30% excess of
  cells by step 5k in 10/10 seeds, `t_rep` fires in 4/10. We therefore also
  report the **census takeover time** per family (first sample with ≥ 30%
  excess over the random-soup baseline) and validate the end state with a
  **random-cell in-situ assay** (16 random cells of the final snapshot, each
  assayed against its own population; fraction heritable). Stage C runs will
  store 8 random tapes per sample so the assay can be applied over time.
  (b) The pre-registered early stop (4 samples after 50% quasispecies
  occupancy) ends runs once a stack family dominates, so later LDIR invasions
  in those runs are censored; succession statistics use only runs that reached
  the horizon, and Stage C re-runs the affected cells without the early stop.
- 2026-10-07 (Stage A running; 31 runs read): added an **in-situ** variant of
  the assay for the final state — partners are drawn from the stored soup
  snapshot instead of uniform random bytes. Reason: members of the LDIR cloud
  take their pointers from the partner's bytes (`POP DE … LDIR`), so against
  random partners they score low although 96–99% of cells carry the block-copy
  pair. Random-partner scores are a lower bound (reported as before); the
  in-situ score is used for the final-state replicator label.
- 2026-10-07 (Stage A running; the first 19 of 630 runs read, all from the
  `none` / 32-step cells): the pre-registered occupancy detector `tq_10` has
  **both failure modes** in the data. False negatives: at mutation 1/4, 0/10
  runs reached 10% quasispecies occupancy, yet every final dominant tape is a
  heritable LDIR replicator (assay gen2 0.97–0.99) living in a diverse cloud
  with a free first byte, so no Hamming ball around one genotype ever fills.
  False positives: at 1/16, 2 of 8 `tq_10` crossings are non-heritable smears
  (`00 41 00 41`, `00 04 00 04`; gen2 0.08). We therefore ADD an assay-based
  emergence time **`t_rep`**: the first sample at which any of the stored top-3
  exemplars has gen2 ≥ 0.3 and exact share ≥ 0.5% (so one random cell does not
  count), computed post hoc from the JSONL with the run's own budget and
  suppression set. `tq_10` stays reported as pre-registered; conclusions about
  emergence use `t_rep`, and every table shows both. The horizon, grid and
  conditions are unchanged.
- 2026-10-07 (Stage A running, no results read): the assay is applied to the
  **top-3 exemplars** at emergence, keeping the best heritability. Reason, from
  the browser probe data: the most common exact genotype in a run can be the
  *sterile offspring* of a copier — `LD HL,$E321 ; PUSH HL` (`21 e3 21 e5`)
  writes `21 e3 21 e3 …` into its neighbour, so the inert `21 e3 ×8` tape
  outnumbers the copier that makes it (assay: offspring gen2 0.08, parent 0.27,
  parent copying confirmed byte-for-byte against a blank neighbour).
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
