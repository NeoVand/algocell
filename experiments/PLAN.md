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

7 × 3 × 3 × 10 = 630 runs. Horizon 300,000 steps. 8,192 pairs are drawn per
step but only those whose two cells are not claimed by another pair run:
measured on the GPU, 3,982 ± 35 pairs per step (48.6%), i.e. 0.40 interactions
per cell per step, ≈ 1.2 × 10^9 interactions and ≈ 1.2 × 10^5 participations per
cell over 300,000 steps (corrected 2026-10-07; the original text said ≈ 120 per
cell, off by three orders of magnitude). Mutation: 8192/2^k bytes per step over
20,000 cells, i.e. one mutated byte per cell every 2^k × 2.44 steps regardless of
L (39 steps at k = 4), so the per-byte rate is proportional to 1/L. Stopping early 4
samples after quasispecies occupancy exceeds 50%. Sampling every 500 steps.

**Stage B — organism size.** L ∈ {4, 9, 16, 25, 36, 49, 64, 81, 100} ×
ablation ∈ {`none`, `block-copy`, `stack-writes`, `no-copy`} × steps ∈
{128, 512} × k = 4 × 10 seeds = 720 runs, same horizon and stopping rule. The
L = 16 cells are shared with Stage A (not re-run).

**Stage C — long horizons, uncensored succession, finer ablations.** *Outlined before
A/B; fixed 2026-10-07 after reading them; re-specified the same day after the design
review (REVIEW.md §4) and before any Stage C result was read. The exact list is
`make_conds.stage_c()` → `conds/stageC.json` (490 runs).* Common to every C/D/E run:
no early stop; sampling every 50 steps until step 5,000 and every 500 thereafter
(emergence in A/B sat at the 500-step resolution floor) **and at steps 1, 2, 3, 5, 8, 13,
21, 34** (the zero flood forms within the first ~30 steps: 6.5% zeros after step 1, 31% by
step 50, so a 50-step grid would record only its plateau); the 256-bin byte histogram, the
zero-byte fraction and the active-pair count in every sample; the shader's per-pair write
counters summarised in every sample (silent-interaction fraction, writes into partner and
self, histogram); **full soup snapshots at 500, 1k, 2k, 3k, 5k, 7.5k, 10k, 15k, 20k, 30k,
50k, 75k, 100k, 150k, 200k, 300k (and 500k, 750k, 1M)** steps, each with an **interaction
census** (before/after state of every pair that interacted in that step: copy events A→B
and B→A at best cyclic shift ≥ 0.75, partial copies, bytes changed, zeros written, copy
offsets, programs destroyed; aggregates plus 128 raw pairs); 16 uniformly random cell
tapes and the top-10 exemplars per sample; full provenance (shader and ISA hashes, git
commit, library versions incl. brotli, adapter) in every record. **Clock:** because the
fraction of drawn pairs that survive the parallel collision claim depends on GPU
scheduling (48–56% depending on grid size and load), every cross-arm comparison in time
is made in **cumulative active interactions per cell** (from the recorded `active_pairs`),
with raw steps shown alongside. **Seeds 101–110**: Stage A/B used seeds 1–10, and
a seed fixes the initial soup and the RNG stream, so the cells that generated the
hypotheses are not re-used.

- **C1 — slow or impossible?** `no-copy`, `rmw-only`, `all-ld` at (128, k=4) and
  (32, k=2), **1,000,000 steps** (sampled every 1,000 after 5,000).
- **C2 — succession without censoring.** `none`, `block-copy`, `ld-mem` at 128 and
  512 steps × k ∈ {2, 4, 6}, 300,000 steps.
- **C3 — finer ablations** at (128, k=4) and (32, k=2), 300,000 steps: `push-only`
  (PUSH+POP, 8), `ex-sp-only` (1), `call-rst` (CALL*, RET*, RST; 34), `ld-imm` (11),
  `ld-reg` (54), `cb-page` (256), `ed-loads` (8), and — added by the review — the
  write/read split of the Stage A arm: **`stack-write-only`** (PUSH, EX (SP),HL, CALL*,
  RST; 22 opcodes, the ones that write through SP) and **`stack-read-only`** (POP, RET*,
  EX DE,HL, EXX, EX AF,AF'; 24 opcodes that write nothing); the Stage A arm
  **`stack-writes`** itself re-run with the new seeds (a direct replication of H2 under the
  C instrumentation); and the unablated comparator `none` at (32, k=2), which C2 lacked. The Stage A/B arm
  `stack-writes` (46) is their union and is misnamed: the LDIR zoo uses POP DE and
  EX DE,HL for pointer set-up, so its 20–100× delay confounds write removal with
  pointer-route removal.
- **C4 — functional fraction over time**, from the random tapes of every run.
- **C5 — size without censoring:** `none` and `stack-writes` at L ∈ {36, 100},
  (128, k=4), for continuity with Stage B's arm; the full size axis without censoring
  is Stage E.

Predictions, stated before any C run is read: C1 — `no-copy` and `rmw-only` remain
at 0/10 at 1M steps; `all-ld` emergence fraction rises above 0.3. C3 —
`stack-write-only` reproduces the Stage A `stack-writes` delay within a factor 2 and
`stack-read-only` is indistinguishable from `none` (seed-paired sign test, ≤ 7/10
concordant); `ex-sp-only` and `push-only` each delay less than `stack-write-only`;
`ld-imm` is the load family whose removal costs most; `cb-page` and `ed-loads` change
nothing at L = 16. C5 — dominant tapes at L = 100 have period ≤ 8 in most seeds.

**Stage D — the L = 9 reversal, confirmatory.** *Fixed 2026-10-07 after reading Stage
B; re-specified the same day after the review and before any Stage D result was read
(the first version re-used seeds 1–20, i.e. Stage B's own initial soups, and did not
separate write-side from read-side stack opcodes). `make_conds.stage_d()` →
`conds/stageD.json` (280 runs).* Stage B observed, post hoc, that removing the 46-opcode
stack arm at L = 9 raises emergence from 4/10 and 1/10 to 9/10 and 10/10 (Fisher
two-sided p = 0.057 and 1.2 × 10⁻⁴; one-sided 0.029 and 6 × 10⁻⁵), with every L = 9
replicator LDIR-based, mostly the 3-byte unit `DEC E ; LDIR`.

Design: L = 9, k = 4, **seeds 1001–1020 (new)**, 300,000 steps, Stage C
instrumentation. Arms × budgets {128, 512}: `none`, `stack-writes` (46, Stage B's
arm), `stack-write-only` (22), `stack-read-only` (24), `push` (PUSH only, 4),
`call-rst-write` (CALL*, RST; 17) = 240 runs; plus `none` at 32 steps and `none` at
k = 6 (128 steps), 20 seeds each.

Outcomes and tests, fixed now. Primary: emergence fraction by `t_rep` (heritable) and
by `t_faith` (faithful); **D1**: `stack-write-only` > `none`, pooled over the two
budgets by a one-sided Cochran–Mantel–Haenszel test at α = 0.05; secondaries per budget
with Holm over the two budgets. **D1b**: `stack-writes` > `none` (direct replication
of Stage B, one-sided). **D2** (restated 2026-10-07 before any Stage D run, after a local seed-1001 run showed
≈ 0.19 zeros within 500 steps with the 22 writing stack opcodes removed — LD (HL),r,
LD (nn),rr and block copies of zero registers also write zeros, so the stack *doubles* the
flood rather than creating it): at step 5,000, zero_frac(`none`) − zero_frac(`stack-write-only`)
≥ 0.10, both reported against the 1/256 random baseline; exploratory: which non-stack
writers carry the residual flood (byte histogram, write counters, interaction census).
**D3**: the fraction of the effect carried by the write side,
(frac(`stack-write-only`) − frac(`none`)) / (frac(`stack-writes`) − frac(`none`)),
is ≥ 0.75, and `stack-read-only` is within 0.1 of `none`; `push` versus
`call-rst-write` says which writer matters (no directional prediction). **D4**: every
heritable L = 9 replicator is LDIR-based with period 3 or 9. **D5** (per-byte-hazard
account of the floor): `none` at 32 steps has a higher emergence fraction than `none`
at 128 steps (one-sided Fisher). The k = 6 arm is exploratory. If D1 fails, the Stage
B observation is reported as an unreplicated single cell.

**Stage E — size-axis controls.** *Specified 2026-10-07 after the review, before any
run. `make_conds.stage_e()` → `conds/stageE.json` (1,230 runs).* Stage B's size axis has
four confounds that could each produce "flat emergence time above L = 16": the
per-byte mutation rate falls as 1/L; the number of 4-byte windows per cell and the
total soup bytes grow with L at fixed 20,000 cells; the Z80 budget per tape byte falls
from 8 (L = 16) to 1.28 (L = 100); and tape parity drives the early stop (even L
phase-lock at q_share ≥ 0.5 and stop at 3.5–5k steps, odd L split into two phase
variants and run to 300k), so "final" states of different L had different ages.
Arms, each for `none` and `stack-write-only`, 128 steps, k = 4 nominal, seeds 101–110,
no early stop, fine early sampling:

- **@nominal** — Stage B's setting without the early stop, at the nine square L, at
  **L ∈ {8, 10, 12, 18, 20, 24, 32, 50}** (headless only; shaders and executors exported
  for them) to break the parity alternation and resolve the floor between 9 and 16, and
  at **L ∈ {3, 5, 6, 7}** to find where life starts (the 3-byte unit `DEC E ; LDIR`
  exists). The three control arms below run at the nine square L only.
- **@mubyte** — per-byte mutation rate held at the L = 16 value: 32·L mutated bytes
  per step instead of 512.
- **@bytes** — total soup held at 320,000 bytes: cells = 320,000/L (80,000 at L = 4 …
  3,200 at L = 100) with the drawn pairs scaled with the cell count.
- **@steps8L** — Z80 budget proportional to length: 8·L steps (32 at L = 4 … 800 at
  L = 100).
- **@musweep** — L = 100, 128 steps, k ∈ {1, 2, 3, 5, 8} (k = 4 is @nominal): the
  maintained period/information of the replicator versus the per-byte mutation rate, the
  error-threshold curve.
- **@budget** — L = 100, k = 4, steps ∈ {32, 64, 256, 1024, 2048} (128 is @nominal):
  faithfulness and whole-tape copiers versus bytes copyable per encounter; prediction:
  faithful fraction rises from ≈ 0 at 128 to ≈ 1 at ≥ 512, and whole-tape copiers with
  cargo appear only at ≥ 512.
- **E6 — within-seed variance:** 10 repeats of seed 1 in three cells (`none` L = 16,
  `stack-writes` L = 16, `none` L = 9) to measure how much of the between-run
  variance is GPU non-determinism; if it matches the between-seed variance, seeds are
  exchangeable replicates. At L = 16 the three control arms coincide with @nominal and
  with the Stage C cells (32·16 = 512 mutations; 320,000/16 = 20,000 cells; 8·16 = 128
  steps): these 100 physically identical conditions are **declared cross-batch replicates**
  and enter the variance analysis, never a pooled table (the analysis asserts that each
  (ablation, L, steps, k, seed, replicate) appears once).
- **E7 — the 32-step · 1/4 block-copy exception** (Stage A: 0/10 vs 10/10 in one of 63
  cells) with 20 new seeds (1001–1020), `none` and `block-copy`.

Predictions: if "flat above 16" is real, the four arms agree on the emergence
fraction and KM median at every L ≥ 16; if the per-byte mutation rate drives it,
@mubyte shows emergence time growing with L; if the lottery (bytes/windows) drives
it, @bytes does; if steps per byte drives faithfulness, @steps8L restores faithful
replication at L = 81–100 at the 128-equivalent budget. Tiling: the first-replicator
period divides 2L in ≥ 90% of runs in every arm (ring arithmetic does not depend on
L, mutation or budget). Floor: the non-square lengths place the floor between L = 9
and L = 16; at L = 12 (3 | 12 and 4 | 24) the 3-byte LDIR unit and the 4-byte stack unit
both tile, so L = 12 should behave like L = 16 rather than like L = 9.

Costs (L40S list price; estimator fitted to the measured per-L rates of Stages A/B plus
host time per sample) are quoted from the preflight logs in the change log. Every stage is
preflighted locally (`preflight.py`: tests, condition integrity, no foreign runs in the
volume directory, a dry run of every distinct parameter signature through the exact Modal
code path) and smoke-tested on Modal (2 short runs + 1 invalid condition) before launch.

**Stage F — ring arithmetic (specified 2026-10-07, before any run; instrument built and
tested the same day).** The pair memory can now be padded to P ≥ 2L bytes: addresses wrap
modulo P, the padding is zero and never written back, and SP aliases the last byte of B
under the ring modulus so the stack family is untouched (verified: `LD BC,nn ; PUSH BC`
scores gen2 0.51–0.65 on P ∈ {32, 33, 34, 35, 37}; the offset-4 shift-copier is heritable on
P = 32 only). `make_conds.stage_f()` → `conds/stageF.json`: `none` and `stack-write-only` at
128 steps, k = 4, seeds 101–110, L = 16 with P ∈ {33 = 3·11, 34 = 2·17, 35 = 5·7, 37 prime}
and L = 36 with P ∈ {73 prime, 74 = 2·37, 75 = 3·5², 79 prime}; the P = 2L rows are Stage E's
@nominal runs (160 runs, ≈ $12).
Predictions. **F1 (divisor law):** the period of every tiled first replicator under
`stack-write-only` divides P (not L): periods {3, 11} at P = 33, {2, 17} at P = 34, {5, 7} at
P = 35; {2, 37} at P = 74; {3, 5, 15, 25} at P = 75. **F2 (prime ring):** at P = 37 and P = 73/79
no tiled LDIR replicator appears; emergence under `stack-write-only` falls to the BC-limited
exact-copier route (`LD DE,L ; LD BC,L ; LDIR`-type designs, three register set-ups instead of
one) and is therefore at least 10× slower by KM median than at P = 2L, or absent within 300k
steps. **F3 (control):** under the full ISA the Load–Push family emerges with the same KM
median (within one 50-step sample) at every P. If F1 fails, the gcd mechanism verified in the
executor does not govern the population dynamics and the "ring arithmetic" claim is dropped.

**F4 — is the L = 9–12 dead zone ring arithmetic? (added 2026-10-07 after Stage E was read
and before any F run; 50 runs, `none` only, 128 steps, k = 4, seeds 101–110.)** Stage E found
the pusher regime at L = 8 and L ≥ 16 but not at L = 9–12, and the isolated assay found the
pusher's own heritability (gen2, 64 random partners, 128 steps) marginal there. An executor
probe on padded rings (`runs/unit_fitness_vs_ring.csv`, run before this paragraph was written)
gives, for `01 c5` tiled to L: **L = 12: 0.20 at P = 24, 0.60 at P = 28, 0.45 at 32, 0.58 at 36,
0.53 at 48**; L = 10: 0.31 at P = 20, 0.28 / 0.36 / 0.29 / 0.37 at P = 24 / 28 / 32 / 40 (no
rescue); L = 9: 0.32 at 18, 0.44 / 0.33 / 0.36 at 20 / 24 / 32; L = 8: 0.53 at 16, 0.62 at 20,
0.55 at 24, **0.28 at 32**; L = 16: 0.51–0.61 at every P ∈ {32 … 48}. Cells: `none` at L = 12
with P ∈ {28, 36}, at L = 10 with P ∈ {28, 40}, and at L = 8 with P = 32 (the P = 2L rows are
Stage E's @nominal runs). Predictions, population level: **F4a** at L = 12 · P ∈ {28, 36} the
pusher emerges as at L = 16 — heritable in ≥ 9/10 seeds, KM median < 2,000 steps, modal
first-replicator period 2 (Stage E at P = 24: 6/10, KM 245,500, all LDIR); **F4b** at L = 10 ·
P ∈ {28, 40} no rescue — ≤ 6/10 heritable or KM median > 50,000 (Stage E: 4/10, median not
reached); **F4c** at L = 8 · P = 32 the pusher is impaired — fewer than 10/10 heritable or KM
median > 2,000 (Stage E at P = 16: 10/10, KM 500). If F4a holds and F4b holds, the dead zone
is a property of the stack-pointer wrap on the ring and not of the tape length as such; if F4a
fails, the isolated-assay heritability does not predict population emergence and the
copy-geometry explanation is dropped. Executor-level note for F1 logged before the run: on
P = 34 the isolated tiled units `04 5e ed b0` and `1d ed b0` are heritable (gen2 0.62, 0.68)
although 4 ∤ 34 and 3 ∤ 34; F1 (emergent periods divide P) stands as written and this probe
is the first thing to check if it fails. Cost: ≈ 1.3 GPU-h ≈ $3 on top of F's ≈ $12.

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
- Anything that looks like a new mechanism is confirmed by assaying its tape with
  the single-pair executor and by its recurrence across seeds. Re-running a seed is
  NOT a replication: the GPU's atomic claim order and a non-atomic mutation write make
  trajectories diverge within a few hundred steps even for the same seed (four local
  seed-1 runs reached different dominant tapes); each run is one sample.

## Change log

- 2026-10-07 (Stage E read; 1,230/1,230 runs, 0 failures): **E-flat**: emergence is 10/10
  at every L ≥ 16 in all four arms (one 7/10 cell, @mubyte L = 81); the emergence time's
  mild growth with L (×8 under @nominal and @bytes, ×20 under @mubyte) vanishes under
  @steps8L (150 steps at every L) — a budget-per-byte effect, not mutation rate or lottery.
  **E-steps confirmed** (faithful 3/10 → 10/10 at L = 81 and 1/10 → 9/10 at L = 100 under
  stack-write-only). **E-tiling confirmed** in 8/8 arms (divides-2L 0.91–1.00). **E-floor not
  supported**: L = 8 is a pusher cell (10/10 at KM 500) while L = 9, 10, 12 are not (4, 4,
  6/10); L = 5 and 6 emerge slowly through whole-tape / LDIR copiers; L = 3, 4, 7 never.
  Post hoc: an isolated-assay sweep shows the pusher's own heritability dips to 0.20–0.32 at
  L = 9–12 and is 0.53–0.72 at L = 8 and L ≥ 16 — a copy-geometry explanation to be tested
  with ring variants. **E4**: faithfulness at L = 100 switches on between 128 and 256 steps;
  no whole-tape first replicators at any budget; at 64 steps 6/10 soups are heritable
  populations without any clone (top share ≤ 0.07%, 69–100% of random cells heritable) —
  the `t_rep` event is a clonality criterion and misses them. **E5 confirmed** for the LDIR
  regime (median period 5 → 32.5 bytes over two decades of mutation rate). **E6**: same-seed
  repeats diverge (0.25–0.38 dex vs 0.27–0.46 dex across seeds): seeds are exchangeable,
  single runs are not reproducible. **E7 confirmed** (18/20 vs 0/20, p = 1.7 × 10⁻⁹).
  Details: results/stageE/FINDINGS.md; generated numbers in stage_e/NUMBERS_E.md.
- 2026-10-07 (Stage C read; 490/490 runs, 0 failures): **C1a confirmed** (no-copy and
  rmw-only 0/40 at 1,000,000 steps; 95% upper bound 0.072 per run), **C1b not met** (all-ld
  2/10, not > 0.3). **C3a**: stack-write-only reproduces the stack-writes suppression (8/10
  vs 8/10, both 10/10 slower than none, p = 0.002) but the KM ratio is ×2.4, outside the
  pre-registered ×2 (paired test cannot separate the arms). **C3b confirmed** (read-only
  5/5 vs none). **C3c half met**: ex-sp-only ×3.5 (yes) but push-only ×992 — removing
  PUSH/POP alone delays more than removing all 22 stack writers. **C3d confirmed**: ld-imm
  2/10 heritable, the strongest single-family suppression at 128 steps (the first replicator
  in every permissive arm is `LD rr,nn ; PUSH rr`, which needs the immediate). **C3e
  confirmed** (cb-page, ed-loads within paired noise). **C5** met for the first replicator
  only (period 2 in 9/10 at L = 100; final dominants period 10 in 7/8). At 32 steps · 1/4 no
  arm differs from none (every p ≥ 0.34). Post hoc: mutation rate selects the family holding
  the soup at 300k (LDIR at 1/4 and ≤ 128 steps, the EX (SP),HL family at 1/16–1/64);
  succession increases heritability (gen2 0.59 → 1.00 at L = 16, 0.64 → 0.83 at 36,
  0.72 → 0.99 at 100) with byte-identical successors in 7–9/10 seeds. C4 not yet computed.
  Details: results/stageC/FINDINGS.md; generated numbers in stage_c/NUMBERS_C.md.
- 2026-10-07 (Stage D read; 280/280 runs): **D1 confirmed** (stack-write-only > none,
  CMH one-sided p = 3.9 × 10⁻⁶), **D1b replicated** (p = 1.2 × 10⁻⁵), **D2 confirmed**
  (+0.14 zero fraction at step 5,000; read-only arm 0.00), **D3** first clause confirmed
  (ratio 1.0–1.08), second clause not met at 512 steps (read-only +0.25, p = 0.095),
  **D4 confirmed** (periods 3 × 119, 9 × 22, 6 × 1; all LDIR), **D5 not supported** (4/20 vs
  6/20). Post hoc observation, logged for Stage C/E: removing CALL/RST alone recovers most
  of the effect (CMH p = 4.5 × 10⁻⁴) while reducing the flood no more than removing PUSH
  does (p = 0.033), so the suppression is attributed to the return-address writers rather
  than to the zero load as such; this is a hypothesis for the census analysis, not a result.
  Details: results/stageD/FINDINGS.md.
- 2026-10-07 (ring-length instrument): `createSimShader`/`createZ80TestShader` take a
  `memLength` (compile-time `MEM_LENGTH`, default 2L → identical behaviour and identical
  executor scores to before); the stack-pointer reset now aliases the end of B under the
  ring modulus (with the default ring this is the same value as before). Exported ring
  variants and Stage F conditions (160 runs) added; not launched.
- 2026-10-07 (second pre-launch review, before any C/D/E run; REVIEW.md §8):
  **superseded Stage D runs archived** (seeds 1–12 + 8 orphans → `runs/stageD_v1_seeds1-20`,
  removed from the volume; every analysis script now selects only summaries whose stem
  is in the stage's condition file). **Recording:** explicit early samples (1…34), full
  snapshots on a log schedule with an interaction census, per-sample write counters,
  16 random tapes, top-10 exemplars, copy offset recorded by the assay (direct test of
  period = gcd(offset, 2L)). **Conditions:** C3 gains `stack-writes` and `none` at
  (32, k=2) (490 runs); E gains the L = 100 mutation and budget sweeps and L ∈ {3,5,6,7}
  (1,230 runs); @bytes grids rounded to the nearest cell count; the L = 16 duplicates
  declared as replicates. **Outcomes:** D2 restated as a contrast (the flood is not
  stack-specific); time expressed in cumulative active interactions per cell;
  `t_faith` censoring by the Stage A/B early stop is reported (4 runs, 3 of them in the
  stack-writes L = 100 · 128 cell). **Analysis corrections:** tiled-only divisor
  statistics (an untiled tape's period trivially divides 2L; six faithful whole-tape
  copiers exist at L = 49–81 and are now a class of their own), Fisher two-sided test
  with a relative tolerance, tolerant period scanning up to L − 1, zoo units by
  per-position majority vote. **Infrastructure:** Modal timeout 1 h with one retry,
  brotli pinned to the local version, tracked/untracked dirtiness recorded separately, a
  `--smoke` launcher mode, the preflight estimator fitted to measured rates with host time.
  Preflight (tests + integrity + volume check + dry run of every distinct signature) passed for
  all three stages; estimates at L40S list price: Stage C 24.7 GPU-h ≈ $48, Stage D 17.8 GPU-h
  ≈ $35, Stage E 69.6 GPU-h ≈ $136 (the L = 100 budget sweep's 1,024/2,048-step runs and the
  mutation sweep account for ≈ $40 of E).
- 2026-10-07 (after the four-part review, before any Stage C/D/E run): **Stages
  C, D and E re-specified** (see the Design section): new seeds for every stage
  (Stage B's seeds generated the hypotheses); fine early sampling (50 steps to
  5,000) because 25–36% of A/B emergence times were exactly the first 500-step
  sample; byte histogram, zero fraction and active-pair count recorded per
  sample; the Stage A arm `stack-writes` (46 opcodes, of which 24 write nothing)
  split into `stack-write-only` (22) and `stack-read-only` (24); Stage D arms
  `push` and `call-rst-write`; Stage E added as the size-axis control set
  (per-byte-constant mutation, constant total bytes, steps ∝ L, non-square L,
  within-seed variance, the 32-step exception). Stage A/B conclusions are
  classified as **exploratory under an outcome (t_rep) adopted after 19 runs were
  read**; the pre-registered primary tq_10 fires on zero-byte floods (no-copy
  L = 36/512: 10/10 crossings without a replicator), so by its letter H3 is
  refuted and H1/H6's time ratios are unresolvable at 500-step resolution.
  Confirmatory claims will come only from C/D/E with the outcomes fixed here:
  heritable (`t_rep`), faithful (`t_faith` = gen2 ≥ 0.3 and ≥ 50% of partners became
  ≥ 75% copies), KM medians censored at the last step, Wilson intervals, exact
  Fisher/CMH for fractions and seed-paired sign tests for within-seed contrasts.
  Analysis changes made at the same time (all post hoc, all applied to A/B):
  mechanism labels under the run's suppression set; shift-invariant occupancy
  `q_shift_share`; `gen2_cond` (heritability given a copy was made);
  `self_preserved_as_B` (vulnerability); in-situ assays return NaN, never 0, when
  no partner has headroom; tolerant period; final-state tables at fixed steps
  (5k/50k/300k) rather than at the early-stop step; the zoo's "final" design taken
  from the in-situ winner. Numbers quoted in findings are generated by
  `findings.py` (NUMBERS.md) and never typed.
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
- 2026-10-07 (Stage B assays recomputed and read; the first Stage C relaunch was
  stopped after 5 results and superseded by the re-specified design, so no old-seed
  Stage C data exist). Three post hoc additions, all reported next to the pre-registered
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
