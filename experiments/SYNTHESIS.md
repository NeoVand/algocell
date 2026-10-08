# Synthesis across Stages A–E (2026-10-07) — what the paper can claim, and on what

Every number below is copied from a FINDINGS document or a generated NUMBERS file (`results/stage{A,B,C,D,E}/`); nothing is typed from memory. Confidence is graded **A** (pre-registered, replicated with independent seeds, p ≤ 10⁻³), **B** (pre-registered, one stage, or post hoc with p ≤ 0.01), **C** (post hoc or descriptive). Stage F (ring arithmetic, F1–F4) is running and is not used here.

## 1. The system and the instrument

20,000 cells hold L-byte tapes (L = 3 … 100). Each step draws 8,192 random pairs; about 48% survive the parallel claim (≈ 0.395 active interactions per cell per step, recorded). A pair's two tapes are concatenated into a 2L-byte ring (padding to P ≥ 2L in Stage F) and executed as one Z80 program for S steps (32 … 2,048), starting at the first byte of A with the stack pointer aliasing the last byte of B. 8,192 / 2ᵏ random bytes are mutated per step. There is no fitness function. The instrument: a prefix-aware opcode suppression that removes instruction families (verified against a reference Z80 with 0 divergences), an **assay** that executes a tape against 64 random partners and scores first-generation copying, second-generation heritability (gen2 ≥ 0.3 = heritable) and faithfulness (≥ 50% of partners become ≥ 75% copies), an **interaction census** of every active pair at log-spaced snapshots, byte-level recording (zero fraction, byte histogram, write counters) at explicit early steps, and full soup snapshots. Stages A (630 runs), B (640), C (490), D (280), E (1,230): 3,270 runs, all pre-registered in PLAN.md before being read; seeds 1–10 for A/B, 101–110 for C/E, 1001–1020 for D and E7.

## 2. Claims

**2.1 Self-replication emerges in every seed of the unablated system** — at every (steps, mutation) cell of Stage A at L = 16 (10/10 in all cells), at every L ≥ 16 in Stage B and in all four Stage E control arms (10/10 except one 7/10 cell), typically within 200–2,000 steps at 128 steps (≈ 80–800 encounters per cell). Confidence **A**. Figures: `stageE/stage_e/E1_fraction`, `stageC/stage_c/C3_forest_st128_k4`.

**2.2 The first replicator is a 2-byte instruction: `LD rr,nn ; PUSH rr`** — the 16-bit immediate *is* the unit, so each PUSH writes one more copy below the stack pointer, into the partner's tail. First in 10/10 runs of `none`, `block-copy`, `ld-mem` (C), modal period 2 at L = 8 and at every L ≥ 16 (E), byte-identical `01 c5 ×8` in 8/10 `none` runs. Confidence **A** (descriptive, every stage). Figure: zoo tables; `stageE/stage_e/E3_period`.

**2.3 Only three parts of the instruction set carry emergence at 128 steps** — (i) the 16-bit immediate loads (`ld-imm`: 2/10 heritable in 300k steps, 10/10 seeds slower than `none`, p = 0.002), (ii) the stack writers (`push-only` ×992, `stack-write-only` ×382, `stack-writes` ×160 in KM median; each 10/10 slower, p = 0.002), (iii) copying itself (`no-copy`, `rmw-only`: 0/40 runs at 1,000,000 steps; 95% upper bound 7% per run). Every other family — register loads, memory loads, the ED-page loads, the CB page, CALL/RET/RST, EX (SP),HL alone, the stack readers, block copy — is within the seed-paired noise of `none` (all p ≥ 0.18). Confidence **A** for (i)–(iii) individually (C3, with D for the stack), **B** for the "nothing else matters" statement (n = 10 per arm). Figure: `C3_forest_st128_k4`.

**2.4 The write side of the stack carries the suppression; the read side carries nothing** — `stack-write-only` (22 opcodes) reproduces `stack-writes` (46) at L = 16 (8/10 vs 8/10; paired 5/4/1) and `stack-read-only` is indistinguishable from `none` (5 slower / 5 faster); at L = 9 the write-only arm beats `none` with CMH p = 3.9 × 10⁻⁶ and the read-only arm does not (p = 0.054). Confidence **A**.

**2.5 Removing instructions can make life emerge more often: the L = 9 reversal** — `stack-write-only` vs `none` at L = 9: 12/20 vs 6/20 (128 steps), 19/20 vs 5/20 (512 steps), CMH p = 3.9 × 10⁻⁶, replicated the Stage B post hoc observation with 20 new seeds. The return-address writers CALL/RST carry most of it (`call-rst-write` 12/20, 14/20, p = 4.5 × 10⁻⁴) while reducing the zero flood no more than removing PUSH does, so the mechanism is not the zero load as such. Confidence **A** (D1, D1b), **B** for the attribution to CALL/RST.

**2.6 The replacement principle** — when the fast mechanism is removed, a slower one takes over: without PUSH the first replicator is an LDIR unit of period 4 or 8, 100–1,000× later, and the delay is longest when the other stack writers remain (`push-only` ×992 > `stack-write-only` ×382 ≈ `stack-writes` ×160). Without immediate loads the same LDIR units appear at 168k and 205k steps in 2/10 runs. Confidence **B** (C3; the ordering push-only > write-only is one paired comparison, 10/0 vs 10/0 against `none`, not tested against each other pre hoc).

**2.7 Above L = 16 size does not matter for whether life emerges, and it matters for when only through the budget per byte** — emergence 10/10 at every L ≥ 16 in @nominal, @mubyte, @bytes and @steps8L; KM medians rise ×8 from L = 16 to 100 under @nominal and @bytes, ×20 under @mubyte (which raises the per-cell mutation load), and are flat (100–200 steps) under @steps8L. Faithfulness at L = 81–100 is restored by scaling the budget (`stack-write-only` 3/10 → 10/10, 1/10 → 9/10). Confidence **A** (E-flat, E-steps; pre-registered, four arms).

**2.8 There is no size floor; there is a dead zone** — `none` emerges 10/10 at L = 8 (KM 500, the pusher), 4/10, 4/10, 6/10 at L = 9, 10, 12 (LDIR units, floods in the rest), 9/10 at L = 5 and 6 (whole-tape and period-3 copiers, slow), 0/10 at L = 3, 4, 7. The pusher's own heritability in isolation is 0.53 at L = 8, 0.32 / 0.31 / 0.20 at L = 9 / 10 / 12, 0.59 at L = 16 and 0.43–0.72 above. Stage F4 (pre-registered) switched the dead zone with memory padding: at L = 12 four or twelve padding bytes give 10/10 pusher emergence at 350–600 steps (vs 6/10 at 245,500), at L = 8 sixteen padding bytes destroy the pusher cell (10/10 → 2/10), at L = 10 eight bytes rescue emergence (10/10 at 500) but leave heritability at the threshold. Mechanism (executor trace): a race between the stack pointer writing B from its last byte downward and the program counter executing B forward; a `LD rr,nn` that straddles a not-yet-written byte poisons every later PUSH; padding shifts the phase at the wrap. Confidence **A** for the dead zone and for its geometric (not size-law) nature; **B** for the race mechanism (demonstrated in single traces; single-encounter averages do not predict the L = 10 margin).

**2.9 Replicators tile their tapes with periods that divide 2L, the pair length — not the memory ring** — in Stage B at every L ≥ 25, in Stage C (all 33 cells) and in all 8 Stage E arms (divides-2L 0.91–1.00 of tiled first replicators). The statement is non-trivial for the LDIR regime (periods 3–50 under `stack-write-only`, of which only 81–93% divide L; e.g. period 8 at L = 36 and L = 100) and trivial for the period-2 pusher. Stage F falsified the pre-registered ring reading: on memory rings padded to P > 2L the emergent periods divide P in 0.00 of cases and keep dividing 2L (0.50–1.00), because only the two tapes are inherited. Confidence **A** for the law as a description of the pair geometry; the gcd-on-the-ring mechanism is withdrawn; the offset form could not be evaluated (modal offset 0). Figures: `E3_period`, `stageF/stage_f/F1_periods`.

**2.10 Mutation rate selects the winning mechanism; the LDIR regime has an error threshold, the pusher does not** — at 300k steps under `none` the EX (SP),HL family holds the soup at 1/16 and 1/64 (8–9/10) while LDIR displaces everything at 1/4 and ≤ 128 steps (8–9/10, take-over medians ≈ 64,000 steps); at L = 100 under `stack-write-only` the median period of the first replicator is 32.5, 22.5, 15, 8, 5, 5 bytes from 1.6 × 10⁻⁵ to 2.0 × 10⁻³ mutations per byte per step, while the pusher's period stays 2 and its heritable fraction 9–10/10 at every rate. Confidence **A** (E5 pre-registered), **B** (C2 succession, descriptive).

**2.11 Succession selects for heritability, and converges on byte-identical tapes** — the modal first tape has gen2 0.59 / 0.64 / 0.72 at L = 16 / 36 / 100 and the modal final dominant 1.00 / 0.83 / 0.99, writing no more partner bytes but damaging itself less; the successors are byte-identical in 9/10, 7/10, 7/10 independent seeds (a pusher with a DJNZ inserted at L = 36; a JP NZ variant at L = 100). The population is only 7–14% heritable for thousands of steps after the pusher appears and reaches 0.78 at 50k when the successor takes over (C4). Confidence **B** (post hoc, but measured on all seeds; the assay of the tapes is deterministic).

**2.12 Faithful replication needs a budget, and below it the soup replicates without clones** — at L = 100 faithfulness switches on between 128 and 256 steps (`stack-write-only` 1/10 → 10/10); at 64 steps no exemplar ever reaches 0.5% share, yet 6/10 final soups have 69–100% of random cells heritable (gen2 ≥ 0.3) with the top genotype at ≤ 0.07%; at 32 steps nothing. Confidence **A** for the threshold (E4), **B** for the sub-clonal regime (post hoc reading of a pre-registered cell).

**2.13 Occupancy- and pattern-based detectors fail, compression detectors fail where a flood precedes life** — the pre-registered occupancy event `tq_10` fires on the zero flood (10/10 for `ld-imm` with 2/10 replicators; 20/20 for `none` at L = 9 with 6/20) and misses the LDIR cloud (1/10 for `stack-write-only` with 8/10); return-address smears spread a byte pattern without heredity (gen2 0.08). Time-resolved against the C4 heritable fraction (7,960 run×step samples, `results/detectors/`): high-order entropy, the non-unique tape fraction and brotli bits per byte reach AUC 0.97 pooled (0.99–1.00 from 30k steps on at L = 16), occupancy of the dominant genotype stays at 0.60–0.75 at every step, and the interaction-level detector (encounters writing ≥ L bytes) is 0.987 at L = 16 but 0.545 at L = 100. By tape length, high-order entropy is **anti-correlated** with life under `none` at L = 25, 49, 64, 81 (AUC 0.07–0.34) because the zero flood compresses better than the replicating soup, while the non-unique fraction and top share stay ≥ 0.85 at every L ≥ 8. Confidence **A** (every stage; two ground truths).

**2.14 Same-seed runs diverge; seeds are exchangeable** — 10 repeats of one seed spread 0.25–0.38 dex against 0.27–0.46 dex across seeds (E6). Confidence **A**. This must be stated in the paper: the simulation is not reproducible run-by-run, and every claim is a population claim.

**2.15 At 32 steps · 1/4 the only route to life is block copy** — `block-copy` 0/20 vs `none` 18/20 (p = 1.7 × 10⁻⁹, E7, new seeds), although the pusher is viable in isolation at that budget (gen2 0.54). Confidence **A** for the effect; the reason the pusher never appears there is open.

## 3. A narrative for the paper

1. *An instrument, not a demonstration.* Prefix-aware ablation + a heritability assay + census; what pattern-counting gets wrong (2.13).
2. *The atlas.* One figure: the forest plot of 16 ablations at L = 16 (2.3), with the zoo of first replicators; the stack write/read split (2.4); the nulls to 1M steps.
3. *Less is more.* The L = 9 reversal (2.5) and the replacement principle (2.6): instruction sets have load-bearing parts and parasitic parts, and removing the parasitic ones speeds emergence.
4. *Geometry.* The ring law (2.9), the size axis under control (2.7), the dead zone and its ring-arithmetic explanation (2.8, F4), the budget threshold and the sub-clonal regime (2.12).
5. *Evolution after emergence.* Mutation selects the family and the error threshold (2.10); succession toward heritability with convergent byte-identical outcomes (2.11, C4).
6. *Methods statements the field needs.* Same-seed divergence (2.14); the assay definitions; the clock in encounters per cell.

Bridges to biology, stated as analogies and kept to one paragraph each: biosignature failure modes (2.13); repeat structure from copy geometry rather than selection (2.9); Eigen's threshold realised as period-vs-mutation (2.10); the pre-emption of emergence by sterile writers, i.e. a "weed" that occupies the niche (2.5, 2.6); collective replication below the budget for individual copies (2.12).

## 4. Figure list (sources exist; publication styling via `figstyle.py`)

| # | Figure | Source | Claim |
|---|---|---|---|
| 1 | The soup, the ring, the pusher (schematic + one snapshot sequence) | snapshots `t500 … t300000` | §1 |
| 2 | Atlas forest plot, L = 16, 128 steps | `stageC/stage_c/C3_forest_st128_k4` | 2.3, 2.4 |
| 3 | Nulls to 1M steps (KM) | `C1_nulls_km` | 2.3 |
| 4 | L = 9 reversal: write vs read side, CALL/RST | Stage D cells (to draw: forest of D arms) | 2.5 |
| 5 | Size axis under four arms (fraction + KM) | `E1_fraction`, `E2_km_steps` | 2.7, 2.8 |
| 6 | Period vs L, divides 2L | `E3_period` | 2.9 |
| 7 | Isolated heritability of the units vs L (dead zone) + F4 | `E8_unit_fitness`, F4 (pending) | 2.8 |
| 8 | Mutation: family succession + error-threshold curve | `C2_mutation`, `E5_mutation`, census families | 2.10 |
| 9 | Succession toward heritability: first vs final, C4 curves | `first_vs_final_none.csv`, `c4/C4_functional_st128_k4` | 2.11 |
| 10 | Budget at L = 100: threshold and sub-clonal regime | `E4_budget` | 2.12 |
| 11 | Detectors vs ground truth (tq_10 inversions, HOE by L) | benchmark tables | 2.13 |
| S1 | Same-seed divergence | E6 numbers | 2.14 |

## 5. Open before writing

- F1–F4 (running): the ring law at the population level and the dead-zone mechanism.
- Why 32-step soups never find the pusher (2.15): an invasion assay (seed the pusher at 1% and measure take-over) at 32 vs 128 steps, local, no cost.
- Stability of the sub-clonal regime beyond 300k steps (2.12): one 1M-step cell at L = 100 · 64 steps if desired (≈ $3).
- The emptiness of L = 3, 4, 7 (discovery vs maintenance): @mubyte at L = 4 is also 0/10, so not the per-byte rate alone.
- The time-resolved detector benchmark on the C/E snapshots (HOE, unique fraction, species entropy vs the C4 ground truth), zero cost.
- A D-stage forest figure and the final-figure styling pass.

## 6. Spend and provenance

At L40S list price: A ≈ $37, B ≈ $48, C ≈ $48, D ≈ $35, E ≈ $136, F ≈ $16 (running); ≈ $320 in total, plus $0.30 for the aborted first Stage C batch (quarantined, excluded). Every run records shader and ISA hashes, adapter, library versions, git commit and dirtiness; every analysis table is regenerated from the raw summaries by the scripts in this directory.
