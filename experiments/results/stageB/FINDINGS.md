# Stage B — organism size (L = 4 … 100), findings

640 runs on Modal L40S: L ∈ {4, 9, 25, 36, 49, 64, 81, 100} × {none, block-copy, stack-writes, no-copy} × {128, 512} Z80 steps × mutation 1/16 × 10 seeds, horizon 300k steps (L = 16 comes from Stage A). 20,000 cells, 8,192 pairs/step. Pre-registration: `experiments/PLAN.md` (H6). Tables: `cells.csv`, `assays.csv`, `succession.csv`, `zoo.csv`, `size_axis/`. Auto-generated report: `../REPORT_stageB.md`.

Emergence below means a heritable replicator by the assay (`t_rep`: a top-3 exemplar with share ≥ 0.5% whose offspring copy, gen2 ≥ 0.3). The occupancy measure `tq_10` is in the tables.

## 1. Size does not slow emergence; it has a floor

| L | unablated, 128 steps | unablated, 512 steps | stack writes removed, 128 | stack writes removed, 512 |
|---|---|---|---|---|
| 4 | 0/10 | 1/10 (50,500) | 0/10 | 0/10 |
| 9 | 4/10 (34,750) | 1/10 (133,500) | 9/10 (92,000) | 10/10 (98,250) |
| 16 (Stage A) | 10/10 (500) | 10/10 (1,000) | 10/10 (50,500) | 10/10 (60,250) |
| 25 | 10/10 (1,500) | 10/10 (500) | 10/10 (11,750) | 10/10 (27,250) |
| 36 | 10/10 (500) | 10/10 (500) | 10/10 (18,000) | 10/10 (26,750) |
| 49 | 10/10 (1,500) | 10/10 (500) | 6/10 (16,250) | 10/10 (52,500) |
| 64 | 10/10 (1,000) | 10/10 (500) | 10/10 (26,750) | 10/10 (19,250) |
| 81 | 10/10 (1,500) | 10/10 (500) | 7/10 (18,500) | 9/10 (6,500) |
| 100 | 10/10 (1,500) | 10/10 (500) | 10/10 (16,500) | 10/10 (5,250) |

(seeds with a replicator / 10, median first-replicator step in parentheses; block-copy removal is indistinguishable from unablated at L ≥ 25 and kills L = 9 entirely: 0/10, 0/10.)

- From L = 16 to L = 100 the time to the first heritable replicator is flat: 500–1,500 steps at 128 steps, 500 at 512 steps. A 100-byte organism replicates as fast as a 16-byte one. **H6's first clause (emergence time grows with L) is false.**
- Below 16 bytes there is a floor. L = 4 produced one replicator in 80 runs. L = 9 is marginal, and it is the only size where the *ablations change direction* (section 3).
- no-copy (112 instructions removed): 0/160 at every L. Rmw-only was not run here (0/90 at L = 16 in Stage A).

## 2. Large organisms are tiled, not complex

- Unablated and block-copy soups at every L ≥ 25 are taken over by the same **Load–Push** family as at L = 16 (`01 c5` = `LD BC,nn ; PUSH BC`, `21 e3` = `LD HL,nn ; EX (SP),HL`), census takeover at 1.5–3k steps in 10/10 seeds, with occasional later LDIR invasions (128 steps: 1/10 at L = 49 and 81, 5/10 at L = 100; 512 steps: 2/10 at L = 25, 3/10 at L = 36).
- The dominant tapes have **minimal period 2** in 20/20 runs at L = 25, 36, 49, 64 and in 18–20/20 at L = 81, 100 (the exceptions are the same period-2 tape with a 3-byte end defect). A 100-byte organism is fifty copies of a 2-byte instruction pair. Every byte is code; there is no room for cargo because a stack replicator *is* its own payload: what it pushes is what it is.
- With stack writes removed the replicators are LDIR shift-copies whose period **divides L** in 75–100% of runs: 3 | 9 (`1d ed b0` = `DEC E ; LDIR`), 5 | 25, 7 | 49, 9 and 27 | 81, 4/8/16/32 | 64. Copying a tape onto itself at a fixed offset makes the offset the period, so even content-agnostic copiers end up tiled. 10–20% of LDIR runs at L ≥ 49 are aperiodic: a ~5-byte LDIR core carrying 40–90 bytes of hitchhiking junk — the only "free tape" seen anywhere, and it is neutral cargo, not function.
- High-order entropy of the final soups: 1.0–1.9 bits/byte under Load–Push, 3.5–5.7 under LDIR, at every L.
- Mechanism classes are the same two at every L ≥ 16; L = 4 and 9 have fewer. **H6's complexity clause (more mechanism classes, longer coexistence tail at large L) is not supported.** The size axis changes nothing qualitative above 16 bytes except the number of repeats.
- Functional fraction of the final population (random cells assayed in situ): LDIR soups 0.9–1.0 at every L; Load–Push soups 0.16–0.59, the rest being sterile pushed debris, as at L = 16.

## 3. At L = 9 the stack is a sterile zero flood, and removing it *helps* (post hoc)

- Unablated L = 9 soups: 34% of all bytes are `00` (26% at L = 4), byte entropy 6.0 bits vs 7.4 when the stack-writing families are removed. The zeros come from `PUSH rr` with still-zero registers and from CALL/RST return addresses (`xx 00`), none of which copies the code that writes them. The dominant tapes are `00 00 …` and `00 39 00 39` (NOP; ADD HL,SP). No stack replicator forms: the 4-byte Load–Push unit and the period-2 pairs do not tile 9.
- With stack writes removed, the period-3 `DEC E ; LDIR` replicator tiles 9 exactly and emerges in 9/10 and 10/10 seeds (medians 92k and 98k steps), versus 4/10 and 1/10 unablated. Fisher exact p = 0.057 (128 steps) and 0.0001 (512 steps), n = 10 per arm.
- This is the opposite of the Stage A result at L = 16, where removing stack writes delays emergence 20–100×. It was not pre-registered and is one cell of the design; it is logged in PLAN as a hypothesis for a confirmatory run (Stage D: L = 9 × {none, stack-writes, push-only, call-rst} × {128, 512} × 20 seeds, ≈ 9 GPU-h).

## 4. A heritable but unfaithful replicator (methodological)

In 3/10 no-copy runs at L = 100 (128 steps) the final top exemplar is a `CALL` chain `cd xx cd xx …` at 0.2% share. Each CALL pushes its 16-bit return address; because PC runs in the unreduced 16-bit space, after `CALL $xxCD` every return address has high byte `$CD`, the CALL opcode itself. The neighbour is flooded with `cd yy` pairs (50% similar to the parent), those offspring do the same (gen2 0.49), but the operand drifts by +4 each generation and the lineage dies by generation 3. It passes the pre-registered gen2 ≥ 0.3 rule and fails `offspring_within_q` (0 partners became ≥ 75% copies). Tables now also report the latter, and a replicator is called *faithful* when ≥ 0.5 of its partners become copies. No other Stage A/B call changes; no-copy remains null.

## 5. Caveats

- The horizon is 300k steps. The L = 9 unablated cells and the L = 49/81 stack-writes cells are right-censored; 0/10 means "not within 300k steps".
- The in-situ assay returns 0 by construction in a soup saturated with copies (no partner with prior similarity < 0.75), so `final_rep_insitu` is a lower bound in Load–Push monocultures (e.g. L = 81, 128 steps: 1/10 in situ, 10/10 against random partners).
- One mutation rate (1/16) was run at L ≠ 16. Stage C adds mutation and finer ablations at L = 16 and the no-stop succession at L = 36 and 100.
