# Stage I — lethal tar in the first machine (pre-registered 2026-10-08 night; run and scored the same night)

Design: 10 worlds at L = 16 under the `zero_halts` rule (a zero byte fetched as an opcode halts the pair for the rest of the
encounter, as an unmatched bracket halts BFF), everything else identical to Stage G L = 16; seeds 4001–4010; local GPU,
1,517 s. Culture tests, partner tests and the census ran under the same rule (`z80_test_lethal_L16.wgsl`). Scoring:
`stage_i_analysis.py` → `NUMBERS_I.md`, `predictions.csv`, `per_world.csv`; pipeline `stage_i_pipeline.sh`.

## Verdicts (pre-registered thresholds)

| | statement | value | outcome |
|---|---|---|---|
| I1 | zero fraction ≥ 0.15 at step 500 in ≥ 7/10 | 10/10 (median 0.183; benign tar 0.37) | met |
| strong form | ≥ 7/10 worlds without a heritable replicator by 300,000 steps | 0/10 | not triggered |
| I2 | "open, then extinct" in ≥ 5/10 | 0/10 | not met |
| I3 | closed dominant by 300,000 steps in ≤ 6/10 (kill ≥ 9/10) | 10/10 | **killed** |
| I4 | closed dominants contain no zero byte | 10/10 | descriptive, holds |

## What happened

The open beginning never happens. No literal pusher establishes in any world (a pusher tape appears in a top-ten
exemplar list 3 times across all 6,980 samples of the ten worlds, never as a heritable class), because its execution
runs into partners that are 18% zero bytes from the first steps and halts there. Life comes anyway, late and born closed:
the first heritable tape in every world is a block-copy replicator carrying `LDIR` or `LDDR` (10/10 `first_has_block`),
copying 1.00 of 256 random partners with 0.00 self-damage from its first appearance, containing no zero byte, at a median
t_rep of 46,250 steps (range 20,000–219,000) against 525 under benign tar: ninety times later. The quasispecies share never
reaches 10% (variants differ in the bytes that do not matter) while the heritable fraction of random cells rises from 0 at
20,000 steps to 0.41 at 30,000, 0.88 at 50,000 and 0.94 thereafter (median over worlds).

## Reading

I2 and I3 were written for a world in which the open pusher establishes and then dies; lethal tar did something
stronger: it removed the open route before it could be taken, so there was no open phase to end. The window model's
reading survives in its limiting form (∫ n dt ≈ 0 for the open form), and the classification gains a third outcome for the
lethal column: where a closed design exists in the instruction set (the Z80's block copier), life begins closed and late;
where none exists (BFF), the open beginning that lethal tar permits briefly ends in extinction. The ninety-fold delay is
the theorem's "exponentially rarer" made visible: without the literal road, life must wait for a closed design. The same
regime appears when the stack writers are removed (Stage C/D: the `LDIR` regime, 160- to 992-fold later), so three
different blocks of the literal road, no literal channel (BFF), no stack writers, lethal zeros, lead to the same place.

## Caveats

Ten worlds, one tape length; the first-tape census of pushers uses exact period-2 matches; the kill of I3 is by the
letter of a prediction that assumed an open phase; the new cell of the classification rests on one machine and one rule.
