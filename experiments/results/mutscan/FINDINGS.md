# Mutational scan of the Stage G first replicators and final dominants (2026-10-08, late night)

Pre-registered in `REVISION_PREREG.md` (M1–M3) before the run; script `mutscan.py`; tables `mutscan_tapes.csv` (160 rows),
`mutscan_sites.csv` (one row per tape × position), per-mutant arrays in `runs/mutscan/`. Every single-byte mutant of every
tape (all 255 values per position at L = 16, 20; 32 seeded values at L = 50, 64) was run as organism A against 32 shared
random partners and its offspring against 32 fresh ones (the culture test's executor and thresholds). *Heritable*: gen2 ≥
0.3. *Transmitted*: the mutant byte is present, at the aligned position, in at least half of the mutant's copies.
*Transmissible site*: a position where at least half of the tested values are both heritable and transmitted. *Capacity*:
Σ_i log₂(1 + 255 × frac_both(i)) bits. All 160 unmutated controls were heritable in the scan's own partner draw.

## Outcomes against the predictions

| | prediction | outcome |
|---|---|---|
| M1 mutational robustness (Cicala et al.'s explanation of the succession) | final more robust than first in ≥ 15 of 20 worlds at L = 16, 20, 50 | **killed**: 2/20, 2/20, 0/20 (and 8/20 at L = 64). Medians first vs final: 0.74 vs 0.59 (L = 16), 0.86 vs 0.82 (L = 20), 0.99 vs 0.95 (L = 50), 0.99 vs 0.99 (L = 64) |
| M2 variation channel: closed finals have more transmissible sites than first replicators | more sites for loop-bearing finals | **killed, reversed**: sites median first vs loop-bearing finals 0 vs 0 (L = 16), 1 vs 0 (L = 20), 23 vs 1 (L = 50), 32 vs 0.5 (L = 64); capacity 32 vs 6, 54 vs 11, 269 vs 42, 325 vs 14 bits |
| M3 matched pair at L = 50 (pusher + JR against the pusher it is made of) | the closed form has more sites | **killed**: 0 of 14 worlds; sites 23 → 1, capacity 272 → 42 bits |

## What the numbers say

1. **The open pusher transmits its mutations; the closed designs do not.** Among heritable single mutants, the mutant
   byte reaches the copies in a median 16% (L = 16), 23% (L = 20), 53% (L = 50) and 62% (L = 64) of cases for the first
   replicator, against 1–4% for the return-based and block-copy closers (the RET NZ design of 17 L = 16 worlds: 1%; the
   pusher + JR of 14 L = 50 worlds: 4%; the LDIR tilings at L = 20 and 64: 0–1%). The pusher's transmissible positions
   are its operand bytes: a changed operand is pushed into the copy and pushed again by the copy. The closers execute
   every byte they copy, so a change is either lethal or repaired.
2. **The exceptions are the first non-coding segments.** Three L = 16 finals and one each at L = 20 and 64 carry a
   block copy (`ed b0`) preceded by a jump or a load that skips a stretch of bytes: those bytes are copied as data and
   never executed, and they are transmissible (10, 10, 4, 11 and 6 sites; transmission 0.34–1.00). Heritable variation
   returns where a genome acquires bytes that are copied but not run.
3. **Mutational robustness does not explain the succession.** The pusher is at least as robust to single mutations as
   its successor at every length (0.99 at L ≥ 50); the RET NZ closer at L = 16 is markedly less robust (0.59 vs 0.74).
   What the closers gain is context independence (Fig. 3a–c), not mutational robustness.

## Reading, and what it changes in the paper

Closure, as it first evolves here, buys fidelity across contexts by executing everything it copies, and in doing so
loses the capacity to inherit variation that the sloppy open copier had. The first closed organisms are canalised: one
string in every context and one string across their own mutants. A channel for inherited variation reopens only when a
genome acquires a segment that is copied as data and skipped as code. This answers the optimistic reviewer's question
("does closure create capacity for innovation?") in the negative for the first closers, and it separates the two
explanations of the succession: not robustness to mutation (killed) but independence from context (measured).
Candidate wording for the paper: "closure trades variation for fidelity". Still to do: a figure (sites and capacity,
first against final, by design), a Methods paragraph, and the invasion test of the matched L = 50 pair.
