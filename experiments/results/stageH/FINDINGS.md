# Stage H — does space sustain the open phase? (pre-registered 2026-10-08 night; run and scored the same night)

Design: 10 worlds at L = 16 with well-mixed pairing (the partner drawn uniformly from the soup instead of one of the four
lattice neighbours; everything else identical to Stage G L = 16), seeds 3001–3010, local GPU, 1,455 s in all. Control: Stage G
L = 16, 20 worlds. Scoring: `stage_h_analysis.py` → `NUMBERS_H.md`, `predictions.csv`, `per_world.csv`, `stageH_measures.*`;
descriptive kin census: `kin_census.py` → `KIN_CENSUS.md`, `kin_census.csv`. Pipeline: `stage_h_pipeline.sh` (assays, c4
census, per-world table, standard chain, publication).

## Verdicts (pre-registered thresholds)

| | statement | value | outcome |
|---|---|---|---|
| H0 | pusher first with t_rep in 100–5,000 in ≥ 7/10 | 8/10 (the other two at t_rep = 50) | met |
| strong form | ≥ 4/10 worlds without a heritable replicator by 300,000 steps | 0/10 | not triggered |
| H1 | h_open (median heritable fraction, steps 2,000–20,000) lower when mixed: median < 0.125 and one-sided P < 0.05 | median 0.125 (IQR 0.125–0.562); P_perm 0.78 | **killed** |
| H2 | closure sooner when mixed: t_close70 median < 30,000 and one-sided P < 0.05 | median 50,000; reached in 7/10 (lattice 20/20); P_perm 0.77 | **killed** |

The kin hypothesis as pre-registered is dead at L = 16: the open phase is exactly as heritable among strangers as among
kin, and closure is not hastened by mixing. As pre-registered, the paper gains a null: the open → closed order of events does
not depend on spatial structure. Every mixed world began with the pusher (`01 c5` × 8 in 9, `11 d5` × 8 in 1), and the 7 that
closed did so with the same design family as the lattice worlds (`RET NZ`/`RET PO` tilings `b5|bd e3 21 e3 21 c0|e0 …` in 6,
an `LDIR`-bearing tape in 1), copying 1.00 of partners with 0.00 self-damage.

## What differed (descriptive, not scored)

- **Timing of the first replicator**: t_rep median 125 (50–300) against 525 on the lattice; tq_10 100–350 against ~1,150.
  Mixed, the pusher spreads exponentially rather than as a wave.
- **Timing of closure is bimodal when mixed**: three worlds closed by step 100, 500 and 5,000 (heritability ≥ 0.7 almost at
  once), four between 30,000 and 100,000, three not by 300,000 (final heritability 0.44–0.63, the pusher still dominant at
  22–28% exact share with 16–17% zero bytes). The lattice is tight: 10,000–200,000, median 30,000, 20/20.
- **Kin census** (encounters of the first tape at the open-phase snapshots, 2,000–20,000): the partner is kin (Hamming ≤ 2)
  in 57% of encounters on the lattice and 11% when mixed. Against kin the pusher is damaged in 2.5% (lattice) and 0% (mixed) of
  encounters and copies into 98–100%; against strangers it is damaged in 34% and 31% and copies into 60% and 54%. The
  mechanism behind the hypothesis is therefore real (kin encounters are harmless, stranger encounters are not), but its
  population consequence is not: the pusher population is as heritable with 11% kin as with 57%.

## Reading (post hoc, labelled)

Space is not what sustains the open phase. What space seems to change is the reliability, not the possibility, of closure:
on the lattice the pusher's slow wave leaves persistent frontiers and tar pockets, and a closed mutant arising anywhere
grows as a compact domain of exact copies; mixed, the sweep is over in a few hundred steps and whether a closed mutant is
present during or soon after it appears to decide between immediate closure and a long open reign. This is a hypothesis
for a follow-up (more worlds; the arrival time of the first closed tape against the sweep time), not a result.

## Caveats

Ten worlds against twenty; one tape length; same-seed GPU non-determinism (population claims over seeds); t_close70 is read
on the sampled steps (50 … 300,000) and censored worlds enter the rank test at the horizon; the kin census uses exact
matches of the first tape and no shift tolerance for copies, so its copy rates are lower bounds.
