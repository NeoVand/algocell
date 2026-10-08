# Analysis A — assembly fires on sterile order (pre-registered in BIOLOGY_PREREG.md, section A; run 2026-10-08)

Code: `biology/assembly.py` (CPU only, no new runs; tests in `biology/test_assembly.py`, 7 passing). Every number below is
in `NUMBERS_ASSEMBLY.md` with the table it came from. Outcomes are stated against the pre-registered predictions and
kill criteria exactly as written; post hoc additions are labelled as such.

## Summary

| | pre-registered prediction | kill | result | verdict |
|---|---|---|---|---|
| P1 a(first tape) ≤ a(tar modal tape before t_rep) | ≥ 80% of 80 worlds | < 50% | 80/80 = 100% (79 equal, 1 strictly less) | prediction met, kill not triggered — but the test has no power (see P1) |
| P2 A_top10 > 10 × baseline at a sample before t_rep | ≥ 75% | < 50% | 71/80 = 88.8% | prediction met, kill not triggered |
| P3 AUC of A_top10 vs `event`, Stage E, L ≥ 25 | ≤ 0.65 | ≥ 0.75 | 0.721 (HOE on the same samples 0.749) | neither: pre-registered grey zone |

The bridge claim (complexity-times-abundance cannot separate the first replicator from the sterile order before it) is
supported by P1 and P2 in the Stage G worlds and is neither confirmed nor killed by the detector test P3: as a detector
of heredity across tape lengths ≥ 25, assembly is no better than the compression biosignature it was compared with.

## P1 — the first replicator is never more complex than the order before it, and could not have been

Registered comparison: a(first_tape) ≤ a(modal tape at the last sample before t_rep) in 80/80 worlds; equal in 79,
strictly less in 1 (L = 16 seed 2003, where the modal tape is `00 41` × 7 with one mismatched byte, a = 7 vs 4).
Two facts make this outcome weaker than it looks and must be read with it:

1. The comparison is degenerate in 59/80 worlds: the modal tape at the last sample before t_rep *is* the first tape.
   t_rep is the first sample at which a top-3 exemplar holding ≥ 0.5% of cells is heritable (`assay_batch.py`,
   SHARE_MIN); the pusher is typically already modal below 0.5% for several samples before that. The 21 non-degenerate
   worlds are the 20 L = 16 worlds (tar `00 41` × 8, NOP / LD B,C, a = 4 = a_first) and one L = 20 world (tar
   `21 e5` × 10, a second pusher, a = 5 = a_first).
2. The direction "≤" was unfalsifiable by construction. Every first tape is a period-2 tape (`01 c5`, `11 d5` or
   `21 e5` repeated), and its Re-Pair index equals the minimum number of joins that can build *any* string of its
   length (brute force over reachable lengths: 4, 5, 7, 6 joins for L = 16, 20, 50, 64; a(first) = 4, 5, 7, 6). No tape
   can be strictly simpler than the first replicator, so P1 could only ever come out as "equal" or "tar more complex".
   The informative content is the 79/80 equalities: the assembly index puts the first replicator and the order before it
   at the same, minimal value.

Post hoc companion (not pre-registered): tar = modal tape at the last sample before the first tape enters the top ten
at all (median arrival step 375; t_rep − arrival median 250 steps). a_first ≤ a_tar in 80/80, equal in 32, strictly
less in 48. The pre-arrival modal tapes are period-2 tapes with one (38) or two (10) mismatched bytes, `00 41` × 8 (18),
`00 29` × 10 (6), `00 01` × 25 (3), all-zero L = 20 (2), and three other period-2 tapes: low-complexity, sterile order.
At L ≥ 20 that order is faint (median modal share 0.10–0.13%, i.e. 20–26 cells of 20,000); at L = 16 it is a real tar
(`00 41`, median share 0.9%).

## P2 — assembly fires before the first heritable replicator in 71/80 worlds

A_top10 exceeds 10 × its step-1 value before t_rep in 71/80 worlds (88.8%; prediction ≥ 75%). Median lead
t_rep / step_first_cross = 19 (min 0.15, max 525); by L: 20/20 at L = 16 (median crossing step 21 vs t_rep 525), 16/20 at
L = 20 (300 vs 800), 19/20 at L = 50 (34 vs 600), 16/20 at L = 64 (3 vs 800). At the last sample before t_rep A_top10
is a median 10 × baseline, at the t_rep sample a median 12.7 ×: the measure reads the same at the tar as at the arrival
of heredity. In the exemplar world (L = 16, seed 2001; figure `assembly_exemplar_L16_s2001`) the crossing is at step 50
(A_top10 = 27.5 against a threshold of 25.6), t_rep = 700, and A_top10 at the tar sample (46.4) is higher than at t_rep
(38.4); the closed successor later lifts it to 2,300.

The nine non-confirming worlds split in two:
- Four L = 64 worlds (seeds 2003, 2008, 2015, 2019) never cross because their step-1 top ten contains one duplicated
  random tape (a = 63, two copies), which alone sets the baseline at e^63 / 20,000 ≈ 1.1 × 10^23. These count against
  P2 under the registered rule and are reported so. They are also the clearest instance of the mechanism under test:
  under the e^a weighting one accidental copy of a random 64-byte string outweighs every periodic class in the soup by
  twenty orders of magnitude. Among the 76 worlds without such a doubleton, 71 cross before t_rep (93.4%).
- Five worlds cross at or after t_rep: L = 20 seeds 2003 (t_rep 350, crossing 1,550), 2007 (300, 2,050), 2008 (700,
  700), 2014 (200, 450) and L = 50 seed 2001 (450, 500).

Post hoc, stricter: the crossing happens before the first tape enters the top ten at all in 66/80 worlds (82.5%;
L = 16: 20/20, L = 20: 12/20, L = 50: 18/20, L = 64: 16/20), so in those worlds the rise of A_top10 is carried entirely
by tapes other than the replicator. Figure `assembly_first_cross_vs_trep` shows all 80 worlds against the diagonal
(the four never-crossing worlds are drawn at the top edge).

## P3 — grey zone: AUC 0.721 pooled over L ≥ 25 (prediction ≤ 0.65, kill ≥ 0.75)

Primary stratum (all 720 Stage E runs with L ≥ 25, samples on the detectors.py snapshot-step schedule, n = 11,520 of
which 7,617 positive): AUC(A_top10) = 0.721, HOE = 0.749 on the same samples. Variants: @nominal arms only (the strata
of `results/detectors/NUMBERS_DETECTORS.md`) 0.706 (HOE 0.771); every recorded sample instead of the schedule 0.695
(HOE 0.691); @nominal, every sample 0.642 (HOE 0.707) — the only variant inside the predicted range, and not the
primary one. The HOE values of the @nominal per-L strata reproduce NUMBERS_DETECTORS.md exactly, so the sample sets
match. Below L = 25 both detectors work (0.932 and 0.960).

Structure of the number: per L (all arms) A_top10 is 0.93 at L = 25, 0.84 at 32–36, 0.81 at 49, 0.93 at 50, 0.73 at
64, 0.65 at 81, 0.71 at 100; under @nominal alone it is inverted at L = 81 (0.357) and 100 (0.380), exactly where HOE
was already known to invert. Per arm it ranges from 0.43 (stack-write-only @musweep) to 0.99 (none @steps8L). Per step
it is 0.75 at step 500, 0.68–0.76 through 50,000, and falls to 0.58 at 100,000 and 0.46 at 300,000: late in the run
the soups that never produced a heritable replicator carry a higher A_top10 than those that did. Verdict: the
pre-registered prediction is not met and the kill criterion is not met; assembly is not a usable detector of heredity
at L ≥ 25 (arm-dependent, inverted in two strata, AUC 0.72 pooled), but it is not as weak as predicted either.

## Exact snapshot assembly (context, `per_snapshot.csv`)

1,519 snapshots (80 worlds). The top-10 proxy A_top10 captures a median 7.3% of the exact A (10th percentile 0.1%):
the exact sum is dominated by low-copy-number, mid-complexity classes — the single largest contributor has a median
copy number of 11 and a median a of 18 (L = 64: n = 4, a = 22), and classes with ≤ 3 copies carry a median 35% of A
(50% at L = 20 and 64). P2 and P3 are, by pre-registration, about A_top10; the exact A exists only at snapshot steps
(85 of them precede t_rep, all at 500–1,000 steps; median A_exact 191 before vs 4.2 × 10^5 after t_rep).

## Deviations from the pre-registration, and why

1. Stage E N_T: the prereg fixes N_T = 20,000 for Stage G; in Stage E the `bytes` arm has 3,200–80,000 cells, so
   A_top10 there uses each run's own cell count (n_i = round(share × cells), divided by cells).
2. P3 primary stratum: "across Stage E worlds" was read as all 1,230 runs of conds/stageE.json (720 with L ≥ 25), with
   the detectors.py sample schedule (16 snapshot steps per run). The @nominal-only subset that detectors.py tabulates
   and an every-sample variant are reported alongside, not instead.
3. Post hoc companions were added and are labelled: the pre-arrival tar for P1, the decomposition of the nine P2
   failures and the 76-world count, crossing-before-arrival, and the composition columns of `per_snapshot.csv`.
4. Snapshots: the 1-million-step worlds (L = 20, 64) have no 7,500-step snapshot on disk (their schedule omitted it)
   and have 500,000 / 750,000 / 1,000,000-step snapshots instead; all files present were used. L = 20 seed 2014 has no
   emergence snapshot file although its summary records first emergence at step 800; nothing was substituted.
5. The outcomes were not appended to BIOLOGY_PREREG.md (outside the paths this analysis was allowed to write).
6. `per_sample_E_all.csv.gz` (all 858,540 Stage E samples, 23 MB) was written by the first pass; writing it is now
   opt-in (`--write-all-samples`) and the file was regenerated by the final run rather than deleted.

## Caveats

- Re-Pair gives an upper bound on the assembly index. It is exact on the three worked examples and, by the
  minimum-joins argument above, for every first tape; the P1 equalities are therefore exact, while a strict inequality
  only says that the tar is not a perfect period-2 or constant tape (the bound may overstate by how much).
- Samples of one run are correlated: the AUCs are Mann–Whitney statistics over (run, step) samples without confidence
  intervals; read the per-L, per-step and per-arm strata, as in detectors.py.
- The ground truth `event` is itself an assay (gen2 ≥ 0.3 on a top-3 exemplar holding ≥ 0.5% of cells); its share
  rule is what makes the registered P1 comparison degenerate.
- The e^a weighting makes A hypersensitive to single copies of complex strings (one random doubleton = 10^23; a
  handful of copies of a period-2 tape with one defect, a = 11, is worth e^11 ≈ 6 × 10^4 each and is what drives the
  L = 64 crossings at steps 2–3). A_top10 is a proxy that sees ~7% of the exact A.
- One exemplar world and 80 worlds in four tape lengths from one ablation (`none`); Stage E spans two ablations and six
  arms, which is where the arm dependence of P3 shows.
