# Pre-registration for the revision experiments (written 2026-10-08, late night, before any run)

Prompted by review 1 (`REVIEW_1_ASSESSMENT.md`). Predictions and kill criteria are fixed here before the data exist;
outcomes will be appended below the line, misses included. Spend: B5 on Modal under the user's blanket authorisation of
2026-10-08 ("more than welcome to use Modal"); M and P on the local GPU.

## B5 — the unrun cell of the classification: BFF as published, unmatched brackets as no-ops

*Why.* Fig. 5d claims a 2 × 2 classification (literal-write instruction present/absent × tar benign/lethal) with one
cell marked "not run". Review 1 (both perspectives) says a classification with an unrun cell is a regime map.

*Design.* `modal_bff.py --variants std --nohalt`, seeds 1–12, 2¹⁷ programs, 16,384 epochs, everything else as the
published variant (no pointer wrap, no literal). Variant name `stdnh`. Scoring by the same pipeline as the other five
variants (`micro/bff_analysis.py`: first heritable replicator, loop content, pointer entry, inflow).

*Predictions.* B5-1: every first replicator is loop-bearing (0 straight-line first replicators), as in `std` (9/9) and
`wrap` (19/19): without a literal channel no open form exists (Theorem 1), and removing the halt does not add one.
B5-2: the transition rate is at least that of `std` (≥ 9 of 12 worlds with a heritable replicator by 16,384 epochs),
since execution is never cut short. B5-3: pointer entry of the first replicator is ≤ 0.05 (closed) in every world, as in
`std`/`wrap`.

*Kill criteria.* A single straight-line (loop-free) first replicator kills "no literal ⇒ born closed" for BFF and the
cell is reported as such. B5-2 failing is informative, not a kill: it would mean halting is not what limits `std`.

*Cost estimate.* Earlier BFF soups: 60 soups = 24.8 GPU-h ≈ $47 at list (≈ $0.80 per soup). 12 soups ≈ $10, wall
≈ 1–1.5 h in parallel. Dry run through the exact path (`--smoke --nohalt`) before launch.

## M — single-byte mutational scan of every Stage G first replicator and final dominant

*Why.* Cicala et al. (2026) explain the Load–Push → LDIR succession by mutational robustness ("LDIR ≫ LDD ≫ Load–Push");
our paper explains the open → closed succession by context independence. Review 1 says we must test the alternative. The
optimistic reviewer asks the larger question: does closure open a channel for inherited variation? One scan answers both.

*Design.* For each of the 80 Stage G worlds, the first heritable replicator and the final dominant (160 tapes). For every
position i and every alternative value v (all 255 at L = 16, 20; 32 seeded values at L = 50, 64), the single mutant
x[i ← v] is run as organism A against 32 shared random partners for 128 instructions (the culture test's executor,
`assay.execute_pairs`), then its offspring against 32 fresh partners. Recorded per mutant: gen2 (heritable iff ≥ 0.3,
the pre-registered threshold), faithful (also ≥ 50% of partners became ≥ 75% copies), and *transmission*: among the
offspring that are copies (≥ 75% at the best cyclic shift), the fraction carrying v at the aligned position of i. A
position is a *transmissible site* if, over the tested values, at least half of the mutants are heritable AND transmit
their byte in at least half of their copies. Per tape: robustness = fraction of heritable mutants; sites = number of
transmissible sites; capacity = Σ_i log₂(1 + 255 × frac_both(i)) bits.

*Predictions.* M1 (Cicala's measure): the final dominant is more mutation-robust than the first replicator in ≥ 15 of 20
worlds at each of L = 16, 20, 50 (where finals are closed in ≥ 19 of 20). If M1 holds, mutational robustness and context
independence co-vary and the succession alone cannot separate them; M3 then does. M2 (variation channel): loop-bearing
finals have more transmissible sites than first replicators (median over worlds, at each L where finals are loop-bearing);
stated bluntly, closure makes room for heritable variation. M3 (matched pair): at L = 50, in the worlds whose final is
the pusher tiling with `20 f0` inserted, the final has strictly more transmissible sites and higher capacity than the
first (the two tapes differ by two bytes, so this isolates the jump).

*Kill criteria.* M2 killed if first replicators have as many or more transmissible sites than loop-bearing finals at two
of the three lengths. M3 killed if the matched finals do not exceed their firsts in a majority of the matched worlds.
Either outcome is reported; a killed M2 means closure buys fidelity, not evolvability, and the paper says so.

*What this does not test.* Fitness in the soup (invasion) and multi-step evolvability; those are the next experiments
(`REVIEW_1_ASSESSMENT.md`, §4).

## P — pointer confinement for every Z80 tape (not run tonight)

One flag in the test shader (any fetch at address ≥ L sets `entered`), rerun of the partner test for the 160 tapes.
Prediction: every loop-bearing final with zero inflow has entered = 0 in all 256 encounters; every first replicator has
entered = 1 in all 256. Kill: a zero-inflow final that enters the partner (then the Z80 has a saturation closer too).

---
## Outcomes (appended after the runs)

**M (run 2026-10-08, late night; `results/mutscan/FINDINGS.md`).** M1 killed (final more mutation-robust than first in
2/20, 2/20, 0/20, 8/20 at L = 16, 20, 50, 64). M2 killed and reversed: transmissible sites median first vs loop-bearing
finals 0 vs 0, 1 vs 0, 23 vs 1, 32 vs 0.5; capacity 32 vs 6, 54 vs 11, 269 vs 42, 325 vs 14 bits. M3 killed: 0 of 14
matched L = 50 pairs. Reading: the first closers execute everything they copy and inherit no variation; the open pusher
inherits mutations in its operand bytes; heritable variation returns only in genomes with a copied-but-skipped segment
(five worlds with a jump or load before a block copy). Cicala et al.'s mutational-robustness explanation of the succession
is not supported by this measure.

**B5 (launched 2026-10-08 21:44 local on Modal, batch `bff_stdnh`, 12 soups; outcome pending).**

## I — head-to-head invasion of the matched pair at L = 50 (written before the run, 2026-10-08 night)

*Why.* Review 1's first priority: show that the closing change itself confers an advantage in competition, not only in
the partner test. At L = 50 the final dominant of 14 worlds is the pusher tiling with `20 f0` (JR NZ) inserted three times,
so ancestor and successor are matched except for the jump.

*Design.* `invasion_pair.py`: soups of 20,000 cells at L = 50, 8,192 pairs per step, 128 instructions, mutation 1/16
(k = 4), square lattice, as in Stage G. Resident: every cell set to one tape (phase 0). Invader: 1% of cells (200, seeded
RNG) set to the other tape at step 0. Two directions (closed into pusher-filled; pusher into closed-filled), 5 seeds each,
20,000 steps, recorded every 250 steps: share of cells within Hamming distance 4 of either tape under any cyclic shift
(nearest class wins; the two tapes differ at 6 positions), zero fraction; every 2,500 steps the heritable fraction of 16
random cells.

*Predictions.* I1: the closed form rises above 50% of cells within 20,000 steps in ≥ 4 of 5 soups. I2: the pusher stays
below 5% of cells in ≥ 4 of 5 soups when seeded into the closed resident. *Kill.* I1 failing kills "the return itself
is selected" in a kin-filled world (the advantage would then exist only among strangers or damaged copies, or not at all).

**I (run 2026-10-08 night; `results/invasion_pair/FINDINGS.md`).** I1 by the pre-registered class share: not met (the
share splits the quasispecies cloud between classes; a definitional miss, reported as such). By the share of cells
carrying the jump word, defined post hoc: the closed form passes 50% at step 110–130 in 5 of 5 soups. I2 met: the pusher
seeded into a closed world leaves no trace (its core share equals the closed world's own cloud). Unregistered control
result: in pusher-only worlds the jump word arises and passes 50% at steps 5,000, 7,500 and 13,250 (3 of 3).
