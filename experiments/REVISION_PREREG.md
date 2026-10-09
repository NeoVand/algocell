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

**B5 (run 2026-10-08/09 on Modal, batch `bff_stdnh`, 12 soups; `results/bff_stdnh/FINDINGS.md`).** B5-1 met (7 of 7 first
replicators loop-bearing). B5-2 not met (7 of 12 transitions). B5-3 not met (one first replicator enters the partner in every
encounter; five are pointer-closed, one intermediate). Unregistered: the replicator is lost again by 16,384 epochs in 4 of 7.

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

## K — closure at an aligned intermediate length (written before the run, 2026-10-08 night; launched on Modal)

*Why.* Review 2 (Required 1) notes that the L = 20 and L = 50 closers jump "through the 16-bit address wrap": with memory
mapped modulo 2L and a 16-bit program counter, a backward relative jump that crosses address 0 lands at (65,536 + target)
mod 2L, which differs from the true-ring target whenever 2L does not divide 65,536. Our traces confirm it: the L = 50
JR NZ closer's second jump (from address 9, offset −16) lands at 31, not 95. Count from `stage_g_runs.csv`: 25 of the 67
closers use a relative jump at a misaligned length (all 20 at L = 50, 5 of 19 at L = 20); the other 42 (RET NZ designs
at L = 16, block copies at L = 20 and 64) do not depend on the wrap. L = 32 (2L = 64 divides 65,536) is the untested
aligned length between 16 and 64.

*Design.* Stage G conditions at L = 32 (`none@closure1M`, 128 instructions, k = 4, square lattice, 1,000,000 steps),
seeds 5001–5020, on Modal (`conds/stageK.json`, batch `stageK`), scored by the Stage G pipeline (culture test, partner
tests, c4 census, closure).

*Predictions.* K1: the first heritable replicator is a load–push word in ≥ 18 of 20 worlds. K2: a closed successor (loop-
bearing, ≥ 0.95 of partners copied, 0.00 self-damage) dominates by one million steps in ≥ 15 of 20 worlds, by a mechanism
that cannot use the wrap (there is none at this length). K3: the median first-replicator step lies between 300 and 1,000.

*Kill.* K2 below 10 of 20 kills "closure evolves at intermediate lengths" as a wrap-independent result; the paper would
then restrict the closure claim to L = 16 and 64 and name the dependence at L = 20 and 50.

*Cost.* 20 runs × 1,000,000 steps ≈ $0.16 each at the recorded rate, ≈ $3–4 with overhead.

**K (run 2026-10-09 early on Modal; `results/stageK/FINDINGS.md`).** K1 met (20/20 load–push first). K2 not met, not
killed: 12/20 closed by one million steps, all by block copy, none by a jump; 8 worlds still open. K3 met (median 425).
Reading: closure at aligned lengths is monotone in L (20/20 at 16 by 300k; 12/20 at 32 and 8/20 at 64 by 1M) and goes by
block copy or return; the jump closures at L = 20 and 50 depend on the wrap and are reported as such.

## L — ten-million-step extensions: does the capacity for inherited variation recover after closure? (written before the run, 2026-10-09)

*Design.* Stage G conditions at L = 16 and L = 20, 10 worlds each (seeds 6001–6010), horizon 10,000,000 steps, samples every
5,000, snapshots at the Stage G steps and at 2, 3, 5, 7.5 and 10 million (`conds/stageL.json`, Modal batch `stageL`). Scored
by the Stage G pipeline plus the mutational scan (`mutscan.py`) of the dominant tape at every snapshot and, once the
executed-address bitmap exists, the copied-but-unexecuted fraction U.

*Predictions.* L1: in ≥ 5 of 10 worlds at each length the dominant tape at ten million steps has more transmissible sites
than the closed dominant at 300,000 (L = 16) or one million (L = 20) steps. L2: every such recovery is in a block-copy
lineage (the dominant carries `LDIR`/`LDDR`) and its transmissible sites are copied-but-unexecuted positions. L3: no
return-based or push-based dominant has more than one transmissible site at any snapshot.

*Kill.* L1 failing at both lengths (capacity stays ≤ 1 site in ≥ 8 of 10 worlds) kills "the genotype reopens" within this
horizon; the paper would then report the first individuals as canalised for as long as we watched.

*Cost.* 20 runs × 10,000,000 steps ≈ $1.6 each at the recorded rate, ≈ $35 with overhead; ~1 h wall in parallel.

## M — the 8080 subset: a second real instruction set (written before the run, 2026-10-09)

*Design.* The Z80 restricted to the Intel 8080 instruction set by suppression (`i8080` in `make_conds.ABLATIONS`: the
whole CB page, the 78 defined ED-page opcodes, and EX AF,AF', DJNZ, the five JR forms and EXX; verified tonight that
`01 c5`, `ad e3 21 c0`, JP, JP (HL) and CALL are untouched). Deviations from a real 8080 are stated in the paper (IX/IY
prefixes keep selecting the IX/IY form of an 8080 instruction; suppressed bytes are NOPs, not the 8080's undocumented
aliases). Stage G conditions at L = 16 (300,000 steps) and L = 32 (one million), 20 worlds each, seeds 7001–7020
(`conds/stageM.json`, Modal batch `stageM`).

*Predictions.* M1: the first heritable replicator is a load–push word in ≥ 18 of 20 worlds at both lengths (the literal
word is 8080 code). M2: at L = 16, a closed successor dominates by 300,000 steps in ≥ 15 of 20 worlds, by a return design
(the Z80's `ad e3 21 e3 21 c0 ad c0` is entirely 8080 code), none by block copy (there is none). M3: at L = 32, closure by
one million steps in ≤ 5 of 20 worlds (the Z80 closed 12 of 20 there, all by block copy, which the 8080 lacks). M4: the
8080 subset explains the published negative result for long 8080 tapes: return closure needs the short ring that aliases
the popped address back into the organism.

*Kill.* M1 failing kills "open first" as a two-machine result. M2 failing (closure at L = 16 in fewer than 10 of 20) means
the return route is not enough and the Z80's closure depended on its extensions; reported as such.

*Cost.* 20 × 300k + 20 × 1M ≈ $4.

**P (run 2026-10-09; `results/exectrace/FINDINGS.md`).** Met: every first replicator fetches partner bytes in all 256
encounters; every loop-bearing final (89 of 89 across G, K, I) never does; pointer closure and zero inflow agree in 106 of
110 finals, the four exceptions being pointer-closed with one to three partner bytes retained; no pointer-open final has
zero inflow. Unregistered: the pusher's transmissible sites are all executed (operands); the closers' are unexecuted.

## D — two dials in the second machine (written before the run, 2026-10-09; NATURE_PLAN Move 3)

*Machinery.* `micro/bff.py` gains `halt_p` (an unmatched bracket halts the encounter with probability p, hashed from pair
index, step and a per-call seed so that the GPU kernel and the CPU reference agree; p = 1 is the published rule, p = 0
the `nohalt` variant) and `lit_rep` (the literal `P a b` writes its two operand bytes r times, write ratio 2r/3; r = 1 is
the published switch). Kernel and reference checked byte for byte under five settings; the 15 BFF tests pass.

**D1, tar lethality as a dial** (Model 5's crossover). Variant `wraplit` (wrap + literal) with p ∈ {0.01, 0.03, 0.1, 0.3},
6 soups each (seeds 1–6), 16,384 epochs; p = 0 (`wraplitnh`, 12 soups) and p = 1 (`wraplit`, 12 soups) exist.
*Predictions.* D1-1: the open all-`P` population persists to 16,384 epochs (share ≥ 10%) in ≥ 4 of 6 soups at p = 0.01
and in ≤ 1 of 6 at p = 0.3. D1-2: the epoch at which the all-`P` share first falls below 1% decreases monotonically in p
over p = 0.03, 0.1, 0.3, 1 (medians), within a factor of three of 1/p scaling. D1-3: no closure by control flow at any p
(no closed heritable tiling exists). *Kill.* D1-1 failing in the direction of extinction at p = 0.01 (≥ 4 of 6 extinct)
kills the window reading: lethality would then be a step, not a dial.

**D2, literal bandwidth as a dial** (the write-ratio boundary). Variant `lit` (no wrap) with r ∈ {2, 3}, 6 soups each
(seeds 1–6); r = 1 exists (12 soups: 8 bits of inflow, open, then a 0.2% minority). *Predictions.* D2-1: at r ≥ 2 the
first replicator's inflow is ≤ 1 bit (the copy completes within one pass: 21 pushes × 4 bytes ≥ 64) against 8 bits at
r = 1: information-closed from birth without a loop and without a wrapping pointer, the theorem's boundary. D2-2: the
first replicator is still the all-`P` tiling, loop-free, pointer-open (enters the partner in every encounter). D2-3: the
collapse into bracket tar is slower or absent at r ≥ 2 (fewer partial copies), measured as the all-`P` share at epoch
1,024. *Kill.* D2-1 failing (≥ 4 bits at r ≥ 2) kills the saturation reading of Theorem 2's boundary.

*Cost.* 36 soups ≈ 25 min each on an L40S ≈ $30.

**M (run 2026-10-09 on Modal; `results/stageM/FINDINGS.md`).** M1 met (20/20 and 20/20 load–push first). M2 met (20/20
closed at L = 16, all by the byte-identical RET NZ design). M3 met (0/20 at L = 32). M4 supported.

### Outcome of L (appended 2026-10-09, after `score_L.py`)
L1 not met (1/10 at L = 16, 0/10 at L = 20); L2 not met at L = 16 (the one recovery is a block-copy lineage with 1 of 4 sites
unexecuted), n/a at L = 20; L3 not met as written (the open pusher at L = 20 holds 2–3 sites and the DJNZ-based closer 2); the
kill criterion fired (final dominant with ≤ 1 site in 9/10 and 8/10 worlds). Every world closed; 17/20 end on a minimal closer
with no transmissible site; genome-bearing block copiers (10–12 sites, 7–10 unexecuted) were the most common tape in 3/20
worlds for 1–3 million steps at < 0.3% share and were replaced. The paper reports the first individuals as canalised for as
long as we watched (ten million steps). Details: `results/stageL/FINDINGS.md`, `SCORING.md`.

### Outcome of D (appended 2026-10-09, after `dials_score.py`; r = 2 pending at the time of writing)
D1-1 half met (6/6 persist at p = 0.01; at p = 0.3 the share falls below 1% in 6/6 at ~850 epochs and recovers to 0.10–0.11 in
5/6); D1-2 not met (no collapse at p ≤ 0.1; a threshold between 0.1 and 0.3, not 1/p); D1-3 met; kill not fired. D2-1 met at
r = 3 (0.00 bits against 8.00 at r = 1); D2-2 met (6/6 loop-free, pointer-open); D2-3 met (collapse absent at r = 3; every
random tape heritable at 16,384 epochs). Details: `results/bff_dials/FINDINGS.md`, `SCORING.md`.
