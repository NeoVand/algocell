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
r = 2 landed 23:51: D2-1 met (0.00 bits, 6/6), D2-2 met (loop-free, pointer-open, 6/6), D2-3 met (all-`P` share 22–31% at epoch 1,024, 0.97 at 16,384, every random tape heritable). All D predictions scored.

### Correction to the outcome of P (2026-10-09)
The outcome note above says "every first replicator fetches partner bytes in all 256 encounters". That is true of the registered
set (the 80 Stage G first replicators) and of Stage K (20 of 20), but not of Stage I: its 10 first replicators, born under lethal
tar, never enter the partner (entered 0.00, inflow 0 bits). The pooled count is 100 of 110. The error reached the findings file
and the v4 draft, both now corrected; the result itself (open under benign tar, born closed under lethal tar) is unchanged.

### Correction to the outcome of M (2026-10-09, round-2 review)
M4 was marked "supported" on the reading that the 8080 subset expresses no closer on a 32-byte ring. That reading is false.
A reviewer constructed a 20-byte closer from 8080 instructions (`21 20 00 31 40 00 16 10 2b 46 2b 4e c5 15 c2 08 00 c3 11
00`, a counted PUSH loop closed by JP NZ, then a jump to itself) plus 12 arbitrary bytes; in our executor under `i8080` it
copies exactly into 256 of 256 random partners, never enters the partner, survives eight serial transfers exactly and scores
1.00 on the culture test (`check_8080_closer.py`, `results/review_r2/check_8080_closer.txt`). M3 (0 of 20 closed at L = 32)
stands as observed; M4 is withdrawn: the barrier at L = 32 is that evolution did not find a closer, not that none exists.

## R2 — four tests asked for by the round-2 review (written before any run, 2026-10-09)

All local GPU, deterministic executor, no Modal. Scripts: `serial_retention.py` (S), `invasion_closer.py` (A),
`marker_check.py` (V), `population_sample.py` (Q). Every quoted number will come from the tables they write.

**S — serial allele retention.** The mutational scan scores transmission in first-generation copies only. S follows each
single-byte allele through serial transfers, with every lineage in the denominator.

*Panel* (tape, executor rule): (1) pusher `01 c5`×8, L = 16; (2) pusher ×25, L = 50; (3) pusher ×32, L = 64; (4) the RET NZ
closer `ad e3 21 e3 21 c0 ad c0`×2, L = 16 (Stage G modal final); (5) the LDIR tiling `1e 04 ed b0`×5, L = 20 (Stage G
modal final); (6) the LDIR tiling `04 5e ed b0`×8, L = 32 (Stage K modal block-copy final); (7) the pusher with three jumps,
L = 50 (Stage G modal final; offspring = parent + one jump); (8) the Stage I final with the most first-generation sites
(`1d 2e 0f c3 …`, seed 4009, L = 16), lethal tar; (9–11) the three transient genomes, the modal tape at the first snapshot
where each was modal (Stage L L = 16 seed 6006 step 1,000,000; L = 20 seed 6003 step 100,000; L = 20 seed 6004 step 500,000);
(12) the 8080 pusher `01 c5`×16, L = 32, under `i8080`; (13, 14) the constructed 8080 closer with two random 12-byte
payloads (seed 20261010), L = 32, under `i8080`.

*Procedure.* Mutants x[i ← v]: all 255 values at every position for L ≤ 32, 32 seeded values per position at L ≥ 50.
Each mutant founds R = 16 lineages; the unmutated tape founds 256. Generation g: the lineage tape runs as A against a
fresh uniform random partner (128 instructions); the partner half after the encounter is the next lineage tape. The
partner sequence of lineage r is shared by all mutants of a genotype (common random numbers). A transfer is a *copy* if
the offspring matches its parent at ≥ 75% of positions at the best cyclic shift; a lineage is *alive* at g if transfers
1..g were all copies. The allele is *present* at g if the mutant's 5-byte cyclic window centred on i occurs cyclically in
the lineage tape (alignment-free, so inserted jumps and shifts do not hide it). A mutant is *identifiable* if its window
does not occur cyclically in the wild type; others are excluded and counted. Background b_g(i, v): the share of alive
control lineages whose tape contains the mutant's window at g.

*Outcomes,* per genotype at g = 1, 2, 4, 8, over identifiable mutants with every lineage counted: **lost** (not alive),
**erased** (alive, window absent), **retained** (alive, window present). A site is *serially transmissible* at g if, for
at least half of its identifiable values, P(present | alive) − b_g ≥ 0.5 (the mutational scan's 0.5 thresholds carried to
generation g). Also reported: the first-generation aligned rate as in `mutscan.py`, for continuity.

*Predictions.* S1: the pusher's sites persist; at L = 50 and 64 its serially transmissible sites at g = 4 are at least half
of its first-generation count at g = 1, and ≥ 10. S2: the evolved Z80 closers (4–7) have ≤ 2 serially transmissible sites
at g = 4, and among their non-lost lineages at g = 1 the erased share exceeds the retained share. S3: the constructed 8080
closer (13, 14) retains its payload: ≥ 11 of 12 payload positions serially transmissible at g = 8, control lineages alive
at g = 8 in ≥ 0.99. S4: each transient genome (9–11) has ≥ 5 serially transmissible sites at g = 4. S5 (descriptive): the
pusher at L = 16 (first-generation median 0 sites) has ≤ 2 at g = 4.
*Kill.* S1 failing narrows "the first replicator transmits its mutations" to first-generation copies, and the contrast in
the paper is restated on S's numbers. S3 failing withdraws "closure does not by itself erase variation".

**A — accessibility or selection.** Stage M dynamics (160 × 125 square lattice, 8,192 pairs, 128 instructions, mutation
1/16 per pair, L = 32). A1: the constructed 8080 closer (payload 1 of S) seeded at 1% into a world filled with the 8080
pusher `01 c5`×16, under `i8080`; 5 seeds; and the pusher at 1% into a closer-filled world, 5 seeds; 3 unseeded controls
per resident; 20,000 steps. A2 (full Z80): the same constructed closer against the evolved LDIR closer `04 5e ed b0`×8,
1% each way, 5 seeds each, 3 controls each, 50,000 steps. Recorded every 10 steps to 500, then every 250: the share of
cells carrying the closer's 20-byte core at any cyclic shift, the share within Hamming 8 of each resident tape at any
shift, and among core carriers the share whose 12 payload bytes equal the seeded payload and the number of distinct
payloads.
*Predictions.* A1-1: the closer passes 50% core share by step 20,000 in ≥ 4 of 5 soups. A1-2: the pusher seeded into the
closer world stays < 5% (Hamming-8 class) in ≥ 4 of 5. A1-3: among core carriers at step 20,000, ≥ 10 distinct payloads
(variation is carried in the soup). A2 is two-sided and descriptive: invader > 50% at 50,000 steps in ≥ 4 of 5 counts as
favoured, < 5% in ≥ 4 of 5 as disfavoured, otherwise neither.
*Kill.* A1-1 failing means the 8080 closer exists but does not win against the pusher; the paper then says expressible
but not competitive, not "not found".

**V — the invasion marker.** The L = 50 invasion endpoint is the share of cells carrying `20 f0` (post hoc). V re-runs the
closed-into-pusher invasion (seeds 1–3, the same initial conditions) and at steps 0, 50, 100, 150, 200, 300, 500 and
1,000 traces 512 random cells against 16 random partners each; a cell is *confined* if no partner byte is fetched in any
of the 16. *Prediction* V1: at every time point the confined share is within 0.10 of the marker share, ≥ 90% of marker
cells are confined and ≤ 10% of marker-free cells are. *Kill.* V1 failing: the figure reports the confined share and the
marker is dropped.

**Q — population sampling.** Final snapshot of every Stage G, K, I and M world, and the ten-million-step snapshot of every
Stage L world: 256 random cells (seeded), each assayed by the culture test (32 partners, gen2 ≥ 0.3) and traced against
16 partners. Per world: heritable share, confined share among heritable cells, share of heritable cells identical (any
cyclic shift) to the modal tape, and the median Hamming distance of heritable cells to the modal tape at its best shift.
Up to 8 random heritable cells per world get the single-mutant scan (16 values per position, 16 partners) for the
distribution of transmissible sites across the population, not only on the modal tape. *Prediction* Q1 (descriptive): in
worlds whose modal tape is closed, ≥ 90% of heritable cells are confined and the median number of transmissible sites of
sampled heritable cells is ≤ 2.

*Cost.* Local GPU; S ≈ minutes; A ≈ 1–2 h; V ≈ 10 min; Q ≈ 30 min.

## R2b — population classes over time (written 2026-10-09 after the first Q results and before this run; exploratory)

*Why.* Q's final-snapshot sample showed that the modal tape can misrepresent a world: at L = 50 no world's modal tape is
confined and 18–39% of heritable cells are; in lethal-tar worlds and some L = 20 and 64 worlds most heritable cells are
distinct closers carrying 9–44 transmissible sites while the modal tape carries none. A clonal population yields a
modal tape and a diverse one hardly does, so modal-tape analysis selects regenerators. Q2 measures classes of cells
directly. Because Q motivated it, Q2 is exploratory; definitions are fixed here before it runs.

*Sample.* Every snapshot of every Stage G, I, K, L and M world (final included): 64 random cells (seeded). Each cell: the
culture test (32 partners; heritable iff gen2 ≥ 0.3) and tracing against 16 random partners (confined iff no partner byte
is fetched in any). Up to 16 random heritable cells per snapshot: the single-mutant scan (16 values per position, 16
partners; transmissible site as in the mutational scan) and the executed union of the 16 traced encounters.

*Classes of heritable cells.* **open**: not confined. **regenerator**: confined, ≤ 2 transmissible sites. **transmitter**:
confined, ≥ 5 transmissible sites. **intermediate**: confined, 3–4 sites. Per snapshot: the heritable share, the share of
heritable cells in each class (classes of unscanned confined cells are inferred from the scanned ones), and the median
number of unexecuted transmissible sites among transmitters. For open cells the scan is first-generation only, which S
showed overstates their sustained transmission; they are reported as open, not by sites.

*Use.* Replaces modal-tape counts wherever the paper speaks about a world's population (closure, variation, the fate of
genomes); modal-tape results stay, labelled as such. No predictions are registered; the outcome is reported as found.

### Outcome of R2 (appended 2026-10-09 after the runs; tables in `results/serial_retention/`, `results/marker_check/`, `results/population/`, `results/invasion_closer/`, numbers in `results/review_r2/ROUND2_NUMBERS.md`)
**S.** S1 not met: the pusher's lineages stop copying (unmutated lineages alive after four transfers 0.05–0.18, after eight 0.00–0.04), so its first-generation sites (16 at L = 50, 30 at L = 64) fall to 0 and 7 at four transfers and 0 at eight; the kill applies and the paper restates the claim on S's numbers (first-generation transmission only). Post hoc, partners drawn from the pusher's own world give the same result (alleles retained at four transfers ≤ 0.01). S2 met (evolved closers 0 sites at g4; erased 0.60–0.88 vs retained 0.00–0.01 at g1 and g8). S3 met (12 of 12 payload sites at g8 for both payloads; control lineages alive 1.00). S4 met (10, 9 and 12 sites at g4, 7–9 of them unexecuted). S5 met (0 sites). Unregistered: the L = 50 closer's lineage stops copying at the third transfer (`l50_lineage.py`).
**V.** V1 not met: marker share 0.73–0.75 at steps 300–1,000 against confined share 0.19–0.25; P(confined | marker) 0.26–0.33. The kill applies: the L = 50 invasion is reported as the spread of the jump, not of closure. The re-run reproduced the marker trajectory statistically, not bitwise (GPU pairing order is non-deterministic, as stated in Methods).
**A1.** A1-1 not met on the registered metric (exact 20-byte core share 0 at 20,000 steps in 5 of 5): the core mutates within a few hundred steps. Post hoc: the closer's loop body passes 50% of cells at steps 300–320 and the pusher class falls below 1% at steps 490–1,000 in 5 of 5; at 20,000 steps 84–98% of random cells are heritable and at most 2% of heritable cells are open, most of them closed with ≥ 5 sites. A1-2 met (pusher class < 5% in 5 of 5). A1-3 not scorable on the registered core metric; post hoc, random transmitter cells carry 15–17 never-executed sites through eight transfers. The kill (closer exists but does not win) did not fire.
**A2.** The evolved block-copy tiling seeded into the closer world: tiling class > 50% at 50,000 steps in 5 of 5 (favoured). The closer seeded into the tiling world: < 5% core share in 5 of 5 (disfavoured; registered core metric, consistent with the population classes).
**Q.** Q1 not met: among the 109 worlds with a confined modal tape, confinement of heritable cells is ≥ 0.9 in 105 (fails: three L = 20 worlds at 0.60–0.61 and one L = 64 world at 0.86); the median number of transmissible sites of sampled heritable cells is 0 (that part met), but the pooled median hides a split by stage, lethal-tar worlds having a median of 12. L = 50: no world's modal tape confined; confined among heritable cells 0.18–0.39.

## R3 — tests asked for by the second round-2 report (written before any run, 2026-10-09)

Local GPU, no Modal. Scripts: `gen_conv_shader.py` + `conv_soups.py` (C), `invasion_aligned.py` (B), `first_closure.py`
(T), `census_period.py` (P), `robust_ref22.py` (F).

**C — harness conventions.** Two derived soup and executor shaders, byte-identical to the Stage G ones except for one
hunk after the register reset of every encounter: *randreg* draws A, F, B, C, D, E, H, L, their alternates, IX and IY
uniformly at random (PC = 0 and SP as before); *randsp* draws SP uniformly from 0–65,535 (registers zero). The draw is
seeded by the step's batch seed and the pair index. 20 worlds each at L = 16 (seeds 8001–8020), Stage G conditions,
300,000 steps; top-3 tapes recorded every 250 steps to 20,000 and every 2,500 after; snapshots at 20,000, 100,000 and
300,000. Every assay (culture test, tracing) runs under the world's own convention.
*Predictions.* C1: the first replicator is a load–push word in ≥ 18 of 20 worlds under randreg (its mechanism reads no
register but SP). C2: under randsp the first replicator is open (pointer enters the partner in ≥ half of 16 encounters)
in ≥ 15 of 20 worlds with a replicator. C3 (descriptive, no direction registered): population closure at 300,000 steps
under each convention, against 20 of 20 in Stage G. *Kill.* C1 failing: "open first" is reported as a property of the
zero-register convention. Also: the partner test (256 partners) of every Stage G and K first and final tape under
randreg and randsp: copies, self-damage, confinement.

**B — invasions at aligned lengths.** Stage G dynamics; L = 16: the return closer `ad e3 21 e3 21 c0 ad c0`×2 against
the pusher `01 c5`×8; L = 32 (Stage K dynamics): the block-copy tiling `04 5e ed b0`×8 against `01 c5`×16. 1% each way,
five seeds each, three unseeded controls per resident, 20,000 steps. Endpoints: share within Hamming ⌈L/4⌉ of each tape
(any shift) every 10 steps to 500 and every 250 after; classes of 64 random cells (culture test, tracing) at steps 1,000,
5,000 and 20,000. *Predictions.* B1: the closer seeded into the pusher world holds most cells (class share > 0.5) and
most heritable cells are confined at 20,000 steps in ≥ 4 of 5 soups at each length. B2: the pusher seeded into the
closer world stays below 5% in ≥ 4 of 5. *Kill.* B1 failing at both lengths: the selection claim is withdrawn.

**T — first-closure timing.** For every world of Stages G (L = 16, 20, 64), L, K, I and M (L = 16), the first sample at
which one of the three most common tapes holding ≥ 0.5% of cells is heritable (gen2 ≥ 0.3, 64 partners) and confined in
16 of 16 traced encounters; censored at the horizon. Kaplan–Meier medians; log-rank tests of benign L = 16 (Stages G and
L, 30 worlds) against lethal L = 16 (Stage I, 10). *Decision rule, registered:* unless benign closure is significantly
earlier (log-rank P < 0.05), the text says "displaced" and restricts the window reading to BFF; in either case it says
"displaced" unless descent is shown.

**P — census by period.** All 16,777,216 three-byte words tiled to 16 bytes (five words and the first byte), executed
for 128 instructions as organism against the all-zero partner: self-writers are tapes whose partner ends equal to the
tape at some cyclic shift; each self-writer traced against the zero partner and 16 random partners (closed if the pointer
fetches no partner byte in any). *Prediction.* P1: no closed self-writer of period 3.

**F — ref. 22's robustness test.** Every Stage K first and final tape (L = 32): 1,000 trials each of 1, 4 and 8
successive random single-byte mutations; a trial is robust if the mutant, run for 512 instructions against the all-zero
partner, leaves the partner equal to the mutant (shift 0; any shift reported too). *Prediction* (from ref. 22): F1, the
block-copy finals are more robust than the pushers at every k. Reported beside the single-mutant scan of the same tapes.

## E — the copy-offset switch: is erasing variation selected, or only likelier? (written before any run, 2026-10-09)

*Finding that motivates it (measured before this registration, `offset_census.py`).* In the 4-byte block-copy core
`XX 5e ed b0` (LD E,(HL); LDIR, with HL = D = B = C = 0), the first byte XX is both the first instruction and the copy
offset: E = XX, so the copy lands at XX mod 2L. Offsets 4 and 8 (and 16 at L = 32) overwrite the ring, the organism's own
body included, with a tiling of its first 4, 8 or 16 bytes: the closer regenerates and erases every other byte. An
offset of exactly L copies the tape verbatim: the closer transmits. Among the 256 values of XX, the working ones are 13
regenerating against 3 transmitting at L = 16 (`04 24 44 64 84 a4`, `08 48 68 88 a8 c8 e8` against `50 90 b0`) and 9
against 2 at L = 32 (`04 44 84`, `08 48 88 c8`, `50 90` against `60 a0`). One byte switches a closer between erasing and
keeping variation, and the erasing values are about four times more numerous.

*Hypotheses.* H-sel: regeneration is selected (a regenerator outcompetes a transmitter with the same core). H-bias:
the two are nearly neutral and mutation of the offset byte, which reaches a regenerating value about four times more
often than a transmitting one, drives populations toward regeneration.

*Design* (`offset_switch.py`, Modal). R and T differ only in byte 0 and in what follows the core: L = 16, R = `44 5e ed
b0`×4, T = `b0 5e ed b0` + 12 random bytes drawn separately for every T cell; L = 32, R = `04 5e ed b0`×8, T = `a0 5e ed
b0` + 28 random bytes. Conditions: L = 16 benign tar, L = 16 lethal tar (zero halts), L = 32 benign tar. Starts: a 50:50
random mixture (10 worlds with mutation off, 100,000 steps; 10 with the standard mutation, 300,000 steps), and 1%
invasions each way with mutation (5 worlds each, 300,000 steps). Every 500 steps (every 50 to 5,000): every cell is
classified by its first four bytes (core `5e ed b0` at bytes 1–3 and the offset class of byte 0: regenerating,
transmitting, broken) and, among transmitting cells, the number of distinct tails. Soup at the end saved.
*Predictions (from H-bias).* E1: with mutation off, the transmitting share among core-carrying cells at 100,000 steps
lies in 0.3–0.7 in ≥ 7 of 10 mixtures at each benign condition. E2: with mutation on, it falls below its start and
lies in 0.05–0.4 at 300,000 steps in ≥ 7 of 10 mixtures (the neutral mutation-bias equilibrium is ≈ 0.19 at L = 16 and
0.18 at L = 32 if only offset-byte mutations act). E3 (lethal tar): no direction registered; observed lethal-tar
populations are mostly transmitters, and this tests whether that is selection.
*Reading.* E1 failing with the regenerating share rising means regeneration is selected at a fixed core (H-sel); E1
failing the other way means transmission is selected. Cost ≈ 90 worlds × 2–3 min on L40S ≈ $8.

### Outcome of C (appended 2026-10-09; `results/conv/REPORT.md`, `conv_worlds.csv`; run on Modal, L40S)
C1 met: under random initial registers the first replicator is a load–push word in 20 of 20 worlds (all open). C2 met:
under a random stack pointer the first replicator is open in 18 of 20 (a load–push word in 5; `e1 e3`, POP HL; EX (SP),HL,
in 13). C3 (descriptive): under random registers no population closed by 300,000 steps (0 of 19 with heritable cells;
heritable share median 0.08); under a random stack pointer 15 of 20 closed, mostly by block-copy closers, many of them
transmitters. Every evolved closer of the standard soups relies on zero registers (the 4-byte LDIR core needs HL = D = B
= C = 0; the return closer needs a zero flag and zero A and L).

## R4 — closure under random registers: accessible or impossible? (written before the runs, 2026-10-09)

Self-initialising closers exist: `21 00 00 11 10 00 01 10 00 ed b0 18 fe` + 3 bytes (LD HL,0; LD DE,16; LD BC,16; LDIR;
JR −2; a 13-byte transmitter at L = 16) and the 20-byte 8080 closer copy exactly under zero, random-register and
random-SP conventions (culture test 1.00, checked before this registration). The 4-byte cores that evolve use the zero
registers the harness provides.
**C4a.** 20 random-register worlds at L = 16 (seeds 8101–8120) for 3,000,000 steps (Modal). *Prediction:* closure (most
heritable cells confined) in ≤ 5 of 20 by 3,000,000 steps.
**C4b.** Random registers, L = 16: the self-initialising transmitter (random 3-byte tail per cell) seeded at 1% into a
world filled with the pusher, 5 seeds, 50,000 steps; and the pusher at 1% into a closer-filled world, 5 seeds.
*Prediction:* the closer holds > 50% of cells and most heritable cells are confined at 50,000 steps in ≥ 4 of 5; the
pusher stays below 5% in ≥ 4 of 5.
**C4c.** Standard convention, L = 16: the 13-byte self-initialising transmitter against the 4-byte regenerator `44 5e ed
b0`×4, 1% each way, 5 seeds each, 100,000 steps, standard mutation. *No direction registered* (a test of core length:
13 executed bytes against 4).
Endpoints: share of cells carrying each design (pusher: Hamming ≤ 4 at any shift; self-initialising closer: its first 11
bytes with at most one mismatch; regenerator: core `5e ed b0` at bytes 1–3 with a regenerating offset byte), every 250
steps; classes of 64 random cells at the end under the world's convention. Cost ≈ $15.

### Outcome of E (appended 2026-10-09; `results/offset/REPORT.md`, `offset_switch.csv`; 90 worlds on Modal, L40S)
E1 not met at any condition (mutation off, T share in 0.3–0.7 at 100,000 steps in 3, 1 and 2 of 10): without mutation
the share drifts to either side (end values 0.22–0.98, 0.00–1.00, 0.00–0.97), as a neutral pair would at this population
size. E2 not met (mutation on, T share in 0.05–0.4 in 3, 0 and 6 of 10). With mutation on, the two benign-or-lethal
conditions where core cells persist converge from every start to a condition-specific share: L = 32 benign, T share
0.07 (0.04–0.31) from 1% R, 0.08 (0.04–0.37) from 1% T and 0.06 (0.03–0.14) from 50:50, below the offset-byte mutation
equilibrium 0.18; L = 16 lethal, 0.87 (0.80–0.90), 0.81 (0.79–0.82; core cells lost late in 3 of 5) and 0.86 (0.82–1.00).
At L = 16 benign with mutation on the first-four-byte classifier fails: core-carrying cells fall to 3–5% of the soup, so
the share is not interpretable there. Post hoc (`offset_encounter.py`, `results/offset/OFFSET_ENCOUNTER.md`; 4,096
encounters per entry, soup cells from the final mix50 soups): the pristine R and T are indistinguishable in every
encounter component measured (benign / lethal: copy into a soup partner 1.000 for both; exact survival as the partner of a
soup executor 0.026 and 0.028 / 0.025 and 0.024; core intact 0.082 and 0.086 / 0.534 and 0.539; a random single mutant
still copies 0.750 and 0.744 / 0.759 and 0.757). *Reading:* neither H-sel nor H-bias as registered. The pair is neutral
in encounters and drifts without mutation; with mutation the direction is set by the environment (regeneration at
L = 32 benign, transmission at L = 16 lethal), so selection acts through what the mutants of each type become.

## E-mech — the mutational flow between regeneration and transmission (written before the run, 2026-10-09)

*Question.* R and T are neutral in encounters, yet mutation pushes the soup to a condition-specific share. Is that
share predicted by where single mutants of each type go, measured in isolation?
*Design* (`offset_flow.py`, local GPU). For each condition (L = 16 benign, L = 16 lethal, L = 32 benign) and type (R as
in E; T with 64 random tails), every single-byte mutant (L positions × 255 values) is executed against 4 random partners;
each offspring is executed against a fresh random partner (gen2). Each mutant is labelled by the functional class of its
gen2 offspring: R (offspring periodic with period < L and carrying a block-copy core), T (offspring equal to its parent
at shift 0 and not periodic below L), or O (anything else), with the first-four-byte label also reported. Flows per
mutation: f_RT, f_TR (class switch), l_R, l_T (loss to O). The two-type neutral model with these flows predicts the
equilibrium T share x* solving f_RT (1 − x) − f_TR x − (l_T − l_R) x (1 − x) = 0.
*Predictions.* EM1: x* < 0.18 at L = 32 benign (regeneration favoured beyond the offset-byte bias). EM2: x* > 0.5 at
L = 16 lethal. *Kill.* Either failing: the mutational-flow explanation is withdrawn and the environment dependence is
reported without a mechanism.

### Outcome of E-mech (appended 2026-10-09; `results/offset/OFFSET_FLOW.md`, `offset_flow.csv.gz`)
As registered (random partners), EM1 failed: x* = 0.42 at L = 32 benign (needed < 0.18); EM2 was met only nominally
(x* = 0.59 at L = 16 lethal), and the random-partner flows are the same under benign and lethal tar (x* 0.583 and
0.589; f_RT 0.009 and 0.010, f_TR 0.003 and 0.003). The registered kill applies: the flows of isolated mutants do not
explain the environment dependence, and it is reported without a mechanism. A check run beside it, with partners drawn
from each condition's own final soups, gives x* = 0.16 at L = 32 benign and 0.75 at L = 16 lethal, but it is circular:
the pools are already at the condition's equilibrium, and a mutant whose copy fails leaves the partner, often a cell of
the majority type, in place. It is not used as evidence. What the environment acts through (the soup's background, the
partial overwrites of byte 0 by halted executors, or a population-level effect) is open.

### Outcomes of B, T, P and F (appended 2026-10-09; `results/invasion_aligned/REPORT.md`, `results/review_r2/FIRST_CLOSURE.md`, `CENSUS_P2.md`, `CENSUS_P3.md`, `ROBUST_REF22.md`)
**B.** B1 met at both lengths (5 of 5): seeded at 1% into a pusher-filled world, the return closer at L = 16 holds
0.855–0.907 of cells at 20,000 steps and the block-copy tiling at L = 32 0.639–0.811, with heritable cells confined
(1.00 in all ten); the closer class first passes half the cells at 400–750 steps (L = 16) and 260–290 (L = 32). B2 met
(5 of 5 at both lengths): the pusher seeded into a closer-filled world is at 0.000 at 20,000 steps. Unseeded pusher
controls keep 0.000–0.113 (L = 16) and 0.207–0.383 (L = 32) pusher-class cells. The selection claim stands at aligned
lengths.
**T.** Benign L = 16 (Stages G and L, 30 worlds) closes earlier than lethal L = 16 (Stage I, 10): KM medians 22,500 and
38,000 steps, log-rank χ² = 7.21, P = 0.007. The registered rule is met; the text still says "displaced" (descent not
shown).
**P.** P1 failed: one closed self-writer of period 3 exists, `b0 5e ed` tiled, a block copier whose copy offset is the
tape length (a transmitter). Period 2: 5 self-writers, all open (the load–push words). The smallest closed self-writer
therefore has period 3, not 4; Corollary 2.1 and Proposition 4 are restated on the census. [Period 4: running.]
**F.** F1 failed: by ref. 22's criterion (exact self-copy into the zero partner after k mutations, 512 instructions)
both pushers and block-copy finals score ≈ 0 at every k (medians 0.000–0.002). Post hoc, the matched pair of the
copy-offset switch: the transmitter scores 0.880, 0.585 and 0.341 at k = 1, 4, 8, the regenerator 0.000–0.001. Ref.
22's criterion measures transmission of mutations, not survival of function; the regenerator, which survives most
single mutations functionally, fails it by erasing them.

### Outcomes of C4b and C4c (appended 2026-10-09; `results/r4/R4_INVASIONS.md`, `r4_inv.csv`; Modal)
The registered design-share endpoints failed on definition, as the motif endpoints of A1 and E did: the
self-initialising closer's motif (its first 11 bytes, at most one mismatch) is below 5% of cells by 550–750 steps in
every world where it was the resident, because the high bytes of its 16-bit operands are neutral (addresses are taken
mod 2L) and drift in a transmitter. Its opcode skeleton is also gone by the end (≤ 0.001 of cells). The registered class
endpoint (64 random cells at the end, under the world's convention) is unambiguous.
**C4b** (random registers, L = 16, 50,000 steps). Closer seeded at 1% into the pusher world: the pusher falls below 5% of
cells by 650–1,300 steps in 5 of 5; at the end 0.81–0.91 of cells are heritable and 0.93–0.98 of those confined, and
0.930–0.936 carry an LDIR. Pusher seeded at 1% into the closer world: 0.000 at the end in 5 of 5 (met); heritable
0.84–0.89, confined 0.93–0.98. Unseeded pusher controls: pusher share 0.04–0.11, heritable 0.03–0.08, none confined.
Reading: under random registers, closure is not reached from a random start in 300,000 steps (C3), but once a
self-initialising closer is present it displaces the pusher within about a thousand steps and its descendants, a
diverse family of LDIR closers that set their own registers, hold the soup closed. The barrier is accessibility.
**C4c** (standard convention, L = 16, 100,000 steps). Neither design persists as such: the 4-byte regenerator's share is
0.003–0.119 with SI13 invading, 0.006–0.645 when it invades SI13 (above half the cells in one world, from 40,250
steps), and 0.039–0.095 alone; the 13-byte closer's motif is gone everywhere. Populations stay closed (heritable
0.88–0.97, confined 0.98–1.00) and 0.94–0.97 of cells carry an LDIR. Seeded at 1% into the SI13 world, the 4-byte
regenerator reaches 0.07–0.36 of cells by 20,000 steps in 5 of 5 (0.000–0.005 in the unseeded SI13 worlds); the reverse
invasion leaves no SI13 motif. With zero registers supplied the 4-byte core invades the 13-byte design's world, but no
winner is declared: the long-run population is a diverse block-copy family that neither motif endpoint sees.

### Outcome of C4a (appended 2026-10-09; `results/r4/R4_LONG.md`, `r4_long.csv`, `R4_CLOSERS.md`; Modal, L40S)
C4a not met: 6 of 20 random-register worlds were closed (most heritable cells confined) at 3,000,000 steps (registered:
≤ 5). Closed worlds by snapshot: 1 at 300,000 steps (seed 8105; with C3, 1 of 40 random-register worlds by 300,000), 3
at one million, 4 at two million, 6 at three million; the other 14 hold the open pusher (heritable share 0.05–0.16,
none confined). Post hoc (`r4_closers.py`, 32 heritable cells per closed world): in all 6 closed worlds every tested
cell carries a block move (LDIR or LDDR), 0.84–1.00 of cells execute a JP or JR, and a median of 5–7 bytes per tape is
never executed; in 5 of 6 the offspring is an exact whole-tape copy in a median of 1.00 of encounters (transmitters), in
one (8107) in none. The closers load their own source and offset (addresses are taken mod 2L, so only the low byte of
HL and DE matters), jump over bytes they never run and copy with a block move. A hand-built 6-byte closer, `2e 00 1e 10
ed b0` (LD L,0; LD E,16; LDIR), scores 1.00 / 1.00 in the culture test under both conventions, while the evolved 4-byte
core `44 5e ed b0` scores 0.34 / 0.13 under random registers. *Reading:* closure under random registers is not
impossible but slow (≥ 100-fold later than with zero registers at L = 16, where 20 of 20 closed by 300,000 steps), and
the closers that evolve there are transmitters carrying never-executed bytes, as under lethal tar.

### Correction to the outcome of S (2026-10-09, audit)
"The L = 50 closer's lineage stops copying at the third transfer" is wrong by one: generations 1 and 2 copy into every
partner, generation 3 (made at the third transfer) copies into 0.008 of partners, and the fourth transfer fails. The
manuscript says "at the fourth transfer".

## DZ — the lethality dial in the Z80 (written before any run, 2026-10-09; round-2 report, optional item 3)

*Question.* Two harsh conditions, lethal tar and random registers, gave closed populations of transmitters, and the
benign one regenerators; benign tar closed L = 16 earlier than lethal tar. Is there a dose–response, and does closure
timing track the open window as Model 5 says?
*Design* (`gen_dial_shader.py`, `dial_soups.py`, `modal_dial.py`). Derived shaders in which a zero byte fetched as the
first byte of an instruction halts the pair with probability p (a fresh draw per fetch), p ∈ {0, 0.01, 0.03, 0.1,
0.3, 1}. Validated before any soup (65,536 random pairs with 25% zero bytes): p = 0 equals the standard executor and
p = 1 the Stage I executor, exactly. L = 16, Stage G conditions (20,000 tapes, square lattice, 8,192 pairs per step,
128 instructions, the standard mutation), 10 worlds per p with seeds 9001–9010 (the same seeds at every
p), 300,000 steps. Top-3 tapes every 50 steps to 2,000, every 250 to 20,000 and every 2,500 after; snapshots at 2,000,
20,000, 100,000 and 300,000. Every assay runs under the world's own p.
*Endpoints.* t_rep: first sample with a heritable top-3 tape holding ≥ 0.5% of cells; whether that first replicator is
open (its pointer enters the partner in at least half of 16 traced encounters). t_closed: first sample with a top-3 tape
(≥ 0.5%) heritable and confined in 16 of 16 encounters; censored at 300,000. Population classes at 300,000 steps (64
random cells; up to 16 heritable cells scanned): share of heritable cells that are transmitters (confined, ≥ 5
transmissible sites) and regenerators (confined, ≤ 2).
*Predictions.* DZ1 (the open beginning): the first replicator is open in ≥ 8 of 10 worlds at p = 0 and ≤ 2 of 10 at
p = 1, and the number of open-first worlds falls with p (Cochran–Armitage trend on the rank of p, one-sided P < 0.05).
DZ2 (Model 5, closure timing tracks the window): first closure is later at higher p: Kaplan–Meier medians non-decreasing
across the six levels up to one reversal, and Spearman correlation between p and t_closed (censored values at the
horizon) positive with permutation P < 0.05. DZ3 (the environment's choice): the transmitter share of heritable cells at
300,000 steps rises with p (Spearman, permutation P < 0.05); transmitters are the majority of heritable cells in ≤ 3 of
10 worlds at p = 0 and ≥ 7 of 10 at p = 1.
*Kill.* DZ2 failing: Model 5's window reading of closure timing is restricted to the benign/lethal contrast. DZ3 failing:
"the environment chooses" is restated as a contrast of two conditions, not a dose–response. *Cost.* 60 worlds × ≈ 3 min
on L40S ≈ $6.

## C4a-rep — replication of closure under random registers (written before the run, 2026-10-09)
*Why.* The independent figure critic noted that the 6 closed C4a worlds are all odd seeds (8101–8111); the chance of
that split is about 0.005, though it was noticed post hoc. Batch seeds come from SplitMix64 of the world seed and the
wall times show no split by GPU type, so no mechanism is known; a replication settles it. *Design.* 20 new
random-register worlds at L = 16, seeds 8201–8220, 3,000,000 steps, as C4a, with the GPU model recorded per world.
*Prediction.* Closure (most heritable cells confined) by 3,000,000 steps in 2–12 of 20; closed worlds not concentrated in
one seed parity (both parities represented if ≥ 4 close). *Reading.* Fewer than 2 closures: the C4a rate is reported as
an upper estimate, pooled with the replication.

### Correction to the outcome of E (2026-10-09, figure critic)
The core was lost (fewer than 1,000 core cells after 20,000 steps) in 7 of the 20 lethal-tar worlds with mutation, not
3: 1 of 5 started from 1% regenerators, 3 of 5 from 1% transmitters and 3 of 10 mixtures. At L = 16 benign with mutation
it was lost in all 20. Fixed-time shares, counting only worlds with ≥ 1,000 core cells (`offset_switch_times.py`,
`results/offset/SWITCH_TIMES.md`): L = 16 lethal, 0.71–0.92 at 20,000 steps in all 20 worlds and 0.79–0.92 at 300,000 in
the 13 that kept the core; L = 32 benign, 0.03–0.37 at 300,000 in all 20.

## PE — parent × environment serial transfer (written before the run, 2026-10-09; third report, top request)

*Question.* Is the regenerate-or-transmit mode a property of the parent, or of the environment it is copied in?
*Design* (`pe_matrix.py`, local GPU; the serial-retention machinery of S: every single-byte mutant, 16 lineages each, 256
unmutated control lineages, eight serial transfers, five-byte window allele). Six parents at L = 16: the pusher `01 c5` × 8;
the evolved return closer (`ad e3 21 e3 21 c0 ad c0` × 2); the block-copy regenerator R = `44 5e ed b0` × 4; the matched
transmitter T = `b0 5e ed b0` + 12 fixed random bytes; the evolved lethal-tar closer of world 4009; the hand-built
self-initialising closer `2e 00 1e 10 ed b0` + 10 fixed random bytes. Six environments: (1) benign rule, uniform random
partners; (2) lethal rule (zero halts), random partners; (3) benign rule, partners drawn from the final soup of Stage G
world 2001 (closed); (4) lethal rule, partners from the final soup of Stage I world 4001; (5) benign rule, partners from
the open phase of world 2001 (the recorded snapshot nearest step 5,000); (6) random registers, random partners.
*Endpoints* per parent and environment: unmutated lineages alive after eight transfers; transmissible sites after one
and eight transfers (S's definition); shares of mutant lineages lost, erased and carrying the allele at eight.
*Predictions.* PE1 (the mode belongs to the parent): wherever a parent's unmutated lineages survive (≥ 0.5 alive at
eight transfers), the two regenerators carry ≤ 2 sites after eight transfers and the three transmitters ≥ 8. PE2: the
pusher's lineages are lost (≤ 0.2 alive at eight) in every environment. PE3 (descriptive): under random registers only
the self-initialising closer survives (≥ 0.9 alive), the zero-register closers ≤ 0.1. *Kill.* PE1 failing in any
environment: the mode is reported as environment-dependent there.

### Outcome of PE (appended 2026-10-09; `results/pe/PE_MATRIX.md`, `pe_matrix.csv`; local GPU)
PE1 met: in all 24 parent–environment cells in which the unmutated lineages survived (all with 1.00 alive at eight
transfers), the regenerators carried 0 sites after eight transfers and the transmitters 9–12 (block-copy T 12 in every
environment, the lethal-tar closer 9, the self-initialising closer 10–11). PE2 not met: the pusher's lineages were lost
everywhere except with partners drawn from the closed benign world, where 0.36 of them were alive at eight transfers
(0.00–0.04 elsewhere), still with 0 sites. PE3 as described: under random registers only the self-initialising closer
survived (1.00, 10 sites); the zero-register closers and the pusher 0.00–0.01. Also: the return closer dies under the
lethal rule (0.00 alive with random partners and with lethal-world partners), while both block copiers survive it.
*Reading:* the mode of heredity belongs to the parent; the environment decides which parents survive.

## E-L32L — the matched pair at L = 32 under lethal tar (written before the run, 2026-10-09)
*Why.* The two informative conditions of E differ in both length and tar (L = 32 benign, regenerators favoured; L = 16
lethal, transmitters favoured), so "the environment chooses" is confounded with length. *Design.* As E, at L = 32 with
lethal tar (shaders derived by `gen_lethal_shader --tape 32`): R = `04 5e ed b0` × 8, T = `a0 5e ed b0` + 28 random
bytes; ten 50:50 worlds and five 1% invasions each way, standard mutation, 300,000 steps (`offset_switch.py`,
`modal_offset.py --part l32lethal`). *Prediction* (environment, not length): the transmitter share among core-carrying
cells at 300,000 steps exceeds 0.5 in ≥ 7 of 10 mixtures (L = 32 benign: 0.03–0.14). *Kill.* If it is below 0.4 in ≥ 7
of 10, the matched-pair evidence for an environmental choice is withdrawn and the L = 16/32 contrast is reported as
possibly a length effect.

### Outcome of C4a-rep (appended 2026-10-09; `results/r4/R4_REP.md`, `r4_rep.csv`, `R4_REP_CLOSERS.md`; Modal)
Met: 2 of 20 closed by 3,000,000 steps (seeds 8204 and 8218, both even, first closed at one million); none by 300,000.
The odd-seed concentration of C4a does not replicate. Pooled: 8 of 40 random-register worlds closed by three million
steps, and 1 of 60 by 300,000 (C3, C4a and C4a-rep). Closers (32 heritable cells each): 8218 is like the C4a closers
(whole-tape exact copy 1.00, LDIR, a jump, 6 never-executed bytes); 8204 is a different design, a loop through a
conditional jump around a single block-move step (`ed a8`, LDD; 0.06 of cells carry LDIR or LDDR), with 7 never-executed
bytes and no whole-tape copy. Across the 8 closed worlds, whole-tape transmitters in 6.

### Outcome of E-L32L (appended 2026-10-09; `results/offset/REPORT.md`, `SWITCH_TIMES.md`; Modal)
Met: at L = 32 with lethal tar the transmitter share among core-carrying cells at 300,000 steps is above 0.5 in 10 of 10
mixtures (0.55–0.87; median 0.77), and converges there from both 1% starts (from 1% regenerators 0.62–0.79, median 0.76;
from 1% transmitters 0.58–0.81, median 0.78); no world lost the core (core-carrying cells 0.88–0.93 of the soup at the
end, medians). Against L = 32 with benign tar (0.03–0.37 in all 20 worlds, medians 0.06–0.08), the tar alone reverses
the equilibrium at the same length. The confound with length is removed.

### Outcome of DZ (appended 2026-10-09; `results/dial/DIAL.md`, `dial_worlds.csv`; 60 worlds on Modal, A10/A10G)
**DZ1 met.** The first replicator is open in 10 of 10 worlds at p = 0, 0.01, 0.03 and 0.1, in 2 of 10 at p = 0.3 and in
0 of 10 at p = 1 (Cochran–Armitage z = −6.10). The open beginning ends at a threshold between p = 0.1 and 0.3, the
same interval as the BFF lethality dial. At p = 0.1 every first replicator is the load–push word `e5 2a` (PUSH HL;
LD HL,(nn)), still open, arriving at a Kaplan–Meier median of 5,000 steps (250–400 at p ≤ 0.03); at p = 0.3 most first
replicators are closed block copiers (KM 40,000), at p = 1 all are closed (KM 52,500; one world without a replicator).
**DZ2 partly met.** Spearman(p, t_closed) = 0.25, permutation P = 0.023 (met), but the KM medians are 18,000, 42,500,
60,000, 75,000, 57,500 and 52,500 steps: they rise to p = 0.1 and fall twice after it, so the registered monotone part
fails. The kill applies: Model 5's window reading of closure timing is restricted to the contrast of the extremes; a
damaged open phase (0.01 ≤ p ≤ 0.1) delays closure relative to both no lethality and full lethality.
**DZ3 partly met.** Spearman(p, transmitter share of heritable cells at 300,000) = 0.50, permutation P < 5 × 10⁻⁵ (met);
medians 0.00, 0.00, 0.00, 0.26, 0.56, 0.78; regenerator medians 1.00, 0.60, 0.41, 0.00, 0.09, 0.03. Transmitter-majority
worlds: 1 of 10 at p = 0 (met, ≤ 3) and 6 of 10 at p = 1 (registered ≥ 7: not met). The dose–response is supported by
the registered trend test; the endpoint count missed by one world.

## E-census — where the tar acts (exploratory, written before the run, 2026-10-09; no prediction registered)
At L = 32, the final soups of the ten 50:50 mutation-on worlds of each tar are sampled as lattice-neighbour pairs (organism
a random cell, partner one of its four neighbours, torus), 200,000 pairs per soup set, and executed under each rule
(benign, lethal), crossing composition with rule. Every cell before and after is classed by its first four bytes (R, T,
core with another offset, no core). Reported: per encounter, the probability that a partner of each class keeps its
class, becomes the other class or loses the core, by the executor's class; and the expected one-encounter change in the
transmitter share. The aim is to locate the asymmetry (composition, rule, or their interaction); results are descriptive.

### Outcome of E-census (exploratory; `results/offset/ENCOUNTER_CENSUS.md`, `INTRUDER.md`)
One round of encounters in the equilibrium soups changes the transmitter share by at most 5 × 10⁻⁴ under either rule and
composition: the equilibrium is maintained by slow processes, and one round does not locate it. One asymmetry is clear
(`offset_intruder.py`, 16,384 encounters per row, executors the core-free cells of each tar's soups): as partners of
executors whose pointer enters them, regenerators keep their class in 0.05–0.24 of encounters and transmitters with a
random body in 0.35–0.49; a transmitter whose body is tiled copies of its core (`a0 5e ed b0` × 8) is as vulnerable as
the regenerator (0.07–0.25), and where the executor does not enter, all three keep their class alike (0.08–0.15). The
lethal rule widens the transmitters' overall advantage (benign-soup executors: 0.154 against 0.092 benign, 0.208 against
0.126 lethal). Reading: a body made of copies of the copying code is hijacked by intruders, and a body of junk protects
against them; this favours transmitters under both rules, more under lethal tar, and so does not by itself explain why
regenerators win under benign tar. Post hoc and descriptive.

### Correction to the outcome of DZ (2026-10-09, figure critic)
At p = 1 one world (9004) has no heritable top-3 tape, so DZ1's count is 0 of 9 worlds with one. The registered closure
endpoint (a top-3 tape at ≥ 0.5% of cells) misses closed transmitter populations: worlds 9007 (p = 0.03) and 9004
(p = 1) are counted as unclosed, yet all their heritable random cells are confined (transmitters 1.00 and 0.875 of
heritable cells). The DZ2 timing test is therefore biased against closure at high p; it is reported as registered, with
this bias stated.

## N1 — the toxic payload: do the bytes a transmitter never runs defend it? (written before any payload byte was inspected, 2026-10-09 night)
Hypothesis (H_tox). A transmitter (core `XX 5e ed b0`, offset d = L) copies d − 4 payload bytes that its own pointer
never fetches (its LDIR, BC = 0, runs to the end of the budget). An executor that falls through into it (fall-through
enters B at its first byte, runs the core, and continues into the payload when its own BC lets the LDIR finish) is
stopped by any byte that halts the pair: 0x76 (HALT) under both rules, 0x00 under lethal tar. Such bytes cost the owner
nothing. H_tox: they are selected in payloads because they stop intruders. Rival (H_dam): zeros enter payloads by damage
(executors push register contents, mostly zero, from the stack top, which is the END of the partner, `sp_init`); 0x76 is
not written by damage. Rival (H_neu): payload bytes are neutral (founding payloads are uniform random bytes; mutation draws
uniform bytes, `mutate_soup`).

Data (no new runs): the final soups of the E runs (`runs/offset/offset/*_final.npy`; L = 16 and 32, benign and lethal,
mutation on: 10 mix50 + 5 R_into_T + 5 T_into_R per cell; mutation off: 10 mix50 at L = 16 both tars and L = 32 benign)
and the lethality-dial worlds (`runs/dial/dial/`, L = 16, p ∈ {0, 0.01, 0.03, 0.1, 0.3, 1}, ten per p, snapshot at 100k).
Transmitter = `classify(...)` isT (L = 32: first byte 0x60 or 0xa0; L = 16: 0x50, 0x90, 0xb0; then `5e ed b0`); in the
dial worlds, cores sit anywhere, so there a transmitter is any cyclic shift of a tape whose bytes 0–3 at that shift are
`XX 5e ed b0` with XX mod 32 = 16 (L = 16). Payload = positions 4 … L − 1 after the core. A world enters a test only with
≥ 100 transmitter cells. The unit of inference is the world.

Statistics per world: Z = share of payload bytes equal to 0x00; H = share equal to 0x76; per-position profiles of both.
Null under H_neu: 1/256 = 0.0039 for each value.

Predictions of H_tox (registered):
- **N1-1 (damage-free marker).** In mutation-on mix50 worlds at L = 32 under benign tar, H exceeds 3/256 in ≥ 6 of the
  worlds that enter; same at L = 16. (Damage does not write 0x76; H_neu predicts 1/256.)
- **N1-2 (tar).** Median Z in lethal mix50 worlds ≥ 3 × median Z in benign mix50 worlds, at L = 32 and at L = 16.
- **N1-3 (position).** Hazard h_j = share of entering executors whose pointer fetches payload position j, measured by
  tracing 16,384 encounters per condition (executors = random cells of the same final soups, hosts = the soup's own
  transmitters as partner B, the soup's rule). H_tox: the per-position share of halting bytes (0x76; plus 0x00 under
  lethal) correlates positively with h_j across positions (Spearman ρ > 0.3, pooled over worlds by position, per
  condition). H_dam: zeros concentrate at the last payload positions (stack writes) rather than at high-hazard positions.
- **N1-4 (dose).** In the dial worlds at 100k steps, Z of transmitter payloads rises with p (Spearman ρ > 0 over worlds
  that enter, one-sided P < 0.05).
- **N1-5 (decision).** Composition supports H_tox if N1-1 holds, or if N1-2 and N1-3 both hold. It supports H_dam if zeros
  are enriched only at the stack-write tail and N1-1 fails. Anything else: composition undecided. Composition alone is
  never taken as proof; N1-6 decides.
- **N1-6 (causal, encounters).** Hosts: up to 400 transmitter tapes per condition, sampled from the final soups in
  proportion to abundance. Variants: native; detox (every payload 0x00 and 0x76 replaced by a uniformly random byte
  outside {0x00, 0x76}); sham (the same number of payload positions, chosen among the other payload bytes, re-drawn
  outside {0x00, 0x76}). Each host is the partner B of 64 executors drawn from its own soup, under its soup's rule; also
  each host executes as A against 64 partners (owner-side check). Outcome: the host keeps its class and its payload
  (all positions except those its own variant changed) after the encounter. H_tox: under lethal tar, native − detox ≥
  0.02 (bootstrap 95% CI over hosts excludes 0) and |native − sham| < 0.01; under benign tar the same holds for hosts that
  carry 0x76. Owner-side: native and detox produce identical copies as A in every encounter (payload never executed).
- **N1-7 (causal, population; registered here, launched only if N1-6 holds).** From the ten lethal L = 32 mix50 final
  soups, two arms per soup with new seeds: detox every transmitter's payload, or sham; 100,000 steps, snapshots as in E.
  H_tox: the transmitter share of core carriers in the detox arm falls below the sham arm by ≥ 0.10 at some snapshot in
  ≥ 7 of 10 soups, and the payload Z of the detox arm climbs back towards the sham arm's.

### Outcome of N1 (appended 2026-10-09 night; `results/toxin/COMPOSITION.md`, `CAUSAL.md`, `*.csv`; `payload_toxin.py`)
H_tox is rejected; the composition favours H_dam (N1-5), and the causal test finds a real but negligible protection.
N1-1 failed: 0x76 is not enriched in benign payloads (median H 0.0038 at L = 32, above 3/256 in 1 of 10 worlds; 0.0018
at L = 16, 2 of 9). N1-2 failed at L = 32 (lethal median Z 0.052 against benign 0.067; ratio 0.77) and was below the
registered factor at L = 16 (0.060 against 0.038; ratio 1.57). N1-3 failed: across payload positions the share of halting
bytes does not rise with the intruders' hazard (ρ = −0.76, −0.04, +0.02, −0.03); zeros rise with position instead (ρ =
+0.66 to +0.86), to 0.32–0.38 at the second-last byte at L = 32 with an even–odd pattern, the footprint of words pushed
from the stack top, which sits at the end of the partner. N1-4 failed at the registered snapshot (dial, 100k: ρ = −0.19,
15 worlds); at 300k ρ = +0.39 (one-sided P = 0.04, 21 worlds), secondary. Mutation-off worlds carry no payload zeros at
all (clonal survivors). N1-6 failed on its registered size: under lethal tar removing the halting bytes lowers the
host's survival as an intact partner by 0.0009 (L = 32; 95% CI 0.0005–0.0013) and 0.0012 (L = 16), the sham changes
nothing, and the owner never fetches its payload (0 of 25,600 encounters per variant) and copies exactly in all of them.
*Reading:* payload zeros are scars of intrusions that transmitters inherit and regenerators erase; their protective value
is about a thousandth per encounter. The environment's choice between regenerators and transmitters is not explained by
a toxic payload. N1-7 is not launched.

## N3 — the line of descent of the first self-confined replicators (written before the recorder was run, 2026-10-09 night)
*Question.* Do the closed (execution-confined) replicators that take over a benign-tar world descend from the open
replicators that came first, and by what event was the first confined copier made?
*Method* (`lod.py`, no shader change). Every step of a world is recorded exactly: the pairs drawn and their active flag,
each active pair's memories before execution (the soup before the step), after execution and before mutation
(`read_pair_memory`), and the soup after mutation. Each changed cell gets a birth record: *copy* (its new tape is within
Hamming L/4 of the executor A's tape at some cyclic shift, and closer to it than to its own old tape), *damage* (within
L/4 of its own old tape), *novel* (neither; both tapes are parents, with per-position attribution), or *mutation* (a
point mutation after the encounter; parent = its record before). Every copy event is re-run on the traced executor:
the executor's record is flagged *confined copier* if its pointer fetched no partner byte, *open copier* otherwise.
Records no living cell descends from are freed; the rest form the exact ancestry graph. The population measure of
closure is the share of copy events whose executor stayed confined (per 500 steps).
*Validation, before any registered run* (V1–V3 must all hold; otherwise no claim is made):
- V1 (known ancestry): the return closer `ad e3 21 e3 21 c0 ad c0` × 2 seeded at 1% into a world of the pusher `01 c5`
  × 8 (L = 16, Stage G dynamics, 5,000 steps): ≥ 99% of 64 sampled confined copiers at the end trace by their line of
  descent (major parent at novel events) to a seeded closer; the pusher's copy events are open in ≥ 95%, the closer's
  confined in ≥ 95%.
- V2 (exact replay): re-running every active pair of the first 200 steps on the traced executor reproduces the recorded
  post-execution memories byte for byte.
- V3 (accounting): per step, cells changed outside active pairs number at most the mutation count.
*Registered runs.* 20 new benign-tar worlds at L = 16, zero registers, Stage G dynamics (seeds 31001–31020), recorded
from step 0 until the confined share of copy events has exceeded 0.5 for 2,000 consecutive steps, then 5,000 more steps
(cap 80,000). At the end, 64 cells are sampled among those whose record was flagged confined in the last 500 steps, and
each line of descent is walked back to step 0. C* = the earliest record on the line flagged confined.
*Predictions.* N3-1 (descent): in ≥ 15 of the worlds that close, the line of descent of the majority of sampled confined
copiers passes through a record flagged open copier before C*. N3-2 (the founding event, my prior, registered as a
prior): C* is made in most worlds by an encounter in which an open copier wrote into a partner and the result keeps
bytes of both (a *copy* or *novel* record whose executor parent is an open copier), not by a point mutation of an open
copier's own tape. Reported regardless: the kind of each C* event, the executor's and partner's tapes, which positions of
C* came from which parent, and the base rate (share of living cells at C*'s birth whose line of descent contains an open
copier within 2,000 steps). *Decision.* N3-1 met: "closed replicators descend from the open ones" may be written.
Independent origin is written only if in ≥ 15 worlds no open copier lies on the line. Otherwise mixed, reported as such.

### Validation of N3 (appended before any registered world was run, 2026-10-09 night; `runs/lod/validate/validate.json`)
V2 met: re-running every active pair of the first 200 steps (795,934 pairs) on the traced executor reproduces the recorded
post-execution memories byte for byte (0 mismatches). V3 met: at most 339 cells per step change outside active pairs
(mutation count 512). V1, ancestry, met: at 5,000 steps all 64 sampled confined copiers trace by their line of descent to
a seeded closer. V1, flags, not met as registered: copy events whose executor descends from the seeded closer are
confined in 0.922 (needed ≥ 0.95), and those whose executor descends from the pusher are open in 0.799 (needed ≥ 0.95).
The registered flag criterion counted lineages, and lineages change: pusher-rooted lineages made 61,399 confined copies
(recombinants, mostly of pusher material, that copy while confined), and closer-rooted mutants that enter the partner.
*Amendment (post hoc, made before the registered worlds; stated as such).* The confinement classifier itself is checked
on genotypes: the exact pusher enters the partner in 1.000 of 4,096 encounters against random, pusher and closer
partners alike; the exact closer in 0.000 of each, and both copy exactly. The classifier is therefore exact; the
lineage-level flag rates are a property of the dynamics and are reported, not used as a gate.

## N2 — in-situ demography of regenerators and transmitters (written before the run, 2026-10-09 night)
*Question.* Isolated single mutants flow alike under both rules (E-mech), and one round of encounters barely moves the
transmitter share (E-census), yet the equilibrium share differs threefold between tars. Which per-capita rate differs?
*Design* (`n2_demography.py`, local GPU). The ten L = 32 mix50 mutation-on final soups of each tar are continued 2,000
steps under their own rule and under the other rule (composition × rule, 40 runs), recording every step exactly (pairs,
pre-execution, post-execution pre-mutation, post-mutation). Classes by the first four bytes: R, T (as in E), M (the core
with another first byte), O (no core). Per class and per cell-step: births (an executor of the class converts a partner
of another class to its class), conversions lost to each other class's executors, losses to O by encounter (damage) and
by mutation, and gains by mutation. Per-capita net growth g_c; the decomposition of g_T − g_R into components.
*Hypothesis H_backup (registered).* A regenerator's body is backup copies of its core; when its first core is damaged it
can still copy by running on into a backup, which works when zeros are NOPs (benign rule) and fails when a zero halts
(lethal rule). Predictions: N2-1, under the benign rule the per-capita rate at which R cells are lost to O (encounter
damage plus mutation, net of mutational gains) is lower than T's, and under the lethal rule it is not lower, in both
compositions (the rule, not the composition, sets the sign). N2-2 (direct test): among R and T mutants whose first core
carries a zero at one of positions 0–3, the share that still copies itself exactly as executor against partners from the
benign soups is higher for R than for T under the benign rule, and the R advantage shrinks under the lethal rule.
*Alternatives reported regardless:* the component of g_T − g_R that changes most between rules, whatever it is.

### Amendment to N3 (2026-10-09 night, after inspecting the founding event of the first finished registered world)
The registered copy rule (new tape within L/4 of the executor's at some shift, and closer to it than to its own old tape,
ties to copy) misclassifies damage as copying when the partner already resembles the executor: in world 31016 the event
that flagged C* as a confined copier (step 3,516) wrote two zero bytes into the tail of a partner that was already 14/16
identical. Amended rule: a copy must also make at least L/4 positions newly match the executor at the copy's shift
(gain ≥ 0.25); everything else as registered. Re-validated (`runs/lod_v2/validate/validate.json`): V2 0 mismatches in
808,114 pairs; V3 ≤ 339; V1 ancestry 64 of 64 to the seeded closer; pusher-rooted copy events open in 0.998 (341
confined against 139,898; under the registered rule 61,399 confined); closer-rooted copy events confined in 0.934 (closer
mutants that enter the partner). The 20 registered worlds are reported under the registered rule and re-run under the
amended rule (same seeds, new output `lod_v2`); both are reported, and the amended rule is the one that measures copying.

### Outcome of N2 (appended 2026-10-09 night; `results/n2/DEMOGRAPHY.md`, `BACKGROUND.md`, `BACKUP.md`)
N2-2 failed: a regenerator with a zero at any of positions 0–3 never copies itself under either rule (0 of 4,096; its
LDIR, BC = 0, consumes the budget before a backup core is reached); H_backup is rejected. N2-1 failed (R's net loss to O is
not lower under the benign rule). The decomposition (post hoc, descriptive): direct R↔T copying is neutral in counts
(within 0.3% in every condition). The background terms that differ are (i) destruction by core-free executors, lower for
T than R (junk resists intruders), partly offset by recolonisation of the emptied cells; (ii) conversion of intruders,
higher for R (an executor that runs into a tiled body of cores is turned into a regenerator; about 3 × 10⁻⁴ per capita
under both rules); (iii) small mutational terms. The lethal rule reduces destruction of T by core-free executors by 9–13%
and of R by 0–6%; summed, the terms predict the observed 2,000-step drift (benign soup under the lethal rule: predicted
+0.031, observed +0.039; lethal soup under the benign rule: predicted −0.059, observed −0.049; own rules ≈ 0).

## N4 — do inherited payload zeros carry the lethal rule's protection of transmitters? (written before the run, 2026-10-09 night)
*Design* (`n4_scars.py`, local). Executors: core-free cells (class O) of the final L = 32 mix50 soups of each tar.
Hosts as partner B, drawn from the same soups: T native (as found), T detox (payload zeros replaced by random non-zero,
non-0x76 bytes), T sham (as many other payload positions re-drawn), and R (as found). 65,536 encounters per host type ×
rule × soup tar. Outcome: destruction = the host loses its class (first four bytes no longer its class).
*Prediction N4-1:* the reduction in T destruction from the benign to the lethal rule is at least twice as large for
native as for detox hosts (both soups). *N4-2:* sham behaves as native (within 25% of the native reduction).
*Kill:* if detox shows ≥ 75% of the native reduction, payload zeros do not carry the rule's protection and the
"scars become a shield" reading is withdrawn.

### Outcome of N4 (appended 2026-10-09 night; `results/n2/SCARS.md`)
Killed as registered: removing the payload zeros keeps 88–94% of the lethal rule's reduction in transmitter destruction
(benign soups: native +0.055, detox +0.048, sham +0.054; lethal soups: +0.041, +0.039, +0.041), so inherited zeros do not
carry the protection. Descriptive (post hoc): most destruction by core-free executors happens without entering (stack
writes); among executors that enter, a regenerator is destroyed in 0.86–0.89 of encounters under either rule (the
intruder runs the tiled LDIR to the end of the budget), a transmitter in 0.58 (benign rule) and 0.48 (lethal rule),
because an intruder wandering through junk is halted by a zero under the lethal rule. The lethal rule lowers destruction
more for transmitters (by 0.041–0.055) than for regenerators (0.028–0.036).

### Addition to N3 (2026-10-09 night, before any amended-rule world was inspected)
Under the registered copy rule, the C* events of the first six worlds are artefacts: none copies itself against random
partners (copy rate 0.00–0.11; three are confined loops that never copy, three enter the partner), and hundreds to
thousands of open-copier records lie on each line after C*. Two changes, made before any amended-rule result was seen:
(i) the founding tape of every C* is now tested directly (64 random partners: enters the partner, copies ≥ 0.75 at best
shift, copies exactly); (ii) a second founding record is reported beside C*: C** = the oldest confined record newer than
the newest open-flagged record on the line (the start of the final confined stretch). Lines now store their full step,
kind and flag arrays and sampled tapes (`lod_v3`, same seeds, amended copy rule). The registered N3-1 is evaluated on C*
as registered; C** is reported as an addition.
Implementation note (2026-10-09 night): memory grows with the ancestry graph (7.1 GB at 52,000 steps under the registered
rule, near the container limit), so the minor parent of a novel record is kept as its tape and flag only (the line of
descent follows major parents) and the container memory is raised (`lod_v4`, same seeds, amended copy rule, C**). No
change to any rule or measure.
Implementation note (2026-10-09 night): the sampled tapes stored per line skip the founding encounters (in world 31012 the
sampled ancestors jump 10 bytes across the transition), so `lod_v5` (same seeds, amended rule, nothing else changed)
also stores every record on the 64 sampled lines (step, kind, flag, tape, overwritten tape, minor parent's tape) and the
soup every 2,000 steps. The redundant `lod_v4` was stopped.

### Outcome of N3 under the registered copy rule (appended 2026-10-09 night; `results/lod/registered_rule/LOD.md`)
18 of 20 worlds closed by the registered measure (31006 and 31017 reached the 80,000-step cap), and in all 18 the
sampled lines pass through open copiers before C*: N3-1 is met as written. It is uninformative, and is not used as
evidence: (i) only 1 of the 24 distinct C* tapes copies itself against random partners (12 enter the partner, the rest
are confined loops that never copy; the registered copy rule counted damage to near-identical partners as copying), and
(ii) the base rate is 1.0 (median): by the time C* appears, every sampled living cell has an open copier within 2,000
steps of its line, so the criterion could not fail. Whether closers descend from open copiers is decided on the founders
of the final confined stretch, identified by genotype, under the amended rule (`lod_v5`).

### Amendment to N3: functional parents (2026-10-09 night, after inspecting lod_v3 worlds 31011 and 31013)
At a recombinant (novel) record the registered line follows the parent that supplied more bytes. When closers overwrite a
pusher cell in several partial writes, the line therefore stays with the pusher cell's material and the "transition"
jumps 12–16 bytes from a pusher to a complete closer (worlds 31011, 31013), so the line traces material, not function.
Amended (`lod_v6`, same seeds, amended copy rule): at a novel record the line follows the parent that supplied more of
the bytes the new tape fetches when it runs as A against a random partner (ties: more bytes overall); damage records now
store the writer's tape as `other`. Re-validated (`runs/lod_v6/validate/validate.json`): V2 0 mismatches; V3 ≤ 342; V1
ancestry 64 of 64 to the seeded closer. `lod_v5` (material parents) was stopped; `lod_v2`/`lod_v3` are kept and reported.

## N5 — the stepwise path from the open pusher to a confined copier (written before the run, 2026-10-09 night)
*Observation that prompted it (exploratory, `results/lod/LANDSCAPE.md`).* In the hypercube between the pusher `21 e5` × 8
(or `01 c5` × 8) and the return closers found on the lines of descent, the shortest single-byte path through copiers to a
confined copier is four steps, the same in all four cubes: PUSH → RET PO at position 5, then at 15, then PUSH → EX (SP),HL
at 9, then at 1. Copy rate against random partners rises (0.59 → 0.64 → 0.97 → 1.00 → 1.00 for `21 e5`) while the share of
encounters in which the pointer enters the partner falls (1.00, 1.00, 1.00, 0.62, 0.00).
*Question.* Is each step favoured in the soup, so that the path is an adaptive walk from heredity without confinement to
a confined copier?
*Design* (`n5_path.py`, local GPU). For each family (`21 e5` with RET PO; `01 c5` with RET PO), genotypes G0 (pusher) … G4
(confined). For each step k = 1 … 4: Gk seeded at 1% (random cells) into a world filled with G(k−1), and G(k−1) seeded at
1% into a world filled with Gk; Stage G dynamics (L = 16, benign, zero registers), 5 seeds per direction, 20,000 steps.
Shares by exact genotype class (Hamming ≤ 2 to the design at shift 0, nearest design wins) every 250 steps. Also, for each
genotype: the culture test (32 partners), exact copy rate and entering rate against random partners.
*Predictions.* N5-1: for each k, Gk exceeds 0.5 of cells at 20,000 steps in ≥ 4 of 5 seeds when seeded into G(k−1).
N5-2: for each k, G(k−1) stays below 0.05 when seeded into Gk in ≥ 4 of 5 seeds. *Reading.* Both for every step: the
path is an adaptive walk. A step that fails N5-1 is reported as neutral or deleterious, and the walk as requiring drift
or recombination there.
*Amendment to N5 (2026-10-09 night, after 60 of 80 runs).* The registered class measure (Hamming ≤ 2 at shift 0) counts
shifted copies as no class; pushers and return closers copy at cyclic shifts, so every design scored 0.000 at 20,000 steps
in all 60 runs, invader and resident alike: the measure is uninformative and is not used. Re-run with the paper's
convention (Hamming to each design at its best cyclic shift, ≤ 2, nearest design wins, ties to none); everything else as
registered; final soups saved (`runs/n5/`).
*Second amendment to N5 (2026-10-09 night, before any trajectory was inspected; only end-of-run shares had been printed).*
At 20,000 steps the worlds have evolved away from both designs (the pusher world closes on its own, as Stage G worlds do),
so resident and invader both read ≈ 0 and the registered endpoint confounds invasion with new evolution. Amended endpoint
(trajectories are recorded every 250 steps): the invader's share at 2,000 steps, in runs where invader + resident hold
≥ 0.5 of cells at that time; N5-1 and N5-2 are evaluated on it with the registered thresholds (≥ 0.5 in ≥ 4 of 5;
< 0.05 in ≥ 4 of 5). Runs that fail the ≥ 0.5 condition are reported as evolved away.
*Outcome of N5 (2026-10-09 night; `results/n5/PATH.md`).* Uninformative as designed, under both the registered and the
amended endpoints: the designs differ by single bytes in a soup that mutates each cell about 0.026 times per step, so
both the resident and the invader leave the Hamming ≤ 2 classes within a few hundred steps (median combined share 0.017
at 500 steps, 0.002 at 2,000). The genotype table is informative: along the four-step path, exact copying collapses
after the first step (0.52 → 0.003 → 0.000) and the last two genotypes are not heritable by the culture test (gen2 0.25
and 0.17, threshold 0.3). The exploratory "gradual path" of the landscape used a lenient copier criterion (≥ 0.75
similar copies) and is withdrawn; it is re-tested with heritability.

### Outcome of N3 under the amended copy rule (lod_v2; appended 2026-10-09 night; `results/lod/amended_rule/LOD.md`)
19 of 20 worlds closed; N3-1 is met as written (19 of 19) and is again uninformative (base rate ≈ 1). C* remains a
transient: of 25 distinct C* tapes, 7 copy themselves against random partners and 18 enter the partner; a median of
about 1,000 open-copier records follow C* on its line. The oldest confined record on a line is not where the confined
lineage begins; the founder is identified by genotype at the start of the final confined stretch, on functional-parent
lines (lod_v6).

### Outcome of N3 on functional-parent lines (lod_v6; appended 2026-10-10 00:35; `results/lod/founders_v6.csv` (the 32 distinct founders; every number below), `CHAIN_v6.md`, `PARTS_v6.md`)
17 of 20 worlds closed within 80,000 steps (exact replay held in all 20: 0 mismatches). On the 64 sampled lines of each
closed world, the founder F of the confined lineage (first confined copier, by genotype, after the last open copier
before the final confined stretch) was found in all 17 worlds: 32 distinct founders.
- *Descent.* Every founder's functional line runs back to open copiers: the pushers `21 e5` × 8 (18 founders), `01 c5` ×
  8 (5) or their EX (SP),HL variants (9). Closed replicators descend from the open ones (with the base-rate caveat that
  open copiers dominated the soup when the founders appeared).
- *How the founder was made (N3-2).* Completed by a recombination in 19, by a partial overwrite from a neighbour in 12,
  by a copy of the partner into the executor's half in 1, by a point mutation in none. The registered prior is right that
  founders are not point mutants and wrong about the writers: the founders' bytes were written mostly by tapes that cannot
  copy themselves (64% of all founder bytes; 7% by open copiers, 8% by confined copiers, 20% kept from the last open
  copier, 0.6% by mutation; per founder, median 78%, IQR 38–94%; 21 of 32 founders got at least half their bytes from
  non-copiers). Each founder's bytes came from a median of 4 distinct events (1–9), a median of 42.5 records and 240.5 steps
  after the last open copier on its line.
- *What was assembled.* In 16 of 17 worlds the founder is a return closer (`XX e3 21 e3 21 RET XX RET` × 2, the family
  of the paper's evolved L = 16 closer, made of the pusher's own load word `21` with EX (SP),HL and a return in place of
  PUSH); one world founded an LDIR block copier; one a push–return closer (`… 01 c5 01 c0 …`).
- *The parts were in circulation.* The two-byte words each founder executes were carried, in the last soup snapshot
  before it appeared, by a median of 0.25% of cells (about 10 times a random tape), 98% of the carriers being
  non-copiers; only 5 of the 32 founders' executed segments existed in any cell before the founder was assembled.
- *Reading.* The first self-confined copiers were not evolved within a lineage by point mutations: they were assembled,
  in cells that could not copy themselves, from words circulating in the open copiers' world, by recombination and
  partial overwrites from several neighbours. Exploratory support (`results/lod/LANDSCAPE.md`): in 3 of 4 hypercubes
  between a pusher and a return closer, no single-byte path through heritable genotypes reaches a heritable confined one.

### Correction to the outcome of N3 (2026-10-10, after the independent figure critic; `results/lod/FOUNDERS2.md`, `founders2.csv`, `chain_v6b.csv`)
The critic found two errors in the analysis reported above, and one missing null.
1. *Chain start.* `lod_chain.py` started each chain at the first open copier at or before the stable-stretch threshold,
   not at the newest open copier older than the founder; open copiers lay between them on 26 of 36 lines. Corrected.
2. *First founders and conversions were pooled.* All 12 founders completed by a partial overwrite are later founders in
   worlds that already held confined copiers, and 11 of the 15 later founders took bytes from a confined copier: they are
   conversions by existing closers, not new assemblies. First founders (the earliest in each world) are now reported
   separately.
3. *Null.* The share of founder bytes from non-copiers (0.88 per first founder, median) matches the share of non-copiers
   in the soup at the snapshot before (0.86): it reflects abundance and is not evidence of anything. The sentence "bytes
   written mostly by tapes that cannot copy themselves" is withdrawn as a finding.
Corrected outcome. *First founders, one per world (17 of 17 closed worlds):* completed by a recombination in 16, by a copy
of the partner in 1, by a point mutation in none; a median of 79 steps and 14 records after the newest open copier on the
line, with only non-copiers in between in 16 of 17; bytes from a median of 3 separate events (1–6), none from a confined
copier. The parts the founders execute are words such as `21 e3` and `21 e0`, which differ from the pusher's `21 e5` in
one byte (two bits): point mutation made the parts, recombination joined them. *Later founders (15 in 10 worlds):*
mostly conversions (12 by partial overwrite, 3 by recombination; 11 with bytes from a confined copier). The descent
statement stands: every founder's line runs back to open pushers.

## N6 — confirmatory replication of the founder result (written 2026-10-10, before any N6 world is run; awaiting spend approval)
*Why.* The N3 founder analysis (assembled by recombination, not point mutation) used definitions fixed after the N3 data
were inspected (copy-gain rule, genotype-defined founder, functional parents, chain start at the newest open copier, first
founders separated). It is therefore exploratory. N6 runs the frozen pipeline on new worlds.
*Frozen pipeline.* `lod.py`, `modal_lod.py`, `lod_traj.py`, `lod_chain.py` (chain start at the newest open copier older than
the founder), `lod_founders2.py` and `lod_parts_first.py` as committed at 8b492ae / 24ec712; no change to any definition
(GAIN = 0.25; functional parents; tape classes against 64 random partners; founder = first confined copier after the newest
open copier older than the final ≥ 90%-confined stretch; first founder = the earliest founder in its world).
*Runs.* 20 new benign-tar worlds at L = 16, zero registers, Stage G dynamics, seeds 32001–32020, recorded from step 0 with
the N3 stopping rule (confined share of copy events > 0.5 for 2,000 steps, then 5,000 more; cap 80,000). Modal, one world
per container (≈ 29 GPU-hours for N3's 20 worlds; estimate $25–35).
*Predictions* (on the first founder of each world that closes; n = number of closed worlds):
- N6-1 (how made): the completing event is a recombination or a copy of the partner (records `novel` or `copyA`) in
  ≥ 0.8 n worlds, and a point mutation (`mut`) in ≤ 0.1 n.
- N6-2 (through non-copiers): only non-copiers lie between the newest open copier and the founder in ≥ 0.75 n worlds.
- N6-3 (parts in circulation): the median share of cells carrying the founder's executed two-byte words, in the last
  snapshot before the founder, is ≥ 5 × the random-tape baseline (0.00024).
- Reported regardless: the median and range of steps and records since the newest open copier, contributing events,
  founder families, the base rate, and later founders separately.
*Kill.* If point mutations complete ≥ 0.25 n first founders, the claim "the first confined copier is assembled, not
point-mutated" is withdrawn from the paper. If fewer than 10 worlds close, N6 is uninformative and the claim stays
exploratory.
*Decision.* N6-1 and N6-2 met: the founder result may be stated as confirmed. N6-1 met alone: "assembled" may be stated
as confirmed, "through non-copiers" stays exploratory. N6-3 is descriptive support only.
*Not registered here.* A lethal-tar arm needs `lod.run` to pass `zero_halts` and the classifier to use the lethal executor;
that code change will be reviewed and tested before it is registered or run.

### Second correction to the outcome of N3 (2026-10-10, after the second independent figure critic; `results/lod/FOUNDING.md`, `founding.csv`, `lod_founding.py`)
The critic found that the record class "novel", reported as "recombination", is a residual class (the new tape is neither
a ≥ 75% copy of its partner nor ≥ 75% its own former bytes), not evidence that two parents' bytes were spliced, and that in
6 of 17 worlds the line record just before the founder already writes the founder into every random partner. Replaying
every founding encounter: the executor was a non-copier in 17 of 17 worlds; the founder was written into the partner's half
in 13 and the executor's own half in 4; a median of 7 of its 16 bytes (1–13) were in neither tape before. In 11 worlds the
executor's pointer ran into its partner and it wrote the founder only with that partner (0–19% of random partners); in 5,
a confined non-copier that writes the founder into every partner and turns itself into it had arisen 1–3 steps earlier (by
a point mutation in 2, a rewrite in 3); in 1 (the block copier), an open producer arose by point mutation one step before.
"Completed by recombination" and "recombination joined the parts" are withdrawn; the record class is renamed *rewrite*.
What stands: no first founder was a point mutation of a copier; every founding executor was a non-copier; the founder
appeared a median of 79 steps after the newest open copier; the words founders execute are one-byte variants of the
pusher's words, carried by about eight times the random-tape share of cells. The "contributing events" measure attributes
bytes along the chain and is not used, since the founding encounter writes most of the founder.

### Amendment to N6 (2026-10-10, before any N6 world is run)
Added predictions, on the first founder of each closed world (`lod_founding.py`, frozen as committed with this amendment):
- N6-4 (the maker): the founding executor is a non-copier in ≥ 0.8 n worlds.
- N6-5 (written, not inherited): no first founder is a point mutation of a copier (open or confined) in ≥ 0.9 n worlds.
Reported regardless: side, producers on the chain and how they arose, bytes in neither parent. N6-1 is kept as registered,
with "novel" read as *rewrite*.

### Third correction to the outcome of N3 (2026-10-10, third critic report; `results/lod/FOUNDING.md` regenerated)
"Written by a non-copier in 17 of 17" holds only under the whole-tape copy test. The 5 producers copy every byte they run
(copier by executed bytes; one is one byte short of the whole-tape threshold) and are confined: by that criterion they are
closers whose unexecuted half has not been rebuilt, made 1–3 steps before the founder by point mutation (2) or rewrite (3).
Robust under both criteria: in 12 of 17 worlds the founder was written by a program that copies neither its tape nor the code
it runs, as a rare outcome with soup partners (median 0.02, at most 0.46) except the open producer of 31005 (1.00); its
pointer entered the partner in the founding encounter in 10 of the 12. No founding event is a point mutation (0 of 17,
against 54 of 212 earlier chain events; P = 0.0068 under exchangeability). Founder 31007 is a copy of an open copier. Bytes
in neither parent, letting each 8-byte half of the founder take its own shifts: median 2 (0–7), withdrawing "a median of 7".
### Second amendment to N6 (2026-10-10, before any N6 world is run)
N6-4 is restated with both criteria: the founding executor is a non-copier by the whole-tape test in ≥ 0.8 n worlds (as
registered) and by executed bytes in ≥ 0.6 n worlds. N6-5 is restated as: no founding event is a point mutation in
≥ 0.9 n worlds. The frozen pipeline now includes `lod_founding.py` as committed with this amendment.

### Fourth correction to the outcome of N3 (2026-10-10, third critic report on the figure; `results/lod/REACH.md`, `lod_reach.py`)
The comparison "0 of 17 founding events are point mutations against 54 of 212 chain events (P = 0.007)" is withdrawn as a
test: record kinds are defined by how many bytes change, and every founder is 4–10 bytes (median 8, best shift) from the tape
its line held before, so the zero follows from the definitions. The fair null is mutational reach: only 14 of the 209
distinct chain tapes have any single-byte mutant that is a confined copier (at most 0.6% of their mutants), and the 54
recorded point mutations were expected to make 0.002 founders. That every founding executor is a whole-tape non-copier has
probability 0.076 under the soups' composition (non-copiers 84–89%). Point mutations made the writer in 3 worlds (31005,
31006, 31019). Stated result: closers descend from the open lineage; the first one is out of reach of a single mutation of
its ancestors and is written in one encounter, mostly by the non-copying tapes that fill the soup.
### Third amendment to N6 (2026-10-10, before any N6 world is run)
N6-5 is withdrawn (a founding event's kind is constrained by its distance from the tape before). Replaced by N6-5′
(mutational reach, `lod_reach.py` frozen with this amendment): the recorded point mutations on the chains are expected to
make fewer than 0.05 founders. N6-4 is reported with the abundance null of `lod_reach.py` and is not decisive alone.

### Wording note to N6 (2026-10-10, during N6, before any N6 world was downloaded or inspected)
A referee-style audit noted that N6-1 still says "recombination". The criterion is defined by record classes (`novel` or
`copyA`), and `novel` was renamed *rewrite* in the second correction to N3: N6-1 reads "the completing event is a rewrite or
a copy of the partner". No criterion, threshold, script or definition changes; the scoring script `n6_score.py` (84a889a)
is unchanged.

## X1–X3 — three referee-driven tests (written 2026-10-10, before any code for them was written or run)

Prompted by an internal hostile review: (i) the serial transfer follows one offspring per transfer, so its loss of the
pusher's lineages measures per-encounter copy success, not heredity in a branching population; (ii) inflow and openness
were measured against uniform random partners, whereas in the soup a pusher meets mostly its own kind; (iii) "open first"
was shown for the first replicator of a world, not along the lines of descent of the closed populations. All three run
locally (no spend). Thresholds as in the serial-retention test (R2/S) unless stated.

### X1 — branching transfer: does the open phase carry heritable variation?
*Genotypes* (the serial-retention panel): pusher16, pusher64, ret16, ldir32 (regenerators) and genome20_s6003
(transmitter). *Mutants:* every position, 32 seeded alternative values per position; 4 replicate populations per mutant;
64 unmutated populations. *Branching:* each generation every member runs as organism (128 instructions, the genotype's
own rule) against k = 2 fresh partners; every offspring that is a ≥ 75% copy of its parent at the best cyclic shift joins
the next generation; a population larger than N = 32 is reduced to a uniform random subsample of 32; G = 8 generations.
*Partner arms:* U, uniform random bytes (all genotypes); K, cells drawn uniformly from the open-phase snapshots (steps 5,000
and 10,000) of the 20 Stage G worlds of the same length (pushers only). *Measures:* a population is alive at g if it has a
member; it carries the allele if at least half of its members contain the mutant's 5-byte window (cyclic); background is
the share of alive unmutated populations that carry the same window by chance; a position is branching-transmissible at g
if, for at least half of its tested values, the share of alive mutant populations carrying the allele exceeds the
background by at least 0.5.
*Predictions.* X1-1: at least 0.5 of pusher16's unmutated populations are alive at g = 8 in arm U. X1-2 (decision): pusher16
has at least one branching-transmissible site at g = 8 in arm U. X1-3: ret16 and ldir32 have at most two sites at g = 8,
genome20_s6003 at least five.
*Reading.* X1-2 met: the open replicator carries heritable variation through branching lineages, and the paper says
heritable variation precedes closure. X1-2 not met in both arms: the paper says that persistent heritable variation arrives
with closed transmitters, and the summary and conclusions change accordingly. If X1-1 fails, branching does not rescue the
pusher's lineages and X1-2 is read as uninformative about alleles. Arm K is reported regardless.

### X2 — inflow and openness against partners from the soup
*Organisms:* the first heritable replicator of each of the 80 Stage G worlds. *Partners:* 256 cells drawn uniformly from
the snapshot of the organism's own world at step 5,000 (L = 16 and 20) or the snapshot nearest 2 × t_rep (L = 50 and 64);
also step 10,000 at L = 16. *Measures:* H(Y | X = x) over the 256 encounters (plug-in, bits) and the share of encounters
in which the pointer fetches a partner byte; compared with the uniform-partner values already reported.
*Predictions.* X2-1: at L = 16 the median in-soup inflow is at least 1 bit. X2-2: the pointer enters the partner in at least
half of the in-soup encounters for at least 90% of first replicators. *Reading.* X2-1 not met: the text says that the
partner dependence of the open replicator's offspring largely vanishes among its own kind, and the claim is restricted to
random partners. X2-2 not met: "open" is restricted to random partners.

### X3 — open first along the lines of descent
*Data:* the line records of every closed lod_v6 world (17) and, when available, of every closed N6 world. *Method:* on
each sampled line (the 64 lines per world of N3), every distinct tape from step 0 to the founder is classed against 64
random partners as in N3 (`lod_traj.classify_tapes`); the first copier on each line is the earliest record whose tape is
an open or confined copier. *Predictions.* X3-1: in at least 90% of closed worlds the first copier on every sampled line is
an open copier. X3-2: in no world does a confined copier on a line precede that line's first open copier by more than
the median founder delay (79 steps). *Reading.* X3-1 not met: "replication begins open" is restricted to the first
replicator of each world and is not said of the ancestry of the closers. For lod_v6 this analysis is registered after
the data exist but before this question was examined; for N6 it is confirmatory.

### Outcome of X2 (2026-10-10; `inflow_insoup.py`, `results/inflow_insoup/INFLOW_INSOUP.md`, `per_world.csv`)
X2-1 met: at L = 16 the median inflow against partners from the step-5,000 snapshot is 3.78 bits (every world above
1 bit; 3.52 with step-10,000 partners; 3.86 against uniform partners). X2-2 met: the pointer enters the partner in every
in-soup encounter for 80 of 80 first replicators. Medians against uniform and in-soup partners: L = 20, 6.17 and 4.52 bits;
L = 50, 7.78 and 6.90; L = 64, 7.98 and 7.33. Caveat recorded with the outcome: partners were drawn uniformly from the
whole soup, in which a median 15% of cells at L = 16 are ≥ 75% copies of the organism, whereas a pusher's lattice
neighbour is within Hamming distance 2 of it in 57% of open-phase encounters (`kin_census.py`); inflow among lattice
neighbours was not measured.

### Outcome of X3 on lod_v6 (2026-10-10; `lod_openfirst.py`, `results/lod/OPEN_FIRST.md`, `open_first.csv`)
X3-1 met: in 17 of 17 closed worlds the first copier on every sampled line is an open copier. X3-2 met: no line's first
copier is confined. The lines of a world share their early ancestry (19 distinct first-copier records over 17 worlds), so
the count of independent observations is 19, not 1,088. Two unregistered sensitivity checks (copy rate strictly above 0.5;
at least one exact self-copy), labelled as such in the report, leave the result unchanged. N6 worlds will be scored when
available.

### Outcome of X1 (2026-10-10; `branching_transfer.py`, `results/branching/BRANCHING.md`, `SUPP_TABLE.md`)
X1-1 met: 48 of 64 unmutated pusher16 populations alive at g = 8 (arm U). X1-2 not met: pusher16 has no
branching-transmissible site at g = 8 in arm U or arm K, and no mutant population carries its allele at g = 8 (0 of 512 in
each arm); the allele share falls from 0.10 at g = 1 to 0 by g = 5. X1-3 met: ret16 0, ldir32 0, genome20_s6003 9 sites
(all never executed). Registered reading applied: the paper says that persistent heritable variation arrives with closed
transmitters. An independent audit (re-run, independent re-implementation, alternative byte-presence scoring) found no bug;
it noted that 48 of 64 overstates control survival by chance (about 0.62 with 2,048 populations, unregistered) and that
arm K's survival is inflated by pool cells that already copy the parent, so arm K's survival is not cited. Mechanism from
its traces: the pusher's bytes are both the words it writes and the opcodes it executes, so a mutant byte is soon run
rather than copied, and the wild-type word is regenerated.

### Outcome of N6 (2026-10-10; `n6_analyse.sh` at 84a889a, frozen scripts unchanged; `results/lod_n6/N6_SCORE.md`)
All 20 worlds closed; replay validation held in every world (0 mismatches). N6-1 met (completed by a rewrite or a copy of
the partner in 18 of 20, by a point mutation in 0; the other 2 by partial overwrite). N6-2 met (only non-copiers between
the newest open copier and the founder in 20 of 20). N6-3 met (median executed-word share 0.00190, 7.9 × baseline).
N6-4 not met (executor a whole-tape non-copier in 20 of 20, but a non-copier by executed bytes in 11 of 20 against 12
required; abundance P = 0.062). N6-5′ met (0.0064 expected founders from 55 recorded point mutations). Kill not triggered.
Registered decision applied: N6-1 and N6-2 met, so the founder result is stated as confirmed; the writer's kind by
executed code is reported as not confirmed. X3 on the N6 worlds (confirmatory): X3-1 met, 20 of 20; X3-2 met.
