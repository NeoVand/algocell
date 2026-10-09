# The plan for Nature (written 2026-10-09, early, after both reviews and tonight's experiments)

## The thesis, sharpened by what the reviews took away and what tonight found

Three transitions, in a fixed order, each measured by intervention, each explained by counting, each switchable:

1. **Heredity without individuality.** The first replicator is the shortest thing that writes itself; it is open, its copies
   depend on the neighbour, and it inherits its own mutations (the mutational scan: 20–32 transmissible positions at
   L ≥ 50). Solid: 100 of 100 worlds across five lengths, both lattices, two dating rules.
2. **Individuality by closure, at the price of variation.** A control cycle keeps execution inside the organism; copies
   stop depending on the partner (0 bits in 63 of 67) and the organism wins the competition against its own ancestor
   within ~120 steps (pair invasion), but it executes everything it copies, so a mutation is either lethal or corrected:
   the first individuals are canalised (capacity 6–42 bits against 32–325 for the open form). New tonight; the review's
   own "bigger question", answered the other way round.
3. **The genotype: information copied but not executed.** Heritable variation reappears only in genomes with a segment
   that a block copy transmits and a jump never runs (5 worlds so far, 4–11 transmissible sites). This is Langton's
   criterion (information used interpreted and uninterpreted) and von Neumann's description/constructor split, arising
   rather than designed, and measurable as U = fraction of copied positions never executed. Hypothesis, with a pilot.

The substrate map (literal channel × by-product lethality) says where each beginning is possible; tonight it has all four
cells. The write-range/execution-range split says which beginnings can evolve further. A lemma to state and test: *a copier
whose written range equals its executed range transmits no neutral variation; evolvability requires copying more than you
run.* Block copies have that for free; push and return designs do not.

## The three moves that make it a Nature paper

**Move 1 — a second real instruction set (the 8080), with predictions registered first.** The 8080 is the Z80 without the
CB/ED/DD/FD pages and without DJNZ, JR, EXX and EX AF,AF'; our ablation machinery produces it by suppression (`z80-only`
pattern set; verify the opcode list against the 8080 manual before running). Predictions: pusher first (`01 c5` is LXI B /
PUSH B); closure at L = 16 by the return design (`ad e3 21 e3 21 c0 ad c0` is all 8080); no block-copy closers anywhere, so
closure at L = 32 and 64 much rarer; under lethal zeros, late life must come from return designs, so rarer still. This also
explains the published negative result (ref. 5 never saw a looping 8080 variant): return closure needs a short ring that
aliases the popped address back into the organism, which a long tape does not give. Cost ≈ $10 on Modal (L = 16 × 20
worlds × 300k; L = 32 × 20 × 1M). Kill: no closure at L = 16 in 15 of 20.

**Move 2 — watch the genotype being born.** (a) Instrument the test executor with an executed-address bitmap per encounter
(one shader change): it gives pointer confinement for every tape in both machines (the reviews' demand), the executed set,
and U. (b) Run the mutational scan and U on the dominant tape of every snapshot of Stages G, K and I (≈ 1,600 tapes, an
hour): capacity over evolutionary time, by lineage type. Pre-register: transmissible sites coincide with copied-but-
unexecuted positions; every closer with U = 0 has ≤ 1 site; the born-closed LDIR replicators of Stage I have U > 0 from
birth; BFF replicators have U > 0. (c) Extend 10 worlds at L = 16 and 10 at L = 20 to ten million steps (≈ $35): does
capacity recover after closure, and only in block-copy lineages? (d) In 8080 mode, with no block copy, recovery should be
rare. If (c) holds, the paper has a third, measured transition and a new title.

**Move 3 — turn the map into a phase diagram.** Two dials, both cheap in BFF (≈ 25 min a soup): tar lethality p (a fetched
unmatched bracket halts with probability 0, 0.01, 0.03, 0.1, 0.3, 1) and literal bandwidth (P writes 1, 2 or 3 bytes).
Model 5 predicts a crossover p* from "open then closed/persisting" to "open then extinct"; the write-ratio reading predicts
open births below ratio 1 and saturation closure at or above. Two dials × 6 levels × 6 soups ≈ $60. Either outcome is a
figure; a crossover is a law.

## The honest floor (what we publish even if Moves 2–3 disappoint)

The reviews' synthesis version: "Heredity can precede individuality in a digital primordial soup", conditional, two real
instruction sets if Move 1 holds, the closure cost as the discovery. That paper exists in the repository today minus the
trim and the statistics.

## Order of work (each step a day or less unless marked)

1. Trim to 3,500 words; one closure definition (confinement = mechanism, independence = outcome); references renumbered and
   pruned; ED items to ten; Fig. 6 redrawn for the argument actually used; Data/Code availability; OSF time stamp.
2. Executed-address bitmap in the test shader; rerun partner tests for all 160 + Stage K + Stage I tapes; the 2 × 2
   agreement table (pointer × inflow) for both machines.
3. 8080 mode: define the pattern set, verify the surviving opcode list, pre-register, launch (Modal).
4. Capacity and U over snapshots (existing data); Stage I and BFF predictions.
5. Statistics: log-rank and bootstrap intervals; threshold grid (Stage J); 64-cell census where recorded.
6. Convention robustness (random initial registers; random SP start), 20 worlds each at L = 16 (Modal ≈ $5): tar should
   change colour or vanish, the pusher should still come first.
7. Ten-million-step extensions (Move 2c) and the dials (Move 3), launched together (≈ $100).
8. Lineage by proximity from the dense snapshots; descent vs displacement by length, stated.
9. Manuscript v4: six figures (encounter; one world; open → closed; the cost of closure and the birth of the genotype; the
   substrate map with two real machines and the dials; theory as proved). Preprint; clean repository; DOI; an outside run.

Budget ≈ $150 on Modal in all; three to four weeks. Every prediction written before the run; every miss reported.
