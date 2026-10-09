# Review 2 (AI, three reviewers with a synthesis, 2026-10-08): what it adds, what we verified, what to do

Source: `~/Downloads/paper-review-heredity-before-individuality.md` (untrusted; read only). Read with `REVIEW_1_ASSESSMENT.md`;
overlaps (novelty, entropy headline, three closures, finite searches, theorem language, 8080) are not repeated here.

## 1. New points, verified tonight against our own data

1. **The 16-bit wrap under the relative-jump closers (Required 1 / M4): confirmed.** Memory is addressed modulo 2L
   (`mem[addr % MEM_LENGTH]`) while the program counter is 16-bit. A backward relative jump that crosses address 0 lands
   at (65,536 + target) mod 2L, which equals the true-ring target only when 2L divides 65,536 (L = 16, 64; not 20, 50).
   Our traces (`results/concept/traces.json`): the L = 50 JR NZ closer jumps 23 → 9 (ordinary) and then 9 → 31 (−5 wraps
   to 65,531, mod 100 = 31); on an aligned ring it would land at 95, in the partner. The L = 20 DJNZ closer jumps 16 → 7
   (−9 → 65,527 mod 40 = 7; aligned: 31). Exact count: 25 of 67 closers use a relative jump at a misaligned length (all
   20 at L = 50, 5 of 19 at L = 20); the other 42 (17 RET NZ + 3 LDIR at L = 16, 14 block copies at L = 20, 8 at L = 64)
   do not depend on the wrap. The review's "up to 39" is an upper bound; 25 is the number. Consequence: the open → closed
   order stands at the aligned lengths (20 of 20 at L = 16, 8 of 20 at L = 64) and the mechanism at L = 50 is specific
   to the aliasing. Stage K (L = 32, aligned, 20 worlds, one million steps) is launched to test closure at an intermediate
   length without the wrap (`REVISION_PREREG.md`). The pair-invasion and mutational-scan results at L = 50 inherit the
   convention and must say so.
2. **"Ninety-fold delay" depends on the dating rule (Required 3): confirmed.** By first clone (t_rep) lethal vs benign is
   46,250 vs 525. By heritable fraction of random cells ≥ 0.1 it is 40,000 vs 1,250 (thirty-fold); at ≥ 0.5 it is 40,000
   vs 30,000: closed, high-fidelity heredity arrives at about the same time with and without an open phase. Two lethal
   worlds reach 0.5 at step 5,000 while their first clone is dated 21,000 and 206,000. This supports the review's point
   that the open phase is not shown to speed closure, and it fits the L = 16 picture where the RET NZ closer shares no
   bytes with the pusher (displacement, not descent) while at L = 50 the closer is the pusher plus a jump (descent, and
   the pair invasion shows the jump sweeps).
3. **Fig. 3d plateaus:** final heritable fraction medians 0.86 (L = 16), 0.93 (20), 0.59 (50), 0.30 (64). "Rises to
   0.75–0.94" holds at L = 16 and 20 only.
4. **Size axis:** alive fractions 0, 0, 0.9, 0.9, 0, 1.0, 0.4, 0.4, 0.6 at L = 3, 4, 5, 6, 7, 8, 9, 10, 12 and 1.0 from 16.
   "No floor" is wrong: nothing at L = 3, 4 and 7.
5. **Stage D labels:** the 19 of 20 at 512 steps is stack-write-only (stack-writes gives 18); name it.
6. **BFF variants mixed on p. 6:** first-sample share 0.58–0.63 is `wraplit`; the 8-bit inflow is `lit` (first share
   0.4–0.9%). Separate them.
7. **References:** 23 of 44 never cited (15, 16, 18, 20–36, 38–40); first-citation order 1, 2, 3, 4, 8, 9, 10, 11, 14,
   17, 5, … Renumber by first citation; cite or drop the rest (Eigen, Tierra, von Neumann, Langton and Amoeba belong in
   the Discussion, as the review says).
8. **Ablations:** Fig. 4a has 15 rows; the text says thirteen. Fix the count and the "input and output" phrase.
9. **Closure is not monotone in L** (20 of 20 at L = 50 by 300k; 19 of 20 at L = 20 by one million): the sentence
   "closure takes longer the larger the organism" must go; with point 1 the L = 50 speed is the jump's.

## 2. Points accepted without new checks

Convention robustness (initial registers, SP start, pair order, start address) untested; pointer entry unrecorded in the
Z80; log-rank tests and bootstrap intervals in place of "seed-paired noise" (same-seed runs diverge); census of ≥ 64 cells;
McNemar's weak null; Theorem 1's context (fixed tonight: a partner holding no byte of the copy in place; add the zero-
context remark); Fig. 6 drawn for the bandwidth reading while the Z80 argument is the budget pigeonhole; Avida's circular
memory (drop Avida from Scope); the 0.2% vs 3 × 10⁻⁴ N mismatch; "faithful" used in two senses; Nature's limit of ten
Extended Data items (we have 14 figures and 2 tables); Data and Code availability statements; a third-party time stamp
for the pre-registration (OSF or Zenodo).

## 3. Points we push back on, with reasons

- *"Reject and resubmit"* (Pessimist): the synthesis's view is right; the aligned-length result and the measured
  convergence survive every objection raised.
- *The BFF literal switch is circular*: the instruction was designed to be a self-writer, and the paper says so; what
  was not designed is what follows (extinction under lethal brackets, persistence under harmless ones), and the no-literal
  cells. Keep the cell, state the design.
- *Theorem 2 "says nothing about copying"*: true and intended; it is the structural half. The content is in Proposition 3
  and the measurements. Say so once, plainly.
- *Model 5 "fits any outcome"*: tonight's pusher-only controls give the first direct estimate of its ingredients (jump
  arises and sweeps within 5,000–13,250 steps in a saturated pusher world); making q(L) quantitative is the right next
  step (the review's Ambition 3; our mutational scan already enumerates every single mutant).

## 4. What tonight's new results do to the verdict

The mutational scan and the pair invasion answer the review's two causal demands (matched-genotype test; invasion of the
closed form into the open population) before the review asked, and they add the dissociation both reviews wanted: the
jump wins the competition and loses the variation channel. They do not touch the wrap question, which is the one new
threat; Stage K is the answer, and the honest fallback is to state the claim at aligned lengths.

## 5. The revision plan, merged from both reviews, in order

1. Stage K result into the closure section (claim restricted or confirmed). 2. Convention paragraph: address mapping, SP
start, organism-first order stated; a robustness stage (random registers, random SP) proposed or run. 3. Pointer entry
recorded in the Z80 (P) and the 2 × 2 agreement table. 4. Timing by heritable fraction beside t_rep; "ninety-fold" →
"thirty-fold by heritable material, and closed heredity arrives at a similar time". 5. Descent vs displacement by
length, stated. 6. Statistics: log-rank, bootstrap intervals, thresholds grid (Stage J), 64-cell census where it exists.
7. References renumbered and pruned; ED items reduced to ten (merge 1–4, 6–7, 9–10; keep 5, 8, 11–14 as data). 8. Title
"Heredity can precede individuality…" or keep the current one with the operational definition in the first paragraph.
9. Trim to 3,500 words. 10. 8080 mode as the second real instruction set (Z80 with the Z80-only opcodes suppressed:
CB/ED/DD/FD pages, DJNZ, JR, EXX, EX AF,AF'), pre-registered: the L = 16 RET NZ closer is 8080-valid, so closure is
predicted there.
