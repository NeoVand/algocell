# BFF (Computational Life) soups with three switches: findings

Pre-registered in `THEORY.md` P1 (a), (b1)/(b2) and (e) on 2026-10-08 before the runs (change log in `PLAN.md`);
60 soups on Modal L40S, each 2¹⁷ programs of 64 bytes, 16,384 epochs, 2¹³ steps per encounter, mutation 2⁻¹² per
byte per epoch, semantics of cubff's default BFF verified from `bff.inc.h` (pointer and heads at byte 0; the pointer
does not wrap; unmatched brackets terminate). Variants: **std** (as published, 24 seeds), **wrap** (pointer wraps
modulo 128, 24 seeds), **wraplit** (wrap plus one added instruction `P a b` that writes its two literal operand
bytes below head1 — the BFF analogue of `LD rr,nn ; PUSH rr`; 12 seeds). Numbers from `NUMBERS_BFF.md` and `runs.csv`
(generated). Cross-machine check: the six seeds also run locally on an Apple GPU give epoch statistics identical to
the Modal runs for every compared epoch (`reproducibility_local_vs_modal.csv`), so unlike the Z80 soup this system is
bitwise reproducible across machines.

Events (recorded per sample, every 64 epochs): `t_top`, the first sample at which the most common tape class is
heritable by the culture test (gen2 ≥ 0.3; 64 random partners) — its tape is the "first replicator"; `t_her`, the first
sample at which ≥ 50% of 32 random tapes are heritable; the pre-registered Z80 share criterion `t_rep` is reported
alongside (it under-fires in BFF because replicator populations are quasispecies: in the first live samples 0.69–0.97 of
random tapes were heritable while the most common exact class held 0.15% of the soup; amendment recorded in
THEORY.md before any verdict). **Closed** = the pointer enters the partner in ≤ 5% of culture-test encounters and
≥ 95% of partners become copies; **open** = enters in ≥ 50%; intermediate otherwise.

## 1. Standard BFF: replicators are born closed (prediction a)

- Transition (heritable top class) in **9/24** worlds within 16,384 epochs (median epoch 6,464 among those); ≥ 50% of
  random tapes heritable in 7/24. The paper reports a transition (its complexity ≥ 1 criterion) in ≈ 40% of runs at
  this mutation rate; our high-order-entropy ≥ 1 criterion fires in 9/24, in 2 of them without any heritable replicator.
- **Every first replicator contains a loop (9/9) and none is open (0/9):** entered median 0.00, copies median 1.00,
  self-damage 0.00. Seven are closed by the strict definition; two are intermediate — seed 11 enters the partner in 3%
  of encounters and copies 0.86, seed 4 enters in 34% and copies 0.72 — both loop-bearing and both tightened by the end
  (final top class closed in 7/9).
- Before the transition the soup shows no chunk phase: mean chunk transfer 0.47 bytes per encounter, copy events
  ≤ 1.2% of encounters — the straight-line "chunk copier" phase we had speculated about does not exist (consistent
  with the one-pass bound and with the search below).
- Reading of (a): the pre-stated strict form ("every first replicator closed with a loop") is **7/9**; the mechanism
  statement — no open, straight-line replicator; loops from birth — holds **9/9**. Both are reported; the strict
  reading is not met by two near-closed cases.

## 2. Wrapping pointer: more transitions, still born closed (reading b2)

- Transition in **19/24** worlds (median epoch 6,912), ≥ 50% heritable in 19/24 — more often than standard BFF (9/24;
  Fisher two-sided p = 0.0077; time to `t_her` with censoring, Mann–Whitney p = 2.9 × 10⁻⁴). The wrapping pointer
  executes random code for 5,000+ steps instead of ≈ 600 and rewrites partners ten times more (chunk transfer 0.98 vs
  0.47 bytes per encounter), which is where its extra transitions come from.
- **First replicators: loop-bearing 19/19, open 0/19**, closed 16/19, intermediate 3/19 (entered 0.00, copies
  0.83–0.89). Final top class closed in 18/19. This is reading **(b2)**: the open phase does not appear when the
  pointer wraps, so pointer range was not the whole story.
- The high-order-entropy detector fires at epoch 64 in **24/24** wrap worlds, in 5 of them without a replicator ever
  appearing: the wrapping soup immediately fills with one-symbol tar (compressible, sterile), the BFF analogue of the
  Z80 zero flood, and the compression detector reports it as the transition.

## 3. The structured search: no straight-line replicator in wrap-BFF

`micro/bff_search.py` (`results/bff_search/NUMBERS_SEARCH.md`): 16,447,860 straight-line programs (prefix ≤ 3
instructions + tiling unit of period 2–5 containing a copy instruction) against 2 random partners each, wrapping
pointer. 130 pass the 75% byte-match criterion against 32 partners; every one is a one-symbol fill (offspring 0.83 one
byte), and the 8 with gen2 ≥ 0.3 are `,`-fills "heritable" only through the partner's stray instructions. No program
copies a genome of period ≥ 2. BFF has no literal write channel (`+`/`−` increment in place, `.`/`,` copy from
memory), and a memory copier's read and write heads start at the same byte, cross after 64 writes and drift.

## 4. Adding a literal write channel switches the open phase on — and it does not last (prediction e)

- **(e1) met, 12/12:** the first replicator in every `wraplit` world is the 64-byte tape of the added instruction
  itself, `PPPP…` — a one-byte tiling whose literal operands are two more copies of itself, the exact analogue of
  `01 c5`: straight-line, no loop, pointer enters the partner in 100% of culture-test encounters, copies 0.88–1.00 of
  random partners, gen2 0.75–0.92, self-damage 0.00.
- **(e2) met:** it appears at the first sample (epoch 64) in 12/12 worlds, holding 53–60% of the soup, with 53–84% of
  random tapes heritable, against a censored median of 16,384 epochs for standard BFF (one-sided Mann–Whitney
  p = 6.6 × 10⁻⁸).
- **(e3) not met — and the alternative is not closure either:** the open wave **collapses in 12/12 worlds**. By epoch
  128 the all-`P` class holds ≈ 2% of the soup, by epoch 256 ≈ 0.03%, and the heritable fraction of random tapes is 0
  from epoch ≈ 256 to the end; the population's mean executed steps fall from ≈ 5,300 to 35–90 per encounter and the
  fraction of encounters whose pointer enters the partner from 0.79 to 0.01–0.03. The soup has become tar that
  terminates any pointer within a few dozen steps (unmatched `]`). The `P` tape survives to the end as a 0.2% minority
  class with gen2 0.70–0.91 — heritable in the culture test, unable to spread in the soup it created. No closed class
  ever appears (0/12).
- Reading, with the mechanism still to be confirmed by a controlled test: the open replicator is partner-dependent in
  the lethal way. Copying into a tar partner is cut short when the pointer runs into the partner's terminating
  brackets (at most 44 of 64 bytes are written per pass), so the offspring carries a tar head and dies on execution;
  and a tar tape that terminates before entering its partner cannot be converted by executing the partner's `P`
  code. Selection for early termination (immunity) races against the open copier and wins within ≈ 200 epochs. In the
  Z80 soup the tar is zeros — no-ops that do not stop the pointer — so the open pusher kept its population for tens of
  thousands of steps, long enough for a closed variant to be found. **Closure needs a window; the window is set by how
  lethal the context is to an open organism.** This refines THEORY.md Lemma 2's "closure is selected because the open
  form is damaged": too little damage (Z80 at L = 64, self-damage 0.06) and closure is slow; too much (BFF with a
  literal) and the open form is gone before closure can evolve.

## 5. What the BFF experiments add

1. Agüera y Arcas et al.'s BFF replicators are born closed; the Z80's open phase is not a universal first stage.
2. The difference is the write channel, not the pointer range: a wrapping pointer alone gives no open phase (b2); one
   literal-push instruction gives an open, straight-line, partner-dependent first replicator in every world (e1).
3. The open phase is a window, not a stage: in BFF-with-literal it lasts ≈ 100 epochs and ends in collapse, because
   BFF's tar is lethal to open organisms; in the Z80 it lasts long enough for closure to be found. Prediction for the
   held cell (e4, standard pointer + literal) and for a "benign tar" BFF variant (e.g. `]` without a matching `[` as a
   no-op instead of a halt): the open phase should then persist and closure should follow — to be pre-registered.
4. Two detector failures: high-order entropy fires on one-symbol tar (wrap: 24/24 at epoch 64, 5 with no replicator
   ever) and misses a soup that is 60% one replicator (wraplit: HOE < 0 throughout, because a one-byte organism
   lowers byte entropy as much as it lowers compressed size).
5. Cost: 60 soups of 19–30 min each on L40S, 24.8 GPU-hours ≈ $47 at list price (the user approved ≈ $45; the overrun comes from the slowest containers).

Figures: `bff_timeseries.{pdf,svg,png}` (high-order entropy, pointer-entered fraction, copy events, heritable fraction
vs epoch, all runs by variant); `bff_first_vs_final_openness.{pdf,svg,png}`. Per-run table `runs.csv`.

## 4b. Benign tar: the open phase becomes permanent (prediction f, `wraplitnh`, 12 seeds)

Same as `wraplit` but an unmatched bracket is a no-op instead of ending the encounter (pre-registered THEORY.md P1(f),
run after a constructive search found no closed heritable replicator among 1.3 million periodic programs of this
language up to period 10).
- **(f1) met, 12/12:** the first replicator is again the all-`P` one-byte tiling, open (pointer enters the partner in
  100% of culture-test encounters), loop-free, copies 1.00 of random partners, at the first sample (epoch 64).
- **(f2) met, 12/12:** the open population **persists**: the heritable fraction of random tapes is ≥ 0.5 at the final
  sample in 12/12 worlds (final median 1.00; peak 1.00 reached at median epoch 128; collapsed 0/12), against 0/12 in
  `wraplit`. The kill criterion (collapse in ≥ 9/12) was not triggered.
- **(f3) met, 0/12 closed:** no closed class ever appears, as the search implied: in this language the literal's
  operands cannot both execute a loop and be written by it, so closure is unreachable.
- Reading, fixed in advance: this is the "M1" cell of the minimal-machine ladder realised inside BFF — a substrate in
  which the open organism is stable and closure is impossible — and it settles the mechanism of the `wraplit` collapse:
  the only change between the two cells is whether an unmatched bracket halts the pointer. Lethal tar: extinction in
  12/12 by epoch ≈ 256. Benign tar: a permanent open regime, 12/12, for 16,384 epochs.
- The high-order-entropy detector fires at the first sample (epoch 0) in 12/12 — before any class holds 0.03% of the
  soup, because 8,192 un-halted steps of random code already restructure the tapes — and is silent (HOE ≤ 0.03) at
  every sample from epoch 64 on, through the organism's whole reign: a one-byte organism has no high-order structure
  to detect. The detector fires before life and goes quiet during it.

## 4c. Literal without the wrapping pointer (`lit`, 12 seeds, all finished; seeds 11 and 12 were re-run after a stopped Modal app lost them at epochs 153 and 10,560)

- **(l1) met, 12/12:** the all-`P` tiling is the first replicator in every world (epoch 64), open and loop-free, copies
  0.91 of random partners (median; the one-pass bound makes copies into partners with hostile tails partial).
- **(l2) partly as predicted:** the open wave is weaker than under the wrapping pointer — the heritable fraction of
  random tapes reaches ≥ 0.5 in only 5/12 worlds (peak median 0.47 at epoch 96) — and the open population dies out in
  every world (final heritable fraction 0.00 in 12/12; the all-`P` class survives only as
  a minority). By the pre-registered letter of (l2) ("collapse in ≥ 9/12", defined as ≥ 0.5 then < 0.1) the count is
  5/12, because 7 worlds never reached 0.5; by its intent (no persistent open population) it is 12/12. Both are
  reported. No closed class (0/12).
- Together with 4b: the pointer wrap decides how strong the open wave is (bandwidth), the tar decides whether it lasts.

## 5b. The classification, completed

| | tar benign to open organisms | tar lethal to open organisms |
|---|---|---|
| **literal write channel** | open, then closed — the Z80 soup (40/40 at 300k; 19/20 by 1M at L = 20) · **open forever — BFF + literal, no-halt (12/12 persist, 0/12 close: closure unreachable)** | open, then extinct — BFF + literal (12/12 collapse) · BFF + literal without wrap (12/12 die out) |
| **no literal write channel** | — (not run) | born closed — BFF as published (9/9) and with a wrapping pointer (19/19) |

The Z80 and the benign-tar BFF cell share the top-left corner and differ in one respect only: the Z80 instruction set
offers closers (a jump that keeps the pointer home while the pushes continue), BFF + `P` offers none that can also be
copied. So "open then closed" needs three things — a literal channel, benign tar, and a reachable closed design — and
the atlas has now exhibited a substrate that has the first two and lacks the third.
