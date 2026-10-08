# Theorems (what can be proved), propositions (computer-assisted), and model statements

Written 2026-10-08 04:50 after the BFF results, at the user's request to turn the theory into theorems where possible.
Each statement says what kind of claim it is: **Theorem** (pen-and-paper proof, no simulation), **Proposition**
(exact statement decided by finite computation, with the script that decides it), **Model** (a result about a
stylised model with stated assumptions, not about the simulator), **Empirical** (what only the soups can tell).
Notation follows THEORY.md: a pair is the organism A (L bytes) followed by the partner B (L bytes); an *encounter* is one
execution of the pair from the first byte of A; a *full copy* means B equals A (up to a cyclic shift or reversal) at
every position after the encounter; *closed* means the instruction pointer never executes an address in B.

---

## Theorem 1 (no open replicator in one-pass BFF)

**Setting.** Standard BFF: 128-byte pair, pointer and both heads start at byte 0, the pointer advances by one after
every instruction unless a bracket jumps, execution ends when the pointer leaves [0, 128), on an unmatched bracket, or
after 2¹³ steps. Instructions: `<` `>` `{` `}` move head0/head1 by ±1; `+` `−` change the byte under head0 by ±1;
`.` writes the byte under head0 to the cell under head1; `,` writes the byte under head1 to the cell under head0;
every other byte is a no-op. Call an execution *straight-line* if no bracket jump is taken.

**Theorem 1.** No straight-line execution of a 64-byte program A produces a full copy of A in B. More precisely, if A
has D distinct byte values, then a straight-line execution writes at most 43 bytes of B that agree with A at any fixed
alignment when D ≥ 3, and writes a non-constant pattern into B only if the program uses at least three distinct
instruction letters.

*Proof.* A straight-line execution executes at most 128 instructions (one per pointer position). Only `.` and `,`
can place a prescribed value into a cell for every partner: `+`/`−` change an unknown partner byte by ±1 and so cannot
produce a prescribed value with certainty. Hence each of the k positions of B that end up equal to the target needs
one write instruction: w ≥ k. A write lands on the cell under the writing head, so the k target positions are among the
positions the heads have occupied; both heads start at byte 0, which lies in A, and a move changes a head's position by
one, so the two heads together occupy at most 2 + v distinct positions, of which byte 0 is not in B: k ≤ 1 + v, i.e.
v ≥ k − 1. The values written are the contents of the *source* head's cell at the time of writing (for `.` the source is
head0, for `,` head1). A written value can only change between two writes if the source head moved or an increment
changed its cell; each such change costs an instruction not already counted among the writer's moves when the source
and the writer are different heads, and when the program alternates `.` and `,` so that a head is sometimes source and
sometimes writer, a single move changes both roles, so we only use: the sequence of written values takes on at least
D' distinct values, where D' is the number of distinct values among the k copied bytes, and distinct source *values*
must come from distinct source *cells* or from increments; the source cells holding A's values lie in A, and the heads
must visit them, so the heads occupy at least k + D' − (cells that are both source and target) distinct positions
other than byte 0; a target cell can serve as a source only after it has been written, and then it provides a value
already written, so it contributes no new value. Therefore 2 + v ≥ 1 + k + D', i.e. v ≥ k + D' − 1, and
128 ≥ w + v ≥ 2k + D' − 1. For a full copy k = 64 and D' = D ≥ 3 this gives 128 ≥ 130, impossible. For a partial copy
of a tape in which every byte differs from its neighbour at the copied positions (D' ≥ k/2 for a period-2 tiling,
D' ≥ k for a tape with k distinct bytes), 128 ≥ 2k + k/2 − 1 gives k ≤ 51, and with the writer's moves and the source's
moves necessarily distinct instructions (two different heads) the sharper 128 ≥ k + (k − 1) + (k − 1) gives k ≤ 43.
If A has D ≤ 2 distinct byte values, then as a program A contains at most two instruction letters (a non-instruction
byte is a no-op); to write a non-constant pattern into B the program needs a write instruction, a move of the writing
head (to reach two positions) and a change of the source (a move of the other head or an increment) — three letters —
so a two-letter program writes a constant into B, which equals A only if A is constant, and a constant program of one
instruction letter either never writes or writes the byte under head0 onto itself. ∎

**Corollary 1.1.** Every full self-replicator in standard BFF executes a bracket jump; since the pointer must return
to an earlier address to execute more than 128 instructions, every full self-replicator contains a loop. (This is the
"born closed" part of the BFF prediction; what the theorem does *not* say is whether the loop keeps the pointer out of
B — that is Theorem 2's business and the experiment's: 28/28 first replicators loop-bearing, 0/28 open.)

**Remark (what the soups measured).** The chunk-transfer statistic before any transition was 0.47 bytes per
encounter (standard) and the largest observed 90th percentile 2 bytes, well below the bound of 43; the bound is not
tight for random programs, it is the ceiling for designed ones.

---

## Theorem 2 (closure requires a cycle)

**Setting.** Any machine with an instruction pointer that, in the absence of a control transfer (jump, call, return,
repeat-in-place, halt-in-place), advances past each executed instruction, so that an execution without control
transfers executes each address at most once; instructions may write into memory. Let β be an upper bound on the number
of bytes an execution can write per byte of instruction stream it consumes (bytes written divided by bytes of code
executed, maximised over all executions).

**Theorem 2.** If β < 1, then a full self-replicator of L bytes whose pointer never leaves its own L bytes executes
some address at least twice. Equivalently: a closed full replicator contains a cycle in its executed control flow.

*Proof.* A closed execution executes only addresses in A; if no address repeats, it consumes at most L bytes of
instruction stream and so writes at most βL < L bytes, fewer than the L bytes a full copy of A into B requires. ∎

**The three machines.** BFF: `.`/`,` write one byte per one-byte instruction but a copy needs a head move per target
cell, so β ≤ ½ for copies (Theorem 1's count); BFF with the literal push `P a b`: two bytes per three bytes of code,
β = ⅔; Z80 `LD rr,nn ; PUSH rr`: two bytes per four bytes of code, β = ½; Z80 `LDIR`: unbounded bytes per instruction,
but `LDIR` re-executes its own address until BC = 0, which is a repeat and so a cycle in the sense of the theorem. This
is why Stage G's syntactic criterion (jump/call/return) under-counted closure and the behavioural one did not: LDIR
closers satisfy Theorem 2's conclusion without a jump.

**Corollary 2.1 (smallest closed organism > smallest open organism, for literal writers).** In a machine with a literal
write channel the smallest open self-writer can be the literal itself (Z80: two bytes, `01 c5`; BFF+P: one byte,
`P`), while a closed replicator needs in addition an executed control transfer, hence at least one more byte and, when
the control transfer carries an operand, two or more. Hence the open self-writer is strictly smaller.

---

## Proposition 3 (the two-byte self-writers of the Z80 are exactly the load–push words) — computer-assisted

**Statement.** Among the 65,536 two-byte words tiled to 16 bytes and executed for 128 Z80 steps as A against the
all-zero partner, exactly six leave the partner holding a tiling of the word: the trivial `00 00` (the partner already
is that tiling) and the five load–push words `01 c5`, `11 d5`, `21 e5` (`LD rr,nn ; PUSH rr`) and `2a e5`, `e5 2a`
(`LD HL,(nn) ; PUSH HL`) — `two_byte_census.py --fixed-points`, `results/census2/fixed_points.csv`, decided
2026-10-08 04:55 by exhaustive computation. None of the five contains a control-transfer or repeat instruction, so by
the pointer-advance rule each executes past its own 16 bytes into the partner: **every two-byte self-writer of this
machine is open**, and no two-byte tiling is closed (Corollary 2.1 made exact at two bytes). The stochastic census
(256 random partners) agrees: the same five words copy ≥ 0.5 of partners (0.66–0.81), none ≥ 0.95, and the 246
`CALL`-smear words that pass gen2 ≥ 0.3 produce no 75% copy.

---

## Proposition 4 (open first: a counting statement about the initial soup)

**Statement.** In a soup of N tapes of L uniformly random bytes, the expected number of occurrences of a given k-byte
word (at any of the L − k + 1 positions of a tape) is N (L − k + 1) / 256ᵏ. For N = 20,000 and L = 16: 4.6 for any given
two-byte word, 0.018 for any three-byte word, 6.1 × 10⁻⁵ for any four-byte word. The probability that at least one copy
of a given two-byte word is present at step 0 is ≥ 1 − e⁻⁴·⁶ ≈ 0.99 (Poisson approximation; exact: 1 − (1 − 256⁻²)^{N(L−1)}).

**Consequence (conditional).** If a two-byte open self-writer exists (Proposition 3) and has positive growth rate in
the soup (empirical: the pusher copies into 0.66–0.73 of random partners and invades from 1% of cells at 128 steps,
`results/invasion/`), and every closed replicator is at least four bytes (Corollary 2.1 plus Proposition 3(ii)), then
with probability ≈ 0.99 an open self-writer is present from the start while a specific closed design is absent with
probability ≥ 0.9999, and the first replicator is open unless the open form fails to grow. This is the status of "open
first": a theorem-grade count plus one empirical premise, not a theorem about the soup.

---

## Model 5 (the closure window) — a model, not a theorem about the soup

**Assumptions.** An open replicator population of size n(t) tapes; each tape-epoch produces a closed functional
variant with probability q (a mutation hitting the right bytes with the right values: q is small, of order
(sites)·μ·256⁻ᵐ for an m-byte closing motif); a closed variant, once present, has a higher per-encounter growth rate
than the open form and fixes with probability of order 1 − 1/fitness-ratio.

**Statement.** The expected number of closed variants produced before the open population is gone is
q ∫ n(t) dt; closure occurs with probability ≈ 1 − exp(−q ∫ n dt). The integral is the *window*. Two regimes follow:
(i) if the context is benign for the open form (its copies are viable in the soup it creates), n(t) stays near the
carrying capacity and the window grows linearly with time, so closure is a matter of waiting (Z80: closure by 300k
steps at L = 16, 50; by 1M in 19/20 at L = 20; 8/20 at L = 64, where the open form is least damaged and q is smallest
because the motif must land in a longer tape); (ii) if the context becomes lethal for the open form faster than q ∫ n dt
reaches order one, the open form goes extinct without closure (BFF with the literal channel: n(t) fell from 0.6 N at
epoch 64 to 3 × 10⁻⁴ N by epoch 256, a window of order 10⁷ tape-epochs, and 0/12 closures).

**What the model does not do.** It does not predict which regime a given machine is in; that depends on the soup's
tar chemistry (BFF's unmatched `]` halts the pointer; the Z80's zeros are no-ops), which is exactly what the
experiments measure. Its testable content is the integral: raising the window at fixed q (a benign-tar BFF variant)
should restore closure; shrinking it in the Z80 (adding a halting byte class to contexts) should remove it.

---

## Corrections these proofs forced

- **THEORY.md P1(e4), held prediction for `lit` (standard pointer + literal), is wrong as written.** The count behind
  it was for the two-byte tiling `P x` (16 pushes per 64 bytes of code, 48 bytes written in one pass). The one-byte
  tiling `P P P …` executes a push every three bytes (its literal operands are two more copies of itself), 22 pushes in
  its own 64 bytes and a further 10 in the copied tail of the partner, 64 bytes in all: a full open self-replicator
  within one pass (β = ⅔, and the organism's code is all pushes). The corrected prediction for `lit` is the same as for
  `wraplit`: the all-`P` tape first, open and loop-free; whether it collapses is the empirical question (its copies into
  tar partners are cut short the same way).
- **The syntactic loop criterion of Stage G** is a proxy for Theorem 2's conclusion; the behavioural partner test is the
  measurement of closure itself. The paper should state closure as Theorem 2 states it (a repeated executed address)
  and measure it by intervention.

## What remains to prove or decide

1. A clean general version of Theorem 1 for arbitrary head-copy machines (the proof above handles the role-swapping case
   by a conservative count; a cleaner invariant would be the number of distinct (position, value) pairs delivered).
2. Proposition 3 for three-byte words (16.7 million tilings; feasible on the GPU with the alphabet prefilter), which
   would make "the smallest closed Z80 replicator has at least four bytes" exact.
3. The model's q for the Z80 closers (`20 f0` at L = 50: two specific bytes at a specific phase — q of order
   μ² · phases) against the measured closure times — a quantitative test of Model 5 without new runs.
