# Supplementary Information

Open replicators evolve closure in a digital primordial soup — supplementary text, assembled 2026-10-08 from the tracked documents of the experiments repository.

## S1. Theorems, propositions and the model (THEOREMS.md)

### Theorems (what can be proved), propositions (computer-assisted), and model statements

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

---

### Part II — statements above the level of bytes (2026-10-08 05:10, after the user asked for substrate-level theorems)

The user's question: can the main claims be theorems for an arbitrary substrate, in the language of theoretical
computer science, information theory or thermodynamics? Answer in three parts: an **architectural theorem** (any
machine with two properties), an **information-theoretic lemma** (substrate-free, definitional), and a
**classification** in place of a universal dynamical law, with the reasons why the dynamics cannot be a theorem.

## Theorem 6 (architectural open-first theorem)

**Hypotheses.** A machine M executes a program held in a shared memory that also holds its environment (the partner), in
encounters of bounded length. (H1) *Default flow:* in the absence of an executed control transfer (jump, call, return,
repeat-in-place, halt) the instruction pointer advances monotonically through memory; the organism's region is followed
by the environment's. (H2) *Bounded write bandwidth:* over any execution, bytes written ≤ β × bytes of instruction
stream consumed, with β < 1. (H3, optional) *Self-consistent literal write:* M has an instruction that writes its own
operand bytes verbatim to a memory pointer that advances, and the operand can encode the instruction's own opcode in
the phase in which it is executed.

Definitions. A pattern x of length L is a *self-replicator in context e* if the encounter (x, e) ends with the
environment's region holding a copy of x. It is *open* if its pointer executes an address in the environment in some
context, *closed* if it executes none in any context.

**Theorem 6.** (i) Under (H1)–(H2), every closed self-replicator executes some address at least twice (contains a
control cycle). (ii) Under (H1)–(H3), there is a straight-line self-replicator of length at most the literal width plus
the opcode width (in the Z80, two bytes; in BFF with `P`, one byte), and every straight-line self-replicator is open.
(iii) Hence, under (H1)–(H3), the minimal open self-replicator is strictly shorter than the minimal closed one; under
(H1)–(H2) without (H3), no straight-line self-replicator exists and every self-replicator contains a cycle (born closed).
(iv) Under uniform random initialisation over an alphabet of size σ, the expected number of occurrences of a fixed
pattern of length k among N regions of length L is N (L − k + 1) σ^−k, so the minimal open pattern is present with
probability exponentially (in the length difference) higher than any minimal closed pattern.

*Proof.* (i) is Theorem 2: confinement without a repeated address consumes at most L bytes of code and writes at most
βL < L bytes. (ii) The literal instruction tiled across the region writes its own operands, which by (H3) are the tiling
itself, at the advancing pointer; it executes no control transfer, so by (H1) its pointer runs on into the environment
and executes whatever is there: it is open by definition, and it is a self-replicator in every context whose code does
not divert the pointer or the write pointer (e.g. the all-no-op context), hence a self-replicator in some context.
Conversely any straight-line program runs into the environment by (H1) and is open. (iii) follows from (i) and (ii):
a closed replicator needs an executed control transfer, which a straight-line literal tiling does not contain, so the
closed one is longer by at least the control transfer's encoding; without (H3) a straight-line program writes only what
it reads, and the counting of Theorem 1 (generalised: a copy of L bytes with D ≥ 3 values needs ≥ 2L + D − 1 instruction
executions against ≤ L/β... in one pass) excludes a straight-line self-copy, so every self-replicator contains a cycle.
(iv) is linearity of expectation. ∎

**What it decides.** Whether a substrate's first replicator is open or closed is decided by (H3): a self-consistent
literal write channel. Z80 (`LD rr,nn ; PUSH rr`: yes) → open first, 80/80 worlds; BFF (no literal store: `+`/`−`
increment in place, `.`/`,` copy from memory) → born closed, 28/28 first replicators loop-bearing; BFF + `P` → open
first, 12/12. The theorem does not say the open phase lasts (see the classification below).

**Scope.** (H1)–(H2) are properties of the von Neumann architecture (shared code and data, sequential control, finite
write bandwidth), not of any particular instruction set: Core War's MARS, Tierra's CPU, Avida's, the Z80 and BFF all
satisfy them. They are not properties of a chemistry; a chemical reading would need an analogue of "a copier that runs
off its template into adjacent material by default" (rolling-circle or run-on polymerisation would be the candidates),
and that is a hypothesis, not a theorem.

## Lemma 7 (information-theoretic form of closure; substrate-free)

Let the offspring of x in context e be o = F(x, e), with e drawn from a context distribution. Define *leakage*
λ(x) = I(o; e | x), the information the offspring carries about the context given the parent. Then:
(i) if λ(x) > 0, then P(o = x | x) < 1: an open replicator cannot be perfectly faithful across contexts;
(ii) if P(o = x | x) = 1 then λ(x) = 0: perfect fidelity across contexts implies closure;
(iii) fidelity is bounded by Fano's inequality in terms of H(o | x) ≥ λ(x): P(o ≠ x | x) ≥ (H(o|x) − 1)/log|O|.

*Proof.* (i) If o = x with probability one given x, then o is a deterministic function of x and I(o; e | x) = 0.
(ii) is the contrapositive of (i). (iii) is Fano. ∎

**Reading.** This is Krakauer et al.'s individuality (the organism's future determined by its own past rather than the
environment's) specialised to one generation and made measurable by intervention: the partner test estimates
P(o ≈ x | x) under uniformly random e, and the pointer-entered record is a sufficient condition for λ(x) = 0 in these
machines (no read of the environment, no information about it). "Selection for fidelity is selection against
information inflow from the environment" is then a theorem, if a modest one: the Z80 pusher has λ > 0 (its copy
succeeds in 0.66 of contexts) and the `RET NZ` closer has λ = 0 (1.00 in all 256). The lemma is near-tautological; its
value is that it says what the measured quantity *is*, and that the definition of closure is the right one.

## The dynamics: a classification, not a law

**Why no theorem.** Whether closure follows the open phase depends on the context distribution the open replicator
itself creates (its copies, its failed copies, the mutants of both) and on how that distribution acts on it. Tonight's
BFF + `P` runs are the counter-example to any universal "open, then closed": open in 12/12 worlds at epoch 64, extinct
by epoch ≈ 256 in 12/12, because partial copies into pointer-halting partners are themselves pointer-halting tar that
kills the next generation. No statement about the machine alone can decide this; it is an ecological feedback.

**Classification (two binary properties of a substrate).**

| | waste benign to open organisms | waste lethal to open organisms |
|---|---|---|
| **self-consistent literal write (H3)** | open first, then closed — the Z80 soup (closure by 300k at L = 16, 50; 19/20 by 1M at L = 20) | open first, then extinct — BFF + `P` (12/12 collapse, 0/12 closure) |
| **no literal write** | born closed — BFF (28/28 loop-bearing first replicators) | born closed or lifeless — not yet run |

Model 5 gives the quantitative form inside the top row: closure occurs with probability ≈ 1 − exp(−q ∫ n dt), where
∫ n dt is the open population's integral over time (the window) and q the per-tape-epoch rate of closing mutations;
benign waste makes the window grow without bound, lethal waste closes it. The two parameters are measured, not derived:
the viability of a copy made into a waste partner (Z80: ≈ 1, zeros are no-ops; BFF: ≈ 0, an unmatched `]` halts) and
the rate at which waste accumulates. The classification is substrate-independent in form and substrate-specific in its
two entries, which is the most that is true.

**Thermodynamics.** What can be said without inventing: every copy erases L bytes of the partner (Landauer: ≥ L kT ln 2
per copy, open or closed alike), so closure does not change the erasure cost; the open organism additionally performs
whatever computation the environment's code dictates when its pointer runs into it — work it did not choose, at the
environment's command — and this is what Maturana and Varela's "operationally closed, thermodynamically open" picks
out: closure cuts the inflow of *control*, not of energy. Whether an inequality separates the two (dissipation per
faithful copy as a function of λ) is a question for a model with explicit costs; none is claimed here.

## Summary for the paper

- Theorem 6 settles *what the first replicator is* for any von Neumann machine: open if the instruction set has a
  self-consistent literal write, closed otherwise.
- Lemma 7 settles *what closure is* in substrate-free terms: zero information inflow from context to offspring, the
  limit that perfect fidelity requires.
- The classification settles *what can follow*: closure or extinction, decided by whether the open organism's own
  waste is lethal to it; the window q ∫ n dt is the quantity to measure and model, and the fourth cell is the next
  experiment.

## S2. Formal definitions, predictions and their pre-registered outcomes (THEORY.md)

### Theory: fixed points, closure and the order of events (draft 2026-10-08)

Written before any new simulator (BFF, minimal machines) is run, so that the predictions below are pre-registered. Numbers quoted from our data are in `results/closure/NUMBERS_CLOSURE.md` and the stage findings.

## 1. The setting, abstractly

A **machine** M is a deterministic byte-code interpreter with a program counter (PC) and some registers. A **world** places two byte strings, A and B, each of length L, side by side in a memory of P ≥ 2L bytes (addresses mod P), zeroes the registers, sets the PC to the first byte of A, and runs S steps. Only A and B are written back; padding is discarded.

Write **E_S(A, B)** for the memory after S steps and **T_S(A, B) ⊆ ℤ_P** for the set of addresses the PC visited. Call A the **organism** and B the **context**.

Our Z80 soup is one instance (P = 2L; S = 128; the stack pointer starts at the last byte of B). The Computational Life BFF soup is another (two 64-byte tapes, one instruction pointer, two data heads).

## 2. Three definitions

**Definition 1 (self-writer; fixed point).** A is a *self-writer against context class 𝒞* if for every B ∈ 𝒞 the second half of E_S(A, B) is a copy of A up to a cyclic shift (similarity ≥ 0.75 in our assay). Writing R(A, B) for "run and read back the second half", a self-writer is a fixed point of R(·, B) on 𝒞. Kleene's recursion theorem guarantees fixed points of program transformations in any machine that can simulate itself; what the theorem does not say is how *small* the smallest one is.

**Definition 2 (closure; forward invariance).** A is *closed* if T_S(A, B) ⊆ [0, L) for every context B, i.e. the organism's bytes are a forward-invariant set of the execution dynamics uniformly over contexts. A is *open* if some context carries the trace outside. A closed organism is *coordinate-free* if it remains closed under every translation of its position and every ring length P ≥ 2L (its control flow uses relative targets), and *coordinate-dependent* otherwise (an absolute target that happens to land inside it modulo P).

**Definition 3 (aliveness; individuality).** For a distribution 𝒟 on contexts, the *aliveness* of A is
 𝒜_𝒟(A) = Pr_{B∼𝒟}[ R(A, B) is a copy of A and R(R(A, B), B′) is a copy of A for B′ ∼ 𝒟 ],
the probability that a random context yields a copy whose copies copy. This is our gen2 assay. For the uniform distribution on random byte strings it is the quantity we report as "partners copied"; closure in the sense of Definition 2 implies 𝒜 = 1 for every 𝒟 whose contexts do not overwrite A. Definition 3 is the replicator case of Krakauer, Bertschinger, Olbrich, Flack and Ay's information-theoretic individuality: how much of the system's future is determined by its own past rather than by its environment.

The ladder we observed is a ladder of 𝒜: the zero flood (not a self-writer), the return-address smear (writes a pattern, gen2 ≈ 0.08), the open pusher (𝒜 ≈ 0.6 against random contexts, 1.0 against blank ones), the closed successor (𝒜 = 1.0).

## 3. Two lemmas (sketches; to be made exact per machine)

**Lemma 1 (open fixed points are cheap).** Suppose M has (i) a *stepping write* W that writes k ≥ 1 bytes from a register to the address in a pointer and moves the pointer by k, and (ii) a *literal load* Λ that loads the k bytes following its opcode into that register. Let λ be the opcode of Λ and ω the opcode of W. Then the string u = (λ ω)^{L/2} … more precisely, the periodic string whose period is the k-byte word consisting of the opcodes in the order they appear in the instruction stream … is a self-writer against the class of *quiet* contexts (contexts whose bytes the machine treats as no-operations): executing Λ loads the next k bytes, which are the opcodes themselves, and W writes them; the stream Λ W Λ W … reproduces the period. Its length is k + 1 bytes of information (in the Z80, k = 2: `01 c5`). *Proof idea:* direct simulation; the only requirement is that the operand bytes of Λ are the opcodes of Λ and W in the stream order. It is not a self-writer against arbitrary contexts: whenever the PC enters the context before the pointer has overwritten it, a Λ whose operand straddles the context loads foreign bytes and every later W propagates them. Hence the smallest fixed point is **open**.

**Lemma 2 (closure needs a cycle).** If S > L and A is closed (Definition 2), then the PC visits some address of A twice, so the control-flow graph of A restricted to A has a cycle, so A contains at least one instruction whose successor address is not the next byte (a jump, relative jump, loop, call or return). *Proof:* with only sequential instructions the PC increases monotonically and leaves [0, L) after at most L steps. *Corollary:* a closed self-writer must also write its own control-flow bytes, so its literal loads must carry them as data; its length is bounded below by the cycle instruction plus the bytes needed to load that instruction's encoding. In the Z80 the smallest closed self-writer we observed is 8 bytes (`ad e3 21 e3 21 c0 ad c0`: the return address it pushes is its own entry point modulo 32); the smallest open one is 2.

**Consequence (order of events).** In a machine where cost(open fixed point) < cost(closed fixed point), random assembly finds the open one first; it is heritable only against quiet contexts; selection in noisy contexts then favours any variant that closes. This predicts the observed order: tar (defaults written by W on empty registers), open replicator, closure. In a machine where the smallest self-writer already needs a cycle (no multi-byte literal write), life is **born closed** and no open phase exists.

## 4. Predictions for new systems (pre-registered here)

**P1 — BFF (Computational Life).** Semantics as published (arXiv 2406.19108 §2 and cubff `bff_noheads`): two 64-byte tapes concatenated into 128 bytes, instruction pointer and both heads start at byte 0, pointer arithmetic modulo 128, ten instruction bytes (`< > { } + - . , [ ]`), every other byte a no-op, `[`/`]` conditional jumps on the byte under head0 (terminate on an unmatched bracket), 2¹³ steps per encounter, 2¹⁷ programs, mutation 0.024%.
One more fact from `bff.inc.h` changes the prediction and is the point: **the BFF instruction pointer does not wrap** — evaluation ends when `pc` reaches 128 (or goes below 0 on an unmatched `]`). The Z80 ring laps; BFF makes a single pass. The general criterion that follows from Lemma 1 is:

> **An open full replicator exists iff a straight-line program can write at least its own length within one encounter**, i.e. (bytes written per instruction-pointer byte) × (instruction-pointer bytes executed per encounter, including laps) ≥ L.

Z80 soup: PUSH writes 2 bytes per instruction, the ring is lapped many times in 128 steps → yes (the pusher). Standard BFF: `.` writes 1 byte and needs `>`/`{` to advance, 3 bytes of code per byte written, one pass of at most 128 bytes → at most ≈ 21 bytes of a 64-byte organism per encounter → **no**. Consequences, fixed before we run anything:
(a) In **standard BFF** every *full* replicator contains a loop and is **closed from birth**: at emergence it copies into ≥ 0.95 of random partners and its instruction pointer never enters the partner. The Z80-type open→closed succession is **absent**. What precedes the looping replicators is a **chunk phase**: straight-line tilings such as `.{>` (copy head0→head1, head1 stepping back from 0 into the end of the partner, head0 stepping forward) spread ≈ 21-byte fragments backwards into partners — heredity without clones, as in our 64-step L = 100 cell. The paper's transition (high-order entropy, unique-token drop) marks the loop take-over; the theory predicts a measurable chunk-copier population before it.
(b) **BFF with a modular instruction pointer** (one switch: `pc` wraps modulo 128) admits the open full replicator: the `.{>` tiling now laps the tape ≈ 64 times in 2¹³ steps and writes the whole organism (backwards) into the partner while executing the partner's bytes. Prediction: **open first, closed later**, as in the Z80 — first replicators straight-line and partner-dependent, later dominants looping and partner-independent. One switch toggles the open phase on and off in the Google system.

> **Correction, 2026-10-08, from the harness tests (`tests/test_bff.py`), before any BFF soup ran.** The mechanism claimed for (b) is wrong. With a wrapping pointer the `.{>` tiling (and `{.>`, and `.`+`{.>`) does run for all 2¹³ steps and does enter the partner, but it does not replicate: the read head and the write head start at the same byte, cross after 64 writes, and from then on the program re-reads its own copy as source while a one-byte phase error per lap accumulates; after 32 random partners the partner is at most a 0.50 copy and the program itself is scrambled in a median > 50% of encounters. The criterion above is therefore **necessary, not sufficient**. A second condition is needed for an open full replicator: *the content written must not depend on the memory being overwritten* — a literal carried in the instruction stream (the Z80 pusher: code = data = literal), or a copy whose head trajectory is idempotent. BFF has no literal store (`+`/`−` increment in place), so an open full replicator in wrap-BFF, if one exists, must be a memory copier with a self-consistent head trajectory; none was found by hand. Status of (b), restated before the runs with both readings fixed: **(b1)** open first, closed later (the original prediction) — then pointer range was the whole story; **(b2)** born closed in both variants, as in (a) — then the open phase requires a literal write channel, and the Z80's open phase is a property of an instruction set with immediate operands and stack writes, not of ring laps alone. (b2) sharpens the theory rather than refuting it, but it is the reading the harness test favours. A constructive search over structured straight-line programs (a prefix of ≤ 3 instructions plus a tiling unit of period ≤ 5, assayed against random partners) is run alongside the soups to settle existence independently of evolution. Prediction (c) is conditional on (b1).
>
> *Search result (same night, `micro/bff_search.py`, `results/bff_search/NUMBERS_SEARCH.md`): 16,447,860 programs, 2 random partners each under a wrapping pointer; 130 pass the 75% byte-match criterion against 32 partners and 8 of those have gen2 ≥ 0.3 — all of them one-symbol fills (`,`×4 + `<` units behind a `}}` prefix) whose offspring is a tape of a single byte, "heritable" only because the partner's own stray instructions write the fill onward; offspring tapes of every passing program are 0.83 one byte. No straight-line program in the family copies a genome of period ≥ 2. This is the BFF analogue of the Z80 zero flood (tar), and it supports reading (b2) before the soups have spoken: the open full replicator needs a literal write channel.*
>
> **Run design, fixed here:** 12 seeds per variant (standard, wrap), 2¹⁷ programs, 16,384 epochs, mutation 2⁻¹², samples every 64 epochs. The paper reports a transition within 16k epochs in 40% of runs at this mutation rate (its Fig. 5), so ≈ 5 transitions per variant are expected; if a variant yields < 4, 12 further seeds are added with no change to the criteria. *Amended 03:30 the same night, before any transition had been observed in the running soups (seeds 1–12 were at epoch ≈ 2,000 with no replicator): the second block of 12 seeds (13–24) per variant is run at once on Modal (user's approval of ≈ $45), so the design is 24 seeds per variant; criteria unchanged.* **First replicator:** the first sample at which a top-3 tape class (a tape and its reverse count as one class) has share ≥ 0.5% and culture-test gen2 ≥ 0.3 — the Z80 criterion. *Amendment 03:45, after the first live samples of the Modal soups and before any verdict was computed: this share threshold is unsuited to BFF, where the first replicating populations are quasispecies — 0.69–0.97 of random tapes heritable while the most common exact class holds 0.15% of the soup. The exemplar used as "first replicator" is therefore the most common class at the first sample where it is heritable (gen2 ≥ 0.3, any share; `t_top`), the population event is a heritable fraction ≥ 0.5 of 32 random tapes (`t_her`), and the pre-registered `t_rep` is reported alongside. All three are computed from the same recorded samples; the closed/open thresholds are unchanged.* **Closed:** the tape's culture test (64 random partners) enters the partner in ≤ 5% of encounters and copies into ≥ 95%; **open:** enters in ≥ 50%. The reading of (a) is confirmed if every first replicator in the standard variant is closed and contains a loop; (b1) if ≥ half of the first replicators in the wrap variant are open; (b2) if ≥ half are closed and loop-bearing.
(e) **BFF with a literal write channel** — *pre-registered 03:50 the same night, before any run of this variant, as the direct test of the refined criterion in the correction above.* One instruction is added to BFF (byte `P`, 0x50, otherwise a no-op): `P a b` writes its two following code bytes below head1 (`head1 −= 2; tape[head1+1] = a; tape[head1] = b`) and skips them — the BFF analogue of `LD rr,nn ; PUSH rr`. Each push costs four bytes of code for two bytes written, exactly the Z80 pusher's ratio, so the one-pass bound bites: with the standard (non-wrapping) pointer the two-byte tiling `P x` copies 48 of its 64 bytes into a quiet partner (32 from its own pass, 16 more because its copy executes) and no more (`tests/test_bff.py::test_literal_push_tiling_is_an_open_replicator_only_with_a_wrapping_pointer`). With the wrapping pointer it laps, copies itself completely with its own bytes intact, runs on into the partner, and copies a fraction strictly between 0.3 and 1.0 of random partners — the Z80 pusher's phenotype. The variant run is therefore **wrap + literal** (`wraplit`); the standard-pointer + literal cell (`lit`) is a held prediction, (e4) below, not run tonight. Predictions, 12 seeds, 2¹⁷ programs, 16,384 epochs, mutation 2⁻¹², same measurements and events as the other variants: **(e1)** the first replicator (`t_top` exemplar) is a straight-line `P`-tiling in ≥ 9/12 worlds — no loop, pointer enters the partner in ≥ 50% of culture-test encounters, copies < 90% of random partners; **(e2)** emergence (`t_her`) is earlier than in standard BFF (median epoch lower, Mann–Whitney one-sided p < 0.05 against the 24 standard worlds); **(e3)** by the end of the run the dominant class in ≥ 6 of the worlds that transitioned is closed (enters ≤ 5%, copies ≥ 95%), i.e. the Z80 order of events — open first, closed later — reproduced inside BFF by adding the literal channel. **(e4, held — corrected 04:50 before any run, see THEOREMS.md)** in `lit` (standard pointer + literal) the two-byte tiling `P x` cannot copy itself in one pass (48 of 64 bytes), but the one-byte tiling `P P P …` can: a push every three bytes, 22 in its own body and 10 more in the copied tail of the partner, 64 bytes in all. Corrected prediction: `lit` behaves like `wraplit` — the all-`P` tape is the first replicator, open and loop-free; whether it then collapses as in `wraplit` is the open question (its copies into pointer-halting partners are cut short in the same way). The original (e4) wording ("the one-pass bound forbids a straight-line full copier") was a counting error for the one-byte unit and is withdrawn. **Kill criteria:** first replicators loop-bearing and closed in ≥ 6/12 `wraplit` worlds refutes (e1) and with it the "literal write channel" reading; no earlier emergence than standard BFF refutes (e2).

> **Results (2026-10-08 04:30; 60 soups; `results/bff/FINDINGS.md`, numbers in `results/bff/NUMBERS_BFF.md`).**
> **(a)** standard BFF, 24 seeds: transition in 9/24; first replicators loop-bearing 9/9, open 0/9, closed by the strict definition 7/9 (two intermediate: entered 0.03 and 0.34). The mechanism statement holds; the strict form is met in 7/9. No chunk phase before the transition (0.47 bytes transferred per encounter, copy events ≤ 1.2%).
> **(b)** wrap, 24 seeds: transition in 19/24 (more than standard, Fisher p = 0.0077); first replicators loop-bearing 19/19, open 0/19, closed 16/19 → **(b2) born closed**; pointer range is not what makes the open phase. The search (above) had already found no straight-line replicator.
> **(e)** wrap + literal, 12 seeds: **(e1) met 12/12** — the first replicator is the one-byte tiling `PPPP…` of the added instruction (its literal operands are itself), open (enters the partner in 100% of encounters), loop-free, copies 0.88–1.00 of random partners, gen2 0.75–0.92; **(e2) met** — present at epoch 64 in 12/12 with 53–60% of the soup and 53–84% of random tapes heritable (p = 6.6 × 10⁻⁸ against standard BFF); **(e3) not met** — no closed class ever appears (0/12). Instead the open wave **collapses in 12/12** by epoch ≈ 256: the soup turns into tar that terminates the pointer within 35–90 steps (unmatched `]`), copies into such partners are partial (≤ 44 of 64 bytes per pass) and dead, tar that halts before entering a partner cannot be converted, and the open replicator survives only as a 0.2% minority. Neither kill criterion was triggered.
> **(f) Benign tar and the `lit` cell — pre-registered 2026-10-08 05:55, user-approved (≈ $10 each), before any run.** Two cells of 12 seeds (1–12), 2¹⁷ programs, 16,384 epochs, same measurements and events.
> *`wraplitnh`* = wrap + literal + **no-halt** (an unmatched bracket is a no-op instead of ending the encounter; tests in `tests/test_bff.py`). A constructive search (`micro/bff_closed_search.py`) over all periodic tilings of {`P`, `[`, `]`, no-op} up to period 8 (77,540 tilings, three partners each) finds **no closed heritable replicator**: the four closed tilings that reach exactly 75% similarity are loops whose pushes write a sterile `[]` fill; 640 open replicators exist (the `P` family). (Period ≤ 10, run before the launch at 10:36: 1,309,528 tilings, three partners each — **0 closed tilings copy ≥ 90% into every partner**, 75 open ones do, 734,576 are closed non-replicators (immune tar); `results/bff_closed_search/`.) Predictions: **(f1)** the first replicator is the open all-`P` tiling in ≥ 10/12 worlds (as in `wraplit`); **(f2) the open population persists**: heritable fraction of random tapes ≥ 0.5 at the final sample (epoch 16,384) in ≥ 6/12 worlds, against 0/12 in `wraplit` — this is the test of the window model's mechanism (collapse was caused by pointer-halting tar); **(f3)** no closed class appears (0/12), since none exists up to period 8; a closed class would mean the search's period bound was too small and is reported as such. **Kill:** collapse (heritable fraction < 0.1 at the end after reaching ≥ 0.5) in ≥ 9/12 worlds refutes the lethal-tar explanation of the `wraplit` collapse.
> *`lit`* = standard pointer + literal (no wrap, halting as published): per the corrected (e4), **(l1)** the all-`P` tiling first, open, in ≥ 10/12; **(l2)** collapse as in `wraplit` (≥ 9/12), since the halting tar is unchanged. A persistent open population here would mean the pointer wrap, not the tar, drove the collapse.
> Interpretation fixed in advance: `wraplitnh` persisting without closure is the "M1" cell of the minimal-machine ladder (P2) realised inside BFF — a substrate where the open organism is stable and closure is impossible — and `wraplit` is the lethal-tar cell; together with the Z80 they fill three cells of the classification in THEOREMS.md Part II.
>
> **Consequence for the theory.** Lemma 2's selection argument needs a time condition: closure can evolve only while the open form persists. The open phase is a *window* whose length is set by how lethal the context is to an open organism — too benign (Z80 zeros are no-ops; L = 64 pusher self-damage 0.06) and closure is slow, too lethal (BFF brackets halt the pointer) and the open form is extinct before a closed variant is found. The Z80 soup sits in between. Predictions to pre-register next: (i) a BFF-with-literal variant whose tar is benign (an unmatched `]` as a no-op instead of a halt) keeps the open phase and then closes; (ii) the `lit` cell (e4); (iii) in the Z80, raising the lethality of contexts (e.g. a random byte density of halting instructions, if any) shortens the open window and, past a threshold, prevents closure.

(c) **Instruction density** (map k byte values to each instruction, 10/256 … 250/256 of bytes active) makes random code noisier without changing the language. In variant (b): the open replicator's partner-copy success falls with density and the time from first replicator to closed dominant falls with it. In standard BFF (a): density changes nothing about closure (the replicators never read the partner), a useful null.
(d) Measurement is direct because we own the interpreter: closure = the instruction-pointer trace stays in [0, 64) for every context (Definition 2, exact; recorded per encounter as "entered the partner"), chunk transfer = bytes of the partner equal to the organism at best alignment including reversal, plus the culture test (Definition 3).
Kill criteria: an open (partner-entering) full replicator in standard BFF refutes the criterion; a looping first replicator in ≥ 50% of modular-IP worlds refutes (b); no density dependence in (b) refutes the noise mechanism.
Note on the paper's long-tape remark ("trivial (non-looping) self-replicators rapidly take over" when the heads start at the program counter): there the execution runs along a 65,536-byte tape from a random position, so a straight-line copier can write far more than its length — the criterion is met, and the open form appears, exactly as predicted.

**P2 — the minimal machine ladder** (one byte = one opcode; byte values outside the active set are no-operations; memory, PC, pointer and registers as in §1; soups as in the Z80 experiment):
- *M0*: no stepping write (only register and in-place operations) → no self-writer; no life at any horizon.
- *M1*: Λ (2-byte literal) + W only → the open 2-byte self-writer appears within ~10³ encounters per cell; no cycle instruction exists, so **closure can never evolve**; the heritable fraction of the population plateaus near the open replicator's 𝒜 against the soup's own contexts.
- *M2*: M1 + one relative jump → open first, then a closed successor containing the jump; time to closure **decreases with the density of active opcodes** (noisier random code = stronger selection for closure), and is longer than the time to emergence by at least an order of magnitude.
- *M3*: a 1-byte copy that needs a loop (no literal) → born closed (𝒜 ≥ 0.95 at emergence), slower emergence than M1 (more bytes must assemble).
Kill criteria: an open replicator in M3, or a closed one in M1, or no life in M1, refutes the lemmas as applied; a closure time that does not fall with opcode density refutes the selection mechanism.

**P3 — geometry vs topology.** The L = 16 Z80 closer (`RET NZ`, absolute target 0x21E3 ≡ 3 mod 32) loses closure on rings where 0x21E3 mod P does not land in its body (e.g. P = 33, 34, 35, 37); the DJNZ (L = 36) and JR NZ (L = 50) closers keep it on every ring. Executor test, no soup needed.
*Result (2026-10-08, run right after this was written; 256 random partners, one 128-step encounter):* the `RET NZ` closer copies 1.00 of partners at P = 32 and **0.00 at every other ring** (33, 34, 35, 37, 40, 48), with self-damage 0.59–1.00: coordinate-dependent, as predicted. The DJNZ closer copies 1.00 at P = 72 but only 0.12–0.73 at P = 73, 74, 75, 79: its *control* stays closed (relative jump) but its *writes* travel through the stack pointer, whose path across the ring is geometric, so part of the copy lands in the discarded padding. The open pusher is ring-insensitive (0.50–0.75 everywhere) because it was never closed. Reading: control closure can be made coordinate-free; the reproductive channel of this machine (writing through a pointer anchored at the end of the partner) cannot. "Topological" applies to the organism's control, not to its means of writing.

**P4 — environment noise selects closure.** In the Z80 soup, the fraction of random contexts the open pusher survives rises toward 1 as the context's density of active (non-NOP) bytes falls (measured: 0.67 at 0% zeros, 0.85 at 70%, 1.00 at 100%). Prediction for M2: with a sparse active set the open replicator is sufficient and closure is not selected within the horizon.

## 5. Thermodynamic reading (directions, not claims)

Steps per encounter are the free energy an encounter can spend; writing over a context is erasure. The open organism *imports* information (it reads context bytes into its registers) and the import is what poisons it; the closed organism only *exports*. Closure is the shutting of the information inflow while the flow of "energy" stays open: Maturana and Varela's "operationally closed, thermodynamically open", made countable. The budget threshold (E4) is then the energy at which one encounter can erase and rewrite a whole genome, and the communal regime below it is heredity carried by partial erasures. Candidate formal connections: England's dissipation bound for self-replication; Still, Sivak, Bell and Crooks on non-predictive information as dissipation (the open organism stores context information that predicts nothing about its own future). To be pursued only if an inequality can be written and checked.

## 6. What would make this a theory rather than a reading

1. Exact statements and proofs of Lemmas 1–2 for a specified machine class, and an exhaustive enumeration of the smallest open and closed self-writers in M1–M3 (finite search).
   *Done for the Z80 itself (2026-10-08, `two_byte_census.py`, `results/census2/NUMBERS_CENSUS2.md`): all 65,536 two-byte words tiled to L = 16 through the culture test (32 partners, 128 steps). 254 words pass gen2 ≥ 0.3, but 246 of them are `CALL` return-address smears that copy into 0.00 of partners at the 75% level (the known weakness of gen2 alone; the faithfulness criterion exists for this reason) and only 8 are faithful. Exactly five words copy into ≥ 0.5 of 256 random partners, all of them load–push pairs — `01 c5`, `11 d5`, `21 e5` (`LD rr,nn ; PUSH rr`, copied 0.66–0.69, self-damage 0.31–0.37) and `2a e5` / `e5 2a` (`LD HL,(nn) ; PUSH HL`, copied 0.70 / 0.81, self-damage 0.39 / 0.14, coordinate-dependent because the load address is taken modulo the ring) — and none copies ≥ 0.95. So Lemma 1 is constructive for this machine (the open two-byte self-writers are the load–push words and nothing else) and Lemma 2's corollary holds at two bytes (no closed two-byte organism exists; the only two-byte cycle, `JR −2`, writes nothing). The literal pusher `01 c5` is the one that appears first in 63/80 Stage G worlds although `e5 2a` copies better in isolation: the literal is coordinate-free and works at every L, the memory-load variant is not.*
   *The minimal machine ladder M0–M3 (P2) is deferred: a first design pass (same night) showed that the "needs a loop" property of M3 is delicate — any self-advancing one-byte copy instruction makes a trivial period-1 open replicator — so the ladder needs its own pre-registration rather than an overnight implementation.*
2. P1–P4 run as pre-registered experiments.
3. Stage G confirming the Z80 order of events with new seeds.
4. A second real instruction set (beyond BFF) to test universality.

## S3. Pre-registration and change log (PLAN.md)

### Instruction-set ablation atlas of self-replication — pre-registered plan

Written 2026-10-07 **before** any sweep ran. Changes after this date are logged
at the bottom with reasons. The point of this file is to keep us honest: the
hypotheses, conditions, outcomes and analysis are fixed here, and we report
every cell of the grid, including the empty ones.

## Question

Which Z80 instruction families does the spontaneous emergence of self-replication
in the Algocell soup depend on, how does the step budget per interaction and the
mutation rate modulate that dependence, and what replication mechanisms appear
when the usual ones are removed?

## System under study

The deployed Algocell simulation, run headlessly with the *exact* shipping WGSL
(exported by `npm run export:sim`; sha in `algocell_exp/shader/meta.json`,
zilion 0.2.0). Square grid only (hex is out of scope for this study), 160×125 =
20,000 cells, 8,192 random neighbour pairs per step, concatenated A+B execution
with all registers zero, byte-replacement mutation of `8192 / 2^k` bytes per
step. No fitness function. GPU scheduling makes runs non-deterministic even for
a fixed seed, so each (condition, seed) is one sample of the process.

**Tape length (organism size)** is a factor: L ∈ {4, 9, 16, 25, 36, 49, 64,
81, 100} bytes (square numbers, so cells tile as √L×√L). Two conventions that
follow from this, fixed before any run:

- *Stack start.* A real Z80 resets SP to 0xFFFF, which for 32-byte pairs
  aliases to the last byte of the second program. For pair lengths that do not
  divide 65536 that alias lands at an arbitrary byte, so SP starts at the
  largest 16-bit value that aliases to the last byte of the pair (`spInit`).
  This is identical to 0xFFFF for L=16, i.e. the deployed system is unchanged.
- *Mutation.* `8192 / 2^k` byte replacements per step regardless of L, so the
  expected number of mutations per **cell** per step is independent of L
  (2.56% at k=4) while the per-byte rate falls as 1/L. We treat mutation as a
  per-organism hazard; the per-byte view is reported alongside.

Larger tapes need more instructions per interaction to be copied at all (a
PUSH-based copier needs L/2 pushes), so the step-budget axis is crossed with L.

## Hypotheses (stated so they can fail)

- **H1 (block copy is dispensable early):** removing LDI/LDD/LDIR/LDDR does not
  change the time to first replicator at the default settings. (Prediction from
  Computational Life: stack replicators come first.)
- **H2 (the stack is the early bottleneck):** removing the stack-writing
  instructions (PUSH, EX (SP),HL, CALL, RST) delays emergence by more than an
  order of magnitude at 128 steps, and the replicators that eventually appear
  are LD-loop or block-copy based.
- **H3 (loads are nearly necessary):** removing all LD families, stack and EX
  ("No-copy", 112 instructions) leaves only read-modify-write and return-address
  pushes as ways to write memory; we predict no replicator within the horizon at
  any step/mutation setting. *The user remembers seeing emergence here with
  short budgets and high mutation; we test that memory as a hypothesis, not a
  fact.*
- **H4 (step budget is the strongest knob):** cutting the step budget from 128
  to 32 delays emergence more than any single-family ablation and changes the
  winning mechanism (pilot: PUSH HL/LD HL,(nn) at 32 steps).
- **H6 (size):** time to emergence grows with L at a fixed budget, because the
  minimal replicator must copy more bytes within the same budget; at 512 steps
  large organisms emerge, and their replicators have more non-replication bytes
  ("free tape"). The user's expectation that larger organisms support more
  complex emergence is tested as: more distinct mechanism classes and a longer
  tail of coexisting quasispecies at large L. We do not assume it.
- **H5 (mutation is non-monotone):** for a fixed budget there is an
  intermediate mutation rate that minimises time to emergence; both very low
  and very high rates delay it (error threshold above).

## Design

Two pre-registered stages; a third is planned only in outline and will be
specified (and logged) after Stages A and B are read.

**Stage A — ablation × budget × mutation at L = 16.** Full factorial:

| factor | levels |
|---|---|
| ablation | `none`; `block-copy`; `stack-writes` = family:stack + family:ex + family:call-ret + family:rst (POP/RET included for simplicity); `ld-mem` = family:ld8-mem + family:ld16-mem (memory loads only); `all-ld` = every LD family; `no-copy` = all-ld + stack + ex + block-copy (the preset); `rmw-only` = family:writes-mem minus incdec-mem/rotate-mem/bit-set-mem (only read-modify-write can write memory) |
| z80 steps | 32, 128, 512 |
| mutation 1/2^k | k = 2, 4, 6 |
| seeds | 10 (seeds 1–10) |

7 × 3 × 3 × 10 = 630 runs. Horizon 300,000 steps. 8,192 pairs are drawn per
step but only those whose two cells are not claimed by another pair run:
measured on the GPU, 3,982 ± 35 pairs per step (48.6%), i.e. 0.40 interactions
per cell per step, ≈ 1.2 × 10^9 interactions and ≈ 1.2 × 10^5 participations per
cell over 300,000 steps (corrected 2026-10-07; the original text said ≈ 120 per
cell, off by three orders of magnitude). Mutation: 8192/2^k bytes per step over
20,000 cells, i.e. one mutated byte per cell every 2^k × 2.44 steps regardless of
L (39 steps at k = 4), so the per-byte rate is proportional to 1/L. Stopping early 4
samples after quasispecies occupancy exceeds 50%. Sampling every 500 steps.

**Stage B — organism size.** L ∈ {4, 9, 16, 25, 36, 49, 64, 81, 100} ×
ablation ∈ {`none`, `block-copy`, `stack-writes`, `no-copy`} × steps ∈
{128, 512} × k = 4 × 10 seeds = 720 runs, same horizon and stopping rule. The
L = 16 cells are shared with Stage A (not re-run).

**Stage C — long horizons, uncensored succession, finer ablations.** *Outlined before
A/B; fixed 2026-10-07 after reading them; re-specified the same day after the design
review (REVIEW.md §4) and before any Stage C result was read. The exact list is
`make_conds.stage_c()` → `conds/stageC.json` (490 runs).* Common to every C/D/E run:
no early stop; sampling every 50 steps until step 5,000 and every 500 thereafter
(emergence in A/B sat at the 500-step resolution floor) **and at steps 1, 2, 3, 5, 8, 13,
21, 34** (the zero flood forms within the first ~30 steps: 6.5% zeros after step 1, 31% by
step 50, so a 50-step grid would record only its plateau); the 256-bin byte histogram, the
zero-byte fraction and the active-pair count in every sample; the shader's per-pair write
counters summarised in every sample (silent-interaction fraction, writes into partner and
self, histogram); **full soup snapshots at 500, 1k, 2k, 3k, 5k, 7.5k, 10k, 15k, 20k, 30k,
50k, 75k, 100k, 150k, 200k, 300k (and 500k, 750k, 1M)** steps, each with an **interaction
census** (before/after state of every pair that interacted in that step: copy events A→B
and B→A at best cyclic shift ≥ 0.75, partial copies, bytes changed, zeros written, copy
offsets, programs destroyed; aggregates plus 128 raw pairs); 16 uniformly random cell
tapes and the top-10 exemplars per sample; full provenance (shader and ISA hashes, git
commit, library versions incl. brotli, adapter) in every record. **Clock:** because the
fraction of drawn pairs that survive the parallel collision claim depends on GPU
scheduling (48–56% depending on grid size and load), every cross-arm comparison in time
is made in **cumulative active interactions per cell** (from the recorded `active_pairs`),
with raw steps shown alongside. **Seeds 101–110**: Stage A/B used seeds 1–10, and
a seed fixes the initial soup and the RNG stream, so the cells that generated the
hypotheses are not re-used.

- **C1 — slow or impossible?** `no-copy`, `rmw-only`, `all-ld` at (128, k=4) and
  (32, k=2), **1,000,000 steps** (sampled every 1,000 after 5,000).
- **C2 — succession without censoring.** `none`, `block-copy`, `ld-mem` at 128 and
  512 steps × k ∈ {2, 4, 6}, 300,000 steps.
- **C3 — finer ablations** at (128, k=4) and (32, k=2), 300,000 steps: `push-only`
  (PUSH+POP, 8), `ex-sp-only` (1), `call-rst` (CALL*, RET*, RST; 34), `ld-imm` (11),
  `ld-reg` (54), `cb-page` (256), `ed-loads` (8), and — added by the review — the
  write/read split of the Stage A arm: **`stack-write-only`** (PUSH, EX (SP),HL, CALL*,
  RST; 22 opcodes, the ones that write through SP) and **`stack-read-only`** (POP, RET*,
  EX DE,HL, EXX, EX AF,AF'; 24 opcodes that write nothing); the Stage A arm
  **`stack-writes`** itself re-run with the new seeds (a direct replication of H2 under the
  C instrumentation); and the unablated comparator `none` at (32, k=2), which C2 lacked. The Stage A/B arm
  `stack-writes` (46) is their union and is misnamed: the LDIR zoo uses POP DE and
  EX DE,HL for pointer set-up, so its 20–100× delay confounds write removal with
  pointer-route removal.
- **C4 — functional fraction over time**, from the random tapes of every run.
- **C5 — size without censoring:** `none` and `stack-writes` at L ∈ {36, 100},
  (128, k=4), for continuity with Stage B's arm; the full size axis without censoring
  is Stage E.

Predictions, stated before any C run is read: C1 — `no-copy` and `rmw-only` remain
at 0/10 at 1M steps; `all-ld` emergence fraction rises above 0.3. C3 —
`stack-write-only` reproduces the Stage A `stack-writes` delay within a factor 2 and
`stack-read-only` is indistinguishable from `none` (seed-paired sign test, ≤ 7/10
concordant); `ex-sp-only` and `push-only` each delay less than `stack-write-only`;
`ld-imm` is the load family whose removal costs most; `cb-page` and `ed-loads` change
nothing at L = 16. C5 — dominant tapes at L = 100 have period ≤ 8 in most seeds.

**Stage D — the L = 9 reversal, confirmatory.** *Fixed 2026-10-07 after reading Stage
B; re-specified the same day after the review and before any Stage D result was read
(the first version re-used seeds 1–20, i.e. Stage B's own initial soups, and did not
separate write-side from read-side stack opcodes). `make_conds.stage_d()` →
`conds/stageD.json` (280 runs).* Stage B observed, post hoc, that removing the 46-opcode
stack arm at L = 9 raises emergence from 4/10 and 1/10 to 9/10 and 10/10 (Fisher
two-sided p = 0.057 and 1.2 × 10⁻⁴; one-sided 0.029 and 6 × 10⁻⁵), with every L = 9
replicator LDIR-based, mostly the 3-byte unit `DEC E ; LDIR`.

Design: L = 9, k = 4, **seeds 1001–1020 (new)**, 300,000 steps, Stage C
instrumentation. Arms × budgets {128, 512}: `none`, `stack-writes` (46, Stage B's
arm), `stack-write-only` (22), `stack-read-only` (24), `push` (PUSH only, 4),
`call-rst-write` (CALL*, RST; 17) = 240 runs; plus `none` at 32 steps and `none` at
k = 6 (128 steps), 20 seeds each.

Outcomes and tests, fixed now. Primary: emergence fraction by `t_rep` (heritable) and
by `t_faith` (faithful); **D1**: `stack-write-only` > `none`, pooled over the two
budgets by a one-sided Cochran–Mantel–Haenszel test at α = 0.05; secondaries per budget
with Holm over the two budgets. **D1b**: `stack-writes` > `none` (direct replication
of Stage B, one-sided). **D2** (restated 2026-10-07 before any Stage D run, after a local seed-1001 run showed
≈ 0.19 zeros within 500 steps with the 22 writing stack opcodes removed — LD (HL),r,
LD (nn),rr and block copies of zero registers also write zeros, so the stack *doubles* the
flood rather than creating it): at step 5,000, zero_frac(`none`) − zero_frac(`stack-write-only`)
≥ 0.10, both reported against the 1/256 random baseline; exploratory: which non-stack
writers carry the residual flood (byte histogram, write counters, interaction census).
**D3**: the fraction of the effect carried by the write side,
(frac(`stack-write-only`) − frac(`none`)) / (frac(`stack-writes`) − frac(`none`)),
is ≥ 0.75, and `stack-read-only` is within 0.1 of `none`; `push` versus
`call-rst-write` says which writer matters (no directional prediction). **D4**: every
heritable L = 9 replicator is LDIR-based with period 3 or 9. **D5** (per-byte-hazard
account of the floor): `none` at 32 steps has a higher emergence fraction than `none`
at 128 steps (one-sided Fisher). The k = 6 arm is exploratory. If D1 fails, the Stage
B observation is reported as an unreplicated single cell.

**Stage E — size-axis controls.** *Specified 2026-10-07 after the review, before any
run. `make_conds.stage_e()` → `conds/stageE.json` (1,230 runs).* Stage B's size axis has
four confounds that could each produce "flat emergence time above L = 16": the
per-byte mutation rate falls as 1/L; the number of 4-byte windows per cell and the
total soup bytes grow with L at fixed 20,000 cells; the Z80 budget per tape byte falls
from 8 (L = 16) to 1.28 (L = 100); and tape parity drives the early stop (even L
phase-lock at q_share ≥ 0.5 and stop at 3.5–5k steps, odd L split into two phase
variants and run to 300k), so "final" states of different L had different ages.
Arms, each for `none` and `stack-write-only`, 128 steps, k = 4 nominal, seeds 101–110,
no early stop, fine early sampling:

- **@nominal** — Stage B's setting without the early stop, at the nine square L, at
  **L ∈ {8, 10, 12, 18, 20, 24, 32, 50}** (headless only; shaders and executors exported
  for them) to break the parity alternation and resolve the floor between 9 and 16, and
  at **L ∈ {3, 5, 6, 7}** to find where life starts (the 3-byte unit `DEC E ; LDIR`
  exists). The three control arms below run at the nine square L only.
- **@mubyte** — per-byte mutation rate held at the L = 16 value: 32·L mutated bytes
  per step instead of 512.
- **@bytes** — total soup held at 320,000 bytes: cells = 320,000/L (80,000 at L = 4 …
  3,200 at L = 100) with the drawn pairs scaled with the cell count.
- **@steps8L** — Z80 budget proportional to length: 8·L steps (32 at L = 4 … 800 at
  L = 100).
- **@musweep** — L = 100, 128 steps, k ∈ {1, 2, 3, 5, 8} (k = 4 is @nominal): the
  maintained period/information of the replicator versus the per-byte mutation rate, the
  error-threshold curve.
- **@budget** — L = 100, k = 4, steps ∈ {32, 64, 256, 1024, 2048} (128 is @nominal):
  faithfulness and whole-tape copiers versus bytes copyable per encounter; prediction:
  faithful fraction rises from ≈ 0 at 128 to ≈ 1 at ≥ 512, and whole-tape copiers with
  cargo appear only at ≥ 512.
- **E6 — within-seed variance:** 10 repeats of seed 1 in three cells (`none` L = 16,
  `stack-writes` L = 16, `none` L = 9) to measure how much of the between-run
  variance is GPU non-determinism; if it matches the between-seed variance, seeds are
  exchangeable replicates. At L = 16 the three control arms coincide with @nominal and
  with the Stage C cells (32·16 = 512 mutations; 320,000/16 = 20,000 cells; 8·16 = 128
  steps): these 100 physically identical conditions are **declared cross-batch replicates**
  and enter the variance analysis, never a pooled table (the analysis asserts that each
  (ablation, L, steps, k, seed, replicate) appears once).
- **E7 — the 32-step · 1/4 block-copy exception** (Stage A: 0/10 vs 10/10 in one of 63
  cells) with 20 new seeds (1001–1020), `none` and `block-copy`.

Predictions: if "flat above 16" is real, the four arms agree on the emergence
fraction and KM median at every L ≥ 16; if the per-byte mutation rate drives it,
@mubyte shows emergence time growing with L; if the lottery (bytes/windows) drives
it, @bytes does; if steps per byte drives faithfulness, @steps8L restores faithful
replication at L = 81–100 at the 128-equivalent budget. Tiling: the first-replicator
period divides 2L in ≥ 90% of runs in every arm (ring arithmetic does not depend on
L, mutation or budget). Floor: the non-square lengths place the floor between L = 9
and L = 16; at L = 12 (3 | 12 and 4 | 24) the 3-byte LDIR unit and the 4-byte stack unit
both tile, so L = 12 should behave like L = 16 rather than like L = 9.

Costs (L40S list price; estimator fitted to the measured per-L rates of Stages A/B plus
host time per sample) are quoted from the preflight logs in the change log. Every stage is
preflighted locally (`preflight.py`: tests, condition integrity, no foreign runs in the
volume directory, a dry run of every distinct parameter signature through the exact Modal
code path) and smoke-tested on Modal (2 short runs + 1 invalid condition) before launch.

**Stage F — ring arithmetic (specified 2026-10-07, before any run; instrument built and
tested the same day).** The pair memory can now be padded to P ≥ 2L bytes: addresses wrap
modulo P, the padding is zero and never written back, and SP aliases the last byte of B
under the ring modulus so the stack family is untouched (verified: `LD BC,nn ; PUSH BC`
scores gen2 0.51–0.65 on P ∈ {32, 33, 34, 35, 37}; the offset-4 shift-copier is heritable on
P = 32 only). `make_conds.stage_f()` → `conds/stageF.json`: `none` and `stack-write-only` at
128 steps, k = 4, seeds 101–110, L = 16 with P ∈ {33 = 3·11, 34 = 2·17, 35 = 5·7, 37 prime}
and L = 36 with P ∈ {73 prime, 74 = 2·37, 75 = 3·5², 79 prime}; the P = 2L rows are Stage E's
@nominal runs (160 runs, ≈ $12).
Predictions. **F1 (divisor law):** the period of every tiled first replicator under
`stack-write-only` divides P (not L): periods {3, 11} at P = 33, {2, 17} at P = 34, {5, 7} at
P = 35; {2, 37} at P = 74; {3, 5, 15, 25} at P = 75. **F2 (prime ring):** at P = 37 and P = 73/79
no tiled LDIR replicator appears; emergence under `stack-write-only` falls to the BC-limited
exact-copier route (`LD DE,L ; LD BC,L ; LDIR`-type designs, three register set-ups instead of
one) and is therefore at least 10× slower by KM median than at P = 2L, or absent within 300k
steps. **F3 (control):** under the full ISA the Load–Push family emerges with the same KM
median (within one 50-step sample) at every P. If F1 fails, the gcd mechanism verified in the
executor does not govern the population dynamics and the "ring arithmetic" claim is dropped.

**F4 — is the L = 9–12 dead zone ring arithmetic? (added 2026-10-07 after Stage E was read
and before any F run; 50 runs, `none` only, 128 steps, k = 4, seeds 101–110.)** Stage E found
the pusher regime at L = 8 and L ≥ 16 but not at L = 9–12, and the isolated assay found the
pusher's own heritability (gen2, 64 random partners, 128 steps) marginal there. An executor
probe on padded rings (`runs/unit_fitness_vs_ring.csv`, run before this paragraph was written)
gives, for `01 c5` tiled to L: **L = 12: 0.20 at P = 24, 0.60 at P = 28, 0.45 at 32, 0.58 at 36,
0.53 at 48**; L = 10: 0.31 at P = 20, 0.28 / 0.36 / 0.29 / 0.37 at P = 24 / 28 / 32 / 40 (no
rescue); L = 9: 0.32 at 18, 0.44 / 0.33 / 0.36 at 20 / 24 / 32; L = 8: 0.53 at 16, 0.62 at 20,
0.55 at 24, **0.28 at 32**; L = 16: 0.51–0.61 at every P ∈ {32 … 48}. Cells: `none` at L = 12
with P ∈ {28, 36}, at L = 10 with P ∈ {28, 40}, and at L = 8 with P = 32 (the P = 2L rows are
Stage E's @nominal runs). Predictions, population level: **F4a** at L = 12 · P ∈ {28, 36} the
pusher emerges as at L = 16 — heritable in ≥ 9/10 seeds, KM median < 2,000 steps, modal
first-replicator period 2 (Stage E at P = 24: 6/10, KM 245,500, all LDIR); **F4b** at L = 10 ·
P ∈ {28, 40} no rescue — ≤ 6/10 heritable or KM median > 50,000 (Stage E: 4/10, median not
reached); **F4c** at L = 8 · P = 32 the pusher is impaired — fewer than 10/10 heritable or KM
median > 2,000 (Stage E at P = 16: 10/10, KM 500). If F4a holds and F4b holds, the dead zone
is a property of the stack-pointer wrap on the ring and not of the tape length as such; if F4a
fails, the isolated-assay heritability does not predict population emergence and the
copy-geometry explanation is dropped. Executor-level note for F1 logged before the run: on
P = 34 the isolated tiled units `04 5e ed b0` and `1d ed b0` are heritable (gen2 0.62, 0.68)
although 4 ∤ 34 and 3 ∤ 34; F1 (emergent periods divide P) stands as written and this probe
is the first thing to check if it fails. Cost: ≈ 1.3 GPU-h ≈ $3 on top of F's ≈ $12.

**Stage G — closure, confirmatory (pre-registered 2026-10-08, before any G run; 80 runs,
`make_conds.stage_g()` → `conds/stageG.json`).** The post hoc analysis `closure.py` over 233
`none` worlds of Stages B, C and E (128 steps, 1/16) found: the first heritable replicator is
straight-line code (control-flow instruction in 18/256) that copies into 54–82% of random
partners and damages itself in up to 37% (phase 1: up to 89%) of encounters; the faithful
dominant at 300k carries a control-flow instruction in 105/233 worlds (paired: 92 gained, 3
lost, McNemar p = 7 × 10⁻²⁴) and every such successor copies into 100% of 256 random partners
with no self-damage; the population's heritable fraction rises from 0.07–0.14 to 0.78 (C4).
New seeds **2001–2020**, `none`, 128 steps, 1/16, recording as in C–F (16 random tapes per
sample for the heritable fraction).

- **G1** — L = 16 and L = 50, 300,000 steps (40 runs). Predictions per L: **(a)** the first
  heritable replicator contains no control-flow instruction (jump, relative jump, DJNZ,
  CALL/RET, RST; linear disassembly) in ≥ 18/20 worlds and copies into < 90% of 256 random
  partners in ≥ 18/20 (executor test, one 128-step encounter); **(b)** at 300k the top
  exemplar is faithful and contains a control-flow instruction in ≥ 16/20 worlds, and the
  modal final tape copies into ≥ 95% of 256 random partners with ≤ 5% self-damage; **(c)** the
  median heritable fraction of 16 random cells is < 0.20 at every sampled step ≤ 5,000 after
  emergence and > 0.70 at 100,000. **Kill criterion:** fewer than 12/20 control-flow dominants
  at 300k, or ≥ 6/20 first replicators with control flow, rejects "open first, closed later"
  as stated for that L.
- **G2** — L = 20 and L = 64, 1,000,000 steps (40 runs; sampled every 1,000 after 5,000).
  Stage E: the pusher is still dominant with no control flow at 300k in 8/10 and 10/10 worlds.
  **Prediction:** ≥ 10/20 worlds per L have a faithful control-flow dominant at 1M that
  copies ≥ 95% of random partners. **Alternative** (closure needs a geometric coincidence that
  only some L offer): < 5/20. The prediction is the first; either result is reported.

Analysis: `stage_g.py` (first/final control flow, executor partner tests, C4 heritable
fractions at 5k/100k/300k/1M), numbers only from generated tables. Cost from the preflight
log (estimate ≈ 9 GPU-h ≈ $17 at list price; approved in principle at "about $10" — launched if
the preflight estimate is ≤ $20, otherwise held for the user).

## Outcomes (all recorded per sample, nothing chosen after the fact)

- `t_02`, `t_10`, `t_50`: first step at which the most common 16-byte tape holds
  ≥ 2%, 10%, 50% of cells (exact match); −1 = censored at the horizon.
- `tq_10`, `tq_50`: the same on **quasispecies occupancy** `q_share`, the
  fraction of cells within Hamming distance ⌈L/4⌉ of the dominant tape (4 for
  L = 16; `q_half_share` uses half that radius). Exact-hash
  share undercounts a replicator because mutation keeps splitting it into
  near-identical variants (local pilot, 2026-10-07, before the sweep: the
  classic replicator plateaus near 10% by exact hash while the grid is visibly
  full of it).
- The dominant tape at each threshold crossing and its **mechanism classes**
  (stack / block-copy / ld-mem / rmw / io), from a linear disassembly of the
  tape's write-capable instructions.
- `motif_share` (secondary): fraction of cells containing the dominant tape's
  most frequent 2-byte word at least 4 times; the "grid looks full of it"
  measure for replicators whose copies drift beyond Hamming 4.
- Species Shannon entropy, Simpson index, richness (≥ 0.1%), unique tapes.
- Byte entropy H0 and high-order entropy (H0 − brotli bits/byte), the
  Computational Life paper's replicator signal.

Primary outcome: `tq_10` (10% quasispecies occupancy is unambiguous
replication; 2% exact share is noisy because return-address smears reach
~0.5%). `t_10` is reported alongside for comparison with exact-species work.

## Analysis

- Per condition: fraction of seeds reaching `tq_10` within the horizon, with
  Wilson 95% intervals; median `tq_10` among those that did; Kaplan–Meier curve
  treating censoring properly (no imputation of censored runs).
- Mechanism table: for each condition, the distribution of mechanism classes of
  the first emergent replicator.
- H1: compare `none` vs `block-copy` at (128, k=4): overlapping intervals and a
  median ratio within [0.5, 2] = supported.
- H2: `stack-writes` vs `none`: median ratio > 10 or ≥ 50% censored = supported.
- H3: any `no-copy` run with `tq_10 > 0` refutes it; we then report the tape.
- H4/H5: monotonicity of median `tq_10` along the steps and mutation axes for
  `none`, with the sign test across seeds.
- H6: median `tq_10` vs L at each budget; number of mechanism classes and the
  species-entropy plateau after emergence vs L.
- Everything is plotted as the full grid (small multiples), not only the cells
  that behave.

## Rules

- No horizon, threshold or condition is changed after seeing results except by
  adding a logged entry below (additions are fine; removals are not).
- Nulls are results. A censored cell is reported as censored, never dropped.
- Single-seed observations (including the user's) are hypotheses.
- Anything that looks like a new mechanism is confirmed by assaying its tape with
  the single-pair executor and by its recurrence across seeds. Re-running a seed is
  NOT a replication: the GPU's atomic claim order and a non-atomic mutation write make
  trajectories diverge within a few hundred steps even for the same seed (four local
  seed-1 runs reached different dominant tapes); each run is one sample.

## Change log

- 2026-10-08 (post hoc analyses on recorded data, no new GPU spend): **census + nascent scan**
  (results/stageD/census): under the full ISA at L = 9 an LDIR-bearing tape enters the top-10 in
  5/20 runs (first at 118,500 steps) while the soup destroys 6.8 programs per 1,000 encounters
  and its only copy events are sterile smears; without the stack writers 20/20 within 3,400
  steps, 6.5 flicker episodes, 1.3 destroyed per 1,000 — the suppression prevents assembly
  rather than killing established copiers. **Invasion assay** (results/invasion): the pusher
  seeded into 1% of cells at 32 steps is extinct by 20k in 20/20 soups while the LDIR unit
  makes 20/20 soups 75–100% heritable; the 32-step puzzle is a viability limit. **Detector
  benchmark** (results/detectors): compression detectors AUC 0.97 against the C4 heritable
  fraction at L = 16; high-order entropy anti-correlated with life under `none` at L ≥ 25
  (the flood compresses); occupancy ≤ 0.75 at every step. **Stage D forest figure** added.
  **Ring sweep** (local, every even P from 2L to 2L + 24 at L = 8, 10, 12, 16; 3 seeds; 20k
  steps; results/ring_sweep): 40 of 52 (L, P) cells are pusher cells; failures are scattered
  resonances (L = 12 fails only at P = 24; L = 8 only at P = 32; L = 16 at P = 52; L = 10 on
  7/12 rings), each reproducible across seeds; no arithmetic rule fits.
- 2026-10-07 (Stage F read; 210/210 runs, 0 failures): **F1 falsified** — on every padded
  ring the tiled first-replicator periods divide P in 0.00 of cases and keep dividing 2L
  (0.50–1.00); as pre-registered, the gcd-on-the-ring mechanism is dropped and the divisor law
  is restated as a property of the pair length 2L. **F2** slowing confirmed (×7.9–10.4 or
  absent on prime rings) but tiled units do appear, and composite padding slows the LDIR
  regime as much (P = 33: 2/10, 35: 2/10, 75: ×14). **F3 falsified** — odd rings slow the
  pusher ×2–16 (P = 35: KM 3,150 vs 200) and break its tiling (whole-tape units at P = 37;
  0/10 faithful at P = 79); even rings are within noise. **F4a confirmed** (L = 12 · P = 28, 36:
  10/10 at KM 350, 600, period 2), **F4b half** (L = 10 · P = 28 rescues emergence, 10/10 at
  KM 500, but heritability stays at the threshold: gen2 0.35–0.40, 25–50% of cells heritable,
  flood 0.15–0.21; P = 40: 2/10 as predicted), **F4c confirmed** (L = 8 · P = 32: 2/10).
  Mechanism from executor traces (stage_f/pusher_trace.csv): a race between the stack pointer
  (writing B from its last byte downward) and the program counter (executing B forward); a
  3-byte LD that straddles a not-yet-written byte loads partner data into the register pair
  and poisons every later PUSH; padding shifts the instruction phase at the wrap so that a
  fresh LD precedes the next PUSH. Single-encounter numbers do not predict the L = 10
  outcome (isolated gen2 0.31/0.36/0.37 for P = 20/28/40). Details: results/stageF/FINDINGS.md.
- 2026-10-07 (Stage F launched, 210 runs incl. F4): preflight OK (47 tests; 210 conditions,
  21 parameter signatures dry-run through the Modal path; volume clean; estimate 8.3 GPU-h
  ≈ $16), Modal smoke OK (2 ok + 1 invalid failed without aborting). One test bound was
  corrected first: the active-pair count per step depends on GPU load (≈3,980 on an idle Mac
  GPU, 4,400–4,600 while another GPU job runs, 3,954 on a dedicated L40S), which is the
  reason time is also recorded as active interactions per cell; the test now checks the
  physical range only. Also computed C4 (functional fraction over time, from the recorded
  random tapes): under `none` at 128 steps the population is 7–14% heritable for thousands
  of steps after the pusher appears and reaches 0.78 at 50k when the EX (SP),HL family
  takes over — results/stageC/c4. The assay pipeline now assays every tape on its run's own
  ring length.
- 2026-10-07 (Stage E read; 1,230/1,230 runs, 0 failures): **E-flat**: emergence is 10/10
  at every L ≥ 16 in all four arms (one 7/10 cell, @mubyte L = 81); the emergence time's
  mild growth with L (×8 under @nominal and @bytes, ×20 under @mubyte) vanishes under
  @steps8L (150 steps at every L) — a budget-per-byte effect, not mutation rate or lottery.
  **E-steps confirmed** (faithful 3/10 → 10/10 at L = 81 and 1/10 → 9/10 at L = 100 under
  stack-write-only). **E-tiling confirmed** in 8/8 arms (divides-2L 0.91–1.00). **E-floor not
  supported**: L = 8 is a pusher cell (10/10 at KM 500) while L = 9, 10, 12 are not (4, 4,
  6/10); L = 5 and 6 emerge slowly through whole-tape / LDIR copiers; L = 3, 4, 7 never.
  Post hoc: an isolated-assay sweep shows the pusher's own heritability dips to 0.20–0.32 at
  L = 9–12 and is 0.53–0.72 at L = 8 and L ≥ 16 — a copy-geometry explanation to be tested
  with ring variants. **E4**: faithfulness at L = 100 switches on between 128 and 256 steps;
  no whole-tape first replicators at any budget; at 64 steps 6/10 soups are heritable
  populations without any clone (top share ≤ 0.07%, 69–100% of random cells heritable) —
  the `t_rep` event is a clonality criterion and misses them. **E5 confirmed** for the LDIR
  regime (median period 5 → 32.5 bytes over two decades of mutation rate). **E6**: same-seed
  repeats diverge (0.25–0.38 dex vs 0.27–0.46 dex across seeds): seeds are exchangeable,
  single runs are not reproducible. **E7 confirmed** (18/20 vs 0/20, p = 1.7 × 10⁻⁹).
  Details: results/stageE/FINDINGS.md; generated numbers in stage_e/NUMBERS_E.md.
- 2026-10-07 (Stage C read; 490/490 runs, 0 failures): **C1a confirmed** (no-copy and
  rmw-only 0/40 at 1,000,000 steps; 95% upper bound 0.072 per run), **C1b not met** (all-ld
  2/10, not > 0.3). **C3a**: stack-write-only reproduces the stack-writes suppression (8/10
  vs 8/10, both 10/10 slower than none, p = 0.002) but the KM ratio is ×2.4, outside the
  pre-registered ×2 (paired test cannot separate the arms). **C3b confirmed** (read-only
  5/5 vs none). **C3c half met**: ex-sp-only ×3.5 (yes) but push-only ×992 — removing
  PUSH/POP alone delays more than removing all 22 stack writers. **C3d confirmed**: ld-imm
  2/10 heritable, the strongest single-family suppression at 128 steps (the first replicator
  in every permissive arm is `LD rr,nn ; PUSH rr`, which needs the immediate). **C3e
  confirmed** (cb-page, ed-loads within paired noise). **C5** met for the first replicator
  only (period 2 in 9/10 at L = 100; final dominants period 10 in 7/8). At 32 steps · 1/4 no
  arm differs from none (every p ≥ 0.34). Post hoc: mutation rate selects the family holding
  the soup at 300k (LDIR at 1/4 and ≤ 128 steps, the EX (SP),HL family at 1/16–1/64);
  succession increases heritability (gen2 0.59 → 1.00 at L = 16, 0.64 → 0.83 at 36,
  0.72 → 0.99 at 100) with byte-identical successors in 7–9/10 seeds. C4 not yet computed.
  Details: results/stageC/FINDINGS.md; generated numbers in stage_c/NUMBERS_C.md.
- 2026-10-07 (Stage D read; 280/280 runs): **D1 confirmed** (stack-write-only > none,
  CMH one-sided p = 3.9 × 10⁻⁶), **D1b replicated** (p = 1.2 × 10⁻⁵), **D2 confirmed**
  (+0.14 zero fraction at step 5,000; read-only arm 0.00), **D3** first clause confirmed
  (ratio 1.0–1.08), second clause not met at 512 steps (read-only +0.25, p = 0.095),
  **D4 confirmed** (periods 3 × 119, 9 × 22, 6 × 1; all LDIR), **D5 not supported** (4/20 vs
  6/20). Post hoc observation, logged for Stage C/E: removing CALL/RST alone recovers most
  of the effect (CMH p = 4.5 × 10⁻⁴) while reducing the flood no more than removing PUSH
  does (p = 0.033), so the suppression is attributed to the return-address writers rather
  than to the zero load as such; this is a hypothesis for the census analysis, not a result.
  Details: results/stageD/FINDINGS.md.
- 2026-10-07 (ring-length instrument): `createSimShader`/`createZ80TestShader` take a
  `memLength` (compile-time `MEM_LENGTH`, default 2L → identical behaviour and identical
  executor scores to before); the stack-pointer reset now aliases the end of B under the
  ring modulus (with the default ring this is the same value as before). Exported ring
  variants and Stage F conditions (160 runs) added; not launched.
- 2026-10-07 (second pre-launch review, before any C/D/E run; REVIEW.md §8):
  **superseded Stage D runs archived** (seeds 1–12 + 8 orphans → `runs/stageD_v1_seeds1-20`,
  removed from the volume; every analysis script now selects only summaries whose stem
  is in the stage's condition file). **Recording:** explicit early samples (1…34), full
  snapshots on a log schedule with an interaction census, per-sample write counters,
  16 random tapes, top-10 exemplars, copy offset recorded by the assay (direct test of
  period = gcd(offset, 2L)). **Conditions:** C3 gains `stack-writes` and `none` at
  (32, k=2) (490 runs); E gains the L = 100 mutation and budget sweeps and L ∈ {3,5,6,7}
  (1,230 runs); @bytes grids rounded to the nearest cell count; the L = 16 duplicates
  declared as replicates. **Outcomes:** D2 restated as a contrast (the flood is not
  stack-specific); time expressed in cumulative active interactions per cell;
  `t_faith` censoring by the Stage A/B early stop is reported (4 runs, 3 of them in the
  stack-writes L = 100 · 128 cell). **Analysis corrections:** tiled-only divisor
  statistics (an untiled tape's period trivially divides 2L; six faithful whole-tape
  copiers exist at L = 49–81 and are now a class of their own), Fisher two-sided test
  with a relative tolerance, tolerant period scanning up to L − 1, zoo units by
  per-position majority vote. **Infrastructure:** Modal timeout 1 h with one retry,
  brotli pinned to the local version, tracked/untracked dirtiness recorded separately, a
  `--smoke` launcher mode, the preflight estimator fitted to measured rates with host time.
  Preflight (tests + integrity + volume check + dry run of every distinct signature) passed for
  all three stages; estimates at L40S list price: Stage C 24.7 GPU-h ≈ $48, Stage D 17.8 GPU-h
  ≈ $35, Stage E 69.6 GPU-h ≈ $136 (the L = 100 budget sweep's 1,024/2,048-step runs and the
  mutation sweep account for ≈ $40 of E).
- 2026-10-07 (after the four-part review, before any Stage C/D/E run): **Stages
  C, D and E re-specified** (see the Design section): new seeds for every stage
  (Stage B's seeds generated the hypotheses); fine early sampling (50 steps to
  5,000) because 25–36% of A/B emergence times were exactly the first 500-step
  sample; byte histogram, zero fraction and active-pair count recorded per
  sample; the Stage A arm `stack-writes` (46 opcodes, of which 24 write nothing)
  split into `stack-write-only` (22) and `stack-read-only` (24); Stage D arms
  `push` and `call-rst-write`; Stage E added as the size-axis control set
  (per-byte-constant mutation, constant total bytes, steps ∝ L, non-square L,
  within-seed variance, the 32-step exception). Stage A/B conclusions are
  classified as **exploratory under an outcome (t_rep) adopted after 19 runs were
  read**; the pre-registered primary tq_10 fires on zero-byte floods (no-copy
  L = 36/512: 10/10 crossings without a replicator), so by its letter H3 is
  refuted and H1/H6's time ratios are unresolvable at 500-step resolution.
  Confirmatory claims will come only from C/D/E with the outcomes fixed here:
  heritable (`t_rep`), faithful (`t_faith` = gen2 ≥ 0.3 and ≥ 50% of partners became
  ≥ 75% copies), KM medians censored at the last step, Wilson intervals, exact
  Fisher/CMH for fractions and seed-paired sign tests for within-seed contrasts.
  Analysis changes made at the same time (all post hoc, all applied to A/B):
  mechanism labels under the run's suppression set; shift-invariant occupancy
  `q_shift_share`; `gen2_cond` (heritability given a copy was made);
  `self_preserved_as_B` (vulnerability); in-situ assays return NaN, never 0, when
  no partner has headroom; tolerant period; final-state tables at fixed steps
  (5k/50k/300k) rather than at the early-stop step; the zoo's "final" design taken
  from the in-situ winner. Numbers quoted in findings are generated by
  `findings.py` (NUMBERS.md) and never typed.
- 2026-10-07 (Stage C relaunched): **runner bug, Stage C aborted and
  re-run; no Stage A/B data affected.** Stage C conditions carry
  `stop_share: -1` to disable the early stop, but `run()` only treated `None`
  as "never" (the CLI converted −1, the Modal path did not), so every Stage C
  run stopped 4 samples in, at 2,500–5,000 steps (0.15 GPU-h, ≈ $0.30
  wasted). Fixed so any non-positive share means never stop, verified
  locally, and the 420 runs relaunched unchanged. The aborted batch is kept
  in `runs/stageC_aborted/` and is not used for any conclusion; its 2,500-step
  snapshots agree with Stage A's early dynamics (`ld-imm` and `push-only`
  0/10 by 2,500 steps, the other finer ablations 9–10/10 by 500–1,000).
  Stage A and B used `stop_share: 0.5` (a positive value), so their early
  stop behaved as pre-registered.
- 2026-10-07 (Stage B assays recomputed and read; the first Stage C relaunch was
  stopped after 5 results and superseded by the re-specified design, so no old-seed
  Stage C data exist). Three post hoc additions, all reported next to the pre-registered
  measures, none replacing them:
  (a) **Faithfulness.** The no-copy soups at L = 100 contain a trace-level
  (top share 0.2%) `CALL`-chain `cd xx cd xx …` whose CALLs push their own
  16-bit return address; the high byte of that address is the CALL opcode
  itself (PC runs in the unreduced 16-bit space, so after `JP`/`CALL $xxCD`
  every return address is `$CDyy`). The neighbour is flooded with `cd yy`
  pairs (50% similar), the offspring do the same (gen2 0.49), but the operand
  drifts by +4 each generation and the lineage dies by generation 3. It passes
  the gen2 ≥ 0.3 heritability rule, so that rule alone is not enough: every
  table now also reports `offspring_within_q` (share of partners that became
  ≥ 75% copies) and a replicator is called **faithful** when it is ≥ 0.5.
  Nothing in Stages A/B changes except this one case, which is labelled
  "heritable, not faithful"; no-copy stays 0/160 by `t_rep`.
  (b) **Tiling.** Dominant tapes' minimal period relative to L (does the
  period divide L; is it ≤ L/2), and the high-order entropy of the final soup
  per family, to test H6's "free tape" clause directly.
  (c) **A reverse ablation at L = 9**, found in the data and NOT
  pre-registered: removing the stack-writing families makes replication
  *more* likely (9/10 and 10/10 vs 4/10 and 1/10; Fisher p = 0.057 at 128
  steps, 0.0001 at 512). Unablated L = 9 soups are a zero flood (34% of all
  bytes are `00`, byte entropy 6.0 bits vs 7.4 without stack writes), written
  by PUSH of still-zero registers and by CALL/RST return addresses; the
  replicator that does emerge is the period-3 `DEC E ; LDIR` tiling 9 bytes
  exactly. This is one cell of the design with n = 10 per arm and is logged
  as a hypothesis for a confirmatory run (Stage D proposal: L = 9 ×
  {none, stack-writes, push-only, call-rst} × {128, 512} × 20 seeds), not as
  a result.
- 2026-10-07 (Stage B complete, first table read): **analysis bug, no data
  affected.** The exported single-pair executor used by the assay had a fixed
  40-byte private memory (sized for the 32/38-byte differential tests), so
  every post hoc assay of a pair longer than 40 bytes (L ≥ 25) indexed out of
  bounds and returned garbage (a known `LD E,49 ; LDIR` copier scored 0.00 at
  L = 49). The executor is now exported per tape length and the Stage B assays
  are recomputed; L ≤ 16 results are unchanged. Also: census takeover
  baselines are now estimated per L from random tapes (≥ 2 PUSH bytes occurs
  in 46% of random 100-byte tapes), and the minimal period of dominant tapes
  is recorded, since the first look suggests large organisms get *tiled* by
  short motifs rather than hosting larger replicators.
- 2026-10-07 (same reading): the assay's gain is now normalised by headroom,
  (after − before)/(1 − before), averaged over partners whose prior similarity
  to the tape is < 0.75. Reason: the in-situ random-cell assay returned 0/16
  heritable cells in soups where 60% of cells carry the Load–Push pair,
  because a copier cannot raise the similarity of a partner that is already a
  copy. Random-partner scores are essentially unchanged (prior similarity
  ≈ 0.06). Threshold stays gen2 ≥ 0.3.
- 2026-10-07 (Stage A 211/630 read, interrupted by the Modal outage): two
  blind spots found, both logged for the analysis and for Stage C:
  (a) `t_rep` is built from the top-3 exact exemplars, so a maximally diverse
  replicator cloud (every member unique; no genotype ≥ 0.5%) is never assayed
  — e.g. stack-writes / 32 steps: the census shows LDIR in ≥ 30% excess of
  cells by step 5k in 10/10 seeds, `t_rep` fires in 4/10. We therefore also
  report the **census takeover time** per family (first sample with ≥ 30%
  excess over the random-soup baseline) and validate the end state with a
  **random-cell in-situ assay** (16 random cells of the final snapshot, each
  assayed against its own population; fraction heritable). Stage C runs will
  store 8 random tapes per sample so the assay can be applied over time.
  (b) The pre-registered early stop (4 samples after 50% quasispecies
  occupancy) ends runs once a stack family dominates, so later LDIR invasions
  in those runs are censored; succession statistics use only runs that reached
  the horizon, and Stage C re-runs the affected cells without the early stop.
- 2026-10-07 (Stage A running; 31 runs read): added an **in-situ** variant of
  the assay for the final state — partners are drawn from the stored soup
  snapshot instead of uniform random bytes. Reason: members of the LDIR cloud
  take their pointers from the partner's bytes (`POP DE … LDIR`), so against
  random partners they score low although 96–99% of cells carry the block-copy
  pair. Random-partner scores are a lower bound (reported as before); the
  in-situ score is used for the final-state replicator label.
- 2026-10-07 (Stage A running; the first 19 of 630 runs read, all from the
  `none` / 32-step cells): the pre-registered occupancy detector `tq_10` has
  **both failure modes** in the data. False negatives: at mutation 1/4, 0/10
  runs reached 10% quasispecies occupancy, yet every final dominant tape is a
  heritable LDIR replicator (assay gen2 0.97–0.99) living in a diverse cloud
  with a free first byte, so no Hamming ball around one genotype ever fills.
  False positives: at 1/16, 2 of 8 `tq_10` crossings are non-heritable smears
  (`00 41 00 41`, `00 04 00 04`; gen2 0.08). We therefore ADD an assay-based
  emergence time **`t_rep`**: the first sample at which any of the stored top-3
  exemplars has gen2 ≥ 0.3 and exact share ≥ 0.5% (so one random cell does not
  count), computed post hoc from the JSONL with the run's own budget and
  suppression set. `tq_10` stays reported as pre-registered; conclusions about
  emergence use `t_rep`, and every table shows both. The horizon, grid and
  conditions are unchanged.
- 2026-10-07 (Stage A running, no results read): the assay is applied to the
  **top-3 exemplars** at emergence, keeping the best heritability. Reason, from
  the browser probe data: the most common exact genotype in a run can be the
  *sterile offspring* of a copier — `LD HL,$E321 ; PUSH HL` (`21 e3 21 e5`)
  writes `21 e3 21 e3 …` into its neighbour, so the inert `21 e3 ×8` tape
  outnumbers the copier that makes it (assay: offspring gen2 0.08, parent 0.27,
  parent copying confirmed byte-for-byte against a blank neighbour).
- 2026-10-07, Stage A stopped after 5 of 630 runs (none read) and relaunched
  with two additions per sample: a **byte-pattern census** (fraction of cells
  containing ED-page block-copy pairs, ≥2 PUSH bytes, EX (SP),HL, RST 38,
  LD (HL),x, CB-page (HL) ops, ≥8 zero bytes; top five 4-grams) and **soup
  snapshots** (brotli-compressed, at the first `tq_10` crossing and at the
  end) for post hoc analysis. Motivation from the three pilot runs: after the
  Load–Push takeover the population became a diverse LDIR-based cloud with no
  single dominant genotype (unique tapes 19k → 7k, HOE → 5 bits/byte), which
  per-genotype measures cannot describe.
- 2026-10-07, Stage A launched (no results read yet): added a **replication
  assay** as an interpretation outcome, computed post hoc from the dominant
  tapes stored at every sample. The tape is executed as program A against N
  random neighbours (and as B) for the condition's step budget and suppression
  set, and the score is the best shift-aligned fraction of its own bytes that
  it writes into the neighbour. Motivation: the pilot's no-copy soup is
  dominated by the RST return-address smear `ff 41 00 41 …`, which spreads a
  byte pattern without copying the code that produces it; occupancy alone
  cannot tell a smear from a replicator. The assay also runs the offspring it
  produced against fresh neighbours (**gen2 score**, heritability): on known
  tapes the Load–Push replicator scores 0.81 / gen2 0.58, the RST smear 0.61 /
  gen2 0.08, and the stack-free LDIR replicator from the pilot 0.98 / gen2 0.99
  under its own ablation. A replicator is one with gen2 ≥ 0.3 (fixed here,
  before any sweep is read). The primary outcome stays `tq_10`; the assay is
  reported next to it and used to label each emergence as self-copying or not.
- 2026-10-07, before any sweep: quasispecies radius changed from a fixed 4 to
  ⌈L/4⌉ and the motif repeat count from 4 to max(2, ⌈L/4⌉), after the
  tape-length axis was added (a fixed radius 4 covers an entire 4-byte tape).
  For L = 16 nothing changes.
- 2026-10-08 (**Stage G run and analysed**; 80/80 ok in 8 min wall clock, preflight estimate 5.8 GPU-h ≈ $11 at
  list price; `stage_g.py`, `results/stageG/FINDINGS.md`): G1 (a) and (b) **met at L = 16 and L = 50** — first
  replicator without control flow and partner-dependent 40/40; 300k dominant closed (copied 1.00, damaged 0.00)
  with control flow 40/40, byte-identical across worlds in 17/20 (`RET NZ` closer) and 14/20 (`JR NZ` pusher).
  G1 (c) met at L = 16, **not met at L = 50** (median heritable fraction 0.59 at 2k, 0.62 at 100k; threshold had
  been calibrated on L = 16). G2 by the pre-registered syntactic criterion: 6/20 at L = 20 (between) and 3/20 at
  L = 64 (alternative); by the behavioural partner test 19/20 and 8/20 — the syntactic proxy omitted LDIR/LDDR,
  which loop in hardware; both counts reported, the pre-registered one decides. 12/20 L = 64 worlds are still
  pusher-dominated at 1M steps. **BFF (THEORY P1):** harness tests (`tests/test_bff.py`) run before any soup
  falsified the mechanism claimed for P1(b) — a straight-line copier under a wrapping pointer does not replicate
  (read and write heads cross and drift) — so THEORY.md was corrected with readings (b1)/(b2) and the run design
  fixed before launch; 24 local soups (12 standard, 12 wrap; 2¹⁷ programs, 16,384 epochs, mutation 2⁻¹²) and a
  structured search over 16.4M straight-line programs (`micro/bff_search.py`) started.
- 2026-10-08 03:45 (**two-byte census; BFF on Modal**): `two_byte_census.py` ran all 65,536 two-byte words tiled to
  L = 16 through the culture test (32 partners) and the partner test (256): five load–push words copy ≥ 0.5 of partners
  (`01 c5`, `11 d5`, `21 e5`, `2a e5`, `e5 2a`), none ≥ 0.95; 246 `CALL` smears pass gen2 ≥ 0.3 with no 75% copy
  (`results/census2/`). The BFF soups were moved to Modal with the user's approval (≈ $45 cap): `modal_bff.py`
  smoke-tested, then seeds 13–24 (03:33) and 1–12 (03:41) for both variants, 48 soups of 2¹⁷ programs × 16,384 epochs;
  on the L40S containers an epoch takes ≈ 0.07 s against ≈ 0.3 s locally, so the batch costs ≈ $36 and finishes within
  the hour. Local seeds 1–6 (standard) are kept as a cross-machine reproducibility check and the other local streams are
  stopped when they finish. The first live samples (03:42) showed quasispecies replicator populations (heritable fraction
  0.69–0.97 with the top exact class at 0.15% of the soup), so the BFF emergence exemplar is the most common class at the
  first sample where it is heritable (`t_top`) and the population event is `t_her` (heritable fraction ≥ 0.5); the
  pre-registered share criterion `t_rep` is reported alongside (THEORY.md P1, amendment 03:45, before any verdict).
- 2026-10-08 03:52 (**BFF literal-push variant launched**): harness tests showed the straight-line literal pusher `P x`
  copies only 48 of 64 bytes with the standard (one-pass) pointer — the one-pass bandwidth bound — and copies itself
  completely under a wrapping pointer with the Z80 pusher's phenotype (0.61 of random partners, self-damage 0.34). THEORY
  P1(e) pre-registered (wrap + literal: open first, closed later; kill criteria; `lit` cell held as e4), then 12 `wraplit`
  soups (seeds 1–12) launched on Modal (≈ $8.5; running total for BFF on Modal ≈ $44, within the ≈ $45 approval).
- 2026-10-08 04:35 (**BFF results, 60 soups, 24.8 GPU-h ≈ $47 at list, approval was ≈ $45**; `results/bff/FINDINGS.md`): (a) standard BFF —
  transitions 9/24, first replicators loop-bearing 9/9, open 0/9, strictly closed 7/9; (b) wrap — 19/24, loop-bearing
  19/19, open 0/19 → reading (b2) born closed; (e) wrap + literal — first replicator the one-byte `P` tiling in 12/12 at
  epoch 64, open and loop-free (e1 met), earliest of all variants (e2 met), **no closure and collapse in 12/12** by
  epoch ≈ 256 as the soup becomes pointer-halting tar (e3 not met; kill criteria not triggered). Theory refined: the
  open phase is a window set by the lethality of the tar; next pre-registrations listed in THEORY.md P1(e). Analysis
  amendments recorded: heritable one-byte tilings are replicators, sterile fills are tar (classification fixed after
  seeing the all-`P` first replicator, before the verdicts were written). Local vs Modal reproducibility: identical.
- 2026-10-08 04:55 (**theorems**, at the user's request; `THEOREMS.md`): Theorem 1 (no open replicator in one-pass
  BFF; ≤ 43 copied bytes for a straight-line execution), Theorem 2 (closure requires a repeated executed address when
  bytes written per code byte < 1; covers LDIR), Proposition 3 (deterministic: the two-byte fixed points of the Z80
  against the zero partner are exactly the five load–push words, all open — `two_byte_census.py --fixed-points`),
  Proposition 4 (open-first as a count: 4.6 expected copies of any two-byte word at step 0, 6 × 10⁻⁵ of any four-byte
  word), Model 5 (the closure window q∫n dt). The counting corrected the held prediction (e4): the all-`P` one-byte
  tiling is a full one-pass replicator in `lit` (22 + 10 pushes = 64 bytes), so `lit` is predicted to behave like
  `wraplit`; withdrawn wording recorded in THEORY.md.
- 2026-10-08 10:36 (**follow-ups launched**; user approved ≈ $10 each; venue decided: Nature): `lit` (standard pointer +
  literal, seeds 1–12) and `wraplitnh` (wrap + literal + no-halt, seeds 1–12) on Modal after a smoke test of the
  no-halt path; predictions and kill criteria in THEORY.md P1(f) and the corrected (e4). Constructive search before
  launch: no closed heritable replicator among 1.3 M periodic P/bracket tilings up to period 10 under no-halt
  (`micro/bff_closed_search.py`). `manuscript/PAPER_PLAN.md` records Nature's official constraints (fetched from the
  formatting and figure guides) and the figure/Extended Data plan; conceptual panels are to be designed with the user.
- 2026-10-08 11:50 (**incident**): the two follow-up batches were launched with `modal run --detach` driving a `starmap`;
  the local client processes died (cause not identified) and, since a detached app keeps only the last triggered
  function alive, 4 `lit` and 5 `wraplitnh` soups were killed mid-run (≈ $6 of GPU time lost; 15 of 24 finished).
  Fix: `modal_bff.py` now spawns each soup (`Function.spawn`) and exits, so containers survive the client; the missing
  seeds were relaunched at 11:50 (finished seeds return at once because `run_soup` skips a present `summary.json`).
  Follow-up spend ≈ $23 against the ≈ $20 approval.

## S4. Generated number tables

### results/stageG/stageG/NUMBERS_G.md

#### Stage G — numbers (generated; do not edit)

Per-world executor tests: 256 random partners, one 128-step encounter; `copied` = fraction of partners that become a ≥ 75% copy (best cyclic shift), `damaged` = fraction of encounters in which the organism loses ≥ 25% of its bytes. Control flow = jump, relative jump, DJNZ, CALL/RET, RST by linear disassembly (pre-registered); `block` = LDIR/LDDR-type repeat instructions (reported separately).

## L = 16 (none@closure, horizon 300,000, 20 worlds)

- first replicator: control flow in 0/20; copies < 90% of random partners in 20/20; modal first tape `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` in 19/20 worlds; median copied 0.68 (range 0.62–0.75), median damaged 0.33 (range 0.27–0.43)
- final dominant (faithful in 20/20): control flow in 20/20; block-repeat (LDIR/LDDR) in 2/20; copies ≥ 95% of partners in 20/20 (with control flow 20/20; with control flow or block-repeat 20/20); modal final tape `ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0` in 17/20 worlds, copied 1.00, damaged 0.00, control flow RET NZ, block -
- paired per world: gained control flow 20, lost 0; gained closure (copied < 0.95 → ≥ 0.95) 20

heritable fraction of 16 random cells (median over worlds) by step:

|                  |   50 |   100 |   150 |   200 |   300 |   500 |   750 |   1000 |   1500 |   2000 |   3000 |   5000 |   7500 |   10000 |   15000 |   20000 |   30000 |   50000 |   75000 |   100000 |   150000 |   200000 |   300000 |
|:-----------------|-----:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|
| median_heritable |    0 |     0 |     0 |     0 |     0 |     0 |     0 |   0.03 |   0.06 |   0.12 |   0.12 |   0.12 |   0.09 |    0.12 |    0.19 |    0.19 |    0.75 |    0.88 |    0.81 |     0.88 |     0.88 |     0.94 |     0.91 |

## L = 20 (none@closure1M, horizon 1,000,000, 20 worlds)

- first replicator: control flow in 0/20; copies < 90% of random partners in 20/20; modal first tape `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` in 12/20 worlds; median copied 0.66 (range 0.56–0.71), median damaged 0.43 (range 0.39–0.48)
- final dominant (faithful in 20/20): control flow in 6/20; block-repeat (LDIR/LDDR) in 15/20; copies ≥ 95% of partners in 19/20 (with control flow 6/20; with control flow or block-repeat 19/20); modal final tape `1e 04 ed b0 1e 04 ed b0 1e 04 ed b0 1e 04 ed b0…` in 4/20 worlds, copied 1.00, damaged 0.00, control flow -, block LDIR
- paired per world: gained control flow 6, lost 0; gained closure (copied < 0.95 → ≥ 0.95) 19

heritable fraction of 16 random cells (median over worlds) by step:

|                  |   50 |   100 |   150 |   200 |   300 |   500 |   750 |   1000 |   1500 |   2000 |   3000 |   5000 |   10000 |   15000 |   20000 |   30000 |   50000 |   75000 |   100000 |   150000 |   200000 |   300000 |   500000 |   750000 |   1000000 |
|:-----------------|-----:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|---------:|---------:|----------:|
| median_heritable |    0 |     0 |     0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |   0.06 |   0.12 |    0.12 |    0.16 |    0.12 |    0.19 |    0.19 |    0.19 |     0.16 |     0.16 |     0.28 |     0.72 |     0.88 |     0.94 |      0.94 |

## L = 50 (none@closure, horizon 300,000, 20 worlds)

- first replicator: control flow in 0/20; copies < 90% of random partners in 20/20; modal first tape `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` in 14/20 worlds; median copied 0.68 (range 0.59–0.78), median damaged 0.11 (range 0.08–0.17)
- final dominant (faithful in 20/20): control flow in 20/20; block-repeat (LDIR/LDDR) in 0/20; copies ≥ 95% of partners in 20/20 (with control flow 20/20; with control flow or block-repeat 20/20); modal final tape `01 c5 01 c5 01 c5 01 c5 01 20 f0 c5 01 c5 01 c5…` in 14/20 worlds, copied 1.00, damaged 0.00, control flow JR NZ,d, block -
- paired per world: gained control flow 20, lost 0; gained closure (copied < 0.95 → ≥ 0.95) 20

heritable fraction of 16 random cells (median over worlds) by step:

|                  |   50 |   100 |   150 |   200 |   300 |   500 |   750 |   1000 |   1500 |   2000 |   3000 |   5000 |   7500 |   10000 |   15000 |   20000 |   30000 |   50000 |   75000 |   100000 |   150000 |   200000 |   300000 |
|:-----------------|-----:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|
| median_heritable |    0 |     0 |     0 |     0 |     0 |     0 |  0.06 |   0.12 |   0.38 |   0.59 |    0.5 |   0.41 |   0.44 |     0.5 |     0.5 |     0.5 |     0.5 |    0.56 |    0.56 |     0.62 |     0.56 |     0.56 |     0.59 |

## L = 64 (none@closure1M, horizon 1,000,000, 20 worlds)

- first replicator: control flow in 0/20; copies < 90% of random partners in 20/20; modal first tape `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` in 18/20 worlds; median copied 0.73 (range 0.68–0.84), median damaged 0.06 (range 0.04–0.10)
- final dominant (faithful in 20/20): control flow in 3/20; block-repeat (LDIR/LDDR) in 7/20; copies ≥ 95% of partners in 8/20 (with control flow 3/20; with control flow or block-repeat 8/20); modal final tape `11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5…` in 5/20 worlds, copied 0.79, damaged 0.09, control flow -, block -
- paired per world: gained control flow 3, lost 0; gained closure (copied < 0.95 → ≥ 0.95) 8

heritable fraction of 16 random cells (median over worlds) by step:

|                  |   50 |   100 |   150 |   200 |   300 |   500 |   750 |   1000 |   1500 |   2000 |   3000 |   5000 |   10000 |   15000 |   20000 |   30000 |   50000 |   75000 |   100000 |   150000 |   200000 |   300000 |   500000 |   750000 |   1000000 |
|:-----------------|-----:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|---------:|---------:|----------:|
| median_heritable |    0 |     0 |     0 |     0 |     0 |     0 |  0.03 |   0.06 |   0.25 |   0.38 |   0.44 |   0.38 |    0.25 |    0.22 |    0.31 |    0.25 |    0.28 |    0.19 |     0.25 |     0.25 |     0.25 |     0.25 |     0.31 |     0.25 |      0.34 |

## Pre-registered verdicts

- **G1, L = 16**: (a) first without control flow 20/20 (≥ 18 needed) and partner-dependent 20/20 (≥ 18): met; (b) control-flow faithful dominant 20/20 (≥ 16), modal final copied 1.00 (≥ 0.95), damaged 0.00 (≤ 0.05): met; (c) median heritable fraction ≤ 5k after emergence all < 0.20, at 100k 0.88 (> 0.70): met; kill criterion not triggered. Behavioural closure (copied ≥ 0.95) in 20/20 finals; with a loop of either kind 20/20.
- **G2, L = 20**: faithful control-flow dominant at 1M copying ≥ 95% of partners in 6/20 (prediction ≥ 10, alternative < 5): between; with a loop of either kind (control flow or LDIR/LDDR) 19/20; closed by partner test regardless of syntax 19/20; finals still the open pusher (no control flow, no block-repeat, copied < 0.95): 1/20.
- **G1, L = 50**: (a) first without control flow 20/20 (≥ 18 needed) and partner-dependent 20/20 (≥ 18): met; (b) control-flow faithful dominant 20/20 (≥ 16), modal final copied 1.00 (≥ 0.95), damaged 0.00 (≤ 0.05): met; (c) median heritable fraction ≤ 5k after emergence max 0.62, at 100k 0.62 (> 0.70): NOT met; kill criterion not triggered. Behavioural closure (copied ≥ 0.95) in 20/20 finals; with a loop of either kind 20/20.
- **G2, L = 64**: faithful control-flow dominant at 1M copying ≥ 95% of partners in 3/20 (prediction ≥ 10, alternative < 5): alternative; with a loop of either kind (control flow or LDIR/LDDR) 8/20; closed by partner test regardless of syntax 8/20; finals still the open pusher (no control flow, no block-repeat, copied < 0.95): 12/20.

## Final-tape classes per L

L = 16:

| final_cf                       | final_block   |   n |   copied |   damaged |
|:-------------------------------|:--------------|----:|---------:|----------:|
| JP (HL)+JP NC,nn+JP nn+JR NZ,d | LDIR          |   1 |     1.00 |      0.00 |
| JP nn                          | -             |   1 |     1.00 |      0.00 |
| JR Z,d                         | LDIR          |   1 |     1.00 |      0.00 |
| RET NZ                         | -             |  17 |     1.00 |      0.00 |

L = 20:

| final_cf        | final_block   |   n |   copied |   damaged |
|:----------------|:--------------|----:|---------:|----------:|
| -               | -             |   1 |     0.69 |      0.36 |
| -               | LDIR          |  13 |     1.00 |      0.00 |
| DJNZ d          | -             |   3 |     1.00 |      0.00 |
| DJNZ d+JP NZ,nn | LDDR          |   1 |     1.00 |      0.00 |
| JR NZ,d         | -             |   1 |     1.00 |      0.00 |
| RET             | LDDR          |   1 |     1.00 |      0.00 |

L = 50:

| final_cf   | final_block   |   n |   copied |   damaged |
|:-----------|:--------------|----:|---------:|----------:|
| JR NZ,d    | -             |  20 |     1.00 |      0.00 |

L = 64:

| final_cf                 | final_block   |   n |   copied |   damaged |
|:-------------------------|:--------------|----:|---------:|----------:|
| -                        | -             |  12 |     0.78 |      0.06 |
| -                        | LDIR          |   5 |     1.00 |      0.00 |
| CALL PE,nn+RST 08+RST 28 | LDDR          |   1 |     1.00 |      0.00 |
| DJNZ d+RST 08            | LDDR          |   1 |     1.00 |      0.00 |
| JP (HL)                  | -             |   1 |     1.00 |      0.00 |

### results/stageG/NUMBERS.md

#### Numbers for the findings (generated; do not edit)

## 1. Emergence by tape length (mutation 1/16)

### none@closure

|   tape |   ('stopped', 128) | ('tfaith', 128)   | ('tq10', 128)   | ('trep', 128)   |
|-------:|-------------------:|:------------------|:----------------|:----------------|
|     16 |                  0 | 20/20             | 20/20           | 20/20 (450)     |
|     50 |                  0 | 20/20             | 20/20           | 20/20 (600)     |

### none@closure1M

|   tape |   ('stopped', 128) | ('tfaith', 128)   | ('tq10', 128)   | ('trep', 128)   |
|-------:|-------------------:|:------------------|:----------------|:----------------|
|     20 |                  0 | 20/20             | 19/20           | 20/20 (700)     |
|     64 |                  0 | 20/20             | 20/20           | 20/20 (750)     |

## 2. L = 9: none vs stack-writes (post hoc; one cell of the design)


## 3. Zero-byte fraction of the soup (from snapshots) and final byte entropy H0

|                             |   n |   zero_final_mean |   zero_final_min |   zero_final_max |   zero_emergence_mean |   H0_final_mean |   steps_run_min |
|:----------------------------|----:|------------------:|-----------------:|-----------------:|----------------------:|----------------:|----------------:|
| ('none@closure', 16, 128)   |  20 |             0.006 |            0.003 |            0.019 |                 0.268 |           3.254 |      300000     |
| ('none@closure', 50, 128)   |  20 |             0.049 |            0.045 |            0.055 |                 0.337 |           2.206 |      300000     |
| ('none@closure1M', 20, 128) |  20 |             0.034 |            0.001 |            0.223 |                 0.166 |           3.923 |           1e+06 |
| ('none@closure1M', 64, 128) |  20 |             0.052 |            0.007 |            0.091 |                 0.282 |           3.815 |           1e+06 |

## 4. Tiling: period of the first heritable replicator tape

|                             |   n |   period_median |   tiled |   whole_tape | periods   |   n_tiled |   divides_L |   divides_2L |   gcd_law |
|:----------------------------|----:|----------------:|--------:|-------------:|:----------|----------:|------------:|-------------:|----------:|
| ('none@closure', 16, 128)   |  20 |               2 |       1 |            0 | 2×20      |        20 |           1 |            1 |         0 |
| ('none@closure', 50, 128)   |  20 |               2 |       1 |            0 | 2×20      |        20 |           1 |            1 |         0 |
| ('none@closure1M', 20, 128) |  20 |               2 |       1 |            0 | 2×20      |        20 |           1 |            1 |         0 |
| ('none@closure1M', 64, 128) |  20 |               2 |       1 |            0 | 2×20      |        20 |           1 |            1 |         0 |

`tiled` = period ≤ L/2; `whole_tape` = period L with copy offset 0 (exact whole-tape copier); `divides_*` and `gcd_law` (period == gcd(offset, 2L)) are computed over tiled tapes only.

### Final-soup high-order entropy (bits/byte), median over seeds

|                        |   128 |
|:-----------------------|------:|
| ('none@closure', 16)   |  2.09 |
| ('none@closure', 50)   |  1.26 |
| ('none@closure1M', 20) |  2.72 |
| ('none@closure1M', 64) |  1.97 |

## 5. Faithfulness and functional fraction of the final population

|                             |   n |   final_replicator |   final_faithful |   func_rnd_median |   func_rnd_faithful_median |   func_insitu_median |   insitu_informative_runs |   trep_unfaithful |
|:----------------------------|----:|-------------------:|-----------------:|------------------:|---------------------------:|---------------------:|--------------------------:|------------------:|
| ('none@closure', 16, 128)   |  20 |                 20 |               20 |              0.86 |                       0.86 |                 0.35 |                        20 |                 0 |
| ('none@closure', 50, 128)   |  20 |                 20 |               20 |              0.56 |                       0.41 |                 0.36 |                        20 |                 0 |
| ('none@closure1M', 20, 128) |  20 |                 20 |               20 |              0.95 |                       0.95 |                 0.9  |                        20 |                 0 |
| ('none@closure1M', 64, 128) |  20 |                 20 |               20 |              0.34 |                       0.22 |                 0.27 |                        20 |                 0 |

## 6. Census family at 300k steps and at the last step

|                             |   n |   stopped_early | family_300k             | family_last             | stack_takeover        | ldir_takeover          |
|:----------------------------|----:|----------------:|:------------------------|:------------------------|:----------------------|:-----------------------|
| ('none@closure', 16, 128)   |  20 |               0 | ex_sp:17, ldir:3        | ex_sp:17, ldir:3        | 17 runs, median 8500  | 3 runs, median 43500   |
| ('none@closure', 50, 128)   |  20 |               0 | push:20                 | push:20                 | 20 runs, median 1350  | 0                      |
| ('none@closure1M', 20, 128) |  20 |               0 | ldir:10, push:5, none:5 | ldir:16, push:3, none:1 | 17 runs, median 18000 | 16 runs, median 247500 |
| ('none@closure1M', 64, 128) |  20 |               0 | push:17, ldir:3         | push:13, ldir:7         | 20 runs, median 1750  | 7 runs, median 419000  |

### results/stageC/stage_c/NUMBERS_C.md

#### Stage C numbers (generated)

Interaction clock: mean active interactions per cell per step over all Stage C runs = 0.3949 (min 0.3941, max 0.3959); 1,000 steps ≈ 395 encounters per cell.

## C1 — 1,000,000-step runs

| label    |   steps |   k |   n |   steps_run_min |   tq_10_n |   t_rep_n |   t_rep_km |   t_faith_n | periods   | mechs                            |   func_rnd_median |   zero_final_median |
|:---------|--------:|----:|----:|----------------:|----------:|----------:|-----------:|------------:|:----------|:---------------------------------|------------------:|--------------------:|
| all-ld   |      32 |   2 |  10 |         1000000 |         2 |         2 |        inf |           2 | 4×2       | block-copy:1, block-copy+stack:1 |             0.672 |             0.0527  |
| no-copy  |      32 |   2 |  10 |         1000000 |         0 |         0 |        inf |           0 | –         | –                                |             0     |             0.194   |
| rmw-only |      32 |   2 |  10 |         1000000 |         0 |         0 |        inf |           0 | –         | –                                |             0     |             0.00403 |
| all-ld   |     128 |   4 |  10 |         1000000 |         1 |         2 |        inf |           2 | 8×2       | block-copy+stack:1, -:1          |             0     |             0.327   |
| no-copy  |     128 |   4 |  10 |         1000000 |         0 |         0 |        inf |           0 | –         | –                                |             0     |             0.246   |
| rmw-only |     128 |   4 |  10 |         1000000 |         0 |         0 |        inf |           0 | –         | –                                |             0     |             0.0042  |

## C3 — ablations at 128 steps, mutation 1/16, L = 16, seeds 101–110

| label            |   n |   steps_run_min |   t_rep_n |   t_rep_km |   t_rep_km_ix |   km_ratio_vs_none |   slower |   faster |   ties |    sign_p |   t_faith_n |   tq_10_n |   tiled_frac |   div2L_tiled | periods   |   zero_final_median |
|:-----------------|----:|----------------:|----------:|-----------:|--------------:|-------------------:|---------:|---------:|-------:|----------:|------------:|----------:|-------------:|--------------:|:----------|--------------------:|
| none             |  10 |          300000 |        10 | 200        |     79        |               1    |      nan |      nan |    nan | nan       |          10 |        10 |            1 |             1 | 2×10      |             0.00421 |
| stack-writes     |  10 |          300000 |         8 |   3.2e+04  |      1.26e+04 |             160    |       10 |        0 |      0 |   0.00195 |           8 |         1 |            1 |             1 | 4×4, 8×4  |             0.00615 |
| stack-write-only |  10 |          300000 |         8 |   7.65e+04 |      3.02e+04 |             382    |       10 |        0 |      0 |   0.00195 |           8 |         1 |            1 |             1 | 4×3, 8×5  |             0.00613 |
| stack-read-only  |  10 |          300000 |        10 | 600        |    237        |               3    |        5 |        5 |      0 |   1       |          10 |        10 |            1 |             1 | 2×10      |             0.0149  |
| push-only        |  10 |          300000 |         6 |   1.98e+05 |      7.84e+04 |             992    |       10 |        0 |      0 |   0.00195 |           6 |         7 |            1 |             1 | 4×3, 8×3  |             0.0142  |
| ex-sp-only       |  10 |          300000 |        10 | 700        |    277        |               3.5  |        7 |        2 |      1 |   0.18    |          10 |        10 |            1 |             1 | 2×10      |             0.0195  |
| call-rst         |  10 |          300000 |        10 | 250        |     98.6      |               1.25 |        6 |        3 |      1 |   0.508   |          10 |        10 |            1 |             1 | 2×10      |             0.0103  |
| ld-imm           |  10 |          300000 |         2 | inf        |    inf        |             inf    |       10 |        0 |      0 |   0.00195 |           2 |        10 |            1 |             1 | 4×2       |             0.418   |
| ld-reg           |  10 |          300000 |        10 | 150        |     59.2      |               0.75 |        5 |        5 |      0 |   1       |          10 |        10 |            1 |             1 | 2×10      |             0.00383 |
| ld-mem           |  10 |          300000 |        10 | 200        |     79        |               1    |        4 |        4 |      2 |   1       |          10 |        10 |            1 |             1 | 2×10      |             0.00388 |
| cb-page          |  10 |          300000 |        10 |   1.3e+03  |    513        |               6.5  |        6 |        4 |      0 |   0.754   |          10 |        10 |            1 |             1 | 2×10      |             0.00402 |
| ed-loads         |  10 |          300000 |        10 | 450        |    178        |               2.25 |        6 |        3 |      1 |   0.508   |          10 |        10 |            1 |             1 | 2×10      |             0.00409 |
| block-copy       |  10 |          300000 |        10 | 350        |    138        |               1.75 |        6 |        4 |      0 |   0.754   |          10 |        10 |            1 |             1 | 2×10      |             0.0038  |
| all-ld           |  10 |         1000000 |         2 | inf        |    inf        |             inf    |       10 |        0 |      0 |   0.00195 |           2 |         1 |            1 |             1 | 8×2       |             0.327   |
| rmw-only         |  10 |         1000000 |         0 | inf        |    inf        |             inf    |       10 |        0 |      0 |   0.00195 |           0 |         0 |          nan |           nan | –         |             0.0042  |
| no-copy          |  10 |         1000000 |         0 | inf        |    inf        |             inf    |       10 |        0 |      0 |   0.00195 |           0 |         0 |          nan |           nan | –         |             0.246   |

`slower/faster/ties`: seed-paired comparison of t_rep against `none` (a censored run counts as slower than any emerged run; two censored runs tie); `sign_p` two-sided exact sign test ignoring ties; `t_rep_km_ix` = KM median in cumulative active interactions per cell.

## C3 — ablations at 32 steps, mutation 1/4, L = 16, seeds 101–110

| label            |   n |   steps_run_min |   t_rep_n |   t_rep_km |   t_rep_km_ix |   km_ratio_vs_none |   slower |   faster |   ties |    sign_p |   t_faith_n |   tq_10_n |   tiled_frac |   div2L_tiled | periods   |   zero_final_median |
|:-----------------|----:|----------------:|----------:|-----------:|--------------:|-------------------:|---------:|---------:|-------:|----------:|------------:|----------:|-------------:|--------------:|:----------|--------------------:|
| none             |  10 |          300000 |         9 |   6.85e+04 |      2.7e+04  |              1     |      nan |      nan |    nan | nan       |           9 |         2 |            1 |             1 | 4×9       |             0.0508  |
| stack-writes     |  10 |          300000 |         5 |   2.28e+05 |      9e+04    |              3.33  |        7 |        3 |      0 |   0.344   |           5 |         0 |            1 |             1 | 4×5       |             0.0106  |
| stack-write-only |  10 |          300000 |        10 |   2.15e+04 |      8.49e+03 |              0.314 |        4 |        6 |      0 |   0.754   |          10 |         1 |            1 |             1 | 4×10      |             0.00929 |
| stack-read-only  |  10 |          300000 |        10 |   9.8e+04  |      3.87e+04 |              1.43  |        6 |        4 |      0 |   0.754   |          10 |         1 |            1 |             1 | 4×10      |             0.0501  |
| push-only        |  10 |          300000 |         9 |   9.3e+04  |      3.67e+04 |              1.36  |        7 |        3 |      0 |   0.344   |           9 |         0 |            1 |             1 | 4×9       |             0.0394  |
| ex-sp-only       |  10 |          300000 |        10 |   7.4e+04  |      2.92e+04 |              1.08  |        4 |        6 |      0 |   0.754   |          10 |         0 |            1 |             1 | 4×10      |             0.0516  |
| call-rst         |  10 |          300000 |         6 |   4.25e+04 |      1.68e+04 |              0.62  |        5 |        4 |      1 |   1       |           6 |         5 |            1 |             1 | 2×4, 4×2  |             0.0245  |
| ld-imm           |  10 |          300000 |         9 |   6.75e+04 |      2.66e+04 |              0.985 |        4 |        5 |      1 |   1       |           9 |         0 |            1 |             1 | 4×9       |             0.0519  |
| ld-reg           |  10 |          300000 |        10 |   1.04e+05 |      4.13e+04 |              1.53  |        7 |        3 |      0 |   0.344   |          10 |         0 |            1 |             1 | 4×10      |             0.0502  |
| cb-page          |  10 |          300000 |         9 |   9.9e+04  |      3.91e+04 |              1.45  |        5 |        5 |      0 |   1       |           9 |         0 |            1 |             1 | 4×9       |             0.0515  |
| ed-loads         |  10 |          300000 |         9 |   6.05e+04 |      2.39e+04 |              0.883 |        5 |        5 |      0 |   1       |           9 |         0 |            1 |             1 | 4×9       |             0.0509  |
| all-ld           |  10 |         1000000 |         2 | inf        |    inf        |            inf     |        9 |        0 |      1 |   0.00391 |           2 |         2 |            1 |             1 | 4×2       |             0.0527  |
| rmw-only         |  10 |         1000000 |         0 | inf        |    inf        |            inf     |        9 |        0 |      1 |   0.00391 |           0 |         0 |          nan |           nan | –         |             0.00403 |
| no-copy          |  10 |         1000000 |         0 | inf        |    inf        |            inf     |        9 |        0 |      1 |   0.00391 |           0 |         0 |          nan |           nan | –         |             0.194   |

`slower/faster/ties`: seed-paired comparison of t_rep against `none` (a censored run counts as slower than any emerged run; two censored runs tie); `sign_p` two-sided exact sign test ignoring ties; `t_rep_km_ix` = KM median in cumulative active interactions per cell.

## C2 — succession without censoring (census family of the dominant unit at fixed steps; medians over seeds)

| label      |   steps |   k |   n | family_5k               | family_50k                      | family_300k             |   ldir_300k |   zero8_300k |   final_zero_frac | ldir_takeover         | stack_takeover        |
|:-----------|--------:|----:|----:|:------------------------|:--------------------------------|:------------------------|------------:|-------------:|------------------:|:----------------------|:----------------------|
| block-copy |     128 |   2 |  10 | none:10                 | none:9, ex_sp:1                 | none:6, ex_sp:4         |     7.5e-05 |     0.201    |          0.229    | 0                     | 10 runs, median 14500 |
| block-copy |     128 |   4 |  10 | none:8, push:1, ex_sp:1 | ex_sp:7, none:3                 | ex_sp:10                |     0       |     0.0014   |          0.0038   | 0                     | 9 runs, median 4750   |
| block-copy |     128 |   6 |  10 | none:10                 | push:7, none:2, ex_sp:1         | ex_sp:6, push:4         |     0       |     0.00045  |          0.0013   | 0                     | 9 runs, median 40500  |
| block-copy |     512 |   2 |  10 | push:10                 | push:10                         | push:10                 |     5e-05   |     0.126    |          0.13     | 0                     | 10 runs, median 1750  |
| block-copy |     512 |   4 |  10 | push:10                 | push:7, ex_sp:3                 | ex_sp:10                |     0       |     0.00753  |          0.0144   | 0                     | 10 runs, median 1150  |
| block-copy |     512 |   6 |  10 | none:9, push:1          | push:8, ex_sp:2                 | push:7, ex_sp:3         |     0       |     0.0651   |          0.0713   | 0                     | 10 runs, median 1825  |
| ld-mem     |     128 |   2 |  10 | none:9, push:1          | push:6, ldir:2, none:2          | ldir:7, ex_sp:3         |     0.794   |     0.0151   |          0.0374   | 7 runs, median 68000  | 9 runs, median 9000   |
| ld-mem     |     128 |   4 |  10 | none:8, push:1, ex_sp:1 | ex_sp:9, none:1                 | ex_sp:10                |     0       |     0.00125  |          0.00388  | 0                     | 10 runs, median 5225  |
| ld-mem     |     128 |   6 |  10 | none:10                 | ex_sp:6, push:4                 | ex_sp:9, ldir:1         |     0       |     0.0003   |          0.000759 | 1 runs, median 53500  | 10 runs, median 21250 |
| ld-mem     |     512 |   2 |  10 | push:10                 | push:10                         | push:10                 |     2.5e-05 |     0.109    |          0.11     | 0                     | 10 runs, median 1175  |
| ld-mem     |     512 |   4 |  10 | push:7, none:2, ex_sp:1 | ex_sp:9, push:1                 | ex_sp:10                |     0       |     0.00585  |          0.0118   | 0                     | 10 runs, median 1275  |
| ld-mem     |     512 |   6 |  10 | none:9, ldir:1          | ex_sp:5, push:2, ldir:2, none:1 | ex_sp:8, ldir:2         |     0       |     0.00128  |          0.00386  | 2 runs, median 4950   | 9 runs, median 1000   |
| none       |      32 |   2 |  10 | none:10                 | none:7, ldir:3                  | ldir:9, none:1          |     0.84    |     0.0081   |          0.0508   | 9 runs, median 64000  | 0                     |
| none       |     128 |   2 |  10 | none:9, ldir:1          | none:6, ldir:3, ex_sp:1         | ldir:8, ex_sp:2         |     0.803   |     0.0243   |          0.0402   | 8 runs, median 63750  | 9 runs, median 15000  |
| none       |     128 |   4 |  10 | none:10                 | ex_sp:8, ldir:1, none:1         | ex_sp:9, ldir:1         |     0       |     0.00185  |          0.00421  | 1 runs, median 38500  | 9 runs, median 3800   |
| none       |     128 |   6 |  10 | none:10                 | ex_sp:5, push:3, none:1, ldir:1 | ex_sp:8, ldir:2         |     0       |     0.000325 |          0.000914 | 2 runs, median 130750 | 7 runs, median 14500  |
| none       |     512 |   2 |  10 | push:10                 | push:10                         | push:10                 |     0       |     0.13     |          0.134    | 0                     | 10 runs, median 1525  |
| none       |     512 |   4 |  10 | push:9, none:1          | ex_sp:7, push:3                 | ex_sp:9, ldir:1         |     0       |     0.0074   |          0.0142   | 1 runs, median 116500 | 10 runs, median 1625  |
| none       |     512 |   6 |  10 | push:5, none:5          | push:8, ex_sp:2                 | push:6, ex_sp:3, ldir:1 |     0       |     0.0577   |          0.0655   | 1 runs, median 260000 | 10 runs, median 4175  |

### C2 emergence by mutation rate

| label      |   steps |   k |   n |   t_rep_n |   t_rep_km |   t_rep_km_ix |   t_faith_n |   final_faithful_n |   func_rnd_median | periods   |   zero_final_median |
|:-----------|--------:|----:|----:|----------:|-----------:|--------------:|------------:|-------------------:|------------------:|:----------|--------------------:|
| block-copy |     128 |   2 |  10 |        10 | 850        |    336        |          10 |                 10 |             0.219 | 2×10      |            0.229    |
| block-copy |     128 |   4 |  10 |        10 | 350        |    138        |          10 |                 10 |             0.844 | 2×10      |            0.0038   |
| block-copy |     128 |   6 |  10 |        10 |   4.1e+03  |      1.62e+03 |          10 |                 10 |             0.969 | 2×10      |            0.0013   |
| block-copy |     512 |   2 |  10 |        10 | 350        |    138        |          10 |                 10 |             0.281 | 2×10      |            0.13     |
| block-copy |     512 |   4 |  10 |        10 | 250        |     98.8      |          10 |                 10 |             0.766 | 2×10      |            0.0144   |
| block-copy |     512 |   6 |  10 |        10 |   1.05e+03 |    414        |          10 |                 10 |             0.406 | 2×10      |            0.0713   |
| ld-mem     |     128 |   2 |  10 |        10 |   1.15e+03 |    454        |          10 |                 10 |             0.656 | 2×10      |            0.0374   |
| ld-mem     |     128 |   4 |  10 |        10 | 200        |     79        |          10 |                 10 |             0.875 | 2×10      |            0.00388  |
| ld-mem     |     128 |   6 |  10 |        10 | 850        |    336        |          10 |                 10 |             0.984 | 2×10      |            0.000759 |
| ld-mem     |     512 |   2 |  10 |        10 | 200        |     79.1      |          10 |                 10 |             0.391 | 2×10      |            0.11     |
| ld-mem     |     512 |   4 |  10 |        10 | 200        |     79        |          10 |                 10 |             0.672 | 2×10      |            0.0118   |
| ld-mem     |     512 |   6 |  10 |        10 | 150        |     59.2      |          10 |                 10 |             0.922 | 2×9, 16×1 |            0.00386  |
| none       |      32 |   2 |  10 |         9 |   6.85e+04 |      2.7e+04  |           9 |                  9 |             0.781 | 4×9       |            0.0508   |
| none       |     128 |   2 |  10 |        10 | 650        |    257        |          10 |                 10 |             0.703 | 2×10      |            0.0402   |
| none       |     128 |   4 |  10 |        10 | 200        |     79        |          10 |                 10 |             0.844 | 2×10      |            0.00421  |
| none       |     128 |   6 |  10 |        10 | 200        |     79        |          10 |                 10 |             0.969 | 2×10      |            0.000914 |
| none       |     512 |   2 |  10 |        10 | 350        |    138        |          10 |                 10 |             0.344 | 2×10      |            0.134    |
| none       |     512 |   4 |  10 |        10 | 350        |    138        |          10 |                 10 |             0.766 | 2×10      |            0.0142   |
| none       |     512 |   6 |  10 |        10 | 450        |    178        |          10 |                 10 |             0.422 | 2×10      |            0.0655   |

## C5 — L ∈ {36, 100} without early stop (periods of the first heritable replicator and of the faithful final dominant)

| label        |   L |   steps |   k |   n |   t_rep_n |   t_rep_km |   t_faith_n |   final_faithful_n | periods                    | trep_period_le8   | final_period_le8   | final_periods   |
|:-------------|----:|--------:|----:|----:|----------:|-----------:|------------:|-------------------:|:---------------------------|:------------------|:-------------------|:----------------|
| none         |  36 |     128 |   4 |  10 |        10 |  350       |          10 |                 10 | 2×10                       | 10/10             | 3/10               | 2×3, 14×7       |
| none         | 100 |     128 |   4 |  10 |        10 |    1.2e+03 |           9 |                  8 | 2×9, 20×1                  | 9/10              | 0/8                | 9×1, 10×7       |
| stack-writes |  36 |     128 |   4 |  10 |        10 |    1.2e+04 |          10 |                 10 | 4×1, 6×2, 8×3, 9×2, 12×2   | 6/10              | 10/10              | 4×6, 6×2, 8×2   |
| stack-writes | 100 |     128 |   4 |  10 |        10 |    8.5e+03 |           1 |                  1 | 5×1, 8×1, 20×4, 25×3, 76×1 | 2/10              | 0/1                | 9×1             |

C1 pooled: no-copy + rmw-only 0/40 runs emerged within 1,000,000 steps → 95% upper bound on the per-run emergence probability within that horizon = 0.072 (exact binomial, one-sided).

## First replicator vs final dominant under `none` (128 steps, 1/16): the modal tapes assayed in isolation

|   L | which                     | identical_in_seeds   |   period | tape                                                                                                        |   score |   gen2 | faithful   |   partner_bytes_changed |   self_bytes_changed |
|----:|:--------------------------|:---------------------|---------:|:------------------------------------------------------------------------------------------------------------|--------:|-------:|:-----------|------------------------:|---------------------:|
|  16 | first (t_rep)             | 8/10                 |        2 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5                                                             |    0.82 |   0.59 | True       |                   15.62 |                 4.50 |
|  16 | final (faithful dominant) | 9/10                 |        8 | ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0                                                             |    1.00 |   1.00 | True       |                   15.94 |                 0.00 |
|  36 | first (t_rep)             | 7/10                 |        2 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 |    0.83 |   0.64 | True       |                   34.12 |                 8.08 |
|  36 | final (faithful dominant) | 7/10                 |       14 | 11 d5 11 d5 11 d5 11 d5 11 10 f0 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 10 f0 d5 11 d5 11 d5 11 d5 11 d5 11 d5 |    0.94 |   0.83 | True       |                   35.81 |                 0.00 |
| 100 | first (t_rep)             | 9/10                 |        2 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 …                                               |    0.86 |   0.72 | True       |                   91.95 |                 1.83 |
| 100 | final (faithful dominant) | 7/10                 |       10 | c2 c5 01 c5 01 c5 01 c5 c2 c5 c2 c5 01 c5 01 c5 01 c5 c2 c5 …                                               |    0.99 |   0.99 | True       |                   98.67 |                 2.00 |

`identical_in_seeds`: seeds whose tape is byte-identical to the modal one; `partner_bytes_changed` / `self_bytes_changed`: mean bytes of the random partner / of the tape itself that differ after one 128-step encounter as program A (64 partners).

## Pre-registered C predictions — the numbers

- C1 no-copy @(128, k4): heritable 0/10 at 1,000,000 steps (KM median NR); faithful 0/10
- C1 no-copy @(32, k2): heritable 0/10 at 1,000,000 steps (KM median NR); faithful 0/10
- C1 rmw-only @(128, k4): heritable 0/10 at 1,000,000 steps (KM median NR); faithful 0/10
- C1 rmw-only @(32, k2): heritable 0/10 at 1,000,000 steps (KM median NR); faithful 0/10
- C1 all-ld @(128, k4): heritable 2/10 at 1,000,000 steps (KM median NR); faithful 2/10
- C1 all-ld @(32, k2): heritable 2/10 at 1,000,000 steps (KM median NR); faithful 2/10
- C3 stack-writes @(128, k4): 8/10 heritable, KM 32,000 vs none 200 (ratio 160.00); paired vs none: 10 slower, 0 faster, 0 ties, sign p = 0.00195
- C3 stack-write-only @(128, k4): 8/10 heritable, KM 76,500 vs none 200 (ratio 382.50); paired vs none: 10 slower, 0 faster, 0 ties, sign p = 0.00195
- C3 stack-read-only @(128, k4): 10/10 heritable, KM 600 vs none 200 (ratio 3.00); paired vs none: 5 slower, 5 faster, 0 ties, sign p = 1
- C3 push-only @(128, k4): 6/10 heritable, KM 198,500 vs none 200 (ratio 992.50); paired vs none: 10 slower, 0 faster, 0 ties, sign p = 0.00195
- C3 ex-sp-only @(128, k4): 10/10 heritable, KM 700 vs none 200 (ratio 3.50); paired vs none: 7 slower, 2 faster, 1 ties, sign p = 0.18
- C3 call-rst @(128, k4): 10/10 heritable, KM 250 vs none 200 (ratio 1.25); paired vs none: 6 slower, 3 faster, 1 ties, sign p = 0.508
- C3 ld-imm @(128, k4): 2/10 heritable, KM NR vs none 200 (ratio nan); paired vs none: 10 slower, 0 faster, 0 ties, sign p = 0.00195
- C3 ld-reg @(128, k4): 10/10 heritable, KM 150 vs none 200 (ratio 0.75); paired vs none: 5 slower, 5 faster, 0 ties, sign p = 1
- C3 ld-mem @(128, k4): 10/10 heritable, KM 200 vs none 200 (ratio 1.00); paired vs none: 4 slower, 4 faster, 2 ties, sign p = 1
- C3 cb-page @(128, k4): 10/10 heritable, KM 1,300 vs none 200 (ratio 6.50); paired vs none: 6 slower, 4 faster, 0 ties, sign p = 0.754
- C3 ed-loads @(128, k4): 10/10 heritable, KM 450 vs none 200 (ratio 2.25); paired vs none: 6 slower, 3 faster, 1 ties, sign p = 0.508
- C3 stack-writes @(32, k2): 5/10 heritable, KM 228,000 vs none 68,500 (ratio 3.33); paired vs none: 7 slower, 3 faster, 0 ties, sign p = 0.344
- C3 stack-write-only @(32, k2): 10/10 heritable, KM 21,500 vs none 68,500 (ratio 0.31); paired vs none: 4 slower, 6 faster, 0 ties, sign p = 0.754
- C3 stack-read-only @(32, k2): 10/10 heritable, KM 98,000 vs none 68,500 (ratio 1.43); paired vs none: 6 slower, 4 faster, 0 ties, sign p = 0.754
- C3 push-only @(32, k2): 9/10 heritable, KM 93,000 vs none 68,500 (ratio 1.36); paired vs none: 7 slower, 3 faster, 0 ties, sign p = 0.344
- C3 ex-sp-only @(32, k2): 10/10 heritable, KM 74,000 vs none 68,500 (ratio 1.08); paired vs none: 4 slower, 6 faster, 0 ties, sign p = 0.754
- C3 call-rst @(32, k2): 6/10 heritable, KM 42,500 vs none 68,500 (ratio 0.62); paired vs none: 5 slower, 4 faster, 1 ties, sign p = 1
- C3 ld-imm @(32, k2): 9/10 heritable, KM 67,500 vs none 68,500 (ratio 0.99); paired vs none: 4 slower, 5 faster, 1 ties, sign p = 1
- C3 ld-reg @(32, k2): 10/10 heritable, KM 104,500 vs none 68,500 (ratio 1.53); paired vs none: 7 slower, 3 faster, 0 ties, sign p = 0.344
- C3 cb-page @(32, k2): 9/10 heritable, KM 99,000 vs none 68,500 (ratio 1.45); paired vs none: 5 slower, 5 faster, 0 ties, sign p = 1
- C3 ed-loads @(32, k2): 9/10 heritable, KM 60,500 vs none 68,500 (ratio 0.88); paired vs none: 5 slower, 5 faster, 0 ties, sign p = 1

### results/stageD/stage_d/NUMBERS_D.md

#### Stage D numbers (generated)

| label            |   steps |   k |   n |   t_rep_n |   t_rep_km |   t_faith_n |   tq_10_n |   fisher_p_greater | periods       |   ldir_first |
|:-----------------|--------:|----:|----:|----------:|-----------:|------------:|----------:|-------------------:|:--------------|-------------:|
| none             |      32 |   4 |  20 |         4 | inf        |           4 |        20 |         nan        | 3×4           |            4 |
| call-rst-write   |     128 |   4 |  20 |        12 |   1.92e+05 |          12 |        18 |           0.0555   | 3×8, 6×1, 9×3 |           10 |
| none             |     128 |   4 |  20 |         6 | inf        |           6 |        20 |         nan        | 3×5, 9×1      |            6 |
| push             |     128 |   4 |  20 |        11 |   2.26e+05 |          11 |        10 |           0.1      | 3×9, 9×2      |           11 |
| stack-read-only  |     128 |   4 |  20 |         8 | inf        |           8 |        20 |           0.371    | 3×8           |            8 |
| stack-write-only |     128 |   4 |  20 |        12 |   1.38e+05 |          12 |         9 |           0.0555   | 3×11, 9×1     |           11 |
| stack-writes     |     128 |   4 |  20 |        12 |   1.52e+05 |          12 |        11 |           0.0555   | 3×9, 9×3      |           10 |
| none             |     128 |   6 |  20 |         3 | inf        |           3 |        20 |         nan        | 3×3           |            3 |
| call-rst-write   |     512 |   4 |  20 |        14 |   1.6e+05  |          14 |        12 |           0.00519  | 3×10, 9×4     |           12 |
| none             |     512 |   4 |  20 |         5 | inf        |           5 |        15 |         nan        | 3×4, 9×1      |            5 |
| push             |     512 |   4 |  20 |         8 | inf        |           8 |         8 |           0.25     | 3×8           |            8 |
| stack-read-only  |     512 |   4 |  20 |        10 |   2.52e+05 |          10 |        14 |           0.0954   | 3×8, 9×2      |            9 |
| stack-write-only |     512 |   4 |  20 |        19 |   9e+04    |          19 |        16 |           5.01e-06 | 3×16, 9×3     |           18 |
| stack-writes     |     512 |   4 |  20 |        18 |   8.25e+04 |          18 |        16 |           3.43e-05 | 3×16, 9×2     |           18 |

## CMH one-sided (arm > none), pooled over 128 and 512 steps at 1/16

- stack-writes: z = 4.23, p = 1.2e-05
- stack-write-only: z = 4.47, p = 3.9e-06
- stack-read-only: z = 1.61, p = 0.054
- push: z = 1.83, p = 0.033
- call-rst-write: z = 3.32, p = 0.00045

### results/stageE/stage_e/NUMBERS_E.md

#### Stage E numbers (generated)

## E1–E3 size axis per arm

| ablation         | arm     |   tape_len |   n |   t_rep_n |   t_rep_km |   t_rep_km_ix |   t_faith_n |   tiled_frac |   div2L_tiled |   period_median |   whole_tape_n |   func_rnd_median |
|:-----------------|:--------|-----------:|----:|----------:|-----------:|--------------:|------------:|-------------:|--------------:|----------------:|---------------:|------------------:|
| none             | bytes   |          4 |  10 |         1 |     inf    |        inf    |           1 |         0.00 |        nan    |            4.00 |              1 |              0.00 |
| none             | bytes   |          9 |  10 |         6 |  172500.00 |      68489.69 |           6 |         0.50 |          1.00 |            6.00 |              3 |              0.86 |
| none             | bytes   |         16 |  10 |        10 |     150.00 |         59.29 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.84 |
| none             | bytes   |         25 |  10 |        10 |     900.00 |        354.59 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.23 |
| none             | bytes   |         36 |  10 |        10 |     800.00 |        314.36 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.52 |
| none             | bytes   |         49 |  10 |        10 |    1200.00 |        471.26 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.34 |
| none             | bytes   |         64 |  10 |        10 |    1500.00 |        590.32 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.28 |
| none             | bytes   |         81 |  10 |        10 |     700.00 |        275.46 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.25 |
| none             | bytes   |        100 |  10 |        10 |    2150.00 |        845.24 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.89 |
| none             | mubyte  |          4 |  10 |         0 |     inf    |        inf    |           0 |       nan    |        nan    |          nan    |              0 |              0.00 |
| none             | mubyte  |          9 |  10 |         1 |     inf    |        inf    |           1 |         0.00 |        nan    |            9.00 |              1 |              0.00 |
| none             | mubyte  |         16 |  10 |        10 |     200.00 |         78.97 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.84 |
| none             | mubyte  |         25 |  10 |        10 |    1000.00 |        394.54 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.27 |
| none             | mubyte  |         36 |  10 |        10 |     650.00 |        256.45 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.67 |
| none             | mubyte  |         49 |  10 |        10 |    1700.00 |        670.92 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.38 |
| none             | mubyte  |         64 |  10 |        10 |    1400.00 |        552.79 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.94 |
| none             | mubyte  |         81 |  10 |         7 |    2950.00 |       1163.82 |           7 |         1.00 |          1.00 |            2.00 |              0 |              0.91 |
| none             | mubyte  |        100 |  10 |        10 |    3950.00 |       1559.21 |           5 |         1.00 |          1.00 |            3.00 |              0 |              0.66 |
| none             | nominal |          3 |  10 |         0 |     inf    |        inf    |           0 |       nan    |        nan    |          nan    |              0 |              0.00 |
| none             | nominal |          4 |  10 |         0 |     inf    |        inf    |           0 |       nan    |        nan    |          nan    |              0 |              0.00 |
| none             | nominal |          5 |  10 |         9 |   49000.00 |      19338.30 |           9 |         0.00 |        nan    |            5.00 |              9 |              0.81 |
| none             | nominal |          6 |  10 |         9 |  151500.00 |      59811.46 |           4 |         0.11 |          1.00 |            4.00 |              2 |              0.75 |
| none             | nominal |          7 |  10 |         0 |     inf    |        inf    |           0 |       nan    |        nan    |          nan    |              0 |              0.00 |
| none             | nominal |          8 |  10 |        10 |     500.00 |        197.87 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.80 |
| none             | nominal |          9 |  10 |         4 |     inf    |        inf    |           4 |         0.50 |          1.00 |            6.00 |              2 |              0.00 |
| none             | nominal |         10 |  10 |         4 |     inf    |        inf    |           4 |         1.00 |          1.00 |            5.00 |              0 |              0.00 |
| none             | nominal |         12 |  10 |         6 |  245500.00 |      97054.38 |           6 |         0.67 |          1.00 |            6.00 |              1 |              0.88 |
| none             | nominal |         16 |  10 |        10 |     200.00 |         78.94 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.86 |
| none             | nominal |         18 |  10 |        10 |    2100.00 |        828.63 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.94 |
| none             | nominal |         20 |  10 |        10 |     950.00 |        375.17 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.12 |
| none             | nominal |         24 |  10 |        10 |     250.00 |         98.61 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.48 |
| none             | nominal |         25 |  10 |        10 |     900.00 |        355.64 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.31 |
| none             | nominal |         32 |  10 |        10 |     350.00 |        138.15 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.31 |
| none             | nominal |         36 |  10 |        10 |     500.00 |        197.30 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.56 |
| none             | nominal |         49 |  10 |        10 |    1100.00 |        434.31 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.38 |
| none             | nominal |         50 |  10 |        10 |     500.00 |        197.87 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.53 |
| none             | nominal |         64 |  10 |        10 |     750.00 |        296.22 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.25 |
| none             | nominal |         81 |  10 |        10 |     950.00 |        375.10 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.34 |
| none             | nominal |        100 |  10 |        10 |    1650.00 |        651.31 |           9 |         0.90 |          1.00 |            2.00 |              0 |              0.94 |
| none             | steps8L |          4 |  10 |         1 |     inf    |        inf    |           1 |         0.00 |        nan    |            4.00 |              1 |              0.00 |
| none             | steps8L |          9 |  10 |         2 |     inf    |        inf    |           2 |         0.50 |          1.00 |            4.50 |              0 |              0.00 |
| none             | steps8L |         16 |  10 |        10 |     150.00 |         59.22 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.86 |
| none             | steps8L |         25 |  10 |        10 |     900.00 |        355.42 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.30 |
| none             | steps8L |         36 |  10 |        10 |     200.00 |         78.94 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.97 |
| none             | steps8L |         49 |  10 |        10 |     150.00 |         59.25 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.62 |
| none             | steps8L |         64 |  10 |        10 |     100.00 |         39.50 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.44 |
| none             | steps8L |         81 |  10 |        10 |     150.00 |         59.25 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.70 |
| none             | steps8L |        100 |  10 |        10 |     150.00 |         59.21 |          10 |         1.00 |          1.00 |            2.00 |              0 |              0.73 |
| stack-write-only | bytes   |          4 |  10 |         3 |     inf    |        inf    |           3 |         0.00 |        nan    |            4.00 |              3 |              0.00 |
| stack-write-only | bytes   |          9 |  10 |        10 |   52500.00 |      20849.42 |          10 |         1.00 |          1.00 |            3.00 |              0 |              0.86 |
| stack-write-only | bytes   |         16 |  10 |         9 |   58000.00 |      22891.97 |           9 |         0.89 |          1.00 |            4.00 |              0 |              0.91 |
| stack-write-only | bytes   |         25 |  10 |        10 |   53500.00 |      21079.58 |          10 |         0.90 |          1.00 |            5.00 |              1 |              0.95 |
| stack-write-only | bytes   |         36 |  10 |        10 |   15500.00 |       6096.63 |          10 |         0.90 |          1.00 |           10.50 |              1 |              0.98 |
| stack-write-only | bytes   |         49 |  10 |         7 |   39500.00 |      15533.30 |           7 |         0.14 |          1.00 |           48.00 |              3 |              0.97 |
| stack-write-only | bytes   |         64 |  10 |        10 |   49500.00 |      19472.15 |          10 |         0.70 |          1.00 |           32.00 |              2 |              0.97 |
| stack-write-only | bytes   |         81 |  10 |        10 |    7500.00 |       2948.75 |          10 |         0.40 |          1.00 |           70.50 |              4 |              0.97 |
| stack-write-only | bytes   |        100 |  10 |        10 |    3850.00 |       1515.02 |           6 |         1.00 |          0.90 |           50.00 |              0 |              0.95 |
| stack-write-only | mubyte  |          4 |  10 |         1 |     inf    |        inf    |           1 |         0.00 |        nan    |            4.00 |              1 |              0.00 |
| stack-write-only | mubyte  |          9 |  10 |         7 |  175000.00 |      69115.92 |           7 |         0.57 |          1.00 |            3.00 |              3 |              0.89 |
| stack-write-only | mubyte  |         16 |  10 |         8 |   85500.00 |      33777.30 |           8 |         1.00 |          1.00 |            4.00 |              0 |              0.91 |
| stack-write-only | mubyte  |         25 |  10 |        10 |   17000.00 |       6704.54 |          10 |         0.90 |          1.00 |            5.00 |              0 |              0.94 |
| stack-write-only | mubyte  |         36 |  10 |        10 |   14500.00 |       5724.91 |          10 |         1.00 |          1.00 |            6.00 |              0 |              0.97 |
| stack-write-only | mubyte  |         49 |  10 |        10 |   10500.00 |       4143.24 |          10 |         0.80 |          1.00 |            7.00 |              0 |              0.97 |
| stack-write-only | mubyte  |         64 |  10 |        10 |   20000.00 |       7896.21 |          10 |         1.00 |          1.00 |            4.00 |              0 |              1.00 |
| stack-write-only | mubyte  |         81 |  10 |         2 |     inf    |        inf    |           1 |         1.00 |          1.00 |            6.00 |              0 |              0.94 |
| stack-write-only | mubyte  |        100 |  10 |        10 |   16500.00 |       6513.07 |           0 |         1.00 |          1.00 |            4.00 |              0 |              0.73 |
| stack-write-only | nominal |          3 |  10 |         0 |     inf    |        inf    |           0 |       nan    |        nan    |          nan    |              0 |              0.00 |
| stack-write-only | nominal |          4 |  10 |         0 |     inf    |        inf    |           0 |       nan    |        nan    |          nan    |              0 |              0.00 |
| stack-write-only | nominal |          5 |  10 |         8 |  161000.00 |      63557.54 |           8 |         0.00 |        nan    |            5.00 |              8 |              0.81 |
| stack-write-only | nominal |          6 |  10 |        10 |  107500.00 |      42454.65 |           5 |         0.10 |          1.00 |            4.00 |              2 |              0.84 |
| stack-write-only | nominal |          7 |  10 |         5 |  265500.00 |     104878.81 |           5 |         0.00 |        nan    |            7.00 |              5 |              0.39 |
| stack-write-only | nominal |          8 |  10 |         6 |  105000.00 |      41628.45 |           6 |         0.50 |          1.00 |            6.00 |              3 |              0.83 |
| stack-write-only | nominal |          9 |  10 |         9 |  108000.00 |      42648.43 |           9 |         0.56 |          1.00 |            3.00 |              3 |              0.84 |
| stack-write-only | nominal |         10 |  10 |         8 |  189500.00 |      74838.30 |           8 |         0.38 |          1.00 |           10.00 |              5 |              0.88 |
| stack-write-only | nominal |         12 |  10 |        10 |   42000.00 |      16574.49 |           9 |         0.90 |          1.00 |            6.00 |              0 |              0.91 |
| stack-write-only | nominal |         16 |  10 |        10 |   33000.00 |      13033.16 |          10 |         0.90 |          1.00 |            6.00 |              1 |              0.89 |
| stack-write-only | nominal |         18 |  10 |         7 |   41000.00 |      16190.25 |           7 |         1.00 |          1.00 |            4.00 |              0 |              0.94 |
| stack-write-only | nominal |         20 |  10 |        10 |   31000.00 |      12253.00 |          10 |         1.00 |          1.00 |            8.00 |              0 |              0.97 |
| stack-write-only | nominal |         24 |  10 |         8 |   16000.00 |       6317.19 |           8 |         1.00 |          1.00 |            8.00 |              0 |              0.94 |
| stack-write-only | nominal |         25 |  10 |        10 |   56500.00 |      22306.01 |          10 |         1.00 |          1.00 |            5.00 |              0 |              0.97 |
| stack-write-only | nominal |         32 |  10 |        10 |   51500.00 |      20341.25 |          10 |         1.00 |          1.00 |            8.00 |              0 |              0.95 |
| stack-write-only | nominal |         36 |  10 |        10 |    8500.00 |       3353.90 |          10 |         1.00 |          1.00 |            8.50 |              0 |              0.98 |
| stack-write-only | nominal |         49 |  10 |         7 |   46500.00 |      18350.83 |           7 |         1.00 |          1.00 |            7.00 |              0 |              0.97 |
| stack-write-only | nominal |         50 |  10 |        10 |   18500.00 |       7298.52 |          10 |         1.00 |          1.00 |           10.00 |              0 |              0.97 |
| stack-write-only | nominal |         64 |  10 |        10 |   22500.00 |       8885.09 |          10 |         1.00 |          1.00 |            8.00 |              0 |              0.98 |
| stack-write-only | nominal |         81 |  10 |         8 |   13000.00 |       5135.42 |           3 |         1.00 |          0.88 |            9.50 |              0 |              0.97 |
| stack-write-only | nominal |        100 |  10 |        10 |    2150.00 |        848.26 |           1 |         1.00 |          1.00 |           15.00 |              0 |              0.89 |
| stack-write-only | steps8L |          4 |  10 |         1 |     inf    |        inf    |           1 |         0.00 |        nan    |            4.00 |              1 |              0.00 |
| stack-write-only | steps8L |          9 |  10 |         8 |   63500.00 |      25073.39 |           8 |         0.75 |          1.00 |            3.00 |              1 |              0.81 |
| stack-write-only | steps8L |         16 |  10 |         9 |   27000.00 |      10659.58 |           9 |         0.89 |          1.00 |            8.00 |              1 |              0.88 |
| stack-write-only | steps8L |         25 |  10 |        10 |   23500.00 |       9270.80 |          10 |         1.00 |          1.00 |            5.00 |              0 |              0.97 |
| stack-write-only | steps8L |         36 |  10 |        10 |   26000.00 |      10266.66 |          10 |         1.00 |          1.00 |            8.50 |              0 |              1.00 |
| stack-write-only | steps8L |         49 |  10 |        10 |   22000.00 |       8715.39 |          10 |         0.80 |          0.50 |           12.00 |              0 |              0.98 |
| stack-write-only | steps8L |         64 |  10 |         8 |   17500.00 |       6918.43 |           8 |         1.00 |          1.00 |           16.00 |              0 |              1.00 |
| stack-write-only | steps8L |         81 |  10 |        10 |   28000.00 |      11068.63 |          10 |         0.80 |          0.75 |           16.00 |              0 |              0.97 |
| stack-write-only | steps8L |        100 |  10 |         9 |   17000.00 |       6713.89 |           9 |         1.00 |          1.00 |           10.00 |              0 |              1.00 |

### Tiling pooled per arm (first heritable replicators, L ≥ 3)

| ablation         | arm     |   first_replicators |   tiled |   divides_L_of_tiled |   divides_2L_of_tiled | non_divisors_2L                                    |
|:-----------------|:--------|--------------------:|--------:|---------------------:|----------------------:|:---------------------------------------------------|
| none             | bytes   |                  77 |      73 |                0.589 |                 1.000 | –                                                  |
| none             | mubyte  |                  68 |      67 |                0.597 |                 1.000 | –                                                  |
| none             | nominal |                 162 |     140 |                0.793 |                 1.000 | –                                                  |
| none             | steps8L |                  73 |      71 |                0.592 |                 1.000 | –                                                  |
| stack-write-only | bytes   |                  79 |      58 |                0.931 |                 0.983 | L100:p16                                           |
| stack-write-only | mubyte  |                  68 |      61 |                0.918 |                 1.000 | –                                                  |
| stack-write-only | nominal |                 166 |     130 |                0.877 |                 0.992 | L81:p10                                            |
| stack-write-only | steps8L |                  75 |      67 |                0.806 |                 0.910 | L49:p8, L49:p12, L49:p8, L49:p12, L81:p12, L81:p20 |

## E4 budget sweep, L = 100

| ablation         |   steps |   n |   t_rep_n |   t_rep_km |   t_faith_n |   whole_tape_n |   tiled_frac |   period_median |   final_rep_n |   func_rnd_median |   func_rnd_min |
|:-----------------|--------:|----:|----------:|-----------:|------------:|---------------:|-------------:|----------------:|--------------:|------------------:|---------------:|
| none             |      32 |  10 |         0 |     inf    |           0 |              0 |       nan    |          nan    |             0 |              0.00 |           0.00 |
| none             |      64 |  10 |         0 |     inf    |           0 |              0 |       nan    |          nan    |             6 |              0.94 |           0.00 |
| none             |     128 |  10 |        10 |    1650.00 |           9 |              0 |         0.90 |            2.00 |             9 |              0.94 |           0.81 |
| none             |     256 |  10 |        10 |     450.00 |          10 |              0 |         1.00 |            2.00 |            10 |              0.97 |           0.91 |
| none             |    1024 |  10 |        10 |     150.00 |          10 |              0 |         1.00 |            2.00 |            10 |              0.92 |           0.75 |
| none             |    2048 |  10 |        10 |     100.00 |          10 |              0 |         1.00 |            2.00 |            10 |              0.75 |           0.41 |
| stack-write-only |      32 |  10 |         0 |     inf    |           0 |              0 |       nan    |          nan    |             0 |              0.00 |           0.00 |
| stack-write-only |      64 |  10 |         0 |     inf    |           0 |              0 |       nan    |          nan    |             6 |              0.75 |           0.00 |
| stack-write-only |     128 |  10 |        10 |    2150.00 |           1 |              0 |         1.00 |           15.00 |             9 |              0.89 |           0.22 |
| stack-write-only |     256 |  10 |        10 |    7500.00 |          10 |              0 |         1.00 |           15.00 |            10 |              1.00 |           0.94 |
| stack-write-only |    1024 |  10 |        10 |    6500.00 |          10 |              0 |         1.00 |            8.00 |            10 |              1.00 |           0.94 |
| stack-write-only |    2048 |  10 |        10 |    2950.00 |          10 |              0 |         0.90 |           22.50 |            10 |              1.00 |           0.97 |

## E5 mutation sweep, L = 100

| ablation         |   k |   per_byte_mutation_rate |   n |   t_rep_n |   t_rep_km |   t_faith_n |   period_median |   tiled_frac |   final_hoe_median |
|:-----------------|----:|-------------------------:|----:|----------:|-----------:|------------:|----------------:|-------------:|-------------------:|
| none             |   1 |                 0.00205  |  10 |         9 |   1.2e+04  |           2 |             4   |          1   |               3.18 |
| none             |   2 |                 0.00102  |  10 |        10 |   2.45e+03 |           8 |             2   |          1   |               2.02 |
| none             |   3 |                 0.000512 |  10 |        10 |   2.1e+03  |           8 |             2   |          1   |               1.8  |
| none             |   4 |                 0.000256 |  10 |        10 |   1.65e+03 |           9 |             2   |          0.9 |               2.86 |
| none             |   5 |                 0.000128 |  10 |        10 |   1.05e+03 |          10 |             2   |          1   |               1.55 |
| none             |   8 |                 1.6e-05  |  10 |        10 | 950        |           9 |             2   |          1   |               1.51 |
| stack-write-only |   1 |                 0.00205  |  10 |         9 |   9.5e+03  |           0 |             5   |          1   |               3.31 |
| stack-write-only |   2 |                 0.00102  |  10 |        10 |   1.55e+04 |           0 |             5   |          1   |               3.17 |
| stack-write-only |   3 |                 0.000512 |  10 |        10 |   6e+03    |           0 |             8   |          1   |               3.75 |
| stack-write-only |   4 |                 0.000256 |  10 |        10 |   2.15e+03 |           1 |            15   |          1   |               5.5  |
| stack-write-only |   5 |                 0.000128 |  10 |        10 |   1.05e+03 |           0 |            22.5 |          0.9 |               5.49 |
| stack-write-only |   8 |                 1.6e-05  |  10 |        10 |   1.25e+03 |           7 |            32.5 |          0.8 |               5.95 |

## E6 within-seed (10 repeats of seed 1) vs between-seed (seeds 101–110; `none` from Stage E @nominal, `stack-writes` from Stage C) variance

- none L = 16: within-seed 10/10 emerged, KM median 150, log10 SD among emerged 0.25; between-seed 20/20 emerged, KM median 200, log10 SD among emerged 0.46
- stack-writes L = 16: within-seed 8/10 emerged, KM median 86,500, log10 SD among emerged 0.38; between-seed 8/10 emerged, KM median 32,000, log10 SD among emerged 0.45
- none L = 9: within-seed 3/10 emerged, KM median NR, log10 SD among emerged 0.33; between-seed 4/10 emerged, KM median NR, log10 SD among emerged 0.27

## Intrinsic heritability (gen2, isolated assay, 128 steps) of the canonical units tiled to each L

|   L |   LDIR-3 1d ed b0 |   LDIR-4 04 5e ed b0 |   pusher 01 c5 |
|----:|------------------:|---------------------:|---------------:|
|   3 |              1.00 |                 0.01 |           0.00 |
|   4 |              0.00 |                 1.00 |           0.39 |
|   5 |              1.00 |                -0.00 |          -0.00 |
|   6 |              1.00 |                 0.00 |           0.04 |
|   7 |              0.00 |                 0.40 |           0.07 |
|   8 |              0.01 |                 1.00 |           0.53 |
|   9 |              1.00 |                 0.00 |           0.32 |
|  10 |              0.05 |                 0.00 |           0.31 |
|  12 |              0.14 |                 1.00 |           0.20 |
|  16 |              0.00 |                 1.00 |           0.59 |
|  18 |              1.00 |                 0.00 |           0.43 |
|  20 |              0.00 |                 1.00 |           0.46 |
|  24 |              1.00 |                 1.00 |           0.60 |
|  25 |              0.39 |                 0.17 |           0.61 |
|  32 |              0.00 |                 1.00 |           0.72 |
|  36 |              0.83 |                 1.00 |           0.64 |
|  49 |              0.53 |                 0.00 |           0.58 |
|  50 |              0.00 |                 0.00 |           0.58 |
|  64 |             -0.00 |                 1.00 |           0.62 |
|  81 |              0.30 |                 0.50 |           0.71 |
| 100 |              0.75 |                 0.30 |           0.72 |

### same at 32 steps

|   L |   LDIR-3 1d ed b0 |   LDIR-4 04 5e ed b0 |   pusher 01 c5 |
|----:|------------------:|---------------------:|---------------:|
|   3 |              1.00 |                 0.01 |           0.00 |
|   4 |              0.00 |                 1.00 |           0.40 |
|   5 |              1.00 |                -0.00 |          -0.00 |
|   6 |              1.00 |                 0.00 |           0.05 |
|   7 |              0.00 |                 0.52 |           0.13 |
|   8 |              0.01 |                 1.00 |           0.53 |
|   9 |              1.00 |                 0.55 |           0.25 |
|  10 |              0.05 |                 0.00 |           0.42 |
|  12 |              0.33 |                 1.00 |           0.46 |
|  16 |              0.00 |                 1.00 |           0.54 |
|  18 |              0.89 |                 0.00 |           0.45 |
|  20 |              0.95 |                 0.70 |           0.51 |
|  24 |              0.92 |                 0.41 |           0.42 |
|  25 |              0.23 |                 0.04 |           0.37 |
|  32 |             -0.00 |                 0.00 |           0.26 |
|  36 |              0.45 |                -0.00 |           0.18 |
|  49 |              0.08 |                -0.00 |           0.09 |
|  50 |              0.22 |                -0.00 |           0.10 |
|  64 |             -0.00 |                -0.00 |           0.08 |
|  81 |              0.03 |                -0.00 |           0.06 |
| 100 |             -0.00 |                -0.00 |           0.03 |

## E7 the 32-step · mutation 1/4 block-copy exception (seeds 1001–1020)

- block-copy: heritable 0/20, faithful 0/20, tq_10 0/20; KM median t_rep NR; periods –
- none: heritable 18/20, faithful 18/20, tq_10 0/20; KM median t_rep 71,000; periods 4×18
- Fisher exact, heritable: none 18/20 vs block-copy 0/20: two-sided p = 3.35e-09, one-sided (none greater) p = 1.68e-09

### results/stageF/stage_f/NUMBERS_F.md

#### Stage F numbers (generated)

## All ring cells (P = 2L rows from Stage E @nominal)

| ablation         |   L |   P | source   |   n |   t_rep_n |   t_rep_km |   t_faith_n |   tiled |   div_P |   div_2L |   div_L | periods                           |   ldir_first |   func_rnd_median |
|:-----------------|----:|----:|:---------|----:|----------:|-----------:|------------:|--------:|--------:|---------:|--------:|:----------------------------------|-------------:|------------------:|
| none             |   8 |  16 | stageE   |  10 |        10 |     500.00 |          10 |      10 |    1.00 |     1.00 |    1.00 | 2×10                              |            0 |              0.80 |
| none             |   8 |  32 | stageF   |  10 |         2 |     inf    |           2 |       1 |    1.00 |     1.00 |    1.00 | 4×1, 8×1                          |            2 |              0.00 |
| none             |  10 |  20 | stageE   |  10 |         4 |     inf    |           4 |       4 |    1.00 |     1.00 |    1.00 | 2×1, 5×3                          |            3 |              0.00 |
| none             |  10 |  28 | stageF   |  10 |        10 |     500.00 |          10 |      10 |    1.00 |     1.00 |    1.00 | 2×10                              |            0 |              0.31 |
| none             |  10 |  40 | stageF   |  10 |         2 |     inf    |           2 |       2 |    1.00 |     1.00 |    1.00 | 5×2                               |            2 |              0.00 |
| none             |  12 |  24 | stageE   |  10 |         6 |  245500.00 |           6 |       4 |    1.00 |     1.00 |    1.00 | 4×2, 6×2, 8×1, 12×1               |            5 |              0.88 |
| none             |  12 |  28 | stageF   |  10 |        10 |     350.00 |          10 |      10 |    1.00 |     1.00 |    1.00 | 2×10                              |            0 |              0.47 |
| none             |  12 |  36 | stageF   |  10 |        10 |     600.00 |          10 |      10 |    1.00 |     1.00 |    1.00 | 2×10                              |            0 |              0.89 |
| none             |  16 |  32 | stageE   |  10 |        10 |     200.00 |          10 |      10 |    1.00 |     1.00 |    1.00 | 2×10                              |            0 |              0.86 |
| none             |  16 |  33 | stageF   |  10 |        10 |    1150.00 |          10 |      10 |    0.00 |     1.00 |    1.00 | 2×10                              |            0 |              0.28 |
| none             |  16 |  34 | stageF   |  10 |        10 |     500.00 |          10 |      10 |    1.00 |     1.00 |    1.00 | 2×10                              |            0 |              0.53 |
| none             |  16 |  35 | stageF   |  10 |        10 |    3150.00 |           8 |      10 |    0.00 |     1.00 |    1.00 | 2×10                              |            0 |              0.23 |
| none             |  16 |  37 | stageF   |  10 |        10 |     800.00 |          10 |       3 |    0.00 |     1.00 |    1.00 | 2×3, 16×7                         |            0 |              0.14 |
| none             |  36 |  72 | stageE   |  10 |        10 |     500.00 |          10 |      10 |    1.00 |     1.00 |    1.00 | 2×10                              |            0 |              0.56 |
| none             |  36 |  73 | stageF   |  10 |        10 |    1100.00 |          10 |      10 |    0.00 |     1.00 |    1.00 | 2×10                              |            0 |              0.48 |
| none             |  36 |  74 | stageF   |  10 |        10 |     250.00 |          10 |      10 |    1.00 |     1.00 |    1.00 | 2×10                              |            0 |              0.69 |
| none             |  36 |  75 | stageF   |  10 |        10 |     950.00 |          10 |      10 |    0.00 |     1.00 |    1.00 | 2×10                              |            0 |              0.48 |
| none             |  36 |  79 | stageF   |  10 |        10 |     400.00 |           0 |       9 |    0.00 |     1.00 |    1.00 | 2×9, 35×1                         |            0 |              0.67 |
| stack-write-only |   8 |  16 | stageE   |  10 |         6 |  105000.00 |           6 |       3 |    1.00 |     1.00 |    1.00 | 4×3, 8×3                          |            4 |              0.83 |
| stack-write-only |  10 |  20 | stageE   |  10 |         8 |  189500.00 |           8 |       3 |    1.00 |     1.00 |    1.00 | 5×3, 10×5                         |            5 |              0.88 |
| stack-write-only |  12 |  24 | stageE   |  10 |        10 |   42000.00 |           9 |       9 |    1.00 |     1.00 |    1.00 | 4×3, 6×6, 8×1                     |            7 |              0.91 |
| stack-write-only |  16 |  32 | stageE   |  10 |        10 |   33000.00 |          10 |       9 |    1.00 |     1.00 |    1.00 | 4×5, 8×4, 16×1                    |            8 |              0.89 |
| stack-write-only |  16 |  33 | stageF   |  10 |         2 |     inf    |           2 |       0 |  nan    |   nan    |  nan    | 11×1, 16×1                        |            2 |              0.00 |
| stack-write-only |  16 |  34 | stageF   |  10 |         6 |  229500.00 |           5 |       1 |    0.00 |     1.00 |    1.00 | 4×1, 9×1, 15×1, 16×3              |            4 |              0.78 |
| stack-write-only |  16 |  35 | stageF   |  10 |         2 |     inf    |           2 |       1 |    0.00 |     0.00 |    0.00 | 6×1, 15×1                         |            2 |              0.00 |
| stack-write-only |  16 |  37 | stageF   |  10 |         1 |     inf    |           1 |       0 |  nan    |   nan    |  nan    | 15×1                              |            0 |              0.00 |
| stack-write-only |  36 |  72 | stageE   |  10 |        10 |    8500.00 |          10 |      10 |    1.00 |     1.00 |    0.90 | 4×1, 6×3, 8×1, 9×1, 12×4          |           10 |              0.98 |
| stack-write-only |  36 |  73 | stageF   |  10 |         6 |   67000.00 |           6 |       2 |    0.00 |     1.00 |    1.00 | 9×1, 12×1, 25×1, 29×1, 35×1, 36×1 |            6 |              0.97 |
| stack-write-only |  36 |  74 | stageF   |  10 |        10 |   24000.00 |          10 |       8 |    0.00 |     0.88 |    0.88 | 4×5, 12×2, 14×1, 22×1, 26×1       |           10 |              0.97 |
| stack-write-only |  36 |  75 | stageF   |  10 |         9 |  119500.00 |           9 |       4 |    0.00 |     0.50 |    0.50 | 9×1, 12×1, 13×2, 20×3, 27×1, 30×1 |            9 |              0.89 |
| stack-write-only |  36 |  79 | stageF   |  10 |         8 |   88000.00 |           8 |       7 |    0.00 |     0.71 |    0.71 | 7×1, 9×2, 14×1, 18×3, 36×1        |            8 |              0.94 |

## F1 — periods of tiled first replicators under stack-write-only: fraction dividing P (prediction ≥ 0.9 at composite P), and the periods seen

- L = 8, P = 16 (stageE): 6/10 heritable, 3 tiled; divides P 1.00, divides 2L 1.00, divides L 1.00; periods 4×3, 8×3; LDIR-first 4
- L = 10, P = 20 (stageE): 8/10 heritable, 3 tiled; divides P 1.00, divides 2L 1.00, divides L 1.00; periods 5×3, 10×5; LDIR-first 5
- L = 12, P = 24 (stageE): 10/10 heritable, 9 tiled; divides P 1.00, divides 2L 1.00, divides L 1.00; periods 4×3, 6×6, 8×1; LDIR-first 7
- L = 16, P = 32 (stageE): 10/10 heritable, 9 tiled; divides P 1.00, divides 2L 1.00, divides L 1.00; periods 4×5, 8×4, 16×1; LDIR-first 8
- L = 16, P = 33 (stageF): 2/10 heritable, 0 tiled; divides P nan, divides 2L nan, divides L nan; periods 11×1, 16×1; LDIR-first 2
- L = 16, P = 34 (stageF): 6/10 heritable, 1 tiled; divides P 0.00, divides 2L 1.00, divides L 1.00; periods 4×1, 9×1, 15×1, 16×3; LDIR-first 4
- L = 16, P = 35 (stageF): 2/10 heritable, 1 tiled; divides P 0.00, divides 2L 0.00, divides L 0.00; periods 6×1, 15×1; LDIR-first 2
- L = 16, P = 37 (stageF): 1/10 heritable, 0 tiled; divides P nan, divides 2L nan, divides L nan; periods 15×1; LDIR-first 0
- L = 36, P = 72 (stageE): 10/10 heritable, 10 tiled; divides P 1.00, divides 2L 1.00, divides L 0.90; periods 4×1, 6×3, 8×1, 9×1, 12×4; LDIR-first 10
- L = 36, P = 73 (stageF): 6/10 heritable, 2 tiled; divides P 0.00, divides 2L 1.00, divides L 1.00; periods 9×1, 12×1, 25×1, 29×1, 35×1, 36×1; LDIR-first 6
- L = 36, P = 74 (stageF): 10/10 heritable, 8 tiled; divides P 0.00, divides 2L 0.88, divides L 0.88; periods 4×5, 12×2, 14×1, 22×1, 26×1; LDIR-first 10
- L = 36, P = 75 (stageF): 9/10 heritable, 4 tiled; divides P 0.00, divides 2L 0.50, divides L 0.50; periods 9×1, 12×1, 13×2, 20×3, 27×1, 30×1; LDIR-first 9
- L = 36, P = 79 (stageF): 8/10 heritable, 7 tiled; divides P 0.00, divides 2L 0.71, divides L 0.71; periods 7×1, 9×2, 14×1, 18×3, 36×1; LDIR-first 8

## F2 — prime rings (P = 37, 73, 79): tiled LDIR replicators and KM median vs P = 2L

- L = 16, P = 37: 1/10 heritable, tiled 0, LDIR-first 0, KM NR vs 33,000 at P = 32 (ratio nan); periods 15×1
- L = 36, P = 73: 6/10 heritable, tiled 2, LDIR-first 6, KM 67,000 vs 8,500 at P = 72 (ratio 7.9); periods 9×1, 12×1, 25×1, 29×1, 35×1, 36×1
- L = 36, P = 79: 8/10 heritable, tiled 7, LDIR-first 8, KM 88,000 vs 8,500 at P = 72 (ratio 10.4); periods 7×1, 9×2, 14×1, 18×3, 36×1

## F3 — control: `none` KM median per P (prediction: within one 50-step sample of P = 2L)

- L = 16, P = 32 (stageE): 10/10, KM 200 (Δ vs P = 2L: +0 steps); modal period 2; periods 2×10
- L = 16, P = 33 (stageF): 10/10, KM 1,150 (Δ vs P = 2L: +950 steps); modal period 2; periods 2×10
- L = 16, P = 34 (stageF): 10/10, KM 500 (Δ vs P = 2L: +300 steps); modal period 2; periods 2×10
- L = 16, P = 35 (stageF): 10/10, KM 3,150 (Δ vs P = 2L: +2950 steps); modal period 2; periods 2×10
- L = 16, P = 37 (stageF): 10/10, KM 800 (Δ vs P = 2L: +600 steps); modal period 16; periods 2×3, 16×7
- L = 36, P = 72 (stageE): 10/10, KM 500 (Δ vs P = 2L: +0 steps); modal period 2; periods 2×10
- L = 36, P = 73 (stageF): 10/10, KM 1,100 (Δ vs P = 2L: +600 steps); modal period 2; periods 2×10
- L = 36, P = 74 (stageF): 10/10, KM 250 (Δ vs P = 2L: -250 steps); modal period 2; periods 2×10
- L = 36, P = 75 (stageF): 10/10, KM 950 (Δ vs P = 2L: +450 steps); modal period 2; periods 2×10
- L = 36, P = 79 (stageF): 10/10, KM 400 (Δ vs P = 2L: -100 steps); modal period 2; periods 2×9, 35×1

## F4 — the dead zone on padded rings (`none`, L = 8, 10, 12)

- L = 8, P = 16 (stageE): 10/10 heritable (Wilson 0.72–1.00), KM 500, faithful 10; modal period 2; periods 2×10; LDIR-first 0; final random-cell replicator fraction 0.80
- L = 8, P = 32 (stageF): 2/10 heritable (Wilson 0.06–0.51), KM NR, faithful 2; modal period 4; periods 4×1, 8×1; LDIR-first 2; final random-cell replicator fraction 0.00
- L = 10, P = 20 (stageE): 4/10 heritable (Wilson 0.17–0.69), KM NR, faithful 4; modal period 5; periods 2×1, 5×3; LDIR-first 3; final random-cell replicator fraction 0.00
- L = 10, P = 28 (stageF): 10/10 heritable (Wilson 0.72–1.00), KM 500, faithful 10; modal period 2; periods 2×10; LDIR-first 0; final random-cell replicator fraction 0.31
- L = 10, P = 40 (stageF): 2/10 heritable (Wilson 0.06–0.51), KM NR, faithful 2; modal period 5; periods 5×2; LDIR-first 2; final random-cell replicator fraction 0.00
- L = 12, P = 24 (stageE): 6/10 heritable (Wilson 0.31–0.83), KM 245,500, faithful 6; modal period 4; periods 4×2, 6×2, 8×1, 12×1; LDIR-first 5; final random-cell replicator fraction 0.88
- L = 12, P = 28 (stageF): 10/10 heritable (Wilson 0.72–1.00), KM 350, faithful 10; modal period 2; periods 2×10; LDIR-first 0; final random-cell replicator fraction 0.47
- L = 12, P = 36 (stageF): 10/10 heritable (Wilson 0.72–1.00), KM 600, faithful 10; modal period 2; periods 2×10; LDIR-first 0; final random-cell replicator fraction 0.89

## Mechanism trace — the pusher `01 c5` tiled to L, executed as A against the same 8 random partners for 8 … 128 steps

Fraction of A's bytes still intact and fraction of B's bytes equal to the parent, by step; the isolated gen2 (64 partners) in the index.

|                |   ('A_intact_frac', 8) |   ('A_intact_frac', 16) |   ('A_intact_frac', 32) |   ('A_intact_frac', 64) |   ('A_intact_frac', 128) |   ('B_copy_frac', 8) |   ('B_copy_frac', 16) |   ('B_copy_frac', 32) |   ('B_copy_frac', 64) |   ('B_copy_frac', 128) |
|:---------------|-----------------------:|------------------------:|------------------------:|------------------------:|-------------------------:|---------------------:|----------------------:|----------------------:|----------------------:|-----------------------:|
| (8, 16, 0.53)  |                      1 |                    0.84 |                    0.67 |                    0.67 |                     0.67 |                 0.62 |                  0.75 |                  0.67 |                  0.66 |                   0.66 |
| (8, 32, 0.28)  |                      1 |                    0.97 |                    0.88 |                    0.78 |                     0.58 |                 0.62 |                  0.64 |                  0.75 |                  0.72 |                   0.58 |
| (10, 20, 0.31) |                      1 |                    0.96 |                    0.78 |                    0.78 |                     0.75 |                 0.42 |                  0.6  |                  0.49 |                  0.61 |                   0.59 |
| (10, 28, 0.36) |                      1 |                    0.94 |                    0.82 |                    0.59 |                     0.41 |                 0.45 |                  0.61 |                  0.74 |                  0.55 |                   0.48 |
| (10, 40, 0.37) |                      1 |                    0.98 |                    0.92 |                    0.76 |                     0.42 |                 0.42 |                  0.46 |                  0.64 |                  0.74 |                   0.41 |
| (12, 24, 0.2)  |                      1 |                    0.91 |                    0.68 |                    0.6  |                     0.55 |                 0.53 |                  0.74 |                  0.65 |                  0.57 |                   0.49 |
| (12, 28, 0.6)  |                      1 |                    0.89 |                    0.78 |                    0.71 |                     0.65 |                 0.53 |                  0.72 |                  0.77 |                  0.68 |                   0.66 |
| (12, 36, 0.58) |                      1 |                    0.96 |                    0.92 |                    0.8  |                     0.77 |                 0.53 |                  0.69 |                  0.74 |                  0.7  |                   0.64 |
| (16, 32, 0.59) |                      1 |                    0.98 |                    0.87 |                    0.59 |                     0.61 |                 0.5  |                  0.58 |                  0.62 |                  0.55 |                   0.59 |
| (16, 33, 0.51) |                      1 |                    0.98 |                    0.88 |                    0.48 |                     0.42 |                 0.5  |                  0.55 |                  0.74 |                  0.23 |                   0.16 |
| (16, 34, 0.59) |                      1 |                    0.98 |                    0.84 |                    0.61 |                     0.62 |                 0.5  |                  0.54 |                  0.7  |                  0.65 |                   0.62 |
| (16, 35, 0.53) |                      1 |                    0.97 |                    0.86 |                    0.77 |                     0.88 |                 0.5  |                  0.52 |                  0.68 |                  0.24 |                   0.48 |
| (16, 37, 0.51) |                      1 |                    0.98 |                    0.93 |                    0.77 |                     0.78 |                 0.5  |                  0.51 |                  0.7  |                  0.36 |                   0.75 |

### results/closure/NUMBERS_CLOSURE.md

#### Closure (generated)

## 1. Control flow in the first replicator vs the final faithful dominant (`none`, 128 steps, 1/16; Stages stageB,stageC,stageE)

- first replicators with any control-flow instruction: 18 / 256
- final faithful dominants with any control-flow instruction: 105 / 233 (runs with both tapes)
- paired: gained control flow 92, lost 3, both 13, neither 125; exact McNemar two-sided p = 7.2e-24

|   L |   n |   first_cf |   final_cf | final_ops                                  |
|----:|----:|-----------:|-----------:|:-------------------------------------------|
|   5 |   9 |          8 |          9 | JP nn:6, JP NZ,nn:1, JP NC,nn:1            |
|   6 |   2 |          0 |          2 | JP NC,nn:1, JP PO,nn:1                     |
|   8 |  10 |          0 |          8 | RET P:3, JP nn+JR Z,d:1, JP nn+RET PO:1    |
|   9 |   8 |          3 |          7 | JP nn:2, JP M,nn:1, JP C,nn+JP nn:1        |
|  10 |   3 |          1 |          2 | JP nn:1, JP NC,nn:1                        |
|  12 |   6 |          3 |          4 | JP PO,nn:1, JP nn:1, JP NZ,nn:1            |
|  16 |  20 |          0 |         20 | RET NZ:18, JR d+RET NZ:1, JP nn+RET Z:1    |
|  18 |  10 |          1 |          1 | JP nn+JR Z,d:1                             |
|  20 |  10 |          0 |          0 |                                            |
|  24 |  10 |          0 |          8 | JR d:6, CALL NC,nn+JR d:1, JR NZ,d:1       |
|  25 |  20 |          0 |          0 |                                            |
|  32 |  10 |          0 |          1 | JR C,d:1                                   |
|  36 |  30 |          0 |         14 | DJNZ d:13, JR d:1                          |
|  49 |  19 |          0 |          0 |                                            |
|  50 |  10 |          0 |         10 | JR NZ,d:10                                 |
|  64 |  20 |          0 |          0 |                                            |
|  81 |  18 |          0 |          2 | RET C+RET NZ+RST 38:1, CALL PE,nn+RST 38:1 |
| 100 |  18 |          0 |         17 | JP NZ,nn:17                                |

## 2–3. The modal first and final tapes per L: convergence, control flow and partner-independence (256 random partners, one 128-step encounter)

|   L | which   |   n_tapes |   seeds_identical |   distinct |   period | control_flow   |   partners_copied |   self_damaged |   gen2_isolated | tape                                                                                                        |
|----:|:--------|----------:|------------------:|-----------:|---------:|:---------------|------------------:|---------------:|----------------:|:------------------------------------------------------------------------------------------------------------|
|   5 | first   |         9 |                 1 |          9 |        5 | -              |              1.00 |           0.00 |            1.00 | 1d 63 ed b0 1e                                                                                              |
|   5 | final   |         9 |                 5 |          5 |        5 | JP nn          |              1.00 |           0.00 |            1.00 | 00 1d c3 ed b0                                                                                              |
|   6 | first   |         9 |                 2 |          8 |        4 | -              |              0.00 |           0.00 |            1.00 | b0 00 14 ed b0 00                                                                                           |
|   6 | final   |         2 |                 1 |          2 |        6 | JP PO,nn       |              1.00 |           0.00 |            1.00 | 1e a2 e2 1c ed b0                                                                                           |
|   8 | first   |        10 |                 8 |          3 |        2 | -              |              0.58 |           0.39 |            0.53 | 01 c5 01 c5 01 c5 01 c5                                                                                     |
|   8 | final   |        10 |                 3 |          8 |        8 | RET P          |              1.00 |           0.00 |            1.00 | bc e3 21 e3 21 f0 bc f0                                                                                     |
|   9 | first   |         8 |                 3 |          6 |        3 | -              |              1.00 |           0.00 |            1.00 | 1d ed b0 1d ed b0 1d ed b0                                                                                  |
|   9 | final   |         8 |                 1 |          8 |        9 | JP (HL)        |              1.00 |           0.00 |            1.00 | 00 48 1e 99 70 ed b0 57 e9                                                                                  |
|  10 | first   |         4 |                 1 |          4 |        2 | -              |              0.55 |           0.40 |            0.31 | 01 c5 01 c5 01 c5 01 c5 01 c5                                                                               |
|  10 | final   |         3 |                 1 |          3 |        5 | -              |              1.00 |           0.00 |            1.00 | 05 6e eb ed b0 05 6e eb ed b0                                                                               |
|  12 | first   |         6 |                 1 |          6 |        4 | -              |              1.00 |           0.00 |            1.00 | 14 14 ed b0 14 14 ed b0 14 14 ed b0                                                                         |
|  12 | final   |         6 |                 1 |          6 |       12 | JR d           |              1.00 |           0.00 |            1.00 | 1e 3c 18 f1 9c ed b0 78 00 0a b3 18                                                                         |
|  16 | first   |        20 |                11 |          3 |        2 | -              |              0.64 |           0.38 |            0.59 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5                                                             |
|  16 | final   |        20 |                18 |          3 |        8 | RET NZ         |              1.00 |           0.00 |            1.00 | ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0                                                             |
|  18 | first   |        10 |                 4 |          4 |        2 | -              |              0.54 |           0.37 |            0.43 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5                                                       |
|  18 | final   |        10 |                 3 |          5 |        4 | -              |              1.00 |           0.00 |            1.00 | b0 62 14 ed b0 62 14 ed b0 62 14 ed b0 62 14 ed b0 62                                                       |
|  20 | first   |        10 |                 6 |          2 |        2 | -              |              0.67 |           0.46 |            0.46 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5                                                 |
|  20 | final   |        10 |                 8 |          3 |        2 | -              |              0.63 |           0.44 |            0.47 | 21 e5 21 e5 21 e5 21 e5 21 e5 21 e5 21 e5 21 e5 21 e5 21 e5                                                 |
|  24 | first   |        10 |                 8 |          2 |        2 | -              |              0.77 |           0.22 |            0.60 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5                                     |
|  24 | final   |        10 |                 6 |          5 |        6 | JR d           |              1.00 |           1.00 |            1.00 | 01 c5 01 18 f8 c5 01 c5 01 18 f8 c5 01 c5 01 18 f8 c5 01 c5 01 18 f8 c5                                     |
|  25 | first   |        20 |                13 |          4 |        2 | -              |              0.79 |           0.89 |            0.70 | c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5                                  |
|  25 | final   |        20 |                 9 |          5 |        2 | -              |              0.84 |           0.90 |            0.70 | c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5                                  |
|  32 | first   |        10 |                 7 |          2 |        2 | -              |              0.80 |           0.21 |            0.72 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5             |
|  32 | final   |        10 |                 7 |          3 |        2 | -              |              0.75 |           0.27 |            0.66 | 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5             |
|  36 | first   |        30 |                20 |          2 |        2 | -              |              0.77 |           0.18 |            0.64 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 |
|  36 | final   |        30 |                14 |          5 |        2 | -              |              0.72 |           0.16 |            0.64 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 |
|  49 | first   |        20 |                18 |          3 |        2 | -              |              0.79 |           0.15 |            0.58 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 …                                                           |
|  49 | final   |        19 |                 6 |          6 |        2 | -              |              0.68 |           0.77 |            0.62 | d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 …                                                           |
|  50 | first   |        10 |                 9 |          2 |        2 | -              |              0.67 |           0.10 |            0.58 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 …                                                           |
|  50 | final   |        10 |                 7 |          2 |       14 | JR NZ,d        |              1.00 |           0.00 |            0.88 | 01 c5 01 c5 01 c5 01 c5 01 20 f0 c5 01 c5 01 c5 …                                                           |
|  64 | first   |        20 |                14 |          3 |        2 | -              |              0.72 |           0.06 |            0.62 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 …                                                           |
|  64 | final   |        20 |                12 |          3 |        2 | -              |              0.79 |           0.05 |            0.73 | 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 …                                                           |
|  81 | first   |        20 |                19 |          2 |        2 | -              |              0.79 |           0.67 |            0.68 | c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 …                                                           |
|  81 | final   |        18 |                12 |          4 |        2 | -              |              0.86 |           0.70 |            0.68 | c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 …                                                           |
| 100 | first   |        30 |                23 |          7 |        2 | -              |              0.82 |           0.02 |            0.72 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 …                                                           |
| 100 | final   |        18 |                17 |          2 |       10 | JP NZ,nn       |              1.00 |           0.00 |            0.99 | c2 c5 01 c5 01 c5 01 c5 c2 c5 c2 c5 01 c5 01 c5 …                                                           |

### results/census2/NUMBERS_CENSUS2.md

#### Two-byte word census at L = 16 (generated)

- 65,536 words tiled to 16 bytes, culture test with 32 random partners, 128 steps: heritable (gen2 ≥ 0.3) 254, faithful 8, score ≥ 0.5 60
- heritable words with a control-flow instruction: 246/254; partner test (256 partners): copied median 0.00 (max 0.81), self-damage median 1.00 (min 0.14); words copying ≥ 0.95 of partners: 0

| word   | mnemonics              | control_flow   |   score |   gen2 | faithful   |   copied |   damaged |
|:-------|:-----------------------|:---------------|--------:|-------:|:-----------|---------:|----------:|
| 2a e5  | LD HL,(nn) ; PUSH HL   | -              |    0.83 |   0.59 | True       |     0.70 |      0.39 |
| e5 2a  | PUSH HL ; LD HL,(nn)   | -              |    0.82 |   0.57 | True       |     0.81 |      0.14 |
| 21 e5  | LD HL,nn ; PUSH HL     | -              |    0.87 |   0.55 | True       |     0.66 |      0.37 |
| 01 c5  | LD BC,nn ; PUSH BC     | -              |    0.84 |   0.55 | True       |     0.69 |      0.34 |
| c5 01  | PUSH BC ; LD BC,nn     | -              |    0.70 |   0.52 | True       |     0.48 |      0.92 |
| 8d d4  | ADC A,L ; CALL NC,nn   | CALL NC,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4b c4  | LD C,E ; CALL NZ,nn    | CALL NZ,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4b d4  | LD C,E ; CALL NC,nn    | CALL NC,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4b f4  | LD C,E ; CALL P,nn     | CALL P,nn      |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 8b d4  | ADC A,E ; CALL NC,nn   | CALL NC,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 8c cc  | ADC A,H ; CALL Z,nn    | CALL Z,nn      |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 8d cc  | ADC A,L ; CALL Z,nn    | CALL Z,nn      |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 8d cd  | ADC A,L ; CALL nn      | CALL nn        |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4b cd  | LD C,E ; CALL nn       | CALL nn        |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 8d f4  | ADC A,L ; CALL P,nn    | CALL P,nn      |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4d c4  | LD C,L ; CALL NZ,nn    | CALL NZ,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4d d4  | LD C,L ; CALL NC,nn    | CALL NC,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 8b f4  | ADC A,E ; CALL P,nn    | CALL P,nn      |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 6b d4  | LD L,E ; CALL NC,nn    | CALL NC,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 8b cd  | ADC A,E ; CALL nn      | CALL nn        |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 6d d4  | LD L,L ; CALL NC,nn    | CALL NC,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| d4 02  | CALL NC,nn ; LD (BC),A | CALL NC,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4d cd  | LD C,L ; CALL nn       | CALL nn        |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4d f4  | LD C,L ; CALL P,nn     | CALL P,nn      |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 6b cd  | LD L,E ; CALL nn       | CALL nn        |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 6b f4  | LD L,E ; CALL P,nn     | CALL P,nn      |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4b e4  | LD C,E ; CALL PO,nn    | CALL PO,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 6d cd  | LD L,L ; CALL nn       | CALL nn        |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 6d f4  | LD L,L ; CALL P,nn     | CALL P,nn      |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 0d cd  | DEC C ; CALL nn        | CALL nn        |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4d e4  | LD C,L ; CALL PO,nn    | CALL PO,nn     |    0.50 |   0.49 | False      |     0.00 |      1.00 |
| 6d e4  | LD L,L ; CALL PO,nn    | CALL PO,nn     |    0.50 |   0.49 | False      |     0.00 |      1.00 |
| 6b e4  | LD L,E ; CALL PO,nn    | CALL PO,nn     |    0.50 |   0.49 | False      |     0.00 |      1.00 |
| 6a e4  | LD L,D ; CALL PO,nn    | CALL PO,nn     |    0.50 |   0.45 | False      |     0.00 |      1.00 |
| 8a d4  | ADC A,D ; CALL NC,nn   | CALL NC,nn     |    0.50 |   0.45 | False      |     0.00 |      1.00 |
| d4 05  | CALL NC,nn ; DEC B     | CALL NC,nn     |    0.44 |   0.45 | False      |     0.00 |      1.00 |
| 8c d4  | ADC A,H ; CALL NC,nn   | CALL NC,nn     |    0.50 |   0.45 | False      |     0.00 |      1.00 |
| 32 c4  | LD (nn),A ; CALL NZ,nn | CALL NZ,nn     |    0.44 |   0.45 | False      |     0.00 |      1.00 |
| 8c f4  | ADC A,H ; CALL P,nn    | CALL P,nn      |    0.50 |   0.45 | False      |     0.00 |      1.00 |
| 4a e4  | LD C,D ; CALL PO,nn    | CALL PO,nn     |    0.50 |   0.45 | False      |     0.00 |      1.00 |

## Deterministic fixed points against the all-zero partner (THEOREMS.md Proposition 3)

- words whose tiling reappears in the zero partner after one 128-step encounter: 6 / 65,536 — `00 00`, `01 c5`, `11 d5`, `21 e5`, `2a e5`, `e5 2a`
- of these, organism intact after the encounter: 6

### results/bff/NUMBERS_BFF.md

#### BFF soups — numbers (generated; do not edit)

Events: `t_top` = first sample at which the most common tape class is heritable (gen2 ≥ 0.3) and not a one-symbol fill (one byte ≥ 90% of the tape; fills are tar, reported as class 'fill'); it is the exemplar used as the first replicator; `t_her` = heritable fraction of 32 random tapes ≥ 0.5; `t_rep` = the pre-registered Z80 criterion (top-3 class share ≥ 0.5% and gen2 ≥ 0.3), kept for the record; `t_hoe1` = high-order entropy ≥ 1 bit/byte; `t_closed` = top class enters the partner ≤ 5% and copies ≥ 95% of random partners; `t_open` = a replicating top-3 class enters the partner ≥ 50%. Culture tests: 64 random partners, 2^13 steps.

## std: 24 runs (24 finished; epochs done median 16,384)

- transitions: top class heritable (t_top) in 9/24 (median 6464 epochs); heritable fraction ≥ 0.5 (t_her) in 7/24 (median 10880); pre-registered share criterion (t_rep) in 2/24; HOE ≥ 1 in 9/24 (median 10816); closed top class in 9/24; an open replicator ever in the top 3 in 3/24
- first replicators: {'closed': 7, 'intermediate': 2}; with a loop 9/9; median entered 0.00, copies 1.00, self-damage 0.00
- final top class: closed in 7/9, with a loop 7/9; final heritable fraction median 0.97 (max over time, median 1.00, reached at median epoch 8896); collapsed (heritable fraction ≥ 0.5 reached, < 0.1 at the end) 0/9; first replicator a one-byte tiling in 0/9
- before t_rep: mean chunk transfer 0.47 bytes/encounter (median over runs), max 90th percentile 2, max copy-event fraction 0.0117

first replicators (BFF string; `·` = non-instruction byte, `0` = zero):

|   seed |    t_top |    t_her |   t_hoe1 | first_class   | first_loop   |   first_entered |   first_copies |   first_self_damage |   first_gen2 |   first_fill | first_pretty                                                     |
|-------:|---------:|---------:|---------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|     10 |  3072.00 |  3392.00 |  3200.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.12 | ·<······[··,···········}·<,]··,<·}···[·····,·,··[·········<····· |
|     11 | 16064.00 | 16384.00 | 16128.00 | intermediate  | True         |            0.03 |           0.86 |                0.00 |         0.73 |         0.14 | ·<·0··[······,·}·····<·]·[··<·<··[·]·<·····}·,······[·····0·<<·> |
|     16 | 12672.00 | 12736.00 |  9024.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.08 | ··········[<,·}····]··}···,,··<·[············{····{}······{·}··} |
|     17 |  6464.00 |  6400.00 |  6336.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.08 | ····<·,·[··[·[·,·}·····<···]·-······]··<·······}·,·[[·····,·<··· |
|      2 |  3648.00 |   nan    |   nan    | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.22 | ·<·······[,·<·}··,···············]···············,··}·<·,[······ |
|     20 |  2368.00 |  4992.00 |  2432.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.41 | ·····[[[[·[[[[[[[[[[[<·,}]······················]},·<[[[[[[[[[[[ |
|     22 | 11136.00 | 11200.00 | 11072.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.08 | ·[·<····}·········,]·}······,··<[···[[··················,·.····· |
|      4 | 10688.00 | 10880.00 | 10880.00 | intermediate  | True         |            0.34 |           0.72 |                0.05 |         0.56 |         0.31 | [[·[··,···<}······]·········]]·········]······}<···,··[·[[[···<· |
|      9 |  1920.00 |   nan    |   nan    | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.16 | ····[············<,·}·]·}····,<·}·<,····}·]·}·,<············[··· |

final top classes:

|   seed |   final_epoch | final_class   | final_loop   |   final_entered |   final_copies |   final_self_damage |   final_gen2 |   final_fill | final_pretty                                                     |
|-------:|--------------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|     10 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.09 | ·····<········,·····,·····[···}·<,··],<·}···········,··[····+,<· |
|     11 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.11 | -·····+[[··[···[··[·············<····,··}·],····<···}···[[·,,,,< |
|     16 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ·····[<,·}····]··}····,··<·[·······················}·····{····-+ |
|     17 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.06 | ··00··<·,+[-·[···,·}·····<···]······-·]·<·····}·,···[··[·,·<·0·· |
|      2 |         16384 | fill          | False        |            1.00 |           0.00 |                0.00 |         0.00 |         1.00 | ································································ |
|     20 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ·,··············[·<·,}]··························},·<········[·· |
|     22 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.03 | ··{[·····<····}·········,]·}·····,··<·[·····+··················- |
|      4 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.08 | ·······[··········[<···,····}······]·············}····,···<[···· |
|      9 |         16384 | fill          | False        |            1.00 |           0.00 |                0.00 |         0.00 |         1.00 | ································································ |

- runs without a replicator by the criterion: seeds [1, 3, 5, 6, 7, 8, 12, 13, 14, 15, 18, 19, 21, 23, 24]; their final HOE median 0.13, final heritable fraction median 0.00

## wrap: 24 runs (24 finished; epochs done median 16,384)

- transitions: top class heritable (t_top) in 19/24 (median 6912 epochs); heritable fraction ≥ 0.5 (t_her) in 19/24 (median 6976); pre-registered share criterion (t_rep) in 14/24; HOE ≥ 1 in 24/24 (median 64); closed top class in 19/24; an open replicator ever in the top 3 in 2/24
- first replicators: {'closed': 16, 'intermediate': 3}; with a loop 19/19; median entered 0.00, copies 1.00, self-damage 0.00
- final top class: closed in 18/19, with a loop 18/19; final heritable fraction median 0.97 (max over time, median 1.00, reached at median epoch 7680); collapsed (heritable fraction ≥ 0.5 reached, < 0.1 at the end) 0/19; first replicator a one-byte tiling in 0/19
- before t_rep: mean chunk transfer 0.98 bytes/encounter (median over runs), max 90th percentile 62, max copy-event fraction 0.8066

first replicators (BFF string; `·` = non-instruction byte, `0` = zero):

|   seed |    t_top |    t_her |   t_hoe1 | first_class   | first_loop   |   first_entered |   first_copies |   first_self_damage |   first_gen2 |   first_fill | first_pretty                                                     |
|-------:|---------:|---------:|---------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|      1 |  7424.00 |  7424.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.16 | >>>>···<<,·<····<[···<·········,·,,}··],,,]··},,·,·········<···[ |
|     10 |  4224.00 |  4224.00 |    64.00 | intermediate  | True         |            0.00 |           0.89 |                0.17 |         0.81 |         0.12 | <<[[><·[·>[,··<·····}···]······--······]···}·····<··,[>·[·<>[[<< |
|     12 |  8576.00 |  8960.00 |    64.00 | intermediate  | True         |            0.00 |           0.83 |                0.00 |         0.62 |         0.22 | 0<[,····<·}··]<·<·<·<]··}·<····,,····<·}··]<·<·<·<]··}·<····,[<0 |
|     13 |  3712.00 |  3712.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.12 | ··<···[,}····<··-,]·········<··<<··<·········],-··<····},[···<·· |
|     14 |  1536.00 |  1536.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.17 | ,,,,,,,,,>>{··<<···[}<·····,·····]·····]·····,·····<}[···<<··{>> |
|     15 |  4480.00 |  4480.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.31 | 0<[[··,···}············<··,···]··,··<············}···,··[[<····0 |
|     16 |  1728.00 |  1728.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.11 | >>>······<<<···············><<[·>····{·.·]{]·.·{····>·[<<>······ |
|     17 |  7680.00 |  7616.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.19 | {<<[[·<·}>·<,······<·>··]············]··>·<······,<·>}·<·[[<<{{{ |
|     18 |  3968.00 |  4032.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.47 | <,,,,·,,,,[,}<,····],,·,·,,·····,<},[,,,,·,,,,,·,,,·,·<<<···>··> |
|      2 | 13120.00 | 13120.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.14 | ····,[··<···,,·····}···]·····,}····},·····]···}·····,,···<··[,·· |
|     20 |   768.00 |   960.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.28 | [···{·····..····,········>···],,,]···>········,····..·····{···[· |
|     22 |  1088.00 |  1088.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.25 | <[,·,[·[[·[·[·,[[·.>{]>··.{····{{····{.··>]{>.·[[,·[·[·[[·[,·,[< |
|     23 |  7552.00 |  7552.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.39 | ··<·····<,·[,·<··}·,·]··········}··········]·,·}··<·,[·,<·····<· |
|     24 | 16320.00 | 16320.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.22 | ··{····[·····<·····}·,]··]·,,··,,··,,·]··],·}·····<·····[····{·· |
|      3 | 12288.00 | 12288.00 |    64.00 | intermediate  | True         |            0.00 |           0.88 |                0.00 |         0.63 |         0.19 | <<>>><><>><<··<[,·<}],]}<·,[····[,·<}],]}<·,[<·················· |
|      6 | 13952.00 | 13888.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.19 | ·······[<,···}······]]]·]·]·]········]·]·]·]]]······}···,<[····· |
|      7 |  7680.00 |  7552.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.12 | ·<0··0·,0[··········[[,·····<}]·-·]}<·····,[[··········[0,·0··0< |
|      8 |  5376.00 |  5376.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.25 | ·········,,·}<·,[·<··,··}]·,-··<·<·<··-,·]}··,··<·[,·<}·,,······ |
|      9 |  6912.00 |  6976.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.12 | ·{····[··········<·····}····,]<<<<],····}·····<··········[····{· |

final top classes:

|   seed |   final_epoch | final_class   | final_loop   |   final_entered |   final_copies |   final_self_damage |   final_gen2 |   final_fill | final_pretty                                                     |
|-------:|--------------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|      1 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ·········[····<···········,·}··]······},,···········<······[···· |
|     10 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ·······[···<··,·····}···]·······-······]···}·····,··<·[[········ |
|     12 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.11 | ·<,··,··[···<}·,,··]··[··············]·,···]·,·}·<··[··,···<···· |
|     13 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.03 | ··[·<····,}·]·························]·},····<·[··············· |
|     14 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ··{·······[}<·····,·····]·······,·····<}[···<·····{··>········[· |
|     15 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.06 | ···<··,·[····<}··,············]···············,··}<·[·····,·<··· |
|     16 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ·······[··,·········[{·.>··]>.·{·····[·························· |
|     17 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ················{[<},············]·······,}<[{···········[······ |
|     18 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.09 | <···········,·····[·}<·······,·····]···,·<}·[·,··+·····<<··<··>> |
|      2 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.09 | ·····[··<····,·····}···]······}···········]···}······,···<··[[·· |
|     20 |         16384 | fill          | False        |            1.00 |           0.02 |                0.09 |         0.00 |         1.00 | ................................................................ |
|     22 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.06 | {·········[·····[·.>{·····················]{>.··[··············{ |
|     23 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ····.·[<,···}·]·····}···,<·················[·········,·········· |
|     24 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.22 | ··{····[·····<·····}·,]··]·,,··,,··,,·]··],·}·····<·····[····{·· |
|      3 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.12 | ·[····<·,[[,·<}·,]}<·,[·,·······+·······],]}<·,[[,·<············ |
|      6 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.06 | >·<·[··················[<,···}······]]]······}···,<[············ |
|      7 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.06 | 0··<,+············[[,·····<}]·-·]}<·····,[[·······+·····,<······ |
|      8 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.03 | ···[·<··,··}]················}··,··<·[·························· |
|      9 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ·{····[··········<·····}····,]····],····}·····<··········[··[·{· |

- runs without a replicator by the criterion: seeds [4, 5, 11, 19, 21]; their final HOE median 0.92, final heritable fraction median 0.00

## wraplit: 12 runs (12 finished; epochs done median 16,384)

- transitions: top class heritable (t_top) in 12/12 (median 64 epochs); heritable fraction ≥ 0.5 (t_her) in 12/12 (median 64); pre-registered share criterion (t_rep) in 12/12; HOE ≥ 1 in 0/12 (median nan); closed top class in 0/12; an open replicator ever in the top 3 in 12/12
- first replicators: {'open': 12}; with a loop 0/12; median entered 1.00, copies 0.91, self-damage 0.00
- final top class: closed in 0/12, with a loop 0/12; final heritable fraction median 0.00 (max over time, median 0.66, reached at median epoch 64); collapsed (heritable fraction ≥ 0.5 reached, < 0.1 at the end) 12/12; first replicator a one-byte tiling in 12/12
- before t_rep: mean chunk transfer 1.53 bytes/encounter (median over runs), max 90th percentile 64, max copy-event fraction 0.7569

first replicators (BFF string; `·` = non-instruction byte, `0` = zero):

|   seed |   t_top |   t_her |   t_hoe1 | first_class   | first_loop   |   first_entered |   first_copies |   first_self_damage |   first_gen2 |   first_fill | first_pretty                                                     |
|-------:|--------:|--------:|---------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|      1 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.91 |                0.00 |         0.88 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     10 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.89 |                0.00 |         0.89 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     11 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           1.00 |                0.00 |         0.91 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     12 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.92 |                0.00 |         0.78 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      2 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.95 |                0.00 |         0.92 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      3 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.92 |                0.00 |         0.86 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      4 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.88 |                0.00 |         0.82 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      5 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.89 |                0.00 |         0.83 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      6 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.89 |                0.00 |         0.75 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      7 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.91 |                0.00 |         0.85 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      8 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.91 |                0.00 |         0.82 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      9 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.91 |                0.00 |         0.82 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |

final top classes:

|   seed |   final_epoch | final_class   | final_loop   |   final_entered |   final_copies |   final_self_damage |   final_gen2 |   final_fill | final_pretty                                                     |
|-------:|--------------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|      1 |         16384 | open          | False        |            1.00 |           0.95 |                0.00 |         0.80 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     10 |         16384 | open          | False        |            1.00 |           0.94 |                0.00 |         0.82 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     11 |         16384 | open          | False        |            1.00 |           0.94 |                0.00 |         0.85 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     12 |         16384 | open          | False        |            1.00 |           0.89 |                0.00 |         0.86 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      2 |         16384 | open          | False        |            1.00 |           0.92 |                0.00 |         0.72 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      3 |         16384 | open          | False        |            1.00 |           0.92 |                0.00 |         0.86 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      4 |         16384 | open          | False        |            1.00 |           0.89 |                0.00 |         0.86 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      5 |         16384 | open          | False        |            1.00 |           0.86 |                0.00 |         0.70 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      6 |         16384 | open          | False        |            1.00 |           0.97 |                0.00 |         0.91 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      7 |         16384 | open          | False        |            1.00 |           0.86 |                0.00 |         0.78 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      8 |         16384 | open          | False        |            1.00 |           0.94 |                0.00 |         0.85 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      9 |         16384 | open          | False        |            1.00 |           0.95 |                0.00 |         0.86 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |

## Transition rates and between-variant tests

| variant   |   runs |   transition (t_top) |   heritable ≥ 0.5 (t_her) |   median t_her (epochs; censored runs at horizon) |   HOE ≥ 1 |   HOE ≥ 1 without a replicator |   first replicator open |   first closed with loop |   final closed |   collapsed |
|:----------|-------:|---------------------:|--------------------------:|--------------------------------------------------:|----------:|-------------------------------:|------------------------:|-------------------------:|---------------:|------------:|
| std       |     24 |                    9 |                         7 |                                             16384 |         9 |                              2 |                       0 |                        7 |              7 |           0 |
| wrap      |     24 |                   19 |                        19 |                                              7552 |        24 |                              5 |                       0 |                       16 |             18 |           0 |
| wraplit   |     12 |                   12 |                        12 |                                                64 |         0 |                              0 |                      12 |                        0 |              0 |          12 |

- std vs wrap: transitions 9/24 vs 19/24 (Fisher two-sided p = 0.00766); t_her with censored runs at the horizon, Mann–Whitney two-sided p = 0.000286
- std vs wraplit: transitions 9/24 vs 12/12 (Fisher two-sided p = 0.000258); t_her with censored runs at the horizon, Mann–Whitney two-sided p = 1.31e-07
- wrap vs wraplit: transitions 19/24 vs 12/12 (Fisher two-sided p = 0.146); t_her with censored runs at the horizon, Mann–Whitney two-sided p = 8.19e-07

## Readings of THEORY.md P1 (computed, pre-stated thresholds)

- (a) standard BFF: first replicators closed 7/9, with a loop 9/9, open 0/9 → NOT as predicted
- (b) wrap BFF: first replicators open 0/19, closed with a loop 16/19 → (b2) born closed
- (e1) wrap + literal: first replicators straight-line and open 12/12 transitions of 12 runs (≥ 9/12 predicted); closed with a loop 0/12 (≥ 6/12 kills) → (e1) met; (e2) earlier emergence than standard BFF: median t_her 64 vs 16384 (one-sided Mann–Whitney p = 6.56e-08) → met; (e3) final dominant closed in 0/12 transitioned worlds (≥ 6 predicted) → not met; collapsed after the open wave in 12/12 (peak heritable fraction median 0.66 at median epoch 64, final median 0.00)
### results/bff_search/NUMBERS_SEARCH.md

#### Search for an open full replicator in wrap-BFF (generated)

- enumerated: 16,447,860 straight-line programs (prefix ≤ 3 instructions over `<>{}+-.,`, tiling unit of period 2–5 with a copy instruction, repeated to 64 bytes); 2 random partners each, 2^13 steps, wrapping pointer; alphabet filter passed (≥ 75% of each partner's bytes in the program's alphabet): 284,192; both partners ≥ 50% copied: 15947
- stage 2 (32 random partners): programs with copies ≥ 0.5: 378; with copies ≥ 0.5 and gen2 ≥ 0.3: 8; best copies: 0.84 (`,,,` + `,,,,<`, self-damage 0.03, gen2 0.03)
- wall time 1094s

| prefix   | unit   |   copies |   score |   gen2 |   self_damage |   entered |   executed |
|:---------|:-------|---------:|--------:|-------:|--------------:|----------:|-----------:|
| ,,,      | ,,,,<  |     0.84 |    0.73 |   0.03 |          0.03 |      1.00 |    7433.12 |
| },{      | ,<,,,  |     0.84 |    0.72 |   0.03 |          0.03 |      1.00 |    7433.12 |
| -><      | ,,<,,  |     0.84 |    0.69 |   0.09 |          0.06 |      1.00 |    7433.12 |
| },,      | ,,,<   |     0.84 |    0.68 |   0.03 |          0.91 |      1.00 |    7433.12 |
| },>      | ,<,,,  |     0.84 |    0.68 |   0.09 |          0.06 |      1.00 |    7433.12 |
| ,,       | ,,,<   |     0.84 |    0.69 |   0.03 |          0.03 |      1.00 |    7433.12 |
| -,       | ,,,<   |     0.84 |    0.68 |   0.03 |          0.91 |      1.00 |    7433.12 |
| +><      | ,,<,,  |     0.84 |    0.69 |   0.09 |          0.06 |      1.00 |    7433.12 |
| ,{.      | ,,<,,  |     0.84 |    0.69 |   0.03 |          0.06 |      1.00 |    7433.12 |
| ,>,      | ,,,,<  |     0.84 |    0.71 |   0.03 |          0.03 |      1.00 |    7433.12 |
| +,,      | ,,<,   |     0.84 |    0.68 |   0.03 |          0.91 |      1.00 |    7433.12 |
| +{.      | ,,,,<  |     0.84 |    0.68 |   0.03 |          0.06 |      1.00 |    7433.12 |
| -{.      | ,,,<,  |     0.84 |    0.68 |   0.03 |          0.06 |      1.00 |    7433.12 |
| +,       | ,,,<   |     0.84 |    0.68 |   0.03 |          0.91 |      1.00 |    7433.12 |
| ,        | <,,,,  |     0.84 |    0.70 |   0.09 |          0.06 |      1.00 |    7433.12 |
| ,>>      | ,,,,<  |     0.84 |    0.70 |   0.03 |          0.03 |      1.00 |    7433.12 |
| ,><      | ,,<,,  |     0.84 |    0.70 |   0.09 |          0.03 |      1.00 |    7433.12 |
| -<       | ,,,,<  |     0.84 |    0.69 |   0.09 |          0.06 |      1.00 |    7433.12 |
| },>      | ,,,,<  |     0.84 |    0.68 |   0.03 |          0.06 |      1.00 |    7433.12 |
| +>,      | ,,,,<  |     0.84 |    0.70 |   0.03 |          0.03 |      1.00 |    7433.12 |

## Nature of the 'copies' (post hoc characterisation, 32 fresh partners each, all 130 programs with copies ≥ 0.75)

- offspring tapes contain on average 9.7 distinct byte values (min 4.2, max 20.7); the modal byte fills 0.83 of the offspring (min 0.67). Every passing program is a constant fill: it writes the byte under a fixed head over the whole ring, and the parent itself is 0.78 that same byte on average, so the 75% match is met by the parent's own low information content, not by a copy.
- the 8 programs with copies ≥ 0.75 and gen2 ≥ 0.3 have units [',,,,<', ',,,<,', ',,<,,', ',<,,,']: comma-fills whose offspring (a tape of `,`) count as heritable only because the partner's own stray head moves and `,` instructions, executed when the pointer runs into the partner, write the fill onward.
- conclusion: no straight-line program in this family copies a genome of period ≥ 2 into its partner under a wrapping pointer; what exists is one-symbol tar, the BFF analogue of the Z80 zero flood. The open full replicator of the Z80 soup needs a literal write channel (code = data = literal), which BFF lacks.

### results/bff_closed_search/NUMBERS_CLOSED_SEARCH_p10.md

#### Closed self-replicators in BFF + P, wrap, no-halt (generated)

- 1,309,528 periodic tilings (period ≤ 10, alphabet `P[]x`, containing P), 3 partners each (all-x, two random), 2^13 steps
- closed AND ≥ 90% copy into every partner: **0** (of which heritable, gen2 ≥ 0.3: 0); open replicators (≥ 90% copy, pointer enters): 75; closed non-replicators (immune): 734576

open replicators by period (count):
|    |   1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 |   9 |   10 |
|:---|----:|----:|----:|----:|----:|----:|----:|----:|----:|-----:|
| n  |   1 |   1 |   1 |   1 |   1 |   1 |   1 |   1 |   1 |   66 |

### results/stageG/c4/NUMBERS_C4.md

#### C4 — functional fraction of random cells at fixed steps (generated)

|                                |   200 |   1000 |   5000 |   50000 |   300000 |   1000000 |
|:-------------------------------|------:|-------:|-------:|--------:|---------:|----------:|
| ('none@closure', 16, 128, 4)   |  0.01 |   0.07 |   0.12 |    0.81 |     0.88 |    nan    |
| ('none@closure', 50, 128, 4)   |  0.01 |   0.12 |   0.39 |    0.57 |     0.61 |    nan    |
| ('none@closure1M', 20, 128, 4) |  0    |   0.01 |   0.14 |    0.34 |     0.56 |      0.81 |
| ('none@closure1M', 64, 128, 4) |  0.01 |   0.11 |   0.38 |    0.28 |     0.32 |      0.51 |

### faithful

|                                |   200 |   1000 |   5000 |   50000 |   300000 |   1000000 |
|:-------------------------------|------:|-------:|-------:|--------:|---------:|----------:|
| ('none@closure', 16, 128, 4)   |  0.01 |   0.07 |   0.12 |    0.8  |     0.88 |    nan    |
| ('none@closure', 50, 128, 4)   |  0    |   0.08 |   0.27 |    0.45 |     0.46 |    nan    |
| ('none@closure1M', 20, 128, 4) |  0    |   0.01 |   0.13 |    0.33 |     0.54 |      0.81 |
| ('none@closure1M', 64, 128, 4) |  0    |   0.08 |   0.27 |    0.2  |     0.28 |      0.46 |


## S5. Literature notes (LITERATURE.md)

### Literature (working document, 2026-10-08)

Every entry below was checked against a source on the date given (search result or page fetch); items marked *to verify* were not. For each thread: what it says, and how it relates to our results (**A** anticipates, **C** contradicts or complicates, **E** is explained by our results, **M** methodological neighbour).

## 1. Soups of random programs (the AI / ALife line)

- **Agüera y Arcas, Alakuijala, Evans, Laurie, Mordvintsev, Niklasson, Randazzo, Versari (2024). Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction.** arXiv:2406.19108 (v2, 2 Aug 2024). BFF: two 64-byte tapes concatenated (128 bytes), IP and both heads at 0, ten instruction bytes, all else no-op, modular pointers, 2¹³ steps per encounter; soup of 2¹⁷ programs, random ordered pairs per epoch, 0.024% mutation; emergence detected by high-order entropy (Shannon entropy minus brotli-q2 compressed size) and by a drop in unique tokens; "40% of runs show a state transition within 16k epochs"; 2D grid variant (240×135) also produces replicators. Z80: 2D grid of **16-byte programs**, adjacent tapes concatenated into a 32-byte buffer, 256 steps, random mutation; "Early generations use stack-based copy mechanism", later "self-replicators that exploit LDIR or LDDR". Long tape (65,536 bytes): when the heads start at the program counter, "trivial (non-looping) self-replicators rapidly take over the universe"; an offset "somewhat larger than 8" on head1 is needed for looping replicators to arise. SUBLEQ: a 60-byte hand-made replicator exists (25 bytes in RSUBLEQ4) but none arose from random soups "even following billions of executions". Relation: **A** for emergence without fitness and for the stack→LDIR Z80 succession (observed, not quantified or explained); **E** for the succession (open pusher → closed replicator), for the entropy detector's successes and failures (the flood compresses), and for the long-tape remark (Lemma 1: trivial straight-line self-writers are the cheapest fixed points wherever the write pointer starts next to the code). **M**: our instrument differs by the prefix-aware ablation, the culture test (gen2), the census, pre-registration.
- **cubff** (github.com/paradigms-of-intelligence/cubff, Apache-2.0): the simulator behind the paper; languages `bff`, `bff_noheads`, `bff8`, `bff_noheads_4bit`, `bff_perm`, `bff_selfmove`, `forth`, `forthcopy`, `forthtrivial`, `subleq`, `rsubleq4`; CPU build with `make CUDA=0`. We implement BFF ourselves (so the culture test and the IP trace are available) and check semantics against `bff.inc.h`.
- **Knierim, Versari, Obryk, Agüera y Arcas, Saurous (2026). BFF: Simple explanations for complex phenomena.** arXiv:2607.01483 (1 Jul 2026). Argues that distribution-tuned random mutation walks find replicators as readily as pairwise interaction, and that limiting ancestry-tree depth/width curbs take-over but not emergence. Relation: **M/C** — bears on *discovery* of replicators; our claims are about what happens after discovery (openness, closure) and about which primitives matter; their result is consistent with Lemma 1 (the fixed point is cheap to find).
- **Cicala, Niklasson, Randazzo, Boukortt, Basti, Etcheverry, Saurous, Laurie, Manyika, Agüera y Arcas, Richards (2026). Coevolution of self-replication and function in a digital primordial soup.** arXiv:2607.09211 (v1 10 Jul, v2 2 Sep 2026). Random 32-byte Z80 programs; solving a polynomial raises interaction probability; a 4-byte LDIR copier is the standard replicator; neutralising the LDIR family except LDD slowed the transition to ~10M epochs; metabolic (runtime) penalties favour conditional execution. Relation: **A** for Z80 soups and for one instruction ablation; **M**: they add a fitness function, we add none; their LDIR copier is a closed-at-birth replicator in our terms. *To extract from the full text:* their statement about the 2024 stack→LDIR observation.
- **Rasmussen, Knudsen, Feldberg, Hindsholm (1990). The Coreworld: emergence and evolution of cooperative structures in a computational chemistry.** Physica D 42:111–134 (DOI 10.1016/0167-2789(90)90070-6). Core War instructions, shared tape, local energy resource; replicators unstable under copying noise (Ofria et al. 2002, IEEE TEC 6(4), report redcode replicators collapse under slight noise). Relation: **A** for random-start soups; **E**: fragility under noise is openness in our terms.
- **Koza (1994). Spontaneous emergence of self-replicating and evolutionarily self-improving computer programs.** Artificial Life III, pp. 225–262. A handful of replicators among 12.5 million random Lisp programs; estimated odds 10⁻⁶–10⁻⁹ for his function set. Relation: **A/M**: random assembly of replicators; our Lemma 1 says the odds depend on whether the substrate admits a tiny open fixed point.
- **Pargellis (1996). The spontaneous generation of digital "Life".** Physica D 91:86–96 (DOI 10.1016/0167-2789(95)00268-5); **(1996)** The evolution of self-replicating computer organisms, Physica D 98:111–127; **(2001)** Digital life behavior in the Amoeba world, Artificial Life 7 (DOI 10.1162/106454601300328025). Random soups of a tailored instruction set; first replicator after ~400 generations on average; Amoeba-II with a self-defined genetic code. Relation: **A** for spontaneous emergence; **M** for the role of the instruction set.
- **C G, LaBar, Hintze, Adami (2017). Origin of life in a digital microcosm.** Phil. Trans. R. Soc. A 375 (arXiv 1701.03993). Exhaustive survey of all 8-instruction Avida genomes (~2.09 × 10¹¹): 9,141 self-replicate; the smallest Avida replicator needs 8 of 26 instructions. Relation: **M**: the enumeration programme we propose for minimal machines (smallest open / closed self-writers) is the same method; **A** for "life is a small fixed point".
- **Adami & LaBar (2015). From entropy to information: biased typewriters and the origin of life.** arXiv 1506.06988 (in *Information and Causality: From Matter to Life*, CUP). **Adami (2015).** Information-theoretic considerations concerning the origin of life. Origins of Life and Evolution of Biospheres (DOI 10.1007/s11084-015-9439-0). The probability of finding a replicator by chance depends on its information content, not its length; biased monomer distributions raise it by orders of magnitude. Relation: **A/E**: the 2-byte pusher has almost no information, which is why it is found in a few hundred encounters; our tiling law (genome = copy unit, independent of organism length) is the same decoupling of length from information.
- **Ray (1991). An approach to the synthesis of life.** In *Artificial Life II* (Langton, Taylor, Farmer, Rasmussen eds.), SFI Studies in the Sciences of Complexity vol. X, Addison-Wesley, pp. 371–408. Tierra: a seeded 80-instruction ancestor evolves parasites, hyper-parasites and shorter replicators. Relation: **M**: designed ancestor, no spontaneous origin; its parasites are open organisms that depend on a host's code — our pusher is partner-dependent in the same sense but arises unseeded.
- **Ofria & Wilke (2004). Avida: a software platform for research in computational evolutionary biology.** Artificial Life 10(2):191–229 (DOI 10.1162/106454604773563612). Seeded self-replicators, explicit fitness via tasks. Relation: **M**.
- **Langton (1984). Self-reproduction in cellular automata.** Physica D 10(1–2):135–144 (DOI 10.1016/0167-2789(84)90256-2). Designed loop; sheath + signal path; the blueprint circulates in a loop rather than sitting on a tape. **Sayama (1999). A new structurally dissolvable self-reproducing loop evolving in a simple cellular automata space.** Artificial Life 5(4):343–365 — evoloops: unplanned evolution towards smaller loops. Relation: **M/A**: in a designed system, selection for faster copying shrinks the organism; our soups select the *shortest* self-writer first and then add a loop — the opposite direction, because the organism is not designed to be closed.
- **Hutton (2002). Evolvable self-replicating molecules in an artificial chemistry.** Artificial Life 8(4):341–356. Squirm3: template replicators appear spontaneously from a random mixture under some conditions and mutants out-compete them. Relation: **A**: spontaneous origin in an artificial chemistry; **M**: template copying is closed by construction (bonds), so the open phase cannot appear there.
- **Fontana & Buss (1994). "The arrival of the fittest": toward a theory of biological organization.** Bull. Math. Biol. 56(1):1–64. AlChemy: λ-expressions as molecules; self-maintaining organisations (level 0: copiers; level 1: closed self-maintaining sets without copiers) and the argument that organisation, not replication, is primary. Relation: **C/E**: they *suppress* copiers to see organisation; we see organisation (closure) arise *from* an open copier under selection, with no suppression.
- **Dittrich, Ziegler & Banzhaf (2001). Artificial chemistries — a review.** Artificial Life 7(3):225–275. Taxonomy (molecules, reactions, algorithm); Turing-gas soups are one class. Relation: **M**. **Sipper (1998). Fifty years of research on self-replication: an overview.** Artificial Life 4(3):237–257 — four model classes from von Neumann on, all designed replicators. Relation: **M**.

## 2. Origin-of-life theory

- **Nowak & Ohtsuki (2008). Prevolutionary dynamics and the origin of evolution.** PNAS 105(39):14924–14927 (DOI 10.1073/pnas.0806714105). "Prelife": selection and mutation among sequences produced without replication; replicators take over once their rate passes a threshold (a phase transition); an error threshold separates life from prelife. Follow-ups: Ohtsuki & Nowak 2009 Proc R Soc B 276:3783–3790; Manapat, Ohtsuki, Bürger, Nowak 2009 (originator dynamics). Relation: **A** for the concept; **E**: our flood and smears are prelife made concrete, with the added finding that prelife can *prevent* life (assembly suppression at L = 9) as well as precede it.
- **Eigen (1971).** Selforganization of matter and the evolution of biological macromolecules, Naturwissenschaften 58(10):465–523 (DOI 10.1007/BF00623322; verified 2026-10-08); **Eigen & Schuster (1977, 1978)** The Hypercycle: A. Emergence of the hypercycle, Naturwissenschaften 64(11):541–565 (DOI 10.1007/BF00450633); B. The abstract hypercycle, 65:7–41 (DOI 10.1007/BF00420631); C. The realistic hypercycle, 65:341–369; book: Springer 1979 (ISBN 978-3-540-09293-3). Error threshold: maintainable genome length ∝ 1/(per-site error rate). Relation: **A/E**: E5 shows the relation for the LDIR regime (period 5 → 32.5 bytes over two decades of mutation); the 2-byte pusher sits below every threshold, so the paradox is a problem of growth, not origin.
- **Szathmáry (2000). The evolution of replicators.** Phil. Trans. R. Soc. B 355. **Maynard Smith & Szathmáry (1995). The Major Transitions in Evolution.** Limited vs unlimited heredity; replicators that transmit only some phenotypic features; the transitions as changes in how information is transmitted. Relation: **A** for limited heredity first; **E**: the open pusher is a limited-heredity replicator (heritable in a fraction of contexts), the closed successor unlimited within its universe.
- **Woese (2002). On the evolution of cells.** PNAS 99(13):8742–8747 (DOI 10.1073/pnas.132266999); progenote: Woese & Fox 1977. Early evolution communal, dominated by horizontal transfer; the Darwinian threshold when vertical descent took over; model support in later work (gene-content models; Koonin 2014 review). Relation: **A** for the concept; **E**: the budget threshold at L = 100 (heredity without clones below one genome per encounter, lineages above) is a mechanistic instance with one control knob.
- **Dyson (1985/1999). Origins of Life.** Cambridge University Press; 2nd ed. 28 Sept 1999, 112 pp., ISBN 9780521626682 (verified 2026-10-08). Metabolism first, sloppy replication, heredity as a late refinement. Relation: **A** in spirit (sloppy before accurate); **C** in that our first replicators are genetic, not metabolic.
- **Benner, Kim, Carrigan (2012). Asphalt, water, and the prebiotic synthesis of ribose, ribonucleosides, and RNA.** Accounts of Chemical Research (issue on chemical evolution). The "asphalt problem" (sugars decay to tar; reactions leave inert side products); Benner's "tar paradox" at Goldschmidt 2013. Relation: **E**: tar as the content-free operation of the same machinery that, loaded, replicates; the L = 9 reversal quantifies tar suppressing assembly.
- **Szostak (2012). The eightfold path to non-enzymatic RNA replication.** J. Syst. Chem. 3:2. **Rajamani, Ichida, Antal, Treco, Leu, Nowak, Szostak, Chen (2010).** Effect of stalling after mismatches on the error catastrophe in nonenzymatic nucleic acid replication, JACS (stalling >100-fold after a mismatch raises the information ceiling). Later: Prywes et al. 2016 eLife 5:e17756 (activated oligonucleotides enable copying of all four letters); fidelity depends strongly on sequence and conditions. Relation: **A/E**: non-enzymatic template copying is real but sloppy and context-dependent — the chemical analogue of the open replicator; our result predicts selection acts first on context-independence.
- **Kauffman (1986). Autocatalytic sets of proteins.** J. Theor. Biol. 119(1):1–24; **Kauffman (1993)** The Origins of Order: Self-Organization and Selection in Evolution, Oxford University Press, New York (ISBN 9780195079517). **Hordijk & Steel (2004). Detecting autocatalytic, self-sustaining sets in chemical reaction systems.** J. Theor. Biol. 227(4):451–461 (DOI 10.1016/j.jtbi.2003.11.020). RAF sets: reflexively autocatalytic and food-generated; emergence of collective autocatalysis above a catalysis-density threshold (Erdős–Rényi argument). Relation: **M**: collective (sub-clonal) heredity is their regime; our 64-step soups with heritable populations and no clone are the program analogue.
- **Lincoln & Joyce (2009). Self-sustained replication of an RNA enzyme.** Science 323(5918):1229–1232 (DOI 10.1126/science.1167856): cross-catalytic ribozyme pair, exponential isothermal amplification from four substrates. **Vaidya, Manapat, Chen, Xulvi-Brunet, Hayden & Lehman (2012). Spontaneous network formation among cooperative RNA replicators.** Nature 491:72–77 (DOI 10.1038/nature11549): fragments assemble into cooperative cycles that out-grow selfish autocatalysts. Relation: **M**: experimental cooperative heredity; our open pusher is the opposite case — a selfish copier that needs a quiet context — and closure is its route to independence.

## 3. Autonomy, closure, individuality, definitions of life

- **Krakauer, Bertschinger, Olbrich, Flack, Ay (2020). The information theory of individuality.** Theory in Biosciences 139:209–223 (DOI 10.1007/s12064-020-00313-7; arXiv 1412.2447). Individuality as information flow from a system's past to its future, relative to the environment; organismal, colonial and driven ("environmentally determined") individuality; no physical boundary required. Relation: **A** for the measure; **E**: our partner-independence is the replicator case of their quantity, and we observe it evolve from ≈ 0.6 (driven) to 1.0 (organismal).
- **Maturana & Varela (1972/1980). Autopoiesis and Cognition.** Reidel. Operational closure; the organism's operations produce the organisation that produces them. **Rosen (1991). Life Itself.** Columbia UP. Closure to efficient causation. **Mossio & Moreno (2010). Organisational closure in biological organisms.** Hist. Phil. Life Sci. 32(2–3):269–288. **Moreno & Mossio (2015). Biological Autonomy: a philosophical and theoretical enquiry.** Springer (History, Philosophy and Theory of the Life Sciences 12). Springer; **Mossio, Bich & Moreno (2013)** on closure and constraints. Relation: **A** conceptually; our contribution is a measurable, evolving closure in a system with no spatial boundary (control closure, Definition 2), and the finding that control closure can be coordinate-free while the write channel stays geometric (P3).
- **Chaitin (1970). To a mathematical definition of "life".** ACM SIGACT News issue 4, pp. 12–18 (DOI 10.1145/1247047.1247052); **Chaitin (1979).** Toward a mathematical definition of "life", in *The Maximum Entropy Formalism* (MIT Press), pp. 477–498. Algorithmic-information-theoretic definition (organisation as mutual information between parts). Relation: **M**: the first attempt at a computational definition; ours is dynamical (forward invariance under execution over contexts) rather than descriptive.
- **Walker & Davies (2013). The algorithmic origins of life.** J. R. Soc. Interface (arXiv 1207.4803). Life begins when information gains causal (top-down) control over its substrate. Relation: **A** in spirit; **E**: closure is the moment the organism's information, not the context's, determines its own propagation.
- **Bartlett & Wong (2020). Defining Lyfe in the Universe: from three privileged functions to four pillars.** Life 10(4):42 (DOI 10.3390/life10040042). Dissipation, autocatalysis, homeostasis, learning. Relation: **M**: our ladder adds an operational criterion (context-independent heredity) that is measurable by intervention.
- **Kleene's recursion theorem**; **von Neumann (1966).** Theory of Self-Reproducing Automata (Burks ed.). Relation: Lemma 1 (existence of fixed points is guaranteed; their minimal size is substrate-dependent); our result dissolves the constructor architecture at the origin.
- **Waddington (1942). Canalization of development and the inheritance of acquired characters.** Nature 150(3811):563–565 (DOI 10.1038/150563a0). **Wagner (2005). Robustness and Evolvability in Living Systems.** Princeton University Press (Princeton Studies in Complexity), 2005; paperback 2007. Relation: **E**: closure is canalization of reproduction against environmental noise.

## 4. Biosignatures and detection

- **Viking labeled-release experiment** (Levin & Straat 1976–; Levin 1997 SPIE; Levin & Straat 2016 Astrobiology): an activity (culture) test; positive responses, heat-sterilised controls weaker; mainstream interpretation chemical (perchlorate/hypochlorite; McKay, Quinn, Stoker 2025 Icarus). Relation: **M**: the archetype of an intervention-based test; our result says state-based signatures fail in principle where sterile order precedes life.
- **Sharma, Czégel, Lachmann, Kempes, Walker, Cronin (2023). Assembly theory explains and quantifies selection and evolution.** Nature 622. Critiques: Uthamacumaran et al. (arXiv 2210.00901) and Zenil's group (arXiv 2403.06629, 2408.15108) argue the assembly index is a compression measure; Hazen et al. find abiotic heteropolyanions with assembly index up to 21 (above the proposed threshold of 15); Jaeger 2024 (PMC10978598) measured critique. Relation: **E**: our detector benchmark shows compression-type and order-type signatures fail where tar is more orderly than life; heredity needs an intervention.
- **Agnostic biosignatures** (Laboratory for Agnostic Biosignatures; Life Detection Knowledge Base taxonomy: chemistry / structure / activity). Relation: **M**: our ladder maps onto the activity category.

## 5. Thermodynamics (directions only)

- **England (2013). Statistical physics of self-replication.** J. Chem. Phys. 139:121923 (DOI 10.1063/1.4818538; arXiv 1209.1179): lower bound on heat dissipated per replication in terms of growth rate, durability and internal entropy. **Kolchinsky (2024)** arXiv 2404.01130 disputes the universal reading. **Still, Sivak, Bell, Crooks (2012). Thermodynamics of prediction.** PRL 109:120604: non-predictive information retained about the environment ("nostalgia") lower-bounds dissipation. Relation: the open organism imports context information that predicts nothing about its own future; closure removes the import. To be pursued only if an inequality can be written for our system.

## 6. To add and verify

Verified 2026-10-08 and placed above: Dittrich, Ziegler & Banzhaf 2001; Fontana & Buss 1994; Langton 1984; Sayama 1999; Hutton 2002; Ray 1991; Ofria & Wilke 2004; Eigen 1971; Dyson 1999; Mossio & Moreno 2010 and Moreno & Mossio 2015; Waddington 1942; Wagner 2005; Lincoln & Joyce 2009; Vaidya et al. 2012; Hordijk & Steel 2004; Kauffman 1986. Verified 2026-10-08 04:00: Sipper 1998; Eigen & Schuster 1977–78 (A, B, C); Kauffman 1993. Still to verify: Cicala et al.'s exact wording on the 2024 stack→LDIR observation.
