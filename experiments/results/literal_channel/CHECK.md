# Independent check of Proposition 9 and Conjecture 9′

Date: 2026-10-11. Branch: `feat/conj9`, based on `feat/experiments-runner`
at `dadd241`. CPU only; no Modal or GPU. The planning note and the Supplementary
Information in `experiments/manuscript/MAIN_nature.md` (especially Proposition 3,
Theorem 6, and the notation preceding them) were read independently.

## Verdict

**Proposition 9 needs a correction.** Its Fine–Wilf argument is otherwise sound,
but its stated copy-length bound is one byte too short for odd widths. Use

\[
N\ge (w+2)+w-\gcd(w+2,w)=2w+2-\gcd(2,w).
\]

Thus the bound is **2w+1 for odd w, 2w for even w**. The original bound is 2w
for both. There are concrete finite-window counterexamples at the original
odd-width bound, not merely a missing justification. At N=16 the corrected
bound holds for every w≤7 and also for w=8. The parity conclusion for an infinite
exact output stream holds without a finite-window qualification.

**Conjecture 9′ is proved for the requested infinite periodic input/output
model, including arbitrarily many distinct load and push opcodes sharing one
register.** No width-1 writer through execution period 64 was found in an exact
symbolic search with a proved reduction to minimal decoder cycles; an independent
union-find enumeration directly covered every token order through period 40.

**A finite-window interpretation must be distinguished.** For an unmodified
periodic input, width 1 has a period-16 mixture that writes a matching 16-byte
factor, then disagrees at byte 17. Therefore an unrestricted finite-window
version of the conjecture is false. This word does **not** reproduce in the
zero-partner 6502 ring harness. The infinite-stream proof must not be presented
as a proof about every finite soup encounter or every 6502 instruction family.

**Width 3 has mixture writers.** The requested expectation “w=3 none” is true
for the repeated-pair family in Proposition 9, and false for mixtures. The
smallest returned mixture has primitive byte period 3 and execution period 6.

## 1. Proposition 9: corrected statement and proof

State all of these assumptions:

* A load has one opcode byte a followed by exactly w literal bytes L. A push has
  one opcode byte b≠a. No prefixes or intervening instructions occur in the
  repeated execution pair `(a L b)`.
* Both instructions use the same data register R. Loading installs L verbatim;
  pushing preserves R and emits a fixed permutation π(L) of its bytes. The
  stack pointer is separate: neither the load nor the data value changes its
  addressing, and pushing changes it by exactly −w.
* The executed byte stream remains `(a L b)^∞` over the part under discussion.
  Source mutation, virgin partner bytes, transient initial pushes and later
  overwrites cannot silently be identified with that prescribed stream.
* Consecutive pushes occupy contiguous w-byte blocks. Reading memory in
  ascending address order reverses the **order of blocks**, not their internal
  byte layout. The copied region consists of these writes, with the same layout
  at every push and no holes or unrelated final writes.
* For a finite copied window, require the corrected bound above. For infinite
  equality, no separate copy-length bound is necessary.

Let E=`(a L b)^∞`, ℓ=w+2, and C=π(L). Every push has the same R, so the written
stream in ascending memory is C^∞, up to phase. A copied N-byte window has
periods ℓ and w. Fine–Wilf gives period g=gcd(ℓ,w)=gcd(2,w) at
N≥ℓ+w−g. This bound is at least ℓ for all w≥1, so the window contains every
residue of the execution period; the g-periodicity extends to all of E.

For odd w, g=1 forces a=b, a contradiction. For even w, g=2 makes E alternate.
Position 0 is a and position w+1 is the odd-position byte b, hence
E=`(a b)^∞` and L=`(b a)^(w/2)`. The written block C must also alternate.
Conversely, such an alternating layout emits a shift of E. For w=2 both
permutations of the two bytes qualify. This is a sufficient construction for
the prescribed periodic-stream model; completion in a particular encounter
still requires its placement, phase, budget and bootstrap conditions.

The original odd-width bound fails sharply in these examples (`a≠b`):

| w | execution period | literal | ascending layout π | output block | matching window |
|---|---|---|---|---|---|
| 1 | `a a b` | `a` | `(0)` | `a` | `a a`, length 2; next bytes differ |
| 3 | `a b a a b` | `b a a` | `(1,0,2)` | `a b a` | `a b a a b a`, length 6; next bytes differ |

The second example includes a complete execution period in the matching window.
`finite_bound_counterexamples.json` records concrete byte instances.

An independent symbolic union-find check of all permutations, without invoking
Fine–Wilf inside the algorithm, gives the following at the corrected bound.
Every literal is a free byte, including opcode bytes; it is not sampled.

| w | layouts checked | layouts permitting a repeated-pair writer |
|---|---:|---:|
| 1 | 1 | 0 |
| 2 | 2 | 2 |
| 3 | 6 | 0 |
| 4 | 24 | 8 |
| 5 | 120 | 0 |
| 6 | 720 | 72 |
| 7 | 5,040 | 0 |
| 8 | 40,320 | 1,152 |

For w=2r the count is 2(r!)²: each parity class of literal positions must map
to one parity class of output positions. See `pair_layouts.csv` (46,233 layouts).

### Relation to Theorem 6 (H1)–(H3)

Proposition 9 is best presented as an **independent structural lemma
characterising when this restricted literal channel can realise (H3)**,
followed by a conditional application of Theorem 6(ii). It is not a corollary
of Theorem 6: that theorem assumes a self-consistent literal write and does not
derive byte-width restrictions. The new result supplies that restriction.

(H1) supplies default flow and the interpretation as an open program when
execution reaches the partner. It is compatible with descending data writes;
instruction flow and stack flow have opposite directions. It does not supply
the partner coverage or execution budget by itself.

(H2) is a machine-wide copy-cost hypothesis. Failure of a repeated literal pair,
or even failure of all width-1 single-register literal mixtures, does not prove
(H2). Memory copiers, stack exchanges, return-address writers and other families
remain outside this analysis. No conclusion that every 6502 replicator must be
closed follows from Proposition 9 or from the proof below.

(H3) must be understood at the **executed phase of the tiling**, allowing the
load–push macro and the emitted layout. Merely having an immediate load and a
push is insufficient. The 8086 SP pair is excluded: loading SP redirects the
stack, and original-8086 PUSH SP emits the decremented SP. In the zero-partner
ring, even a legitimate literal channel can fail to bootstrap in one phase.

## 2. Census agreement and exceptions of scope

Both repository Z80 lists agree: `review_r2/census_p2.csv` and
`census6502/control_z80_words.csv` contain the same five nontrivial period-2
words. Their literal subset is exactly:

| word | channel | literal at the load | ascending push bytes |
|---|---|---|---|
| `01 c5` | LD BC,nn; PUSH BC | `c5 01` | `c5 01` |
| `11 d5` | LD DE,nn; PUSH DE | `d5 11` | `d5 11` |
| `21 e5` | LD HL,nn; PUSH HL | `e5 21` | `e5 21` |

All three have the predicted alternating form and L=`b a`. The other two,
`2a e5` and `e5 2a`, alternate too, but opcode 2a is **LD HL,(nn)**: its two
following bytes give a memory address, not an immediate register value. They
are memory-fed stack copiers, outside Proposition 9. Proposition 3 correctly
groups them as load–push words; that grouping must not be reinterpreted as five
literal-channel witnesses. `00 00` is the trivial already-zero fixed point and
is not in these nontrivial lists.

The requested temporary 8086 census directory was present. Its log lists 29
self-writers from 55,937 evaluated unequal-byte words, with 9,343 unknown words
(256 equal-byte words were skipped). Its interpreter describes itself as
partial and unvalidated; this audit establishes list agreement, not independent
validation of the entire 8086 ISA census.

Its **13 literal-channel entries** are:

* load-first: `b9 51`, `ba 52`, `bb 53`, `bd 55`, `be 56`, `bf 57`;
* push-first: `50 b8`, `51 b9`, `52 ba`, `53 bb`, `55 bd`, `56 be`, `57 bf`.

All have `(a b)^∞` up to phase and L=`b a` at their load boundary. A push-first
word begins with an initial-register push; the repeated load–push pair follows
afterwards. **There are no nonalternating literal-push exceptions.**

The remaining 16 listed 8086 words lie outside the literal grammar:
`50 0d`, `ff 30`, `ff 31`, `ff 32`, `ff 33`, `ff 34`, `ff 35`, `ff 37`,
`ad 50`, `c4 52`, `07 a4`, `07 a5`, `50 ad`, `57 c7`, `0e ea`, `35 ff`.
They use arithmetic, memory/indirect operands, string copies, segment operations,
stores or control transfer; notably `0e ea` includes a far jump.

The absence of `b8 50`, despite the presence of `50 b8`, illustrates the
bootstrap qualification. Replaying the supplied interpreter at its zero-partner
settings produces `f8 50` repeated for `b8 50`, but an exact partner for
`50 b8` and `b9 51`. The trace shows that later loads in the former encounter
read altered operands: AX first changes from b850 to 50f8 at step 34. Both
orientations of the SP pair (`bc 54`, `54 bc`) also fail. These are finite
harness effects, not exceptions to the prescribed-stream lemma.

`census_audit.csv` records all 39 source rows (the five Z80 rows occur in two
sources); `census_provenance.json` records source hashes and replay outcomes.
`8086_harness_trace.csv` records the concrete replay. No complete census was
rerun, and none of these replays used a GPU.

## 3. Exhaustive symbolic search

Here **k is an execution byte period**, so k=(w+1)nL+nP. Primitive byte period
can be smaller: the w=2 witness has execution period 4, primitive period 2.
Both counts are positive. Pure pushes with a preinitialised opcode-valued R
would be a degenerate constant writer; that initial-state resource is excluded
by the census's zero R and nonzero push opcode, and must be excluded explicitly
in an abstract statement with unconstrained initial R.

Every load operand is a free byte. An operand can equal any load or push opcode;
there is no “operand alphabet disjoint from opcodes” shortcut. All token orders
are considered, modulo rotating an execution cycle to start at a load. Loads
and pushes use disjoint opcode sets, all loads set the same R, and all pushes
have the same byte layout.

### Exact reduction and coverage through 64

For a candidate with execution period k and output-block period p=w nP,
infinite equality implies code period g=gcd(k,p). The SMT search represents
each of the g code bytes as a free 8-bit variable, all token orders as Boolean
load/push choices, all register updates and the descending output order, and
every output phase. Register state is cyclic; the first token is a load, so
the concrete replay does not rely on an arbitrary initial register value.

There is a complete reduction to **minimal decoder cycles**. If the input's
primitive byte period is d, its next instruction is determined by its first-byte
residue modulo d. A minimal recurrent cycle visits each such residue at most
once. If its t=nL+nP instructions consume k bytes and emit p bytes, then d
divides k and, for a self-writer, d divides p. Thus t≤d≤gcd(k,p)=g. Cycles with
t>g can be discarded; repeated traversal of a cycle cannot restore an absent
infinite-stream match. Distinct boundary residues modulo g are imposed on
remaining SMT candidates. This is a proved symbolic reduction, **not literal
enumeration of trillions of concrete words**.

For width 1, g=gcd(k,nP)≤nP<t, so every minimal-cycle candidate is rejected.
`minimal_symbolic_search.csv` contains **992 width-1 count cases through k=64,
all UNSAT, no unknowns**. Their unreduced start-at-load token-order counts sum
to 17,167,680,177,532; those orders are covered by the reduction, not visited
one by one. The whole reduced run, including w=2/3 controls through k=16,
took about 0.55 seconds on this CPU.

Several distinct load and push opcodes are covered exactly. Map all load
opcode byte values to A, all push opcode values to B, and other bytes to C.
This map preserves instruction lengths, decoder boundaries, literal copying,
and equality of streams. Any multi-opcode counterexample would map to a
single-load/single-push counterexample. Therefore the single-opcode UNSAT
result covers **any finite numbers of distinct load and push opcodes** with
the stated shared-register semantics. No assumption prevents literals from
mapping to A or B. A further direct SMT check, without collapsing opcode
identities, uses load opcodes {01,02,03} and push opcodes {80,81}, all acting on
the same abstract register: every count case through k=16 is UNSAT.

### Independent direct enumeration through 40

`direct_search.py` uses no Z3 and no minimal-cycle pruning. It recursively
enumerates every load/push token order beginning at a load, then solves byte
equalities with union-find. Opcode positions fix constants; literal positions
remain free, and a conflict occurs only when unequal opcode constants are
forced together. Code period g is imposed and all g output phases are checked.
This is exact for the infinite-stream problem because both periods must divide
the matching stream.

**Width 1: none through execution period 40**, over **165,580,120** directly
enumerated token orders, in **216.93 seconds**. The reduced search supplies the
range through 64; the direct enumeration supplies an independent finite check
of its conclusion. See `direct_search.csv` and `direct_run.json`. An initial
unreduced SMT exploration was discarded because the proved reduction and
direct enumeration gave complete, reproducible coverage;
no incomplete or timed-out SMT result is counted as UNSAT.

### Width controls and the unexpected width-3 mixture

The reduced control search through execution period 16 has 35 w=2 count cases
(2 SAT), and 24 w=3 cases (3 SAT), with no unknowns. Each returned SAT model was
independently decoded and compared over lcm(k,p) bytes. The direct enumerator
through k=16 independently confirms these witnesses and also retains
repetitions of smaller execution cycles. Full SMT constraints without the
minimal-cycle reduction were cross-checked against the direct enumerator for
every count case through k=16 and widths 1, 2 and 3; see `verification.csv`.

* **w=2:** E=`(a b)^∞`; load `a [b a]`, push `b`. Execution period 4; ascending
  output `b a` repeated. Both layouts work, as Proposition 9 predicts.
* **w=3, repeated pair:** no writer, under either infinite equality or the
  corrected finite bound; all six layouts were checked.
* **w=3, mixture:** E=`(a b b)^∞`; execute `a [b b a]; b; b`. The execution
  word is `a b b a b b` (k=6). Each push writes `b b a`, so the ascending
  output is `(b b a)^∞`, a shift of E. **This is a counterexample to an odd-width
  prohibition for arbitrary mixtures**, not to Proposition 9. Every permutation
  of this literal has one a and two b's and is a rotation of E.

More generally, for each w≥2, identity layout and the word
`(a b^(w−1))^∞` execute a load of `b^(w−1) a` followed by w−1 pushes. The
execution period is 2w, and every output block is a shift of the same period-w
word. Hence the repeated-pair even/odd boundary does not extend to mixtures.

## 4. Proof of Conjecture 9′ in the infinite-stream model — PROVED

Let E have primitive byte period d. Decode E sequentially using only one-byte
push opcodes and load opcodes followed by one literal byte. At a token boundary,
the next opcode and next boundary depend only on the current address modulo d,
not on R: loads consume two bytes, pushes one. The finite graph of d boundary
residues has an eventual directed cycle with t≤d distinct vertices.

If the cycle contains nL>0 loads and nP>0 pushes, then nP<t≤d. Once the cycle
has repeated, its loads reset the register in the same way on every traversal.
The chronological emitted word per cycle therefore has length nP. Reading the
descending stack in ascending order reverses that word, so the steady written
stream has period nP as well.

If the output is a shift of E^∞, then nP is a period of E. Every period of an
infinite periodic word is a multiple of its primitive period d. But
0<nP<d, a contradiction. Thus no such width-1 periodic self-writer exists.

A recurrent all-load cycle produces no unbounded output. A recurrent all-push
cycle visits every byte residue (each step is +1), so E consists solely of push
opcodes and the unchanged R emits a constant. This can match only a constant
push-opcode word with that same preinitialised R, the excluded degenerate
initial-state case. In particular it cannot match the 6502's zero initial A
and nonzero PHA opcode. Finite initial transients cannot alter the steady-period
contradiction. Multiple opcode identities do not affect the graph argument.

For the expansion interpretation, the byte displacement around the width-1
cycle is P_e=2nL+nP. It is a positive multiple of d, but
P_e≤2t−nP<2d, so P_e=d. Therefore the claimed expansion
P_e/nP=1+2nL/nP>1 is real, and it forces a period contraction inconsistent with
minimality of d. The needed extra ingredient is deterministic periodic
decoding; the numerical inequality alone is not a nonperiodicity theorem.

**Toeplitz claim: not proved for arbitrary mixtures.** This argument proves
absence of a periodic solution, not that a nonperiodic solution exists, or that
every such solution has Toeplitz structure. The repeated-pair recurrence in the
planning note does not establish a common Toeplitz recurrence for variable
push runs and descending output phases. The minimal-period proof replaces
that unproved part and is sufficient for the infinite-stream conjecture.

## 5. Finite copy caveat and concrete period-16 factor

The planning note defines self-writing by a finite N=16 partner. Infinite
stream equality is stronger. In general a finite matching window only forces
the gcd period when N≥k+p−gcd(k,p); no such bound is stated for mixtures.

`finite_window.py` performs a separate exhaustive symbolic test for N=16 and
execution periods ≤16, without imposing the gcd period. It tests all input
and output phases. There are no solutions through k=15; at k=16 there are
five satisfiable token orders. Its first concrete witness is

```
a9 a9 a9 48 48 a9 a9 48 48 48 a9 a9 a9 48 48 48
```

It decodes as `LLPLPPPLLPP`. There are five loads and six pushes per execution
period. The chronological push word is `48 a9 a9 a9 48 48`; its reverse has
period six. Executing the unmodified tiling for 39 instructions (57 fetched
bytes, 21 pushes), then reading the newest 16 writes in ascending memory gives

```
a9 a9 48 48 48 a9 a9 a9 48 48 48 a9 a9 a9 48 48
```

This is exactly the 16-byte tape rotated by five bytes. The next written byte
is 48 and the next target byte is a9. The source has primitive period 16 and
the output period is six, so this is a finite factor match, not an infinite
periodic self-writer. Here the general Fine–Wilf threshold would be 20 bytes.

The supplied reference 6502 interpreter was also tried on this tape against a
zero 16-byte partner at budgets 39, 128 and 512, under both BRK modes: no exact
copy. At the end of the source's first pass only six partner bytes have been
written; the pointer reaches virgin zeros, and the presumed unmodified tiling
is not what executes. Consequently this witness disproves a **finite-factor
version on a prescribed periodic input**, not the actual zero-partner census
claim. `6502_finite_harness.json` records the source hash and all six replays.
It demonstrates why that claim needs an explicit bootstrap/finite-copy
argument rather than the infinite-stream proof alone. No all-period result
about the full 6502 ISA or arbitrary finite shared-memory encounters is claimed.

## Reproduction and artifacts

Use the requested interpreter. The only extra dependency is `z3-solver==5.1.0.0`,
installed outside the repository for this run. From the worktree root:

```sh
/Users/neo/repos/algocell/experiments/.venv/bin/python -m pip install --target /private/tmp/algocell-literal-channel-deps -r experiments/results/literal_channel/requirements.txt
PYTHONPATH=/private/tmp/algocell-literal-channel-deps /Users/neo/repos/algocell/experiments/.venv/bin/python experiments/results/literal_channel/search.py --minimal-cycle --prefix minimal_
/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/results/literal_channel/direct_search.py --max-period 40
/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/results/literal_channel/finite_window.py
/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/results/literal_channel/audit_census.py
PYTHONPATH=/private/tmp/algocell-literal-channel-deps /Users/neo/repos/algocell/experiments/.venv/bin/python experiments/results/literal_channel/verify.py
PYTHONPATH=/private/tmp/algocell-literal-channel-deps /Users/neo/repos/algocell/experiments/.venv/bin/python -c 'from experiments.results.literal_channel.search import pair_checks; from pathlib import Path; pair_checks(Path("experiments/results/literal_channel"))'
```

The search CSVs give every count case or directly enumerated execution period;
the JSONs give versions, timings, concrete witnesses and source provenance.
Only files under `experiments/results/literal_channel/` are part of this change.
The planning note and manuscript have not been modified.
