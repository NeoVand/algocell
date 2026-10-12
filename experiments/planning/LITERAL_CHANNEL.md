# The literal channel: why the Z80 and the 8086 have a pusher and the 6502 cannot (draft, 2026-10-11)

Status: Proposition 9 below is proved. Conjecture 9′ is not; it is checked only for periods ≤ 4 by the exhaustive
census. The plan at the end says how this goes into the paper.

## Setting

A **straight-line self-writer** of period k is a word T of k bytes with two properties:

- the tiling T^∞, executed sequentially from position 0, never transfers control;
- it writes into the partner an exact copy of T^∞ at some cyclic shift.

This is the census's definition: "exact into the zero partner", L = 16.

A **literal channel** of width w is a pair of instructions, both acting on the same w-byte register R:

- a *load-literal* with a one-byte opcode `a`, followed by w literal bytes that become the contents of R;
- a *push* with a one-byte opcode `b ≠ a`, which writes the w bytes of R to the w addresses just below the stack
  pointer, in a fixed byte layout π, and lowers the stack pointer by w.

The pair has length ℓ = w + 2. Examples:

| Machine | Load-literal | Push | w |
|---|---|---|---|
| Z80 | `LD rr,nn` (01, 11, 21) | `PUSH rr` (c5, d5, e5) | 2 |
| 8086 | `MOV r16,imm16` (b8–bf) | `PUSH r16` (50–57) | 2 |
| 6502 | `LDA #n` (a9) | `PHA` (48) | 1 |

The 6502 has no other literal channel. `LDX #` and `LDY #` load registers that no instruction can push, `PHP` pushes
the status register rather than a literal, and the 6502 has no 16-bit immediate load.

**Fine and Wilf's theorem.** A word of length at least p + q − gcd(p, q) that has periods p and q also has period
gcd(p, q).

## Proposition 9 (literal-channel parity)

Let the self-writer be the repeated pair, T^∞ = (a L b)^∞ with |L| = w. Suppose the copy it writes is at least
ℓ + w − 2 bytes long. (At L = 16 this holds whenever w ≤ 7.) Then:

- (i) If w is **odd**, no such self-writer exists, at any period.
- (ii) If w is **even**, one exists only with T^∞ = (a b)^∞, L = (b a)^{w/2}, and a layout π that writes L as an
  alternating word. For w = 2 every layout qualifies, so `a b` is a period-2 self-writer whenever the channel exists.

**Proof.**
1. The executed stream E = (a L b)^∞ has period ℓ.
2. Every push writes the same block π(L), one block lower each time, so the written region read in ascending memory
   is (π(L))^∞, which has period w.
3. Self-writing means this region equals a shift of E over the copied window. That window therefore has periods ℓ and
   w, and by Fine–Wilf it has period g = gcd(ℓ, w) = gcd(w + 2, w) = gcd(2, w).
4. **w odd:** g = 1, so the window is constant, and in particular a = b. This contradicts a ≠ b.
5. **w even:** g = 2, so E = (u v)^∞.
   - Position 0 of the pair is a, so u = a.
   - Position w + 1 is b, and it is odd, so v = b.
   - The literal fills positions 1 … w, which alternate starting at an odd position, so L = (b a)^{w/2}.
   - The written block π(L) must itself be a factor of (a b)^∞, so it alternates.

   For w = 2 both orders of a 2-byte block alternate. ∎

**Witnesses for (ii), from the census.**
- Z80: `01 c5`. Here `LD BC,$01C5` is followed by `PUSH BC`, which writes c5 01. Also `21 e5`, which became the
  dominant tape in a Z80 soup during the ZSoup lockstep test. These are among the census's 5 period-2 self-writers.
- 8086: `B9 51` (`MOV CX,$51B9`; `PUSH CX`), in the 8086 partial census.

**Consequence for the 6502.** Its only literal channel has w = 1, so (i) rules out the pusher mechanism at every
period, not just at k ≤ 4. On the Z80 and the 8086 the channel has w = 2, so a 2-byte open self-writer is
guaranteed. The width of the literal channel decides whether a machine can begin with an open, pattern-like
replicator.

## Conjecture 9′ (any mixture, width 1)

**Statement.** No straight-line self-writer of any period executes only 1-byte literal loads and pushes, in any
order and with any literal values.

**Why it should hold.** With w = 1, every written byte is an isolated copy of one literal byte. The copy keeps no
adjacency from the tiling, but the tiling must contain opcode–operand adjacencies, because every load opcode is
followed by its literal. A tiling would have to equal a stream sampled from itself at expansion
P_e / n_P = 1 + 2 n_L / n_P > 1, where P_e is the length of the execution period and n_L and n_P count the loads
and pushes in it. For the single repeated pair this self-similarity forces the non-periodic Toeplitz word
x_{3i+1} = a, x_{3i+3} = b, x_{3i+2} = x_{i+1}. Hence the impossibility.

**Status.** Not proved for general mixtures. It is verified for periods ≤ 4 by the census, which checks every byte
word, so it covers every mechanism at those periods.

## What stays outside the propositions (covered only by the census, periods ≤ 4)

Straight-line writes that are not literal pushes:

- memory-to-register-to-memory copies through indexed loads, such as the census's open stack copiers
  `TSX; LDA zz,X; PHA`, which copy exactly into about 1 partner in 1,000;
- `JSR`, which pushes a return address (a value of the program counter, not a literal; the census's "75%
  smears");
- read-modify-write instructions, and the undocumented stores.

None gives an exact straight-line self-writer at k ≤ 4.

## What this does not predict

- The arrival rate (only "later than the Z80").
- Which loops appear, for example the ring-wrapping forward branch.
- The class mix.

## Plan for the paper (to do in the next session)

1. **Methods, "What is proved".** Add Proposition 9 with the proof above, about half a page, after Proposition 8.
   State Conjecture 9′ as a conjecture, with the census as its finite check.
2. **Main text, new section on the second machine (S1).** One paragraph, before the 6502 results:
   - why the 6502 should begin closed: its literal channel is one byte wide (Proposition 9);
   - the census confirms there is no straight-line self-writer of any kind up to period 4;
   - the prediction was registered before any 6502 soup.
3. **Discussion.** The width of the literal channel is a property of an instruction set that predicts how replication
   begins: Z80 and 8086 open (w = 2), 6502 closed (w = 1). The 8086 is the natural third test: there the parity
   predicts an open beginning. It would be registered before running, and its write-ratio fork is noted in the ISA
   research.
4. **Extended Data.** A table of literal channels and period-2 census results for the Z80, 8086 and 6502.
5. **Before submission:**
   - Run an exhaustive symbolic search of the load–push token grammar up to period 64, on the CPU and cheaply, either
     to prove Conjecture 9′ or to find a counterexample.
   - Have the proof of Proposition 9 checked by an independent reviewer, and against the census word lists for the Z80
     and 8086, period 2.

## Correction after the independent check (2026-10-11 night; `results/literal_channel/CHECK.md`, merged)

- **Proposition 9's proof holds** with the copy-length bound corrected: 2w + 1 for odd w, 2w for even w. It is best
  read as an independent characterisation of Theorem 6's (H3) for a single repeated pair.
- **Parity is the wrong general line.**
  - Width-3 *mixtures* do self-write: `a [b b a]; b; b` writes a shift of `(a b b)^∞`.
  - Width 1 is the special case. For any mixture of width-1 loads and pushes sharing one register, no infinite
    periodic self-writer exists. This is **proved**, and an exact symbolic search through k = 64 plus enumeration
    through k = 40 (165,580,120 token orders) agree.
  - A finite coincidence exists in the abstract model (a period-16 mixture matching exactly 16 bytes, then diverging).
    It fails in the real 6502 harness, so the finite statement must carry the copy-length condition.
- **The substrate invariant is therefore the literal-channel *width*.** A one-byte channel cannot write its own
  instruction stream as literals; channels of two or more bytes can. The Z80 and 8086 (width 2) admit a pusher; the
  6502 (width 1) does not. "Parity" survives only as the statement about one repeated pair.
