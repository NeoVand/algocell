# Theory: fixed points, closure and the order of events (draft 2026-10-08)

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
Consequences of the lemmas, fixed before we run anything:
(a) An **open straight-line self-writer exists** in BFF at cost 3: the tiled string `.{>` (copy head0→head1, step head1 back, step head0 forward) with head1 wrapping from 0 to byte 127, the last byte of the partner, writes the organism backwards into the partner one byte per three instructions while the instruction pointer runs forward through the organism and then into the partner. Lemma 1 applies, so **straight-line replicators should be the first to appear**. (The paper's long-tape remark that "trivial (non-looping) self-replicators rapidly take over" when the heads start at the program counter is the same object.)
(b) BFF's random code is **quiet**: only 10 of 256 byte values are instructions, so a random partner is 96% no-ops. By P4 the open replicator is barely handicapped there; prediction: the straight-line copier copies into ≥ 0.90 of random BFF partners (vs 0.5–0.8 for the Z80 pusher), so **selection for closure is weak in standard BFF** and the straight-line form should persist much longer than in the Z80 soup, or indefinitely.
(c) The looping copiers the paper reports (`[` … `]` around a copy step) are closed (the instruction pointer cycles inside the loop) and cheaper in information (≈ 5 bytes copy the whole tape); if they displace the open form in standard BFF, the theory says the reason is cost/robustness, not closure, and distinguishes the two by **instruction density**: mapping k byte values to each instruction (k = 1 … 25, i.e. 10/256 … 250/256 of bytes active) makes random code noisier without changing the language. **Prediction: the time from the first replicator to a closed dominant falls monotonically with instruction density, and the open replicator's partner-copy success falls with it.** This is the decisive test of the mechanism, and it is a one-parameter manipulation of the Google system.
(d) Measurement is direct in BFF because we own the interpreter: closure = the instruction-pointer trace stays in [0, 64) for every context (Definition 2, exact), in addition to the culture test (Definition 3).
Kill criteria: a looping first replicator in ≥ 50% of standard-BFF worlds refutes (a); no dependence of time-to-closure on density refutes (c) and with it the noise mechanism.

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
2. P1–P4 run as pre-registered experiments.
3. Stage G confirming the Z80 order of events with new seeds.
4. A second real instruction set (beyond BFF) to test universality.
