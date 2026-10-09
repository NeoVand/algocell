# Response to the reviews: Heredity before individuality in a digital primordial soup (revision 1)

*To the reviewers. Thank you for two careful reviews. We checked every factual point against the primary sources and our own data before acting; most were right, several were decisive, and they led to eight new pre-registered experiments. Below, each point is followed by what we did and where it now sits in the manuscript. Points we could not address are listed at the end, as are errors we found ourselves.*

## What is new in this revision

1. **Pointer confinement measured, not inferred.** A traced copy of the executor records every fetched address. Every traced first replicator under benign tar enters the partner in all 256 encounters (100 of 100); every loop-bearing final dominant never does (89 of 89); confinement and zero information inflow agree in 106 of 110 finals (main text, Closure; Extended Data Fig. 4).
2. **The matched-genotype tests you asked for.** A single-byte mutational scan of all 160 first and final tapes, and a head-to-head invasion of the matched L = 50 pair (pusher against pusher with its jump), both pre-registered. The closed form wins in about fifty encounters per cell, but it is not more robust to mutation, so robustness does not explain the succession, and it transmits almost no variation (Fig. 4; Extended Data Fig. 6).
3. **A second real instruction set.** The Z80 restricted to its Intel 8080 subset: the load–push word comes first in every world; at L = 16 the same return design closes 20 of 20 worlds; at L = 32, where the subset has no closer, none of 20 closes in a million steps (Extended Data Fig. 5).
4. **Closure without the address wrap.** At the aligned length L = 32, where no relative jump can use the 16-bit wrap, 12 of 20 worlds close by one million steps, all by block copy (Stage K).
5. **Ten million steps.** Twenty worlds at L = 16 and 20: all close, and the capacity for inherited variation does not recover; the pre-registered kill criterion for a recovery fired, and we report it (Fig. 4d).
6. **The unrun cell of the map.** BFF as published with harmless brackets: 7 of 12 soups produce life, every first replicator carries a loop, and four lose it again (Fig. 5e).
7. **Two dials in the second machine.** Tar lethality behaves as a threshold between p = 0.1 and 0.3, not as the registered 1/p dial (a failed prediction, reported); raising the literal's write ratio to two or three copies gives a first replicator that is information-closed from birth without a loop, the boundary case of Theorem 2 (Fig. 5c,d).

## Point by point

**Prior work and novelty.** You were right: the load–push first replicator, the block-copy takeover and serial-transfer detection were already described (refs 16, 22, 23). The novelty sentence is gone; the introduction credits all three and states the contribution as the open/closed characterisation measured by intervention, its mechanism, the theorems, and the experiments that test it.

**The entropy headline.** "Seven of eight bits" and "most of what a child looks like" are gone. The text now gives the exact-copy fraction of the first replicator (a median 58%, 32%, 5% and almost never at L = 16, 20, 50 and 64), states that every position varies across partners, and reports H(o | x) as a plug-in entropy over 256 partners with its 8-bit ceiling.

**Three kinds of closure conflated.** Loop (syntax), pointer confinement (mechanism) and context independence of the offspring (outcome) are now named, measured and reported separately; their agreement and their four exceptions are given (Closure; Extended Data Fig. 4). In BFF the three come apart, and the text says where.

**Descent or displacement.** At L = 16 the return closer shares no bytes with the pusher, so the text says succession; at L = 50 the closer is the pusher with a jump inserted, and the pair invasion shows the jump selected among kin. A further check we added: at L = 50 the closer writes its parent with one more jump inserted, the same string in every encounter, and the plain pusher remains the most common exact tape at 6% while jump-carrying tapes hold 70–76% of the cells; the definition of "final dominant" is now in Methods.

**The 16-bit address wrap.** Confirmed and quantified: 25 of the 67 closers are relative jumps whose target depends on the wrap (all 20 at L = 50, 5 at L = 20). The convention is stated in the main text and Methods, the L = 50 results are labelled as results about the machine so defined, and closure is stated at the aligned lengths, where it is slower and monotone in L.

**The ninety-fold delay.** It depended on the dating rule, as you said. Both datings are now reported: the first clone arrives 84-fold later under lethal tar (Kaplan–Meier medians 38,000 against 450 steps), heritable material 32-fold later (40,000 against 1,250); Extended Data Fig. 8 shows the two worlds in which they differ most.

**Heritable-fraction plateaus, the size axis, the BFF variants.** "Rises to 0.75–0.94" is now stated for L = 16 only, with about one half at L = 50. "No floor" is gone; the pusher-first claim is restricted to tapes of 16 to 64 bytes, below which block copiers come first. The literal push with and without a wrapping pointer are no longer pooled.

**Theorems.** Theorem 1 is quantified over partners holding no byte of the copy in place. Theorem 2 states its two provisos, and the main text now uses the one that holds for the Z80 (the execution budget outlasts a pass over the organism); Fig. 6 is redrawn for that argument and for Proposition 3. Theorem 6(iii) concludes "loop-bearing", not "closed". Finite searches are stated as finite ("none found up to period 10").

**Model 5 fits any outcome.** Its ingredients are now measured: an open population at carrying capacity closes by mutation within 5,000–13,250 steps against of order 10⁵ from a random start, and the lethality dial locates the collapse between p = 0.1 and 0.3. We say plainly that the model does not predict which regime a machine is in.

**The variation channel.** Your best suggestion, and the most consequential result of the revision: the first replicator transmits mutations at a median 23 of 50 positions, all of them executed operands; the closers transmit almost none, and the few they do are mostly bytes they copy but never execute (82 of 130 sites). Genomes carrying such bytes arise in 3 of 20 ten-million-step worlds and lose to smaller closers. The first individuals are canalised.

**Statements of scope and convention.** Units of fixed size, start address and pairing are imposed by the harness and are stated as limits; changing zero to a halt is described as a change of the machine's semantics. The literal-push instruction in BFF was designed to write itself, and the text says so; what follows it was not designed.

**Presentation.** References renumbered by first citation (36; uncited entries removed); Extended Data reduced to eight figures and two tables, all cited and all drawn from generated tables; Data and Code availability statements added; Kaplan–Meier medians used throughout.

## What we did not do

- **A robustness stage for the conventions** (random initial registers and stack pointer, start address, pair order) has not been run.
- **Log-rank tests, bootstrap intervals and the grid of culture-test thresholds** are not yet in the text; effects are reported as Kaplan–Meier medians with seed-paired sign tests and exact tests.
- **A third-party time stamp** of the pre-registrations (OSF or Zenodo) is pending; the registrations and their dated change log are in the repository.
- **Chemistry** remains a set of predictions, stated as such.

## Errors we found ourselves

An independent audit of every number in the main text against the generated tables, run before resubmission, corrected these, all now fixed: the traced count of first replicators that enter the partner (100 of 100 under benign tar; the ten born under lethal tar are confined), an inverted sentence on inflow, a detector claim broader than its table, a "nineteen-fold" that was a timing ratio rather than a rise, emergence medians that mixed plain and Kaplan–Meier statistics, "about forty" encounters per cell (about fifty), and "all by block copy" at L = 16, where most closers are returns.
