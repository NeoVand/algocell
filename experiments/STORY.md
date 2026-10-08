# The story (2026-10-08) — a Quanta-style treatment, then the audit behind it

Every number below is in a generated table (`results/*/NUMBERS*.md`, `results/closure/NUMBERS_CLOSURE.md`, `results/stageD/census/`). The piece is an exercise to find the spine of the paper, not a press release.

---

## In a Digital Primordial Soup, Life's First Invention Is a Loop

*Thousands of simulated worlds of random machine code agree on how self-copying begins: with a two-byte program that works only when its neighbours are quiet. Then evolution teaches it to stop listening.*

The smallest living thing in this universe is two bytes long. Written in the machine language of the Z80, the processor that powered the ZX Spectrum and the Game Boy, it reads: load this number into a register; push the register onto the stack. The number it loads is itself. Dropped into a crowd of random programs that take turns running one another, the pair of bytes writes copies of itself into whatever program it happens to be touching, two bytes at a time, until the neighbour is a copy. A quine that fits inside the punctuation of a text message.

It is also, by any reasonable standard, a bad organism. It reads before it writes. The processor that runs it does not stop at the end of its own code; it runs straight on into the neighbour, executing whatever bytes it finds there. If the neighbour contains random instructions, as it usually does early on, one of them will sooner or later be read as the number to push, and from that moment the little copier spreads garbage instead of itself. In a test against 256 random partners, the two-byte program made a faithful copy into about two thirds of them and wrecked its own body in more than a third of the encounters. Against partners made entirely of zeros, which the Z80 treats as "do nothing", it succeeded every time.

A new study of more than 3,400 such worlds, run on cloud graphics processors and planned in advance down to the statistical tests, argues that this bad organism is not a curiosity but the rule: the first self-replicator to appear in a soup of random programs is almost always this kind of open, environment-dependent copier, and the first thing evolution does with it is to close it.

### Order before life

The experiment follows a line of work that began in 2024, when a team at Google showed that soups of random programs in a minimal language spontaneously give rise to self-replicators, with no fitness function telling them to. The new work keeps the recipe, 20,000 programs on a grid, random pairs executed together for 128 instructions, a few hundred random byte mutations per round, and asks what the Google-style result actually depends on, by deleting parts of the instruction set one family at a time, changing the length of the programs from 3 to 100 bytes, the number of instructions an encounter is allowed, the mutation rate, and even the arithmetic of the memory the programs share.

The first thing the soups do is not live. Within fifty rounds, nearly a third of all the bytes in a fresh soup have become zeros, written there by stack instructions pushing empty registers, and a handful of byte patterns produced by subroutine-call instructions spread through the grid without ever copying the code that makes them. The compression test for life that the 2024 paper used, which looks for soups that have become more orderly than chance, lights up at this point. Nothing is alive. In worlds of longer programs the test later ran backwards: the sterile flood compresses better than the living soup that replaces it, so the detector scored life as death. Only a direct test, lifting a program out of the soup, placing it with fresh random partners and checking whether its copies also copy, separated the two. The authors call this the culture test, and they treat the state of the soup, however orderly, as inadmissible evidence of life.

### Three load-bearing parts

Out of that flood, in every world whose programs were at least 16 bytes long, the two-byte copier appeared within a few hundred rounds. Deleting most of the Z80's instruction families changed nothing: the register loads, the memory loads, the bit operations, the subroutine calls, the stack reads, the block-copy instruction, each could go. Three deletions mattered. Remove the instructions that load a literal number and life came in two of ten worlds instead of ten. Remove the instructions that write through the stack pointer and it still came, but a hundred to a thousand times later, carried by a different, slower mechanism built on the Z80's block-copy instruction. Remove every instruction that can write to memory by stepping along it and nothing came, in forty worlds run for a million rounds each.

The deletions also produced the study's strangest result. For nine-byte programs, a size at which the two-byte copier cannot establish itself, removing the stack-writing instructions made life more likely, not less: 19 of 20 worlds instead of 5 of 20 at one setting. The culprits were the two instructions that write return addresses. Their sterile patterns occupy the soup and destroy five times more programs per encounter than the soup does without them, and the census of every interaction shows what that does: in the full machine, a block-copy program reached even half a percent of the soup in 5 of 20 worlds, after a median of 118,000 rounds; without the stack writers, in 20 of 20, within 3,400. The sterile order does not kill the nascent copiers. It stops them from ever assembling. Prebiotic chemists have a word for the problem of useful molecules being swamped by sterile byproducts of the same chemistry. They call it tar.

### Closing the loop

The headline is what happens after the two-byte copier wins. For thousands of rounds the soup is full of its pattern and almost empty of heredity: when the authors lifted random cells out and cultured them, only seven to fourteen percent passed. Then, in world after world, a descendant takes over that carries exactly one more kind of instruction. At 16 bytes it is a conditional return, which uses the two bytes the program has just written as the address to jump back to; its own body doubles as its own entry point. At 36 bytes it is a decrement-and-loop. At 50 and 100 bytes, a conditional jump. In each case the effect is the same: the processor's program counter never leaves the organism. Cultured against 256 random partners, every one of these successors copied into all of them and damaged itself in none. Of 233 worlds in which both the first and the final replicator could be recovered, 92 had gained a control-flow instruction between the two and 3 had lost one. The heritable fraction of the population rose to nearly eighty percent as the loop took over. And the successors were not merely similar across worlds: at 16 bytes, 18 of 20 independent worlds, starting from different random soups, ended with the same eight-byte genome to the byte; at 100 bytes, 17 of 18.

The authors are careful about what they are not claiming. The first copier is heritable by their own test, barely. What the loop adds is not more copying but independence: the same throughput, into any partner, without reading it. In the vocabulary of origin-of-life theory this is closure, the moment an organism's operations stop depending on what surrounds it. Here it is a measurable number, the fraction of random partners a program copies into, and it goes from two thirds to one.

### Budgets and thresholds

Two further results round out the picture. When an encounter is too short to copy a whole genome, 64 instructions for a 100-byte program, the soups became heritable without ever producing a clone: in six of ten worlds, 69 to 100 percent of cultured cells passed the test while no genotype held more than seven hundredths of a percent of the soup. Each encounter moved a chunk; the population carried the heredity. At 128 instructions, lineages appeared. And when the slower, block-copy regime was pushed across a hundredfold range of mutation rates, the length of the repeated unit it could maintain fell from 32 bytes to 5, the curve that the chemist Manfred Eigen predicted for molecular replicators half a century ago, drawn here by programs that no one designed.

Whether any of this transfers from a 1970s microprocessor to chemistry is the obvious question, and the authors do not pretend to answer it. Their machine is one machine. But the mechanism behind the loop does not seem to care what the machine is. Any replicator that works by writing itself into a neighbour that will also be executed is open to that neighbour until something closes it. If that is right, then the first replicators, wherever they arise, are sloppy, environment-dependent and only partly heritable, their order is indistinguishable from the sterile order around them, and the step that makes them worth calling alive is the one that makes them stop listening.

---

## The audit: what this story rests on

**Spine (in order of the piece), with evidence class.** A = pre-registered and replicated with independent seeds; B = post hoc but measured on every seed with a mechanism check; C = suggestive.

1. **Order before life.** The flood (0.05–0.07 zeros after one round, 0.30–0.33 by round 50, in every arm; `stageD/NUMBERS`), the smears, `tq_10` firing on the flood (20/20 at L = 9 with 6/20 replicators), high-order entropy anti-correlated with life under `none` at L ≥ 25 (AUC 0.07–0.34) while 0.97–1.00 at L = 16 (`results/detectors`). **A** for the facts, **B** for the detector verdict (the ground truth is our own assay).
2. **The open two-byte copier.** First replicator in 10/10 worlds at every L ≥ 16 (A, B, C, E); modal first tape `01 c5` or a register variant at every L ≥ 8 (`results/closure`); partner-copy 0.54–0.82 and self-damage 0.02–0.89 against 256 random partners; 1.00 against blank partners; population heritability 0.07–0.14 for thousands of rounds (C4). **A** for emergence, **B** for the openness measurements (executor experiments, deterministic).
3. **Closure.** 92 gained / 3 lost control flow, McNemar p = 7 × 10⁻²⁴ over 233 worlds from three stages; every control-flow successor 1.00 / 0.00 against random partners; byte-identical convergence 18/20 (L = 16), 17/18 (L = 100), 7/10 (L = 36, Stage C); heritable fraction 0.78 at 50k, 0.88 at 300k (C4). **B**: post hoc, strong, mechanistic, but not yet pre-registered. This is the headline and it deserves a confirmatory run (below). Caveat to state: at L = 20, 25, 32, 49, 64, 81 the pusher still reigns at 300k rounds with no closure; the transition is in progress, not universal within the horizon.
4. **Three load-bearing parts, the replacement principle, the nulls.** **A** (C3, C1, D).
5. **Less is more and its mechanism.** D1 p = 3.9 × 10⁻⁶ (**A**); CALL/RST attribution p = 4.5 × 10⁻⁴ (**B**); assembly-prevention from the census and the nascent-copier scan (**B**).
6. **Budget threshold and heredity without clones.** E4 threshold **A**; the sub-clonal regime **B** (post hoc reading of a pre-registered cell; 6/10 soups; 0 of 675 sampled written pairs at 300k received a whole genome).
7. **Error-threshold curve.** E5 **A** (as a monotone trend; not fitted to Eigen's formula).
8. **Convergence across worlds; same-seed divergence.** **A** (E6) and **B** (closure table).
9. **Tiling follows the pair length, not the memory ring.** F1 falsified as pre-registered. **A**. Supplement material, but it is the honest counterweight to the ring story and shows the pre-registration working.

**What I would cut from the main text.** The fine atlas rows (supplement table), the ring sweep and F2/F3 details (supplement), the detector tables beyond one figure, the variance analysis (methods), the dead-zone resonances (one paragraph plus supplement).

**What we must not claim.** That this is chemistry; that closure is universal (it is in progress in half the worlds at 300k); that the first copier is "not alive" (it passes our own test); that high-order entropy is useless (it is excellent at L = 16 against the heritable fraction).

**What would make the headline indisputable.** A pre-registered confirmation of claim 3 with new seeds: at L = 16 and L = 50, 20 seeds each, predictions (i) the first replicator has no control flow in ≥ 90% of worlds, (ii) by 300k the faithful dominant has control flow in ≥ 80% and copies ≥ 95% of random partners, (iii) the heritable fraction is < 0.2 at 5k and > 0.7 at 100k; plus 1M-round runs at L = 20 and 64 (20 seeds) predicting that closure arrives later in ≥ 50% of worlds. About 4 GPU-hours, roughly $10. Second, for generality beyond one machine: the mechanism predicts that any open replicator is partner-dependent; a second instruction set would test it, and that is the extension that would justify a Nature or Science main-journal attempt. Without it, the realistic ambitious targets are PNAS, Nature Communications or Science Advances, with the ALife venues as a fallback that would under-sell the work.

**Paper spine.** Title: *Life closes the loop: spontaneous self-replicators begin open and evolve independence from their surroundings.* Abstract claims, in order: three load-bearing primitives and a replacement principle; sterile order that precedes life, fools compression biosignatures and can prevent life from assembling; the open two-byte replicator; closure by control flow with byte-identical convergence; budget threshold and heredity without clones; error threshold. Figures: (1) the soup, the flood, the two-byte quine; (2) the atlas forest plot; (3) the L = 9 reversal and the census; (4) openness vs closure (`closure_partner_independence` + the RET NZ trace + C4 curves); (5) size and budget (E1/E2/E4); (6) mutation (C2/E5); (7) detectors vs the culture test. Methods: the instrument, the assay, pre-registration, the GPU non-determinism statement.

---

## v2 plan (2026-10-08 night): one idea, escalating implications

The piece above tours the results. The new version is organised around one idea: **the first living thing was not an
individual**. It was a word that wrote itself into its neighbours, lived off them, and later learned to keep to itself;
we can now say in bits how much of it was its neighbourhood (a median 7 of 8 bits of its offspring are the partner's) and
watch that number fall to zero when a single jump arrives. Structure: (1) the wave on the lattice (stills); (2) the word
that writes itself, and base pairing as the same kind of channel; (3) the catch: seven bits of neighbour, a colony, cheap
among kin, costly at the frontier; (4) the loop, RET NZ's product doubling as its control (the RNA-world property, found
at 16 bytes), zero bits, convergence; (5) why: write less than you read and you must leave or revisit, and the BFF
flooding organism as the proof of the boundary (write two for one, overwrite everything, never need a loop); (6) the order
of events as a prediction for chemistry, and life beginning at minimum complexity (assembly index equal to the tar's),
so that compression and assembly both fire on the dead; (7) the window: tar lethality and the three fates; (8) what it is
not, what would test it (a well-mixed Z80 soup; by-product screens in template chemistry), and the definition of life
that falls out: not complexity, not a boundary, but a measurable independence.

Dots, with evidence class (A measured on every seed; B measured, post hoc; C hypothesis with a stated test):
- Template copying is a literal channel: the template is instruction and data at once, like `LD BC,nn ; PUSH BC`; the
  theorem then says the smallest replicators arise there and are open. (C, by analogy; the theorem is A.)
- The RNA-world property in silicon: RET NZ's written bytes are its return addresses; product = control. (A)
- Cycle or saturation: Theorem 2 needs write ratio < 1. BFF's wrapping literal pusher writes two bytes per cell executed,
  floods the ring, has 0.04 bits of inflow with an open pointer: closed without a loop, as the theorem allows. Template
  copying has ratio 1, so chemistry may close by exhaustive copying; overhead (instructions longer than outputs) forces
  cycles. (A for the machines; C for chemistry.)
- Life begins at minimum complexity: the first replicator's assembly index is the lowest any string of its length can
  have and equals the tar's; complexity thresholds describe evolved life. Detection at the origin needs intervention. (A)
- Individuality in bits: 7 → 0 across the transition; the Darwinian threshold as inflow → 0; the sub-clonal regime
  (heredity without clones) as the all-horizontal extreme, measurable the same way. (A for the drop; C for the reading.)
- Space and kin: pairs are lattice neighbours, so an open replicator's partners are mostly its own copies; openness is
  cheap in a kin neighbourhood and costly at fronts and tar pockets. Prediction: in a well-mixed Z80 soup the open phase
  is shorter or absent and closure is selected faster, or life fails to establish. One-line shader change. (C)
- Tar decides what can begin and how the open phase ends: inert versus inhibitory by-products. (A for the machines; C.)
