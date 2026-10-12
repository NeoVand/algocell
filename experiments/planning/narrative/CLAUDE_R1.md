# Claude, Round 1 (written independently of Astra's draft) — 2026-10-11

## 1. What we actually saw, in one paragraph

Random machine code, paired with lattice neighbours and mutated, makes replicators. On the Z80 the first ones copy a
*pattern*, not an individual.

- **The pusher.** `01 c5` (LD BC,nn; PUSH BC) is straight-line code. Its pointer runs out of its own bytes into the
  partner, and it writes its tiling wherever it goes.
- **It carries no heritable variation.** X1: lineages persist under branching, but 0 of 512 mutants keep their
  allele.
- **It is permeable.** X2: 3.8 bits of in-soup inflow, and the pointer enters the partner in 80 of 80 encounters.
- **Individuals arrive later, by construction.** Closed programs whose execution never leaves their own bytes appear
  only afterwards. Along every line of descent the first copier is open (37 of 37). Closed founders are written by
  other programs (rewrite), not reached by point mutation (N3, N6).
- **Closed programs carry genomes.** A closed copier's copied-but-unexecuted bytes are exactly the heritable sites:
  d − 4 on the Z80 (Proposition 8). The environment chooses between regenerators and transmitters (the copy-offset
  switch, lethal tar).
- **The second machine.** On the 6502, chosen and predicted in advance, the order is different: replication begins
  closed in every world, with no open phase, about 2,500 times later, and heritable classes exist from the start.

## 2. The explanatory core (my thesis)

**In a substrate where code is data, every byte of a replicator is either *used* (executed) or *transcribed*
(copied without being executed). Heredity lives only in the transcribed bytes. Whether a substrate's first
replicators can exist without that division of labour is fixed by one property of its copying chemistry: whether it
can write its own instructions back as literals in a self-consistent tiling.**

Four consequences follow, each provable or proved:

1. **Open first, where a self-consistent literal tiling exists.** Theorem 6 and Proposition 3 cover the Z80.
   Proposition 9 (literal-channel parity, proved) says a repeated load-and-push pair self-writes only if its literal
   is an even number of bytes wide. The Z80 and the 8086 have width 2; the 6502 has width 1. Conjecture 9′, for any
   mixture of width-1 loads and pushes, is being checked by exhaustive symbolic search.
2. **No heredity without transcription.**
   - Lemma 11 (new, trivial but central): open straight-line code fetches every byte of the organism.
   - Lemma 10 (new, a non-interference argument): an unexecuted byte that is only ever copied is heritable for every
     value.
   - So the open replicators replicate patterns without genomes, and X1's 0 of 512 becomes a theorem. The
     computer-assisted part is the executed sites of the pusher: 16 × 255 single mutants, exhaustively.
3. **Individuality and heredity are the same event.**
   - Theorem 2: confinement forces control flow to loop or halt, and that control flow is what separates using from
     transcribing.
   - Lemma 7: perfect fidelity requires zero inflow, so a faithful replicator must be informationally closed.
   - Proposition 12 (new, generalising Proposition 8): a copier with a c-byte core inside a copy period d carries
     exactly d − c inert sites.
   - Closure, zero inflow and the appearance of a genome are three views of one transition.
4. **The order is a substrate property, so it can vary between machines, and it did.**
   - Even literal channel (Z80, BFF with its literal `P`): patterns first, then individuals.
   - Odd channel (6502): individuals with genomes from the first replicator.
   - We predicted the 6502 outcome from the census before running it, and it held. P1's loop criterion had a blind
     spot on a ring; it is stated honestly, with the exploratory 20 of 20 loops.

**Why this is a good explanation (Deutsch).**
- **Hard to vary.** Change the literal width and the predicted order flips. Change "executed versus transcribed" and
  the heredity counts stop matching Proposition 8 and X1.
- **Reach.** It predicts the 8086 begins open (width 2; its census already lists `B9 51`). It predicts heritable
  variation on the 6502 from the start, which is confirmed. It predicts that any designed von Neumann replicator needs
  an uninterpreted description, which is von Neumann's own argument, now *derived* for undesigned soups.
- **New questions.** What plays the literal channel in chemistry? Is there a phase diagram, with the open-pattern
  phase against the closed-individual phase, and inflow as an order parameter?

**Rivals our data can address.**
- (a) "Closure is just better replication." It does not explain why heredity appears *with* closure, or why the 6502
  skips the open phase.
- (b) "The pusher is a Z80 accident." Proposition 9 says it is generic to width-2 literal channels, and BFF's `P`
  shows an open start too.
- (c) "Selection for fidelity drives closure." Partly, through Model 5's window and the lethality dial. But founders
  arise by rewrite (constructed by other programs), not by gradual selection of point mutants.

## 3. The biology hook: a Darwinian threshold, generated rather than assumed

Woese argued that early life was communal, with rampant horizontal transfer and no stable individual lineages. At a
"Darwinian threshold", organisms became integrated enough for vertical inheritance to dominate (Woese 2002;
Vetsigian, Woese & Goldenfeld 2006).

Our open phase maps onto the communal side. Copying there is pattern-level, permeable (inflow) and horizontal: writers
copy their code into others, and founders are built by others. It carries no individual heredity. Closure maps onto
the threshold: zero inflow, vertical inheritance, a genome.

In our system the threshold is not assumed; it emerges. Whether a substrate has a pre-Darwinian phase at all is set by
its copying chemistry (the literal channel). That gives the paper its line: **a Darwinian threshold, generated
spontaneously, and predictable from the substrate.** It must be said as an analogy with a mechanism, not as a claim
about Earth's history.

## 4. Paper architecture I would propose

**Working title:** "Replicators before individuals: a Darwinian threshold in digital primordial soups" (or "Life
begins as patterns in self-organizing machine code").

**Main text (about 3,500 words), five figures:**

1. **The instrument and the event.**
   - A soup of real-ISA programs on a validated substrate: Zilion, one sentence on the validation.
   - Spontaneous replication.
   - The first replicator is open: its pointer runs into neighbours.
2. **Patterns, not genomes.**
   - Executed versus transcribed bytes.
   - X1: lineages persist, but no allele is kept.
   - X2: inflow.
   - Lemmas 10 and 11 as a schematic.
3. **The threshold.**
   - Closure, zero inflow and the appearance of heritable sites.
   - Founders built by rewrite (N3/N6, open-first along every line).
   - The copy-period law (d − c), as the genome's size set by one byte.
4. **Why the order is a property of the substrate.**
   - Literal-channel parity, as a schematic.
   - The census for Z80, 6502 and 8086.
   - The prediction.
5. **The test on an undesigned second machine.**
   - The 6502 begins closed (20 of 20), about 2,500 times later, with heritable classes from the start.
   - The registered verdicts as stated, P1's criterion included.

**Extended Data and Supplement.**
- The ablation atlas, as a phase-diagram figure.
- The lethality dial and the closure window (Model 5).
- The BFF arms, including the literal channel in BFF.
- The emulator validation and impact study (with flags named).
- The census tables.
- The full theorem proofs: main text holds the statements in boxes, Methods holds sketches, SI holds the proofs.
- The statistics.

**What leaves the main text:** stage-by-stage history, most ablation detail, the BFF write-ratio detail, every
correction narrative. The registered record stays fully available in the SI and in a public registry file.

**Zilion in the paper:** a Methods subsection and one main-text sentence. A separate tool paper (JOSS or the ALife
journal) with datasets, agent skills and browser reproduction carries the community story.

## 5. Honest assessment

- **Methodology: strong for the field, not novel enough to be the headline.**
  - Real instruction sets at GPU scale.
  - Per-instruction validation against hardware-derived vectors.
  - An independent oracle.
  - Bit-identical pipelines.
  - Registered predictions on a second machine.

  This is unusually rigorous for ALife, and it is what licenses strong claims.
- **Math: elementary but explanatory.** Combinatorics on words, non-interference, counting. Its value is reach: it
  predicts across machines. Depth would come from a general theorem connecting substrate properties to a phase
  structure (open against closed). That is the main thing I would *try* to prove before submission. I would not try to
  add more lemmas.
- **Surprise: high for "replication without heredity first" and for the substrate-predicted order.** It contradicts the
  default picture that replicators arrive as individuals with genomes (Tierra/Avida assume it; BFF reported
  replicators without this distinction). The 6502 test makes it a prediction, not a story.
- **Biggest risks.**
  - Overreach on biology.
  - A reader asking "what is the literal channel in chemistry?"
  - The P1 miss, which is defensible if stated plainly.
  - The impact-study flags. They touch the Extended Data, not the headline.

## 6. Theory programme, by explanatory reach (not by count)

1. **Order theorem (main).** Lemmas 10–11, Proposition 12 and Corollary 13: a substrate-level statement of "patterns
   before genomes", with the exhaustive pusher-mutant check.
2. **Literal-channel parity.** Proposition 9 is done. Conjecture 9′ is being searched. Then generalise beyond
   load/push to "self-consistent straight-line writers", ideally as a single combinatorics-on-words theorem.
3. **Closure as a phase transition (model).** The open-pattern phase against the closed-individual phase. Order
   parameter: inflow or confinement. Control parameters: lethality, mutation, length. The ablation atlas and the dial
   data as the phase diagram. A mean-field model from Model 5, to be tested against the data we already have.
4. **Heredity capacity.** Heritable bits = 8 × inert sites. Lemma 7 links fidelity, inflow and closure.
5. **Arrival-time law (model).** Order of magnitude for the Z80 from Proposition 4 counts. For the 6502, a pathway
   through near-copiers.
6. **An abstract frame** (fetch-execute automata on a shared tape) in which all of the above hold. This gives the
   bridge to Turing machines and automata, and an honest paragraph on chemistry.

## 7. Questions for the user

- Are we comfortable leading with the Darwinian-threshold framing, stated as an analogy with a mechanism?
- Do we run the 8086 (width 2: an open start predicted) as a third registered machine before submission? It is the
  strongest single addition to *reach*.
- Should Zilion get its own tool paper, so this paper stays about the science?
