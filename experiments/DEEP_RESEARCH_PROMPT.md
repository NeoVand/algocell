You are a senior research analyst helping the authors of an artificial-life paper that is under revision at Nature (second round of review). Do a deep, critical literature review. The goal is to find what we have missed, what threatens our novelty, what reviewers will attack, which theory we can borrow to turn observations into theorems, and which known results or analogies would sharpen two experiments we run tonight. Be sceptical and concrete. A short list of papers that really matter, read closely, is worth more than a long list of loosely related ones.

# 1. The paper in brief

**Working title:** "Heredity can precede individuality in a digital primordial soup". The title is under revision.

**Substrate:**
- A lattice of 160 × 125 cells. Each cell holds a tape of L bytes of real Z80 machine code (L = 16–100).
- At each step, neighbouring cells on a square lattice with 4 neighbours are paired at random.
- In each encounter, the two tapes are joined into one 2L-byte memory. A real Z80 core runs 128 instructions, starting at the first byte of tape A, with registers zeroed. Both halves are then written back.
- Random point mutations happen at a low rate.
- There are no fitness functions; the dynamics alone produce self-copiers.
- **Heredity** is scored by a "culture test": copy into random partners, then copy the copies. A second-generation score of at least 0.3 counts as heritable.
- **By-products ("tar")** are mostly zero bytes. Under *benign tar*, 0x00 is a NOP. Under *lethal tar*, fetching 0x00 as an opcode halts the pair for the rest of the encounter, as an unmatched bracket halts a program in BFF. A *lethality dial* makes a zero halt with probability p.
- **Comparisons:**
  - an 8080-like subset of the Z80, used as an ablation;
  - BFF, the Brainfuck-like soup of Agüera y Arcas et al. (2024, "Computational life"), with variants that add a wrapping pointer and a "literal push" instruction.

**Main claims and results.** We refer to these by ID.
- **C1, the first replicator is open.** In benign-tar worlds, the first replicators are "pushers" (for example `01 c5` tiled) whose instruction pointer always runs into the partner's half and writes there. All 100 benign-tar first replicators enter the partner in every traced encounter. They copy only with the help of a neighbour, so they are not self-contained individuals.
- **C2, closure follows.** Later, "closed" replicators appear and take over. A closed replicator's pointer never leaves its own L bytes: there are loop-bearing return closers and block-copy tilings built around LDIR, such as `04 5e ed b0`. No loop-bearing final dominant ever enters the partner (89 of 89). Open-then-closed happens in 47 of 80 Z80 worlds and in 20 of 20 worlds of the 8080 subset at L = 16.
- **C3, the substrate decides the beginning.** Two properties set whether life starts open:
  - whether there is a literal-write channel (an instruction that writes a constant);
  - whether by-products are lethal.

  With lethal tar, or with no literal channel, replicators are born closed or loop-bearing. On the dial, the open beginning ends at a threshold between p = 0.1 and 0.3. Closure is slowest at intermediate lethality.
- **C4, closure either regenerates or transmits.**
  - Regenerators (at most 2 transmissible mutation sites) overwrite their own body with copies of their copying code. Mutations outside the core are erased, a kind of self-repair.
  - Transmitters (at least 5 sites) copy bytes they never execute, so those bytes carry heritable variation.
  - Open pushers' lineages stop copying under serial transfer: after 8 transfers, 0.00–0.04 of unmutated lineages are still alive.
- **C5, a one-byte switch with a proof (Proposition 8).** In the block-copy core `XX 5e ed b0`, the first byte sets the copy offset d = XX mod 2L. The overlapping LDIR copy tiles the first d bytes, as an overlapping memmove does. When d divides L, exactly d − 4 sites are transmitted: 2,289 of 2,343 cells match exactly. The core and its variant `1e NN ed b0` are carried by 2,347 of 15,655 confined cells.
- **C6, the environment sets the regenerator/transmitter balance.** We competed a matched pair (same core, different first byte). The pair is neutral within single encounters and drifts without mutation. With mutation it reaches an equilibrium set by the environment: at L = 32, transmitters reach 0.55–0.87 under lethal tar against 0.03–0.37 under benign tar, and the transmitter share rises with p on the dial.

  Why is open. Our pre-registered explanation by mutational flows failed. An exploratory census found that a body of tiled copies of the copying code is hijacked by intruding executors, while a body of junk protects. That favours transmitters under both tars, so it is not the whole story.
- **C7, closure without supplied registers.** With all registers randomised at each encounter, 8 of 40 worlds close by 3 million steps. They do it with self-initialising closers that leave 5–7 bytes unexecuted, and 6 of the 8 are transmitters.
- **C8, theory.**
  - **Theorem 1:** a per-encounter count at a fixed shift.
  - **Theorem 2:** a machine running 128 instructions inside L ≤ 100 cells must revisit a cell, a cycle by pigeonhole. Closure therefore implies a loop.
  - **Proposition 3:** two-byte words that write themselves into an all-zero partner.
  - **Lemma 7:** the information inflow H(offspring | partner) of the first replicator is 3.9, 6.2, 7.8 and 8.0 bits across lengths.
  - **Model 5:** a closure-window model, restricted after the dial result.
  - We are asked to prefer substrate-level theorems over loose theory.

**Tonight's two experiments**, planned and not yet run:
- **A, the toxic payload.** Under lethal tar, zeros in a transmitter's never-executed payload cost the owner nothing but halt any intruder whose pointer runs into them. Hypothesis: the payload evolves into a defence, a phenotype expressed only in other organisms' execution, and this is why lethal environments favour transmitters.
  - **Tests:** zero enrichment in payloads against drift; dose-response with p; position of zeros against intruders' entry points; knockouts in encounters and in whole populations (detoxified against sham).
  - **Rival explanation:** zeros accumulate from damage, not selection.
- **B, genealogy.** An exact per-step record of which tape each cell copied from, with founder labels carried forward. This decides whether closed organisms *descend* from open ones (with or without recombination) or arise independently and displace them. Validation uses a seeded-closer world with known ancestry and a control with shuffled pairs.

# 2. Research workstreams

Cover every workstream. If you must choose, go deepest on W1, W2, W4 and W5.

**W1. Novelty threats and closest prior work (highest priority).**
- Find every work on spontaneous emergence of self-replicators from random code or artificial chemistries. Include at least:
  - Rasmussen's Coreworld/Venus;
  - Ray's Tierra;
  - Avida (Adami, Ofria, Lenski);
  - Pargellis's Amoeba ("spontaneous generation of digital life");
  - Fontana's AlChemy and Fontana & Buss;
  - Stringmol (Hickinbotham, Clark, Stepney);
  - Nanopond;
  - combinator chemistries (Kruszewski & Mikolov);
  - Agüera y Arcas et al. 2024 (BFF, Forth, Z80, 8080) and the book chapters that discuss it;
  - every 2024–2026 follow-up on arXiv, bioRxiv or ALife proceedings: BFF variants, SUBLEQ/RISC-V/Forth soups, "computational life", "primordial soup of programs".
- For each, ask: did it report (a) replicators that depend on or write into neighbours before self-contained ones, (b) a transition from open to closed or self-confined copying, (c) self-repairing versus variation-transmitting replicators, (d) environment-dependent selection between them, or (e) unexecuted payloads with a function?
- Quote or paraphrase the exact claim, with location. Rate each threat to C1–C7 as none, partial or serious.
- Check whether "digital primordial soup" or "heredity before individuality" has been used as a phrase or thesis before.

**W2. Theory of individuality and its order relative to heredity.**
- Map what we can cite and where our result fits:
  - major transitions (Maynard Smith & Szathmáry);
  - transitions in individuality (Michod; Godfrey-Smith's Darwinian populations and "marginal" cases);
  - the "Darwinian threshold" and communal pre-cellular evolution (Woese 2002; Vetsigian, Woese & Goldenfeld 2006);
  - information-theoretic individuality (Krakauer et al. 2020) and autonomy measures (Bertschinger et al.);
  - organisational closure and closure of constraints (Rosen; Maturana & Varela; Moreno & Mossio; Montévil & Mossio);
  - Clarke and Pradeu on biological individuality.
- Which accounts predict heredity before individuality, which predict the reverse, and which would call our "open replicators" not individuals at all? Is our pointer confinement a defensible operational analogue of "closure"? Which terms should we avoid?
- Identify any information-theoretic individuality measure we could compute from our traced encounters, and say exactly how.

**W3. Prebiotic chemistry analogues.**
- Template replicators that need partners or surfaces; replicases that copy others ("the replicase's dilemma").
- Parasites in replicator systems and spatial rescue (Eigen's error threshold and hypercycle; Boerlijst & Hogeweg; Takeuchi & Hogeweg; Szathmáry's stochastic corrector).
- Cross-replicating ribozymes (Joyce lab); autocatalytic sets and "evolution before genes" (Kauffman; Hordijk & Steel; Segré's GARD; Vasas, Szathmáry & Santos 2010 and 2012).
- What do these say about whether non-autonomous (open) replication precedes autonomous replication? Which experimental chemistry results could a Nature reviewer from origin-of-life research hold up against C1–C3?

**W4. Analogues and prior art for the toxic-payload hypothesis (experiment A).**
- **Core War:** DAT bombs that kill processes that execute them, imps, papers, scanners, and evolved warriors (including 2024–2026 work on evolving Core War programs, possibly with LLMs). How exactly does "memory the owner never runs but intruders execute and die on" appear there?
- **Biology:**
  - the "bodyguard" hypothesis for non-coding DNA (Hsu 1975);
  - selfish DNA (Orgel & Crick; Doolittle & Sapienza);
  - non-coding DNA as a buffer against insertion or transposons;
  - superinfection exclusion in phages;
  - restriction–modification and toxin–antitoxin systems;
  - bacteriocins, spite and costless harm;
  - weaponised waste, such as yeast making ethanol to poison competitors ("make–accumulate–consume");
  - Dawkins's extended phenotype, a trait expressed in another organism's machinery.
- **Digital evolution:** host–parasite coevolution and defences, for example Zaman et al. 2014 in Avida, and parasites and hyper-parasites in Tierra.
- What predictions do these literatures make that we could test tonight, such as position, dose-response, cost, or arms races? What controls did they use to separate adaptive defence from damage?

**W5. Methods for genealogy and causal analysis (experiment B).**
- Line-of-descent and knockout analysis in Avida (Lenski et al. 2003, "The evolutionary origin of complex features").
- Phylogeny tracking and ancestry-based metrics (Dolson, Lalejini, Ofria; phylotrackpy; "Interpreting the tape of life").
- Hereditary stratigraphy (Moreno, Rodriguez-Papa, Ofria).
- How to attribute ancestry when copies are partial, shifted or chimeric: thresholds, recombination detection, and similarity at the best cyclic shift.
- Standard ways to distinguish descent from independent origin and displacement. Pitfalls such as survivorship bias and the base rate of the dominant class at the reference time.
- Recommend a decision rule and a null model.

**W6. The copy-offset switch (C5) and the regenerate/transmit trade-off (C4, C6).**
- **Novelty check:** is Proposition 8 elementary? Overlapping LDIR or memmove filling memory with a period-d pattern is a known Z80 idiom, and there are combinatorics-on-words results (Fine–Wilf, periodicity lemmas). Tell us how to cite and frame it honestly, and whether a stronger, non-trivial theorem is within reach. For example: which offsets can evolution reach by one mutation, and the stationary distribution under the switch.
- **Biological analogues:** robustness against evolvability (Wagner); "survival of the flattest" (Wilke et al. 2001, Nature); mutator and anti-mutator alleles and the drift barrier (Lynch); proofreading and repair as erasing variation; the Weismann barrier; environment-dependent selection on evolvability.
- Which theories predict that a harsher environment favours variation-transmitting replicators?

**W7. Theory to make claims rigorous.**
- Results we can import to prove our claims at the level of the substrate:
  - cycle and pigeonhole bounds for deterministic machines confined to bounded memory (linear-bounded automata, eventual periodicity);
  - fixed points and self-reproduction (Kleene's recursion theorem, von Neumann's constructor and uninterpreted description copying, quines);
  - information-theoretic bounds on replication fidelity and information inflow (Eigen; Adami's genomic information; channel views of heredity).
- For each, state the exact theorem and the assumptions we would need to check.

**W8. Critiques of digital-soup and ALife claims, and how successful papers answered them.**
- Collect the standard objections: toy substrate, ad hoc definitions of life or replication, dependence on convention (registers, wrap-around), generality across instruction sets, and the gap from digital to chemical life.
- Collect the published responses.
- Find the ALife and digital-evolution papers that reached Nature, Science or PNAS (for example Lenski et al. 2003; Wilke et al. 2001; Blount et al. 2008) and summarise how they framed significance for biologists in the first paragraph.

**W9. Technical facts reviewers will check.**
- Confirm from primary datasheets or opcode tables how a real Intel 8080 decodes the opcodes that the Z80 uses as prefixes or for extra instructions. The reviewer's claim is that CB is an alternate JMP, D9 an alternate RET, and DD, ED and FD alternate CALLs.
- Undocumented Z80 behaviour that could matter: flags, the R register, block-instruction timing and interruptibility, LDIR with overlapping source and destination.
- Cite a primary source for each.

**W10. Positioning, title and Nature practicalities.**
- Which communities (origin of life, evolutionary transitions, ALife, philosophy of biology) would care, and what each would need to see.
- Propose 3–5 titles of at most 75 characters that make no claim beyond the evidence.
- Nature's current policies on code and data availability *during peer review* (repository access for reviewers, Code Ocean and similar), and the limits on Extended Data and length.

# 3. Rules

- **No invented references.** Every citation needs authors, year, title, venue, and a DOI, arXiv ID or stable URL.
- Mark each source as **[full text read]**, **[abstract only]** or **[cited secondhand]**.
- If you cannot verify something, say so. Never fill gaps from memory without flagging it.
- Quote sparingly: at most one short quote of 25 words or fewer per source. Otherwise paraphrase.
- Tie every finding to our claim IDs (C1–C8) or to experiments A and B, and say what we should do: cite, reframe, add a control, run a new test, or drop a claim.
- Prefer primary sources. Include 2024–2026 preprints and say they are not peer-reviewed.
- Disagree with us where the literature does. A finding that weakens a claim is as valuable as one that supports it.

# 4. Deliverable

A single report with these sections:
1. **Executive summary** in 10 bullets: the most important things we did not know.
2. **Novelty threats**, ranked serious, partial or none, each with the exact overlapping claim and how to answer or concede it.
3. **Must-cite list**, 15–25 entries with one line each on why, ranked.
4. **Findings by workstream** (W1–W10).
5. **Reviewer attack surface:** the ten most likely objections, each with the best available counter-evidence or an honest concession.
6. **Recommendations for tonight's experiments A and B:** controls, statistics, decision rules, and predictions taken from the literature.
7. **Theory leads:** theorems we can import or prove, with exact statements.
8. **Framing and titles.**
9. **Bibliography**, in Nature style, with DOIs or arXiv IDs and verification marks.
