# Literature verification against primary sources

Checked on 2026-10-09. Every quote below is verbatim and at most 25 words. Where the source uses LaTeX, maths is shown as plain text and flagged. Page numbers for Agüera y Arcas et al. are PDF pages of arXiv v2.

"Reviews" means the three AI-written reviews: DR1 = `dr1_alife.md`, DR2 = `dr2_alife.md`, DR3 = `dr3_artificial_life_literature_review.md`.

**Sources read in full:**
- arXiv HTML for 2607.09211 v1 and v2, 2607.01483 v1, 2609.10817 v1 and 2601.03335 v1.
- arXiv PDF for 2406.19108 v2, including a rendered Fig. 2.
- ICWS'94 annotated draft v3.2 (koth.org).
- The Intel 8080 Assembly Language Programming Manual, Rev C (98-004C, 1976). I read the scanned PDF and rendered p. xix.
- The `1801BM1/vm80a` Verilog decoder.
- MAME `i8085.cpp`.

## Verdicts at a glance

| Item | Exists | Main correction to the reviews |
|---|---|---|
| 1. Cicala et al. 2026 | Yes (v1, v2) | The turnover happens **without** tasks. DR3 says the opposite and is wrong. The v2 appendix already shows the `XX 5E … ED B0` motif, with byte 0 annotated as the "partner offset". |
| 2. Knierim et al. 2026 | Yes (v1 only) | Random search beats BFF only with **tuned** byte distributions. DR3's "phase-shifting looping replicators" is not in the paper. |
| 3. Agüera y Arcas et al. 2024 | Yes (v1, v2) | All of DR2's quotes check out. DR3 overstates zero-poisoning. |
| 4. Core War / DRQ | Yes | DAT kill is confirmed (ICWS'94 §5.2, §5.5.1). I found no passive own-body lethal payload. DR3's "carpet traps" could not be found. DRQ is in the GECCO '26 proceedings. |
| 5. Other soups 2024–26 | — | One new Z80 soup paper (Jha et al., arXiv:2609.10817) that no review mentions. None of these papers has LDIR genealogy or a functional unexecuted payload. |
| 6. 8080 aliases | Confirmed | Confirmed from a decoder reconstructed from the die, plus emulator sources. Intel's own manual lists these codes as "---". DR3's citation of the Intel manual is wrong. |

---

## 1. Cicala et al. 2026, arXiv:2607.09211

**Exists:** yes.

**Citation:** Cicala, F., Niklasson, E., Randazzo, E., Boukortt, S., Basti, A., Etcheverry, M., Saurous, R. A., Laurie, B., Manyika, J., Agüera y Arcas, B. & Richards, B. A.
- Title: "Coevolution of self-replication and function in a digital primordial soup". v1 is titled "Co-evolution of …".
- Preprint arXiv:2607.09211 [cs.NE]. v1 was posted on 10 Jul 2026 and v2 on 2 Sep 2026.
- Not peer-reviewed. There is no journal reference, and a Crossref title search found no published version.

### Differences between v1 and v2 (verified)

- **Title:** "Co-evolution" in v1 became "Coevolution" in v2.
- **§4.1 registers:** v1 sets "register A and the stack pointer" to 0xFF. v2 says "registers A and F, and the stack pointer".
- **Robustness sentence (§2.3):** only v1 says "The transition is intrinsically driven by differences in mutational robustness among the replicators (which we quantify below), with tasks serving to accelerate it." v2 replaces it with "The takeover does not depend on the tasks: LDIR replicators are more robust to mutation than Load–Push replicators (see Fig. 2F)".
- **No-task LDD control (§2.3):** v1 says it "fails to complete within one million epochs". v2 says "within ten million epochs", and §4.4 adds that these runs used 10^7 epochs.
- **Discussion caveat:** v1 notes that "our programs were initialized with elements that have predefined computational semantics". v2 drops this caveat.
- **New in v2:**
  - §2.6, a hard-wired-copy control;
  - Appendices A–C, which include annotated program listings;
  - in §4.5, the clause that the copy "wraps harmlessly".

### (a) Load–Push first, LDIR later

- §2.3 (both versions): "Lacking a loop, full replication requires a long chain of these pairs, meaning a Load–Push replicator consumes the entire 32-byte tape."
- §2.3: "A second replicator, the ‘Load Increment Repeat’ (LDIR) replicator, reliably took over later."
- §2.3: "the Load–Push replicators were the first to appear and dominate the grids". This refers to Fig. 2D, the mean over 100 seeds.
- **What the curves count (§4.4).**
  - Families are counted by byte pattern: "we used a byte-matching search to count known replicators of each type". The patterns themselves came from traces: "We identified these key instructions by manually inspecting our runs and looking at execution traces to see what elements of tape were responsible for replication."
  - The "LDIR" curve is the sum over four patterns: `ED B0`, `ED B8`, `ED A0` and `ED A8` (LDIR, LDDR, LDI and LDD).
  - The Load–Push curve is the sum over four 4-byte patterns: `01 C5 01 C5`, `11 D5 11 D5`, `21 E5 21 E5` and `E5 2A E5 2A`.
  - A tape counts if it contains the pattern anywhere. **My inference:** such a count cannot tell executed bytes from inert ones.

### (b) Their explanation

They give both: an intrinsic robustness hierarchy, plus acceleration by task pressure.

- v2 §2.3: "Robustness to mutation drives the population toward more compact replicators on its own, but the pressure to solve mathematical tasks accelerates that shift".
- The robustness account is labelled a hypothesis: "To verify our hypothesis that LDIR replicators naturally come to dominate because there is a robustness hierarchy of the replicators".
- **How they test it (Fig. 2F, §4.5).**
  - A single-genome assay: 1, 4 or 8 successive point mutations, each followed by one execution against a zeroed partner; 100 trials per group.
  - Results appear as bars with Wilson intervals, compared by one-sided Z-tests. No percentages are given in the text.
  - The assay is not a causal test inside the soup.
- **Why tasks accelerate it.** The free tape of an LDIR replicator can hold task code. A program that also passes validation is chosen to interact more often.

### (c) Does the transition happen without task pressure?

**Yes**, in both versions. v2 §2.3: "removing the task-based gating of interactions (see 4 Methods for details) slowed the Load–Push-to-LDIR transition but did not prevent it".

- **Caveat (§4.4).** The no-task control also changes the interaction rate: "Note that this also raises the interaction probability of every program to 1". The base rate is otherwise 0.3. So the control changes two things at once.
- **The LDD case (LDIR blocked) is different.** "without it, the transition from Load–Push is extremely slow and fails to complete within ten million epochs" (v2). v1 says one million epochs.

### (d) The LDIR test genome

- §4.5 (both versions): "The LDIR replicator was the four bytes [0x1E, 0x20, 0xED, 0xB0], which set register E to 32 and then executed the LDIR instruction".
  - It is padded: "followed by 28 uniformly random bytes drawn independently in each trial to fill the tape to 32 bytes".
  - On the padding: "the content of this trailing region does not affect replication and is reproduced faithfully along with the functional prefix" (both versions).
  - BC = 0 (v2 only): "so the copy wraps harmlessly under modulo- 2ℓ addressing until the instruction budget is exhausted". The source writes 2ℓ in LaTeX.
- **The other canonical genomes.**
  - LDD: 11 bytes `2E 1F 1E 3F 0E 20 ED A8 28 FC 76`, "again padded with zeros to 32 bytes".
  - Load–Push: `01 C5` repeated 16 times.
- **Not mentioned by any review: their evolved programs use the `XX 5E` motif, with byte 0 as the destination offset (v2, Appendix B).** Examples:
  - B.1.1, tape `E0 5E 0E 09 ED B0 14 5A 76 41 00 …`. It is annotated "01: 5E LD E, (HL) ; E = mem[00h] = E0h == 20h mod 64 (partner offset)", where `==` is the source's own notation. It copies only part of itself: "04: ED B0 LDIR ; copy the 9-byte functional core to the partner half". The partner's bytes 09–1F are therefore not overwritten.
  - B.1.5 and B.1.6, tapes `20 36 5E 7A 4B ED B0 …` and `20 9A 5E 14 4B ED B0 …`. Here byte 0 also sets the count: "04: 4B LD C, E ; BC = 32 (byte counter)".
  - **What they do not do:** they never treat byte 0 as a switch, never vary it systematically, and never count its admissible values.

### (e) Execution conventions (v2 §2.1, §4.1, §4.3, Table 1, Algorithm 1)

| Convention | What the paper says |
|---|---|
| Tape | ℓ = 32 bytes. Pairs are concatenated into a 64-byte memory, and "every memory access is taken modulo 2ℓ" (LaTeX in source). |
| Registers (interaction) | "registers HL, BC, and E, and the program counter PC are set to zero". Also "registers A and F, and the stack pointer, are set to 0xFF". D = 0 during interaction and D = x during validation: "the register initialization is identical, except that register D is initialized to zero rather than the task input." IX, IY, the alternate registers, I and R are **not stated**. |
| SP | "the stack pointer value 0xFF resolves to the last byte of the tape, so stack writes grow backwards from the end of the memory" |
| Budget | B = 512 instructions. |
| LDIR counting | **Not stated in words.** The v2 numbers imply one step per byte transferred. B.1.2, which has `LD C,2Eh` (BC = 46) plus 9 other instructions, reports "Validation cost: 55 CPU steps (46 replication steps + 9 arithmetic steps; …)". B.1.1 reports 15 steps, which is 6 instructions plus 9 iterations (my arithmetic). §4.5 says the BC = 0 copy runs "until the instruction budget is exhausted". |
| Count overrun | "counter values exceeding 32 overrun harmlessly under mod-64 addressing, rewriting already-copied bytes with identical values" (App. B) |
| Mutation | "each program has a 1/64 chance per epoch of one byte being reinitialized". It is applied at the start of each epoch. |
| Lattice / pairing | 32 niches of 128×128, giving 2^19 programs. The partner is a random von Neumann neighbour with probability 0.95; otherwise it is drawn uniformly from the whole population (π = 0.05). Boundary conditions are **not stated**. The selected program P1 always comes first: "execution begins at the first byte of the first program". Pairs containing a duplicate are filtered out, so about 56% of programs are selected per step. |
| Selection | p_base = 0.3. p_succ = 1, minus a metabolic penalty C·k/B with C = 0.3. |
| Duration and seeds | 10^6 epochs (10^7 for Fig. 2E). 100 seeds per condition. The ancestry matrix uses G = 2,000 seeds. |

### (f) Unexecuted bytes, zeros, halting, toxicity, lineage, recombination

- **Unexecuted bytes.**
  - Appendix B uses a colour class "Unexecuted / trailing bytes" and labels such regions "[Unused trailing genomic bytes]".
  - In B.1.6 the trailing bytes act only through the copy in the partner half: "their clone at 3Eh--3Fh decodes as DEC A; RET".
  - No defensive or protective role is proposed.
- **Zero bytes.**
  - The paper says nothing about zeros being special. 0x00 is a Z80 NOP.
  - Partners are zeroed in the validation and robustness assays, and the LDD genome is zero-padded.
- **Halting.**
  - HALT (0x76) ends execution. Under the metabolic penalty, "evolved programs consistently began incorporating a HALT instruction, which terminates execution as soon as the correct output is computed" (§2.4).
  - Some programs halt conditionally on register D (Fig. 3B).
  - In B.1.3 and B.1.4, programs skip LDIR during validation by consuming `ED B0` as the operand of `LD BC,(nn)`.
- **Toxicity or lethality.** I searched for "toxic", "lethal" and "poison" and found zero occurrences.
- **Lineage.**
  - Ancestry is tracked only as a niche label: "This label was then carried through interactions to record the most recent solved niche from which each lineage received material." (§4.7)
  - They warn: "Such a label is not the same as identifying the niche that contributed the code responsible for the solution." (§2.6)
  - There is **no analysis of whether LDIR replicators descend from Load–Push replicators.**
- **Recombination (§3).**
  - "Our interaction protocol does not preclude genetic mixing, and accidental recombination may have played a role in evolutionary transitions."
  - "Most of the dominant replicators we manually inspected reproduced asexually" (v2 wording).

### (g) Lethality of by-products or a literal-write channel

**Neither is varied.** The only change to the instruction set is in Fig. 2E. There they "turn the key instruction byte pairs for all the LDIR-family replicators except LDD into the equivalent of NO-OPs".

**LDD when LDIR is blocked:** "In their absence, we observe the consistent emergence of a different replication mechanism based on the LDD instruction". This happens consistently only under task pressure.

### Discrepancies with the reviews

- **DR3 is wrong on the mechanism.** It says Cicala "attributed the structural turnover entirely to external task selection". On that basis it claims the turnover "occurs spontaneously in the complete absence of extrinsic task selection" as our distinction. Both versions report the no-task transition and attribute it to robustness.
- **DR3's epoch count is unsupported.** It says "over 100,000 epochs"; that number is not in either version. Runs last 10^6 epochs.
- **DR2 and DR3 partly misstate the first-byte point.** Both say Cicala "never treat the first byte as an offset switch" or "did not observe the one-byte copy-offset switch". Partly wrong: the v2 listings annotate byte 0 as the "partner offset". What is new in our paper is the *switch* analysis (regenerate vs transmit) and the d − 4 count.
- **DR2 misidentifies the reference genome.** It says "Your C5 core is their reference genome". Their reference genome is `1E 20 ED B0` (load an immediate into E), not `XX 5E ED B0`. Their *evolved* v2 listings do contain `XX 5E … ED B0`.
- **DR2 overstates the BC inconsistency.** It says §4.4 "says BC = 32" and that this is internally inconsistent. The §4.4 sentence only describes what LDIR does in general. The evolved counters are 9, 32 and 46, with overrun described as harmless.
- **DR2's §4.4 quote is not verbatim.** It is stitched together. The actual text: "not an exhaustive list of possible replicators, and likewise not a perfect search for replicators of this type".
- **DR2's robustness quote is from v1 only.** DR2 cites it to v1 correctly.
- **DR2's register list is v2 only.** "SP, A and F = 0xFF" does not match v1.
- **DR1 misplaces the cause of block-copy.** It says "metabolic constraints promote … block-copy replication". The paper credits the faster move to LDIR to task pressure. The metabolic penalty drives HALT and conditional halting.

### Implications

- Cite Cicala v2 as the first quantitative report of the Load–Push → LDIR turnover, **including without tasks**, and of the `XX 5E … ED B0` motif with byte 0 as destination. Our novelty has to rest on:
  - trace-based closure;
  - the lethality and literal-channel dial;
  - regenerators vs transmitters, and the d − 4 switch analysis;
  - descent analysis and the proofs.
- Their conventions differ from ours: zero is a NOP, 32-byte tapes, budget 512 with LDIR counted per byte, and P1 always first. State these differences explicitly.

---

## 2. Knierim et al. 2026, arXiv:2607.01483

**Exists:** yes.

**Citation:** Knierim, C., Versari, L., Obryk, R., Agüera y Arcas, B. & Saurous, R. A.
- Title: "BFF: Simple explanations for complex phenomena".
- Preprint arXiv:2607.01483v1 [cs.NE], 1 Jul 2026. Only v1 exists.
- The HTML header shows the date "August 24, 2026".
- Not peer-reviewed. The paper covers BFF only, not Z80.

### Verified quotes

- **Abstract.**
  - "we explore the alternate hypothesis that self-replicators can be found at least as easily using simple mutation random walks in program space"
  - On ancestry caps: "showing instead that it merely stops self-replicators from taking over the soup".
- **§1:** "distributionally tuned random mutation is at least as powerful as pairwise interactions for finding self-replicators".
- **§1.2.**
  - Program interaction in BFF "is not an unusually powerful search operator".
  - Mergers: "We define a merger to be a consecutive copy of multiple bytes that have not been previously copied together."
- **§3.3:** "Thus compositionality is not needed for the discovery of self-replicators in the BFF system."
- **Detector (§2).**
  - A program is chained through 5 executions against noise, across 9 branches.
  - "For the first halves, we compute the number of bytes that match the original program in at least three of the final tapes."
  - The score is the minimum over the two halves and runs from 0 to 64. The replicator threshold is 48: "This is a somewhat arbitrary choice, and results do not change significantly for thresholds of at least ≈ 20". The source writes ≈20 in LaTeX.
  - The detector "does not fire for “semi-replicators” that copy themselves into a different position".
- **Non-executed bytes (§2):** "some self-replicators copy their functionality and most of their bytes but have some non-executed bytes that are kept from the original program". No function is attributed to these bytes.
- **Numbers (§3.1–3.2).**
  - BFF needs about 2.5·10^6 interactions (5·10^6 programs tested) to reach its first replicator.
  - Uniform random sampling needs 2.9·10^7 programs: "pure random sampling is about 6 times slower than running BFF".
  - Tuned distributions are faster: 1.7·10^6 (the BFF-derived distribution), 4.5·10^5 (CUST) and 9.4·10^4 (CUST64).
  - Uniform random walks at p = 1/200 to 1/50 need 9.5·10^8 to 2.0·10^8 programs (Table 1), which is *slower* than BFF.

### Discrepancies with the reviews

- **DR2 overgeneralises.** It says "random-walk mutation finds self-replicators faster than paired interaction". That holds only with tuned byte distributions; with uniform bytes, random processes are slower.
- **DR3's claims are not in the paper.** It says Knierim noted "looping replicators preserved unexecuted trailing bytes that shifted across replication cycles" and "phase displacements across generations".
  - The only mention of shifting is footnote 2: a non-replicator that scores highly because "in each execution the program shifts itself by two positions". Its copy mechanism eventually destroys itself.
  - It is offered as an example of a detector artefact, not as a class of looping replicator.
- **DR1 and the rest of DR2 are accurate.**

### Implications

- Cite Knierim for the detector and for their point that emergence is a search problem, distinct from takeover.
- Expect reviewers to ask for a baseline: random search on Z80 bytes with a tuned distribution.

---

## 3. Agüera y Arcas et al. 2024, arXiv:2406.19108

**Exists:** yes.

**Citation:** Agüera y Arcas, B., Alakuijala, J., Evans, J., Laurie, B., Mordvintsev, A., Niklasson, E., Randazzo, E. & Versari, L.
- Title: "Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction".
- arXiv:2406.19108. v1 was posted on 27 Jun 2024 and v2 on 2 Aug 2024. Page numbers below refer to the v2 PDF.

### Z80 soup (§3.3, pp. 16–17; Fig. 12)

- **Set-up and conventions.**
  - "We study a 2D grid of 16-byte programs initialized with uniform random noise."
  - Each step picks an adjacent pair and they "concatenate them in random order (“AB” or “BA”). Then we reset the Z80 emulator and run 256 instruction steps".
  - Addresses are taken modulo the "concatenated tape length (32 bytes)".
  - SP: "at initialization Z80 sets the stack pointer at the end of the address space". This "gives tape A a simple mechanism of writing to tape B".
  - Mutation: "mutations are applied to random bytes of the grid". The rate is not given.
  - **Not stated:** the other registers (beyond "reset"), grid size, mutation rate and how LDIR steps are counted. The emulator is `superzazu/z80` (footnote 4).
- **Succession.**
  - "Most of the time we see the development of an “ecosystem” of stack-based self-replicators that eventually gets replaced" by self-replicators using “LDIR” or “LDDR”.
  - Fig. 12 caption: "Then the grid is overtaken by more robust self-replicators that use memory copy instructions."

### 8080 (§3.3, p. 17)

- "We have also tried the 8080 CPU in the long-tape setting."
- "This produces replicators which seem to always be two bytes repeated, for example, 01 c5".
- "00 in 8080 is a no-op, so this is harmless."
- "Note that these replicators are non-looping, which, in long-tape BFF, are not able to take over all of memory."
- "Perhaps for this reason we have never seen a looping variant emerge in 8080."

### BFF zero-poisoning (Fig. 2, p. 7; rendered and checked)

- Fig. 2 panel text: "The first replicator can copy "0", but can't write over "0", which leads to zeros multiplying in the soup."
- Right-hand annotation: "~14% of the soup are zeros" "during the zero-poinsoning phase" (the misspelling is in the original).
- Recovery: "This replicator has a more robust "[<,}]" structure that can overwrite zeros."

### Tracer tokens and the first BFF replicator (p. 5; Figs. 2–3, pp. 7–8)

- "Tokens are tuples of (epoch, position, char) packed into 64-bit integers."
- Fig. 2, epoch 2354: "The first self-replicator emerges in a complex rewrite event, triggering a rapid cascade of replications."
- Same panel: "A few bytes persist on the tape from the epoch 1, but most were copied from another tape."
- Also: "The first replicator accidentally overwrites itself with the content from another tape, but its copies survive."

### Long-tape BFF (§2.4, pp. 11–12)

- If the heads start at the program counter, "we find that trivial (non-looping) self-replicators rapidly take over the" universe.
- "Adding an offset to head1 is sufficient to allow looping self-replicators to arise."
- "Anecdotally, we appear to require an offset somewhat larger than 8 to work (e.g., 12 or 16)."

### Long-tape Forth (§3.1.2, p. 15)

- Replicators "tend to consist of a fairly long non-functional head followed by a relatively short functional replicating tail."
- The reason given: "beginning to execute partway through a replicator will generally lead to an error". In turn, "adding non-functional code before the replicator decreases the probability of that occurrence".
- "In the replicator above, the functional tail involves the last 7 instructions, starting at PUSH 1."
- Also: "Note that this loop actually copies the head of the “next” replicator rather than its own head."
- **Not in any review:** early long-tape Forth replicators were open too. "“Bad” replicators … arise within the first few seconds. These generally do not loop but instead consist only of a series of PUSHes and COPYs".

### Discrepancies with the reviews

- **DR2:** every quote and location checks out.
- **DR1 on the 8080:** it describes the paper as using "8080-like" soups. The 8080 appears **only** in the long-tape setting, never in paired-tape soups.
- **DR3 on zero-poisoning:** it describes "fragile replicators collapse into dead, zero-filled states". That is overstated. The paper shows a peak of about 14% zeros and stalled replication, followed by takeover by a zero-tolerant replicator.

### Implications

- Our 8080 result comes from a paired-tape soup. Contrast it explicitly with their long-tape report: "never seen a looping variant".
- Cite Forth long-tape as a second "open first" precedent, alongside the Z80 stack copiers.
- Cite the Forth non-functional head as the nearest protective-payload precedent. It protects against the soup's own random entry points, not against foreign executors.

---

## 4. Core War, and Digital Red Queen

### DAT kills the process that executes it: confirmed

**Source:** ICWS'94 annotated draft v3.2. Base draft by Mark Durham; revision by Stefan Strack (2 Feb 1994); HTML by Stephen Beitzel (1995). Copy at koth.org/info/icws94.html.

- §5.2, lines 0503–0504: "Attempted execution of a DAT instruction by a task effectively removes that task from the warrior's task queue."
- §5.5.1, lines 0684–0685: "No additional processing takes place. This effectively removes the current task from the current warrior's task queue."
- §5.2, lines 0509–0510: "A warrior is no longer executing when its task queue is empty."
- **Background is lethal by default (§4.2–4.3).** Empty core is normally DAT: the loader may preload "an instruction such as "DAT #0, #0" into all of core". The standard KOTH setting lists "Initial Instruction: DAT.F $0, $0".
- **This is the closest analogue of lethal tar:** in Core War, unoccupied memory kills by default.

### Passive own-body memory that kills foreign processes: not found

**What I checked:**
- the ICWS'94 draft (its only warrior is Dwarf, §2.7, a bomber);
- the strategy section of the Wikipedia "Core War" page (secondary);
- the Digital Red Queen paper;
- comments in archived vampire/pit warrior source files;
- secondary descriptions of imp gates.

**Every lethal mechanism I found is active:**
- **Bombs** are thrown into the opponent's memory.
- **Imp gates** are cells that a looping process keeps decrementing. This comes from secondary tutorials; ICWS'94 does not define imp gates.
- **Vampires (closest).** Wikipedia: "A vampire tries to make its opponent's processes jump into a piece of its own code called a "pit"." The pit is code inside the vampire's own body that traps foreign processes. But those processes reach it only through JMP "fangs" that the vampire actively scatters.
- **Decoys** are passive but are not lethal. They mislead scanners.

**DR3's "carpet traps"** ("non-executed DAT blocks positioned behind an active copy loop") could not be found under that name. "DAT carpet" in Core War usage means an active core-clear.

**Absence of evidence is not proof.** I did not read Dewdney's Scientific American columns (1984, 1985, 1987) or the rec.games.corewar archives.

### Digital Red Queen

**Exists:** yes.

**Citation:** Kumar, A., Bahlous-Boldi, R., Sharma, P., Isola, P., Risi, S., Tang, Y. & Ha, D.
- Title: "Digital Red Queen: Adversarial Program Evolution in Core War with LLMs".
- arXiv:2601.03335v1, 6 Jan 2026.
- **Also published in the GECCO '26 proceedings**, DOI 10.1145/3795095.3805116 (Crossref: issued 10 Jul 2026). I did not read the ACM version.
- Quote from §2 Related Work, paragraph "Core War": "A program can inject a DAT instruction in front of an opponent’s process, terminating it when that process attempts to execute it."

### Discrepancies with the reviews

- **DR1 on DRQ's status.** It says DRQ "should not be presented as a peer-reviewed scientific precedent". It is in the GECCO '26 proceedings.
- **DR2 on imp gates.** It says "DAT bombs and imp gates are memory the owner never runs but that kills foreign processes". Imp gates are actively maintained, and bombs are thrown into enemy territory; neither is a passive payload.
- **DR3's "carpet traps"** remain unverified.

### Implications

- Cite ICWS'94 §5.2 and §5.5.1 for DAT semantics, and §4.2 for core pre-filled with DAT.
- Present the vampire "pit" as the nearest Core War analogue to a lethal payload inside one's own body. The difference to state: our candidate payload is never executed by its owner and needs no action by its owner.

---

## 5. Other machine-code soups, 2024–2026

**How I searched:**
- the Semantic Scholar list of papers citing 2406.19108 (25 indexed);
- arXiv API metadata searches: Z80; BFF + replicators; "primordial soup" + programs; SUBLEQ; RISC-V + replicators; Forth + replicators; "self-replicating programs"; "computational life";
- two web searches, including the ALIFE 2025 programme.

I could not run full-text searches of the ALIFE proceedings.

### Relevant papers found

1. **Cicala et al. 2026** (arXiv:2607.09211; see item 1).
   - (i) Yes: Load–Push copiers come first and write into the partner via SP.
   - (ii) No payload with a defensive function. Trailing bytes act only through their copy in the partner (App. B.1.6).
   - (iii) No: ancestry is a niche label only.
2. **Jha, K., Cicala, F., Agüera y Arcas, B., Richards, B. A., Jaques, N., Kleiman-Weiner, M. & Niklasson, E.** "Tapes Together Strong: The Co-evolution of Computation and Cooperation". arXiv:2609.10817v1 [cs.MA], 9 Sep 2026. **No review mentions it.**
   - The setting: programs "written in a modified Z80 assembly language, initialized as 32-byte sequences of completely random bytes". Pairs share a 64-byte cyclic tape, with one CPU per program, an energy budget, and a STEAL instruction.
   - (i) No replicator taxonomy; replication frequency is measured by LDI counts.
   - (ii) No.
   - (iii) No: lineages are cooperators versus defectors, plus Hamming distance.
   - Its conventions are very different from ours: "A program enters a halt state either by successfully completing L targeted write operations into its opponent’s memory segment" (L is LaTeX in the source).
3. **Knierim et al. 2026** (arXiv:2607.01483; BFF only). Non-executed bytes are noted but given no function.

### Screened and excluded

These do not study machine-code soups:
- 2412.17799 (ASAL);
- 2509.03534 (AlChemy, lambda calculus);
- 2509.23212;
- 2607.02954 (Microcosmos);
- 2609.19902 (self-replicating NCA);
- 2610.12251 (McCulloch–Pitts networks);
- the GECCO '26 Companion paper on Barricelli and symbiogenesis.

### Not found

- No RISC-V, SUBLEQ, 8080 or Forth soup paper after 2024.
- No paper reporting the genealogy of LDIR replicators.
- No paper reporting unexecuted payload bytes with a defensive function in a machine-code soup.

### Implications

- Cite Jha et al. 2026 as a parallel Z80 soup with different conventions.
- Our descent analysis and payload analysis appear to have no direct precedent, within the limits of this search.

---

## 6. Undocumented Intel 8080 opcodes

**Exists:** the claim is confirmed by the best available evidence. I found no published test on genuine Intel silicon.

### Sources, best first

1. **Intel's own manual.** Intel 8080 Assembly Language Programming Manual, Rev C (98-004C, 1976), p. xix, table "8080 CPU INSTRUCTIONS IN OPERATION CODE SEQUENCE". It prints 08, 10, 18, 20, 28, 30, 38, CB, D9, DD, ED and FD as "---". Intel therefore leaves these codes **undefined**. Checked against the rendered page.
2. **Decoder reconstructed from the die (near-primary).** `vm80a` by Vslav ("1801BM1"), 2014–2018, CC-BY 3.0. It is gate-level Verilog derived from the die of the KR580VM80A, a Soviet clone of the early i8080A. In `org/rtl/vm80a.v` (lines 673, 705–708):
   - `id_nop = cmp(i, 8'b00xxx000, 8'b00111000)` matches 00, 08, 10, 18, 20, 28, 30, 38.
   - `id_ret = cmp(i, 8'b110x1001, 8'b00010000)` matches C9, D9.
   - `id_call = cmp(i, 8'b11xx1101, 8'b00110000)` matches CD, DD, ED, FD.
   - `id_jmp = cmp(i, 8'b1100x011, 8'b00001000)` matches C3, CB.
   - I enumerated all 256 bytes against these masks.
   - **Caveat:** this is a clone, not Intel silicon. Zeptobars reports: "Layout of KR580VM80A is quite similar though not identical to i8080, but there were no differing (vs i8080) opcodes identified."
3. **Emulator source.** The MAME `src/devices/cpu/i8085/i8085.cpp` changelog (K. Strzecha, 20 Jul 2002) says: "Undocumented i8080 opcodes added: 08h, 10h, 18h, 20h, 28h, 30h, 38h - NOP; 0CBh - JMP; 0D9h - RET; 0DDh, 0EDh, 0FDh - CALL". The information is credited to A. V. Ignatichev.
   - The same file shows the **8085 differs** for every one of these codes: 08 = DSUB, 10 = ARHL, 18 = RDEL, 20 = RIM, 28 = LDHI, 30 = SIM, 38 = LDSI, CB = RSTV, D9 = SHLX, DD = JNX5, ED = LHLX, FD = JX5.
4. **Opcode table (secondary).** The pastraiser.com 8080 table lists these codes as \*NOP, \*JMP a16, \*RET and \*CALL a16, with the footnote "All instructions marked by "*" are only alternative opcodes for existing instructions." The direct download was blocked, so I read the page through a summarising fetch tool.

### Discrepancies with the reviews

- **DR3 cites the wrong source.** It says Intel's 1976 manual (98-004C) verifies the alias decoding. That manual lists these codes as "---".
- **DR1 and DR2 are correct**, but cite only secondary sources.

### Implications

- On a real 8080:
  - `ED B0 nn` is a 3-byte CALL to address nn·256 + B0h;
  - CB is JMP;
  - 00 is NOP.
- An ablation that decodes these codes as NOP or trap, or treats 00 as lethal, is an "8080-like subset", not an 8080. Rename it, or implement the aliases.
- Never write "8085" for "8080": on the 8085 these codes mean something else.
